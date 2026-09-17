"""
3-Head Hybrid Retrieval:
  Head 1 -- Qdrant dense + BM25 sparse (hybrid)
  Head 2 -- Neo4j KG traversal (gracefully disabled if Neo4j not running)
  Head 3 -- SymPy symbolic solver
Uses Groq for LLM calls.
"""
import json, os, re, time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseVector,
    NamedVector, NamedSparseVector,
)
import sympy as sp
from dotenv import load_dotenv, find_dotenv
from groq import Groq
import groq as groq_lib
from tqdm import tqdm

load_dotenv(find_dotenv())
client_llm = Groq()

# ── Config ────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-base-en-v1.5"   # base fits in 4 GB VRAM
QDRANT_PATH = "./phase3/qdrant_storage"
COLLECTION  = "nsrag_corpus"
TOP_K       = 5

embedder = SentenceTransformer(EMBED_MODEL, device="cuda")

@dataclass
class RetrievedPassage:
    title:  str
    text:   str
    score:  float
    source: str   # "dense" | "sparse" | "kg" | "solver"
    hop:    int = 0


def _groq_call(prompt: str, max_tokens: int = 300, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            resp = client_llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except groq_lib.RateLimitError:
            wait = 2 ** attempt * 10
            print(f"\n  [rate limit] waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(5)
    return ""


# ══════════════════════════════════════════════════════════════════════
# HEAD 1 -- Qdrant Hybrid (Dense + BM25 Sparse)
# ══════════════════════════════════════════════════════════════════════
class HybridRetriever:
    def __init__(self):
        os.makedirs(QDRANT_PATH, exist_ok=True)
        self.qdrant = QdrantClient(path=QDRANT_PATH)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if COLLECTION not in existing:
            self.qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config={
                    "dense": VectorParams(size=768, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                },
            )
            print(f"Created Qdrant collection: {COLLECTION}")

    def _bm25_sparse(self, text: str) -> Tuple[List[int], List[float]]:
        """TF-weighted sparse vector over 30k token hash space."""
        from collections import Counter
        tokens = re.findall(r"\w+", text.lower())
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        indices = [abs(hash(t)) % 30000 for t in tf]
        values  = [round(c / total, 6) for c in tf.values()]
        # Qdrant requires unique indices
        deduped: Dict[int, float] = {}
        for idx, val in zip(indices, values):
            deduped[idx] = deduped.get(idx, 0) + val
        return list(deduped.keys()), list(deduped.values())

    def is_indexed(self) -> bool:
        info = self.qdrant.get_collection(COLLECTION)
        return info.points_count > 0

    def index_corpus(self, corpus: List[Dict], batch_size: int = 64):
        if self.is_indexed():
            print(f"Collection already has {self.qdrant.get_collection(COLLECTION).points_count} points, skipping re-index.")
            return
        print(f"Indexing {len(corpus)} passages into Qdrant...")
        texts = [f"{d['title']} {d['text']}" for d in corpus]
        points = []
        for i in range(0, len(corpus), batch_size):
            batch_texts  = texts[i:i+batch_size]
            batch_corpus = corpus[i:i+batch_size]
            dense_embs   = embedder.encode(
                batch_texts, normalize_embeddings=True,
                device="cuda", show_progress_bar=False,
            )
            for j, (doc, emb) in enumerate(zip(batch_corpus, dense_embs)):
                idx, vals = self._bm25_sparse(doc["text"])
                points.append(PointStruct(
                    id=i + j,
                    vector={
                        "dense":  emb.tolist(),
                        "sparse": SparseVector(indices=idx, values=vals),
                    },
                    payload={"title": doc["title"], "text": doc["text"]},
                ))
        self.qdrant.upsert(collection_name=COLLECTION, points=points)
        print(f"Indexed {len(points)} passages.")

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[RetrievedPassage]:
        q_emb = embedder.encode(
            [query], normalize_embeddings=True, device="cuda"
        )[0].tolist()

        results = self.qdrant.query_points(
            collection_name=COLLECTION,
            query=q_emb,
            using="dense",
            limit=top_k,
            with_payload=True,
        ).points
        return [
            RetrievedPassage(
                title=r.payload["title"],
                text=r.payload["text"],
                score=r.score,
                source="dense",
            )
            for r in results
        ]


# ══════════════════════════════════════════════════════════════════════
# HEAD 2 -- Knowledge Graph (Neo4j) — gracefully disabled if unavailable
# ══════════════════════════════════════════════════════════════════════
class KGRetriever:
    def __init__(self, uri="bolt://localhost:7687",
                 user="neo4j", password="password123"):
        self.available = False
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self.available = True
            print("Neo4j connected.")
        except Exception as e:
            print(f"Neo4j unavailable ({e}) -- KG head disabled.")

    def extract_entities(self, text: str) -> List[str]:
        if not self.available:
            return []
        raw = _groq_call(
            f"Extract 2-3 key named entities from this question. "
            f"Return as JSON array only, no explanation: {text}",
            max_tokens=80,
        )
        try:
            # handle both ["a","b"] and {"entities":["a","b"]}
            raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            return parsed.get("entities", parsed.get("items", []))
        except Exception:
            return []

    def retrieve(self, query: str, entities: List[str],
                 hops: int = 2) -> List[RetrievedPassage]:
        if not self.available or not entities:
            return []
        passages = []
        with self.driver.session() as session:
            for entity in entities[:2]:
                result = session.run(
                    f"""
                    MATCH path = (start:Entity {{name: $name}})-[*1..{hops}]-(end:Entity)
                    RETURN start.name AS src,
                           [r IN relationships(path) | type(r)] AS rels,
                           end.name AS tgt,
                           end.description AS desc
                    LIMIT 10
                    """,
                    name=entity,
                )
                for record in result:
                    text = (f"{record['src']} "
                            f"{'->'.join(record['rels'])} "
                            f"{record['tgt']}: {record['desc'] or ''}")
                    passages.append(RetrievedPassage(
                        title=f"KG: {record['src']}",
                        text=text, score=0.85, source="kg",
                    ))
        return passages

    def build_from_corpus(self, corpus: List[Dict], limit: int = 500):
        if not self.available:
            return
        print(f"Building KG from {min(len(corpus), limit)} passages...")
        for doc in corpus[:limit]:
            raw = _groq_call(
                f'Extract entity-relation-entity triples from this text.\n'
                f'Return ONLY a JSON array: [["entity1","relation","entity2"],...]\n'
                f'Max 3 triples. Text: {doc["text"][:300]}',
                max_tokens=150,
            )
            try:
                raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()
                parsed = json.loads(raw)
                triples = parsed if isinstance(parsed, list) else parsed.get("triples", [])
                with self.driver.session() as session:
                    for triple in triples:
                        if len(triple) != 3:
                            continue
                        e1, rel, e2 = triple
                        rel_clean = re.sub(r"[^A-Za-z0-9_]", "_", rel.upper())
                        session.run(
                            f"MERGE (a:Entity {{name:$e1}}) "
                            f"MERGE (b:Entity {{name:$e2}}) "
                            f"MERGE (a)-[:{rel_clean}]->(b)",
                            e1=e1, e2=e2,
                        )
            except Exception:
                continue
            time.sleep(0.5)
        print("KG built.")


# ══════════════════════════════════════════════════════════════════════
# HEAD 3 -- SymPy Symbolic Solver
# ══════════════════════════════════════════════════════════════════════
class SymbolicSolver:
    TRIGGER_PATTERNS = [
        r"\b(solve|calculate|compute|evaluate|find|derive)\b",
        r"\b(equation|formula|integral|derivative|limit)\b",
        r"[=\+\-\*/\^]",
        r"\b(x|y|z)\s*=",
        r"\d+\s*[\+\-\*/]\s*\d+",
    ]

    def should_trigger(self, question: str) -> bool:
        q = question.lower()
        return any(re.search(p, q) for p in self.TRIGGER_PATTERNS)

    def extract_and_solve(self, question: str) -> Optional[RetrievedPassage]:
        code_str = _groq_call(
            f"Convert this math/physics question into SymPy Python code.\n"
            f"Return ONLY executable Python using sympy. End with: result = <answer>\n"
            f"If not solvable symbolically, write: result = None\n"
            f"Question: {question}",
            max_tokens=300,
        )
        try:
            code_str = re.sub(r"```(?:python)?|```", "", code_str).strip()
            safe_ns = {
                "sp": sp, "symbols": sp.symbols, "Symbol": sp.Symbol,
                "solve": sp.solve, "simplify": sp.simplify,
                "expand": sp.expand, "factor": sp.factor,
                "integrate": sp.integrate, "diff": sp.diff,
                "Matrix": sp.Matrix, "pi": sp.pi, "E": sp.E,
                "sqrt": sp.sqrt, "Rational": sp.Rational,
                "__builtins__": {},
            }
            exec(code_str, safe_ns)  # noqa: S102
            result = safe_ns.get("result")
            if result is not None and result is not sp.nan:
                return RetrievedPassage(
                    title="Symbolic Solution",
                    text=f"Mathematical solution: {sp.latex(result)} = {result}",
                    score=0.99,
                    source="solver",
                )
        except Exception:
            pass
        return None


# ══════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════
class RetrievalRouter:
    def __init__(self):
        self.dense  = HybridRetriever()
        self.kg     = KGRetriever()
        self.solver = SymbolicSolver()

    def route(self, sub_question, context: Dict = None) -> List[RetrievedPassage]:
        domain   = sub_question.domain
        question = sub_question.question
        if context:
            question = question + " " + " ".join(str(v) for v in context.values())

        passages = []

        # Always: dense retrieval
        passages += self.dense.retrieve(question)

        # Scientific domain: add KG
        if domain in ("physics", "chemistry", "biology"):
            entities = self.kg.extract_entities(question)
            passages += self.kg.retrieve(question, entities)

        # Math/physics with equation signals: add solver
        if domain in ("math", "physics") and self.solver.should_trigger(question):
            sol = self.solver.extract_and_solve(question)
            if sol:
                passages.append(sol)

        # Deduplicate and rank
        seen, unique = set(), []
        for p in sorted(passages, key=lambda x: x.score, reverse=True):
            key = (p.title, p.source)
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique[:TOP_K + 2]


# ══════════════════════════════════════════════════════════════════════
# EVAL -- Recall@k + MRR on HotpotQA
# ══════════════════════════════════════════════════════════════════════
def evaluate_retrieval(data_path: str, n: int = 200):
    with open(data_path) as f:
        data = json.load(f)[:n]

    # Build corpus from gold contexts
    corpus: Dict[str, Dict] = {}
    for item in data:
        ctx = item.get("context", {})
        for title, sents in zip(ctx.get("title", []), ctx.get("sentences", [])):
            if title not in corpus:
                corpus[title] = {"title": title, "text": " ".join(sents)}
    corpus_list = list(corpus.values())

    retriever = HybridRetriever()
    retriever.index_corpus(corpus_list)

    recall_5, recall_10, mrr_scores = [], [], []

    for item in tqdm(data, desc="Retrieval eval"):
        sf = item.get("supporting_facts", {})
        gold_titles = list(set(sf.get("title", [])))
        if not gold_titles:
            continue

        passages_5  = retriever.retrieve(item["question"], top_k=5)
        passages_10 = retriever.retrieve(item["question"], top_k=10)
        ret_5  = [p.title for p in passages_5]
        ret_10 = [p.title for p in passages_10]

        recall_5.append(float(any(gt in ret_5  for gt in gold_titles)))
        recall_10.append(float(any(gt in ret_10 for gt in gold_titles)))

        rr = 0.0
        for rank, title in enumerate(ret_10, 1):
            if title in gold_titles:
                rr = 1.0 / rank
                break
        mrr_scores.append(rr)

    n_eval = len(recall_5)
    metrics = {
        "n":         n_eval,
        "Recall@5":  round(100 * sum(recall_5)  / n_eval, 2),
        "Recall@10": round(100 * sum(recall_10) / n_eval, 2),
        "MRR":       round(sum(mrr_scores)       / n_eval, 4),
    }

    print("\n" + "="*50)
    print("PHASE 3 -- RETRIEVAL METRICS")
    print("="*50)
    print(json.dumps(metrics, indent=2))
    print("\nAcceptance gates:")
    print(f"  Recall@5  >= 72%:  {'PASS' if metrics['Recall@5']  >= 72   else 'FAIL'}")
    print(f"  Recall@10 >= 82%:  {'PASS' if metrics['Recall@10'] >= 82   else 'FAIL'}")
    print(f"  MRR       >= 0.55: {'PASS' if metrics['MRR']       >= 0.55 else 'FAIL'}")

    os.makedirs("phase3/results", exist_ok=True)
    with open("phase3/results/retrieval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    evaluate_retrieval("phase1/data/hotpotqa/dev_1k.json", n=200)
