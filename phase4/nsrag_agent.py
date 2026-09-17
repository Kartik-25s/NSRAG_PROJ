"""
NSRAG Agent -- full multi-hop loop with EKA alignment.
Uses Groq (free) for all LLM calls.
Run from repo root: python phase4/01_nsrag_agent.py
"""
import json, os, re, time, sys, itertools
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from groq import Groq
import groq as groq_lib

load_dotenv(find_dotenv())
_keys = [k.strip() for k in [
    os.environ.get("GROQ_API_KEY",  ""),
    os.environ.get("GROQ_API_KEY2", ""),
    os.environ.get("GROQ_API_KEY3", ""),
    os.environ.get("GROQ_API_KEY4", ""),
    os.environ.get("GROQ_API_KEY5", ""),
    os.environ.get("GROQ_API_KEY6", ""),
    os.environ.get("GROQ_API_KEY7", ""),
    os.environ.get("GROQ_API_KEY8", ""),
    os.environ.get("GROQ_API_KEY9", ""),
    os.environ.get("GROQ_API_KEY10",""),
] if k.strip()]
_clients = [Groq(api_key=k) for k in _keys]
_client_pool = itertools.cycle(_clients)
print(f"Using {len(_clients)} Groq client(s) in round-robin")

sys.path.insert(0, ".")
from phase2.decomposer import decompose, topological_sort, QuestionDAG, SubQuestion
from phase3.hybrid_retriever import RetrievalRouter, RetrievedPassage
from eval.evaluate import evaluate


# ── Groq helper ──────────────────────────────────────────────────────
def _groq(prompt: str, max_tokens: int = 200, retries: int = 6) -> str:
    for attempt in range(retries):
        try:
            resp = next(_client_pool).chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            time.sleep(0.3)  # gentle pacing to avoid burst
            return resp.choices[0].message.content.strip()
        except groq_lib.RateLimitError:
            wait = (attempt + 1) * 5  # linear: 5s, 10s, 15s... (not exponential)
            print(f"\n  [rate limit] waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            if attempt == retries - 1:
                return ""
            time.sleep(2)
    return ""


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)


# ── Data structures ──────────────────────────────────────────────────
@dataclass
class HopRecord:
    hop_num:    int
    sub_q:      str
    domain:     str
    evidence:   List[str]
    answer:     str
    confidence: float
    utility:    float


@dataclass
class NSRAGResult:
    question:     str
    final_answer: str
    hop_trace:    List[HopRecord]
    total_hops:   int
    confidence:   float
    latency:      float


# ── EKA — Early Knowledge Alignment ─────────────────────────────────
def early_knowledge_align(question: str, dag: QuestionDAG,
                           router: RetrievalRouter) -> QuestionDAG:
    """
    Retrieve broad seed context BEFORE executing the DAG.
    Refine sub-questions if they reference things not in the corpus.
    """
    seed_passages = router.dense.retrieve(question, top_k=3)
    seed_text     = " ".join(p.text[:200] for p in seed_passages)

    if not dag.sub_questions:
        return dag

    plan = json.dumps([
        {"id": sq.id, "question": sq.question,
         "domain": sq.domain, "depends_on": sq.depends_on}
        for sq in dag.sub_questions
    ])

    raw = _groq(
        f"Given this question and seed context, verify the decomposition plan "
        f"is grounded in available knowledge. If a sub-question references "
        f"something not in the context, refine it. Return the same JSON structure.\n\n"
        f"Original question: {question}\n"
        f"Seed context: {seed_text}\n"
        f"Current plan: {plan}\n\n"
        f'Return JSON: {{"sub_questions": [...], "estimated_hops": N}}',
        max_tokens=500,
    )
    try:
        data    = _extract_json(raw)
        refined = data.get("sub_questions", [])
        if refined and len(refined) == len(dag.sub_questions):
            dag.sub_questions = [
                SubQuestion(
                    id=sq["id"], question=sq["question"],
                    domain=sq.get("domain", "general"),
                    depends_on=sq.get("depends_on", []),
                )
                for sq in refined
            ]
    except Exception:
        pass   # fallback: keep original plan
    return dag


# ── Sub-question answerer ────────────────────────────────────────────
def answer_subquestion(sub_q: str, evidence: List[RetrievedPassage],
                        intermediate: Dict) -> tuple:
    ctx = "\n".join(
        f"[{p.source.upper()}] {p.title}: {p.text[:200]}"
        for p in evidence[:3]
    )
    dep_ctx = ""
    if intermediate:
        dep_ctx = "\nPrevious answers:\n" + "\n".join(
            f"  - {v}" for v in intermediate.values()
        )

    ans = _groq(
        f"Answer this sub-question using only the context. "
        f"Be concise (1-10 words). If unsure say UNKNOWN.{dep_ctx}\n\n"
        f"Context:\n{ctx}\n\nSub-question: {sub_q}\nAnswer:",
        max_tokens=60,
    )
    conf = 0.3 if "UNKNOWN" in ans.upper() else 0.85
    return ans, conf


# ── Utility scorer ───────────────────────────────────────────────────
def compute_utility(answer: str, prev_answers: Dict) -> float:
    if "UNKNOWN" in answer.upper():
        return 0.1
    if not prev_answers:
        return 0.9
    prev_text = " ".join(prev_answers.values()).lower()
    return 0.2 if answer.lower() in prev_text else 0.8


# ── Final synthesizer ────────────────────────────────────────────────
def synthesize_final(question: str, hop_trace: List[HopRecord]) -> tuple:
    trace_text = "\n".join(
        f"Step {h.hop_num}: {h.sub_q} -> {h.answer} (conf={h.confidence:.2f})"
        for h in hop_trace
    )
    ans = _groq(
        f"Using these reasoning steps, give the final answer in 1-5 words.\n\n"
        f"Question: {question}\nReasoning:\n{trace_text}\n\nFinal answer:",
        max_tokens=30,
    )
    conf = sum(h.confidence for h in hop_trace) / max(len(hop_trace), 1)
    return ans, conf


# ── Main NSRAG Agent ─────────────────────────────────────────────────
class NSRAGAgent:
    def __init__(self, router: Optional[RetrievalRouter] = None):
        self.router = router or RetrievalRouter()

    def _fallback(self, question: str, t0: float) -> NSRAGResult:
        """Single-hop fallback when decomposition fails."""
        evidence  = self.router.dense.retrieve(question)
        ans, conf = answer_subquestion(question, evidence, {})
        return NSRAGResult(
            question=question, final_answer=ans,
            hop_trace=[], total_hops=1,
            confidence=conf, latency=round(time.time() - t0, 2),
        )

    def run(self, question: str, max_hops: int = 5) -> NSRAGResult:
        t0 = time.time()

        # 1. Decompose
        dag = decompose(question)
        if not dag.valid or not dag.sub_questions:
            return self._fallback(question, t0)

        # 2. EKA alignment (skipped to reduce API calls)
        # dag = early_knowledge_align(question, dag, self.router)

        # 3. Execute hops in topological order
        ordered      = topological_sort(dag)[:max_hops]
        hop_trace    = []
        intermediate = {}   # sq_id -> answer

        for hop_num, sq in enumerate(ordered, 1):
            context  = {k: v for k, v in intermediate.items()
                        if k in sq.depends_on}
            evidence = self.router.route(sq, context)
            ans, conf = answer_subquestion(sq.question, evidence, context)
            utility   = compute_utility(ans, intermediate)

            hop_trace.append(HopRecord(
                hop_num=hop_num, sub_q=sq.question, domain=sq.domain,
                evidence=[p.title for p in evidence[:3]],
                answer=ans, confidence=conf, utility=utility,
            ))
            intermediate[sq.id] = ans

            # Early stop: confident + diminishing utility
            if conf > 0.9 and utility < 0.25 and hop_num >= 2:
                break

        # 4. Synthesize
        final_ans, final_conf = synthesize_final(question, hop_trace)
        return NSRAGResult(
            question=question, final_answer=final_ans,
            hop_trace=hop_trace, total_hops=len(hop_trace),
            confidence=final_conf, latency=round(time.time() - t0, 2),
        )


# ── Batch evaluation ─────────────────────────────────────────────────
def run_evaluation(
    data_path: str,
    n: int = 500,
    output_path: str = "phase4/results/nsrag_hotpotqa.json",
):
    with open(data_path) as f:
        data = json.load(f)[:n]

    # Re-use the indexed Qdrant corpus from phase3
    router = RetrievalRouter()

    # Build index if not already done
    corpus: dict = {}
    for item in data:
        ctx = item.get("context", {})
        for title, sents in zip(ctx.get("title", []), ctx.get("sentences", [])):
            if title not in corpus:
                corpus[title] = {"title": title, "text": " ".join(sents)}
    router.dense.index_corpus(list(corpus.values()))

    agent       = NSRAGAgent(router=router)
    predictions = []
    hop_counts  = []
    ckpt_path   = output_path.replace(".json", "_checkpoint.json")

    # Resume from checkpoint
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            predictions = json.load(f)
        done_ids = {p["id"] for p in predictions}
        data = [d for d in data if d.get("id", "") not in done_ids]
        print(f"Resuming: {len(predictions)} done, {len(data)} remaining")

    for item in tqdm(data, desc="NSRAG Agent",
                     initial=len(predictions), total=n):
        result = agent.run(item["question"])
        predictions.append({
            "id":           item.get("id", ""),
            "question":     item["question"],
            "prediction":   result.final_answer,
            "gold_answers": [item["answer"]],
            "hops":         result.total_hops,
            "latency":      result.latency,
            "hop_trace":    [asdict(h) for h in result.hop_trace],
        })
        hop_counts.append(result.total_hops)

        # checkpoint every item so no progress is lost on crash
        with open(ckpt_path, "w") as f:
            json.dump(predictions, f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    metrics = evaluate(predictions)
    all_hops = [p["hops"] for p in predictions]
    metrics["avg_hops"] = round(sum(all_hops) / len(all_hops), 2)
    metrics["system"]   = "NSRAG-v1"

    print("\n" + "=" * 55)
    print("PHASE 4 -- NSRAG AGENT RESULTS (HotpotQA)")
    print("=" * 55)
    print(json.dumps(metrics, indent=2))
    print("\nAcceptance gates:")
    print(f"  EM  >= 10: {'PASS' if metrics['EM'] >= 10 else 'FAIL'}")
    print(f"  F1  >= 20: {'PASS' if metrics['F1'] >= 20 else 'FAIL'}")
    print(f"  Avg hops 2.1-2.8: {'PASS' if 2.1 <= metrics['avg_hops'] <= 2.8 else 'CHECK'}")

    with open(output_path.replace(".json", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ── Demo ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = NSRAGAgent()
    q = ("What government position was held by the "
         "Canadian-born writer of 'The Blind Assassin'?")
    result = agent.run(q)
    print(f"\nQuestion:  {result.question}")
    print(f"Answer:    {result.final_answer}")
    print(f"Hops:      {result.total_hops}  Conf: {result.confidence:.2f}")
    print(f"Latency:   {result.latency}s")
    print("\nHop trace:")
    for h in result.hop_trace:
        print(f"  [{h.hop_num}][{h.domain}] {h.sub_q}")
        print(f"      -> {h.answer}  (evidence: {h.evidence[:2]})")
