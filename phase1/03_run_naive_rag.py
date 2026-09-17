"""Naive RAG with checkpointing + Groq rate-limit retry."""
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, json, time, pickle, re, string, os
from collections import Counter
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq
import groq as groq_lib

load_dotenv("../.env")
import itertools

# Pre-create one client per key — avoids per-call allocation overhead
_keys = [k.strip() for k in [
    os.environ.get("GROQ_API_KEY", ""),
    os.environ.get("GROQ_API_KEY2", ""),
    os.environ.get("GROQ_API_KEY3", ""),
] if k.strip()]
_clients = [Groq(api_key=k) for k in _keys]
_client_pool = itertools.cycle(_clients)
print(f"Using {len(_clients)} Groq client(s) in round-robin")

MODEL  = "BAAI/bge-base-en-v1.5"
INDEX  = "data/hotpotqa/index.faiss"
CORPUS = "data/hotpotqa/index_corpus.pkl"
DATA   = "data/hotpotqa/dev_1k.json"
CKPT   = "results/naive_rag_checkpoint.json"
OUT    = "results/naive_rag_hotpotqa.json"
SCORES = "results/naive_rag_hotpotqa_scores.json"

# ── eval ────────────────────────────────────────────────────────────
def normalize(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def exact_match(pred, golds):
    return float(any(normalize(pred) == normalize(g) for g in golds))

def f1_score(pred, golds):
    def _f1(p, g):
        pt = normalize(p).split(); gt = normalize(g).split()
        common = Counter(pt) & Counter(gt)
        n = sum(common.values())
        if not n: return 0.0
        return 2*(n/len(pt))*(n/len(gt)) / (n/len(pt)+n/len(gt))
    return max(_f1(pred, g) for g in golds)

def evaluate(preds):
    em, f1 = [], []
    for p in preds:
        golds = p["gold_answers"]
        if isinstance(golds, str): golds = [golds]
        em.append(exact_match(p["prediction"], golds))
        f1.append(f1_score(p["prediction"], golds))
    n = len(preds)
    return {"n": n, "EM": round(100*sum(em)/n, 2), "F1": round(100*sum(f1)/n, 2)}

# ── load index ──────────────────────────────────────────────────────
print("Loading embedder...")
embedder = SentenceTransformer(MODEL, device="cuda")
index    = faiss.read_index(INDEX)
with open(CORPUS, "rb") as f:
    corpus = pickle.load(f)
print(f"Index: {index.ntotal} vectors  Corpus: {len(corpus)} passages")

def retrieve(question, top_k=5):
    q_emb = embedder.encode([question], normalize_embeddings=True,
                             device="cuda", convert_to_numpy=True).astype(np.float32)
    _, idxs = index.search(q_emb, top_k)
    return [corpus[i] for i in idxs[0]]

def generate(question, passages, retries=5):
    ctx = "\n\n".join(
        f"[{i+1}] {p['title']}: {p['text'][:400]}" for i, p in enumerate(passages)
    )
    for attempt in range(retries):
        try:
            resp = next(_client_pool).chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content":
                    f"Answer in 1-5 words using ONLY the context.\n\n"
                    f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"}],
                temperature=0, max_tokens=50,
            )
            return resp.choices[0].message.content.strip()
        except groq_lib.RateLimitError:
            wait = 2 ** attempt * 10   # 10s, 20s, 40s, 80s, 160s
            print(f"\n  [rate limit] waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"\n  [error] {e} — retrying in 5s...")
            time.sleep(5)
    return ""

# ── resume from checkpoint ──────────────────────────────────────────
os.makedirs("results", exist_ok=True)
if os.path.exists(CKPT):
    with open(CKPT) as f:
        predictions = json.load(f)
    done_ids = {p["id"] for p in predictions}
    print(f"Resuming from checkpoint: {len(predictions)} done")
else:
    predictions, done_ids = [], set()

with open(DATA) as f:
    data = json.load(f)

remaining = [item for item in data if item.get("id","") not in done_ids]
print(f"Remaining: {len(remaining)}/1000")

latencies = []
for item in tqdm(remaining, desc="Naive RAG", initial=len(predictions), total=1000):
    q, ans = item["question"], item["answer"]
    t0 = time.time()
    passages = retrieve(q)
    pred     = generate(q, passages)
    lat      = time.time() - t0
    predictions.append({
        "id": item.get("id",""), "question": q,
        "prediction": pred, "gold_answers": [ans],
        "hops": 2, "latency": round(lat, 3),
        "retrieved": [p["title"] for p in passages],
    })
    latencies.append(lat)
    # save checkpoint every 50
    if len(predictions) % 50 == 0:
        with open(CKPT, "w") as f:
            json.dump(predictions, f)

# final save
with open(OUT, "w") as f:
    json.dump(predictions, f, indent=2)
if os.path.exists(CKPT):
    os.remove(CKPT)

scores = evaluate(predictions)
scores["avg_latency"] = round(sum(latencies)/len(latencies), 2) if latencies else 0
scores["system"] = "NaiveRAG-bge-base-groq-llama3.1-8b"
print(json.dumps(scores, indent=2))

with open(SCORES, "w") as f:
    json.dump(scores, f, indent=2)
print("Done. Saved:", OUT)
