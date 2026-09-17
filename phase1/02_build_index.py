"""Build FAISS index from HotpotQA corpus. Saves index + corpus to disk."""
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, json, time, pickle

MODEL = "BAAI/bge-base-en-v1.5"   # base not large — fits in 4GB VRAM
BATCH = 64
DATA  = "data/hotpotqa/dev_1k.json"
OUT   = "data/hotpotqa/index"

print(f"Loading embedder: {MODEL}")
embedder = SentenceTransformer(MODEL, device="cuda")
print("Loaded on GPU")

with open(DATA) as f:
    data = json.load(f)

docs = {}
for item in data:
    ctx = item.get("context", {})
    for title, sents in zip(ctx.get("title", []), ctx.get("sentences", [])):
        if title not in docs:
            docs[title] = {"title": title, "text": " ".join(sents)}

corpus = list(docs.values())
print(f"Corpus: {len(corpus)} unique passages")

texts = [f"{d['title']} {d['text']}" for d in corpus]
t0 = time.time()
print(f"Encoding {len(texts)} passages (batch={BATCH})...")
embs = embedder.encode(
    texts,
    batch_size=BATCH,
    show_progress_bar=True,
    normalize_embeddings=True,
    device="cuda",
    convert_to_numpy=True,
)
print(f"Encoded in {time.time()-t0:.1f}s  shape={embs.shape}")

dim   = embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embs.astype(np.float32))
print(f"FAISS index: {index.ntotal} vectors, dim={dim}")

faiss.write_index(index, f"{OUT}.faiss")
with open(f"{OUT}_corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)
print(f"Saved -> {OUT}.faiss + {OUT}_corpus.pkl")
