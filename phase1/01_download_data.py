"""Download and sample all datasets locally."""
from datasets import load_dataset
import json, random, os

random.seed(42)

def save_sample(dataset, split, path, n=1000):
    data = list(dataset[split])
    sample = random.sample(data, min(n, len(data)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(sample, f)
    print(f"  Saved {len(sample)} -> {path}")
    return sample

print("HotpotQA...")
hp = load_dataset("hotpot_qa", "fullwiki", trust_remote_code=False)
save_sample(hp, "validation", "data/hotpotqa/dev_1k.json")

print("MuSiQue...")
try:
    mq = load_dataset("musique_qa", trust_remote_code=False)
    save_sample(mq, "validation", "data/musique/dev_1k.json")
except Exception as e:
    print(f"  Primary failed ({e}), trying mirror...")
    mq = load_dataset("alakahanar/musique", trust_remote_code=False)
    save_sample(mq, "validation", "data/musique/dev_1k.json")

print("2WikiMultiHopQA...")
w2 = load_dataset("voidful/2wikimultihopqa", trust_remote_code=False)
save_sample(w2, "validation", "data/2wikimultihop/dev_1k.json")

print("TheoremQA...")
tq = load_dataset("TIGER-Lab/TheoremQA", trust_remote_code=False)
with open("data/theoremqa/test.json", "w") as f:
    json.dump(list(tq["test"]), f)
print(f"  TheoremQA: {len(tq['test'])} samples")

print("\nAll datasets ready.")
