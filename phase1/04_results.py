"""Print Phase 1 results table."""
import json, glob

results_all = {}

for path in glob.glob("results/*_scores.json"):
    with open(path) as f:
        d = json.load(f)
    name = d.get("system", path)
    results_all[name] = d

print("\n" + "="*65)
print(f"{'PHASE 1 BASELINE RESULTS':^65}")
print(f"{'HotpotQA fullwiki — 1000 dev samples':^65}")
print("="*65)
print(f"{'System':<32} {'EM':>7} {'F1':>7} {'Latency':>10}")
print("-"*65)
for name, res in results_all.items():
    em  = res.get("EM",  "—")
    f1  = res.get("F1",  "—")
    lat = res.get("avg_latency", "—")
    lat_str = f"{lat}s" if isinstance(lat, (int, float)) else lat
    print(f"{name:<32} {str(em):>7} {str(f1):>7} {lat_str:>10}")
print("="*65)

print("\nLiterature reference ranges:")
for s, (e, f) in [
    ("Naive RAG",  ("35-45", "48-58")),
    ("IRCoT",      ("50-56", "63-68")),
    ("HopRAG",     ("52-58", "65-72")),
]:
    print(f"  {s:<20} EM: {e:>8}   F1: {f}")

with open("results/phase1_summary.json", "w") as f:
    json.dump(results_all, f, indent=2)
print("\nSaved: results/phase1_summary.json")
