"""Shared eval harness — exact match + F1, used by all phases."""
import re, string
from collections import Counter
from typing import List, Dict


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def exact_match(pred: str, golds: List[str]) -> float:
    return float(any(normalize(pred) == normalize(g) for g in golds))


def f1_score(pred: str, golds: List[str]) -> float:
    def _f1(p, g):
        pt = normalize(p).split()
        gt = normalize(g).split()
        common = Counter(pt) & Counter(gt)
        n = sum(common.values())
        if not n:
            return 0.0
        return 2 * (n / len(pt)) * (n / len(gt)) / (n / len(pt) + n / len(gt))
    return max(_f1(pred, g) for g in golds)


def evaluate(predictions: List[Dict]) -> Dict:
    em_scores, f1_scores, hop_data = [], [], {}
    for item in predictions:
        pred  = item.get("prediction", "")
        golds = item.get("gold_answers", [])
        if isinstance(golds, str):
            golds = [golds]
        em = exact_match(pred, golds)
        f1 = f1_score(pred, golds)
        em_scores.append(em)
        f1_scores.append(f1)
        h = item.get("hops", -1)
        hop_data.setdefault(h, {"em": [], "f1": []})
        hop_data[h]["em"].append(em)
        hop_data[h]["f1"].append(f1)

    n = len(predictions)
    return {
        "n":  n,
        "EM": round(100 * sum(em_scores) / n, 2),
        "F1": round(100 * sum(f1_scores) / n, 2),
        "by_hop": {
            str(k): {
                "EM": round(100 * sum(v["em"]) / len(v["em"]), 2),
                "F1": round(100 * sum(v["f1"]) / len(v["f1"]), 2),
                "n":  len(v["em"]),
            }
            for k, v in sorted(hop_data.items())
        },
    }
