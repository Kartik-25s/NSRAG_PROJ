"""
Symbolic Query Decomposer + Hop Planner
Converts a complex question -> typed DAG of sub-questions
Uses Groq (free) instead of OpenAI.
"""
import json, os, time, re
from dataclasses import dataclass, asdict
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv
from groq import Groq
import groq as groq_lib
from tqdm import tqdm

load_dotenv(find_dotenv())
client = Groq()

# ── Data structures ──────────────────────────────────────────────────
@dataclass
class SubQuestion:
    id: int
    question: str
    domain: str          # physics | math | chemistry | biology | general
    depends_on: List[int]
    answer: Optional[str] = None
    confidence: float = 0.0

@dataclass
class QuestionDAG:
    original: str
    sub_questions: List[SubQuestion]
    estimated_hops: int
    valid: bool = True
    error: Optional[str] = None

# ── Decomposer prompt ────────────────────────────────────────────────
DECOMPOSE_PROMPT = """You are a multi-hop question decomposer. Most questions require exactly 2 hops.

Break the question into the MINIMUM number of sub-questions needed (usually exactly 2):
- Sub-question 1: find an intermediate fact
- Sub-question 2: use that fact to answer the original question

Rules:
1. Use EXACTLY 2 sub-questions unless the question clearly needs 3 (very rare)
2. Each sub-question answered with a single fact lookup
3. Domain tag: physics | math | chemistry | biology | general
4. Sub-question 2 must depend on sub-question 1

Output ONLY valid JSON, no markdown:
{{
  "sub_questions": [
    {{
      "id": 1,
      "question": "...",
      "domain": "general",
      "depends_on": []
    }},
    {{
      "id": 2,
      "question": "...",
      "domain": "general",
      "depends_on": [1]
    }}
  ],
  "estimated_hops": 2
}}

Question: {question}"""

def _call_groq(prompt: str, retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
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

def _extract_json(text: str) -> dict:
    """Robustly extract JSON even if model wraps it in markdown."""
    # strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    # find first { ... }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)

def decompose(question: str, retries: int = 3) -> QuestionDAG:
    for attempt in range(retries):
        try:
            raw  = _call_groq(DECOMPOSE_PROMPT.format(question=question))
            data = _extract_json(raw)

            raw_sqs = data["sub_questions"][:3]  # cap at 3 sub-questions
            sqs = [
                SubQuestion(
                    id=sq["id"],
                    question=sq["question"],
                    domain=sq.get("domain", "general"),
                    depends_on=[d for d in sq.get("depends_on", [])
                                if any(r["id"] == d for r in raw_sqs)],
                )
                for sq in raw_sqs
            ]

            # Validate: dependency IDs exist and are strictly less than current id
            ids = {sq.id for sq in sqs}
            for sq in sqs:
                for dep in sq.depends_on:
                    if dep not in ids or dep >= sq.id:
                        raise ValueError(f"Bad dependency: sq{sq.id} -> sq{dep}")

            est_hops = min(data.get("estimated_hops", len(sqs)), len(sqs))
            return QuestionDAG(
                original=question,
                sub_questions=sqs,
                estimated_hops=est_hops,
                valid=True,
            )

        except Exception as e:
            if attempt == retries - 1:
                return QuestionDAG(
                    original=question, sub_questions=[],
                    estimated_hops=-1, valid=False, error=str(e)
                )
            time.sleep(2 ** attempt)

# ── Topological sort executor ────────────────────────────────────────
def topological_sort(dag: QuestionDAG) -> List[SubQuestion]:
    visited, order = set(), []
    sq_map = {sq.id: sq for sq in dag.sub_questions}

    def visit(sq_id):
        if sq_id in visited: return
        visited.add(sq_id)
        for dep in sq_map[sq_id].depends_on:
            visit(dep)
        order.append(sq_map[sq_id])

    for sq in dag.sub_questions:
        visit(sq.id)
    return order

# ── Domain tagger ────────────────────────────────────────────────────
DOMAIN_KEYWORDS = {
    "physics":   ["force", "energy", "momentum", "velocity", "mass", "gravity",
                  "electric", "magnetic", "quantum", "photon", "wave",
                  "entropy", "pressure", "temperature", "speed", "acceleration"],
    "math":      ["equation", "integral", "derivative", "matrix", "vector",
                  "probability", "theorem", "proof", "polynomial", "function",
                  "limit", "series", "calculate", "compute", "solve", "formula"],
    "chemistry": ["element", "compound", "reaction", "molecule", "bond", "acid",
                  "base", "oxidation", "catalyst", "molar", "periodic", "isotope"],
    "biology":   ["protein", "gene", "cell", "dna", "rna", "enzyme", "organism",
                  "species", "evolution", "metabolism", "chromosome"],
}

def validate_domain(question: str, predicted_domain: str) -> str:
    q = question.lower()
    scores = {d: sum(1 for kw in kws if kw in q) for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return predicted_domain
    if best != predicted_domain and scores[best] >= 2:
        return best
    return predicted_domain

# ── Batch evaluation ─────────────────────────────────────────────────
def evaluate_decomposer(
    data_path: str,
    n: int = 50,
    output_path: str = "phase2/results/decomposer_eval.json",
):
    with open(data_path) as f:
        data = json.load(f)[:n]

    results = []
    valid_count  = 0
    hop_errors   = []
    domain_correct = 0
    domain_total   = 0

    for item in tqdm(data, desc="Decomposing"):
        q   = item["question"]
        dag = decompose(q)

        if dag.valid:
            valid_count += 1

        # HotpotQA ground-truth is always 2-hop
        pred_hops = dag.estimated_hops if dag.estimated_hops > 0 else 0
        hop_errors.append(abs(pred_hops - 2))

        for sq in dag.sub_questions:
            corrected = validate_domain(sq.question, sq.domain)
            if corrected == sq.domain:
                domain_correct += 1
            domain_total += 1

        results.append({
            "question":        q,
            "dag":             asdict(dag),
            "valid":           dag.valid,
            "estimated_hops":  dag.estimated_hops,
            "error":           dag.error,
        })
        time.sleep(0.5)  # gentle pacing

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    metrics = {
        "n":                    n,
        "dag_validity_rate":    round(100 * valid_count / n, 1),
        "hop_accuracy_within1": round(100 * sum(1 for e in hop_errors if e <= 1) / n, 1),
        "domain_tag_accuracy":  round(100 * domain_correct / max(domain_total, 1), 1),
        "avg_hops_predicted":   round(
            sum(r["estimated_hops"] for r in results if r["estimated_hops"] > 0)
            / max(valid_count, 1), 2
        ),
    }

    print("\n" + "="*50)
    print("PHASE 2 -- DECOMPOSER METRICS")
    print("="*50)
    print(json.dumps(metrics, indent=2))
    print("\nAcceptance gates:")
    print(f"  DAG validity  >= 90%:   {'PASS' if metrics['dag_validity_rate']    >= 90 else 'FAIL'}")
    print(f"  Hop accuracy  >= 75%:   {'PASS' if metrics['hop_accuracy_within1'] >= 75 else 'FAIL'}")
    print(f"  Domain acc    >= 80%:   {'PASS' if metrics['domain_tag_accuracy']  >= 80 else 'FAIL'}")

    with open(output_path.replace(".json", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    # ── Single question smoke test ──────────────────────────────────
    test_q = ("What government position was held by the "
              "Canadian-born writer of 'The Blind Assassin'?")
    dag = decompose(test_q)
    print("\nDECOMPOSITION TEST:")
    print(f"  Original: {dag.original}")
    for sq in dag.sub_questions:
        deps = f" (needs: {sq.depends_on})" if sq.depends_on else ""
        print(f"  [{sq.id}] [{sq.domain}] {sq.question}{deps}")
    print(f"  Estimated hops: {dag.estimated_hops}  Valid: {dag.valid}")
    if dag.error:
        print(f"  Error: {dag.error}")

    print()

    # ── Full 50-sample eval on HotpotQA ────────────────────────────
    evaluate_decomposer(
        "phase1/data/hotpotqa/dev_1k.json",
        n=50,
        output_path="phase2/results/decomposer_eval.json",
    )
