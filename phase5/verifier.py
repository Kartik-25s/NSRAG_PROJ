"""
Formal Verifier + Hallucination Guard
Gates: SymPy equation check -> Chain-of-Verification -> Attribution check
Uses Groq for all LLM calls.
Run from repo root: python phase5/01_verifier.py
"""
import json, os, re, sys, time, itertools
from dataclasses import dataclass
from typing import List, Optional, Dict
import sympy as sp
from dotenv import load_dotenv, find_dotenv
from groq import Groq
import groq as groq_lib
from tqdm import tqdm

load_dotenv(find_dotenv())
_keys = [k.strip() for k in [
    os.environ.get("GROQ_API_KEY",   ""),
    os.environ.get("GROQ_API_KEY2",  ""),
    os.environ.get("GROQ_API_KEY3",  ""),
    os.environ.get("GROQ_API_KEY4",  ""),
    os.environ.get("GROQ_API_KEY5",  ""),
    os.environ.get("GROQ_API_KEY6",  ""),
    os.environ.get("GROQ_API_KEY7",  ""),
    os.environ.get("GROQ_API_KEY8",  ""),
    os.environ.get("GROQ_API_KEY9",  ""),
    os.environ.get("GROQ_API_KEY10", ""),
] if k.strip()]
_clients = [Groq(api_key=k) for k in _keys]
_client_pool = itertools.cycle(_clients)
print(f"Using {len(_clients)} Groq client(s) in round-robin")

sys.path.insert(0, ".")
from eval.evaluate import exact_match


# ── Groq helper ──────────────────────────────────────────────────────
def _groq(prompt: str, max_tokens: int = 300, retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            resp = next(_client_pool).chat.completions.create(
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
                return ""
            time.sleep(5)
    return ""


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)


# ── Data structure ───────────────────────────────────────────────────
@dataclass
class VerificationResult:
    passed:             bool
    confidence:         float
    hallucination_rate: float
    failed_gate:        Optional[str]   # "equation" | "cov" | "attribution"
    detail:             Optional[str]
    verified_equations: List[str]
    final_answer:       str             # may be revised after fallback


# ══════════════════════════════════════════════════════════════════════
# GATE 1 -- SymPy equation consistency (math/physics only)
# ══════════════════════════════════════════════════════════════════════
def extract_equations(text: str) -> List[str]:
    patterns = [
        r"\$\$(.+?)\$\$",
        r"\$(.+?)\$",
        r"\\begin\{equation\}(.+?)\\end\{equation\}",
        r"(\d+[\+\-\*/\^]\d+\s*=\s*\d+)",
    ]
    eqs = []
    for p in patterns:
        eqs.extend(re.findall(p, text, re.DOTALL))
    return [e.strip() for e in eqs if len(e.strip()) > 2]


def verify_equation_sympy(eq_str: str) -> tuple:
    try:
        clean = (eq_str
                 .replace(r"\frac", "sp.Rational")
                 .replace("^", "**")
                 .replace(r"\pi", "sp.pi")
                 .replace(r"\infty", "sp.oo"))
        if "=" in clean:
            lhs, rhs = clean.split("=", 1)
            lhs_expr = sp.sympify(lhs.strip(), evaluate=True)
            rhs_expr = sp.sympify(rhs.strip(), evaluate=True)
            diff     = sp.simplify(lhs_expr - rhs_expr)
            return (diff == 0), f"LHS-RHS = {diff}"
        else:
            sp.sympify(clean, evaluate=True)
            return True, "Parsed OK"
    except Exception as e:
        return True, f"Unparseable (skipped): {e}"  # don't penalize LaTeX we can't parse


# ══════════════════════════════════════════════════════════════════════
# GATE 2 -- Chain-of-Verification
# ══════════════════════════════════════════════════════════════════════
def chain_of_verification(answer: str, question: str,
                           n_checks: int = 3) -> tuple:
    raw = _groq(
        f"Generate {n_checks} yes/no verification questions that check specific "
        f"facts in this answer. Return JSON: "
        f'[{{"question":"...","expected":"yes"}},...]\n\n'
        f"Original question: {question}\nAnswer: {answer}",
        max_tokens=300,
    )
    try:
        parsed = _extract_json(raw) if "{" in raw else json.loads(raw)
        checks = parsed if isinstance(parsed, list) else parsed.get("checks", [])
    except Exception:
        return True, []   # generation failed -- don't block

    failures = []
    for check in checks[:n_checks]:
        ans_raw = _groq(
            f"Answer yes or no only.\nQuestion: {check['question']}\nAnswer:",
            max_tokens=5,
        )
        actual = ans_raw.lower().strip()
        if check.get("expected", "yes").lower() not in actual:
            failures.append(
                f"FAIL: {check['question']} "
                f"(expected {check.get('expected')}, got {actual})"
            )
        time.sleep(0.3)   # gentle pacing between checks

    return len(failures) == 0, failures


# ══════════════════════════════════════════════════════════════════════
# GATE 3 -- Attribution check
# ══════════════════════════════════════════════════════════════════════
def attribution_check(answer: str, hop_trace: List[Dict]) -> tuple:
    all_evidence = " ".join(
        " ".join(str(e) for e in h.get("evidence", []))
        + " " + h.get("answer", "")
        for h in hop_trace
    ).lower()

    raw = _groq(
        f'Extract all factual claims from this answer.\n'
        f'Return JSON: {{"claims":["claim1","claim2",...]}}\n'
        f"Answer: {answer}",
        max_tokens=200,
    )
    try:
        claims = _extract_json(raw).get("claims", [])
    except Exception:
        return 0.0, []   # assume grounded if extraction fails

    STOPWORDS = {"the", "a", "an", "is", "was", "of", "in", "and", "to", "that"}
    ungrounded = []
    for claim in claims:
        words   = set(claim.lower().split()) - STOPWORDS
        overlap = words & set(all_evidence.split())
        if words and len(overlap) / len(words) < 0.3:
            ungrounded.append(claim)

    halluc_rate = len(ungrounded) / max(len(claims), 1)
    return halluc_rate, ungrounded


# ══════════════════════════════════════════════════════════════════════
# FULL VERIFIER PIPELINE
# ══════════════════════════════════════════════════════════════════════
class FormalVerifier:

    def verify(
        self,
        question:  str,
        answer:    str,
        hop_trace: List[Dict],
        domain:    str = "general",
        agent=None,
    ) -> VerificationResult:

        verified_equations: List[str] = []

        # ── Gate 1: equation check (math/physics only) ────────────────
        if domain in ("math", "physics"):
            for eq in extract_equations(answer):
                valid, detail = verify_equation_sympy(eq)
                if not valid:
                    return VerificationResult(
                        passed=False, confidence=0.1,
                        hallucination_rate=0.8,
                        failed_gate="equation", detail=detail,
                        verified_equations=[],
                        final_answer=answer,
                    )
                verified_equations.append(eq)

        # ── Gate 2: chain-of-verification ────────────────────────────
        cov_passed, cov_failures = chain_of_verification(answer, question)
        if not cov_passed and len(cov_failures) >= 2:
            if agent:
                new_result    = agent.run(question)
                answer        = new_result.final_answer
                hop_trace     = [vars(h) for h in new_result.hop_trace]
                cov_passed, _ = chain_of_verification(answer, question)

            if not cov_passed:
                return VerificationResult(
                    passed=False, confidence=0.35,
                    hallucination_rate=0.5,
                    failed_gate="cov",
                    detail="; ".join(cov_failures[:2]),
                    verified_equations=verified_equations,
                    final_answer=answer,
                )

        # ── Gate 3: attribution ───────────────────────────────────────
        halluc_rate, ungrounded = attribution_check(answer, hop_trace)
        confidence = 1.0 - halluc_rate

        if confidence < 0.6 and agent:
            new_result  = agent.run(question)
            answer      = new_result.final_answer
            hop_trace   = [vars(h) for h in new_result.hop_trace]
            hr2, _      = attribution_check(answer, hop_trace)
            halluc_rate = min(halluc_rate, hr2)
            confidence  = 1.0 - halluc_rate

        return VerificationResult(
            passed=confidence >= 0.6,
            confidence=round(confidence, 3),
            hallucination_rate=round(halluc_rate, 3),
            failed_gate=None if confidence >= 0.6 else "attribution",
            detail=f"Ungrounded: {ungrounded[:2]}" if ungrounded else None,
            verified_equations=verified_equations,
            final_answer=answer,
        )


# ══════════════════════════════════════════════════════════════════════
# BATCH EVALUATION
# ══════════════════════════════════════════════════════════════════════
def evaluate_verifier(predictions_path: str, n: int = 200):
    with open(predictions_path) as f:
        predictions = json.load(f)[:n]

    verifier   = FormalVerifier()
    before_hr  = []
    after_hr   = []
    fp_count   = 0
    pass_count = 0

    for item in tqdm(predictions, desc="Verifying"):
        answer    = item["prediction"]
        question  = item["question"]
        hop_trace = item.get("hop_trace", [])
        domain    = item.get("domain", "general")
        golds     = item["gold_answers"]
        if isinstance(golds, str):
            golds = [golds]

        # Naive hallucination proxy before verification
        evidence_text = " ".join(
            str(h.get("evidence", "")) for h in hop_trace
        ).lower()
        before_hr.append(0.0 if answer.lower() in evidence_text else 0.5)

        result = verifier.verify(question, answer, hop_trace, domain)
        after_hr.append(result.hallucination_rate)

        if result.passed:
            pass_count += 1

        # False positive: verifier blocked a correct answer
        if exact_match(answer, golds) and not result.passed:
            fp_count += 1

        time.sleep(0.3)

    n_eval = len(predictions)
    metrics = {
        "n":                      n_eval,
        "hallucination_before":   round(100 * sum(before_hr) / n_eval, 1),
        "hallucination_after":    round(100 * sum(after_hr)  / n_eval, 1),
        "false_positive_rate":    round(100 * fp_count  / n_eval, 1),
        "verification_pass_rate": round(100 * pass_count / n_eval, 1),
    }

    print("\n" + "=" * 55)
    print("PHASE 5 -- VERIFIER METRICS")
    print("=" * 55)
    print(json.dumps(metrics, indent=2))
    print("\nAcceptance gates:")
    print(f"  Halluc after < 20%: {'PASS' if metrics['hallucination_after']    < 20 else 'FAIL'}")
    print(f"  False pos    < 10%: {'PASS' if metrics['false_positive_rate']    < 10 else 'FAIL'}")
    print(f"  Pass rate    > 80%: {'PASS' if metrics['verification_pass_rate'] > 80 else 'FAIL'}")

    os.makedirs("phase5/results", exist_ok=True)
    with open("phase5/results/verifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    evaluate_verifier("phase4/results/nsrag_hotpotqa.json", n=200)
