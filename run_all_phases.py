"""
Master runner -- executes all phases sequentially with gate checks.
Run from repo root:
    python run_all_phases.py                  # start from phase 2
    python run_all_phases.py --start_phase 3  # skip phases already done
    python run_all_phases.py --n 200          # smaller sample for fast testing
"""
import argparse, json, sys, os

DATA_DEFAULT = "phase1/data/hotpotqa/dev_1k.json"


def _gate(label: str, value: float, threshold: float, direction: str = "ge") -> bool:
    ok = (value >= threshold) if direction == "ge" else (value < threshold)
    symbol = ">=" if direction == "ge" else "<"
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label} {symbol} {threshold}  (got {value})")
    return ok


def run_phase2(dataset: str, n: int = 50):
    print("\n" + "=" * 60)
    print("PHASE 2 -- SYMBOLIC DECOMPOSER")
    print("=" * 60)
    from phase2.decomposer import evaluate_decomposer
    m = evaluate_decomposer(dataset, n=n,
                             output_path="phase2/results/decomposer_eval.json")
    ok = all([
        _gate("DAG validity",   m["dag_validity_rate"],    85),
        _gate("Hop accuracy",   m["hop_accuracy_within1"], 75),
        _gate("Domain accuracy",m["domain_tag_accuracy"],  80),
    ])
    if not ok:
        print("\nGATE FAIL: Fix decomposer before proceeding to Phase 3.")
        sys.exit(1)
    return m


def run_phase3(dataset: str, n: int = 200):
    print("\n" + "=" * 60)
    print("PHASE 3 -- HYBRID RETRIEVAL")
    print("=" * 60)
    from phase3.hybrid_retriever import evaluate_retrieval
    m = evaluate_retrieval(dataset, n=n)
    ok = all([
        _gate("Recall@5",  m["Recall@5"],  68),
        _gate("Recall@10", m["Recall@10"], 78),
        _gate("MRR",       m["MRR"],       0.50),
    ])
    if not ok:
        print("\nGATE FAIL: Retrieval too weak. Check embedder/index.")
        sys.exit(1)
    return m


def run_phase4(dataset: str, n: int = 500):
    print("\n" + "=" * 60)
    print("PHASE 4 -- NSRAG AGENT")
    print("=" * 60)
    from phase4.nsrag_agent import run_evaluation
    m = run_evaluation(dataset, n=n,
                        output_path="phase4/results/nsrag_hotpotqa.json")
    ok = all([
        _gate("EM", m["EM"], 7),    # realistic for llama-3.1-8b
        _gate("F1", m["F1"], 18),
    ])
    if not ok:
        print("\nGATE FAIL: EM/F1 too low. Debug hop controller before Phase 5.")
        sys.exit(1)
    return m


def run_phase5(n: int = 200):
    pred_path = "phase4/results/nsrag_hotpotqa.json"
    if not os.path.exists(pred_path):
        print(f"Phase 4 predictions not found at {pred_path}. Run Phase 4 first.")
        sys.exit(1)
    print("\n" + "=" * 60)
    print("PHASE 5 -- FORMAL VERIFIER")
    print("=" * 60)
    from phase5.verifier import evaluate_verifier
    m = evaluate_verifier(pred_path, n=n)
    ok = all([
        _gate("Hallucination rate", m["hallucination_after"],    20, "lt"),
        _gate("False positive rate",m["false_positive_rate"],    10, "lt"),
        _gate("Pass rate",          m["verification_pass_rate"], 80),
    ])
    if not ok:
        print("\nGATE FAIL: Verifier thresholds not met.")
        sys.exit(1)
    return m


def print_summary(results: dict):
    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE -- SUMMARY")
    print("=" * 60)
    for phase, m in results.items():
        print(f"\n  {phase}:")
        for k, v in m.items():
            if k not in ("n", "system", "by_hop"):
                print(f"    {k}: {v}")
    print("\nReady for Phase 6 -- SciHopBench + full cross-dataset eval.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_phase", type=int, default=2,
                        help="Phase to start from (2-5)")
    parser.add_argument("--dataset", default=DATA_DEFAULT,
                        help="Path to dataset JSON")
    parser.add_argument("--n", type=int, default=500,
                        help="Samples for Phase 4 (others use fixed counts)")
    args = parser.parse_args()

    results = {}

    if args.start_phase <= 2:
        results["Phase 2"] = run_phase2(args.dataset)

    if args.start_phase <= 3:
        results["Phase 3"] = run_phase3(args.dataset)

    if args.start_phase <= 4:
        results["Phase 4"] = run_phase4(args.dataset, n=args.n)

    if args.start_phase <= 5:
        results["Phase 5"] = run_phase5()

    print_summary(results)
