"""Post-hoc analysis of selector study results.

Computes Codex-recommended metrics:
- Oracle recovery: (selector - baseline) / (oracle - baseline)
- k-scaling: fit log(1 - oracle) vs k to estimate hit probability p and correlation rho
- Effective diversity: entropy(answer clusters) / raw k
- Majority failure rate: oracle_exists && majority_wrong
- Per-task category breakdown
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def load_results(results_path: str) -> dict:
    with open(results_path) as f:
        return json.load(f)


def load_candidates(jsonl_path: str) -> Dict[str, list]:
    candidates: Dict[str, list] = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            tid = rec["task_id"]
            if tid not in candidates:
                candidates[tid] = []
            candidates[tid].append(rec)
    return candidates


def compute_answer_anywhere(candidates: Dict[str, list], k_values: List[int]) -> dict:
    """Answer-anywhere analysis: does correct answer appear anywhere in response?

    Separates convergence (answer placement) from computation (answer derivation).
    """
    metrics = {}
    for k in k_values:
        greedy_last = 0
        greedy_anywhere = 0
        pert_last = 0
        pert_anywhere = 0
        total = 0

        for tid, cands in candidates.items():
            greedy = cands[0]
            correct_ans = None
            for c in cands:
                if c["correct"]:
                    correct_ans = c["extracted_answer"]
                    break
            if correct_ans is None:
                total += 1
                continue

            total += 1
            if greedy["correct"]:
                greedy_last += 1
            if correct_ans in greedy.get("all_integers", []):
                greedy_anywhere += 1

            subset = cands[1:k + 1]
            any_last = any(c["correct"] for c in subset)
            any_anywhere = any(
                correct_ans in c.get("all_integers", []) for c in subset
            )
            if any_last:
                pert_last += 1
            if any_anywhere:
                pert_anywhere += 1

        metrics[k] = {
            "greedy_last_int": greedy_last / total if total else 0,
            "greedy_answer_anywhere": greedy_anywhere / total if total else 0,
            "oracle_last_int": pert_last / total if total else 0,
            "oracle_answer_anywhere": pert_anywhere / total if total else 0,
            "total": total,
        }
    return metrics


def compute_scaling_metrics(candidates: Dict[str, list], k_values: List[int]) -> dict:
    """Compute oracle accuracy at each k prefix and fit the basin model."""
    metrics = {}

    for k in k_values:
        oracle_hits = 0
        majority_correct = 0
        majority_wrong_oracle_exists = 0
        total = 0
        per_task_diversity = []

        for tid, cands in candidates.items():
            greedy = cands[0]
            subset = cands[1:k + 1] if len(cands) > k else cands[1:]
            if not subset:
                continue
            total += 1

            oracle_exists = any(c["correct"] for c in subset)
            if oracle_exists:
                oracle_hits += 1

            answers = [c["extracted_answer"] for c in subset if c.get("extracted_answer") is not None]
            if answers:
                majority_answer = Counter(answers).most_common(1)[0][0]
                correct_ans = None
                for c in cands:
                    if c["correct"]:
                        correct_ans = c["extracted_answer"]
                        break
                if correct_ans is not None and majority_answer == correct_ans:
                    majority_correct += 1
                elif oracle_exists:
                    majority_wrong_oracle_exists += 1

                unique_answers = len(set(answers))
                entropy = 0.0
                for count in Counter(answers).values():
                    p = count / len(answers)
                    if p > 0:
                        entropy -= p * math.log2(p)
                per_task_diversity.append(entropy / math.log2(max(len(answers), 2)))

        oracle_acc = oracle_hits / total if total > 0 else 0
        majority_acc = majority_correct / total if total > 0 else 0
        mean_diversity = sum(per_task_diversity) / len(per_task_diversity) if per_task_diversity else 0

        metrics[k] = {
            "oracle_acc": oracle_acc,
            "majority_acc": majority_acc,
            "oracle_not_majority": majority_wrong_oracle_exists,
            "total_tasks": total,
            "mean_effective_diversity": mean_diversity,
        }

    return metrics


def fit_basin_model(scaling_metrics: dict) -> dict:
    """Fit log(1 - oracle) = k_eff * log(1 - p).

    Returns estimated hit probability p and effective k_eff per k value.
    """
    k_values = sorted(scaling_metrics.keys())
    if len(k_values) < 2:
        return {"p_hat": None, "rho_hat": None, "fit_points": []}

    fit_points = []
    for k in k_values:
        oracle = scaling_metrics[k]["oracle_acc"]
        if oracle >= 1.0:
            log_miss = float("-inf")
        elif oracle <= 0.0:
            log_miss = 0.0
        else:
            log_miss = math.log(1 - oracle)
        fit_points.append({"k": k, "oracle_acc": oracle, "log_1_minus_oracle": log_miss})

    valid_points = [(p["k"], p["log_1_minus_oracle"]) for p in fit_points
                    if p["log_1_minus_oracle"] != float("-inf") and p["log_1_minus_oracle"] != 0.0]

    if len(valid_points) < 2:
        return {"p_hat": None, "rho_hat": None, "fit_points": fit_points}

    sum_k = sum(k for k, _ in valid_points)
    sum_log = sum(l for _, l in valid_points)
    sum_kl = sum(k * l for k, l in valid_points)
    sum_k2 = sum(k * k for k, _ in valid_points)
    n = len(valid_points)

    denom = n * sum_k2 - sum_k * sum_k
    if abs(denom) < 1e-12:
        return {"p_hat": None, "rho_hat": None, "fit_points": fit_points}

    slope = (n * sum_kl - sum_k * sum_log) / denom
    p_hat = 1 - math.exp(slope) if slope < 0 else None

    return {"p_hat": p_hat, "slope": slope, "fit_points": fit_points}


def bootstrap_ci(
    per_task_correct: List[bool],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for accuracy (resampling tasks)."""
    import random
    rng = random.Random(seed)
    n = len(per_task_correct)
    if n == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0}

    observed = sum(per_task_correct) / n
    boot_accs = []
    for _ in range(n_boot):
        sample = [per_task_correct[rng.randint(0, n - 1)] for _ in range(n)]
        boot_accs.append(sum(sample) / n)
    boot_accs.sort()
    lo = boot_accs[int(n_boot * alpha / 2)]
    hi = boot_accs[int(n_boot * (1 - alpha / 2))]
    return {"mean": observed, "ci_lo": lo, "ci_hi": hi, "n": n}


def compute_bootstrap_cis(results: dict, candidates: Optional[Dict[str, list]] = None) -> dict:
    """Compute bootstrap CIs for all selectors at each k.

    Requires per-task correctness data from selector_results.
    """
    cis = {}
    for k_str, selectors in results.get("selector_results", {}).items():
        if k_str == "temp_baseline":
            continue
        k_cis = {}
        for sel_name, sel_data in selectors.items():
            if sel_name == "random_mean":
                continue
            per_task = sel_data.get("per_task_correct")
            if per_task is not None:
                k_cis[sel_name] = bootstrap_ci(per_task)
            else:
                acc = sel_data.get("accuracy", 0)
                total = sel_data.get("total", 0)
                if total > 0:
                    per_task_approx = [True] * int(acc * total) + [False] * (total - int(acc * total))
                    k_cis[sel_name] = bootstrap_ci(per_task_approx)
        cis[k_str] = k_cis
    return cis


def compute_selector_recovery(results: dict) -> dict:
    """Compute oracle recovery for each selector at each k."""
    recovery = {}
    for k_str, selectors in results.get("selector_results", {}).items():
        if k_str == "temp_baseline":
            continue
        oracle = selectors.get("oracle", {}).get("accuracy", 0)
        majority = selectors.get("majority", {}).get("accuracy", 0)
        headroom = oracle - majority

        k_recovery = {}
        for sel_name, sel_data in selectors.items():
            if sel_name in ("oracle", "random_mean"):
                continue
            sel_acc = sel_data.get("accuracy", 0)
            if headroom > 0.001:
                rec = (sel_acc - majority) / headroom
            else:
                rec = 0.0
            k_recovery[sel_name] = {
                "accuracy": sel_acc,
                "recovery": rec,
                "regret": oracle - sel_acc,
            }
        recovery[k_str] = {
            "oracle": oracle,
            "majority": majority,
            "headroom": headroom,
            "selectors": k_recovery,
        }
    return recovery


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_selector_results.py <results.json> [candidates.jsonl]")
        sys.exit(1)

    results_path = sys.argv[1]
    results = load_results(results_path)

    print("=" * 70)
    print("SELECTOR STUDY — POST-HOC ANALYSIS")
    print("=" * 70)
    print(f"Model: {results['metadata']['model']}")
    print(f"k={results['metadata']['k']}, n_test={results['metadata']['n_test']}")
    print(f"Geometry: {results['metadata'].get('geometry', 'hyperbolic')}")
    print()

    # Selector recovery
    recovery = compute_selector_recovery(results)
    print("--- Oracle Recovery by k ---")
    for k_str in sorted(recovery.keys(), key=lambda x: int(x)):
        data = recovery[k_str]
        print(f"\nk={k_str}: oracle={data['oracle']:.1%}, majority={data['majority']:.1%}, headroom={data['headroom']:.1%}")
        for sel_name in ["greedy", "plurality_confidence", "consistency_filtered",
                         "scratchpad_majority", "grounded_majority", "composite"]:
            if sel_name in data["selectors"]:
                s = data["selectors"][sel_name]
                print(f"  {sel_name:25s}: acc={s['accuracy']:.1%}, recovery={s['recovery']:+.1%}, regret={s['regret']:.1%}")

    # Bootstrap confidence intervals
    cis = compute_bootstrap_cis(results)
    if cis:
        print("\n--- Bootstrap 95% CIs (10,000 resamples) ---")
        for k_str in sorted(cis.keys(), key=lambda x: int(x)):
            print(f"\nk={k_str}:")
            for sel_name in ["greedy", "oracle", "majority", "plurality_confidence",
                             "consistency_filtered", "composite"]:
                if sel_name in cis[k_str]:
                    ci = cis[k_str][sel_name]
                    print(f"  {sel_name:25s}: {ci['mean']:.1%} [{ci['ci_lo']:.1%}, {ci['ci_hi']:.1%}]")

    # k-scaling analysis (requires candidates JSONL)
    if len(sys.argv) >= 3:
        jsonl_path = sys.argv[2]
        print(f"\n--- k-Scaling Analysis (from {jsonl_path}) ---")
        candidates = load_candidates(jsonl_path)

        k_values = [k for k in [1, 3, 5, 10, 15, 20] if k <= results["metadata"]["k"]]
        scaling = compute_scaling_metrics(candidates, k_values)

        print(f"\n{'k':>4s}  {'Oracle':>8s}  {'Majority':>8s}  {'Oracle-not-Maj':>14s}  {'Eff.Diversity':>13s}")
        for k in k_values:
            m = scaling[k]
            print(f"{k:4d}  {m['oracle_acc']:8.1%}  {m['majority_acc']:8.1%}  {m['oracle_not_majority']:14d}  {m['mean_effective_diversity']:13.3f}")

        # Answer-anywhere analysis (convergence vs computation)
        aa = compute_answer_anywhere(candidates, k_values)
        print(f"\n--- Answer-Anywhere Analysis (Convergence vs Computation) ---")
        print(f"{'k':>4s}  {'Greedy-Last':>11s}  {'Greedy-Any':>10s}  {'Oracle-Last':>11s}  {'Oracle-Any':>10s}")
        for k in k_values:
            m = aa[k]
            print(f"{k:4d}  {m['greedy_last_int']:11.1%}  {m['greedy_answer_anywhere']:10.1%}  {m['oracle_last_int']:11.1%}  {m['oracle_answer_anywhere']:10.1%}")
        max_k = max(k_values)
        m = aa[max_k]
        conv_gap = m["oracle_answer_anywhere"] - m["oracle_last_int"]
        comp_gap = m["oracle_answer_anywhere"] - m["greedy_answer_anywhere"]
        print(f"\nAt k={max_k}:")
        print(f"  Convergence gap (oracle any - oracle last): {conv_gap:+.1%}")
        print(f"  Computation gap (oracle any - greedy any):  {comp_gap:+.1%}")

        # Basin model fit
        basin = fit_basin_model(scaling)
        print(f"\nBasin model fit:")
        if basin["p_hat"] is not None:
            print(f"  Estimated per-candidate hit probability p = {basin['p_hat']:.4f}")
            print(f"  Slope of log(1-oracle) vs k = {basin['slope']:.4f}")
        else:
            print("  Could not fit basin model (insufficient data or oracle=1.0)")

        print(f"\n  Fit points:")
        for fp in basin["fit_points"]:
            print(f"    k={fp['k']:3d}: oracle={fp['oracle_acc']:.1%}, log(1-oracle)={fp['log_1_minus_oracle']:.3f}")

    # Temperature baseline comparison
    if "temp_baseline" in results.get("selector_results", {}):
        tb = results["selector_results"]["temp_baseline"]
        print(f"\n--- Temperature Baseline ---")
        print(f"Temperature majority (k={tb['k']}, temp={tb['temperature']}): {tb['accuracy']:.1%}")

        max_k = str(max(int(k) for k in results["selector_results"] if k.isdigit()))
        pert_majority = results["selector_results"][max_k]["majority"]["accuracy"]
        pert_oracle = results["selector_results"][max_k]["oracle"]["accuracy"]
        print(f"Perturbation majority (k={max_k}): {pert_majority:.1%}")
        print(f"Perturbation oracle (k={max_k}): {pert_oracle:.1%}")
        print(f"Perturbation majority vs temperature: {pert_majority - tb['accuracy']:+.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
