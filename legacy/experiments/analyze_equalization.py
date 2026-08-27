"""Analyze solve-count equalization and oracle coverage from sensitivity results.

Runs Monte Carlo test against heterogeneous iid Bernoulli null, computes oracle
coverage curves, and compares across token counts.

Usage:
    python experiments/analyze_equalization.py experiments/sensitivity_sweet_spot_random_noise_t2_results.json
    python experiments/analyze_equalization.py --compare experiments/*_t2_*.json experiments/*_t3_*.json
"""

import json
import sys
import numpy as np
from pathlib import Path
from itertools import combinations


def load_results(path: Path) -> dict:
    data = json.loads(path.read_text())
    sr = data.get("sensitivity_results", data.get("noise_results", []))
    n_latents = len(sr)
    n_tasks = data.get("n_tasks", 25)

    # Build binary matrix: latents x tasks (if per-task data available)
    has_task_data = sr and "task_results" in sr[0]
    if has_task_data:
        matrix = np.zeros((n_latents, n_tasks), dtype=int)
        for i, latent in enumerate(sr):
            for j, tr in enumerate(latent["task_results"]):
                matrix[i, j] = int(tr["correct"])
        solve_counts = matrix.sum(axis=1)
        task_solve_rates = matrix.mean(axis=0)
    else:
        # Old format: only aggregate n_correct
        matrix = None
        solve_counts = np.array([s.get("n_correct", 0) for s in sr])
        task_solve_rates = None

    return {
        "path": str(path),
        "num_soft_tokens": data.get("num_soft_tokens"),
        "n_latents": n_latents,
        "n_tasks": n_tasks,
        "baseline_accuracy": data.get("baseline_accuracy"),
        "matrix": matrix,
        "solve_counts": solve_counts,
        "task_solve_rates": task_solve_rates,
        "has_task_data": has_task_data,
    }


def monte_carlo_equalization(solve_counts: np.ndarray, task_solve_rates: np.ndarray,
                              n_sim: int = 100_000, seed: int = 42) -> dict:
    """Test whether observed SD of solve counts is lower than heterogeneous iid null."""
    rng = np.random.default_rng(seed)
    n_latents = len(solve_counts)
    n_tasks = len(task_solve_rates)
    observed_sd = float(np.std(solve_counts, ddof=0))

    sim_sds = np.empty(n_sim)
    for i in range(n_sim):
        sim_matrix = rng.random((n_latents, n_tasks)) < task_solve_rates[None, :]
        sim_counts = sim_matrix.sum(axis=1)
        sim_sds[i] = np.std(sim_counts, ddof=0)

    p_value = float(np.mean(sim_sds <= observed_sd))
    expected_sd = float(np.median(sim_sds))

    return {
        "observed_sd": observed_sd,
        "expected_sd_median": expected_sd,
        "expected_sd_mean": float(np.mean(sim_sds)),
        "ratio": observed_sd / expected_sd if expected_sd > 0 else float("inf"),
        "p_value": p_value,
        "n_sim": n_sim,
        "percentile_5": float(np.percentile(sim_sds, 5)),
        "percentile_95": float(np.percentile(sim_sds, 95)),
    }


def oracle_coverage(matrix: np.ndarray) -> dict:
    """Compute oracle coverage at various k values."""
    n_latents, n_tasks = matrix.shape
    results = {}

    # Full oracle
    any_solved = matrix.max(axis=0)
    results["full_oracle"] = int(any_solved.sum())

    # Oracle at each k (average over all k-combinations)
    for k in range(1, min(n_latents + 1, 11)):
        if k > n_latents:
            break
        coverages = []
        combos = list(combinations(range(n_latents), k))
        if len(combos) > 1000:
            # Sample for large k
            rng = np.random.default_rng(42)
            indices = rng.choice(len(combos), 1000, replace=False)
            combos = [combos[i] for i in indices]
        for combo in combos:
            sub = matrix[list(combo), :]
            coverages.append(int(sub.max(axis=0).sum()))
        results[f"k={k}"] = {
            "mean": float(np.mean(coverages)),
            "min": int(np.min(coverages)),
            "max": int(np.max(coverages)),
            "n_combos": len(combos),
        }

    return results


def frozen_tasks(matrix: np.ndarray) -> dict:
    """Identify tasks that are never solved (frozen) across all latents."""
    never_solved = np.where(matrix.max(axis=0) == 0)[0]
    always_solved = np.where(matrix.min(axis=0) == 1)[0]
    return {
        "frozen": never_solved.tolist(),
        "n_frozen": len(never_solved),
        "always_solved": always_solved.tolist(),
        "n_always_solved": len(always_solved),
    }


def analyze_file(path: Path) -> dict:
    data = load_results(path)
    if data["has_task_data"]:
        mc = monte_carlo_equalization(data["solve_counts"], data["task_solve_rates"])
        oracle = oracle_coverage(data["matrix"])
        frozen = frozen_tasks(data["matrix"])
    else:
        mc = {"observed_sd": float(np.std(data["solve_counts"], ddof=0)),
              "expected_sd_median": None, "expected_sd_mean": None,
              "ratio": None, "p_value": None, "n_sim": 0,
              "percentile_5": None, "percentile_95": None}
        oracle = {"full_oracle": None}
        frozen = {"frozen": [], "n_frozen": None, "always_solved": [], "n_always_solved": None}

    return {
        "file": data["path"],
        "num_soft_tokens": data["num_soft_tokens"],
        "n_latents": data["n_latents"],
        "baseline_accuracy": data["baseline_accuracy"],
        "solve_counts": data["solve_counts"].tolist(),
        "mean_accuracy": float(data["solve_counts"].mean() / data["n_tasks"]),
        "equalization": mc,
        "oracle": oracle,
        "frozen": frozen,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_equalization.py <results.json> [...]")
        sys.exit(1)

    compare = "--compare" in sys.argv
    paths = [Path(a) for a in sys.argv[1:] if a != "--compare" and Path(a).exists()]

    for path in sorted(paths):
        result = analyze_file(path)
        t = result["num_soft_tokens"]
        n = result["n_latents"]
        sc = result["solve_counts"]
        mc = result["equalization"]

        print(f"\n{'='*60}")
        print(f"  {path.name}  (t={t}, n={n})")
        print(f"{'='*60}")
        print(f"  Baseline: {result['baseline_accuracy']:.0%}")
        print(f"  Mean accuracy: {result['mean_accuracy']:.1%}")
        print(f"  Solve counts: {sc}")
        print(f"  SD: {mc['observed_sd']:.2f}", end="")
        if mc['expected_sd_median'] is not None:
            print(f"  (expected: {mc['expected_sd_median']:.2f})")
            print(f"  Ratio (obs/exp): {mc['ratio']:.3f}")
            print(f"  p-value: {mc['p_value']:.4f}  (100k MC sims)")
            print(f"  95% CI of null SD: [{mc['percentile_5']:.2f}, {mc['percentile_95']:.2f}]")
        else:
            print("  (no per-task data for MC test)")
        print()
        if result["oracle"]["full_oracle"] is not None:
            print(f"  Oracle coverage:")
            for key, val in result["oracle"].items():
                if key == "full_oracle":
                    print(f"    Full oracle: {val}/{25}")
                elif isinstance(val, dict):
                    print(f"    {key}: {val['mean']:.1f}/25 (range {val['min']}-{val['max']})")
            print()
            frozen = result["frozen"]
            print(f"  Frozen tasks: {frozen['n_frozen']} {frozen['frozen']}")
            print(f"  Always solved: {frozen['n_always_solved']} {frozen['always_solved']}")
        else:
            print("  (no per-task data for oracle/frozen analysis)")

    if compare and len(paths) > 1:
        print(f"\n{'='*60}")
        print("  COMPARISON")
        print(f"{'='*60}")
        for path in sorted(paths):
            r = analyze_file(path)
            mc = r["equalization"]
            orc = r["oracle"]
            k3 = orc.get("k=3", {})
            p_str = f"{mc['p_value']:.3f}" if mc['p_value'] is not None else "N/A"
            k3_str = f"{k3['mean']:.0f}" if isinstance(k3, dict) and 'mean' in k3 else "N/A"
            full_str = str(orc['full_oracle']) if orc['full_oracle'] is not None else "N/A"
            t_str = str(r['num_soft_tokens']) if r['num_soft_tokens'] is not None else "?"
            print(f"  t={t_str:>2} n={r['n_latents']:>2}: "
                  f"SD={mc['observed_sd']:.2f} p={p_str} "
                  f"oracle(k=3)={k3_str}/25 "
                  f"full={full_str}/25")


if __name__ == "__main__":
    main()
