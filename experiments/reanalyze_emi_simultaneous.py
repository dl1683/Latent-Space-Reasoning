"""Reanalyze EMI results with simultaneous max-statistic bootstrap band.

The runner (run_execution_mode_invariance.py) uses independent per-comparison
bootstrap CIs. This script reanalyzes from saved .npz with the Codex-specified
simultaneous correction: for each bootstrap replicate, compute ALL test
statistics across word×mode-pair comparisons, take the max, and use the
(1-α) quantile of the max distribution as the simultaneous threshold.

If independent and simultaneous results agree, no multiplicity issue.
If they disagree on borderline cases, the simultaneous result is canonical.

Usage: python experiments/reanalyze_emi_simultaneous.py
"""
import json
import sys
from pathlib import Path

import numpy as np


VARIABLES = ["x", "y", "z"]
OUTER_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
CORE_WORDS = ["CP", "CPC", "CPCP", "PC", "PCP", "PCPC", "CPCC", "PCPP", "CPCPC"]
FULL_WORDS = ["C", "CC", "P", "PP", "CPP", "PCC"]
MODES = ["L", "W", "G", "F"]
MODE_PAIRS = [("L", "W"), ("W", "G"), ("L", "G"), ("L", "F"), ("W", "F"), ("G", "F")]
EPSILON_TV = 0.06
BOOTSTRAP_N = 100000
BOOTSTRAP_SEED = 42
ALPHA = 0.05


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


def load_dists(npz_path, words):
    """Load distributions from .npz into {word: {mode: array(n_cells, 11)}}."""
    data = np.load(npz_path)
    dists = {}
    for word in words:
        dists[word] = {}
        for mode in MODES:
            key = f"{word}_{mode}"
            if key in data:
                dists[word][mode] = data[key]
    return dists


def build_strata():
    """Return stratum indices for x, y, z variables."""
    cells = [(v, o) for v in VARIABLES for o in OUTER_VALUES]
    strata = {v: [] for v in VARIABLES}
    for i, (v, o) in enumerate(cells):
        strata[v].append(i)
    return strata


def compute_tv_per_cell(da, db, n_cells):
    return [tv(da[i], db[i]) for i in range(n_cells)]


def simultaneous_bootstrap(dists, words, strata, n_boot, seed, alpha=0.05):
    """Simultaneous max-statistic bootstrap band across all comparisons.

    For each bootstrap replicate:
      1. Resample cells within each stratum (same resampling for all comparisons)
      2. Compute stratified mean TV for every (word, mode_pair)
      3. Record the max across all comparisons

    The simultaneous threshold = percentile(1-alpha) of the max distribution.
    """
    rng = np.random.RandomState(seed)
    n_cells = sum(len(idx) for idx in strata.values())
    stratum_list = [strata[v] for v in VARIABLES]

    # Pre-compute per-cell TV for every comparison
    comparisons = []
    observed = {}
    for word in words:
        for m1, m2 in MODE_PAIRS:
            if m1 not in dists[word] or m2 not in dists[word]:
                continue
            da = dists[word][m1]
            db = dists[word][m2]
            tv_per_cell = compute_tv_per_cell(da, db, n_cells)
            comp_key = f"{word}_{m1}v{m2}"
            comparisons.append((comp_key, tv_per_cell))
            strata_vals = [[tv_per_cell[i] for i in s_idx] for s_idx in stratum_list]
            observed[comp_key] = float(np.mean([v for sv in strata_vals for v in sv]))

    n_comp = len(comparisons)
    max_stats = np.empty(n_boot)

    for b in range(n_boot):
        # Same resampling indices for all comparisons (joint resampling)
        boot_indices = {}
        for s_idx in stratum_list:
            n_s = len(s_idx)
            boot_indices[id(s_idx)] = rng.randint(0, n_s, size=n_s)

        boot_max = 0.0
        for comp_key, tv_per_cell in comparisons:
            boot_mean_parts = []
            for s_idx in stratum_list:
                idx = boot_indices[id(s_idx)]
                boot_mean_parts.extend([tv_per_cell[s_idx[i]] for i in idx])
            m = np.mean(boot_mean_parts)
            if m > boot_max:
                boot_max = m
        max_stats[b] = boot_max

    threshold = float(np.percentile(max_stats, 100 * (1 - alpha)))

    # Also compute per-comparison independent CIs for comparison
    independent_cis = {}
    for comp_key, tv_per_cell in comparisons:
        means = np.empty(n_boot)
        rng2 = np.random.RandomState(seed)
        for b in range(n_boot):
            boot_vals = []
            for s_idx in stratum_list:
                n_s = len(s_idx)
                idx = rng2.randint(0, n_s, size=n_s)
                boot_vals.extend([tv_per_cell[s_idx[i]] for i in idx])
            means[b] = np.mean(boot_vals)
        lb = float(np.percentile(means, 100 * alpha / 2))
        ub = float(np.percentile(means, 100 * (1 - alpha / 2)))
        independent_cis[comp_key] = (lb, ub)

    return observed, threshold, max_stats, independent_cis


def gate(observed_tv, eps, threshold_simul=None, ci_indep=None):
    """Gate a comparison.

    Simultaneous: observed <= eps AND below simultaneous threshold → INVARIANT
    Independent: CI upper bound <= eps → INVARIANT
    """
    results = {}
    if threshold_simul is not None:
        if observed_tv <= eps and observed_tv <= threshold_simul:
            results["simultaneous"] = "INVARIANT"
        elif observed_tv > eps:
            results["simultaneous"] = "DIVERGENT"
        else:
            results["simultaneous"] = "UNRESOLVED"

    if ci_indep is not None:
        lb, ub = ci_indep
        if ub <= eps:
            results["independent"] = "INVARIANT"
        elif lb > eps:
            results["independent"] = "DIVERGENT"
        else:
            results["independent"] = "UNRESOLVED"

    return results


def main():
    result_dir = Path("experiments/results/execution_mode_invariance")

    # Try core first, then full
    core_path = result_dir / "emi_core_dists.npz"
    full_path = result_dir / "emi_full_dists.npz"
    runner_result = result_dir / "emi_result.json"

    if not core_path.exists():
        print(f"ERROR: {core_path} not found. EMI must complete first.")
        sys.exit(1)

    # Load runner's independent results for comparison
    runner_eps_mode = None
    if runner_result.exists():
        with open(runner_result) as f:
            runner = json.load(f)
        runner_eps_mode = runner.get("eps_mode", 0.01)
        print(f"Runner eps_mode: {runner_eps_mode}")
    else:
        runner_eps_mode = 0.01
        print(f"Runner result not found yet, using eps_mode={runner_eps_mode}")

    eps_mode = runner_eps_mode
    strata = build_strata()

    # --- Core words ---
    print(f"\n{'='*60}")
    print(f"SIMULTANEOUS MAX-STATISTIC REANALYSIS (eps_mode={eps_mode})")
    print(f"{'='*60}\n")

    words = CORE_WORDS
    npz_path = core_path
    if full_path.exists():
        words = CORE_WORDS + FULL_WORDS
        npz_path = full_path
        print(f"Using full distributions ({len(words)} words)")
    else:
        print(f"Using core distributions ({len(words)} words)")

    dists = load_dists(npz_path, words)
    print(f"Loaded {len(dists)} words, {BOOTSTRAP_N} bootstrap replicates\n")

    observed, threshold, max_stats, indep_cis = simultaneous_bootstrap(
        dists, words, strata, BOOTSTRAP_N, BOOTSTRAP_SEED, ALPHA)

    print(f"Simultaneous threshold (1-alpha={1-ALPHA}): {threshold:.6f}")
    print(f"Max-statistic distribution: median={np.median(max_stats):.6f}, "
          f"p95={np.percentile(max_stats, 95):.6f}, "
          f"p99={np.percentile(max_stats, 99):.6f}\n")

    # Compare independent vs simultaneous for each comparison
    print(f"{'Comparison':<22s} {'Obs TV':>8s} {'Indep CI':>18s} "
          f"{'Indep':>10s} {'Simul':>10s} {'Match':>6s}")
    print("-" * 80)

    disagreements = []
    for comp_key in sorted(observed.keys()):
        obs = observed[comp_key]
        ilb, iub = indep_cis[comp_key]
        gates = gate(obs, eps_mode, threshold, indep_cis[comp_key])
        ig = gates["independent"]
        sg = gates["simultaneous"]
        match = "YES" if ig == sg else "NO"
        if ig != sg:
            disagreements.append(comp_key)
        print(f"  {comp_key:<20s} {obs:8.6f} [{ilb:.6f},{iub:.6f}] "
              f"{ig:>10s} {sg:>10s} {match:>6s}")

    print(f"\n{len(disagreements)} disagreements out of {len(observed)} comparisons")
    if disagreements:
        print(f"Disagreements: {', '.join(disagreements)}")
        print("\nSimultaneous result is canonical per Codex spec.")
    else:
        print("Independent and simultaneous agree — no multiplicity concern.")

    # Save reanalysis
    out = {
        "eps_mode": eps_mode,
        "alpha": ALPHA,
        "n_boot": BOOTSTRAP_N,
        "seed": BOOTSTRAP_SEED,
        "n_comparisons": len(observed),
        "simultaneous_threshold": threshold,
        "max_stat_median": float(np.median(max_stats)),
        "max_stat_p95": float(np.percentile(max_stats, 95)),
        "max_stat_p99": float(np.percentile(max_stats, 99)),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
        "comparisons": {},
    }
    for comp_key in sorted(observed.keys()):
        obs = observed[comp_key]
        ilb, iub = indep_cis[comp_key]
        gates = gate(obs, eps_mode, threshold, indep_cis[comp_key])
        out["comparisons"][comp_key] = {
            "observed_tv": obs,
            "independent_ci": [ilb, iub],
            "independent_gate": gates["independent"],
            "simultaneous_gate": gates["simultaneous"],
        }

    out_path = result_dir / "emi_simultaneous_reanalysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
