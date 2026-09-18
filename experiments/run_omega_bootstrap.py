"""Bootstrap confidence intervals for core omega amplification claims.

Analysis-only — no model forward passes. Reuses observations from the fast sweep
(3,528 readouts) to compute CIs, p-values, and effect sizes for key claims.
"""

import json
import numpy as np
from pathlib import Path


def load_observations(path):
    with open(path) as f:
        return json.load(f)


def tv(a, b):
    return 0.5 * np.abs(np.array(a) - np.array(b)).sum()


def compute_omega_cells(obs_idx, pair, continuations, cells, tau):
    """For a given pair and tau, compute per-cell omega for each continuation.

    Returns dict: continuation_name -> list of (cell_key, baseline_tv, post_tv) for eligible cells.
    """
    surf_a, surf_b, pair_name = pair
    results = {}

    for cont_name in continuations:
        if cont_name == "none":
            continue
        cell_data = []
        for cell in cells:
            depth, var, val = cell
            key_a_base = (depth, var, val, surf_a, "none")
            key_b_base = (depth, var, val, surf_b, "none")

            if key_a_base not in obs_idx or key_b_base not in obs_idx:
                continue

            d_base = tv(obs_idx[key_a_base], obs_idx[key_b_base])
            if d_base > tau:
                continue

            key_a = (depth, var, val, surf_a, cont_name)
            key_b = (depth, var, val, surf_b, cont_name)

            if key_a not in obs_idx or key_b not in obs_idx:
                continue

            d_post = tv(obs_idx[key_a], obs_idx[key_b])
            cell_data.append((cell, d_base, d_post))

        results[cont_name] = cell_data

    return results


def bootstrap_mean(values, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, (0.0, 0.0)
    arr = np.array(values)
    means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(arr.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def bootstrap_diff(values_a, values_b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n_a, n_b = len(values_a), len(values_b)
    if n_a == 0 or n_b == 0:
        return 0.0, (0.0, 0.0), 1.0
    arr_a, arr_b = np.array(values_a), np.array(values_b)
    obs_diff = arr_a.mean() - arr_b.mean()
    diffs = []
    for _ in range(n_boot):
        ma = rng.choice(arr_a, size=n_a, replace=True).mean()
        mb = rng.choice(arr_b, size=n_b, replace=True).mean()
        diffs.append(ma - mb)
    diffs = np.array(diffs)
    ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    p_val = float((diffs <= 0).mean())
    return float(obs_diff), ci, p_val


def permutation_test(values_a, values_b, n_perm=5000, seed=42):
    rng = np.random.default_rng(seed)
    arr_a, arr_b = np.array(values_a), np.array(values_b)
    obs_diff = arr_a.mean() - arr_b.mean()
    combined = np.concatenate([arr_a, arr_b])
    n_a = len(arr_a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_diff = combined[:n_a].mean() - combined[n_a:].mean()
        if perm_diff >= obs_diff:
            count += 1
    return float(obs_diff), float(count / n_perm)


def cohens_d(values_a, values_b):
    a, b = np.array(values_a), np.array(values_b)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float('nan')
    pooled_std = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2))
    if pooled_std == 0:
        return float('inf') if a.mean() != b.mean() else 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def main():
    print("=== Omega Bootstrap Analysis ===\n")

    obs = load_observations("experiments/results/omega_sweep_fast/observations.json")
    print(f"Loaded {len(obs)} observations")

    obs_idx = {}
    for o in obs:
        key = (o["depth"], o["var"], o["val"], o["surface"], o["continuation"])
        obs_idx[key] = np.array(o["dist"], dtype=np.float64)

    cells = [(4, "x", 1), (4, "x", 5), (4, "y", 2), (4, "y", 6), (4, "z", 3), (4, "z", 7)]

    equiv_pairs = [
        ("assert_novar_1", "assert_var_1", "strong_assert"),
        ("assert_novar_2", "assert_var_2", "strong_assert_2"),
        ("assert_novar_4", "assert_var_4", "moderate_assert"),
        ("mislead_var_1", "mislead_novar_1", "misleading"),
        ("mislead_var_4", "mislead_novar_3", "misleading_2"),
        ("rewrite_1", "rewrite_4", "rewrite_pair"),
        ("boundary_1", "boundary_2", "boundary_pair"),
        ("observe_1", "observe_2", "observe_pair"),
    ]

    conts_of_interest = ["neutral_1", "neutral_2", "heavy_1", "heavy_2",
                         "identity", "type_check", "pass", "observe",
                         "bare_nl", "comment_var", "print_var"]

    for tau in [0.01, 0.02, 0.05]:
        print(f"\n{'='*70}")
        print(f"  TAU = {tau}")
        print(f"{'='*70}")

        all_cont_tvs = {}

        for pair in equiv_pairs:
            pair_name = pair[2]
            omega_cells = compute_omega_cells(obs_idx, pair, conts_of_interest + ["none"], cells, tau)

            for cont_name, cell_data in omega_cells.items():
                if cont_name not in all_cont_tvs:
                    all_cont_tvs[cont_name] = []
                for cell, d_base, d_post in cell_data:
                    all_cont_tvs[cont_name].append(d_post)

        print(f"\n--- Per-continuation bootstrap (pooled across all pairs) ---")
        print(f"{'Continuation':<18s} {'N':>4s} {'Mean':>8s} {'95% CI':>20s} {'P(mean<=tau)':>14s} {'Status':>10s}")

        cont_stats = {}
        for cont_name in sorted(all_cont_tvs.keys()):
            tvs = all_cont_tvs[cont_name]
            if not tvs:
                continue
            mean_tv, ci = bootstrap_mean(tvs, n_boot=2000)
            n_below = sum(1 for v in tvs if v <= tau)
            rng = np.random.default_rng(42)
            arr = np.array(tvs)
            n = len(arr)
            boot_means = [rng.choice(arr, size=n, replace=True).mean() for _ in range(2000)]
            p_preserve = sum(1 for m in boot_means if m <= tau) / len(boot_means)

            status = "PRESERVE" if mean_tv <= tau else "AMPLIFY"
            print(f"  {cont_name:<16s} {len(tvs):4d} {mean_tv:8.5f} [{ci[0]:.5f}, {ci[1]:.5f}] {p_preserve:14.4f} {status:>10s}")
            cont_stats[cont_name] = {"tvs": tvs, "mean": mean_tv, "ci": ci, "p_preserve": p_preserve}

        print(f"\n--- Key comparisons ---")

        if "identity" in cont_stats and "neutral_1" in cont_stats:
            id_tvs = cont_stats["identity"]["tvs"]
            n1_tvs = cont_stats["neutral_1"]["tvs"]
            diff, ci, p_val = bootstrap_diff(id_tvs, n1_tvs)
            d = cohens_d(id_tvs, n1_tvs)
            print(f"  Identity vs Neutral_1: diff={diff:.5f}, 95% CI=[{ci[0]:.5f}, {ci[1]:.5f}], "
                  f"P(diff<=0)={p_val:.4f}, Cohen's d={d:.3f}")

        if "identity" in cont_stats and "neutral_2" in cont_stats:
            id_tvs = cont_stats["identity"]["tvs"]
            n2_tvs = cont_stats["neutral_2"]["tvs"]
            diff, ci, p_val = bootstrap_diff(id_tvs, n2_tvs)
            d = cohens_d(id_tvs, n2_tvs)
            print(f"  Identity vs Neutral_2: diff={diff:.5f}, 95% CI=[{ci[0]:.5f}, {ci[1]:.5f}], "
                  f"P(diff<=0)={p_val:.4f}, Cohen's d={d:.3f}")

        if "type_check" in cont_stats and "neutral_1" in cont_stats:
            tc_tvs = cont_stats["type_check"]["tvs"]
            n1_tvs = cont_stats["neutral_1"]["tvs"]
            diff, ci, p_val = bootstrap_diff(tc_tvs, n1_tvs)
            d = cohens_d(tc_tvs, n1_tvs)
            print(f"  Type_check vs Neutral_1: diff={diff:.5f}, 95% CI=[{ci[0]:.5f}, {ci[1]:.5f}], "
                  f"P(diff<=0)={p_val:.4f}, Cohen's d={d:.3f}")

        if "identity" in cont_stats and "pass" in cont_stats:
            id_tvs = cont_stats["identity"]["tvs"]
            pass_tvs = cont_stats["pass"]["tvs"]
            diff, ci, p_val = bootstrap_diff(id_tvs, pass_tvs)
            d = cohens_d(id_tvs, pass_tvs)
            print(f"  Identity vs Pass: diff={diff:.5f}, 95% CI=[{ci[0]:.5f}, {ci[1]:.5f}], "
                  f"P(diff<=0)={p_val:.4f}, Cohen's d={d:.3f}")

        print(f"\n--- Permutation tests ---")

        if "identity" in cont_stats and "neutral_1" in cont_stats:
            id_tvs = cont_stats["identity"]["tvs"]
            n1_tvs = cont_stats["neutral_1"]["tvs"]
            diff, p_val = permutation_test(id_tvs, n1_tvs)
            print(f"  Identity vs Neutral_1 (permutation): diff={diff:.5f}, p={p_val:.4f}")

        if "identity" in cont_stats and "neutral_2" in cont_stats:
            id_tvs = cont_stats["identity"]["tvs"]
            n2_tvs = cont_stats["neutral_2"]["tvs"]
            diff, p_val = permutation_test(id_tvs, n2_tvs)
            print(f"  Identity vs Neutral_2 (permutation): diff={diff:.5f}, p={p_val:.4f}")

        amplifiers = [c for c, s in cont_stats.items() if s["mean"] > tau]
        preservers = [c for c, s in cont_stats.items() if s["mean"] <= tau]
        print(f"\n  Amplifiers ({len(amplifiers)}): {', '.join(sorted(amplifiers))}")
        print(f"  Preservers ({len(preservers)}): {', '.join(sorted(preservers))}")

    result_dir = Path("experiments/results/omega_bootstrap")
    result_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for tau in [0.01, 0.02, 0.05]:
        all_cont_tvs = {}
        for pair in equiv_pairs:
            omega_cells = compute_omega_cells(obs_idx, pair, conts_of_interest + ["none"], cells, tau)
            for cont_name, cell_data in omega_cells.items():
                if cont_name not in all_cont_tvs:
                    all_cont_tvs[cont_name] = []
                for cell, d_base, d_post in cell_data:
                    all_cont_tvs[cont_name].append(d_post)

        tau_summary = {}
        for cont_name, tvs in all_cont_tvs.items():
            if not tvs:
                continue
            mean_tv, ci = bootstrap_mean(tvs)
            tau_summary[cont_name] = {
                "n": len(tvs),
                "mean": mean_tv,
                "ci_95": list(ci),
                "amplifies": mean_tv > tau,
            }
        summary[str(tau)] = tau_summary

    with open(result_dir / "bootstrap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {result_dir}/bootstrap_summary.json")


if __name__ == "__main__":
    main()
