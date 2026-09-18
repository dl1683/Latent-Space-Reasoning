"""Omega Baseline Predictor (Experiment #46).

Can baseline distance properties predict omega sensitivity?

The finding from exps 38-43 is that omega is highly pair-dependent.
This experiment asks: given ONLY the baseline readout distributions
of a surface pair (no continuations), can we predict whether that
pair will be amplification-sensitive?

Features extracted from baseline:
  1. Baseline TV between pair members
  2. Entropy of each pair member's distribution
  3. Peak probability (max bin value)
  4. Distribution shape (how many bins have >1% probability)

Target: max omega across all tested continuations

Uses existing data from exps #38 (partial) and #43 (authority).
ANALYSIS ONLY — no new model runs.
"""

import json
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from omega import tv


def load_observations(result_dir):
    with open(Path(result_dir) / "observations.json") as f:
        obs = json.load(f)
    idx = {}
    for o in obs:
        key = (o["depth"], o["var"], o["val"], o["surface"], o["continuation"])
        idx[key] = np.array(o["dist"], dtype=np.float64)
    return idx


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def extract_baseline_features(dist_a, dist_b):
    """Extract features from a pair of baseline distributions."""
    tv_val = tv(dist_a, dist_b)
    ent_a = entropy(dist_a)
    ent_b = entropy(dist_b)
    peak_a = float(np.max(dist_a))
    peak_b = float(np.max(dist_b))
    active_a = int(np.sum(dist_a > 0.01))
    active_b = int(np.sum(dist_b > 0.01))
    kl_ab = float(np.sum(dist_a * np.log2(
        np.clip(dist_a, 1e-10, 1) / np.clip(dist_b, 1e-10, 1))))
    kl_ba = float(np.sum(dist_b * np.log2(
        np.clip(dist_b, 1e-10, 1) / np.clip(dist_a, 1e-10, 1))))

    return {
        "tv": tv_val,
        "entropy_a": ent_a,
        "entropy_b": ent_b,
        "entropy_mean": (ent_a + ent_b) / 2,
        "entropy_diff": abs(ent_a - ent_b),
        "peak_a": peak_a,
        "peak_b": peak_b,
        "peak_mean": (peak_a + peak_b) / 2,
        "active_bins_a": active_a,
        "active_bins_b": active_b,
        "kl_ab": kl_ab,
        "kl_ba": kl_ba,
        "kl_sym": (kl_ab + kl_ba) / 2,
    }


EQUIV_PAIRS = [
    ("assert_novar_1", "assert_var_1", "strong_assert"),
    ("assert_novar_2", "assert_var_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CONTS_PARTIAL = [
    "baseline_none", "bare_newline", "pass_stmt", "var_only",
    "var_equals", "identity_no_nl", "identity_full", "dangling_assign",
    "no_lhs", "cross_var_assign", "literal_assign", "none_assign",
    "comment_var", "type_check", "observe",
]

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]


def run_baseline_predictor():
    print("=== Omega Baseline Predictor (Exp #46) ===")
    print("Analysis-only: uses existing data from exp #38\n")

    obs_idx = load_observations("experiments/results/omega_partial")
    print(f"Loaded {len(obs_idx)} observations\n")

    summary = {"type": "analysis_only", "source": "exp_38", "cells": []}

    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        print(f"\n{'='*50}")
        print(f"  Pair: {pair_name}")
        print(f"{'='*50}")

        for depth, var, val in CELLS:
            key_a_base = (depth, var, val, surf_a, "baseline_none")
            key_b_base = (depth, var, val, surf_b, "baseline_none")

            if key_a_base not in obs_idx or key_b_base not in obs_idx:
                continue

            dist_a = obs_idx[key_a_base]
            dist_b = obs_idx[key_b_base]
            features = extract_baseline_features(dist_a, dist_b)

            # Compute omega for all continuations on this cell
            cell_omegas = {}
            for cont in CONTS_PARTIAL:
                if cont == "baseline_none":
                    continue
                key_a = (depth, var, val, surf_a, cont)
                key_b = (depth, var, val, surf_b, cont)
                if key_a in obs_idx and key_b in obs_idx:
                    cell_omegas[cont] = tv(obs_idx[key_a], obs_idx[key_b])

            if not cell_omegas:
                continue

            max_omega = max(cell_omegas.values())
            mean_omega = np.mean(list(cell_omegas.values()))
            n_amplifying_001 = sum(1 for v in cell_omegas.values() if v > 0.01)
            n_amplifying_005 = sum(1 for v in cell_omegas.values() if v > 0.05)
            top_cont = max(cell_omegas, key=cell_omegas.get)

            cell_data = {
                "pair": pair_name,
                "cell": f"d{depth}_{var}_{val}",
                "features": features,
                "max_omega": float(max_omega),
                "mean_omega": float(mean_omega),
                "n_amplifying_001": n_amplifying_001,
                "n_amplifying_005": n_amplifying_005,
                "top_continuation": top_cont,
            }
            summary["cells"].append(cell_data)

            print(f"\n  Cell d{depth}_{var}_{val}:")
            print(f"    Baseline TV: {features['tv']:.6f}")
            print(f"    Entropy mean: {features['entropy_mean']:.3f}")
            print(f"    Peak mean: {features['peak_mean']:.3f}")
            print(f"    Active bins: {features['active_bins_a']}, "
                  f"{features['active_bins_b']}")
            print(f"    Max omega: {max_omega:.5f} ({top_cont})")
            print(f"    Amplifying at 0.01: {n_amplifying_001}/14, "
                  f"at 0.05: {n_amplifying_005}/14")

    # Cross-cell correlation analysis
    print(f"\n\n{'='*50}")
    print(f"  Cross-cell correlation: baseline TV vs max omega")
    print(f"{'='*50}\n")

    cells_data = summary["cells"]
    if len(cells_data) >= 3:
        tvs = [c["features"]["tv"] for c in cells_data]
        max_omegas = [c["max_omega"] for c in cells_data]
        ent_means = [c["features"]["entropy_mean"] for c in cells_data]
        peak_means = [c["features"]["peak_mean"] for c in cells_data]
        kl_syms = [c["features"]["kl_sym"] for c in cells_data]

        def spearman(a, b):
            n = len(a)
            if n < 3:
                return 0.0
            a_r = np.argsort(np.argsort(a)).astype(float)
            b_r = np.argsort(np.argsort(b)).astype(float)
            d = a_r - b_r
            return float(1.0 - 6 * np.sum(d**2) / (n * (n**2 - 1)))

        correlations = {
            "baseline_tv_vs_max_omega": spearman(tvs, max_omegas),
            "entropy_vs_max_omega": spearman(ent_means, max_omegas),
            "peak_vs_max_omega": spearman(peak_means, max_omegas),
            "kl_sym_vs_max_omega": spearman(kl_syms, max_omegas),
        }
        summary["correlations"] = correlations

        for name, rho in correlations.items():
            print(f"  {name:35s}: rho = {rho:.3f}")

        # Per-pair breakdown
        print(f"\n  Per-pair breakdown:")
        for pair_name in set(c["pair"] for c in cells_data):
            pc = [c for c in cells_data if c["pair"] == pair_name]
            if len(pc) >= 3:
                ptv = [c["features"]["tv"] for c in pc]
                pmo = [c["max_omega"] for c in pc]
                rho = spearman(ptv, pmo)
                print(f"    {pair_name:22s}: "
                      f"TV range=[{min(ptv):.5f}, {max(ptv):.5f}], "
                      f"omega range=[{min(pmo):.5f}, {max(pmo):.5f}], "
                      f"rho={rho:.3f}")

    result_dir = Path("experiments/results/omega_baseline_predictor")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {result_dir}/")
    return summary


if __name__ == "__main__":
    run_baseline_predictor()
