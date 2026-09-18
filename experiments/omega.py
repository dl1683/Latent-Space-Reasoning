"""Omega modulus computation for behavioral distance transfer tests.

Implements the CR-15.10.1 omega modulus on finite panels of 11-bin
readout distributions from SVB experiments.

    omega_a^W(tau) = sup { d_W(T_a x, T_a y) : d_W(x,y) <= tau }

where d_W(x,y) = max over (w,c) in W of TV(readout(x,w), readout(y,w)).
"""

import json
import numpy as np
from itertools import combinations
from pathlib import Path


def tv(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()


def cosine_sim(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_d_W(dists_x, dists_y, W_keys):
    """Behavioral distance: max TV over question family W.

    dists_x, dists_y: dicts mapping action_key -> 11-bin array
    W_keys: list of action keys defining the question family

    Returns (max_tv, coverage) where coverage = fraction of W_keys present in both.
    """
    max_tv = 0.0
    covered = 0
    for k in W_keys:
        if k in dists_x and k in dists_y:
            d = tv(dists_x[k], dists_y[k])
            max_tv = max(max_tv, d)
            covered += 1
    return max_tv


def compute_omega_panel(state_pairs, action_key, W_keys, tau):
    """Compute omega_a^W(tau) on a finite panel of state pairs.

    state_pairs: list of (dists_x, dists_y, label) where each dists_*
                 maps action_key -> 11-bin array
    action_key:  the action a whose omega we're computing
    W_keys:      question family keys for d_W computation
    tau:         tolerance threshold

    Returns dict with:
      eligible_pairs: list of (label, d_pre, d_post) for pairs with d_pre <= tau
      panel_modulus:  max d_post over eligible pairs (= omega estimate)
      n_eligible:     count of eligible pairs
      n_total:        total pairs checked
    """
    eligible = []
    for dists_x, dists_y, label in state_pairs:
        d_pre = compute_d_W(dists_x, dists_y, W_keys)
        if d_pre <= tau:
            dists_x_post = {k: dists_x.get(f"{action_key}_{k}", dists_x.get(action_key))
                            for k in W_keys}
            dists_y_post = {k: dists_y.get(f"{action_key}_{k}", dists_y.get(action_key))
                            for k in W_keys}
            d_post = compute_d_W(dists_x_post, dists_y_post, W_keys)
            eligible.append((label, float(d_pre), float(d_post)))

    if not eligible:
        return {
            "eligible_pairs": [],
            "panel_modulus": None,
            "n_eligible": 0,
            "n_total": len(state_pairs),
            "tau": tau,
            "action": action_key,
            "preserves": None,
            "assessable": False,
        }

    panel_mod = max(e[2] for e in eligible)
    return {
        "eligible_pairs": eligible,
        "panel_modulus": panel_mod,
        "n_eligible": len(eligible),
        "n_total": len(state_pairs),
        "tau": tau,
        "action": action_key,
        "preserves": panel_mod <= tau,
        "assessable": True,
    }


def evaluate_transfer(omega_train, d_test_pairs, tau):
    """Evaluate whether omega from W_train predicts d_W_test outcomes.

    omega_train: dict from compute_omega_panel (under W_train)
    d_test_pairs: list of (label, d_W_test_post) for the same eligible pairs
    tau: tolerance

    Returns prediction accuracy and comparison stats.
    """
    if not omega_train["eligible_pairs"]:
        return {"error": "no eligible pairs", "n": 0}

    predicted_preserves = omega_train["preserves"]
    actual_preserved = sum(1 for _, d in d_test_pairs if d <= tau)
    actual_separated = sum(1 for _, d in d_test_pairs if d > tau)
    n = len(d_test_pairs)

    return {
        "omega_prediction": "all_preserve" if predicted_preserves else "at_least_one_separates",
        "omega_value": omega_train["panel_modulus"],
        "actual_all_preserved": actual_separated == 0,
        "actual_preserved": actual_preserved,
        "actual_separated": actual_separated,
        "preservation_rate": actual_preserved / n if n > 0 else 0.0,
        "n": n,
        "tau": tau,
    }


def load_congruence_data(path):
    """Load congruence experiment results into state-pair format.

    Returns:
      cells: dict mapping (var, val) -> dict of arm -> 11-bin dist
      equiv_pairs: list of pair definitions
      composers: list of composer strings
    """
    with open(path) as f:
        data = json.load(f)

    cells = {}
    for key, obs in data["observations"].items():
        cell_id = (obs["var"], obs["val"])
        if cell_id not in cells:
            cells[cell_id] = {}
        cells[cell_id][obs["arm"]] = np.array(obs["dist"], dtype=np.float64)

    return {
        "cells": cells,
        "equiv_pairs": data["equiv_pairs"],
        "composers": data["composers"],
    }


def build_congruence_state_pairs(congruence_data, pair_name):
    """Build state pairs from congruence data for a given equiv_pair.

    Each pair compares surface a vs a' for the same cell.
    Returns list of (dists_a, dists_a', label) where each dists maps
    arm_suffix -> 11-bin array.
    """
    cells = congruence_data["cells"]
    pairs = []

    for (var, val), arms in cells.items():
        dists_a = {}
        dists_a_prime = {}

        prefix_a = f"{pair_name}_a"
        prefix_ap = f"{pair_name}_a_prime"

        for arm_key, dist in arms.items():
            if arm_key == "baseline":
                dists_a["baseline"] = dist
                dists_a_prime["baseline"] = dist
            elif arm_key == prefix_ap:
                dists_a_prime["immediate"] = dist
            elif arm_key.startswith(prefix_ap + "_"):
                suffix = arm_key[len(prefix_ap) + 1:]
                dists_a_prime[suffix] = dist
            elif arm_key == prefix_a:
                dists_a["immediate"] = dist
            elif arm_key.startswith(prefix_a + "_"):
                suffix = arm_key[len(prefix_a) + 1:]
                dists_a[suffix] = dist

        if "immediate" in dists_a and "immediate" in dists_a_prime:
            label = f"{var}_{val}_{pair_name}"
            pairs.append((dists_a, dists_a_prime, label))

    return pairs


def run_congruence_omega_test(congruence_path, tau_values=None):
    """Run the full omega transfer test on congruence data.

    W_train = {immediate readout after surface}
    W_test  = {readout after surface + c0, readout after surface + c1}

    Tests whether omega_c0 and omega_c1 under W_train predict
    preservation/separation under W_test.
    """
    if tau_values is None:
        tau_values = [0.005, 0.01, 0.02, 0.05, 0.10]

    data = load_congruence_data(congruence_path)
    results = {}

    for pair_info in data["equiv_pairs"]:
        pair_name = pair_info["name"]
        pairs = build_congruence_state_pairs(data, pair_name)

        if not pairs:
            results[pair_name] = {"error": "no pairs built"}
            continue

        pair_result = {"n_cells": len(pairs), "tau_results": {}}

        for tau in tau_values:
            W_train = ["immediate"]

            d_train_values = []
            for dists_a, dists_ap, label in pairs:
                d = compute_d_W(dists_a, dists_ap, W_train)
                d_train_values.append((label, d))

            eligible_labels = [l for l, d in d_train_values if d <= tau]
            n_eligible = len(eligible_labels)

            c0_post = []
            c1_post = []
            cosine_sims = []

            for dists_a, dists_ap, label in pairs:
                d_pre = compute_d_W(dists_a, dists_ap, W_train)
                if d_pre > tau:
                    continue

                d_c0 = tv(dists_a.get("c0", np.zeros(11)),
                          dists_ap.get("c0", np.zeros(11)))
                d_c1 = tv(dists_a.get("c1", np.zeros(11)),
                          dists_ap.get("c1", np.zeros(11)))
                c0_post.append((label, float(d_c0)))
                c1_post.append((label, float(d_c1)))

                cos = cosine_sim(dists_a["immediate"], dists_ap["immediate"])
                cosine_sims.append((label, cos))

            omega_c0 = max((d for _, d in c0_post), default=0.0)
            omega_c1 = max((d for _, d in c1_post), default=0.0)

            c0_preserved = sum(1 for _, d in c0_post if d <= tau)
            c1_preserved = sum(1 for _, d in c1_post if d <= tau)
            n = len(c0_post)

            tau_result = {
                "n_eligible": n_eligible,
                "omega_c0": float(omega_c0),
                "omega_c1": float(omega_c1),
                "c0_preserves_prediction": omega_c0 <= tau,
                "c1_preserves_prediction": omega_c1 <= tau,
                "c0_actual_preserved": c0_preserved,
                "c0_actual_separated": n - c0_preserved,
                "c1_actual_preserved": c1_preserved,
                "c1_actual_separated": n - c1_preserved,
                "c0_preservation_rate": c0_preserved / n if n > 0 else None,
                "c1_preservation_rate": c1_preserved / n if n > 0 else None,
                "cosine_mean": float(np.mean([c for _, c in cosine_sims])) if cosine_sims else None,
                "cosine_min": float(np.min([c for _, c in cosine_sims])) if cosine_sims else None,
                "d_train_stats": {
                    "mean": float(np.mean([d for _, d in d_train_values])),
                    "median": float(np.median([d for _, d in d_train_values])),
                    "min": float(np.min([d for _, d in d_train_values])),
                    "max": float(np.max([d for _, d in d_train_values])),
                },
                "c0_post_stats": {
                    "mean": float(np.mean([d for _, d in c0_post])) if c0_post else None,
                    "max": float(np.max([d for _, d in c0_post])) if c0_post else None,
                },
                "c1_post_stats": {
                    "mean": float(np.mean([d for _, d in c1_post])) if c1_post else None,
                    "max": float(np.max([d for _, d in c1_post])) if c1_post else None,
                },
            }
            pair_result["tau_results"][str(tau)] = tau_result

        results[pair_name] = pair_result

    return results


def load_gate1b_data(path):
    """Load gate1b experiment into cell-indexed structure.

    Returns dict mapping (depth, var, val) -> dict of surface_key -> 11-bin dist.
    surface_key = "{role}_{split}_{idx}" (e.g. "ASSERT_train_0").
    """
    with open(path) as f:
        data = json.load(f)

    cells = {}
    for key, obs in data["observations"].items():
        cell_id = (obs["depth"], obs["var"], obs["val"])
        if cell_id not in cells:
            cells[cell_id] = {}
        surface_key = key.split("_", 3)[3]  # strip "d{depth}_{var}_{val}_"
        cells[cell_id][surface_key] = np.array(obs["dist"], dtype=np.float64)

    roles = data["config"]["intensional_roles"]
    return {"cells": cells, "roles": roles}


def run_gate1b_omega_test(gate1b_path, tau_values=None):
    """Run omega transfer test on gate1b data.

    W_train = train surfaces of {ASSERT, ASSERT_VAR, BOUNDARY, BASELINE}
    W_test  = holdout surfaces of {MISLEADING_ASSERT, MISLEADING_ASSERT_NOVAR, REWRITE, OBSERVE}

    For each same-depth cell pair, compute d_W_train and d_W_test.
    Test: does d_W_train ≤ tau predict d_W_test ≤ tau?
    """
    if tau_values is None:
        tau_values = [0.01, 0.02, 0.05, 0.10, 0.20]

    data = load_gate1b_data(gate1b_path)
    cells = data["cells"]

    all_keys = set()
    for c in cells.values():
        all_keys.update(c.keys())

    w_train_keys = sorted(k for k in all_keys
                          if any(k.startswith(p) for p in
                                 ["ASSERT_train", "ASSERT_VAR_train",
                                  "BOUNDARY_train", "BASELINE_train"]))
    w_test_keys = sorted(k for k in all_keys
                         if any(k.startswith(p) for p in
                                ["MISLEADING_ASSERT_holdout",
                                 "MISLEADING_ASSERT_NOVAR_holdout",
                                 "REWRITE_holdout", "OBSERVE_holdout"]))

    results = {"w_train_keys": w_train_keys, "w_test_keys": w_test_keys,
               "by_depth": {}}

    for depth in [2, 3, 4]:
        depth_cells = {k: v for k, v in cells.items() if k[0] == depth}
        cell_ids = sorted(depth_cells.keys())
        n = len(cell_ids)

        pairs_data = []
        for i, j in combinations(range(n), 2):
            ci, cj = cell_ids[i], cell_ids[j]
            di, dj = depth_cells[ci], depth_cells[cj]

            d_train = compute_d_W(di, dj, w_train_keys)
            d_test = compute_d_W(di, dj, w_test_keys)

            cos_vals = []
            for k in w_train_keys:
                if k in di and k in dj:
                    cos_vals.append(cosine_sim(di[k], dj[k]))
            cos_mean = float(np.mean(cos_vals)) if cos_vals else None

            label = f"{ci}-{cj}"
            same_var = ci[1] == cj[1]
            same_val = ci[2] == cj[2]
            pairs_data.append({
                "label": label, "d_train": d_train, "d_test": d_test,
                "cosine_mean": cos_mean, "same_var": same_var,
                "same_val": same_val,
            })

        tau_results = {}
        for tau in tau_values:
            eligible = [p for p in pairs_data if p["d_train"] <= tau]
            n_eligible = len(eligible)

            if n_eligible == 0:
                tau_results[str(tau)] = {"n_eligible": 0, "n_total": len(pairs_data)}
                continue

            omega_transfer = max(p["d_test"] for p in eligible)
            test_preserved = sum(1 for p in eligible if p["d_test"] <= tau)

            cos_eligible = [p for p in pairs_data
                            if p["cosine_mean"] is not None
                            and p["cosine_mean"] >= 1.0 - tau]
            cos_test_preserved = sum(1 for p in cos_eligible if p["d_test"] <= tau)

            same_var_eligible = [p for p in eligible if p["same_var"]]
            same_var_preserved = sum(1 for p in same_var_eligible
                                     if p["d_test"] <= tau)

            d_train_vals = [p["d_train"] for p in eligible]
            d_test_vals = [p["d_test"] for p in eligible]

            tau_results[str(tau)] = {
                "n_eligible": n_eligible,
                "n_total": len(pairs_data),
                "omega_transfer": float(omega_transfer),
                "predicts_preserve": omega_transfer <= tau,
                "test_preserved": test_preserved,
                "test_preservation_rate": test_preserved / n_eligible,
                "d_train_stats": {
                    "mean": float(np.mean(d_train_vals)),
                    "max": float(np.max(d_train_vals)),
                },
                "d_test_stats": {
                    "mean": float(np.mean(d_test_vals)),
                    "median": float(np.median(d_test_vals)),
                    "max": float(np.max(d_test_vals)),
                },
                "cosine_baseline": {
                    "n_eligible": len(cos_eligible),
                    "test_preserved": cos_test_preserved,
                    "rate": cos_test_preserved / len(cos_eligible) if cos_eligible else None,
                },
                "same_var_breakdown": {
                    "n_same_var": len(same_var_eligible),
                    "preserved": same_var_preserved,
                    "n_cross_var": n_eligible - len(same_var_eligible),
                    "cross_preserved": test_preserved - same_var_preserved,
                },
            }

        results["by_depth"][str(depth)] = {
            "n_cells": n,
            "n_pairs": len(pairs_data),
            "tau_results": tau_results,
        }

    return results


if __name__ == "__main__":
    import sys
    congruence_path = Path("experiments/results/svb_qwen3_congruence/result.json")
    if not congruence_path.exists():
        print(f"Congruence data not found at {congruence_path}")
        sys.exit(1)

    print("=== Omega Modulus Transfer Test: Congruence Data ===\n")
    results = run_congruence_omega_test(congruence_path)

    for pair_name, pr in results.items():
        print(f"\n--- {pair_name} ({pr.get('n_cells', 0)} cells) ---")
        if "error" in pr:
            print(f"  Error: {pr['error']}")
            continue
        for tau_str, tr in pr.get("tau_results", {}).items():
            tau = float(tau_str)
            print(f"\n  tau = {tau}:")
            print(f"    Eligible pairs: {tr['n_eligible']}")
            if tr["n_eligible"] == 0:
                print("    (no eligible pairs at this tolerance)")
                continue
            print(f"    omega_c0 = {tr['omega_c0']:.6f}  "
                  f"(predicts {'PRESERVE' if tr['c0_preserves_prediction'] else 'SEPARATE'})")
            print(f"    omega_c1 = {tr['omega_c1']:.6f}  "
                  f"(predicts {'PRESERVE' if tr['c1_preserves_prediction'] else 'SEPARATE'})")
            print(f"    c0 actual: {tr['c0_actual_preserved']}/{tr['n_eligible']} preserved  "
                  f"({tr['c0_preservation_rate']:.1%})")
            print(f"    c1 actual: {tr['c1_actual_preserved']}/{tr['n_eligible']} preserved  "
                  f"({tr['c1_preservation_rate']:.1%})")
            if tr["cosine_mean"] is not None:
                print(f"    Cosine sim (baseline): mean={tr['cosine_mean']:.4f} "
                      f"min={tr['cosine_min']:.4f}")

    gate1b_path = Path("experiments/results/svb_qwen3_gate1b/result.json")
    if gate1b_path.exists():
        print("\n\n=== Omega Transfer Test: Gate1b Data ===\n")
        g_results = run_gate1b_omega_test(gate1b_path)
        print(f"W_train: {len(g_results['w_train_keys'])} surface keys")
        print(f"W_test:  {len(g_results['w_test_keys'])} surface keys")

        for depth_str, dr in g_results["by_depth"].items():
            print(f"\n--- Depth {depth_str} ({dr['n_cells']} cells, "
                  f"{dr['n_pairs']} pairs) ---")
            for tau_str, tr in dr["tau_results"].items():
                tau = float(tau_str)
                print(f"\n  tau = {tau}:")
                print(f"    Eligible (d_train <= tau): {tr['n_eligible']}/{tr['n_total']}")
                if tr["n_eligible"] == 0:
                    continue
                print(f"    omega_transfer = {tr['omega_transfer']:.6f}  "
                      f"(predicts {'PRESERVE' if tr['predicts_preserve'] else 'SEPARATE'})")
                print(f"    W_test preserved: {tr['test_preserved']}/{tr['n_eligible']}  "
                      f"({tr['test_preservation_rate']:.1%})")
                print(f"    d_test stats: mean={tr['d_test_stats']['mean']:.4f}  "
                      f"median={tr['d_test_stats']['median']:.4f}  "
                      f"max={tr['d_test_stats']['max']:.4f}")
                cb = tr["cosine_baseline"]
                if cb["rate"] is not None:
                    print(f"    Cosine baseline: {cb['test_preserved']}/{cb['n_eligible']} "
                          f"preserved ({cb['rate']:.1%})")
                sv = tr["same_var_breakdown"]
                if sv["n_same_var"] > 0:
                    print(f"    Same-var pairs: {sv['preserved']}/{sv['n_same_var']} preserved  "
                          f"Cross-var: {sv['cross_preserved']}/{sv['n_cross_var']} preserved")
