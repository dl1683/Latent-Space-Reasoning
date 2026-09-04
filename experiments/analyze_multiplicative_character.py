"""Multiplicative character analysis for Gate 1b data.

Tests the promotion ladder from Codex:
  Step 1: Cellwise L_M/L_A ratio stability (not just depth means)
  Step 2: R-preservation and S=C+L preservation
  Step 3: Surface descent (paraphrases yield same a_u)

Uses BASELINE to compute relative gains: a_u(z) = L(T_u z) / L(z).
The separable law: log L(T_u z_d) = g(d,z) + theta_u.
"""
import json
import sys
from collections import defaultdict

import numpy as np


def analyze_character(result_path):
    with open(result_path) as f:
        data = json.load(f)

    observations = data["observations"]

    role_data = defaultdict(lambda: defaultdict(list))
    for key, obs in observations.items():
        if obs["val"] == 9:
            continue
        ctx = (obs["depth"], obs["var"], obs["val"])
        dist = np.array(obs["dist"])
        C = float(dist[int(obs["val"])])
        L = float(dist[9])
        R = 1.0 - C - L
        role_data[obs["role"]][ctx].append({
            "C": C, "L": L, "R": R, "S": C + L,
            "dist": dist, "surface": obs.get("surface", ""),
            "split": obs.get("split", "train"),
        })

    print("=== MULTIPLICATIVE CHARACTER ANALYSIS ===\n")

    # Step 1: Cellwise ratio stability
    print("--- STEP 1: Cellwise L ratio stability ---\n")
    ratios_by_depth = defaultdict(list)
    log_gains = defaultdict(list)
    all_ratios = []

    for ctx in role_data["BASELINE"]:
        bl = role_data["BASELINE"][ctx]
        ast = role_data["ASSERT"].get(ctx, [])
        mis = role_data["MISLEADING_ASSERT"].get(ctx, [])

        if not bl or not ast or not mis:
            continue

        L_bl = np.mean([x["L"] for x in bl])
        L_ast = np.mean([x["L"] for x in ast])
        L_mis = np.mean([x["L"] for x in mis])

        depth = ctx[0]

        if L_ast > 1e-6:
            ratio = L_mis / L_ast
            ratios_by_depth[depth].append(ratio)
            all_ratios.append(ratio)

        if L_bl > 1e-6:
            g_ast = np.log(max(L_ast, 1e-10)) - np.log(L_bl)
            g_mis = np.log(max(L_mis, 1e-10)) - np.log(L_bl)
            log_gains[("ASSERT", depth)].append(g_ast)
            log_gains[("MISLEADING_ASSERT", depth)].append(g_mis)

    print(f"  {'Depth':>6} {'n':>5} {'mean_ratio':>11} {'std':>8} {'CV':>8} {'med':>8}")
    for d in sorted(ratios_by_depth):
        arr = np.array(ratios_by_depth[d])
        cv = arr.std() / arr.mean() if arr.mean() > 0 else float('inf')
        print(f"  d{d:>5} {len(arr):>5} {arr.mean():11.3f} {arr.std():8.3f} {cv:8.3f} {np.median(arr):8.3f}")

    if all_ratios:
        arr = np.array(all_ratios)
        cv = arr.std() / arr.mean()
        print(f"  {'ALL':>6} {len(arr):>5} {arr.mean():11.3f} {arr.std():8.3f} {cv:8.3f} {np.median(arr):8.3f}")

        depth_means = [np.mean(ratios_by_depth[d]) for d in sorted(ratios_by_depth)]
        depth_range = max(depth_means) - min(depth_means)
        print(f"\n  Depth-mean range: {depth_range:.3f} (want: small)")
        print(f"  Overall CV: {cv:.3f} (want: << 1 for stable character)")

    # Step 1b: Separable log-gain test
    print(f"\n--- STEP 1b: Separable log-gain (theta_u independent of depth) ---\n")
    for role in ["ASSERT", "MISLEADING_ASSERT"]:
        depths = sorted(set(d for (r, d) in log_gains if r == role))
        if depths:
            means = [np.mean(log_gains[(role, d)]) for d in depths]
            stds = [np.std(log_gains[(role, d)]) for d in depths]
            print(f"  {role}:")
            for d, m, s in zip(depths, means, stds):
                print(f"    d{d}: log_gain = {m:+.4f} ± {s:.4f}")
            range_lg = max(means) - min(means)
            print(f"    depth range: {range_lg:.4f} (want: small for separability)")

    # Step 2: Coordinate preservation
    print(f"\n--- STEP 2: R-preservation and S=C+L preservation ---\n")
    for role in ["ASSERT", "ASSERT_VAR", "MISLEADING_ASSERT", "MISLEADING_ASSERT_NOVAR", "BASELINE"]:
        role_obs = []
        for ctx, entries in role_data[role].items():
            for e in entries:
                role_obs.append(e)
        if role_obs:
            Rs = [x["R"] for x in role_obs]
            Ss = [x["S"] for x in role_obs]
            print(f"  {role:<30} R_mean={np.mean(Rs):.4f}±{np.std(Rs):.4f}  S_mean={np.mean(Ss):.4f}±{np.std(Ss):.4f}")

    # Cellwise R and S comparison (role vs baseline)
    print(f"\n  Cellwise delta (role - BASELINE):")
    for role in ["ASSERT", "MISLEADING_ASSERT"]:
        dR_list, dS_list = [], []
        for ctx in role_data["BASELINE"]:
            bl = role_data["BASELINE"][ctx]
            ro = role_data[role].get(ctx, [])
            if not bl or not ro:
                continue
            R_bl = np.mean([x["R"] for x in bl])
            S_bl = np.mean([x["S"] for x in bl])
            R_ro = np.mean([x["R"] for x in ro])
            S_ro = np.mean([x["S"] for x in ro])
            dR_list.append(R_ro - R_bl)
            dS_list.append(S_ro - S_bl)
        if dR_list:
            print(f"  {role:<30} dR={np.mean(dR_list):+.4f}±{np.std(dR_list):.4f}  dS={np.mean(dS_list):+.4f}±{np.std(dS_list):.4f}")

    # Step 3: Surface descent (within-class consistency of a_u)
    print(f"\n--- STEP 3: Surface descent (same class, same a_u?) ---\n")
    for role in ["ASSERT", "ASSERT_VAR", "MISLEADING_ASSERT", "MISLEADING_ASSERT_NOVAR"]:
        surface_ratios = defaultdict(list)
        for ctx in role_data["BASELINE"]:
            bl = role_data["BASELINE"][ctx]
            entries = role_data[role].get(ctx, [])
            if not bl or not entries:
                continue
            L_bl = np.mean([x["L"] for x in bl])
            if L_bl < 1e-6:
                continue
            for e in entries:
                ratio = e["L"] / L_bl
                surface_ratios[e["surface"]].append(ratio)

        if surface_ratios:
            print(f"  {role}:")
            surf_means = []
            for surf, ratios in sorted(surface_ratios.items(), key=lambda x: np.mean(x[1])):
                arr = np.array(ratios)
                surf_means.append(arr.mean())
                surf_display = repr(surf)[:40]
                print(f"    {surf_display:<42} a_u={arr.mean():.3f}±{arr.std():.3f} (n={len(arr)})")
            if len(surf_means) >= 2:
                within_cv = np.std(surf_means) / np.mean(surf_means) if np.mean(surf_means) > 0 else float('inf')
                print(f"    Within-class CV of a_u: {within_cv:.3f} (want: small)")

    # Step 4: K_a vs logit-bias rival — CROSS-FIT version (Codex review 2026-09-04)
    # Prior version was tautological: fit a_u/δ_u per cell, predicted same cell.
    # Fix: fit one a_u (or b_u=e^δ_u) per role on TRAIN split, predict HOLDOUT.
    #
    # K_a model:  L' = a_u·L, C' = C+(1-a_u)·L, R' = R
    # Logit-bias: b = e^δ_u, Z = 1+(b-1)·L, L' = b·L/Z, C' = C/Z, R' = R/Z
    # Fingerprint difference: K_a preserves R; logit-bias rescales all non-shadow.
    print(f"\n--- STEP 4: K_a vs logit-bias rival (cross-fit) ---\n")
    print("  K_a:        L'=a·L, C'=C+(1-a)L, R'=R          (C-L exchange)")
    print("  Logit-bias: b=e^d, Z=1+(b-1)L, L'=bL/Z, R'=R/Z (uniform rescale)")
    print("  Fit on TRAIN split, predict HOLDOUT.\n")

    for role in ["ASSERT", "MISLEADING_ASSERT"]:
        train_ratios = []
        train_log_bias = []

        for ctx in role_data["BASELINE"]:
            bl_train = [x for x in role_data["BASELINE"][ctx] if x["split"] == "train"]
            ro_train = [x for x in role_data[role].get(ctx, []) if x["split"] == "train"]
            if not bl_train or not ro_train:
                continue
            L_bl = np.mean([x["L"] for x in bl_train])
            C_bl = np.mean([x["C"] for x in bl_train])
            L_ro = np.mean([x["L"] for x in ro_train])
            C_ro = np.mean([x["C"] for x in ro_train])
            if L_bl < 1e-6 or C_bl < 1e-6 or L_ro < 1e-10 or C_ro < 1e-10:
                continue
            train_ratios.append(L_ro / L_bl)
            train_log_bias.append(np.log(L_ro / C_ro) - np.log(L_bl / C_bl))

        if not train_ratios:
            print(f"  {role}: insufficient train data\n")
            continue

        a_u_fit = float(np.median(train_ratios))
        delta_u_fit = float(np.median(train_log_bias))
        b_u_fit = np.exp(delta_u_fit)
        print(f"  {role} (fitted on train):")
        print(f"    a_u = {a_u_fit:.4f}  (K_a parameter)")
        print(f"    d_u = {delta_u_fit:+.4f}, b = {b_u_fit:.4f}  (logit-bias parameter)")

        ka_L_err, ka_R_err, ka_C_err = [], [], []
        lb_L_err, lb_R_err, lb_C_err = [], [], []
        obs_dR, obs_RC_ratio_change = [], []

        for ctx in role_data["BASELINE"]:
            bl_hold = [x for x in role_data["BASELINE"][ctx] if x["split"] == "holdout"]
            ro_hold = [x for x in role_data[role].get(ctx, []) if x["split"] == "holdout"]
            if not bl_hold or not ro_hold:
                continue
            L_bl = np.mean([x["L"] for x in bl_hold])
            C_bl = np.mean([x["C"] for x in bl_hold])
            R_bl = np.mean([x["R"] for x in bl_hold])
            L_ro = np.mean([x["L"] for x in ro_hold])
            C_ro = np.mean([x["C"] for x in ro_hold])
            R_ro = np.mean([x["R"] for x in ro_hold])
            if L_bl < 1e-6 or C_bl < 1e-6:
                continue

            ka_L = a_u_fit * L_bl
            ka_C = C_bl + (1 - a_u_fit) * L_bl
            ka_R = R_bl

            Z = 1 + (b_u_fit - 1) * L_bl
            lb_L = b_u_fit * L_bl / Z
            lb_C = C_bl / Z
            lb_R = R_bl / Z

            ka_L_err.append(L_ro - ka_L)
            ka_C_err.append(C_ro - ka_C)
            ka_R_err.append(R_ro - ka_R)
            lb_L_err.append(L_ro - lb_L)
            lb_C_err.append(C_ro - lb_C)
            lb_R_err.append(R_ro - lb_R)
            obs_dR.append(R_ro - R_bl)
            if C_bl > 1e-6 and C_ro > 1e-6 and R_bl > 1e-6:
                obs_RC_ratio_change.append((R_ro / C_ro) - (R_bl / C_bl))

        if not ka_L_err:
            print(f"    No holdout data for cross-validation\n")
            continue

        ka_L_arr = np.array(ka_L_err)
        ka_C_arr = np.array(ka_C_err)
        ka_R_arr = np.array(ka_R_err)
        lb_L_arr = np.array(lb_L_err)
        lb_C_arr = np.array(lb_C_err)
        lb_R_arr = np.array(lb_R_err)
        dR_arr = np.array(obs_dR)

        ka_total = np.mean(np.abs(ka_L_arr) + np.abs(ka_C_arr) + np.abs(ka_R_arr))
        lb_total = np.mean(np.abs(lb_L_arr) + np.abs(lb_C_arr) + np.abs(lb_R_arr))

        print(f"\n    Holdout prediction (n={len(ka_L_err)} cells):")
        print(f"    {'':>20} {'K_a':>12} {'Logit-bias':>12}")
        print(f"    {'L residual mean':>20} {ka_L_arr.mean():+12.5f} {lb_L_arr.mean():+12.5f}")
        print(f"    {'C residual mean':>20} {ka_C_arr.mean():+12.5f} {lb_C_arr.mean():+12.5f}")
        print(f"    {'R residual mean':>20} {ka_R_arr.mean():+12.5f} {lb_R_arr.mean():+12.5f}")
        print(f"    {'|L| + |C| + |R| MAE':>20} {ka_total:12.5f} {lb_total:12.5f}")

        print(f"\n    R-PRESERVATION TEST (K_a predicts ΔR=0):")
        print(f"      Observed ΔR: mean={dR_arr.mean():+.5f} ± {dR_arr.std():.5f}")
        logit_pred_dR = -dR_arr  # placeholder; compute properly
        # Logit-bias predicted ΔR = R_bl(1/Z - 1) = -R_bl(b-1)L/(1+(b-1)L)
        print(f"      K_a pred ΔR=0: residual |ΔR| mean={np.abs(dR_arr).mean():.5f}")
        lb_pred_dR_vals = []
        for ctx in role_data["BASELINE"]:
            bl_hold = [x for x in role_data["BASELINE"][ctx] if x["split"] == "holdout"]
            if not bl_hold:
                continue
            L_bl = np.mean([x["L"] for x in bl_hold])
            R_bl = np.mean([x["R"] for x in bl_hold])
            if L_bl < 1e-6:
                continue
            Z = 1 + (b_u_fit - 1) * L_bl
            lb_pred_dR_vals.append(R_bl / Z - R_bl)
        if lb_pred_dR_vals:
            lb_dR_pred = np.array(lb_pred_dR_vals)
            print(f"      Logit-bias pred ΔR: mean={lb_dR_pred.mean():+.5f}")

        if obs_RC_ratio_change:
            rc_arr = np.array(obs_RC_ratio_change)
            print(f"\n    R/C RATIO CHANGE (logit-bias predicts 0):")
            print(f"      Observed Δ(R/C): mean={rc_arr.mean():+.5f} ± {rc_arr.std():.5f}")

        winner = "K_a" if ka_total < lb_total else "Logit-bias"
        print(f"\n    WINNER (lower cross-fit MAE): {winner} ({ka_total:.5f} vs {lb_total:.5f})")
        print()

    # Summary
    print(f"\n=== PROMOTION LADDER SUMMARY ===\n")
    if all_ratios:
        arr = np.array(all_ratios)
        cv = arr.std() / arr.mean()
        if cv < 0.3:
            print(f"  Step 1: PASS — cellwise ratio stable (CV={cv:.3f})")
        elif cv < 0.5:
            print(f"  Step 1: MARGINAL — ratio moderately stable (CV={cv:.3f})")
        else:
            print(f"  Step 1: FAIL — ratio not stable (CV={cv:.3f})")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/results/svb_qwen3_gate1b/result.json"
    analyze_character(path)
