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
