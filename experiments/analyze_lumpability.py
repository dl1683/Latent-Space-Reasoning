"""F8 Lumpability test: does the (C,L,R) simplex projection commute with suffix actions?

If κr(z) = κr(z') implies κr(T_u z) = κr(T_u z'), then K_u is a well-defined
operator on the simplex. If not, K_u is only a fitted summary.

Test: find pairs of contexts (depth, var, val) with similar BASELINE (C,L,R)
and check whether their post-suffix (C,L,R) are also similar. Systematic
divergence means the (C,L,R) projection loses information the model uses.
"""
import json
import sys
from collections import defaultdict

import numpy as np


def analyze_lumpability(result_path):
    with open(result_path) as f:
        data = json.load(f)

    observations = data["observations"]

    # Collect BASELINE and role distributions per context
    baseline = {}
    role_data = defaultdict(dict)

    for key, obs in observations.items():
        if obs["val"] == 9:
            continue
        ctx = (obs["depth"], obs["var"], obs["val"])
        dist = np.array(obs["dist"])
        C = float(dist[int(obs["val"])])
        L = float(dist[9])
        R = 1.0 - C - L

        if obs["role"] == "BASELINE":
            baseline[ctx] = {"C": C, "L": L, "R": R, "dist": dist}
        else:
            if obs["role"] not in role_data:
                role_data[obs["role"]] = {}
            if ctx not in role_data[obs["role"]]:
                role_data[obs["role"]][ctx] = []
            role_data[obs["role"]][ctx].append({"C": C, "L": L, "R": R, "dist": dist})

    print("=== F8 LUMPABILITY TEST ===\n")
    print(f"Baseline contexts: {len(baseline)}\n")

    # For each role, check: do contexts with similar baseline CLR
    # produce similar post-suffix CLR?
    test_roles = ["ASSERT", "ASSERT_VAR", "MISLEADING_ASSERT", "MISLEADING_ASSERT_NOVAR",
                  "REWRITE", "OBSERVE", "BOUNDARY"]

    for role in test_roles:
        if role not in role_data:
            continue

        ctxs = sorted(set(baseline.keys()) & set(role_data[role].keys()))
        if len(ctxs) < 2:
            continue

        # Get baseline and mean post-suffix CLR for each context
        bl_clr = np.array([[baseline[c]["C"], baseline[c]["L"], baseline[c]["R"]] for c in ctxs])
        post_clr = np.array([[np.mean([e["C"] for e in role_data[role][c]]),
                              np.mean([e["L"] for e in role_data[role][c]]),
                              np.mean([e["R"] for e in role_data[role][c]])]
                             for c in ctxs])

        # For each pair, compute baseline similarity and post-suffix similarity
        n = len(ctxs)
        bl_dists = []
        post_dists = []
        for i in range(n):
            for j in range(i+1, n):
                bl_d = np.sum(np.abs(bl_clr[i] - bl_clr[j]))
                post_d = np.sum(np.abs(post_clr[i] - post_clr[j]))
                bl_dists.append(bl_d)
                post_dists.append(post_d)

        bl_arr = np.array(bl_dists)
        post_arr = np.array(post_dists)

        # Lumpability check: among pairs with small baseline distance,
        # is the post-suffix distance also small?
        thresholds = [0.02, 0.05, 0.10]
        print(f"  {role}:")
        for thresh in thresholds:
            mask = bl_arr < thresh
            if mask.sum() > 0:
                similar_post = post_arr[mask]
                all_post = post_arr
                print(f"    BL_dist<{thresh:.2f}: {mask.sum():>5} pairs, "
                      f"post_dist={similar_post.mean():.4f}±{similar_post.std():.4f} "
                      f"(vs all={all_post.mean():.4f})")

        # Correlation: if lumpable, baseline distance should predict post distance
        if len(bl_dists) > 10:
            corr = np.corrcoef(bl_arr, post_arr)[0, 1]
            print(f"    Correlation(bl_dist, post_dist) = {corr:.3f}")

        # Check residual: after accounting for baseline CLR, how much
        # variance in post CLR is unexplained?
        if n >= 5:
            from numpy.linalg import lstsq
            # Fit: post_L = a * bl_L + b
            X = np.column_stack([bl_clr[:, 1], np.ones(n)])
            y = post_clr[:, 1]
            coeffs, residuals, _, _ = lstsq(X, y, rcond=None)
            y_pred = X @ coeffs
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            print(f"    R²(bl_L → post_L) = {r2:.3f} (perfect lumpability = 1.0)")
        print()

    # Summary
    print("=== LUMPABILITY SUMMARY ===\n")
    print("If R² ≈ 1.0 for all roles: (C,L,R) projection is approximately lumpable.")
    print("If R² << 1.0: the model uses information beyond (C,L,R) to determine")
    print("post-suffix behavior. K_u is a fitted summary, not a well-defined operator.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/results/svb_qwen3_gate1b/result.json"
    analyze_lumpability(path)
