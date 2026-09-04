"""Per-surface equivalence analysis for Gate 2 representative selection.

For Gate 2 composition to be meaningful, we need specific surface pairs
that are behaviorally equivalent (TV < eps_eq across contexts). This
script finds them from Gate 1b data.

Equivalence classes tested:
  Class A: ASSERT ∪ ASSERT_VAR (true invariance assertions)
  Class M: MISLEADING_ASSERT ∪ MISLEADING_ASSERT_NOVAR (misleading assertions)

For each pair of surfaces within a class, computes TV across all shared
contexts and reports whether the pair qualifies as a Gate 2 representative.
"""
import json
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np


def tv_distance(p, q):
    return 0.5 * np.sum(np.abs(np.array(p) - np.array(q)))


def analyze_surface_equivalence(result_path, eps_eq=0.02, eps_sep=0.03):
    with open(result_path) as f:
        data = json.load(f)

    observations = data["observations"]

    class_A_roles = {"ASSERT", "ASSERT_VAR"}
    class_M_roles = {"MISLEADING_ASSERT", "MISLEADING_ASSERT_NOVAR"}

    surface_data = defaultdict(lambda: defaultdict(list))

    for key, obs in observations.items():
        if obs["val"] == 9:
            continue
        role = obs["role"]
        surface = obs.get("surface", "unknown")
        ctx = (obs["depth"], obs["var"], obs["val"])
        surface_data[(role, surface)][ctx].append(np.array(obs["dist"]))

    surfaces_by_class = {"A": [], "M": []}
    for (role, surface) in surface_data:
        if role in class_A_roles:
            surfaces_by_class["A"].append((role, surface))
        elif role in class_M_roles:
            surfaces_by_class["M"].append((role, surface))

    print("=== PER-SURFACE EQUIVALENCE ANALYSIS ===\n")

    for cls_name, surfaces in sorted(surfaces_by_class.items()):
        print(f"--- Class {cls_name}: {len(surfaces)} surfaces ---\n")
        for role, surface in sorted(surfaces):
            n_ctx = len(surface_data[(role, surface)])
            print(f"  {role:30s} | {surface!r:50s} | {n_ctx} contexts")
        print()

        if len(surfaces) < 2:
            print(f"  Only {len(surfaces)} surface(s), no pairs to test.\n")
            continue

        pair_results = []
        for (r1, s1), (r2, s2) in combinations(surfaces, 2):
            shared_ctxs = set(surface_data[(r1, s1)].keys()) & set(surface_data[(r2, s2)].keys())
            if not shared_ctxs:
                continue

            tvs = []
            for ctx in shared_ctxs:
                for d1 in surface_data[(r1, s1)][ctx]:
                    for d2 in surface_data[(r2, s2)][ctx]:
                        tvs.append(tv_distance(d1, d2))

            arr = np.array(tvs)
            cross_role = r1 != r2
            pair_results.append({
                "s1": f"{r1}:{s1}",
                "s2": f"{r2}:{s2}",
                "cross_role": cross_role,
                "n": len(tvs),
                "mean": arr.mean(),
                "max": arr.max(),
                "p95": np.percentile(arr, 95),
                "p99": np.percentile(arr, 99),
                "median": np.median(arr),
            })

        pair_results.sort(key=lambda x: x["max"])

        print(f"  {'Pair':<65} {'cross':>5} {'n':>5} {'mean':>7} {'p95':>7} {'max':>7} {'pass':>5}")
        for pr in pair_results:
            s1_short = pr["s1"].split(":")[1][:25]
            s2_short = pr["s2"].split(":")[1][:25]
            label = f"{s1_short} vs {s2_short}"
            cross = "Y" if pr["cross_role"] else "N"
            passes = "YES" if pr["max"] < eps_eq else ("~" if pr["p95"] < eps_eq else "NO")
            print(f"  {label:<65} {cross:>5} {pr['n']:>5} {pr['mean']:7.4f} {pr['p95']:7.4f} {pr['max']:7.4f} {passes:>5}")

        qualifying = [pr for pr in pair_results if pr["max"] < eps_eq]
        cross_role_qualifying = [pr for pr in qualifying if pr["cross_role"]]

        print(f"\n  Qualifying pairs (max TV < {eps_eq}): {len(qualifying)} total, {len(cross_role_qualifying)} cross-role")
        if cross_role_qualifying:
            print(f"\n  BEST CROSS-ROLE CANDIDATES FOR GATE 2:")
            for pr in cross_role_qualifying[:5]:
                print(f"    {pr['s1']} <-> {pr['s2']}  (max={pr['max']:.4f})")
        elif qualifying:
            print(f"\n  WARNING: Only within-role pairs qualify. No cross-role equivalence at eps_eq={eps_eq}.")
            print(f"  Consider relaxing eps_eq or using different surfaces.")
            print(f"\n  Best cross-role pairs (not qualifying but closest):")
            cross_role_all = [pr for pr in pair_results if pr["cross_role"]]
            for pr in cross_role_all[:5]:
                print(f"    {pr['s1']} <-> {pr['s2']}  (max={pr['max']:.4f}, p95={pr['p95']:.4f})")
        else:
            print(f"\n  WARNING: NO pairs qualify at eps_eq={eps_eq}.")
            print(f"  Best within-role pairs:")
            for pr in pair_results[:5]:
                print(f"    {pr['s1']} <-> {pr['s2']}  (max={pr['max']:.4f})")
        print()

    # Between-class separation check
    print("--- BETWEEN-CLASS SEPARATION ---\n")
    all_A = [(r, s) for r, s in surface_data if r in class_A_roles]
    all_M = [(r, s) for r, s in surface_data if r in class_M_roles]

    between_tvs = []
    for (r1, s1) in all_A:
        for (r2, s2) in all_M:
            shared = set(surface_data[(r1, s1)].keys()) & set(surface_data[(r2, s2)].keys())
            for ctx in shared:
                for d1 in surface_data[(r1, s1)][ctx]:
                    for d2 in surface_data[(r2, s2)][ctx]:
                        between_tvs.append(tv_distance(d1, d2))

    if between_tvs:
        arr = np.array(between_tvs)
        print(f"  A vs M: n={len(between_tvs)}, mean={arr.mean():.4f}, min={arr.min():.4f}, p5={np.percentile(arr, 5):.4f}")
        passes_sep = np.percentile(arr, 5) > eps_sep
        print(f"  Separation (p5 > {eps_sep}): {'YES' if passes_sep else 'NO'}")

    print(f"\n=== SUMMARY ===")
    print(f"  eps_eq={eps_eq}, eps_sep={eps_sep}")
    print(f"  For Gate 2, need cross-role pairs in each class with max TV < eps_eq")
    print(f"  AND between-class separation with p5 TV > eps_sep.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/results/svb_qwen3_gate1b/result.json"
    analyze_surface_equivalence(path)
