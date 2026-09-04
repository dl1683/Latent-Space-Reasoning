"""Fiber-square analysis for Gate 1b data.

Tests whether Gate 1b results satisfy the fiber-square criterion:
  a0 ~= a1  (ASSERT ~= ASSERT_VAR)         -- within-class equivalence
  b0 ~= b1  (MISLEADING_NOVAR ~= MISLEADING) -- within-class equivalence
  a_i !~= b_j                               -- between-class separation

Uses TV distance between response laws, not just mean L.
Codex criterion: UCB(sup_s TV(B_s(u), B_s(v))) < eps_eq for equivalence,
                 LCB(sup_s TV(B_s(u), B_s(w))) > eps_sep for separation.

The fiber square proves: the identity fiber q^{-1}(id) contains at least
two behavioral points with multiple surface representatives each.
"""
import json
import sys
from collections import defaultdict

import numpy as np


def tv_distance(p, q):
    """Total variation distance between two distributions."""
    return 0.5 * np.sum(np.abs(np.array(p) - np.array(q)))


def analyze_fiber_square(result_path):
    with open(result_path) as f:
        data = json.load(f)

    observations = data["observations"]

    fiber_roles = {
        "a0": "ASSERT",
        "a1": "ASSERT_VAR",
        "b0": "MISLEADING_ASSERT_NOVAR",
        "b1": "MISLEADING_ASSERT",
    }

    contexts = defaultdict(lambda: defaultdict(list))
    for key, obs in observations.items():
        if obs["role"] not in fiber_roles.values():
            continue
        if obs["val"] == 9:
            continue
        ctx = (obs["depth"], obs["var"], obs["val"])
        contexts[ctx][obs["role"]].append(np.array(obs["dist"]))

    print("=== FIBER SQUARE ANALYSIS ===\n")
    print(f"Contexts: {len(contexts)} (depth x var x val, excluding val=9)\n")

    pairs_within = [("a0", "a1"), ("b0", "b1")]
    pairs_between = [("a0", "b0"), ("a0", "b1"), ("a1", "b0"), ("a1", "b1")]

    within_tvs = defaultdict(list)
    between_tvs = defaultdict(list)

    for ctx, role_dists in contexts.items():
        for label, (r1_label, r2_label) in [("within", p) for p in pairs_within] + \
                                             [("between", p) for p in pairs_between]:
            r1 = fiber_roles[r1_label]
            r2 = fiber_roles[r2_label]
            if r1 not in role_dists or r2 not in role_dists:
                continue
            for d1 in role_dists[r1]:
                for d2 in role_dists[r2]:
                    tv = tv_distance(d1, d2)
                    if label == "within":
                        within_tvs[f"{r1_label}-{r2_label}"].append(tv)
                    else:
                        between_tvs[f"{r1_label}-{r2_label}"].append(tv)

    print("--- WITHIN-CLASS TV (want: SMALL) ---\n")
    print(f"  {'Pair':<20} {'n':>6} {'mean_TV':>9} {'max_TV':>9} {'p95_TV':>9}")
    for pair_name in ["a0-a1", "b0-b1"]:
        tvs = within_tvs.get(pair_name, [])
        if tvs:
            arr = np.array(tvs)
            print(f"  {pair_name:<20} {len(tvs):>6} {arr.mean():9.4f} {arr.max():9.4f} {np.percentile(arr, 95):9.4f}")

    print("\n--- BETWEEN-CLASS TV (want: LARGE) ---\n")
    print(f"  {'Pair':<20} {'n':>6} {'mean_TV':>9} {'min_TV':>9} {'p5_TV':>9}")
    for pair_name in ["a0-b0", "a0-b1", "a1-b0", "a1-b1"]:
        tvs = between_tvs.get(pair_name, [])
        if tvs:
            arr = np.array(tvs)
            print(f"  {pair_name:<20} {len(tvs):>6} {arr.mean():9.4f} {arr.min():9.4f} {np.percentile(arr, 5):9.4f}")

    all_within = []
    for tvs in within_tvs.values():
        all_within.extend(tvs)
    all_between = []
    for tvs in between_tvs.values():
        all_between.extend(tvs)

    if all_within and all_between:
        within_arr = np.array(all_within)
        between_arr = np.array(all_between)

        print(f"\n--- AGGREGATE ---\n")
        print(f"  Within-class TV:  mean={within_arr.mean():.4f}, max={within_arr.max():.4f}, p95={np.percentile(within_arr, 95):.4f}")
        print(f"  Between-class TV: mean={between_arr.mean():.4f}, min={between_arr.min():.4f}, p5={np.percentile(between_arr, 5):.4f}")

        gap = np.percentile(between_arr, 5) - np.percentile(within_arr, 95)
        print(f"\n  Separation gap (p5_between - p95_within): {gap:+.4f}")

        print(f"\n=== FIBER SQUARE VERDICT ===\n")
        if gap > 0:
            print(f"  -> FIBER SQUARE ESTABLISHED")
            print(f"     Within-class TV upper bound ({np.percentile(within_arr, 95):.4f}) < "
                  f"Between-class TV lower bound ({np.percentile(between_arr, 5):.4f})")
            print(f"     The identity fiber contains >= 2 behavioral points with multiple representatives.")
        elif within_arr.mean() < between_arr.mean() * 0.5:
            print(f"  -> PARTIAL: within < between on average but distributions overlap")
            print(f"     Within mean={within_arr.mean():.4f} < Between mean={between_arr.mean():.4f}")
            print(f"     Need tighter equivalence margins or more contexts.")
        else:
            print(f"  -> NO FIBER SQUARE: within-class and between-class TV not cleanly separated")

    print(f"\n--- PER-SPLIT (train vs holdout) ---\n")
    for split in ["train", "holdout"]:
        split_within = defaultdict(list)
        split_between = defaultdict(list)
        for key, obs in observations.items():
            if obs["role"] not in fiber_roles.values() or obs["val"] == 9 or obs["split"] != split:
                continue
            ctx = (obs["depth"], obs["var"], obs["val"], obs["split"])
            for key2, obs2 in observations.items():
                if obs2["role"] not in fiber_roles.values() or obs2["val"] == 9 or obs2["split"] != split:
                    continue
                if (obs2["depth"], obs2["var"], obs2["val"], obs2["split"]) != ctx:
                    continue
                if key >= key2:
                    continue
                tv = tv_distance(obs["dist"], obs2["dist"])
                r1_class = "a" if obs["role"] in ["ASSERT", "ASSERT_VAR"] else "b"
                r2_class = "a" if obs2["role"] in ["ASSERT", "ASSERT_VAR"] else "b"
                if r1_class == r2_class:
                    split_within[split].append(tv)
                else:
                    split_between[split].append(tv)

        w = split_within.get(split, [])
        b = split_between.get(split, [])
        if w and b:
            print(f"  {split:>8}: within={np.mean(w):.4f} (n={len(w)}), between={np.mean(b):.4f} (n={len(b)}), ratio={np.mean(b)/np.mean(w):.1f}x")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/results/svb_qwen3_gate1b/result.json"
    analyze_fiber_square(path)
