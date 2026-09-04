"""Gate 2 composition analysis: tests F-comp (representative independence).

If a0 ≡_M a1 and b0 ≡_M b1, then all four compositions
a0*b0, a0*b1, a1*b0, a1*b1 should produce the same behavioral law.

Also tests multiplicative character composition: chi(A*B) ≈ chi(A)*chi(B).
"""
import json
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np


def tv_distance(p, q):
    return 0.5 * np.sum(np.abs(np.array(p) - np.array(q)))


def analyze_composition(result_path, gate1b_path=None):
    with open(result_path) as f:
        data = json.load(f)

    observations = data["observations"]

    roles = list(data["config"]["intensional_roles"].keys())
    role_dists = defaultdict(lambda: defaultdict(list))

    for key, obs in observations.items():
        if obs["val"] == 9:
            continue
        ctx = (obs["depth"], obs["var"], obs["val"])
        role_dists[obs["role"]][ctx].append(np.array(obs["dist"]))

    print("=== GATE 2: COMPOSITION (F-COMP) ===\n")

    # Pairwise TV between all role pairs, per context
    pair_tvs = defaultdict(list)
    for ctx in set().union(*[set(role_dists[r].keys()) for r in roles]):
        for r1, r2 in combinations(roles, 2):
            if ctx not in role_dists[r1] or ctx not in role_dists[r2]:
                continue
            for d1 in role_dists[r1][ctx]:
                for d2 in role_dists[r2][ctx]:
                    tv = tv_distance(d1, d2)
                    pair_tvs[(r1, r2)].append(tv)

    print(f"  {'Pair':<55} {'n':>5} {'mean_TV':>9} {'p95_TV':>9} {'max_TV':>9}")
    for (r1, r2), tvs in sorted(pair_tvs.items()):
        arr = np.array(tvs)
        print(f"  {r1} vs {r2:<30} {len(tvs):>5} {arr.mean():9.4f} "
              f"{np.percentile(arr, 95):9.4f} {arr.max():9.4f}")

    # Max pairwise TV (the decisive metric)
    all_tvs = []
    for tvs in pair_tvs.values():
        all_tvs.extend(tvs)

    if all_tvs:
        arr = np.array(all_tvs)
        print(f"\n  Overall max pairwise TV: {arr.max():.4f}")
        print(f"  Overall p95 pairwise TV: {np.percentile(arr, 95):.4f}")
        print(f"  Overall mean pairwise TV: {arr.mean():.4f}")

        eps_eq = 0.01
        print(f"\n=== VERDICT (eps_eq={eps_eq}) ===\n")
        if np.percentile(arr, 95) < eps_eq:
            print(f"  -> F-COMP PASS: all compositions behaviorally equivalent")
            print(f"     Monoid multiplication is well-defined on equivalence classes.")
        elif arr.mean() < eps_eq:
            print(f"  -> F-COMP MARGINAL: mean equivalent but tails exceed threshold")
        else:
            print(f"  -> F-COMP FAIL: compositions are NOT representative-independent")

    # If Gate 1b data available, compute character values
    if gate1b_path:
        print(f"\n--- MULTIPLICATIVE CHARACTER COMPOSITION TEST ---\n")
        with open(gate1b_path) as f:
            g1b = json.load(f)

        # Get baseline L and single-role L from Gate 1b
        bl_L = defaultdict(list)
        assert_L = defaultdict(list)
        misleading_L = defaultdict(list)

        for key, obs in g1b["observations"].items():
            if obs["val"] == 9:
                continue
            ctx = (obs["depth"], obs["var"], obs["val"])
            L = float(np.array(obs["dist"])[9])
            if obs["role"] == "BASELINE":
                bl_L[ctx].append(L)
            elif obs["role"] == "ASSERT":
                assert_L[ctx].append(L)
            elif obs["role"] == "MISLEADING_ASSERT":
                misleading_L[ctx].append(L)

        # Compute chi(A), chi(M) from Gate 1b
        chi_A, chi_M = [], []
        for ctx in bl_L:
            L_bl = np.mean(bl_L[ctx])
            if L_bl < 1e-6:
                continue
            if ctx in assert_L:
                chi_A.append(np.mean(assert_L[ctx]) / L_bl)
            if ctx in misleading_L:
                chi_M.append(np.mean(misleading_L[ctx]) / L_bl)

        # Compute chi(A*M) from composition data
        comp_L = defaultdict(list)
        for key, obs in observations.items():
            if obs["val"] == 9:
                continue
            ctx = (obs["depth"], obs["var"], obs["val"])
            L = float(np.array(obs["dist"])[9])
            comp_L[ctx].append(L)

        chi_AM = []
        for ctx in bl_L:
            L_bl = np.mean(bl_L[ctx])
            if L_bl < 1e-6 or ctx not in comp_L:
                continue
            chi_AM.append(np.mean(comp_L[ctx]) / L_bl)

        if chi_A and chi_M and chi_AM:
            print(f"  chi(A) = {np.mean(chi_A):.4f} ± {np.std(chi_A):.4f}")
            print(f"  chi(M) = {np.mean(chi_M):.4f} ± {np.std(chi_M):.4f}")
            print(f"  chi(A*M) observed = {np.mean(chi_AM):.4f} ± {np.std(chi_AM):.4f}")
            predicted = np.mean(chi_A) * np.mean(chi_M)
            print(f"  chi(A)*chi(M) predicted = {predicted:.4f}")
            error = abs(np.mean(chi_AM) - predicted) / predicted if predicted > 0 else float('inf')
            print(f"  Relative error: {error:.3f} ({error*100:.1f}%)")

            if error < 0.1:
                print(f"\n  -> CHARACTER COMPOSITION: CONSISTENT ({error*100:.1f}% error)")
            elif error < 0.25:
                print(f"\n  -> CHARACTER COMPOSITION: MARGINAL ({error*100:.1f}% error)")
            else:
                print(f"\n  -> CHARACTER COMPOSITION: FAILS ({error*100:.1f}% error)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/results/svb_qwen3_gate2_composition/result.json"
    g1b = sys.argv[2] if len(sys.argv) > 2 else \
        "experiments/results/svb_qwen3_gate1b/result.json"
    analyze_composition(path, g1b)
