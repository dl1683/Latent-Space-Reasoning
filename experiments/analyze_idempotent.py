"""Idempotent analysis: tests B^2 = B for a candidate generator.

Combines Gate 1b data (baseline epsilon, single pass) with idempotent
test data (pass^2, pass^3) to check:
  1. Nontriviality: LCB D(epsilon, u) > eps_sep
  2. First stability: UCB D(u, u^2) < eps_eq
  3. Power stability: UCB D(u^2, u^3) < eps_eq

True idempotence: B^2 = B implies B^n = B for all n >= 1.
So stability must hold at ALL powers, not just the first.
"""
import json
import sys
from collections import defaultdict

import numpy as np


def tv_distance(p, q):
    return 0.5 * np.sum(np.abs(np.array(p) - np.array(q)))


def analyze_idempotent(gate1b_path, idempotent_path):
    with open(gate1b_path) as f:
        g1b = json.load(f)
    with open(idempotent_path) as f:
        idem = json.load(f)

    # Extract d4 distributions from Gate 1b for BASELINE and BOUNDARY(pass)
    baseline_dists = {}
    pass_dists = {}
    newline_dists = {}
    newline2_dists = {}
    newline3_dists = {}

    for key, obs in g1b["observations"].items():
        if obs["depth"] != 4 or obs["val"] == 9:
            continue
        ctx = (obs["var"], obs["val"])
        if obs["role"] == "BASELINE":
            baseline_dists[ctx] = np.array(obs["dist"])
        elif obs["role"] == "BOUNDARY":
            surf = obs["surface"]
            if surf == "pass\n":
                pass_dists[ctx] = np.array(obs["dist"])
            elif surf == "\n":
                newline_dists[ctx] = np.array(obs["dist"])
            elif surf == "\n\n":
                newline2_dists[ctx] = np.array(obs["dist"])
            elif surf == "\n\n\n":
                newline3_dists[ctx] = np.array(obs["dist"])

    # Extract pass^2 and pass^3 from idempotent test
    pass2_dists = {}
    pass3_dists = {}
    for key, obs in idem["observations"].items():
        if obs["val"] == 9:
            continue
        ctx = (obs["var"], obs["val"])
        if obs["role"] == "PASS_x2":
            pass2_dists[ctx] = np.array(obs["dist"])
        elif obs["role"] == "PASS_x3":
            pass3_dists[ctx] = np.array(obs["dist"])

    print("=== IDEMPOTENT TEST: pass\\n ===\n")
    print(f"  Contexts at d4: baseline={len(baseline_dists)}, pass={len(pass_dists)}, "
          f"pass^2={len(pass2_dists)}, pass^3={len(pass3_dists)}\n")

    # Compute pairwise TV distances
    pairs = [
        ("epsilon vs pass", baseline_dists, pass_dists),
        ("pass vs pass^2", pass_dists, pass2_dists),
        ("pass^2 vs pass^3", pass2_dists, pass3_dists),
        ("epsilon vs pass^2", baseline_dists, pass2_dists),
        ("epsilon vs pass^3", baseline_dists, pass3_dists),
    ]

    print(f"  {'Comparison':<25} {'n':>4} {'mean_TV':>9} {'med_TV':>9} {'p95_TV':>9} {'mean_dL':>9}")
    for name, d1, d2 in pairs:
        tvs, dLs = [], []
        for ctx in d1:
            if ctx in d2:
                tv = tv_distance(d1[ctx], d2[ctx])
                dL = float(d2[ctx][9] - d1[ctx][9])
                tvs.append(tv)
                dLs.append(dL)
        if tvs:
            arr = np.array(tvs)
            print(f"  {name:<25} {len(tvs):>4} {arr.mean():9.4f} {np.median(arr):9.4f} "
                  f"{np.percentile(arr, 95):9.4f} {np.mean(dLs):+9.4f}")

    # Verdict
    eps_eq = 0.01
    eps_sep = 0.02

    tv_eps_u = []
    tv_u_u2 = []
    tv_u2_u3 = []
    for ctx in baseline_dists:
        if ctx in pass_dists:
            tv_eps_u.append(tv_distance(baseline_dists[ctx], pass_dists[ctx]))
        if ctx in pass_dists and ctx in pass2_dists:
            tv_u_u2.append(tv_distance(pass_dists[ctx], pass2_dists[ctx]))
        if ctx in pass2_dists and ctx in pass3_dists:
            tv_u2_u3.append(tv_distance(pass2_dists[ctx], pass3_dists[ctx]))

    print(f"\n=== VERDICT (eps_eq={eps_eq}, eps_sep={eps_sep}) ===\n")

    if tv_eps_u:
        lcb = np.percentile(tv_eps_u, 5)
        print(f"  Nontriviality: LCB(TV(eps,u)) = {lcb:.4f}", end="")
        if lcb > eps_sep:
            print(f" > {eps_sep} -> PASS (u != identity)")
        else:
            print(f" <= {eps_sep} -> FAIL (u may be trivial)")

    if tv_u_u2:
        ucb = np.percentile(tv_u_u2, 95)
        print(f"  First stability: UCB(TV(u,u^2)) = {ucb:.4f}", end="")
        if ucb < eps_eq:
            print(f" < {eps_eq} -> PASS (u^2 ~= u)")
        else:
            print(f" >= {eps_eq} -> FAIL (u^2 != u)")

    if tv_u2_u3:
        ucb = np.percentile(tv_u2_u3, 95)
        print(f"  Power stability: UCB(TV(u^2,u^3)) = {ucb:.4f}", end="")
        if ucb < eps_eq:
            print(f" < {eps_eq} -> PASS (u^3 ~= u^2)")
        else:
            print(f" >= {eps_eq} -> FAIL (u^3 != u^2)")

    # Also show newline comparison for reference
    print(f"\n--- NEWLINE COMPARISON (from Gate 1b) ---\n")
    nl_pairs = [
        ("newline vs newline^2", newline_dists, newline2_dists),
        ("newline^2 vs newline^3", newline2_dists, newline3_dists),
    ]
    for name, d1, d2 in nl_pairs:
        tvs = [tv_distance(d1[c], d2[c]) for c in d1 if c in d2]
        if tvs:
            arr = np.array(tvs)
            print(f"  {name:<25} n={len(tvs):>3} mean_TV={arr.mean():.4f} p95={np.percentile(arr, 95):.4f}")


if __name__ == "__main__":
    g1b = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/results/svb_qwen3_gate1b/result.json"
    idem = sys.argv[2] if len(sys.argv) > 2 else \
        "experiments/results/svb_qwen3_idempotent_pass/result.json"
    analyze_idempotent(g1b, idem)
