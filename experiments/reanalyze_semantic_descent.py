"""Re-analyze Gate 1 results with F6 baseline comparison.

Loads saved result.json and runs the updated analysis including
statement-type vs intensional-role variance comparison.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def reanalyze(result_path):
    with open(result_path) as f:
        data = json.load(f)

    observations = data["observations"]

    roles = ["ASSERT", "MISLEADING_ASSERT", "REWRITE", "OBSERVE", "BOUNDARY"]

    stmt_type_map = {
        "ASSERT": "comment",
        "MISLEADING_ASSERT": "comment",
        "REWRITE": "assignment",
        "OBSERVE": "expression",
        "BOUNDARY": "delimiter",
    }

    for split in ["train", "holdout"]:
        print(f"\n=== {split.upper()} SET ===\n")

        role_L = defaultdict(list)
        surface_L = defaultdict(list)
        stmt_L = defaultdict(list)

        for key, obs in observations.items():
            if obs["split"] != split:
                continue
            dist = np.array(obs["dist"])
            L = float(dist[9]) if obs["val"] != 9 else 0.0
            role_L[obs["role"]].append(L)
            surface_L[(obs["role"], obs["surface"])].append(L)
            stmt_L[stmt_type_map.get(obs["role"], "?")].append(L)

        print(f"  {'Role':<25} {'n':>5} {'L_mean':>8} {'L_std':>8} {'L_med':>8}")
        for role in roles:
            vals = role_L.get(role, [])
            if vals:
                arr = np.array(vals)
                print(f"  {role:<25} {len(vals):>5} {arr.mean():8.4f} {arr.std():8.4f} {np.median(arr):8.4f}")

        within_vars = []
        for (role, surf), vals in surface_L.items():
            if len(vals) >= 2:
                within_vars.append(np.var(vals))

        role_means = [np.mean(v) for v in role_L.values() if v]
        within_mean = np.mean(within_vars) if within_vars else float('nan')
        between_role = np.var(role_means) if len(role_means) >= 2 else float('nan')

        stmt_means = [np.mean(v) for v in stmt_L.values() if v]
        between_stmt = np.var(stmt_means) if len(stmt_means) >= 2 else float('nan')

        print(f"\n  Within-surface L variance (mean): {within_mean:.6f}")
        print(f"  Between-ROLE L variance:          {between_role:.6f}")
        print(f"  Between-STMT-TYPE L variance:     {between_stmt:.6f}")

        if within_mean > 0:
            role_ratio = between_role / within_mean
            stmt_ratio = between_stmt / within_mean
            print(f"  Role/Within ratio:   {role_ratio:.2f}")
            print(f"  Stmt/Within ratio:   {stmt_ratio:.2f}")
            if role_ratio > stmt_ratio * 1.2:
                print(f"  -> Role explains {role_ratio/stmt_ratio:.1f}x more than stmt-type")
            else:
                print(f"  -> Stmt-type baseline is competitive")

        print(f"\n  Statement-type breakdown:")
        print(f"  {'Type':<15} {'n':>5} {'L_mean':>8} {'L_std':>8}")
        for stype in ["comment", "assignment", "expression", "delimiter"]:
            vals = stmt_L.get(stype, [])
            if vals:
                print(f"  {stype:<15} {len(vals):>5} {np.mean(vals):8.4f} {np.std(vals):8.4f}")

    print("\n=== DECISIVE: MISLEADING_ASSERT vs ASSERT ===\n")
    assert_L = [L for key, obs in observations.items()
                if obs["role"] == "ASSERT"
                for L in [float(np.array(obs["dist"])[9]) if obs["val"] != 9 else 0.0]]
    misleading_L = [L for key, obs in observations.items()
                    if obs["role"] == "MISLEADING_ASSERT"
                    for L in [float(np.array(obs["dist"])[9]) if obs["val"] != 9 else 0.0]]

    if assert_L and misleading_L:
        a_mean, m_mean = np.mean(assert_L), np.mean(misleading_L)
        a_std, m_std = np.std(assert_L), np.std(misleading_L)
        diff = m_mean - a_mean
        pooled_se = np.sqrt(a_std**2/len(assert_L) + m_std**2/len(misleading_L))
        t_stat = diff / pooled_se if pooled_se > 0 else 0

        print(f"  ASSERT:              L={a_mean:.4f} ± {a_std:.4f} (n={len(assert_L)})")
        print(f"  MISLEADING_ASSERT:   L={m_mean:.4f} ± {m_std:.4f} (n={len(misleading_L)})")
        print(f"  Diff (misleading-true): {diff:+.4f}")
        print(f"  t-statistic:         {t_stat:.2f}")

        if diff > 0.02 and t_stat > 2:
            print(f"\n  VERDICT: Model follows CONTENT (misleading comments suppress less)")
            print(f"  -> Supports intensional descent over lexical cueing")
        elif abs(diff) < 0.01:
            print(f"\n  VERDICT: Model follows FORM (comment form dominates)")
            print(f"  -> Supports lexical cueing hypothesis")
        else:
            print(f"\n  VERDICT: AMBIGUOUS (small or non-significant difference)")

    print("\n=== PER-DEPTH ROLE MEANS ===\n")
    for depth in sorted(set(obs["depth"] for obs in observations.values())):
        print(f"  d{depth}:")
        print(f"    {'Role':<25} {'L_mean':>8}")
        for role in roles:
            vals = [float(np.array(obs["dist"])[9]) if obs["val"] != 9 else 0.0
                    for obs in observations.values()
                    if obs["role"] == role and obs["depth"] == depth]
            if vals:
                print(f"    {role:<25} {np.mean(vals):8.4f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/results/svb_qwen3_semantic_descent/result.json"
    reanalyze(path)
