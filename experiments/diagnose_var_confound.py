"""Diagnose variable-mention confound from Gate 1 results.

Within MISLEADING_ASSERT, compare L for surfaces that mention {var}
vs the one that doesn't ("# Updating the value.\n").
"""
import json
import sys
import numpy as np
from collections import defaultdict


def diagnose(result_path):
    with open(result_path) as f:
        data = json.load(f)

    observations = data["observations"]

    misleading_var = []
    misleading_novar = []
    assert_all = []

    for key, obs in observations.items():
        if obs["val"] == 9:
            continue
        dist = np.array(obs["dist"])
        L = float(dist[9])

        if obs["role"] == "MISLEADING_ASSERT":
            if "{var}" in obs["surface"]:
                misleading_var.append(L)
            else:
                misleading_novar.append(L)
        elif obs["role"] == "ASSERT":
            assert_all.append(L)

    print("=== VARIABLE-MENTION CONFOUND DIAGNOSTIC ===\n")
    print(f"  ASSERT (no var mention):           L={np.mean(assert_all):.4f} +/- {np.std(assert_all):.4f} (n={len(assert_all)})")
    print(f"  MISLEADING_ASSERT (var mention):   L={np.mean(misleading_var):.4f} +/- {np.std(misleading_var):.4f} (n={len(misleading_var)})")
    print(f"  MISLEADING_ASSERT (no var):        L={np.mean(misleading_novar):.4f} +/- {np.std(misleading_novar):.4f} (n={len(misleading_novar)})")

    diff_content = np.mean(misleading_novar) - np.mean(assert_all)
    diff_varmention = np.mean(misleading_var) - np.mean(misleading_novar)

    print(f"\n  Content effect (misleading_novar - assert): {diff_content:+.4f}")
    print(f"  Var-mention effect (misleading_var - misleading_novar): {diff_varmention:+.4f}")

    if abs(diff_content) > 2 * abs(diff_varmention):
        print(f"\n  -> CONTENT DOMINATES: {abs(diff_content)/abs(diff_varmention):.1f}x larger than var-mention effect")
    elif abs(diff_varmention) > 2 * abs(diff_content):
        print(f"\n  -> VAR-MENTION DOMINATES: {abs(diff_varmention)/abs(diff_content):.1f}x larger than content effect")
    else:
        print(f"\n  -> MIXED: both effects are comparable")

    print("\n=== PER-SURFACE BREAKDOWN (MISLEADING_ASSERT) ===\n")
    surface_L = defaultdict(list)
    for key, obs in observations.items():
        if obs["role"] == "MISLEADING_ASSERT" and obs["val"] != 9:
            dist = np.array(obs["dist"])
            L = float(dist[9])
            label = obs["surface"].strip()
            if len(label) > 40:
                label = label[:37] + "..."
            surface_L[label].append(L)

    print(f"  {'Surface':<45} {'L_mean':>8} {'n':>5} {'has_var':>8}")
    for surf, vals in sorted(surface_L.items(), key=lambda x: np.mean(x[1])):
        has_var = "YES" if "{var}" in surf else "no"
        print(f"  {surf:<45} {np.mean(vals):8.4f} {len(vals):>5} {has_var:>8}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/results/svb_qwen3_semantic_descent/result.json"
    diagnose(path)
