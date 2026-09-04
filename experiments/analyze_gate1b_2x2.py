"""Gate 1b: 2x2 ANOVA for content x variable-mention.

Tests whether the ASSERT vs MISLEADING_ASSERT difference is driven by
content (what the comment says) or variable-mention (whether {var} appears).

Four conditions:
  ASSERT:                   true assertion, no var mention
  ASSERT_VAR:               true assertion, var mention
  MISLEADING_ASSERT:        misleading assertion, var mention
  MISLEADING_ASSERT_NOVAR:  misleading assertion, no var mention

Content effect: (MISLEADING_ASSERT + MISLEADING_ASSERT_NOVAR) - (ASSERT + ASSERT_VAR)
Var-mention effect: (ASSERT_VAR + MISLEADING_ASSERT) - (ASSERT + MISLEADING_ASSERT_NOVAR)
"""
import json
import sys

import numpy as np
from collections import defaultdict


def analyze_2x2(result_path):
    with open(result_path) as f:
        data = json.load(f)

    observations = data["observations"]

    cells = {
        "ASSERT": [],
        "ASSERT_VAR": [],
        "MISLEADING_ASSERT": [],
        "MISLEADING_ASSERT_NOVAR": [],
    }

    for key, obs in observations.items():
        if obs["role"] not in cells:
            continue
        if obs["val"] == 9:
            continue
        dist = np.array(obs["dist"])
        L = float(dist[9])
        cells[obs["role"]].append(L)

    print("=== GATE 1b: 2x2 CONTENT x VAR-MENTION ===\n")
    print(f"  {'Condition':<30} {'n':>5} {'L_mean':>8} {'L_std':>8} {'L_med':>8}")
    for name in ["ASSERT", "ASSERT_VAR", "MISLEADING_ASSERT", "MISLEADING_ASSERT_NOVAR"]:
        vals = cells[name]
        if vals:
            arr = np.array(vals)
            print(f"  {name:<30} {len(vals):>5} {arr.mean():8.4f} {arr.std():8.4f} {np.median(arr):8.4f}")

    a = np.mean(cells["ASSERT"])
    av = np.mean(cells["ASSERT_VAR"])
    m = np.mean(cells["MISLEADING_ASSERT"])
    mn = np.mean(cells["MISLEADING_ASSERT_NOVAR"])

    content_effect = ((m + mn) / 2) - ((a + av) / 2)
    var_effect = ((av + m) / 2) - ((a + mn) / 2)
    interaction = (m - mn) - (av - a)

    print(f"\n  Content main effect (misleading - true):  {content_effect:+.4f}")
    print(f"  Var-mention main effect (var - novar):    {var_effect:+.4f}")
    print(f"  Interaction (content x var):              {interaction:+.4f}")

    grand_mean = (a + av + m + mn) / 4
    ss_content = 2 * len(cells["ASSERT"]) * content_effect**2
    ss_var = 2 * len(cells["ASSERT"]) * var_effect**2

    print(f"\n  |Content effect| / |Var effect| = {abs(content_effect)/max(abs(var_effect), 1e-10):.1f}x")

    print("\n=== VERDICT ===\n")
    if abs(content_effect) > 2 * abs(var_effect):
        print(f"  -> CONTENT DOMINATES ({abs(content_effect)/abs(var_effect):.1f}x)")
        print(f"     The model reads what comments SAY, not just whether they mention the variable.")
        print(f"     This supports intensional content sensitivity.")
    elif abs(var_effect) > 2 * abs(content_effect):
        print(f"  -> VAR-MENTION DOMINATES ({abs(var_effect)/abs(content_effect):.1f}x)")
        print(f"     The model responds to variable name presence, not comment meaning.")
        print(f"     This is a lexical priming effect, not intensional descent.")
    elif abs(content_effect) > 0.01 and abs(var_effect) > 0.01:
        print(f"  -> BOTH CONTRIBUTE")
        print(f"     Content and var-mention both affect L. Need to quantify relative importance.")
    elif abs(content_effect) < 0.01 and abs(var_effect) < 0.01:
        print(f"  -> NEITHER EFFECT (both < 0.01)")
        print(f"     The Gate 1 effect may not replicate or is confounded differently.")
    else:
        print(f"  -> AMBIGUOUS: content={content_effect:+.4f}, var={var_effect:+.4f}")

    print("\n=== PER-DEPTH BREAKDOWN ===\n")
    depth_cells = defaultdict(lambda: defaultdict(list))
    for key, obs in observations.items():
        if obs["role"] not in cells or obs["val"] == 9:
            continue
        dist = np.array(obs["dist"])
        L = float(dist[9])
        depth_cells[obs["depth"]][obs["role"]].append(L)

    print(f"  {'Depth':>6} {'content':>9} {'var_ment':>9} {'interact':>9} {'|c/v|':>7}")
    for d in sorted(depth_cells):
        dc = depth_cells[d]
        if all(dc[r] for r in cells):
            da = np.mean(dc["ASSERT"])
            dav = np.mean(dc["ASSERT_VAR"])
            dm = np.mean(dc["MISLEADING_ASSERT"])
            dmn = np.mean(dc["MISLEADING_ASSERT_NOVAR"])
            dc_eff = ((dm + dmn) / 2) - ((da + dav) / 2)
            dv_eff = ((dav + dm) / 2) - ((da + dmn) / 2)
            di_eff = (dm - dmn) - (dav - da)
            ratio = abs(dc_eff) / max(abs(dv_eff), 1e-10)
            print(f"  d{d:>5} {dc_eff:+9.4f} {dv_eff:+9.4f} {di_eff:+9.4f} {ratio:7.1f}x")

    print("\n=== HOLDOUT GENERALIZATION ===\n")
    for split in ["train", "holdout"]:
        split_cells = defaultdict(list)
        for key, obs in observations.items():
            if obs["role"] not in cells or obs["val"] == 9 or obs["split"] != split:
                continue
            dist = np.array(obs["dist"])
            L = float(dist[9])
            split_cells[obs["role"]].append(L)

        if all(split_cells[r] for r in cells):
            sa = np.mean(split_cells["ASSERT"])
            sav = np.mean(split_cells["ASSERT_VAR"])
            sm = np.mean(split_cells["MISLEADING_ASSERT"])
            smn = np.mean(split_cells["MISLEADING_ASSERT_NOVAR"])
            sc = ((sm + smn) / 2) - ((sa + sav) / 2)
            sv = ((sav + sm) / 2) - ((sa + smn) / 2)
            print(f"  {split:>8}: content={sc:+.4f}, var_mention={sv:+.4f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/results/svb_qwen3_gate1b/result.json"
    analyze_2x2(path)
