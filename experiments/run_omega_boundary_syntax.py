"""Omega Boundary-Preserving Syntax Crossover (Experiment #47).

Fable R3 Rank 1: prerequisite for all #38 (partial identity) claims.

Problem: #38 concatenated continuation+query without separator, creating
malformed Python: 'x' + 'f()' -> 'xf()', 'x = ' + 'f()' -> 'x = f()',
'x = x' + 'f()' -> 'x = xf()'. Only newline made it valid.

This experiment uses PROPER boundary separation for all arms so the
query always starts as valid Python. Includes the malformed original
as an explicit diagnostic to measure confound magnitude.

Design: 6 cells x 2 pair families x 8 arms x 2 readouts = 192.
Arms all end with newline (valid boundary) EXCEPT the glued diagnostic.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv


MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "query_template": "f()\nprint({var})  # Output: ",
    "templates": {
        "depth4_single": (
            "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
            "    def g():\n        {var} = 999\n        def h():\n"
            "            {var} = 9999\n            def k():\n"
            "                {var} = 99999\n                return {var}\n"
            "            k()\n            return {var}\n"
            "        h()\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
    },
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

EQUIV_PAIRS = [
    ("assert_novar_2", "assert_var_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
]

SURFACES = {
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
}

ARMS = {
    "empty": "",
    "bare_var_nl": "{var}\n",
    "bare_var_semi": "{var};\n",
    "identity_nl": "{var} = {var}\n",
    "identity_semi": "{var} = {var};\n",
    "glued_identity": "{var} = {var}",
    "nonvar_expr": "_ = 42\n",
    "pass_stmt": "pass\n",
}


def run_boundary_syntax():
    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    observations = []
    t0 = time.time()
    n_readouts = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)

        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for pair_a, pair_b, pair_name in EQUIV_PAIRS:
            surf_a = SURFACES[pair_a].replace("{var}", var)
            surf_b = SURFACES[pair_b].replace("{var}", var)

            for arm_name, arm_tmpl in ARMS.items():
                arm = arm_tmpl.replace("{var}", var)
                suffix_a = surf_a + arm + query
                suffix_b = surf_b + arm + query

                dist_a = adapter.get_dist_from_state(prefix_state, suffix_a)
                dist_b = adapter.get_dist_from_state(prefix_state, suffix_b)
                d = tv(dist_a, dist_b)
                n_readouts += 2

                has_newline_boundary = arm.endswith("\n") or arm == ""
                observations.append({
                    "depth": depth, "var": var, "val": val,
                    "pair": pair_name,
                    "arm": arm_name,
                    "arm_text": arm,
                    "valid_boundary": has_newline_boundary,
                    "tv": float(d),
                    "dist_a": dist_a.tolist(),
                    "dist_b": dist_b.tolist(),
                })

                if n_readouts % 48 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_readouts}/192 readouts, {elapsed:.0f}s",
                          flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done: {n_readouts} readouts in {elapsed:.1f}s")
    del adapter
    return observations, elapsed


def analyze(observations):
    print("\n=== BOUNDARY-SYNTAX ANALYSIS ===\n")

    for pair_name in ["strong_assert_2", "boundary_pair"]:
        pair_obs = [o for o in observations if o["pair"] == pair_name]
        print(f"\n--- {pair_name} ---")

        baseline_tvs = [o["tv"] for o in pair_obs if o["arm"] == "empty"]
        if baseline_tvs:
            mean_bl = np.mean(baseline_tvs)
            print(f"  Baseline (empty): mean TV = {mean_bl:.6f}")

        for arm_name in ARMS:
            arm_tvs = [o["tv"] for o in pair_obs if o["arm"] == arm_name]
            if not arm_tvs:
                continue
            valid = any(o["valid_boundary"] for o in pair_obs
                        if o["arm"] == arm_name)
            mean_tv = np.mean(arm_tvs)
            max_tv = np.max(arm_tvs)
            ratio = max_tv / mean_bl if mean_bl > 0.001 else float("inf")
            marker = "" if valid else " [MALFORMED]"
            print(f"  {arm_name:20s}: mean={mean_tv:.6f} max={max_tv:.6f} "
                  f"ratio={ratio:.2f}x{marker}")

    print("\n=== KEY COMPARISON: valid vs glued identity ===")
    for pair_name in ["strong_assert_2", "boundary_pair"]:
        pair_obs = [o for o in observations if o["pair"] == pair_name]
        valid_id = [o["tv"] for o in pair_obs if o["arm"] == "identity_nl"]
        glued_id = [o["tv"] for o in pair_obs if o["arm"] == "glued_identity"]
        if valid_id and glued_id:
            print(f"  {pair_name}:")
            print(f"    identity_nl (valid):  mean={np.mean(valid_id):.6f} "
                  f"max={np.max(valid_id):.6f}")
            print(f"    glued (malformed):    mean={np.mean(glued_id):.6f} "
                  f"max={np.max(glued_id):.6f}")
            diff = np.mean(glued_id) - np.mean(valid_id)
            print(f"    difference:           {diff:+.6f} "
                  f"({'glued higher' if diff > 0 else 'valid higher'})")

    print("\n=== CONFOUND CHECK: var-mentioning vs non-var ===")
    for pair_name in ["strong_assert_2", "boundary_pair"]:
        pair_obs = [o for o in observations if o["pair"] == pair_name]
        var_arms = ["bare_var_nl", "bare_var_semi", "identity_nl",
                     "identity_semi"]
        novar_arms = ["nonvar_expr", "pass_stmt"]
        var_tvs = [o["tv"] for o in pair_obs
                   if o["arm"] in var_arms and o["valid_boundary"]]
        novar_tvs = [o["tv"] for o in pair_obs
                     if o["arm"] in novar_arms]
        if var_tvs and novar_tvs:
            print(f"  {pair_name}:")
            print(f"    var-mentioning (valid): mean={np.mean(var_tvs):.6f}")
            print(f"    non-var (valid):        mean={np.mean(novar_tvs):.6f}")
            ratio = np.mean(var_tvs) / np.mean(novar_tvs) if np.mean(novar_tvs) > 0.001 else float("inf")
            print(f"    ratio: {ratio:.2f}x")

    return {
        "type": "boundary_syntax_crossover",
        "n_observations": len(observations),
        "arms": list(ARMS.keys()),
        "pairs": [p[2] for p in EQUIV_PAIRS],
    }


if __name__ == "__main__":
    out_dir = Path("experiments/results/omega_boundary_syntax")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Omega Boundary-Preserving Syntax Crossover (Exp #47) ===")
    print("Testing: does variable amplification survive valid Python?\n")

    observations, elapsed = run_boundary_syntax()

    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f, indent=2)

    summary = analyze(observations)
    summary["n_readouts"] = len(observations) * 2
    summary["elapsed_s"] = elapsed

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")
