"""Omega Template Sensitivity: Does the code structure affect omega?

Tests different code templates beyond depth variation:
- Single variable: standard nested shadowing
- Two variables: x and y shadowing independently
- Linear (no nesting): sequential shadowing without def
- Loop: variable inside loop body
- Class: variable in method

Key question: Is the omega effect specific to nested-function shadowing,
or does it appear in other code structures?

Fast: 5 templates × 3 vars × 2 vals × 2 pairs × 6 conts = 360 readouts (~2 min)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix


MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "query_template": "f()\nprint({var})  # Output: ",
}

TEMPLATES = {
    "depth4_nested": (
        "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
        "    def g():\n        {var} = 999\n        def h():\n"
        "            {var} = 9999\n            def k():\n"
        "                {var} = 99999\n                return {var}\n"
        "            k()\n            return {var}\n"
        "        h()\n        return {var}\n"
        "    g()\n    return {var}\n"
    ),
    "depth2_nested": (
        "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
        "    def g():\n        {var} = 999\n        return {var}\n"
        "    g()\n    return {var}\n"
    ),
    "linear_shadow": (
        "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
        "    {var} = 999\n    {var} = 9999\n"
        "    return {var}\n"
    ),
    "loop_body": (
        "{var} = {outer_val}\ndef f():\n"
        "    for i in range(3):\n        {var} = i * 10\n"
        "    return {var}\n"
    ),
    "conditional": (
        "{var} = {outer_val}\ndef f():\n"
        "    if True:\n        {var} = 99\n"
        "    else:\n        {var} = 999\n"
        "    return {var}\n"
    ),
}

QUERY_TEMPLATES = {
    "depth4_nested": "f()\nprint({var})  # Output: ",
    "depth2_nested": "f()\nprint({var})  # Output: ",
    "linear_shadow": "f()\nprint({var})  # Output: ",
    "loop_body": "f()\nprint({var})  # Output: ",
    "conditional": "f()\nprint({var})  # Output: ",
}

SURFACE_PAIRS = [
    ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ("# Values preserved.\n", "# {var} retains its value.\n", "moderate_assert"),
]

CONTINUATIONS = {
    "none": "",
    "neutral": "# State unchanged.\n",
    "heavy": "# Reassigning {var} now.\n",
    "pass": "pass\n",
    "identity": "{var} = {var}\n",
    "type_check": "type({var})\n",
}

CELLS = [
    (v, val)
    for v in ["x", "y", "z"]
    for val in [1, 5]
]


def run_template_sweep():
    print("=== Omega Template Sensitivity Sweep ===")
    n_total = len(TEMPLATES) * len(CELLS) * len(SURFACE_PAIRS) * 2 * len(CONTINUATIONS)
    print(f"Templates: {len(TEMPLATES)}, Cells: {len(CELLS)}, "
          f"Pairs: {len(SURFACE_PAIRS)}, Continuations: {len(CONTINUATIONS)}")
    print(f"Total readouts: {n_total}")
    print()

    adapter = ModelAdapter(MODEL_CFG)
    observations = []
    t0 = time.time()
    n = 0

    for tmpl_name, tmpl in TEMPLATES.items():
        query_tmpl = QUERY_TEMPLATES[tmpl_name]

        for var, val in CELLS:
            prefix_text = build_prefix(tmpl, var=var, outer_val=val)
            query = query_tmpl.replace("{var}", var)
            prefix_state = adapter.get_state_after_prefix(prefix_text)

            for surf_a_tmpl, surf_b_tmpl, pair_name in SURFACE_PAIRS:
                for side, surf_tmpl in [("a", surf_a_tmpl), ("b", surf_b_tmpl)]:
                    surface_text = surf_tmpl.replace("{var}", var)

                    for cont_name, cont_tmpl in CONTINUATIONS.items():
                        cont_text = cont_tmpl.replace("{var}", var)
                        full_suffix = surface_text + cont_text + query

                        dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                        n += 1

                        observations.append({
                            "template": tmpl_name, "var": var, "val": val,
                            "pair": pair_name, "side": side,
                            "continuation": cont_name,
                            "dist": dist.tolist(),
                        })

                        if n % 100 == 0:
                            elapsed = time.time() - t0
                            print(f"  {n}/{n_total} readouts, {elapsed:.0f}s, "
                                  f"{n/elapsed:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    result_dir = Path("experiments/results/omega_templates")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["template"], obs["var"], obs["val"],
               obs["pair"], obs["side"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    tau = 0.01
    print(f"\n=== Template × Continuation Omega (tau={tau}) ===\n")

    tmpl_stats = {}
    for tmpl_name in TEMPLATES:
        print(f"--- {tmpl_name} ---")
        for pair_name in [p[2] for p in SURFACE_PAIRS]:
            for cont_name in sorted(CONTINUATIONS.keys()):
                tvs = []
                for var, val in CELLS:
                    ka_base = (tmpl_name, var, val, pair_name, "a", "none")
                    kb_base = (tmpl_name, var, val, pair_name, "b", "none")
                    if ka_base not in obs_idx or kb_base not in obs_idx:
                        continue
                    d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
                    if d_pre > tau:
                        continue

                    ka = (tmpl_name, var, val, pair_name, "a", cont_name)
                    kb = (tmpl_name, var, val, pair_name, "b", cont_name)
                    if ka in obs_idx and kb in obs_idx:
                        d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                        tvs.append(d)

                if tvs:
                    mean_tv = float(np.mean(tvs))
                    max_tv = float(np.max(tvs))
                    status = "PRESERVE" if max_tv <= tau else "AMPLIFY"
                    print(f"  {pair_name:20s} {cont_name:15s} "
                          f"mean={mean_tv:.6f} max={max_tv:.6f} n={len(tvs)} {status}")
                    tmpl_stats.setdefault(tmpl_name, {}).setdefault(pair_name, {})[cont_name] = {
                        "mean": mean_tv, "max": max_tv, "n": len(tvs),
                        "preserves": bool(max_tv <= tau),
                    }
        print()

    with open(result_dir / "template_stats.json", "w") as f:
        json.dump(tmpl_stats, f, indent=2)

    print(f"\nSaved {n} observations to {result_dir}/")
    del adapter
    return observations


if __name__ == "__main__":
    run_template_sweep()
