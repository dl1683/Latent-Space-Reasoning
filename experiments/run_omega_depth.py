"""Omega Depth Sensitivity: Does amplification scale with nesting depth?

Tests the same surface+continuation at depths 2, 3, 4.
Key question: Does deeper nesting amplify the omega effect?

Fast experiment: 3 depths × 6 vars × 4 vals × 8 surfaces × 8 continuations = 4,608 readouts
But with KV caching on 72 prefixes, should be ~23 min at 3.3/s.

Actually let's keep it fast: 3 depths × 3 vars × 2 vals × 8 surfaces × 8 conts = 1,152 readouts (~6 min).
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix, build_query


MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "query_template": "f()\nprint({var})  # Output: ",
    "templates": {
        "depth2_single": (
            "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
            "    def g():\n        {var} = 999\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
        "depth3_single": (
            "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
            "    def g():\n        {var} = 999\n        def h():\n"
            "            {var} = 9999\n            return {var}\n"
            "        h()\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
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

SURFACES = {
    "none": "",
    "assert_novar": "# State unchanged.\n",
    "assert_var": "# {var} stays the same.\n",
    "mislead_var": "# Reassigning {var} now.\n",
    "mislead_novar": "# Overwriting the variable.\n",
    "rewrite": "{var} = {var}\n",
    "observe": "_ = {var}\n",
    "boundary": "\n",
}

CONTINUATIONS = {
    "none": "",
    "neutral": "# No changes.\n",
    "heavy": "# Reassigning {var} now.\n",
    "pass": "pass\n",
    "identity": "{var} = {var}\n",
    "bare_nl": "\n",
    "double_neutral": "# No changes.\n# No changes.\n",
    "neutral_then_heavy": "# No changes.\n# Reassigning {var} now.\n",
}

CELLS = [
    (d, v, val)
    for d in [2, 3, 4]
    for v in ["x", "y", "z"]
    for val in [1, 5]
]

EQUIV_PAIRS = [
    ("assert_novar", "assert_var", "strong_assert_2"),
    ("mislead_var", "mislead_novar", "misleading"),
    ("none", "boundary", "baseline_boundary"),
]


def run_depth_sweep():
    print("=== Omega Depth Sensitivity Sweep ===")
    n_total = len(CELLS) * len(SURFACES) * len(CONTINUATIONS)
    print(f"Cells: {len(CELLS)} ({len(set(d for d,_,_ in CELLS))} depths)")
    print(f"Surfaces: {len(SURFACES)}, Continuations: {len(CONTINUATIONS)}")
    print(f"Total readouts: {n_total}")
    print()

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    observations = []
    t0 = time.time()
    n = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"][f"depth{depth}_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_name, surf_tmpl in SURFACES.items():
            surface_text = surf_tmpl.replace("{var}", var)

            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                full_suffix = surface_text + cont_text + query

                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                n += 1

                observations.append({
                    "depth": depth, "var": var, "val": val,
                    "surface": surf_name, "continuation": cont_name,
                    "dist": dist.tolist(),
                })

                if n % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n}/{n_total} readouts, {elapsed:.0f}s, "
                          f"{n/elapsed:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    result_dir = Path("experiments/results/omega_depth")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    tau = 0.01
    print(f"\n=== Depth × Continuation Omega Table (tau={tau}) ===\n")

    for pair_a, pair_b, pair_name in EQUIV_PAIRS:
        print(f"--- Pair: {pair_name} ({pair_a} vs {pair_b}) ---")
        print(f"{'Continuation':25s} {'d2_mean':>10s} {'d2_max':>10s} "
              f"{'d3_mean':>10s} {'d3_max':>10s} "
              f"{'d4_mean':>10s} {'d4_max':>10s}")

        for cont_name in sorted(CONTINUATIONS.keys()):
            row_parts = []
            for depth in [2, 3, 4]:
                tvs = []
                for _, var, val in [c for c in CELLS if c[0] == depth]:
                    ka_none = (depth, var, val, pair_a, "none")
                    kb_none = (depth, var, val, pair_b, "none")
                    if ka_none not in obs_idx or kb_none not in obs_idx:
                        continue
                    d_pre = 0.5 * np.abs(obs_idx[ka_none] - obs_idx[kb_none]).sum()
                    if d_pre > tau:
                        continue

                    ka = (depth, var, val, pair_a, cont_name)
                    kb = (depth, var, val, pair_b, cont_name)
                    if ka in obs_idx and kb in obs_idx:
                        d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                        tvs.append(d)

                if tvs:
                    row_parts.append(f"{np.mean(tvs):10.6f} {np.max(tvs):10.6f}")
                else:
                    row_parts.append(f"{'n/a':>10s} {'n/a':>10s}")

            print(f"{cont_name:25s} {' '.join(row_parts)}")
        print()

    depth_summary = {}
    for depth in [2, 3, 4]:
        depth_conts = {}
        for cont_name in CONTINUATIONS:
            all_tvs = []
            for pair_a, pair_b, pair_name in EQUIV_PAIRS:
                for _, var, val in [c for c in CELLS if c[0] == depth]:
                    ka_none = (depth, var, val, pair_a, "none")
                    kb_none = (depth, var, val, pair_b, "none")
                    if ka_none not in obs_idx or kb_none not in obs_idx:
                        continue
                    d_pre = 0.5 * np.abs(obs_idx[ka_none] - obs_idx[kb_none]).sum()
                    if d_pre > tau:
                        continue
                    ka = (depth, var, val, pair_a, cont_name)
                    kb = (depth, var, val, pair_b, cont_name)
                    if ka in obs_idx and kb in obs_idx:
                        d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                        all_tvs.append(d)
            if all_tvs:
                depth_conts[cont_name] = {
                    "mean": float(np.mean(all_tvs)),
                    "max": float(np.max(all_tvs)),
                    "n": len(all_tvs),
                }
        depth_summary[f"depth{depth}"] = depth_conts

    with open(result_dir / "depth_summary.json", "w") as f:
        json.dump(depth_summary, f, indent=2)

    print(f"\nSaved {n} observations to {result_dir}/")
    del adapter
    return observations


if __name__ == "__main__":
    run_depth_sweep()
