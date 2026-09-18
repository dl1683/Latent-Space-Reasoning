"""Pair x Depth Interaction: Do new pairs work at all depths?

Tests the 5 best surface pairs across depths 2, 3, and 4.
If pairs that work at depth 4 also work at depth 2, the finding
is more robust. If they only work at depth 4, depth dependence
is a confound.

3 depths x 3 vars x 2 vals x 5 pairs x 2 sides x 4 continuations = 720 readouts
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix


TEMPLATES = {
    2: (
        "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
        "    def g():\n        {var} = 999\n        return {var}\n"
        "    g()\n    return {var}\n"
    ),
    3: (
        "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
        "    def g():\n        {var} = 999\n        def h():\n"
        "            {var} = 9999\n            return {var}\n"
        "        h()\n        return {var}\n"
        "    g()\n    return {var}\n"
    ),
    4: (
        "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
        "    def g():\n        {var} = 999\n        def h():\n"
        "            {var} = 9999\n            def k():\n"
        "                {var} = 99999\n                return {var}\n"
        "            k()\n            return {var}\n"
        "        h()\n        return {var}\n"
        "    g()\n    return {var}\n"
    ),
}

QUERY = "f()\nprint({var})  # Output: "

SURFACE_PAIRS = [
    ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ("# ok\n", "# okay\n", "terse_ok_okay"),
    ("# OK.\n", "# Good.\n", "minimal_ok_good"),
    ("# OK.\n", "# Fine.\n", "minimal_ok_fine"),
    ("# Confirmed.\n", "# Verified.\n", "novar_confirmed"),
]

CONTINUATIONS = {
    "none": "",
    "identity": "{var} = {var}\n",
    "type_check": "type({var})\n",
    "heavy": "# Reassigning {var} now.\n",
}

CELLS_PER_DEPTH = [
    ("x", 1), ("x", 5),
    ("y", 2), ("y", 6),
    ("z", 3), ("z", 7),
]

MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
}


def run_pair_depth():
    print("=== Pair x Depth Interaction ===")
    n_total = len(TEMPLATES) * len(CELLS_PER_DEPTH) * len(SURFACE_PAIRS) * 2 * len(CONTINUATIONS)
    print(f"Depths: {len(TEMPLATES)}, Cells/depth: {len(CELLS_PER_DEPTH)}, "
          f"Pairs: {len(SURFACE_PAIRS)}, Continuations: {len(CONTINUATIONS)}")
    print(f"Total readouts: {n_total}")
    print()

    adapter = ModelAdapter(MODEL_CFG)
    observations = []
    t0 = time.time()
    n = 0

    for depth, tmpl in TEMPLATES.items():
        for var, val in CELLS_PER_DEPTH:
            prefix_text = build_prefix(tmpl, var=var, outer_val=val)
            query = QUERY.replace("{var}", var)
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
                            "depth": depth, "var": var, "val": val,
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

    result_dir = Path("experiments/results/omega_pair_depth")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"], obs["pair"],
               obs["side"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    for tau in [0.01, 0.02, 0.05]:
        print(f"\n=== Depth x Pair Eligibility (tau={tau}) ===")
        print(f"{'Depth':<6s} {'Pair':<20s} {'eligible':>8s} {'total':>6s} "
              f"{'%':>6s}")
        print("-" * 50)

        for depth in sorted(TEMPLATES.keys()):
            for pair_name in [p[2] for p in SURFACE_PAIRS]:
                eligible = 0
                total = 0
                for var, val in CELLS_PER_DEPTH:
                    ka = (depth, var, val, pair_name, "a", "none")
                    kb = (depth, var, val, pair_name, "b", "none")
                    if ka in obs_idx and kb in obs_idx:
                        total += 1
                        d = 0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum()
                        if d <= tau:
                            eligible += 1
                if total > 0:
                    pct = 100 * eligible / total
                    print(f"  d={depth}  {pair_name:<20s} {eligible:>8d} "
                          f"{total:>6d} {pct:5.0f}%")

        print(f"\n--- Omega by Depth x Pair (tau={tau}) ---")
        for depth in sorted(TEMPLATES.keys()):
            for pair_name in [p[2] for p in SURFACE_PAIRS]:
                for cont_name in sorted(CONTINUATIONS.keys()):
                    if cont_name == "none":
                        continue
                    tvs = []
                    for var, val in CELLS_PER_DEPTH:
                        ka_base = (depth, var, val, pair_name, "a", "none")
                        kb_base = (depth, var, val, pair_name, "b", "none")
                        if ka_base not in obs_idx or kb_base not in obs_idx:
                            continue
                        d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
                        if d_pre > tau:
                            continue
                        ka = (depth, var, val, pair_name, "a", cont_name)
                        kb = (depth, var, val, pair_name, "b", cont_name)
                        if ka in obs_idx and kb in obs_idx:
                            d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                            tvs.append(d)

                    if tvs:
                        mean_tv = float(np.mean(tvs))
                        max_tv = float(np.max(tvs))
                        status = "PRESERVE" if max_tv <= tau else "AMPLIFY"
                        print(f"  d={depth} {pair_name:<20s} {cont_name:<12s} "
                              f"mean={mean_tv:.5f} max={max_tv:.5f} "
                              f"n={len(tvs)} {status}")

    print(f"\nSaved {n} observations to {result_dir}/")
    del adapter
    return observations


if __name__ == "__main__":
    run_pair_depth()
