"""Omega Value Sensitivity: Does the outer value affect the omega effect?

Tests a wider range of outer values (0-20) at depth 4 with core surfaces/continuations.
Key question: Is amplification uniform across values, or do certain values resist it?

Fast: 1 depth × 3 vars × 10 vals × 6 surfaces × 6 conts = 1,080 readouts (~5.5 min).
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
    "rewrite": "{var} = {var}\n",
    "boundary": "\n",
}

CONTINUATIONS = {
    "none": "",
    "neutral": "# State unchanged.\n",
    "heavy": "# Reassigning {var} now.\n",
    "pass": "pass\n",
    "identity": "{var} = {var}\n",
    "neutral_then_heavy": "# State unchanged.\n# Reassigning {var} now.\n",
}

VALUES = [0, 1, 2, 3, 5, 7, 10, 13, 17, 20]

CELLS = [
    (4, v, val)
    for v in ["x", "y", "z"]
    for val in VALUES
]

EQUIV_PAIRS = [
    ("assert_novar", "assert_var", "strong_assert"),
]


def run_value_sweep():
    print("=== Omega Value Sensitivity Sweep ===")
    n_total = len(CELLS) * len(SURFACES) * len(CONTINUATIONS)
    print(f"Cells: {len(CELLS)} (values: {VALUES})")
    print(f"Surfaces: {len(SURFACES)}, Continuations: {len(CONTINUATIONS)}")
    print(f"Total readouts: {n_total}")
    print()

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    observations = []
    t0 = time.time()
    n = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
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

    result_dir = Path("experiments/results/omega_values")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    tau = 0.01
    print(f"\n=== Value × Continuation Table (tau={tau}) ===\n")

    for pair_a, pair_b, pair_name in EQUIV_PAIRS:
        print(f"--- Pair: {pair_name} ---")
        header = f"{'val':>5s}"
        for cont_name in sorted(CONTINUATIONS.keys()):
            header += f" {cont_name:>12s}"
        print(header)

        for val in VALUES:
            row = f"{val:5d}"
            for cont_name in sorted(CONTINUATIONS.keys()):
                tvs = []
                for var in ["x", "y", "z"]:
                    ka_none = (var, val, pair_a, "none")
                    kb_none = (var, val, pair_b, "none")
                    if ka_none not in obs_idx or kb_none not in obs_idx:
                        continue
                    d_pre = 0.5 * np.abs(obs_idx[ka_none] - obs_idx[kb_none]).sum()
                    if d_pre > tau:
                        continue
                    ka = (var, val, pair_a, cont_name)
                    kb = (var, val, pair_b, cont_name)
                    if ka in obs_idx and kb in obs_idx:
                        d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                        tvs.append(d)
                if tvs:
                    row += f" {np.mean(tvs):12.6f}"
                else:
                    row += f" {'n/a':>12s}"
            print(row)
        print()

    val_summary = {}
    for val in VALUES:
        cont_data = {}
        for cont_name in CONTINUATIONS:
            tvs = []
            for pair_a, pair_b, _ in EQUIV_PAIRS:
                for var in ["x", "y", "z"]:
                    ka_none = (var, val, pair_a, "none")
                    kb_none = (var, val, pair_b, "none")
                    if ka_none not in obs_idx or kb_none not in obs_idx:
                        continue
                    d_pre = 0.5 * np.abs(obs_idx[ka_none] - obs_idx[kb_none]).sum()
                    if d_pre > tau:
                        continue
                    ka = (var, val, pair_a, cont_name)
                    kb = (var, val, pair_b, cont_name)
                    if ka in obs_idx and kb in obs_idx:
                        d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                        tvs.append(d)
            if tvs:
                cont_data[cont_name] = {
                    "mean": float(np.mean(tvs)), "max": float(np.max(tvs)),
                    "n": len(tvs),
                }
        val_summary[val] = cont_data

    with open(result_dir / "value_summary.json", "w") as f:
        json.dump(val_summary, f, indent=2)

    eligible_counts = {}
    for val in VALUES:
        count = 0
        for pair_a, pair_b, _ in EQUIV_PAIRS:
            for var in ["x", "y", "z"]:
                ka = (var, val, pair_a, "none")
                kb = (var, val, pair_b, "none")
                if ka in obs_idx and kb in obs_idx:
                    d = 0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum()
                    if d <= tau:
                        count += 1
        eligible_counts[val] = count

    print("Eligible cells per value:")
    for val, count in sorted(eligible_counts.items()):
        print(f"  val={val}: {count}/3 eligible")

    print(f"\nSaved {n} observations to {result_dir}/")
    del adapter
    return observations


if __name__ == "__main__":
    run_value_sweep()
