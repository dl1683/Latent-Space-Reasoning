"""Omega Perturbation Diversity: Does omega generalize to novel perturbation types?

Tests 14 novel continuation types beyond the original 6, covering:
- Destructive: del, None assignment, reassignment to 0
- Arithmetic identity: +0, *1, //1
- Observation: print, repr, str, len
- Control flow: assert, if-pass, while-False
- Collection: [var], {var: var}

6 cells x 2 pairs x 2 sides x 15 continuations = 360 readouts (~2 min)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix


TEMPLATE = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            def k():\n"
    "                {var} = 99999\n                return {var}\n"
    "            k()\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

QUERY = "f()\nprint({var})  # Output: "

SURFACE_PAIRS = [
    ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ("# OK.\n", "# Good.\n", "minimal_ok_good"),
]

CONTINUATIONS = {
    "none": "",
    "identity": "{var} = {var}\n",
    "del_var": "del {var}\n",
    "assign_none": "{var} = None\n",
    "assign_zero": "{var} = 0\n",
    "add_zero": "{var} = {var} + 0\n",
    "mul_one": "{var} = {var} * 1\n",
    "floor_one": "{var} = {var} // 1\n",
    "print_var": "print({var})\n",
    "repr_var": "repr({var})\n",
    "str_var": "str({var})\n",
    "assert_var": "assert {var}\n",
    "if_pass": "if {var}: pass\n",
    "list_wrap": "[{var}]\n",
    "bool_var": "bool({var})\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 6), (4, "y", 8),
]


def run_perturbation_diversity():
    print("=== Omega Perturbation Diversity ===")
    result_dir = Path("experiments/results/omega_perturbation_diversity")
    result_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "query_template": QUERY,
        "templates": {"depth4_single": TEMPLATE},
    }

    n_total = len(CELLS) * len(SURFACE_PAIRS) * 2 * len(CONTINUATIONS)
    print(f"Readouts: {n_total}")

    adapter = ModelAdapter(cfg)
    observations = []
    t0 = time.time()
    n = 0

    for depth, var, val in CELLS:
        prefix_text = build_prefix(TEMPLATE, var=var, outer_val=val)
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

    elapsed = time.time() - t0
    print(f"Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    analyze_diversity(observations)

    del adapter
    print(f"\nSaved {len(observations)} observations to {result_dir}/")
    return observations


def analyze_diversity(observations):
    print("\n=== Perturbation Diversity Analysis ===")

    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["pair"], obs["side"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    for tau in [0.01, 0.02, 0.05]:
        print(f"\n--- tau={tau} ---")
        print(f"{'Continuation':<18s} {'Pair':<20s} "
              f"{'mean':>8s} {'max':>8s} {'n':>3s} {'status':>8s}")

        for pair_name in [p[2] for p in SURFACE_PAIRS]:
            for cont_name in sorted(CONTINUATIONS.keys()):
                if cont_name == "none":
                    continue
                tvs = []
                for _, var, val in CELLS:
                    ka_base = (var, val, pair_name, "a", "none")
                    kb_base = (var, val, pair_name, "b", "none")
                    if ka_base not in obs_idx or kb_base not in obs_idx:
                        continue
                    d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
                    if d_pre > tau:
                        continue
                    ka = (var, val, pair_name, "a", cont_name)
                    kb = (var, val, pair_name, "b", cont_name)
                    if ka in obs_idx and kb in obs_idx:
                        d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                        tvs.append(d)

                if tvs:
                    mean_tv = float(np.mean(tvs))
                    max_tv = float(np.max(tvs))
                    status = "PRESERVE" if max_tv <= tau else "AMPLIFY"
                    print(f"  {cont_name:<16s} {pair_name:<20s} "
                          f"{mean_tv:8.5f} {max_tv:8.5f} {len(tvs):3d} "
                          f"{status:>8s}")

        print(f"\n  Classification at tau={tau}:")
        for category, conts in [
            ("Semantic identity", ["identity", "add_zero", "mul_one", "floor_one"]),
            ("Observation only", ["print_var", "repr_var", "str_var", "bool_var"]),
            ("Destructive", ["del_var", "assign_none", "assign_zero"]),
            ("Control flow", ["assert_var", "if_pass"]),
            ("Collection wrap", ["list_wrap"]),
        ]:
            amplify_count = 0
            total_count = 0
            for cont_name in conts:
                for pair_name in [p[2] for p in SURFACE_PAIRS]:
                    tvs = []
                    for _, var, val in CELLS:
                        ka_base = (var, val, pair_name, "a", "none")
                        kb_base = (var, val, pair_name, "b", "none")
                        if ka_base not in obs_idx or kb_base not in obs_idx:
                            continue
                        d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
                        if d_pre > tau:
                            continue
                        ka = (var, val, pair_name, "a", cont_name)
                        kb = (var, val, pair_name, "b", cont_name)
                        if ka in obs_idx and kb in obs_idx:
                            d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                            tvs.append(d)
                    if tvs:
                        total_count += 1
                        if max(tvs) > tau:
                            amplify_count += 1
            if total_count > 0:
                print(f"    {category}: {amplify_count}/{total_count} amplify")


if __name__ == "__main__":
    run_perturbation_diversity()
