"""Omega Multi-Variable Scope: Does a second variable in scope change omega?

Prior experiments use single-variable code. Here we test whether having a
second variable in scope changes the amplification pattern for the first.

Template A: two variables, perturbation targets var1
Template B: two variables interleaved, perturbation targets var1

2 templates x 3 cell configs x 2 pairs x 2 sides x 6 continuations = 144 readouts
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix


TEMPLATE_DUAL_INDEPENDENT = (
    "{var} = {outer_val}\nother = 42\n"
    "def f():\n    {var} = 99\n    other = 43\n"
    "    def g():\n        {var} = 999\n        other = 44\n"
    "        def h():\n            {var} = 9999\n            other = 45\n"
    "            def k():\n                {var} = 99999\n                other = 46\n"
    "                return {var}\n"
    "            k()\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

TEMPLATE_DUAL_INTERLEAVED = (
    "{var} = {outer_val}\nother = {var} + 1\n"
    "def f():\n    {var} = 99\n    other = {var} * 2\n"
    "    def g():\n        {var} = 999\n        other = {var} - 1\n"
    "        def h():\n            {var} = 9999\n            other = {var} + 3\n"
    "            def k():\n                {var} = 99999\n                other = {var}\n"
    "                return {var}\n"
    "            k()\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

TEMPLATES = {
    "dual_independent": TEMPLATE_DUAL_INDEPENDENT,
    "dual_interleaved": TEMPLATE_DUAL_INTERLEAVED,
}

QUERY = "f()\nprint({var})  # Output: "

SURFACE_PAIRS = [
    ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ("# OK.\n", "# Good.\n", "minimal_ok_good"),
]

CONTINUATIONS = {
    "none": "",
    "neutral": "# State unchanged.\n",
    "heavy": "# Reassigning {var} now.\n",
    "identity": "{var} = {var}\n",
    "type_check": "type({var})\n",
    "pass": "pass\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "x", 7), (4, "y", 8),
]


def run_multivar():
    print("=== Omega Multi-Variable Scope ===")
    result_dir = Path("experiments/results/omega_multivar")
    result_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "query_template": QUERY,
        "templates": {},
    }

    n_total = len(TEMPLATES) * len(CELLS) * len(SURFACE_PAIRS) * 2 * len(CONTINUATIONS)
    print(f"Readouts: {n_total}")

    adapter = ModelAdapter(cfg)
    observations = []
    t0 = time.time()
    n = 0

    for tmpl_name, template in TEMPLATES.items():
        print(f"\n  Template: {tmpl_name}")
        for depth, var, val in CELLS:
            prefix_text = build_prefix(template, var=var, outer_val=val)
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
                            "template": tmpl_name,
                            "depth": depth, "var": var, "val": val,
                            "pair": pair_name, "side": side,
                            "continuation": cont_name,
                            "dist": dist.tolist(),
                        })

    elapsed = time.time() - t0
    print(f"\nDone: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    analyze_multivar(observations)
    del adapter
    print(f"\nSaved {len(observations)} observations to {result_dir}/")
    return observations


def analyze_multivar(observations):
    print("\n=== Multi-Variable Analysis ===")

    obs_idx = {}
    for obs in observations:
        key = (obs["template"], obs["var"], obs["val"], obs["pair"],
               obs["side"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    templates = sorted(set(obs["template"] for obs in observations))

    for tau in [0.01, 0.02, 0.05]:
        print(f"\n--- tau={tau} ---")
        print(f"{'Template':<22s} {'Pair':<20s} {'Cont':<12s} "
              f"{'mean':>8s} {'max':>8s} {'n':>3s} {'status':>8s}")

        for tmpl in templates:
            for pair_name in [p[2] for p in SURFACE_PAIRS]:
                for cont_name in sorted(CONTINUATIONS.keys()):
                    if cont_name == "none":
                        continue
                    tvs = []
                    for _, var, val in CELLS:
                        ka = (tmpl, var, val, pair_name, "a", "none")
                        kb = (tmpl, var, val, pair_name, "b", "none")
                        if ka not in obs_idx or kb not in obs_idx:
                            continue
                        d_pre = 0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum()
                        if d_pre > tau:
                            continue
                        ka2 = (tmpl, var, val, pair_name, "a", cont_name)
                        kb2 = (tmpl, var, val, pair_name, "b", cont_name)
                        if ka2 in obs_idx and kb2 in obs_idx:
                            d = float(0.5 * np.abs(obs_idx[ka2] - obs_idx[kb2]).sum())
                            tvs.append(d)
                    if tvs:
                        mean_tv = float(np.mean(tvs))
                        max_tv = float(np.max(tvs))
                        status = "PRESERVE" if max_tv <= tau else "AMPLIFY"
                        print(f"  {tmpl:<20s} {pair_name:<20s} {cont_name:<12s} "
                              f"{mean_tv:8.5f} {max_tv:8.5f} {len(tvs):3d} "
                              f"{status:>8s}")

    print("\n  Comparison: single-var (prior) vs dual-var")
    print("  If patterns match, omega is robust to multi-variable scope.")
    print("  If dual_interleaved differs from dual_independent, variable")
    print("  interaction affects omega.")


if __name__ == "__main__":
    run_multivar()
