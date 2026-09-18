"""Omega Cross-Variable Transfer: Does omega generalize across variables?

The strongest test of omega's predictive power:
- Compute omega from x-variable cells only
- Predict preservation for y,z-variable cells
- Compare against action-only baseline

If omega transfers across variables, it's measuring a genuine property
of the continuation, not memorizing cell-specific artifacts.

Fast: 6 cells × 2 pairs × 2 sides × 10 conts = 240 readouts (~1 min)
But we need 3 separate variable groups, so just reuse sweep data.
Actually, let's run fresh with wider variable coverage.

9 cells × 2 pairs × 2 sides × 12 conts = 432 readouts (~2 min)
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

SURFACE_PAIRS = [
    ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ("# Values preserved.\n", "# {var} retains its value.\n", "moderate_assert"),
]

CONTINUATIONS = {
    "none": "",
    "neutral_1": "# State unchanged.\n",
    "neutral_2": "# Values preserved.\n",
    "heavy_1": "# Reassigning {var} now.\n",
    "heavy_2": "# Updating {var}.\n",
    "pass": "pass\n",
    "identity": "{var} = {var}\n",
    "type_check": "type({var})\n",
    "observe": "_ = {var}\n",
    "verified": "# Verified.\n",
    "bare_nl": "\n",
    "neutral_then_heavy": "# State unchanged.\n# Reassigning {var} now.\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 6), (4, "y", 8),
    (4, "z", 3), (4, "z", 5), (4, "z", 7),
]


def run_transfer_test():
    print("=== Omega Cross-Variable Transfer Test ===")
    n_total = len(CELLS) * len(SURFACE_PAIRS) * 2 * len(CONTINUATIONS)
    print(f"Cells: {len(CELLS)}, Pairs: {len(SURFACE_PAIRS)}, "
          f"Continuations: {len(CONTINUATIONS)}")
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

    result_dir = Path("experiments/results/omega_transfer")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["pair"], obs["side"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    VARS = ["x", "y", "z"]
    HEAVY_KW = ["heavy", "reassign", "updat", "chang"]

    for tau in [0.01, 0.02, 0.05]:
        print(f"\n=== Cross-Variable Transfer (tau={tau}) ===")

        for cal_var in VARS:
            eval_vars = [v for v in VARS if v != cal_var]

            for pair_name in [p[2] for p in SURFACE_PAIRS]:
                cal_omega = {}
                for cont_name in CONTINUATIONS:
                    tvs = []
                    for _, var, val in [c for c in CELLS if c[1] == cal_var]:
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
                        cal_omega[cont_name] = max(tvs)

                omega_correct = 0
                baseline_correct = 0
                total = 0

                for cont_name in CONTINUATIONS:
                    if cont_name not in cal_omega:
                        continue

                    eval_tvs = []
                    for _, var, val in [c for c in CELLS if c[1] in eval_vars]:
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
                            eval_tvs.append(d)

                    if not eval_tvs:
                        continue

                    actual = "preserve" if max(eval_tvs) <= tau else "separate"
                    omega_pred = "preserve" if cal_omega[cont_name] <= tau else "separate"

                    cn_lower = cont_name.lower()
                    is_heavy = any(k in cn_lower for k in HEAVY_KW)
                    if cont_name == "neutral_then_heavy":
                        is_heavy = True
                    baseline_pred = "separate" if is_heavy else "preserve"

                    total += 1
                    omega_correct += int(omega_pred == actual)
                    baseline_correct += int(baseline_pred == actual)

                    if omega_pred != baseline_pred:
                        marker = "omega" if omega_pred == actual else "baseline"
                        print(f"  cal={cal_var} pair={pair_name:20s} "
                              f"{cont_name:20s} omega={omega_pred:8s} "
                              f"base={baseline_pred:8s} actual={actual:8s} [{marker}]")

                if total > 0:
                    print(f"  cal={cal_var} pair={pair_name}: "
                          f"omega={omega_correct}/{total} ({100*omega_correct/total:.0f}%), "
                          f"baseline={baseline_correct}/{total} ({100*baseline_correct/total:.0f}%)")

    print(f"\nSaved {n} observations to {result_dir}/")
    del adapter
    return observations


if __name__ == "__main__":
    run_transfer_test()
