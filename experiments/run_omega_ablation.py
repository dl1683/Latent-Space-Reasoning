"""Token Ablation: What drives identity's amplification?

The identity continuation ({var} = {var}\n) produces ~0.047 amplification.
This experiment ablates specific tokens to find what drives it:
- Just the variable name
- Just the equals sign
- Assignment to a literal
- Assignment to a different variable
- Reversed assignment
- Partial tokens

6 cells × 2 pairs × 2 sides × 20 variants = 480 readouts (~2 min)
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

ABLATIONS = {
    # Controls
    "none": "",
    "identity": "{var} = {var}\n",
    "type_check": "type({var})\n",

    # Ablation: remove assignment target
    "just_var": "{var}\n",
    "just_equals_var": "= {var}\n",
    "var_equals": "{var} =\n",
    "var_equals_space": "{var} = \n",

    # Ablation: change RHS
    "assign_zero": "{var} = 0\n",
    "assign_one": "{var} = 1\n",
    "assign_none": "{var} = None\n",
    "assign_str": "{var} = ''\n",
    "assign_self_plus": "{var} = {var} + 0\n",

    # Ablation: change LHS
    "assign_to_underscore": "_ = {var}\n",
    "assign_to_temp": "tmp = {var}\n",

    # Ablation: multiple identity
    "double_identity": "{var} = {var}\n{var} = {var}\n",
    "triple_identity": "{var} = {var}\n{var} = {var}\n{var} = {var}\n",

    # Ablation: related operations
    "augmented_assign": "{var} += 0\n",
    "comparison": "{var} == {var}\n",
    "is_check": "{var} is {var}\n",
    "repr_call": "repr({var})\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 6), (4, "y", 8),
]


def run_ablation():
    print("=== Token Ablation on Identity ===")
    n_total = len(CELLS) * len(SURFACE_PAIRS) * 2 * len(ABLATIONS)
    print(f"Cells: {len(CELLS)}, Pairs: {len(SURFACE_PAIRS)}, "
          f"Ablations: {len(ABLATIONS)}")
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

                for abl_name, abl_tmpl in ABLATIONS.items():
                    abl_text = abl_tmpl.replace("{var}", var)
                    full_suffix = surface_text + abl_text + query

                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    n += 1

                    observations.append({
                        "depth": depth, "var": var, "val": val,
                        "pair": pair_name, "side": side,
                        "ablation": abl_name,
                        "dist": dist.tolist(),
                    })

                    if n % 100 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n}/{n_total} readouts, {elapsed:.0f}s, "
                              f"{n/elapsed:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    result_dir = Path("experiments/results/omega_ablation")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["pair"], obs["side"], obs["ablation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    print("\n=== Ablation Omega Table (tau=0.01) ===")
    print(f"{'Ablation':<25s} {'pair':<20s} {'mean_tv':>8s} {'max_tv':>8s} "
          f"{'n':>3s} {'status':>8s}")
    print("-" * 75)

    ablation_stats = {}
    for tau in [0.01, 0.02, 0.05]:
        ablation_stats[str(tau)] = {}
        if tau != 0.01:
            print(f"\n=== Ablation Omega Table (tau={tau}) ===")

        for pair_name in [p[2] for p in SURFACE_PAIRS]:
            for abl_name in ABLATIONS:
                tvs = []
                for _, var, val in CELLS:
                    ka_base = (var, val, pair_name, "a", "none")
                    kb_base = (var, val, pair_name, "b", "none")
                    if ka_base not in obs_idx or kb_base not in obs_idx:
                        continue
                    d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
                    if d_pre > tau:
                        continue

                    ka = (var, val, pair_name, "a", abl_name)
                    kb = (var, val, pair_name, "b", abl_name)
                    if ka in obs_idx and kb in obs_idx:
                        d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                        tvs.append(d)

                if tvs:
                    mean_tv = float(np.mean(tvs))
                    max_tv = float(np.max(tvs))
                    status = "PRESERVE" if max_tv <= tau else "AMPLIFY"
                    if tau == 0.01 or (tau > 0.01 and abl_name in [
                        "none", "identity", "assign_zero", "assign_one",
                        "just_var", "augmented_assign", "comparison",
                        "double_identity"
                    ]):
                        print(f"  {abl_name:<23s} {pair_name:<20s} "
                              f"{mean_tv:8.5f} {max_tv:8.5f} {len(tvs):3d} "
                              f"{status:>8s}")
                    ablation_stats[str(tau)].setdefault(pair_name, {})[abl_name] = {
                        "mean": mean_tv, "max": max_tv, "n": len(tvs),
                        "preserves": bool(max_tv <= tau),
                    }

    with open(result_dir / "ablation_stats.json", "w") as f:
        json.dump(ablation_stats, f, indent=2)

    print(f"\nSaved {n} observations to {result_dir}/")
    del adapter
    return ablation_stats


if __name__ == "__main__":
    run_ablation()
