"""Omega Structure Transfer: Does omega trained on one code structure predict another?

Tests whether omega estimated from nested functions transfers to:
- For loops with variable accumulation
- Class with method modifying attribute
- List comprehension with variable
- Try/except with variable

If omega transfers across code structures, it captures something about
the model's variable tracking, not just pattern matching on function nesting.

4 structures x 6 cells x 2 pairs x 2 sides x 6 continuations = 576 readouts (~3 min)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix


TEMPLATES = {
    "nested_func": (
        "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
        "    def g():\n        {var} = 999\n        def h():\n"
        "            {var} = 9999\n            def k():\n"
        "                {var} = 99999\n                return {var}\n"
        "            k()\n            return {var}\n"
        "        h()\n        return {var}\n"
        "    g()\n    return {var}\n"
    ),
    "for_loop": (
        "{var} = {outer_val}\nfor i in range(5):\n    {var} = {var} + i\n"
        "    for j in range(3):\n        {var} = {var} * 1\n"
        "        for k in range(2):\n            {var} = {var} + 0\n"
    ),
    "class_method": (
        "class Obj:\n    def __init__(self):\n        self.{var} = {outer_val}\n"
        "    def update(self):\n        self.{var} = self.{var}\n"
        "    def check(self):\n        return self.{var}\n"
        "obj = Obj()\nobj.update()\n{var} = obj.check()\n"
    ),
    "try_except": (
        "{var} = {outer_val}\ntry:\n    {var} = int(str({var}))\n"
        "    try:\n        {var} = {var} + 0\n"
        "    except ValueError:\n        {var} = 0\n"
        "except Exception:\n    {var} = -1\n"
    ),
}

QUERIES = {
    "nested_func": "f()\nprint({var})  # Output: ",
    "for_loop": "print({var})  # Output: ",
    "class_method": "print({var})  # Output: ",
    "try_except": "print({var})  # Output: ",
}

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
    (4, "x", 1), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 6), (4, "y", 8),
]


def run_structure_transfer():
    print("=== Omega Structure Transfer ===")
    result_dir = Path("experiments/results/omega_structure_transfer")
    result_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "query_template": "",
        "templates": {},
    }

    n_total = len(TEMPLATES) * len(CELLS) * len(SURFACE_PAIRS) * 2 * len(CONTINUATIONS)
    print(f"Readouts: {n_total}")

    adapter = ModelAdapter(cfg)
    observations = []
    t0 = time.time()
    n = 0

    for struct_name, template in TEMPLATES.items():
        query_tmpl = QUERIES[struct_name]
        print(f"\n  Structure: {struct_name}")

        for depth, var, val in CELLS:
            prefix_text = build_prefix(template, var=var, outer_val=val)
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
                            "structure": struct_name,
                            "var": var, "val": val,
                            "pair": pair_name, "side": side,
                            "continuation": cont_name,
                            "dist": dist.tolist(),
                        })

    elapsed = time.time() - t0
    print(f"\nDone: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    analyze_structure_transfer(observations)
    del adapter
    print(f"\nSaved {len(observations)} observations to {result_dir}/")
    return observations


def analyze_structure_transfer(observations):
    print("\n=== Structure Transfer Analysis ===")

    obs_idx = {}
    for obs in observations:
        key = (obs["structure"], obs["var"], obs["val"], obs["pair"],
               obs["side"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    structures = sorted(set(obs["structure"] for obs in observations))

    for tau in [0.01, 0.02, 0.05]:
        print(f"\n--- tau={tau} ---")
        # Per-structure omega table
        struct_omega = {}
        for struct in structures:
            cont_omega = {}
            for pair_name in [p[2] for p in SURFACE_PAIRS]:
                for cont_name in sorted(CONTINUATIONS.keys()):
                    if cont_name == "none":
                        continue
                    tvs = []
                    for _, var, val in CELLS:
                        ka = (struct, var, val, pair_name, "a", "none")
                        kb = (struct, var, val, pair_name, "b", "none")
                        if ka not in obs_idx or kb not in obs_idx:
                            continue
                        d_pre = 0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum()
                        if d_pre > tau:
                            continue
                        ka_c = (struct, var, val, pair_name, "a", cont_name)
                        kb_c = (struct, var, val, pair_name, "b", cont_name)
                        if ka_c in obs_idx and kb_c in obs_idx:
                            d = float(0.5 * np.abs(obs_idx[ka_c] - obs_idx[kb_c]).sum())
                            tvs.append(d)
                    if tvs:
                        key = (pair_name, cont_name)
                        if key not in cont_omega:
                            cont_omega[key] = []
                        cont_omega[key].extend(tvs)
            struct_omega[struct] = cont_omega

        # Show omega by structure
        print(f"\n  {'Structure':<15s} {'Pair':<18s} {'Cont':<12s} "
              f"{'mean':>8s} {'max':>8s} {'n':>3s} {'status':>8s}")
        for struct in structures:
            for pair_name in [p[2] for p in SURFACE_PAIRS]:
                for cont_name in sorted(CONTINUATIONS.keys()):
                    if cont_name == "none":
                        continue
                    key = (pair_name, cont_name)
                    if key in struct_omega[struct]:
                        tvs = struct_omega[struct][key]
                        mean_tv = float(np.mean(tvs))
                        max_tv = float(np.max(tvs))
                        status = "PRESERVE" if max_tv <= tau else "AMPLIFY"
                        print(f"  {struct:<15s} {pair_name:<18s} {cont_name:<12s} "
                              f"{mean_tv:8.5f} {max_tv:8.5f} {len(tvs):3d} "
                              f"{status:>8s}")

        # Cross-structure transfer: train on nested_func, predict others
        print(f"\n  Cross-structure transfer (train=nested_func):")
        ref = "nested_func"
        ref_omega = struct_omega.get(ref, {})
        for target in structures:
            if target == ref:
                continue
            target_omega = struct_omega.get(target, {})
            correct = 0
            total = 0
            for key in target_omega:
                if key not in ref_omega:
                    continue
                ref_max = max(ref_omega[key])
                tgt_max = max(target_omega[key])
                ref_pred = "preserve" if ref_max <= tau else "separate"
                tgt_actual = "preserve" if tgt_max <= tau else "separate"
                correct += int(ref_pred == tgt_actual)
                total += 1
            if total > 0:
                print(f"    {ref} -> {target}: {100*correct/total:.0f}% ({correct}/{total})")


if __name__ == "__main__":
    run_structure_transfer()
