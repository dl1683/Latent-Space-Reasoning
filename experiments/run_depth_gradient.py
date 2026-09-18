"""Experiment #70: Depth Gradient.

If the identity paradox (x=x causes divergence) is driven by scope-chain
traversal, then DEEPER nesting should amplify the effect. At depth 1 (flat scope),
x=x is trivial. At depth 4 (4 nested functions), x=x requires traversing the
entire scope chain.

Tests: depth 1, 2, 3, 4 with identity and heavy continuations.
2 surface pairs, 3 variable-value combos = rich grid.

Design: 4 depths x 3 continuations x 3 vals x 2 pairs x 2 surfaces = 144 readouts. ~3 min.
"""
import gc
import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv

DEPTH_TEMPLATES = {
    1: "{var} = {outer_val}\ndef f():\n    {var} = 99\n    return {var}\n",
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

MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "query_template": "f()\nprint({var})  # Output: ",
    "templates": {},
}

SURFACES = {
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
}

EQUIV_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert"),
    ("boundary_1", "boundary_2", "boundary_pair"),
]

CONTINUATIONS = {
    "empty": "",
    "identity": "{var} = {var}\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
}

CELLS = [("x", 1), ("y", 2), ("y", 6)]
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def run():
    print("=== Exp #70: Depth Gradient ===")
    n_readouts = len(DEPTH_TEMPLATES) * len(CONTINUATIONS) * len(CELLS) * len(SURFACES)
    n_comparisons = len(DEPTH_TEMPLATES) * len(CONTINUATIONS) * len(CELLS) * len(EQUIV_PAIRS)
    print(f"Readouts: {n_readouts}, Comparisons: {n_comparisons}")

    adapter = ModelAdapter(MODEL_CFG)
    import torch
    adapter.mdl.half()
    gc.collect()
    print("Model loaded (float16)")

    query_tmpl = MODEL_CFG["query_template"]
    t0 = time.time()

    all_dists = {}
    n_done = 0

    for depth, tmpl in DEPTH_TEMPLATES.items():
        for var, val in CELLS:
            prefix_text = build_prefix(tmpl, var=var, outer_val=val)
            query = query_tmpl.replace("{var}", var)
            prefix_state = adapter.get_state_after_prefix(prefix_text)

            for surf_name, surf_tmpl in SURFACES.items():
                surf_text = surf_tmpl.replace("{var}", var)
                for cont_name, cont_tmpl in CONTINUATIONS.items():
                    cont_text = cont_tmpl.replace("{var}", var)
                    full_suffix = surf_text + cont_text + query
                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    all_dists[(depth, var, val, surf_name, cont_name)] = dist
                    n_done += 1

            del prefix_state
            gc.collect()

        elapsed = time.time() - t0
        print(f"  Depth {depth} done, {n_done}/{n_readouts} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} readouts in {elapsed:.1f}s")

    # Compute TV for each condition
    results = []
    for depth in DEPTH_TEMPLATES:
        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            for var, val in CELLS:
                for cont_name in CONTINUATIONS:
                    ka = (depth, var, val, surf_a, cont_name)
                    kb = (depth, var, val, surf_b, cont_name)
                    if ka in all_dists and kb in all_dists:
                        cell_tv = float(tv(all_dists[ka], all_dists[kb]))
                        flip = (DIGIT_LABELS[int(np.argmax(all_dists[ka]))] !=
                                DIGIT_LABELS[int(np.argmax(all_dists[kb]))])
                        results.append({
                            "depth": depth,
                            "pair": pair_name,
                            "cell": f"{var}_{val}",
                            "continuation": cont_name,
                            "tv": cell_tv,
                            "flip": flip,
                        })

    # Print depth x continuation matrix for each pair
    for pair_name in ["strong_assert", "boundary_pair"]:
        print(f"\n=== {pair_name}: mean TV by depth x continuation ===")
        print(f"{'Depth':>6s}", end="")
        for cont in CONTINUATIONS:
            print(f" {cont:>12s}", end="")
        print()
        print("-" * 44)

        for depth in sorted(DEPTH_TEMPLATES.keys()):
            print(f"{depth:6d}", end="")
            for cont in CONTINUATIONS:
                tvs = [r["tv"] for r in results
                       if r["depth"] == depth and r["pair"] == pair_name and r["continuation"] == cont]
                if tvs:
                    print(f" {np.mean(tvs):12.4f}", end="")
                else:
                    print(f" {'N/A':>12s}", end="")
            print()

    # Depth scaling analysis
    print(f"\n=== DEPTH SCALING (identity amplification ratio = identity_TV / empty_TV) ===")
    for pair_name in ["strong_assert", "boundary_pair"]:
        print(f"\n  {pair_name}:")
        for depth in sorted(DEPTH_TEMPLATES.keys()):
            empty_tvs = [r["tv"] for r in results
                         if r["depth"] == depth and r["pair"] == pair_name and r["continuation"] == "empty"]
            id_tvs = [r["tv"] for r in results
                      if r["depth"] == depth and r["pair"] == pair_name and r["continuation"] == "identity"]
            heavy_tvs = [r["tv"] for r in results
                         if r["depth"] == depth and r["pair"] == pair_name and r["continuation"] == "heavy"]

            e_mean = np.mean(empty_tvs) if empty_tvs else 0.001
            i_mean = np.mean(id_tvs) if id_tvs else 0
            h_mean = np.mean(heavy_tvs) if heavy_tvs else 0

            i_ratio = i_mean / e_mean if e_mean > 0.001 else float('inf')
            h_ratio = h_mean / e_mean if e_mean > 0.001 else float('inf')
            print(f"    depth={depth}: empty={e_mean:.4f} identity={i_mean:.4f} ({i_ratio:.1f}x) "
                  f"heavy={h_mean:.4f} ({h_ratio:.1f}x)")

    summary = {
        "experiment": 70,
        "name": "depth_gradient",
        "n_readouts": n_done,
        "n_comparisons": len(results),
        "results": results,
    }

    result_dir = Path("experiments/results/omega_depth_gradient")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
