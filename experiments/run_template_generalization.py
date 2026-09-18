"""Experiment #82: Template Generalization.

All experiments so far used the same depth-3 nested scope PREFIX_TEMPLATE.
Do the key findings generalize to a different template?

Test: 3 prefix templates, same conditions, same cells.

Template A (existing): depth-3 nested functions with var reassignments
Template B: flat function, single scope, no nesting
Template C: class method with self reference

Conditions: empty, identity, heavy, collision (4 key conditions)
Pairs: strong_assert, boundary_pair
Cells: 3

Design: 3 templates x 4 conditions x 3 cells x 2 pairs x 2 surfaces
= 144 readouts, ~50s.
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

MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "query_template": "f()\nprint({var})  # Output: ",
    "templates": {},
}

CELLS = [("x", 1), ("y", 2), ("y", 6)]

SURFACES = {
    "assert_var": "# {var} stays the same.\n",
    "assert_novar": "# State unchanged.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
}

EQUIV_PAIRS = [
    ("assert_var", "assert_novar", "strong_assert"),
    ("boundary_1", "boundary_2", "boundary_pair"),
]

# Template A: existing depth-3 nested (same as all prior experiments)
TEMPLATE_A = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

# Template B: flat function, single scope
TEMPLATE_B = (
    "{var} = {outer_val}\ndef f():\n"
    "    {var} = 42\n"
    "    {var} = {var} * 2\n"
    "    return {var}\n"
)

# Template C: class method with self
TEMPLATE_C = (
    "class State:\n"
    "    def __init__(self):\n"
    "        self.{var} = {outer_val}\n"
    "    def f(self):\n"
    "        self.{var} = 99\n"
    "        return self.{var}\n"
    "s = State()\ndef f():\n"
    "    return s.{var}\n"
)

# Template C needs different query
QUERY_C = "s.f()\nprint(s.{var})  # Output: "

TEMPLATES = {
    "depth3_nested": (TEMPLATE_A, None),
    "flat_single": (TEMPLATE_B, None),
    "class_method": (TEMPLATE_C, QUERY_C),
}

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def run():
    print("=== Exp #82: Template Generalization ===")

    adapter = ModelAdapter(MODEL_CFG)
    import torch
    adapter.mdl.half()
    gc.collect()
    print("Model loaded (float16)")

    default_query = MODEL_CFG["query_template"]
    t0 = time.time()

    all_dists = {}
    n_done = 0

    for var, val in CELLS:
        for tmpl_name, (tmpl, custom_query) in TEMPLATES.items():
            prefix_text = build_prefix(tmpl, var=var, outer_val=val)
            query = (custom_query or default_query).replace("{var}", var)
            prefix_state = adapter.get_state_after_prefix(prefix_text)

            id_text = f"{var} = {var}\n"
            heavy_text = f"{var} = {var} + 1  # WARNING: intentional mutation\n"
            collision_text = f"_ = {val}\n"

            for surf_name, surf_tmpl in SURFACES.items():
                surf_text = surf_tmpl.replace("{var}", var)
                for cont_name, cont_text in [("empty", ""), ("identity", id_text),
                                              ("heavy", heavy_text), ("collision", collision_text)]:
                    full_suffix = surf_text + cont_text + query
                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    all_dists[(var, val, tmpl_name, surf_name, cont_name)] = dist
                    n_done += 1

            del prefix_state
            gc.collect()

        elapsed = time.time() - t0
        print(f"  Cell {var}_{val} done, {n_done} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} readouts in {elapsed:.1f}s")

    results = []
    for tmpl_name in TEMPLATES:
        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            for var, val in CELLS:
                for cont_name in ["empty", "identity", "heavy", "collision"]:
                    ka = (var, val, tmpl_name, surf_a, cont_name)
                    kb = (var, val, tmpl_name, surf_b, cont_name)
                    if ka in all_dists and kb in all_dists:
                        cell_tv = float(tv(all_dists[ka], all_dists[kb]))
                        flip = (DIGIT_LABELS[int(np.argmax(all_dists[ka]))] !=
                                DIGIT_LABELS[int(np.argmax(all_dists[kb]))])
                        results.append({
                            "template": tmpl_name,
                            "pair": pair_name,
                            "cell": f"{var}_{val}",
                            "continuation": cont_name,
                            "tv": cell_tv,
                            "flip": flip,
                        })

    # Main table
    for pair_name in ["strong_assert", "boundary_pair"]:
        print(f"\n=== TEMPLATE GENERALIZATION: {pair_name} ===")
        print(f"{'Template':>20s} {'empty':>10s} {'identity':>10s} {'id_amp':>8s} {'heavy':>10s} {'hvy_amp':>8s} {'collision':>10s} {'col_amp':>8s}")
        print("-" * 90)

        for tmpl_name in TEMPLATES:
            empty_tvs = [r["tv"] for r in results if r["template"] == tmpl_name and r["pair"] == pair_name and r["continuation"] == "empty"]
            id_tvs = [r["tv"] for r in results if r["template"] == tmpl_name and r["pair"] == pair_name and r["continuation"] == "identity"]
            heavy_tvs = [r["tv"] for r in results if r["template"] == tmpl_name and r["pair"] == pair_name and r["continuation"] == "heavy"]
            col_tvs = [r["tv"] for r in results if r["template"] == tmpl_name and r["pair"] == pair_name and r["continuation"] == "collision"]

            e = np.mean(empty_tvs) if empty_tvs else 0.001
            i = np.mean(id_tvs) if id_tvs else 0
            h = np.mean(heavy_tvs) if heavy_tvs else 0
            c = np.mean(col_tvs) if col_tvs else 0

            print(f"{tmpl_name:>20s} {e:10.4f} {i:10.4f} {i/e if e>0.001 else 0:7.1f}x {h:10.4f} {h/e if e>0.001 else 0:7.1f}x {c:10.4f} {c/e if e>0.001 else 0:7.1f}x")

    # Check which findings generalize
    print("\n=== GENERALIZATION SCORECARD ===")
    print("Finding: heavy always amplifies")
    for tmpl_name in TEMPLATES:
        for pair_name in ["strong_assert", "boundary_pair"]:
            empty_tvs = [r["tv"] for r in results if r["template"] == tmpl_name and r["pair"] == pair_name and r["continuation"] == "empty"]
            heavy_tvs = [r["tv"] for r in results if r["template"] == tmpl_name and r["pair"] == pair_name and r["continuation"] == "heavy"]
            e = np.mean(empty_tvs)
            h = np.mean(heavy_tvs)
            amplifies = h > e
            print(f"  {tmpl_name:>20s} x {pair_name:>15s}: heavy_delta = {h-e:+.4f} {'OK' if amplifies else 'FAIL'}")

    print("\nFinding: identity pair-dependent (amplify on strong_assert, varies on boundary)")
    for tmpl_name in TEMPLATES:
        sa_empty = np.mean([r["tv"] for r in results if r["template"] == tmpl_name and r["pair"] == "strong_assert" and r["continuation"] == "empty"])
        sa_id = np.mean([r["tv"] for r in results if r["template"] == tmpl_name and r["pair"] == "strong_assert" and r["continuation"] == "identity"])
        bp_empty = np.mean([r["tv"] for r in results if r["template"] == tmpl_name and r["pair"] == "boundary_pair" and r["continuation"] == "empty"])
        bp_id = np.mean([r["tv"] for r in results if r["template"] == tmpl_name and r["pair"] == "boundary_pair" and r["continuation"] == "identity"])

        sa_amp = sa_id / sa_empty if sa_empty > 0.001 else 0
        bp_amp = bp_id / bp_empty if bp_empty > 0.001 else 0
        print(f"  {tmpl_name:>20s}: sa_id_amp={sa_amp:.1f}x, bp_id_amp={bp_amp:.1f}x, ratio={sa_amp/bp_amp if bp_amp > 0.1 else float('inf'):.1f}x")

    summary = {
        "experiment": 82,
        "name": "template_generalization",
        "n_readouts": n_done,
        "n_comparisons": len(results),
        "results": results,
    }

    result_dir = Path("experiments/results/omega_template_generalization")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
