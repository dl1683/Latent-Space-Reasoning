"""Experiment #78: Throwaway Variable Boundary.

Exp #77 showed _ and __ don't break collision cancellation but _x does.
Where is the boundary between "throwaway" and "real" variable names?

Test: double collision with different target variable names, ordered by
how "throwaway" they look to a language model.

Target names (most throwaway → most real):
- _ (canonical throwaway)
- __ (double underscore throwaway)
- ___ (triple — still throwaway?)
- _0 (underscore + digit)
- _x (underscore + letter — broke cancellation in #77)
- tmp (common temp variable)
- result (meaningful name)
- counter (very meaningful)

Also test single versions for baseline comparison.

Design: 8 names × (single + double) × 3 cells × 2 pairs × 2 surfaces
= 192 readouts, ~60s.
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

PREFIX_TEMPLATE = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]

TARGET_NAMES = ["_", "__", "___", "_0", "_x", "tmp", "result", "counter"]


def get_conditions(var, val):
    conds = {"empty": ""}
    for tname in TARGET_NAMES:
        conds[f"single_{tname}"] = f"{tname} = {val}\n"
        conds[f"double_{tname}"] = f"{tname} = {val}\n{tname} = {val}\n"
    return conds


def run():
    print("=== Exp #78: Throwaway Variable Boundary ===")

    adapter = ModelAdapter(MODEL_CFG)
    import torch
    adapter.mdl.half()
    gc.collect()
    print("Model loaded (float16)")

    query_tmpl = MODEL_CFG["query_template"]
    t0 = time.time()

    all_dists = {}
    n_done = 0

    for var, val in CELLS:
        prefix_text = build_prefix(PREFIX_TEMPLATE, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        conditions = get_conditions(var, val)

        for surf_name, surf_tmpl in SURFACES.items():
            surf_text = surf_tmpl.replace("{var}", var)
            for cond_name, cond_text in conditions.items():
                full_suffix = surf_text + cond_text + query
                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                all_dists[(var, val, surf_name, cond_name)] = dist
                n_done += 1

        del prefix_state
        gc.collect()
        elapsed = time.time() - t0
        print(f"  Cell {var}_{val} done, {n_done} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} readouts in {elapsed:.1f}s")

    results = []
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for var, val in CELLS:
            conditions = get_conditions(var, val)
            for cond_name in conditions:
                ka = (var, val, surf_a, cond_name)
                kb = (var, val, surf_b, cond_name)
                if ka in all_dists and kb in all_dists:
                    cell_tv = float(tv(all_dists[ka], all_dists[kb]))
                    flip = (DIGIT_LABELS[int(np.argmax(all_dists[ka]))] !=
                            DIGIT_LABELS[int(np.argmax(all_dists[kb]))])
                    results.append({
                        "pair": pair_name,
                        "cell": f"{var}_{val}",
                        "continuation": cond_name,
                        "tv": cell_tv,
                        "flip": flip,
                    })

    for pair_name in ["strong_assert", "boundary_pair"]:
        print(f"\n=== THROWAWAY BOUNDARY: {pair_name} ===")
        print(f"{'Target':>10s} {'single_TV':>10s} {'s_amp':>8s} {'double_TV':>10s} {'d_amp':>8s} {'ratio':>8s}")
        print("-" * 60)

        pair_res = [r for r in results if r["pair"] == pair_name]
        empty_tv = np.mean([r["tv"] for r in pair_res if r["continuation"] == "empty"])

        for tname in TARGET_NAMES:
            s_tvs = [r["tv"] for r in pair_res if r["continuation"] == f"single_{tname}"]
            d_tvs = [r["tv"] for r in pair_res if r["continuation"] == f"double_{tname}"]
            if s_tvs and d_tvs:
                s_mean = np.mean(s_tvs)
                d_mean = np.mean(d_tvs)
                s_amp = s_mean / empty_tv if empty_tv > 0.001 else 0
                d_amp = d_mean / empty_tv if empty_tv > 0.001 else 0
                ratio = d_mean / s_mean if s_mean > 0.001 else 0
                print(f"{tname:>10s} {s_mean:10.4f} {s_amp:7.1f}x {d_mean:10.4f} {d_amp:7.1f}x {ratio:7.2f}x")

    # Cancellation index: double/single ratio (< 1 means cancellation)
    print("\n=== CANCELLATION INDEX (double/single, <1 = cancellation) ===")
    print(f"{'Target':>10s} {'strong_a':>10s} {'boundary':>10s}")
    print("-" * 35)
    for tname in TARGET_NAMES:
        vals = {}
        for pair_name in ["strong_assert", "boundary_pair"]:
            pair_res = [r for r in results if r["pair"] == pair_name]
            s_tvs = [r["tv"] for r in pair_res if r["continuation"] == f"single_{tname}"]
            d_tvs = [r["tv"] for r in pair_res if r["continuation"] == f"double_{tname}"]
            if s_tvs and d_tvs:
                s_mean = np.mean(s_tvs)
                d_mean = np.mean(d_tvs)
                vals[pair_name] = d_mean / s_mean if s_mean > 0.001 else 0
        sa = vals.get("strong_assert", 0)
        bp = vals.get("boundary_pair", 0)
        print(f"{tname:>10s} {sa:10.2f}x {bp:10.2f}x")

    summary = {
        "experiment": 78,
        "name": "throwaway_boundary",
        "n_readouts": n_done,
        "n_comparisons": len(results),
        "results": results,
    }

    result_dir = Path("experiments/results/omega_throwaway_boundary")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
