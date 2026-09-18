"""Experiment #76: Stacking Grid on New Pairs (Fable R7b E1).

Fable's critique: composition algebra observed only on 2 pairs (strong_assert,
boundary_pair). Need ≥2 more pairs to know which algebra is the norm.

Test: full stacking grid (6 singles + 10 2-element compositions) on 2 new pairs:
- assert_weak: ("# probably same\n", "# likely unchanged\n")
- comment_generic: ("# note\n", "# info\n")

These are semantically weaker than strong_assert, expected to be intermediate.

Design: 16 conditions x 3 cells x 2 new pairs x 2 surfaces = 192 readouts, ~60s.
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
    "assert_weak_1": "# probably same\n",
    "assert_weak_2": "# likely unchanged\n",
    "comment_generic_1": "# note\n",
    "comment_generic_2": "# info\n",
}

EQUIV_PAIRS = [
    ("assert_weak_1", "assert_weak_2", "assert_weak"),
    ("comment_generic_1", "comment_generic_2", "comment_generic"),
]

PREFIX_TEMPLATE = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def get_conditions(var, val):
    singles = {
        "empty": "",
        "identity": f"{var} = {var}\n",
        "collision": f"_ = {val}\n",
        "print_val": f"print({val})\n",
        "isinstance": f"isinstance({var}, int)\n",
        "heavy": f"{var} = {var} + 1  # WARNING: intentional mutation\n",
    }
    stacks = {
        "identity+collision": f"{var} = {var}\n_ = {val}\n",
        "collision+identity": f"_ = {val}\n{var} = {var}\n",
        "identity+isinstance": f"{var} = {var}\nisinstance({var}, int)\n",
        "isinstance+identity": f"isinstance({var}, int)\n{var} = {var}\n",
        "collision+isinstance": f"_ = {val}\nisinstance({var}, int)\n",
        "isinstance+collision": f"isinstance({var}, int)\n_ = {val}\n",
        "identity+heavy": f"{var} = {var}\n{var} = {var} + 1  # WARNING: intentional mutation\n",
        "heavy+identity": f"{var} = {var} + 1  # WARNING: intentional mutation\n{var} = {var}\n",
        "double_identity": f"{var} = {var}\n{var} = {var}\n",
        "double_collision": f"_ = {val}\n_ = {val}\n",
    }
    return {**singles, **stacks}


def run():
    print("=== Exp #76: Stacking Grid on New Pairs (Fable E1) ===")

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
            surf_text = surf_tmpl
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

    # Print tables
    for pair_name in ["assert_weak", "comment_generic"]:
        print(f"\n=== STACKING GRID: {pair_name} ===")
        print(f"{'Condition':>25s} {'mean_TV':>10s} {'amp':>10s} {'flips':>6s}")
        print("-" * 55)

        pair_results = [r for r in results if r["pair"] == pair_name]
        empty_tv = np.mean([r["tv"] for r in pair_results if r["continuation"] == "empty"])

        cond_names = list(get_conditions("x", 1).keys())
        for cond in cond_names:
            cond_tvs = [r["tv"] for r in pair_results if r["continuation"] == cond]
            cond_flips = [r["flip"] for r in pair_results if r["continuation"] == cond]
            if cond_tvs:
                mean_tv = np.mean(cond_tvs)
                amp = mean_tv / empty_tv if empty_tv > 0.001 else float('inf')
                n_flips = sum(cond_flips)
                print(f"{cond:>25s} {mean_tv:10.4f} {amp:10.1f}x {n_flips:6d}/3")

    # Cross-pair comparison
    print("\n=== CROSS-PAIR COMPARISON (amp vs strong_assert / boundary_pair) ===")
    print("(Requires loading prior data for full comparison)")
    print("For now, compare assert_weak vs comment_generic:")
    print(f"\n{'Condition':>25s} {'aw_amp':>10s} {'cg_amp':>10s} {'ratio':>10s}")
    print("-" * 60)

    cond_names = list(get_conditions("x", 1).keys())
    aw_empty = np.mean([r["tv"] for r in results if r["pair"] == "assert_weak" and r["continuation"] == "empty"])
    cg_empty = np.mean([r["tv"] for r in results if r["pair"] == "comment_generic" and r["continuation"] == "empty"])

    for cond in cond_names:
        aw_tvs = [r["tv"] for r in results if r["pair"] == "assert_weak" and r["continuation"] == cond]
        cg_tvs = [r["tv"] for r in results if r["pair"] == "comment_generic" and r["continuation"] == cond]
        if aw_tvs and cg_tvs:
            aw_mean = np.mean(aw_tvs)
            cg_mean = np.mean(cg_tvs)
            aw_amp = aw_mean / aw_empty if aw_empty > 0.001 else 0
            cg_amp = cg_mean / cg_empty if cg_empty > 0.001 else 0
            ratio = aw_amp / cg_amp if cg_amp > 0.1 else float('inf')
            print(f"{cond:>25s} {aw_amp:10.1f}x {cg_amp:10.1f}x {ratio:10.2f}")

    # Delta TV table
    for pair_name in ["assert_weak", "comment_generic"]:
        print(f"\n=== DELTA TV (condition - empty) : {pair_name} ===")
        pair_results = [r for r in results if r["pair"] == pair_name]
        empty_tv = np.mean([r["tv"] for r in pair_results if r["continuation"] == "empty"])

        for cond in cond_names:
            cond_tvs = [r["tv"] for r in pair_results if r["continuation"] == cond]
            if cond_tvs:
                mean_tv = np.mean(cond_tvs)
                delta = mean_tv - empty_tv
                print(f"  {cond:>25s}: delta_TV = {delta:+.4f}")

    summary = {
        "experiment": 76,
        "name": "stacking_newpairs",
        "n_readouts": n_done,
        "n_comparisons": len(results),
        "results": results,
    }

    result_dir = Path("experiments/results/omega_stacking_newpairs")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
