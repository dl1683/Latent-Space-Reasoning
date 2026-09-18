"""Experiment #77: Verbatim vs Renamed Repeat (Fable R7b E3).

Fable's hypothesis: double_collision self-cancels because the model enters
"copy mode" — two identical lines triggers repetition suppression. If true,
two DIFFERENT collision lines with same val should NOT self-cancel.

Test: compare verbatim repeat (val=val) vs renamed repeat (_ = val, __ = val)
vs mixed (val first, var second) vs identity patterns.

Conditions:
- empty
- single_collision: _ = val
- double_collision_verbatim: _ = val / _ = val (same line twice)
- double_collision_renamed: _ = val / __ = val (different target, same value)
- double_collision_mixed: _ = val / _x = val (yet another target)
- triple_collision_verbatim: _ = val x3
- triple_collision_mixed: _ = val / __ = val / _x = val
- single_identity: var = var
- double_identity_verbatim: var = var x2
- double_identity_renamed: var = var / _var = var (impossible, different)
  Actually identity is self-referential so rename means different pattern.
  Use: var = var / _ = var (self-assignment vs variable read)

Design: 10 conditions x 3 cells x 2 pairs x 2 surfaces = 120 readouts.
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


def get_conditions(var, val):
    return {
        "empty": "",
        "single_collision": f"_ = {val}\n",
        "double_verbatim": f"_ = {val}\n_ = {val}\n",
        "double_renamed": f"_ = {val}\n__ = {val}\n",
        "double_mixed": f"_ = {val}\n_x = {val}\n",
        "triple_verbatim": f"_ = {val}\n_ = {val}\n_ = {val}\n",
        "triple_mixed": f"_ = {val}\n__ = {val}\n_x = {val}\n",
        "single_identity": f"{var} = {var}\n",
        "double_identity": f"{var} = {var}\n{var} = {var}\n",
        "identity_plus_read": f"{var} = {var}\n_ = {var}\n",
    }


def run():
    print("=== Exp #77: Verbatim vs Renamed Repeat (Fable E3) ===")

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
        print(f"\n=== RENAME REPEAT: {pair_name} ===")
        print(f"{'Condition':>25s} {'mean_TV':>10s} {'amp':>10s} {'delta':>10s} {'flips':>6s}")
        print("-" * 65)

        pair_results = [r for r in results if r["pair"] == pair_name]
        empty_tv = np.mean([r["tv"] for r in pair_results if r["continuation"] == "empty"])

        for cond in get_conditions("x", 1).keys():
            cond_tvs = [r["tv"] for r in pair_results if r["continuation"] == cond]
            cond_flips = [r["flip"] for r in pair_results if r["continuation"] == cond]
            if cond_tvs:
                mean_tv = np.mean(cond_tvs)
                amp = mean_tv / empty_tv if empty_tv > 0.001 else float('inf')
                delta = mean_tv - empty_tv
                n_flips = sum(cond_flips)
                print(f"{cond:>25s} {mean_tv:10.4f} {amp:10.1f}x {delta:+10.4f} {n_flips:6d}/3")

    # Verbatim vs renamed comparison
    print("\n=== VERBATIM vs RENAMED (strong_assert) ===")
    sa = [r for r in results if r["pair"] == "strong_assert"]
    for prefix in ["double", "triple"]:
        verb_tvs = [r["tv"] for r in sa if r["continuation"] == f"{prefix}_verbatim"]
        if prefix == "double":
            ren_tvs = [r["tv"] for r in sa if r["continuation"] == "double_renamed"]
            mix_tvs = [r["tv"] for r in sa if r["continuation"] == "double_mixed"]
        else:
            ren_tvs = []
            mix_tvs = [r["tv"] for r in sa if r["continuation"] == "triple_mixed"]

        if verb_tvs:
            print(f"  {prefix}_verbatim: {np.mean(verb_tvs):.4f}")
        if ren_tvs:
            print(f"  {prefix}_renamed:  {np.mean(ren_tvs):.4f}")
        if mix_tvs:
            print(f"  {prefix}_mixed:    {np.mean(mix_tvs):.4f}")

    # Per-cell detail
    print("\n=== PER-CELL DETAIL (strong_assert) ===")
    cond_order = list(get_conditions("x", 1).keys())
    print(f"{'Cell':>8s}", end="")
    for cond in cond_order:
        print(f" {cond[:12]:>12s}", end="")
    print()
    for cell in ["x_1", "y_2", "y_6"]:
        print(f"{cell:>8s}", end="")
        for cond in cond_order:
            tvs = [r["tv"] for r in results
                   if r["pair"] == "strong_assert" and r["continuation"] == cond and r["cell"] == cell]
            if tvs:
                print(f" {tvs[0]:12.4f}", end="")
        print()

    summary = {
        "experiment": 77,
        "name": "rename_repeat",
        "n_readouts": n_done,
        "n_comparisons": len(results),
        "results": results,
    }

    result_dir = Path("experiments/results/omega_rename_repeat")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
