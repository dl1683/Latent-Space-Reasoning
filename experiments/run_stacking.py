"""Experiment #73: Amplification Stacking.

We now have a taxonomy of amplification primitives:
- Self-assignment (var=var): ~13x
- Value collision (_ = val): ~11x
- Value mention (print(val)): ~8x
- Deep inspection (isinstance): ~9x

Key question: Do these STACK (multiplicative/additive) or SATURATE (shared mechanism)?

Tests all combinations of 2-element compositions from these primitives.
If mechanisms are independent: combined > max(individual).
If shared: combined ≈ max(individual).

Design: 10 2-element compositions + 5 singles + 1 empty = 16 conditions x
3 cells x 2 pairs x 2 surfaces = 192 readouts. ~4 min.
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
    mismatch_val = [v for v in [1, 2, 3, 5, 6, 7] if v != val][0]
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
        "identity+print": f"{var} = {var}\nprint({val})\n",
        "identity+isinstance": f"{var} = {var}\nisinstance({var}, int)\n",
        "collision+print": f"_ = {val}\nprint({val})\n",
        "collision+isinstance": f"_ = {val}\nisinstance({var}, int)\n",
        "identity+heavy": f"{var} = {var}\n{var} = {var} + 1  # WARNING: intentional mutation\n",
        "collision+heavy": f"_ = {val}\n{var} = {var} + 1  # WARNING: intentional mutation\n",
        "double_identity": f"{var} = {var}\n{var} = {var}\n",
        "double_collision": f"_ = {val}\n_ = {val}\n",
    }
    return {**singles, **stacks}


def run():
    print("=== Exp #73: Amplification Stacking ===")

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
                        "condition": cond_name,
                        "tv": cell_tv,
                        "flip": flip,
                    })

    # Singles
    print("\n=== SINGLES (strong_assert) ===")
    print(f"{'Condition':>25s} {'mean_TV':>10s} {'vs_empty':>10s}")
    print("-" * 50)
    empty_tvs = [r["tv"] for r in results
                 if r["pair"] == "strong_assert" and r["condition"] == "empty"]
    empty_mean = np.mean(empty_tvs) if empty_tvs else 0.001

    for cond in ["empty", "identity", "collision", "print_val", "isinstance", "heavy"]:
        tvs = [r["tv"] for r in results
               if r["pair"] == "strong_assert" and r["condition"] == cond]
        if tvs:
            m = np.mean(tvs)
            ratio = m / empty_mean if empty_mean > 0.001 else float('inf')
            print(f"{cond:>25s} {m:10.4f} {ratio:10.1f}x")

    # Stacks
    print("\n=== STACKS vs INDIVIDUAL MAX (strong_assert) ===")
    print(f"{'Stack':>25s} {'stack_TV':>10s} {'max_single':>10s} {'super':>10s} {'additivity':>10s}")
    print("-" * 70)

    singles_mean = {}
    for cond in ["empty", "identity", "collision", "print_val", "isinstance", "heavy"]:
        tvs = [r["tv"] for r in results
               if r["pair"] == "strong_assert" and r["condition"] == cond]
        singles_mean[cond] = np.mean(tvs) if tvs else 0

    stack_pairs = [
        ("identity+collision", "identity", "collision"),
        ("collision+identity", "collision", "identity"),
        ("identity+print", "identity", "print_val"),
        ("identity+isinstance", "identity", "isinstance"),
        ("collision+print", "collision", "print_val"),
        ("collision+isinstance", "collision", "isinstance"),
        ("identity+heavy", "identity", "heavy"),
        ("collision+heavy", "collision", "heavy"),
        ("double_identity", "identity", "identity"),
        ("double_collision", "collision", "collision"),
    ]

    for stack_name, comp_a, comp_b in stack_pairs:
        tvs = [r["tv"] for r in results
               if r["pair"] == "strong_assert" and r["condition"] == stack_name]
        if tvs:
            stack_mean = np.mean(tvs)
            max_single = max(singles_mean[comp_a], singles_mean[comp_b])
            sum_single = singles_mean[comp_a] + singles_mean[comp_b]
            super_ratio = stack_mean / max_single if max_single > 0.001 else float('inf')
            add_ratio = stack_mean / sum_single if sum_single > 0.001 else float('inf')
            print(f"{stack_name:>25s} {stack_mean:10.4f} {max_single:10.4f} "
                  f"{super_ratio:10.1f}x {add_ratio:10.1f}x")

    # Commutativity check
    print("\n=== COMMUTATIVITY CHECK ===")
    ic_tvs = [r["tv"] for r in results
              if r["pair"] == "strong_assert" and r["condition"] == "identity+collision"]
    ci_tvs = [r["tv"] for r in results
              if r["pair"] == "strong_assert" and r["condition"] == "collision+identity"]
    if ic_tvs and ci_tvs:
        ic = np.mean(ic_tvs)
        ci = np.mean(ci_tvs)
        print(f"  identity+collision: {ic:.4f}")
        print(f"  collision+identity: {ci:.4f}")
        print(f"  Ratio: {ic/ci:.2f}x (1.0 = commutative)")

    summary = {
        "experiment": 73,
        "name": "amplification_stacking",
        "n_readouts": n_done,
        "n_comparisons": len(results),
        "results": results,
    }

    result_dir = Path("experiments/results/omega_stacking")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
