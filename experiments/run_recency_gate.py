"""Experiment #75: Recency Gate (Fable R7b E2).

Fable's key critique: on boundary_pair, identity's effect may be purely positional
(last-line-before-query wins). On strong_assert, it may be chain-based (survives
appended lines).

Test: identity (var=var) followed by k neutral lines (pass), k=0,1,2,4.
If positional: boundary_pair identity collapses at k=1.
If chain-based: strong_assert identity survives to k=4.

Also test heavy + k pass lines as control (heavy changes the actual value,
so its effect should survive regardless of position).

Design: 2 continuations x 5 k-values x 3 cells x 2 pairs x 2 surfaces = 120 readouts.
Plus empty control x 5 k-values = +60 readouts. Total: 180 readouts, ~4 min.
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
K_VALUES = [0, 1, 2, 4]


def run():
    print("=== Exp #75: Recency Gate (Fable E2) ===")

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

        id_text = f"{var} = {var}\n"
        heavy_text = f"{var} = {var} + 1  # WARNING: intentional mutation\n"

        for surf_name, surf_tmpl in SURFACES.items():
            surf_text = surf_tmpl.replace("{var}", var)
            for k in K_VALUES:
                pad = "pass\n" * k

                for cont_name, cont_text in [("empty", ""), ("identity", id_text), ("heavy", heavy_text)]:
                    full_suffix = surf_text + cont_text + pad + query
                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    all_dists[(var, val, surf_name, cont_name, k)] = dist
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
            for cont_name in ["empty", "identity", "heavy"]:
                for k in K_VALUES:
                    ka = (var, val, surf_a, cont_name, k)
                    kb = (var, val, surf_b, cont_name, k)
                    if ka in all_dists and kb in all_dists:
                        cell_tv = float(tv(all_dists[ka], all_dists[kb]))
                        flip = (DIGIT_LABELS[int(np.argmax(all_dists[ka]))] !=
                                DIGIT_LABELS[int(np.argmax(all_dists[kb]))])
                        results.append({
                            "pair": pair_name,
                            "cell": f"{var}_{val}",
                            "continuation": cont_name,
                            "k_pass": k,
                            "tv": cell_tv,
                            "flip": flip,
                        })

    # Recency gate: identity decay by pair
    for pair_name in ["strong_assert", "boundary_pair"]:
        print(f"\n=== RECENCY GATE: {pair_name} ===")
        print(f"{'k':>4s} {'empty':>10s} {'identity':>10s} {'id_amp':>10s} {'heavy':>10s} {'hvy_amp':>10s}")
        print("-" * 50)

        for k in K_VALUES:
            empty_tvs = [r["tv"] for r in results
                         if r["pair"] == pair_name and r["continuation"] == "empty" and r["k_pass"] == k]
            id_tvs = [r["tv"] for r in results
                      if r["pair"] == pair_name and r["continuation"] == "identity" and r["k_pass"] == k]
            heavy_tvs = [r["tv"] for r in results
                         if r["pair"] == pair_name and r["continuation"] == "heavy" and r["k_pass"] == k]

            e = np.mean(empty_tvs) if empty_tvs else 0.001
            i = np.mean(id_tvs) if id_tvs else 0
            h = np.mean(heavy_tvs) if heavy_tvs else 0
            i_amp = i / e if e > 0.001 else float('inf')
            h_amp = h / e if e > 0.001 else float('inf')
            print(f"{k:4d} {e:10.4f} {i:10.4f} {i_amp:10.1f}x {h:10.4f} {h_amp:10.1f}x")

    # Also show delta TV (Fable's recommended metric)
    for pair_name in ["strong_assert", "boundary_pair"]:
        print(f"\n=== DELTA TV (continuation - empty) : {pair_name} ===")
        print(f"{'k':>4s} {'identity_delta':>15s} {'heavy_delta':>15s}")
        print("-" * 40)

        for k in K_VALUES:
            empty_tvs = [r["tv"] for r in results
                         if r["pair"] == pair_name and r["continuation"] == "empty" and r["k_pass"] == k]
            id_tvs = [r["tv"] for r in results
                      if r["pair"] == pair_name and r["continuation"] == "identity" and r["k_pass"] == k]
            heavy_tvs = [r["tv"] for r in results
                         if r["pair"] == pair_name and r["continuation"] == "heavy" and r["k_pass"] == k]

            e = np.mean(empty_tvs) if empty_tvs else 0
            i = np.mean(id_tvs) if id_tvs else 0
            h = np.mean(heavy_tvs) if heavy_tvs else 0
            print(f"{k:4d} {i-e:15.4f} {h-e:15.4f}")

    # Per-cell identity decay
    print("\n=== PER-CELL IDENTITY TV BY K (strong_assert) ===")
    print(f"{'Cell':>8s}", end="")
    for k in K_VALUES:
        print(f" {'k='+str(k):>10s}", end="")
    print()
    for cell in ["x_1", "y_2", "y_6"]:
        print(f"{cell:>8s}", end="")
        for k in K_VALUES:
            tvs = [r["tv"] for r in results
                   if r["pair"] == "strong_assert" and r["continuation"] == "identity"
                   and r["cell"] == cell and r["k_pass"] == k]
            if tvs:
                print(f" {tvs[0]:10.4f}", end="")
        print()

    print("\n=== PER-CELL IDENTITY TV BY K (boundary_pair) ===")
    print(f"{'Cell':>8s}", end="")
    for k in K_VALUES:
        print(f" {'k='+str(k):>10s}", end="")
    print()
    for cell in ["x_1", "y_2", "y_6"]:
        print(f"{cell:>8s}", end="")
        for k in K_VALUES:
            tvs = [r["tv"] for r in results
                   if r["pair"] == "boundary_pair" and r["continuation"] == "identity"
                   and r["cell"] == cell and r["k_pass"] == k]
            if tvs:
                print(f" {tvs[0]:10.4f}", end="")
        print()

    summary = {
        "experiment": 75,
        "name": "recency_gate",
        "n_readouts": n_done,
        "n_comparisons": len(results),
        "results": results,
    }

    result_dir = Path("experiments/results/omega_recency_gate")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
