"""Experiment #80: Surface Type Dissection.

Open signal s261: Why do boundary and comment_generic have opposite identity
responses (6.0x vs 0.3x) despite similar baselines (0.013 vs 0.019)?

Hypothesis: Identity amplification depends on whether the surface pair differs
in CODE STRUCTURE or COMMENT CONTENT.
- Structural surface differences (blank vs statement) -> identity amplifies
- Semantic surface differences (comment A vs comment B) -> identity suppresses

Test: 6 surface pairs systematically varying the TYPE of difference:
1. stmt_vs_blank: ("pass\n", "\n") — structural, like boundary
2. stmt_vs_stmt: ("pass\n", "_ = 0\n") — both statements, differ in effect
3. comment_short: ("# a\n", "# b\n") — minimal comment difference
4. comment_long: ("# note here\n", "# info here\n") — like comment_generic
5. cross_type: ("pass\n", "# note\n") — code vs comment
6. blank_vs_blank: ("  \n", "   \n") — minimal whitespace difference

3 continuations (empty/identity/heavy) x 3 cells x 6 pairs = 108 readouts.
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

SURFACE_PAIRS = {
    "stmt_vs_blank": ("pass\n", "\n"),
    "stmt_vs_stmt": ("pass\n", "_ = 0\n"),
    "comment_short": ("# a\n", "# b\n"),
    "comment_long": ("# note here\n", "# info here\n"),
    "cross_type": ("pass\n", "# note\n"),
    "blank_vs_blank": ("  \n", "   \n"),
}

PREFIX_TEMPLATE = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def run():
    print("=== Exp #80: Surface Type Dissection ===")

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

        for pair_name, (surf_a, surf_b) in SURFACE_PAIRS.items():
            for cont_name, cont_text in [("empty", ""), ("identity", id_text), ("heavy", heavy_text)]:
                suffix_a = surf_a + cont_text + query
                suffix_b = surf_b + cont_text + query

                dist_a = adapter.get_dist_from_state(prefix_state, suffix_a)
                dist_b = adapter.get_dist_from_state(prefix_state, suffix_b)
                all_dists[(var, val, pair_name, "a", cont_name)] = dist_a
                all_dists[(var, val, pair_name, "b", cont_name)] = dist_b
                n_done += 2

        del prefix_state
        gc.collect()
        elapsed = time.time() - t0
        print(f"  Cell {var}_{val} done, {n_done} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} readouts in {elapsed:.1f}s")

    results = []
    for pair_name in SURFACE_PAIRS:
        for var, val in CELLS:
            for cont_name in ["empty", "identity", "heavy"]:
                da = all_dists.get((var, val, pair_name, "a", cont_name))
                db = all_dists.get((var, val, pair_name, "b", cont_name))
                if da is not None and db is not None:
                    cell_tv = float(tv(da, db))
                    flip = (DIGIT_LABELS[int(np.argmax(da))] !=
                            DIGIT_LABELS[int(np.argmax(db))])
                    results.append({
                        "pair": pair_name,
                        "cell": f"{var}_{val}",
                        "continuation": cont_name,
                        "tv": cell_tv,
                        "flip": flip,
                    })

    # Main table
    print("\n=== SURFACE TYPE TABLE ===")
    print(f"{'Pair':>20s} {'type':>15s} {'baseline':>10s} {'identity':>10s} {'id_amp':>10s} {'id_delta':>10s} {'heavy':>10s} {'hvy_amp':>10s}")
    print("-" * 105)

    type_labels = {
        "stmt_vs_blank": "structural",
        "stmt_vs_stmt": "structural",
        "comment_short": "semantic",
        "comment_long": "semantic",
        "cross_type": "cross",
        "blank_vs_blank": "whitespace",
    }

    for pair_name in SURFACE_PAIRS:
        empty_tvs = [r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "empty"]
        id_tvs = [r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "identity"]
        heavy_tvs = [r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "heavy"]

        e = np.mean(empty_tvs) if empty_tvs else 0.001
        i = np.mean(id_tvs) if id_tvs else 0
        h = np.mean(heavy_tvs) if heavy_tvs else 0
        i_amp = i / e if e > 0.001 else float('inf')
        h_amp = h / e if e > 0.001 else float('inf')

        print(f"{pair_name:>20s} {type_labels[pair_name]:>15s} {e:10.4f} {i:10.4f} {i_amp:10.1f}x {i-e:+10.4f} {h:10.4f} {h_amp:10.1f}x")

    # Summary by surface type
    print("\n=== BY SURFACE TYPE ===")
    for stype in ["structural", "semantic", "cross", "whitespace"]:
        matching = [p for p, t in type_labels.items() if t == stype]
        all_id_amp = []
        all_hvy_amp = []
        for p in matching:
            empty_tvs = [r["tv"] for r in results if r["pair"] == p and r["continuation"] == "empty"]
            id_tvs = [r["tv"] for r in results if r["pair"] == p and r["continuation"] == "identity"]
            heavy_tvs = [r["tv"] for r in results if r["pair"] == p and r["continuation"] == "heavy"]
            e = np.mean(empty_tvs) if empty_tvs else 0.001
            i = np.mean(id_tvs) if id_tvs else 0
            h = np.mean(heavy_tvs) if heavy_tvs else 0
            all_id_amp.append(i / e if e > 0.001 else 0)
            all_hvy_amp.append(h / e if e > 0.001 else 0)
        print(f"  {stype:>12s}: mean id_amp={np.mean(all_id_amp):.1f}x, mean hvy_amp={np.mean(all_hvy_amp):.1f}x")

    summary = {
        "experiment": 80,
        "name": "surface_type",
        "n_readouts": n_done,
        "n_comparisons": len(results),
        "results": results,
    }

    result_dir = Path("experiments/results/omega_surface_type")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
