"""Experiment #81: Pre-registered Identity Prediction Test (OMT-0 Endgame).

PRE-REGISTERED PREDICTIONS (written before any data collection):

4 novel surface pairs, never tested before. For each, we predict whether
identity amplifies (id_amp > 2x) or suppresses (id_amp <= 2x), and the
approximate magnitude, using the two-factor model:
  Factor 1: meaningfulness (arbitrary=0, meaningful=1)
  Factor 2: baseline level (lower baseline -> higher amplification within group)

PAIR 1: var_echo ("# {var}\n", "# .\n")
  - Classification: ARBITRARY (single character / variable name in comment)
  - Expected baseline: ~0.01-0.02 (short, structurally similar comments)
  - PREDICTION: AMPLIFIES, id_amp = 3-8x
  - Reasoning: arbitrary tokens, low baseline -> chain-resolution regime

PAIR 2: english_synonym ("# same\n", "# equal\n")
  - Classification: MEANINGFUL (real English words with similar meaning)
  - Expected baseline: ~0.01-0.03 (similar short words)
  - PREDICTION: SUPPRESSES, id_amp = 0.3-1.0x
  - Reasoning: meaningful tokens -> language model competition

PAIR 3: code_return ("return {var}\n", "return {val}\n")
  - Classification: ARBITRARY (code syntax, different reference targets)
  - Expected baseline: ~0.02-0.05 (return var vs return literal)
  - PREDICTION: AMPLIFIES, id_amp = 2-6x
  - Reasoning: code tokens, moderate baseline, structurally similar

PAIR 4: pragma ("# type: ignore\n", "# noqa\n")
  - Classification: ARBITRARY (tool pragmas, not meaningful to LM as language)
  - Expected baseline: ~0.01-0.03 (both are pragmatic comments)
  - PREDICTION: AMPLIFIES, id_amp = 3-7x
  - Reasoning: arbitrary/technical tokens, expected low baseline

GATE: >= 3/4 directional predictions correct = model validates.
STRONG GATE: >= 3/4 magnitude predictions within predicted range.

Design: 4 pairs x 3 conditions (empty/identity/heavy) x 3 cells x 2 surfaces
= 72 readouts, ~30s.
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
    "var_echo": ("# {var}\n", "# .\n"),
    "english_synonym": ("# same\n", "# equal\n"),
    "code_return": ("return {var}\n", "return {val}\n"),
    "pragma": ("# type: ignore\n", "# noqa\n"),
}

PREDICTIONS = {
    "var_echo": {"direction": "amplify", "range": (3.0, 8.0), "meaningful": False},
    "english_synonym": {"direction": "suppress", "range": (0.3, 1.0), "meaningful": True},
    "code_return": {"direction": "amplify", "range": (2.0, 6.0), "meaningful": False},
    "pragma": {"direction": "amplify", "range": (3.0, 7.0), "meaningful": False},
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
    print("=== Exp #81: Pre-registered Identity Prediction Test ===")
    print("(Predictions registered in source code BEFORE data collection)\n")

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

        for pair_name, (surf_a_tmpl, surf_b_tmpl) in SURFACE_PAIRS.items():
            surf_a = surf_a_tmpl.replace("{var}", var).replace("{val}", str(val))
            surf_b = surf_b_tmpl.replace("{var}", var).replace("{val}", str(val))

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

    # Score predictions
    print("\n" + "=" * 80)
    print("=== PRE-REGISTERED PREDICTION SCORECARD ===")
    print("=" * 80)

    direction_correct = 0
    range_correct = 0
    total = 0

    for pair_name, pred in PREDICTIONS.items():
        empty_tvs = [r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "empty"]
        id_tvs = [r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "identity"]
        heavy_tvs = [r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "heavy"]

        baseline = np.mean(empty_tvs) if empty_tvs else 0.001
        id_tv = np.mean(id_tvs) if id_tvs else 0
        heavy_tv = np.mean(heavy_tvs) if heavy_tvs else 0
        id_amp = id_tv / baseline if baseline > 0.001 else 0

        actual_dir = "amplify" if id_amp > 2.0 else "suppress"
        dir_match = actual_dir == pred["direction"]
        lo, hi = pred["range"]
        range_match = lo <= id_amp <= hi

        direction_correct += dir_match
        range_correct += range_match
        total += 1

        dir_icon = "OK" if dir_match else "WRONG"
        range_icon = "OK" if range_match else "MISS"

        print(f"\n  {pair_name}:")
        print(f"    Predicted: {pred['direction']} ({lo:.1f}-{hi:.1f}x)")
        print(f"    Actual:    {actual_dir} ({id_amp:.1f}x)")
        print(f"    Baseline:  {baseline:.4f}")
        print(f"    Identity TV: {id_tv:.4f}")
        print(f"    Heavy TV:  {heavy_tv:.4f}")
        print(f"    Direction: {dir_icon}    Range: {range_icon}")

    print(f"\n{'=' * 80}")
    print(f"DIRECTION GATE: {direction_correct}/{total} correct  {'PASS' if direction_correct >= 3 else 'FAIL'}")
    print(f"RANGE GATE:     {range_correct}/{total} correct  {'PASS' if range_correct >= 3 else 'FAIL'}")
    print(f"{'=' * 80}")

    if direction_correct >= 3:
        print("\nVERDICT: Two-factor model VALIDATES on held-out data.")
        print("Omega modulus is predictable from question family properties.")
    else:
        print("\nVERDICT: Two-factor model FAILS on held-out data.")
        print("Need more factors or the model is overfitting training pairs.")

    summary = {
        "experiment": 81,
        "name": "preregistered_predict",
        "n_readouts": n_done,
        "n_comparisons": len(results),
        "predictions": PREDICTIONS,
        "direction_correct": direction_correct,
        "range_correct": range_correct,
        "total": total,
        "results": results,
    }

    result_dir = Path("experiments/results/omega_preregistered_predict")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
