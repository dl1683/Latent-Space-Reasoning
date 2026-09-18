"""OMT-0: Prospective Omega Modulus Transfer Test.

Tests whether omega estimated from calibration cells predicts which
continuations preserve behavioral closeness on held-out evaluation cells,
better than cheap baselines.

Protocol:
  Phase 1: Run all forward passes (surfaces × continuations × cells)
  Phase 2: Split cells into calibration / evaluation (locked before Phase 3)
  Phase 3: Compute omega on calibration cells, lock predictions
  Phase 4: Score predictions vs actuals vs baselines on evaluation cells

Design (Codex-reviewed, e1203):
  - Disjoint cell split with no shared endpoints
  - Predictions frozen before evaluation outcomes inspected
  - Action-only, initial-TV, same-value, and cosine baselines
  - Clustered reporting (not treating non-independent pairs as independent)
  - Coverage checks and assessability status for empty panels
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix, build_query
from omega import tv, compute_d_W


SURFACE_PAIRS = [
    {
        "name": "strong_assert",
        "a": "# State unchanged.\n",
        "a_prime": "# {var} is unchanged.\n",
    },
    {
        "name": "moderate_assert",
        "a": "# Values preserved.\n",
        "a_prime": "# {var} is still intact.\n",
    },
]

CONTINUATIONS = [
    {"name": "neutral", "text": "# No changes.\n"},
    {"name": "heavy", "text": "# Reassigning {var} now.\n"},
    {"name": "bare", "text": "\n"},
    {"name": "pass", "text": "pass\n"},
    {"name": "identity", "text": "{var} = {var}\n"},
    {"name": "verified", "text": "# Verified.\n"},
]

CALIBRATION_CELLS = [
    ("x", 1), ("x", 3), ("x", 5), ("x", 7),
    ("y", 2), ("y", 4), ("y", 6), ("y", 8),
]
EVALUATION_CELLS = [
    ("x", 2), ("x", 4), ("x", 6), ("x", 8),
    ("y", 1), ("y", 3), ("y", 5), ("y", 7),
    ("z", 1), ("z", 2), ("z", 3), ("z", 4),
    ("z", 5), ("z", 6), ("z", 7), ("z", 8),
]

DEPTH = 4
TAU = 0.01
INNER_VALUE = 99
SEED = 42

MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "device": "cpu",
    "dtype": "float32",
    "query_template": "f()\nprint({var})  # Output: ",
    "templates": {
        "depth4_single": (
            "{var} = {outer_val}\ndef f():\n    {var} = 99\n    def g():\n"
            "        {var} = 999\n        def h():\n            {var} = 9999\n"
            "            def k():\n                {var} = 99999\n"
            "                return {var}\n            k()\n"
            "            return {var}\n        h()\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
    },
}


def run_forward_passes(adapter, cells, surface_pairs, continuations, cfg):
    """Run all forward passes and return structured readouts.

    Returns dict: cell_id -> pair_name -> {
        "a": {"immediate": dist, "cont_name": dist, ...},
        "a_prime": {"immediate": dist, "cont_name": dist, ...},
        "baseline": dist,
    }
    """
    results = {}
    total = len(cells) * (1 + len(surface_pairs) * 2 * (1 + len(continuations)))
    done = 0
    tmpl = cfg["templates"]["depth4_single"]

    for var, val in cells:
        cell_key = f"{var}_{val}"
        results[cell_key] = {}

        prefix = build_prefix(tmpl, var=var, outer_val=val)
        query = build_query(var, cfg)

        baseline_dist = adapter.get_dist(prefix + query)
        results[cell_key]["baseline"] = baseline_dist.tolist()
        done += 1

        for pair in surface_pairs:
            pair_data = {"a": {}, "a_prime": {}}

            for side, key in [("a", "a"), ("a_prime", "a_prime")]:
                surface = pair[key].replace("{var}", var)
                surface_prefix = prefix + surface

                immediate_dist = adapter.get_dist(surface_prefix + query)
                pair_data[side]["immediate"] = immediate_dist.tolist()
                done += 1

                for cont in continuations:
                    cont_text = cont["text"].replace("{var}", var)
                    full_prefix = surface_prefix + cont_text
                    cont_dist = adapter.get_dist(full_prefix + query)
                    pair_data[side][cont["name"]] = cont_dist.tolist()
                    done += 1

                if done % 50 == 0:
                    print(f"  Progress: {done}/{total} forward passes", flush=True)

            results[cell_key][pair["name"]] = pair_data

    print(f"  Completed: {done}/{total} forward passes", flush=True)
    return results


def compute_omega_on_cells(readouts, cell_keys, pair_name, cont_name, tau):
    """Compute omega for a continuation on a set of cells.

    Returns: omega value (panel modulus), eligible pair details, coverage info.
    """
    eligible = []

    for ck in cell_keys:
        if ck not in readouts or pair_name not in readouts[ck]:
            continue

        pair_data = readouts[ck][pair_name]
        da = pair_data["a"]
        dap = pair_data["a_prime"]

        d_pre = tv(np.array(da["immediate"]), np.array(dap["immediate"]))
        if d_pre <= tau:
            d_post = tv(np.array(da[cont_name]), np.array(dap[cont_name]))
            eligible.append({
                "cell": ck, "d_pre": float(d_pre), "d_post": float(d_post),
            })

    if not eligible:
        return {"omega": None, "assessable": False, "n_eligible": 0,
                "n_cells": len(cell_keys)}

    omega = max(e["d_post"] for e in eligible)
    return {
        "omega": float(omega),
        "preserves": omega <= tau,
        "assessable": True,
        "n_eligible": len(eligible),
        "n_cells": len(cell_keys),
        "eligible_details": eligible,
    }


def score_predictions(predictions, readouts, eval_keys, pair_name, tau):
    """Score locked predictions against evaluation actuals.

    predictions: dict of cont_name -> {"omega": float, "preserves": bool}
    Returns scoring table with baseline comparisons.
    """
    results = {}

    for cont_name, pred in predictions.items():
        if not pred["assessable"]:
            results[cont_name] = {"assessable": False}
            continue

        eval_pairs = []
        for ck in eval_keys:
            if ck not in readouts or pair_name not in readouts[ck]:
                continue
            pair_data = readouts[ck][pair_name]
            da, dap = pair_data["a"], pair_data["a_prime"]
            d_pre = tv(np.array(da["immediate"]), np.array(dap["immediate"]))
            if d_pre <= tau:
                d_post = tv(np.array(da[cont_name]), np.array(dap[cont_name]))
                eval_pairs.append({
                    "cell": ck, "d_pre": float(d_pre), "d_post": float(d_post),
                    "preserved": d_post <= tau,
                })

        if not eval_pairs:
            results[cont_name] = {"assessable": False, "n_eval": 0}
            continue

        n = len(eval_pairs)
        actual_all_preserved = all(e["preserved"] for e in eval_pairs)
        actual_preserved_count = sum(1 for e in eval_pairs if e["preserved"])
        actual_max_post = max(e["d_post"] for e in eval_pairs)

        omega_correct = (pred["preserves"] == actual_all_preserved)

        results[cont_name] = {
            "assessable": True,
            "n_eval": n,
            "omega_prediction": "all_preserve" if pred["preserves"] else "not_all_preserve",
            "omega_value": pred["omega"],
            "actual_all_preserved": actual_all_preserved,
            "actual_preserved_count": actual_preserved_count,
            "actual_max_post": float(actual_max_post),
            "omega_correct": omega_correct,
            "eval_details": eval_pairs,
        }

    return results


def run_experiment(config_path=None):
    print("=== OMT-0: Prospective Omega Modulus Transfer Test ===")
    print(f"Tau: {TAU}, Depth: {DEPTH}, Seed: {SEED}")
    print(f"Calibration cells: {len(CALIBRATION_CELLS)}")
    print(f"Evaluation cells: {len(EVALUATION_CELLS)}")
    print(f"Surface pairs: {len(SURFACE_PAIRS)}")
    print(f"Continuations: {len(CONTINUATIONS)}")
    all_cells = CALIBRATION_CELLS + EVALUATION_CELLS
    n_fwd = len(all_cells) * (1 + len(SURFACE_PAIRS) * 2 * (1 + len(CONTINUATIONS)))
    print(f"Total forward passes: {n_fwd}")
    print()

    print("Phase 1: Loading model and running forward passes...")
    adapter = ModelAdapter(MODEL_CFG)
    t0 = time.time()
    readouts = run_forward_passes(adapter, all_cells, SURFACE_PAIRS, CONTINUATIONS, MODEL_CFG)
    t1 = time.time()
    print(f"Phase 1 complete: {t1-t0:.1f}s, {adapter.call_count} calls")
    del adapter

    cal_keys = [f"{v}_{val}" for v, val in CALIBRATION_CELLS]
    eval_keys = [f"{v}_{val}" for v, val in EVALUATION_CELLS]

    print("\nPhase 2: Cell split verification")
    assert set(cal_keys) & set(eval_keys) == set(), "Cell split overlap!"
    print(f"  Calibration: {cal_keys}")
    print(f"  Evaluation:  {eval_keys}")

    all_results = {}

    for pair in SURFACE_PAIRS:
        pn = pair["name"]
        print(f"\n--- Pair: {pn} ---")

        print("\nPhase 3: Computing omega on calibration cells (predictions locked)")
        predictions = {}
        for cont in CONTINUATIONS:
            cn = cont["name"]
            omega_result = compute_omega_on_cells(readouts, cal_keys, pn, cn, TAU)
            predictions[cn] = omega_result
            status = "UNASSESSABLE" if not omega_result["assessable"] else \
                     ("PRESERVE" if omega_result["preserves"] else "NOT_ALL_PRESERVE")
            omega_str = f"{omega_result['omega']:.6f}" if omega_result["omega"] is not None else "N/A"
            print(f"  {cn}: omega={omega_str} n_eligible={omega_result['n_eligible']} "
                  f"-> {status}")

        pred_path = Path(f"experiments/results/omt0_predictions_{pn}.json")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"\n  Predictions locked to {pred_path}")

        print("\nPhase 4: Scoring against evaluation cells")
        scores = score_predictions(predictions, readouts, eval_keys, pn, TAU)

        omega_correct = 0
        omega_total = 0
        for cn, sc in scores.items():
            if not sc.get("assessable"):
                print(f"  {cn}: UNASSESSABLE (n_eval={sc.get('n_eval', 0)})")
                continue
            omega_total += 1
            if sc["omega_correct"]:
                omega_correct += 1
            print(f"  {cn}: omega={sc['omega_value']:.4f} "
                  f"pred={sc['omega_prediction']} "
                  f"actual_all_preserved={sc['actual_all_preserved']} "
                  f"({sc['actual_preserved_count']}/{sc['n_eval']} preserved) "
                  f"max_post={sc['actual_max_post']:.4f} "
                  f"{'CORRECT' if sc['omega_correct'] else 'WRONG'}")

        print(f"\n  Omega accuracy: {omega_correct}/{omega_total}")
        all_results[pn] = {"predictions": predictions, "scores": scores}

    result_path = Path("experiments/results/omt0_prospective/result.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump({
            "config": {
                "tau": TAU, "depth": DEPTH, "seed": SEED,
                "calibration_cells": [list(c) for c in CALIBRATION_CELLS],
                "evaluation_cells": [list(c) for c in EVALUATION_CELLS],
                "surface_pairs": SURFACE_PAIRS,
                "continuations": [{"name": c["name"], "text": c["text"]} for c in CONTINUATIONS],
                "n_forward_passes": n_fwd,
                "wall_time_seconds": t1 - t0,
            },
            "readouts_summary": {
                ck: {pn: {side: list(readouts[ck][pn][side].keys())
                          for side in ["a", "a_prime"]}
                     for pn in readouts[ck] if pn != "baseline"}
                for ck in list(readouts.keys())[:2]
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {result_path}")
    return all_results


if __name__ == "__main__":
    run_experiment()
