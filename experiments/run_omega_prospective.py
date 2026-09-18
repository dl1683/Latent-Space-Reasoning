"""Omega Prospective Consequence Test (Experiment #53).

Train omega thresholds on a TRAIN split of cells, test whether they
predict actual argmax flips on HELD-OUT cells. The key experiment:
can omega generalize as a predictive safety tool?
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv


MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "query_template": "f()\nprint({var})  # Output: ",
    "templates": {
        "depth4_single": (
            "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
            "    def g():\n        {var} = 999\n        def h():\n"
            "            {var} = 9999\n            def k():\n"
            "                {var} = 99999\n                return {var}\n"
            "            k()\n            return {var}\n"
            "        h()\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
    },
}

SURFACES = {
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
    "minimal_ok": "# OK.\n",
    "minimal_good": "# Good.\n",
}

EQUIV_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CONTINUATIONS = {
    "empty": "",
    "identity": "{var} = {var}\n",
    "nonvar_42": "_ = 42\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "neutral_pass": "pass\n",
    "type_check": "type({var})\n",
    "safety_comment": "# Do not modify {var}.\n",
    "deprecation": "# Deprecated: {var} is no longer used.\n",
}

# Wider cell grid for train/test splitting
ALL_CELLS = [
    (4, "x", 1), (4, "x", 3), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 4), (4, "y", 6), (4, "y", 8),
    (4, "z", 1), (4, "z", 3), (4, "z", 5), (4, "z", 9),
]

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]

# All possible 6-train/6-test splits (C(12,6) = 924, too many)
# Use stratified splits: one var each side
SPLITS = [
    {"train": [(4,"x",1),(4,"x",3),(4,"y",2),(4,"y",4),(4,"z",1),(4,"z",3)],
     "test":  [(4,"x",5),(4,"x",7),(4,"y",6),(4,"y",8),(4,"z",5),(4,"z",9)]},
    {"train": [(4,"x",5),(4,"x",7),(4,"y",6),(4,"y",8),(4,"z",5),(4,"z",9)],
     "test":  [(4,"x",1),(4,"x",3),(4,"y",2),(4,"y",4),(4,"z",1),(4,"z",3)]},
    {"train": [(4,"x",1),(4,"x",7),(4,"y",2),(4,"y",8),(4,"z",1),(4,"z",9)],
     "test":  [(4,"x",3),(4,"x",5),(4,"y",4),(4,"y",6),(4,"z",3),(4,"z",5)]},
    {"train": [(4,"x",3),(4,"x",5),(4,"y",4),(4,"y",6),(4,"z",3),(4,"z",5)],
     "test":  [(4,"x",1),(4,"x",7),(4,"y",2),(4,"y",8),(4,"z",1),(4,"z",9)]},
]


def collect_readouts(adapter, cells):
    query_tmpl = MODEL_CFG["query_template"]
    observations = []
    for depth, var, val in cells:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_name, surf_tmpl in SURFACES.items():
            surface_text = surf_tmpl.replace("{var}", var)
            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                full_suffix = surface_text + cont_text + query
                dist = adapter.get_dist_from_state(prefix_state, full_suffix)

                argmax_idx = int(np.argmax(dist))
                argmax_label = DIGIT_LABELS[argmax_idx]
                correct_digit = str(val)

                observations.append({
                    "depth": depth, "var": var, "val": val,
                    "surface": surf_name, "continuation": cont_name,
                    "dist": dist.tolist(),
                    "argmax_label": argmax_label,
                    "argmax_prob": float(dist[argmax_idx]),
                    "correct_digit": correct_digit,
                    "is_correct": argmax_label == correct_digit,
                })
    return observations


def compute_omega_and_flips(observations):
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = obs

    cells_in_data = list(set(
        (o["depth"], o["var"], o["val"]) for o in observations))

    results = []
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for cont_name in CONTINUATIONS:
            for depth, var, val in cells_in_data:
                key_a = (depth, var, val, surf_a, cont_name)
                key_b = (depth, var, val, surf_b, cont_name)
                if key_a not in obs_idx or key_b not in obs_idx:
                    continue
                oa = obs_idx[key_a]
                ob = obs_idx[key_b]
                cell_tv = tv(oa["dist"], ob["dist"])
                flip = oa["argmax_label"] != ob["argmax_label"]
                results.append({
                    "pair": pair_name,
                    "continuation": cont_name,
                    "cell": f"{var}_{val}",
                    "tv": float(cell_tv),
                    "flip": flip,
                    "correct_a": oa["is_correct"],
                    "correct_b": ob["is_correct"],
                })
    return results


def find_best_threshold(cell_data, thresholds):
    best_f1 = 0
    best_t = None
    best_stats = None
    for t in thresholds:
        above = [c for c in cell_data if c["tv"] >= t]
        below = [c for c in cell_data if c["tv"] < t]
        tp = sum(1 for c in above if c["flip"])
        fp = sum(1 for c in above if not c["flip"])
        fn = sum(1 for c in below if c["flip"])
        tn = sum(1 for c in below if not c["flip"])
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        stats = {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                 "precision": p, "recall": r, "f1": f1}
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_stats = stats
    return best_t, best_stats


def run_prospective():
    print("=== Omega Prospective Consequence Test ===")
    n_cells = len(ALL_CELLS)
    n_readouts = n_cells * len(SURFACES) * len(CONTINUATIONS)
    print(f"Total: {n_readouts} readouts ({n_cells} cells × "
          f"{len(SURFACES)} surfaces × {len(CONTINUATIONS)} continuations)")
    print(f"Splits: {len(SPLITS)} train/test configurations")

    adapter = ModelAdapter(MODEL_CFG)
    t0 = time.time()

    print("Collecting all readouts...")
    all_obs = collect_readouts(adapter, ALL_CELLS)
    elapsed = time.time() - t0
    print(f"Done: {len(all_obs)} readouts in {elapsed:.1f}s")

    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    split_results = []
    for i, split in enumerate(SPLITS):
        train_cells_set = set(tuple(c) for c in split["train"])
        test_cells_set = set(tuple(c) for c in split["test"])

        train_obs = [o for o in all_obs
                     if (o["depth"], o["var"], o["val"]) in train_cells_set]
        test_obs = [o for o in all_obs
                    if (o["depth"], o["var"], o["val"]) in test_cells_set]

        train_data = compute_omega_and_flips(train_obs)
        test_data = compute_omega_and_flips(test_obs)

        train_t, train_stats = find_best_threshold(train_data, thresholds)

        # Apply train threshold to test data
        test_above = [c for c in test_data if c["tv"] >= train_t]
        test_below = [c for c in test_data if c["tv"] < train_t]
        tp = sum(1 for c in test_above if c["flip"])
        fp = sum(1 for c in test_above if not c["flip"])
        fn = sum(1 for c in test_below if c["flip"])
        tn = sum(1 for c in test_below if not c["flip"])
        test_p = tp / (tp + fp) if (tp + fp) > 0 else 0
        test_r = tp / (tp + fn) if (tp + fn) > 0 else 0
        test_f1 = 2 * test_p * test_r / (test_p + test_r) if (test_p + test_r) > 0 else 0

        # Correlation on test set
        tvs = np.array([c["tv"] for c in test_data])
        flips = np.array([1 if c["flip"] else 0 for c in test_data])
        if flips.std() > 0 and tvs.std() > 0:
            test_corr = float(np.corrcoef(tvs, flips)[0, 1])
        else:
            test_corr = 0.0

        # Also test all thresholds on test data for comparison
        test_best_t, test_best_stats = find_best_threshold(test_data, thresholds)

        sr = {
            "split_idx": i,
            "train_cells": [list(c) for c in split["train"]],
            "test_cells": [list(c) for c in split["test"]],
            "n_train_obs": len(train_obs),
            "n_test_obs": len(test_obs),
            "n_train_pairs": len(train_data),
            "n_test_pairs": len(test_data),
            "train_threshold": train_t,
            "train_f1": train_stats["f1"],
            "train_precision": train_stats["precision"],
            "train_recall": train_stats["recall"],
            "test_threshold_from_train": train_t,
            "test_f1": test_f1,
            "test_precision": test_p,
            "test_recall": test_r,
            "test_tp": tp, "test_fp": fp, "test_fn": fn, "test_tn": tn,
            "test_correlation": test_corr,
            "test_oracle_threshold": test_best_t,
            "test_oracle_f1": test_best_stats["f1"] if test_best_stats else 0,
            "n_train_flips": sum(1 for c in train_data if c["flip"]),
            "n_test_flips": sum(1 for c in test_data if c["flip"]),
        }
        split_results.append(sr)

    # Aggregate
    train_f1s = [s["train_f1"] for s in split_results]
    test_f1s = [s["test_f1"] for s in split_results]
    test_corrs = [s["test_correlation"] for s in split_results]
    oracle_f1s = [s["test_oracle_f1"] for s in split_results]

    summary = {
        "n_readouts": len(all_obs),
        "n_cells": n_cells,
        "n_splits": len(SPLITS),
        "splits": split_results,
        "aggregate": {
            "train_f1_mean": float(np.mean(train_f1s)),
            "train_f1_std": float(np.std(train_f1s)),
            "test_f1_mean": float(np.mean(test_f1s)),
            "test_f1_std": float(np.std(test_f1s)),
            "test_correlation_mean": float(np.mean(test_corrs)),
            "test_correlation_std": float(np.std(test_corrs)),
            "oracle_f1_mean": float(np.mean(oracle_f1s)),
            "f1_gap": float(np.mean(oracle_f1s) - np.mean(test_f1s)),
            "generalization_ratio": float(np.mean(test_f1s) / np.mean(train_f1s))
                if np.mean(train_f1s) > 0 else 0,
        },
    }

    result_dir = Path("experiments/results/omega_prospective")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(all_obs, f)

    # Print
    print(f"\n=== PROSPECTIVE RESULTS ({len(SPLITS)} splits) ===")
    print(f"Total readouts: {len(all_obs)}")
    print(f"\nAggregate:")
    agg = summary["aggregate"]
    print(f"  Train F1:      {agg['train_f1_mean']:.3f} ± {agg['train_f1_std']:.3f}")
    print(f"  Test F1:       {agg['test_f1_mean']:.3f} ± {agg['test_f1_std']:.3f}")
    print(f"  Oracle F1:     {agg['oracle_f1_mean']:.3f}")
    print(f"  F1 gap:        {agg['f1_gap']:.3f}")
    print(f"  Generalization: {agg['generalization_ratio']:.1%}")
    print(f"  Test corr:     {agg['test_correlation_mean']:.3f} ± {agg['test_correlation_std']:.3f}")

    for sr in split_results:
        print(f"\n  Split {sr['split_idx']}:")
        print(f"    Train: t={sr['train_threshold']}, F1={sr['train_f1']:.3f}, "
              f"P={sr['train_precision']:.2f}, R={sr['train_recall']:.2f}")
        print(f"    Test:  t={sr['test_threshold_from_train']}, F1={sr['test_f1']:.3f}, "
              f"P={sr['test_precision']:.2f}, R={sr['test_recall']:.2f}")
        print(f"    Test correlation: {sr['test_correlation']:.3f}")
        print(f"    Flips: train={sr['n_train_flips']}, test={sr['n_test_flips']}")
        print(f"    Oracle: t={sr['test_oracle_threshold']}, F1={sr['test_oracle_f1']:.3f}")


if __name__ == "__main__":
    run_prospective()
