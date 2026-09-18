"""Omega Cross-Model Consequence Test (Experiment #55).

Tests whether the tau=0.3 threshold from exp #49 (Qwen3-1.7B) generalizes
to other model sizes. Runs consequence prediction on 0.6B and 4B.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv


def make_cfg(model_id):
    return {
        "model_id": model_id,
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
    "safety_comment": "# Do not modify {var}.\n",
    "deprecation": "# Deprecated: {var} is no longer used.\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
MODELS = ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-4B-Base"]


def run_for_model(model_id):
    print(f"\n=== {model_id} ===")
    cfg = make_cfg(model_id)
    adapter = ModelAdapter(cfg)
    query_tmpl = cfg["query_template"]
    observations = []
    t0 = time.time()

    for depth, var, val in CELLS:
        tmpl = cfg["templates"]["depth4_single"]
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

    elapsed = time.time() - t0
    print(f"  {len(observations)} readouts in {elapsed:.1f}s")

    import gc, torch
    del adapter
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return observations


def analyze_model(observations, model_id):
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = obs

    all_pairs = []
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for cont_name in CONTINUATIONS:
            for depth, var, val in CELLS:
                oa = obs_idx[(depth, var, val, surf_a, cont_name)]
                ob = obs_idx[(depth, var, val, surf_b, cont_name)]
                cell_tv = tv(oa["dist"], ob["dist"])
                flip = oa["argmax_label"] != ob["argmax_label"]
                all_pairs.append({
                    "pair": pair_name, "continuation": cont_name,
                    "cell": f"{var}_{val}",
                    "tv": float(cell_tv), "flip": flip,
                })

    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    tvs = np.array([p["tv"] for p in all_pairs])
    flips = np.array([1 if p["flip"] else 0 for p in all_pairs])
    if flips.std() > 0 and tvs.std() > 0:
        corr = float(np.corrcoef(tvs, flips)[0, 1])
    else:
        corr = 0.0

    threshold_results = {}
    best_f1 = 0
    best_t = None
    for t in thresholds:
        above = [p for p in all_pairs if p["tv"] >= t]
        below = [p for p in all_pairs if p["tv"] < t]
        tp = sum(1 for p in above if p["flip"])
        fp = sum(1 for p in above if not p["flip"])
        fn = sum(1 for p in below if p["flip"])
        tn = sum(1 for p in below if not p["flip"])
        p_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p_val * r / (p_val + r) if (p_val + r) > 0 else 0
        threshold_results[str(t)] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(p_val, 3), "recall": round(r, 3),
            "f1": round(f1, 3),
        }
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    # Using tau=0.3 from 1.7B
    transferred = threshold_results.get("0.3", {})

    total_correct = sum(1 for o in observations if o["is_correct"])
    total_flips = sum(1 for p in all_pairs if p["flip"])

    return {
        "model_id": model_id,
        "n_readouts": len(observations),
        "overall_accuracy": round(total_correct / len(observations), 3),
        "total_flips": total_flips,
        "total_pairs": len(all_pairs),
        "flip_rate": round(total_flips / len(all_pairs), 3),
        "tv_flip_correlation": round(corr, 3),
        "best_threshold": best_t,
        "best_f1": round(best_f1, 3),
        "transferred_threshold_0.3": transferred,
        "threshold_results": threshold_results,
    }


def run_crossmodel():
    print("=== Cross-Model Consequence Test ===")
    results = {}

    for model_id in MODELS:
        obs = run_for_model(model_id)
        model_key = model_id.split("/")[-1]
        analysis = analyze_model(obs, model_id)
        results[model_key] = analysis

        result_dir = Path(f"experiments/results/omega_crossmodel_consequence")
        result_dir.mkdir(parents=True, exist_ok=True)
        with open(result_dir / f"{model_key}_observations.json", "w") as f:
            json.dump(obs, f)

    # Comparison
    print("\n=== CROSS-MODEL COMPARISON ===")
    print(f"{'Model':20s} {'Acc':>5s} {'Corr':>6s} {'BestT':>6s} {'BestF1':>6s} "
          f"{'T=0.3 P':>7s} {'T=0.3 R':>7s} {'T=0.3 F1':>8s}")
    for model_key, r in results.items():
        t03 = r["transferred_threshold_0.3"]
        print(f"{model_key:20s} {r['overall_accuracy']:5.1%} {r['tv_flip_correlation']:6.3f} "
              f"{r['best_threshold']:6.2f} {r['best_f1']:6.3f} "
              f"{t03.get('precision', 0):7.3f} {t03.get('recall', 0):7.3f} "
              f"{t03.get('f1', 0):8.3f}")

    results["comparison"] = {
        "reference_model": "Qwen3-1.7B",
        "reference_threshold": 0.3,
        "reference_f1": 0.75,
        "reference_correlation": 0.677,
    }

    with open(result_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    run_crossmodel()
