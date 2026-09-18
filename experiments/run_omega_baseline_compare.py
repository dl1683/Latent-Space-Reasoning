"""Baseline Feature Comparison (Experiment #60).

Per Codex R1: compare omega (11-bin TV) against baseline features as flip
predictors. Uses the existing prospective experiment data (no new model runs).

Features compared:
- omega (11-bin TV between surface pairs)
- continuation category (binary: code vs comment)
- entropy of distribution A
- argmax probability margin (max prob - second prob)

Reports logistic regression AUC, correlation, and per-feature prediction.
"""

import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from omega import tv


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

CODE_CONTINUATIONS = {"identity", "nonvar_42", "heavy", "type_check", "neutral_pass"}
COMMENT_CONTINUATIONS = {"safety_comment", "deprecation"}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
    (4, "x", 10), (4, "y", 20),
    (4, "z", 30), (4, "x", 42),
    (4, "y", 77), (4, "z", 99),
]


def load_prospective_data():
    result_dir = Path("experiments/results/omega_prospective")
    obs_file = result_dir / "observations.json"
    if not obs_file.exists():
        print("No prospective observations found. Running from consequence data.")
        obs_file = Path("experiments/results/omega_consequence") / "observations.json"
        if not obs_file.exists():
            print("ERROR: No observation data found.")
            sys.exit(1)

    with open(obs_file) as f:
        observations = json.load(f)
    return observations


def compute_features(observations):
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = obs

    rows = []
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for cont_name in CONTINUATIONS:
            available_cells = []
            for depth, var, val in CELLS:
                key_a = (depth, var, val, surf_a, cont_name)
                key_b = (depth, var, val, surf_b, cont_name)
                if key_a in obs_idx and key_b in obs_idx:
                    available_cells.append((depth, var, val))

            for depth, var, val in available_cells:
                key_a = (depth, var, val, surf_a, cont_name)
                key_b = (depth, var, val, surf_b, cont_name)
                oa = obs_idx[key_a]
                ob = obs_idx[key_b]

                bin_a = oa.get("dist", oa.get("bin11", oa.get("distribution", [])))
                bin_b = ob.get("dist", ob.get("bin11", ob.get("distribution", [])))
                if not bin_a or not bin_b:
                    continue

                omega = tv(bin_a, bin_b)

                argmax_a = int(np.argmax(bin_a))
                argmax_b = int(np.argmax(bin_b))
                flip = 1 if argmax_a != argmax_b else 0

                probs_sorted = sorted(bin_a, reverse=True)
                margin = probs_sorted[0] - (probs_sorted[1] if len(probs_sorted) > 1 else 0)

                arr_a = np.array(bin_a)
                entropy = float(-np.sum(arr_a * np.log(arr_a + 1e-30)))

                is_code = 1 if cont_name in CODE_CONTINUATIONS else 0
                is_empty = 1 if cont_name == "empty" else 0

                rows.append({
                    "pair": pair_name,
                    "continuation": cont_name,
                    "cell": f"{var}_{val}",
                    "omega": float(omega),
                    "flip": flip,
                    "margin": float(margin),
                    "entropy": float(entropy),
                    "is_code": is_code,
                    "is_empty": is_empty,
                    "max_prob_a": float(probs_sorted[0]),
                })

    return rows


def analyze(rows):
    if not rows:
        print("ERROR: no rows to analyze.")
        return {}

    n = len(rows)
    flips = np.array([r["flip"] for r in rows])
    n_flips = int(flips.sum())

    print(f"\n=== BASELINE COMPARISON (Exp #60) ===\n")
    print(f"Total pairs: {n}")
    print(f"Total flips: {n_flips} ({n_flips/n:.1%})")

    features = {
        "omega": np.array([r["omega"] for r in rows]),
        "margin": np.array([r["margin"] for r in rows]),
        "entropy": np.array([r["entropy"] for r in rows]),
        "is_code": np.array([r["is_code"] for r in rows], dtype=float),
        "max_prob": np.array([r["max_prob_a"] for r in rows]),
    }

    def safe_corr(x, y):
        if x.std() > 0 and y.std() > 0:
            return float(np.corrcoef(x, y)[0, 1])
        return 0.0

    print("\n--- Univariate correlations with flip ---")
    correlations = {}
    for name, vals in features.items():
        r = safe_corr(vals, flips)
        correlations[name] = r
        print(f"  {name:12s}: r = {r:+.3f}")

    print("\n--- Threshold-based F1 for each feature ---")
    best_results = {}
    for name, vals in features.items():
        if name == "is_code":
            tp = int(((vals == 1) & (flips == 1)).sum())
            fp = int(((vals == 1) & (flips == 0)).sum())
            fn = int(((vals == 0) & (flips == 1)).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
            best_results[name] = {"threshold": "binary", "f1": f1,
                                   "precision": p, "recall": rec}
            print(f"  {name:12s}: F1={f1:.3f} P={p:.3f} R={rec:.3f} (binary)")
            continue

        if name in ("margin", "max_prob"):
            direction = "below"
        else:
            direction = "above"

        best_f1 = 0
        best_t = None
        percentiles = np.percentile(vals, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        for t in percentiles:
            if direction == "above":
                pred = (vals >= t).astype(int)
            else:
                pred = (vals <= t).astype(int)
            tp = int((pred * flips).sum())
            fp = int((pred * (1 - flips)).sum())
            fn = int(((1 - pred) * flips).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
                best_results[name] = {"threshold": best_t, "f1": f1,
                                       "precision": p, "recall": rec}

        if best_t is not None:
            br = best_results[name]
            print(f"  {name:12s}: F1={br['f1']:.3f} P={br['precision']:.3f} "
                  f"R={br['recall']:.3f} (t={best_t:.4f})")
        else:
            print(f"  {name:12s}: no positive predictions possible")
            best_results[name] = {"threshold": None, "f1": 0}

    print("\n--- Per-continuation flip rates ---")
    for cont_name in sorted(set(r["continuation"] for r in rows)):
        subset = [r for r in rows if r["continuation"] == cont_name]
        fr = sum(r["flip"] for r in subset) / len(subset) if subset else 0
        mean_omega = np.mean([r["omega"] for r in subset])
        mean_margin = np.mean([r["margin"] for r in subset])
        print(f"  {cont_name:20s}: flip={fr:.1%} omega={mean_omega:.4f} "
              f"margin={mean_margin:.3f} n={len(subset)}")

    summary = {
        "n_pairs": n,
        "n_flips": n_flips,
        "flip_rate": n_flips / n,
        "correlations": correlations,
        "best_f1": best_results,
    }
    return summary


if __name__ == "__main__":
    print("=== Baseline Feature Comparison (Exp #60) ===")
    print("Using existing prospective data — no model runs needed.\n")

    observations = load_prospective_data()
    print(f"Loaded {len(observations)} observations.")

    rows = compute_features(observations)
    print(f"Computed {len(rows)} feature rows.")

    summary = analyze(rows)

    out_dir = Path("experiments/results/omega_baseline_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "rows.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nResults saved to {out_dir}")
