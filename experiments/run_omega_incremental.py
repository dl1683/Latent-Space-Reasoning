"""Omega Incremental Value Test (Experiment #62).

Tests whether omega adds ANY predictive value beyond simpler features:
- Model: logistic regression of flip ~ margin + continuation_type + omega
- Compare: margin-only, continuation-only, margin+continuation, full model
- If omega adds nothing: the measure is epiphenomenal for flip prediction

Uses existing prospective data. Zero compute.
"""

import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from omega import tv


def load_data():
    p = Path("experiments/results/omega_prospective/observations.json")
    with open(p) as f:
        return json.load(f)


EQUIV_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CONTINUATIONS_LIST = ["empty", "identity", "nonvar_42", "heavy",
                      "neutral_pass", "type_check", "safety_comment", "deprecation"]

CODE_CONTS = {"identity", "nonvar_42", "heavy", "type_check", "neutral_pass"}

CELLS = [
    (4, "x", 1), (4, "x", 5), (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7), (4, "x", 10), (4, "y", 20),
    (4, "z", 30), (4, "x", 42), (4, "y", 77), (4, "z", 99),
]


def build_rows(observations):
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = obs

    rows = []
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for cont_name in CONTINUATIONS_LIST:
            for depth, var, val in CELLS:
                key_a = (depth, var, val, surf_a, cont_name)
                key_b = (depth, var, val, surf_b, cont_name)
                if key_a not in obs_idx or key_b not in obs_idx:
                    continue

                oa = obs_idx[key_a]
                ob = obs_idx[key_b]
                da = oa.get("dist", [])
                db = ob.get("dist", [])
                if not da or not db:
                    continue

                omega = float(tv(da, db))
                argmax_a = int(np.argmax(da))
                argmax_b = int(np.argmax(db))
                flip = 1 if argmax_a != argmax_b else 0

                sorted_a = sorted(da, reverse=True)
                margin_a = sorted_a[0] - sorted_a[1] if len(sorted_a) > 1 else 1.0
                sorted_b = sorted(db, reverse=True)
                margin_b = sorted_b[0] - sorted_b[1] if len(sorted_b) > 1 else 1.0
                min_margin = min(margin_a, margin_b)

                arr_a = np.array(da)
                entropy = float(-np.sum(arr_a * np.log(arr_a + 1e-30)))

                is_code = 1 if cont_name in CODE_CONTS else 0
                is_identity = 1 if cont_name == "identity" else 0
                is_heavy = 1 if cont_name == "heavy" else 0
                is_type_check = 1 if cont_name == "type_check" else 0

                rows.append({
                    "omega": omega,
                    "flip": flip,
                    "min_margin": min_margin,
                    "entropy": entropy,
                    "is_code": is_code,
                    "is_identity": is_identity,
                    "is_heavy": is_heavy,
                    "is_type_check": is_type_check,
                    "pair": pair_name,
                    "continuation": cont_name,
                })

    return rows


def logistic_predict(X, w, b):
    z = X @ w + b
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def fit_logistic(X, y, lr=0.1, n_iter=2000, reg=0.01):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        p = logistic_predict(X, w, b)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        grad_w = (1/n) * (X.T @ (p - y)) + reg * w
        grad_b = (1/n) * np.sum(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def evaluate_model(X, y, w, b):
    p = logistic_predict(X, w, b)
    pred = (p >= 0.5).astype(int)
    tp = int((pred * y).sum())
    fp = int((pred * (1 - y)).sum())
    fn = int(((1 - pred) * y).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) > 0 and len(neg) > 0:
        auc_val = 0
        for pv in pos:
            auc_val += np.sum(pv > neg) + 0.5 * np.sum(pv == neg)
        auc_val /= (len(pos) * len(neg))
    else:
        auc_val = 0.5

    return {"f1": f1, "precision": precision, "recall": recall,
            "auc": float(auc_val), "n_pred_pos": int(pred.sum())}


def run_incremental(rows):
    y = np.array([r["flip"] for r in rows])
    n = len(y)
    n_flips = int(y.sum())

    print(f"Rows: {n}, Flips: {n_flips} ({n_flips/n:.1%})\n")

    omega_arr = np.array([r["omega"] for r in rows]).reshape(-1, 1)
    margin_arr = np.array([r["min_margin"] for r in rows]).reshape(-1, 1)
    entropy_arr = np.array([r["entropy"] for r in rows]).reshape(-1, 1)
    is_heavy = np.array([r["is_heavy"] for r in rows]).reshape(-1, 1)
    is_identity = np.array([r["is_identity"] for r in rows]).reshape(-1, 1)
    is_type_check = np.array([r["is_type_check"] for r in rows]).reshape(-1, 1)

    cont_features = np.hstack([is_heavy, is_identity, is_type_check])

    def normalize(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1
        return (X - mu) / sd

    models = {
        "omega_only": normalize(omega_arr),
        "margin_only": normalize(margin_arr),
        "entropy_only": normalize(entropy_arr),
        "continuation_only": cont_features.astype(float),
        "margin+continuation": np.hstack([normalize(margin_arr), cont_features]),
        "omega+margin": np.hstack([normalize(omega_arr), normalize(margin_arr)]),
        "omega+continuation": np.hstack([normalize(omega_arr), cont_features]),
        "omega+margin+continuation": np.hstack([normalize(omega_arr), normalize(margin_arr), cont_features]),
        "full": np.hstack([normalize(omega_arr), normalize(margin_arr),
                           normalize(entropy_arr), cont_features]),
    }

    results = {}
    print(f"{'Model':40s} {'AUC':>6s} {'F1':>6s} {'P':>6s} {'R':>6s}")
    print("-" * 70)

    for name, X in models.items():
        w, b = fit_logistic(X, y)
        ev = evaluate_model(X, y, w, b)
        results[name] = ev
        print(f"{name:40s} {ev['auc']:.3f}  {ev['f1']:.3f}  "
              f"{ev['precision']:.3f}  {ev['recall']:.3f}")

    auc_margin_cont = results["margin+continuation"]["auc"]
    auc_full = results["omega+margin+continuation"]["auc"]
    delta = auc_full - auc_margin_cont

    print(f"\n=== KEY RESULT ===")
    print(f"Margin+continuation AUC: {auc_margin_cont:.3f}")
    print(f"+ omega AUC:             {auc_full:.3f}")
    print(f"Incremental AUC:         {delta:+.3f}")

    if abs(delta) < 0.02:
        print("VERDICT: Omega adds NO incremental value for flip prediction")
    elif delta > 0.02:
        print(f"VERDICT: Omega adds modest value (+{delta:.3f} AUC)")
    else:
        print(f"VERDICT: Omega HURTS prediction ({delta:+.3f} AUC)")

    return results


if __name__ == "__main__":
    print("=== Omega Incremental Value Test (Exp #62) ===\n")

    observations = load_data()
    rows = build_rows(observations)

    results = run_incremental(rows)

    out_dir = Path("experiments/results/omega_incremental")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir}")
