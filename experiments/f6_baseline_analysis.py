"""F6 Baseline Analysis: nested ladder comparison.

Tests whether the proposed intensional role quotient adds predictive
information beyond cheap context-tracking features.

M_state: pre-suffix (C,L,R) + depth + var + val
M_cheap: M_state + statement_type + token_count + char_count + has_var_mention
M_cheap_role: M_cheap + role

Requires result.json from an experiment with a BASELINE role (e.g., Gate 1b).
"""
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error


def extract_features(obs, baseline_lookup):
    """Extract feature vectors for the three nested models."""
    depth = obs["depth"]
    var = obs["var"]
    val = obs["val"]
    role = obs["role"]
    surface = obs["surface"]

    base_key = (depth, var, val)
    baseline_clr = baseline_lookup.get(base_key)
    if baseline_clr is None:
        return None

    depth_feat = [1 if depth == d else 0 for d in [2, 3, 4]]
    var_feat = [1 if var == v else 0 for v in ["x", "y", "z"]]
    val_feat = [val / 9.0]

    x_state = list(baseline_clr) + depth_feat + var_feat + val_feat

    stmt_map = {"comment": [1,0,0,0], "assignment": [0,1,0,0],
                "expression": [0,0,1,0], "delimiter": [0,0,0,1],
                "baseline": [0,0,0,0]}
    stmt = classify_statement(surface, role)
    stmt_feat = stmt_map.get(stmt, [0,0,0,0])
    tok_count = [len(surface.split()) / 10.0]
    char_count = [len(surface) / 50.0]
    has_var = [1.0 if any(v in surface for v in ["x", "y", "z"]) else 0.0]

    x_cheap = x_state + stmt_feat + tok_count + char_count + has_var

    role_map = {"ASSERT": [1,0,0,0,0,0,0,0],
                "ASSERT_VAR": [0,1,0,0,0,0,0,0],
                "MISLEADING_ASSERT": [0,0,1,0,0,0,0,0],
                "MISLEADING_ASSERT_NOVAR": [0,0,0,1,0,0,0,0],
                "REWRITE": [0,0,0,0,1,0,0,0],
                "OBSERVE": [0,0,0,0,0,1,0,0],
                "BOUNDARY": [0,0,0,0,0,0,1,0],
                "BASELINE": [0,0,0,0,0,0,0,1]}
    role_feat = role_map.get(role, [0]*8)

    x_role = x_cheap + role_feat

    return x_state, x_cheap, x_role


def classify_statement(surface, role):
    if surface.startswith("#"):
        return "comment"
    if "=" in surface and not surface.startswith("_") and not surface.startswith("type"):
        return "assignment"
    if surface.strip() in ("", "\n", "\n\n", "\n\n\n", "pass", "pass  ", "..."):
        return "delimiter"
    return "expression"


def run_f6(result_path):
    with open(result_path) as f:
        data = json.load(f)

    observations = data["observations"]

    baseline_lookup = {}
    for key, obs in observations.items():
        if obs["role"] == "BASELINE":
            dist = np.array(obs["dist"])
            val = obs["val"]
            C = float(dist[val])
            L = float(dist[9]) if val != 9 else 0.0
            R = 1.0 - C - L
            baseline_lookup[(obs["depth"], obs["var"], obs["val"])] = [C, L, R]

    if not baseline_lookup:
        print("ERROR: No BASELINE role found. F6 requires pre-suffix responses.")
        print("Run Gate 1b (svb_qwen3_gate1b.json) which includes BASELINE.")
        return

    print(f"Baseline lookup: {len(baseline_lookup)} entries\n")

    X_state, X_cheap, X_role = [], [], []
    Y = []
    groups = []

    for key, obs in observations.items():
        if obs["role"] == "BASELINE":
            continue
        if obs["val"] == 9:
            continue

        feats = extract_features(obs, baseline_lookup)
        if feats is None:
            continue

        dist = np.array(obs["dist"])
        val = obs["val"]
        C = float(dist[val])
        L = float(dist[9])
        R = 1.0 - C - L
        y = [C, L, R]

        x_s, x_c, x_r = feats
        X_state.append(x_s)
        X_cheap.append(x_c)
        X_role.append(x_r)
        Y.append(y)
        groups.append(obs["surface"])

    X_state = np.array(X_state)
    X_cheap = np.array(X_cheap)
    X_role = np.array(X_role)
    Y = np.array(Y)
    groups = np.array(groups)

    unique_surfaces = np.unique(groups)
    surface_to_group = {s: i for i, s in enumerate(unique_surfaces)}
    group_ids = np.array([surface_to_group[s] for s in groups])

    print(f"Samples: {len(Y)}, Surfaces: {len(unique_surfaces)}")
    print(f"Feature dims: state={X_state.shape[1]}, cheap={X_cheap.shape[1]}, role={X_role.shape[1]}\n")

    n_splits = min(5, len(unique_surfaces))
    gkf = GroupKFold(n_splits=n_splits)

    results = {}
    for name, X in [("M_state", X_state), ("M_cheap", X_cheap), ("M_cheap+role", X_role)]:
        fold_mses = []
        for train_idx, test_idx in gkf.split(X, Y, group_ids):
            model = Ridge(alpha=1.0)
            model.fit(X[train_idx], Y[train_idx])
            pred = model.predict(X[test_idx])
            tv = np.mean(np.sum(np.abs(pred - Y[test_idx]), axis=1) / 2)
            fold_mses.append(tv)

        mean_tv = np.mean(fold_mses)
        std_tv = np.std(fold_mses)
        results[name] = (mean_tv, std_tv)
        print(f"  {name:<15} TV = {mean_tv:.4f} +/- {std_tv:.4f}")

    print("\n=== F6 VERDICT ===\n")
    tv_cheap = results["M_cheap"][0]
    tv_role = results["M_cheap+role"][0]
    tv_state = results["M_state"][0]

    improvement = tv_cheap - tv_role
    print(f"  Role improvement over cheap: {improvement:+.4f} TV")

    if improvement > 0.01:
        print(f"  -> PASS: role adds {improvement:.4f} TV predictive information beyond cheap features")
    elif improvement > 0.005:
        print(f"  -> MARGINAL: small role improvement ({improvement:.4f})")
    else:
        print(f"  -> FAIL (F6): cheap context tracking explains the role effect")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/results/svb_qwen3_gate1b/result.json"
    run_f6(path)
