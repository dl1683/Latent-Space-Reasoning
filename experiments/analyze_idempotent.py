"""Idempotent test: compare suffix_count=1 (Gate 1b) vs suffix_count=2 distributions.

Tests u² ≡_M u for each null-witness class. If TV(response(u), response(u²))
< eps_eq, that class is idempotent. Also tests the multiplicative prediction:
if K_a model holds, L(u²) = a_u² × L(baseline).

Can also run in the legacy mode comparing pass vs pass^2 vs pass^3 if the
idempotent data contains PASS_x2 / PASS_x3 roles.
"""
import json
import sys
from collections import defaultdict

import numpy as np


def tv_distance(p, q):
    return 0.5 * np.sum(np.abs(np.array(p) - np.array(q)))


def analyze_idempotent(gate1b_path, idempotent_path, eps_eq=0.01):
    with open(gate1b_path) as f:
        g1b = json.load(f)
    with open(idempotent_path) as f:
        idem = json.load(f)

    g1b_obs = g1b["observations"]
    idem_obs = idem["observations"]

    idem_roles = set(obs["role"] for obs in idem_obs.values())
    if "PASS_x2" in idem_roles:
        return _analyze_legacy(g1b, idem)

    test_roles = sorted(set(obs["role"] for obs in idem_obs.values() if obs["role"] != "BASELINE"))

    g1b_by_ctx = defaultdict(lambda: defaultdict(list))
    idem_by_ctx = defaultdict(lambda: defaultdict(list))
    baseline_by_ctx = {}

    for key, obs in g1b_obs.items():
        if obs["val"] == 9:
            continue
        ctx = (obs["depth"], obs["var"], obs["val"])
        if obs["role"] == "BASELINE":
            baseline_by_ctx[ctx] = np.array(obs["dist"])
        elif obs["role"] in test_roles:
            g1b_by_ctx[obs["role"]][ctx].append(np.array(obs["dist"]))

    for key, obs in idem_obs.items():
        if obs["val"] == 9 or obs["role"] == "BASELINE":
            continue
        ctx = (obs["depth"], obs["var"], obs["val"])
        idem_by_ctx[obs["role"]][ctx].append(np.array(obs["dist"]))

    print("=== IDEMPOTENT TEST: u² ≡_M u? ===\n")

    for role in test_roles:
        shared = sorted(set(g1b_by_ctx[role].keys()) & set(idem_by_ctx[role].keys()))
        if not shared:
            print(f"  {role}: no shared contexts\n")
            continue

        tvs = []
        a_u_vals = []
        a_u2_vals = []
        pred_errors = []

        for ctx in shared:
            for d1 in g1b_by_ctx[role][ctx]:
                for d2 in idem_by_ctx[role][ctx]:
                    tvs.append(tv_distance(d1, d2))

            bl = baseline_by_ctx.get(ctx)
            if bl is not None and float(bl[9]) > 0.001:
                bl_L = float(bl[9])
                for d1 in g1b_by_ctx[role][ctx]:
                    a_u_vals.append(float(d1[9]) / bl_L)
                for d2 in idem_by_ctx[role][ctx]:
                    s2_L = float(d2[9])
                    a_u2_vals.append(s2_L / bl_L)
                    if a_u_vals:
                        pred = a_u_vals[-1] ** 2 * bl_L
                        pred_errors.append(abs(s2_L - pred))

        arr = np.array(tvs)
        is_idem = arr.max() < eps_eq

        print(f"  {role}:")
        print(f"    Contexts: {len(shared)}, TV pairs: {len(tvs)}")
        print(f"    TV(u, u²): mean={arr.mean():.4f}, med={np.median(arr):.4f}, "
              f"p95={np.percentile(arr, 95):.4f}, max={arr.max():.4f}")
        print(f"    Idempotent (max TV < {eps_eq}): {'YES' if is_idem else 'NO'}")

        if a_u_vals:
            a1 = np.array(a_u_vals)
            print(f"    a_u = L(u)/L(bl): mean={a1.mean():.4f} ± {a1.std():.4f}")
        if a_u2_vals:
            a2 = np.array(a_u2_vals)
            print(f"    a_u² = L(u²)/L(bl): mean={a2.mean():.4f} ± {a2.std():.4f}")
            print(f"    Predicted a_u² (from a_u²): {a1.mean()**2:.4f}")
        if pred_errors:
            pe = np.array(pred_errors)
            print(f"    Cellwise |L_actual - a²×L_bl|: mean={pe.mean():.4f}, max={pe.max():.4f}")
        print()

    print("=== SUMMARY ===\n")
    print("Idempotent = u² ≡_M u (full response-law level).")
    print("Multiplicative = L(u²) ≈ a_u² × L(bl) (K_a composition verified on L coordinate).")
    print("Both can coexist only if a_u ∈ {0, 1}.")


def _analyze_legacy(g1b, idem):
    """Legacy mode: pass vs pass^2 vs pass^3."""
    baseline_dists = {}
    pass_dists = {}
    pass2_dists = {}
    pass3_dists = {}

    for key, obs in g1b["observations"].items():
        if obs["depth"] != 4 or obs["val"] == 9:
            continue
        ctx = (obs["var"], obs["val"])
        if obs["role"] == "BASELINE":
            baseline_dists[ctx] = np.array(obs["dist"])
        elif obs["role"] == "BOUNDARY" and obs["surface"] == "pass\n":
            pass_dists[ctx] = np.array(obs["dist"])

    for key, obs in idem["observations"].items():
        if obs["val"] == 9:
            continue
        ctx = (obs["var"], obs["val"])
        if obs["role"] == "PASS_x2":
            pass2_dists[ctx] = np.array(obs["dist"])
        elif obs["role"] == "PASS_x3":
            pass3_dists[ctx] = np.array(obs["dist"])

    print("=== IDEMPOTENT TEST (LEGACY): pass\\n ===\n")

    pairs = [
        ("epsilon vs pass", baseline_dists, pass_dists),
        ("pass vs pass^2", pass_dists, pass2_dists),
        ("pass^2 vs pass^3", pass2_dists, pass3_dists),
    ]

    print(f"  {'Comparison':<25} {'n':>4} {'mean_TV':>9} {'p95_TV':>9}")
    for name, d1, d2 in pairs:
        tvs = [tv_distance(d1[c], d2[c]) for c in d1 if c in d2]
        if tvs:
            arr = np.array(tvs)
            print(f"  {name:<25} {len(tvs):>4} {arr.mean():9.4f} {np.percentile(arr, 95):9.4f}")


if __name__ == "__main__":
    g1b = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/results/svb_qwen3_gate1b/result.json"
    idem = sys.argv[2] if len(sys.argv) > 2 else \
        "experiments/results/svb_qwen3_idempotent/result.json"
    analyze_idempotent(g1b, idem)
