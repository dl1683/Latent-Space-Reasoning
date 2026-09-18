"""Omega Composition Order Test (Experiment #62).

Tests whether omega is order-dependent under composed continuations.

Key question: does the ORDER of continuations matter, or does only the
last one dominate (as digit-copying hypothesis predicts)?

Compositions tested:
- identity;heavy vs heavy;identity
- nonvar_42;identity vs identity;nonvar_42
- heavy;nonvar_42 vs nonvar_42;heavy

Controls: each atomic continuation alone.

Predictions under rival (digit-copying, last-token-wins):
  identity;heavy ≈ heavy (last = heavy, digit copying of leading digit)
  heavy;identity ≈ identity (last = identity)
  nonvar_42;heavy ≈ heavy (last = heavy)

Predictions under compositional omega:
  identity;heavy ≠ heavy;identity (order genuinely changes omega)
  TV of composition ≠ TV of last element alone

~360 readouts, fast.
"""

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
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "nonvar_42": "_ = 42\n",
    "id_then_heavy": "{var} = {var}\n{var} = {var} + 1  # WARNING: intentional mutation\n",
    "heavy_then_id": "{var} = {var} + 1  # WARNING: intentional mutation\n{var} = {var}\n",
    "nv42_then_id": "_ = 42\n{var} = {var}\n",
    "id_then_nv42": "{var} = {var}\n_ = 42\n",
    "heavy_then_nv42": "{var} = {var} + 1  # WARNING: intentional mutation\n_ = 42\n",
    "nv42_then_heavy": "_ = 42\n{var} = {var} + 1  # WARNING: intentional mutation\n",
}

QUERY = "f()\nprint({var})  # Output: "

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
TAU = 0.05


def run():
    print("=== Composition Order Test (Exp #62) ===")
    adapter = ModelAdapter({"model_id": MODEL_CFG["model_id"]})
    template = MODEL_CFG["templates"]["depth4_single"]

    observations = []
    t0 = time.time()
    total = len(CELLS) * len(SURFACES) * len(CONTINUATIONS)
    done = 0

    for depth, var, val in CELLS:
        prefix_code = template.format(var=var, outer_val=val)
        query = QUERY.format(var=var)
        prefix_state = adapter.get_state_after_prefix(prefix_code)

        for surf_name, surf_text in SURFACES.items():
            surface = surf_text.format(var=var)

            for cont_name, cont_text in CONTINUATIONS.items():
                continuation = cont_text.format(var=var)
                suffix = surface + continuation + query
                dist = adapter.get_dist_from_state(prefix_state, suffix)
                done += 1

                argmax_idx = int(np.argmax(dist))

                observations.append({
                    "depth": depth, "var": var, "val": val,
                    "surface": surf_name, "continuation": cont_name,
                    "dist": dist.tolist(),
                    "argmax_label": DIGIT_LABELS[argmax_idx],
                    "argmax_prob": float(dist[argmax_idx]),
                    "correct_digit": str(val),
                })

                if done % 50 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    print(f"  {done}/{total} ({rate:.1f}/s, "
                          f"ETA {(total-done)/rate:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {len(observations)} readouts in {elapsed:.1f}s")

    out_dir = Path("experiments/results/omega_comp_order")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f, indent=2)

    summary = analyze(observations)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")
    return summary


def analyze(observations):
    idx = {}
    for r in observations:
        key = (r['var'], r['val'], r['surface'], r['continuation'])
        idx[key] = r

    pair_omega = {}
    for s1, s2, pair_name in EQUIV_PAIRS:
        cont_omega = {}
        for cont_name in CONTINUATIONS:
            tvs = []
            flips = 0
            for depth, var, val in CELLS:
                base_k1 = (var, val, s1, "empty")
                base_k2 = (var, val, s2, "empty")
                if base_k1 not in idx or base_k2 not in idx:
                    continue
                baseline = tv(idx[base_k1]['dist'], idx[base_k2]['dist'])
                if baseline > TAU:
                    continue

                k1 = (var, val, s1, cont_name)
                k2 = (var, val, s2, cont_name)
                if k1 not in idx or k2 not in idx:
                    continue
                cell_tv = float(tv(idx[k1]['dist'], idx[k2]['dist']))
                tvs.append(cell_tv)
                if idx[k1]['argmax_label'] != idx[k2]['argmax_label']:
                    flips += 1

            if tvs:
                cont_omega[cont_name] = {
                    "mean_tv": round(float(np.mean(tvs)), 6),
                    "max_tv": round(float(np.max(tvs)), 6),
                    "n_eligible": len(tvs),
                    "n_flips": flips,
                }
        pair_omega[pair_name] = cont_omega

    order_tests = [
        ("id_then_heavy", "heavy_then_id", "identity+heavy order"),
        ("nv42_then_id", "id_then_nv42", "nonvar42+identity order"),
        ("heavy_then_nv42", "nv42_then_heavy", "heavy+nonvar42 order"),
    ]

    last_token_tests = [
        ("id_then_heavy", "heavy", "id;heavy vs atomic heavy"),
        ("heavy_then_id", "identity", "heavy;id vs atomic identity"),
        ("nv42_then_heavy", "heavy", "nv42;heavy vs atomic heavy"),
        ("id_then_nv42", "nonvar_42", "id;nv42 vs atomic nonvar42"),
        ("nv42_then_id", "identity", "nv42;id vs atomic identity"),
        ("heavy_then_nv42", "nonvar_42", "heavy;nv42 vs atomic nonvar42"),
    ]

    order_results = []
    for ab, ba, label in order_tests:
        ab_tvs = []
        ba_tvs = []
        for pn, po in pair_omega.items():
            if ab in po and ba in po:
                ab_tvs.append(po[ab]["mean_tv"])
                ba_tvs.append(po[ba]["mean_tv"])
        if ab_tvs:
            m_ab = float(np.mean(ab_tvs))
            m_ba = float(np.mean(ba_tvs))
            order_results.append({
                "pair": label, "ab": ab, "ba": ba,
                "mean_omega_ab": round(m_ab, 6),
                "mean_omega_ba": round(m_ba, 6),
                "ratio": round(m_ab / m_ba, 3) if m_ba > 0.0001 else None,
                "order_matters": abs(m_ab - m_ba) > 0.01,
            })

    lt_results = []
    for composed, atomic, label in last_token_tests:
        comp_tvs = []
        atom_tvs = []
        for pn, po in pair_omega.items():
            if composed in po and atomic in po:
                comp_tvs.append(po[composed]["mean_tv"])
                atom_tvs.append(po[atomic]["mean_tv"])
        if comp_tvs:
            m_c = float(np.mean(comp_tvs))
            m_a = float(np.mean(atom_tvs))
            lt_results.append({
                "label": label, "composed": composed, "atomic": atomic,
                "mean_omega_composed": round(m_c, 6),
                "mean_omega_atomic": round(m_a, 6),
                "ratio": round(m_c / m_a, 3) if m_a > 0.001 else None,
                "last_token_wins": abs(m_c - m_a) / max(m_a, 0.001) < 0.3,
            })

    n_order = sum(1 for t in order_results if t["order_matters"])
    n_lt = sum(1 for t in lt_results if t["last_token_wins"])

    if n_order >= 2:
        verdict = "COMPOSITIONAL: order matters"
    elif n_lt >= 4:
        verdict = "LAST_TOKEN_WINS: composed ≈ last atomic element"
    else:
        verdict = "MIXED"

    print(f"\n=== ORDER TESTS ===")
    for t in order_results:
        flag = "ORDER MATTERS" if t["order_matters"] else "~same"
        print(f"  {t['pair']}: A;B={t['mean_omega_ab']:.4f} B;A={t['mean_omega_ba']:.4f} "
              f"ratio={t['ratio']} {flag}")

    print(f"\n=== LAST-TOKEN TESTS ===")
    for t in lt_results:
        flag = "LAST_TOKEN" if t["last_token_wins"] else "NOT last-token"
        print(f"  {t['label']}: comp={t['mean_omega_composed']:.4f} "
              f"atom={t['mean_omega_atomic']:.4f} ratio={t['ratio']} {flag}")

    print(f"\nVERDICT: {verdict}")

    return {
        "n_readouts": len(observations),
        "pair_omega": pair_omega,
        "order_tests": order_results,
        "last_token_tests": lt_results,
        "verdict": verdict,
    }


if __name__ == "__main__":
    run()
