"""Experiment #68: Non-Commutativity Structure.

Exp #65 found neutral_then_identity amplifies 2.1x above identity alone.
This experiment maps the FULL composition algebra: for every ordered pair
(A, B) of continuations, compute TV(surface_a + A + B + Q, surface_b + A + B + Q).

If composition were commutative, TV(A then B) = TV(B then A).
If it were additive, TV(A then B) = TV(A) + TV(B).
If it were multiplicative, TV(A then B) = TV(A) * TV(B) / TV(empty).

We measure all three to find the actual algebraic structure.

Design: 6 continuations, all 36 ordered pairs (including self-composition).
2 surface pairs, 3 cells = 216 readouts. ~10 minutes.
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
}

EQUIV_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
]

CONTINUATIONS = {
    "empty": "",
    "identity": "{var} = {var}\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "nonvar_42": "_ = 42\n",
    "safety_comment": "# Do not modify {var}.\n",
    "deprecation": "# Deprecated: {var} is no longer used.\n",
}

CELLS = [(4, "x", 1), (4, "y", 2), (4, "y", 6)]
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
TAU = 0.05


def run():
    print("=== Exp #68: Non-Commutativity Structure ===")
    cont_names = list(CONTINUATIONS.keys())
    n_combos = len(cont_names) ** 2
    n_readouts = n_combos * len(EQUIV_PAIRS) * len(CELLS) * 2
    print(f"Composition pairs: {n_combos} ({len(cont_names)}x{len(cont_names)})")
    print(f"Total readouts: {n_readouts}")

    adapter = ModelAdapter(MODEL_CFG)
    import torch
    adapter.mdl.half()
    gc.collect()
    print("Model loaded (float16)")

    query_tmpl = MODEL_CFG["query_template"]
    t0 = time.time()

    observations = []
    n_done = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_name, surf_tmpl in SURFACES.items():
            surface_text = surf_tmpl.replace("{var}", var)

            for cont_a_name in cont_names:
                cont_a_text = CONTINUATIONS[cont_a_name].replace("{var}", var)
                for cont_b_name in cont_names:
                    cont_b_text = CONTINUATIONS[cont_b_name].replace("{var}", var)

                    full_suffix = surface_text + cont_a_text + cont_b_text + query
                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    n_done += 1

                    argmax_idx = int(np.argmax(dist))
                    observations.append({
                        "depth": depth, "var": var, "val": val,
                        "surface": surf_name,
                        "cont_a": cont_a_name,
                        "cont_b": cont_b_name,
                        "dist": dist.tolist(),
                        "argmax_label": DIGIT_LABELS[argmax_idx],
                        "argmax_prob": float(dist[argmax_idx]),
                        "margin": float(np.sort(dist)[-1] - np.sort(dist)[-2]),
                    })

                    if n_done % 72 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n_done}/{n_readouts} readouts, {elapsed:.0f}s",
                              flush=True)

        del prefix_state
        gc.collect()

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} readouts in {elapsed:.1f}s")

    idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["surface"], obs["cont_a"], obs["cont_b"])
        idx[key] = obs

    # Compute TV for each (pair, cell, cont_a, cont_b)
    results = []
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for depth, var, val in CELLS:
            for ca in cont_names:
                for cb in cont_names:
                    ka = (var, val, surf_a, ca, cb)
                    kb = (var, val, surf_b, ca, cb)
                    if ka in idx and kb in idx:
                        cell_tv = float(tv(idx[ka]["dist"], idx[kb]["dist"]))
                        flip = idx[ka]["argmax_label"] != idx[kb]["argmax_label"]
                        results.append({
                            "pair": pair_name, "cell": f"{var}_{val}",
                            "cont_a": ca, "cont_b": cb,
                            "tv": cell_tv, "flip": flip,
                        })

    # Build composition matrix: mean TV over cells for each (pair, cont_a, cont_b)
    from collections import defaultdict
    for pair_name in ["strong_assert_2", "boundary_pair"]:
        print(f"\n=== COMPOSITION MATRIX: {pair_name} ===")
        matrix = {}
        for ca in cont_names:
            for cb in cont_names:
                tvs = [r["tv"] for r in results
                       if r["pair"] == pair_name and r["cont_a"] == ca and r["cont_b"] == cb]
                if tvs:
                    matrix[(ca, cb)] = np.mean(tvs)

        # Print as table
        short = {c: c[:6] for c in cont_names}
        header = f"{'A\\B':>10s}" + "".join(f"{short[c]:>8s}" for c in cont_names)
        print(header)
        print("-" * len(header))
        for ca in cont_names:
            row = f"{short[ca]:>10s}"
            for cb in cont_names:
                v = matrix.get((ca, cb), float('nan'))
                row += f"{v:8.4f}"
            print(row)

        # Check commutativity: TV(A,B) vs TV(B,A)
        asym_pairs = []
        for i, ca in enumerate(cont_names):
            for j, cb in enumerate(cont_names):
                if i < j and (ca, cb) in matrix and (cb, ca) in matrix:
                    ab = matrix[(ca, cb)]
                    ba = matrix[(cb, ca)]
                    ratio = ab / ba if ba > 0.001 else float('inf')
                    asym_pairs.append((ca, cb, ab, ba, ratio))

        print(f"\nCommutativity check (TV(A→B) vs TV(B→A)):")
        for ca, cb, ab, ba, ratio in sorted(asym_pairs, key=lambda x: -abs(x[4] - 1)):
            print(f"  {short[ca]}→{short[cb]}: {ab:.4f} vs {short[cb]}→{short[ca]}: {ba:.4f} "
                  f"(ratio={ratio:.2f}x)")

        # Check additivity: TV(A,B) vs TV(A,empty) + TV(empty,B)
        print(f"\nAdditivity check:")
        for ca in cont_names:
            for cb in cont_names:
                if ca == "empty" or cb == "empty":
                    continue
                ab = matrix.get((ca, cb), 0)
                a0 = matrix.get((ca, "empty"), 0)
                ob = matrix.get(("empty", cb), 0)
                if a0 > 0.001 and ob > 0.001:
                    add_pred = a0 + ob
                    ratio = ab / add_pred if add_pred > 0.001 else float('inf')
                    if abs(ratio - 1) > 0.3:
                        print(f"  {short[ca]}+{short[cb]}: actual={ab:.4f}, "
                              f"additive={add_pred:.4f} (ratio={ratio:.2f}x)")

    summary = {
        "experiment": 68,
        "name": "noncommutativity_structure",
        "n_readouts": n_done,
        "n_compositions": len(results),
        "observations": observations,
        "composition_results": results,
    }

    result_dir = Path("experiments/results/omega_noncommutativity")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
