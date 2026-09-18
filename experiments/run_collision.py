"""Experiment #71: Literal-Value Collision Dynamics.

Exp #59 found that when `_ = k` literal matches the cell value, TV is HIGHER
(interference, not reinforcement). This is counterintuitive: matching should
anchor, not destabilize.

Hypothesis: literal-value COLLISION creates constructive interference in the
model's scope chain — two causal paths (cell assignment + literal) produce
the same value, creating representational ambiguity about which path is
authoritative. This amplifies surface sensitivity.

Alternative: matching literals simply activate the cell value's representation
more strongly, amplifying ALL downstream effects including surface sensitivity.

Design: 6 cell values x 3 literal conditions (match/mismatch/control) x
2 surface pairs x 2 surfaces = 144 readouts. Plus identity/heavy overlay
on match vs control = +48 readouts. Total: 192 readouts, ~4 min.
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

CELLS = [("x", 1), ("x", 2), ("x", 3), ("y", 5), ("y", 6), ("y", 7)]

SURFACES = {
    "assert_var": "# {var} stays the same.\n",
    "assert_novar": "# State unchanged.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
}

EQUIV_PAIRS = [
    ("assert_var", "assert_novar", "strong_assert"),
    ("boundary_1", "boundary_2", "boundary_pair"),
]

PREFIX_TEMPLATE = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def get_mismatch_val(val):
    candidates = [v for v in [1, 2, 3, 5, 6, 7] if v != val]
    return candidates[0]


def run():
    print("=== Exp #71: Literal-Value Collision Dynamics ===")

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

        match_literal = f"_ = {val}\n"
        mismatch_val = get_mismatch_val(val)
        mismatch_literal = f"_ = {mismatch_val}\n"

        literal_conditions = {
            "control": "",
            "match": match_literal,
            "mismatch": mismatch_literal,
        }

        continuation_overlays = {
            "none": "",
            "identity": f"{var} = {var}\n",
            "heavy": f"{var} = {var} + 1  # WARNING: intentional mutation\n",
        }

        for surf_name, surf_tmpl in SURFACES.items():
            surf_text = surf_tmpl.replace("{var}", var)

            for lit_name, lit_text in literal_conditions.items():
                for cont_name, cont_text in continuation_overlays.items():
                    if cont_name != "none" and lit_name == "mismatch":
                        continue
                    full_suffix = surf_text + lit_text + cont_text + query
                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    key = (var, val, surf_name, lit_name, cont_name)
                    all_dists[key] = dist
                    n_done += 1

        del prefix_state
        gc.collect()
        elapsed = time.time() - t0
        print(f"  Cell {var}_{val} done, {n_done} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} readouts in {elapsed:.1f}s")

    results = []
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for var, val in CELLS:
            for lit_name in ["control", "match", "mismatch"]:
                for cont_name in ["none", "identity", "heavy"]:
                    if cont_name != "none" and lit_name == "mismatch":
                        continue
                    ka = (var, val, surf_a, lit_name, cont_name)
                    kb = (var, val, surf_b, lit_name, cont_name)
                    if ka in all_dists and kb in all_dists:
                        cell_tv = float(tv(all_dists[ka], all_dists[kb]))
                        flip = (DIGIT_LABELS[int(np.argmax(all_dists[ka]))] !=
                                DIGIT_LABELS[int(np.argmax(all_dists[kb]))])
                        results.append({
                            "pair": pair_name,
                            "cell": f"{var}_{val}",
                            "var": var,
                            "val": val,
                            "literal": lit_name,
                            "continuation": cont_name,
                            "tv": cell_tv,
                            "flip": flip,
                        })

    # Analysis: collision effect
    print("\n=== COLLISION EFFECT (match vs control, no continuation) ===")
    print(f"{'Pair':>15s} {'match_TV':>10s} {'control_TV':>10s} {'mismatch_TV':>10s} {'match/ctrl':>10s} {'mis/ctrl':>10s}")
    print("-" * 60)
    for pair_name in ["strong_assert", "boundary_pair"]:
        match_tvs = [r["tv"] for r in results if r["pair"] == pair_name
                     and r["literal"] == "match" and r["continuation"] == "none"]
        ctrl_tvs = [r["tv"] for r in results if r["pair"] == pair_name
                    and r["literal"] == "control" and r["continuation"] == "none"]
        mis_tvs = [r["tv"] for r in results if r["pair"] == pair_name
                   and r["literal"] == "mismatch" and r["continuation"] == "none"]
        m = np.mean(match_tvs) if match_tvs else 0
        c = np.mean(ctrl_tvs) if ctrl_tvs else 0.001
        mi = np.mean(mis_tvs) if mis_tvs else 0
        print(f"{pair_name:>15s} {m:10.4f} {c:10.4f} {mi:10.4f} "
              f"{m/c:10.1f}x {mi/c:10.1f}x")

    # Per-cell collision breakdown
    print("\n=== PER-CELL COLLISION (strong_assert, no continuation) ===")
    print(f"{'Cell':>8s} {'control':>10s} {'match':>10s} {'mismatch':>10s} {'match/ctrl':>10s}")
    print("-" * 45)
    for var, val in CELLS:
        ctrl = [r["tv"] for r in results if r["cell"] == f"{var}_{val}"
                and r["pair"] == "strong_assert" and r["literal"] == "control"
                and r["continuation"] == "none"]
        match = [r["tv"] for r in results if r["cell"] == f"{var}_{val}"
                 and r["pair"] == "strong_assert" and r["literal"] == "match"
                 and r["continuation"] == "none"]
        mis = [r["tv"] for r in results if r["cell"] == f"{var}_{val}"
               and r["pair"] == "strong_assert" and r["literal"] == "mismatch"
               and r["continuation"] == "none"]
        c = ctrl[0] if ctrl else 0.001
        m = match[0] if match else 0
        mi = mis[0] if mis else 0
        ratio = m / c if c > 0.001 else float('inf')
        print(f"{var}_{val:>5d} {c:10.4f} {m:10.4f} {mi:10.4f} {ratio:10.1f}x")

    # Interaction: collision x identity/heavy
    print("\n=== COLLISION x CONTINUATION INTERACTION (strong_assert) ===")
    print(f"{'Condition':>20s} {'mean_TV':>10s} {'vs_control_none':>15s}")
    print("-" * 50)
    ctrl_none = np.mean([r["tv"] for r in results if r["pair"] == "strong_assert"
                         and r["literal"] == "control" and r["continuation"] == "none"])
    for lit in ["control", "match"]:
        for cont in ["none", "identity", "heavy"]:
            tvs = [r["tv"] for r in results if r["pair"] == "strong_assert"
                   and r["literal"] == lit and r["continuation"] == cont]
            if tvs:
                m = np.mean(tvs)
                ratio = m / ctrl_none if ctrl_none > 0.001 else float('inf')
                print(f"{lit}+{cont:>12s} {m:10.4f} {ratio:15.1f}x")

    summary = {
        "experiment": 71,
        "name": "collision_dynamics",
        "n_readouts": n_done,
        "n_comparisons": len(results),
        "results": results,
    }

    result_dir = Path("experiments/results/omega_collision")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
