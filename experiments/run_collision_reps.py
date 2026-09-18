"""Experiment #74: Collision Repetition Curve.

Exp #73 found that double_collision = 0.5x (self-cancelling).
Does collision amplification follow an inverted-U with repetition count?

Test: 0, 1, 2, 3, 4 repetitions of `_ = val`.
If inverted-U: peaks at 1, drops at 2+.
If monotonic decay: each repetition resolves more ambiguity.
If U-shaped: 2 cancels but 3+ re-amplifies.

Also test: repetitions of mismatch `_ = other_val` as control.

Design: 5 reps x 2 literal types x 3 cells x 2 pairs x 2 surfaces = 120 readouts. ~2 min.
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

CELLS = [("x", 1), ("y", 2), ("y", 6)]

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


def run():
    print("=== Exp #74: Collision Repetition Curve ===")

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

        mismatch_val = [v for v in [1, 2, 3, 5, 6, 7] if v != val][0]

        for reps in range(5):
            match_text = f"_ = {val}\n" * reps
            mismatch_text = f"_ = {mismatch_val}\n" * reps

            for surf_name, surf_tmpl in SURFACES.items():
                surf_text = surf_tmpl.replace("{var}", var)

                suffix_m = surf_text + match_text + query
                dist_m = adapter.get_dist_from_state(prefix_state, suffix_m)
                all_dists[(var, val, surf_name, "match", reps)] = dist_m
                n_done += 1

                suffix_x = surf_text + mismatch_text + query
                dist_x = adapter.get_dist_from_state(prefix_state, suffix_x)
                all_dists[(var, val, surf_name, "mismatch", reps)] = dist_x
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
            for lit_type in ["match", "mismatch"]:
                for reps in range(5):
                    ka = (var, val, surf_a, lit_type, reps)
                    kb = (var, val, surf_b, lit_type, reps)
                    if ka in all_dists and kb in all_dists:
                        cell_tv = float(tv(all_dists[ka], all_dists[kb]))
                        flip = (DIGIT_LABELS[int(np.argmax(all_dists[ka]))] !=
                                DIGIT_LABELS[int(np.argmax(all_dists[kb]))])
                        results.append({
                            "pair": pair_name,
                            "cell": f"{var}_{val}",
                            "literal_type": lit_type,
                            "reps": reps,
                            "tv": cell_tv,
                            "flip": flip,
                        })

    print("\n=== REPETITION CURVE (strong_assert) ===")
    print(f"{'Reps':>6s} {'match_TV':>10s} {'match_amp':>10s} {'mismatch_TV':>12s} {'mis_amp':>10s}")
    print("-" * 52)
    base_tvs = [r["tv"] for r in results
                if r["pair"] == "strong_assert" and r["literal_type"] == "match" and r["reps"] == 0]
    base_mean = np.mean(base_tvs) if base_tvs else 0.001

    for reps in range(5):
        match_tvs = [r["tv"] for r in results
                     if r["pair"] == "strong_assert" and r["literal_type"] == "match"
                     and r["reps"] == reps]
        mis_tvs = [r["tv"] for r in results
                   if r["pair"] == "strong_assert" and r["literal_type"] == "mismatch"
                   and r["reps"] == reps]
        m = np.mean(match_tvs) if match_tvs else 0
        mi = np.mean(mis_tvs) if mis_tvs else 0
        m_amp = m / base_mean if base_mean > 0.001 else 0
        mi_amp = mi / base_mean if base_mean > 0.001 else 0
        print(f"{reps:6d} {m:10.4f} {m_amp:10.1f}x {mi:12.4f} {mi_amp:10.1f}x")

    print("\n=== REPETITION CURVE (boundary_pair) ===")
    print(f"{'Reps':>6s} {'match_TV':>10s} {'mismatch_TV':>12s}")
    print("-" * 32)
    for reps in range(5):
        match_tvs = [r["tv"] for r in results
                     if r["pair"] == "boundary_pair" and r["literal_type"] == "match"
                     and r["reps"] == reps]
        mis_tvs = [r["tv"] for r in results
                   if r["pair"] == "boundary_pair" and r["literal_type"] == "mismatch"
                   and r["reps"] == reps]
        m = np.mean(match_tvs) if match_tvs else 0
        mi = np.mean(mis_tvs) if mis_tvs else 0
        print(f"{reps:6d} {m:10.4f} {mi:12.4f}")

    summary = {
        "experiment": 74,
        "name": "collision_repetition",
        "n_readouts": n_done,
        "n_comparisons": len(results),
        "results": results,
    }

    result_dir = Path("experiments/results/omega_collision_reps")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
