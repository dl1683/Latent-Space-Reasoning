"""Experiment #79: Regime Classifier (Fable R8 E4).

Test whether the three regimes (chain-resolution, recency, already-divergent)
are a continuum parameterized by empty baseline, or genuinely discrete.

Design: 8 surface pairs spanning a semantic gradient from strongly assertive
to semantically empty. Measure empty baseline + identity + heavy on each.
Plot identity_amp vs empty_baseline.

If identity_amp = f(1/baseline) with a clean monotonic relationship,
the regimes are one continuum. If there are discrete clusters or
non-monotonic jumps, the regimes are genuinely discrete.

Pairs (ordered by expected assertiveness):
1. strong_explicit: ("# x is definitely {val}\n", "# x equals {val} for sure\n")
2. strong_assert: ("# {var} stays the same.\n", "# State unchanged.\n") [existing]
3. assert_weak: ("# probably same\n", "# likely unchanged\n") [existing]
4. hedged: ("# maybe {val}?\n", "# could be {val}\n")
5. comment_generic: ("# note\n", "# info\n") [existing]
6. whitespace_long: ("    \n", "     \n")
7. boundary: ("\n", "pass\n") [existing]
8. code_noise: ("_ = 0\n", "_ = None\n")

Design: 8 pairs × 3 conditions (empty/identity/heavy) × 3 cells × 2 surfaces
= 144 readouts, ~50s.
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

SURFACE_PAIRS = {
    "strong_explicit": ("# {var} is definitely {val}\n", "# {var} equals {val} for sure\n"),
    "strong_assert": ("# {var} stays the same.\n", "# State unchanged.\n"),
    "medium_assert": ("# {var} unchanged\n", "# no change to {var}\n"),
    "assert_weak": ("# probably same\n", "# likely unchanged\n"),
    "hedged": ("# maybe {val}?\n", "# could be {val}\n"),
    "comment_generic": ("# note\n", "# info\n"),
    "boundary": ("\n", "pass\n"),
    "code_noise": ("_ = 0\n", "_ = None\n"),
}

PREFIX_TEMPLATE = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def run():
    print("=== Exp #79: Regime Classifier (Fable E4) ===")

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

        id_text = f"{var} = {var}\n"
        heavy_text = f"{var} = {var} + 1  # WARNING: intentional mutation\n"

        for pair_name, (surf_a_tmpl, surf_b_tmpl) in SURFACE_PAIRS.items():
            surf_a = surf_a_tmpl.replace("{var}", var).replace("{val}", str(val))
            surf_b = surf_b_tmpl.replace("{var}", var).replace("{val}", str(val))

            for cont_name, cont_text in [("empty", ""), ("identity", id_text), ("heavy", heavy_text)]:
                suffix_a = surf_a + cont_text + query
                suffix_b = surf_b + cont_text + query

                dist_a = adapter.get_dist_from_state(prefix_state, suffix_a)
                dist_b = adapter.get_dist_from_state(prefix_state, suffix_b)
                all_dists[(var, val, pair_name, "a", cont_name)] = dist_a
                all_dists[(var, val, pair_name, "b", cont_name)] = dist_b
                n_done += 2

        del prefix_state
        gc.collect()
        elapsed = time.time() - t0
        print(f"  Cell {var}_{val} done, {n_done} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} readouts in {elapsed:.1f}s")

    results = []
    for pair_name in SURFACE_PAIRS:
        for var, val in CELLS:
            for cont_name in ["empty", "identity", "heavy"]:
                da = all_dists.get((var, val, pair_name, "a", cont_name))
                db = all_dists.get((var, val, pair_name, "b", cont_name))
                if da is not None and db is not None:
                    cell_tv = float(tv(da, db))
                    flip = (DIGIT_LABELS[int(np.argmax(da))] !=
                            DIGIT_LABELS[int(np.argmax(db))])
                    results.append({
                        "pair": pair_name,
                        "cell": f"{var}_{val}",
                        "continuation": cont_name,
                        "tv": cell_tv,
                        "flip": flip,
                    })

    # Main table: baseline, identity, heavy, identity_amp, heavy_amp
    print("\n=== REGIME CLASSIFIER TABLE ===")
    print(f"{'Pair':>20s} {'baseline':>10s} {'identity':>10s} {'id_amp':>10s} {'heavy':>10s} {'hvy_amp':>10s} {'id_delta':>10s} {'hvy_delta':>10s}")
    print("-" * 95)

    pair_data = {}
    for pair_name in SURFACE_PAIRS:
        empty_tvs = [r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "empty"]
        id_tvs = [r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "identity"]
        heavy_tvs = [r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "heavy"]

        e = np.mean(empty_tvs) if empty_tvs else 0.001
        i = np.mean(id_tvs) if id_tvs else 0
        h = np.mean(heavy_tvs) if heavy_tvs else 0
        i_amp = i / e if e > 0.001 else float('inf')
        h_amp = h / e if e > 0.001 else float('inf')

        pair_data[pair_name] = {"baseline": e, "identity": i, "heavy": h,
                                 "id_amp": i_amp, "hvy_amp": h_amp}

        print(f"{pair_name:>20s} {e:10.4f} {i:10.4f} {i_amp:10.1f}x {h:10.4f} {h_amp:10.1f}x {i-e:+10.4f} {h-e:+10.4f}")

    # Sort by baseline and check monotonicity
    sorted_pairs = sorted(pair_data.items(), key=lambda x: x[1]["baseline"])
    print("\n=== SORTED BY BASELINE (low -> high) ===")
    print(f"{'Pair':>20s} {'baseline':>10s} {'id_amp':>10s} {'hvy_amp':>10s} {'regime':>15s}")
    print("-" * 70)

    for pair_name, d in sorted_pairs:
        regime = "chain-res" if d["baseline"] < 0.01 else "recency" if d["baseline"] < 0.03 else "divergent"
        print(f"{pair_name:>20s} {d['baseline']:10.4f} {d['id_amp']:10.1f}x {d['hvy_amp']:10.1f}x {regime:>15s}")

    # Monotonicity check
    baselines = [d["baseline"] for _, d in sorted_pairs]
    id_amps = [d["id_amp"] for _, d in sorted_pairs]
    hvy_amps = [d["hvy_amp"] for _, d in sorted_pairs]

    from scipy.stats import spearmanr
    rho_id, p_id = spearmanr(baselines, id_amps)
    rho_hvy, p_hvy = spearmanr(baselines, hvy_amps)

    print(f"\nSpearman correlation (baseline vs id_amp):  rho={rho_id:.3f}, p={p_id:.4f}")
    print(f"Spearman correlation (baseline vs hvy_amp): rho={rho_hvy:.3f}, p={p_hvy:.4f}")

    if abs(rho_id) > 0.7 and p_id < 0.05:
        print("=> CONTINUUM: identity amplification is monotonically related to baseline")
    else:
        print("=> NOT a clean continuum -- regimes may be discrete")

    # Per-cell detail for outlier checking
    print("\n=== PER-CELL EMPTY BASELINES ===")
    for pair_name in SURFACE_PAIRS:
        tvs = [r["tv"] for r in results if r["pair"] == pair_name and r["continuation"] == "empty"]
        cells = [r["cell"] for r in results if r["pair"] == pair_name and r["continuation"] == "empty"]
        detail = ", ".join(f"{c}={t:.4f}" for c, t in zip(cells, tvs))
        print(f"  {pair_name:>20s}: {detail}")

    summary = {
        "experiment": 79,
        "name": "regime_classifier",
        "n_readouts": n_done,
        "n_comparisons": len(results),
        "pair_data": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in pair_data.items()},
        "results": results,
    }

    result_dir = Path("experiments/results/omega_regime_classifier")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
