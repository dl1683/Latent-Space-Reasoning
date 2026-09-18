"""Omega Repetition Disentanglement (Experiment #40).

Tests whether the phase-transition interference (rep_2 < rep_1) is driven by
token count, statement order, exact repetition, or content variation.

Sequences tested (I = identity `x = x\n`, N = neutral `# OK.\n`):
  I, N, II, IN, NI, NN, I_padded, N_padded, I_paraphrase, N_varied
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix


MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B",
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
    "assert_novar_1": "# No changes.\n",
    "assert_var_1": "# {var} is unchanged.\n",
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "minimal_ok": "# OK.\n",
    "minimal_good": "# Good.\n",
}

EQUIV_PAIRS = [
    ("assert_novar_1", "assert_var_1", "strong_assert"),
    ("assert_novar_2", "assert_var_2", "strong_assert_2"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]


def make_continuations(var):
    """Build the 10 continuation sequences for disentanglement."""
    I = f"{var} = {var}\n"
    N = "# OK.\n"
    N2 = "# Good.\n"
    padding = "pass\n"

    return {
        "none":           "",
        "I":              I,
        "N":              N,
        "II":             I + I,
        "IN":             I + N,
        "NI":             N + I,
        "NN":             N + N,
        "I_padded":       I + padding,       # same token count as II
        "N_padded":       N + padding,       # same token count as NN
        "I_paraphrase":   f"_ = {var}\n{var} = _\n",  # identity through temp
        "N_varied":       N + N2,            # two DIFFERENT neutrals
        "III":            I + I + I,
        "NNN":            N + N + N,
        "INI":            I + N + I,
        "NIN":            N + I + N,
    }


def tv(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()


def run_experiment():
    print("=== Omega Repetition Disentanglement (Exp #40) ===\n")

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    observations = []
    t0 = time.time()
    n_readouts = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        continuations = make_continuations(var)

        for surf_name, surf_tmpl in SURFACES.items():
            surface_text = surf_tmpl.replace("{var}", var)

            for cont_name, cont_text in continuations.items():
                full_suffix = surface_text + cont_text + query
                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                n_readouts += 1

                observations.append({
                    "depth": depth, "var": var, "val": val,
                    "surface": surf_name, "continuation": cont_name,
                    "dist": dist.tolist(),
                })

                if n_readouts % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_readouts} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done: {n_readouts} readouts in {elapsed:.1f}s\n")

    result_dir = Path("experiments/results/omega_repetition")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    # Compute pair distances and omega
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    for tau in [0.01, 0.05]:
        print(f"\n=== OMEGA TABLE (tau={tau}) ===")
        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            # Find eligible cells (baseline TV <= tau)
            eligible_cells = []
            for depth, var, val in CELLS:
                key_a = (depth, var, val, surf_a, "none")
                key_b = (depth, var, val, surf_b, "none")
                if key_a in obs_idx and key_b in obs_idx:
                    baseline_tv = tv(obs_idx[key_a], obs_idx[key_b])
                    if baseline_tv <= tau:
                        eligible_cells.append((depth, var, val))

            if not eligible_cells:
                print(f"\n  {pair_name}: 0 eligible cells")
                continue

            print(f"\n  {pair_name}: {len(eligible_cells)} eligible cells")
            cont_names = sorted(make_continuations("x").keys())
            results_table = {}

            for cn in cont_names:
                tvs = []
                for depth, var, val in eligible_cells:
                    key_a = (depth, var, val, surf_a, cn)
                    key_b = (depth, var, val, surf_b, cn)
                    if key_a in obs_idx and key_b in obs_idx:
                        tvs.append(tv(obs_idx[key_a], obs_idx[key_b]))
                if tvs:
                    mean_tv = np.mean(tvs)
                    max_tv = np.max(tvs)
                    status = "PRESERVE" if max_tv <= tau else "AMPLIFY"
                    results_table[cn] = {
                        "mean_tv": float(mean_tv),
                        "max_tv": float(max_tv),
                        "n": len(tvs),
                        "status": status,
                    }
                    print(f"    {cn:20s}: mean={mean_tv:.5f} max={max_tv:.5f} "
                          f"n={len(tvs)} {status}")

    # Key comparisons for disentanglement
    print("\n\n=== DISENTANGLEMENT ANALYSIS ===\n")
    print("Testing competing predictions:\n")

    for tau in [0.01, 0.05]:
        print(f"--- tau={tau} ---")
        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            eligible_cells = []
            for depth, var, val in CELLS:
                key_a = (depth, var, val, surf_a, "none")
                key_b = (depth, var, val, surf_b, "none")
                if key_a in obs_idx and key_b in obs_idx:
                    if tv(obs_idx[key_a], obs_idx[key_b]) <= tau:
                        eligible_cells.append((depth, var, val))

            if not eligible_cells:
                continue

            def mean_omega(cont_name):
                tvs = []
                for depth, var, val in eligible_cells:
                    key_a = (depth, var, val, surf_a, cont_name)
                    key_b = (depth, var, val, surf_b, cont_name)
                    if key_a in obs_idx and key_b in obs_idx:
                        tvs.append(tv(obs_idx[key_a], obs_idx[key_b]))
                return np.mean(tvs) if tvs else None

            I_val = mean_omega("I")
            II_val = mean_omega("II")
            IN_val = mean_omega("IN")
            NI_val = mean_omega("NI")
            NN_val = mean_omega("NN")
            N_val = mean_omega("N")
            I_pad = mean_omega("I_padded")
            N_var = mean_omega("N_varied")
            I_para = mean_omega("I_paraphrase")

            print(f"\n  {pair_name} ({len(eligible_cells)} cells):")
            if I_val is not None and II_val is not None and I_pad is not None:
                print(f"    Token count: II={II_val:.5f} vs I_padded={I_pad:.5f} "
                      f"({'TOKEN COUNT' if abs(II_val - I_pad) < 0.005 else 'CONTENT MATTERS'})")
            if NI_val is not None and IN_val is not None:
                print(f"    Order:       NI={NI_val:.5f} vs IN={IN_val:.5f} "
                      f"({'LAST WINS' if NI_val > IN_val else 'FIRST WINS' if IN_val > NI_val else 'ORDER NEUTRAL'})")
            if II_val is not None and I_val is not None:
                print(f"    Repetition:  II={II_val:.5f} vs I={I_val:.5f} "
                      f"({'SUPPRESSION' if II_val < I_val else 'ADDITIVE' if II_val > I_val else 'FLAT'})")
            if NN_val is not None and N_var is not None:
                print(f"    Variation:   NN={NN_val:.5f} vs N_varied={N_var:.5f} "
                      f"({'VARIETY HELPS' if N_var > NN_val else 'REPETITION HELPS'})")
            if I_para is not None and I_val is not None:
                print(f"    Paraphrase:  I_para={I_para:.5f} vs I={I_val:.5f} "
                      f"({'EQUIVALENT' if abs(I_para - I_val) < 0.005 else 'DIFFERENT'})")

    # Save summary
    summary = {
        "n_readouts": n_readouts,
        "elapsed_s": elapsed,
        "n_cells": len(CELLS),
        "n_pairs": len(EQUIV_PAIRS),
        "n_continuations": len(make_continuations("x")),
    }
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {result_dir}/")
    del adapter


if __name__ == "__main__":
    run_experiment()
