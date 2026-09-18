"""Measurement Stability: Is the readout deterministic?

Since we use torch.float32 on CPU with no sampling (just softmax readout),
measurements should be perfectly deterministic. But if there's any source
of non-determinism (thread scheduling, numerical precision), omega values
could be measuring noise.

This test: load model twice, run identical readouts, compare distributions.
If TV between runs > tau, omega is measuring noise.

6 cells × 4 surfaces × 2 continuations × 2 runs = 96 readouts (~30s)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix


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
    "assert_novar": "# State unchanged.\n",
    "assert_var": "# {var} stays the same.\n",
    "moderate_novar": "# Values preserved.\n",
    "moderate_var": "# {var} retains its value.\n",
}

CONTINUATIONS = {
    "none": "",
    "identity": "{var} = {var}\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 6), (4, "y", 8),
]


def run_one_pass(adapter, label):
    query_tmpl = MODEL_CFG["query_template"]
    results = {}
    n = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_name, surf_tmpl in SURFACES.items():
            surface_text = surf_tmpl.replace("{var}", var)
            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                full_suffix = surface_text + cont_text + query
                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                key = (var, val, surf_name, cont_name)
                results[key] = dist
                n += 1

    print(f"  {label}: {n} readouts")
    return results


def run_stability():
    print("=== Measurement Stability Test ===")
    print(f"Cells: {len(CELLS)}, Surfaces: {len(SURFACES)}, "
          f"Continuations: {len(CONTINUATIONS)}")
    n_per_pass = len(CELLS) * len(SURFACES) * len(CONTINUATIONS)
    print(f"Readouts per pass: {n_per_pass}")
    print()

    t0 = time.time()

    print("Pass 1: loading model fresh...")
    adapter1 = ModelAdapter(MODEL_CFG)
    results1 = run_one_pass(adapter1, "pass_1")
    del adapter1

    print("Pass 2: loading model fresh...")
    adapter2 = ModelAdapter(MODEL_CFG)
    results2 = run_one_pass(adapter2, "pass_2")
    del adapter2

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    tvs = []
    max_diffs = []
    print(f"\n=== Per-Readout Stability ===")
    print(f"{'Key':<40s} {'tv':>10s} {'max_diff':>10s}")
    print("-" * 65)

    for key in sorted(results1.keys()):
        d1 = results1[key]
        d2 = results2[key]
        tv = float(0.5 * np.abs(d1 - d2).sum())
        md = float(np.abs(d1 - d2).max())
        tvs.append(tv)
        max_diffs.append(md)
        if tv > 0.001:
            var, val, surf, cont = key
            print(f"  {var}={val} {surf} {cont:<10s} {tv:10.6f} {md:10.6f}")

    print(f"\n=== Summary ===")
    print(f"  Mean TV between runs: {np.mean(tvs):.8f}")
    print(f"  Max TV between runs:  {np.max(tvs):.8f}")
    print(f"  Std TV between runs:  {np.std(tvs):.8f}")
    print(f"  Mean max-diff:        {np.mean(max_diffs):.8f}")
    print(f"  Max max-diff:         {np.max(max_diffs):.8f}")

    if np.max(tvs) < 1e-6:
        print(f"\n  RESULT: Perfectly deterministic (max TV < 1e-6)")
    elif np.max(tvs) < 0.001:
        print(f"\n  RESULT: Near-deterministic (max TV < 0.001)")
    elif np.max(tvs) < 0.01:
        print(f"\n  RESULT: Low noise (max TV < 0.01 = tau)")
    else:
        print(f"\n  WARNING: Noise floor ({np.max(tvs):.4f}) may affect omega!")

    result_dir = Path("experiments/results/omega_stability")
    result_dir.mkdir(parents=True, exist_ok=True)
    stability = {
        "mean_tv": float(np.mean(tvs)),
        "max_tv": float(np.max(tvs)),
        "std_tv": float(np.std(tvs)),
        "n_readouts": len(tvs),
        "all_tvs": [float(t) for t in tvs],
    }
    with open(result_dir / "stability.json", "w") as f:
        json.dump(stability, f, indent=2)

    print(f"\nSaved to {result_dir}/")
    return stability


if __name__ == "__main__":
    run_stability()
