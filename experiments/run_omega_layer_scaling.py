"""Omega Layer-wise Fingerprint at Different Model Sizes.

Runs the same layer-wise omega measurement as run_omega_layers.py but on
Qwen3-0.6B to test whether the onset layer shifts with model capacity.

Both Qwen3-0.6B and 1.7B have 28 layers. If the onset is at a fixed
fraction of depth (~70%, layer 20), it should appear at L20 in both.
If onset depends on capacity, it might shift later in the smaller model.
"""

import json
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import build_prefix
from run_omega_sweep import MODEL_CFG
from run_omega_layers import (
    LayerProbeAdapter, LAYER_CHECKPOINTS, CELLS, PAIRS,
    SURFACES, CONTINUATIONS, tv, compute_layer_omega, print_summary, build_summary
)

MODEL_ID = "Qwen/Qwen3-0.6B"
RESULT_DIR = Path("experiments/results/omega_layer_scaling")


def main():
    print(f"=== Layer-wise Omega @ {MODEL_ID} ===", flush=True)

    adapter = LayerProbeAdapter(MODEL_ID)
    query_tmpl = MODEL_CFG["query_template"]
    observations = []
    t0 = time.time()
    n_readouts = 0

    for depth, var, val in CELLS:
        tmpl_key = f"depth{depth}_single"
        tmpl = MODEL_CFG["templates"][tmpl_key]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)

        for surf_name, surf_tmpl in SURFACES.items():
            surface_text = surf_tmpl.replace("{var}", var)
            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                full_text = prefix_text + surface_text + cont_text + query

                layer_dists = adapter.get_layer_dists(full_text, LAYER_CHECKPOINTS)
                n_readouts += 1

                obs = {
                    "depth": depth,
                    "var": var,
                    "val": val,
                    "surface": surf_name,
                    "continuation": cont_name,
                    "layer_dists": {str(k): v.tolist() for k, v in layer_dists.items()},
                }
                observations.append(obs)

                if n_readouts % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_readouts} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n_readouts} readouts in {elapsed:.1f}s, "
          f"{adapter.call_count} model calls", flush=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULT_DIR / "observations.json", "w") as f:
        json.dump(observations, f)

    print("\n=== Computing layer-wise omega ===", flush=True)
    layer_omega = compute_layer_omega(observations, PAIRS)

    with open(RESULT_DIR / "layer_omega.json", "w") as f:
        json.dump(layer_omega, f, indent=2)

    print_summary(layer_omega)

    summary = build_summary(layer_omega, n_readouts, elapsed)
    summary["model"] = MODEL_ID
    with open(RESULT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Compare with 1.7B if available
    ref_path = Path("experiments/results/omega_layers/summary.json")
    if ref_path.exists():
        with open(ref_path) as f:
            ref = json.load(f)
        print("\n=== Comparison: 0.6B vs 1.7B ===")
        print(f"{'Layer':>8}  {'0.6B identity':>14}  {'1.7B identity':>14}  {'ratio':>8}")
        ref_ident = ref.get("identity_omega_by_layer", {})
        cur_ident = summary.get("identity_omega_by_layer", {})
        for lk in [str(L) for L in LAYER_CHECKPOINTS] + ["final"]:
            v06 = cur_ident.get(lk, None)
            v17 = ref_ident.get(lk, None)
            s06 = f"{v06:.6f}" if v06 is not None else "---"
            s17 = f"{v17:.6f}" if v17 is not None else "---"
            if v06 and v17 and v17 > 0:
                ratio = f"{v06/v17:.2f}x"
            else:
                ratio = "---"
            print(f"{lk:>8}  {s06:>14}  {s17:>14}  {ratio:>8}")

    del adapter
    print(f"\nResults saved to {RESULT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
