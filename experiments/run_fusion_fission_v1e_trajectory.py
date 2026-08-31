"""
Fusion-Fission v1e: Residual stream trajectory analysis.

Since fusion is a whole-state property (v1d), track how the hidden
state for each world CHANGES from layer to layer. Measure:
- Update direction similarity: do A-different worlds update in the
  same direction at fused layers?
- Convergence/divergence rate: do world trajectories converge (fuse)
  or diverge (fission) at each layer?
- Update magnitude: how much does the state change at each layer?

The hypothesis: at fused layers, the layer-to-layer updates for
A-different worlds are MORE SIMILAR in direction (they converge),
while at separate layers they diverge.
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
DTYPE = torch.float32

SEPARATE_LAYERS = [5, 6, 8, 22, 23, 24]
FUSED_LAYERS = [2, 7, 13, 14, 18]

WORLDS = [
    {"hesk": "red",  "vorn": "red",  "label": "00"},
    {"hesk": "red",  "vorn": "blue", "label": "01"},
    {"hesk": "blue", "vorn": "red",  "label": "10"},
    {"hesk": "blue", "vorn": "blue", "label": "11"},
]


def make_prompt(hesk_color, vorn_color):
    return f"HESK is {hesk_color}. VORN is {vorn_color}. Remember these facts.\n"


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/results/fusion_fission_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Fusion-Fission v1e: Trajectory Analysis ===\n")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()

    # Get hidden states for all 4 worlds (last token)
    all_hs = {}
    for w in WORLDS:
        prompt = make_prompt(w["hesk"], w["vorn"])
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        all_hs[w["label"]] = [h[0, -1].numpy() for h in out.hidden_states]

    n_layers = len(all_hs["00"]) - 1
    print(f"Layers: {n_layers}\n")

    results = []

    for layer in range(1, n_layers + 1):
        # State at this layer and previous layer (last token)
        prev = {label: all_hs[label][layer - 1] for label in ["00", "01", "10", "11"]}
        curr = {label: all_hs[label][layer] for label in ["00", "01", "10", "11"]}

        # Update vectors: how much did the state change at this layer
        updates = {label: curr[label] - prev[label] for label in ["00", "01", "10", "11"]}

        # Update magnitudes
        magnitudes = {label: float(np.linalg.norm(updates[label])) for label in ["00", "01", "10", "11"]}

        # KEY METRIC 1: Update direction similarity for A-different pairs
        # Pairs that differ only in A: (00,10) and (01,11)
        # If fusion is happening, A-different worlds should update similarly
        update_sim_a_pair1 = cosine_sim(updates["00"], updates["10"])  # differ in A
        update_sim_a_pair2 = cosine_sim(updates["01"], updates["11"])  # differ in A
        update_sim_a_mean = (update_sim_a_pair1 + update_sim_a_pair2) / 2

        # Pairs that differ only in B: (00,01) and (10,11)
        update_sim_b_pair1 = cosine_sim(updates["00"], updates["01"])  # differ in B
        update_sim_b_pair2 = cosine_sim(updates["10"], updates["11"])  # differ in B
        update_sim_b_mean = (update_sim_b_pair1 + update_sim_b_pair2) / 2

        # Same-world pair (sanity)
        update_sim_same = cosine_sim(updates["00"], updates["00"])  # always 1

        # KEY METRIC 2: State convergence/divergence
        # Distance between A-different worlds at this layer vs previous
        dist_a_curr = np.linalg.norm(curr["00"] - curr["10"]) + np.linalg.norm(curr["01"] - curr["11"])
        dist_a_prev = np.linalg.norm(prev["00"] - prev["10"]) + np.linalg.norm(prev["01"] - prev["11"])
        convergence_a = float(dist_a_prev - dist_a_curr)  # positive = converging

        dist_b_curr = np.linalg.norm(curr["00"] - curr["01"]) + np.linalg.norm(curr["10"] - curr["11"])
        dist_b_prev = np.linalg.norm(prev["00"] - prev["01"]) + np.linalg.norm(prev["10"] - prev["11"])
        convergence_b = float(dist_b_prev - dist_b_curr)  # positive = converging

        tag = ""
        if layer in FUSED_LAYERS: tag = " [FUSED]"
        elif layer in SEPARATE_LAYERS: tag = " [SEPARATE]"

        lr = {
            "layer": layer,
            "update_sim_a_different": float(update_sim_a_mean),
            "update_sim_b_different": float(update_sim_b_mean),
            "convergence_a": convergence_a,
            "convergence_b": convergence_b,
            "avg_update_magnitude": float(np.mean(list(magnitudes.values()))),
        }
        results.append(lr)

        # Direction: at fused layers, A-different worlds should update MORE similarly
        print(f"  Layer {layer:2d}{tag:12s}  "
              f"upd_sim(A-diff)={update_sim_a_mean:.4f}  "
              f"upd_sim(B-diff)={update_sim_b_mean:.4f}  "
              f"conv_A={convergence_a:+.2f}  "
              f"conv_B={convergence_b:+.2f}  "
              f"mag={lr['avg_update_magnitude']:.1f}")

    # Summary
    print("\n=== Summary ===")
    fused_upd_a = [r["update_sim_a_different"] for r in results if r["layer"] in FUSED_LAYERS]
    sep_upd_a = [r["update_sim_a_different"] for r in results if r["layer"] in SEPARATE_LAYERS]
    fused_conv_a = [r["convergence_a"] for r in results if r["layer"] in FUSED_LAYERS]
    sep_conv_a = [r["convergence_a"] for r in results if r["layer"] in SEPARATE_LAYERS]
    fused_conv_b = [r["convergence_b"] for r in results if r["layer"] in FUSED_LAYERS]
    sep_conv_b = [r["convergence_b"] for r in results if r["layer"] in SEPARATE_LAYERS]

    print(f"  FUSED:    update_sim(A-diff) = {np.mean(fused_upd_a):.4f}")
    print(f"  SEPARATE: update_sim(A-diff) = {np.mean(sep_upd_a):.4f}")
    print(f"  FUSED:    convergence_A = {np.mean(fused_conv_a):+.3f}, convergence_B = {np.mean(fused_conv_b):+.3f}")
    print(f"  SEPARATE: convergence_A = {np.mean(sep_conv_a):+.3f}, convergence_B = {np.mean(sep_conv_b):+.3f}")

    if np.mean(fused_upd_a) > np.mean(sep_upd_a):
        verdict = "FUSED layers update A-different worlds MORE SIMILARLY: fusion = trajectory convergence"
    else:
        verdict = "No clear update-similarity difference: fusion is not simple trajectory convergence"

    print(f"\n  Verdict: {verdict}")

    out_path = out_dir / "fusion_fission_v1e_trajectory.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "verdict": verdict}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
