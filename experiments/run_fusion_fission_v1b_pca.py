"""
Fusion-Fission v1b: PCA probe at fused vs separate layers.

Question: Can a linear subspace find the compositional structure that
global cosine similarity misses?
- YES → structure is low-rank linear (R^n but not global-R^n)
- NO → structure is genuinely nonlinear — native math territory

Uses the same 4 worlds from v1. At each layer, extracts per-token
hidden states and runs PCA to see if worlds separate.
"""

import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
DTYPE = torch.float32

WORLDS = [
    {"hesk": "red",  "vorn": "red",  "label": "00"},
    {"hesk": "red",  "vorn": "blue", "label": "01"},
    {"hesk": "blue", "vorn": "red",  "label": "10"},
    {"hesk": "blue", "vorn": "blue", "label": "11"},
]

def make_prompt(hesk_color, vorn_color):
    return (
        f"HESK is {hesk_color}. VORN is {vorn_color}. "
        f"Remember these facts.\n"
    )


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/results/fusion_fission_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Fusion-Fission v1b: PCA Probe ===\n")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()

    # Get hidden states for all 4 worlds
    prompts = {w["label"]: make_prompt(w["hesk"], w["vorn"]) for w in WORLDS}

    # Verify tokenization lengths match
    lens = {label: len(tok(p).input_ids) for label, p in prompts.items()}
    print(f"Token lengths: {lens}")
    if len(set(lens.values())) > 1:
        print("WARNING: Token lengths differ. PCA comparison may be unreliable.")

    all_hs = {}
    for label, prompt in prompts.items():
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        all_hs[label] = [h[0].numpy() for h in out.hidden_states]  # [n_layers+1][seq_len, d]

    n_layers = len(all_hs["00"]) - 1  # exclude embedding layer
    seq_len = all_hs["00"][0].shape[0]
    d_model = all_hs["00"][0].shape[1]
    print(f"Layers: {n_layers}, Seq len: {seq_len}, d_model: {d_model}\n")

    # Key layers from v1 results
    separate_layers = [5, 6, 8, 22, 23, 24]
    fused_layers = [2, 7, 13, 14, 18]

    pca_results = []

    for layer in range(n_layers + 1):
        # Stack all 4 worlds' representations: [4, seq_len, d]
        stacked = np.stack([all_hs[label][layer] for label in ["00", "01", "10", "11"]])

        # Method 1: Mean-pool over sequence -> [4, d]
        mean_reps = stacked.mean(axis=1)

        # Center
        mean_reps_centered = mean_reps - mean_reps.mean(axis=0)

        # PCA via SVD
        U, S, Vt = np.linalg.svd(mean_reps_centered, full_matrices=False)

        # Project onto top 2 PCs
        proj = mean_reps_centered @ Vt[:2].T  # [4, 2]

        # Check if worlds separate by fact A (HESK) and fact B (VORN)
        # Fact A: 00,01 should cluster vs 10,11
        # Fact B: 00,10 should cluster vs 01,11
        coords = {label: proj[i] for i, label in enumerate(["00", "01", "10", "11"])}

        # A-separation: distance between A=red cluster and A=blue cluster
        a_red_center = (coords["00"] + coords["01"]) / 2
        a_blue_center = (coords["10"] + coords["11"]) / 2
        a_sep = np.linalg.norm(a_blue_center - a_red_center)

        # B-separation: distance between B=red cluster and B=blue cluster
        b_red_center = (coords["00"] + coords["10"]) / 2
        b_blue_center = (coords["01"] + coords["11"]) / 2
        b_sep = np.linalg.norm(b_blue_center - b_red_center)

        # Variance explained
        var_explained = S[:2] ** 2 / (S ** 2).sum() if (S ** 2).sum() > 0 else [0, 0]

        # Method 2: Per-token PCA at the last token position
        last_tok_reps = stacked[:, -1, :]  # [4, d]
        lt_centered = last_tok_reps - last_tok_reps.mean(axis=0)
        U_lt, S_lt, Vt_lt = np.linalg.svd(lt_centered, full_matrices=False)
        proj_lt = lt_centered @ Vt_lt[:2].T
        coords_lt = {label: proj_lt[i] for i, label in enumerate(["00", "01", "10", "11"])}

        # Check if PCA2 separates by A vs B at last token
        a_sep_lt = np.linalg.norm(
            (coords_lt["10"] + coords_lt["11"]) / 2 -
            (coords_lt["00"] + coords_lt["01"]) / 2
        )
        b_sep_lt = np.linalg.norm(
            (coords_lt["01"] + coords_lt["11"]) / 2 -
            (coords_lt["00"] + coords_lt["10"]) / 2
        )

        # Orthogonality: angle between A-direction and B-direction
        a_dir = (coords_lt["10"] + coords_lt["11"]) / 2 - (coords_lt["00"] + coords_lt["01"]) / 2
        b_dir = (coords_lt["01"] + coords_lt["11"]) / 2 - (coords_lt["00"] + coords_lt["10"]) / 2
        a_norm = np.linalg.norm(a_dir)
        b_norm = np.linalg.norm(b_dir)
        if a_norm > 1e-8 and b_norm > 1e-8:
            ortho = abs(np.dot(a_dir, b_dir) / (a_norm * b_norm))
        else:
            ortho = float('nan')

        status_tag = ""
        if layer in separate_layers:
            status_tag = " [SEPARATE]"
        elif layer in fused_layers:
            status_tag = " [FUSED]"

        result = {
            "layer": layer,
            "mean_pool_a_sep": float(a_sep),
            "mean_pool_b_sep": float(b_sep),
            "var_explained_pc1": float(var_explained[0]),
            "var_explained_pc2": float(var_explained[1]) if len(var_explained) > 1 else 0,
            "last_tok_a_sep": float(a_sep_lt),
            "last_tok_b_sep": float(b_sep_lt),
            "ab_orthogonality": float(ortho),
            "last_tok_coords": {k: v.tolist() for k, v in coords_lt.items()},
        }
        pca_results.append(result)

        print(f"  Layer {layer:2d}{status_tag:12s}  "
              f"mean(A={a_sep:.4f}, B={b_sep:.4f})  "
              f"last(A={a_sep_lt:.4f}, B={b_sep_lt:.4f})  "
              f"ortho={ortho:.3f}  "
              f"var={var_explained[0]:.3f}/{var_explained[1] if len(var_explained)>1 else 0:.3f}")

    # Summary: compare PCA separation at FUSED vs SEPARATE layers
    print("\n=== Summary: PCA separation at behavioral FUSED vs SEPARATE layers ===")
    fused_a = [r["last_tok_a_sep"] for r in pca_results if r["layer"] in fused_layers]
    fused_b = [r["last_tok_b_sep"] for r in pca_results if r["layer"] in fused_layers]
    sep_a = [r["last_tok_a_sep"] for r in pca_results if r["layer"] in separate_layers]
    sep_b = [r["last_tok_b_sep"] for r in pca_results if r["layer"] in separate_layers]

    print(f"  FUSED layers:    A-sep mean={np.mean(fused_a):.4f}  B-sep mean={np.mean(fused_b):.4f}")
    print(f"  SEPARATE layers: A-sep mean={np.mean(sep_a):.4f}  B-sep mean={np.mean(sep_b):.4f}")

    fused_ortho = [r["ab_orthogonality"] for r in pca_results if r["layer"] in fused_layers and not np.isnan(r["ab_orthogonality"])]
    sep_ortho = [r["ab_orthogonality"] for r in pca_results if r["layer"] in separate_layers and not np.isnan(r["ab_orthogonality"])]
    if fused_ortho and sep_ortho:
        print(f"  FUSED layers:    AB-orthogonality mean={np.mean(fused_ortho):.4f}")
        print(f"  SEPARATE layers: AB-orthogonality mean={np.mean(sep_ortho):.4f}")

    verdict = ""
    if np.mean(sep_a) > 2 * np.mean(fused_a) or np.mean(sep_b) > 2 * np.mean(fused_b):
        verdict = "PCA SEPARATES: structure is low-rank linear (R^n subspace)"
    elif np.mean(fused_a) > np.mean(sep_a) * 0.8:
        verdict = "PCA DOES NOT SEPARATE: structure may be genuinely nonlinear"
    else:
        verdict = "INCONCLUSIVE: PCA shows some but not decisive separation"

    print(f"\n  Verdict: {verdict}")

    out_path = out_dir / "fusion_fission_v1b_pca.json"
    with open(out_path, "w") as f:
        json.dump({"pca_results": pca_results, "verdict": verdict}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
