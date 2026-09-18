"""Cross-matrix SVD analysis: W_gate @ W_up^T.

The cross-matrix C = W_gate @ W_up^T captures row-pairing information that
Gram operators miss. Its SVD reveals whether the gate-up coupling has
structure beyond what random pairing would give.

Key test: compute singular values of C for the original and permuted pairing,
see if the original has distinctive spectral structure.

Model: SmolLM2-135M (CPU-only)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch


def analyze_cross_matrix():
    from transformers import AutoModelForCausalLM

    print("Loading SmolLM2-135M...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", dtype=torch.float32, device_map="cpu"
    )
    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    m = model.config.intermediate_size
    print(f"  Loaded in {time.time()-t0:.1f}s (d={d}, m={m}, layers={n_layers})")

    rng = np.random.default_rng(42)
    results = {}

    for l in range(n_layers):
        gate_w = model.model.layers[l].mlp.gate_proj.weight.detach().numpy()
        up_w = model.model.layers[l].mlp.up_proj.weight.detach().numpy()

        # Cross-matrix: C = W_gate @ W_up^T, shape (m, m), rank <= d
        # Too expensive to compute full m x m SVD.
        # Instead compute SVD of the d x d matrix: W_gate^T @ W_up (= V S U^T factors of C)
        # The nonzero singular values of C = W_gate @ W_up^T are the same as
        # those of G = W_gate^T @ W_up (d x d matrix, much smaller)
        G_orig = gate_w.T @ up_w  # d x d

        # Random permutation control
        perm = rng.permutation(m)
        up_perm = up_w[perm]
        G_perm = gate_w.T @ up_perm  # d x d

        # SVD of both
        sv_orig = np.linalg.svd(G_orig, compute_uv=False)
        sv_perm = np.linalg.svd(G_perm, compute_uv=False)

        # Also compute for the Gram operators (for comparison)
        S_gate = gate_w.T @ gate_w / m  # d x d
        S_up = up_w.T @ up_w / m  # d x d
        sv_gate = np.linalg.svd(S_gate, compute_uv=False)
        sv_up = np.linalg.svd(S_up, compute_uv=False)

        # Nuclear norm (sum of singular values) — measures total coupling
        nuc_orig = float(np.sum(sv_orig))
        nuc_perm = float(np.sum(sv_perm))

        # Frobenius norm (sqrt of sum of squared sv) — measures energy
        frob_orig = float(np.sqrt(np.sum(sv_orig ** 2)))
        frob_perm = float(np.sqrt(np.sum(sv_perm ** 2)))

        # Spectral norm (max sv) — measures peak coupling
        spec_orig = float(sv_orig[0])
        spec_perm = float(sv_perm[0])

        # Effective rank of cross-matrix
        sv_norm = sv_orig / np.sum(sv_orig)
        eff_rank = float(np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-15))))

        sv_norm_p = sv_perm / np.sum(sv_perm)
        eff_rank_perm = float(np.exp(-np.sum(sv_norm_p * np.log(sv_norm_p + 1e-15))))

        # Power-law fit on cross-matrix singular values
        log_k = np.log(np.arange(1, len(sv_orig) + 1))
        log_sv = np.log(sv_orig + 1e-15)
        slope, _ = np.polyfit(log_k[:d // 2], log_sv[:d // 2], 1)
        alpha_cross = -slope

        log_sv_p = np.log(sv_perm + 1e-15)
        slope_p, _ = np.polyfit(log_k[:d // 2], log_sv_p[:d // 2], 1)
        alpha_perm = -slope_p

        results[f"L{l}"] = {
            "nuc_orig": nuc_orig,
            "nuc_perm": nuc_perm,
            "nuc_ratio": nuc_orig / (nuc_perm + 1e-10),
            "frob_orig": frob_orig,
            "frob_perm": frob_perm,
            "frob_ratio": frob_orig / (frob_perm + 1e-10),
            "spec_orig": spec_orig,
            "spec_perm": spec_perm,
            "spec_ratio": spec_orig / (spec_perm + 1e-10),
            "eff_rank_orig": eff_rank,
            "eff_rank_perm": eff_rank_perm,
            "alpha_cross": alpha_cross,
            "alpha_perm": alpha_perm,
            "top5_sv_orig": sv_orig[:5].tolist(),
            "top5_sv_perm": sv_perm[:5].tolist(),
        }

        if l % 5 == 0 or l == n_layers - 1:
            print(
                f"  L{l:2d}: nuc_ratio={nuc_orig/nuc_perm:.4f}, "
                f"spec_ratio={spec_orig/spec_perm:.4f}, "
                f"erank={eff_rank:.1f}/{eff_rank_perm:.1f}, "
                f"alpha={alpha_cross:.3f}/{alpha_perm:.3f}"
            )

    print("\n" + "=" * 70)
    print("CROSS-MATRIX SVD ANALYSIS")
    print("=" * 70)

    nuc_ratios = [results[f"L{l}"]["nuc_ratio"] for l in range(n_layers)]
    spec_ratios = [results[f"L{l}"]["spec_ratio"] for l in range(n_layers)]
    frob_ratios = [results[f"L{l}"]["frob_ratio"] for l in range(n_layers)]
    eranks = [results[f"L{l}"]["eff_rank_orig"] for l in range(n_layers)]
    eranks_p = [results[f"L{l}"]["eff_rank_perm"] for l in range(n_layers)]
    alphas = [results[f"L{l}"]["alpha_cross"] for l in range(n_layers)]
    alphas_p = [results[f"L{l}"]["alpha_perm"] for l in range(n_layers)]

    print(f"\nNuclear norm ratio (original/permuted):")
    print(f"  Mean: {np.mean(nuc_ratios):.4f}, range: {min(nuc_ratios):.4f}-{max(nuc_ratios):.4f}")
    print(f"\nSpectral norm ratio (original/permuted):")
    print(f"  Mean: {np.mean(spec_ratios):.4f}, range: {min(spec_ratios):.4f}-{max(spec_ratios):.4f}")
    print(f"\nFrobenius norm ratio (original/permuted):")
    print(f"  Mean: {np.mean(frob_ratios):.4f}, range: {min(frob_ratios):.4f}-{max(frob_ratios):.4f}")
    print(f"\nEffective rank:")
    print(f"  Original: mean={np.mean(eranks):.1f}, range={min(eranks):.1f}-{max(eranks):.1f}")
    print(f"  Permuted: mean={np.mean(eranks_p):.1f}, range={min(eranks_p):.1f}-{max(eranks_p):.1f}")
    print(f"\nPower-law exponent of cross-matrix SV:")
    print(f"  Original: mean={np.mean(alphas):.3f}, range={min(alphas):.3f}-{max(alphas):.3f}")
    print(f"  Permuted: mean={np.mean(alphas_p):.3f}, range={min(alphas_p):.3f}-{max(alphas_p):.3f}")

    if np.mean(nuc_ratios) > 1.05:
        print(f"\n>> Cross-matrix has HIGHER coupling than random pairing (ratio > 1)")
    elif np.mean(nuc_ratios) < 0.95:
        print(f"\n>> Cross-matrix has LOWER coupling than random pairing (ratio < 1)")
    else:
        print(f"\n>> Cross-matrix coupling is SIMILAR to random pairing")

    summary = {
        "model": "SmolLM2-135M",
        "d": d,
        "m": m,
        "n_layers": n_layers,
        "nuc_ratios": nuc_ratios,
        "spec_ratios": spec_ratios,
        "frob_ratios": frob_ratios,
        "eff_ranks_orig": eranks,
        "eff_ranks_perm": eranks_p,
        "alphas_orig": alphas,
        "alphas_perm": alphas_p,
        "per_layer": results,
    }

    out_dir = Path(__file__).parent / "results" / "row_permutation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_matrix_svd.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    analyze_cross_matrix()
