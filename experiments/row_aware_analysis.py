"""Row-aware analysis of gate/up pairing in SwiGLU MLPs.

Computes per-neuron row-wise statistics that are sensitive to row permutation,
complementing the D_k analysis (which is blind to row pairing).

Model: SmolLM2-135M (CPU-only)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch


def analyze_row_pairing():
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

    results = {}

    for l in range(n_layers):
        gate_w = model.model.layers[l].mlp.gate_proj.weight.detach().numpy()
        up_w = model.model.layers[l].mlp.up_proj.weight.detach().numpy()
        down_w = model.model.layers[l].mlp.down_proj.weight.detach().numpy()

        # Row-wise cosine similarity: cos(g_i, u_i) for each neuron i
        gate_norms = np.linalg.norm(gate_w, axis=1)
        up_norms = np.linalg.norm(up_w, axis=1)
        row_dots = np.sum(gate_w * up_w, axis=1)
        row_cosines = row_dots / (gate_norms * up_norms + 1e-10)

        # Diagonal of W_gate @ W_up^T (unnormalized)
        diag_cross = row_dots

        # Row norm ratio: ||g_i|| / ||u_i||
        norm_ratios = gate_norms / (up_norms + 1e-10)

        # Down_proj column norms: ||d_i|| tells us how much neuron i contributes
        down_col_norms = np.linalg.norm(down_w, axis=0)

        # Weighted cosine: weight each neuron's cosine by its output contribution
        weighted_cos = np.average(row_cosines, weights=down_col_norms)

        # Cross-matrix diagonal dominance: |C_ii| / max_j |C_ij|
        # Only compute for a sample of neurons (full cross-matrix is m x m)
        sample_idx = np.linspace(0, m - 1, min(100, m), dtype=int)
        gate_sample = gate_w[sample_idx]
        cross_rows = gate_sample @ up_w.T  # (100, m)
        diag_vals = np.abs(np.array([cross_rows[k, sample_idx[k]] for k in range(len(sample_idx))]))
        row_maxes = np.max(np.abs(cross_rows), axis=1)
        diag_dominance = diag_vals / (row_maxes + 1e-10)

        results[f"L{l}"] = {
            "cos_mean": float(np.mean(row_cosines)),
            "cos_std": float(np.std(row_cosines)),
            "cos_median": float(np.median(row_cosines)),
            "cos_q10": float(np.percentile(row_cosines, 10)),
            "cos_q90": float(np.percentile(row_cosines, 90)),
            "cos_min": float(np.min(row_cosines)),
            "cos_max": float(np.max(row_cosines)),
            "cos_frac_positive": float(np.mean(row_cosines > 0)),
            "weighted_cos": weighted_cos,
            "norm_ratio_mean": float(np.mean(norm_ratios)),
            "norm_ratio_std": float(np.std(norm_ratios)),
            "diag_dominance_mean": float(np.mean(diag_dominance)),
            "diag_dominance_std": float(np.std(diag_dominance)),
            "down_col_norm_mean": float(np.mean(down_col_norms)),
            "down_col_norm_cv": float(np.std(down_col_norms) / np.mean(down_col_norms)),
        }

        if l % 5 == 0 or l == n_layers - 1:
            r = results[f"L{l}"]
            print(
                f"  L{l:2d}: cos={r['cos_mean']:+.4f} +/- {r['cos_std']:.4f}, "
                f"wcos={weighted_cos:+.4f}, "
                f"frac+={r['cos_frac_positive']:.3f}, "
                f"diag_dom={r['diag_dominance_mean']:.4f}"
            )

    print("\n" + "=" * 70)
    print("ROW-AWARE ANALYSIS SUMMARY")
    print("=" * 70)

    cos_means = [results[f"L{l}"]["cos_mean"] for l in range(n_layers)]
    wcos_means = [results[f"L{l}"]["weighted_cos"] for l in range(n_layers)]
    frac_pos = [results[f"L{l}"]["cos_frac_positive"] for l in range(n_layers)]
    diag_doms = [results[f"L{l}"]["diag_dominance_mean"] for l in range(n_layers)]

    print(f"\nRow-wise cosine similarity (gate_i, up_i):")
    print(f"  Mean across layers:     {np.mean(cos_means):+.4f}")
    print(f"  Range:                  {min(cos_means):+.4f} to {max(cos_means):+.4f}")
    print(f"  Fraction positive:      {np.mean(frac_pos):.3f}")
    print(f"  Weighted cosine range:  {min(wcos_means):+.4f} to {max(wcos_means):+.4f}")

    print(f"\nRandom baseline: mean cosine ~ 0, frac positive ~ 0.5")

    early = cos_means[:n_layers // 3]
    mid = cos_means[n_layers // 3:2 * n_layers // 3]
    late = cos_means[2 * n_layers // 3:]
    print(f"\nDepth profile:")
    print(f"  Early (L0-L{n_layers//3-1}):  cos = {np.mean(early):+.4f}")
    print(f"  Mid   (L{n_layers//3}-L{2*n_layers//3-1}): cos = {np.mean(mid):+.4f}")
    print(f"  Late  (L{2*n_layers//3}-L{n_layers-1}): cos = {np.mean(late):+.4f}")

    print(f"\nCross-matrix diagonal dominance:")
    print(f"  Mean: {np.mean(diag_doms):.4f}")
    print(f"  Range: {min(diag_doms):.4f} to {max(diag_doms):.4f}")
    print(f"  Random baseline: 1/sqrt(m) ~ {1/np.sqrt(m):.4f}")

    summary = {
        "model": "SmolLM2-135M",
        "d": d,
        "m": m,
        "n_layers": n_layers,
        "cos_means_all_layers": cos_means,
        "wcos_means_all_layers": wcos_means,
        "frac_positive_all_layers": frac_pos,
        "diag_dominance_all_layers": diag_doms,
        "overall_cos_mean": float(np.mean(cos_means)),
        "overall_frac_positive": float(np.mean(frac_pos)),
        "random_cos_baseline": 0.0,
        "random_frac_positive_baseline": 0.5,
        "per_layer": results,
    }

    out_dir = Path(__file__).parent / "results" / "row_permutation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "row_aware_analysis.json"
    def to_native(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=to_native)
    print(f"\nResults: {out_path}")

    return summary


if __name__ == "__main__":
    analyze_row_pairing()
