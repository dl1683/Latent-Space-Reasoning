"""D-scaling test for mixed moments across the SmolLM2 scale ladder.

Tests the theorem's prediction: D1_norm ~ d^{2alpha-1} for shared
eigenvector structure in power-law frame operators.

Scale ladder: SmolLM2-135M (d=576), SmolLM2-360M (d=960), SmolLM2-1.7B (d=2048).
Same architecture family, different dimensions. CPU-only, no inference.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def normalized_d1(S_gate, S_up, d):
    """Compute scale-invariant D1 between two frame operators."""
    tau_gate = np.trace(S_gate) / d
    tau_up = np.trace(S_up) / d
    S_norm = S_gate / tau_gate
    T_norm = S_up / tau_up
    S_c = S_norm - np.eye(d)
    T_c = T_norm - np.eye(d)
    return float(np.trace(S_c @ T_c) / d)


def eigvec_overlap_top_k(S, T, k=4):
    """Top-k eigenvector subspace overlap (mean of top-k singular values)."""
    _, evecs_S = np.linalg.eigh(S)
    _, evecs_T = np.linalg.eigh(T)
    evecs_S = evecs_S[:, ::-1][:, :k]
    evecs_T = evecs_T[:, ::-1][:, :k]
    svs = np.linalg.svd(evecs_S.T @ evecs_T, compute_uv=False)
    return float(np.mean(svs))


def powerlaw_exponent(S, d):
    """Fit power-law exponent to eigenspectrum."""
    evals = np.sort(np.linalg.eigvalsh(S))[::-1]
    evals_pos = evals[evals > 0]
    log_k = np.log(np.arange(1, len(evals_pos) + 1))
    log_e = np.log(evals_pos)
    # Linear fit on log-log (tail: skip top 20%)
    n_skip = max(1, len(evals_pos) // 5)
    slope, _ = np.polyfit(log_k[n_skip:], log_e[n_skip:], 1)
    return -slope  # alpha (positive)


def analyze_scale(model_id, label):
    """Load model, compute D1_norm at each layer."""
    from transformers import AutoModelForCausalLM

    print(f"\nLoading {label}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map="cpu"
    )
    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded in {time.time()-t0:.1f}s (d={d}, layers={n_layers})")

    d1_norms = []
    overlaps = []
    alphas_gate = []
    alphas_up = []

    for l in range(n_layers):
        gate_w = model.model.layers[l].mlp.gate_proj.weight.detach().numpy()
        up_w = model.model.layers[l].mlp.up_proj.weight.detach().numpy()

        m = gate_w.shape[0]
        S_gate = gate_w.T @ gate_w / m
        S_up = up_w.T @ up_w / m

        d1 = normalized_d1(S_gate, S_up, d)
        ov = eigvec_overlap_top_k(S_gate, S_up, k=4)

        d1_norms.append(d1)
        overlaps.append(ov)

        # Compute alpha at select layers
        if l in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
            a_gate = powerlaw_exponent(S_gate, d)
            a_up = powerlaw_exponent(S_up, d)
            alphas_gate.append((l, a_gate))
            alphas_up.append((l, a_up))
            print(f"  L{l}: D1_norm={d1:+.4f}, overlap={ov:.4f}, alpha_gate={a_gate:.3f}, alpha_up={a_up:.3f}")
        elif l % 5 == 0:
            print(f"  L{l}: D1_norm={d1:+.4f}, overlap={ov:.4f}")

    # Summary stats
    early = d1_norms[: n_layers // 3]
    mid = d1_norms[n_layers // 3: 2 * n_layers // 3]
    late = d1_norms[2 * n_layers // 3:]

    result = {
        "model_id": model_id,
        "label": label,
        "d": d,
        "n_layers": n_layers,
        "d1_norm_all": d1_norms,
        "d1_norm_mean": float(np.mean(d1_norms)),
        "d1_norm_early": float(np.mean(early)),
        "d1_norm_mid": float(np.mean(mid)),
        "d1_norm_late": float(np.mean(late)),
        "d1_norm_max": float(max(d1_norms)),
        "overlap_mean": float(np.mean(overlaps)),
        "alphas_gate": alphas_gate,
        "alphas_up": alphas_up,
    }

    print(f"\n  Summary for {label} (d={d}):")
    print(f"    D1_norm: mean={np.mean(d1_norms):+.4f}, early={np.mean(early):+.4f}, mid={np.mean(mid):+.4f}, late={np.mean(late):+.4f}, max={max(d1_norms):+.4f}")
    print(f"    Overlap: mean={np.mean(overlaps):.4f}")

    del model
    return result


def main():
    models = [
        ("HuggingFaceTB/SmolLM2-135M", "SmolLM2-135M"),
        ("HuggingFaceTB/SmolLM2-360M", "SmolLM2-360M"),
        ("HuggingFaceTB/SmolLM2-1.7B", "SmolLM2-1.7B"),
    ]

    all_results = {}
    for model_id, label in models:
        result = analyze_scale(model_id, label)
        all_results[label] = result

    # D-scaling analysis
    print("\n" + "=" * 70)
    print("D-SCALING TEST: D1_norm vs model dimension")
    print("=" * 70)

    ds = []
    d1_means = []
    d1_lates = []
    d1_maxes = []

    for label, res in sorted(all_results.items(), key=lambda x: x[1]["d"]):
        d = res["d"]
        ds.append(d)
        d1_means.append(res["d1_norm_mean"])
        d1_lates.append(res["d1_norm_late"])
        d1_maxes.append(res["d1_norm_max"])
        print(f"{label:>16} (d={d:>5}): mean={res['d1_norm_mean']:+.4f}, late={res['d1_norm_late']:+.4f}, max={res['d1_norm_max']:+.4f}")

    # Fit power law: D1 = a * d^b
    if len(ds) >= 2:
        log_d = np.log(ds)
        for name, vals in [("mean", d1_means), ("late", d1_lates), ("max", d1_maxes)]:
            log_v = np.log(np.maximum(vals, 1e-10))
            if len(ds) >= 3:
                b, log_a = np.polyfit(log_d, log_v, 1)
            else:
                b = (log_v[1] - log_v[0]) / (log_d[1] - log_d[0])
            print(f"\n  D1_norm_{name} scaling exponent: {b:+.3f}")
            print(f"    Predicted (2*alpha-1 with alpha=0.8): +0.600")
            print(f"    Predicted (2*alpha-1 with alpha=0.6): +0.200")

    # Save
    out_dir = Path(__file__).parent / "results" / "frame_composition"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "d_scaling_test.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
