"""Compute mixed moments D_k for real model frame operators.

Tests whether adjacent MLP components (gate_proj / up_proj) and adjacent
layers share eigenvector structure. If D1 >> 0, the standard practice
of analyzing components independently misses essential information.

CPU-only. Loads pretrained weights, computes frame operators, eigenvectors,
and mixed moments. No inference, no GPU.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def mixed_moments_numpy(S, T, k_max=3):
    """Mixed moments D_k(S,T) = (1/d) tr[(S-τI)(T-τI)]^k."""
    d = S.shape[0]
    tau_S = np.trace(S) / d
    tau_T = np.trace(T) / d
    S_c = S - tau_S * np.eye(d)
    T_c = T - tau_T * np.eye(d)
    M = S_c @ T_c
    moments = {}
    for k in range(1, k_max + 1):
        if k == 1:
            moments[k] = float(np.trace(M) / d)
        else:
            Mk = np.linalg.matrix_power(M, k)
            moments[k] = float(np.trace(Mk) / d)
    return moments


def eigenvector_overlap(U1, U2, top_k=None):
    """Compute principal angle cosines between top-k eigenvectors."""
    if top_k:
        U1 = U1[:, :top_k]
        U2 = U2[:, :top_k]
    overlap = U1.T @ U2
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    return singular_values


def frame_operator(W):
    """Compute frame operator S = WW^T / m, where W is (d, m) or (m, d)."""
    if W.shape[0] > W.shape[1]:
        # W is (m, d), typical for nn.Linear weight shapes
        S = W.T @ W / W.shape[0]
    else:
        # W is (d, m)
        S = W @ W.T / W.shape[1]
    return S


def analyze_model(model_id, label):
    """Load model, compute mixed moments between frame operators."""
    from transformers import AutoModelForCausalLM

    print(f"\nLoading {label} ({model_id})...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="cpu"
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  d={d_model}, n_layers={n_layers}")

    results = {
        "model_id": model_id,
        "label": label,
        "d_model": d_model,
        "n_layers": n_layers,
        "within_layer": [],  # D1(gate, up) per layer
        "across_layers_gate": [],  # D1(gate_l, gate_{l+1})
        "across_layers_up": [],  # D1(up_l, up_{l+1})
        "cross_component_across": [],  # D1(gate_l, up_{l+1})
    }

    # Extract all frame operators and eigenvectors
    gates = []
    ups = []

    for l in range(n_layers):
        # Get weight matrices
        try:
            gate_w = model.model.layers[l].mlp.gate_proj.weight.detach().numpy()
            up_w = model.model.layers[l].mlp.up_proj.weight.detach().numpy()
        except AttributeError:
            try:
                # Phi-style: fc1/fc2
                gate_w = model.model.layers[l].mlp.fc1.weight.detach().numpy()
                up_w = model.model.layers[l].mlp.fc2.weight.detach().numpy()
            except AttributeError:
                print(f"  Skipping layer {l}: unknown MLP structure")
                continue

        S_gate = frame_operator(gate_w)
        S_up = frame_operator(up_w)
        gates.append(S_gate)
        ups.append(S_up)

    print(f"  Extracted {len(gates)} layers")

    # 1. Within-layer: D1(gate, up) at each layer
    print("  Computing within-layer mixed moments...")
    for l in range(len(gates)):
        mm = mixed_moments_numpy(gates[l], ups[l], k_max=3)
        # Also compute eigenvector overlap
        _, evecs_gate = np.linalg.eigh(gates[l])
        _, evecs_up = np.linalg.eigh(ups[l])
        evecs_gate = evecs_gate[:, ::-1]  # descending order
        evecs_up = evecs_up[:, ::-1]
        overlap_cos = eigenvector_overlap(evecs_gate, evecs_up, top_k=min(32, d_model))
        top_overlap = float(np.mean(overlap_cos[:4]))

        results["within_layer"].append({
            "layer": l,
            "D1": mm[1],
            "D2": mm[2],
            "D3": mm[3],
            "top4_eigvec_overlap": top_overlap,
        })

    # 2. Across layers: D1(gate_l, gate_{l+1})
    print("  Computing across-layer mixed moments...")
    for l in range(len(gates) - 1):
        mm_gate = mixed_moments_numpy(gates[l], gates[l + 1], k_max=3)
        mm_up = mixed_moments_numpy(ups[l], ups[l + 1], k_max=3)
        mm_cross = mixed_moments_numpy(gates[l], ups[l + 1], k_max=3)

        results["across_layers_gate"].append({
            "layers": [l, l + 1],
            "D1": mm_gate[1],
            "D2": mm_gate[2],
            "D3": mm_gate[3],
        })
        results["across_layers_up"].append({
            "layers": [l, l + 1],
            "D1": mm_up[1],
            "D2": mm_up[2],
            "D3": mm_up[3],
        })
        results["cross_component_across"].append({
            "layers": [l, l + 1],
            "D1": mm_cross[1],
            "D2": mm_cross[2],
            "D3": mm_cross[3],
        })

    # Print summary
    print(f"\n  Within-layer D1(gate, up):")
    print(f"  {'Layer':<8} {'D1':>12} {'D2':>14} {'top4 overlap':>14}")
    print(f"  {'-'*48}")
    for entry in results["within_layer"]:
        print(
            f"  {entry['layer']:<8} {entry['D1']:>+12.4f} "
            f"{entry['D2']:>+14.4f} {entry['top4_eigvec_overlap']:>14.4f}"
        )

    d1_within = [e["D1"] for e in results["within_layer"]]
    print(f"\n  D1 range: [{min(d1_within):+.4f}, {max(d1_within):+.4f}]")
    print(f"  D1 mean:  {np.mean(d1_within):+.4f}")

    d1_across_gate = [e["D1"] for e in results["across_layers_gate"]]
    print(f"\n  Across-layer D1 (gate-gate): [{min(d1_across_gate):+.4f}, {max(d1_across_gate):+.4f}], mean={np.mean(d1_across_gate):+.4f}")

    d1_across_up = [e["D1"] for e in results["across_layers_up"]]
    print(f"  Across-layer D1 (up-up):     [{min(d1_across_up):+.4f}, {max(d1_across_up):+.4f}], mean={np.mean(d1_across_up):+.4f}")

    # Comparison with synthetic predictions
    alpha = 0.8  # from entry 89
    predicted_threshold = 0  # D1 > 0 means shared structure; D1 ≈ 0 means independent
    n_shared = sum(1 for d1 in d1_within if d1 > 0.1)
    print(f"\n  Layers with D1 > 0.1 (within): {n_shared}/{len(d1_within)}")

    # Delete model to free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    models = [
        ("HuggingFaceTB/SmolLM2-135M", "SmolLM2-135M"),
        ("Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
    ]

    all_results = {}
    for model_id, label in models:
        try:
            results = analyze_model(model_id, label)
            all_results[label] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    out_dir = Path(__file__).parent / "results" / "frame_composition"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "real_model_mixed_moments.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("REAL MODEL MIXED MOMENTS: GATE-UP EIGENVECTOR ALIGNMENT")
    print("=" * 70)
    for label, res in all_results.items():
        d = res["d_model"]
        d1_within = [e["D1"] for e in res["within_layer"]]
        d1_across = [e["D1"] for e in res["across_layers_gate"]]
        print(f"\n{label} (d={d}):")
        print(f"  Within-layer D1(gate,up): mean={np.mean(d1_within):+.4f}, range=[{min(d1_within):+.4f}, {max(d1_within):+.4f}]")
        print(f"  Across-layer D1(gate,gate): mean={np.mean(d1_across):+.4f}, range=[{min(d1_across):+.4f}, {max(d1_across):+.4f}]")

        # Compare with synthetic prediction
        # For shared frames at this d with alpha=0.8:
        # D1 ~ (1-0.8)^2 * zeta(1.6) * d^(0.6) ≈ 0.04 * 3.19 * d^0.6
        predicted_if_fully_shared = 0.128 * (d ** 0.6)
        print(f"  Synthetic prediction if fully shared (alpha=0.8): D1 ≈ {predicted_if_fully_shared:.1f}")
        mean_d1 = np.mean(d1_within)
        if predicted_if_fully_shared > 0:
            effective_sharing = mean_d1 / predicted_if_fully_shared
            print(f"  Effective sharing fraction: {effective_sharing:.4f}")


if __name__ == "__main__":
    main()
