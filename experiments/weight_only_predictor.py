"""Weight-only tensor predictor: ||ΔT||_F from weights alone.

Derivation: for linearized gated MLP, the tensor T encodes the bilinear map.
||ΔT||_F^2 = tr(GG^T ⊙ D^TD ⊙ ΔUΔU^T)
where ⊙ is Hadamard (elementwise) product. No input x needed.

This tests whether weight structure ALONE predicts permutation damage,
or whether input-specific gating patterns are essential.

Model: SmolLM2-135M (CPU-only)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch


def partial_permutation(m, frac, rng):
    perm = np.arange(m)
    n_swap = max(2, int(m * frac))
    swap_idx = rng.choice(m, n_swap, replace=False)
    shuffled = rng.permutation(swap_idx)
    perm[swap_idx] = shuffled
    return perm


def run_test():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading SmolLM2-135M...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", dtype=torch.float32, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    m = model.config.intermediate_size
    print(f"  Loaded in {time.time()-t0:.1f}s (d={d}, m={m}, layers={n_layers})")

    test_prompts = [
        "Machine learning algorithms can",
        "The largest planet in our solar system",
        "for i in range(10):",
        "In the year 2025, technology",
        "The sum of angles in a triangle",
    ]

    print("\nExtracting hidden states...")
    model.eval()
    hidden_by_layer = {l: [] for l in range(n_layers)}
    with torch.no_grad():
        for prompt in test_prompts:
            tokens = tokenizer(prompt, return_tensors="pt")
            outputs = model(**tokens, output_hidden_states=True)
            for l in range(n_layers):
                hidden_by_layer[l].append(outputs.hidden_states[l][0, -1, :])
    del outputs
    print(f"  Got {len(test_prompts)} x {n_layers} hidden states")

    test_layers = [0, 7, 15, 22, 29]
    fracs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    n_perms = 10
    rng = np.random.default_rng(42)

    results = {}

    for l in test_layers:
        gate_w = model.model.layers[l].mlp.gate_proj.weight.detach().numpy()
        up_w = model.model.layers[l].mlp.up_proj.weight.detach().numpy()
        down_w = model.model.layers[l].mlp.down_proj.weight.detach().numpy()

        # Precompute Gram matrices for weight-only predictor
        GGT = gate_w @ gate_w.T  # m x m
        DTD = down_w.T @ down_w  # m x m

        records = []

        for frac in fracs:
            for trial in range(n_perms):
                perm = partial_permutation(m, frac, rng)
                up_perm = up_w[perm]
                delta_u = up_w - up_perm  # m x d

                # Weight-only tensor norm: ||ΔT||_F^2 = tr(GGT ⊙ DTD ⊙ ΔUΔU^T)
                DUUT = delta_u @ delta_u.T  # m x m
                tensor_norm_sq = float(np.sum(GGT * DTD * DUUT))
                tensor_norm = np.sqrt(max(0, tensor_norm_sq))

                # Weight-only DU norm: ||DΔU||_F
                du_norm = float(np.linalg.norm(down_w @ delta_u, 'fro'))

                # Actual errors on held-out prompts
                actual_errors = []
                tensor_input_errors = []
                for x_torch in hidden_by_layer[l]:
                    x = x_torch.numpy()
                    gx = gate_w @ x
                    ux = up_w @ x
                    ux_perm = up_perm @ x
                    silu_gx = gx * (1 / (1 + np.exp(-gx)))  # silu in numpy
                    y_orig = down_w @ (silu_gx * ux)
                    y_perm = down_w @ (silu_gx * ux_perm)
                    actual_errors.append(float(np.linalg.norm(y_orig - y_perm)))

                    # Input-conditioned tensor predictor (no silu)
                    tensor_input_errors.append(
                        float(np.linalg.norm(down_w @ (gx * (ux - ux_perm))))
                    )

                mean_actual = float(np.mean(actual_errors))
                mean_tensor_input = float(np.mean(tensor_input_errors))

                records.append({
                    "frac": frac,
                    "trial": trial,
                    "actual": mean_actual,
                    "weight_tensor": tensor_norm,
                    "weight_du": du_norm,
                    "input_tensor": mean_tensor_input,
                })

        actuals = [r["actual"] for r in records]
        w_tensors = [r["weight_tensor"] for r in records]
        w_dus = [r["weight_du"] for r in records]
        i_tensors = [r["input_tensor"] for r in records]

        def corr(a, b):
            a, b = np.array(a), np.array(b)
            if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        def r2_scaled(pred, actual):
            p, a = np.array(pred), np.array(actual)
            if np.std(p) < 1e-10:
                return 0.0
            slope = np.dot(p, a) / np.dot(p, p)
            scaled = p * slope
            ss_res = np.sum((a - scaled) ** 2)
            ss_tot = np.sum((a - np.mean(a)) ** 2)
            if ss_tot < 1e-10:
                return 0.0
            return float(1 - ss_res / ss_tot)

        results[f"L{l}"] = {
            "corr_weight_tensor": corr(w_tensors, actuals),
            "corr_weight_du": corr(w_dus, actuals),
            "corr_input_tensor": corr(i_tensors, actuals),
            "r2_weight_tensor": r2_scaled(w_tensors, actuals),
            "r2_weight_du": r2_scaled(w_dus, actuals),
            "r2_input_tensor": r2_scaled(i_tensors, actuals),
        }

        r = results[f"L{l}"]
        print(
            f"  L{l:2d}: "
            f"wt_tensor r={r['corr_weight_tensor']:.4f} R2={r['r2_weight_tensor']:.4f} | "
            f"wt_DU r={r['corr_weight_du']:.4f} R2={r['r2_weight_du']:.4f} | "
            f"in_tensor r={r['corr_input_tensor']:.4f} R2={r['r2_input_tensor']:.4f}"
        )

    print("\n" + "=" * 70)
    print("WEIGHT-ONLY vs INPUT-CONDITIONED PREDICTION")
    print("=" * 70)

    for metric, label in [
        ("r2_weight_tensor", "Weight-only tensor"),
        ("r2_weight_du", "Weight-only DU"),
        ("r2_input_tensor", "Input-conditioned tensor"),
    ]:
        vals = [results[f"L{l}"][metric] for l in test_layers]
        print(f"  {label:30s}: mean R2 = {np.mean(vals):.4f} ({min(vals):.4f} to {max(vals):.4f})")

    mean_wt = np.mean([results[f"L{l}"]["r2_weight_tensor"] for l in test_layers])
    mean_du = np.mean([results[f"L{l}"]["r2_weight_du"] for l in test_layers])
    mean_it = np.mean([results[f"L{l}"]["r2_input_tensor"] for l in test_layers])

    print(f"\n  Weight-only tensor vs DU improvement: {mean_wt - mean_du:+.4f}")
    print(f"  Input-conditioned vs weight-only improvement: {mean_it - mean_wt:+.4f}")

    if mean_wt > 0.5 and mean_wt > mean_du + 0.05:
        print(f"\n  >> WEIGHT-ONLY TENSOR WORKS: gate structure in weights predicts damage")
    elif mean_it > 0.5 and mean_it > mean_wt + 0.1:
        print(f"\n  >> INPUT-CONDITIONED NEEDED: weight-only is insufficient")
    else:
        print(f"\n  >> INCONCLUSIVE or DU suffices")

    summary = {
        "model": "SmolLM2-135M",
        "d": d, "m": m,
        "test_layers": test_layers,
        "n_test_prompts": len(test_prompts),
        "mean_r2_weight_tensor": mean_wt,
        "mean_r2_weight_du": mean_du,
        "mean_r2_input_tensor": mean_it,
        "per_layer": results,
    }

    out_dir = Path(__file__).parent / "results" / "row_permutation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "weight_only_predictor.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    run_test()
