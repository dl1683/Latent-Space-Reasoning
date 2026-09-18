"""Row-permutation challenge: D_k is blind to row pairing, but SwiGLU is not.

Permuting up_proj rows preserves W_up^T W_up (and hence all D_k) exactly,
but changes which gate neuron gates which up neuron. This demonstrates that
Gram-operator-based invariants are insufficient for understanding MLP composition.

Prescribed by Droid adversarial review (entry 102).

Model: SmolLM2-135M (CPU-only, d=576, 30 layers)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch


def normalized_d1(S, T, d):
    tau_S = np.trace(S) / d
    tau_T = np.trace(T) / d
    S_c = S / tau_S - np.eye(d)
    T_c = T / tau_T - np.eye(d)
    return float(np.trace(S_c @ T_c) / d)


def mlp_forward(gate_w, up_w, down_w, x):
    """SwiGLU MLP forward: y = down @ (silu(gate @ x) * up @ x)"""
    gate_out = gate_w @ x
    gate_act = gate_out * torch.sigmoid(gate_out)  # silu
    up_out = up_w @ x
    hidden = gate_act * up_out
    return down_w @ hidden


def run_challenge():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading SmolLM2-135M...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", dtype=torch.float32, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded in {time.time()-t0:.1f}s (d={d}, layers={n_layers})")

    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "In quantum mechanics, the wave function",
        "2 + 2 = 4, therefore 3 + 3 =",
        "The quick brown fox jumps over the",
        "import numpy as np\nnp.array([1, 2, 3])",
        "According to Einstein's theory of",
        "Once upon a time in a land far",
        "The derivative of sin(x) is",
        "SELECT * FROM users WHERE",
    ]

    print("\nExtracting hidden states...")
    all_hidden = []
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt")
            outputs = model(**tokens, output_hidden_states=True)
            for layer_idx in range(n_layers):
                hs = outputs.hidden_states[layer_idx][0, -1, :]  # last token
                all_hidden.append((layer_idx, hs))
    del outputs

    hidden_by_layer = {}
    for layer_idx, hs in all_hidden:
        if layer_idx not in hidden_by_layer:
            hidden_by_layer[layer_idx] = []
        hidden_by_layer[layer_idx].append(hs)
    del all_hidden

    print(f"  Got hidden states: {len(prompts)} prompts x {n_layers} layers")

    rng = np.random.default_rng(42)
    results = {}

    for l in range(n_layers):
        gate_w = model.model.layers[l].mlp.gate_proj.weight.detach()
        up_w = model.model.layers[l].mlp.up_proj.weight.detach()
        down_w = model.model.layers[l].mlp.down_proj.weight.detach()
        m = gate_w.shape[0]

        perm = rng.permutation(m)
        perm_t = torch.tensor(perm, dtype=torch.long)

        up_perm = up_w[perm_t]
        gate_perm = gate_w[perm_t]
        down_ctrl = down_w[:, perm_t]

        S_gate = (gate_w.T @ gate_w / m).numpy()
        S_up_orig = (up_w.T @ up_w / m).numpy()
        S_up_perm = (up_perm.T @ up_perm / m).numpy()

        d1_before = normalized_d1(S_gate, S_up_orig, d)
        d1_after = normalized_d1(S_gate, S_up_perm, d)
        d1_diff = abs(d1_before - d1_after)

        gram_error = float(np.max(np.abs(S_up_orig - S_up_perm)))

        rel_errors_uponly = []
        rel_errors_ctrl = []

        for x in hidden_by_layer[l]:
            y_normal = mlp_forward(gate_w, up_w, down_w, x)
            y_up_perm = mlp_forward(gate_w, up_perm, down_w, x)
            y_ctrl = mlp_forward(gate_perm, up_perm, down_ctrl, x)

            norm_y = torch.norm(y_normal).item()
            if norm_y > 1e-10:
                rel_errors_uponly.append(
                    torch.norm(y_normal - y_up_perm).item() / norm_y
                )
                rel_errors_ctrl.append(
                    torch.norm(y_normal - y_ctrl).item() / norm_y
                )

        mean_err_uponly = float(np.mean(rel_errors_uponly))
        mean_err_ctrl = float(np.mean(rel_errors_ctrl))

        results[f"L{l}"] = {
            "d1_before": d1_before,
            "d1_after": d1_after,
            "d1_diff": d1_diff,
            "gram_max_error": gram_error,
            "rel_error_up_only": mean_err_uponly,
            "rel_error_control": mean_err_ctrl,
            "rel_errors_uponly_all": rel_errors_uponly,
            "rel_errors_ctrl_all": rel_errors_ctrl,
        }

        if l % 5 == 0 or l == n_layers - 1:
            print(
                f"  L{l:2d}: D1 diff={d1_diff:.2e}, "
                f"Gram err={gram_error:.2e}, "
                f"up-perm err={mean_err_uponly:.4f}, "
                f"ctrl err={mean_err_ctrl:.2e}"
            )

    print("\n" + "=" * 70)
    print("ROW-PERMUTATION CHALLENGE RESULTS")
    print("=" * 70)

    all_up_errs = [r["rel_error_up_only"] for r in results.values()]
    all_ctrl_errs = [r["rel_error_control"] for r in results.values()]
    all_d1_diffs = [r["d1_diff"] for r in results.values()]
    all_gram_errs = [r["gram_max_error"] for r in results.values()]

    print(f"\nP1 (up-perm changes output):  mean rel error = {np.mean(all_up_errs):.4f}")
    print(f"    Range: {min(all_up_errs):.4f} - {max(all_up_errs):.4f}")
    verdict_p1 = "CONFIRMED" if np.mean(all_up_errs) > 0.1 else "WEAK" if np.mean(all_up_errs) > 0.01 else "FAILED"
    print(f"    Verdict: {verdict_p1}")

    print(f"\nP2 (D_k preserved):  max D1 diff = {max(all_d1_diffs):.2e}")
    print(f"    Max Gram error = {max(all_gram_errs):.2e}")
    verdict_p2 = "CONFIRMED" if max(all_d1_diffs) < 1e-6 else "FAILED"
    print(f"    Verdict: {verdict_p2}")

    print(f"\nP3 (control preserves output): mean rel error = {np.mean(all_ctrl_errs):.2e}")
    print(f"    Range: {min(all_ctrl_errs):.2e} - {max(all_ctrl_errs):.2e}")
    verdict_p3 = "CONFIRMED" if np.mean(all_ctrl_errs) < 1e-5 else "FAILED"
    print(f"    Verdict: {verdict_p3}")

    ratio = np.mean(all_up_errs) / max(np.mean(all_ctrl_errs), 1e-15)
    print(f"\nSeparation ratio (up-perm / control): {ratio:.0f}x")

    summary = {
        "model": "SmolLM2-135M",
        "d": d,
        "n_layers": n_layers,
        "n_prompts": len(prompts),
        "seed": 42,
        "P1_up_perm_mean_error": float(np.mean(all_up_errs)),
        "P1_verdict": verdict_p1,
        "P2_max_d1_diff": float(max(all_d1_diffs)),
        "P2_max_gram_error": float(max(all_gram_errs)),
        "P2_verdict": verdict_p2,
        "P3_control_mean_error": float(np.mean(all_ctrl_errs)),
        "P3_verdict": verdict_p3,
        "separation_ratio": ratio,
        "per_layer": results,
    }

    out_dir = Path(__file__).parent / "results" / "row_permutation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "challenge_result.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults: {out_path}")

    return summary


if __name__ == "__main__":
    run_challenge()
