"""Test whether silu nonlinearity amplifies row-permutation sensitivity.

Compares MLP output change under row permutation in three regimes:
1. Full SwiGLU: y = down @ (silu(gate @ x) * up @ x)
2. Linear-gated: y = down @ ((gate @ x) * up @ x)  [silu removed]
3. Pure bilinear: y = down @ (gate @ x + up @ x)    [no gating at all]

If the bilinear form is LESS sensitive, then silu's sparsity amplifies
pairing sensitivity. If equally sensitive, it's the 3-way tensor structure.

Model: SmolLM2-135M (CPU-only, uses hidden states from forward pass)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch


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
    model.eval()
    hidden_by_layer = {l: [] for l in range(n_layers)}
    with torch.no_grad():
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt")
            outputs = model(**tokens, output_hidden_states=True)
            for l in range(n_layers):
                hidden_by_layer[l].append(outputs.hidden_states[l][0, -1, :])
    del outputs
    print(f"  Got {len(prompts)} x {n_layers} hidden states")

    rng = np.random.default_rng(42)
    results = {}

    for l in range(n_layers):
        gate_w = model.model.layers[l].mlp.gate_proj.weight.detach()
        up_w = model.model.layers[l].mlp.up_proj.weight.detach()
        down_w = model.model.layers[l].mlp.down_proj.weight.detach()
        m = gate_w.shape[0]

        perm = torch.tensor(rng.permutation(m), dtype=torch.long)
        up_perm = up_w[perm]

        errs_swiglu = []
        errs_linear_gated = []
        errs_additive = []

        for x in hidden_by_layer[l]:
            g = gate_w @ x
            u_orig = up_w @ x
            u_perm = up_perm @ x

            # SwiGLU: silu(g) * u
            silu_g = g * torch.sigmoid(g)
            y_swiglu = down_w @ (silu_g * u_orig)
            y_swiglu_perm = down_w @ (silu_g * u_perm)

            # Linear-gated: g * u (no silu)
            y_linear = down_w @ (g * u_orig)
            y_linear_perm = down_w @ (g * u_perm)

            # Additive: g + u (no gating)
            y_add = down_w @ (g + u_orig)
            y_add_perm = down_w @ (g + u_perm)

            norm_sw = torch.norm(y_swiglu).item()
            norm_li = torch.norm(y_linear).item()
            norm_ad = torch.norm(y_add).item()

            if norm_sw > 1e-10:
                errs_swiglu.append(torch.norm(y_swiglu - y_swiglu_perm).item() / norm_sw)
            if norm_li > 1e-10:
                errs_linear_gated.append(torch.norm(y_linear - y_linear_perm).item() / norm_li)
            if norm_ad > 1e-10:
                errs_additive.append(torch.norm(y_add - y_add_perm).item() / norm_ad)

        results[f"L{l}"] = {
            "swiglu": float(np.mean(errs_swiglu)),
            "linear_gated": float(np.mean(errs_linear_gated)),
            "additive": float(np.mean(errs_additive)),
        }

        if l % 5 == 0 or l == n_layers - 1:
            r = results[f"L{l}"]
            print(
                f"  L{l:2d}: SwiGLU={r['swiglu']:.4f}, "
                f"linear_gated={r['linear_gated']:.4f}, "
                f"additive={r['additive']:.4f}"
            )

    print("\n" + "=" * 70)
    print("NONLINEARITY AMPLIFICATION TEST")
    print("=" * 70)

    sw = [results[f"L{l}"]["swiglu"] for l in range(n_layers)]
    lg = [results[f"L{l}"]["linear_gated"] for l in range(n_layers)]
    ad = [results[f"L{l}"]["additive"] for l in range(n_layers)]

    print(f"\nMean relative error from row permutation:")
    print(f"  SwiGLU (silu(g)*u):    {np.mean(sw):.4f}")
    print(f"  Linear gated (g*u):    {np.mean(lg):.4f}")
    print(f"  Additive (g+u):        {np.mean(ad):.4f}")
    print(f"\nRatio SwiGLU/linear:     {np.mean(sw)/np.mean(lg):.4f}")
    print(f"Ratio SwiGLU/additive:   {np.mean(sw)/np.mean(ad):.4f}")
    print(f"Ratio linear/additive:   {np.mean(lg)/np.mean(ad):.4f}")

    if np.mean(ad) < 0.01:
        print("\n>> Additive mode nearly invariant to row permutation (as expected)")
    if abs(np.mean(sw) - np.mean(lg)) / np.mean(sw) < 0.2:
        print(">> SwiGLU and linear-gated have SIMILAR sensitivity")
        print("   => The 3-way tensor structure drives sensitivity, not silu")
    elif np.mean(sw) > np.mean(lg) * 1.5:
        print(">> SwiGLU is MORE sensitive than linear-gated")
        print("   => silu AMPLIFIES pairing sensitivity through sparsity")
    else:
        print(">> Linear-gated is MORE sensitive than SwiGLU")
        print("   => silu DAMPENS pairing sensitivity")

    summary = {
        "model": "SmolLM2-135M",
        "n_layers": n_layers,
        "n_prompts": len(prompts),
        "mean_swiglu": float(np.mean(sw)),
        "mean_linear_gated": float(np.mean(lg)),
        "mean_additive": float(np.mean(ad)),
        "per_layer": results,
    }

    out_dir = Path(__file__).parent / "results" / "row_permutation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bilinear_vs_gated.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    run_test()
