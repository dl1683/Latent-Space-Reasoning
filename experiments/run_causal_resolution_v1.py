"""
Causal resolution v1: Can we steer fact retrieval by swapping states at L25?

The logit-lens resolution shows L25 amplifies the queried fact. If this is
causally active (not just correlational), swapping the hidden state at L25
from a query-A run into a query-B run should change which fact the model
retrieves.

Design:
- Run model on "ZOG: big. MIP: small.\nZOG:" → expects "big"
- Run model on "ZOG: big. MIP: small.\nMIP:" → expects "small"
- At layer L, inject ZOG-query's hidden state into MIP-query's forward pass
- If the output flips from "small" to "big" at resolution layers but not
  at earlier layers, the resolution layer is causally determining retrieval.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from datetime import datetime

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "causal_resolution_v1")


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    return model, tok


CONFIGS = [
    {
        "name": "ZOG_MIP_bigsmall",
        "storage": "ZOG: big. MIP: small.",
        "query_a": "ZOG",
        "query_b": "MIP",
        "expected_a": "big",
        "expected_b": "small",
    },
    {
        "name": "PLIM_KROT_hotcold",
        "storage": "PLIM: hot. KROT: cold.",
        "query_a": "PLIM",
        "query_b": "KROT",
        "expected_a": "hot",
        "expected_b": "cold",
    },
    {
        "name": "HESK_VORN_redblue",
        "storage": "HESK: red. VORN: blue.",
        "query_a": "HESK",
        "query_b": "VORN",
        "expected_a": "red",
        "expected_b": "blue",
    },
]


def get_hidden_states(model, tok, prompt):
    """Capture hidden states at every layer."""
    states = {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    states[idx] = output[0].detach().clone()
                else:
                    states[idx] = output.detach().clone()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(i)))

    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids)
    for h in hooks:
        h.remove()
    return states, out


def run_with_injection(model, tok, prompt, donor_state, inject_layer):
    """Run the model but replace hidden state at inject_layer with donor_state."""
    injected = [False]

    def inject_hook(module, input, output):
        if not injected[0]:
            injected[0] = True
            if isinstance(output, tuple):
                return (donor_state,) + output[1:]
            return donor_state

    hook = model.model.layers[inject_layer].register_forward_hook(inject_hook)

    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids)
    hook.remove()

    logits = out.logits[0, -1]
    top5_idx = torch.topk(logits, 5).indices
    top5 = [tok.decode([int(t)]).strip() for t in top5_idx]
    return top5, logits


def run_config(model, tok, cfg):
    prompt_a = f"{cfg['storage']}\n{cfg['query_a']}:"
    prompt_b = f"{cfg['storage']}\n{cfg['query_b']}:"

    ids_a = tok(prompt_a, return_tensors="pt").input_ids
    ids_b = tok(prompt_b, return_tensors="pt").input_ids
    if ids_a.shape != ids_b.shape:
        print(f"  WARNING: different sequence lengths ({ids_a.shape[1]} vs {ids_b.shape[1]}), skipping")
        return None

    states_a, out_a = get_hidden_states(model, tok, prompt_a)
    states_b, out_b = get_hidden_states(model, tok, prompt_b)

    top5_a = [tok.decode([int(t)]).strip() for t in torch.topk(out_a.logits[0, -1], 5).indices]
    top5_b = [tok.decode([int(t)]).strip() for t in torch.topk(out_b.logits[0, -1], 5).indices]
    print(f"  Baseline A ({cfg['query_a']}): {top5_a[0]!r} (expected {cfg['expected_a']!r})")
    print(f"  Baseline B ({cfg['query_b']}): {top5_b[0]!r} (expected {cfg['expected_b']!r})")

    n_layers = model.config.num_hidden_layers
    layer_results = []

    for layer in range(n_layers):
        top5_injected, logits_inj = run_with_injection(
            model, tok, prompt_b, states_a[layer], layer
        )
        greedy = top5_injected[0]

        flipped_to_a = (greedy == cfg["expected_a"])
        stayed_b = (greedy == cfg["expected_b"])

        logits_inj_np = logits_inj.detach()
        val_a_tok = tok.encode(" " + cfg["expected_a"])[0]
        val_b_tok = tok.encode(" " + cfg["expected_b"])[0]
        logit_a = float(logits_inj_np[val_a_tok])
        logit_b = float(logits_inj_np[val_b_tok])

        layer_results.append({
            "layer": layer,
            "greedy": greedy,
            "flipped_to_a": flipped_to_a,
            "stayed_b": stayed_b,
            "logit_a": round(logit_a, 4),
            "logit_b": round(logit_b, 4),
            "logit_diff": round(logit_a - logit_b, 4),
            "top3": top5_injected[:3],
        })

    first_flip = None
    for lr in layer_results:
        if lr["flipped_to_a"]:
            first_flip = lr["layer"]
            break

    return {
        "config": cfg["name"],
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "baseline_a": top5_a[0],
        "baseline_b": top5_b[0],
        "injection_direction": f"A({cfg['query_a']})->B({cfg['query_b']})",
        "layers": layer_results,
        "first_flip_layer": first_flip,
        "n_flipped": sum(1 for lr in layer_results if lr["flipped_to_a"]),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tok = load_model()

    all_results = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===")
        result = run_config(model, tok, cfg)
        if result is None:
            continue
        all_results.append(result)

        print(f"\n  {'Layer':>5s} {'greedy':>8s} {'flip?':>6s} {'logit_A':>8s} {'logit_B':>8s} {'diff':>8s}")
        for lr in result["layers"]:
            L = lr["layer"]
            if L % 3 == 0 or L >= 18:
                flip = "FLIP" if lr["flipped_to_a"] else ""
                print(f"  {L:5d} {lr['greedy']:>8s} {flip:>6s} {lr['logit_a']:8.2f} {lr['logit_b']:8.2f} {lr['logit_diff']:+8.2f}")

        print(f"\n  First flip: L{result['first_flip_layer']}")
        print(f"  Total flipped: {result['n_flipped']}/{len(result['layers'])}")

    print("\n=== SUMMARY ===")
    for r in all_results:
        print(f"  {r['config']:30s} {r['injection_direction']:25s} "
              f"first_flip=L{r['first_flip_layer']} "
              f"flipped={r['n_flipped']}/{len(r['layers'])}")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "causal_resolution_v1",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_ID,
            "purpose": "Causal test: does injecting query-A state at layer L into query-B "
                       "forward pass flip the output from expected-B to expected-A?",
            "prediction": "Injection at L25 (resolution layer) should flip the output; "
                         "injection at early layers may not (resolution happens later)",
            "n_configs": len(all_results),
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
