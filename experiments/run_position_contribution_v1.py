"""
Position contribution v1: Which positions carry the resolution signal?

The resolution layer (L25) amplifies the queried fact in the value space.
This experiment decomposes the attention output by position: for each
source position, compute the attention-weighted value contribution to the
last token, apply logit lens, measure JSD between fact-worlds.

This tells us: is the resolution signal coming from the queried entity's
position, the irrelevant entity's position, or distributed?
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import json
import os
from datetime import datetime

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "position_contribution_v1")

CONFIGS = [
    {
        "name": "PLIM_KROT_hotcold_queryB",
        "entity_a": "PLIM", "entity_b": "KROT",
        "val0": "hot", "val1": "cold",
        "query": "KROT",
        "template": "{A}: {va}. {B}: {vb}.\n{Q}:",
    },
    {
        "name": "HESK_VORN_redblue_queryA",
        "entity_a": "HESK", "entity_b": "VORN",
        "val0": "red", "val1": "blue",
        "query": "HESK",
        "template": "{A}: {va}. {B}: {vb}.\n{Q}:",
    },
]


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map=DEVICE, trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tok


def make_worlds(cfg):
    t = cfg["template"]
    A, B = cfg["entity_a"], cfg["entity_b"]
    v0, v1 = cfg["val0"], cfg["val1"]
    Q = cfg["query"]
    def fill(va, vb):
        return t.format(A=A, B=B, va=va, vb=vb, Q=Q)
    return {"w00": fill(v0, v0), "w10": fill(v1, v0)}


def find_token_span(tok, prompt, target_str):
    ids = tok(prompt, return_tensors="pt").input_ids[0]
    decoded = [tok.decode([int(t)]) for t in ids]
    recon = ""
    ranges = []
    for dtok in decoded:
        ranges.append((len(recon), len(recon) + len(dtok)))
        recon += dtok
    positions = []
    idx = recon.lower().find(target_str.lower())
    if idx >= 0:
        end = idx + len(target_str)
        for i, (cs, ce) in enumerate(ranges):
            if ce > idx and cs < end:
                positions.append(i)
    return positions


def get_per_position_value_contributions(model, tok, prompt, target_layers):
    """For target layers, capture per-position value contribution to last token."""
    results = {}
    hooks = []

    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        def make_hook(l_idx):
            def hook_fn(module, input, output):
                attn_output, attn_weights = output[0], output[1]
                results[l_idx] = {
                    "attn_weights": attn_weights.detach().clone(),
                    "attn_output": attn_output.detach().clone(),
                }
            return hook_fn

        hooks.append(attn.register_forward_hook(make_hook(layer_idx)))

    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids, output_attentions=True)

    for h in hooks:
        h.remove()

    per_pos = {}
    for l_idx in target_layers:
        attn_w = out.attentions[l_idx][0]
        n_heads, seq_len, _ = attn_w.shape
        last_pos = seq_len - 1

        layer = model.model.layers[l_idx]
        normed = layer.input_layernorm(
            get_residual_at_layer(model, tok, prompt, l_idx)
        )

        head_dim = model.config.hidden_size // model.config.num_attention_heads
        v_proj = layer.self_attn.v_proj(normed)
        v_heads = v_proj[0].view(seq_len, n_heads, head_dim).transpose(0, 1)

        per_pos[l_idx] = {}
        for pos in range(seq_len):
            weight_to_pos = attn_w[:, last_pos, pos]
            weighted_v = (weight_to_pos.unsqueeze(-1) * v_heads[:, pos, :]).sum(dim=0)
            per_pos[l_idx][pos] = weighted_v.detach()

    return per_pos


def get_residual_at_layer(model, tok, prompt, target_layer):
    """Get the residual stream (input) at target_layer."""
    result = {}
    def hook_fn(module, input, output):
        result[0] = input[0].detach().clone()

    hook = model.model.layers[target_layer].input_layernorm.register_forward_hook(hook_fn)
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        model(ids)
    hook.remove()
    return result[0]


def logit_lens_dist(model, hidden_state):
    normed = model.model.norm(hidden_state.unsqueeze(0).unsqueeze(0))
    logits = model.lm_head(normed)
    return F.softmax(logits[0, 0].detach(), dim=-1)


def js_dist(p, q):
    m = (p + q) / 2
    eps = 1e-10
    jsd = (
        0.5 * ((p + eps) * ((p + eps) / (m + eps)).log()).sum()
        + 0.5 * ((q + eps) * ((q + eps) / (m + eps)).log()).sum()
    )
    return math.sqrt(max(0, float(jsd)))


def run_config(model, tok, cfg):
    worlds = make_worlds(cfg)
    Q = cfg["query"]
    A, B = cfg["entity_a"], cfg["entity_b"]
    n_layers = model.config.num_hidden_layers

    target_layers = [10, 15, 20, 21, 22, 23, 24, 25, 26, 27]

    prompt_w00 = worlds["w00"]
    prompt_w10 = worlds["w10"]

    ids = tok(prompt_w00, return_tensors="pt").input_ids[0]
    tokens = [tok.decode([int(t)]) for t in ids]
    seq_len = len(tokens)

    q_span = find_token_span(tok, prompt_w00, Q)
    a_span = find_token_span(tok, prompt_w00, A)
    b_span = find_token_span(tok, prompt_w00, B)

    print(f"  Tokens: {tokens}")
    print(f"  Query span ({Q}): {q_span}")
    print(f"  Entity A span ({A}): {a_span}")
    print(f"  Entity B span ({B}): {b_span}")

    per_pos_w00 = get_per_position_value_contributions(model, tok, prompt_w00, target_layers)
    per_pos_w10 = get_per_position_value_contributions(model, tok, prompt_w10, target_layers)

    layer_results = []
    for layer in target_layers:
        pos_data = []
        for pos in range(seq_len):
            v_w00 = per_pos_w00[layer][pos]
            v_w10 = per_pos_w10[layer][pos]

            d_w00 = logit_lens_dist(model, v_w00)
            d_w10 = logit_lens_dist(model, v_w10)
            jsd = js_dist(d_w00, d_w10)

            label = ""
            if pos in a_span[:2]:
                label = f"entity_A({A})"
            elif pos in b_span[:2]:
                label = f"entity_B({B})"
            elif pos in q_span:
                label = f"query({Q})"
            else:
                label = f"pos{pos}({tokens[pos].strip()})"

            pos_data.append({
                "position": pos,
                "token": tokens[pos],
                "label": label,
                "jsd_between_worlds": round(jsd, 6),
            })

        layer_results.append({
            "layer": layer,
            "positions": pos_data,
        })

    return {
        "config": cfg["name"],
        "queried_entity": Q,
        "tokens": tokens,
        "query_span": q_span,
        "entity_a_span": a_span,
        "entity_b_span": b_span,
        "layers": layer_results,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tok = load_model()

    all_results = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===")
        result = run_config(model, tok, cfg)
        all_results.append(result)

        for lr in result["layers"]:
            print(f"\n  Layer {lr['layer']}:")
            for pd in lr["positions"]:
                bar = "#" * int(pd["jsd_between_worlds"] * 50)
                print(f"    pos {pd['position']:2d} {pd['label']:25s} jsd={pd['jsd_between_worlds']:.4f} {bar}")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "position_contribution_v1",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_ID,
            "purpose": "Decompose resolution by position: which tokens' value contributions carry the resolution signal?",
            "n_configs": len(all_results),
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
