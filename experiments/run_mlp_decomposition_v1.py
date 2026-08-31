"""
MLP decomposition v1: Where in the layer does resolution happen?

The attention control experiment showed resolution is NOT attention routing.
This experiment hooks INSIDE each layer to decompose the contribution:
- Post-attention state (residual + attention output, BEFORE MLP)
- Post-MLP state (residual + attention + MLP, the full layer output)

If JSD ratio spikes only after MLP (not after attention alone), the MLP is
the resolution mechanism.
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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "mlp_decomposition_v1")

CONFIGS = [
    {
        "name": "ZOG_MIP_bigsmall_queryA",
        "entity_a": "ZOG", "entity_b": "MIP",
        "val0": "big", "val1": "small",
        "query": "ZOG",
        "template": "{A}: {va}. {B}: {vb}.\n{Q}:",
    },
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
        MODEL_ID, dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
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
    return {
        "w00": fill(v0, v0),
        "w01": fill(v0, v1),
        "w10": fill(v1, v0),
        "w11": fill(v1, v1),
    }


def get_decomposed_states(model, tok, prompt):
    """Capture post-attention and post-MLP states at every layer."""
    post_attn_states = {}
    post_mlp_states = {}
    hooks = []

    for i, layer in enumerate(model.model.layers):
        def make_post_attn_hook(idx):
            def hook_fn(module, input, output):
                post_attn_states[idx] = input[0].detach().clone()
            return hook_fn

        def make_post_mlp_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    post_mlp_states[idx] = output[0].detach().clone()
                else:
                    post_mlp_states[idx] = output.detach().clone()
            return hook_fn

        hooks.append(layer.post_attention_layernorm.register_forward_hook(make_post_attn_hook(i)))
        hooks.append(layer.register_forward_hook(make_post_mlp_hook(i)))

    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()

    return post_attn_states, post_mlp_states


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

    all_post_attn = {}
    all_post_mlp = {}
    for wname, prompt in worlds.items():
        pa, pm = get_decomposed_states(model, tok, prompt)
        all_post_attn[wname] = pa
        all_post_mlp[wname] = pm

    layer_results = []
    for layer in range(n_layers):
        attn_dists = {}
        mlp_dists = {}
        for wname in worlds:
            pa_hs = all_post_attn[wname][layer][0, -1]
            pm_hs = all_post_mlp[wname][layer][0, -1]
            attn_dists[wname] = logit_lens_dist(model, pa_hs)
            mlp_dists[wname] = logit_lens_dist(model, pm_hs)

        d_A_attn = js_dist(attn_dists["w00"], attn_dists["w10"])
        d_B_attn = js_dist(attn_dists["w00"], attn_dists["w01"])
        d_A_mlp = js_dist(mlp_dists["w00"], mlp_dists["w10"])
        d_B_mlp = js_dist(mlp_dists["w00"], mlp_dists["w01"])

        if Q == A:
            q_attn, i_attn = d_A_attn, d_B_attn
            q_mlp, i_mlp = d_A_mlp, d_B_mlp
        else:
            q_attn, i_attn = d_B_attn, d_A_attn
            q_mlp, i_mlp = d_B_mlp, d_A_mlp

        ratio_attn = q_attn / (i_attn + 1e-6)
        ratio_mlp = q_mlp / (i_mlp + 1e-6)

        layer_results.append({
            "layer": layer,
            "post_attn_queried_jsd": round(q_attn, 6),
            "post_attn_irrelevant_jsd": round(i_attn, 6),
            "post_attn_ratio": round(ratio_attn, 4),
            "post_mlp_queried_jsd": round(q_mlp, 6),
            "post_mlp_irrelevant_jsd": round(i_mlp, 6),
            "post_mlp_ratio": round(ratio_mlp, 4),
            "mlp_boost": round(ratio_mlp - ratio_attn, 4),
        })

    best_attn = max(layer_results, key=lambda r: r["post_attn_ratio"])
    best_mlp = max(layer_results, key=lambda r: r["post_mlp_ratio"])

    return {
        "config": cfg["name"],
        "queried_entity": Q,
        "irrelevant_entity": B if Q == A else A,
        "layers": layer_results,
        "peak_post_attn": {"layer": best_attn["layer"], "ratio": best_attn["post_attn_ratio"]},
        "peak_post_mlp": {"layer": best_mlp["layer"], "ratio": best_mlp["post_mlp_ratio"]},
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tok = load_model()

    all_results = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===")
        result = run_config(model, tok, cfg)
        all_results.append(result)

        print(f"  {'Layer':>5s} {'attn_q':>8s} {'attn_i':>8s} {'attn_r':>7s}  |  {'mlp_q':>8s} {'mlp_i':>8s} {'mlp_r':>7s}  {'boost':>7s}")
        for lr in result["layers"]:
            L = lr["layer"]
            if L % 3 == 0 or L >= 20:
                print(f"  {L:5d} {lr['post_attn_queried_jsd']:8.4f} {lr['post_attn_irrelevant_jsd']:8.4f} {lr['post_attn_ratio']:7.2f}"
                      f"  |  {lr['post_mlp_queried_jsd']:8.4f} {lr['post_mlp_irrelevant_jsd']:8.4f} {lr['post_mlp_ratio']:7.2f}"
                      f"  {lr['mlp_boost']:+7.2f}")

        pa = result["peak_post_attn"]
        pm = result["peak_post_mlp"]
        print(f"\n  Peak post-attn: L{pa['layer']} ratio={pa['ratio']:.2f}")
        print(f"  Peak post-MLP:  L{pm['layer']} ratio={pm['ratio']:.2f}")

    print("\n=== SUMMARY ===")
    for r in all_results:
        pa, pm = r["peak_post_attn"], r["peak_post_mlp"]
        print(f"  {r['config']:40s} attn_peak@L{pa['layer']} r={pa['ratio']:6.2f}"
              f"  mlp_peak@L{pm['layer']} r={pm['ratio']:6.2f}")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "mlp_decomposition_v1",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_ID,
            "purpose": "Decompose resolution into attention vs MLP contribution",
            "prediction": "If MLP is the resolution mechanism, post-MLP ratio should "
                         "spike at L21-25 while post-attn ratio stays flat",
            "n_configs": len(all_results),
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
