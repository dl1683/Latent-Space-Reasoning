"""
Attention control for logit-lens resolution v1.

Tests the trivial confound: does the resolution layer (L21-25) simply reflect
attention routing? If the last token attends more to the queried entity's
value token at resolution layers, the JSD ratio might follow mechanically.

Measures: attention from the query position (last token) to each entity's
value token, across all layers and heads. Compares attention selectivity
ratio to JSD ratio from logit_lens_resolution_v1.
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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "attention_control_v1")

CONFIGS = [
    {
        "name": "ZOG_MIP_bigsmall_queryA",
        "entity_a": "ZOG", "entity_b": "MIP",
        "val0": "big", "val1": "small",
        "query": "ZOG",
        "template": "{A}: {va}. {B}: {vb}.\n{Q}:",
    },
    {
        "name": "PLIM_KROT_hotcold_queryA",
        "entity_a": "PLIM", "entity_b": "KROT",
        "val0": "hot", "val1": "cold",
        "query": "PLIM",
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

    return {
        "w00": fill(v0, v0),
        "w01": fill(v0, v1),
        "w10": fill(v1, v0),
        "w11": fill(v1, v1),
    }


def find_token_span(tok, prompt, target_str):
    """Find token positions covering target_str in prompt, handling multi-token spans."""
    ids = tok(prompt, return_tensors="pt").input_ids[0]
    decoded_tokens = [tok.decode([int(t)]) for t in ids]
    char_pos = 0
    token_char_ranges = []
    reconstructed = ""
    for i, dtok in enumerate(decoded_tokens):
        token_char_ranges.append((len(reconstructed), len(reconstructed) + len(dtok)))
        reconstructed += dtok

    positions = []
    start = 0
    while True:
        idx = reconstructed.lower().find(target_str.lower(), start)
        if idx == -1:
            break
        end = idx + len(target_str)
        for i, (cs, ce) in enumerate(token_char_ranges):
            if ce > idx and cs < end:
                positions.append(i)
        start = idx + 1
    return sorted(set(positions))


def get_attention_weights(model, tok, prompt):
    """Get attention weights from all layers, all heads."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    attentions = []
    for layer_attn in out.attentions:
        attentions.append(layer_attn[0].detach().cpu())
    return attentions, ids[0]


def analyze_attention(model, tok, cfg):
    """For one config, measure attention selectivity across worlds and layers."""
    worlds = make_worlds(cfg)
    Q = cfg["query"]
    A, B = cfg["entity_a"], cfg["entity_b"]

    queried_entity = Q
    irrelevant_entity = B if Q == A else A

    n_layers = model.config.num_hidden_layers

    attentions_by_world = {}
    for wname, prompt in worlds.items():
        attns, token_ids = get_attention_weights(model, tok, prompt)
        q_assignment = f"{queried_entity}:"
        i_assignment = f"{irrelevant_entity}:"
        q_span = find_token_span(tok, prompt, q_assignment)
        i_span = find_token_span(tok, prompt, i_assignment)
        # Only take the FIRST occurrence (the assignment, not the query)
        prompt_tokens = [tok.decode([int(t)]) for t in token_ids]
        mid = len(token_ids) // 2
        q_span_first = [p for p in q_span if p < mid]
        i_span_first = [p for p in i_span if p < mid]
        if not q_span_first:
            q_span_first = q_span[:2]
        if not i_span_first:
            i_span_first = i_span[:2]
        attentions_by_world[wname] = {
            "attns": attns,
            "q_span": q_span_first,
            "i_span": i_span_first,
            "n_tokens": len(token_ids),
        }

    all_layer_data = []
    for layer in range(n_layers):
        attn_to_queried_vals = []
        attn_to_irrelevant_vals = []

        for wname, wdata in attentions_by_world.items():
            layer_attn = wdata["attns"][layer]
            last_pos = wdata["n_tokens"] - 1
            mean_attn = layer_attn[:, last_pos, :].mean(dim=0)

            attn_to_q = sum(float(mean_attn[p]) for p in wdata["q_span"])
            attn_to_i = sum(float(mean_attn[p]) for p in wdata["i_span"])
            attn_to_queried_vals.append(attn_to_q)
            attn_to_irrelevant_vals.append(attn_to_i)

        mean_attn_queried = sum(attn_to_queried_vals) / len(attn_to_queried_vals)
        mean_attn_irrelevant = sum(attn_to_irrelevant_vals) / len(attn_to_irrelevant_vals)
        attn_ratio = mean_attn_queried / (mean_attn_irrelevant + 1e-10)

        all_layer_data.append({
            "layer": layer,
            "attn_to_queried": round(mean_attn_queried, 6),
            "attn_to_irrelevant": round(mean_attn_irrelevant, 6),
            "attn_ratio": round(attn_ratio, 4),
        })

    return {
        "config": cfg["name"],
        "queried_entity": queried_entity,
        "irrelevant_entity": irrelevant_entity,
        "layers": all_layer_data,
    }


def load_logit_lens_results():
    """Load JSD ratios from logit_lens_resolution_v1 for comparison."""
    path = os.path.join(os.path.dirname(__file__), "results",
                        "logit_lens_resolution_v1", "results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    jsd_by_config = {}
    for r in data["results"]:
        jsd_by_config[r["config"]] = {
            lr["layer"]: lr["ratio"] for lr in r["layers"]
        }
    return jsd_by_config


def compute_correlation(xs, ys):
    """Pearson correlation between two lists."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    return cov / (sx * sy)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tok = load_model()
    jsd_data = load_logit_lens_results()

    all_results = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===")
        result = analyze_attention(model, tok, cfg)
        all_results.append(result)

        jsd_ratios = jsd_data.get(cfg["name"], {})

        print(f"  {'Layer':>5s} {'attn_q':>8s} {'attn_i':>8s} {'attn_r':>7s} {'jsd_r':>7s}")
        attn_rs = []
        jsd_rs = []
        for lr in result["layers"]:
            L = lr["layer"]
            jsd_r = jsd_ratios.get(L, float('nan'))
            if L % 3 == 0 or L >= 20:
                jsd_str = f"{jsd_r:7.2f}" if not math.isnan(jsd_r) else "    N/A"
                print(f"  {L:5d} {lr['attn_to_queried']:8.4f} {lr['attn_to_irrelevant']:8.4f} "
                      f"{lr['attn_ratio']:7.2f} {jsd_str}")
            if not math.isnan(jsd_r):
                attn_rs.append(lr["attn_ratio"])
                jsd_rs.append(jsd_r)

        corr = compute_correlation(attn_rs, jsd_rs)
        result["correlation_with_jsd"] = round(corr, 4)
        print(f"\n  Pearson(attn_ratio, jsd_ratio) = {corr:.4f}")

        if corr > 0.7:
            print(f"  WARNING: High correlation — attention MAY explain JSD selectivity")
        elif corr < 0.3:
            print(f"  RESULT: Low correlation — resolution layer is NOT just attention routing")

    print("\n=== SUMMARY ===")
    for r in all_results:
        peak_layer = max(r["layers"], key=lambda x: x["attn_ratio"])
        print(f"  {r['config']:40s} peak_attn@L{peak_layer['layer']:2d} "
              f"attn_ratio={peak_layer['attn_ratio']:6.2f} "
              f"corr_with_jsd={r['correlation_with_jsd']:.4f}")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "attention_control_v1",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_ID,
            "purpose": "Test whether attention routing explains logit-lens resolution layer",
            "kill_condition": "If correlation(attn_ratio, jsd_ratio) > 0.7 across configs, "
                             "resolution layer reduces to attention patterns",
            "n_configs": len(all_results),
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
