"""
Entropy structure v1: How does output distribution entropy evolve across layers?

Uses logit lens at each layer to get pseudo-distributions, then measures:
1. Shannon entropy of the distribution (bits) — how "decided" the model is
2. KL divergence between fact-worlds — how much the distribution discriminates
3. Top-k concentration — what fraction of probability mass is in the top-k tokens

If the resolution layer is where the model commits to a fact, entropy should
drop sharply there (the distribution narrows to the correct answer). This
connects the resolution layer finding to the distributional congruence failure:
entropy captures HOW MUCH distributional structure exists beyond the argmax.
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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "entropy_structure_v1")

CONFIGS = [
    {
        "name": "ZOG_MIP_bigsmall",
        "template": "ZOG: {v0}. MIP: {v1}.\nZOG:",
        "val0": ("big", "small"),
        "val1": ("hot", "cold"),
    },
    {
        "name": "PLIM_KROT_hotcold_queryB",
        "template": "PLIM: {v0}. KROT: {v1}.\nKROT:",
        "val0": ("hot", "cold"),
        "val1": ("red", "blue"),
    },
    {
        "name": "HESK_VORN_redblue_queryA",
        "template": "HESK: {v0}. VORN: {v1}.\nHESK:",
        "val0": ("red", "blue"),
        "val1": ("big", "small"),
    },
]


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    return model, tok


def get_all_layer_residuals(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    residuals = {}
    for i, hs in enumerate(out.hidden_states[1:]):
        residuals[i] = hs
    final_logits = out.logits[0, -1]
    return residuals, final_logits


def logit_lens_dist(model, hidden_state):
    normed = model.model.norm(hidden_state.unsqueeze(0).unsqueeze(0))
    logits = model.lm_head(normed)
    return F.softmax(logits[0, 0].detach(), dim=-1)


def shannon_entropy(p):
    eps = 1e-10
    return float(-(p * (p + eps).log2()).sum())


def top_k_mass(p, k=10):
    k = min(k, p.numel())
    topk = torch.topk(p, k).values
    return float(topk.sum())


def js_dist(p, q):
    m = (p + q) / 2
    eps = 1e-10
    jsd = (
        0.5 * ((p + eps) * ((p + eps) / (m + eps)).log()).sum()
        + 0.5 * ((q + eps) * ((q + eps) / (m + eps)).log()).sum()
    )
    return math.sqrt(max(0, float(jsd)))


def run_config(model, tok, cfg):
    n_layers = model.config.num_hidden_layers
    v0a, v0b = cfg["val0"]
    v1a, v1b = cfg["val1"]

    worlds = {
        "w00": cfg["template"].format(v0=v0a, v1=v1a),
        "w10": cfg["template"].format(v0=v0b, v1=v1a),
        "w01": cfg["template"].format(v0=v0a, v1=v1b),
        "w11": cfg["template"].format(v0=v0b, v1=v1b),
    }

    print(f"  Worlds:")
    for k, v in worlds.items():
        print(f"    {k}: {repr(v)}")

    world_residuals = {}
    world_final_logits = {}
    for wname, prompt in worlds.items():
        residuals, final_logits = get_all_layer_residuals(model, tok, prompt)
        world_residuals[wname] = residuals
        world_final_logits[wname] = final_logits

    layer_data = []
    for layer_idx in range(n_layers):
        dists = {}
        for wname in worlds:
            hidden = world_residuals[wname][layer_idx][0, -1]
            dists[wname] = logit_lens_dist(model, hidden)

        entropies = {w: shannon_entropy(d) for w, d in dists.items()}
        top10_masses = {w: top_k_mass(d, 10) for w, d in dists.items()}
        top1_masses = {w: top_k_mass(d, 1) for w, d in dists.items()}

        jsd_queried = js_dist(dists["w00"], dists["w10"])
        jsd_irrelevant = js_dist(dists["w00"], dists["w01"])
        jsd_ratio = jsd_queried / max(jsd_irrelevant, 1e-6)

        greedy_tokens = {w: torch.argmax(d).item() for w, d in dists.items()}
        greedy_words = {w: tok.decode([greedy_tokens[w]]).strip() for w in greedy_tokens}

        layer_data.append({
            "layer": layer_idx,
            "entropies": {w: round(e, 4) for w, e in entropies.items()},
            "avg_entropy": round(sum(entropies.values()) / len(entropies), 4),
            "top1_mass": {w: round(m, 6) for w, m in top1_masses.items()},
            "top10_mass": {w: round(m, 6) for w, m in top10_masses.items()},
            "jsd_queried": round(jsd_queried, 6),
            "jsd_irrelevant": round(jsd_irrelevant, 6),
            "jsd_ratio": round(jsd_ratio, 4),
            "greedy_tokens": greedy_words,
        })

    final_dists = {w: F.softmax(world_final_logits[w], dim=-1) for w in worlds}
    final_entropies = {w: shannon_entropy(d) for w, d in final_dists.items()}
    final_greedy = {w: tok.decode([torch.argmax(d).item()]).strip() for w, d in final_dists.items()}

    return {
        "config": cfg["name"],
        "worlds": {k: v for k, v in worlds.items()},
        "n_layers": n_layers,
        "layers": layer_data,
        "final_output": {
            "entropies": {w: round(e, 4) for w, e in final_entropies.items()},
            "greedy_tokens": final_greedy,
        },
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tok = load_model()

    all_results = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===")
        result = run_config(model, tok, cfg)
        all_results.append(result)

        print(f"\n  Layer-by-layer entropy + JSD:")
        print(f"  {'L':>3} {'avg_H':>8} {'jsd_q':>8} {'jsd_i':>8} {'ratio':>8} {'top1_w00':>10} {'greedy':>20}")
        for ld in result["layers"]:
            greedy_str = "/".join(ld["greedy_tokens"].values())
            print(f"  {ld['layer']:3d} {ld['avg_entropy']:8.2f} {ld['jsd_queried']:8.4f} {ld['jsd_irrelevant']:8.4f} {ld['jsd_ratio']:8.2f} {ld['top1_mass']['w00']:10.4f} {greedy_str:>20}")

        print(f"\n  Final output: {result['final_output']}")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "entropy_structure_v1",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_ID,
            "purpose": "Measure how output distribution entropy evolves across layers via logit lens. Connect resolution layer finding to distributional congruence failure.",
            "n_configs": len(all_results),
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
