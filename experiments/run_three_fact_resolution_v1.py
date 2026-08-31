"""
Three-fact resolution v1: Does the resolution layer generalize to 3+ facts?

With entities A, B, C, the model must suppress BOTH irrelevant facts when
querying one. If resolution generalizes, the queried fact's JSD spikes at
L21-25 while BOTH irrelevant facts are suppressed. If it's just pairwise
competition, the pattern may break with three competing facts.
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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "three_fact_resolution_v1")


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    return model, tok


CONFIGS = [
    {
        "name": "ZOG_MIP_PLIM_queryA",
        "entities": ["ZOG", "MIP", "PLIM"],
        "vals": [("big", "small"), ("hot", "cold"), ("red", "blue")],
        "query": "ZOG",
        "template": "{E0}: {v0}. {E1}: {v1}. {E2}: {v2}.\n{Q}:",
    },
    {
        "name": "ZOG_MIP_PLIM_queryB",
        "entities": ["ZOG", "MIP", "PLIM"],
        "vals": [("big", "small"), ("hot", "cold"), ("red", "blue")],
        "query": "MIP",
        "template": "{E0}: {v0}. {E1}: {v1}. {E2}: {v2}.\n{Q}:",
    },
    {
        "name": "ZOG_MIP_PLIM_queryC",
        "entities": ["ZOG", "MIP", "PLIM"],
        "vals": [("big", "small"), ("hot", "cold"), ("red", "blue")],
        "query": "PLIM",
        "template": "{E0}: {v0}. {E1}: {v1}. {E2}: {v2}.\n{Q}:",
    },
    {
        "name": "HESK_VORN_KROT_queryA",
        "entities": ["HESK", "VORN", "KROT"],
        "vals": [("big", "small"), ("hot", "cold"), ("red", "blue")],
        "query": "HESK",
        "template": "{E0}: {v0}. {E1}: {v1}. {E2}: {v2}.\n{Q}:",
    },
]


def make_worlds_3fact(cfg):
    """Generate 8 worlds (2^3 combinations) for 3 entities."""
    t = cfg["template"]
    entities = cfg["entities"]
    vals = cfg["vals"]
    Q = cfg["query"]
    worlds = {}
    for a in range(2):
        for b in range(2):
            for c in range(2):
                key = f"w{a}{b}{c}"
                prompt = t.format(
                    E0=entities[0], v0=vals[0][a],
                    E1=entities[1], v1=vals[1][b],
                    E2=entities[2], v2=vals[2][c],
                    Q=Q,
                )
                worlds[key] = prompt
    return worlds


def get_all_hidden_states(model, tok, prompt):
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

    states["final"] = F.softmax(out.logits[0, -1], dim=-1)
    return states


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


def verify_baseline(model, tok, worlds, cfg):
    """Check whether the model can retrieve each entity's value."""
    query_idx = cfg["entities"].index(cfg["query"])
    results = {}
    for wname, prompt in list(worlds.items())[:4]:
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        top5_idx = torch.topk(logits, 5).indices
        top5 = [tok.decode([int(t)]).strip() for t in top5_idx]
        expected_val_idx = int(wname[1 + query_idx])
        expected = cfg["vals"][query_idx][expected_val_idx]
        results[wname] = {
            "greedy": top5[0],
            "expected": expected,
            "correct": top5[0] == expected,
            "top5": top5,
        }
    return results


def run_config(model, tok, cfg):
    worlds = make_worlds_3fact(cfg)
    n_layers = model.config.num_hidden_layers
    query_idx = cfg["entities"].index(cfg["query"])

    baseline = verify_baseline(model, tok, worlds, cfg)
    n_correct = sum(1 for v in baseline.values() if v["correct"])

    all_states = {}
    for wname, prompt in worlds.items():
        all_states[wname] = get_all_hidden_states(model, tok, prompt)

    ref_world = "w000"

    layer_results = []
    for layer in range(n_layers):
        dists = {}
        for wname in worlds:
            hs = all_states[wname][layer][0, -1]
            dists[wname] = logit_lens_dist(model, hs)

        entity_jsd = []
        for e_idx in range(3):
            flip_worlds = []
            for wname in worlds:
                bits = list(wname[1:])
                if bits[e_idx] == '0':
                    bits[e_idx] = '1'
                    partner = 'w' + ''.join(bits)
                    flip_worlds.append((wname, partner))

            avg_jsd = 0.0
            count = 0
            for w1, w2 in flip_worlds:
                if w1 in dists and w2 in dists:
                    avg_jsd += js_dist(dists[w1], dists[w2])
                    count += 1
            avg_jsd /= max(count, 1)
            entity_jsd.append(avg_jsd)

        q_jsd = entity_jsd[query_idx]
        irrel_jsds = [entity_jsd[i] for i in range(3) if i != query_idx]

        layer_results.append({
            "layer": layer,
            "queried_jsd": round(q_jsd, 6),
            "irrelevant1_jsd": round(irrel_jsds[0], 6),
            "irrelevant2_jsd": round(irrel_jsds[1], 6),
            "ratio_vs_irrel1": round(q_jsd / (irrel_jsds[0] + 1e-6), 4),
            "ratio_vs_irrel2": round(q_jsd / (irrel_jsds[1] + 1e-6), 4),
            "ratio_vs_max_irrel": round(q_jsd / (max(irrel_jsds) + 1e-6), 4),
        })

    best = max(layer_results, key=lambda r: r["ratio_vs_max_irrel"])

    return {
        "config": cfg["name"],
        "queried_entity": cfg["query"],
        "irrelevant_entities": [e for i, e in enumerate(cfg["entities"]) if i != query_idx],
        "baseline": baseline,
        "baseline_correct": f"{n_correct}/4",
        "layers": layer_results,
        "peak": {
            "layer": best["layer"],
            "ratio_vs_max_irrel": best["ratio_vs_max_irrel"],
            "queried_jsd": best["queried_jsd"],
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

        bl = result["baseline"]
        print(f"  Baseline: {result['baseline_correct']} correct")
        for w, info in bl.items():
            mark = "OK" if info["correct"] else "FAIL"
            print(f"    {w}: greedy={info['greedy']!r} expected={info['expected']!r} [{mark}]")

        print(f"\n  {'Layer':>5s} {'q_jsd':>8s} {'i1_jsd':>8s} {'i2_jsd':>8s} {'r_i1':>7s} {'r_i2':>7s} {'r_max':>7s}")
        for lr in result["layers"]:
            L = lr["layer"]
            if L % 3 == 0 or L >= 20:
                print(f"  {L:5d} {lr['queried_jsd']:8.4f} {lr['irrelevant1_jsd']:8.4f} "
                      f"{lr['irrelevant2_jsd']:8.4f} {lr['ratio_vs_irrel1']:7.2f} "
                      f"{lr['ratio_vs_irrel2']:7.2f} {lr['ratio_vs_max_irrel']:7.2f}")

        p = result["peak"]
        print(f"\n  Peak: L{p['layer']}, ratio_vs_max_irrel={p['ratio_vs_max_irrel']:.2f}, q_jsd={p['queried_jsd']:.4f}")

    print("\n=== SUMMARY ===")
    for r in all_results:
        p = r["peak"]
        print(f"  {r['config']:40s} peak@L{p['layer']:2d} ratio={p['ratio_vs_max_irrel']:7.2f} "
              f"q_jsd={p['queried_jsd']:.4f} baseline={r['baseline_correct']}")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "three_fact_resolution_v1",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_ID,
            "purpose": "Test whether resolution layer generalizes to 3-fact worlds",
            "prediction": "If resolution is a general mechanism, queried JSD should spike "
                         "at L21-25 while BOTH irrelevant facts are suppressed",
            "n_configs": len(all_results),
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
