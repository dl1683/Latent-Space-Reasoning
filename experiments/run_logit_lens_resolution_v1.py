"""
Logit-Lens Resolution v1: Layer-dependent behavioral distance via logit lens.

At each layer, apply the model's final layernorm + unembedding to get a
pseudo-distribution over the vocabulary. Measure sqrt(JSD) between worlds
that differ in one fact. Compare the "queried" fact's distance to the
"irrelevant" fact's distance across layers.

Key finding: a resolution layer (21-25) where the model amplifies the
queried fact's behavioral signature while suppressing the irrelevant fact.
Cosine similarity stays flat (~0.98) throughout, blind to this structure.
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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "logit_lens_resolution_v1")


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    return model, tok


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


CONFIGS = [
    {
        "name": "ZOG_MIP_bigsmall_queryA",
        "entity_a": "ZOG", "entity_b": "MIP",
        "val0": "big", "val1": "small",
        "query": "ZOG",
        "template": "{A}: {va}. {B}: {vb}.\n{Q}:",
    },
    {
        "name": "ZOG_MIP_bigsmall_queryB",
        "entity_a": "ZOG", "entity_b": "MIP",
        "val0": "big", "val1": "small",
        "query": "MIP",
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
    {
        "name": "HESK_VORN_redblue_queryA_T3",
        "entity_a": "HESK", "entity_b": "VORN",
        "val0": "red", "val1": "blue",
        "query": "HESK",
        "template": "The color of {A} is {va}. The color of {B} is {vb}.\n{Q}:",
    },
]


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


def verify_baseline(model, tok, worlds, cfg):
    results = {}
    for name, prompt in worlds.items():
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        top5_idx = torch.topk(logits, 5).indices
        top5 = [tok.decode([int(t)]) for t in top5_idx]
        greedy = top5[0].strip()

        Q = cfg["query"]
        A, B = cfg["entity_a"], cfg["entity_b"]
        v0, v1 = cfg["val0"], cfg["val1"]
        if Q == A:
            expected = v0 if name in ("w00", "w01") else v1
        else:
            expected = v0 if name in ("w00", "w10") else v1

        results[name] = {
            "greedy": greedy,
            "expected": expected,
            "correct": greedy == expected,
            "top5": top5,
        }
    return results


def run_config(model, tok, cfg):
    worlds = make_worlds(cfg)
    n_layers = model.config.num_hidden_layers
    Q = cfg["query"]
    A, B = cfg["entity_a"], cfg["entity_b"]

    baseline = verify_baseline(model, tok, worlds, cfg)
    n_correct = sum(1 for v in baseline.values() if v["correct"])

    all_states = {}
    for name, prompt in worlds.items():
        all_states[name] = get_all_hidden_states(model, tok, prompt)

    layer_results = []
    for layer in range(n_layers):
        dists = {}
        for name in worlds:
            hs = all_states[name][layer][0, -1]
            dists[name] = logit_lens_dist(model, hs)

        cos_w00_w10 = float(F.cosine_similarity(
            all_states["w00"][layer][0, -1].unsqueeze(0),
            all_states["w10"][layer][0, -1].unsqueeze(0),
        ))

        d_A_diff = js_dist(dists["w00"], dists["w10"])
        d_B_diff = js_dist(dists["w00"], dists["w01"])
        d_both = js_dist(dists["w00"], dists["w11"])
        d_cross = js_dist(dists["w01"], dists["w10"])

        if Q == A:
            d_queried, d_irrel = d_A_diff, d_B_diff
        else:
            d_queried, d_irrel = d_B_diff, d_A_diff

        ratio = d_queried / (d_irrel + 1e-6)
        layer_results.append({
            "layer": layer,
            "queried_jsd": round(d_queried, 6),
            "irrelevant_jsd": round(d_irrel, 6),
            "ratio": round(ratio, 4),
            "cosine_w00_w10": round(cos_w00_w10, 6),
            "both_diff_jsd": round(d_both, 6),
            "cross_jsd": round(d_cross, 6),
        })

    final_dists = {name: all_states[name]["final"] for name in worlds}
    d_A_f = js_dist(final_dists["w00"], final_dists["w10"])
    d_B_f = js_dist(final_dists["w00"], final_dists["w01"])
    if Q == A:
        d_qf, d_if = d_A_f, d_B_f
    else:
        d_qf, d_if = d_B_f, d_A_f

    best = max(layer_results, key=lambda r: r["ratio"])

    return {
        "config": cfg["name"],
        "model": MODEL_ID,
        "queried_entity": Q,
        "irrelevant_entity": B if Q == A else A,
        "baseline": baseline,
        "baseline_correct": f"{n_correct}/4",
        "layers": layer_results,
        "final": {
            "queried_jsd": round(d_qf, 6),
            "irrelevant_jsd": round(d_if, 6),
            "ratio": round(d_qf / (d_if + 1e-6), 4),
        },
        "peak": {
            "layer": best["layer"],
            "ratio": best["ratio"],
            "queried_jsd": best["queried_jsd"],
            "irrelevant_jsd": best["irrelevant_jsd"],
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

        print(f"\n  {'Layer':>5s} {'queried':>8s} {'irrelev':>8s} {'ratio':>7s} {'cosine':>8s}")
        for lr in result["layers"]:
            L = lr["layer"]
            if L % 3 == 0 or L >= 20:
                print(f"  {L:5d} {lr['queried_jsd']:8.4f} {lr['irrelevant_jsd']:8.4f} {lr['ratio']:7.2f} {lr['cosine_w00_w10']:8.6f}")
        f = result["final"]
        print(f"  final {f['queried_jsd']:8.4f} {f['irrelevant_jsd']:8.4f} {f['ratio']:7.2f}")

        p = result["peak"]
        print(f"\n  Peak: layer {p['layer']}, ratio={p['ratio']:.2f}, "
              f"queried={p['queried_jsd']:.4f}, irrel={p['irrelevant_jsd']:.4f}")

    print("\n=== SUMMARY ===")
    for r in all_results:
        p = r["peak"]
        print(f"  {r['config']:40s} peak@L{p['layer']:2d} ratio={p['ratio']:7.2f} "
              f"q={p['queried_jsd']:.4f} i={p['irrelevant_jsd']:.4f} "
              f"baseline={r['baseline_correct']}")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "logit_lens_resolution_v1",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_ID,
            "n_configs": len(all_results),
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
