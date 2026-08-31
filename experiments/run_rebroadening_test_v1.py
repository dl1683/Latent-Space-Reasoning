"""
Re-broadening test v1: Is the final output's distributional structure meaningful?

The commitment bottleneck (entropy_structure_v1) shows:
- L25: entropy 0.05-0.30 bits (near-certain)
- Final output: entropy 5.5-7.7 bits (re-broadened)

This experiment asks: does the re-broadened distribution carry MEANINGFUL
information about the history, or is it generic noise?

Test design:
1. Take same-place histories (same greedy answers, different presentation order)
2. Get their full output distributions
3. Check which tokens differ most between history variants
4. If the differences are in semantically meaningful tokens (related to the
   history content), the re-broadening carries information
5. If the differences are random/generic, it's noise

Also investigates the L27 vs final-output entropy discrepancy.
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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "rebroadening_test_v1")

ENTITIES = ["ZOG", "MIP", "PLIM"]
VALUES = {"ZOG": ("big", "small"), "MIP": ("hot", "cold"), "PLIM": ("red", "blue")}


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    return model, tok


def get_output_dist_and_hidden(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    final_dist = F.softmax(out.logits[0, -1], dim=-1)
    hidden_states = out.hidden_states
    return final_dist, hidden_states


def logit_lens_dist(model, hidden_state):
    normed = model.model.norm(hidden_state.unsqueeze(0).unsqueeze(0))
    logits = model.lm_head(normed)
    return F.softmax(logits[0, 0].detach(), dim=-1)


def shannon_entropy(p):
    eps = 1e-10
    return float(-(p * (p + eps).log2()).sum())


def js_dist(p, q):
    m = (p + q) / 2
    eps = 1e-10
    jsd = (
        0.5 * ((p + eps) * ((p + eps) / (m + eps)).log()).sum()
        + 0.5 * ((q + eps) * ((q + eps) / (m + eps)).log()).sum()
    )
    return math.sqrt(max(0, float(jsd)))


def top_divergent_tokens(p, q, tok, k=20):
    diff = (p - q).abs()
    topk = torch.topk(diff, k)
    results = []
    for i in range(k):
        idx = topk.indices[i].item()
        results.append({
            "token": tok.decode([idx]).strip(),
            "token_id": idx,
            "abs_diff": round(float(topk.values[i]), 6),
            "prob_p": round(float(p[idx]), 6),
            "prob_q": round(float(q[idx]), 6),
        })
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tok = load_model()
    n_layers = model.config.num_hidden_layers

    histories = {
        "std": "ZOG: big. MIP: hot. PLIM: red.",
        "rev": "PLIM: red. MIP: hot. ZOG: big.",
        "dup": "ZOG: big. MIP: hot. PLIM: red. ZOG: big.",
    }

    print("=== Part 1: L27 vs final output discrepancy ===")
    prompt = histories["std"] + "\nZOG:"
    final_dist, hidden_states = get_output_dist_and_hidden(model, tok, prompt)

    print(f"  Number of hidden states: {len(hidden_states)}")
    print(f"  Hidden state shapes: {hidden_states[0].shape}, {hidden_states[-1].shape}")

    last_layer_hidden = hidden_states[-1][0, -1]
    last_layer_dist = logit_lens_dist(model, last_layer_hidden)

    print(f"  Last hidden state entropy (logit lens): {shannon_entropy(last_layer_dist):.4f}")
    print(f"  Final output entropy:                   {shannon_entropy(final_dist):.4f}")
    print(f"  JSD(last_hidden_logitlens, final):       {js_dist(last_layer_dist, final_dist):.6f}")

    print(f"  Last hidden greedy: {tok.decode([torch.argmax(last_layer_dist).item()])}")
    print(f"  Final output greedy: {tok.decode([torch.argmax(final_dist).item()])}")

    second_to_last = hidden_states[-2][0, -1]
    second_to_last_dist = logit_lens_dist(model, second_to_last)
    print(f"  Second-to-last hidden entropy: {shannon_entropy(second_to_last_dist):.4f}")

    for i in [25, 26, 27, 28]:
        if i < len(hidden_states):
            h = hidden_states[i][0, -1]
            d = logit_lens_dist(model, h)
            print(f"  hidden_states[{i}] entropy: {shannon_entropy(d):.4f}, greedy: {tok.decode([torch.argmax(d).item()])}")

    print("\n=== Part 2: Distributional differences between same-place histories ===")
    query_entities = ["ZOG", "MIP", "PLIM"]
    all_results = []

    for qe in query_entities:
        print(f"\n  Query: {qe}")
        dists = {}
        for hname, history in histories.items():
            prompt = history + f"\n{qe}:"
            dist, _ = get_output_dist_and_hidden(model, tok, prompt)
            dists[hname] = dist
            greedy = tok.decode([torch.argmax(dist).item()]).strip()
            entropy = shannon_entropy(dist)
            print(f"    {hname}: greedy={greedy}, entropy={entropy:.4f}")

        pairs = [("std", "rev"), ("std", "dup"), ("rev", "dup")]
        pair_results = []
        for a, b in pairs:
            jsd = js_dist(dists[a], dists[b])
            top_tokens = top_divergent_tokens(dists[a], dists[b], tok, k=20)
            print(f"    JSD({a}, {b}) = {jsd:.6f}")
            print(f"      Top divergent tokens: {[t['token'] for t in top_tokens[:10]]}")
            pair_results.append({
                "pair": f"{a}-{b}",
                "jsd": round(jsd, 6),
                "top_divergent": top_tokens,
            })

        expected_answer = VALUES[qe][0]
        all_results.append({
            "query_entity": qe,
            "expected_answer": expected_answer,
            "entropies": {h: round(shannon_entropy(dists[h]), 4) for h in dists},
            "greedy_tokens": {h: tok.decode([torch.argmax(dists[h]).item()]).strip() for h in dists},
            "pairs": pair_results,
        })

    print("\n=== Part 3: Are the divergent tokens semantically related to history? ===")
    for r in all_results:
        print(f"\n  Query: {r['query_entity']} (expected: {r['expected_answer']})")
        for pr in r["pairs"]:
            print(f"    {pr['pair']}: JSD={pr['jsd']:.4f}")
            history_tokens = set()
            for e in ENTITIES:
                for v in VALUES[e]:
                    history_tokens.add(v.lower())
                history_tokens.add(e.lower())
            history_related = []
            other = []
            for td in pr["top_divergent"][:10]:
                if td["token"].lower() in history_tokens:
                    history_related.append(td["token"])
                else:
                    other.append(td["token"])
            print(f"      History-related: {history_related}")
            print(f"      Other: {other}")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "rebroadening_test_v1",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_ID,
            "purpose": "Test whether the re-broadened final distribution carries meaningful history information, and investigate L27 vs final-output entropy discrepancy",
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
