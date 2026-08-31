"""
Fusion-Fission v1c: Attention pattern analysis at fused vs separate layers.

Question: WHY do facts fuse at some layers and separate at others?
Hypothesis: At fused layers, attention creates cross-talk between
fact-A and fact-B token positions.
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
DTYPE = torch.float32

SEPARATE_LAYERS = [5, 6, 8, 22, 23, 24]
FUSED_LAYERS = [2, 7, 13, 14, 18]


def make_prompt(nameA, valA, nameB, valB):
    return f"{nameA} is {valA}. {nameB} is {valB}. Remember these facts.\n"


def find_fact_positions(tok, prompt, nameA, nameB):
    """Use character offsets to find which tokens belong to each fact."""
    encoding = tok(prompt, return_offsets_mapping=True)
    ids = encoding.input_ids
    offsets = encoding.offset_mapping
    tokens = [tok.decode([t]) for t in ids]

    # Find character spans for each fact in the prompt
    # "HESK is red. VORN is blue. Remember these facts.\n"
    fact_a_str = f"{nameA} is"
    fact_b_str = f"{nameB} is"

    a_char_start = prompt.find(nameA)
    # End of fact A = the period after the first value
    first_period = prompt.find('.', a_char_start)
    a_char_end = first_period

    b_char_start = prompt.find(nameB)
    second_period = prompt.find('.', b_char_start)
    b_char_end = second_period

    a_pos = []
    b_pos = []

    for i, (start, end) in enumerate(offsets):
        if start is None or end is None:
            continue
        if start >= a_char_start and end <= a_char_end + 1:
            a_pos.append(i)
        elif start >= b_char_start and end <= b_char_end + 1:
            b_pos.append(i)

    return {
        "tokens": tokens,
        "a_positions": a_pos,
        "b_positions": b_pos,
    }


def analyze_attention_layer(attn_weights, a_pos, b_pos, seq_len):
    n_heads = attn_weights.shape[0]
    results = []

    for h in range(n_heads):
        w = attn_weights[h].numpy()

        a_attends_b = 0.0
        b_attends_a = 0.0
        a_attends_a = 0.0
        b_attends_b = 0.0

        for ai in a_pos:
            row = w[ai]
            a_attends_b += sum(row[bi] for bi in b_pos if bi <= ai)
            a_attends_a += sum(row[ai2] for ai2 in a_pos if ai2 <= ai)

        for bi in b_pos:
            row = w[bi]
            b_attends_a += sum(row[ai] for ai in a_pos if ai <= bi)
            b_attends_b += sum(row[bi2] for bi2 in b_pos if bi2 <= bi)

        n_a = max(len(a_pos), 1)
        n_b = max(len(b_pos), 1)

        results.append({
            "head": h,
            "a_to_b": float(a_attends_b / n_a),
            "b_to_a": float(b_attends_a / n_b),
            "a_self": float(a_attends_a / n_a),
            "b_self": float(b_attends_b / n_b),
            "cross": float((a_attends_b + b_attends_a) / (n_a + n_b)),
            "self": float((a_attends_a + b_attends_b) / (n_a + n_b)),
        })

    return results


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/results/fusion_fission_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Fusion-Fission v1c: Attention Analysis ===\n")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()

    prompt = make_prompt("HESK", "red", "VORN", "blue")
    print(f"Prompt: {repr(prompt)}")

    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    spans = find_fact_positions(tok, prompt, "HESK", "VORN")
    print(f"Tokens: {spans['tokens']}")
    print(f"A positions: {spans['a_positions']} = {[spans['tokens'][i] for i in spans['a_positions']]}")
    print(f"B positions: {spans['b_positions']} = {[spans['tokens'][i] for i in spans['b_positions']]}")

    if not spans['a_positions'] or not spans['b_positions']:
        print("ERROR: Could not find both fact positions. Check tokenizer offset_mapping support.")
        # Fallback: hardcode known tokenization
        # 'H'=0, 'ES'=1, 'K'=2, ' is'=3, ' red'=4, '.'=5
        # ' V'=6, 'ORN'=7, ' is'=8, ' blue'=9, '.'=10
        print("Using hardcoded positions for HESK/VORN prompt.")
        spans['a_positions'] = [0, 1, 2, 3, 4]
        spans['b_positions'] = [6, 7, 8, 9]
        print(f"A positions: {spans['a_positions']} = {[spans['tokens'][i] for i in spans['a_positions']]}")
        print(f"B positions: {spans['b_positions']} = {[spans['tokens'][i] for i in spans['b_positions']]}")

    with torch.no_grad():
        out = model(ids, output_attentions=True)

    n_layers = len(out.attentions)
    seq_len = ids.shape[1]
    print(f"Layers with attention: {n_layers}, Seq: {seq_len}\n")

    all_layer_results = []

    for layer in range(n_layers):
        attn = out.attentions[layer][0]
        heads = analyze_attention_layer(attn, spans["a_positions"], spans["b_positions"], seq_len)

        avg_cross = np.mean([h["cross"] for h in heads])
        avg_self = np.mean([h["self"] for h in heads])
        max_cross = max(h["cross"] for h in heads)
        max_cross_head = max(range(len(heads)), key=lambda i: heads[i]["cross"])
        ratio = avg_cross / avg_self if avg_self > 1e-8 else float('inf')

        # Also compute: how much does the LAST token attend to A vs B positions?
        last_tok_attn = out.attentions[layer][0, :, -1, :]  # [n_heads, seq]
        last_to_a = sum(last_tok_attn[:, ai].mean().item() for ai in spans["a_positions"])
        last_to_b = sum(last_tok_attn[:, bi].mean().item() for bi in spans["b_positions"])

        tag = ""
        if layer in FUSED_LAYERS: tag = " [FUSED]"
        elif layer in SEPARATE_LAYERS: tag = " [SEPARATE]"

        lr = {
            "layer": layer,
            "avg_cross": float(avg_cross),
            "avg_self": float(avg_self),
            "cross_self_ratio": float(ratio),
            "max_cross": float(max_cross),
            "max_cross_head": int(max_cross_head),
            "last_tok_to_a": float(last_to_a),
            "last_tok_to_b": float(last_to_b),
        }
        all_layer_results.append(lr)

        print(f"  Layer {layer:2d}{tag:12s}  "
              f"cross={avg_cross:.4f}  self={avg_self:.4f}  "
              f"ratio={ratio:.3f}  "
              f"last->A={last_to_a:.3f} last->B={last_to_b:.3f}")

    # Summary
    print("\n=== Summary: cross-fact attention at FUSED vs SEPARATE ===")
    fused_cross = [lr["avg_cross"] for lr in all_layer_results if lr["layer"] in FUSED_LAYERS]
    sep_cross = [lr["avg_cross"] for lr in all_layer_results if lr["layer"] in SEPARATE_LAYERS]
    fused_ratio = [lr["cross_self_ratio"] for lr in all_layer_results if lr["layer"] in FUSED_LAYERS]
    sep_ratio = [lr["cross_self_ratio"] for lr in all_layer_results if lr["layer"] in SEPARATE_LAYERS]
    fused_last_a = [lr["last_tok_to_a"] for lr in all_layer_results if lr["layer"] in FUSED_LAYERS]
    fused_last_b = [lr["last_tok_to_b"] for lr in all_layer_results if lr["layer"] in FUSED_LAYERS]
    sep_last_a = [lr["last_tok_to_a"] for lr in all_layer_results if lr["layer"] in SEPARATE_LAYERS]
    sep_last_b = [lr["last_tok_to_b"] for lr in all_layer_results if lr["layer"] in SEPARATE_LAYERS]

    print(f"  FUSED layers:    avg cross-fact = {np.mean(fused_cross):.4f}, ratio = {np.mean(fused_ratio):.3f}")
    print(f"  SEPARATE layers: avg cross-fact = {np.mean(sep_cross):.4f}, ratio = {np.mean(sep_ratio):.3f}")
    print(f"  FUSED layers:    last->A = {np.mean(fused_last_a):.3f}, last->B = {np.mean(fused_last_b):.3f}")
    print(f"  SEPARATE layers: last->A = {np.mean(sep_last_a):.3f}, last->B = {np.mean(sep_last_b):.3f}")

    if np.mean(fused_cross) > np.mean(sep_cross) * 1.3:
        verdict = "CROSS-FACT ATTENTION HIGHER AT FUSED LAYERS: fusion is attention-mediated"
    elif np.mean(sep_cross) > np.mean(fused_cross) * 1.3:
        verdict = "CROSS-FACT ATTENTION HIGHER AT SEPARATE LAYERS: counterintuitive"
    else:
        verdict = "NO CLEAR ATTENTION DIFFERENCE: fusion mechanism is not primarily attention cross-talk"

    print(f"\n  Verdict: {verdict}")

    out_path = out_dir / "fusion_fission_v1c_attention.json"
    with open(out_path, "w") as f:
        json.dump({
            "prompt": prompt,
            "a_positions": spans["a_positions"],
            "b_positions": spans["b_positions"],
            "tokens": spans["tokens"],
            "verdict": verdict,
            "layer_results": all_layer_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
