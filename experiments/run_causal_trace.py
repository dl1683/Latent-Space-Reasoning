"""Causal Tracing Probe: Identify which layers carry the perturbation effect.

Codex-recommended mechanism probe (E, reduced scope).
For each layer, run perturbed forward pass but restore baseline hidden state
at that layer. Measure change in <think> probability to identify the
causal bottleneck layer(s).

Method:
  1. Forward pass with NO prefix (baseline) → save all hidden states
  2. Forward pass WITH 2-tok noise prefix (perturbed) → get <think> prob
  3. For each layer L: perturbed forward, but hook layer L to swap in
     baseline residual stream → measure <think> prob
  4. Layer where restoration most reduces <think> prob = causal bottleneck

Uses hooks on transformer layers to patch residual stream mid-forward-pass.
Fast: one forward pass per layer per task (~2 min for 5 tasks × 36 layers).
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from latent_reasoning.core.encoder import LLMEncoder

THINK_TOKEN_ID = 151667


def build_prompt(tokenizer, query: str) -> str:
    """Build chat-formatted prompt."""
    system_msg = "Answer to the best of your ability."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


def forward_with_hidden(model, tokenizer, prompt, soft_prompt, device):
    """Forward pass returning logits and all hidden states."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(input_ids)

        if soft_prompt is not None:
            sp = soft_prompt.to(model.dtype).to(device)
            combined_embeds = torch.cat([sp, text_embeds], dim=1)
            soft_mask = torch.ones(
                1, sp.size(1), dtype=attention_mask.dtype, device=device
            )
            combined_mask = torch.cat([soft_mask, attention_mask], dim=1)
        else:
            combined_embeds = text_embeds
            combined_mask = attention_mask

        outputs = model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    return outputs


def get_think_prob(logits_last):
    """Extract <think> probability from logits at last position."""
    probs = F.softmax(logits_last.float(), dim=-1)
    return probs[THINK_TOKEN_ID].item()


def patched_forward(model, tokenizer, prompt, soft_prompt, device,
                    patch_layer, baseline_hidden, n_soft_tokens):
    """Run perturbed forward, but at patch_layer restore baseline residual.

    baseline_hidden: hidden state tensor from baseline forward at patch_layer+1
                     (index +1 because hidden_states[0] = embedding output).
    n_soft_tokens: number of soft prompt tokens prepended (for position alignment).
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    hook_handle = None

    def patch_hook(module, input, output):
        """Replace output residual at last position with baseline value."""
        # output is a tuple: (hidden_states, ...) for Qwen3 decoder layers
        hidden = output[0]  # (batch, seq_len, dim)
        # Get baseline hidden at the LAST position (no soft prefix in baseline)
        # In perturbed: last position = -1, in baseline: last position = -1
        baseline_val = baseline_hidden[0, -1, :].to(hidden.dtype).to(hidden.device)
        # Patch last position
        hidden = hidden.clone()
        hidden[0, -1, :] = baseline_val
        return (hidden,) + output[1:]

    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(input_ids)
        sp = soft_prompt.to(model.dtype).to(device)
        combined_embeds = torch.cat([sp, text_embeds], dim=1)
        soft_mask = torch.ones(
            1, sp.size(1), dtype=attention_mask.dtype, device=device
        )
        combined_mask = torch.cat([soft_mask, attention_mask], dim=1)

        # Register hook on the target layer
        hook_handle = model.model.layers[patch_layer].register_forward_hook(patch_hook)

        try:
            outputs = model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                use_cache=False,
            )
        finally:
            hook_handle.remove()

    return outputs.logits[0, -1, :]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Causal tracing probe (E)")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--n-tasks", type=int, default=5,
                        help="Number of tasks (default 5 for speed)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("CAUSAL TRACING PROBE")
    print("Which layers carry the perturbation effect?")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    encoder = LLMEncoder(args.model, quantization="4bit")
    model = encoder.model
    tokenizer = encoder.tokenizer
    device = encoder._device

    # Calibrate
    with torch.no_grad():
        embed_weight = model.get_input_embeddings().weight
        target_rms = embed_weight.float().square().mean().sqrt().item()
    embed_dim = embed_weight.shape[1]
    n_layers = len(model.model.layers)
    print(f"  Embed dim: {embed_dim}, Layers: {n_layers}, Target RMS: {target_rms:.5f}")

    # Generate tasks
    from run_latent_sensitivity import generate_nested_tasks
    tasks = generate_nested_tasks(n_tasks=args.n_tasks, seed=42,
                                  difficulty_filter="sweet_spot")
    print(f"  Tasks: {len(tasks)}")

    # Make 2-tok noise (the condition with largest effect)
    noise_gen = torch.Generator().manual_seed(2024)
    noise_2tok = torch.randn(1, 2, embed_dim, generator=noise_gen)
    current_rms = noise_2tok.square().mean().sqrt().clamp_min(1e-8)
    noise_2tok = noise_2tok * (target_rms / current_rms)

    all_results = []
    for ti, task in enumerate(tasks):
        prompt = build_prompt(tokenizer, task.prompt)
        print(f"\n  [{ti+1}/{len(tasks)}] {task.task_id}:")

        # 1. Baseline forward (no prefix)
        base_out = forward_with_hidden(model, tokenizer, prompt, None, device)
        base_think_prob = get_think_prob(base_out.logits[0, -1, :])
        base_hidden_states = [h.cpu() for h in base_out.hidden_states]
        print(f"    baseline <think> prob: {base_think_prob:.4f}")

        # 2. Perturbed forward (2-tok noise)
        pert_out = forward_with_hidden(model, tokenizer, prompt, noise_2tok, device)
        pert_think_prob = get_think_prob(pert_out.logits[0, -1, :])
        print(f"    perturbed <think> prob: {pert_think_prob:.4f}")
        print(f"    delta: {pert_think_prob - base_think_prob:+.4f}")

        del base_out, pert_out
        gc.collect()
        torch.cuda.empty_cache()

        # 3. Patched forward at each layer
        layer_effects = []
        for li in range(n_layers):
            t0 = time.time()
            # baseline_hidden at layer li+1 (hidden_states[0] = embeddings)
            patched_logits = patched_forward(
                model, tokenizer, prompt, noise_2tok, device,
                patch_layer=li,
                baseline_hidden=base_hidden_states[li + 1],
                n_soft_tokens=2,
            )
            patched_think_prob = get_think_prob(patched_logits)
            elapsed = time.time() - t0

            # Effect = how much patching REDUCES think prob vs full perturbation
            reduction = pert_think_prob - patched_think_prob
            layer_effects.append({
                "layer": li,
                "patched_think_prob": patched_think_prob,
                "reduction_from_pert": reduction,
                "restored_toward_base": (pert_think_prob - patched_think_prob) / max(pert_think_prob - base_think_prob, 1e-8),
                "elapsed": elapsed,
            })

            del patched_logits
            gc.collect()

        # Find most impactful layers
        sorted_by_reduction = sorted(layer_effects, key=lambda x: abs(x["reduction_from_pert"]), reverse=True)
        top3 = sorted_by_reduction[:3]
        print(f"    Top causal layers: "
              + ", ".join(f"L{l['layer']}({l['reduction_from_pert']:+.4f})" for l in top3))

        task_result = {
            "task_id": task.task_id,
            "correct_answer": task.correct_answer,
            "base_think_prob": base_think_prob,
            "pert_think_prob": pert_think_prob,
            "delta_think_prob": pert_think_prob - base_think_prob,
            "layer_effects": layer_effects,
        }
        all_results.append(task_result)

        del base_hidden_states
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate: mean effect per layer
    print("\n" + "=" * 60)
    print("AGGREGATE: Per-layer causal effect on <think> probability")
    print("=" * 60)

    for li in range(n_layers):
        reductions = [r["layer_effects"][li]["reduction_from_pert"] for r in all_results]
        restorations = [r["layer_effects"][li]["restored_toward_base"] for r in all_results]
        mean_red = np.mean(reductions)
        mean_rest = np.mean(restorations)

        if abs(mean_red) > 0.01 or abs(mean_rest) > 0.05:
            print(f"  Layer {li:2d}: mean_reduction={mean_red:+.4f}, "
                  f"mean_restoration={mean_rest:+.2%}")

    # Top 5 layers by aggregate effect
    agg_reductions = []
    for li in range(n_layers):
        reductions = [r["layer_effects"][li]["reduction_from_pert"] for r in all_results]
        agg_reductions.append((li, np.mean(reductions), np.std(reductions)))

    agg_reductions.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\n  Top 5 causal layers (by mean |reduction|):")
    for li, mean_r, std_r in agg_reductions[:5]:
        print(f"    Layer {li:2d}: {mean_r:+.4f} +/- {std_r:.4f}")

    # Save
    output_path = args.output or str(
        Path(__file__).parent / "causal_trace_results.json"
    )
    output = {
        "experiment": "causal_tracing",
        "model": args.model,
        "n_tasks": len(tasks),
        "n_layers": n_layers,
        "perturbation": "noise_2tok",
        "target_metric": "think_prob",
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
