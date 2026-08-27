"""Think-Gate Probe: Measure <think> token probability under different perturbation conditions.

Codex-recommended highest-ROI mechanism probe for NeurIPS paper.
Tests PGRMS claim: does perturbation raise the probability of entering reasoning mode?

Uses single forward pass (NOT generate) to extract logits at first decode position.
Compares <think> log-prob across: baseline, zero, mean, noise (1/2/3/8 tok), discrete tokens.

Fast: one forward pass per condition per task (~5 min total for 25 tasks × 8 conditions).
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.core.encoder import LLMEncoder


THINK_TOKEN_ID = 151667  # <think> in Qwen3 vocab


def build_prompt(tokenizer, query: str) -> str:
    """Build chat-formatted prompt (same as harness)."""
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


def get_first_token_logits(model, tokenizer, prompt: str,
                           soft_prompt: torch.Tensor | None = None,
                           device: torch.device = None) -> dict:
    """Do a single forward pass and extract logits at the first decode position.

    Returns dict with think_logit, think_log_prob, think_rank, top5_tokens, top5_probs.
    """
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
            use_cache=False,
        )

    # Logits at the LAST position = first decode token prediction
    logits = outputs.logits[0, -1, :].float()  # (vocab_size,)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)

    think_logit = logits[THINK_TOKEN_ID].item()
    think_log_prob = log_probs[THINK_TOKEN_ID].item()
    think_prob = probs[THINK_TOKEN_ID].item()

    # Rank of <think> token (1-indexed)
    sorted_indices = logits.argsort(descending=True)
    think_rank = (sorted_indices == THINK_TOKEN_ID).nonzero(as_tuple=True)[0].item() + 1

    # Top-5 tokens
    top5_ids = sorted_indices[:5].tolist()
    top5_probs = probs[sorted_indices[:5]].tolist()
    top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids]

    return {
        "think_logit": think_logit,
        "think_log_prob": think_log_prob,
        "think_prob": think_prob,
        "think_rank": think_rank,
        "top5_tokens": top5_tokens,
        "top5_ids": top5_ids,
        "top5_probs": top5_probs,
    }


def generate_tasks(n_tasks: int = 25, seed: int = 42):
    """Generate the same 25 nested arithmetic tasks used in other experiments."""
    sys.path.insert(0, str(Path(__file__).parent))
    from run_latent_sensitivity import generate_nested_tasks
    return generate_nested_tasks(
        n_tasks=n_tasks, seed=seed, difficulty_filter="sweet_spot"
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Think-gate probe")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--n-tasks", type=int, default=25)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("THINK-GATE PROBE")
    print("Measuring <think> token probability under perturbation")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    encoder = LLMEncoder(args.model, quantization="4bit")
    model = encoder.model
    tokenizer = encoder.tokenizer
    device = encoder._device

    # Calibrate RMS
    with torch.no_grad():
        embed_weight = model.get_input_embeddings().weight
        target_rms = embed_weight.float().square().mean().sqrt().item()
    embed_dim = embed_weight.shape[1]
    print(f"  Embed dim: {embed_dim}, Target RMS: {target_rms:.5f}")

    # Generate tasks
    tasks = generate_tasks(args.n_tasks)
    print(f"  Tasks: {len(tasks)}")

    # Define conditions
    noise_gen = torch.Generator().manual_seed(2024)

    def make_noise(n_tokens, rms=target_rms):
        sp = torch.randn(1, n_tokens, embed_dim, generator=noise_gen)
        current_rms = sp.square().mean().sqrt().clamp_min(1e-8)
        sp = sp * (rms / current_rms)
        return sp

    def make_discrete(char):
        token_ids = tokenizer.encode(char, add_special_tokens=False)
        tid = token_ids[0]
        emb = model.get_input_embeddings()(
            torch.tensor([[tid]], device=device)
        )
        return emb.float().cpu()

    conditions = {}

    # Baseline (no prefix)
    conditions["baseline"] = None

    # Zero embedding
    conditions["zero_2tok"] = torch.zeros(1, 2, embed_dim)

    # Mean embedding (2 tokens)
    with torch.no_grad():
        mean_emb = embed_weight.float().mean(dim=0, keepdim=True)
        mean_rms = mean_emb.square().mean().sqrt().clamp_min(1e-8)
        mean_emb = mean_emb * (target_rms / mean_rms)
    conditions["mean_2tok"] = mean_emb.unsqueeze(0).expand(1, 2, embed_dim).clone()

    # Random noise at different token counts
    for k in [1, 2, 3, 8]:
        conditions[f"noise_{k}tok"] = make_noise(k)

    # Discrete tokens (Shi et al.)
    for char in ["/", "?"]:
        for k in [2]:
            emb = make_discrete(char)
            conditions[f"disc_{char}_{k}tok"] = emb.expand(1, k, embed_dim).clone()

    print(f"\n  Conditions: {list(conditions.keys())}")

    # Run probe
    results = []
    for ti, task in enumerate(tasks):
        prompt = build_prompt(tokenizer, task.prompt)
        task_result = {
            "task_id": task.task_id,
            "prompt_preview": task.prompt[:80],
            "correct_answer": task.correct_answer,
            "conditions": {},
        }

        for cond_name, sp in conditions.items():
            t0 = time.time()
            info = get_first_token_logits(model, tokenizer, prompt, sp, device)
            elapsed = time.time() - t0
            info["elapsed"] = elapsed
            task_result["conditions"][cond_name] = info

        # Print summary for this task
        baseline_lp = task_result["conditions"]["baseline"]["think_log_prob"]
        noise2_lp = task_result["conditions"]["noise_2tok"]["think_log_prob"]
        baseline_rank = task_result["conditions"]["baseline"]["think_rank"]
        noise2_rank = task_result["conditions"]["noise_2tok"]["think_rank"]

        print(f"  [{ti+1}/{len(tasks)}] {task.task_id}: "
              f"base_rank={baseline_rank}, noise2_rank={noise2_rank}, "
              f"base_lp={baseline_lp:.2f}, noise2_lp={noise2_lp:.2f}, "
              f"delta_lp={noise2_lp - baseline_lp:+.2f}")

        results.append(task_result)
        gc.collect()

    # Aggregate stats
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    for cond_name in conditions:
        log_probs = [r["conditions"][cond_name]["think_log_prob"] for r in results]
        ranks = [r["conditions"][cond_name]["think_rank"] for r in results]
        probs = [r["conditions"][cond_name]["think_prob"] for r in results]
        rank1_count = sum(1 for r in ranks if r == 1)

        print(f"\n  {cond_name:20s}: "
              f"mean_lp={np.mean(log_probs):7.3f} +/- {np.std(log_probs):.3f}, "
              f"mean_rank={np.mean(ranks):6.1f}, "
              f"rank=1: {rank1_count}/{len(ranks)}, "
              f"mean_prob={np.mean(probs):.4f}")

    # Save
    output_path = args.output or str(
        Path(__file__).parent / "think_gate_probe_results.json"
    )
    output = {
        "experiment": "think_gate_probe",
        "model": args.model,
        "n_tasks": len(tasks),
        "think_token_id": THINK_TOKEN_ID,
        "target_rms": target_rms,
        "embed_dim": embed_dim,
        "conditions": list(conditions.keys()),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
