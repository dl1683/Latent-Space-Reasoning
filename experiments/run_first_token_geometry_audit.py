"""First-token distribution-geometry audit for perturbation mechanisms.

This script measures how candidate prefix interventions move the next-token
distribution before full generation. It is intended as a cheap diagnostic pass:
if a perturbation wins downstream, these metrics help distinguish clean steering
from broad distribution drift or attention-sink disruption.

Example:
    python experiments/run_first_token_geometry_audit.py --model Qwen/Qwen3-0.6B --n-tasks 8
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from harness import auto_calibrate, generate_nested_expression_tasks

from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.decode.distribution_geometry import compare_logit_geometry
from latent_reasoning.decode.projection import (
    latent_to_soft_prompt,
    make_row_orthonormal_W,
)


def build_chat_prompt(encoder: LLMEncoder, query: str, *, enable_thinking: bool = True) -> str:
    """Build the same general chat prompt shape used by the experiment harness."""
    system_msg = "Answer to the best of your ability."
    if hasattr(encoder.tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ]
        try:
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if not enable_thinking:
                kwargs["enable_thinking"] = False
            return encoder.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            pass
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def first_token_logits(
    encoder: LLMEncoder,
    prompt: str,
    *,
    soft_prompt: Tensor | None = None,
) -> Tensor:
    """Return logits for the next generated token without sampling or generation."""
    inputs = encoder.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(encoder._device) for k, v in inputs.items()}

    with torch.no_grad():
        if soft_prompt is None:
            outputs = encoder.model(**inputs)
        else:
            sp = soft_prompt.to(device=encoder._device, dtype=encoder.model.dtype)
            text_embeds = encoder.model.get_input_embeddings()(inputs["input_ids"])
            combined_embeds = torch.cat([sp, text_embeds], dim=1)
            soft_mask = torch.ones(
                1,
                sp.size(1),
                dtype=inputs["attention_mask"].dtype,
                device=encoder._device,
            )
            combined_mask = torch.cat([soft_mask, inputs["attention_mask"]], dim=1)
            outputs = encoder.model(inputs_embeds=combined_embeds, attention_mask=combined_mask)

    return outputs.logits[:, -1, :].detach().float().cpu()


def random_soft_prompt(
    *,
    num_tokens: int,
    embed_dim: int,
    target_rms: float,
    seed: int,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    prompt = torch.randn(1, num_tokens, embed_dim, generator=generator, device=device)
    rms = prompt.square().mean().sqrt().clamp_min(1e-8)
    return prompt * (target_rms / rms)


def zero_soft_prompt(*, num_tokens: int, embed_dim: int, device: torch.device) -> Tensor:
    return torch.zeros(1, num_tokens, embed_dim, device=device)


def mean_embedding_soft_prompt(
    encoder: LLMEncoder,
    *,
    num_tokens: int,
    target_rms: float,
) -> Tensor:
    with torch.no_grad():
        mean_embedding = encoder.model.get_input_embeddings().weight.float().mean(dim=0)
        prompt = mean_embedding.view(1, 1, -1).repeat(1, num_tokens, 1)
        rms = prompt.square().mean().sqrt().clamp_min(1e-8)
        return (prompt * (target_rms / rms)).to(encoder._device)


def latent_projected_soft_prompt(
    *,
    latent_dim: int,
    embed_dim: int,
    num_tokens: int,
    target_rms: float,
    seed: int,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(latent_dim, generator=generator, device=device)
    projection = make_row_orthonormal_W(
        latent_dim,
        num_tokens * embed_dim,
        seed=seed + 1000,
    ).to(device)
    return latent_to_soft_prompt(
        latent,
        projection,
        curvature=0.5,
        embed_dim=embed_dim,
        num_tokens=num_tokens,
        target_rms=target_rms,
        use_logmap=False,
    )


def build_soft_prompts(
    encoder: LLMEncoder,
    *,
    modes: list[str],
    n_prefixes: int,
    num_tokens: int,
    embed_dim: int,
    target_rms: float,
    latent_dim: int,
    seed: int,
) -> dict[str, list[Tensor]]:
    prompts: dict[str, list[Tensor]] = {}
    for mode in modes:
        mode_prompts = []
        for idx in range(n_prefixes):
            prompt_seed = seed + idx * 7919
            if mode == "random_noise":
                sp = random_soft_prompt(
                    num_tokens=num_tokens,
                    embed_dim=embed_dim,
                    target_rms=target_rms,
                    seed=prompt_seed,
                    device=encoder._device,
                )
            elif mode == "zero_embedding":
                sp = zero_soft_prompt(
                    num_tokens=num_tokens,
                    embed_dim=embed_dim,
                    device=encoder._device,
                )
            elif mode == "mean_embedding":
                sp = mean_embedding_soft_prompt(
                    encoder,
                    num_tokens=num_tokens,
                    target_rms=target_rms,
                )
            elif mode == "latent_projected":
                sp = latent_projected_soft_prompt(
                    latent_dim=latent_dim,
                    embed_dim=embed_dim,
                    num_tokens=num_tokens,
                    target_rms=target_rms,
                    seed=prompt_seed,
                    device=encoder._device,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")
            mode_prompts.append(sp)
        prompts[mode] = mode_prompts
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--n-tasks", type=int, default=12)
    parser.add_argument("--n-prefixes", type=int, default=3)
    parser.add_argument("--num-soft-tokens", type=int, default=2)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", choices=["easy_nested", "sweet_spot"], default="sweet_spot")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["random_noise", "latent_projected", "zero_embedding", "mean_embedding"],
        choices=["random_noise", "latent_projected", "zero_embedding", "mean_embedding"],
    )
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/first_token_geometry_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading model...")
    encoder = LLMEncoder(model_name=args.model, quantization=args.quantization)
    cal = auto_calibrate(encoder)
    embed_dim = cal["embed_dim"]
    target_rms = cal["embedding_rms"]
    latent_dim = encoder.latent_dim

    _, tasks = generate_nested_expression_tasks(
        n_train=0,
        n_test=args.n_tasks,
        seed=args.seed,
        difficulty=args.difficulty,
    )
    soft_prompts = build_soft_prompts(
        encoder,
        modes=args.modes,
        n_prefixes=args.n_prefixes,
        num_tokens=args.num_soft_tokens,
        embed_dim=embed_dim,
        target_rms=target_rms,
        latent_dim=latent_dim,
        seed=args.seed,
    )

    per_task = []
    aggregate: dict[str, list[dict[str, float]]] = {mode: [] for mode in args.modes}

    for task_idx, task in enumerate(tasks, start=1):
        prompt = build_chat_prompt(encoder, task.prompt, enable_thinking=not args.no_think)
        baseline_logits = first_token_logits(encoder, prompt)
        task_record = {
            "task_id": task.task_id,
            "prompt": task.prompt,
            "correct_answer": task.correct_answer,
            "modes": {},
        }

        print(f"[{task_idx}/{len(tasks)}] {task.task_id}")
        for mode, mode_prompts in soft_prompts.items():
            prefix_records = []
            for prefix_idx, soft_prompt in enumerate(mode_prompts):
                candidate_logits = first_token_logits(encoder, prompt, soft_prompt=soft_prompt)
                metrics = compare_logit_geometry(
                    baseline_logits,
                    candidate_logits,
                    topk=args.topk,
                ).mean_dict()
                prefix_records.append({"prefix_index": prefix_idx, "geometry": metrics})
                aggregate[mode].append(metrics)
            task_record["modes"][mode] = prefix_records
        per_task.append(task_record)

    summary = {}
    for mode, rows in aggregate.items():
        if not rows:
            continue
        keys = rows[0].keys()
        summary[mode] = {
            key: sum(row[key] for row in rows) / len(rows)
            for key in keys
        }

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "quantization": args.quantization,
        "difficulty": args.difficulty,
        "n_tasks": len(tasks),
        "n_prefixes": args.n_prefixes,
        "num_soft_tokens": args.num_soft_tokens,
        "target_rms": target_rms,
        "topk": args.topk,
        "modes": args.modes,
        "summary": summary,
        "per_task": per_task,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved audit to {args.output}")


if __name__ == "__main__":
    main()
