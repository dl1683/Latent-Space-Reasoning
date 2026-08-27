"""How much output diversity comes from numerical noise alone?

Batched and sequential greedy generation disagreed on 67% of checks while
running the capability-limited diversity study. Batching changes only the
reduction order inside matmuls -- the mathematical input is identical -- so if
that reroutes a *greedy* trajectory, the question of what embedding
perturbation actually contributes becomes sharp:

  If bit-level floating-point noise produces as much trajectory diversity as a
  deliberate 2-token embedding perturbation, then "perturbation" is not a
  semantic intervention on the latent space. It is a random number generator,
  and any diversity-based gain it shows would be reproduced by literally any
  perturbation of comparable or smaller magnitude -- including none at all.

Three conditions, all greedy (do_sample=False), all on the same task:

  A. REPEAT      -- the same single-sequence call run twice. Establishes that
                    the sequential path is deterministic at all.
  B. IDENTICAL   -- k byte-identical rows in one batch. Mathematically the same
                    input k times; any divergence is pure numerical noise.
  C. PERTURBED   -- k rows differing by the study's random embedding soft
                    prompts. This is the actual intervention.

Reported per condition: how many distinct completions, how many distinct final
answers, and how long the shared prefix is before the first divergence.

Usage::

    python probe_greedy_trajectory_stability.py --n-tasks 5 --k 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR.parent / "src"))

from harness import extract_answer  # noqa: E402
from run_capability_limited_diversity_study import (  # noqa: E402
    build_noise_soft_prompts,
    generate_batch,
)
from run_latent_sensitivity import (  # noqa: E402
    decode_with_raw_soft_prompt,
    generate_nested_tasks,
)
from latent_reasoning.core.encoder import LLMEncoder  # noqa: E402


def shared_prefix_len(texts: list[str]) -> int:
    """Characters common to every completion before the first divergence."""
    if not texts:
        return 0
    ref = texts[0]
    for i in range(len(ref)):
        ch = ref[i]
        if any(i >= len(t) or t[i] != ch for t in texts[1:]):
            return i
    return min(len(t) for t in texts)


def describe(texts: list[str], answers: list) -> dict:
    return {
        "n": len(texts),
        "distinct_completions": len(set(texts)),
        "distinct_answers": len({str(a) for a in answers}),
        "shared_prefix_chars": shared_prefix_len(texts),
        "answers": [a for a in answers],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--difficulty", default="wide_mult")
    p.add_argument("--n-tasks", type=int, default=5)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    tasks = generate_nested_tasks(
        n_tasks=args.n_tasks, difficulty_filter=args.difficulty)[:args.n_tasks]

    encoder = LLMEncoder(model_name=args.model, quantization="none", dtype="bfloat16")
    embed_dim = encoder.model.get_input_embeddings().embedding_dim
    rms = encoder.model.get_input_embeddings().weight.float().square().mean().sqrt().item()

    perturbed_sps = build_noise_soft_prompts(args.k, embed_dim, rms)
    # Condition B: one soft prompt, repeated. Every row is byte-identical.
    identical_sps = [perturbed_sps[0].clone() for _ in range(args.k)]

    records = []
    for ti, task in enumerate(tasks):
        print(f"\n=== task {ti + 1}/{len(tasks)}: {task.prompt.splitlines()[-1]} "
              f"(answer {task.correct_answer}) ===", flush=True)

        # A. determinism of the sequential path
        a1, _ = decode_with_raw_soft_prompt(
            encoder, perturbed_sps[0], task.prompt,
            max_new_tokens=args.max_new_tokens, enable_thinking=False)
        a2, _ = decode_with_raw_soft_prompt(
            encoder, perturbed_sps[0], task.prompt,
            max_new_tokens=args.max_new_tokens, enable_thinking=False)
        repeat = describe([a1, a2], [extract_answer(a1), extract_answer(a2)])

        # B. identical rows in one batch -> pure numerical noise
        outs_b = generate_batch(encoder, task.prompt, args.k, args.max_new_tokens,
                                soft_prompts=identical_sps)
        ident = describe([t for t, _ in outs_b],
                         [extract_answer(t) for t, _ in outs_b])

        # C. the actual intervention
        outs_c = generate_batch(encoder, task.prompt, args.k, args.max_new_tokens,
                                soft_prompts=perturbed_sps)
        pert = describe([t for t, _ in outs_c],
                        [extract_answer(t) for t, _ in outs_c])

        for name, d in (("A repeat(seq)", repeat), ("B identical(batch)", ident),
                        ("C perturbed(batch)", pert)):
            print(f"  {name:<20} distinct completions {d['distinct_completions']}/{d['n']}"
                  f"  distinct answers {d['distinct_answers']}"
                  f"  shared prefix {d['shared_prefix_chars']} chars", flush=True)

        records.append({
            "task_id": task.task_id,
            "correct_answer": task.correct_answer,
            "repeat_sequential": repeat,
            "identical_batch": ident,
            "perturbed_batch": pert,
        })

    out = Path(args.output) if args.output else (
        EXPERIMENTS_DIR / "greedy_trajectory_stability.json")
    out.write_text(json.dumps({"model": args.model, "difficulty": args.difficulty,
                               "k": args.k, "tasks": records}, indent=2))

    print("\n" + "=" * 74)
    print("SUMMARY (greedy decoding throughout)")
    print("=" * 74)
    for key, label in (("repeat_sequential", "A same call, twice"),
                       ("identical_batch", "B identical rows, one batch"),
                       ("perturbed_batch", "C perturbed rows, one batch")):
        dc = sum(r[key]["distinct_completions"] for r in records)
        da = sum(r[key]["distinct_answers"] for r in records)
        n = sum(r[key]["n"] for r in records)
        print(f"{label:<32} {dc} distinct completions of {n};"
              f" {da} distinct answers across {len(records)} tasks")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
