"""Latent Sensitivity Test: Do different random latents produce different accuracy?

Codex-recommended first experiment. Answers THE core question:
"Is the latent-to-quality landscape exploitable?"

If accuracy varies across random latents -> landscape is exploitable -> evolution can help.
If accuracy is stable -> landscape is flat -> evolution is pointless.

Design:
- Generate random multi-step arithmetic tasks (varying difficulty)
- Run zero-shot baseline (no conditioning) for reference
- Test N random latents with soft prompt conditioning on same tasks
- Measure accuracy variance across latents
- Statistical test: Cochran's Q (are all latents equally accurate?)
"""

from __future__ import annotations

import gc
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from harness import (
    DecodeConfig,
    DecodeMode,
    auto_calibrate,
    decode_latent,
    verify_answer,
)
from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.decode.projection import make_row_orthonormal_W
from latent_reasoning.decode.steering import make_steer_projection


def decode_with_raw_soft_prompt(
    encoder: LLMEncoder,
    soft_prompt: torch.Tensor,
    query: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    enable_thinking: bool = True,
) -> str:
    """Decode using a pre-built soft prompt tensor (bypasses latent projection).

    Used for control experiments where soft prompt tokens are generated
    directly (e.g., random noise, mean embeddings) rather than projected
    from a latent vector via W.
    """
    system_msg = "Answer to the best of your ability."
    user_msg = query or ""

    if hasattr(encoder.tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        try:
            template_kwargs = dict(tokenize=False, add_generation_prompt=True)
            if not enable_thinking:
                template_kwargs["enable_thinking"] = False
            prompt = encoder.tokenizer.apply_chat_template(
                messages, **template_kwargs,
            )
        except Exception:
            prompt = (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
    else:
        prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: "

    inputs = encoder.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(encoder._device) for k, v in inputs.items()}

    sp = soft_prompt.to(encoder.model.dtype).to(encoder._device)
    with torch.no_grad():
        text_embeds = encoder.model.get_input_embeddings()(inputs["input_ids"])
        combined_embeds = torch.cat([sp, text_embeds], dim=1)

        soft_mask = torch.ones(
            1, sp.size(1),
            dtype=inputs["attention_mask"].dtype,
            device=encoder._device,
        )
        combined_mask = torch.cat([soft_mask, inputs["attention_mask"]], dim=1)

        generate_kwargs = dict(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=encoder.tokenizer.pad_token_id,
            eos_token_id=encoder.tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False

        output_ids = encoder.model.generate(**generate_kwargs)

    new_tokens = output_ids[0, :]
    text = encoder.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def safe_print(text: str) -> None:
    """Print with ASCII fallback for Windows cp1252 compatibility."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


# =====================================================================
# Task generation — random multi-step arithmetic chains
# =====================================================================

@dataclass
class SensitivityTask:
    task_id: str
    prompt: str
    correct_answer: int
    n_steps: int
    difficulty: str


def generate_tasks(
    n_easy: int = 5,
    n_medium: int = 10,
    n_hard: int = 10,
    seed: int = 42,
) -> List[SensitivityTask]:
    """Generate arithmetic chain tasks at three difficulty levels.

    Easy: 2-3 steps, small numbers, only + and *
    Medium: 4-5 steps, medium numbers, all operations
    Hard: 6-8 steps, larger numbers, includes mod and multi-digit
    """
    rng = random.Random(seed)
    tasks: List[SensitivityTask] = []

    def _make_chain(n_steps: int, num_range: tuple, ops: list) -> tuple:
        value = rng.randint(*num_range)
        var = "a"
        lines = [f"  {var} = {value}"]

        for step in range(n_steps - 1):
            nxt = chr(ord("a") + step + 1)

            # Filter ops based on current value to avoid edge cases
            valid = list(ops)
            if value <= 5 and "-" in valid:
                valid.remove("-")
            if value <= 2 and "%" in valid:
                valid.remove("%")
            if not valid:
                valid = ["+"]

            op = rng.choice(valid)

            if op == "+":
                operand = rng.randint(num_range[0] // 2, num_range[1])
                new_val = value + operand
            elif op == "-":
                operand = rng.randint(1, max(1, value - 1))
                new_val = value - operand
            elif op == "*":
                operand = rng.randint(2, 7)
                new_val = value * operand
            elif op == "%":
                operand = rng.randint(3, max(4, min(13, value)))
                new_val = value % operand
            elif op == "//":
                operand = rng.randint(2, max(3, min(7, value // 2)))
                new_val = value // operand
            else:
                operand = rng.randint(1, 10)
                new_val = value + operand

            lines.append(f"  {nxt} = {var} {op} {operand}")
            var = nxt
            value = new_val

        prompt = (
            "Solve step by step:\n"
            + "\n".join(lines)
            + f"\nWhat is {var}? Answer with just the number."
        )
        return prompt, value

    idx = 0

    # Easy tasks
    for _ in range(n_easy):
        n_steps = rng.randint(2, 3)
        prompt, answer = _make_chain(n_steps, (5, 30), ["+", "*"])
        tasks.append(SensitivityTask(
            f"sens_{idx:03d}", prompt, answer, n_steps, "easy"))
        idx += 1

    # Medium tasks
    for _ in range(n_medium):
        n_steps = rng.randint(4, 5)
        prompt, answer = _make_chain(n_steps, (10, 50), ["+", "-", "*", "%"])
        tasks.append(SensitivityTask(
            f"sens_{idx:03d}", prompt, answer, n_steps, "medium"))
        idx += 1

    # Hard tasks
    for _ in range(n_hard):
        n_steps = rng.randint(6, 8)
        prompt, answer = _make_chain(n_steps, (20, 99), ["+", "-", "*", "%", "//"])
        tasks.append(SensitivityTask(
            f"sens_{idx:03d}", prompt, answer, n_steps, "hard"))
        idx += 1

    return tasks


def generate_nested_tasks(
    n_tasks: int = 40,
    seed: int = 42,
    difficulty_filter: Optional[str] = None,
) -> List[SensitivityTask]:
    """Generate nested expression tasks WITHOUT step-by-step scaffolding.

    These are harder because:
    - No variable assignment scaffolding (must parse expression tree)
    - Multi-digit arithmetic (numbers 10-999)
    - Multiple branches to track simultaneously
    - Modulo by larger divisors requires mental division

    Difficulty levels:
    - easy_nested: single operation, 2-digit numbers (~92% baseline)
    - sweet_spot: 2-3 ops, 2-digit*2-digit core, targeting ~60% baseline
    - medium_nested: 2-3 operations, some nesting (~42% baseline)
    - hard_nested: 3-4 operations with branches (~8% baseline)
    - brutal_nested: 5+ operations, deep nesting, large numbers (~12% baseline)
    """
    rng = random.Random(seed)
    tasks: List[SensitivityTask] = []

    def _eval_safe(expr: str) -> Optional[int]:
        """Safely evaluate an arithmetic expression."""
        try:
            result = eval(expr)  # noqa: S307 - only evaluating our own expressions
            if isinstance(result, float):
                result = int(result)
            return result
        except (ZeroDivisionError, ValueError):
            return None

    def _make_easy_nested(idx: int) -> SensitivityTask:
        """2-digit * 2-digit + 1-digit, or 2-digit * 1-digit + 2-digit."""
        patterns = [
            lambda: (f"{rng.randint(11,49)} * {rng.randint(11,49)} + {rng.randint(10,99)}", 2),
            lambda: (f"{rng.randint(10,99)} * {rng.randint(3,9)} - {rng.randint(10,50)}", 2),
            lambda: (f"({rng.randint(10,99)} + {rng.randint(10,99)}) * {rng.randint(3,9)}", 2),
            lambda: (f"{rng.randint(100,500)} - {rng.randint(10,99)} * {rng.randint(2,5)}", 2),
        ]
        gen = rng.choice(patterns)
        expr, n_ops = gen()
        answer = _eval_safe(expr)
        prompt = (
            f"Compute the following. Show your work, "
            f"then state the final answer.\n{expr}"
        )
        return SensitivityTask(
            f"nest_{idx:03d}", prompt, answer, n_ops, "easy_nested")

    def _make_sweet_spot(idx: int) -> SensitivityTask:
        """2-3 operations with 2-digit*2-digit core. Targets ~60% baseline.

        Key difficulty: larger multiplications (20-99 * 20-99) which are
        borderline for Qwen3-4B. Plus one additional simple operation.
        """
        a = rng.randint(20, 99)
        b = rng.randint(20, 99)
        c = rng.randint(10, 99)
        d = rng.randint(10, 60)
        patterns = [
            # Two multiplications added: a*b + c*d
            (f"{a} * {b} + {c} * {d}", 3),
            # Multiplication then modulo by small number
            (f"({a} * {b} + {c}) % {rng.randint(13, 29)}", 3),
            # Two multiplications subtracted
            (f"{a} * {b} - {c} * {rng.randint(2, 8)}", 3),
            # FOIL: (a+b)*(c-d) where result is multi-digit
            (f"({a} + {b}) * ({c} - {d})", 3),
            # Triple multiplication with smaller numbers
            (f"{rng.randint(5, 15)} * {rng.randint(10, 30)} * {rng.randint(3, 9)}", 3),
            # a*b with integer division
            (f"({a} * {b}) // {rng.randint(3, 9)} + {c}", 3),
        ]
        expr, n_ops = rng.choice(patterns)
        answer = _eval_safe(expr)
        if answer is None or (isinstance(answer, int) and abs(answer) > 100000):
            # Fallback to safe pattern
            expr = f"{a} * {b} + {c}"
            answer = _eval_safe(expr)
            n_ops = 2
        prompt = (
            f"Compute the following expression. Show your work, "
            f"then state the final answer as a single number.\n{expr}"
        )
        return SensitivityTask(
            f"nest_{idx:03d}", prompt, answer, n_ops, "sweet_spot")

    def _make_medium_nested(idx: int) -> SensitivityTask:
        """2-3 operations with parenthesized nesting."""
        a, b, c, d = (rng.randint(10, 99) for _ in range(4))
        m = rng.randint(7, 23)
        patterns = [
            (f"({a} * {b} + {c}) % {m}", 3),
            (f"({a} + {b}) * ({c} - {d})", 3),
            (f"({a} * {b}) // {rng.randint(3, 9)} + {c} * {d}", 4),
            (f"({a} * {b} - {c}) % {m} + {d}", 4),
            (f"({a} + {b} * {c}) // {rng.randint(5, 15)}", 3),
        ]
        expr, n_ops = rng.choice(patterns)
        answer = _eval_safe(expr)
        if answer is None:
            expr = f"({a} * {b} + {c}) % {m}"
            answer = _eval_safe(expr)
            n_ops = 3
        prompt = (
            f"Compute the following expression. Show your work, "
            f"then state the final answer as a single number.\n{expr}"
        )
        return SensitivityTask(
            f"nest_{idx:03d}", prompt, answer, n_ops, "medium_nested")

    def _make_hard_nested(idx: int) -> SensitivityTask:
        """3-4 operations with branches (two sub-expressions combined)."""
        a, b = rng.randint(20, 99), rng.randint(11, 49)
        c, d = rng.randint(10, 80), rng.randint(5, 30)
        e = rng.randint(3, 9)
        m = rng.randint(11, 37)
        patterns = [
            (f"({a} * {b} + {c}) % {m} * (({d} + {rng.randint(10,50)}) // {e})", 6),
            (f"({a} * {b}) // {rng.randint(3,7)} + ({c} * {d}) % {m}", 5),
            (f"(({a} + {b}) * {rng.randint(3,8)} - {c}) % {m} + {d} * {e}", 6),
            (f"({a} * {b} - {c} * {d}) // {rng.randint(2,5)} + {rng.randint(10,99)}", 5),
        ]
        expr, n_ops = rng.choice(patterns)
        answer = _eval_safe(expr)
        if answer is None or answer < 0:
            # Fallback to safe pattern
            expr = f"({a} * {b} + {c}) % {m} + {d} * {e}"
            answer = _eval_safe(expr)
            n_ops = 4
        prompt = (
            f"Compute the following expression. Show your work step by step, "
            f"then state the final answer as a single number.\n{expr}"
        )
        return SensitivityTask(
            f"nest_{idx:03d}", prompt, answer, n_ops, "hard_nested")

    def _make_brutal_nested(idx: int) -> SensitivityTask:
        """5+ operations, deep nesting, 3-digit numbers, multiple branches."""
        a = rng.randint(100, 499)
        b = rng.randint(11, 49)
        c = rng.randint(50, 199)
        d = rng.randint(20, 80)
        e = rng.randint(10, 40)
        f = rng.randint(3, 9)
        g = rng.randint(10, 30)
        m1 = rng.randint(13, 47)
        m2 = rng.randint(7, 19)
        patterns = [
            (f"(({a} * {b} + {c}) % {m1}) * (({d} * {f} - {e}) % {m2} + {g})", 8),
            (f"(({a} + {b} * {c}) // {rng.randint(5,15)}) * {f} + ({d} * {e}) % {m1}", 7),
            (f"({a} * {b}) % {m1} + (({c} - {d}) * {f}) // {rng.randint(2,5)} - {g}", 7),
            (f"(({a} * {b} + {c} * {d}) % {m1}) * (({e} + {g}) // {f})", 7),
        ]
        expr, n_ops = rng.choice(patterns)
        answer = _eval_safe(expr)
        if answer is None or answer < 0:
            expr = f"(({a} * {b} + {c}) % {m1}) * {f} + {g}"
            answer = _eval_safe(expr)
            n_ops = 5
        prompt = (
            f"Compute the following expression carefully. Show every step "
            f"of your work, then state the final answer as a single number.\n{expr}"
        )
        return SensitivityTask(
            f"nest_{idx:03d}", prompt, answer, n_ops, "brutal_nested")

    # If a difficulty filter is set, generate all tasks at that level
    if difficulty_filter:
        makers = {
            "easy_nested": _make_easy_nested,
            "sweet_spot": _make_sweet_spot,
            "medium_nested": _make_medium_nested,
            "hard_nested": _make_hard_nested,
            "brutal_nested": _make_brutal_nested,
        }
        maker = makers.get(difficulty_filter)
        if maker is None:
            raise ValueError(f"Unknown difficulty: {difficulty_filter}")
        idx = 0
        for _ in range(n_tasks):
            t = maker(idx)
            if t.correct_answer is not None:
                tasks.append(t)
                idx += 1
        return tasks

    # Default distribution: 8 easy, 12 medium, 12 hard, 8 brutal
    dist = {
        "easy": max(1, n_tasks // 5),
        "medium": max(1, n_tasks * 3 // 10),
        "hard": max(1, n_tasks * 3 // 10),
    }
    dist["brutal"] = n_tasks - dist["easy"] - dist["medium"] - dist["hard"]

    idx = 0
    for _ in range(dist["easy"]):
        t = _make_easy_nested(idx)
        if t.correct_answer is not None:
            tasks.append(t)
            idx += 1

    for _ in range(dist["medium"]):
        t = _make_medium_nested(idx)
        if t.correct_answer is not None:
            tasks.append(t)
            idx += 1

    for _ in range(dist["hard"]):
        t = _make_hard_nested(idx)
        if t.correct_answer is not None:
            tasks.append(t)
            idx += 1

    for _ in range(dist["brutal"]):
        t = _make_brutal_nested(idx)
        if t.correct_answer is not None:
            tasks.append(t)
            idx += 1

    return tasks


# =====================================================================
# Zero-shot baseline (same chat template, no conditioning)
# =====================================================================

def run_zero_shot(
    encoder: LLMEncoder, prompt: str,
    max_new_tokens: int = 1024, enable_thinking: bool = True,
) -> str:
    """Run with chat template but no soft prompt conditioning."""
    system_msg = "Answer to the best of your ability."
    if hasattr(encoder.tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        try:
            template_kwargs = dict(
                tokenize=False, add_generation_prompt=True,
            )
            if not enable_thinking:
                template_kwargs["enable_thinking"] = False
            formatted = encoder.tokenizer.apply_chat_template(
                messages, **template_kwargs,
            )
        except Exception:
            formatted = (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
    else:
        formatted = f"System: {system_msg}\n\nUser: {prompt}\n\nAssistant: "

    inputs = encoder.tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(encoder._device) for k, v in inputs.items()}

    with torch.no_grad():
        out = encoder.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=encoder.tokenizer.pad_token_id,
            repetition_penalty=1.2,
        )

    resp = encoder.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()

    if "<think>" in resp and "</think>" in resp:
        think_end = resp.find("</think>") + len("</think>")
        resp = resp[think_end:].strip()
    elif resp.startswith("<think>"):
        for starter in ["1.", "Step 1", "## Step", "Here's", "Here is"]:
            if starter in resp:
                resp = resp[resp.index(starter):]
                break

    return resp if resp else "No response"


# =====================================================================
# Main experiment
# =====================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Latent sensitivity test")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--n-latents", type=int, default=20)
    parser.add_argument("--n-easy", type=int, default=5)
    parser.add_argument("--n-medium", type=int, default=10)
    parser.add_argument("--n-hard", type=int, default=10)
    parser.add_argument(
        "--diagnostic", action="store_true",
        help="Quick run: 5 latents, 3+5+5 tasks")
    parser.add_argument(
        "--task-type", default="chain", choices=["chain", "nested"],
        help="Task type: chain (sequential) or nested (expression trees)")
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Calibration mode: run baseline only to find 50-70%% sweet spot")
    parser.add_argument("--n-calibrate", type=int, default=40,
        help="Number of tasks to generate in calibration mode")
    parser.add_argument(
        "--difficulty", default=None,
        choices=["easy_nested", "sweet_spot", "medium_nested", "hard_nested",
                 "brutal_nested"],
        help="Generate ALL nested tasks at this difficulty level")
    parser.add_argument(
        "--decode-mode", default="soft_prompt",
        choices=["soft_prompt", "multi_scale"],
        help="Conditioning mode: soft_prompt (default) or multi_scale (+ steering)")
    parser.add_argument("--steer-layers", default="22,25,28",
        help="Comma-separated layer indices for multi_scale steering")
    parser.add_argument("--steer-scale", type=float, default=1.0,
        help="Scaling factor for intermediate layer steering vectors")
    parser.add_argument("--no-think", action="store_true",
        help="Disable Qwen3 thinking mode (faster, shorter outputs)")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
        help="Max generation tokens")
    parser.add_argument(
        "--control-mode", default="latent_projected",
        choices=["latent_projected", "random_noise", "mean_embedding"],
        help="Control mode: latent_projected (normal W projection), "
             "random_noise (random embeddings at target RMS), "
             "mean_embedding (mean token embedding repeated)")
    parser.add_argument("--n-tasks", type=int, default=None,
        help="Override total number of tasks (for nested/sweet_spot modes)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.diagnostic:
        args.n_latents = 5
        args.n_easy = 3
        args.n_medium = 5
        args.n_hard = 5

    # Generate tasks based on type
    if args.task_type == "nested":
        n_tasks_gen = args.n_calibrate if args.calibrate else (
            args.n_easy + args.n_medium + args.n_hard)
        tasks = generate_nested_tasks(
            n_tasks=n_tasks_gen, difficulty_filter=args.difficulty)
    else:
        tasks = generate_tasks(
            n_easy=args.n_easy,
            n_medium=args.n_medium,
            n_hard=args.n_hard,
        )
    # Override task count if requested
    if args.n_tasks is not None and args.n_tasks < len(tasks):
        tasks = tasks[:args.n_tasks]
    n_tasks = len(tasks)

    run_mode = "CALIBRATE" if args.calibrate else (
        "DIAGNOSTIC" if args.diagnostic else "FULL")

    print("=" * 70)
    print("LATENT SENSITIVITY TEST")
    print(f"Model: {args.model} ({args.quantization})")
    print(f"Task type: {args.task_type}")
    print(f"Tasks: {n_tasks}")
    if not args.calibrate:
        print(f"Latents: {args.n_latents}")
    print(f"Mode: {run_mode}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    encoder = LLMEncoder(model_name=args.model, quantization=args.quantization)
    cal = auto_calibrate(encoder)
    d_latent = encoder.latent_dim
    embed_dim = cal["embed_dim"]
    target_rms = cal["embedding_rms"]

    print(f"  Latent dim: {d_latent}")
    print(f"  Embed dim:  {embed_dim}")
    print(f"  Target RMS: {target_rms:.5f}")

    # Shared W matrix
    W = make_row_orthonormal_W(d_latent, 8 * embed_dim, seed=1234)
    W = W.to(encoder._device)

    # ---- Phase 1: Zero-shot baseline ----
    print(f"\n{'=' * 40}")
    print("PHASE 1: Zero-shot baseline")
    print(f"{'=' * 40}")
    baseline_results = []
    start = time.time()
    for i, task in enumerate(tasks):
        t0 = time.time()
        resp = run_zero_shot(
            encoder, task.prompt,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=not args.no_think,
        )
        elapsed = time.time() - t0
        correct = verify_answer(resp, task.correct_answer)
        baseline_results.append({
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "n_steps": task.n_steps,
            "correct_answer": task.correct_answer,
            "response": resp[:500],
            "correct": correct,
            "time": round(elapsed, 1),
        })
        mark = "OK" if correct else "WRONG"
        safe_print(
            f"  [{i + 1}/{n_tasks}] {task.task_id} "
            f"({task.difficulty}/{task.n_steps}step): "
            f"{mark} (expect={task.correct_answer}, t={elapsed:.1f}s)")

    baseline_elapsed = time.time() - start
    baseline_accuracy = (
        sum(1 for r in baseline_results if r["correct"]) / len(baseline_results)
    )
    # Collect all difficulty levels present
    all_diffs = sorted(set(r["difficulty"] for r in baseline_results))
    per_diff = {}
    for diff in all_diffs:
        sub = [r for r in baseline_results if r["difficulty"] == diff]
        if sub:
            per_diff[diff] = sum(1 for r in sub if r["correct"]) / len(sub)

    print(f"\nBaseline accuracy: {baseline_accuracy:.1%}")
    for d, a in per_diff.items():
        print(f"  {d}: {a:.1%}")
    print(f"Baseline time: {baseline_elapsed:.0f}s")

    # ---- Calibration mode: early exit ----
    if args.calibrate:
        print(f"\n{'=' * 40}")
        print("CALIBRATION COMPLETE")
        print(f"{'=' * 40}")

        # Find tasks in the 50-70% sweet spot per difficulty
        sweet_spot = [r for r in baseline_results if not r["correct"]]
        n_wrong = len(sweet_spot)
        n_right = len(baseline_results) - n_wrong
        print(f"\n  Correct: {n_right}/{n_tasks} ({n_right/n_tasks:.0%})")
        print(f"  Wrong:   {n_wrong}/{n_tasks} ({n_wrong/n_tasks:.0%})")

        if 0.30 <= baseline_accuracy <= 0.80:
            print("\n  >> SWEET SPOT ACHIEVED <<")
            print("  This difficulty range is suitable for sensitivity testing.")
        elif baseline_accuracy > 0.80:
            print("\n  >> TOO EASY -- increase difficulty <<")
        else:
            print("\n  >> TOO HARD -- decrease difficulty <<")

        # Per-task detail for wrong answers
        if n_wrong > 0:
            print(f"\nWrong answers ({n_wrong}):")
            for r in baseline_results:
                if not r["correct"]:
                    safe_print(
                        f"  {r['task_id']} ({r['difficulty']}): "
                        f"expect={r['correct_answer']}, "
                        f"resp='{r['response'][:80]}...'")

        # Save calibration results
        cal_output = {
            "experiment": "latent_sensitivity_calibration",
            "model": args.model,
            "quantization": args.quantization,
            "task_type": args.task_type,
            "n_tasks": n_tasks,
            "baseline_accuracy": baseline_accuracy,
            "per_difficulty": per_diff,
            "baseline_results": baseline_results,
            "elapsed_s": baseline_elapsed,
            "calibration": cal,
        }
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = (Path(__file__).parent
                        / f"calibration_{args.task_type}_results.json")
        with open(out_path, "w") as f_out:
            json.dump(cal_output, f_out, indent=2, default=str)
        print(f"\nCalibration saved to: {out_path}")
        print(f"Total time: {baseline_elapsed / 60:.1f} min")

        # Cleanup
        del encoder
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    # ---- Phase 2: Latent sensitivity ----
    print(f"\n{'=' * 40}")
    print("PHASE 2: Latent sensitivity")
    print(f"{'=' * 40}")

    control_mode = args.control_mode
    num_soft_tokens = 8

    # Generate control soft prompts or latents based on mode
    if control_mode == "random_noise":
        print(f"CONTROL MODE: random_noise")
        print(f"  Generating {args.n_latents} random noise soft prompts "
              f"({num_soft_tokens} tokens x {embed_dim}d, target_rms={target_rms:.5f})")
        noise_gen = torch.Generator().manual_seed(2024)
        control_soft_prompts = []
        for _ in range(args.n_latents):
            sp = torch.randn(1, num_soft_tokens, embed_dim, generator=noise_gen)
            current_rms = sp.square().mean().sqrt().clamp_min(1e-8)
            sp = sp * (target_rms / current_rms)
            control_soft_prompts.append(sp)

    elif control_mode == "mean_embedding":
        print(f"CONTROL MODE: mean_embedding")
        print(f"  Computing mean token embedding and repeating {num_soft_tokens} times")
        with torch.no_grad():
            embed_weight = encoder.model.get_input_embeddings().weight
            mean_emb = embed_weight.float().mean(dim=0, keepdim=True)  # (1, embed_dim)
            mean_rms = mean_emb.square().mean().sqrt().clamp_min(1e-8)
            mean_emb = mean_emb * (target_rms / mean_rms)
            sp = mean_emb.unsqueeze(0).expand(1, num_soft_tokens, embed_dim).clone()
        control_soft_prompts = [sp] * args.n_latents
        print(f"  Note: all {args.n_latents} use identical soft prompt (control)")

    else:
        # Default: latent_projected — generate random latents
        print(f"Generating {args.n_latents} random latents...")
        latent_gen = torch.Generator().manual_seed(2024)
        ball_radius = (1.0 / math.sqrt(0.5)) * 0.95
        target_norm = 0.5 * ball_radius

        latents = []
        for _ in range(args.n_latents):
            z = torch.randn(1, d_latent, generator=latent_gen)
            z = z * (target_norm / z.norm())
            latents.append(z)
        control_soft_prompts = None  # use decode_latent path

    # Decode config (only needed for latent_projected mode)
    decode_mode = (DecodeMode.MULTI_SCALE if args.decode_mode == "multi_scale"
                   else DecodeMode.SOFT_PROMPT)

    # Build layer projections for multi_scale mode
    layer_projections = None
    if decode_mode == DecodeMode.MULTI_SCALE:
        layer_indices = [int(x) for x in args.steer_layers.split(",")]
        layer_projections = {}
        for li_idx in layer_indices:
            layer_projections[li_idx] = make_steer_projection(
                d_latent, embed_dim, seed=li_idx * 1000 + 42,
            ).to(encoder._device)
        safe_print(
            f"  Multi-scale steering at layers: {layer_indices}, "
            f"scale={args.steer_scale}")

    cfg = DecodeConfig(
        geometry="euclidean",
        mode=decode_mode,
        W_soft=W,
        embed_dim=embed_dim,
        num_soft_tokens=num_soft_tokens,
        target_rms=target_rms,
        curvature=0.5,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,  # Greedy for determinism
        enable_thinking=not args.no_think,
        layer_projections=layer_projections,
        steer_scale=args.steer_scale,
        hidden_rms=target_rms,
    )

    sensitivity_results: List[Dict] = []
    phase2_start = time.time()

    for li in range(args.n_latents):
        label = ("Noise" if control_mode == "random_noise"
                 else "MeanEmb" if control_mode == "mean_embedding"
                 else "Latent")
        print(f"\n  --- {label} {li + 1}/{args.n_latents} ---")
        task_results = []
        for ti, task in enumerate(tasks):
            t0 = time.time()
            try:
                if control_soft_prompts is not None:
                    resp = decode_with_raw_soft_prompt(
                        encoder, control_soft_prompts[li], task.prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.0,
                        enable_thinking=not args.no_think,
                    )
                else:
                    resp = decode_latent(encoder, latents[li], task.prompt, cfg)
            except Exception as e:
                print(f"    [{ti + 1}/{n_tasks}] {task.task_id}: ERROR {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                try:
                    if control_soft_prompts is not None:
                        resp = decode_with_raw_soft_prompt(
                            encoder, control_soft_prompts[li], task.prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=0.0,
                            enable_thinking=not args.no_think,
                        )
                    else:
                        resp = decode_latent(encoder, latents[li], task.prompt, cfg)
                except Exception:
                    resp = "ERROR"
            elapsed = time.time() - t0
            correct = verify_answer(resp, task.correct_answer)
            task_results.append({
                "task_id": task.task_id,
                "difficulty": task.difficulty,
                "correct_answer": task.correct_answer,
                "response": resp[:500],
                "correct": correct,
                "time": round(elapsed, 1),
            })
            mark = "OK" if correct else "X "
            safe_print(
                f"    [{ti + 1}/{n_tasks}] {task.task_id}: "
                f"{mark} ({elapsed:.1f}s)")

        acc = sum(1 for r in task_results if r["correct"]) / len(task_results)
        n_correct = sum(1 for r in task_results if r["correct"])
        print(f"  Latent {li + 1} accuracy: {acc:.1%} ({n_correct}/{n_tasks})")

        sensitivity_results.append({
            "latent_idx": li,
            "accuracy": acc,
            "n_correct": n_correct,
            "n_total": n_tasks,
            "task_results": task_results,
        })

        # Per-latent cleanup to prevent VRAM accumulation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    phase2_elapsed = time.time() - phase2_start

    # ---- Phase 3: Statistical analysis ----
    print(f"\n{'=' * 40}")
    print("PHASE 3: Analysis")
    print(f"{'=' * 40}")

    accuracies = np.array([r["accuracy"] for r in sensitivity_results])
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0
    min_acc = float(np.min(accuracies))
    max_acc = float(np.max(accuracies))
    range_acc = max_acc - min_acc

    print(f"\nAccuracy across {args.n_latents} latents:")
    print(f"  Mean:  {mean_acc:.1%}")
    print(f"  Std:   {std_acc:.1%}")
    print(f"  Min:   {min_acc:.1%} (latent {int(np.argmin(accuracies))})")
    print(f"  Max:   {max_acc:.1%} (latent {int(np.argmax(accuracies))})")
    print(f"  Range: {range_acc:.1%}")

    # Per-difficulty breakdown
    print("\nPer-difficulty accuracy (mean across latents):")
    for diff in all_diffs:
        diff_accs = []
        for sr in sensitivity_results:
            sub = [r for r in sr["task_results"] if r["difficulty"] == diff]
            if sub:
                diff_accs.append(
                    sum(1 for r in sub if r["correct"]) / len(sub))
        if diff_accs:
            print(
                f"  {diff}: mean={np.mean(diff_accs):.1%}, "
                f"std={np.std(diff_accs):.1%}, "
                f"range=[{np.min(diff_accs):.1%}, {np.max(diff_accs):.1%}]")

    # Cochran's Q test
    cochran_q = None
    cochran_p = None
    if args.n_latents >= 3:
        n_l = len(sensitivity_results)
        binary = np.zeros((n_l, n_tasks))
        for li, sr in enumerate(sensitivity_results):
            for ti, tr in enumerate(sr["task_results"]):
                binary[li, ti] = 1.0 if tr["correct"] else 0.0

        # Cochran's Q: rows=latents (treatments), cols=tasks (subjects)
        # T_j = per-treatment (per-latent) totals = sum across tasks
        # T_i = per-subject (per-task) totals = sum across latents
        T_j = binary.sum(axis=1)   # per-latent totals (treatments)
        T_i = binary.sum(axis=0)   # per-task totals (subjects)
        N = binary.sum()
        k = n_l

        numer = (k - 1) * (k * float((T_j ** 2).sum()) - N ** 2)
        denom = k * N - float((T_i ** 2).sum())

        if denom > 0:
            from scipy.stats import chi2
            cochran_q = float(numer / denom)
            cochran_p = float(1.0 - chi2.cdf(cochran_q, k - 1))
            print(f"\nCochran's Q test (are all latents equally accurate?):")
            print(f"  Q = {cochran_q:.3f}, p = {cochran_p:.4f}")
            sig = "YES" if cochran_p < 0.05 else "NO"
            print(f"  Significant at alpha=0.05: {sig}")
        else:
            print("\nCochran's Q: degenerate (all latents identical)")

    # Per-latent binomial tests vs baseline rate
    binomial_results = []
    if baseline_accuracy > 0 and baseline_accuracy < 1:
        from scipy.stats import binomtest
        print(f"\nPer-latent binomial test (H0: accuracy = baseline {baseline_accuracy:.1%}):")
        for li, sr in enumerate(sensitivity_results):
            n_correct = sr["n_correct"]
            p_val = binomtest(n_correct, n_tasks, baseline_accuracy,
                              alternative="greater").pvalue
            acc = sr["accuracy"]
            sig = "*" if p_val < 0.05 else " "
            print(f"  L{li}: {acc:.0%} ({n_correct}/{n_tasks}), "
                  f"p={p_val:.4f} {sig}")
            binomial_results.append({
                "latent_idx": li,
                "accuracy": acc,
                "n_correct": n_correct,
                "binomial_p": float(p_val),
                "significant_05": p_val < 0.05,
            })

    # Per-task variance: which tasks are most sensitive to latent?
    print("\nPer-task sensitivity (tasks where latents disagree):")
    task_var = []
    for ti, task in enumerate(tasks):
        task_correct = [
            sr["task_results"][ti]["correct"] for sr in sensitivity_results
        ]
        n_right = sum(task_correct)
        n_wrong = len(task_correct) - n_right
        frac = n_right / len(task_correct)
        task_var.append((task.task_id, task.difficulty, frac, n_right, n_wrong))

    # Sort by most variable (closest to 50%)
    task_var.sort(key=lambda x: abs(x[2] - 0.5))
    for tid, diff, frac, nr, nw in task_var[:10]:
        print(f"  {tid} ({diff}): {frac:.0%} correct ({nr} right, {nw} wrong)")

    # Verdict
    print(f"\n{'=' * 40}")
    print("COMPARISON")
    print(f"{'=' * 40}")
    print(f"  Zero-shot baseline: {baseline_accuracy:.1%}")
    print(f"  Mean conditioned:   {mean_acc:.1%}")
    print(f"  Best latent:        {max_acc:.1%}")
    print(f"  Worst latent:       {min_acc:.1%}")
    delta = mean_acc - baseline_accuracy
    print(f"  Conditioning delta: {delta:+.1%}")

    if range_acc > 0.15:
        verdict = "STRONGLY EXPLOITABLE"
        detail = f"accuracy varies {range_acc:.0%} across random latents"
    elif range_acc > 0.08:
        verdict = "EXPLOITABLE"
        detail = f"accuracy varies {range_acc:.0%} across random latents"
    elif range_acc > 0.03:
        verdict = "WEAKLY EXPLOITABLE"
        detail = f"accuracy varies {range_acc:.0%} -- marginal signal"
    else:
        verdict = "NOT EXPLOITABLE"
        detail = f"accuracy stable (<{range_acc:.0%} range) across latents"

    print(f"\n  VERDICT: {verdict}")
    print(f"  {detail}")

    total_elapsed = time.time() - start

    # ---- Save results ----
    output = {
        "experiment": "latent_sensitivity",
        "model": args.model,
        "quantization": args.quantization,
        "task_type": args.task_type,
        "decode_mode": args.decode_mode,
        "control_mode": control_mode,
        "mode": run_mode,
        "n_latents": args.n_latents,
        "n_tasks": n_tasks,
        "d_latent": d_latent,
        "embed_dim": embed_dim,
        "target_rms": target_rms,
        "calibration": cal,
        "baseline_accuracy": baseline_accuracy,
        "baseline_per_difficulty": per_diff,
        "baseline_results": baseline_results,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "min_accuracy": min_acc,
        "max_accuracy": max_acc,
        "range_accuracy": range_acc,
        "latent_accuracies": accuracies.tolist(),
        "cochrans_q": cochran_q,
        "binomial_tests": binomial_results,
        "cochrans_p": cochran_p,
        "verdict": verdict,
        "baseline_elapsed_s": baseline_elapsed,
        "sensitivity_elapsed_s": phase2_elapsed,
        "total_elapsed_s": total_elapsed,
        "sensitivity_results": sensitivity_results,
        "task_sensitivity": [
            {"task_id": t[0], "difficulty": t[1], "frac_correct": t[2]}
            for t in task_var
        ],
    }

    if args.output:
        out_path = Path(args.output)
    else:
        diff_tag = f"_{args.difficulty}" if args.difficulty else ""
        ctrl_tag = f"_{control_mode}" if control_mode != "latent_projected" else ""
        out_path = (Path(__file__).parent
                    / f"sensitivity{diff_tag}{ctrl_tag}_results.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {out_path}")
    print(f"Total time: {total_elapsed / 60:.1f} min")

    # Cleanup
    del encoder
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
