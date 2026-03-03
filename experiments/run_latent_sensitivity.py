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


# =====================================================================
# Zero-shot baseline (same chat template, no conditioning)
# =====================================================================

def run_zero_shot(encoder: LLMEncoder, prompt: str) -> str:
    """Run with chat template but no soft prompt conditioning."""
    system_msg = "Answer to the best of your ability."
    if hasattr(encoder.tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        try:
            formatted = encoder.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
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
            max_new_tokens=1024,
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
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.diagnostic:
        args.n_latents = 5
        args.n_easy = 3
        args.n_medium = 5
        args.n_hard = 5

    n_tasks = args.n_easy + args.n_medium + args.n_hard

    print("=" * 70)
    print("LATENT SENSITIVITY TEST")
    print(f"Model: {args.model} ({args.quantization})")
    print(f"Latents: {args.n_latents}")
    print(f"Tasks: {n_tasks} (easy={args.n_easy}, med={args.n_medium}, hard={args.n_hard})")
    mode = "DIAGNOSTIC" if args.diagnostic else "FULL"
    print(f"Mode: {mode}")
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

    # Generate tasks
    print(f"\nGenerating {n_tasks} tasks...")
    tasks = generate_tasks(
        n_easy=args.n_easy,
        n_medium=args.n_medium,
        n_hard=args.n_hard,
    )

    # ---- Phase 1: Zero-shot baseline ----
    print(f"\n{'=' * 40}")
    print("PHASE 1: Zero-shot baseline")
    print(f"{'=' * 40}")
    baseline_results = []
    start = time.time()
    for i, task in enumerate(tasks):
        t0 = time.time()
        resp = run_zero_shot(encoder, task.prompt)
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
    per_diff = {}
    for diff in ("easy", "medium", "hard"):
        sub = [r for r in baseline_results if r["difficulty"] == diff]
        if sub:
            per_diff[diff] = sum(1 for r in sub if r["correct"]) / len(sub)

    print(f"\nBaseline accuracy: {baseline_accuracy:.1%}")
    for d, a in per_diff.items():
        print(f"  {d}: {a:.1%}")
    print(f"Baseline time: {baseline_elapsed:.0f}s")

    # ---- Phase 2: Latent sensitivity ----
    print(f"\n{'=' * 40}")
    print("PHASE 2: Latent sensitivity")
    print(f"{'=' * 40}")

    # Generate random latents (Euclidean, normalized)
    print(f"Generating {args.n_latents} random latents...")
    latent_gen = torch.Generator().manual_seed(2024)
    ball_radius = (1.0 / math.sqrt(0.5)) * 0.95
    target_norm = 0.5 * ball_radius

    latents = []
    for _ in range(args.n_latents):
        z = torch.randn(1, d_latent, generator=latent_gen)
        z = z * (target_norm / z.norm())
        latents.append(z)

    # Decode config: Euclidean soft prompt, greedy decoding
    cfg = DecodeConfig(
        geometry="euclidean",
        mode=DecodeMode.SOFT_PROMPT,
        W_soft=W,
        embed_dim=embed_dim,
        num_soft_tokens=8,
        target_rms=target_rms,
        curvature=0.5,
        max_new_tokens=1024,
        temperature=0.0,  # Greedy for determinism
    )

    sensitivity_results: List[Dict] = []
    phase2_start = time.time()

    for li, latent in enumerate(latents):
        print(f"\n  --- Latent {li + 1}/{args.n_latents} ---")
        task_results = []
        for ti, task in enumerate(tasks):
            t0 = time.time()
            resp = decode_latent(encoder, latent, task.prompt, cfg)
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
    for diff in ("easy", "medium", "hard"):
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

        T_j = binary.sum(axis=0)   # per-task totals
        T_i = binary.sum(axis=1)   # per-latent totals
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
        "mode": mode,
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
        out_path = Path(__file__).parent / "latent_sensitivity_results.json"

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
