"""Selector Realization Study — Phase 1: Arithmetic.

Frozen-selector experiment measuring how well local heuristics can
recover oracle gains from perturbation-based trajectory diversity.

Design approved by Codex (2026-06-27):
- Generate k=20 perturbations per task, report prefix curves at {3,5,10,20}
- 100 test tasks (sweet_spot difficulty)
- Majority vote is the central opponent
- Candidate-level JSONL output with full features
- All selectors frozen before generation begins
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.decode.projection import make_row_orthonormal_W
from experiments.harness import (
    Task,
    DecodeConfig,
    DecodeMode,
    auto_calibrate,
    check_soft_prompt_compatibility,
    decode_latent,
    extract_answer,
    generate_nested_expression_tasks,
    load_gsm8k_tasks,
    verify_answer,
    _make_noise,
    _apply_mutation,
    _parse_integers,
)


# =====================================================================
# Candidate dataclass — stores everything per perturbation
# =====================================================================

@dataclass
class CandidateResult:
    task_id: str
    perturbation_idx: int  # 0 = greedy baseline, 1..k = perturbations
    raw_output: str
    stripped_response: str
    extracted_answer: Optional[int]
    correct: bool
    token_count: int
    has_eos: bool
    truncated: bool
    has_think_tags: bool
    all_integers: List[int]
    prompt_integers: List[int]


# =====================================================================
# Selector features — computed per candidate
# =====================================================================

@dataclass
class SelectorFeatures:
    answer_exists: bool
    answer_frequency: int  # how many of k candidates gave same answer
    answer_is_majority: bool
    agrees_with_greedy: bool
    scratchpad_consistent: bool  # intermediate steps support final answer
    prompt_grounded: bool  # final answer uses numbers from prompt
    no_truncation: bool
    no_loops: bool
    response_length: int
    unique_integer_count: int


def compute_selector_features(
    candidate: CandidateResult,
    all_candidates: List[CandidateResult],
    greedy_answer: Optional[int],
) -> SelectorFeatures:
    """Compute local selector features for a single candidate."""
    answers = [c.extracted_answer for c in all_candidates if c.extracted_answer is not None]
    answer_counts = Counter(answers)
    majority_answer = answer_counts.most_common(1)[0][0] if answer_counts else None

    ans = candidate.extracted_answer
    freq = answer_counts.get(ans, 0) if ans is not None else 0

    # Scratchpad consistency: check if intermediate integers support the final answer
    scratchpad_ok = _check_scratchpad_consistency(candidate)

    # Prompt grounding: final answer involves numbers from the prompt
    prompt_grounded = _check_prompt_grounding(candidate)

    # Loop detection: repeated phrases in output
    no_loops = not _detect_loops(candidate.stripped_response)

    return SelectorFeatures(
        answer_exists=ans is not None,
        answer_frequency=freq,
        answer_is_majority=(ans == majority_answer) if ans is not None else False,
        agrees_with_greedy=(ans == greedy_answer) if ans is not None else False,
        scratchpad_consistent=scratchpad_ok,
        prompt_grounded=prompt_grounded,
        no_truncation=not candidate.truncated,
        no_loops=no_loops,
        response_length=len(candidate.stripped_response),
        unique_integer_count=len(set(candidate.all_integers)),
    )


def _check_scratchpad_consistency(candidate: CandidateResult) -> bool:
    """Check if intermediate arithmetic in the response supports the final answer."""
    if candidate.extracted_answer is None:
        return False
    ints = candidate.all_integers
    if len(ints) < 2:
        return True  # nothing to contradict
    # Simple heuristic: the final answer should appear as a computation result,
    # not be a completely disconnected number
    final = ints[-1]
    # Check if any pair of earlier numbers could produce the final via basic ops
    for i, a in enumerate(ints[:-1]):
        for b in ints[i + 1:-1]:
            if a + b == final or a * b == final or a - b == final or b - a == final:
                return True
            if b != 0 and a // b == final:
                return True
            if a != 0 and b // a == final:
                return True
    # Also check if final appears earlier (self-reference in computation)
    if final in ints[:-1]:
        return True
    return False


def _check_prompt_grounding(candidate: CandidateResult) -> bool:
    """Check if the candidate's answer involves numbers from the prompt."""
    if candidate.extracted_answer is None:
        return False
    # If the prompt integers are present in the response, the model is grounded
    prompt_nums = set(candidate.prompt_integers)
    response_nums = set(candidate.all_integers)
    # At least some prompt numbers should appear in the response
    overlap = prompt_nums & response_nums
    return len(overlap) >= max(1, len(prompt_nums) // 2)


def _detect_loops(text: str, window: int = 50, threshold: int = 3) -> bool:
    """Detect repetitive loops in generated text."""
    if len(text) < window * 2:
        return False
    # Check for repeated substrings
    for size in range(window, max(10, window // 5), -5):
        for start in range(0, len(text) - size * threshold, size):
            chunk = text[start:start + size]
            count = text.count(chunk)
            if count >= threshold:
                return True
    return False


# =====================================================================
# Frozen selectors — each returns the index of the selected candidate
# =====================================================================

def select_greedy(candidates: List[CandidateResult], features: List[SelectorFeatures]) -> int:
    """Always pick the greedy baseline (index 0)."""
    return 0


def select_random(candidates: List[CandidateResult], features: List[SelectorFeatures], rng) -> int:
    """Pick a random perturbation (indices 1..k)."""
    perturbed = list(range(1, len(candidates)))
    return rng.choice(perturbed) if perturbed else 0


def select_oracle(candidates: List[CandidateResult], features: List[SelectorFeatures]) -> int:
    """Pick a correct candidate if one exists, else fall back to majority."""
    for i, c in enumerate(candidates):
        if c.correct:
            return i
    return select_majority(candidates, features)


def select_majority(candidates: List[CandidateResult], features: List[SelectorFeatures]) -> int:
    """Majority vote: pick candidate whose answer matches the most common answer."""
    answers = [c.extracted_answer for c in candidates if c.extracted_answer is not None]
    if not answers:
        return 0
    majority_answer = Counter(answers).most_common(1)[0][0]
    for i, c in enumerate(candidates):
        if c.extracted_answer == majority_answer:
            return i
    return 0


def select_plurality_confidence(candidates: List[CandidateResult], features: List[SelectorFeatures]) -> int:
    """Majority vote, but only override greedy if plurality is strong (>50% of candidates)."""
    answers = [c.extracted_answer for c in candidates if c.extracted_answer is not None]
    if not answers:
        return 0
    counts = Counter(answers)
    majority_answer, majority_count = counts.most_common(1)[0]
    # Only override greedy if majority is >50% of valid candidates
    if majority_count > len(answers) / 2:
        for i, c in enumerate(candidates):
            if c.extracted_answer == majority_answer:
                return i
    return 0  # fall back to greedy


def select_consistency_filtered(candidates: List[CandidateResult], features: List[SelectorFeatures]) -> int:
    """Filter out truncated/looping candidates, then majority vote on remainder."""
    valid = [(i, c) for i, (c, f) in enumerate(zip(candidates, features))
             if f.no_truncation and f.no_loops and f.answer_exists]
    if not valid:
        return select_majority(candidates, features)
    answers = [c.extracted_answer for _, c in valid]
    majority_answer = Counter(answers).most_common(1)[0][0]
    for i, c in valid:
        if c.extracted_answer == majority_answer:
            return i
    return valid[0][0]


def select_scratchpad_majority(candidates: List[CandidateResult], features: List[SelectorFeatures]) -> int:
    """Filter to scratchpad-consistent candidates, then majority vote."""
    consistent = [(i, c) for i, (c, f) in enumerate(zip(candidates, features))
                  if f.scratchpad_consistent and f.answer_exists]
    if not consistent:
        return select_majority(candidates, features)
    answers = [c.extracted_answer for _, c in consistent]
    majority_answer = Counter(answers).most_common(1)[0][0]
    for i, c in consistent:
        if c.extracted_answer == majority_answer:
            return i
    return consistent[0][0]


def select_grounded_majority(candidates: List[CandidateResult], features: List[SelectorFeatures]) -> int:
    """Filter to prompt-grounded candidates, then majority vote."""
    grounded = [(i, c) for i, (c, f) in enumerate(zip(candidates, features))
                if f.prompt_grounded and f.answer_exists]
    if not grounded:
        return select_majority(candidates, features)
    answers = [c.extracted_answer for _, c in grounded]
    majority_answer = Counter(answers).most_common(1)[0][0]
    for i, c in grounded:
        if c.extracted_answer == majority_answer:
            return i
    return grounded[0][0]


def select_composite(candidates: List[CandidateResult], features: List[SelectorFeatures]) -> int:
    """Composite selector: score each candidate by multiple features, pick highest.
    Falls back to majority if no clear winner."""
    scores = []
    for i, (c, f) in enumerate(zip(candidates, features)):
        if not f.answer_exists:
            scores.append(-1.0)
            continue
        score = 0.0
        score += f.answer_frequency * 2.0  # strong signal from agreement
        score += 1.0 if f.scratchpad_consistent else 0.0
        score += 1.0 if f.prompt_grounded else 0.0
        score += 0.5 if f.no_truncation else 0.0
        score += 0.5 if f.no_loops else 0.0
        score += 0.5 if f.agrees_with_greedy else 0.0
        scores.append(score)
    return max(range(len(scores)), key=lambda i: scores[i])


SELECTORS = {
    "greedy": select_greedy,
    "oracle": select_oracle,
    "majority": select_majority,
    "plurality_confidence": select_plurality_confidence,
    "consistency_filtered": select_consistency_filtered,
    "scratchpad_majority": select_scratchpad_majority,
    "grounded_majority": select_grounded_majority,
    "composite": select_composite,
}


# =====================================================================
# Generation — produce all candidates for all tasks
# =====================================================================

def _append_task_candidates(path: Path, task_id: str, candidates: List[CandidateResult]) -> None:
    """Append one task's candidates to incremental JSONL (crash-safe)."""
    with open(path, "a", encoding="utf-8") as f:
        for c in candidates:
            record = {
                "task_id": c.task_id,
                "perturbation_idx": c.perturbation_idx,
                "extracted_answer": c.extracted_answer,
                "correct": c.correct,
                "token_count": c.token_count,
                "has_eos": c.has_eos,
                "truncated": c.truncated,
                "has_think_tags": c.has_think_tags,
                "response_length": len(c.stripped_response),
                "raw_length": len(c.raw_output),
                "all_integers": c.all_integers,
                "prompt_integers": c.prompt_integers,
                "stripped_response": c.stripped_response,
                "raw_output": c.raw_output,
            }
            f.write(json.dumps(record, default=str) + "\n")


def generate_candidates(
    encoder: LLMEncoder,
    tasks: List[Task],
    decode_cfg: DecodeConfig,
    k: int = 20,
    noise_scale: float = 0.1,
    curvature: float = 0.5,
    seed: int = 42,
    geometry: str = "hyperbolic",
    incremental_path: Optional[Path] = None,
    resume_existing: Optional[Dict[str, List[CandidateResult]]] = None,
) -> Dict[str, List[CandidateResult]]:
    """Generate greedy baseline + k perturbation candidates per task.

    Returns dict mapping task_id -> list of CandidateResult (index 0 = greedy).
    If incremental_path is set, writes candidates task-by-task for crash safety.
    If resume_existing is set, skips tasks already present in it.
    """
    d_latent = encoder.latent_dim
    ball_radius = (1.0 / math.sqrt(curvature)) * 0.95
    rng = torch.Generator()

    results: Dict[str, List[CandidateResult]] = {}
    if resume_existing:
        results.update(resume_existing)

    for t_idx, task in enumerate(tasks):
        if task.task_id in results:
            n_correct = sum(1 for c in results[task.task_id] if c.correct)
            print(
                f"[{t_idx + 1}/{len(tasks)}] {task.task_id}: "
                f"{n_correct}/{len(results[task.task_id])} correct (RESUMED)",
                flush=True,
            )
            continue

        prompt_ints = _parse_integers(task.prompt)
        candidates = []

        # 0: Greedy baseline (zero latent)
        zero_latent = torch.zeros(1, d_latent, device=encoder._device)
        response, raw = decode_latent(encoder, zero_latent, task.prompt, decode_cfg)
        candidates.append(_make_candidate(task, 0, response, raw, prompt_ints, decode_cfg.max_new_tokens))

        # 1..k: Perturbations
        for p in range(k):
            pert_seed = int(hashlib.sha256(f"{seed}:{task.task_id}:{p}".encode()).hexdigest()[:8], 16)
            rng.manual_seed(pert_seed)
            noise = _make_noise(
                (1, d_latent), noise_scale, d_latent, rng, device=encoder._device,
            )
            latent = _apply_mutation(
                zero_latent, noise, curvature, ball_radius, geometry=geometry,
            )
            response, raw = decode_latent(encoder, latent, task.prompt, decode_cfg)
            candidates.append(_make_candidate(task, p + 1, response, raw, prompt_ints, decode_cfg.max_new_tokens))

        results[task.task_id] = candidates

        if incremental_path:
            _append_task_candidates(incremental_path, task.task_id, candidates)

        n_correct = sum(1 for c in candidates if c.correct)
        print(
            f"[{t_idx + 1}/{len(tasks)}] {task.task_id}: "
            f"{n_correct}/{len(candidates)} correct, "
            f"greedy={'OK' if candidates[0].correct else 'FAIL'}",
            flush=True,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


_tokenizer_ref = None

def _make_candidate(
    task: Task, idx: int, response: str, raw: str,
    prompt_ints: List[int], max_tokens: int,
) -> CandidateResult:
    """Build a CandidateResult from a decoded response."""
    all_ints = _parse_integers(response)
    answer = all_ints[-1] if all_ints else None
    correct = answer == task.correct_answer if answer is not None else False

    # Use tokenizer for accurate token count if available
    if _tokenizer_ref is not None:
        token_count = len(_tokenizer_ref.encode(raw, add_special_tokens=False))
    else:
        token_count = len(raw) // 4
    truncated = token_count >= max_tokens - 5
    has_eos = not truncated
    has_think = "<think>" in raw

    return CandidateResult(
        task_id=task.task_id,
        perturbation_idx=idx,
        raw_output=raw,
        stripped_response=response,
        extracted_answer=answer,
        correct=correct,
        token_count=token_count,
        has_eos=has_eos,
        truncated=truncated,
        has_think_tags=has_think,
        all_integers=all_ints,
        prompt_integers=prompt_ints,
    )


# =====================================================================
# Temperature baseline — best-of-N with temperature sampling
# =====================================================================

def generate_temperature_candidates(
    encoder: LLMEncoder,
    tasks: List[Task],
    decode_cfg: DecodeConfig,
    k: int = 10,
    temperature: float = 0.7,
) -> Dict[str, List[CandidateResult]]:
    """Generate k candidates using temperature sampling (no perturbation)."""
    temp_cfg = DecodeConfig(
        mode=decode_cfg.mode,
        geometry=decode_cfg.geometry,
        curvature=decode_cfg.curvature,
        W_soft=decode_cfg.W_soft,
        embed_dim=decode_cfg.embed_dim,
        num_soft_tokens=decode_cfg.num_soft_tokens,
        target_rms=decode_cfg.target_rms,
        max_new_tokens=decode_cfg.max_new_tokens,
        temperature=temperature,
        enable_thinking=decode_cfg.enable_thinking,
    )

    d_latent = encoder.latent_dim
    zero_latent = torch.zeros(1, d_latent, device=encoder._device)
    results: Dict[str, List[CandidateResult]] = {}

    for t_idx, task in enumerate(tasks):
        prompt_ints = _parse_integers(task.prompt)
        candidates = []
        for s in range(k):
            torch.manual_seed(42_000 + t_idx * 100 + s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42_000 + t_idx * 100 + s)
            response, raw = decode_latent(encoder, zero_latent, task.prompt, temp_cfg)
            candidates.append(_make_candidate(task, s, response, raw, prompt_ints, temp_cfg.max_new_tokens))
        results[task.task_id] = candidates
        n_correct = sum(1 for c in candidates if c.correct)
        print(f"[temp {t_idx + 1}/{len(tasks)}] {task.task_id}: {n_correct}/{k} correct", flush=True)

    return results


# =====================================================================
# Analysis — apply selectors and compute metrics
# =====================================================================

def apply_selectors(
    all_candidates: Dict[str, List[CandidateResult]],
    k_values: List[int] = [3, 5, 10, 20],
) -> Dict[str, Dict[str, dict]]:
    """Apply all frozen selectors at each k value.

    Returns: {k: {selector_name: {accuracy, correct_tasks, n_tasks, selected_answers}}}
    """
    import random as pyrandom
    rng = pyrandom.Random(42)
    results = {}

    for k in k_values:
        k_results = {}
        for sel_name, sel_fn in SELECTORS.items():
            correct = 0
            total = 0
            details = []

            for task_id, candidates in all_candidates.items():
                # Use first k+1 candidates (0=greedy + k perturbations)
                subset = candidates[:k + 1]
                features = [
                    compute_selector_features(c, subset, subset[0].extracted_answer)
                    for c in subset
                ]
                if sel_name == "random":
                    idx = select_random(subset, features, rng)
                elif sel_name in ("greedy", "oracle"):
                    idx = sel_fn(subset, features)
                else:
                    idx = sel_fn(subset, features)

                selected = subset[idx]
                total += 1
                if selected.correct:
                    correct += 1
                details.append({
                    "task_id": task_id,
                    "selected_idx": idx,
                    "selected_answer": selected.extracted_answer,
                    "correct": selected.correct,
                })

            k_results[sel_name] = {
                "accuracy": correct / total if total > 0 else 0.0,
                "correct": correct,
                "total": total,
                "details": details,
            }

        # Add random selector
        correct = 0
        total = 0
        for task_id, candidates in all_candidates.items():
            subset = candidates[:k + 1]
            perturbed = subset[1:]
            if perturbed:
                selected = rng.choice(perturbed)
            else:
                selected = subset[0]
            total += 1
            if selected.correct:
                correct += 1
        k_results["random"] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        }

        # Random mean (expected accuracy over all perturbations)
        correct_sum = 0
        count_sum = 0
        for task_id, candidates in all_candidates.items():
            perturbed = candidates[1:k + 1]
            for c in perturbed:
                count_sum += 1
                if c.correct:
                    correct_sum += 1
        k_results["random_mean"] = {
            "accuracy": correct_sum / count_sum if count_sum > 0 else 0.0,
            "correct": correct_sum,
            "total": count_sum,
        }

        results[str(k)] = k_results

    return results


def compute_diagnostics(
    all_candidates: Dict[str, List[CandidateResult]],
    k: int = 10,
) -> dict:
    """Compute key diagnostics: oracle headroom over majority, task categories."""
    diagnostics = {
        "k": k,
        "oracle_not_majority_count": 0,
        "oracle_and_majority_count": 0,
        "neither_count": 0,
        "greedy_correct_count": 0,
        "task_categories": [],
    }

    for task_id, candidates in all_candidates.items():
        subset = candidates[:k + 1]
        greedy = subset[0]
        answers = [c.extracted_answer for c in subset if c.extracted_answer is not None]
        majority_answer = Counter(answers).most_common(1)[0][0] if answers else None
        has_correct = any(c.correct for c in subset)

        majority_is_right = any(
            c.correct and c.extracted_answer == majority_answer
            for c in subset
        )
        oracle_exists = has_correct

        if oracle_exists and majority_is_right:
            diagnostics["oracle_and_majority_count"] += 1
            cat = "both_correct"
        elif oracle_exists and not majority_is_right:
            diagnostics["oracle_not_majority_count"] += 1
            cat = "oracle_only"
        elif not oracle_exists:
            diagnostics["neither_count"] += 1
            cat = "neither"
        else:
            cat = "majority_only"

        if greedy.correct:
            diagnostics["greedy_correct_count"] += 1

        diagnostics["task_categories"].append({
            "task_id": task_id,
            "category": cat,
            "greedy_correct": greedy.correct,
            "majority_answer": majority_answer,
            "majority_correct": majority_is_right,
            "oracle_exists": oracle_exists,
            "n_correct_candidates": sum(1 for c in subset if c.correct),
        })

    return diagnostics


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Selector Realization Study — Phase 1: Arithmetic")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--curvature", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--task-source", choices=["nested", "gsm8k"], default="nested",
                        help="Task source: nested arithmetic expressions or GSM8K word problems")
    parser.add_argument("--no-temperature-baseline", action="store_true")
    parser.add_argument("--geometry", choices=["hyperbolic", "euclidean"], default="hyperbolic",
                        help="Noise geometry: hyperbolic (Poincare ball) or euclidean (L2 ball)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Resume from candidates JSONL")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path("eval_results/selector_study")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace("/", "_")

    print("=" * 70, flush=True)
    print("SELECTOR REALIZATION STUDY — Phase 1: Arithmetic", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model} ({args.quantization})", flush=True)
    print(f"k={args.k} perturbations, {args.n_test} test tasks", flush=True)
    print(f"noise_scale={args.noise_scale}, curvature={args.curvature}, geometry={args.geometry}", flush=True)
    print(f"max_new_tokens={args.max_new_tokens}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(flush=True)

    # Generate tasks
    print(f"Generating tasks (source={args.task_source})...", flush=True)
    if args.task_source == "gsm8k":
        _, test_tasks = load_gsm8k_tasks(
            n_test=args.n_test, n_train=50, seed=args.seed,
        )
    else:
        _, test_tasks = generate_nested_expression_tasks(
            n_train=50, n_test=args.n_test,
            seed=args.seed, difficulty="sweet_spot",
        )
    print(f"Test tasks: {len(test_tasks)}", flush=True)

    # Load model
    print("Loading model...", flush=True)
    encoder = LLMEncoder(
        model_name=args.model,
        quantization=args.quantization,
    )
    cal = auto_calibrate(encoder)
    print(f"Calibration: embed_dim={cal['embed_dim']}, rms={cal['embedding_rms']:.5f}", flush=True)

    if not check_soft_prompt_compatibility(encoder):
        print("ERROR: Model does not support inputs_embeds.", flush=True)
        sys.exit(1)

    global _tokenizer_ref
    _tokenizer_ref = encoder.tokenizer

    # Create projection matrix
    d_latent = encoder.latent_dim
    num_soft_tokens = 8
    d_out = num_soft_tokens * cal["embed_dim"]
    W = make_row_orthonormal_W(d_latent, d_out, seed=1234).to(encoder._device)

    decode_cfg = DecodeConfig(
        mode=DecodeMode.SOFT_PROMPT,
        geometry=args.geometry,
        curvature=args.curvature,
        W_soft=W,
        embed_dim=cal["embed_dim"],
        num_soft_tokens=num_soft_tokens,
        target_rms=cal["embedding_rms"],
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,  # GREEDY — Codex mandated explicit temp=0
        enable_thinking=not args.no_think,
    )

    # Phase 1: Generate all candidates (with incremental crash-safe saving)
    print("\n--- Phase 1: Generating perturbation candidates ---", flush=True)
    t0 = time.time()

    candidates_path = output_dir / f"candidates_{model_tag}_{timestamp}.jsonl"
    resume_existing = None

    if args.resume:
        print(f"Resuming from {args.resume}...", flush=True)
        resume_existing = _load_candidates(args.resume, test_tasks)
        print(f"  Loaded {len(resume_existing)} completed tasks", flush=True)
    else:
        # Check for previous incremental file to auto-resume
        existing_jsonl = sorted(output_dir.glob("candidates_*.jsonl"))
        if existing_jsonl:
            latest = existing_jsonl[-1]
            try:
                resume_existing = _load_candidates(str(latest), test_tasks)
                if resume_existing:
                    print(f"Auto-resuming from {latest.name} ({len(resume_existing)} tasks)", flush=True)
                    candidates_path = latest
            except Exception as e:
                print(f"Could not resume from {latest}: {e}", flush=True)
                resume_existing = None

    all_candidates = generate_candidates(
        encoder, test_tasks, decode_cfg,
        k=args.k, noise_scale=args.noise_scale,
        curvature=args.curvature, seed=args.seed,
        geometry=args.geometry,
        incremental_path=candidates_path,
        resume_existing=resume_existing,
    )

    gen_time = time.time() - t0
    print(f"\nGeneration complete in {gen_time:.1f}s", flush=True)

    # If we used incremental saving, the file already has all data.
    # Write a clean final copy if we resumed (incremental may have duplicates).
    if resume_existing:
        final_path = output_dir / f"candidates_{model_tag}_{timestamp}_final.jsonl"
        _save_candidates(all_candidates, final_path)
        print(f"Saved final candidates to {final_path}", flush=True)
    else:
        print(f"Candidates saved incrementally to {candidates_path}", flush=True)

    # Phase 2: Temperature baseline (if enabled)
    temp_candidates = None
    if not args.no_temperature_baseline:
        print("\n--- Phase 2: Temperature baseline (k=10, temp=0.7) ---", flush=True)
        t0 = time.time()
        temp_candidates = generate_temperature_candidates(
            encoder, test_tasks, decode_cfg, k=10, temperature=0.7,
        )
        temp_time = time.time() - t0
        print(f"Temperature baseline complete in {temp_time:.1f}s", flush=True)
        temp_path = output_dir / f"temp_candidates_{model_tag}_{timestamp}.jsonl"
        _save_candidates(temp_candidates, temp_path)
        print(f"Saved temperature candidates to {temp_path}", flush=True)

    # Phase 3: Apply selectors
    print("\n--- Phase 3: Applying selectors ---", flush=True)
    k_values = [k for k in [3, 5, 10, 20] if k <= args.k]
    selector_results = apply_selectors(all_candidates, k_values)

    # Temperature baseline majority vote
    if temp_candidates:
        temp_correct = 0
        temp_total = 0
        for task_id, candidates in temp_candidates.items():
            answers = [c.extracted_answer for c in candidates if c.extracted_answer is not None]
            if answers:
                majority = Counter(answers).most_common(1)[0][0]
                # Check if majority is correct
                correct_ans = [t.correct_answer for t in test_tasks if t.task_id == task_id][0]
                if majority == correct_ans:
                    temp_correct += 1
            temp_total += 1
        selector_results["temp_baseline"] = {
            "accuracy": temp_correct / temp_total if temp_total > 0 else 0.0,
            "correct": temp_correct,
            "total": temp_total,
            "method": "majority_vote",
            "k": 10,
            "temperature": 0.7,
        }

    # Phase 4: Diagnostics
    print("\n--- Phase 4: Diagnostics ---", flush=True)
    diagnostics = compute_diagnostics(all_candidates, k=min(10, args.k))

    # Phase 5: Print results
    print("\n" + "=" * 70, flush=True)
    print("RESULTS", flush=True)
    print("=" * 70, flush=True)

    headline_k = str(max(k_values))
    greedy_acc = selector_results.get(headline_k, {}).get("greedy", {}).get("accuracy", 0)
    oracle_acc = selector_results.get(headline_k, {}).get("oracle", {}).get("accuracy", 0)
    majority_acc = selector_results.get(headline_k, {}).get("majority", {}).get("accuracy", 0)

    print(f"\nGreedy baseline:    {greedy_acc:.1%}", flush=True)
    print(f"Oracle (k={headline_k}):    {oracle_acc:.1%}", flush=True)
    print(f"Majority (k={headline_k}):  {majority_acc:.1%}", flush=True)
    print(f"Oracle headroom over majority: {oracle_acc - majority_acc:.1%}", flush=True)
    print(f"Oracle-not-majority tasks: {diagnostics['oracle_not_majority_count']}", flush=True)

    print("\n--- Selector accuracy by k ---", flush=True)
    for k_str in sorted(selector_results.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        if k_str == "temp_baseline":
            continue
        print(f"\nk={k_str}:", flush=True)
        k_data = selector_results[k_str]
        for sel_name in ["greedy", "random_mean", "majority", "plurality_confidence",
                         "consistency_filtered", "scratchpad_majority", "grounded_majority",
                         "composite", "oracle"]:
            if sel_name in k_data:
                acc = k_data[sel_name]["accuracy"]
                n = k_data[sel_name].get("correct", 0)
                print(f"  {sel_name:25s}: {acc:.1%} ({n}/{k_data[sel_name].get('total', '?')})", flush=True)

    if "temp_baseline" in selector_results:
        tb = selector_results["temp_baseline"]
        print(f"\nTemperature baseline (k=10, temp=0.7): {tb['accuracy']:.1%} ({tb['correct']}/{tb['total']})", flush=True)

    # Oracle recovery rates
    print("\n--- Oracle Recovery (over majority) ---", flush=True)
    for k_str in sorted(selector_results.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        if k_str == "temp_baseline":
            continue
        k_data = selector_results[k_str]
        maj = k_data.get("majority", {}).get("accuracy", 0)
        orc = k_data.get("oracle", {}).get("accuracy", 0)
        headroom = orc - maj
        if headroom > 0:
            for sel_name in ["plurality_confidence", "consistency_filtered",
                             "scratchpad_majority", "grounded_majority", "composite"]:
                if sel_name in k_data:
                    sel_acc = k_data[sel_name]["accuracy"]
                    recovery = (sel_acc - maj) / headroom if headroom > 0 else 0
                    print(f"  k={k_str} {sel_name:25s}: recovery={recovery:+.1%}", flush=True)

    # Save full results
    results_path = output_dir / f"selector_results_{model_tag}_{timestamp}.json"
    full_results = {
        "metadata": {
            "model": args.model,
            "quantization": args.quantization,
            "k": args.k,
            "n_test": args.n_test,
            "noise_scale": args.noise_scale,
            "curvature": args.curvature,
            "geometry": args.geometry,
            "task_source": args.task_source,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "seed": args.seed,
            "timestamp": timestamp,
            "candidates_file": str(candidates_path),
        },
        "selector_results": {
            k: {
                sel: {key: val for key, val in data.items() if key != "details"}
                for sel, data in k_data.items()
            }
            for k, k_data in selector_results.items()
        },
        "diagnostics": {
            k: v for k, v in diagnostics.items() if k != "task_categories"
        },
        "task_categories": diagnostics["task_categories"],
    }
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}", flush=True)

    print("\nDone.", flush=True)


def _save_candidates(
    all_candidates: Dict[str, List[CandidateResult]],
    path: Path,
) -> None:
    """Save all candidates to JSONL (one line per candidate)."""
    with open(path, "w", encoding="utf-8") as f:
        for task_id, candidates in all_candidates.items():
            for c in candidates:
                record = {
                    "task_id": c.task_id,
                    "perturbation_idx": c.perturbation_idx,
                    "extracted_answer": c.extracted_answer,
                    "correct": c.correct,
                    "token_count": c.token_count,
                    "has_eos": c.has_eos,
                    "truncated": c.truncated,
                    "has_think_tags": c.has_think_tags,
                    "response_length": len(c.stripped_response),
                    "raw_length": len(c.raw_output),
                    "all_integers": c.all_integers,
                    "prompt_integers": c.prompt_integers,
                    "stripped_response": c.stripped_response,
                    "raw_output": c.raw_output,
                }
                f.write(json.dumps(record, default=str) + "\n")


def _load_candidates(path: str, tasks: List[Task]) -> Dict[str, List[CandidateResult]]:
    """Load candidates from JSONL for resume."""
    task_map = {t.task_id: t for t in tasks}
    results: Dict[str, List[CandidateResult]] = {}
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            tid = record["task_id"]
            if tid not in results:
                results[tid] = []
            task = task_map.get(tid)
            correct = record.get("correct", False)
            results[tid].append(CandidateResult(
                task_id=tid,
                perturbation_idx=record["perturbation_idx"],
                raw_output=record.get("stripped_response", ""),
                stripped_response=record.get("stripped_response", ""),
                extracted_answer=record.get("extracted_answer"),
                correct=correct,
                token_count=record.get("token_count", 0),
                has_eos=record.get("has_eos", True),
                truncated=record.get("truncated", False),
                has_think_tags=record.get("has_think_tags", False),
                all_integers=record.get("all_integers", []),
                prompt_integers=record.get("prompt_integers", []),
            ))
    return results


if __name__ == "__main__":
    main()
