"""
Verifiable Evolution V14 - Dual Steering Decode (Input + Output Conditioning)

Adds logit-level Newton steering alongside V13's soft prompt conditioning.
Based on "The Information Geometry of Softmax" (arXiv:2602.15293, Feb 2026).

Two complementary conditioning channels:
- Input channel (V13): Soft prompt tokens shape what the model ATTENDS to
- Output channel (V14): Dual steering shapes what the model OUTPUTS

The paper proves that naive vector addition in logit/representation space
commits a geometric type error. The correct approach is a regularized Newton
step in the dual (probability) coordinate system.

Algorithm per token:
1. p = softmax(logits)
2. Woodbury solve: v = (diag(p+alpha) - pp^T)^{-1} @ omega_W  [O(V)]
3. logits_steered = logits + eta * v/||v||
4. KL safety cap: if KL(p || softmax(steered)) > cap, auto-downscale eta

Codex review fixes (C -> targeting A-):
1. Use encoder.latent_dim everywhere (not hardcoded 1024)
2. No epsilon parameter (mathematically inert due to normalization)
3. No tanh_squash in steering branch (only NaN guard + final L2-norm)
4. KL cap with automatic eta downscaling (prevents temperature interaction)
5. Bonferroni p < 0.025 for 2 comparisons (not 0.05)

Three conditions:
C1) hyp_mobius_rng: Hyperbolic Mobius evolution + RNG-seed decode (V12 best)
C2) hyp_mobius_softprompt: Hyperbolic Mobius evolution + soft prompt (V13)
C3) hyp_mobius_softprompt_dual: Soft prompt + dual Newton steering (V14 new)

Pre-registered primary: C2 vs C3 (does dual steering add value?)
Pre-registered secondary: C1 vs C3 (full V14 system vs RNG baseline)
Bonferroni threshold: p < 0.025 (2 comparisons)
Pre-registered curvature: c=0.5 (unchanged from V7)

Inherits ALL V11/V12/V13 fixes.

Calibration values for Qwen3-4B:
- Embed dim: 2560, Vocab: 151936
- Element RMS: 0.02195, Mean token L2 norm: 1.097
- r_max (logmap squash): 1.405
"""

import argparse
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from scipy import stats as sp_stats
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.utils import hyperbolic as hyp

from dual_steering import (
    DualSteeringProcessor,
    compute_steering_direction,
    make_steer_projection,
)


# =====================================================================
# Calibration constants (measured from Qwen3-4B embedding table)
# =====================================================================
EMBED_DIM = 2560
NUM_SOFT_TOKENS = 8
SOFT_PROMPT_FLAT_DIM = NUM_SOFT_TOKENS * EMBED_DIM  # 20480
EMBEDDING_RMS_ELEM = 0.02195  # Per-element RMS of Qwen3-4B embeddings
MEAN_TOKEN_L2_NORM = 1.097    # Mean L2 norm of real token embeddings

# Steering hyperparameters (Codex-reviewed)
DEFAULT_ETA = 0.05      # Conservative start (Codex: "0.1 aggressive at temp=0.3")
DEFAULT_ALPHA = 0.01    # Tikhonov regularization
DEFAULT_KL_CAP = 0.5    # Per-token KL divergence safety cap


# =====================================================================
# Task generation (identical to V11/V12/V13)
# =====================================================================

@dataclass
class Task:
    task_id: str
    prompt: str
    correct_answer: int
    depth: int


def generate_all_unique_tasks(branching: int, depths: list) -> dict:
    """Enumerate ALL unique tasks per depth."""
    tasks_by_depth = {}
    for depth in depths:
        tasks = []
        for i, path in enumerate(product(range(branching), repeat=depth)):
            path_list = list(path)
            answer = sum(path_list) * (depth + 1) + depth * 7
            prompt = (
                f"Calculate: sum([{','.join(map(str, path_list))}]) * {depth + 1}"
                f" + {depth} * 7 = ?\nAnswer with just the number."
            )
            tasks.append(Task(
                task_id=f"d{depth}_u{i}",
                prompt=prompt,
                correct_answer=answer,
                depth=depth,
            ))
        tasks_by_depth[depth] = tasks
    return tasks_by_depth


def split_train_test(
    tasks_by_depth: dict,
    n_test_per_depth: int,
    n_train_per_depth: int,
    seed: int = 7777,
) -> Tuple[list, list]:
    """Deterministically split into NON-OVERLAPPING train and test sets."""
    rng = random.Random(seed)
    test_tasks = []
    train_tasks = []
    for depth in sorted(tasks_by_depth.keys()):
        pool = tasks_by_depth[depth][:]
        rng.shuffle(pool)
        n_needed = n_test_per_depth + n_train_per_depth
        if len(pool) < n_needed:
            raise ValueError(
                f"Depth {depth}: need {n_needed} but only {len(pool)}. "
                f"Increase branching."
            )
        test_tasks.extend(pool[:n_test_per_depth])
        train_tasks.extend(pool[n_test_per_depth:n_needed])
    return train_tasks, test_tasks


# =====================================================================
# Answer verification (identical to V11/V12/V13)
# =====================================================================

def verify_answer(response: str, expected: int) -> bool:
    numbers = re.findall(r'-?\d+', response)
    if not numbers:
        return False
    return int(numbers[-1]) == expected


def dense_score(response: str, expected: int) -> float:
    numbers = re.findall(r'-?\d+', response)
    if not numbers:
        return 0.0
    last_num = int(numbers[-1])
    if last_num == expected:
        return 1.0
    distance = abs(last_num - expected)
    return min(1.0 / (1.0 + distance), 0.99)


# =====================================================================
# Fixed Orthogonal Projection (V13, unchanged)
# =====================================================================

def make_row_orthonormal_W(d_latent: int, d_out: int,
                            seed: int = 1234) -> Tensor:
    """Create a fixed row-orthonormal projection matrix W.

    W has shape (d_latent, d_out) with orthonormal rows: W W^T = I.
    """
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d_out, d_latent, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A, mode="reduced")  # Q: (d_out, d_latent)
    W = Q.T.contiguous()  # W: (d_latent, d_out), rows are orthonormal
    return W


def radial_tanh_squash(v: Tensor, r_max: float, eps: float = 1e-8) -> Tensor:
    """Smooth radial squash to prevent boundary blow-up in logmap0 output."""
    r = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    r_new = r_max * torch.tanh(r / r_max)
    return v * (r_new / r)


def latent_to_soft_prompt(
    latent: Tensor,
    W: Tensor,
    curvature: float,
    d_latent: int,
    target_rms: float = EMBEDDING_RMS_ELEM,
) -> Tensor:
    """Convert a Poincare ball latent to soft prompt embeddings.

    Pipeline: logmap0 -> NaN guard -> tanh squash -> W project -> reshape -> RMS cal
    """
    lat = latent.squeeze().float()

    # 1. Map to tangent space
    tangent = hyp.logmap0(lat, curvature)

    # 2. NaN guard
    tangent = torch.nan_to_num(tangent, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Compute r_max from latent dim (not hardcoded)
    r_ref = math.sqrt(d_latent) * EMBEDDING_RMS_ELEM
    r_max = 2.0 * r_ref

    # 4. Radial tanh squash
    tangent = radial_tanh_squash(tangent, r_max)

    # 5. Project (preserve inner products) - ensure same device
    W_dev = W.to(device=tangent.device, dtype=tangent.dtype)
    flat = tangent @ W_dev  # (d_latent,) @ (d_latent, 20480) -> (20480,)

    # 6. Reshape to token sequence
    soft_prompt = flat.view(NUM_SOFT_TOKENS, EMBED_DIM)

    # 7. Scale: match per-element RMS to real token embeddings
    current_rms = soft_prompt.square().mean().sqrt().clamp_min(1e-8)
    soft_prompt = soft_prompt * (target_rms / current_rms)

    return soft_prompt.unsqueeze(0)  # (1, 8, 2560)


def decode_with_projection(
    encoder: LLMEncoder,
    latent: Tensor,
    query: str,
    W: Tensor,
    curvature: float,
    d_latent: int,
    max_new_tokens: int = 250,
    temperature: float = 0.3,
    logits_processor_list=None,
) -> str:
    """Decode using soft prompt, optionally with logits processors (for steering).

    Args:
        logits_processor_list: Optional list of LogitsProcessor instances.
            For V14 dual steering, pass [DualSteeringProcessor(...)].
    """
    # Generate soft prompt embeddings
    with torch.no_grad():
        soft_prompt = latent_to_soft_prompt(latent, W, curvature, d_latent)
        soft_prompt = soft_prompt.to(encoder.model.dtype).to(encoder._device)

    # Build the text prompt
    system_msg = "Answer to the best of your ability."
    user_msg = query if query else ""

    if hasattr(encoder.tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        try:
            prompt = encoder.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
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

    with torch.no_grad():
        text_embeds = encoder.model.get_input_embeddings()(inputs["input_ids"])
        combined_embeds = torch.cat([soft_prompt, text_embeds], dim=1)

        soft_mask = torch.ones(
            1, soft_prompt.size(1),
            dtype=inputs["attention_mask"].dtype,
            device=encoder._device,
        )
        combined_mask = torch.cat([soft_mask, inputs["attention_mask"]], dim=1)

        gen_kwargs = dict(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=encoder.tokenizer.pad_token_id,
            eos_token_id=encoder.tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )

        if logits_processor_list:
            gen_kwargs["logits_processor"] = logits_processor_list

        if temperature < 0.01:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
            gen_kwargs["top_k"] = 50

        outputs = encoder.model.generate(**gen_kwargs)

    prompt_len = combined_embeds.size(1)
    generated_ids = outputs[0, prompt_len:]
    generated = encoder.tokenizer.decode(generated_ids, skip_special_tokens=True)

    response = generated.strip()
    if "<think>" in response and "</think>" in response:
        think_end = response.find("</think>") + len("</think>")
        response = response[think_end:].strip()
    elif response.startswith("<think>"):
        for starter in ["1.", "Step 1", "## Step", "Here's", "Here is"]:
            if starter in response:
                response = response[response.index(starter):]
                break

    return response if response else generated.strip()


# =====================================================================
# Evolution (V12 Mobius mutation, adapted for three decode modes)
# =====================================================================

@dataclass
class Candidate:
    latent: Tensor
    fitness: float = 0.0


def _make_noise(shape, noise_scale: float, dim: int, rng: torch.Generator,
                device=None) -> Tensor:
    per_dim = noise_scale / math.sqrt(max(dim, 1))
    noise = torch.randn(shape, generator=rng) * per_dim
    if device is not None and device != torch.device("cpu"):
        noise = noise.to(device)
    return noise


def _apply_mobius_mutation(parent: Tensor, noise: Tensor, curvature: float,
                           ball_radius: float) -> Tensor:
    """Mobius addition mutation (V12 best operator)."""
    noise_in_ball = hyp.expmap0(noise.squeeze(), curvature)
    mutated = hyp.mobius_add(parent.squeeze(), noise_in_ball, curvature)
    mutated = hyp.project_to_ball(mutated, curvature, 0.95)
    return mutated.unsqueeze(0)


def evaluate_dense(latent, tasks, encoder, curvature, d_latent,
                   W=None, W_steer=None, lm_head_weight=None,
                   eta=DEFAULT_ETA, alpha=DEFAULT_ALPHA, kl_cap=DEFAULT_KL_CAP):
    """Evaluate with dense scoring.

    Decode mode determined by which projections are provided:
    - W=None: RNG-seed decode
    - W only: Soft prompt decode (V13)
    - W + W_steer + lm_head_weight: Soft prompt + dual steering (V14)
    """
    scores = {}
    for task in tasks:
        if W is not None and W_steer is not None and lm_head_weight is not None:
            # V14: Soft prompt + dual steering
            with torch.no_grad():
                omega_W = compute_steering_direction(
                    latent, W_steer, lm_head_weight, curvature, encoder._device,
                )
            processor = DualSteeringProcessor(
                omega_W=omega_W, eta=eta, alpha=alpha, kl_cap=kl_cap,
            )
            response = decode_with_projection(
                encoder, latent, task.prompt, W, curvature, d_latent,
                logits_processor_list=[processor],
            )
        elif W is not None:
            # V13: Soft prompt only
            response = decode_with_projection(
                encoder, latent, task.prompt, W, curvature, d_latent,
            )
        else:
            # V12: RNG-seed decode
            response = encoder.decode(
                latent, query=task.prompt, max_new_tokens=250,
                temperature=0.3, hyperbolic=True, curvature=curvature,
            )
        scores[task.task_id] = dense_score(response, task.correct_answer)
    mean_score = sum(scores.values()) / len(scores) if scores else 0.0
    return mean_score, scores


def evaluate_binary(latent, tasks, encoder, curvature, d_latent,
                    W=None, W_steer=None, lm_head_weight=None,
                    eta=DEFAULT_ETA, alpha=DEFAULT_ALPHA, kl_cap=DEFAULT_KL_CAP):
    """Evaluate with binary scoring."""
    results = {}
    for task in tasks:
        if W is not None and W_steer is not None and lm_head_weight is not None:
            with torch.no_grad():
                omega_W = compute_steering_direction(
                    latent, W_steer, lm_head_weight, curvature, encoder._device,
                )
            processor = DualSteeringProcessor(
                omega_W=omega_W, eta=eta, alpha=alpha, kl_cap=kl_cap,
            )
            response = decode_with_projection(
                encoder, latent, task.prompt, W, curvature, d_latent,
                logits_processor_list=[processor],
            )
        elif W is not None:
            response = decode_with_projection(
                encoder, latent, task.prompt, W, curvature, d_latent,
            )
        else:
            response = encoder.decode(
                latent, query=task.prompt, max_new_tokens=250,
                temperature=0.3, hyperbolic=True, curvature=curvature,
            )
        results[task.task_id] = verify_answer(response, task.correct_answer)
    return results


def run_evolution(
    encoder, train_tasks, seed_latent, curvature=0.5,
    generations=3, population_size=4, tasks_per_gen=8,
    noise_scale=0.1, condition_seed=0, d_latent=None,
    W=None, W_steer=None, lm_head_weight=None,
    eta=DEFAULT_ETA, alpha=DEFAULT_ALPHA, kl_cap=DEFAULT_KL_CAP,
) -> Tuple[Tensor, list]:
    """Run Mobius evolution. Decode mode set by which projections are provided."""
    fitness_curve = []
    dim = seed_latent.numel()
    device = seed_latent.device
    ball_radius = (1.0 / math.sqrt(curvature)) * 0.95

    if d_latent is None:
        d_latent = dim

    # Initialize in hyperbolic space
    target_init_norm = 0.5 * ball_radius
    seed_norm = seed_latent.squeeze().norm().item()
    hyp_target = min(target_init_norm * math.sqrt(curvature), 0.999)
    tangent_norm = math.atanh(hyp_target) / math.sqrt(curvature)
    init_scale = tangent_norm / max(seed_norm, 1e-8)
    seed_latent = hyp.expmap0(
        seed_latent.squeeze() * init_scale, curvature
    ).unsqueeze(0)

    # Isolated RNGs
    mut_rng = torch.Generator()
    mut_rng.manual_seed(condition_seed)
    task_rng = random.Random(condition_seed + 7)

    population = [Candidate(latent=seed_latent.clone())]
    for _ in range(population_size - 1):
        noise = _make_noise(seed_latent.shape, noise_scale, dim, mut_rng, device)
        mutated = _apply_mobius_mutation(seed_latent, noise, curvature, ball_radius)
        population.append(Candidate(latent=mutated))

    global_best = Candidate(latent=seed_latent.clone(), fitness=-1.0)

    for gen in range(generations):
        gen_tasks = task_rng.sample(
            train_tasks, min(tasks_per_gen, len(train_tasks))
        )

        for cand in population:
            score, _ = evaluate_dense(
                cand.latent, gen_tasks, encoder, curvature, d_latent,
                W=W, W_steer=W_steer, lm_head_weight=lm_head_weight,
                eta=eta, alpha=alpha, kl_cap=kl_cap,
            )
            cand.fitness = score

        gen_best = max(population, key=lambda c: c.fitness)
        if gen_best.fitness > global_best.fitness:
            global_best = Candidate(
                latent=gen_best.latent.clone(), fitness=gen_best.fitness,
            )

        fitnesses = [c.fitness for c in population]
        curve_entry = {
            "gen": gen + 1,
            "best": max(fitnesses),
            "mean": sum(fitnesses) / len(fitnesses),
            "min": min(fitnesses),
        }
        fitness_curve.append(curve_entry)

        # Selection + reproduction
        population.sort(key=lambda c: c.fitness, reverse=True)
        n_elite = max(2, population_size // 2)
        elite = population[:n_elite]

        new_pop = [Candidate(latent=e.latent.clone(), fitness=e.fitness)
                    for e in elite]
        while len(new_pop) < population_size:
            parent = elite[task_rng.randint(0, len(elite) - 1)]
            noise = _make_noise(
                parent.latent.shape, noise_scale, dim, mut_rng, device,
            )
            mutated = _apply_mobius_mutation(
                parent.latent, noise, curvature, ball_radius,
            )
            new_pop.append(Candidate(latent=mutated))

        population = new_pop
        print(
            f"  [GEN {gen+1}] best={curve_entry['best']:.3f}"
            f" mean={curve_entry['mean']:.3f}",
            flush=True,
        )

    return global_best.latent, fitness_curve


# =====================================================================
# Statistics (permutation test, Bonferroni-corrected for V14)
# =====================================================================

BONFERRONI_THRESHOLD = 0.025  # 2 pre-registered comparisons: 0.05 / 2

def exact_sign_flip_pvalue(diffs, alternative="greater"):
    """Exact paired sign-flip permutation test."""
    n = len(diffs)
    observed = sum(diffs)

    count_extreme = 0
    total = 2 ** n
    for mask in range(total):
        flipped = 0.0
        for i in range(n):
            sign = 1 if (mask >> i) & 1 else -1
            flipped += sign * diffs[i]
        if alternative == "greater" and flipped >= observed:
            count_extreme += 1
        elif alternative == "less" and flipped <= observed:
            count_extreme += 1
        elif alternative == "two-sided" and abs(flipped) >= abs(observed):
            count_extreme += 1

    return count_extreme / total


def compute_statistics(results_by_condition, task_ids, n_secondary=1):
    """Compute statistics with permutation tests and Bonferroni correction."""
    conditions = list(results_by_condition.keys())
    n_seeds = len(results_by_condition[conditions[0]])

    acc_by_cond = {}
    for cond in conditions:
        acc_by_cond[cond] = [
            sum(r.values()) / len(r) for r in results_by_condition[cond]
        ]

    output = {"per_condition": {}, "pairwise": {}}

    for cond in conditions:
        accs = acc_by_cond[cond]
        output["per_condition"][cond] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs, ddof=1)) if n_seeds > 1 else 0.0,
            "per_seed": accs,
        }

    for i, cond_a in enumerate(conditions):
        for cond_b in conditions[i + 1:]:
            pair_key = f"{cond_a}_vs_{cond_b}"
            accs_a = acc_by_cond[cond_a]
            accs_b = acc_by_cond[cond_b]

            diffs = [b - a for a, b in zip(accs_a, accs_b)]
            mean_diff = float(np.mean(diffs))

            if n_seeds >= 3:
                std_diff = float(np.std(diffs, ddof=1))
                se = std_diff / np.sqrt(len(diffs))
                ci_95 = (
                    mean_diff - 1.96 * se,
                    mean_diff + 1.96 * se,
                )
                p_perm = exact_sign_flip_pvalue(diffs, "greater")
                t_stat, p_t = sp_stats.ttest_rel(accs_b, accs_a)
            else:
                std_diff = float("nan")
                ci_95 = (float("nan"), float("nan"))
                p_perm = float("nan")
                t_stat = float("nan")
                p_t = float("nan")

            # Per-seed McNemar
            per_seed_mcnemar = []
            for seed_idx in range(n_seeds):
                res_a = results_by_condition[cond_a][seed_idx]
                res_b = results_by_condition[cond_b][seed_idx]
                b_cnt = sum(
                    1 for tid in res_a
                    if res_b.get(tid, False) and not res_a[tid]
                )
                c_cnt = sum(
                    1 for tid in res_a
                    if not res_b.get(tid, False) and res_a[tid]
                )
                if b_cnt + c_cnt > 0:
                    chi2 = (abs(b_cnt - c_cnt) - 1) ** 2 / (b_cnt + c_cnt)
                    p_mc = float(1 - sp_stats.chi2.cdf(chi2, 1))
                else:
                    chi2 = 0.0
                    p_mc = 1.0
                per_seed_mcnemar.append({
                    "seed": seed_idx, "b": b_cnt, "c": c_cnt,
                    "chi2": float(chi2), "p": p_mc,
                })

            output["pairwise"][pair_key] = {
                "diff_mean": mean_diff,
                "diff_std": std_diff,
                "diff_ci_95": list(ci_95),
                "p_permutation": float(p_perm),
                "t_stat": float(t_stat),
                "p_ttest": float(p_t),
                "per_seed_mcnemar": per_seed_mcnemar,
                "bonferroni_threshold": BONFERRONI_THRESHOLD,
            }

    # Per-depth
    depth_stats = {}
    for depth in [2, 3]:
        dtasks = [tid for tid in task_ids if tid.startswith(f"d{depth}_")]
        depth_stats[depth] = {}
        for cond in conditions:
            daccs = [
                sum(1 for tid in dtasks if r.get(tid, False))
                / max(len(dtasks), 1)
                for r in results_by_condition[cond]
            ]
            depth_stats[depth][cond] = {
                "mean": float(np.mean(daccs)),
                "std": float(np.std(daccs, ddof=1)) if n_seeds > 1 else 0.0,
            }
    output["per_depth"] = depth_stats

    return output


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V14: Dual Steering Decode (input + output conditioning)"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--test-tasks-per-depth", type=int, default=20)
    parser.add_argument("--train-tasks-per-depth", type=int, default=150)
    parser.add_argument("--branching", type=int, default=15)
    parser.add_argument("--evo-gens", type=int, default=3)
    parser.add_argument("--evo-pop", type=int, default=4)
    parser.add_argument("--evo-tasks", type=int, default=8)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA,
                        help="Steering step size")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Tikhonov regularization")
    parser.add_argument("--kl-cap", type=float, default=DEFAULT_KL_CAP,
                        help="Per-token KL divergence cap")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Run 1 seed for quick sanity check")
    args = parser.parse_args()

    if args.diagnostic:
        args.seeds = 1

    curvature = 0.5
    ball_radius = (1.0 / math.sqrt(curvature)) * 0.95

    print("=" * 70, flush=True)
    print("VERIFIABLE EVOLUTION V14 - DUAL STEERING DECODE", flush=True)
    print("=" * 70, flush=True)
    print("V14 INNOVATION (adds output-channel steering to V13 soft prompt):", flush=True)
    print("  1. V13 soft prompt: input conditioning (shapes attention)", flush=True)
    print("  2. Dual Newton steering: output conditioning (shapes logits)", flush=True)
    print("  3. Regularized Newton step in dual (probability) space", flush=True)
    print("  4. KL safety cap with auto eta downscaling", flush=True)
    print(f"  Steering: eta={args.eta}, alpha={args.alpha}, kl_cap={args.kl_cap}", flush=True)
    print("INHERITED from V11/V12/V13:", flush=True)
    print("  Matched ball radii, RNG isolation, unique tasks, Mobius mutation,", flush=True)
    print("  per-seed McNemar, global best, strict parsing, dense score,", flush=True)
    print("  row-orthonormal projection, tanh squash, RMS calibration", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    b = args.branching
    print(f"Branching: {b} (d2: {b**2}, d3: {b**3} unique)", flush=True)
    print(f"Test: {args.test_tasks_per_depth * 2} ({args.test_tasks_per_depth}/depth)", flush=True)
    print(f"Train: {args.train_tasks_per_depth * 2} ({args.train_tasks_per_depth}/depth)", flush=True)
    print(f"Evolution: {args.evo_gens} gens, pop={args.evo_pop}, tasks/gen={args.evo_tasks}", flush=True)
    print(f"Curvature: {curvature} (pre-registered from V7)", flush=True)
    print(f"Ball radius: {ball_radius:.3f}", flush=True)
    print("Conditions: hyp_mobius_rng, hyp_mobius_softprompt, hyp_mobius_softprompt_dual", flush=True)
    print("Primary: hyp_mobius_softprompt vs hyp_mobius_softprompt_dual", flush=True)
    print("Secondary: hyp_mobius_rng vs hyp_mobius_softprompt_dual", flush=True)
    print(f"Bonferroni threshold: p < {BONFERRONI_THRESHOLD}", flush=True)
    print("=" * 70, flush=True)

    # Generate task pool
    print("\nGenerating unique task pool...", flush=True)
    tasks_by_depth = generate_all_unique_tasks(args.branching, depths=[2, 3])
    for depth, tasks in sorted(tasks_by_depth.items()):
        print(f"  Depth {depth}: {len(tasks)} unique tasks", flush=True)

    train_tasks, test_tasks = split_train_test(
        tasks_by_depth, args.test_tasks_per_depth,
        args.train_tasks_per_depth, seed=7777,
    )
    test_task_ids = [t.task_id for t in test_tasks]

    train_ids = {t.task_id for t in train_tasks}
    test_ids = {t.task_id for t in test_tasks}
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} overlapping tasks!"
    print(f"  Train: {len(train_tasks)}, Test: {len(test_tasks)}, Overlap: 0", flush=True)

    # Load model
    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(model_name=args.model, quantization="4bit")
    d_latent = encoder.latent_dim  # Use actual latent dim (Codex fix #1)
    print(f"  Latent dim: {d_latent}", flush=True)

    # Runtime assertion: check lm_head dtype (Codex recommendation)
    lm_head_weight = encoder.model.lm_head.weight
    print(f"  lm_head dtype: {lm_head_weight.dtype}", flush=True)
    print(f"  lm_head shape: {lm_head_weight.shape}", flush=True)
    assert lm_head_weight.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"lm_head appears quantized ({lm_head_weight.dtype}), steering may fail"
    )

    # Create fixed orthogonal projections
    print("\nCreating projections...", flush=True)

    # V13 soft prompt projection
    W_soft = make_row_orthonormal_W(d_latent=d_latent, d_out=SOFT_PROMPT_FLAT_DIM, seed=1234)
    print(f"  W_soft shape: {W_soft.shape}", flush=True)
    WWT_soft = W_soft @ W_soft.T
    print(f"  W_soft W_soft^T diag check: {WWT_soft.diag()[:3].tolist()}", flush=True)
    print(f"  W_soft W_soft^T off-diag max: {(WWT_soft - torch.eye(d_latent)).abs().max().item():.6f}", flush=True)

    # V14 steering projection
    W_steer = make_steer_projection(d_latent=d_latent, d_hidden=EMBED_DIM, seed=5678)
    print(f"  W_steer shape: {W_steer.shape}", flush=True)
    WWT_steer = W_steer @ W_steer.T
    print(f"  W_steer W_steer^T diag check: {WWT_steer.diag()[:3].tolist()}", flush=True)
    print(f"  W_steer W_steer^T off-diag max: {(WWT_steer - torch.eye(d_latent)).abs().max().item():.6f}", flush=True)

    all_conditions = [
        "hyp_mobius_rng",
        "hyp_mobius_softprompt",
        "hyp_mobius_softprompt_dual",
    ]
    all_results = {c: [] for c in all_conditions}
    all_fitness_curves = {c: [] for c in all_conditions}
    latency_log = {c: [] for c in all_conditions}

    for seed_idx in range(args.seeds):
        seed = 1000 + seed_idx * 111

        print(f"\n{'#' * 70}", flush=True)
        print(f"# SEED {seed_idx + 1}/{args.seeds} (seed={seed})", flush=True)
        print(f"{'#' * 70}", flush=True)

        random.seed(seed)
        torch.manual_seed(seed)

        seed_latent = encoder.encode(
            "You calculate expressions and give numeric answers."
        )

        condition_seed = seed  # SAME for all evolved conditions

        # C1: Hyp Mobius + RNG-seed decode (V12 best)
        print(f"\n[HYP_MOBIUS_RNG] Evolution (RNG-seed decode)...", flush=True)
        t0 = time.time()
        evolved_rng, curve_rng = run_evolution(
            encoder, train_tasks, seed_latent.clone(),
            curvature=curvature,
            generations=args.evo_gens,
            population_size=args.evo_pop,
            tasks_per_gen=args.evo_tasks,
            condition_seed=condition_seed,
            d_latent=d_latent,
            W=None,
        )
        all_fitness_curves["hyp_mobius_rng"].append(curve_rng)

        print(f"\n[HYP_MOBIUS_RNG] Testing...", flush=True)
        rng_res = evaluate_binary(
            evolved_rng, test_tasks, encoder, curvature, d_latent, W=None,
        )
        t1 = time.time()
        latency_log["hyp_mobius_rng"].append(t1 - t0)
        all_results["hyp_mobius_rng"].append(rng_res)
        acc = sum(rng_res.values()) / len(rng_res)
        d2 = sum(1 for t in test_tasks if t.depth == 2 and rng_res[t.task_id]) / args.test_tasks_per_depth
        d3 = sum(1 for t in test_tasks if t.depth == 3 and rng_res[t.task_id]) / args.test_tasks_per_depth
        print(f"  Accuracy: {acc*100:.1f}% (D2: {d2*100:.1f}%, D3: {d3*100:.1f}%) [{t1-t0:.0f}s]", flush=True)

        # C2: Hyp Mobius + soft prompt decode (V13)
        print(f"\n[HYP_MOBIUS_SOFTPROMPT] Evolution (soft prompt decode)...", flush=True)
        t0 = time.time()
        evolved_sp, curve_sp = run_evolution(
            encoder, train_tasks, seed_latent.clone(),
            curvature=curvature,
            generations=args.evo_gens,
            population_size=args.evo_pop,
            tasks_per_gen=args.evo_tasks,
            condition_seed=condition_seed,
            d_latent=d_latent,
            W=W_soft,
        )
        all_fitness_curves["hyp_mobius_softprompt"].append(curve_sp)

        print(f"\n[HYP_MOBIUS_SOFTPROMPT] Testing...", flush=True)
        sp_res = evaluate_binary(
            evolved_sp, test_tasks, encoder, curvature, d_latent, W=W_soft,
        )
        t1 = time.time()
        latency_log["hyp_mobius_softprompt"].append(t1 - t0)
        all_results["hyp_mobius_softprompt"].append(sp_res)
        acc = sum(sp_res.values()) / len(sp_res)
        d2 = sum(1 for t in test_tasks if t.depth == 2 and sp_res[t.task_id]) / args.test_tasks_per_depth
        d3 = sum(1 for t in test_tasks if t.depth == 3 and sp_res[t.task_id]) / args.test_tasks_per_depth
        print(f"  Accuracy: {acc*100:.1f}% (D2: {d2*100:.1f}%, D3: {d3*100:.1f}%) [{t1-t0:.0f}s]", flush=True)

        # C3: Hyp Mobius + soft prompt + dual steering (V14 innovation)
        print(f"\n[HYP_MOBIUS_SOFTPROMPT_DUAL] Evolution (soft prompt + dual steering)...", flush=True)
        t0 = time.time()
        evolved_dual, curve_dual = run_evolution(
            encoder, train_tasks, seed_latent.clone(),
            curvature=curvature,
            generations=args.evo_gens,
            population_size=args.evo_pop,
            tasks_per_gen=args.evo_tasks,
            condition_seed=condition_seed,
            d_latent=d_latent,
            W=W_soft, W_steer=W_steer, lm_head_weight=lm_head_weight,
            eta=args.eta, alpha=args.alpha, kl_cap=args.kl_cap,
        )
        all_fitness_curves["hyp_mobius_softprompt_dual"].append(curve_dual)

        print(f"\n[HYP_MOBIUS_SOFTPROMPT_DUAL] Testing...", flush=True)
        dual_res = evaluate_binary(
            evolved_dual, test_tasks, encoder, curvature, d_latent,
            W=W_soft, W_steer=W_steer, lm_head_weight=lm_head_weight,
            eta=args.eta, alpha=args.alpha, kl_cap=args.kl_cap,
        )
        t1 = time.time()
        latency_log["hyp_mobius_softprompt_dual"].append(t1 - t0)
        all_results["hyp_mobius_softprompt_dual"].append(dual_res)
        acc = sum(dual_res.values()) / len(dual_res)
        d2 = sum(1 for t in test_tasks if t.depth == 2 and dual_res[t.task_id]) / args.test_tasks_per_depth
        d3 = sum(1 for t in test_tasks if t.depth == 3 and dual_res[t.task_id]) / args.test_tasks_per_depth
        print(f"  Accuracy: {acc*100:.1f}% (D2: {d2*100:.1f}%, D3: {d3*100:.1f}%) [{t1-t0:.0f}s]", flush=True)

    # ---- Statistics ----
    print(f"\n{'=' * 70}", flush=True)
    print("STATISTICAL ANALYSIS", flush=True)
    print(f"{'=' * 70}", flush=True)

    stats_result = compute_statistics(all_results, test_task_ids)

    print("\nOverall Accuracy (mean +/- std):", flush=True)
    for cond in all_conditions:
        s = stats_result["per_condition"][cond]
        print(f"  {cond:35s}: {s['mean']*100:.1f}% +/- {s['std']*100:.1f}%", flush=True)

    print("\nLatency (seconds per seed):", flush=True)
    for cond in all_conditions:
        times = latency_log[cond]
        mean_t = np.mean(times) if times else 0
        print(f"  {cond:35s}: {mean_t:.0f}s", flush=True)
    # Compute steering overhead
    if latency_log["hyp_mobius_softprompt"] and latency_log["hyp_mobius_softprompt_dual"]:
        sp_mean = np.mean(latency_log["hyp_mobius_softprompt"])
        dual_mean = np.mean(latency_log["hyp_mobius_softprompt_dual"])
        overhead_pct = ((dual_mean - sp_mean) / sp_mean * 100) if sp_mean > 0 else 0
        print(f"  Steering overhead: {overhead_pct:+.1f}%", flush=True)

    print("\nPairwise Comparisons:", flush=True)
    for pair_key, ps in stats_result["pairwise"].items():
        print(f"\n  {pair_key}:", flush=True)
        print(f"    Diff: {ps['diff_mean']*100:+.1f}%", flush=True)
        if not np.isnan(ps['diff_ci_95'][0]):
            print(f"    95% CI: [{ps['diff_ci_95'][0]*100:.1f}%, {ps['diff_ci_95'][1]*100:.1f}%]", flush=True)
        p_perm = ps['p_permutation']
        print(f"    Permutation test (one-sided): p={p_perm:.4f}", flush=True)
        print(f"    Paired t: t={ps['t_stat']:.3f}, p={ps['p_ttest']:.4f}", flush=True)
        print(f"    Bonferroni threshold: p < {BONFERRONI_THRESHOLD}", flush=True)

    print("\nPer-depth:", flush=True)
    for depth in [2, 3]:
        print(f"  Depth {depth}:", flush=True)
        for cond in all_conditions:
            ds = stats_result["per_depth"][depth][cond]
            print(f"    {cond:35s}: {ds['mean']*100:.1f}% +/- {ds['std']*100:.1f}%", flush=True)

    # ---- Primary verdict ----
    print(f"\n{'=' * 70}", flush=True)
    print("PRE-REGISTERED PRIMARY COMPARISON", flush=True)
    print("Hyp Mobius + Soft Prompt vs Hyp Mobius + Soft Prompt + Dual Steering", flush=True)
    print(f"(Same evolution, added output-channel Newton steering)", flush=True)
    print(f"{'=' * 70}", flush=True)

    primary_key = "hyp_mobius_softprompt_vs_hyp_mobius_softprompt_dual"
    if primary_key in stats_result["pairwise"]:
        kp = stats_result["pairwise"][primary_key]
        sp_s = stats_result["per_condition"]["hyp_mobius_softprompt"]
        dual_s = stats_result["per_condition"]["hyp_mobius_softprompt_dual"]

        print(f"  Soft Prompt only:         {sp_s['mean']*100:.1f}% +/- {sp_s['std']*100:.1f}%", flush=True)
        print(f"  Soft Prompt + Dual Steer: {dual_s['mean']*100:.1f}% +/- {dual_s['std']*100:.1f}%", flush=True)
        print(f"  Difference:               {kp['diff_mean']*100:+.1f}%", flush=True)
        if not np.isnan(kp['p_permutation']):
            print(f"  Permutation p (one-sided): {kp['p_permutation']:.4f}", flush=True)

        # Use Bonferroni-corrected threshold (Codex fix #5)
        if (not np.isnan(kp['p_permutation'])
                and kp['p_permutation'] < BONFERRONI_THRESHOLD
                and kp['diff_mean'] > 0):
            verdict = f"SIGNIFICANT: Dual Steering > Soft Prompt Only (p < {BONFERRONI_THRESHOLD}, Bonferroni)"
        elif (not np.isnan(kp['p_permutation'])
              and kp['p_permutation'] < BONFERRONI_THRESHOLD
              and kp['diff_mean'] < 0):
            verdict = f"SIGNIFICANT: Soft Prompt Only > Dual Steering (unexpected, p < {BONFERRONI_THRESHOLD})"
        else:
            verdict = f"NOT SIGNIFICANT (p >= {BONFERRONI_THRESHOLD}, Bonferroni-corrected)"
    else:
        verdict = "KEY COMPARISON NOT FOUND"

    print(f"\nPRIMARY VERDICT: {verdict}", flush=True)

    # ---- Secondary verdict ----
    print(f"\n{'=' * 70}", flush=True)
    print("PRE-REGISTERED SECONDARY COMPARISON", flush=True)
    print("Hyp Mobius + RNG-Seed vs Hyp Mobius + Soft Prompt + Dual Steering", flush=True)
    print(f"{'=' * 70}", flush=True)

    secondary_key = "hyp_mobius_rng_vs_hyp_mobius_softprompt_dual"
    if secondary_key in stats_result["pairwise"]:
        ks = stats_result["pairwise"][secondary_key]
        rng_s = stats_result["per_condition"]["hyp_mobius_rng"]
        dual_s = stats_result["per_condition"]["hyp_mobius_softprompt_dual"]

        print(f"  RNG-Seed:                 {rng_s['mean']*100:.1f}% +/- {rng_s['std']*100:.1f}%", flush=True)
        print(f"  Soft Prompt + Dual Steer: {dual_s['mean']*100:.1f}% +/- {dual_s['std']*100:.1f}%", flush=True)
        print(f"  Difference:               {ks['diff_mean']*100:+.1f}%", flush=True)
        if not np.isnan(ks['p_permutation']):
            print(f"  Permutation p (one-sided): {ks['p_permutation']:.4f}", flush=True)

        if (not np.isnan(ks['p_permutation'])
                and ks['p_permutation'] < BONFERRONI_THRESHOLD
                and ks['diff_mean'] > 0):
            secondary_verdict = f"SIGNIFICANT: Dual Steer > RNG-Seed (p < {BONFERRONI_THRESHOLD}, Bonferroni)"
        else:
            secondary_verdict = f"NOT SIGNIFICANT (p >= {BONFERRONI_THRESHOLD})"
    else:
        secondary_verdict = "KEY COMPARISON NOT FOUND"

    print(f"\nSECONDARY VERDICT: {secondary_verdict}", flush=True)

    # ---- Fitness curves ----
    print(f"\n{'=' * 70}", flush=True)
    print("FITNESS CURVES", flush=True)
    print(f"{'=' * 70}", flush=True)
    for cond in all_conditions:
        print(f"\n[{cond.upper()}]", flush=True)
        for si, curve in enumerate(all_fitness_curves[cond]):
            gens = " -> ".join(f"{e['best']:.3f}" for e in curve)
            print(f"  Seed {si+1}: best fitness {gens}", flush=True)

    print(f"\n{'=' * 70}", flush=True)

    # ---- Save results ----
    results = {
        "config": {
            "model": args.model,
            "seeds": args.seeds,
            "test_tasks_per_depth": args.test_tasks_per_depth,
            "train_tasks_per_depth": args.train_tasks_per_depth,
            "branching": args.branching,
            "curvature": curvature,
            "ball_radius": ball_radius,
            "evo_gens": args.evo_gens,
            "evo_pop": args.evo_pop,
            "evo_tasks": args.evo_tasks,
            "conditions": all_conditions,
            "primary_comparison": primary_key,
            "secondary_comparison": secondary_key,
            "bonferroni_threshold": BONFERRONI_THRESHOLD,
            "num_soft_tokens": NUM_SOFT_TOKENS,
            "soft_prompt_dim": SOFT_PROMPT_FLAT_DIM,
            "embedding_rms": EMBEDDING_RMS_ELEM,
            "d_latent": d_latent,
            "projection_seed_soft": 1234,
            "projection_seed_steer": 5678,
            "eta": args.eta,
            "alpha": args.alpha,
            "kl_cap": args.kl_cap,
            "lm_head_dtype": str(lm_head_weight.dtype),
            "lm_head_shape": list(lm_head_weight.shape),
        },
        "v14_innovation": [
            "Dual steering: regularized Newton step in dual (probability) space",
            "Output-channel conditioning via lm_head routing",
            "KL safety cap with automatic eta downscaling",
            "Woodbury O(V) solve per token (~1ms overhead)",
            "Codex fixes: latent_dim from encoder, no epsilon, no tanh in steer, Bonferroni",
        ],
        "inherited_fixes": [
            "All V11 fixes (10 Codex V10 issues)",
            "V12 Mobius mutation operator",
            "V13 fixed orthogonal projection soft prompts",
        ],
        "statistics": stats_result,
        "fitness_curves": all_fitness_curves,
        "latency": {
            cond: {
                "per_seed_seconds": latency_log[cond],
                "mean_seconds": float(np.mean(latency_log[cond])) if latency_log[cond] else 0,
            }
            for cond in all_conditions
        },
        "primary_verdict": verdict,
        "secondary_verdict": secondary_verdict,
    }

    suffix = "_diagnostic" if args.diagnostic else ""
    results_path = Path(__file__).parent / f"v14_results{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}", flush=True)


if __name__ == "__main__":
    main()
