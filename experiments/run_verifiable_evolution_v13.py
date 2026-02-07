"""
Verifiable Evolution V13 - Fixed Orthogonal Projection to Soft Prompts

Replaces the 31-bit RNG-seed conditioning mechanism with a fixed random
orthogonal projection from the latent (1024d tangent space) to soft prompt
tokens (8 x 2560 = 20,480 continuous values). ~650x more information bandwidth.

Key insight (from research): The projection does NOT need training. A fixed
row-orthonormal matrix preserves inner products (Johnson-Lindenstrauss), so
similar latents -> similar soft prompts, different latents -> different prompts.
The evolutionary optimizer searches the Poincare ball; the projection is just
a deterministic change of coordinates.

Based on Codex V13 design review (B- -> A- with fixes):
1. Row-orthonormal W projection (W W^T = I_1024)
2. Radial tanh squash for logmap0 output (prevents boundary blow-up)
3. Embedding RMS calibration (match soft prompt scale to real tokens)
4. Plain model control (no latent conditioning)

Three conditions:
C0) no_evolution: Raw latent, no optimization (QC baseline)
C1) hyp_mobius_rng: Hyperbolic Mobius evolution + RNG-seed decode (V12 best)
C2) hyp_mobius_softprompt: Hyperbolic Mobius evolution + fixed projection decode

Pre-registered primary: C1 vs C2 (does soft prompt beat RNG-seed?)
Pre-registered secondary: C0 vs C2 (does evolution + soft prompt beat baseline?)
Pre-registered curvature: c=0.5 (unchanged from V7)

Inherits ALL V11/V12 fixes.

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
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from scipy import stats as sp_stats
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.utils import hyperbolic as hyp


# =====================================================================
# Calibration constants (measured from Qwen3-4B embedding table)
# =====================================================================
EMBED_DIM = 2560
NUM_SOFT_TOKENS = 8
SOFT_PROMPT_FLAT_DIM = NUM_SOFT_TOKENS * EMBED_DIM  # 20480
EMBEDDING_RMS_ELEM = 0.02195  # Per-element RMS of Qwen3-4B embeddings
MEAN_TOKEN_L2_NORM = 1.097    # Mean L2 norm of real token embeddings
R_REF_1024 = math.sqrt(1024) * EMBEDDING_RMS_ELEM  # ~0.7025
R_MAX_LOGMAP = 2.0 * R_REF_1024  # ~1.405 (Codex spec: locked)


# =====================================================================
# Task generation (identical to V11/V12)
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
# Answer verification (identical to V11/V12)
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
# Fixed Orthogonal Projection (V13 core innovation)
# =====================================================================

def make_row_orthonormal_W(d_latent: int = 1024, d_out: int = SOFT_PROMPT_FLAT_DIM,
                            seed: int = 1234) -> Tensor:
    """Create a fixed row-orthonormal projection matrix W.

    W has shape (d_latent, d_out) with orthonormal rows: W W^T = I.
    This preserves inner products (JL property): <u,v> = <uW, vW>.

    Construction: Generate random (d_out, d_latent) matrix, QR decompose,
    take Q^T as W.
    """
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d_out, d_latent, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A, mode="reduced")  # Q: (d_out, d_latent)
    W = Q.T.contiguous()  # W: (d_latent, d_out), rows are orthonormal
    return W


def radial_tanh_squash(v: Tensor, r_max: float, eps: float = 1e-8) -> Tensor:
    """Smooth radial squash to prevent boundary blow-up in logmap0 output.

    Maps any norm to [0, r_max) via tanh saturation.
    Preserves direction, only affects magnitude.
    """
    r = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    r_new = r_max * torch.tanh(r / r_max)
    return v * (r_new / r)


def latent_to_soft_prompt(
    latent: Tensor,
    W: Tensor,
    curvature: float,
    target_rms: float = EMBEDDING_RMS_ELEM,
) -> Tensor:
    """Convert a Poincare ball latent to soft prompt embeddings.

    Pipeline:
    1. logmap0: Poincare ball -> tangent space at origin
    2. NaN guard: Replace any NaN/Inf from boundary points
    3. Radial tanh squash: Prevent blow-up, smooth saturation
    4. Project via row-orthonormal W: (1024,) -> (20480,)
    5. Reshape to (num_tokens, embed_dim)
    6. Scale to match real token embedding RMS

    Returns: (1, NUM_SOFT_TOKENS, EMBED_DIM) tensor
    """
    lat = latent.squeeze().float()

    # 1. Map to tangent space
    tangent = hyp.logmap0(lat, curvature)

    # 2. NaN guard
    tangent = torch.nan_to_num(tangent, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Radial tanh squash
    tangent = radial_tanh_squash(tangent, R_MAX_LOGMAP)

    # 4. Project (preserve inner products)
    flat = tangent @ W  # (1024,) @ (1024, 20480) -> (20480,)

    # 5. Reshape to token sequence
    soft_prompt = flat.view(NUM_SOFT_TOKENS, EMBED_DIM)

    # 6. Scale: match per-element RMS to real token embeddings
    current_rms = soft_prompt.square().mean().sqrt().clamp_min(1e-8)
    soft_prompt = soft_prompt * (target_rms / current_rms)

    return soft_prompt.unsqueeze(0)  # (1, 8, 2560)


def decode_with_projection(
    encoder: LLMEncoder,
    latent: Tensor,
    query: str,
    W: Tensor,
    curvature: float,
    max_new_tokens: int = 250,
    temperature: float = 0.3,
) -> str:
    """Decode using fixed orthogonal projection to soft prompts.

    This bypasses the RNG-seed mechanism entirely. The latent directly
    conditions generation via 8 prepended soft prompt tokens.
    """
    # Generate soft prompt embeddings
    with torch.no_grad():
        soft_prompt = latent_to_soft_prompt(latent, W, curvature)
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

        if temperature < 0.01:
            outputs = encoder.model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=encoder.tokenizer.pad_token_id,
                eos_token_id=encoder.tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )
        else:
            outputs = encoder.model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                pad_token_id=encoder.tokenizer.pad_token_id,
                eos_token_id=encoder.tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )

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
# Evolution (V12 Mobius mutation, adapted for two decode modes)
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


def evaluate_dense(latent, tasks, encoder, curvature, W=None):
    """Evaluate with dense scoring. Uses soft prompt if W is provided."""
    scores = {}
    for task in tasks:
        if W is not None:
            response = decode_with_projection(
                encoder, latent, task.prompt, W, curvature,
            )
        else:
            response = encoder.decode(
                latent, query=task.prompt, max_new_tokens=250,
                temperature=0.3, hyperbolic=True, curvature=curvature,
            )
        scores[task.task_id] = dense_score(response, task.correct_answer)
    mean_score = sum(scores.values()) / len(scores) if scores else 0.0
    return mean_score, scores


def evaluate_binary(latent, tasks, encoder, curvature, W=None):
    """Evaluate with binary scoring. Uses soft prompt if W is provided."""
    results = {}
    for task in tasks:
        if W is not None:
            response = decode_with_projection(
                encoder, latent, task.prompt, W, curvature,
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
    noise_scale=0.1, condition_seed=0, W=None,
) -> Tuple[Tensor, list]:
    """Run Mobius evolution. Uses soft prompt decode if W is provided."""
    fitness_curve = []
    dim = seed_latent.numel()
    device = seed_latent.device
    ball_radius = (1.0 / math.sqrt(curvature)) * 0.95

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
                cand.latent, gen_tasks, encoder, curvature, W=W,
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
# Statistics (permutation test per Codex V13 spec)
# =====================================================================

def exact_sign_flip_pvalue(diffs, alternative="greater"):
    """Exact paired sign-flip permutation test.

    For small n (5-15), this is more appropriate than t-test.
    Tests H0: median(diffs) = 0.
    """
    n = len(diffs)
    observed = sum(diffs)

    # Enumerate all 2^n sign combinations
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
    """Compute statistics with permutation tests."""
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
                # Permutation test (one-sided: H1: diff > 0)
                p_perm = exact_sign_flip_pvalue(diffs, "greater")
                # Also paired t for comparison
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
        description="V13: Fixed orthogonal projection to soft prompts"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--test-tasks-per-depth", type=int, default=20)
    parser.add_argument("--train-tasks-per-depth", type=int, default=150)
    parser.add_argument("--branching", type=int, default=15)
    parser.add_argument("--evo-gens", type=int, default=3)
    parser.add_argument("--evo-pop", type=int, default=4)
    parser.add_argument("--evo-tasks", type=int, default=8)
    parser.add_argument("--diagnostic", action="store_true",
                        help="Run 1 seed for quick sanity check")
    args = parser.parse_args()

    if args.diagnostic:
        args.seeds = 1

    curvature = 0.5
    ball_radius = (1.0 / math.sqrt(curvature)) * 0.95

    print("=" * 70, flush=True)
    print("VERIFIABLE EVOLUTION V13 - FIXED ORTHOGONAL PROJECTION", flush=True)
    print("=" * 70, flush=True)
    print("V13 INNOVATION (replaces 31-bit RNG seed with 20K-dim soft prompt):", flush=True)
    print("  1. Row-orthonormal projection W (1024 -> 20480, W W^T = I)", flush=True)
    print("  2. Radial tanh squash for logmap0 (prevents boundary blow-up)", flush=True)
    print("  3. Embedding RMS calibration (matches real token scale)", flush=True)
    print(f"  Bandwidth: {SOFT_PROMPT_FLAT_DIM} continuous values vs 31 bits", flush=True)
    print("INHERITED from V11/V12:", flush=True)
    print("  Matched ball radii, RNG isolation, unique tasks, Mobius mutation,", flush=True)
    print("  per-seed McNemar, global best, strict parsing, dense score", flush=True)
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
    print("Conditions: no_evolution, hyp_mobius_rng, hyp_mobius_softprompt", flush=True)
    print("Primary: hyp_mobius_rng vs hyp_mobius_softprompt (RNG vs soft prompt)", flush=True)
    print("Secondary: no_evolution vs hyp_mobius_softprompt (baseline vs evolved+SP)", flush=True)
    print(f"Calibration: embed_rms={EMBEDDING_RMS_ELEM:.5f}, r_max={R_MAX_LOGMAP:.4f}", flush=True)
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

    # Create fixed orthogonal projection
    print("\nCreating row-orthonormal projection W...", flush=True)
    W = make_row_orthonormal_W(d_latent=1024, d_out=SOFT_PROMPT_FLAT_DIM, seed=1234)
    print(f"  W shape: {W.shape}", flush=True)
    print(f"  W W^T diagonal check: {(W @ W.T).diag()[:5].tolist()}", flush=True)
    print(f"  W W^T off-diag max: {((W @ W.T) - torch.eye(1024)).abs().max().item():.6f}", flush=True)

    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(model_name=args.model, quantization="4bit")

    all_conditions = ["no_evolution", "hyp_mobius_rng", "hyp_mobius_softprompt"]
    all_results = {c: [] for c in all_conditions}
    all_fitness_curves = {"hyp_mobius_rng": [], "hyp_mobius_softprompt": []}

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

        # C0: No-evolution baseline (norm-matched, no soft prompt)
        print("\n[NO_EVOLUTION] Testing (no optimization, RNG decode)...", flush=True)
        no_evo_latent = seed_latent.clone()
        target_init_norm = 0.5 * ball_radius
        no_evo_norm = no_evo_latent.squeeze().norm().item()
        if no_evo_norm > 0:
            no_evo_latent = no_evo_latent * (target_init_norm / no_evo_norm)
        no_evo_res = evaluate_binary(
            no_evo_latent, test_tasks, encoder, curvature=1.0, W=None,
        )
        all_results["no_evolution"].append(no_evo_res)
        acc = sum(no_evo_res.values()) / len(no_evo_res)
        d2 = sum(1 for t in test_tasks if t.depth == 2 and no_evo_res[t.task_id]) / args.test_tasks_per_depth
        d3 = sum(1 for t in test_tasks if t.depth == 3 and no_evo_res[t.task_id]) / args.test_tasks_per_depth
        print(f"  Accuracy: {acc*100:.1f}% (D2: {d2*100:.1f}%, D3: {d3*100:.1f}%)", flush=True)

        condition_seed = seed  # SAME for both evolved conditions

        # C1: Hyp Mobius + RNG-seed decode (V12 best)
        print(f"\n[HYP_MOBIUS_RNG] Evolution (RNG-seed decode)...", flush=True)
        evolved_rng, curve_rng = run_evolution(
            encoder, train_tasks, seed_latent.clone(),
            curvature=curvature,
            generations=args.evo_gens,
            population_size=args.evo_pop,
            tasks_per_gen=args.evo_tasks,
            condition_seed=condition_seed,
            W=None,  # RNG-seed decode
        )
        all_fitness_curves["hyp_mobius_rng"].append(curve_rng)

        print(f"\n[HYP_MOBIUS_RNG] Testing...", flush=True)
        rng_res = evaluate_binary(
            evolved_rng, test_tasks, encoder, curvature, W=None,
        )
        all_results["hyp_mobius_rng"].append(rng_res)
        acc = sum(rng_res.values()) / len(rng_res)
        d2 = sum(1 for t in test_tasks if t.depth == 2 and rng_res[t.task_id]) / args.test_tasks_per_depth
        d3 = sum(1 for t in test_tasks if t.depth == 3 and rng_res[t.task_id]) / args.test_tasks_per_depth
        print(f"  Accuracy: {acc*100:.1f}% (D2: {d2*100:.1f}%, D3: {d3*100:.1f}%)", flush=True)

        # C2: Hyp Mobius + soft prompt decode (V13 innovation)
        print(f"\n[HYP_MOBIUS_SOFTPROMPT] Evolution (soft prompt decode)...", flush=True)
        evolved_sp, curve_sp = run_evolution(
            encoder, train_tasks, seed_latent.clone(),
            curvature=curvature,
            generations=args.evo_gens,
            population_size=args.evo_pop,
            tasks_per_gen=args.evo_tasks,
            condition_seed=condition_seed,
            W=W,  # Soft prompt decode
        )
        all_fitness_curves["hyp_mobius_softprompt"].append(curve_sp)

        print(f"\n[HYP_MOBIUS_SOFTPROMPT] Testing...", flush=True)
        sp_res = evaluate_binary(
            evolved_sp, test_tasks, encoder, curvature, W=W,
        )
        all_results["hyp_mobius_softprompt"].append(sp_res)
        acc = sum(sp_res.values()) / len(sp_res)
        d2 = sum(1 for t in test_tasks if t.depth == 2 and sp_res[t.task_id]) / args.test_tasks_per_depth
        d3 = sum(1 for t in test_tasks if t.depth == 3 and sp_res[t.task_id]) / args.test_tasks_per_depth
        print(f"  Accuracy: {acc*100:.1f}% (D2: {d2*100:.1f}%, D3: {d3*100:.1f}%)", flush=True)

    # ---- Statistics ----
    print(f"\n{'=' * 70}", flush=True)
    print("STATISTICAL ANALYSIS", flush=True)
    print(f"{'=' * 70}", flush=True)

    stats_result = compute_statistics(all_results, test_task_ids)

    print("\nOverall Accuracy (mean +/- std):", flush=True)
    for cond in all_conditions:
        s = stats_result["per_condition"][cond]
        print(f"  {cond:26s}: {s['mean']*100:.1f}% +/- {s['std']*100:.1f}%", flush=True)

    print("\nPairwise Comparisons:", flush=True)
    for pair_key, ps in stats_result["pairwise"].items():
        print(f"\n  {pair_key}:", flush=True)
        print(f"    Diff: {ps['diff_mean']*100:+.1f}%", flush=True)
        if not np.isnan(ps['diff_ci_95'][0]):
            print(f"    95% CI: [{ps['diff_ci_95'][0]*100:.1f}%, {ps['diff_ci_95'][1]*100:.1f}%]", flush=True)
        p_perm = ps['p_permutation']
        print(f"    Permutation test (one-sided): p={p_perm:.4f}", flush=True)
        print(f"    Paired t: t={ps['t_stat']:.3f}, p={ps['p_ttest']:.4f}", flush=True)

    print("\nPer-depth:", flush=True)
    for depth in [2, 3]:
        print(f"  Depth {depth}:", flush=True)
        for cond in all_conditions:
            ds = stats_result["per_depth"][depth][cond]
            print(f"    {cond:26s}: {ds['mean']*100:.1f}% +/- {ds['std']*100:.1f}%", flush=True)

    # ---- Primary verdict ----
    print(f"\n{'=' * 70}", flush=True)
    print("PRE-REGISTERED PRIMARY COMPARISON", flush=True)
    print("Hyp Mobius + RNG-Seed vs Hyp Mobius + Soft Prompt", flush=True)
    print(f"(Same evolution, different conditioning: 31-bit seed vs {SOFT_PROMPT_FLAT_DIM}-dim continuous)", flush=True)
    print(f"{'=' * 70}", flush=True)

    primary_key = "hyp_mobius_rng_vs_hyp_mobius_softprompt"
    if primary_key in stats_result["pairwise"]:
        kp = stats_result["pairwise"][primary_key]
        rng_s = stats_result["per_condition"]["hyp_mobius_rng"]
        sp_s = stats_result["per_condition"]["hyp_mobius_softprompt"]

        print(f"  Hyp Mobius + RNG:         {rng_s['mean']*100:.1f}% +/- {rng_s['std']*100:.1f}%", flush=True)
        print(f"  Hyp Mobius + SoftPrompt:  {sp_s['mean']*100:.1f}% +/- {sp_s['std']*100:.1f}%", flush=True)
        print(f"  Difference:               {kp['diff_mean']*100:+.1f}%", flush=True)
        if not np.isnan(kp['p_permutation']):
            print(f"  Permutation p (one-sided): {kp['p_permutation']:.4f}", flush=True)

        if (not np.isnan(kp['p_permutation'])
                and kp['p_permutation'] < 0.05
                and kp['diff_mean'] > 0):
            verdict = "SIGNIFICANT: Soft Prompt > RNG-Seed (p < 0.05, pre-registered)"
        elif (not np.isnan(kp['p_permutation'])
              and kp['p_permutation'] < 0.05
              and kp['diff_mean'] < 0):
            verdict = "SIGNIFICANT: RNG-Seed > Soft Prompt (unexpected)"
        else:
            verdict = "NOT SIGNIFICANT (p >= 0.05)"
    else:
        verdict = "KEY COMPARISON NOT FOUND"

    print(f"\nPRIMARY VERDICT: {verdict}", flush=True)

    # ---- Fitness curves ----
    print(f"\n{'=' * 70}", flush=True)
    print("FITNESS CURVES", flush=True)
    print(f"{'=' * 70}", flush=True)
    for cond in ["hyp_mobius_rng", "hyp_mobius_softprompt"]:
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
            "num_soft_tokens": NUM_SOFT_TOKENS,
            "soft_prompt_dim": SOFT_PROMPT_FLAT_DIM,
            "embedding_rms": EMBEDDING_RMS_ELEM,
            "r_max_logmap": R_MAX_LOGMAP,
            "projection_seed": 1234,
        },
        "v13_innovation": [
            f"Fixed orthogonal projection: 1024d latent -> {SOFT_PROMPT_FLAT_DIM}d soft prompt",
            "Row-orthonormal W (W W^T = I_1024, preserves inner products)",
            "Radial tanh squash for logmap0 (smooth boundary saturation)",
            f"Embedding RMS calibration: {EMBEDDING_RMS_ELEM:.5f}",
            f"Information bandwidth: {SOFT_PROMPT_FLAT_DIM} continuous vs 31-bit seed",
        ],
        "inherited_fixes": [
            "All V11 fixes (10 Codex V10 issues)",
            "V12 Mobius mutation operator",
        ],
        "statistics": stats_result,
        "fitness_curves": all_fitness_curves,
        "primary_verdict": verdict,
    }

    suffix = "_diagnostic" if args.diagnostic else ""
    results_path = Path(__file__).parent / f"v13_results{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}", flush=True)


if __name__ == "__main__":
    main()
