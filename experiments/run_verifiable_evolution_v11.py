"""
Verifiable Evolution V11 - Addressing ALL 10 Codex V10 Review Issues

Changes from V10 (Codex review issues fixed):
1. FIXED: Ball radius mismatch -> same L2 ball for euc_constrained and hyperbolic
2. FIXED: RNG contamination -> torch.Generator for noise, SAME seed all conditions
3. FIXED: Train/test leakage -> enumerate ALL unique tasks, deterministic split
4. FIXED: McNemar pseudo-replication -> per-seed McNemar only, paired t-test primary
5. FIXED: Global best not tracked -> track best-evaluated across all generations
6. FIXED: dense_score depth bias -> 1/(1+abs_distance), no |expected| normalization
7. FIXED: Loose answer parsing -> only LAST number in response counts
8. ADDED: No-evolution baseline condition
9. FIXED: Noise not dimension-normalized -> noise_scale / sqrt(dim)
10. FIXED: Statistical logic -> Bonferroni correction, pre-registered primary test

Four conditions:
A) no_evolution: Raw seed latent, no optimization (baseline)
B) euc_constrained: L2 ball, SAME radius as hyperbolic, flat mutations
C) hyperbolic: Poincare ball c=0.5, expmap/logmap mutations
D) euc_unconstrained: No constraint (ablation)

Pre-registered primary comparison: B vs C (same ball radius, different geometry)
Pre-registered curvature: c=0.5 (from V7, unchanged)
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
from scipy import stats
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.core.encoder import LLMEncoder


# =====================================================================
# Task generation: unique tasks, guaranteed non-overlapping train/test
# =====================================================================

@dataclass
class Task:
    task_id: str
    prompt: str
    correct_answer: int
    depth: int


def generate_all_unique_tasks(branching: int, depths: list) -> dict:
    """Enumerate ALL unique tasks per depth. No duplicates possible.

    branching=15 gives: depth-2: 225 unique, depth-3: 3375 unique.
    """
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
                f"Depth {depth}: need {n_needed} unique tasks but only {len(pool)}. "
                f"Increase branching factor."
            )
        test_tasks.extend(pool[:n_test_per_depth])
        train_tasks.extend(pool[n_test_per_depth:n_needed])
    return train_tasks, test_tasks


# =====================================================================
# Answer verification: STRICT (last number only)
# =====================================================================

def verify_answer(response: str, expected: int) -> bool:
    """Only the LAST number in the response counts as the answer."""
    numbers = re.findall(r'-?\d+', response)
    if not numbers:
        return False
    return int(numbers[-1]) == expected


def dense_score(response: str, expected: int) -> float:
    """Dense reward: last number only, absolute distance (no depth bias).

    Score = 1.0 for exact, 1/(1+|distance|) for near miss, 0 for no number.
    No normalization by |expected| so all depths penalized equally.
    """
    numbers = re.findall(r'-?\d+', response)
    if not numbers:
        return 0.0
    last_num = int(numbers[-1])
    if last_num == expected:
        return 1.0
    distance = abs(last_num - expected)
    return min(1.0 / (1.0 + distance), 0.99)


# =====================================================================
# Evolution: RNG isolation, global best, dimension-normalized noise
# =====================================================================

@dataclass
class Candidate:
    latent: Tensor
    fitness: float = 0.0


def _make_noise(
    shape, noise_scale: float, dim: int, rng: torch.Generator, device=None,
) -> Tensor:
    """Dimension-normalized noise with isolated RNG.

    Total L2 norm ~ noise_scale (not noise_scale * sqrt(dim)).
    """
    per_dim = noise_scale / math.sqrt(max(dim, 1))
    noise = torch.randn(shape, generator=rng) * per_dim
    if device is not None and device != torch.device("cpu"):
        noise = noise.to(device)
    return noise


def _apply_mutation(
    parent: Tensor, noise: Tensor, condition: str, curvature: float,
    ball_radius: float,
) -> Tensor:
    """Apply mutation according to the condition's geometry."""
    if condition == "hyperbolic":
        from latent_reasoning.utils import hyperbolic as hyp
        lat = parent.squeeze()
        tan = hyp.logmap0(lat, curvature)
        tan = tan + noise.squeeze()
        mutated = hyp.expmap0(tan, curvature)
        mutated = hyp.project_to_ball(mutated, curvature, 0.95)
        return mutated.unsqueeze(0)
    elif condition == "euc_constrained":
        mutated = parent + noise
        norm = mutated.squeeze().norm()
        if norm > ball_radius:
            mutated = mutated * (ball_radius / norm.item())
        return mutated
    else:  # euc_unconstrained
        return parent + noise


def evaluate_on_tasks_dense(
    latent: Tensor, tasks: list, encoder: LLMEncoder,
    hyperbolic: bool, curvature: float,
) -> Tuple[float, dict]:
    """Evaluate with dense reward. Returns (mean_score, per_task_scores)."""
    scores = {}
    for task in tasks:
        response = encoder.decode(
            latent, query=task.prompt, max_new_tokens=250,
            temperature=0.3, hyperbolic=hyperbolic, curvature=curvature,
        )
        scores[task.task_id] = dense_score(response, task.correct_answer)
    mean_score = sum(scores.values()) / len(scores) if scores else 0.0
    return mean_score, scores


def evaluate_on_tasks_binary(
    latent: Tensor, tasks: list, encoder: LLMEncoder,
    hyperbolic: bool, curvature: float,
) -> dict:
    """Evaluate with binary scoring (exact match, last number only)."""
    results = {}
    for task in tasks:
        response = encoder.decode(
            latent, query=task.prompt, max_new_tokens=250,
            temperature=0.3, hyperbolic=hyperbolic, curvature=curvature,
        )
        results[task.task_id] = verify_answer(response, task.correct_answer)
    return results


def run_evolution(
    encoder: LLMEncoder,
    train_tasks: list,
    seed_latent: Tensor,
    condition: str,
    curvature: float = 0.5,
    generations: int = 3,
    population_size: int = 4,
    tasks_per_gen: int = 8,
    noise_scale: float = 0.1,
    condition_seed: int = 0,
) -> Tuple[Tensor, list]:
    """Run evolution with RNG isolation and global best tracking.

    Key V11 fixes:
    - torch.Generator for mutation noise (not affected by decode's global seed)
    - Global best tracking across all generations
    - Matched ball radius for constrained conditions
    - Dimension-normalized noise
    - Matched initialization L2 norm
    - SAME RNG seed for all conditions (only geometry differs)
    """
    fitness_curve = []
    dim = seed_latent.numel()
    device = seed_latent.device

    # Ball radius: SAME for euc_constrained and hyperbolic
    ball_radius = (1.0 / math.sqrt(curvature)) * 0.95

    # Matched initialization: both constrained conditions start at same L2 norm
    target_init_norm = 0.5 * ball_radius  # ~0.672 for c=0.5

    if condition == "hyperbolic":
        from latent_reasoning.utils import hyperbolic as hyp
        # Compute scale so that expmap0 output has ||result|| = target_init_norm
        # ||expmap0(v, c)|| = tanh(sqrt(c) * ||v||) / sqrt(c)
        # Need: tanh(sqrt(c) * s * ||seed||) / sqrt(c) = target
        # So: s * ||seed|| = atanh(target * sqrt(c)) / sqrt(c)
        seed_norm = seed_latent.squeeze().norm().item()
        hyp_target = min(target_init_norm * math.sqrt(curvature), 0.999)
        tangent_norm = math.atanh(hyp_target) / math.sqrt(curvature)
        init_scale = tangent_norm / max(seed_norm, 1e-8)
        seed_latent = hyp.expmap0(
            seed_latent.squeeze() * init_scale, curvature
        ).unsqueeze(0)
    elif condition == "euc_constrained":
        # Same L2 norm as hyperbolic init
        norm = seed_latent.squeeze().norm().item()
        if norm > 0:
            seed_latent = seed_latent * (target_init_norm / norm)
    # euc_unconstrained: no normalization (ablation condition)

    # Isolated RNG for mutation noise (immune to decode's torch.manual_seed)
    mut_rng = torch.Generator()
    mut_rng.manual_seed(condition_seed)

    # Isolated RNG for task sampling
    task_rng = random.Random(condition_seed + 7)

    # Initialize population
    population = [Candidate(latent=seed_latent.clone())]
    for _ in range(population_size - 1):
        noise = _make_noise(seed_latent.shape, noise_scale, dim, mut_rng, device)
        mutated = _apply_mutation(
            seed_latent, noise, condition, curvature, ball_radius
        )
        population.append(Candidate(latent=mutated))

    # Global best tracking (across ALL generations)
    global_best = Candidate(latent=seed_latent.clone(), fitness=-1.0)

    is_hyp = (condition == "hyperbolic")

    for gen in range(generations):
        gen_tasks = task_rng.sample(
            train_tasks, min(tasks_per_gen, len(train_tasks))
        )

        # Evaluate with dense reward
        for cand in population:
            score, _ = evaluate_on_tasks_dense(
                cand.latent, gen_tasks, encoder, is_hyp, curvature,
            )
            cand.fitness = score

        # Update global best
        gen_best = max(population, key=lambda c: c.fitness)
        if gen_best.fitness > global_best.fitness:
            global_best = Candidate(
                latent=gen_best.latent.clone(), fitness=gen_best.fitness,
            )

        # Log curve
        fitnesses = [c.fitness for c in population]
        curve_entry = {
            "gen": gen + 1,
            "best": max(fitnesses),
            "mean": sum(fitnesses) / len(fitnesses),
            "min": min(fitnesses),
        }
        fitness_curve.append(curve_entry)

        # Selection: top half
        population.sort(key=lambda c: c.fitness, reverse=True)
        n_elite = max(2, population_size // 2)
        elite = population[:n_elite]

        # Reproduce
        new_pop = [Candidate(latent=e.latent.clone(), fitness=e.fitness)
                    for e in elite]
        while len(new_pop) < population_size:
            parent = elite[task_rng.randint(0, len(elite) - 1)]
            noise = _make_noise(
                parent.latent.shape, noise_scale, dim, mut_rng, device,
            )
            mutated = _apply_mutation(
                parent.latent, noise, condition, curvature, ball_radius,
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
# Statistics: per-seed McNemar, Bonferroni, pre-registered primary
# =====================================================================

def compute_statistics(results_by_condition: dict, task_ids: list) -> dict:
    """Statistics with proper corrections for all Codex issues."""
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

    n_pairs = len(conditions) * (len(conditions) - 1) // 2

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
                t_stat, p_value = stats.ttest_rel(accs_b, accs_a)
                ci_95 = stats.t.interval(
                    0.95, len(diffs) - 1, loc=mean_diff, scale=se,
                )
                p_bonferroni = min(1.0, float(p_value) * n_pairs)
            else:
                std_diff = float("nan")
                t_stat = float("nan")
                p_value = float("nan")
                ci_95 = (float("nan"), float("nan"))
                p_bonferroni = float("nan")

            # Per-seed McNemar (NOT pooled -- fixes pseudo-replication)
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
                    p_mc = float(1 - stats.chi2.cdf(chi2, 1))
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
                "t_stat": float(t_stat),
                "p_value_raw": float(p_value),
                "p_value_bonferroni": float(p_bonferroni),
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
    parser = argparse.ArgumentParser(description="V11: All 10 Codex fixes")
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

    curvature = 0.5  # Pre-registered from V7
    ball_radius = (1.0 / math.sqrt(curvature)) * 0.95

    print("=" * 70, flush=True)
    print("VERIFIABLE EVOLUTION V11 - ALL 10 CODEX V10 FIXES", flush=True)
    print("=" * 70, flush=True)
    print("FIXES FROM V10 CODEX REVIEW:", flush=True)
    print("  1. Matched ball radii (same L2 radius for constrained+hyp)", flush=True)
    print("  2. RNG isolation (torch.Generator, SAME seed all conditions)", flush=True)
    print("  3. No train/test leakage (unique enumerated tasks)", flush=True)
    print("  4. Per-seed McNemar (no pseudo-replication)", flush=True)
    print("  5. Global best tracking across generations", flush=True)
    print("  6. Strict answer parsing (last number only)", flush=True)
    print("  7. Depth-unbiased dense score (absolute distance)", flush=True)
    print("  8. No-evolution baseline condition", flush=True)
    print("  9. Dimension-normalized noise (scale/sqrt(dim))", flush=True)
    print(" 10. Bonferroni correction for multiple comparisons", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    b = args.branching
    print(f"Branching: {b} (d2: {b**2} unique, d3: {b**3} unique)", flush=True)
    print(f"Test: {args.test_tasks_per_depth * 2} ({args.test_tasks_per_depth}/depth)", flush=True)
    print(f"Train: {args.train_tasks_per_depth * 2} ({args.train_tasks_per_depth}/depth)", flush=True)
    print(f"Evolution: {args.evo_gens} gens, pop={args.evo_pop}, tasks/gen={args.evo_tasks}", flush=True)
    print(f"Curvature: {curvature} (pre-registered from V7)", flush=True)
    print(f"Ball radius: {ball_radius:.3f} (SAME for euc_constrained and hyperbolic)", flush=True)
    print("Conditions: no_evolution, euc_constrained, hyperbolic, euc_unconstrained", flush=True)
    print("Primary comparison: euc_constrained vs hyperbolic (pre-registered)", flush=True)
    print("=" * 70, flush=True)

    # Generate unique task pool
    print("\nGenerating unique task pool...", flush=True)
    tasks_by_depth = generate_all_unique_tasks(args.branching, depths=[2, 3])
    for depth, tasks in sorted(tasks_by_depth.items()):
        print(f"  Depth {depth}: {len(tasks)} unique tasks", flush=True)

    train_tasks, test_tasks = split_train_test(
        tasks_by_depth, args.test_tasks_per_depth,
        args.train_tasks_per_depth, seed=7777,
    )
    test_task_ids = [t.task_id for t in test_tasks]

    # Verify zero overlap
    train_ids = {t.task_id for t in train_tasks}
    test_ids = {t.task_id for t in test_tasks}
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} overlapping tasks!"
    print(f"  Train: {len(train_tasks)}, Test: {len(test_tasks)}, Overlap: 0 (verified)", flush=True)

    # Load model
    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(model_name=args.model, quantization="4bit")

    evolved_conditions = ["euc_constrained", "hyperbolic", "euc_unconstrained"]
    all_conditions = ["no_evolution"] + evolved_conditions

    all_results = {c: [] for c in all_conditions}
    all_fitness_curves = {c: [] for c in evolved_conditions}

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

        # --- No-evolution baseline (norm-matched to evolved conditions) ---
        print("\n[NO_EVOLUTION] Testing (no optimization)...", flush=True)
        no_evo_latent = seed_latent.clone()
        target_init_norm = 0.5 * ball_radius  # Same as evolved conditions
        no_evo_norm = no_evo_latent.squeeze().norm().item()
        if no_evo_norm > 0:
            no_evo_latent = no_evo_latent * (target_init_norm / no_evo_norm)
        no_evo_res = evaluate_on_tasks_binary(
            no_evo_latent, test_tasks, encoder, hyperbolic=False, curvature=1.0,
        )
        all_results["no_evolution"].append(no_evo_res)
        acc = sum(no_evo_res.values()) / len(no_evo_res)
        d2 = sum(
            1 for t in test_tasks
            if t.depth == 2 and no_evo_res[t.task_id]
        ) / args.test_tasks_per_depth
        d3 = sum(
            1 for t in test_tasks
            if t.depth == 3 and no_evo_res[t.task_id]
        ) / args.test_tasks_per_depth
        print(f"  Accuracy: {acc*100:.1f}% (D2: {d2*100:.1f}%, D3: {d3*100:.1f}%)", flush=True)

        # --- Evolved conditions ---
        for cond_idx, cond in enumerate(evolved_conditions):
            is_hyp = (cond == "hyperbolic")

            # SAME seed for all conditions -> only geometry differs
            # (Codex review: per-condition seeds create RNG confound)
            condition_seed = seed

            print(f"\n[{cond.upper()}] Evolution...", flush=True)
            evolved_latent, curve = run_evolution(
                encoder, train_tasks, seed_latent.clone(),
                condition=cond,
                curvature=curvature,
                generations=args.evo_gens,
                population_size=args.evo_pop,
                tasks_per_gen=args.evo_tasks,
                condition_seed=condition_seed,
            )
            all_fitness_curves[cond].append(curve)

            print(f"\n[{cond.upper()}] Testing...", flush=True)
            test_res = evaluate_on_tasks_binary(
                evolved_latent, test_tasks, encoder,
                is_hyp, curvature if is_hyp else 1.0,
            )
            all_results[cond].append(test_res)

            acc = sum(test_res.values()) / len(test_res)
            d2 = sum(
                1 for t in test_tasks
                if t.depth == 2 and test_res[t.task_id]
            ) / args.test_tasks_per_depth
            d3 = sum(
                1 for t in test_tasks
                if t.depth == 3 and test_res[t.task_id]
            ) / args.test_tasks_per_depth
            print(f"  Accuracy: {acc*100:.1f}% (D2: {d2*100:.1f}%, D3: {d3*100:.1f}%)", flush=True)

    # ---- Fitness curves ----
    print(f"\n{'=' * 70}", flush=True)
    print("FITNESS CURVES", flush=True)
    print(f"{'=' * 70}", flush=True)
    for cond in evolved_conditions:
        print(f"\n[{cond.upper()}]", flush=True)
        for si, curve in enumerate(all_fitness_curves[cond]):
            gens = " -> ".join(f"{e['best']:.3f}" for e in curve)
            print(f"  Seed {si+1}: best fitness {gens}", flush=True)

    # ---- Statistics ----
    print(f"\n{'=' * 70}", flush=True)
    print("STATISTICAL ANALYSIS", flush=True)
    print(f"{'=' * 70}", flush=True)

    stats_result = compute_statistics(all_results, test_task_ids)

    print("\nOverall Accuracy (mean +/- std):", flush=True)
    for cond in all_conditions:
        s = stats_result["per_condition"][cond]
        print(f"  {cond:22s}: {s['mean']*100:.1f}% +/- {s['std']*100:.1f}%", flush=True)

    print("\nPairwise Comparisons:", flush=True)
    for pair_key, ps in stats_result["pairwise"].items():
        print(f"\n  {pair_key}:", flush=True)
        print(f"    Diff: {ps['diff_mean']*100:+.1f}%", flush=True)
        if not np.isnan(ps['diff_ci_95'][0]):
            print(f"    95% CI: [{ps['diff_ci_95'][0]*100:.1f}%, {ps['diff_ci_95'][1]*100:.1f}%]", flush=True)
        p_raw = ps['p_value_raw']
        p_bonf = ps['p_value_bonferroni']
        print(f"    Paired t: t={ps['t_stat']:.3f}, p_raw={p_raw:.4f}, p_bonf={p_bonf:.4f}", flush=True)
        mc_ps = [f"{m['p']:.3f}" for m in ps['per_seed_mcnemar']]
        print(f"    Per-seed McNemar p: {mc_ps}", flush=True)

    print("\nPer-depth:", flush=True)
    for depth in [2, 3]:
        print(f"  Depth {depth}:", flush=True)
        for cond in all_conditions:
            ds = stats_result["per_depth"][depth][cond]
            print(f"    {cond:22s}: {ds['mean']*100:.1f}% +/- {ds['std']*100:.1f}%", flush=True)

    # ---- Pre-registered primary comparison ----
    print(f"\n{'=' * 70}", flush=True)
    print("PRE-REGISTERED PRIMARY COMPARISON", flush=True)
    print("Euclidean Constrained vs Hyperbolic", flush=True)
    print(f"(Same L2 ball radius={ball_radius:.3f}, different geometry)", flush=True)
    print(f"{'=' * 70}", flush=True)

    key_pair = "euc_constrained_vs_hyperbolic"
    if key_pair in stats_result["pairwise"]:
        kp = stats_result["pairwise"][key_pair]
        euc_s = stats_result["per_condition"]["euc_constrained"]
        hyp_s = stats_result["per_condition"]["hyperbolic"]

        print(f"  Euc Constrained: {euc_s['mean']*100:.1f}% +/- {euc_s['std']*100:.1f}%", flush=True)
        print(f"  Hyperbolic c=0.5: {hyp_s['mean']*100:.1f}% +/- {hyp_s['std']*100:.1f}%", flush=True)
        print(f"  Difference:       {kp['diff_mean']*100:+.1f}%", flush=True)
        if not np.isnan(kp['p_value_raw']):
            print(f"  p-value (raw):    {kp['p_value_raw']:.4f}", flush=True)

        # Use raw p-value for pre-registered primary (no Bonferroni needed)
        if (not np.isnan(kp['p_value_raw'])
                and kp['p_value_raw'] < 0.05
                and kp['diff_mean'] > 0):
            verdict = "SIGNIFICANT: Hyperbolic > Euclidean (p < 0.05, pre-registered)"
        elif (not np.isnan(kp['p_value_raw'])
              and kp['p_value_raw'] < 0.05
              and kp['diff_mean'] < 0):
            verdict = "SIGNIFICANT: Euclidean > Hyperbolic (p < 0.05)"
        else:
            verdict = "NOT SIGNIFICANT (p >= 0.05)"
    else:
        verdict = "KEY COMPARISON NOT FOUND"

    print(f"\nVERDICT: {verdict}", flush=True)
    print("=" * 70, flush=True)

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
            "primary_comparison": key_pair,
        },
        "codex_fixes": [
            "Matched ball radii (same L2 for constrained+hyp)",
            "RNG isolation (torch.Generator, SAME seed all conditions)",
            "Unique tasks, no train/test leakage",
            "Per-seed McNemar (no pseudo-replication)",
            "Global best tracking across generations",
            "Strict answer parsing (last number only)",
            "Depth-unbiased dense score (absolute distance)",
            "No-evolution baseline",
            "Dimension-normalized noise (scale/sqrt(dim))",
            "Bonferroni correction for multiple comparisons",
        ],
        "statistics": stats_result,
        "fitness_curves": {
            c: all_fitness_curves[c] for c in evolved_conditions
        },
        "verdict": verdict,
    }

    suffix = "_diagnostic" if args.diagnostic else ""
    results_path = Path(__file__).parent / f"v11_results{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}", flush=True)


if __name__ == "__main__":
    main()
