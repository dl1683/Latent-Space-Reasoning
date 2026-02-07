"""
Verifiable Evolution V12 - Mobius Mutations + Mutation Operator Ablation

Based on Codex V12 design review and 2025-2026 research findings:

1. HypLoRA (NeurIPS 2025 Spotlight): logmap0/expmap0 round-trip causes
   CANCELLATION EFFECT. Perturbation magnitude drops ~40% near boundary.
2. Codex recommends Mobius addition as primary (uniform perturbation).
3. Local expmap as secondary exploration condition.

Six conditions (mutation operator ablation):
A) no_evolution: Raw seed latent, no optimization (baseline)
B) euc_constrained: L2 ball, flat mutations (Euclidean control)
C) hyp_origin_roundtrip: logmap0 -> add noise -> expmap0 (V11 approach)
D) hyp_mobius: mobius_add(parent, noise_in_ball) (Codex-recommended primary)
E) hyp_local_expmap: expmap(noise, parent) (boundary-exploring)
F) euc_unconstrained: No constraint (ablation)

Pre-registered primary comparison: B vs D (flat L2 ball vs Mobius hyperbolic)
Pre-registered secondary: C vs D (quantify cancellation effect)
Pre-registered curvature: c=0.5 (from V7, unchanged)

Inherits ALL V11 fixes (10 Codex V10 issues resolved).
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
from latent_reasoning.utils import hyperbolic as hyp


# =====================================================================
# Task generation (identical to V11)
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
# Answer verification (identical to V11)
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
# Evolution: Multiple mutation operators
# =====================================================================

@dataclass
class Candidate:
    latent: Tensor
    fitness: float = 0.0


def _make_noise(
    shape, noise_scale: float, dim: int, rng: torch.Generator, device=None,
) -> Tensor:
    """Dimension-normalized noise with isolated RNG."""
    per_dim = noise_scale / math.sqrt(max(dim, 1))
    noise = torch.randn(shape, generator=rng) * per_dim
    if device is not None and device != torch.device("cpu"):
        noise = noise.to(device)
    return noise


def _apply_mutation(
    parent: Tensor, noise: Tensor, condition: str, curvature: float,
    ball_radius: float,
) -> Tensor:
    """Apply mutation according to the condition's geometry and operator.

    V12 mutation operators:
    - hyp_origin_roundtrip: logmap0 -> add noise -> expmap0 (V11 style)
    - hyp_mobius: mobius_add(parent, noise_in_ball, c) (Codex-recommended)
    - hyp_local_expmap: expmap(noise, parent, c) (boundary-exploring)
    - euc_constrained: parent + noise, clamped to ball
    - euc_unconstrained: parent + noise
    """
    if condition == "hyp_origin_roundtrip":
        lat = parent.squeeze()
        tan = hyp.logmap0(lat, curvature)
        tan = tan + noise.squeeze()
        mutated = hyp.expmap0(tan, curvature)
        mutated = hyp.project_to_ball(mutated, curvature, 0.95)
        return mutated.unsqueeze(0)

    elif condition == "hyp_mobius":
        # Map noise to a small point in the Poincare ball, then Mobius-add
        noise_in_ball = hyp.expmap0(noise.squeeze(), curvature)
        mutated = hyp.mobius_add(parent.squeeze(), noise_in_ball, curvature)
        mutated = hyp.project_to_ball(mutated, curvature, 0.95)
        return mutated.unsqueeze(0)

    elif condition == "hyp_local_expmap":
        # Exponential map at the parent (not origin) - local perturbation
        mutated = hyp.expmap(noise, parent, curvature)
        mutated = hyp.project_to_ball(mutated.squeeze(), curvature, 0.95)
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
) -> Tuple[Tensor, list, list]:
    """Run evolution with V11 rigor + V12 mutation operators.

    Returns (best_latent, fitness_curve, radius_diagnostics).
    """
    fitness_curve = []
    radius_diagnostics = []
    dim = seed_latent.numel()
    device = seed_latent.device

    ball_radius = (1.0 / math.sqrt(curvature)) * 0.95
    target_init_norm = 0.5 * ball_radius

    is_hyp = condition.startswith("hyp_")

    if is_hyp:
        seed_norm = seed_latent.squeeze().norm().item()
        hyp_target = min(target_init_norm * math.sqrt(curvature), 0.999)
        tangent_norm = math.atanh(hyp_target) / math.sqrt(curvature)
        init_scale = tangent_norm / max(seed_norm, 1e-8)
        seed_latent = hyp.expmap0(
            seed_latent.squeeze() * init_scale, curvature
        ).unsqueeze(0)
    elif condition == "euc_constrained":
        norm = seed_latent.squeeze().norm().item()
        if norm > 0:
            seed_latent = seed_latent * (target_init_norm / norm)

    # Isolated RNGs (same seed for all conditions)
    mut_rng = torch.Generator()
    mut_rng.manual_seed(condition_seed)
    task_rng = random.Random(condition_seed + 7)

    population = [Candidate(latent=seed_latent.clone())]
    for _ in range(population_size - 1):
        noise = _make_noise(seed_latent.shape, noise_scale, dim, mut_rng, device)
        mutated = _apply_mutation(
            seed_latent, noise, condition, curvature, ball_radius
        )
        population.append(Candidate(latent=mutated))

    global_best = Candidate(latent=seed_latent.clone(), fitness=-1.0)

    for gen in range(generations):
        gen_tasks = task_rng.sample(
            train_tasks, min(tasks_per_gen, len(train_tasks))
        )

        for cand in population:
            score, _ = evaluate_on_tasks_dense(
                cand.latent, gen_tasks, encoder, is_hyp, curvature,
            )
            cand.fitness = score

        gen_best = max(population, key=lambda c: c.fitness)
        if gen_best.fitness > global_best.fitness:
            global_best = Candidate(
                latent=gen_best.latent.clone(), fitness=gen_best.fitness,
            )

        fitnesses = [c.fitness for c in population]
        norms = [c.latent.squeeze().norm().item() for c in population]
        curve_entry = {
            "gen": gen + 1,
            "best": max(fitnesses),
            "mean": sum(fitnesses) / len(fitnesses),
            "min": min(fitnesses),
        }
        fitness_curve.append(curve_entry)

        # Radius-stratified diagnostics (Codex V12 recommendation)
        radius_entry = {
            "gen": gen + 1,
            "mean_norm": sum(norms) / len(norms),
            "min_norm": min(norms),
            "max_norm": max(norms),
            "norm_as_fraction_of_ball": [n / ball_radius for n in norms],
            "best_norm": gen_best.latent.squeeze().norm().item(),
        }
        radius_diagnostics.append(radius_entry)

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
            mutated = _apply_mutation(
                parent.latent, noise, condition, curvature, ball_radius,
            )
            new_pop.append(Candidate(latent=mutated))

        population = new_pop
        print(
            f"  [GEN {gen+1}] best={curve_entry['best']:.3f}"
            f" mean={curve_entry['mean']:.3f}"
            f" norm={radius_entry['mean_norm']:.3f}/{ball_radius:.3f}",
            flush=True,
        )

    return global_best.latent, fitness_curve, radius_diagnostics


# =====================================================================
# Statistics (inherited from V11 + V12 extensions)
# =====================================================================

def compute_statistics(results_by_condition: dict, task_ids: list) -> dict:
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
    parser = argparse.ArgumentParser(description="V12: Mobius mutations + ablation")
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
    print("VERIFIABLE EVOLUTION V12 - MOBIUS MUTATIONS + ABLATION", flush=True)
    print("=" * 70, flush=True)
    print("NEW IN V12 (based on Codex design review + 2025 research):", flush=True)
    print("  1. Mobius addition mutation (uniform perturbation, no cancellation)", flush=True)
    print("  2. Local expmap mutation (boundary-exploring)", flush=True)
    print("  3. Origin round-trip retained (V11 style, for comparison)", flush=True)
    print("  4. Radius-stratified diagnostics per generation", flush=True)
    print("  5. 6 conditions for clean mutation operator ablation", flush=True)
    print("INHERITED from V11 (all 10 Codex V10 fixes):", flush=True)
    print("  Matched ball radii, RNG isolation, unique tasks, per-seed McNemar,", flush=True)
    print("  global best, strict parsing, dense score, Bonferroni, dim-normalized noise", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    b = args.branching
    print(f"Branching: {b} (d2: {b**2} unique, d3: {b**3} unique)", flush=True)
    print(f"Test: {args.test_tasks_per_depth * 2} ({args.test_tasks_per_depth}/depth)", flush=True)
    print(f"Train: {args.train_tasks_per_depth * 2} ({args.train_tasks_per_depth}/depth)", flush=True)
    print(f"Evolution: {args.evo_gens} gens, pop={args.evo_pop}, tasks/gen={args.evo_tasks}", flush=True)
    print(f"Curvature: {curvature} (pre-registered from V7)", flush=True)
    print(f"Ball radius: {ball_radius:.3f}", flush=True)
    print("Conditions: no_evolution, euc_constrained, hyp_origin_roundtrip,", flush=True)
    print("            hyp_mobius, hyp_local_expmap, euc_unconstrained", flush=True)
    print("Primary: euc_constrained vs hyp_mobius (flat vs best geometric)", flush=True)
    print("Secondary: hyp_origin_roundtrip vs hyp_mobius (cancellation effect)", flush=True)
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

    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(model_name=args.model, quantization="4bit")

    evolved_conditions = [
        "euc_constrained",
        "hyp_origin_roundtrip",
        "hyp_mobius",
        "hyp_local_expmap",
        "euc_unconstrained",
    ]
    all_conditions = ["no_evolution"] + evolved_conditions

    all_results = {c: [] for c in all_conditions}
    all_fitness_curves = {c: [] for c in evolved_conditions}
    all_radius_diag = {c: [] for c in evolved_conditions}

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

        # No-evolution baseline (norm-matched)
        print("\n[NO_EVOLUTION] Testing (no optimization)...", flush=True)
        no_evo_latent = seed_latent.clone()
        target_init_norm = 0.5 * ball_radius
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

        # Evolved conditions
        for cond in evolved_conditions:
            is_hyp = cond.startswith("hyp_")
            condition_seed = seed  # SAME for all conditions

            print(f"\n[{cond.upper()}] Evolution...", flush=True)
            evolved_latent, curve, rad_diag = run_evolution(
                encoder, train_tasks, seed_latent.clone(),
                condition=cond,
                curvature=curvature,
                generations=args.evo_gens,
                population_size=args.evo_pop,
                tasks_per_gen=args.evo_tasks,
                condition_seed=condition_seed,
            )
            all_fitness_curves[cond].append(curve)
            all_radius_diag[cond].append(rad_diag)

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

    # ---- Radius diagnostics ----
    print(f"\n{'=' * 70}", flush=True)
    print("RADIUS DIAGNOSTICS (mean norm / ball_radius per gen)", flush=True)
    print(f"{'=' * 70}", flush=True)
    for cond in evolved_conditions:
        print(f"\n[{cond.upper()}]", flush=True)
        for si, rad in enumerate(all_radius_diag[cond]):
            norms_str = " -> ".join(
                f"{e['mean_norm']:.3f}" for e in rad
            )
            print(f"  Seed {si+1}: mean_norm {norms_str} (ball={ball_radius:.3f})", flush=True)

    # ---- Statistics ----
    print(f"\n{'=' * 70}", flush=True)
    print("STATISTICAL ANALYSIS", flush=True)
    print(f"{'=' * 70}", flush=True)

    stats_result = compute_statistics(all_results, test_task_ids)

    print("\nOverall Accuracy (mean +/- std):", flush=True)
    for cond in all_conditions:
        s = stats_result["per_condition"][cond]
        print(f"  {cond:24s}: {s['mean']*100:.1f}% +/- {s['std']*100:.1f}%", flush=True)

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
            print(f"    {cond:24s}: {ds['mean']*100:.1f}% +/- {ds['std']*100:.1f}%", flush=True)

    # ---- Pre-registered primary ----
    print(f"\n{'=' * 70}", flush=True)
    print("PRE-REGISTERED PRIMARY COMPARISON", flush=True)
    print("Euclidean Constrained vs Hyperbolic Mobius", flush=True)
    print(f"(Same L2 ball radius={ball_radius:.3f}, different geometry+operator)", flush=True)
    print(f"{'=' * 70}", flush=True)

    primary_key = "euc_constrained_vs_hyp_mobius"
    if primary_key in stats_result["pairwise"]:
        kp = stats_result["pairwise"][primary_key]
        euc_s = stats_result["per_condition"]["euc_constrained"]
        mob_s = stats_result["per_condition"]["hyp_mobius"]

        print(f"  Euc Constrained:  {euc_s['mean']*100:.1f}% +/- {euc_s['std']*100:.1f}%", flush=True)
        print(f"  Hyp Mobius c=0.5: {mob_s['mean']*100:.1f}% +/- {mob_s['std']*100:.1f}%", flush=True)
        print(f"  Difference:       {kp['diff_mean']*100:+.1f}%", flush=True)
        if not np.isnan(kp['p_value_raw']):
            print(f"  p-value (raw):    {kp['p_value_raw']:.4f}", flush=True)

        if (not np.isnan(kp['p_value_raw'])
                and kp['p_value_raw'] < 0.05
                and kp['diff_mean'] > 0):
            verdict = "SIGNIFICANT: Hyp Mobius > Euclidean (p < 0.05, pre-registered)"
        elif (not np.isnan(kp['p_value_raw'])
              and kp['p_value_raw'] < 0.05
              and kp['diff_mean'] < 0):
            verdict = "SIGNIFICANT: Euclidean > Hyp Mobius (p < 0.05)"
        else:
            verdict = "NOT SIGNIFICANT (p >= 0.05)"
    else:
        verdict = "KEY COMPARISON NOT FOUND"

    print(f"\nPRIMARY VERDICT: {verdict}", flush=True)

    # ---- Secondary: cancellation effect ----
    secondary_key = "hyp_origin_roundtrip_vs_hyp_mobius"
    if secondary_key in stats_result["pairwise"]:
        sp = stats_result["pairwise"][secondary_key]
        ort_s = stats_result["per_condition"]["hyp_origin_roundtrip"]
        mob_s = stats_result["per_condition"]["hyp_mobius"]

        print(f"\nSECONDARY: Cancellation Effect (origin round-trip vs Mobius)", flush=True)
        print(f"  Origin round-trip: {ort_s['mean']*100:.1f}% +/- {ort_s['std']*100:.1f}%", flush=True)
        print(f"  Mobius addition:   {mob_s['mean']*100:.1f}% +/- {mob_s['std']*100:.1f}%", flush=True)
        print(f"  Difference:        {sp['diff_mean']*100:+.1f}%", flush=True)
        if not np.isnan(sp['p_value_raw']):
            print(f"  p-value (raw):     {sp['p_value_raw']:.4f}", flush=True)

        if (not np.isnan(sp['p_value_raw'])
                and sp['p_value_raw'] < 0.05
                and sp['diff_mean'] > 0):
            cancel_verdict = "CANCELLATION CONFIRMED: Mobius > Origin round-trip"
        else:
            cancel_verdict = "No significant cancellation effect detected"
        print(f"  VERDICT: {cancel_verdict}", flush=True)

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
        },
        "v12_changes": [
            "Mobius addition mutation (no cancellation, uniform perturbation)",
            "Local expmap mutation (boundary-exploring)",
            "Origin round-trip retained for cancellation ablation",
            "Radius-stratified diagnostics per generation",
            "6-condition mutation operator ablation",
        ],
        "inherited_v11_fixes": [
            "Matched ball radii",
            "RNG isolation (same seed all conditions)",
            "Unique tasks, no leakage",
            "Per-seed McNemar",
            "Global best tracking",
            "Strict answer parsing",
            "Depth-unbiased dense score",
            "No-evolution baseline (norm-matched)",
            "Dimension-normalized noise",
            "Bonferroni correction",
        ],
        "statistics": stats_result,
        "fitness_curves": {c: all_fitness_curves[c] for c in evolved_conditions},
        "radius_diagnostics": {c: all_radius_diag[c] for c in evolved_conditions},
        "primary_verdict": verdict,
    }

    suffix = "_diagnostic" if args.diagnostic else ""
    results_path = Path(__file__).parent / f"v12_results{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}", flush=True)


if __name__ == "__main__":
    main()
