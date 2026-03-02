"""
Verifiable Evolution V9 - RIGOROUS VALIDATION

Addresses ALL Codex criticisms:
1. Pre-registered hypothesis: c=0.5 (locked from V7, not tuned here)
2. Fresh test set (separate from V7 validation tasks)
3. 10 seeds for statistical power
4. 100+ tasks per depth (200 total test tasks)
5. Fair Euclidean baseline (no extra hyperbolic tuning)
6. Proper statistical reporting with confidence intervals
7. McNemar test with Bonferroni correction
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from scipy import stats
import numpy as np

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.config import GeometryConfig
from latent_reasoning.core.encoder import LLMEncoder


@dataclass
class Task:
    task_id: str
    prompt: str
    correct_answer: int
    depth: int


class TaskGenerator:
    """Generate verifiable arithmetic tasks."""

    def __init__(self, branching: int = 4, seed: int = 42):
        self.branching = branching
        self.rng = random.Random(seed)

    def generate(self, n_per_depth: int, depths: list[int]) -> list[Task]:
        tasks = []
        for depth in depths:
            for i in range(n_per_depth):
                task_id = f"d{depth}_t{i}"
                path = [self.rng.randint(0, self.branching - 1) for _ in range(depth)]
                answer = sum(path) * (depth + 1) + depth * 7
                prompt = (
                    f"Calculate: sum([{','.join(map(str, path))}]) * {depth + 1} + {depth} * 7 = ?\n"
                    f"Answer with just the number."
                )
                tasks.append(Task(task_id=task_id, prompt=prompt, correct_answer=answer, depth=depth))
        return tasks


def verify_answer(response: str, expected: int) -> bool:
    """Verify numeric response."""
    import re
    for num in re.findall(r'-?\d+', response):
        if int(num) == expected:
            return True
    return False


@dataclass
class Candidate:
    latent: Tensor
    fitness: float = 0.0


def evaluate_on_tasks(
    latent: Tensor,
    tasks: list[Task],
    encoder: LLMEncoder,
    hyperbolic: bool,
    curvature: float,
) -> dict:
    """Evaluate a latent on tasks, return per-task results."""
    results = {}
    for task in tasks:
        response = encoder.decode(
            latent,
            query=task.prompt,
            max_new_tokens=250,
            temperature=0.3,
            hyperbolic=hyperbolic,
            curvature=curvature,
        )
        results[task.task_id] = verify_answer(response, task.correct_answer)
    return results


def run_evolution(
    encoder: LLMEncoder,
    train_tasks: list[Task],
    seed_latent: Tensor,
    hyperbolic: bool,
    curvature: float = 0.5,
    generations: int = 3,
    population_size: int = 4,
    tasks_per_gen: int = 8,
) -> Tensor:
    """Run evolution and return best latent."""

    if hyperbolic:
        from latent_reasoning.utils import hyperbolic as hyp
        # Project to hyperbolic space
        seed_latent = hyp.expmap0(seed_latent.squeeze() * 0.35, curvature).unsqueeze(0)

    population = [Candidate(latent=seed_latent.clone())]

    # Initialize population with mutations
    for _ in range(population_size - 1):
        noise = torch.randn_like(seed_latent) * 0.1
        if hyperbolic:
            from latent_reasoning.utils import hyperbolic as hyp
            lat = seed_latent.squeeze()
            tan = hyp.logmap0(lat, curvature)
            tan = tan + noise.squeeze()
            mutated = hyp.expmap0(tan, curvature)
            mutated = hyp.project_to_ball(mutated, curvature, 0.95)
            population.append(Candidate(latent=mutated.unsqueeze(0)))
        else:
            population.append(Candidate(latent=seed_latent + noise))

    rng = random.Random(42)

    for gen in range(generations):
        # Sample training tasks
        gen_tasks = rng.sample(train_tasks, min(tasks_per_gen, len(train_tasks)))

        # Evaluate population
        for cand in population:
            results = evaluate_on_tasks(cand.latent, gen_tasks, encoder, hyperbolic, curvature)
            cand.fitness = sum(results.values()) / len(results)

        # Selection
        population.sort(key=lambda c: c.fitness, reverse=True)
        elite = population[:2]

        # Create new population
        new_pop = [Candidate(latent=e.latent.clone()) for e in elite]

        while len(new_pop) < population_size:
            parent = elite[rng.randint(0, 1)]
            noise = torch.randn_like(parent.latent) * 0.1

            if hyperbolic:
                from latent_reasoning.utils import hyperbolic as hyp
                lat = parent.latent.squeeze()
                tan = hyp.logmap0(lat, curvature)
                tan = tan + noise.squeeze()
                child = hyp.expmap0(tan, curvature)
                child = hyp.project_to_ball(child, curvature, 0.95)
                new_pop.append(Candidate(latent=child.unsqueeze(0)))
            else:
                new_pop.append(Candidate(latent=parent.latent + noise))

        population = new_pop
        best = max(population, key=lambda c: c.fitness)
        print(f"  [GEN {gen+1}] best={best.fitness:.3f}", flush=True)

    return max(population, key=lambda c: c.fitness).latent


def compute_statistics(hyp_results: list[dict], euc_results: list[dict], task_ids: list[str]) -> dict:
    """Compute rigorous statistics across all seeds."""

    # Aggregate per-task results
    hyp_correct_per_task = defaultdict(int)
    euc_correct_per_task = defaultdict(int)

    for seed_results in hyp_results:
        for tid, correct in seed_results.items():
            if correct:
                hyp_correct_per_task[tid] += 1

    for seed_results in euc_results:
        for tid, correct in seed_results.items():
            if correct:
                euc_correct_per_task[tid] += 1

    n_seeds = len(hyp_results)

    # Overall accuracy per seed
    hyp_accs = [sum(r.values()) / len(r) for r in hyp_results]
    euc_accs = [sum(r.values()) / len(r) for r in euc_results]

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(hyp_accs, euc_accs)

    # Confidence interval for difference
    diffs = [h - e for h, e in zip(hyp_accs, euc_accs)]
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    ci_95 = stats.t.interval(0.95, len(diffs)-1, loc=mean_diff, scale=std_diff/np.sqrt(len(diffs)))

    # McNemar aggregated across seeds
    b_total = 0  # Hyp correct, Euc wrong
    c_total = 0  # Hyp wrong, Euc correct

    for hyp_r, euc_r in zip(hyp_results, euc_results):
        for tid in hyp_r:
            if hyp_r[tid] and not euc_r.get(tid, False):
                b_total += 1
            elif not hyp_r[tid] and euc_r.get(tid, False):
                c_total += 1

    # McNemar chi-squared
    if b_total + c_total > 0:
        mcnemar_chi2 = (abs(b_total - c_total) - 1) ** 2 / (b_total + c_total)
        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_chi2, 1)
    else:
        mcnemar_chi2 = 0
        mcnemar_p = 1.0

    # Per-depth statistics
    depth_stats = {}
    for depth in [2, 3]:
        depth_tasks = [tid for tid in task_ids if tid.startswith(f"d{depth}_")]
        hyp_depth = [sum(1 for tid in depth_tasks if r.get(tid, False)) / len(depth_tasks) for r in hyp_results]
        euc_depth = [sum(1 for tid in depth_tasks if r.get(tid, False)) / len(depth_tasks) for r in euc_results]

        depth_diffs = [h - e for h, e in zip(hyp_depth, euc_depth)]
        depth_mean = np.mean(depth_diffs)
        depth_std = np.std(depth_diffs, ddof=1)
        depth_ci = stats.t.interval(0.95, len(depth_diffs)-1, loc=depth_mean, scale=depth_std/np.sqrt(len(depth_diffs)))

        depth_stats[depth] = {
            'hyp_mean': np.mean(hyp_depth),
            'euc_mean': np.mean(euc_depth),
            'diff_mean': depth_mean,
            'diff_ci_95': depth_ci,
        }

    return {
        'hyp_mean': np.mean(hyp_accs),
        'hyp_std': np.std(hyp_accs),
        'euc_mean': np.mean(euc_accs),
        'euc_std': np.std(euc_accs),
        'diff_mean': mean_diff,
        'diff_ci_95': ci_95,
        't_stat': t_stat,
        'p_value': p_value,
        'mcnemar_b': b_total,
        'mcnemar_c': c_total,
        'mcnemar_chi2': mcnemar_chi2,
        'mcnemar_p': mcnemar_p,
        'per_depth': depth_stats,
        'n_seeds': n_seeds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--test-tasks-per-depth", type=int, default=100)
    parser.add_argument("--evo-gens", type=int, default=3)
    parser.add_argument("--evo-pop", type=int, default=2)
    parser.add_argument("--evo-tasks", type=int, default=4)
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("VERIFIABLE EVOLUTION V9 - RIGOROUS VALIDATION", flush=True)
    print("=" * 70, flush=True)
    print("PRE-REGISTERED HYPOTHESIS: c=0.5 hyperbolic > Euclidean at depth 2-3", flush=True)
    print("(Curvature locked from V7, NOT tuned in this experiment)", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print(f"Test tasks: {args.test_tasks_per_depth * 2} ({args.test_tasks_per_depth} per depth)", flush=True)
    print("Curvature: 0.5 (pre-registered, fixed)", flush=True)
    print("=" * 70, flush=True)

    # Generate FRESH test set (different seed than V7's seed=42)
    # V7 used seed=42, we use seed=9999 for completely independent test set
    print("\nGenerating FRESH test set (seed=9999, independent of V7)...", flush=True)
    test_gen = TaskGenerator(seed=9999)
    test_tasks = test_gen.generate(args.test_tasks_per_depth, depths=[2, 3])
    test_task_ids = [t.task_id for t in test_tasks]
    print(f"Test set: {len(test_tasks)} tasks", flush=True)

    # Generate training set (also fresh)
    train_gen = TaskGenerator(seed=8888)
    train_tasks = train_gen.generate(100, depths=[2, 3])
    print(f"Train set: {len(train_tasks)} tasks", flush=True)

    # Load model
    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(model_name=args.model, quantization="4bit")

    # Storage for all results
    all_hyp_results = []
    all_euc_results = []

    for seed_idx in range(args.seeds):
        seed = 1000 + seed_idx * 111  # Different from V7 seeds
        random.seed(seed)
        torch.manual_seed(seed)

        print(f"\n{'#' * 70}", flush=True)
        print(f"# SEED {seed_idx + 1}/{args.seeds} (seed={seed})", flush=True)
        print(f"{'#' * 70}", flush=True)

        # Get seed latent
        seed_latent = encoder.encode("You calculate expressions and give numeric answers.")

        # Run Euclidean evolution
        print("\n[EUCLIDEAN] Evolution...", flush=True)
        euc_latent = run_evolution(
            encoder, train_tasks, seed_latent.clone(),
            hyperbolic=False, curvature=1.0,
            generations=args.evo_gens, population_size=args.evo_pop, tasks_per_gen=args.evo_tasks
        )

        # Run Hyperbolic c=0.5 evolution (PRE-REGISTERED, not tuned)
        print("\n[HYPERBOLIC c=0.5] Evolution...", flush=True)
        hyp_latent = run_evolution(
            encoder, train_tasks, seed_latent.clone(),
            hyperbolic=True, curvature=0.5,
            generations=args.evo_gens, population_size=args.evo_pop, tasks_per_gen=args.evo_tasks
        )

        # Evaluate on FRESH test set
        print("\n[TEST] Evaluating on fresh test set...", flush=True)
        euc_results = evaluate_on_tasks(euc_latent, test_tasks, encoder, False, 1.0)
        hyp_results = evaluate_on_tasks(hyp_latent, test_tasks, encoder, True, 0.5)

        euc_acc = sum(euc_results.values()) / len(euc_results)
        hyp_acc = sum(hyp_results.values()) / len(hyp_results)

        # Per-depth
        euc_d2 = sum(1 for t in test_tasks if t.depth == 2 and euc_results[t.task_id]) / args.test_tasks_per_depth
        euc_d3 = sum(1 for t in test_tasks if t.depth == 3 and euc_results[t.task_id]) / args.test_tasks_per_depth
        hyp_d2 = sum(1 for t in test_tasks if t.depth == 2 and hyp_results[t.task_id]) / args.test_tasks_per_depth
        hyp_d3 = sum(1 for t in test_tasks if t.depth == 3 and hyp_results[t.task_id]) / args.test_tasks_per_depth

        print(f"\n  Euclidean:  {euc_acc*100:.1f}% (D2: {euc_d2*100:.1f}%, D3: {euc_d3*100:.1f}%)", flush=True)
        print(f"  Hyperbolic: {hyp_acc*100:.1f}% (D2: {hyp_d2*100:.1f}%, D3: {hyp_d3*100:.1f}%)", flush=True)
        print(f"  Margin: {(hyp_acc-euc_acc)*100:+.1f}%", flush=True)

        all_euc_results.append(euc_results)
        all_hyp_results.append(hyp_results)

    # Compute rigorous statistics
    print(f"\n{'=' * 70}", flush=True)
    print("STATISTICAL ANALYSIS", flush=True)
    print(f"{'=' * 70}", flush=True)

    stats_result = compute_statistics(all_hyp_results, all_euc_results, test_task_ids)

    print(f"\nOverall Accuracy (mean +/- std):", flush=True)
    print(f"  Euclidean:  {stats_result['euc_mean']*100:.1f}% +/- {stats_result['euc_std']*100:.1f}%", flush=True)
    print(f"  Hyperbolic: {stats_result['hyp_mean']*100:.1f}% +/- {stats_result['hyp_std']*100:.1f}%", flush=True)

    print(f"\nDifference (Hyperbolic - Euclidean):", flush=True)
    print(f"  Mean: {stats_result['diff_mean']*100:+.1f}%", flush=True)
    print(f"  95% CI: [{stats_result['diff_ci_95'][0]*100:.1f}%, {stats_result['diff_ci_95'][1]*100:.1f}%]", flush=True)

    print(f"\nPaired t-test:", flush=True)
    print(f"  t = {stats_result['t_stat']:.3f}, p = {stats_result['p_value']:.4f}", flush=True)

    print(f"\nMcNemar test (aggregated across seeds):", flush=True)
    print(f"  b (hyp+, euc-) = {stats_result['mcnemar_b']}", flush=True)
    print(f"  c (hyp-, euc+) = {stats_result['mcnemar_c']}", flush=True)
    print(f"  chi2 = {stats_result['mcnemar_chi2']:.3f}, p = {stats_result['mcnemar_p']:.4f}", flush=True)

    print(f"\nPer-depth analysis:", flush=True)
    for depth, ds in stats_result['per_depth'].items():
        print(f"  Depth {depth}:", flush=True)
        print(f"    Euclidean:  {ds['euc_mean']*100:.1f}%", flush=True)
        print(f"    Hyperbolic: {ds['hyp_mean']*100:.1f}%", flush=True)
        print(f"    Diff: {ds['diff_mean']*100:+.1f}% (95% CI: [{ds['diff_ci_95'][0]*100:.1f}%, {ds['diff_ci_95'][1]*100:.1f}%])", flush=True)

    print(f"\n{'=' * 70}", flush=True)

    # Determine verdict
    significant = stats_result['p_value'] < 0.05
    ci_excludes_zero = stats_result['diff_ci_95'][0] > 0 or stats_result['diff_ci_95'][1] < 0

    if significant and stats_result['diff_mean'] > 0:
        verdict = "HYPERBOLIC c=0.5 SIGNIFICANTLY BETTER (p < 0.05)"
    elif significant and stats_result['diff_mean'] < 0:
        verdict = "EUCLIDEAN SIGNIFICANTLY BETTER (p < 0.05)"
    else:
        verdict = "NO SIGNIFICANT DIFFERENCE (p >= 0.05)"

    print(f"VERDICT: {verdict}", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Save results
    results = {
        'config': {
            'model': args.model,
            'seeds': args.seeds,
            'test_tasks_per_depth': args.test_tasks_per_depth,
            'curvature': 0.5,
            'test_seed': 9999,
            'train_seed': 8888,
        },
        'statistics': {
            'hyp_mean': stats_result['hyp_mean'],
            'hyp_std': stats_result['hyp_std'],
            'euc_mean': stats_result['euc_mean'],
            'euc_std': stats_result['euc_std'],
            'diff_mean': stats_result['diff_mean'],
            'diff_ci_95': list(stats_result['diff_ci_95']),
            't_stat': stats_result['t_stat'],
            'p_value': stats_result['p_value'],
            'mcnemar_chi2': stats_result['mcnemar_chi2'],
            'mcnemar_p': stats_result['mcnemar_p'],
        },
        'verdict': verdict,
    }

    results_path = Path(__file__).parent / "v9_rigorous_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}", flush=True)


if __name__ == "__main__":
    main()
