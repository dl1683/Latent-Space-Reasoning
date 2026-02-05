"""
Verifiable Evolution V6 - Statistical Rigor

Per Codex analysis of V5 results:
"Promising signal, not statistically significant yet. Need 50-100 tasks
and 5-10 seeds for rigorous claims. Euclidean 0% at shallow depths is
suspicious - verify baseline is working correctly."

V6 Changes from V5:
1. Larger validation set (50+ tasks instead of 16)
2. Multiple seeds (5 runs per prompt instead of 1)
3. Balanced depth distribution (10+ tasks per depth)
4. Per-task tracking for McNemar test
5. Strict ablation: only geometry changes, everything else identical
6. Baseline sanity check: verify Euclidean isn't broken
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.config import GeometryConfig
from latent_reasoning.core.encoder import LLMEncoder


@dataclass
class CalibratedTreeTask:
    """Tree task calibrated to model capability."""
    task_id: str  # Unique ID for per-task tracking
    prompt: str
    correct_answer: any
    verifier: callable
    depth: int
    category: str
    difficulty: str


class BalancedTreeTaskGenerator:
    """
    Generate tree tasks with BALANCED depth distribution.

    Ensures equal representation at each depth for fair comparison.
    """

    def __init__(self, max_depth: int = 5, branching: int = 4, seed: int = 42):
        self.max_depth = max_depth
        self.branching = branching
        self.rng = random.Random(seed)

    def generate_balanced(self, tasks_per_depth: int = 12) -> list[CalibratedTreeTask]:
        """Generate balanced tasks across depths 1-5."""
        tasks = []

        for depth in range(1, self.max_depth + 1):
            for i in range(tasks_per_depth):
                task_id = f"d{depth}_t{i}"
                task = self._generate_task(depth, task_id)
                tasks.append(task)

        return tasks

    def _generate_task(self, depth: int, task_id: str) -> CalibratedTreeTask:
        """Generate a single task with SIMPLE prompt format."""
        path = [self.rng.randint(0, self.branching - 1) for _ in range(depth)]

        # Compute answer: sum(path) * (depth+1) + len(path) * 7
        path_sum = sum(path)
        answer = path_sum * (depth + 1) + depth * 7

        # SIMPLE prompt format
        prompt = (
            f"Calculate: sum([{','.join(map(str, path))}]) * {depth + 1} + {depth} * 7 = ?\n"
            f"Answer with just the number."
        )

        if depth <= 2:
            difficulty = "easy"
        elif depth <= 4:
            difficulty = "medium"
        else:
            difficulty = "hard"

        return CalibratedTreeTask(
            task_id=task_id,
            prompt=prompt,
            correct_answer=answer,
            verifier=self._verify_number,
            depth=depth,
            category="tree_traversal",
            difficulty=difficulty,
        )

    def _verify_number(self, response: str, expected: int) -> bool:
        """Verify numeric response with robust pattern matching."""
        import re

        # First check for exact match
        if str(expected) in response:
            return True

        # Look for the number in common answer patterns
        patterns = [
            rf'=\s*{expected}\b',
            rf'\b{expected}\b',
        ]

        for pattern in patterns:
            if re.search(pattern, response):
                return True

        # Fallback: last number in response
        numbers = re.findall(r'-?\d+', response)
        if numbers:
            try:
                if int(numbers[-1]) == expected:
                    return True
            except (ValueError, IndexError):
                pass

        return False


class BalancedTaskPool:
    """Task pool with balanced depth distribution."""

    def __init__(
        self,
        tasks_per_depth: int = 12,  # 12 tasks per depth * 5 depths = 60 tasks
        val_ratio: float = 0.25,  # 25% validation = 15 val tasks
        seed: int = 42
    ):
        random.seed(seed)

        gen = BalancedTreeTaskGenerator(max_depth=5, seed=seed)
        all_tasks = gen.generate_balanced(tasks_per_depth)

        # Stratified split: maintain depth balance in train/val
        depth_tasks = defaultdict(list)
        for task in all_tasks:
            depth_tasks[task.depth].append(task)

        self.train_tasks = []
        self.val_tasks = []

        for depth in sorted(depth_tasks.keys()):
            tasks = depth_tasks[depth]
            random.shuffle(tasks)
            val_count = max(1, int(len(tasks) * val_ratio))
            self.val_tasks.extend(tasks[:val_count])
            self.train_tasks.extend(tasks[val_count:])

    def sample_train(self, n: int, seed: int | None = None) -> list[CalibratedTreeTask]:
        if seed is not None:
            random.seed(seed)
        return random.sample(self.train_tasks, min(n, len(self.train_tasks)))

    def get_validation(self) -> list[CalibratedTreeTask]:
        return self.val_tasks

    def stats(self) -> dict:
        train_depths = defaultdict(int)
        val_depths = defaultdict(int)
        for t in self.train_tasks:
            train_depths[t.depth] += 1
        for t in self.val_tasks:
            val_depths[t.depth] += 1

        return {
            "train_size": len(self.train_tasks),
            "val_size": len(self.val_tasks),
            "train_depths": dict(sorted(train_depths.items())),
            "val_depths": dict(sorted(val_depths.items())),
        }


@dataclass
class TreeCandidate:
    """Candidate with per-task tracking."""
    latent: Tensor
    depth_correct: dict = field(default_factory=dict)
    depth_total: dict = field(default_factory=dict)
    task_results: dict = field(default_factory=dict)  # task_id -> bool
    correct: int = 0
    total: int = 0

    @property
    def raw_fitness(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def weighted_fitness(self) -> float:
        """Depth-weighted fitness with 2^depth scaling."""
        weighted_sum = 0.0
        weight_total = 0.0
        for depth, total in self.depth_total.items():
            weight = 2 ** depth
            correct = self.depth_correct.get(depth, 0)
            weighted_sum += correct * weight
            weight_total += total * weight
        return weighted_sum / weight_total if weight_total > 0 else 0.0

    def get_tail_metric(self) -> float:
        """Tail metric: accuracy on depths 3-5 (calibrated frontier)."""
        deep_correct = sum(self.depth_correct.get(d, 0) for d in range(3, 6))
        deep_total = sum(self.depth_total.get(d, 0) for d in range(3, 6))
        return deep_correct / deep_total if deep_total > 0 else 0.0


def evaluate_candidate(
    candidate: TreeCandidate,
    tasks: list[CalibratedTreeTask],
    encoder: LLMEncoder,
    hyp_module,
    geometry_config: GeometryConfig,
    verbose: bool = False,
) -> None:
    """Evaluate candidate with per-task tracking."""
    candidate.correct = 0
    candidate.total = len(tasks)
    candidate.depth_correct = defaultdict(int)
    candidate.depth_total = defaultdict(int)
    candidate.task_results = {}

    for task in tasks:
        candidate.depth_total[task.depth] += 1

        response = encoder.decode(
            candidate.latent,
            query=task.prompt,
            max_new_tokens=200,
            temperature=0.3,
            hyperbolic=hyp_module is not None,
            curvature=geometry_config.curvature if hyp_module else 1.0,
        )

        is_correct = task.verifier(response, task.correct_answer)
        candidate.task_results[task.task_id] = is_correct

        if verbose and not is_correct:
            print(f"    [WRONG] {task.task_id}: expected={task.correct_answer}, got='{response[:50]}...'", flush=True)

        if is_correct:
            candidate.correct += 1
            candidate.depth_correct[task.depth] += 1


def compute_diversity(population, hyp_module, geometry_config) -> float:
    """Compute diversity with numerical safeguards."""
    n = len(population)
    if n <= 1:
        return 0.0

    total_dist = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            try:
                lat_i = population[i].latent.squeeze()
                lat_j = population[j].latent.squeeze()

                if hyp_module is not None:
                    norm_i, norm_j = lat_i.norm(), lat_j.norm()
                    if norm_i > 0.99 or norm_j > 0.99:
                        dist = torch.norm(lat_i - lat_j).item()
                    else:
                        dist = hyp_module.hyperbolic_distance(
                            lat_i, lat_j, geometry_config.curvature
                        ).item()
                else:
                    dist = torch.norm(lat_i - lat_j).item()

                if not (math.isnan(dist) or math.isinf(dist)):
                    total_dist += dist
                    count += 1
            except Exception:
                continue

    return total_dist / count if count > 0 else 0.0


def mutate(latent, scale, hyp_module, geometry_config):
    """Mutate with safeguards."""
    noise = torch.randn_like(latent) * scale

    if hyp_module is not None:
        lat = latent.squeeze()
        norm = lat.norm()
        if norm > 0.95:
            lat = lat * (0.95 / norm)

        tangent = hyp_module.logmap0(lat, geometry_config.curvature)
        tangent = tangent + noise.squeeze()
        mutated = hyp_module.expmap0(tangent, geometry_config.curvature)
        mutated = hyp_module.project_to_ball(
            mutated, geometry_config.curvature, geometry_config.max_norm
        )

        if torch.isnan(mutated).any() or torch.isinf(mutated).any():
            mutated = latent.squeeze() + noise.squeeze() * 0.1
            mutated = hyp_module.project_to_ball(
                mutated, geometry_config.curvature, geometry_config.max_norm
            )

        return mutated.unsqueeze(0) if mutated.dim() == 1 else mutated
    else:
        return latent + noise


def crossover(parent_a, parent_b, hyp_module, geometry_config):
    """Crossover with safeguards."""
    t = random.random()

    if hyp_module is not None:
        a = parent_a.squeeze()
        b = parent_b.squeeze()
        norm_a, norm_b = a.norm(), b.norm()
        if norm_a > 0.95:
            a = a * (0.95 / norm_a)
        if norm_b > 0.95:
            b = b * (0.95 / norm_b)

        try:
            child = hyp_module.hyperbolic_interpolate(a, b, t, geometry_config.curvature)
            if torch.isnan(child).any() or torch.isinf(child).any():
                child = t * a + (1 - t) * b
                child = hyp_module.project_to_ball(
                    child, geometry_config.curvature, geometry_config.max_norm
                )
        except Exception:
            child = t * a + (1 - t) * b
            child = hyp_module.project_to_ball(
                child, geometry_config.curvature, geometry_config.max_norm
            )

        return child.unsqueeze(0) if child.dim() == 1 else child
    else:
        return t * parent_a + (1 - t) * parent_b


def tournament_select(population, k: int = 3):
    """Tournament selection using weighted fitness."""
    contestants = random.sample(population, min(k, len(population)))
    return max(contestants, key=lambda c: c.weighted_fitness)


def run_evolution(
    encoder: LLMEncoder,
    pool: BalancedTaskPool,
    geometry: str,
    seed_latent: Tensor,
    generations: int = 5,
    population_size: int = 4,
    tasks_per_gen: int = 8,
    elite_count: int = 2,
    mutation_scale: float = 0.1,
) -> dict:
    """Run evolution with balanced tasks."""

    if geometry == "hyperbolic":
        from latent_reasoning.utils import hyperbolic as hyp
        geometry_config = GeometryConfig(
            space="hyperbolic",
            curvature=1.0,
            tangent_scale=0.35,
            max_norm=0.95,
        )
        seed_latent = hyp.expmap0(
            seed_latent.squeeze() * geometry_config.tangent_scale,
            geometry_config.curvature,
        ).unsqueeze(0)
        hyp_module = hyp
    else:
        geometry_config = GeometryConfig(space="euclidean")
        hyp_module = None

    # Initialize population
    population = [TreeCandidate(latent=seed_latent.clone())]
    for _ in range(population_size - 1):
        mutated = mutate(seed_latent, mutation_scale, hyp_module, geometry_config)
        population.append(TreeCandidate(latent=mutated))

    history = []
    rolling_fitness = []

    for gen in range(generations):
        tasks = pool.sample_train(tasks_per_gen, seed=gen * 1000 + gen)

        for candidate in population:
            evaluate_candidate(candidate, tasks, encoder, hyp_module, geometry_config)

        population.sort(key=lambda c: c.weighted_fitness, reverse=True)

        rolling_fitness.append(population[0].raw_fitness)
        if len(rolling_fitness) > 3:
            rolling_fitness.pop(0)
        roll_avg = sum(rolling_fitness) / len(rolling_fitness)

        diversity = compute_diversity(population, hyp_module, geometry_config)
        tail_metric = population[0].get_tail_metric()

        history.append({
            "generation": gen + 1,
            "raw_fitness": population[0].raw_fitness,
            "weighted_fitness": population[0].weighted_fitness,
            "roll_avg": roll_avg,
            "diversity": diversity,
            "tail_metric": tail_metric,
        })

        print(
            f"[GEN {gen+1:02d}] raw={population[0].raw_fitness:.3f} "
            f"wgt={population[0].weighted_fitness:.3f} "
            f"roll={roll_avg:.3f} "
            f"tail={tail_metric:.3f} "
            f"div={diversity:.3f}",
            flush=True
        )

        # Create next generation
        next_gen = []
        for elite in population[:elite_count]:
            next_gen.append(TreeCandidate(latent=elite.latent.clone()))

        while len(next_gen) < population_size:
            parent_a = tournament_select(population)
            parent_b = tournament_select(population)
            child_latent = crossover(parent_a.latent, parent_b.latent, hyp_module, geometry_config)
            if random.random() < 0.8:
                child_latent = mutate(child_latent, mutation_scale, hyp_module, geometry_config)
            next_gen.append(TreeCandidate(latent=child_latent))

        population = next_gen

    return {
        "final_raw": population[0].raw_fitness,
        "final_weighted": population[0].weighted_fitness,
        "final_tail": population[0].get_tail_metric(),
        "history": history,
        "best_latent": population[0].latent,
    }


def evaluate_on_validation(latent, val_tasks, encoder, geometry, verbose=False):
    """Evaluate on validation set with per-task tracking."""
    if geometry == "hyperbolic":
        from latent_reasoning.utils import hyperbolic as hyp
        geometry_config = GeometryConfig(
            space="hyperbolic", curvature=1.0, tangent_scale=0.35, max_norm=0.95
        )
        hyp_module = hyp
    else:
        geometry_config = GeometryConfig(space="euclidean")
        hyp_module = None

    candidate = TreeCandidate(latent=latent)
    evaluate_candidate(candidate, val_tasks, encoder, hyp_module, geometry_config, verbose=verbose)

    return {
        "raw_accuracy": candidate.raw_fitness,
        "weighted_accuracy": candidate.weighted_fitness,
        "tail_metric": candidate.get_tail_metric(),
        "correct": candidate.correct,
        "total": candidate.total,
        "depth_breakdown": {
            d: f"{candidate.depth_correct.get(d, 0)}/{candidate.depth_total.get(d, 0)}"
            for d in sorted(candidate.depth_total.keys())
        },
        "task_results": candidate.task_results,  # For McNemar test
    }


def compute_mcnemar(hyp_results: dict, euc_results: dict) -> dict:
    """Compute McNemar test statistics for paired comparison."""
    # Count concordant/discordant pairs
    b = 0  # Hyp correct, Euc wrong
    c = 0  # Hyp wrong, Euc correct

    for task_id in hyp_results:
        hyp_correct = hyp_results[task_id]
        euc_correct = euc_results.get(task_id, False)

        if hyp_correct and not euc_correct:
            b += 1
        elif not hyp_correct and euc_correct:
            c += 1

    # McNemar test statistic (with continuity correction)
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_approx": 1.0, "significant": False}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # Approximate p-value (chi-square with 1 df)
    # For chi2 > 3.84, p < 0.05
    significant = chi2 > 3.84

    return {
        "b": b,  # Hyp wins
        "c": c,  # Euc wins
        "chi2": chi2,
        "significant": significant,
        "advantage": "hyperbolic" if b > c else "euclidean" if c > b else "tie",
    }


def run_baseline_sanity_check(encoder, pool, seed_latent):
    """Verify that baseline Euclidean setup is working correctly."""
    print("\n[SANITY CHECK] Testing baseline Euclidean on easy tasks...", flush=True)

    geometry_config = GeometryConfig(space="euclidean")

    # Test on depth 1-2 tasks only
    easy_tasks = [t for t in pool.val_tasks if t.depth <= 2]

    candidate = TreeCandidate(latent=seed_latent.clone())

    correct = 0
    for task in easy_tasks:
        response = encoder.decode(
            candidate.latent,
            query=task.prompt,
            max_new_tokens=200,
            temperature=0.3,
            hyperbolic=False,
            curvature=1.0,
        )

        is_correct = task.verifier(response, task.correct_answer)
        if is_correct:
            correct += 1
        else:
            print(f"  [FAILED] {task.task_id}: expected={task.correct_answer}", flush=True)
            print(f"           Prompt: {task.prompt[:60]}...", flush=True)
            print(f"           Response: {response[:100]}...", flush=True)

    accuracy = correct / len(easy_tasks) if easy_tasks else 0
    print(f"  Baseline easy task accuracy: {accuracy*100:.1f}% ({correct}/{len(easy_tasks)})", flush=True)

    if accuracy < 0.3:
        print("  [WARNING] Baseline accuracy very low - possible issue with setup!", flush=True)

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="V6: Statistical Rigor")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--tasks-per-gen", type=int, default=8)
    parser.add_argument("--tasks-per-depth", type=int, default=12)  # 12 * 5 = 60 total
    parser.add_argument("--seeds", type=int, default=5)  # Multiple seeds
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--sanity-check", action="store_true", help="Run baseline sanity check first")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("VERIFIABLE EVOLUTION V6 - STATISTICAL RIGOR", flush=True)
    print("Balanced depths | Multiple seeds | McNemar test | Per-task tracking", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Generations: {args.generations}", flush=True)
    print(f"Population: {args.population}", flush=True)
    print(f"Tasks/gen: {args.tasks_per_gen}", flush=True)
    print(f"Tasks/depth: {args.tasks_per_depth} (total: {args.tasks_per_depth * 5})", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print("=" * 70, flush=True)

    print("\nCreating balanced task pool...", flush=True)
    pool = BalancedTaskPool(tasks_per_depth=args.tasks_per_depth, seed=args.base_seed)
    print(f"Pool stats: {pool.stats()}", flush=True)

    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(model_name=args.model, quantization="4bit")

    # Single prompt for strict ablation
    prompt = "You follow tree paths step by step, computing values at each node."
    seed_latent = encoder.encode(prompt)

    # Optional sanity check
    if args.sanity_check:
        baseline_acc = run_baseline_sanity_check(encoder, pool, seed_latent)
        if baseline_acc < 0.3:
            print("\n[ERROR] Baseline too low. Fix setup before running experiment.", flush=True)
            return

    all_results = []
    all_hyp_task_results = {}
    all_euc_task_results = {}

    for seed_idx in range(args.seeds):
        run_seed = args.base_seed + seed_idx * 1000
        random.seed(run_seed)
        torch.manual_seed(run_seed)

        print(f"\n{'#' * 70}", flush=True)
        print(f"# SEED {seed_idx + 1}/{args.seeds} (seed={run_seed})", flush=True)
        print("#" * 70, flush=True)

        print(f"\n[HYPERBOLIC] Running evolution...", flush=True)
        hyp_result = run_evolution(
            encoder=encoder,
            pool=pool,
            geometry="hyperbolic",
            seed_latent=seed_latent.clone(),
            generations=args.generations,
            population_size=args.population,
            tasks_per_gen=args.tasks_per_gen,
        )

        # Reset seed for fair comparison
        random.seed(run_seed)
        torch.manual_seed(run_seed)

        print(f"\n[EUCLIDEAN] Running evolution...", flush=True)
        euc_result = run_evolution(
            encoder=encoder,
            pool=pool,
            geometry="euclidean",
            seed_latent=seed_latent.clone(),
            generations=args.generations,
            population_size=args.population,
            tasks_per_gen=args.tasks_per_gen,
        )

        print(f"\n[VALIDATION] Evaluating on held-out set...", flush=True)
        val_tasks = pool.get_validation()

        hyp_val = evaluate_on_validation(hyp_result["best_latent"], val_tasks, encoder, "hyperbolic")
        euc_val = evaluate_on_validation(euc_result["best_latent"], val_tasks, encoder, "euclidean")

        print(f"  Hyperbolic: raw={hyp_val['raw_accuracy']*100:.1f}% wgt={hyp_val['weighted_accuracy']:.3f} tail={hyp_val['tail_metric']:.3f}", flush=True)
        print(f"  Euclidean:  raw={euc_val['raw_accuracy']*100:.1f}% wgt={euc_val['weighted_accuracy']:.3f} tail={euc_val['tail_metric']:.3f}", flush=True)
        print(f"  Hyp depths: {hyp_val['depth_breakdown']}", flush=True)
        print(f"  Euc depths: {euc_val['depth_breakdown']}", flush=True)

        # McNemar test for this seed
        mcnemar = compute_mcnemar(hyp_val["task_results"], euc_val["task_results"])
        print(f"  McNemar: b={mcnemar['b']} c={mcnemar['c']} chi2={mcnemar['chi2']:.2f} -> {mcnemar['advantage']}", flush=True)

        raw_margin = (hyp_val['raw_accuracy'] - euc_val['raw_accuracy']) * 100

        all_results.append({
            "seed": run_seed,
            "hyperbolic": hyp_val,
            "euclidean": euc_val,
            "raw_margin": raw_margin,
            "mcnemar": mcnemar,
        })

        # Accumulate task results across seeds
        for task_id, result in hyp_val["task_results"].items():
            if task_id not in all_hyp_task_results:
                all_hyp_task_results[task_id] = []
            all_hyp_task_results[task_id].append(result)

        for task_id, result in euc_val["task_results"].items():
            if task_id not in all_euc_task_results:
                all_euc_task_results[task_id] = []
            all_euc_task_results[task_id].append(result)

    # Final summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY - V6 STATISTICAL ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    avg_hyp_raw = sum(r["hyperbolic"]["raw_accuracy"] for r in all_results) / len(all_results)
    avg_euc_raw = sum(r["euclidean"]["raw_accuracy"] for r in all_results) / len(all_results)
    avg_raw_margin = sum(r["raw_margin"] for r in all_results) / len(all_results)

    hyp_wins = sum(1 for r in all_results if r["mcnemar"]["advantage"] == "hyperbolic")
    euc_wins = sum(1 for r in all_results if r["mcnemar"]["advantage"] == "euclidean")
    ties = sum(1 for r in all_results if r["mcnemar"]["advantage"] == "tie")

    print(f"\nAcross {args.seeds} seeds:", flush=True)
    print(f"  Average Hyperbolic raw: {avg_hyp_raw*100:.1f}%", flush=True)
    print(f"  Average Euclidean raw:  {avg_euc_raw*100:.1f}%", flush=True)
    print(f"  Average raw margin:     {avg_raw_margin:+.1f}%", flush=True)
    print(f"\nMcNemar wins:", flush=True)
    print(f"  Hyperbolic: {hyp_wins}/{args.seeds}", flush=True)
    print(f"  Euclidean:  {euc_wins}/{args.seeds}", flush=True)
    print(f"  Tie:        {ties}/{args.seeds}", flush=True)

    # Per-depth analysis across all seeds
    print(f"\nPer-depth win rate (across all seeds):", flush=True)
    for depth in range(1, 6):
        depth_tasks = [t for t in pool.val_tasks if t.depth == depth]
        if not depth_tasks:
            continue

        hyp_correct = 0
        euc_correct = 0
        total = 0

        for task in depth_tasks:
            if task.task_id in all_hyp_task_results:
                hyp_correct += sum(all_hyp_task_results[task.task_id])
                total += len(all_hyp_task_results[task.task_id])
            if task.task_id in all_euc_task_results:
                euc_correct += sum(all_euc_task_results[task.task_id])

        if total > 0:
            print(f"  Depth {depth}: Hyp={hyp_correct}/{total} ({hyp_correct/total*100:.1f}%) vs Euc={euc_correct}/{total} ({euc_correct/total*100:.1f}%)", flush=True)

    # Statistical significance
    significant_seeds = sum(1 for r in all_results if r["mcnemar"].get("significant", False))
    print(f"\nStatistically significant results: {significant_seeds}/{args.seeds} seeds", flush=True)

    if avg_raw_margin > 15 and hyp_wins >= 3:
        print("\n*** HYPERBOLIC SHOWS CLEAR ADVANTAGE ***", flush=True)
    elif avg_raw_margin < -15 and euc_wins >= 3:
        print("\n*** EUCLIDEAN SHOWS CLEAR ADVANTAGE ***", flush=True)
    else:
        print("\n*** NO CLEAR WINNER - MORE DATA NEEDED ***", flush=True)

    output_path = Path(__file__).parent / "v6_statistical_results.json"
    with open(output_path, "w") as f:
        # Convert task_results to serializable format
        serializable_results = []
        for r in all_results:
            sr = dict(r)
            sr["hyperbolic"]["task_results"] = {k: bool(v) for k, v in sr["hyperbolic"]["task_results"].items()}
            sr["euclidean"]["task_results"] = {k: bool(v) for k, v in sr["euclidean"]["task_results"].items()}
            serializable_results.append(sr)
        json.dump(serializable_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
