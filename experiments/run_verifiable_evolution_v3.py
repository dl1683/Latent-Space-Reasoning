"""
Verifiable Evolution V3 - Hierarchical Tasks + Coverage-Based Fitness

Key insight from Codex:
"Hyperbolic should win when the task's useful variation is hierarchical or tree-like,
 when the fitness landscape has many deep, branching basins, and when evaluation
 rewards coverage of rare niches."

V3 Changes:
1. Use ONLY hierarchical tasks (nested arithmetic, multi-hop reasoning)
2. Coverage-based fitness: reward solving DIFFERENT categories/difficulties
3. Tail metrics: track worst-category accuracy (reveals diversity benefits)
4. Per-category breakdown in results

Why this should show hyperbolic advantage:
- Hierarchical tasks have tree-like solution structure
- Coverage fitness rewards exploration of the space
- Hyperbolic volume growth means more "room" in periphery
- Fitness sharing + coverage = double incentive for diversity
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.config import GeometryConfig
from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.verification.verifiable_tasks import (
    VerifiableTask,
    NestedArithmeticGenerator,
    MultiHopReasoningGenerator,
)


@dataclass
class HierarchicalTaskPool:
    """Task pool with ONLY hierarchical tasks."""

    nested_arithmetic: list[VerifiableTask] = field(default_factory=list)
    multi_hop: list[VerifiableTask] = field(default_factory=list)
    train_tasks: list[VerifiableTask] = field(default_factory=list)
    val_tasks: list[VerifiableTask] = field(default_factory=list)

    @classmethod
    def create(cls, pool_size: int = 200, val_ratio: float = 0.2, seed: int = 42):
        """Create hierarchical task pool."""
        random.seed(seed)

        pool = cls()

        # Generate hierarchical tasks only
        nested_gen = NestedArithmeticGenerator()
        multi_hop_gen = MultiHopReasoningGenerator()

        pool.nested_arithmetic = nested_gen.generate(pool_size // 2)
        pool.multi_hop = multi_hop_gen.generate(pool_size // 2)

        # Combine and split
        all_tasks = pool.nested_arithmetic + pool.multi_hop
        random.shuffle(all_tasks)

        val_size = int(len(all_tasks) * val_ratio)
        pool.val_tasks = all_tasks[:val_size]
        pool.train_tasks = all_tasks[val_size:]

        return pool

    def sample_train(self, n: int, seed: int | None = None) -> list[VerifiableTask]:
        """Sample n tasks from training pool."""
        if seed is not None:
            random.seed(seed)
        return random.sample(self.train_tasks, min(n, len(self.train_tasks)))

    def get_validation(self) -> list[VerifiableTask]:
        """Get full validation set."""
        return self.val_tasks

    def stats(self) -> dict:
        """Get pool statistics."""
        train_cats = defaultdict(int)
        val_cats = defaultdict(int)
        train_diff = defaultdict(int)
        val_diff = defaultdict(int)

        for t in self.train_tasks:
            train_cats[t.category] += 1
            train_diff[t.difficulty] += 1
        for t in self.val_tasks:
            val_cats[t.category] += 1
            val_diff[t.difficulty] += 1

        return {
            "train_size": len(self.train_tasks),
            "val_size": len(self.val_tasks),
            "train_categories": dict(train_cats),
            "val_categories": dict(val_cats),
            "train_difficulty": dict(train_diff),
            "val_difficulty": dict(val_diff),
        }


@dataclass
class CoverageCandidate:
    """Candidate with coverage-based fitness tracking."""
    latent: Tensor

    # Per-category accuracy
    category_correct: dict = field(default_factory=dict)
    category_total: dict = field(default_factory=dict)

    # Per-difficulty accuracy
    difficulty_correct: dict = field(default_factory=dict)
    difficulty_total: dict = field(default_factory=dict)

    # Raw correctness
    correct: int = 0
    total: int = 0

    # Coverage bonus
    coverage_bonus: float = 0.0

    @property
    def raw_fitness(self) -> float:
        """Raw accuracy without coverage bonus."""
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def coverage_fitness(self) -> float:
        """Fitness with coverage bonus."""
        base = self.raw_fitness
        # Add bonus for covering more categories/difficulties
        return base + self.coverage_bonus

    def compute_coverage_bonus(self, alpha: float = 0.1) -> float:
        """
        Compute coverage bonus based on category/difficulty spread.

        Bonus for:
        - Solving tasks in MORE categories (breadth)
        - Solving tasks at higher difficulties (depth)
        """
        # Category coverage: bonus for each category with > 50% accuracy
        cat_coverage = 0
        for cat, total in self.category_total.items():
            if total > 0:
                acc = self.category_correct.get(cat, 0) / total
                if acc >= 0.5:
                    cat_coverage += 1

        # Difficulty bonus: weighted by difficulty level
        diff_weights = {"easy": 0.5, "medium": 1.0, "hard": 1.5}
        diff_bonus = 0
        for diff, total in self.difficulty_total.items():
            if total > 0:
                acc = self.difficulty_correct.get(diff, 0) / total
                diff_bonus += acc * diff_weights.get(diff, 1.0)

        self.coverage_bonus = alpha * (cat_coverage + diff_bonus)
        return self.coverage_bonus

    def get_tail_metric(self) -> float:
        """Get worst-category accuracy (tail metric)."""
        worst = 1.0
        for cat, total in self.category_total.items():
            if total > 0:
                acc = self.category_correct.get(cat, 0) / total
                worst = min(worst, acc)
        return worst if self.category_total else 0.0


def evaluate_candidate(
    candidate: CoverageCandidate,
    tasks: list[VerifiableTask],
    encoder: LLMEncoder,
    hyp_module,
    geometry_config: GeometryConfig,
) -> None:
    """Evaluate candidate with detailed category/difficulty tracking."""
    candidate.correct = 0
    candidate.total = len(tasks)
    candidate.category_correct = defaultdict(int)
    candidate.category_total = defaultdict(int)
    candidate.difficulty_correct = defaultdict(int)
    candidate.difficulty_total = defaultdict(int)

    for task in tasks:
        # Track totals
        candidate.category_total[task.category] += 1
        candidate.difficulty_total[task.difficulty] += 1

        # Generate response
        response = encoder.decode(
            candidate.latent,
            query=task.prompt,
            max_new_tokens=100,
            temperature=0.3,
            hyperbolic=hyp_module is not None,
            curvature=geometry_config.curvature if hyp_module else 1.0,
        )

        # Verify
        is_correct = task.verifier(response, task.correct_answer)
        if is_correct:
            candidate.correct += 1
            candidate.category_correct[task.category] += 1
            candidate.difficulty_correct[task.difficulty] += 1

    # Compute coverage bonus
    candidate.compute_coverage_bonus()


def compute_diversity(
    population: list[CoverageCandidate],
    hyp_module,
    geometry_config: GeometryConfig,
) -> float:
    """Compute average pairwise distance."""
    n = len(population)
    if n <= 1:
        return 0.0

    total_dist = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            try:
                if hyp_module is not None:
                    dist = hyp_module.hyperbolic_distance(
                        population[i].latent.squeeze(),
                        population[j].latent.squeeze(),
                        geometry_config.curvature,
                    ).item()
                else:
                    dist = torch.norm(
                        population[i].latent - population[j].latent
                    ).item()

                if not (torch.isnan(torch.tensor(dist)) or torch.isinf(torch.tensor(dist))):
                    total_dist += dist
                    count += 1
            except Exception:
                continue

    return total_dist / count if count > 0 else 0.0


def mutate(
    latent: Tensor,
    scale: float,
    hyp_module,
    geometry_config: GeometryConfig,
) -> Tensor:
    """Mutate latent in appropriate geometry."""
    noise = torch.randn_like(latent) * scale

    if hyp_module is not None:
        # Hyperbolic mutation
        tangent = hyp_module.logmap0(latent.squeeze(), geometry_config.curvature)
        tangent = tangent + noise.squeeze()
        mutated = hyp_module.expmap0(tangent, geometry_config.curvature)
        mutated = hyp_module.project_to_ball(
            mutated, geometry_config.curvature, geometry_config.max_norm
        )
        return mutated.unsqueeze(0) if mutated.dim() == 1 else mutated
    else:
        return latent + noise


def crossover(
    parent_a: Tensor,
    parent_b: Tensor,
    hyp_module,
    geometry_config: GeometryConfig,
) -> Tensor:
    """Crossover two parents."""
    t = random.random()

    if hyp_module is not None:
        child = hyp_module.hyperbolic_interpolate(
            parent_a.squeeze(),
            parent_b.squeeze(),
            t,
            geometry_config.curvature,
        )
        return child.unsqueeze(0) if child.dim() == 1 else child
    else:
        return t * parent_a + (1 - t) * parent_b


def tournament_select(
    population: list[CoverageCandidate],
    use_coverage: bool = True,
    k: int = 3,
) -> CoverageCandidate:
    """Tournament selection with optional coverage fitness."""
    contestants = random.sample(population, min(k, len(population)))

    def get_fitness(c):
        if use_coverage:
            return c.coverage_fitness
        return c.raw_fitness

    return max(contestants, key=get_fitness)


def run_hierarchical_evolution(
    encoder: LLMEncoder,
    pool: HierarchicalTaskPool,
    geometry: str,
    seed_latent: Tensor,
    generations: int = 10,
    population_size: int = 6,
    tasks_per_gen: int = 12,
    elite_count: int = 2,
    mutation_scale: float = 0.1,
    use_coverage_fitness: bool = True,
) -> dict:
    """Run evolution with hierarchical tasks and coverage fitness."""

    # Setup geometry
    if geometry == "hyperbolic":
        from latent_reasoning.utils import hyperbolic as hyp
        geometry_config = GeometryConfig(
            space="hyperbolic",
            curvature=1.0,
            tangent_scale=0.35,
            max_norm=0.98,
        )
        # Map seed to hyperbolic
        seed_latent = hyp.expmap0(
            seed_latent.squeeze() * geometry_config.tangent_scale,
            geometry_config.curvature,
        ).unsqueeze(0)
        hyp_module = hyp
    else:
        geometry_config = GeometryConfig(space="euclidean")
        hyp_module = None

    # Initialize population
    population = [CoverageCandidate(latent=seed_latent.clone())]
    for _ in range(population_size - 1):
        mutated = mutate(seed_latent, mutation_scale, hyp_module, geometry_config)
        population.append(CoverageCandidate(latent=mutated))

    history = []
    rolling_fitness = []

    for gen in range(generations):
        # Sample tasks for this generation
        tasks = pool.sample_train(tasks_per_gen, seed=gen * 1000 + gen)

        # Evaluate all candidates
        for candidate in population:
            evaluate_candidate(candidate, tasks, encoder, hyp_module, geometry_config)

        # Sort by coverage fitness
        population.sort(key=lambda c: c.coverage_fitness, reverse=True)

        # Track rolling fitness (3-gen average)
        rolling_fitness.append(population[0].raw_fitness)
        if len(rolling_fitness) > 3:
            rolling_fitness.pop(0)
        roll_avg = sum(rolling_fitness) / len(rolling_fitness)

        # Compute metrics
        diversity = compute_diversity(population, hyp_module, geometry_config)
        avg_fitness = sum(c.raw_fitness for c in population) / len(population)
        tail_metric = population[0].get_tail_metric()

        # Category breakdown for best
        cat_breakdown = {
            cat: f"{population[0].category_correct.get(cat, 0)}/{population[0].category_total.get(cat, 0)}"
            for cat in population[0].category_total.keys()
        }

        history.append({
            "generation": gen + 1,
            "raw_fitness": population[0].raw_fitness,
            "coverage_fitness": population[0].coverage_fitness,
            "roll_avg": roll_avg,
            "avg_fitness": avg_fitness,
            "diversity": diversity,
            "tail_metric": tail_metric,
            "category_breakdown": cat_breakdown,
        })

        print(
            f"[GEN {gen+1:02d}] raw={population[0].raw_fitness:.3f} "
            f"cov={population[0].coverage_fitness:.3f} "
            f"roll={roll_avg:.3f} "
            f"tail={tail_metric:.3f} "
            f"div={diversity:.3f}"
        )

        # Create next generation
        next_gen = []

        # Keep elites
        for elite in population[:elite_count]:
            next_gen.append(CoverageCandidate(latent=elite.latent.clone()))

        # Fill with offspring
        while len(next_gen) < population_size:
            parent_a = tournament_select(population, use_coverage_fitness)
            parent_b = tournament_select(population, use_coverage_fitness)

            child_latent = crossover(parent_a.latent, parent_b.latent, hyp_module, geometry_config)

            if random.random() < 0.8:
                child_latent = mutate(child_latent, mutation_scale, hyp_module, geometry_config)

            next_gen.append(CoverageCandidate(latent=child_latent))

        population = next_gen

    return {
        "final_fitness": population[0].raw_fitness,
        "final_coverage": population[0].coverage_fitness,
        "final_roll_avg": roll_avg,
        "history": history,
        "best_latent": population[0].latent,
    }


def evaluate_on_validation(
    latent: Tensor,
    val_tasks: list[VerifiableTask],
    encoder: LLMEncoder,
    geometry: str,
) -> dict:
    """Evaluate on held-out validation set."""

    if geometry == "hyperbolic":
        from latent_reasoning.utils import hyperbolic as hyp
        geometry_config = GeometryConfig(
            space="hyperbolic",
            curvature=1.0,
            tangent_scale=0.35,
            max_norm=0.98,
        )
        hyp_module = hyp
    else:
        geometry_config = GeometryConfig(space="euclidean")
        hyp_module = None

    candidate = CoverageCandidate(latent=latent)
    evaluate_candidate(candidate, val_tasks, encoder, hyp_module, geometry_config)

    return {
        "accuracy": candidate.raw_fitness,
        "correct": candidate.correct,
        "total": candidate.total,
        "tail_metric": candidate.get_tail_metric(),
        "category_breakdown": {
            cat: f"{candidate.category_correct.get(cat, 0)}/{candidate.category_total.get(cat, 0)}"
            for cat in candidate.category_total.keys()
        },
        "difficulty_breakdown": {
            diff: f"{candidate.difficulty_correct.get(diff, 0)}/{candidate.difficulty_total.get(diff, 0)}"
            for diff in candidate.difficulty_total.keys()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="V3: Hierarchical Tasks + Coverage Fitness")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--tasks-per-gen", type=int, default=12)
    parser.add_argument("--pool-size", type=int, default=150)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("VERIFIABLE EVOLUTION V3 - HIERARCHICAL TASKS + COVERAGE FITNESS")
    print("Nested arithmetic | Multi-hop reasoning | Coverage bonus | Tail metrics")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Tasks/gen: {args.tasks_per_gen}")
    print(f"Pool size: {args.pool_size}")
    print(f"Runs: {args.runs}")
    print("=" * 70)

    # Create hierarchical task pool
    print("\nCreating hierarchical task pool...")
    pool = HierarchicalTaskPool.create(pool_size=args.pool_size, seed=args.seed)
    print(f"Pool stats: {pool.stats()}")

    # Load model
    print("\nLoading model...")
    encoder = LLMEncoder(model_name=args.model, quantize=True)

    # Prompts focused on hierarchical reasoning
    prompts = [
        "You solve nested expressions step by step, working from innermost parentheses outward.",
        "You follow chains of operations carefully, tracking each step's result.",
    ]

    results = []

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx * 1000
        random.seed(run_seed)
        torch.manual_seed(run_seed)

        print(f"\n{'#' * 70}")
        print(f"# RUN {run_idx + 1}/{args.runs} (seed={run_seed})")
        print("#" * 70)

        for prompt_idx, prompt in enumerate(prompts):
            print(f"\nPrompt: {prompt[:60]}...")

            # Encode seed latent
            seed_latent = encoder.encode(prompt)

            # Run HYPERBOLIC evolution
            print(f"\n[HYPERBOLIC] Running evolution...")
            hyp_result = run_hierarchical_evolution(
                encoder=encoder,
                pool=pool,
                geometry="hyperbolic",
                seed_latent=seed_latent.clone(),
                generations=args.generations,
                population_size=args.population,
                tasks_per_gen=args.tasks_per_gen,
            )

            # Reset random state for paired comparison
            random.seed(run_seed)
            torch.manual_seed(run_seed)

            # Run EUCLIDEAN evolution
            print(f"\n[EUCLIDEAN] Running evolution...")
            euc_result = run_hierarchical_evolution(
                encoder=encoder,
                pool=pool,
                geometry="euclidean",
                seed_latent=seed_latent.clone(),
                generations=args.generations,
                population_size=args.population,
                tasks_per_gen=args.tasks_per_gen,
            )

            # Validate on held-out set
            print(f"\n[VALIDATION] Evaluating on held-out set...")
            val_tasks = pool.get_validation()

            hyp_val = evaluate_on_validation(
                hyp_result["best_latent"], val_tasks, encoder, "hyperbolic"
            )
            euc_val = evaluate_on_validation(
                euc_result["best_latent"], val_tasks, encoder, "euclidean"
            )

            print(f"  Hyperbolic val: {hyp_val['accuracy']*100:.1f}% ({hyp_val['correct']}/{hyp_val['total']}) tail={hyp_val['tail_metric']:.3f}")
            print(f"  Euclidean val:  {euc_val['accuracy']*100:.1f}% ({euc_val['correct']}/{euc_val['total']}) tail={euc_val['tail_metric']:.3f}")
            print(f"  Hyp categories: {hyp_val['category_breakdown']}")
            print(f"  Euc categories: {euc_val['category_breakdown']}")

            margin = (hyp_val['accuracy'] - euc_val['accuracy']) * 100
            tail_margin = (hyp_val['tail_metric'] - euc_val['tail_metric']) * 100

            if margin > 2:
                winner = "HYPERBOLIC"
            elif margin < -2:
                winner = "EUCLIDEAN"
            else:
                winner = "TIE"

            print(f"\n[RESULT] {winner} (accuracy margin: {margin:+.1f}%, tail margin: {tail_margin:+.1f}%)")

            results.append({
                "run": run_idx + 1,
                "prompt_idx": prompt_idx,
                "prompt": prompt[:60],
                "hyperbolic": {
                    "val_accuracy": hyp_val['accuracy'],
                    "val_tail": hyp_val['tail_metric'],
                    "category_breakdown": hyp_val['category_breakdown'],
                    "difficulty_breakdown": hyp_val['difficulty_breakdown'],
                    "history": hyp_result['history'],
                },
                "euclidean": {
                    "val_accuracy": euc_val['accuracy'],
                    "val_tail": euc_val['tail_metric'],
                    "category_breakdown": euc_val['category_breakdown'],
                    "difficulty_breakdown": euc_val['difficulty_breakdown'],
                    "history": euc_result['history'],
                },
                "winner": winner,
                "margin": margin,
                "tail_margin": tail_margin,
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    hyp_wins = sum(1 for r in results if r['winner'] == "HYPERBOLIC")
    euc_wins = sum(1 for r in results if r['winner'] == "EUCLIDEAN")
    ties = sum(1 for r in results if r['winner'] == "TIE")

    avg_margin = sum(r['margin'] for r in results) / len(results)
    avg_tail_margin = sum(r['tail_margin'] for r in results) / len(results)

    print(f"Hyperbolic wins: {hyp_wins}")
    print(f"Euclidean wins: {euc_wins}")
    print(f"Ties: {ties}")
    print(f"Average accuracy margin: {avg_margin:+.1f}%")
    print(f"Average tail margin: {avg_tail_margin:+.1f}%")

    # Save results
    output_path = Path(__file__).parent / "v3_hierarchical_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
