"""
Verifiable Evolution V4 - True Tree-Structured Tasks

Per Codex analysis for "shakingly good" results:
"Tasks with true tree geometry and depth pressure"
- Depth 6-10, branching factor 3-6
- Reward scales with depth (exponentially)
- Rare-leaf bonus (deeper = rarer = more valuable)
- Tail metrics: 5th-percentile depth accuracy

V4 Changes from V3:
1. True tree tasks: traversal, hierarchical classification, multi-hop
2. Depth-weighted fitness: deeper correct answers worth exponentially more
3. Rarity bonus: rare leaves (deep paths) give bonus
4. Numerical safeguards for hyperbolic diversity calculation
5. Curvature annealing: start low, increase as population stabilizes
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.config import GeometryConfig
from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.verification.tree_tasks import (
    TreeTask,
    TreeTaskPool,
)


@dataclass
class TreeCandidate:
    """Candidate with depth-weighted fitness."""
    latent: Tensor

    # Per-depth tracking
    depth_correct: dict = field(default_factory=dict)
    depth_total: dict = field(default_factory=dict)

    # Raw stats
    correct: int = 0
    total: int = 0

    # Weighted metrics
    depth_weighted_score: float = 0.0
    rarity_bonus: float = 0.0

    @property
    def raw_fitness(self) -> float:
        """Simple accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def weighted_fitness(self) -> float:
        """Depth-weighted + rarity bonus fitness."""
        return self.depth_weighted_score + self.rarity_bonus

    def compute_weighted_score(self, depth_weight_base: float = 2.0) -> float:
        """
        Compute depth-weighted score.

        Weight = base^depth (exponential scaling)
        Deeper correct answers worth more.
        """
        weighted_sum = 0.0
        weight_total = 0.0

        for depth, total in self.depth_total.items():
            weight = depth_weight_base ** depth
            correct = self.depth_correct.get(depth, 0)

            weighted_sum += correct * weight
            weight_total += total * weight

        self.depth_weighted_score = weighted_sum / weight_total if weight_total > 0 else 0.0
        return self.depth_weighted_score

    def get_tail_metric(self) -> float:
        """Get worst-depth accuracy (hardest tasks)."""
        # Focus on depths 5+ (the "hard" ones)
        deep_correct = sum(self.depth_correct.get(d, 0) for d in range(5, 10))
        deep_total = sum(self.depth_total.get(d, 0) for d in range(5, 10))

        return deep_correct / deep_total if deep_total > 0 else 0.0


def evaluate_candidate(
    candidate: TreeCandidate,
    tasks: list[TreeTask],
    encoder: LLMEncoder,
    hyp_module,
    geometry_config: GeometryConfig,
) -> None:
    """Evaluate candidate on tree tasks with depth tracking."""
    candidate.correct = 0
    candidate.total = len(tasks)
    candidate.depth_correct = defaultdict(int)
    candidate.depth_total = defaultdict(int)
    candidate.rarity_bonus = 0.0

    for task in tasks:
        candidate.depth_total[task.depth] += 1

        # Generate response
        response = encoder.decode(
            candidate.latent,
            query=task.prompt,
            max_new_tokens=150,  # Slightly longer for tree tasks
            temperature=0.3,
            hyperbolic=hyp_module is not None,
            curvature=geometry_config.curvature if hyp_module else 1.0,
        )

        # Verify
        is_correct = task.verifier(response, task.correct_answer)
        if is_correct:
            candidate.correct += 1
            candidate.depth_correct[task.depth] += 1
            # Rarity bonus for solving rare (deep) tasks
            candidate.rarity_bonus += (1 - task.rarity) * 0.05

    # Compute weighted score
    candidate.compute_weighted_score()


def compute_diversity_safe(
    population: list[TreeCandidate],
    hyp_module,
    geometry_config: GeometryConfig,
) -> float:
    """
    Compute diversity with numerical safeguards.

    Fixes div=0.0 bug by:
    1. Clamping latent norms before distance computation
    2. Using epsilon-safe operations
    3. Fallback to Euclidean if hyperbolic fails
    """
    n = len(population)
    if n <= 1:
        return 0.0

    total_dist = 0.0
    count = 0
    nan_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            try:
                lat_i = population[i].latent.squeeze()
                lat_j = population[j].latent.squeeze()

                if hyp_module is not None:
                    # Clamp norms to safe range before distance computation
                    norm_i = lat_i.norm()
                    norm_j = lat_j.norm()

                    # If norms are too close to boundary, use Euclidean fallback
                    if norm_i > 0.99 or norm_j > 0.99:
                        dist = torch.norm(lat_i - lat_j).item()
                    else:
                        dist = hyp_module.hyperbolic_distance(
                            lat_i, lat_j, geometry_config.curvature
                        ).item()
                else:
                    dist = torch.norm(lat_i - lat_j).item()

                # Check for valid distance
                if math.isnan(dist) or math.isinf(dist):
                    nan_count += 1
                    # Fallback to Euclidean
                    dist = torch.norm(lat_i - lat_j).item()
                    if math.isnan(dist) or math.isinf(dist):
                        continue

                total_dist += dist
                count += 1

            except Exception as e:
                continue

    return total_dist / count if count > 0 else 0.0


def mutate_safe(
    latent: Tensor,
    scale: float,
    hyp_module,
    geometry_config: GeometryConfig,
) -> Tensor:
    """Mutate with numerical safeguards."""
    noise = torch.randn_like(latent) * scale

    if hyp_module is not None:
        lat = latent.squeeze()

        # Ensure latent is safely inside ball before logmap
        norm = lat.norm()
        if norm > 0.95:
            lat = lat * (0.95 / norm)

        tangent = hyp_module.logmap0(lat, geometry_config.curvature)

        # Add noise in tangent space
        tangent = tangent + noise.squeeze()

        # Map back with safe projection
        mutated = hyp_module.expmap0(tangent, geometry_config.curvature)
        mutated = hyp_module.project_to_ball(
            mutated, geometry_config.curvature, geometry_config.max_norm
        )

        # Ensure result is valid
        if torch.isnan(mutated).any() or torch.isinf(mutated).any():
            # Fallback: just use noise in Euclidean sense, then project
            mutated = latent.squeeze() + noise.squeeze() * 0.1
            mutated = hyp_module.project_to_ball(
                mutated, geometry_config.curvature, geometry_config.max_norm
            )

        return mutated.unsqueeze(0) if mutated.dim() == 1 else mutated
    else:
        return latent + noise


def crossover_safe(
    parent_a: Tensor,
    parent_b: Tensor,
    hyp_module,
    geometry_config: GeometryConfig,
) -> Tensor:
    """Crossover with numerical safeguards."""
    t = random.random()

    if hyp_module is not None:
        a = parent_a.squeeze()
        b = parent_b.squeeze()

        # Ensure both parents are safely inside ball
        norm_a, norm_b = a.norm(), b.norm()
        if norm_a > 0.95:
            a = a * (0.95 / norm_a)
        if norm_b > 0.95:
            b = b * (0.95 / norm_b)

        try:
            child = hyp_module.hyperbolic_interpolate(
                a, b, t, geometry_config.curvature
            )

            if torch.isnan(child).any() or torch.isinf(child).any():
                # Fallback to Euclidean interpolation
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


def tournament_select(
    population: list[TreeCandidate],
    use_weighted: bool = True,
    k: int = 3,
) -> TreeCandidate:
    """Tournament selection."""
    contestants = random.sample(population, min(k, len(population)))

    def get_fitness(c):
        return c.weighted_fitness if use_weighted else c.raw_fitness

    return max(contestants, key=get_fitness)


def run_tree_evolution(
    encoder: LLMEncoder,
    pool: TreeTaskPool,
    geometry: str,
    seed_latent: Tensor,
    generations: int = 10,
    population_size: int = 8,
    tasks_per_gen: int = 15,
    elite_count: int = 2,
    mutation_scale: float = 0.1,
    curvature_annealing: bool = True,
) -> dict:
    """Run evolution with tree tasks and depth-weighted fitness."""

    # Setup geometry
    if geometry == "hyperbolic":
        from latent_reasoning.utils import hyperbolic as hyp

        # Start with lower curvature, anneal up
        initial_curvature = 0.5 if curvature_annealing else 1.0
        geometry_config = GeometryConfig(
            space="hyperbolic",
            curvature=initial_curvature,
            tangent_scale=0.35,
            max_norm=0.95,  # Slightly more conservative
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
    population = [TreeCandidate(latent=seed_latent.clone())]
    for _ in range(population_size - 1):
        mutated = mutate_safe(seed_latent, mutation_scale, hyp_module, geometry_config)
        population.append(TreeCandidate(latent=mutated))

    history = []
    rolling_fitness = []

    for gen in range(generations):
        # Curvature annealing: increase curvature over generations
        if curvature_annealing and hyp_module is not None:
            # Anneal from 0.5 to 1.5 over generations
            progress = gen / max(1, generations - 1)
            geometry_config.curvature = 0.5 + progress * 1.0

        # Sample tasks
        tasks = pool.sample_train(tasks_per_gen, seed=gen * 1000 + gen)

        # Evaluate
        for candidate in population:
            evaluate_candidate(candidate, tasks, encoder, hyp_module, geometry_config)

        # Sort by weighted fitness
        population.sort(key=lambda c: c.weighted_fitness, reverse=True)

        # Rolling fitness
        rolling_fitness.append(population[0].raw_fitness)
        if len(rolling_fitness) > 3:
            rolling_fitness.pop(0)
        roll_avg = sum(rolling_fitness) / len(rolling_fitness)

        # Metrics
        diversity = compute_diversity_safe(population, hyp_module, geometry_config)
        avg_fitness = sum(c.raw_fitness for c in population) / len(population)
        tail_metric = population[0].get_tail_metric()

        # Depth breakdown for best
        depth_breakdown = {
            d: f"{population[0].depth_correct.get(d, 0)}/{population[0].depth_total.get(d, 0)}"
            for d in sorted(population[0].depth_total.keys())
        }

        history.append({
            "generation": gen + 1,
            "raw_fitness": population[0].raw_fitness,
            "weighted_fitness": population[0].weighted_fitness,
            "roll_avg": roll_avg,
            "avg_fitness": avg_fitness,
            "diversity": diversity,
            "tail_metric": tail_metric,
            "curvature": geometry_config.curvature if hyp_module else 1.0,
            "depth_breakdown": depth_breakdown,
        })

        curv_str = f" c={geometry_config.curvature:.2f}" if hyp_module else ""
        print(
            f"[GEN {gen+1:02d}] raw={population[0].raw_fitness:.3f} "
            f"wgt={population[0].weighted_fitness:.3f} "
            f"roll={roll_avg:.3f} "
            f"tail={tail_metric:.3f} "
            f"div={diversity:.3f}{curv_str}"
        )

        # Create next generation
        next_gen = []

        # Keep elites
        for elite in population[:elite_count]:
            next_gen.append(TreeCandidate(latent=elite.latent.clone()))

        # Fill with offspring
        while len(next_gen) < population_size:
            parent_a = tournament_select(population, use_weighted=True)
            parent_b = tournament_select(population, use_weighted=True)

            child_latent = crossover_safe(parent_a.latent, parent_b.latent, hyp_module, geometry_config)

            if random.random() < 0.8:
                child_latent = mutate_safe(child_latent, mutation_scale, hyp_module, geometry_config)

            next_gen.append(TreeCandidate(latent=child_latent))

        population = next_gen

    return {
        "final_raw": population[0].raw_fitness,
        "final_weighted": population[0].weighted_fitness,
        "final_roll_avg": roll_avg,
        "final_tail": population[0].get_tail_metric(),
        "history": history,
        "best_latent": population[0].latent,
    }


def evaluate_on_validation(
    latent: Tensor,
    val_tasks: list[TreeTask],
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
            max_norm=0.95,
        )
        hyp_module = hyp
    else:
        geometry_config = GeometryConfig(space="euclidean")
        hyp_module = None

    candidate = TreeCandidate(latent=latent)
    evaluate_candidate(candidate, val_tasks, encoder, hyp_module, geometry_config)

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
    }


def main():
    parser = argparse.ArgumentParser(description="V4: True Tree Tasks + Depth Fitness")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--tasks-per-gen", type=int, default=15)
    parser.add_argument("--pool-size", type=int, default=200)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-annealing", action="store_true", help="Disable curvature annealing")
    args = parser.parse_args()

    print("=" * 70)
    print("VERIFIABLE EVOLUTION V4 - TRUE TREE TASKS + DEPTH FITNESS")
    print("Tree traversal | Hierarchical classification | Multi-hop | Depth rewards")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Tasks/gen: {args.tasks_per_gen}")
    print(f"Pool size: {args.pool_size}")
    print(f"Runs: {args.runs}")
    print(f"Curvature annealing: {not args.no_annealing}")
    print("=" * 70)

    # Create tree task pool
    print("\nCreating tree task pool...")
    pool = TreeTaskPool(pool_size=args.pool_size, seed=args.seed)
    print(f"Pool stats: {pool.stats()}")

    # Load model
    print("\nLoading model...")
    encoder = LLMEncoder(model_name=args.model, quantization="4bit")

    # Prompts focused on tree/hierarchical reasoning
    prompts = [
        "You follow hierarchical paths step by step, tracking each level carefully.",
        "You solve tree-structured problems by working from root to leaves systematically.",
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

            # Encode seed
            seed_latent = encoder.encode(prompt)

            # Run HYPERBOLIC
            print(f"\n[HYPERBOLIC] Running evolution...")
            hyp_result = run_tree_evolution(
                encoder=encoder,
                pool=pool,
                geometry="hyperbolic",
                seed_latent=seed_latent.clone(),
                generations=args.generations,
                population_size=args.population,
                tasks_per_gen=args.tasks_per_gen,
                curvature_annealing=not args.no_annealing,
            )

            # Reset for paired comparison
            random.seed(run_seed)
            torch.manual_seed(run_seed)

            # Run EUCLIDEAN
            print(f"\n[EUCLIDEAN] Running evolution...")
            euc_result = run_tree_evolution(
                encoder=encoder,
                pool=pool,
                geometry="euclidean",
                seed_latent=seed_latent.clone(),
                generations=args.generations,
                population_size=args.population,
                tasks_per_gen=args.tasks_per_gen,
            )

            # Validation
            print(f"\n[VALIDATION] Evaluating on held-out set...")
            val_tasks = pool.get_validation()

            hyp_val = evaluate_on_validation(
                hyp_result["best_latent"], val_tasks, encoder, "hyperbolic"
            )
            euc_val = evaluate_on_validation(
                euc_result["best_latent"], val_tasks, encoder, "euclidean"
            )

            print(f"  Hyperbolic: raw={hyp_val['raw_accuracy']*100:.1f}% wgt={hyp_val['weighted_accuracy']:.3f} tail={hyp_val['tail_metric']:.3f}")
            print(f"  Euclidean:  raw={euc_val['raw_accuracy']*100:.1f}% wgt={euc_val['weighted_accuracy']:.3f} tail={euc_val['tail_metric']:.3f}")
            print(f"  Hyp depths: {hyp_val['depth_breakdown']}")
            print(f"  Euc depths: {euc_val['depth_breakdown']}")

            # Margins
            raw_margin = (hyp_val['raw_accuracy'] - euc_val['raw_accuracy']) * 100
            wgt_margin = (hyp_val['weighted_accuracy'] - euc_val['weighted_accuracy']) * 100
            tail_margin = (hyp_val['tail_metric'] - euc_val['tail_metric']) * 100

            if wgt_margin > 3:
                winner = "HYPERBOLIC"
            elif wgt_margin < -3:
                winner = "EUCLIDEAN"
            else:
                winner = "TIE"

            print(f"\n[RESULT] {winner}")
            print(f"  Raw margin: {raw_margin:+.1f}%")
            print(f"  Weighted margin: {wgt_margin:+.1f}%")
            print(f"  Tail margin: {tail_margin:+.1f}%")

            results.append({
                "run": run_idx + 1,
                "prompt_idx": prompt_idx,
                "prompt": prompt[:60],
                "hyperbolic": {
                    "raw_accuracy": hyp_val['raw_accuracy'],
                    "weighted_accuracy": hyp_val['weighted_accuracy'],
                    "tail_metric": hyp_val['tail_metric'],
                    "depth_breakdown": hyp_val['depth_breakdown'],
                    "history": hyp_result['history'],
                },
                "euclidean": {
                    "raw_accuracy": euc_val['raw_accuracy'],
                    "weighted_accuracy": euc_val['weighted_accuracy'],
                    "tail_metric": euc_val['tail_metric'],
                    "depth_breakdown": euc_val['depth_breakdown'],
                    "history": euc_result['history'],
                },
                "winner": winner,
                "raw_margin": raw_margin,
                "weighted_margin": wgt_margin,
                "tail_margin": tail_margin,
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - V4 TREE TASKS")
    print("=" * 70)

    hyp_wins = sum(1 for r in results if r['winner'] == "HYPERBOLIC")
    euc_wins = sum(1 for r in results if r['winner'] == "EUCLIDEAN")
    ties = sum(1 for r in results if r['winner'] == "TIE")

    avg_raw_margin = sum(r['raw_margin'] for r in results) / len(results)
    avg_wgt_margin = sum(r['weighted_margin'] for r in results) / len(results)
    avg_tail_margin = sum(r['tail_margin'] for r in results) / len(results)

    print(f"Hyperbolic wins: {hyp_wins}")
    print(f"Euclidean wins: {euc_wins}")
    print(f"Ties: {ties}")
    print(f"Average raw margin: {avg_raw_margin:+.1f}%")
    print(f"Average weighted margin: {avg_wgt_margin:+.1f}%")
    print(f"Average tail margin: {avg_tail_margin:+.1f}%")

    # Save results
    output_path = Path(__file__).parent / "v4_tree_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
