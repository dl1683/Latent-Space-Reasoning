"""
Verifiable Evolution Experiment V2 - Improved Design

Implements Codex's recommendations for valid geometry comparison:
1. Fixed task pool with train/validation split
2. Same seeds for paired comparison (common random numbers)
3. Rolling fitness average for selection
4. Final validation evaluation for comparison
5. Stratified sampling by category

Key insight: Separate selection (stochastic) from measurement (stable).
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import deque

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.config import GeometryConfig
from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.verification.verifiable_tasks import (
    VerifiableTask,
    FixedTaskPool,
    create_fixed_pool,
)


def evaluate_on_tasks(
    encoder: LLMEncoder,
    latent: torch.Tensor,
    tasks: list[VerifiableTask],
    pool: FixedTaskPool,
    hyperbolic: bool = False,
    curvature: float = 1.0,
) -> tuple[int, int, list[str]]:
    """Evaluate a latent on a set of tasks."""
    responses = []
    correct = 0

    for task in tasks:
        response = encoder.decode(
            latent,
            query=task.prompt,
            max_new_tokens=100,
            temperature=0.3,
            hyperbolic=hyperbolic,
            curvature=curvature,
        )
        responses.append(response)

        if pool.evaluate(task, response):
            correct += 1

    return correct, len(tasks), responses


def run_evolution_with_pool(
    encoder: LLMEncoder,
    seed_latent: torch.Tensor,
    pool: FixedTaskPool,
    geometry_config: GeometryConfig,
    max_generations: int = 10,
    population_size: int = 6,
    tasks_per_gen: int = 16,
    elite_count: int = 2,
    mutation_scale: float = 0.1,
    rolling_window: int = 3,
    task_seed: int = 0,
) -> dict:
    """
    Run evolution with fixed pool and rolling fitness.

    Returns dict with best latent, fitness history, and diversity metrics.
    """
    is_hyperbolic = geometry_config.space == "hyperbolic"

    # Load hyperbolic utils if needed
    hyp = None
    if is_hyperbolic:
        from latent_reasoning.utils import hyperbolic as hyp_module
        hyp = hyp_module

    # Initialize population
    population = [seed_latent.clone()]
    for _ in range(population_size - 1):
        mutated = mutate(seed_latent, mutation_scale, hyp, geometry_config)
        population.append(mutated)

    # Track fitness history for rolling average
    fitness_histories = [deque(maxlen=rolling_window) for _ in range(population_size)]

    history = []
    best_fitness = 0.0
    best_latent = seed_latent.clone()

    for gen in range(max_generations):
        # Sample tasks for this generation (seeded for reproducibility)
        gen_seed = task_seed + gen * 1000
        tasks = pool.sample_train(tasks_per_gen, seed=gen_seed)

        # Evaluate all candidates
        gen_fitnesses = []
        for i, latent in enumerate(population):
            correct, total, _ = evaluate_on_tasks(
                encoder, latent, tasks, pool,
                hyperbolic=is_hyperbolic,
                curvature=geometry_config.curvature if is_hyperbolic else 1.0,
            )
            fitness = correct / total
            gen_fitnesses.append(fitness)
            fitness_histories[i].append(fitness)

        # Compute rolling average fitness for selection
        rolling_fitnesses = [
            sum(h) / len(h) if h else 0.0 for h in fitness_histories
        ]

        # Compute diversity
        diversity = compute_diversity(population, hyp, geometry_config)

        # Track best (use rolling fitness)
        best_idx = max(range(len(rolling_fitnesses)), key=lambda i: rolling_fitnesses[i])
        if rolling_fitnesses[best_idx] > best_fitness:
            best_fitness = rolling_fitnesses[best_idx]
            best_latent = population[best_idx].clone()

        # Log
        avg_fitness = sum(gen_fitnesses) / len(gen_fitnesses)
        avg_rolling = sum(rolling_fitnesses) / len(rolling_fitnesses)
        history.append({
            "generation": gen,
            "best_raw": max(gen_fitnesses),
            "best_rolling": rolling_fitnesses[best_idx],
            "avg_raw": avg_fitness,
            "avg_rolling": avg_rolling,
            "diversity": diversity,
        })

        print(
            f"[GEN {gen+1:02d}] raw={max(gen_fitnesses):.3f} "
            f"roll={rolling_fitnesses[best_idx]:.3f} "
            f"avg={avg_fitness:.3f} div={diversity:.3f}",
            flush=True,
        )

        # Create next generation (except last)
        if gen < max_generations - 1:
            population, fitness_histories = create_next_generation(
                population,
                rolling_fitnesses,
                fitness_histories,
                mutation_scale,
                elite_count,
                hyp,
                geometry_config,
            )

    return {
        "best_latent": best_latent,
        "best_rolling_fitness": best_fitness,
        "history": history,
        "final_population": population,
    }


def mutate(latent: torch.Tensor, scale: float, hyp, geometry_config: GeometryConfig) -> torch.Tensor:
    """Mutate a latent vector."""
    noise = torch.randn_like(latent) * scale

    if hyp is not None:
        tangent = hyp.logmap0(latent.squeeze(), geometry_config.curvature)
        tangent = tangent + noise.squeeze()
        mutated = hyp.expmap0(tangent, geometry_config.curvature)
        mutated = hyp.project_to_ball(mutated, geometry_config.curvature, geometry_config.max_norm)
        return mutated.unsqueeze(0) if mutated.dim() == 1 else mutated
    else:
        return latent + noise


def crossover(parent_a: torch.Tensor, parent_b: torch.Tensor, hyp, geometry_config: GeometryConfig) -> torch.Tensor:
    """Crossover two parents."""
    t = random.random()

    if hyp is not None:
        child = hyp.hyperbolic_interpolate(
            parent_a.squeeze(),
            parent_b.squeeze(),
            t,
            geometry_config.curvature,
        )
        return child.unsqueeze(0) if child.dim() == 1 else child
    else:
        return t * parent_a + (1 - t) * parent_b


def compute_diversity(population: list[torch.Tensor], hyp, geometry_config: GeometryConfig) -> float:
    """Compute average pairwise distance."""
    n = len(population)
    if n <= 1:
        return 0.0

    total_dist = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            try:
                if hyp is not None:
                    dist = hyp.hyperbolic_distance(
                        population[i].squeeze(),
                        population[j].squeeze(),
                        geometry_config.curvature,
                    ).item()
                else:
                    dist = torch.norm(population[i] - population[j]).item()

                if not (torch.isnan(torch.tensor(dist)) or torch.isinf(torch.tensor(dist))):
                    total_dist += dist
                    count += 1
            except Exception:
                continue

    return total_dist / count if count > 0 else 0.0


def create_next_generation(
    population: list[torch.Tensor],
    fitnesses: list[float],
    fitness_histories: list[deque],
    mutation_scale: float,
    elite_count: int,
    hyp,
    geometry_config: GeometryConfig,
) -> tuple[list[torch.Tensor], list[deque]]:
    """Create next generation through selection, crossover, mutation."""
    # Sort by fitness
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)

    next_gen = []
    next_histories = []

    # Keep elites
    for i in range(elite_count):
        idx = sorted_indices[i]
        next_gen.append(population[idx].clone())
        next_histories.append(fitness_histories[idx].copy())

    # Fill rest with offspring
    while len(next_gen) < len(population):
        # Tournament selection
        parent_a = tournament_select(population, fitnesses)
        parent_b = tournament_select(population, fitnesses)

        # Crossover
        child = crossover(parent_a, parent_b, hyp, geometry_config)

        # Mutation
        if random.random() < 0.8:
            child = mutate(child, mutation_scale, hyp, geometry_config)

        next_gen.append(child)
        next_histories.append(deque(maxlen=fitness_histories[0].maxlen))

    return next_gen, next_histories


def tournament_select(population: list[torch.Tensor], fitnesses: list[float], k: int = 3) -> torch.Tensor:
    """Tournament selection."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


def run_paired_comparison(
    encoder: LLMEncoder,
    prompt: str,
    pool: FixedTaskPool,
    max_generations: int = 10,
    population_size: int = 6,
    tasks_per_gen: int = 16,
    run_seed: int = 42,
) -> dict:
    """
    Run paired comparison of hyperbolic vs euclidean.

    Uses same seeds for fair comparison (common random numbers).
    """
    # Encode seed latent
    seed_latent = encoder.encode(prompt)

    # Geometry configs
    hyp_config = GeometryConfig(
        space="hyperbolic",
        curvature=1.0,
        tangent_scale=0.35,
        max_norm=0.98,
    )
    euc_config = GeometryConfig(space="euclidean")

    # Map seed to hyperbolic for hyperbolic runs
    from latent_reasoning.utils import hyperbolic as hyp
    hyp_seed = hyp.expmap0(
        seed_latent.squeeze() * hyp_config.tangent_scale,
        hyp_config.curvature,
    ).unsqueeze(0)

    # Set same seeds for paired comparison
    random.seed(run_seed)
    torch.manual_seed(run_seed)

    print(f"\n[HYPERBOLIC] Running evolution...", flush=True)
    hyp_result = run_evolution_with_pool(
        encoder=encoder,
        seed_latent=hyp_seed,
        pool=pool,
        geometry_config=hyp_config,
        max_generations=max_generations,
        population_size=population_size,
        tasks_per_gen=tasks_per_gen,
        mutation_scale=0.15,
        task_seed=run_seed,
    )

    # Reset seeds for euclidean
    random.seed(run_seed)
    torch.manual_seed(run_seed)

    print(f"\n[EUCLIDEAN] Running evolution...", flush=True)
    euc_result = run_evolution_with_pool(
        encoder=encoder,
        seed_latent=seed_latent,
        pool=pool,
        geometry_config=euc_config,
        max_generations=max_generations,
        population_size=population_size,
        tasks_per_gen=tasks_per_gen,
        mutation_scale=0.1,
        task_seed=run_seed,
    )

    # Final validation evaluation
    print(f"\n[VALIDATION] Evaluating on held-out set...", flush=True)
    val_tasks = pool.get_validation()

    hyp_val_correct, hyp_val_total, _ = evaluate_on_tasks(
        encoder, hyp_result["best_latent"], val_tasks, pool,
        hyperbolic=True, curvature=hyp_config.curvature,
    )
    hyp_val_acc = hyp_val_correct / hyp_val_total

    euc_val_correct, euc_val_total, _ = evaluate_on_tasks(
        encoder, euc_result["best_latent"], val_tasks, pool,
        hyperbolic=False, curvature=1.0,
    )
    euc_val_acc = euc_val_correct / euc_val_total

    print(f"  Hyperbolic val: {hyp_val_acc:.1%} ({hyp_val_correct}/{hyp_val_total})", flush=True)
    print(f"  Euclidean val:  {euc_val_acc:.1%} ({euc_val_correct}/{euc_val_total})", flush=True)

    return {
        "hyperbolic": {
            "evolution_fitness": hyp_result["best_rolling_fitness"],
            "validation_accuracy": hyp_val_acc,
            "validation_correct": hyp_val_correct,
            "validation_total": hyp_val_total,
            "history": hyp_result["history"],
            "final_diversity": hyp_result["history"][-1]["diversity"] if hyp_result["history"] else 0.0,
        },
        "euclidean": {
            "evolution_fitness": euc_result["best_rolling_fitness"],
            "validation_accuracy": euc_val_acc,
            "validation_correct": euc_val_correct,
            "validation_total": euc_val_total,
            "history": euc_result["history"],
            "final_diversity": euc_result["history"][-1]["diversity"] if euc_result["history"] else 0.0,
        },
        "winner": "hyperbolic" if hyp_val_acc > euc_val_acc else ("euclidean" if euc_val_acc > hyp_val_acc else "tie"),
        "margin": hyp_val_acc - euc_val_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Run verifiable evolution V2 - improved design")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Model to use")
    parser.add_argument("--generations", type=int, default=10, help="Max generations")
    parser.add_argument("--population", type=int, default=6, help="Population size")
    parser.add_argument("--tasks", type=int, default=16, help="Tasks per generation")
    parser.add_argument("--pool-size", type=int, default=500, help="Fixed task pool size")
    parser.add_argument("--runs", type=int, default=5, help="Number of paired runs")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    args = parser.parse_args()

    # Quick mode
    if args.quick:
        args.generations = 5
        args.population = 4
        args.tasks = 8
        args.pool_size = 100
        args.runs = 2

    print("=" * 70, flush=True)
    print("VERIFIABLE EVOLUTION V2 - IMPROVED DESIGN", flush=True)
    print("Fixed task pool | Paired seeds | Rolling fitness | Validation eval", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Generations: {args.generations}", flush=True)
    print(f"Population: {args.population}", flush=True)
    print(f"Tasks/gen: {args.tasks}", flush=True)
    print(f"Pool size: {args.pool_size}", flush=True)
    print(f"Paired runs: {args.runs}", flush=True)
    print("=" * 70, flush=True)

    # Create fixed task pool
    print("\nCreating fixed task pool...", flush=True)
    pool = create_fixed_pool(pool_size=args.pool_size, val_ratio=0.2, seed=args.seed)
    print(f"Pool stats: {pool.stats()}", flush=True)

    # Load model
    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(
        model_name=args.model,
        quantization="4bit",
        extraction_layer=-4,
        pooling="mean",
    )

    # Test prompts
    prompts = [
        "You are a precise mathematical reasoner. Think step by step.",
        "You are a logical problem solver. Break down problems carefully.",
    ]

    # Run paired comparisons
    all_results = []
    hyp_wins = 0
    euc_wins = 0
    ties = 0

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx * 1000

        print(f"\n{'#' * 70}", flush=True)
        print(f"# RUN {run_idx + 1}/{args.runs} (seed={run_seed})", flush=True)
        print(f"{'#' * 70}", flush=True)

        for prompt in prompts:
            print(f"\nPrompt: {prompt[:50]}...", flush=True)

            result = run_paired_comparison(
                encoder=encoder,
                prompt=prompt,
                pool=pool,
                max_generations=args.generations,
                population_size=args.population,
                tasks_per_gen=args.tasks,
                run_seed=run_seed,
            )

            result["prompt"] = prompt
            result["run_seed"] = run_seed
            all_results.append(result)

            if result["winner"] == "hyperbolic":
                hyp_wins += 1
            elif result["winner"] == "euclidean":
                euc_wins += 1
            else:
                ties += 1

            print(f"\n[RESULT] {result['winner'].upper()} (margin: {result['margin']:+.1%})", flush=True)

    # Final summary
    print("\n" + "=" * 70, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 70, flush=True)

    hyp_val_accs = [r["hyperbolic"]["validation_accuracy"] for r in all_results]
    euc_val_accs = [r["euclidean"]["validation_accuracy"] for r in all_results]
    hyp_divs = [r["hyperbolic"]["final_diversity"] for r in all_results]
    euc_divs = [r["euclidean"]["final_diversity"] for r in all_results]

    print(f"\nValidation Accuracy:")
    print(f"  Hyperbolic: {sum(hyp_val_accs)/len(hyp_val_accs):.1%} (avg)")
    print(f"  Euclidean:  {sum(euc_val_accs)/len(euc_val_accs):.1%} (avg)")

    print(f"\nFinal Diversity:")
    print(f"  Hyperbolic: {sum(hyp_divs)/len(hyp_divs):.3f} (avg)")
    print(f"  Euclidean:  {sum(euc_divs)/len(euc_divs):.3f} (avg)")
    div_ratio = sum(hyp_divs) / max(sum(euc_divs), 0.001)
    print(f"  Ratio (H/E): {div_ratio:.1f}x")

    print(f"\nHead-to-head (validation accuracy):")
    print(f"  Hyperbolic wins: {hyp_wins}")
    print(f"  Euclidean wins:  {euc_wins}")
    print(f"  Ties:            {ties}")

    # Statistical significance (simple paired t-test approximation)
    diffs = [h - e for h, e in zip(hyp_val_accs, euc_val_accs)]
    mean_diff = sum(diffs) / len(diffs)
    print(f"\nMean difference (H - E): {mean_diff:+.2%}")

    # Save results
    output_dir = Path(__file__).parent / "verifiable_evolution_results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"v2_results_{timestamp}.json"

    full_results = {
        "timestamp": timestamp,
        "config": {
            "model": args.model,
            "generations": args.generations,
            "population": args.population,
            "tasks_per_gen": args.tasks,
            "pool_size": args.pool_size,
            "runs": args.runs,
            "seed": args.seed,
        },
        "summary": {
            "hyperbolic_mean_val": sum(hyp_val_accs) / len(hyp_val_accs),
            "euclidean_mean_val": sum(euc_val_accs) / len(euc_val_accs),
            "hyperbolic_wins": hyp_wins,
            "euclidean_wins": euc_wins,
            "ties": ties,
            "mean_diff": mean_diff,
            "hyperbolic_mean_diversity": sum(hyp_divs) / len(hyp_divs),
            "euclidean_mean_diversity": sum(euc_divs) / len(euc_divs),
        },
        "runs": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}", flush=True)

    # Verdict
    print("\n" + "=" * 70, flush=True)
    if hyp_wins > euc_wins + 1:
        print("VERDICT: HYPERBOLIC WINS - Better validation accuracy", flush=True)
    elif euc_wins > hyp_wins + 1:
        print("VERDICT: EUCLIDEAN WINS - Better validation accuracy", flush=True)
    else:
        print("VERDICT: NO CLEAR WINNER - Results within noise margin", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
