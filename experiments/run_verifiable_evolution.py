"""
Verifiable Evolution Experiment

Compare hyperbolic vs euclidean evolution using GROUND TRUTH fitness.
This is the critical test - does hyperbolic actually find better solutions
when selection is based on real correctness?

Key insight from Codex:
"The breakthrough isn't better scoring - it's making truth observable."
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.config import GeometryConfig
from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.verification.verifiable_tasks import (
    VerifiableTask,
    VerifiableTaskSuite,
    create_task_suite,
)
from latent_reasoning.verification.verifiable_evolution import (
    VerifiableEvolutionLoop,
    VerifiableEvolutionResult,
)


def run_baseline_accuracy(
    encoder: LLMEncoder,
    task_suite: VerifiableTaskSuite,
    prompts: list[str],
    tasks_per_prompt: int = 20,
) -> dict:
    """
    Measure baseline accuracy without evolution.

    This answers: "How accurate is the model on verifiable tasks
    without any latent evolution?"
    """
    results = {
        "total_correct": 0,
        "total_tasks": 0,
        "per_prompt": [],
    }

    for prompt in prompts:
        print(f"\n[BASELINE] Testing prompt: {prompt[:50]}...")

        # Encode prompt
        latent = encoder.encode(prompt)

        # Generate tasks
        tasks = task_suite.generate_batch(tasks_per_prompt)

        correct = 0
        for task in tasks:
            response = encoder.decode(
                latent,
                query=task.prompt,
                max_new_tokens=100,
                temperature=0.3,
            )

            if task_suite.evaluate_response(task, response):
                correct += 1

        accuracy = correct / len(tasks)
        results["total_correct"] += correct
        results["total_tasks"] += len(tasks)
        results["per_prompt"].append({
            "prompt": prompt,
            "correct": correct,
            "total": len(tasks),
            "accuracy": accuracy,
        })

        print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(tasks)})")

    results["overall_accuracy"] = results["total_correct"] / results["total_tasks"]
    return results


def run_evolution_experiment(
    encoder: LLMEncoder,
    prompt: str,
    geometry: str,
    geometry_config: GeometryConfig,
    max_generations: int = 20,
    population_size: int = 8,
    tasks_per_evaluation: int = 10,
    seed: int | None = None,
) -> dict:
    """Run evolution experiment with specified geometry."""

    print(f"\n{'='*60}")
    print(f"[{geometry.upper()}] Running evolution on: {prompt[:50]}...")
    print(f"{'='*60}")

    # Encode seed latent
    seed_latent = encoder.encode(prompt)

    # Map to hyperbolic if needed
    if geometry == "hyperbolic":
        from latent_reasoning.utils import hyperbolic as hyp
        seed_latent = hyp.expmap0(
            seed_latent.squeeze() * geometry_config.tangent_scale,
            geometry_config.curvature,
        )
        seed_latent = seed_latent.unsqueeze(0)

    # Create evolution loop
    loop = VerifiableEvolutionLoop(
        encoder=encoder,
        geometry_config=geometry_config,
        population_size=population_size,
        tasks_per_evaluation=tasks_per_evaluation,
        mutation_scale=0.1 if geometry == "euclidean" else 0.15,
        elite_count=2,
        seed=seed,
    )

    # Run evolution
    start_time = time.time()
    result = loop.run(
        seed_latent,
        max_generations=max_generations,
        target_fitness=0.95,
        patience=8,
    )
    elapsed = time.time() - start_time

    # Extract final stats
    return {
        "geometry": geometry,
        "prompt": prompt,
        "best_fitness": result.best_fitness,
        "best_correct": result.best_correct,
        "best_total": result.best_total,
        "generations": result.generations,
        "elapsed_seconds": elapsed,
        "history": result.history,
        "final_population_size": len(result.final_population),
    }


def compare_geometries(
    encoder: LLMEncoder,
    prompts: list[str],
    max_generations: int = 20,
    population_size: int = 8,
    tasks_per_evaluation: int = 10,
    runs_per_condition: int = 3,
    seed: int = 42,
) -> dict:
    """Compare hyperbolic vs euclidean evolution."""

    # Geometry configs
    hyperbolic_config = GeometryConfig(
        space="hyperbolic",
        curvature=1.0,
        tangent_scale=0.35,
        max_norm=0.98,
    )

    euclidean_config = GeometryConfig(space="euclidean")

    results = {
        "hyperbolic": [],
        "euclidean": [],
        "comparison": {},
    }

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n{'#'*70}")
        print(f"# PROMPT {prompt_idx + 1}/{len(prompts)}: {prompt[:60]}...")
        print(f"{'#'*70}")

        for run in range(runs_per_condition):
            run_seed = seed + prompt_idx * 100 + run
            random.seed(run_seed)
            torch.manual_seed(run_seed)

            # Hyperbolic
            hyp_result = run_evolution_experiment(
                encoder=encoder,
                prompt=prompt,
                geometry="hyperbolic",
                geometry_config=hyperbolic_config,
                max_generations=max_generations,
                population_size=population_size,
                tasks_per_evaluation=tasks_per_evaluation,
                seed=run_seed,
            )
            hyp_result["run"] = run
            results["hyperbolic"].append(hyp_result)

            # Euclidean
            euc_result = run_evolution_experiment(
                encoder=encoder,
                prompt=prompt,
                geometry="euclidean",
                geometry_config=euclidean_config,
                max_generations=max_generations,
                population_size=population_size,
                tasks_per_evaluation=tasks_per_evaluation,
                seed=run_seed,
            )
            euc_result["run"] = run
            results["euclidean"].append(euc_result)

            # Quick comparison
            print(f"\n[RUN {run+1}] Hyperbolic: {hyp_result['best_fitness']:.1%} | Euclidean: {euc_result['best_fitness']:.1%}")

    # Aggregate comparison
    hyp_fitnesses = [r["best_fitness"] for r in results["hyperbolic"]]
    euc_fitnesses = [r["best_fitness"] for r in results["euclidean"]]

    results["comparison"] = {
        "hyperbolic_mean": sum(hyp_fitnesses) / len(hyp_fitnesses),
        "euclidean_mean": sum(euc_fitnesses) / len(euc_fitnesses),
        "hyperbolic_max": max(hyp_fitnesses),
        "euclidean_max": max(euc_fitnesses),
        "hyperbolic_wins": sum(1 for h, e in zip(hyp_fitnesses, euc_fitnesses) if h > e),
        "euclidean_wins": sum(1 for h, e in zip(hyp_fitnesses, euc_fitnesses) if e > h),
        "ties": sum(1 for h, e in zip(hyp_fitnesses, euc_fitnesses) if h == e),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run verifiable evolution experiment")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Model to use")
    parser.add_argument("--generations", type=int, default=15, help="Max generations")
    parser.add_argument("--population", type=int, default=8, help="Population size")
    parser.add_argument("--tasks", type=int, default=10, help="Tasks per evaluation")
    parser.add_argument("--runs", type=int, default=2, help="Runs per condition")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    args = parser.parse_args()

    # Quick mode for testing
    if args.quick:
        args.generations = 5
        args.population = 4
        args.tasks = 5
        args.runs = 1

    print("="*70)
    print("VERIFIABLE EVOLUTION EXPERIMENT")
    print("Ground-truth fitness: Actual correctness on verifiable tasks")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Tasks/eval: {args.tasks}")
    print(f"Runs: {args.runs}")
    print("="*70)

    # Load model
    print("\nLoading model...")
    encoder = LLMEncoder(
        model_name=args.model,
        quantization="4bit",
        extraction_layer=-4,
        pooling="mean",
    )

    # Create task suite
    task_suite = create_task_suite(seed=args.seed)

    # Test prompts - reasoning-focused
    prompts = [
        "You are a precise mathematical reasoner. Think step by step.",
        "You are a logical problem solver. Break down problems carefully.",
        "You are an analytical thinker. Consider each step methodically.",
    ]

    # Run baseline first
    print("\n" + "="*70)
    print("PHASE 1: BASELINE ACCURACY (No Evolution)")
    print("="*70)

    baseline_results = run_baseline_accuracy(
        encoder, task_suite, prompts[:2], tasks_per_prompt=20
    )

    print(f"\nBaseline overall accuracy: {baseline_results['overall_accuracy']:.1%}")

    if args.baseline_only:
        print("\n[Baseline only mode - stopping here]")
        return

    # Run comparison
    print("\n" + "="*70)
    print("PHASE 2: EVOLUTION COMPARISON (Hyperbolic vs Euclidean)")
    print("="*70)

    comparison_results = compare_geometries(
        encoder=encoder,
        prompts=prompts[:2],  # Use 2 prompts for quick testing
        max_generations=args.generations,
        population_size=args.population,
        tasks_per_evaluation=args.tasks,
        runs_per_condition=args.runs,
        seed=args.seed,
    )

    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    comp = comparison_results["comparison"]
    print(f"\nBaseline accuracy: {baseline_results['overall_accuracy']:.1%}")
    print(f"\nAfter evolution:")
    print(f"  Hyperbolic mean: {comp['hyperbolic_mean']:.1%}")
    print(f"  Euclidean mean:  {comp['euclidean_mean']:.1%}")
    print(f"  Hyperbolic max:  {comp['hyperbolic_max']:.1%}")
    print(f"  Euclidean max:   {comp['euclidean_max']:.1%}")
    print(f"\nHead-to-head:")
    print(f"  Hyperbolic wins: {comp['hyperbolic_wins']}")
    print(f"  Euclidean wins:  {comp['euclidean_wins']}")
    print(f"  Ties:            {comp['ties']}")

    # Improvement over baseline
    hyp_improvement = comp['hyperbolic_mean'] - baseline_results['overall_accuracy']
    euc_improvement = comp['euclidean_mean'] - baseline_results['overall_accuracy']

    print(f"\nImprovement over baseline:")
    print(f"  Hyperbolic: {hyp_improvement:+.1%}")
    print(f"  Euclidean:  {euc_improvement:+.1%}")

    # Save results
    output_dir = Path(__file__).parent / "verifiable_evolution_results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"

    full_results = {
        "timestamp": timestamp,
        "config": {
            "model": args.model,
            "generations": args.generations,
            "population": args.population,
            "tasks_per_eval": args.tasks,
            "runs": args.runs,
            "seed": args.seed,
        },
        "baseline": baseline_results,
        "evolution": comparison_results,
    }

    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    # Key verdict
    print("\n" + "="*70)
    if hyp_improvement > euc_improvement + 0.05:
        print("VERDICT: HYPERBOLIC WINS - Better ground-truth accuracy")
    elif euc_improvement > hyp_improvement + 0.05:
        print("VERDICT: EUCLIDEAN WINS - Better ground-truth accuracy")
    else:
        print("VERDICT: NO SIGNIFICANT DIFFERENCE")
    print("="*70)


if __name__ == "__main__":
    main()
