"""
Experiment: Euclidean vs Hyperbolic Evolution Comparison

This experiment compares the performance of evolution in Euclidean latent space
versus evolution in hyperbolic (Poincaré ball) latent space.

Hypothesis: Hyperbolic geometry better matches hierarchical reasoning structures,
leading to:
1. Better diversity maintenance during evolution
2. Higher quality final solutions
3. More efficient exploration of the solution space
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch

from latent_reasoning.config import Config, GeometryConfig
from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.core.judge import create_scorer_from_config
from latent_reasoning.core.panel import JudgePanel
from latent_reasoning.evolution.loop import EvolutionLoop
from latent_reasoning.evolution.crossover import population_diversity, population_diversity_hyperbolic


# Test prompts covering different reasoning types
TEST_PROMPTS = [
    # Simple reasoning
    "Explain the concept of recursion in programming.",
    # Planning task
    "Create a plan for building a web application with user authentication.",
    # Analysis task
    "Analyze the trade-offs between monolithic and microservice architectures.",
    # Problem solving
    "How would you optimize a slow database query that takes 10 seconds?",
    # Multi-step reasoning
    "Design an algorithm to find the shortest path in a weighted graph.",
]


def create_euclidean_config(chains: int = 10, generations: int = 15) -> Config:
    """Create config for Euclidean evolution."""
    config = Config()
    config.geometry = GeometryConfig(space="euclidean")
    config.evolution.chains = chains
    config.evolution.generations = generations
    config.evolution.temperature = 0.5
    config.evolution.temperature_decay = 0.95
    config.output.verbosity = "silent"
    return config


def create_hyperbolic_config(chains: int = 10, generations: int = 15) -> Config:
    """Create config for Hyperbolic evolution."""
    config = Config()
    config.geometry = GeometryConfig(
        space="hyperbolic",
        curvature=1.0,
        tangent_scale=0.35,
        max_norm=0.98,
        mutation_noise_scale=0.35,
        barycenter_iterations=7,
        merge_threshold=0.15,
    )
    config.evolution.chains = chains
    config.evolution.generations = generations
    config.evolution.temperature = 0.5
    config.evolution.temperature_decay = 0.95
    config.output.verbosity = "silent"
    return config


def run_single_experiment(
    prompt: str,
    encoder: LLMEncoder,
    judge_panel: JudgePanel,
    config: Config,
    geometry_name: str,
) -> Dict[str, Any]:
    """Run a single evolution experiment and collect metrics."""

    # Encode the prompt
    seed = encoder.encode(prompt)

    # Create evolution loop
    evolution_loop = EvolutionLoop(
        judge_panel=judge_panel,
        config=config.evolution,
        geometry_config=config.geometry,
    )

    # Run evolution
    start_time = time.time()
    result = evolution_loop.run(seed, max_evaluations=config.budget.max_evaluations)
    elapsed_time = time.time() - start_time

    # Compute final diversity
    # Handle both ChainState objects and raw Tensors
    latents = []
    for s in result.survivors:
        if hasattr(s, 'latent'):
            latents.append(s.latent)
        elif isinstance(s, torch.Tensor):
            latents.append(s)
        else:
            continue
    if not latents:
        final_diversity = 0.0
    elif config.geometry.space == "hyperbolic":
        final_diversity = population_diversity_hyperbolic(latents, config.geometry.curvature)
    else:
        final_diversity = population_diversity(latents)

    # Extract diversity trajectory from history
    diversity_trajectory = []
    score_trajectory = []
    for h in result.history:
        if isinstance(h, dict):
            diversity_trajectory.append(h.get("diversity", 0.0))
            score_trajectory.append(h.get("best_raw_score", h.get("best_score", 0.0)))
        else:
            # Handle non-dict history entries
            diversity_trajectory.append(0.0)
            score_trajectory.append(0.0)

    return {
        "geometry": geometry_name,
        "prompt": prompt[:50] + "...",
        "best_score": result.best_score,
        "final_diversity": final_diversity,
        "generations": result.generations,
        "total_evaluations": result.total_evaluations,
        "converged": result.converged,
        "stop_reason": result.stop_reason,
        "elapsed_time": elapsed_time,
        "diversity_trajectory": diversity_trajectory,
        "score_trajectory": score_trajectory,
    }


def run_comparison(
    n_runs: int = 3,
    chains: int = 8,
    generations: int = 15,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run full comparison between Euclidean and Hyperbolic evolution.

    Args:
        n_runs: Number of runs per prompt per geometry
        chains: Population size
        generations: Number of evolution generations
        device: Device for computation
    """
    print("=" * 60)
    print("Hyperbolic vs Euclidean Evolution Comparison")
    print("=" * 60)

    # Create shared components
    print("\nLoading encoder...")
    encoder = LLMEncoder(
        model_name="Qwen/Qwen3-1.7B",
        extraction_layer=-4,
        pooling="mean",
        device_preference=device,
        quantization="4bit",
        latent_dim=1024,
    )

    print("Loading scorer...")
    scorer_config = Config().judges.scorers[0]
    scorer = create_scorer_from_config(
        scorer_config,
        device=device,
        encoder_latent_dim=1024,
    )

    judge_panel = JudgePanel(
        scorers=[scorer],
        modifiers=[],
        aggregation="mean",
        calibrate=True,
    )

    # Create configs
    euclidean_config = create_euclidean_config(chains, generations)
    hyperbolic_config = create_hyperbolic_config(chains, generations)

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_runs": n_runs,
            "chains": chains,
            "generations": generations,
            "device": device,
            "prompts": TEST_PROMPTS,
        },
        "euclidean": [],
        "hyperbolic": [],
    }

    # Run experiments
    for prompt_idx, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[{prompt_idx + 1}/{len(TEST_PROMPTS)}] Prompt: {prompt[:50]}...")

        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}:")

            # Euclidean
            print("    Euclidean...", end="", flush=True)
            try:
                euc_result = run_single_experiment(
                    prompt, encoder, judge_panel, euclidean_config, "euclidean"
                )
                euc_result["prompt_idx"] = prompt_idx
                euc_result["run"] = run
                results["euclidean"].append(euc_result)
                print(f" score={euc_result['best_score']:.4f}, div={euc_result['final_diversity']:.4f}")
            except Exception as e:
                print(f" ERROR: {e}")

            # Hyperbolic
            print("    Hyperbolic...", end="", flush=True)
            try:
                hyp_result = run_single_experiment(
                    prompt, encoder, judge_panel, hyperbolic_config, "hyperbolic"
                )
                hyp_result["prompt_idx"] = prompt_idx
                hyp_result["run"] = run
                results["hyperbolic"].append(hyp_result)
                print(f" score={hyp_result['best_score']:.4f}, div={hyp_result['final_diversity']:.4f}")
            except Exception as e:
                print(f" ERROR: {e}")

    # Compute summary statistics
    results["summary"] = compute_summary(results)

    return results


def compute_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary statistics from results."""
    summary = {}

    for geometry in ["euclidean", "hyperbolic"]:
        runs = results[geometry]
        if not runs:
            continue

        scores = [r["best_score"] for r in runs]
        diversities = [r["final_diversity"] for r in runs]
        times = [r["elapsed_time"] for r in runs]

        summary[geometry] = {
            "mean_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "mean_diversity": sum(diversities) / len(diversities),
            "mean_time": sum(times) / len(times),
            "n_runs": len(runs),
        }

    # Compute improvement
    if "euclidean" in summary and "hyperbolic" in summary:
        euc = summary["euclidean"]
        hyp = summary["hyperbolic"]

        summary["comparison"] = {
            "score_improvement": (hyp["mean_score"] - euc["mean_score"]) / (abs(euc["mean_score"]) + 1e-8) * 100,
            "diversity_improvement": (hyp["mean_diversity"] - euc["mean_diversity"]) / (abs(euc["mean_diversity"]) + 1e-8) * 100,
            "time_ratio": hyp["mean_time"] / (euc["mean_time"] + 1e-8),
        }

    return summary


def print_summary(results: Dict[str, Any]) -> None:
    """Print summary of results."""
    summary = results.get("summary", {})

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for geometry in ["euclidean", "hyperbolic"]:
        if geometry not in summary:
            continue
        s = summary[geometry]
        print(f"\n{geometry.upper()}:")
        print(f"  Mean Score:     {s['mean_score']:.4f}")
        print(f"  Max Score:      {s['max_score']:.4f}")
        print(f"  Min Score:      {s['min_score']:.4f}")
        print(f"  Mean Diversity: {s['mean_diversity']:.4f}")
        print(f"  Mean Time:      {s['mean_time']:.2f}s")

    if "comparison" in summary:
        c = summary["comparison"]
        print("\nCOMPARISON (Hyperbolic vs Euclidean):")
        print(f"  Score Improvement:     {c['score_improvement']:+.2f}%")
        print(f"  Diversity Improvement: {c['diversity_improvement']:+.2f}%")
        print(f"  Time Ratio:            {c['time_ratio']:.2f}x")


def main():
    """Run the experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare Euclidean vs Hyperbolic evolution")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per prompt")
    parser.add_argument("--chains", type=int, default=8, help="Population size")
    parser.add_argument("--generations", type=int, default=15, help="Evolution generations")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    results = run_comparison(
        n_runs=args.runs,
        chains=args.chains,
        generations=args.generations,
        device=args.device,
    )

    print_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"experiments/results_hyperbolic_{timestamp}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serializable items
    def make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
