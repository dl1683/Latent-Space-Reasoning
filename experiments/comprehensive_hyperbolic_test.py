"""
Comprehensive Hyperbolic Evolution Testing

Tests multiple prompts across different domains to thoroughly evaluate
the hyperbolic geometry approach.
"""

import sys
import io
import json
import time
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
from latent_reasoning.config import Config, GeometryConfig
from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.core.judge import create_scorer_from_config
from latent_reasoning.core.panel import JudgePanel
from latent_reasoning.evolution.loop import EvolutionLoop
from latent_reasoning.evolution.crossover import population_diversity, population_diversity_hyperbolic


# Comprehensive test prompts covering different reasoning types
TEST_PROMPTS = {
    "math": [
        "Prove that the square root of 2 is irrational.",
        "Explain why 0.999... equals 1.",
    ],
    "coding": [
        "Implement a binary search tree in Python with insert and search methods.",
        "Write a function to detect a cycle in a linked list.",
    ],
    "reasoning": [
        "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    ],
    "planning": [
        "Design a system for a ride-sharing app like Uber.",
        "Create a migration plan for moving a monolithic application to microservices.",
    ],
    "analysis": [
        "Compare and contrast REST and GraphQL APIs.",
        "What are the trade-offs between SQL and NoSQL databases?",
    ],
}


def run_single_test(prompt, encoder, judge_panel, geometry_config, geometry_name):
    """Run evolution and return metrics + decoded output."""
    config = Config()
    config.geometry = geometry_config
    config.evolution.chains = 10
    config.evolution.generations = 15
    config.output.verbosity = "silent"

    seed = encoder.encode(prompt)

    evolution_loop = EvolutionLoop(
        judge_panel=judge_panel,
        config=config.evolution,
        geometry_config=config.geometry,
    )

    start_time = time.time()
    result = evolution_loop.run(seed, max_evaluations=500)
    elapsed = time.time() - start_time

    # Decode - pass hyperbolic flag if using hyperbolic geometry
    is_hyperbolic = geometry_name == "hyperbolic"
    decoded = encoder.decode(
        result.best_latent,
        query=prompt,
        max_new_tokens=400,
        temperature=0.7,
        hyperbolic=is_hyperbolic,
        curvature=geometry_config.curvature if is_hyperbolic else 1.0,
    )

    # Compute diversity
    latents = []
    for s in result.survivors:
        if hasattr(s, 'latent'):
            latents.append(s.latent)
        elif isinstance(s, torch.Tensor):
            latents.append(s)

    if latents and geometry_name == "hyperbolic":
        diversity = population_diversity_hyperbolic(latents, geometry_config.curvature)
    elif latents:
        diversity = population_diversity(latents)
    else:
        diversity = 0.0

    return {
        "score": result.best_score,
        "generations": result.generations,
        "survivors": len(result.survivors),
        "diversity": diversity,
        "time": elapsed,
        "decoded": decoded[:500],  # Truncate for storage
        "stop_reason": result.stop_reason,
    }


def main():
    print("=" * 70)
    print("COMPREHENSIVE HYPERBOLIC EVOLUTION TEST")
    print("=" * 70)

    # Load components
    print("\nLoading encoder...")
    encoder = LLMEncoder(
        model_name="Qwen/Qwen3-1.7B",
        extraction_layer=-4,
        pooling="mean",
        device_preference="cuda",
        quantization="4bit",
        latent_dim=1024,
    )

    print("Loading scorer...")
    scorer_config = Config().judges.scorers[0]
    scorer = create_scorer_from_config(
        scorer_config,
        device="cuda",
        encoder_latent_dim=1024,
    )

    judge_panel = JudgePanel(
        scorers=[scorer],
        modifiers=[],
        aggregation="mean",
        calibrate=True,
    )

    # Geometry configs
    euclidean_config = GeometryConfig(space="euclidean")
    hyperbolic_config = GeometryConfig(
        space="hyperbolic",
        curvature=1.0,
        tangent_scale=0.35,
        max_norm=0.98,
    )

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": "Qwen/Qwen3-1.7B",
        },
        "categories": {},
    }

    total_euc_score = 0
    total_hyp_score = 0
    total_euc_div = 0
    total_hyp_div = 0
    total_euc_survivors = 0
    total_hyp_survivors = 0
    count = 0

    for category, prompts in TEST_PROMPTS.items():
        print(f"\n{'='*70}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'='*70}")

        results["categories"][category] = []

        for prompt in prompts:
            print(f"\nPrompt: {prompt[:60]}...")
            count += 1

            # Euclidean
            print("  Euclidean...", end="", flush=True)
            try:
                euc = run_single_test(prompt, encoder, judge_panel, euclidean_config, "euclidean")
                print(f" score={euc['score']:.3f}, survivors={euc['survivors']}, div={euc['diversity']:.4f}")
                total_euc_score += euc['score']
                total_euc_div += euc['diversity']
                total_euc_survivors += euc['survivors']
            except Exception as e:
                print(f" ERROR: {e}")
                euc = {"error": str(e)}

            # Hyperbolic
            print("  Hyperbolic...", end="", flush=True)
            try:
                hyp = run_single_test(prompt, encoder, judge_panel, hyperbolic_config, "hyperbolic")
                print(f" score={hyp['score']:.3f}, survivors={hyp['survivors']}, div={hyp['diversity']:.4f}")
                total_hyp_score += hyp['score']
                total_hyp_div += hyp['diversity']
                total_hyp_survivors += hyp['survivors']
            except Exception as e:
                print(f" ERROR: {e}")
                hyp = {"error": str(e)}

            results["categories"][category].append({
                "prompt": prompt,
                "euclidean": euc,
                "hyperbolic": hyp,
            })

            # Show decoded outputs for comparison
            if "decoded" in euc and "decoded" in hyp:
                print(f"\n  --- EUCLIDEAN OUTPUT ---")
                print(f"  {euc['decoded'][:200]}...")
                print(f"\n  --- HYPERBOLIC OUTPUT ---")
                print(f"  {hyp['decoded'][:200]}...")

    # Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    if count > 0:
        print(f"\nEUCLIDEAN (n={count}):")
        print(f"  Mean Score:     {total_euc_score/count:.4f}")
        print(f"  Mean Diversity: {total_euc_div/count:.4f}")
        print(f"  Mean Survivors: {total_euc_survivors/count:.1f}")

        print(f"\nHYPERBOLIC (n={count}):")
        print(f"  Mean Score:     {total_hyp_score/count:.4f}")
        print(f"  Mean Diversity: {total_hyp_div/count:.4f}")
        print(f"  Mean Survivors: {total_hyp_survivors/count:.1f}")

        print(f"\nIMPROVEMENT:")
        score_diff = (total_hyp_score - total_euc_score) / count
        print(f"  Score: {score_diff:+.4f} ({score_diff/(total_euc_score/count)*100:+.1f}%)")
        div_ratio = (total_hyp_div/count) / (total_euc_div/count + 1e-8)
        print(f"  Diversity: {div_ratio:.1f}x")
        surv_ratio = (total_hyp_survivors/count) / (total_euc_survivors/count + 1e-8)
        print(f"  Survivors: {surv_ratio:.1f}x")

    # Save results
    output_path = Path(f"experiments/comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    def make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(results), f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
