"""
Manual Verification: Decode and inspect actual outputs from Euclidean vs Hyperbolic evolution.

This script shows the ACTUAL decoded text outputs so we can manually verify
that the system is producing coherent, quality responses.
"""

import sys
import io

# Fix unicode encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
from latent_reasoning.config import Config, GeometryConfig
from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.core.judge import create_scorer_from_config
from latent_reasoning.core.panel import JudgePanel
from latent_reasoning.evolution.loop import EvolutionLoop


def run_and_decode(
    prompt: str,
    encoder: LLMEncoder,
    judge_panel: JudgePanel,
    config: Config,
    geometry_name: str,
):
    """Run evolution and decode the best result."""
    print(f"\n{'='*60}")
    print(f"GEOMETRY: {geometry_name.upper()}")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}\n")

    # Encode
    seed = encoder.encode(prompt)

    # Create evolution loop
    evolution_loop = EvolutionLoop(
        judge_panel=judge_panel,
        config=config.evolution,
        geometry_config=config.geometry,
    )

    # Run evolution
    print("Running evolution...")
    result = evolution_loop.run(seed, max_evaluations=config.budget.max_evaluations)

    print(f"\nEvolution completed:")
    print(f"  - Generations: {result.generations}")
    print(f"  - Best score: {result.best_score:.4f}")
    print(f"  - Survivors: {len(result.survivors)}")
    print(f"  - Stop reason: {result.stop_reason}")

    # Decode the best latent
    print("\nDecoding best latent...")
    decoded = encoder.decode(
        result.best_latent,
        query=prompt,
        max_new_tokens=512,
        temperature=0.7,
    )

    print(f"\n{'='*60}")
    print("DECODED OUTPUT:")
    print(f"{'='*60}")
    print(decoded)
    print(f"{'='*60}\n")

    return {
        "geometry": geometry_name,
        "prompt": prompt,
        "score": result.best_score,
        "decoded": decoded,
        "generations": result.generations,
        "survivors": len(result.survivors),
    }


def main():
    # Test prompts
    prompts = [
        "Explain recursion in programming with a simple example.",
        "What are the key steps to optimize a slow database query?",
    ]

    print("Loading encoder (this may take a moment)...")
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

    # Create configs
    euclidean_config = Config()
    euclidean_config.geometry = GeometryConfig(space="euclidean")
    euclidean_config.evolution.chains = 8
    euclidean_config.evolution.generations = 10
    euclidean_config.output.verbosity = "minimal"

    hyperbolic_config = Config()
    hyperbolic_config.geometry = GeometryConfig(
        space="hyperbolic",
        curvature=1.0,
        tangent_scale=0.35,
        max_norm=0.98,
    )
    hyperbolic_config.evolution.chains = 8
    hyperbolic_config.evolution.generations = 10
    hyperbolic_config.output.verbosity = "minimal"

    results = []

    for prompt in prompts:
        print(f"\n\n{'#'*70}")
        print(f"TESTING PROMPT: {prompt}")
        print(f"{'#'*70}")

        # Baseline: Just decode the seed (no evolution)
        print("\n" + "="*60)
        print("BASELINE (No Evolution)")
        print("="*60)
        seed = encoder.encode(prompt)
        baseline_decoded = encoder.decode(
            seed,
            query=prompt,
            max_new_tokens=512,
            temperature=0.7,
        )
        print(f"Prompt: {prompt}\n")
        print("DECODED OUTPUT:")
        print("-"*40)
        print(baseline_decoded)
        print("-"*40)

        # Euclidean evolution
        euc_result = run_and_decode(
            prompt, encoder, judge_panel, euclidean_config, "euclidean"
        )
        results.append(euc_result)

        # Hyperbolic evolution
        hyp_result = run_and_decode(
            prompt, encoder, judge_panel, hyperbolic_config, "hyperbolic"
        )
        results.append(hyp_result)

    # Summary
    print("\n\n" + "#"*70)
    print("SUMMARY COMPARISON")
    print("#"*70)

    for i in range(0, len(results), 2):
        euc = results[i]
        hyp = results[i + 1]
        print(f"\nPrompt: {euc['prompt'][:50]}...")
        print(f"  Euclidean:  score={euc['score']:.4f}, survivors={euc['survivors']}")
        print(f"  Hyperbolic: score={hyp['score']:.4f}, survivors={hyp['survivors']}")


if __name__ == "__main__":
    main()
