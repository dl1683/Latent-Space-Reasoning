"""
Rigorous Hyperbolic vs Euclidean Evaluation

Generates FULL outputs (not truncated) with multiple runs per condition
for proper LLM-as-judge evaluation.
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


# Focus on prompts that require real reasoning
EVAL_PROMPTS = [
    # Math reasoning
    "Prove that the square root of 2 is irrational.",
    # Coding - algorithmic thinking
    "Write a function to detect a cycle in a linked list and explain your approach.",
    # Classic reasoning trap
    "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
    # System design - requires structured thinking
    "Design a system for a ride-sharing app like Uber.",
    # Analysis - compare/contrast
    "Compare and contrast REST and GraphQL APIs.",
]

RUNS_PER_CONDITION = 3  # Multiple runs for statistical validity


def run_evolution(prompt, encoder, judge_panel, geometry_config, geometry_name):
    """Run evolution and return full decoded output."""
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

    # Decode FULL output - no truncation
    is_hyperbolic = geometry_name == "hyperbolic"
    decoded = encoder.decode(
        result.best_latent,
        query=prompt,
        max_new_tokens=1024,  # Full output
        temperature=0.7,
        hyperbolic=is_hyperbolic,
        curvature=geometry_config.curvature if is_hyperbolic else 1.0,
    )

    return {
        "decoded": decoded,
        "score": result.best_score,
        "generations": result.generations,
        "survivors": len(result.survivors),
        "time": elapsed,
        "stop_reason": result.stop_reason,
    }


def main():
    print("=" * 70)
    print("RIGOROUS HYPERBOLIC VS EUCLIDEAN EVALUATION")
    print(f"Runs per condition: {RUNS_PER_CONDITION}")
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
            "runs_per_condition": RUNS_PER_CONDITION,
        },
        "evaluations": [],
    }

    for prompt_idx, prompt in enumerate(EVAL_PROMPTS):
        print(f"\n{'='*70}")
        print(f"PROMPT {prompt_idx + 1}/{len(EVAL_PROMPTS)}: {prompt[:60]}...")
        print(f"{'='*70}")

        eval_entry = {
            "prompt": prompt,
            "euclidean_runs": [],
            "hyperbolic_runs": [],
        }

        # Run multiple times for each geometry
        for run in range(RUNS_PER_CONDITION):
            print(f"\n  Run {run + 1}/{RUNS_PER_CONDITION}")

            # Euclidean
            print(f"    Euclidean...", end="", flush=True)
            euc = run_evolution(prompt, encoder, judge_panel, euclidean_config, "euclidean")
            print(f" done (score={euc['score']:.3f})")
            eval_entry["euclidean_runs"].append(euc)

            # Hyperbolic
            print(f"    Hyperbolic...", end="", flush=True)
            hyp = run_evolution(prompt, encoder, judge_panel, hyperbolic_config, "hyperbolic")
            print(f" done (score={hyp['score']:.3f})")
            eval_entry["hyperbolic_runs"].append(hyp)

        results["evaluations"].append(eval_entry)

        # Show sample outputs for this prompt
        print(f"\n  --- SAMPLE EUCLIDEAN OUTPUT (run 1) ---")
        print(f"  {eval_entry['euclidean_runs'][0]['decoded'][:300]}...")
        print(f"\n  --- SAMPLE HYPERBOLIC OUTPUT (run 1) ---")
        print(f"  {eval_entry['hyperbolic_runs'][0]['decoded'][:300]}...")

    # Save results
    output_path = Path(f"experiments/rigorous_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

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

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

    # Quick summary
    print("\nQUICK SUMMARY (for LLM-as-judge evaluation):")
    for eval_entry in results["evaluations"]:
        euc_scores = [r["score"] for r in eval_entry["euclidean_runs"]]
        hyp_scores = [r["score"] for r in eval_entry["hyperbolic_runs"]]
        print(f"\n{eval_entry['prompt'][:50]}...")
        print(f"  Euclidean mean score: {sum(euc_scores)/len(euc_scores):.4f}")
        print(f"  Hyperbolic mean score: {sum(hyp_scores)/len(hyp_scores):.4f}")


if __name__ == "__main__":
    main()
