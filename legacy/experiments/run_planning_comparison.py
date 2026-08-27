"""
3+1 Way Comparison: Planning Tasks with LLM-as-Judge Evaluation

Conditions:
1. Greedy baseline (temp=0, no perturbation) — 1 deterministic output per task
2. Sampling control (temp=0.7, no perturbation, 5 seeds) — isolates sampling from intervention
3. Random perturbation + greedy (temp=0, 2-tok noise, 5 seeds) — our arithmetic approach
4. Evolution (temp=0.7, trained latent scorer, 5 seeds) — the original pipeline

Evaluation: Save all outputs for blind LLM-as-judge pairwise ranking.

Design informed by Codex review:
- Sampling control is mandatory (separates sampling benefit from evolution benefit)
- Log token counts, truncation, response length for every output
- Unit of analysis is the task, not the output
- 5 tasks = pilot study, not statistical claim
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from latent_reasoning.core.encoder import LLMEncoder
from experiments.harness import auto_calibrate
from experiments.run_latent_sensitivity import decode_with_raw_soft_prompt


# =============================================================================
# Task Definitions — Complex Planning Problems
# =============================================================================

PLANNING_TASKS = [
    {
        "id": "plan_01_fraud_detection",
        "category": "system_architecture",
        "prompt": (
            "Design a system architecture for a real-time fraud detection pipeline "
            "that must handle 100,000 transactions per second with less than 10ms latency, "
            "comply with PCI-DSS requirements, support model A/B testing without downtime, "
            "and degrade gracefully under 3x load spikes. Include specific technology choices, "
            "data flow, and failure handling for each component."
        ),
    },
    {
        "id": "plan_02_incident_response",
        "category": "security",
        "prompt": (
            "You are a security consultant. A company running 47 microservices discovered "
            "that 3 services were compromised via a supply chain attack through a shared npm "
            "dependency. The attacker has had access for approximately 72 hours. Design a "
            "containment, investigation, and recovery plan that minimizes business disruption "
            "while preserving forensic evidence. Address lateral movement risk, credential "
            "rotation strategy, communication protocol, and post-incident hardening."
        ),
    },
    {
        "id": "plan_03_data_platform",
        "category": "resource_planning",
        "prompt": (
            "You have $50K budget and 3 engineers for 6 months to build an MVP data platform "
            "for a healthcare startup. The platform must handle HIPAA-compliant data ingestion "
            "from 12 different EHR systems (each with different APIs and data formats), provide "
            "real-time operational dashboards for clinic managers, and support ML model training "
            "for patient risk stratification. Design the architecture, technology stack, project "
            "timeline with milestones, and explain which features to defer vs build first."
        ),
    },
    {
        "id": "plan_04_cache_debugging",
        "category": "debugging",
        "prompt": (
            "A Redis cluster (6 nodes, 3 masters + 3 replicas) serving a high-traffic "
            "e-commerce site starts experiencing intermittent cache inconsistencies during "
            "a flash sale with 10x normal traffic. Symptoms: some users see stale prices "
            "(up to 30 minutes old), others see items marked available when sold out, and "
            "the checkout service occasionally returns duplicate order IDs. Design a "
            "systematic investigation plan covering: immediate triage, root cause hypotheses "
            "ranked by probability, diagnostic commands to run, data to collect, and "
            "resolution strategies for each hypothesis."
        ),
    },
    {
        "id": "plan_05_db_migration",
        "category": "decision_analysis",
        "prompt": (
            "A CTO must choose between three database migration strategies for moving "
            "from Oracle to PostgreSQL: (A) Big-bang migration over a weekend with rollback "
            "plan, (B) Gradual strangler-fig pattern over 6 months with dual-write, "
            "(C) Shadow-read approach for 3 months then hard cutover. Context: 100TB of data, "
            "200 tables with complex triggers and stored procedures, 15 engineering teams "
            "depending on the database, zero-downtime requirement for revenue-critical "
            "services, and a regulatory audit in 4 months that requires documented data "
            "lineage. Analyze each option with specific risks, costs, timeline, team impact, "
            "and make a recommendation with justification."
        ),
    },
]


# =============================================================================
# Condition Runners
# =============================================================================

def run_greedy_baseline(encoder, task, max_new_tokens=1024):
    """Condition 1: Greedy baseline — deterministic, no perturbation."""
    start = time.time()
    response = encoder.generate_baseline(
        query=task["prompt"],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    elapsed = time.time() - start
    return {
        "condition": "greedy_baseline",
        "task_id": task["id"],
        "seed": None,
        "response": response,
        "response_length": len(response),
        "word_count": len(response.split()),
        "elapsed_seconds": elapsed,
    }


def run_sampling_control(encoder, task, seed, max_new_tokens=1024):
    """Condition 2: Sampling control — temp=0.7, no perturbation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    start = time.time()
    response = encoder.generate_baseline(
        query=task["prompt"],
        max_new_tokens=max_new_tokens,
        temperature=0.7,
    )
    elapsed = time.time() - start
    return {
        "condition": "sampling_control",
        "task_id": task["id"],
        "seed": seed,
        "response": response,
        "response_length": len(response),
        "word_count": len(response.split()),
        "elapsed_seconds": elapsed,
    }


def run_random_perturbation(encoder, task, seed, calibration, max_new_tokens=1024):
    """Condition 3: Random 2-tok noise + greedy decoding."""
    n_tokens = 2
    embed_dim = calibration["embed_dim"]
    rms = calibration["embedding_rms"]

    rng = torch.Generator().manual_seed(seed)
    sp = torch.randn(1, n_tokens, embed_dim, generator=rng) * rms

    start = time.time()
    response, gen_meta = decode_with_raw_soft_prompt(
        encoder, sp, task["prompt"],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    elapsed = time.time() - start
    return {
        "condition": "random_perturbation",
        "task_id": task["id"],
        "seed": seed,
        "response": response,
        "response_length": len(response),
        "word_count": len(response.split()),
        "elapsed_seconds": elapsed,
        "generated_tokens": gen_meta.get("generated_tokens"),
        "terminated_by_eos": gen_meta.get("terminated_by_eos"),
    }


def run_evolution(encoder, task, seed, checkpoint_path, max_new_tokens=1024):
    """Condition 4: Evolution with trained latent scorer."""
    from latent_reasoning.config import Config, ScorerConfig
    from latent_reasoning.engine import Engine

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Configure engine to use existing encoder and trained scorer
    config = Config()
    config.encoder.model = "Qwen/Qwen3-4B"
    config.encoder.quantization = "4bit"
    config.encoder.latent_dim = encoder.latent_dim

    # Use trained latent scorer (proper Pydantic config)
    config.judges.scorers = [ScorerConfig(
        type="trained_latent",
        checkpoint=str(checkpoint_path),
        latent_dim=encoder.latent_dim,
    )]
    config.judges.modifiers = []

    # Evolution params — moderate (not too expensive for pilot)
    config.evolution.chains = 6
    config.evolution.generations = 10
    config.evolution.temperature = 0.5
    config.evolution.temperature_decay = 0.95

    # Decode with sampling (evolution needs temp>0 to vary output)
    config.synthesis.max_tokens = max_new_tokens
    config.synthesis.temperature = 0.7
    config.synthesis.decode_strategy = "best"

    config.output.verbosity = "minimal"

    # Create engine, reusing the loaded encoder
    engine = Engine(config=config, encoder=encoder, verbosity="minimal")

    start = time.time()
    result = engine.run(task["prompt"])
    elapsed = time.time() - start

    return {
        "condition": "evolution",
        "task_id": task["id"],
        "seed": seed,
        "response": result.plan,
        "response_length": len(result.plan),
        "word_count": len(result.plan.split()),
        "elapsed_seconds": elapsed,
        "evolution_generations": result.generations,
        "evolution_evaluations": result.evaluations,
        "evolution_score": result.confidence,
        "stop_reason": result.stop_reason,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Planning task 4-way comparison")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-evolution", action="store_true",
                        help="Skip evolution condition (faster for debugging)")
    args = parser.parse_args()

    output_path = args.output or str(
        Path(__file__).parent / "planning_comparison_results.json"
    )
    seeds = list(range(42, 42 + args.n_seeds))

    print(f"{'='*70}")
    print(f"PLANNING TASK 4-WAY COMPARISON")
    print(f"Model: {args.model}, Quantization: {args.quantization}")
    print(f"Seeds: {seeds}, Max tokens: {args.max_new_tokens}")
    print(f"Tasks: {len(PLANNING_TASKS)}")
    print(f"{'='*70}", flush=True)

    # Load model
    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(
        model_name=args.model,
        device_preference="auto",
        quantization=args.quantization,
    )
    calibration = auto_calibrate(encoder)
    print(f"Calibration: embed_dim={calibration['embed_dim']}, "
          f"rms={calibration['embedding_rms']:.4f}", flush=True)

    # Find trained scorer checkpoint
    checkpoint_path = (
        Path(__file__).parent.parent
        / "checkpoints" / "latent_scorer" / "final_model.pt"
    )
    if not checkpoint_path.exists():
        print(f"WARNING: Trained scorer not found at {checkpoint_path}")
        print("Evolution condition will be skipped.", flush=True)
        args.skip_evolution = True

    all_results = {
        "metadata": {
            "model": args.model,
            "quantization": args.quantization,
            "n_seeds": args.n_seeds,
            "seeds": seeds,
            "max_new_tokens": args.max_new_tokens,
            "n_tasks": len(PLANNING_TASKS),
            "calibration": calibration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "conditions": [
                "greedy_baseline",
                "sampling_control",
                "random_perturbation",
            ] + ([] if args.skip_evolution else ["evolution"]),
        },
        "tasks": [t["id"] for t in PLANNING_TASKS],
        "outputs": [],
    }

    for task_idx, task in enumerate(PLANNING_TASKS):
        print(f"\n{'#'*70}")
        print(f"# TASK {task_idx+1}/{len(PLANNING_TASKS)}: {task['id']}")
        print(f"# Category: {task['category']}")
        print(f"{'#'*70}", flush=True)

        # Condition 1: Greedy baseline
        print(f"\n  [1/4] Greedy baseline...", end="", flush=True)
        result = run_greedy_baseline(encoder, task, args.max_new_tokens)
        all_results["outputs"].append(result)
        print(f" {result['word_count']} words, {result['elapsed_seconds']:.1f}s",
              flush=True)

        # Condition 2: Sampling control (5 seeds)
        for seed_idx, seed in enumerate(seeds):
            print(f"  [2/4] Sampling control seed={seed}...", end="", flush=True)
            result = run_sampling_control(encoder, task, seed, args.max_new_tokens)
            all_results["outputs"].append(result)
            print(f" {result['word_count']} words, {result['elapsed_seconds']:.1f}s",
                  flush=True)

        # Condition 3: Random perturbation (5 seeds)
        for seed_idx, seed in enumerate(seeds):
            print(f"  [3/4] Random perturbation seed={seed}...", end="", flush=True)
            result = run_random_perturbation(
                encoder, task, seed, calibration, args.max_new_tokens
            )
            all_results["outputs"].append(result)
            print(f" {result['word_count']} words, {result['elapsed_seconds']:.1f}s",
                  flush=True)

        # Condition 4: Evolution (5 seeds)
        if not args.skip_evolution:
            for seed_idx, seed in enumerate(seeds):
                print(f"  [4/4] Evolution seed={seed}...", end="", flush=True)
                try:
                    result = run_evolution(
                        encoder, task, seed, checkpoint_path, args.max_new_tokens
                    )
                    all_results["outputs"].append(result)
                    print(f" {result['word_count']} words, "
                          f"score={result['evolution_score']:.3f}, "
                          f"{result['elapsed_seconds']:.1f}s", flush=True)
                except Exception as e:
                    print(f" ERROR: {e}", flush=True)
                    all_results["outputs"].append({
                        "condition": "evolution",
                        "task_id": task["id"],
                        "seed": seed,
                        "error": str(e),
                    })

        # Memory cleanup between tasks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    print(f"\n{'='*70}")
    print(f"Saving results to {output_path}", flush=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    conditions = set(o["condition"] for o in all_results["outputs"] if "error" not in o)
    for cond in sorted(conditions):
        outputs = [o for o in all_results["outputs"]
                   if o["condition"] == cond and "error" not in o]
        word_counts = [o["word_count"] for o in outputs]
        times = [o["elapsed_seconds"] for o in outputs]
        print(f"\n{cond}:")
        print(f"  Outputs: {len(outputs)}")
        print(f"  Words: mean={sum(word_counts)/len(word_counts):.0f}, "
              f"min={min(word_counts)}, max={max(word_counts)}")
        print(f"  Time: mean={sum(times)/len(times):.1f}s, "
              f"total={sum(times):.1f}s")

    total_outputs = len([o for o in all_results["outputs"] if "error" not in o])
    print(f"\nTotal outputs: {total_outputs}")
    print(f"Ready for LLM-as-judge evaluation.", flush=True)


if __name__ == "__main__":
    main()
