"""
Evolution condition for 3-way planning comparison.

Runs evolution on 5 planning tasks × 5 seeds, decodes with soft prompt at temp=0.
Merges with existing baseline + perturbation data from planning_comparison_results.json.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.config import Config, ScorerConfig
from latent_reasoning.engine import Engine

# Same 5 tasks as run_planning_comparison.py
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = args.output or str(
        Path(__file__).parent / "planning_evolution_results.json"
    )
    seeds = list(range(42, 42 + args.n_seeds))

    checkpoint_path = (
        Path(__file__).parent.parent
        / "checkpoints" / "latent_scorer" / "final_model.pt"
    )
    if not checkpoint_path.exists():
        print(f"ERROR: Trained scorer not found at {checkpoint_path}")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"EVOLUTION PLANNING COMPARISON")
    print(f"Model: {args.model}, Quantization: {args.quantization}")
    print(f"Seeds: {seeds}, Max tokens: {args.max_new_tokens}")
    print(f"Tasks: {len(PLANNING_TASKS)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}", flush=True)

    # Load model
    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(
        model_name=args.model,
        device_preference="auto",
        quantization=args.quantization,
    )

    results = {
        "metadata": {
            "model": args.model,
            "quantization": args.quantization,
            "n_seeds": args.n_seeds,
            "seeds": seeds,
            "max_new_tokens": args.max_new_tokens,
            "checkpoint": str(checkpoint_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "decode_method": "decode_with_soft_prompt (temp=0, greedy)",
        },
        "outputs": [],
    }

    for task_idx, task in enumerate(PLANNING_TASKS):
        print(f"\n{'#'*70}")
        print(f"# TASK {task_idx+1}/{len(PLANNING_TASKS)}: {task['id']}")
        print(f"{'#'*70}", flush=True)

        for seed in seeds:
            print(f"  Evolution seed={seed}...", end="", flush=True)

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            try:
                # Configure evolution
                config = Config()
                config.encoder.model = args.model
                config.encoder.quantization = args.quantization
                config.encoder.latent_dim = encoder.latent_dim

                config.judges.scorers = [ScorerConfig(
                    type="trained_latent",
                    checkpoint=str(checkpoint_path),
                    latent_dim=encoder.latent_dim,
                )]
                config.judges.modifiers = []

                # Evolution params
                config.evolution.chains = 6
                config.evolution.generations = 10
                config.evolution.temperature = 0.5
                config.evolution.temperature_decay = 0.95

                # We'll decode manually after evolution
                config.synthesis.max_tokens = args.max_new_tokens
                config.synthesis.temperature = 0.0
                config.synthesis.decode_strategy = "best"
                config.output.verbosity = "minimal"

                engine = Engine(config=config, encoder=encoder, verbosity="minimal")

                start = time.time()
                result = engine.run(task["prompt"])
                evo_elapsed = time.time() - start

                # Get the evolved latent from survivors
                if result.survivors:
                    best_latent = max(result.survivors, key=lambda s: s.score).latent
                else:
                    best_latent = encoder.encode(task["prompt"])

                # Decode using soft prompt injection at temp=0
                # This actually uses the latent meaningfully (unlike decode() at temp=0
                # which only uses it for RNG seeds, useless with greedy)
                start2 = time.time()
                soft_prompt_response = encoder.decode_with_soft_prompt(
                    best_latent,
                    query=task["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                )
                decode_elapsed = time.time() - start2

                entry = {
                    "condition": "evolution",
                    "task_id": task["id"],
                    "seed": seed,
                    "response_evo_decode": result.plan,  # Standard decode (may be same as baseline)
                    "response_soft_prompt": soft_prompt_response,  # Soft prompt decode
                    "response_length_evo": len(result.plan),
                    "response_length_soft": len(soft_prompt_response),
                    "word_count_evo": len(result.plan.split()),
                    "word_count_soft": len(soft_prompt_response.split()),
                    "evolution_elapsed": evo_elapsed,
                    "decode_elapsed": decode_elapsed,
                    "evolution_generations": result.generations,
                    "evolution_evaluations": result.evaluations,
                    "evolution_score": result.confidence,
                    "stop_reason": result.stop_reason,
                }
                results["outputs"].append(entry)
                print(f" evo={entry['word_count_evo']}w, soft={entry['word_count_soft']}w, "
                      f"score={result.confidence:.3f}, {evo_elapsed:.1f}s", flush=True)

            except Exception as e:
                import traceback
                print(f" ERROR: {e}", flush=True)
                traceback.print_exc()
                results["outputs"].append({
                    "condition": "evolution",
                    "task_id": task["id"],
                    "seed": seed,
                    "error": str(e),
                })

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Saved {len(results['outputs'])} evolution outputs to {output_path}")

    # Quick summary
    ok = [o for o in results["outputs"] if "error" not in o]
    if ok:
        evo_wc = [o["word_count_evo"] for o in ok]
        soft_wc = [o["word_count_soft"] for o in ok]
        print(f"Evo-decode words: mean={sum(evo_wc)/len(evo_wc):.0f}")
        print(f"Soft-prompt words: mean={sum(soft_wc)/len(soft_wc):.0f}")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
