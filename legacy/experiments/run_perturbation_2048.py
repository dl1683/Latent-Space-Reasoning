"""
Re-run baseline + perturbation at max_new_tokens=2048 for fair comparison with evolution.
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
from experiments.harness import auto_calibrate
from experiments.run_latent_sensitivity import decode_with_raw_soft_prompt


PLANNING_TASKS = [
    {
        "id": "plan_01_fraud_detection",
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
    seeds = [42, 43, 44, 45, 46]
    max_new_tokens = 2048
    output_path = str(Path(__file__).parent / "planning_bp_2048_results.json")

    print(f"{'='*70}")
    print(f"BASELINE + PERTURBATION @ 2048 TOKENS")
    print(f"Seeds: {seeds}, Max tokens: {max_new_tokens}")
    print(f"{'='*70}", flush=True)

    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(
        model_name="Qwen/Qwen3-4B",
        device_preference="auto",
        quantization="4bit",
    )
    calibration = auto_calibrate(encoder)
    print(f"Calibration: embed_dim={calibration['embed_dim']}, "
          f"rms={calibration['embedding_rms']:.4f}", flush=True)

    results = {
        "metadata": {
            "model": "Qwen/Qwen3-4B",
            "quantization": "4bit",
            "max_new_tokens": max_new_tokens,
            "seeds": seeds,
            "calibration": calibration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "outputs": [],
    }

    for task_idx, task in enumerate(PLANNING_TASKS):
        print(f"\n{'#'*70}")
        print(f"# TASK {task_idx+1}/{len(PLANNING_TASKS)}: {task['id']}")
        print(f"{'#'*70}", flush=True)

        # Baseline (deterministic, just 1)
        print(f"  Baseline...", end="", flush=True)
        start = time.time()
        response = encoder.generate_baseline(
            query=task["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        elapsed = time.time() - start
        results["outputs"].append({
            "condition": "greedy_baseline",
            "task_id": task["id"],
            "response": response,
            "word_count": len(response.split()),
            "elapsed": elapsed,
        })
        print(f" {len(response.split())}w, {elapsed:.1f}s", flush=True)

        # Perturbation (5 seeds)
        for seed in seeds:
            print(f"  Perturbation seed={seed}...", end="", flush=True)
            rng = torch.Generator().manual_seed(seed)
            sp = torch.randn(1, 2, calibration["embed_dim"],
                            generator=rng) * calibration["embedding_rms"]
            start = time.time()
            response, meta = decode_with_raw_soft_prompt(
                encoder, sp, task["prompt"],
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            elapsed = time.time() - start
            results["outputs"].append({
                "condition": "random_perturbation",
                "task_id": task["id"],
                "seed": seed,
                "response": response,
                "word_count": len(response.split()),
                "elapsed": elapsed,
                "generated_tokens": meta.get("generated_tokens"),
                "terminated_by_eos": meta.get("terminated_by_eos"),
            })
            print(f" {len(response.split())}w, {elapsed:.1f}s", flush=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Saved to {output_path}")
    print(f"Total outputs: {len(results['outputs'])}")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
