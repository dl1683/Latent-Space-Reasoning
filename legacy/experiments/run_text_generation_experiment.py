"""Text generation experiment: perturbation on small model vs bigger model baseline.

Generates responses to open-ended text tasks using:
  1. Qwen3-4B 4-bit greedy baseline
  2. Qwen3-4B 4-bit with perturbation (5 seeds, pick best by length/completion)
  3. Qwen3-14B 4-bit greedy baseline

Saves all outputs to a JSON file for LLM-as-judge evaluation afterward.
Checkpoints after every task so progress survives interruptions.
"""

from __future__ import annotations

import gc
import json
import time
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR.parent / "src"))

TASKS = [
    {
        "id": "text_001",
        "category": "explanation",
        "prompt": "Explain how a hash table works, including collision resolution strategies. Give concrete examples.",
    },
    {
        "id": "text_002",
        "category": "analysis",
        "prompt": "What are the trade-offs between microservices and monolithic architecture? When would you choose each?",
    },
    {
        "id": "text_003",
        "category": "reasoning",
        "prompt": "A company has 100 employees. 60% use Windows, 40% use Mac. 30% of Windows users and 50% of Mac users use Chrome. What percentage of all employees use Chrome? Show your reasoning step by step.",
    },
    {
        "id": "text_004",
        "category": "creative",
        "prompt": "Write a short technical blog post (3-4 paragraphs) explaining why database indexing matters, aimed at junior developers.",
    },
    {
        "id": "text_005",
        "category": "planning",
        "prompt": "Design a caching strategy for a social media feed that serves 10 million users. Consider cache invalidation, consistency, and cold start problems.",
    },
    {
        "id": "text_006",
        "category": "debugging",
        "prompt": "A web application loads slowly. The database queries return in 50ms, the API responds in 200ms, but the page takes 5 seconds to load. What could be the cause? List possible issues and how you'd diagnose each one.",
    },
    {
        "id": "text_007",
        "category": "explanation",
        "prompt": "Explain the CAP theorem in distributed systems. Give a real-world example of a system that chooses CP over AP, and one that chooses AP over CP.",
    },
    {
        "id": "text_008",
        "category": "reasoning",
        "prompt": "You have 8 identical-looking balls. One is heavier than the rest. You have a balance scale. What is the minimum number of weighings needed to find the heavy ball? Prove your answer.",
    },
    {
        "id": "text_009",
        "category": "analysis",
        "prompt": "Compare REST, GraphQL, and gRPC for API design. For each, describe a scenario where it is the best choice and why.",
    },
    {
        "id": "text_010",
        "category": "planning",
        "prompt": "You need to migrate a production PostgreSQL database from one cloud provider to another with less than 5 minutes of downtime. Describe your migration plan step by step.",
    },
    {
        "id": "text_011",
        "category": "creative",
        "prompt": "Write a clear, concise commit message and PR description for a change that refactors a monolithic user authentication module into separate services for login, session management, and password reset.",
    },
    {
        "id": "text_012",
        "category": "reasoning",
        "prompt": "A recursive function computes fibonacci(n) with time complexity O(2^n). Explain why, then show how memoization reduces it to O(n). What is the space complexity trade-off?",
    },
    {
        "id": "text_013",
        "category": "debugging",
        "prompt": "A distributed system has three services: A calls B, B calls C. Users report intermittent 500 errors. Service A logs show timeouts to B, but B's logs show all requests complete successfully in under 100ms. What could explain this discrepancy?",
    },
    {
        "id": "text_014",
        "category": "explanation",
        "prompt": "Explain how gradient descent works in machine learning. Start from the intuition, then build up to the math. Include the learning rate's role and what happens when it's too high or too low.",
    },
    {
        "id": "text_015",
        "category": "analysis",
        "prompt": "What are the security implications of storing JWTs in localStorage vs httpOnly cookies? Which approach would you recommend for a banking application and why?",
    },
]

OUTPUT_PATH = EXPERIMENTS_DIR / "text_generation_results.json"
CKPT_PATH = EXPERIMENTS_DIR / ".text_generation_checkpoint.json"
N_PERTURBATION_SEEDS = 5


def _load_checkpoint():
    if CKPT_PATH.exists():
        with open(CKPT_PATH) as f:
            return json.load(f)
    return None


def _save_checkpoint(data):
    tmp = CKPT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(CKPT_PATH)


def generate_baseline(encoder, prompt, max_new_tokens=1024):
    """Standard greedy generation."""
    import torch

    system_msg = "Answer to the best of your ability."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    try:
        formatted = encoder.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        formatted = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    inputs = encoder.tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(encoder._device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        out = encoder.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=encoder.tokenizer.pad_token_id,
            eos_token_id=encoder.tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    elapsed = time.time() - t0

    n_prompt = inputs["input_ids"].shape[1]
    n_generated = out[0].shape[0] - n_prompt
    eos_id = encoder.tokenizer.eos_token_id
    terminated_by_eos = bool(eos_id is not None and out[0][-1].item() == eos_id)
    resp = encoder.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    return {
        "response": resp,
        "generated_tokens": int(n_generated),
        "terminated_by_eos": terminated_by_eos,
        "time": round(elapsed, 1),
    }


def generate_with_perturbation(encoder, prompt, seed, embed_dim, embedding_rms,
                                num_soft_tokens=2, max_new_tokens=1024):
    """Generation with random embedding perturbation prefix."""
    import torch
    from run_latent_sensitivity import decode_with_raw_soft_prompt

    torch.manual_seed(42 + seed * 7)
    noise = torch.randn(1, num_soft_tokens, embed_dim) * embedding_rms

    t0 = time.time()
    resp, meta = decode_with_raw_soft_prompt(
        encoder, noise, prompt, max_new_tokens=max_new_tokens,
    )
    elapsed = time.time() - t0

    return {
        "response": resp,
        "seed": seed,
        "generated_tokens": meta.get("generated_tokens", 0),
        "terminated_by_eos": meta.get("terminated_by_eos", False),
        "time": round(elapsed, 1),
    }


def pick_best_perturbation(results):
    """Select best perturbation output: prefer completed (EOS), then longest."""
    completed = [r for r in results if r["terminated_by_eos"]]
    pool = completed if completed else results
    return max(pool, key=lambda r: len(r["response"]))


def run_model(model_name, quantization, tasks, ckpt_results, model_key):
    """Run all tasks for a single model configuration."""
    import torch
    from latent_reasoning.core.encoder import LLMEncoder
    from harness import auto_calibrate

    print(f"\nLoading {model_name} {quantization}...")
    encoder = LLMEncoder(model_name=model_name, quantization=quantization)
    cal = auto_calibrate(encoder)
    print(f"  embed_dim={cal['embed_dim']}, rms={cal['embedding_rms']:.5f}")

    results = ckpt_results.get(model_key, {})

    for ti, task in enumerate(tasks):
        task_id = task["id"]
        if task_id in results:
            print(f"  [{ti+1}/{len(tasks)}] {task_id} — cached")
            continue

        print(f"  [{ti+1}/{len(tasks)}] {task_id} ({task['category']})...", end="", flush=True)
        task_result = {"task_id": task_id, "category": task["category"], "prompt": task["prompt"]}

        # Baseline (greedy)
        baseline = generate_baseline(encoder, task["prompt"])
        task_result["baseline"] = baseline

        # Perturbation seeds (only for 4B)
        if "4B" in model_name and "14B" not in model_name and "32B" not in model_name:
            pert_results = []
            for seed in range(N_PERTURBATION_SEEDS):
                pr = generate_with_perturbation(
                    encoder, task["prompt"], seed,
                    cal["embed_dim"], cal["embedding_rms"],
                )
                pert_results.append(pr)
            task_result["perturbation_seeds"] = pert_results
            task_result["perturbation_best"] = pick_best_perturbation(pert_results)

        results[task_id] = task_result
        ckpt_results[model_key] = results
        _save_checkpoint(ckpt_results)

        n_tok = baseline["generated_tokens"]
        print(f" baseline={n_tok}tok", end="")
        if "perturbation_best" in task_result:
            pb = task_result["perturbation_best"]
            print(f", best_pert={pb['generated_tokens']}tok (seed {pb['seed']})", end="")
        print(f" [{baseline['time']}s]")

        gc.collect()
        torch.cuda.empty_cache()

    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Text generation: perturbation vs scaling")
    parser.add_argument("--skip-14b", action="store_true", help="Skip 14B model")
    args = parser.parse_args()

    print("=" * 70)
    print("TEXT GENERATION EXPERIMENT")
    print(f"Tasks: {len(TASKS)}, Perturbation seeds: {N_PERTURBATION_SEEDS}")
    print("=" * 70)

    ckpt = _load_checkpoint() or {}

    # Phase 1: Qwen3-4B (baseline + perturbation)
    print("\n" + "=" * 70)
    print("PHASE 1: Qwen3-4B 4-bit (baseline + perturbation)")
    print("=" * 70)
    ckpt["4b_4bit"] = run_model("Qwen/Qwen3-4B", "4bit", TASKS, ckpt, "4b_4bit")

    # Phase 2: Qwen3-14B (baseline only)
    if not args.skip_14b:
        print("\n" + "=" * 70)
        print("PHASE 2: Qwen3-14B 4-bit (baseline only)")
        print("=" * 70)
        ckpt["14b_4bit"] = run_model("Qwen/Qwen3-14B", "4bit", TASKS, ckpt, "14b_4bit")

    # Save final results
    output = {
        "experiment": "text_generation_perturbation_vs_scaling",
        "n_tasks": len(TASKS),
        "n_perturbation_seeds": N_PERTURBATION_SEEDS,
        "max_new_tokens": 1024,
        "models": {
            "small": "Qwen/Qwen3-4B (4-bit)",
            "large": "Qwen/Qwen3-14B (4-bit)",
        },
        "results": ckpt,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {OUTPUT_PATH}")

    if CKPT_PATH.exists():
        CKPT_PATH.unlink()
        print("Checkpoint removed (experiment complete)")

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    if "4b_4bit" in ckpt:
        results_4b = ckpt["4b_4bit"]
        eos_baseline = sum(1 for r in results_4b.values() if r["baseline"]["terminated_by_eos"])
        print(f"4B baseline: {eos_baseline}/{len(results_4b)} completed (EOS)")
        if any("perturbation_best" in r for r in results_4b.values()):
            eos_pert = sum(1 for r in results_4b.values()
                          if r.get("perturbation_best", {}).get("terminated_by_eos", False))
            print(f"4B best perturbation: {eos_pert}/{len(results_4b)} completed (EOS)")
    if "14b_4bit" in ckpt:
        results_14b = ckpt["14b_4bit"]
        eos_14b = sum(1 for r in results_14b.values() if r["baseline"]["terminated_by_eos"])
        print(f"14B baseline: {eos_14b}/{len(results_14b)} completed (EOS)")

    print(f"\nRun LLM-as-judge evaluation on: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
