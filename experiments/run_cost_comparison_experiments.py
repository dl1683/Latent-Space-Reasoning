"""Cost comparison experiments: scaling ladder + temperature head-to-head.

Runs two experiments for the cost analysis article:

1. SCALING LADDER: Qwen3-14B and Qwen3-32B baseline on the same 25 sweet_spot
   arithmetic tasks. Determines the crossover point where perturbation×10 on 4B
   beats raw parameter scaling.

2. TEMPERATURE HEAD-TO-HEAD: Temperature sampling at t=0.3, 0.6, 0.9 on Qwen3-4B
   with the same 25 tasks and 10 seeds each. Determines whether token-level
   diversity (temperature) achieves the same gains as embedding-level perturbation.

Both reuse the existing baseline from the t2 n=10 results to save time.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
VENV_PYTHON = str(Path(__file__).parent.parent / ".venv" / "Scripts" / "python.exe")
RUNNER = str(EXPERIMENTS_DIR / "run_latent_sensitivity.py")

EXISTING_4B_BASELINE = str(
    EXPERIMENTS_DIR / "sensitivity_sweet_spot_random_noise_t2_results.json"
)


def run_cmd(args: list[str], label: str) -> int:
    print(f"\n{'=' * 70}")
    print(f"STARTING: {label}")
    print(f"Command: {' '.join(args)}")
    print(f"{'=' * 70}\n", flush=True)

    t0 = time.time()
    result = subprocess.run(args, cwd=str(EXPERIMENTS_DIR))
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"FINISHED: {label} ({elapsed / 60:.1f} min, exit={result.returncode})")
    print(f"{'=' * 70}\n", flush=True)
    return result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cost comparison experiments")
    parser.add_argument(
        "--experiment", default="all",
        choices=["all", "ladder", "temperature"],
        help="Which experiment to run: ladder (14B+32B baselines), "
             "temperature (t=0.3/0.6/0.9 on 4B), or all"
    )
    parser.add_argument(
        "--skip-32b", action="store_true",
        help="Skip 32B in scaling ladder (saves ~20GB VRAM)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("COST COMPARISON EXPERIMENTS")
    print("=" * 70)
    print(f"Experiment: {args.experiment}")
    print(f"Runner: {RUNNER}")
    print(f"Baseline reuse: {EXISTING_4B_BASELINE}")
    print()

    # ================================================================
    # EXPERIMENT 1: SCALING LADDER
    # ================================================================
    if args.experiment in ("all", "ladder"):
        # --- 14B baseline ---
        run_cmd([
            VENV_PYTHON, RUNNER,
            "--model", "Qwen/Qwen3-14B",
            "--quantization", "4bit",
            "--task-type", "nested",
            "--difficulty", "sweet_spot",
            "--n-tasks", "25",
            "--n-latents", "1",
            "--control-mode", "random_noise",
            "--num-soft-tokens", "2",
            "--max-new-tokens", "1024",
            "--output", str(EXPERIMENTS_DIR / "scaling_ladder_14b_4bit_baseline.json"),
        ], "Qwen3-14B 4-bit baseline (25 sweet_spot tasks)")

        gc.collect()
        if hasattr(__builtins__, '__import__'):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- 32B baseline ---
        if not args.skip_32b:
            run_cmd([
                VENV_PYTHON, RUNNER,
                "--model", "Qwen/Qwen3-32B",
                "--quantization", "4bit",
                "--task-type", "nested",
                "--difficulty", "sweet_spot",
                "--n-tasks", "25",
                "--n-latents", "1",
                "--control-mode", "random_noise",
                "--num-soft-tokens", "2",
                "--max-new-tokens", "1024",
                "--output", str(EXPERIMENTS_DIR / "scaling_ladder_32b_4bit_baseline.json"),
            ], "Qwen3-32B 4-bit baseline (25 sweet_spot tasks)")

            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ================================================================
    # EXPERIMENT 2: TEMPERATURE HEAD-TO-HEAD
    # ================================================================
    if args.experiment in ("all", "temperature"):
        # The runner uses temperature=0.0 (greedy) by default.
        # For temperature sampling, we need to modify the approach.
        # The runner's decode_with_raw_soft_prompt uses temperature=0.0 hardcoded.
        # So we run a dedicated temperature comparison script.
        run_temperature_comparison()


def run_temperature_comparison():
    """Run temperature sampling comparison on 4B with 10 seeds."""
    print("\n" + "=" * 70)
    print("TEMPERATURE HEAD-TO-HEAD COMPARISON")
    print("=" * 70)

    sys.path.insert(0, str(EXPERIMENTS_DIR))
    sys.path.insert(0, str(EXPERIMENTS_DIR.parent / "src"))

    import torch
    from run_latent_sensitivity import (
        generate_nested_tasks,
        run_zero_shot,
        extract_answer,
        verify_answer,
        safe_print,
        auto_calibrate,
    )
    from latent_reasoning.core.encoder import LLMEncoder

    tasks = generate_nested_tasks(n_tasks=25, difficulty_filter="sweet_spot")
    print(f"Tasks: {len(tasks)} sweet_spot nested arithmetic")

    # Load 4B model
    print("\nLoading Qwen3-4B 4-bit...")
    encoder = LLMEncoder(model_name="Qwen/Qwen3-4B", quantization="4bit")
    cal = auto_calibrate(encoder)
    print(f"Calibration: embed_dim={cal['embed_dim']}, rms={cal['embedding_rms']:.5f}")

    # Load existing baseline + perturbation results for comparison
    with open(EXISTING_4B_BASELINE) as f:
        existing = json.load(f)
    print(f"Loaded existing baseline: {existing['baseline_accuracy']:.0%}")
    print(f"Loaded existing perturbation mean: {existing['mean_accuracy']:.0%}")

    temperatures = [0.3, 0.6, 0.9]
    n_seeds = 10
    results = {}

    for temp in temperatures:
        print(f"\n{'=' * 50}")
        print(f"TEMPERATURE = {temp}, {n_seeds} seeds")
        print(f"{'=' * 50}")

        seed_results = []
        for seed_idx in range(n_seeds):
            torch.manual_seed(42 + seed_idx * 7)
            task_results = []
            for ti, task in enumerate(tasks):
                t0 = time.time()
                # Use the same generation infrastructure but with temperature
                system_msg = "Answer to the best of your ability."
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": task.prompt},
                ]
                try:
                    formatted = encoder.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    formatted = (
                        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                        f"<|im_start|>user\n{task.prompt}<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )

                inputs = encoder.tokenizer(formatted, return_tensors="pt")
                inputs = {k: v.to(encoder._device) for k, v in inputs.items()}

                with torch.no_grad():
                    out = encoder.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=temp,
                        pad_token_id=encoder.tokenizer.pad_token_id,
                        repetition_penalty=1.2,
                    )

                n_prompt = inputs["input_ids"].shape[1]
                n_generated = out[0].shape[0] - n_prompt
                eos_id = encoder.tokenizer.eos_token_id
                terminated_by_eos = bool(
                    eos_id is not None and out[0][-1].item() == eos_id
                )

                resp = encoder.tokenizer.decode(out[0], skip_special_tokens=True).strip()
                elapsed = time.time() - t0
                correct = verify_answer(resp, task.correct_answer)
                extracted = extract_answer(resp)
                tps = n_generated / elapsed if elapsed > 0 else 0

                task_results.append({
                    "task_id": task.task_id,
                    "correct_answer": task.correct_answer,
                    "correct": correct,
                    "extracted_answer": extracted,
                    "generated_tokens": n_generated,
                    "terminated_by_eos": terminated_by_eos,
                    "time": round(elapsed, 1),
                    "tokens_per_sec": round(tps, 1),
                })

            acc = sum(1 for r in task_results if r["correct"]) / len(task_results)
            n_correct = sum(1 for r in task_results if r["correct"])
            print(f"  Seed {seed_idx}: {acc:.0%} ({n_correct}/{len(tasks)})")

            seed_results.append({
                "seed_idx": seed_idx,
                "accuracy": acc,
                "n_correct": n_correct,
                "task_results": task_results,
            })

            gc.collect()
            torch.cuda.empty_cache()

        # Compute statistics for this temperature
        accs = [s["accuracy"] for s in seed_results]
        import numpy as np
        accs_np = np.array(accs)

        # Plurality vote
        from collections import Counter
        plurality_correct = 0
        for ti, task in enumerate(tasks):
            answers = []
            for sr in seed_results:
                ext = sr["task_results"][ti]["extracted_answer"]
                if ext is not None:
                    answers.append(ext)
            if answers:
                most_common = Counter(answers).most_common(1)[0][0]
                if most_common == task.correct_answer:
                    plurality_correct += 1
        plurality_acc = plurality_correct / len(tasks)

        # Oracle (any seed correct)
        oracle_correct = 0
        for ti, task in enumerate(tasks):
            if any(sr["task_results"][ti]["correct"] for sr in seed_results):
                oracle_correct += 1
        oracle_acc = oracle_correct / len(tasks)

        results[f"temp_{temp}"] = {
            "temperature": temp,
            "n_seeds": n_seeds,
            "mean_accuracy": float(accs_np.mean()),
            "std_accuracy": float(accs_np.std()),
            "min_accuracy": float(accs_np.min()),
            "max_accuracy": float(accs_np.max()),
            "plurality_accuracy": plurality_acc,
            "oracle_accuracy": oracle_acc,
            "seed_results": seed_results,
        }

        print(f"\n  Temperature {temp} summary:")
        print(f"    Mean: {accs_np.mean():.1%} +/- {accs_np.std():.1%}")
        print(f"    Plurality@{n_seeds}: {plurality_acc:.0%}")
        print(f"    Oracle@{n_seeds}: {oracle_acc:.0%}")

    # Save
    output = {
        "experiment": "temperature_vs_perturbation",
        "model": "Qwen/Qwen3-4B",
        "quantization": "4bit",
        "n_tasks": len(tasks),
        "n_seeds": n_seeds,
        "max_new_tokens": 1024,
        "reference": {
            "perturbation_baseline": existing["baseline_accuracy"],
            "perturbation_mean": existing["mean_accuracy"],
            "perturbation_plurality": None,  # compute from existing
            "perturbation_oracle": None,
        },
        "temperature_results": results,
    }

    # Compute perturbation plurality/oracle from existing data
    pert_plurality = 0
    pert_oracle = 0
    for ti, task in enumerate(tasks):
        answers = []
        any_correct = False
        for sr in existing["sensitivity_results"]:
            tr = sr["task_results"][ti]
            ext = tr.get("extracted_answer")
            if ext is not None:
                answers.append(ext)
            if tr["correct"]:
                any_correct = True
        if answers:
            from collections import Counter
            most_common = Counter(answers).most_common(1)[0][0]
            if most_common == task.correct_answer:
                pert_plurality += 1
        if any_correct:
            pert_oracle += 1

    output["reference"]["perturbation_plurality"] = pert_plurality / len(tasks)
    output["reference"]["perturbation_oracle"] = pert_oracle / len(tasks)

    out_path = EXPERIMENTS_DIR / "temperature_vs_perturbation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Summary comparison
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON: Temperature vs Perturbation")
    print(f"{'=' * 70}")
    print(f"{'Method':<30} {'Mean':>8} {'Plural':>8} {'Oracle':>8}")
    print("-" * 56)
    print(f"{'Greedy baseline':<30} {existing['baseline_accuracy']:>7.0%} {'--':>8} {'--':>8}")
    print(f"{'Perturbation t2 x10':<30} {existing['mean_accuracy']:>7.0%} "
          f"{output['reference']['perturbation_plurality']:>7.0%} "
          f"{output['reference']['perturbation_oracle']:>7.0%}")
    for key, res in results.items():
        label = f"Temperature {res['temperature']} x{n_seeds}"
        print(f"{label:<30} {res['mean_accuracy']:>7.0%} "
              f"{res['plurality_accuracy']:>7.0%} "
              f"{res['oracle_accuracy']:>7.0%}")

    del encoder
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
