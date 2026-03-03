"""Run conditioning comparison across multiple models sequentially.

Each model runs the same 20 diverse questions under 3 conditions:
- Pure model (no conditioning)
- Soft prompt (latent projected through W)
- RNG seed (old method)

Models are tested from smallest to largest to fail fast on issues.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import time
from pathlib import Path

# Models to test: varying sizes and architectures
# Each: (hf_id, quantization, description)
MODELS = [
    ("Qwen/Qwen3-0.6B", "none", "0.6B transformer, FP16"),
    ("Qwen/Qwen3-4B", "4bit", "4B transformer, Q4 (current default)"),
    ("Qwen/Qwen3-8B", "4bit", "8B transformer, Q4"),
    ("Qwen/Qwen3-14B", "4bit", "14B transformer, Q4"),
]

SCRIPT = str(Path(__file__).parent / "run_conditioning_comparison.py")


def main():
    print("=" * 70)
    print("MULTI-MODEL CONDITIONING COMPARISON")
    print(f"Models: {len(MODELS)}")
    for hf_id, quant, desc in MODELS:
        print(f"  - {hf_id} ({quant}): {desc}")
    print("=" * 70)

    results_dir = Path(__file__).parent
    completed = []
    failed = []

    for i, (hf_id, quant, desc) in enumerate(MODELS):
        print(f"\n{'#' * 70}")
        print(f"# MODEL {i+1}/{len(MODELS)}: {hf_id} ({quant})")
        print(f"# {desc}")
        print(f"{'#' * 70}")

        model_short = hf_id.split("/")[-1].lower().replace("-", "_")
        out_path = results_dir / f"conditioning_comparison_{model_short}.json"

        if out_path.exists():
            print(f"\nSKIPPING: {hf_id} (results already exist at {out_path})")
            completed.append((hf_id, 0))
            continue

        cmd = [
            sys.executable, "-u", SCRIPT,
            "--model", hf_id,
            "--quantization", quant,
            "--output", str(out_path),
        ]

        start = time.time()
        try:
            result = subprocess.run(
                cmd, timeout=7200,  # 2h max per model
                capture_output=False,  # Stream output directly
            )
            elapsed = time.time() - start

            if result.returncode == 0:
                print(f"\nSUCCESS: {hf_id} in {elapsed/60:.1f} min")
                print(f"  Results: {out_path}")
                completed.append((hf_id, elapsed))
            else:
                print(f"\nFAILED: {hf_id} (exit code {result.returncode})")
                failed.append((hf_id, f"exit code {result.returncode}"))
        except subprocess.TimeoutExpired:
            print(f"\nTIMEOUT: {hf_id} (exceeded 2h)")
            failed.append((hf_id, "timeout"))
        except Exception as e:
            print(f"\nERROR: {hf_id}: {e}")
            failed.append((hf_id, str(e)))

        # Force GPU cleanup between models
        gc.collect()

    # Final summary
    print(f"\n{'=' * 70}")
    print("MULTI-MODEL SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nCompleted ({len(completed)}):")
    for hf_id, elapsed in completed:
        print(f"  {hf_id}: {elapsed/60:.1f} min")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for hf_id, reason in failed:
            print(f"  {hf_id}: {reason}")

    print(f"\nResult files in: {results_dir}")


if __name__ == "__main__":
    main()
