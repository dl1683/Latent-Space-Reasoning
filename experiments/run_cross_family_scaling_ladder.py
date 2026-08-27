"""Cross-family scaling ladder: does the flat-scaling result survive off Qwen3?

The published ladder in README.md is confounded twice over. Every rung is a
Qwen3 model, and every rung is 4-bit quantized. Its most striking entry --
Qwen3-32B scoring 0% -- is not obviously a statement about parameter count at
all: the model produced verbose natural-language explanations, exhausted its
1024-token budget, and never stated an answer. That is the signature of a
degenerate decode, and 4-bit NF4 quantization is a plausible cause.

This script runs the same protocol on a model that shares neither confound:

  * ``google/gemma-4-31B-it`` at bfloat16 -- different family, no quantization,
    comparable parameter count (31B vs 32B).

and optionally on the missing control:

  * ``Qwen/Qwen3-32B`` at bfloat16 -- same model as the 0% rung, unquantized.
    This separates "parameter scaling is flat" from "4-bit broke the 32B".
    Not run by default: the weights are ~62 GB and are not in the local cache.

Protocol is held identical to the published ladder: 25 seeded nested-arithmetic
tasks, greedy decoding, 1024 new tokens, and perturbation via two random
embedding-space tokens scaled to the model's native embedding RMS
(``--control-mode random_noise --num-soft-tokens 2``).

``--difficulty`` selects the tier. ``sweet_spot`` is the published rung and is
the default, but both 31B-class models saturate it, so a perturbation
measurement needs ``frontier_nested``.

Beyond accuracy this records *why* a model fails: mean generated tokens, the
fraction of generations terminated by EOS, and the fraction from which any
integer could be extracted. Without those, "0%" cannot be distinguished from
"never finished talking".

Usage::

    python run_cross_family_scaling_ladder.py                  # Gemma arm only
    python run_cross_family_scaling_ladder.py --include-qwen32b-bf16
    python run_cross_family_scaling_ladder.py --difficulty frontier_nested
    python run_cross_family_scaling_ladder.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
RUNNER = str(EXPERIMENTS_DIR / "run_latent_sensitivity.py")

N_TASKS = 25
N_LATENTS = 10
MAX_NEW_TOKENS = 1024

ARMS = {
    "gemma4_31b_bf16": {
        "model": "google/gemma-4-31B-it",
        "quantization": "none",
        "dtype": "bfloat16",
        "note": "cross-family, unquantized 31B",
    },
    "qwen3_32b_bf16": {
        "model": "Qwen/Qwen3-32B",
        "quantization": "none",
        "dtype": "bfloat16",
        "note": "quantization control for the published 0% rung; ~62 GB download",
    },
}


def result_path(arm: str, difficulty: str) -> Path:
    return EXPERIMENTS_DIR / f"cross_family_ladder_{arm}_{difficulty}_results.json"


def resolved_revision(model_id: str) -> str | None:
    """The exact commit of the local cache this run will load.

    Worth recording: a model id is not a fixed object. Upstream can publish a
    new commit that changes the chat template -- and therefore the prompt the
    model actually sees -- without touching a single weight.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        from huggingface_hub.file_download import repo_folder_name

        ref = (
            Path(HF_HUB_CACHE)
            / repo_folder_name(repo_id=model_id, repo_type="model")
            / "refs" / "main"
        )
        return ref.read_text().strip() if ref.exists() else None
    except Exception:
        return None


def run_arm(arm: str, difficulty: str) -> int:
    spec = ARMS[arm]
    out = result_path(arm, difficulty)
    cmd = [
        sys.executable, RUNNER,
        "--model", spec["model"],
        "--quantization", spec["quantization"],
        "--dtype", spec["dtype"],
        "--task-type", "nested",
        "--difficulty", difficulty,
        "--n-tasks", str(N_TASKS),
        "--n-latents", str(N_LATENTS),
        "--control-mode", "random_noise",
        "--num-soft-tokens", "2",
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--output", str(out),
    ]
    print("=" * 70)
    print(f"ARM: {arm} @ {difficulty}  ({spec['note']})")
    print(" ".join(cmd))
    print("=" * 70, flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(EXPERIMENTS_DIR)).returncode
    print(f"\nARM {arm} finished in {(time.time() - t0) / 60:.1f} min (exit={rc})\n",
          flush=True)
    return rc


def _decode_diagnostics(records: list[dict]) -> dict:
    """Why did this arm score what it scored?"""
    if not records:
        return {}
    n = len(records)
    tokens = [r.get("generated_tokens") or 0 for r in records]
    return {
        "n_generations": n,
        "mean_generated_tokens": sum(tokens) / n,
        "frac_hit_token_cap": sum(t >= MAX_NEW_TOKENS for t in tokens) / n,
        "frac_terminated_by_eos": sum(bool(r.get("terminated_by_eos")) for r in records) / n,
        "frac_answer_extractable": sum(
            r.get("extracted_answer") is not None for r in records
        ) / n,
        "frac_closed_reasoning": sum(
            bool(r.get("closed_reasoning")) for r in records
        ) / n,
    }


def analyze(arm: str, difficulty: str) -> dict | None:
    path = result_path(arm, difficulty)
    if not path.exists():
        return None
    data = json.loads(path.read_text())

    baseline = data["baseline_results"]
    seeds = data["sensitivity_results"]
    tasks = [r["task_id"] for r in baseline]

    # Plurality@k: the modal extracted answer across seeds, ties broken by the
    # order Counter reports (first-seen). Tasks where no seed produced an
    # integer count as wrong rather than being dropped.
    plurality = 0
    oracle = 0
    for ti, task in enumerate(baseline):
        expected = task["correct_answer"]
        answers = [
            s["task_results"][ti]["extracted_answer"]
            for s in seeds
            if s["task_results"][ti]["extracted_answer"] is not None
        ]
        if answers and Counter(answers).most_common(1)[0][0] == expected:
            plurality += 1
        if any(s["task_results"][ti]["correct"] for s in seeds):
            oracle += 1

    accs = [s["accuracy"] for s in seeds]
    n_tasks = len(tasks)
    pert_records = [tr for s in seeds for tr in s["task_results"]]

    return {
        "arm": arm,
        "difficulty": difficulty,
        "model": data["model"],
        "revision": resolved_revision(data["model"]),
        "quantization": data["quantization"],
        "dtype": data.get("dtype"),
        "n_tasks": n_tasks,
        "n_seeds": len(seeds),
        "baseline_accuracy": data["baseline_accuracy"],
        "perturbation_mean_accuracy": sum(accs) / len(accs),
        "perturbation_min_accuracy": min(accs),
        "perturbation_max_accuracy": max(accs),
        "plurality_at_k": plurality / n_tasks,
        "oracle_at_k": oracle / n_tasks,
        "baseline_decode": _decode_diagnostics(baseline),
        "perturbation_decode": _decode_diagnostics(pert_records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-qwen32b-bf16", action="store_true",
                        help="also run the unquantized Qwen3-32B control "
                             "(~62 GB download if not cached)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="skip generation; summarize existing result files")
    parser.add_argument("--difficulty", default="sweet_spot",
                        help="nested-arithmetic tier. Use frontier_nested for "
                             "models that saturate brutal_nested.")
    args = parser.parse_args()

    arms = ["gemma4_31b_bf16"]
    if args.include_qwen32b_bf16:
        arms.append("qwen3_32b_bf16")

    if not args.analyze_only:
        for arm in arms:
            if run_arm(arm, args.difficulty) != 0:
                print(f"ARM {arm} failed; stopping.", file=sys.stderr)
                return 1

    summary = [s for s in (analyze(a, args.difficulty) for a in arms) if s is not None]
    out = EXPERIMENTS_DIR / f"cross_family_ladder_summary_{args.difficulty}.json"
    out.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 78)
    print(f"CROSS-FAMILY SCALING LADDER @ {args.difficulty}")
    print("=" * 78)
    hdr = f"{'arm':<20}{'base':>7}{'pert':>7}{'plur':>7}{'orac':>7}{'eos%':>7}{'ans%':>7}"
    print(hdr)
    print("-" * 78)
    for s in summary:
        print(f"{s['arm']:<20}"
              f"{s['baseline_accuracy']:>7.0%}"
              f"{s['perturbation_mean_accuracy']:>7.0%}"
              f"{s['plurality_at_k']:>7.0%}"
              f"{s['oracle_at_k']:>7.0%}"
              f"{s['perturbation_decode'].get('frac_terminated_by_eos', 0):>7.0%}"
              f"{s['perturbation_decode'].get('frac_answer_extractable', 0):>7.0%}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
