"""The scaling ladder, with and without the truncation artifact.

The published ladder in README.md reports Qwen3 accuracy falling as parameters
rise (1.7B 28%, 4B 32%, 8B 24%, 14B 36%, 32B 0%) and concludes that parameter
scaling is flat or worse on nested arithmetic.

Every rung of that ladder was truncated. Re-reading the stored result files:

    rung          accuracy   mean tokens   hit 1024 cap   terminated by EOS
    1.7B  4-bit        28%           990            80%                 20%
    4B    4-bit        32%           934            76%                 24%
    8B    4-bit        24%          1020            96%                  4%
    14B   4-bit        36%           983            76%                 24%
    32B   4-bit         0%          1024           100%                  0%

Three quarters of every rung never finished generating. These models enter Qwen3
thinking mode, and the score is then read off whatever text survived truncation
by taking its last integer. Larger models think longer, truncate more often, and
therefore score *worse* -- which is why the published curve is inverted rather
than flat. The 32B is simply the limiting case at 100% truncation.

Confirmed directly: Qwen3-32B at bfloat16 scores 4% with thinking on and
**100%** with thinking off, at the same 1024-token cap on the same 25 tasks.

This script measures both arms across the ladder so the artifact and the
underlying capability can be told apart:

  * ``think``   -- default template, thinking on. Reproduces the published
                   protocol on this hardware.
  * ``nothink`` -- ``--no-think``, same 1024-token cap. What the models can
                   actually do when not spending the budget on a reasoning
                   trace they never close.

Both arms run at bfloat16. Quantization is held constant and out of the picture
because it was already shown not to be the driver (32B: 0% at 4-bit vs 4% at
bfloat16 -- unchanged).

Usage::

    python run_true_scaling_ladder.py
    python run_true_scaling_ladder.py --arms nothink
    python run_true_scaling_ladder.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
RUNNER = str(EXPERIMENTS_DIR / "run_latent_sensitivity.py")

N_TASKS = 25
MAX_NEW_TOKENS = 1024
DIFFICULTY = "sweet_spot"

RUNGS = [
    ("qwen3_1_7b", "Qwen/Qwen3-1.7B", 1.7),
    ("qwen3_4b", "Qwen/Qwen3-4B", 4.0),
    ("qwen3_8b", "Qwen/Qwen3-8B", 8.0),
    ("qwen3_14b", "Qwen/Qwen3-14B", 14.0),
    ("qwen3_32b", "Qwen/Qwen3-32B", 32.0),
]

ARMS = ("think", "nothink")


def result_path(rung: str, arm: str) -> Path:
    return EXPERIMENTS_DIR / f"true_ladder_{rung}_{arm}_results.json"


def run_rung(rung: str, model: str, arm: str) -> int:
    out = result_path(rung, arm)
    cmd = [
        sys.executable, RUNNER,
        "--model", model,
        "--quantization", "none",
        "--dtype", "bfloat16",
        "--task-type", "nested",
        "--difficulty", DIFFICULTY,
        "--calibrate",
        "--n-calibrate", str(N_TASKS),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--output", str(out),
    ]
    if arm == "nothink":
        cmd.append("--no-think")

    print("=" * 70)
    print(f"RUNG: {rung} / {arm}")
    print(" ".join(cmd))
    print("=" * 70, flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(EXPERIMENTS_DIR)).returncode
    print(f"\n{rung}/{arm} done in {(time.time() - t0) / 60:.1f} min (exit={rc})\n",
          flush=True)
    return rc


def analyze(rung: str, arm: str) -> dict | None:
    path = result_path(rung, arm)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    rs = data["baseline_results"]
    n = len(rs)
    return {
        "rung": rung,
        "arm": arm,
        "model": data["model"],
        "dtype": data.get("dtype"),
        "accuracy": data["baseline_accuracy"],
        "mean_generated_tokens": sum(r["generated_tokens"] for r in rs) / n,
        "frac_hit_cap": sum(r["generated_tokens"] >= MAX_NEW_TOKENS for r in rs) / n,
        "frac_terminated_by_eos": sum(bool(r["terminated_by_eos"]) for r in rs) / n,
        "frac_closed_reasoning": sum(bool(r.get("closed_reasoning")) for r in rs) / n,
        "n_tasks": n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if not args.analyze_only:
        # nothink first: it is much cheaper (nothing hits the cap) and it is the
        # arm that carries the finding, so a later failure still leaves it done.
        for arm in sorted(args.arms, key=lambda a: a != "nothink"):
            for rung, model, _ in RUNGS:
                if result_path(rung, arm).exists():
                    print(f"skip {rung}/{arm} (already present)", flush=True)
                    continue
                if run_rung(rung, model, arm) != 0:
                    print(f"{rung}/{arm} FAILED; continuing", file=sys.stderr)

    summary = [
        s for s in (analyze(r, a) for a in ARMS for r, _, _ in RUNGS) if s is not None
    ]
    out = EXPERIMENTS_DIR / "true_ladder_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    by_key = {(s["rung"], s["arm"]): s for s in summary}
    print("\n" + "=" * 84)
    print("TRUE SCALING LADDER (25 sweet_spot tasks, bf16, 1024-token cap)")
    print("=" * 84)
    print(f"{'model':<14}{'params':>8}"
          f"{'think acc':>11}{'cap%':>7}"
          f"{'nothink acc':>13}{'cap%':>7}{'tokens':>9}")
    print("-" * 84)
    for rung, _, params in RUNGS:
        t = by_key.get((rung, "think"))
        nt = by_key.get((rung, "nothink"))
        row = f"{rung:<14}{params:>7.1f}B"
        row += f"{t['accuracy']:>11.0%}{t['frac_hit_cap']:>7.0%}" if t else f"{'-':>11}{'-':>7}"
        row += (f"{nt['accuracy']:>13.0%}{nt['frac_hit_cap']:>7.0%}"
                f"{nt['mean_generated_tokens']:>9.0f}") if nt else f"{'-':>13}{'-':>7}{'-':>9}"
        print(row)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
