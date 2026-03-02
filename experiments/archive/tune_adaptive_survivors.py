"""Sweep adaptive-survivor hyperparameters for low-resource benchmarks."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure local experiments scripts are importable when run as a script.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_adaptive_survivors import (
    DEFAULT_QUERIES,
    _build_base_config,
    _build_comparison,
    run_paired_benchmark,
)


def _objective(comparison: dict) -> float:
    quality_delta = comparison.get("quality_delta_adaptive_minus_fixed") or 0.0
    eval_gain = comparison.get("evaluation_reduction_ratio") or 0.0

    quality_penalty = 0.0
    if quality_delta < 0:
        quality_penalty = abs(quality_delta) * 10.0

    # Tune primarily on stable signals; wall-clock latency stays diagnostic.
    quality_bonus = max(0.0, quality_delta) * 0.25
    return (2.5 * eval_gain) + quality_bonus - quality_penalty


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune adaptive survivor settings with a lightweight benchmark sweep."
    )
    parser.add_argument("--profile", default="configs/aim_v1_low_resource.yaml")
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-gpt2")
    parser.add_argument("--chains", type=int, default=3)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable per-trial warmup run before measured queries",
    )
    parser.add_argument("--output", default="experiments/aim_v1_adaptive_tuning.json")
    args = parser.parse_args()

    sweep = [
        {"min_survivors": 1, "survivor_decay": 0.50, "survivor_decay_patience": 1},
        {"min_survivors": 2, "survivor_decay": 0.50, "survivor_decay_patience": 1},
        {"min_survivors": 2, "survivor_decay": 0.75, "survivor_decay_patience": 1},
        {"min_survivors": 2, "survivor_decay": 0.75, "survivor_decay_patience": 2},
    ]

    base = _build_base_config(
        profile=Path(args.profile),
        model=args.model,
        max_tokens=args.max_tokens,
        generations=args.generations,
        chains=args.chains,
        score_cache=False,
    )

    fixed_cfg = deepcopy(base)
    fixed_cfg.evolution.selection.adaptive_survivors = False

    rows = []
    reference_fixed = None
    for params in sweep:
        adaptive_cfg = deepcopy(base)
        adaptive_cfg.evolution.selection.adaptive_survivors = True
        adaptive_cfg.evolution.selection.min_survivors = params["min_survivors"]
        adaptive_cfg.evolution.selection.survivor_decay = params["survivor_decay"]
        adaptive_cfg.evolution.selection.survivor_decay_patience = params["survivor_decay_patience"]

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fixed_candidate, adaptive, _ = run_paired_benchmark(
                fixed_cfg=fixed_cfg,
                adaptive_cfg=adaptive_cfg,
                queries=DEFAULT_QUERIES,
                seed_base=args.seed_base,
                repeats=max(1, args.repeats),
                warmup=not args.no_warmup,
            )

        if reference_fixed is None:
            reference_fixed = fixed_candidate

        comparison = _build_comparison(reference_fixed, adaptive)
        rows.append(
            {
                "params": params,
                "comparison": comparison,
                "fixed_summary": reference_fixed.summary,
                "adaptive_summary": adaptive.summary,
                "objective": _objective(comparison),
            }
        )

    rows.sort(key=lambda row: row["objective"], reverse=True)
    best = rows[0] if rows else None

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "chains": args.chains,
        "generations": args.generations,
        "max_tokens": args.max_tokens,
        "repeats": max(1, args.repeats),
        "warmup": not args.no_warmup,
        "fixed_summary": reference_fixed.summary if reference_fixed is not None else None,
        "candidates": rows,
        "best": best,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote tuning results: {output_path}")
    if best:
        print(f"Best params: {best['params']}")
        print(f"Best comparison: {best['comparison']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
