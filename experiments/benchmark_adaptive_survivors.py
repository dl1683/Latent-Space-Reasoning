"""Benchmark adaptive survivor budget vs fixed survivor budget."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from latent_reasoning.config import Config, ScorerConfig
from latent_reasoning.engine import Engine
from latent_reasoning.eval import summarize_compare_results


DEFAULT_QUERIES = [
    "List 3 practical steps to reduce API latency for a small startup.",
    "Design a minimal monitoring plan for a low-budget backend service.",
    "How can I make model inference cheaper without losing too much quality?",
    "Give a transparent and auditable rollout plan for a new AI feature.",
    "What are lightweight safeguards for prompt-injection in production?",
    "How should I prioritize reliability work with a very small engineering team?",
]


@dataclass
class BenchmarkModeResult:
    name: str
    runs: list[dict]
    summary: dict
    trial_summaries: list[dict] = field(default_factory=list)


def _as_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _numeric_values(results: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for result in results:
        value = _as_float(result.get(key))
        if value is not None:
            values.append(value)
    return values


def _add_mode_extras(summary: dict, runs: list[dict]) -> None:
    """Add metrics not covered by summarize_compare_results."""
    total_compare = _numeric_values(runs, "total_compare_duration_s")
    run_times = _numeric_values(runs, "latent_run_duration_s")
    evolution_times = _numeric_values(runs, "latent_evolution_duration_s")
    non_evolution_times = _numeric_values(runs, "latent_non_evolution_duration_s")
    evals = _numeric_values(runs, "evaluations")

    summary["avg_total_compare_duration_s"] = mean(total_compare) if total_compare else None
    summary["avg_latent_run_duration_s"] = mean(run_times) if run_times else None
    summary["avg_latent_evolution_duration_s"] = mean(evolution_times) if evolution_times else None
    summary["avg_latent_non_evolution_duration_s"] = (
        mean(non_evolution_times) if non_evolution_times else None
    )

    if evals and evolution_times:
        total_evals = sum(evals)
        summary["avg_evolution_time_per_eval_s"] = (
            sum(evolution_times) / total_evals if total_evals > 0 else None
        )
    else:
        summary["avg_evolution_time_per_eval_s"] = None


def _add_trial_aggregates(summary: dict, trial_summaries: list[dict]) -> None:
    metrics = [
        "avg_latent_score",
        "avg_evaluations",
        "avg_latent_duration_s",
        "avg_total_compare_duration_s",
        "avg_latent_evolution_duration_s",
        "avg_latent_non_evolution_duration_s",
        "avg_evaluations_per_quality",
        "avg_evolution_time_per_eval_s",
    ]
    for metric in metrics:
        values: list[float] = []
        for trial_summary in trial_summaries:
            value = _as_float(trial_summary.get(metric))
            if value is not None:
                values.append(value)
        if values:
            summary[f"mean_trial_{metric}"] = mean(values)
            summary[f"median_trial_{metric}"] = median(values)
        else:
            summary[f"mean_trial_{metric}"] = None
            summary[f"median_trial_{metric}"] = None


def _build_base_config(
    profile: Path,
    model: str,
    max_tokens: int,
    generations: int,
    chains: int,
    score_cache: bool,
) -> Config:
    cfg = Config.from_yaml(profile)
    cfg.encoder.model = model
    cfg.encoder.device = "cpu"
    cfg.encoder.quantization = "none"
    cfg.synthesis.max_tokens = max_tokens
    cfg.synthesis.temperature = 0.0
    cfg.evolution.generations = generations
    cfg.evolution.chains = chains
    cfg.evolution.score_cache = score_cache
    cfg.budget.max_evaluations = max(cfg.budget.max_evaluations, chains * generations * 3)
    cfg.output.verbosity = "silent"
    # Disable checkpoint/history writes for lower-noise latency measurements.
    cfg.output.save_history = False

    # Use local trained scorer to avoid extra model downloads during benchmarking.
    cfg.judges.scorers = [
        ScorerConfig(
            type="trained_latent",
            checkpoint="checkpoints/latent_scorer/final_model.pt",
            latent_dim=cfg.encoder.latent_dim,
        )
    ]
    cfg.judges.modifiers = []
    return cfg


def _run_mode(
    mode_name: str,
    cfg: Config,
    queries: list[str],
    seed_base: int,
    trial_index: int = 0,
    warmup_query: str | None = None,
) -> BenchmarkModeResult:
    engine = Engine(config=cfg)
    runs: list[dict] = []

    # Exclude one startup run from metrics to reduce model init noise.
    if warmup_query:
        torch.manual_seed(seed_base - 1)
        engine.compare(warmup_query)

    for idx, query in enumerate(queries):
        # Force deterministic mutation/generation stochasticity per query.
        seed = seed_base + idx
        torch.manual_seed(seed)

        result = engine.compare(query)
        result["mode"] = mode_name
        result["trial_index"] = trial_index
        result["query_index"] = idx
        result["seed"] = seed
        runs.append(result)

    summary = summarize_compare_results(runs)
    _add_mode_extras(summary, runs)
    summary["total_evaluations"] = sum(float(r["evaluations"]) for r in runs)
    summary["mean_normalized_quality"] = mean((float(r["latent_score"]) + 1.0) / 2.0 for r in runs)
    return BenchmarkModeResult(
        name=mode_name,
        runs=runs,
        summary=summary,
        trial_summaries=[summary],
    )


def run_paired_benchmark(
    fixed_cfg: Config,
    adaptive_cfg: Config,
    queries: list[str],
    seed_base: int,
    repeats: int,
    warmup: bool,
) -> tuple[BenchmarkModeResult, BenchmarkModeResult, list[dict]]:
    """Run paired fixed/adaptive trials with counterbalanced execution order."""
    fixed_trials: list[BenchmarkModeResult] = []
    adaptive_trials: list[BenchmarkModeResult] = []
    trial_orders: list[dict] = []

    for trial_idx in range(repeats):
        trial_seed_base = seed_base + (trial_idx * 10_000)
        warmup_query = queries[0] if warmup and queries else None

        if trial_idx % 2 == 0:
            mode_order = [
                ("fixed", fixed_cfg, fixed_trials),
                ("adaptive", adaptive_cfg, adaptive_trials),
            ]
        else:
            mode_order = [
                ("adaptive", adaptive_cfg, adaptive_trials),
                ("fixed", fixed_cfg, fixed_trials),
            ]

        trial_orders.append(
            {
                "trial_index": trial_idx,
                "seed_base": trial_seed_base,
                "mode_order": [item[0] for item in mode_order],
            }
        )

        for mode_name, cfg, target_trials in mode_order:
            trial_result = _run_mode(
                mode_name=mode_name,
                cfg=cfg,
                queries=queries,
                seed_base=trial_seed_base,
                trial_index=trial_idx,
                warmup_query=warmup_query,
            )
            target_trials.append(trial_result)

    def _aggregate(name: str, trials: list[BenchmarkModeResult]) -> BenchmarkModeResult:
        all_runs: list[dict] = []
        trial_summaries: list[dict] = []
        for trial in trials:
            all_runs.extend(trial.runs)
            trial_summaries.append(trial.summary)

        summary = summarize_compare_results(all_runs)
        _add_mode_extras(summary, all_runs)
        summary["num_trials"] = len(trials)
        summary["runs_per_trial"] = len(queries)
        summary["total_evaluations"] = sum(float(r["evaluations"]) for r in all_runs)
        summary["mean_normalized_quality"] = mean(
            (float(r["latent_score"]) + 1.0) / 2.0 for r in all_runs
        )
        _add_trial_aggregates(summary, trial_summaries)

        return BenchmarkModeResult(
            name=name,
            runs=all_runs,
            summary=summary,
            trial_summaries=trial_summaries,
        )

    return _aggregate("fixed", fixed_trials), _aggregate("adaptive", adaptive_trials), trial_orders


def _summary_value(summary: dict, metric: str) -> float | None:
    """Prefer robust median-over-trials metric when available."""
    for key in (f"median_trial_{metric}", f"mean_trial_{metric}", metric):
        value = _as_float(summary.get(key))
        if value is not None:
            return value
    return None


def _build_comparison(fixed: BenchmarkModeResult, adaptive: BenchmarkModeResult) -> dict:
    fixed_score = _summary_value(fixed.summary, "avg_latent_score")
    adaptive_score = _summary_value(adaptive.summary, "avg_latent_score")
    fixed_evals = _summary_value(fixed.summary, "avg_evaluations")
    adaptive_evals = _summary_value(adaptive.summary, "avg_evaluations")
    fixed_latency = _summary_value(fixed.summary, "avg_latent_duration_s")
    adaptive_latency = _summary_value(adaptive.summary, "avg_latent_duration_s")
    fixed_evolution_latency = _summary_value(fixed.summary, "avg_latent_evolution_duration_s")
    adaptive_evolution_latency = _summary_value(adaptive.summary, "avg_latent_evolution_duration_s")
    fixed_evolution_per_eval = _summary_value(fixed.summary, "avg_evolution_time_per_eval_s")
    adaptive_evolution_per_eval = _summary_value(adaptive.summary, "avg_evolution_time_per_eval_s")

    quality_delta = None
    if fixed_score is not None and adaptive_score is not None:
        quality_delta = adaptive_score - fixed_score

    eval_reduction_ratio = None
    if fixed_evals not in (None, 0) and adaptive_evals is not None:
        eval_reduction_ratio = (fixed_evals - adaptive_evals) / fixed_evals

    latency_reduction_ratio = None
    if fixed_latency not in (None, 0) and adaptive_latency is not None:
        latency_reduction_ratio = (fixed_latency - adaptive_latency) / fixed_latency

    evolution_latency_reduction_ratio = None
    if (
        fixed_evolution_latency not in (None, 0)
        and adaptive_evolution_latency is not None
    ):
        evolution_latency_reduction_ratio = (
            fixed_evolution_latency - adaptive_evolution_latency
        ) / fixed_evolution_latency

    evolution_time_per_eval_reduction_ratio = None
    if (
        fixed_evolution_per_eval not in (None, 0)
        and adaptive_evolution_per_eval is not None
    ):
        evolution_time_per_eval_reduction_ratio = (
            fixed_evolution_per_eval - adaptive_evolution_per_eval
        ) / fixed_evolution_per_eval

    return {
        "quality_delta_adaptive_minus_fixed": quality_delta,
        "evaluation_reduction_ratio": eval_reduction_ratio,
        "latency_reduction_ratio": latency_reduction_ratio,
        "evolution_latency_reduction_ratio": evolution_latency_reduction_ratio,
        "evolution_time_per_eval_reduction_ratio": evolution_time_per_eval_reduction_ratio,
        "metric_basis": "median_trial_mean_with_fallback",
    }


def _to_markdown(
    output: dict,
    fixed: BenchmarkModeResult,
    adaptive: BenchmarkModeResult,
) -> str:
    comparison = output["comparison"]
    lines = [
        "# AIM-v1 Adaptive Survivor Benchmark",
        "",
        f"- Generated: `{output['generated_at_utc']}`",
        f"- Model: `{output['model']}`",
        f"- Queries: `{output['num_queries']}`",
        f"- Repeats: `{output['repeats']}`",
        f"- Warmup per trial: `{output['warmup']}`",
        "",
        "## Fixed vs Adaptive",
        "",
        "| Metric | Fixed | Adaptive |",
        "|---|---:|---:|",
        f"| Avg latent score | {fixed.summary.get('avg_latent_score')} | {adaptive.summary.get('avg_latent_score')} |",
        f"| Median trial avg latent score | {fixed.summary.get('median_trial_avg_latent_score')} | {adaptive.summary.get('median_trial_avg_latent_score')} |",
        f"| Avg evaluations | {fixed.summary.get('avg_evaluations')} | {adaptive.summary.get('avg_evaluations')} |",
        f"| Median trial avg evaluations | {fixed.summary.get('median_trial_avg_evaluations')} | {adaptive.summary.get('median_trial_avg_evaluations')} |",
        f"| Avg latent duration (s) | {fixed.summary.get('avg_latent_duration_s')} | {adaptive.summary.get('avg_latent_duration_s')} |",
        f"| Median trial avg latent duration (s) | {fixed.summary.get('median_trial_avg_latent_duration_s')} | {adaptive.summary.get('median_trial_avg_latent_duration_s')} |",
        f"| Avg evolution duration (s) | {fixed.summary.get('avg_latent_evolution_duration_s')} | {adaptive.summary.get('avg_latent_evolution_duration_s')} |",
        f"| Median trial avg evolution duration (s) | {fixed.summary.get('median_trial_avg_latent_evolution_duration_s')} | {adaptive.summary.get('median_trial_avg_latent_evolution_duration_s')} |",
        f"| Avg evolution time/eval (s) | {fixed.summary.get('avg_evolution_time_per_eval_s')} | {adaptive.summary.get('avg_evolution_time_per_eval_s')} |",
        f"| Avg evaluations/quality | {fixed.summary.get('avg_evaluations_per_quality')} | {adaptive.summary.get('avg_evaluations_per_quality')} |",
        "",
        "## Deltas",
        "",
        f"- Quality delta (adaptive - fixed): `{comparison.get('quality_delta_adaptive_minus_fixed')}`",
        f"- Evaluation reduction ratio: `{comparison.get('evaluation_reduction_ratio')}`",
        f"- Latency reduction ratio: `{comparison.get('latency_reduction_ratio')}`",
        f"- Evolution latency reduction ratio: `{comparison.get('evolution_latency_reduction_ratio')}`",
        f"- Evolution time/eval reduction ratio: `{comparison.get('evolution_time_per_eval_reduction_ratio')}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark adaptive survivor budget against fixed survivors."
    )
    parser.add_argument(
        "--profile",
        default="configs/aim_v1_low_resource.yaml",
        help="Base config profile path",
    )
    parser.add_argument(
        "--model",
        default="hf-internal-testing/tiny-random-gpt2",
        help="Model id/path for low-resource benchmark runs",
    )
    parser.add_argument("--queries-file", default=None, help="Optional text file with one query per line")
    parser.add_argument("--chains", type=int, default=3, help="Evolution chains for both modes")
    parser.add_argument("--generations", type=int, default=4, help="Evolution generations for both modes")
    parser.add_argument("--max-tokens", type=int, default=96, help="Max decode tokens")
    parser.add_argument(
        "--score-cache",
        action="store_true",
        help="Enable evolution score-cache during benchmark runs",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Paired trial repeats")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable per-trial warmup run before measured queries",
    )
    parser.add_argument("--adaptive-min-survivors", type=int, default=1, help="Adaptive min_survivors")
    parser.add_argument("--adaptive-decay", type=float, default=0.5, help="Adaptive survivor_decay")
    parser.add_argument("--adaptive-patience", type=int, default=1, help="Adaptive survivor_decay_patience")
    parser.add_argument("--seed-base", type=int, default=1234, help="Base random seed")
    parser.add_argument(
        "--output-json",
        default="experiments/aim_v1_adaptive_survivor_benchmark.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--output-md",
        default="experiments/aim_v1_adaptive_survivor_benchmark.md",
        help="Output Markdown path",
    )
    args = parser.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries_file:
        queries_path = Path(args.queries_file)
        queries = [line.strip() for line in queries_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not queries:
            raise ValueError("queries-file produced no non-empty queries")

    base = _build_base_config(
        profile=Path(args.profile),
        model=args.model,
        max_tokens=args.max_tokens,
        generations=args.generations,
        chains=args.chains,
        score_cache=args.score_cache,
    )

    fixed_cfg = deepcopy(base)
    fixed_cfg.evolution.selection.adaptive_survivors = False

    adaptive_cfg = deepcopy(base)
    adaptive_cfg.evolution.selection.adaptive_survivors = True
    adaptive_cfg.evolution.selection.min_survivors = args.adaptive_min_survivors
    adaptive_cfg.evolution.selection.survivor_decay = args.adaptive_decay
    adaptive_cfg.evolution.selection.survivor_decay_patience = args.adaptive_patience

    fixed, adaptive, trial_orders = run_paired_benchmark(
        fixed_cfg=fixed_cfg,
        adaptive_cfg=adaptive_cfg,
        queries=queries,
        seed_base=args.seed_base,
        repeats=max(1, args.repeats),
        warmup=not args.no_warmup,
    )

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "model": args.model,
        "num_queries": len(queries),
        "queries": queries,
        "repeats": max(1, args.repeats),
        "warmup": not args.no_warmup,
        "trial_orders": trial_orders,
        "fixed": {
            "summary": fixed.summary,
            "trial_summaries": fixed.trial_summaries,
            "runs": fixed.runs,
        },
        "adaptive": {
            "summary": adaptive.summary,
            "trial_summaries": adaptive.trial_summaries,
            "runs": adaptive.runs,
        },
        "comparison": _build_comparison(fixed, adaptive),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_to_markdown(output, fixed, adaptive), encoding="utf-8")

    print(f"Wrote benchmark JSON: {output_json}")
    print(f"Wrote benchmark Markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
