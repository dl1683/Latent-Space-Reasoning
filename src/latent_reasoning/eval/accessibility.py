"""
Accessibility-first evaluation helpers.

This module summarizes comparison outputs into quality-vs-cost metrics that can
be tracked over time for low-resource, low-cost decision making.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


def _numeric_values(results: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for result in results:
        value = result.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def load_compare_results(paths: list[str | Path]) -> list[dict[str, Any]]:
    """
    Load one or more compare result files.

    Each file can contain either a single comparison dict or a list of dicts.
    """
    loaded: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        # Use utf-8-sig to tolerate Windows-authored JSON files with BOM.
        with path.open(encoding="utf-8-sig") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            loaded.append(payload)
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    loaded.append(item)
        else:
            raise ValueError(f"Unsupported JSON payload in {path}: expected dict or list")

    return loaded


def summarize_compare_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarize compare outputs into aggregate quality/cost metrics.

    Scores in this project are in [-1, 1]. We convert to [0, 1] with:
    normalized_quality = (score + 1) / 2
    """
    if not results:
        raise ValueError("No compare results provided")

    scores = _numeric_values(results, "latent_score")
    baseline_times = _numeric_values(results, "baseline_duration_s")
    latent_times = _numeric_values(results, "latent_duration_s")
    latent_run_times = _numeric_values(results, "latent_run_duration_s")
    latent_evolution_times = _numeric_values(results, "latent_evolution_duration_s")
    latent_non_evolution_times = _numeric_values(results, "latent_non_evolution_duration_s")
    overhead_ratios = _numeric_values(results, "latency_overhead_ratio")
    evaluations = _numeric_values(results, "evaluations")
    generations = _numeric_values(results, "generations")

    baseline_lengths = [
        float(len(result.get("baseline", "")))
        for result in results
        if isinstance(result.get("baseline"), str)
    ]
    latent_lengths = [
        float(len(result.get("latent_reasoning", "")))
        for result in results
        if isinstance(result.get("latent_reasoning"), str)
    ]

    # Per-run compute-efficiency proxy: evaluations required per normalized quality point.
    # Lower is better.
    evaluations_per_quality: list[float] = []
    for result in results:
        evals = result.get("evaluations")
        score = result.get("latent_score")
        if isinstance(evals, int | float) and isinstance(score, int | float):
            normalized_quality = (float(score) + 1.0) / 2.0
            if normalized_quality > 0:
                evaluations_per_quality.append(float(evals) / normalized_quality)

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_runs": len(results),
        "avg_latent_score": mean(scores) if scores else None,
        "median_latent_score": median(scores) if scores else None,
        "avg_baseline_duration_s": mean(baseline_times) if baseline_times else None,
        "avg_latent_duration_s": mean(latent_times) if latent_times else None,
        "avg_latent_run_duration_s": mean(latent_run_times) if latent_run_times else None,
        "avg_latent_evolution_duration_s": (
            mean(latent_evolution_times) if latent_evolution_times else None
        ),
        "avg_latent_non_evolution_duration_s": (
            mean(latent_non_evolution_times) if latent_non_evolution_times else None
        ),
        "avg_latency_overhead_ratio": mean(overhead_ratios) if overhead_ratios else None,
        "avg_evaluations": mean(evaluations) if evaluations else None,
        "avg_generations": mean(generations) if generations else None,
        "avg_baseline_length_chars": mean(baseline_lengths) if baseline_lengths else None,
        "avg_latent_length_chars": mean(latent_lengths) if latent_lengths else None,
        "avg_evaluations_per_quality": (
            mean(evaluations_per_quality) if evaluations_per_quality else None
        ),
    }
    return summary
