"""Geometry-focused static ARC benchmark sweep utilities.

This script runs batched ARC sweeps with decode mode set to
``geometry_feedback`` and reports both utility metrics and geometry-trace
diagnostics.

Important: current ARC-AGI-3 is an official interactive agent benchmark, not
the static grid task format used here. Use ``run_arc3_official_harness.py`` for
the true ARC-AGI-3 harness. This script remains useful for cheap static ARC
proxy experiments while the interactive agent integration is developed.

It is intended to replace older ad-hoc static ARC sweep scripts with:

1) explicit geometry-parameter grid search
2) per-run geometry trace summaries
3) geometry-aware frontier extraction
4) CSV/JSON export for downstream analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_reasoning.eval.arc_agi2 import run_arc_evaluation


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    clean = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if not clean:
        return 0.0
    if q <= 0:
        return clean[0]
    if q >= 1:
        return clean[-1]
    index = q * (len(clean) - 1)
    left = int(index)
    right = min(left + 1, len(clean) - 1)
    weight = index - left
    return clean[left] * (1.0 - weight) + clean[right] * weight


def _parse_list(raw: str, cast_fn, *, min_len: int = 1) -> List[Any]:
    items = [chunk.strip() for chunk in (raw.split(",") if isinstance(raw, str) else []) if chunk.strip()]
    parsed = [cast_fn(chunk) for chunk in items]
    if min_len > 0 and not parsed:
        raise ValueError(f"Expected at least {min_len} values, got 0")
    return parsed


def _parse_arc_strategies(raw: str) -> List[str]:
    """Parse and validate ARC strategy names from comma-separated input."""
    valid = {"single", "adaptive", "repair", "consensus", "geometry_bandit", "self_improving"}
    items = [chunk.strip().lower() for chunk in (raw.split(",") if isinstance(raw, str) else []) if chunk.strip()]
    if not items:
        raise ValueError("Expected at least one strategy in --arc-strategies")

    normalized: List[str] = []
    for item in items:
        if item not in valid:
            raise ValueError(
                f"Unsupported ARC strategy '{item}'. Supported: single, adaptive, repair, "
                "consensus, geometry_bandit, self_improving"
            )
        normalized.append(item)
    return normalized


def _serialize_args_for_json(args: argparse.Namespace) -> Dict[str, Any]:
    raw = vars(args)
    return {key: str(value) if isinstance(value, Path) else value for key, value in raw.items()}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return maximum if value > maximum else minimum if value < minimum else value


def _dedupe_configs(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    unique_configs: List[Dict[str, Any]] = []
    for config in configs:
        key = _format_config_id(config)
        if key in seen:
            continue
        seen.add(key)
        unique_configs.append(config)
    return unique_configs


def _format_config_id(cfg: Dict[str, Any]) -> str:
    controller = str(cfg.get("geometry_feedback_controller", "legacy")).strip().lower()
    return "_".join(
        [
            f"s{cfg['arc_strategy']}",
            f"c{cfg['chains']}",
            f"g{cfg['generations']}",
            f"kl{cfg['geometry_feedback_target_forward_kl']}",
            f"tol{cfg['geometry_feedback_kl_tolerance']}",
            f"eta{cfg['geometry_feedback_steering_eta']}",
            f"topk{cfg['geometry_feedback_topk']}",
            f"alpha{cfg['geometry_feedback_alpha']}",
            f"ctrl{controller}",
            f"kp{cfg['geometry_feedback_controller_kp']}",
            f"ki{cfg['geometry_feedback_controller_ki']}",
            f"kd{cfg['geometry_feedback_controller_kd']}",
            f"ema{cfg['geometry_feedback_controller_error_ema']}",
        ]
    )


def _summarize_geometry_trace(
    trace: Sequence[Dict[str, Any]],
    target_forward_kl: float,
    kl_tolerance: float,
) -> Dict[str, float]:
    if not trace:
        return {
            "steps": 0,
            "forward_kl_mean": 0.0,
            "forward_kl_p50": 0.0,
            "forward_kl_p90": 0.0,
            "forward_kl_max": 0.0,
            "js_mean": 0.0,
            "js_max": 0.0,
            "candidate_entropy_mean": 0.0,
            "reference_entropy_mean": 0.0,
            "entropy_delta_mean": 0.0,
            "topk_overlap_mean": 0.0,
            "topk_overlap_min": 0.0,
            "weighted_rank_drift_mean": 0.0,
            "weighted_rank_drift_max": 0.0,
            "top1_changed_rate": 0.0,
            "eta_mean": 0.0,
            "eta_max": 0.0,
            "compliance_rate": 0.0,
            "entropy_delta_flip_rate": 0.0,
        }

    forward_kl = [_to_float(item.get("forward_kl")) for item in trace]
    js = [_to_float(item.get("js")) for item in trace]
    candidate_entropy = [_to_float(item.get("candidate_entropy")) for item in trace]
    reference_entropy = [_to_float(item.get("reference_entropy")) for item in trace]
    entropy_delta = [_to_float(item.get("entropy_delta")) for item in trace]
    topk_overlap = [_to_float(item.get("topk_overlap")) for item in trace]
    weighted_rank_drift = [_to_float(item.get("weighted_rank_drift")) for item in trace]
    eta = [_to_float(item.get("eta")) for item in trace]
    top1_changed = [_to_float(item.get("top1_changed")) for item in trace]

    if target_forward_kl > 0.0:
        low = max(0.0, target_forward_kl * (1.0 - kl_tolerance))
        high = target_forward_kl * (1.0 + kl_tolerance)
        compliance_hits = [1.0 for x in forward_kl if low <= x <= high]
    else:
        compliance_hits = [1.0 for x in forward_kl if x <= max(0.0, kl_tolerance)]
    compliance_rate = _safe_divide(len(compliance_hits), len(forward_kl))

    flips = 0
    for idx in range(1, len(entropy_delta)):
        if entropy_delta[idx] == 0.0 or entropy_delta[idx - 1] == 0.0:
            continue
        flips += int((entropy_delta[idx] < 0) != (entropy_delta[idx - 1] < 0))
    flip_rate = _safe_divide(flips, max(0, len(entropy_delta) - 1))

    return {
        "steps": len(trace),
        "forward_kl_mean": _mean(forward_kl),
        "forward_kl_p50": _percentile(forward_kl, 0.50),
        "forward_kl_p90": _percentile(forward_kl, 0.90),
        "forward_kl_max": max(forward_kl),
        "js_mean": _mean(js),
        "js_max": max(js),
        "candidate_entropy_mean": _mean(candidate_entropy),
        "reference_entropy_mean": _mean(reference_entropy),
        "entropy_delta_mean": _mean(entropy_delta),
        "topk_overlap_mean": _mean(topk_overlap),
        "topk_overlap_min": min(topk_overlap),
        "weighted_rank_drift_mean": _mean(weighted_rank_drift),
        "weighted_rank_drift_max": max(weighted_rank_drift),
        "top1_changed_rate": _safe_divide(sum(top1_changed), len(top1_changed)),
        "eta_mean": _mean(eta),
        "eta_max": max(eta),
        "compliance_rate": compliance_rate,
        "entropy_delta_flip_rate": flip_rate,
    }


def _flatten_task_traces(task_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for summary in task_summaries:
        points.extend(list(summary.get("trace", [])))
    return points


def _collect_task_traces(results) -> List[Dict[str, Any]]:
    task_summaries: List[Dict[str, Any]] = []
    for task in results.task_results:
        trace = list(getattr(task, "lr_decode_trace", []) or [])
        summary = _summarize_geometry_trace(
            trace,
            results.geometry_feedback_target_forward_kl,
            results.geometry_feedback_kl_tolerance,
        )
        task_summaries.append(
            {
                "task_id": task.task_id,
                "test_index": _to_int(task.test_index),
                "trace_steps": summary["steps"],
                "lr_correct": bool(getattr(task, "lr_correct", False)),
                "lr_parse_attempts": _to_int(getattr(task, "lr_parse_attempts", 0)),
                "lr_time": _to_float(getattr(task, "lr_time", 0.0)),
                "lr_best_partial": _to_float(getattr(task, "lr_best_partial", 0.0)),
                "trace": trace,
                "summary": summary,
            }
        )
    return task_summaries


def _run_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    results = run_arc_evaluation(
        encoder=args.model,
        max_tasks=args.max_tasks,
        chains=config["chains"],
        generations=config["generations"],
        max_tokens=args.max_tokens,
        decode_mode=args.decode_mode,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        arc_version=args.arc_version,
        lr_retries=args.lr_retries,
        arc_strategy=config["arc_strategy"],
        reasoning_mode=args.reasoning_mode,
        trajectory_steps=args.trajectory_steps,
        trajectory_decode_interval=args.trajectory_decode_interval,
        trajectory_step_scale=args.trajectory_step_scale,
        geometry_feedback_target_forward_kl=config["geometry_feedback_target_forward_kl"],
        geometry_feedback_kl_tolerance=config["geometry_feedback_kl_tolerance"],
        geometry_feedback_steering_eta=config["geometry_feedback_steering_eta"],
        geometry_feedback_alpha=config["geometry_feedback_alpha"],
        geometry_feedback_kl_cap=config["geometry_feedback_kl_cap"],
        geometry_feedback_topk=config["geometry_feedback_topk"],
        geometry_feedback_eta_min=config["geometry_feedback_eta_min"],
        geometry_feedback_eta_max=config["geometry_feedback_eta_max"],
        geometry_feedback_eta_growth=config["geometry_feedback_eta_growth"],
        geometry_feedback_eta_decay=config["geometry_feedback_eta_decay"],
        geometry_feedback_controller=config["geometry_feedback_controller"],
        geometry_feedback_controller_kp=config["geometry_feedback_controller_kp"],
        geometry_feedback_controller_ki=config["geometry_feedback_controller_ki"],
        geometry_feedback_controller_kd=config["geometry_feedback_controller_kd"],
        geometry_feedback_controller_error_ema=config["geometry_feedback_controller_error_ema"],
    )

    task_traces = _collect_task_traces(results)
    tasks_with_trace = sum(1 for record in task_traces if record["trace_steps"] > 0)
    trace_points = _flatten_task_traces(task_traces)
    trace_steps_total = sum(record["trace_steps"] for record in task_traces)
    geometry_rate = _safe_divide(trace_steps_total, max(1, _to_int(results.total_tests)))
    trace_summary = _summarize_geometry_trace(
        trace_points,
        results.geometry_feedback_target_forward_kl,
        results.geometry_feedback_kl_tolerance,
    )

    run_total = _to_float(results.total_time, 0.0)
    total_tests = max(1, _to_int(results.total_tests))
    total_correct = _to_int(results.lr_correct)
    total_parsed = _to_int(results.lr_parsed)

    return {
        "run_id": _format_config_id(config),
        "config": config,
        "meta": {
            "model": args.model,
            "arc_version": args.arc_version,
            "arc_strategy": config["arc_strategy"],
            "max_tasks": args.max_tasks,
            "max_tokens": args.max_tokens,
            "decode_mode": args.decode_mode,
            "reasoning_mode": args.reasoning_mode,
            "trajectory_steps": args.trajectory_steps,
            "trajectory_decode_interval": args.trajectory_decode_interval,
            "trajectory_step_scale": args.trajectory_step_scale,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": {
            "total_tests": _to_int(results.total_tests),
            "lr_correct": total_correct,
            "lr_parsed": total_parsed,
            "lr_accuracy": _safe_divide(total_correct, total_tests),
            "lr_parse_rate": _safe_divide(total_parsed, total_tests),
            "total_time": run_total,
            "time_per_task": _safe_divide(run_total, total_tests),
            "trace_steps_total": trace_steps_total,
            "tasks_with_trace": tasks_with_trace,
            "trace_steps_per_test": _safe_divide(trace_steps_total, total_tests),
            "geometry": {
                "trace_step_density": geometry_rate,
                **trace_summary,
            },
            "geometry_task_summary": {
                "n_tasks_total": len(task_traces),
                "n_with_nonempty_trace": tasks_with_trace,
                "mean_trace_steps": _safe_divide(trace_steps_total, max(1, tasks_with_trace)),
            },
        },
        "task_trace_summaries": task_traces if args.include_task_traces else [],
    }


def _build_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    strategy_values = args.arc_strategies if args.arc_strategies else args.arc_strategy
    strategy_list = _parse_arc_strategies(strategy_values)
    return [
        {
            "arc_strategy": str(cfg_strategy).strip().lower(),
            "chains": _to_int(chains),
            "generations": _to_int(generations),
            "geometry_feedback_target_forward_kl": _to_float(cfg_target_kl),
            "geometry_feedback_kl_tolerance": _to_float(cfg_kl_tol),
            "geometry_feedback_steering_eta": _to_float(cfg_eta),
            "geometry_feedback_alpha": _to_float(cfg_alpha),
            "geometry_feedback_kl_cap": _to_float(cfg_kl_cap),
            "geometry_feedback_topk": _to_int(cfg_topk),
            "geometry_feedback_controller": str(cfg_controller).strip().lower(),
            "geometry_feedback_controller_kp": _to_float(cfg_controller_kp),
            "geometry_feedback_controller_ki": _to_float(cfg_controller_ki),
            "geometry_feedback_controller_kd": _to_float(cfg_controller_kd),
            "geometry_feedback_controller_error_ema": _to_float(cfg_controller_error_ema),
            "geometry_feedback_eta_min": _to_float(cfg_eta_min),
            "geometry_feedback_eta_max": _to_float(cfg_eta_max),
            "geometry_feedback_eta_growth": _to_float(cfg_eta_growth),
            "geometry_feedback_eta_decay": _to_float(cfg_eta_decay),
        }
        for chains, generations, cfg_strategy, cfg_target_kl, cfg_kl_tol, cfg_eta, cfg_alpha, cfg_kl_cap, cfg_topk, cfg_controller, cfg_controller_kp, cfg_controller_ki, cfg_controller_kd, cfg_controller_error_ema, cfg_eta_min, cfg_eta_max, cfg_eta_growth, cfg_eta_decay in product(
            _parse_list(args.chains, int),
            _parse_list(args.generations, int),
            strategy_list,
            _parse_list(args.geometry_feedback_target_forward_kl, float),
            _parse_list(args.geometry_feedback_kl_tolerance, float),
            _parse_list(args.geometry_feedback_steering_eta, float),
            _parse_list(args.geometry_feedback_alpha, float),
            _parse_list(args.geometry_feedback_kl_cap, float),
            _parse_list(args.geometry_feedback_topk, int),
            _parse_list(args.geometry_feedback_controller, str),
            _parse_list(args.geometry_feedback_controller_kp, float),
            _parse_list(args.geometry_feedback_controller_ki, float),
            _parse_list(args.geometry_feedback_controller_kd, float),
            _parse_list(args.geometry_feedback_controller_error_ema, float),
            _parse_list(args.geometry_feedback_eta_min, float),
            _parse_list(args.geometry_feedback_eta_max, float),
            _parse_list(args.geometry_feedback_eta_growth, float),
            _parse_list(args.geometry_feedback_eta_decay, float),
        )
    ]


def _is_dominated(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    # b dominates a if b is >= on positive metrics and <= on time
    keys_better = [
        ("lr_accuracy", b["summary"]["lr_accuracy"] >= a["summary"]["lr_accuracy"]),
        ("lr_parse_rate", b["summary"]["lr_parse_rate"] >= a["summary"]["lr_parse_rate"]),
        ("geometry_accuracy_proxy", b["summary"]["geometry"]["compliance_rate"] >= a["summary"]["geometry"]["compliance_rate"]),
        ("time", b["summary"]["time_per_task"] <= a["summary"]["time_per_task"]),
    ]
    better_or_equal = all(flag for _, flag in keys_better)
    strictly_better = any(
        [
            b["summary"]["lr_accuracy"] > a["summary"]["lr_accuracy"],
            b["summary"]["lr_parse_rate"] > a["summary"]["lr_parse_rate"],
            b["summary"]["geometry"]["compliance_rate"] > a["summary"]["geometry"]["compliance_rate"],
            b["summary"]["time_per_task"] < a["summary"]["time_per_task"],
        ]
    )
    return better_or_equal and strictly_better


def _compute_frontier(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    frontier: List[Dict[str, Any]] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if candidate is other:
                continue
            if _is_dominated(candidate, other):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    frontier.sort(key=lambda item: item["summary"]["lr_accuracy"], reverse=True)
    return frontier


def _run_score(
    row: Dict[str, Any],
    args: argparse.Namespace,
) -> float:
    metrics = row["summary"]
    geometry = metrics["geometry"]
    utility = (
        args.objective_accuracy_weight * metrics["lr_accuracy"]
        + args.objective_parse_weight * metrics["lr_parse_rate"]
        + args.objective_geometry_weight * geometry["compliance_rate"]
    )
    return utility / (1.0 + args.objective_time_penalty * metrics["time_per_task"])


def _neighbor_configs(config: Dict[str, Any], step_scale: float) -> List[Dict[str, Any]]:
    step_scale = max(1e-6, step_scale)
    neighbors: List[Dict[str, Any]] = []

    def add_candidate(overrides: Dict[str, Any]) -> None:
        candidate = dict(config)
        candidate.update(overrides)
        if candidate == config:
            return

        candidate["chains"] = _to_int(candidate["chains"], 1)
        candidate["generations"] = _to_int(candidate["generations"], 1)
        candidate["geometry_feedback_topk"] = _to_int(candidate["geometry_feedback_topk"], 4)

        if candidate["chains"] < 1 or candidate["generations"] < 1:
            return
        if candidate["geometry_feedback_topk"] < 4:
            candidate["geometry_feedback_topk"] = 4

        for value in [
            candidate["geometry_feedback_target_forward_kl"],
            candidate["geometry_feedback_kl_tolerance"],
            candidate["geometry_feedback_steering_eta"],
            candidate["geometry_feedback_alpha"],
            candidate["geometry_feedback_kl_cap"],
            candidate["geometry_feedback_eta_min"],
            candidate["geometry_feedback_eta_max"],
            candidate["geometry_feedback_eta_growth"],
            candidate["geometry_feedback_eta_decay"],
        ]:
            if value is None:
                return

        if candidate["geometry_feedback_eta_min"] > candidate["geometry_feedback_eta_max"]:
            return
        if candidate["geometry_feedback_eta_growth"] < 1.0:
            return
        if candidate["geometry_feedback_eta_decay"] < 0.2:
            return
        neighbors.append(candidate)

    base_kl = config["geometry_feedback_target_forward_kl"]
    add_candidate({"geometry_feedback_target_forward_kl": _clamp(base_kl * (1.0 - step_scale), 0.001, 0.8)})
    add_candidate({"geometry_feedback_target_forward_kl": _clamp(base_kl * (1.0 + step_scale), 0.001, 0.8)})

    base_tolerance = config["geometry_feedback_kl_tolerance"]
    add_candidate({"geometry_feedback_kl_tolerance": _clamp(base_tolerance * (1.0 - step_scale), 0.01, 0.95)})
    add_candidate({"geometry_feedback_kl_tolerance": _clamp(base_tolerance * (1.0 + step_scale), 0.01, 0.95)})

    base_eta = config["geometry_feedback_steering_eta"]
    add_candidate({"geometry_feedback_steering_eta": _clamp(base_eta * (1.0 - step_scale), 1e-4, 0.5)})
    add_candidate({"geometry_feedback_steering_eta": _clamp(base_eta * (1.0 + step_scale), 1e-4, 0.5)})

    base_alpha = config["geometry_feedback_alpha"]
    add_candidate({"geometry_feedback_alpha": _clamp(base_alpha * (1.0 - step_scale), 1e-4, 0.1)})
    add_candidate({"geometry_feedback_alpha": _clamp(base_alpha * (1.0 + step_scale), 1e-4, 0.1)})

    base_cap = config["geometry_feedback_kl_cap"]
    add_candidate({"geometry_feedback_kl_cap": _clamp(base_cap * (1.0 - step_scale), 0.05, 2.0)})
    add_candidate({"geometry_feedback_kl_cap": _clamp(base_cap * (1.0 + step_scale), 0.05, 2.0)})

    base_topk = config["geometry_feedback_topk"]
    add_candidate(
        {
            "geometry_feedback_topk": _clamp(base_topk * (1.0 - step_scale), 4, 400),
        }
    )
    add_candidate(
        {
            "geometry_feedback_topk": _clamp(base_topk * (1.0 + step_scale), 4, 400),
        }
    )

    base_eta_min = config["geometry_feedback_eta_min"]
    base_eta_max = config["geometry_feedback_eta_max"]
    add_candidate(
        {
            "geometry_feedback_eta_min": _clamp(base_eta_min * (1.0 - step_scale), 1e-4, 1.0),
            "geometry_feedback_eta_max": _clamp(base_eta_max, 1e-4, 1.0),
        }
    )
    add_candidate(
        {
            "geometry_feedback_eta_min": _clamp(base_eta_min * (1.0 + step_scale), 1e-4, 1.0),
            "geometry_feedback_eta_max": _clamp(base_eta_max, 1e-4, 1.0),
        }
    )
    add_candidate(
        {
            "geometry_feedback_eta_max": _clamp(base_eta_max * (1.0 - step_scale), 1e-4, 1.0),
            "geometry_feedback_eta_min": _clamp(base_eta_min, 1e-4, 1.0),
        }
    )
    add_candidate(
        {
            "geometry_feedback_eta_max": _clamp(base_eta_max * (1.0 + step_scale), 1e-4, 1.0),
            "geometry_feedback_eta_min": _clamp(base_eta_min, 1e-4, 1.0),
        }
    )
    add_candidate(
        {
            "geometry_feedback_eta_growth": _clamp(
                1.0 + (config["geometry_feedback_eta_growth"] - 1.0) * (1.0 - step_scale),
                1.0,
                1.5,
            ),
            "geometry_feedback_eta_decay": _clamp(config["geometry_feedback_eta_decay"], 0.2, 0.999),
        }
    )
    add_candidate(
        {
            "geometry_feedback_eta_growth": _clamp(
                1.0 + (config["geometry_feedback_eta_growth"] - 1.0) * (1.0 + step_scale),
                1.0,
                1.5,
            ),
            "geometry_feedback_eta_decay": _clamp(config["geometry_feedback_eta_decay"], 0.2, 0.999),
        }
    )
    add_candidate(
        {
            "geometry_feedback_eta_decay": _clamp(config["geometry_feedback_eta_decay"] * (1.0 - step_scale), 0.2, 0.999),
            "geometry_feedback_eta_growth": _clamp(config["geometry_feedback_eta_growth"], 1.0, 1.5),
        }
    )
    add_candidate(
        {
            "geometry_feedback_eta_decay": _clamp(config["geometry_feedback_eta_decay"] * (1.0 + step_scale), 0.2, 0.999),
            "geometry_feedback_eta_growth": _clamp(config["geometry_feedback_eta_growth"], 1.0, 1.5),
        }
    )

    chain_delta = max(1, round(config["chains"] * step_scale / 2))
    add_candidate({"chains": config["chains"] - chain_delta})
    add_candidate({"chains": config["chains"] + chain_delta})
    generation_delta = max(1, round(config["generations"] * step_scale / 2))
    add_candidate({"generations": config["generations"] - generation_delta})
    add_candidate({"generations": config["generations"] + generation_delta})

    return _dedupe_configs(neighbors)


def _run_config_batch(
    configs: List[Dict[str, Any]],
    args: argparse.Namespace,
    run_counter_start: int,
    max_to_run: int,
    seen: set[str],
) -> List[Dict[str, Any]]:
    batch: List[Dict[str, Any]] = []
    run_counter = run_counter_start
    remaining = max(0, max_to_run)
    for config in configs:
        if remaining <= 0:
            break
        run_id = _format_config_id(config)
        if run_id in seen:
            continue
        seen.add(run_id)
        print(f"[{run_counter}] running", run_id)
        run_counter += 1
        batch.append(_run_config(config, args))
        remaining -= 1
    return batch


def _run_adaptive_search(
    args: argparse.Namespace,
    base_configs: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    total_budget = args.max_configs if args.max_configs and args.max_configs > 0 else max(24, args.adaptive_rounds * 8 * max(1, args.adaptive_top_k))
    seed_budget = min(len(base_configs), args.adaptive_seed_size or len(base_configs), total_budget)
    seen: set[str] = set()
    all_results: List[Dict[str, Any]] = []

    if seed_budget <= 0:
        return all_results, []

    run_counter = 1
    frontier_log: List[Dict[str, Any]] = []
    seed_candidates = base_configs[:seed_budget]
    seed_results = _run_config_batch(seed_candidates, args, run_counter, total_budget, seen)
    all_results.extend(seed_results)
    run_counter += len(seed_results)
    remaining = total_budget - len(seed_results)

    best_run = max(seed_results, key=lambda item: _run_score(item, args))
    best_score = _run_score(best_run, args)

    for round_idx in range(1, args.adaptive_rounds + 1):
        if remaining <= 0:
            break
        ranked = sorted(
            all_results,
            key=lambda item: _run_score(item, args),
            reverse=True,
        )
        parents = ranked[: max(1, args.adaptive_top_k)]
        round_scale = args.adaptive_step_scale * (args.adaptive_decay ** (round_idx - 1))

        frontier_log.append(
            {
                "round": round_idx,
                "n_parents": len(parents),
                "step_scale": round_scale,
                "n_seen": len(seen),
            }
        )

        neighbor_pool: List[Dict[str, Any]] = []
        for parent in parents:
            neighbor_pool.extend(_neighbor_configs(parent["config"], round_scale))

        neighbor_pool = _dedupe_configs(neighbor_pool)
        proposals: List[Dict[str, Any]] = []
        for candidate in neighbor_pool:
            run_id = _format_config_id(candidate)
            if run_id in seen:
                continue
            proposals.append(candidate)
            if len(proposals) >= remaining:
                break

        if not proposals:
            break

        random.shuffle(proposals)
        next_results = _run_config_batch(
            proposals,
            args,
            run_counter,
            remaining,
            seen,
        )
        if not next_results:
            break
        all_results.extend(next_results)
        run_counter += len(next_results)
        remaining = total_budget - len(all_results)

        current_best = max(
            next_results,
            key=lambda item: _run_score(item, args),
            default=best_run,
        )
        current_score = _run_score(current_best, args)
        frontier_log[-1]["n_candidates"] = len(proposals)
        frontier_log[-1]["n_ran"] = len(next_results)
        frontier_log[-1]["best_score"] = current_score
        frontier_log[-1]["best_run_id"] = current_best["run_id"]

        if current_score <= best_score + args.adaptive_improvement_threshold:
            break
        best_score = max(best_score, current_score)
        best_run = current_best

    frontier_log.sort(key=lambda item: item["round"])
    return all_results, frontier_log


def _export_summary_csv(results: List[Dict[str, Any]], output_file: Path) -> None:
    if not results:
        output_file.write_text("[]\n", encoding="utf-8")
        return

    fieldnames = [
        "run_id",
        "arc_strategy",
        "chains",
        "generations",
        "geometry_feedback_target_forward_kl",
        "geometry_feedback_kl_tolerance",
        "geometry_feedback_steering_eta",
        "geometry_feedback_alpha",
        "geometry_feedback_kl_cap",
        "geometry_feedback_topk",
        "geometry_feedback_eta_min",
        "geometry_feedback_eta_max",
        "geometry_feedback_eta_growth",
        "geometry_feedback_eta_decay",
        "geometry_feedback_controller",
        "geometry_feedback_controller_kp",
        "geometry_feedback_controller_ki",
        "geometry_feedback_controller_kd",
        "geometry_feedback_controller_error_ema",
        "total_tests",
        "lr_correct",
        "lr_accuracy",
        "lr_parsed",
        "lr_parse_rate",
        "time_per_task",
        "trace_steps_per_test",
        "trace_step_density",
        "forward_kl_mean",
        "forward_kl_p50",
        "forward_kl_p90",
        "forward_kl_max",
        "js_mean",
        "topk_overlap_mean",
        "topk_overlap_min",
        "weighted_rank_drift_mean",
        "weighted_rank_drift_max",
        "top1_changed_rate",
        "eta_mean",
        "eta_max",
        "compliance_rate",
        "entropy_delta_flip_rate",
        "entropy_delta_mean",
        "candidate_entropy_mean",
        "reference_entropy_mean",
        "geometry_trace_rate",
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            cfg = row["config"]
            summary = row["summary"]
            geom = summary["geometry"]
            writer.writerow(
                {
                    "run_id": row["run_id"],
                    "arc_strategy": cfg["arc_strategy"],
                    "chains": cfg["chains"],
                    "generations": cfg["generations"],
                    "geometry_feedback_target_forward_kl": cfg["geometry_feedback_target_forward_kl"],
                    "geometry_feedback_kl_tolerance": cfg["geometry_feedback_kl_tolerance"],
                    "geometry_feedback_steering_eta": cfg["geometry_feedback_steering_eta"],
                    "geometry_feedback_alpha": cfg["geometry_feedback_alpha"],
                    "geometry_feedback_kl_cap": cfg["geometry_feedback_kl_cap"],
                    "geometry_feedback_topk": cfg["geometry_feedback_topk"],
                    "geometry_feedback_eta_min": cfg["geometry_feedback_eta_min"],
                    "geometry_feedback_eta_max": cfg["geometry_feedback_eta_max"],
                    "geometry_feedback_eta_growth": cfg["geometry_feedback_eta_growth"],
                    "geometry_feedback_eta_decay": cfg["geometry_feedback_eta_decay"],
                    "geometry_feedback_controller": cfg["geometry_feedback_controller"],
                    "geometry_feedback_controller_kp": cfg["geometry_feedback_controller_kp"],
                    "geometry_feedback_controller_ki": cfg["geometry_feedback_controller_ki"],
                    "geometry_feedback_controller_kd": cfg["geometry_feedback_controller_kd"],
                    "geometry_feedback_controller_error_ema": cfg["geometry_feedback_controller_error_ema"],
                    "total_tests": summary["total_tests"],
                    "lr_correct": summary["lr_correct"],
                    "lr_accuracy": summary["lr_accuracy"],
                    "lr_parsed": summary["lr_parsed"],
                    "lr_parse_rate": summary["lr_parse_rate"],
                    "time_per_task": summary["time_per_task"],
                    "trace_steps_per_test": summary["trace_steps_per_test"],
                    "trace_step_density": geom["trace_step_density"],
                    "forward_kl_mean": geom["forward_kl_mean"],
                    "forward_kl_p50": geom["forward_kl_p50"],
                    "forward_kl_p90": geom["forward_kl_p90"],
                    "forward_kl_max": geom["forward_kl_max"],
                    "js_mean": geom["js_mean"],
                    "topk_overlap_mean": geom["topk_overlap_mean"],
                    "topk_overlap_min": geom["topk_overlap_min"],
                    "weighted_rank_drift_mean": geom["weighted_rank_drift_mean"],
                    "weighted_rank_drift_max": geom["weighted_rank_drift_max"],
                    "top1_changed_rate": geom["top1_changed_rate"],
                    "eta_mean": geom["eta_mean"],
                    "eta_max": geom["eta_max"],
                    "compliance_rate": geom["compliance_rate"],
                    "entropy_delta_flip_rate": geom["entropy_delta_flip_rate"],
                    "entropy_delta_mean": geom["entropy_delta_mean"],
                    "candidate_entropy_mean": geom["candidate_entropy_mean"],
                    "reference_entropy_mean": geom["reference_entropy_mean"],
                    "geometry_trace_rate": summary["geometry_task_summary"]["n_with_nonempty_trace"]
                    / max(1, summary["total_tests"]),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--arc-version", default="3")
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--decode-mode", default="geometry_feedback")
    parser.add_argument("--lr-retries", type=int, default=1)
    parser.add_argument(
        "--arc-strategy",
        default=None,
        help="Deprecated alias for --arc-strategies (single value or comma-separated list)",
    )
    parser.add_argument(
        "--arc-strategies",
        default="adaptive,repair,geometry_bandit",
        help="Comma-separated ARC strategies to benchmark. Supported: single, adaptive, repair, consensus, geometry_bandit, self_improving",
    )
    parser.add_argument("--reasoning-mode", default="hybrid")
    parser.add_argument("--trajectory-steps", type=int, default=3)
    parser.add_argument("--trajectory-decode-interval", type=int, default=1)
    parser.add_argument("--trajectory-step-scale", type=float, default=0.15)
    parser.add_argument("--max-tokens", type=int, default=512)

    parser.add_argument("--chains", default="4")
    parser.add_argument("--generations", default="5")

    parser.add_argument(
        "--geometry-feedback-target-forward-kl",
        default="0.03,0.06",
    )
    parser.add_argument(
        "--geometry-feedback-kl-tolerance",
        default="0.25,0.45",
    )
    parser.add_argument(
        "--geometry-feedback-steering-eta",
        default="0.025,0.05",
    )
    parser.add_argument(
        "--geometry-feedback-alpha",
        default="0.005,0.01",
    )
    parser.add_argument(
        "--geometry-feedback-kl-cap",
        default="0.4,0.8",
    )
    parser.add_argument("--geometry-feedback-topk", default="25,50")
    parser.add_argument("--geometry-feedback-eta-min", default="0.01,0.02")
    parser.add_argument("--geometry-feedback-eta-max", default="0.2,0.5")
    parser.add_argument("--geometry-feedback-eta-growth", default="1.03,1.06")
    parser.add_argument("--geometry-feedback-eta-decay", default="0.85,0.95")
    parser.add_argument("--geometry-feedback-controller", default="legacy,pid")
    parser.add_argument("--geometry-feedback-controller-kp", default="0.0,0.25")
    parser.add_argument("--geometry-feedback-controller-ki", default="0.0")
    parser.add_argument("--geometry-feedback-controller-kd", default="0.0")
    parser.add_argument("--geometry-feedback-controller-error-ema", default="0.2")

    parser.add_argument(
        "--search-mode",
        choices=["grid", "adaptive"],
        default="adaptive",
        help="grid: fixed Cartesian sweep; adaptive: iterative neighborhood search around top candidates",
    )
    parser.add_argument("--adaptive-rounds", type=int, default=2)
    parser.add_argument("--adaptive-top-k", type=int, default=2)
    parser.add_argument("--adaptive-seed-size", type=int, default=4)
    parser.add_argument("--adaptive-step-scale", type=float, default=0.35)
    parser.add_argument("--adaptive-decay", type=float, default=0.55)
    parser.add_argument("--adaptive-improvement-threshold", type=float, default=0.0025)

    parser.add_argument("--objective-accuracy-weight", type=float, default=0.65)
    parser.add_argument("--objective-parse-weight", type=float, default=0.20)
    parser.add_argument("--objective-geometry-weight", type=float, default=0.15)
    parser.add_argument("--objective-time-penalty", type=float, default=0.01)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--json-output", type=Path, default=Path("eval_results/arc3_geometry_sweep.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("eval_results/arc3_geometry_sweep.csv"))
    parser.add_argument("--include-task-traces", action="store_true", help="Include per-task geometry traces in JSON output")
    parser.add_argument("--max-configs", type=int, default=0, help="Stop after N configs for a quick smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.decode_mode != "geometry_feedback":
        raise ValueError("This script is geometry-first and requires --decode-mode geometry_feedback")

    configs = _build_configs(args)
    random.seed(args.random_seed)

    if args.search_mode == "adaptive":
        print(f"Planned adaptive search candidates: {len(configs)}")
        results, search_log = _run_adaptive_search(args, configs)
        search_metadata = {
            "mode": "adaptive",
            "adaptive_rounds": args.adaptive_rounds,
            "adaptive_top_k": args.adaptive_top_k,
            "adaptive_seed_size": args.adaptive_seed_size,
            "adaptive_step_scale": args.adaptive_step_scale,
            "adaptive_decay": args.adaptive_decay,
            "adaptive_improvement_threshold": args.adaptive_improvement_threshold,
            "round_log": search_log,
        }
    else:
        print(f"Planned geometry sweep configs: {len(configs)}")
        if args.max_configs and args.max_configs > 0:
            configs = configs[: args.max_configs]
        results = _run_config_batch(configs, args, run_counter_start=1, max_to_run=len(configs), seen=set())
        search_metadata = {
            "mode": "grid",
            "grid_size": len(configs),
        }

    if not results:
        raise ValueError("No configurations were run. Check seed/budget settings.")

    frontier = _compute_frontier(results)
    best_accuracy = max(results, key=lambda item: item["summary"]["lr_accuracy"])
    best_parse = max(results, key=lambda item: item["summary"]["lr_parse_rate"])
    best_geometry = max(results, key=lambda item: item["summary"]["geometry"]["compliance_rate"])

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_args": _serialize_args_for_json(args),
        "search": search_metadata,
        "n_configs": len(results),
        "frontier": [item["run_id"] for item in frontier],
        "best": {
            "max_accuracy_run_id": best_accuracy["run_id"],
            "max_accuracy": best_accuracy["summary"]["lr_accuracy"],
            "max_parse_rate_run_id": best_parse["run_id"],
            "max_parse_rate": best_parse["summary"]["lr_parse_rate"],
            "best_geometry_compliance_run_id": best_geometry["run_id"],
            "best_geometry_compliance": best_geometry["summary"]["geometry"]["compliance_rate"],
        },
        "runs": results,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _export_summary_csv(results, args.csv_output)

    print(f"Saved sweep summary to {args.json_output}")
    print(f"Saved CSV summary to {args.csv_output}")
    print("Best accuracy:", best_accuracy["run_id"], best_accuracy["summary"]["lr_accuracy"])
    print("Best parse rate:", best_parse["run_id"], best_parse["summary"]["lr_parse_rate"])
    print("Best geometry compliance:", best_geometry["run_id"], best_geometry["summary"]["geometry"]["compliance_rate"])
    print("Pareto-like frontier size:", len(frontier))


if __name__ == "__main__":
    main()
