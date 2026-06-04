"""Sweep label-free diffusion repairability gates against reference repairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_AUDIT = Path("eval_results/diffusion_language/diffusion_repairability_geometry_audit.json")
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/diffusion_repairability_geometry_sweep.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--quality-thresholds", default="0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--gap-mins", default="0,1,2,3,4")
    parser.add_argument("--gap-maxs", default="4,5,6,7,8,9,10,11,12")
    parser.add_argument("--coverage-mins", default="0.00,0.20,0.30,0.40,0.50,0.60")
    parser.add_argument("--coverage-maxs", default="0.40,0.60,0.80,1.00")
    parser.add_argument(
        "--skeleton-step-maxs",
        default="none,10,16,20,24,32",
        help=(
            "Comma-separated first-denoise-skeleton max steps to test. "
            "Use 'none' for the current no-step-limit policy."
        ),
    )
    parser.add_argument(
        "--phase-window-confirmation-scores",
        default="",
        help=(
            "Comma-separated fresh score JSON files that confirm phase-window "
            "operating points with actual CUDA generations."
        ),
    )
    parser.add_argument("--source-min-chars", type=int, default=240)
    parser.add_argument("--promotion-margin", type=float, default=0.02)
    parser.add_argument("--base-generation-budget", type=float, default=2.0)
    parser.add_argument("--repair-generation-budget", type=float, default=1.0)
    return parser.parse_args()


def build_repairability_geometry_sweep(
    *,
    audit_path: Path,
    quality_thresholds: list[float],
    gap_mins: list[int],
    gap_maxs: list[int],
    coverage_mins: list[float],
    coverage_maxs: list[float],
    skeleton_step_maxs: list[int | None] | None = None,
    phase_window_confirmation_score_paths: list[Path] | None = None,
    source_min_chars: int = 240,
    promotion_margin: float = 0.02,
    base_generation_budget: float = 2.0,
    repair_generation_budget: float = 1.0,
) -> dict[str, object]:
    audit = _read_json(audit_path)
    rows = _list_of_dicts(audit.get("rows"))
    scores_path = Path(str(audit.get("scores_path", "")))
    scores = _read_json(scores_path) if scores_path.exists() else {}
    current_gate = _current_gate(scores)
    baseline_score = _planning_arm_score(scores, "fixed", rows)
    random_score = _planning_arm_score(scores, "random", rows)
    total_positive_lift = sum(_positive_reference_delta(row, promotion_margin) for row in rows)
    skeleton_step_maxs = skeleton_step_maxs or [None]
    sweep_rows = []
    for quality_threshold in quality_thresholds:
        for gap_min in gap_mins:
            for gap_max in gap_maxs:
                if gap_max < gap_min:
                    continue
                for coverage_min in coverage_mins:
                    for coverage_max in coverage_maxs:
                        if coverage_max < coverage_min:
                            continue
                        for skeleton_step_max in skeleton_step_maxs:
                            sweep_rows.append(
                                _evaluate_gate(
                                    rows=rows,
                                    quality_threshold=quality_threshold,
                                    gap_min=gap_min,
                                    gap_max=gap_max,
                                    coverage_min=coverage_min,
                                    coverage_max=coverage_max,
                                    skeleton_step_max=skeleton_step_max,
                                    source_min_chars=source_min_chars,
                                    promotion_margin=promotion_margin,
                                    baseline_score=baseline_score,
                                    random_score=random_score,
                                    total_positive_lift=total_positive_lift,
                                    base_generation_budget=base_generation_budget,
                                    repair_generation_budget=repair_generation_budget,
                                    current_gate=current_gate,
                                )
                            )
    frontier = _pareto_frontier(sweep_rows)
    current_row = next((row for row in sweep_rows if row["is_current_gate"]), {})
    phase_window_rows = _phase_window_tradeoff_rows(sweep_rows)
    best_score = max((_float(row["selected_score"]) for row in sweep_rows), default=0.0)
    best_zero_miss = [
        row
        for row in sweep_rows
        if int(row["missed_repair_count"]) == 0 and _float(row["selected_score"]) >= best_score - 1e-12
    ]
    return {
        "audit_path": str(audit_path),
        "baseline_score": baseline_score,
        "current_gate": current_gate,
        "current_gate_result": current_row,
        "frontier": frontier,
        "generated_by": "experiments/sweep_diffusion_repairability_geometry.py",
        "promotion_margin": promotion_margin,
        "random_score": random_score,
        "row_count": len(rows),
        "schema": "diffusion_repairability_geometry_sweep.v1",
        "source_min_chars": source_min_chars,
        "summary": {
            "best_score": best_score,
            "current_gate_exact_frontier_representative": bool(
                current_row and any(_same_gate(current_row, row) for row in frontier)
            ),
            "current_gate_on_frontier": bool(
                current_row and any(_same_score_cost(current_row, row) for row in frontier)
            ),
            "frontier_count": len(frontier),
            "phase_window_count": len(phase_window_rows),
            "sweep_count": len(sweep_rows),
            "zero_miss_best_score_count": len(best_zero_miss),
            "zero_waste_zero_miss_count": sum(
                1
                for row in sweep_rows
                if int(row["wasted_spend_count"]) == 0 and int(row["missed_repair_count"]) == 0
            ),
        },
        "phase_window_tradeoff_rows": phase_window_rows,
        "phase_window_confirmation_rows": _phase_window_confirmation_rows(
            phase_window_confirmation_score_paths or []
        ),
        "sweep_rows": sweep_rows,
        "top_score_rows": sorted(
            sweep_rows,
            key=lambda row: (
                -_float(row["selected_score"]),
                _float(row["relative_generation_budget"]),
                int(row["wasted_spend_count"]),
                int(row["missed_repair_count"]),
            ),
        )[:20],
        "zero_miss_best_score_rows": sorted(
            best_zero_miss,
            key=lambda row: (
                _float(row["relative_generation_budget"]),
                int(row["wasted_spend_count"]),
                -_float(row["lift_per_extra_generation_vs_greedy"]),
            ),
        )[:20],
    }


def render_markdown(sweep: dict[str, object]) -> str:
    summary = _dict(sweep.get("summary"))
    current = _dict(sweep.get("current_gate_result"))
    lines = [
        "# Diffusion Repairability Geometry Sweep",
        "",
        "This file is generated by `experiments/sweep_diffusion_repairability_geometry.py`.",
        "It sweeps label-free source-quality, prompt-gap, and prompt-coverage gates against the ungated repair reference.",
        "",
        "## Summary",
        "",
        f"- Audit: `{sweep.get('audit_path', '')}`",
        f"- Rows: `{sweep.get('row_count', 0)}`",
        f"- Baseline greedy planning score: `{_format_float(sweep.get('baseline_score'))}`",
        f"- Random perturbation planning score: `{_format_float(sweep.get('random_score'))}`",
        f"- Sweep points: `{summary.get('sweep_count', 0)}`",
        f"- Pareto frontier points: `{summary.get('frontier_count', 0)}`",
        f"- Phase-window tradeoff points: `{summary.get('phase_window_count', 0)}`",
        f"- Best score: `{_format_float(summary.get('best_score'))}`",
        f"- Zero-waste/zero-miss gates: `{summary.get('zero_waste_zero_miss_count', 0)}`",
        f"- Current gate score/cost on frontier: `{summary.get('current_gate_on_frontier', False)}`",
        f"- Current gate exact frontier representative: `{summary.get('current_gate_exact_frontier_representative', False)}`",
        "",
        "## Current Gate",
        "",
        _gate_table([current]) if current else "_Current gate was not included in this sweep._",
        "",
        "## Phase Window Tradeoff",
        "",
        (
            "These rows show the best-scoring gate for each maximum first-skeleton "
            "denoise step. Lower caps spend less repair compute but can miss late "
            "productive skeletons."
        ),
        "",
        _gate_table(_list_of_dicts(sweep.get("phase_window_tradeoff_rows"))),
        "",
        "## Fresh Phase-Window Confirmations",
        "",
        _confirmation_table(_list_of_dicts(sweep.get("phase_window_confirmation_rows"))),
        "",
        "## Pareto Frontier",
        "",
        _gate_table(_list_of_dicts(sweep.get("frontier"))),
        "",
        "## Best Zero-Miss Gates",
        "",
        _gate_table(_list_of_dicts(sweep.get("zero_miss_best_score_rows"))),
        "",
        "## Top Score Gates",
        "",
        _gate_table(_list_of_dicts(sweep.get("top_score_rows"))),
    ]
    return "\n".join(lines) + "\n"


def _evaluate_gate(
    *,
    rows: list[dict[str, object]],
    quality_threshold: float,
    gap_min: int,
    gap_max: int,
    coverage_min: float,
    coverage_max: float,
    skeleton_step_max: int | None,
    source_min_chars: int,
    promotion_margin: float,
    baseline_score: float,
    random_score: float,
    total_positive_lift: float,
    base_generation_budget: float,
    repair_generation_budget: float,
    current_gate: dict[str, object],
) -> dict[str, object]:
    selected_scores = []
    spent_tasks = []
    productive_spends = 0
    wasted_spends = 0
    missed_repairs = 0
    captured_positive_lift = 0.0
    for row in rows:
        spend = _gate_spends(
            row,
            quality_threshold=quality_threshold,
            gap_min=gap_min,
            gap_max=gap_max,
            coverage_min=coverage_min,
            coverage_max=coverage_max,
            skeleton_step_max=skeleton_step_max,
            source_min_chars=source_min_chars,
        )
        no_repair_baseline_score = _no_repair_baseline_score(row)
        reference_score = _optional_float(row.get("reference_repair_score"))
        if reference_score is None:
            reference_score = _float(row.get("selected_repair_score"))
        reference_delta = reference_score - no_repair_baseline_score
        if spend:
            spent_tasks.append(str(row.get("task_id", "")))
            selected_scores.append(reference_score)
            if reference_delta > promotion_margin:
                productive_spends += 1
                captured_positive_lift += reference_delta
            else:
                wasted_spends += 1
        else:
            selected_scores.append(no_repair_baseline_score)
            if reference_delta > promotion_margin:
                missed_repairs += 1
    selected_score = mean(selected_scores) if selected_scores else 0.0
    spent_count = len(spent_tasks)
    row_count = len(rows)
    relative_budget = (
        base_generation_budget + (spent_count * repair_generation_budget / row_count)
        if row_count
        else 0.0
    )
    lift_vs_greedy = selected_score - baseline_score
    lift_vs_random = selected_score - random_score
    return {
        "captured_reference_lift_fraction": (
            captured_positive_lift / total_positive_lift if total_positive_lift > 0 else None
        ),
        "coverage_max": coverage_max,
        "coverage_min": coverage_min,
        "f1": _f1(productive_spends, wasted_spends, missed_repairs),
        "gap_max": gap_max,
        "gap_min": gap_min,
        "is_current_gate": _same_gate(
            {
                "coverage_max": coverage_max,
                "coverage_min": coverage_min,
                "gap_max": gap_max,
                "gap_min": gap_min,
                "quality_threshold": quality_threshold,
                "skeleton_step_max": skeleton_step_max,
                "source_min_chars": source_min_chars,
            },
            current_gate,
        ),
        "lift_per_extra_generation_vs_greedy": (
            lift_vs_greedy / (relative_budget - 1.0) if relative_budget > 1.0 else None
        ),
        "lift_per_extra_generation_vs_random": (
            lift_vs_random / (relative_budget - 1.0) if relative_budget > 1.0 else None
        ),
        "lift_vs_greedy": lift_vs_greedy,
        "lift_vs_random": lift_vs_random,
        "missed_repair_count": missed_repairs,
        "productive_spend_count": productive_spends,
        "quality_threshold": quality_threshold,
        "record_count": base_generation_budget * row_count + spent_count * repair_generation_budget,
        "relative_generation_budget": relative_budget,
        "selected_score": selected_score,
        "skeleton_step_max": skeleton_step_max,
        "source_min_chars": source_min_chars,
        "spent_count": spent_count,
        "spent_tasks": spent_tasks,
        "wasted_spend_count": wasted_spends,
    }


def _gate_spends(
    row: dict[str, object],
    *,
    quality_threshold: float,
    gap_min: int,
    gap_max: int,
    coverage_min: float,
    coverage_max: float,
    skeleton_step_max: int | None,
    source_min_chars: int,
) -> bool:
    source_needs_repair = (
        _float(row.get("source_planning_quality")) < quality_threshold
        or int(row.get("source_chars", 0)) < source_min_chars
    )
    return (
        source_needs_repair
        and gap_min <= int(row.get("prompt_gap_count", 0)) <= gap_max
        and coverage_min <= _float(row.get("prompt_coverage")) <= coverage_max
        and _skeleton_step_allowed(row, max_step=skeleton_step_max)
    )


def _skeleton_step_allowed(row: dict[str, object], *, max_step: int | None) -> bool:
    if max_step is None:
        return True
    first_step = _optional_int(row.get("first_repairable_denoise_skeleton_step"))
    return first_step is not None and first_step <= max_step


def _current_gate(scores: dict[str, object]) -> dict[str, object]:
    return {
        "coverage_max": _float(scores.get("repair_source_prompt_coverage_max"), default=1.0),
        "coverage_min": _float(scores.get("repair_source_prompt_coverage_min"), default=0.0),
        "gap_max": int(scores.get("repair_source_prompt_gap_max", 999)),
        "gap_min": int(scores.get("repair_source_prompt_gap_min", 0)),
        "quality_threshold": _float(scores.get("repair_source_quality_threshold"), default=0.0),
        "skeleton_step_max": _optional_int(scores.get("repair_denoise_skeleton_max_step")),
        "source_min_chars": int(scores.get("repair_source_min_chars", 240)),
    }


def _planning_arm_score(
    scores: dict[str, object],
    arm: str,
    rows: list[dict[str, object]],
) -> float:
    by_family = _dict(scores.get("by_family_arm"))
    planning = _dict(by_family.get("planning"))
    arm_score = _dict(planning.get(arm)).get("mean_task_score")
    if isinstance(arm_score, int | float) and not isinstance(arm_score, bool):
        return float(arm_score)
    if arm == "fixed":
        return _mean(row.get("source_task_score") for row in rows) or 0.0
    return 0.0


def _pareto_frontier(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frontier = []
    best_score = float("-inf")
    for row in sorted(
        rows,
        key=lambda item: (
            _float(item["relative_generation_budget"]),
            -_float(item["selected_score"]),
            int(item["wasted_spend_count"]),
            int(item["missed_repair_count"]),
        ),
    ):
        score = _float(row["selected_score"])
        if score > best_score + 1e-12:
            frontier.append(row)
            best_score = score
    return frontier


def _phase_window_tradeoff_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[int | None, list[dict[str, object]]] = {}
    for row in rows:
        skeleton_step_max = _optional_int(row.get("skeleton_step_max"))
        groups.setdefault(skeleton_step_max, []).append(row)
    tradeoff_rows = []
    for group in groups.values():
        best = sorted(
            group,
            key=lambda row: (
                -_float(row["selected_score"]),
                _float(row["relative_generation_budget"]),
                int(row["wasted_spend_count"]),
                int(row["missed_repair_count"]),
            ),
        )[0]
        tradeoff_rows.append(best)
    return sorted(
        tradeoff_rows,
        key=lambda row: (
            _optional_int(row.get("skeleton_step_max")) is not None,
            _optional_int(row.get("skeleton_step_max")) or 0,
        ),
    )


def _phase_window_confirmation_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        scores = _read_json(path)
        planning = _dict(_dict(scores.get("by_family_arm")).get("planning"))
        fixed = _dict(planning.get("fixed"))
        random = _dict(planning.get("random"))
        repair = _dict(planning.get("repair_selected"))
        repair_score = _float(repair.get("mean_task_score"))
        fixed_score = _float(fixed.get("mean_task_score"))
        random_score = _float(random.get("mean_task_score"))
        gate_rows = _list_of_dicts(scores.get("repair_spend_gate_rows"))
        rows.append(
            {
                "all_generation_count": int(scores.get("all_generation_count", 0)),
                "delta_vs_fixed": repair_score - fixed_score,
                "delta_vs_random": repair_score - random_score,
                "gate_reason_counts": _reason_counts(gate_rows),
                "late_skeleton_skip_count": sum(
                    1
                    for row in gate_rows
                    if row.get("reason") == "late_repairable_denoise_skeleton"
                ),
                "relative_generation_budget": _float(repair.get("mean_generation_budget_per_task")),
                "repair_score": repair_score,
                "repair_pack": str(scores.get("repair_pack", "")),
                "run_id": str(scores.get("run_id", "")),
                "scores_path": str(path),
                "skeleton_step_max": _optional_int(
                    scores.get("repair_denoise_skeleton_max_step")
                ),
                "spent_count": sum(1 for row in gate_rows if row.get("should_run") is True),
                "spent_tasks": [
                    str(row.get("task_id", ""))
                    for row in gate_rows
                    if row.get("should_run") is True
                ],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            _optional_int(row.get("skeleton_step_max")) is not None,
            _optional_int(row.get("skeleton_step_max")) or 0,
            str(row.get("run_id", "")),
        ),
    )


def _reason_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason", ""))
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _positive_reference_delta(row: dict[str, object], promotion_margin: float) -> float:
    reference_score = _optional_float(row.get("reference_repair_score"))
    if reference_score is None:
        return 0.0
    delta = reference_score - _no_repair_baseline_score(row)
    return delta if delta > promotion_margin else 0.0


def _no_repair_baseline_score(row: dict[str, object]) -> float:
    baseline_score = _optional_float(row.get("no_repair_baseline_score"))
    if baseline_score is not None:
        return baseline_score
    return _float(row.get("source_task_score"))


def _f1(productive: int, wasted: int, missed: int) -> float | None:
    precision = productive / (productive + wasted) if productive + wasted else None
    recall = productive / (productive + missed) if productive + missed else None
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return 2 * precision * recall / (precision + recall)


def _gate_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_No gates._"
    lines = [
        (
            "| Score | Cost | Lift/Extra | Spent | Prod | Waste | Miss | Q< | Gap | "
            "Coverage | Skeleton <= | Tasks |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{_format_float(row.get('selected_score'))} | "
            f"{_format_float(row.get('relative_generation_budget'))} | "
            f"{_format_float(row.get('lift_per_extra_generation_vs_greedy'))} | "
            f"{int(row.get('spent_count', 0))} | "
            f"{int(row.get('productive_spend_count', 0))} | "
            f"{int(row.get('wasted_spend_count', 0))} | "
            f"{int(row.get('missed_repair_count', 0))} | "
            f"{_format_float(row.get('quality_threshold'))} | "
            f"{int(row.get('gap_min', 0))}-{int(row.get('gap_max', 0))} | "
            f"{_format_float(row.get('coverage_min'))}-{_format_float(row.get('coverage_max'))} | "
            f"{_format_optional_int(row.get('skeleton_step_max'))} | "
            f"{', '.join(str(task) for task in row.get('spent_tasks', []))} |"
        )
    return "\n".join(lines)


def _confirmation_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_No fresh phase-window confirmations supplied._"
    lines = [
        (
            "| Run | Policy | Score | Cost | Generations | Skeleton <= | Spent | Late Skips | "
            "Delta Fixed | Delta Random | Tasks |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('run_id', '')}` | "
            f"`{row.get('repair_pack', '')}` | "
            f"{_format_float(row.get('repair_score'))} | "
            f"{_format_float(row.get('relative_generation_budget'))} | "
            f"{int(row.get('all_generation_count', 0))} | "
            f"{_format_optional_int(row.get('skeleton_step_max'))} | "
            f"{int(row.get('spent_count', 0))} | "
            f"{int(row.get('late_skeleton_skip_count', 0))} | "
            f"{_format_float(row.get('delta_vs_fixed'))} | "
            f"{_format_float(row.get('delta_vs_random'))} | "
            f"{', '.join(str(task) for task in row.get('spent_tasks', []))} |"
        )
    return "\n".join(lines)


def _same_gate(left: dict[str, object], right: dict[str, object]) -> bool:
    return (
        abs(_float(left.get("coverage_max")) - _float(right.get("coverage_max"))) < 1e-12
        and abs(_float(left.get("coverage_min")) - _float(right.get("coverage_min"))) < 1e-12
        and int(left.get("gap_max", -1)) == int(right.get("gap_max", -2))
        and int(left.get("gap_min", -1)) == int(right.get("gap_min", -2))
        and abs(_float(left.get("quality_threshold")) - _float(right.get("quality_threshold"))) < 1e-12
        and _optional_int(left.get("skeleton_step_max")) == _optional_int(right.get("skeleton_step_max"))
        and int(left.get("source_min_chars", -1)) == int(right.get("source_min_chars", -2))
    )


def _same_score_cost(left: dict[str, object], right: dict[str, object]) -> bool:
    return (
        abs(_float(left.get("selected_score")) - _float(right.get("selected_score"))) < 1e-12
        and abs(
            _float(left.get("relative_generation_budget"))
            - _float(right.get("relative_generation_budget"))
        )
        < 1e-12
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_optional_ints(value: str) -> list[int | None]:
    parsed: list[int | None] = []
    for item in value.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"none", "null", "off"}:
            parsed.append(None)
        else:
            parsed.append(int(token))
    return parsed


def _path_csv(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def _mean(values: Any) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, int | float)]
    if not numbers:
        return None
    return mean(numbers)


def _float(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return default


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and not isinstance(value, bool) and value.is_integer():
        return int(value)
    return None


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _format_float(value: object) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return ""
    return f"{float(value):.6f}"


def _format_optional_int(value: object) -> str:
    integer = _optional_int(value)
    return "" if integer is None else str(integer)


def main() -> int:
    args = parse_args()
    sweep = build_repairability_geometry_sweep(
        audit_path=args.audit,
        quality_thresholds=_parse_floats(args.quality_thresholds),
        gap_mins=_parse_ints(args.gap_mins),
        gap_maxs=_parse_ints(args.gap_maxs),
        coverage_mins=_parse_floats(args.coverage_mins),
        coverage_maxs=_parse_floats(args.coverage_maxs),
        skeleton_step_maxs=_parse_optional_ints(args.skeleton_step_maxs),
        phase_window_confirmation_score_paths=_path_csv(args.phase_window_confirmation_scores),
        source_min_chars=args.source_min_chars,
        promotion_margin=args.promotion_margin,
        base_generation_budget=args.base_generation_budget,
        repair_generation_budget=args.repair_generation_budget,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(sweep, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(render_markdown(sweep), encoding="utf-8")
    print(
        json.dumps(
            {
                "frontier": sweep["summary"]["frontier_count"],
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "sweep_points": sweep["summary"]["sweep_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
