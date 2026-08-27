"""Evaluate fitted diffusion spend head on an independent transfer slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ALL_REPAIRABLE_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_transfer_all_repairable_frontier_v1_scores.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/diffusion_independent_spend_transfer_eval.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_INDEPENDENT_SPEND_TRANSFER.md")
DEFAULT_SOURCE_QUALITY_MAX = 0.301429
DEFAULT_LEARNED_SOURCE_QUALITY_MAX = 0.256429
DEFAULT_LEARNED_PROMPT_GAP_MAX = 8
DEFAULT_CALIBRATED_BLOCKED_PROMPT_GAP = 7
DEFAULT_PROFIT_LIFT_MIN = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-repairable-scores", type=Path, default=DEFAULT_ALL_REPAIRABLE_SCORES)
    parser.add_argument("--source-quality-max", type=float, default=DEFAULT_SOURCE_QUALITY_MAX)
    parser.add_argument(
        "--learned-source-quality-max",
        type=float,
        default=DEFAULT_LEARNED_SOURCE_QUALITY_MAX,
    )
    parser.add_argument(
        "--learned-prompt-gap-max",
        type=int,
        default=DEFAULT_LEARNED_PROMPT_GAP_MAX,
    )
    parser.add_argument(
        "--calibrated-blocked-prompt-gap",
        type=int,
        default=DEFAULT_CALIBRATED_BLOCKED_PROMPT_GAP,
    )
    parser.add_argument("--profit-lift-min", type=float, default=DEFAULT_PROFIT_LIFT_MIN)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation = evaluate_independent_spend_transfer(
        all_repairable_scores_path=args.all_repairable_scores,
        calibrated_blocked_prompt_gap=args.calibrated_blocked_prompt_gap,
        learned_prompt_gap_max=args.learned_prompt_gap_max,
        learned_source_quality_max=args.learned_source_quality_max,
        profit_lift_min=args.profit_lift_min,
        source_quality_max=args.source_quality_max,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(evaluation), encoding="utf-8")
    print(
        json.dumps(
            {
                "decomposed_error_count": _dict(evaluation.get("summary")).get(
                    "decomposed_error_count", 0
                ),
                "calibrated_availability_error_count": _dict(evaluation.get("summary")).get(
                    "calibrated_availability_error_count", 0
                ),
                "json_output": str(args.json_output),
                "learned_availability_error_count": _dict(evaluation.get("summary")).get(
                    "learned_availability_error_count", 0
                ),
                "report_output": str(args.report_output),
                "single_repairability_error_count": _dict(evaluation.get("summary")).get(
                    "single_repairability_error_count", 0
                ),
                "target_count": _dict(evaluation.get("summary")).get("target_count", 0),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def evaluate_independent_spend_transfer(
    *,
    all_repairable_scores_path: Path,
    calibrated_blocked_prompt_gap: int = DEFAULT_CALIBRATED_BLOCKED_PROMPT_GAP,
    learned_prompt_gap_max: int = DEFAULT_LEARNED_PROMPT_GAP_MAX,
    learned_source_quality_max: float = DEFAULT_LEARNED_SOURCE_QUALITY_MAX,
    profit_lift_min: float = DEFAULT_PROFIT_LIFT_MIN,
    source_quality_max: float = DEFAULT_SOURCE_QUALITY_MAX,
) -> dict[str, object]:
    scores = json.loads(all_repairable_scores_path.read_text(encoding="utf-8"))
    comparison_by_task = {
        str(row.get("task_id", "")): row for row in _list_of_dicts(scores.get("comparison_rows"))
    }
    rows = [
        _target_row(
            gate_row,
            calibrated_blocked_prompt_gap=calibrated_blocked_prompt_gap,
            comparison_by_task=comparison_by_task,
            learned_prompt_gap_max=learned_prompt_gap_max,
            learned_source_quality_max=learned_source_quality_max,
            profit_lift_min=profit_lift_min,
            source_quality_max=source_quality_max,
        )
        for gate_row in _list_of_dicts(scores.get("repair_spend_gate_rows"))
        if str(gate_row.get("task_id", "")).startswith("plan_")
    ]
    return {
        "generated_by": "experiments/evaluate_diffusion_independent_spend_transfer.py",
        "inputs": {"all_repairable_scores": str(all_repairable_scores_path)},
        "calibrated_blocked_prompt_gap": calibrated_blocked_prompt_gap,
        "learned_prompt_gap_max": learned_prompt_gap_max,
        "learned_source_quality_max": learned_source_quality_max,
        "rows": rows,
        "schema": "diffusion_independent_spend_transfer_eval.v1",
        "profit_lift_min": profit_lift_min,
        "source_quality_max": source_quality_max,
        "summary": _summary(rows),
    }


def render_markdown(evaluation: dict[str, object]) -> str:
    summary = _dict(evaluation.get("summary"))
    lines = [
        "# Diffusion Independent Spend Transfer",
        "",
        "This file is generated by `experiments/evaluate_diffusion_independent_spend_transfer.py`.",
        (
            "It evaluates the fitted diffusion spend head on an independent planning "
            "slice by deriving repair-value labels from an all-repairable GPU run."
        ),
        "",
        "## Summary",
        "",
        f"- All-repairable scores: `{_dict(evaluation.get('inputs')).get('all_repairable_scores', '')}`",
        f"- Profit lift threshold: `{_format_float(evaluation.get('profit_lift_min'))}`",
        f"- Target rows: `{summary.get('target_count', 0)}`",
        f"- Profitable repair rows: `{summary.get('profitable_count', 0)}`",
        f"- Single repairability errors: `{summary.get('single_repairability_error_count', 0)}`",
        f"- Decomposed spend-head errors: `{summary.get('decomposed_error_count', 0)}`",
        (
            "- Trajectory-relative spend-head errors: "
            f"`{summary.get('trajectory_relative_error_count', 0)}`"
        ),
        (
            "- Learned availability predictor errors: "
            f"`{summary.get('learned_availability_error_count', 0)}`"
        ),
        (
            "- Calibrated availability predictor errors: "
            f"`{summary.get('calibrated_availability_error_count', 0)}`"
        ),
        f"- Absolute error reduction: `{summary.get('absolute_error_reduction', 0)}`",
        f"- Relative error reduction: `{_format_float(summary.get('relative_error_reduction'))}`",
        f"- Decomposed selected tasks: {_join_tasks(summary.get('decomposed_selected_tasks'))}",
        (
            "- Trajectory-relative selected tasks: "
            f"{_join_tasks(summary.get('trajectory_relative_selected_tasks'))}"
        ),
        (
            "- Learned availability selected tasks: "
            f"{_join_tasks(summary.get('learned_availability_selected_tasks'))}"
        ),
        (
            "- Calibrated availability selected tasks: "
            f"{_join_tasks(summary.get('calibrated_availability_selected_tasks'))}"
        ),
        f"- Profitable tasks: {_join_tasks(summary.get('profitable_tasks'))}",
        "",
        "## Rows",
        "",
        (
            "| Task | Profitable | Repair Lift | Source Quality | Gap | First Step | "
            "Selected Lift | Source-Selected Delta | Single Repairability | Decomposed Spend | "
            "Trajectory-Relative Spend | Learned Availability | Calibrated Availability | "
            "Decomposed Error | Trajectory-Relative Error | Learned Error | Calibrated Error | "
            "Reason |"
        ),
        (
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | "
            "--- | --- | --- | --- | --- | --- | --- | --- |"
        ),
    ]
    for row in _list_of_dicts(evaluation.get("rows")):
        lines.append(
            "| "
            f"`{row.get('task_id', '')}` | "
            f"{bool(row.get('profitable'))} | "
            f"{_format_float(row.get('repair_lift'))} | "
            f"{_format_float(row.get('source_quality'))} | "
            f"{int(row.get('prompt_gap_count', 0))} | "
            f"{_format_optional(row.get('first_repairable_step'))} | "
            f"{_format_float(row.get('selected_repair_lift'))} | "
            f"{_format_float(row.get('source_task_delta_vs_trajectory'))} | "
            f"{bool(row.get('single_repairability_prediction'))} | "
            f"{bool(row.get('decomposed_prediction'))} | "
            f"{bool(row.get('trajectory_relative_prediction'))} | "
            f"{bool(row.get('learned_availability_prediction'))} | "
            f"{bool(row.get('calibrated_availability_prediction'))} | "
            f"{bool(row.get('decomposed_error'))} | "
            f"{bool(row.get('trajectory_relative_error'))} | "
            f"{bool(row.get('learned_availability_error'))} | "
            f"{bool(row.get('calibrated_availability_error'))} | "
            f"`{row.get('gate_reason', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "This is an independent spend-head transfer check, not a full four-head "
                "source/retention/realization transfer audit. It tests whether the fitted "
                "marginal-value spend rule remains better than spending on every "
                "repairable denoise skeleton when the planning prompts are new. The "
                "trajectory-relative head adds one more information channel: whether "
                "the repair source is already below the selected trajectory state. "
                "Labels come from repair-oracle lift, so a repair can be counted as "
                "available even when the selected repair arm is held back by a "
                "promotion margin. The learned availability predictor reuses the "
                "v3-fitted thresholds exactly: `prompt_gap_count <= 8`, "
                "`source_quality <= 0.256429`, and "
                "`source_task_delta_vs_trajectory >= 0`. The calibrated "
                "availability predictor removes the failed absolute source-quality "
                "ceiling and uses the v3/v4 boundary: repairable source, "
                "`prompt_gap_count != 7`, and "
                "`source_task_delta_vs_trajectory >= 0`."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _target_row(
    gate_row: dict[str, object],
    *,
    calibrated_blocked_prompt_gap: int,
    comparison_by_task: dict[str, dict[str, object]],
    learned_prompt_gap_max: int,
    learned_source_quality_max: float,
    profit_lift_min: float,
    source_quality_max: float,
) -> dict[str, object]:
    task_id = str(gate_row.get("task_id", ""))
    comparison = comparison_by_task.get(task_id, {})
    single_prediction = bool(gate_row.get("should_run"))
    selected_repair_lift = _float(comparison.get("repair_delta_vs_evolved"))
    repair_lift = _repair_candidate_lift(comparison)
    trajectory_task_score = _float(comparison.get("trajectory_task_score"))
    source_task_delta_vs_trajectory = _float(gate_row.get("source_task_score")) - trajectory_task_score
    if not single_prediction:
        repair_lift = 0.0
        selected_repair_lift = 0.0
    profitable = repair_lift > profit_lift_min
    source_quality = _float(gate_row.get("source_quality"))
    prompt_gap_count = int(_float(gate_row.get("prompt_gap_count")))
    decomposed_prediction = single_prediction and source_quality <= source_quality_max
    trajectory_relative_prediction = (
        decomposed_prediction and source_task_delta_vs_trajectory >= 0.0
    )
    learned_availability_prediction = (
        single_prediction
        and prompt_gap_count <= learned_prompt_gap_max
        and source_quality <= learned_source_quality_max
        and source_task_delta_vs_trajectory >= 0.0
    )
    calibrated_availability_prediction = (
        single_prediction
        and prompt_gap_count != calibrated_blocked_prompt_gap
        and source_task_delta_vs_trajectory >= 0.0
    )
    return {
        "calibrated_availability_error": calibrated_availability_prediction != profitable,
        "calibrated_availability_prediction": calibrated_availability_prediction,
        "calibrated_blocked_prompt_gap": calibrated_blocked_prompt_gap,
        "decomposed_error": decomposed_prediction != profitable,
        "decomposed_prediction": decomposed_prediction,
        "first_repairable_step": gate_row.get("first_repairable_denoise_skeleton_step"),
        "gate_reason": str(gate_row.get("reason", "")),
        "learned_availability_error": learned_availability_prediction != profitable,
        "learned_availability_prediction": learned_availability_prediction,
        "learned_prompt_gap_max": learned_prompt_gap_max,
        "learned_source_quality_max": learned_source_quality_max,
        "profitable": profitable,
        "prompt_gap_count": prompt_gap_count,
        "repair_lift": repair_lift,
        "selected_repair_lift": selected_repair_lift,
        "single_repairability_error": single_prediction != profitable,
        "single_repairability_prediction": single_prediction,
        "source_quality": source_quality,
        "source_quality_max": source_quality_max,
        "source_task_score": _float(gate_row.get("source_task_score")),
        "source_task_delta_vs_trajectory": source_task_delta_vs_trajectory,
        "task_id": task_id,
        "trajectory_relative_error": trajectory_relative_prediction != profitable,
        "trajectory_relative_prediction": trajectory_relative_prediction,
        "trajectory_task_score": trajectory_task_score,
    }


def _repair_candidate_lift(comparison: dict[str, object]) -> float:
    if "repair_delta_vs_evolved" in comparison:
        return _float(comparison.get("repair_delta_vs_evolved"))
    if "repair_task_score" in comparison and "trajectory_task_score" in comparison:
        return _float(comparison.get("repair_task_score")) - _float(
            comparison.get("trajectory_task_score")
        )
    return _float(comparison.get("repair_delta_vs_evolved"))


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    calibrated_availability_errors = sum(
        int(row.get("calibrated_availability_error", False)) for row in rows
    )
    decomposed_errors = sum(int(row.get("decomposed_error", False)) for row in rows)
    learned_availability_errors = sum(
        int(row.get("learned_availability_error", False)) for row in rows
    )
    single_errors = sum(int(row.get("single_repairability_error", False)) for row in rows)
    trajectory_relative_errors = sum(
        int(row.get("trajectory_relative_error", False)) for row in rows
    )
    absolute_reduction = single_errors - decomposed_errors
    return {
        "absolute_error_reduction": absolute_reduction,
        "calibrated_availability_absolute_error_reduction": (
            single_errors - calibrated_availability_errors
        ),
        "calibrated_availability_error_count": calibrated_availability_errors,
        "calibrated_availability_error_rate": (
            calibrated_availability_errors / len(rows) if rows else 0.0
        ),
        "calibrated_availability_selected_tasks": [
            str(row.get("task_id", ""))
            for row in rows
            if row.get("calibrated_availability_prediction")
        ],
        "decomposed_error_count": decomposed_errors,
        "decomposed_error_rate": decomposed_errors / len(rows) if rows else 0.0,
        "decomposed_selected_tasks": [
            str(row.get("task_id", "")) for row in rows if row.get("decomposed_prediction")
        ],
        "learned_availability_absolute_error_reduction": (
            single_errors - learned_availability_errors
        ),
        "learned_availability_error_count": learned_availability_errors,
        "learned_availability_error_rate": (
            learned_availability_errors / len(rows) if rows else 0.0
        ),
        "learned_availability_selected_tasks": [
            str(row.get("task_id", ""))
            for row in rows
            if row.get("learned_availability_prediction")
        ],
        "profitable_count": sum(int(row.get("profitable", False)) for row in rows),
        "profitable_tasks": [str(row.get("task_id", "")) for row in rows if row.get("profitable")],
        "relative_error_reduction": absolute_reduction / single_errors if single_errors else 0.0,
        "single_repairability_error_count": single_errors,
        "single_repairability_error_rate": single_errors / len(rows) if rows else 0.0,
        "target_count": len(rows),
        "trajectory_relative_absolute_error_reduction": (
            single_errors - trajectory_relative_errors
        ),
        "trajectory_relative_error_count": trajectory_relative_errors,
        "trajectory_relative_error_rate": (
            trajectory_relative_errors / len(rows) if rows else 0.0
        ),
        "trajectory_relative_selected_tasks": [
            str(row.get("task_id", ""))
            for row in rows
            if row.get("trajectory_relative_prediction")
        ],
    }


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


def _format_optional(value: object) -> str:
    if value is None:
        return ""
    return _format_float(value)


def _join_tasks(value: object) -> str:
    values = [str(item) for item in value] if isinstance(value, list) else []
    return ", ".join(f"`{item}`" for item in values) if values else "`none`"


if __name__ == "__main__":
    raise SystemExit(main())
