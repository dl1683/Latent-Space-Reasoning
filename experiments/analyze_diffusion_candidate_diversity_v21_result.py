"""Summarize the v21 candidate-diversity result against the frozen gates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_FREEZE = Path("eval_results/diffusion_language/candidate_diversity_v21_freeze.json")
DEFAULT_SCORES = Path("eval_results/diffusion_language/candidate_diversity_v21_label_scores.json")
DEFAULT_TARGETS = Path("eval_results/diffusion_language/candidate_diversity_v21_targets.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/candidate_diversity_v21_result.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_CANDIDATE_DIVERSITY_V21_RESULT.md")
DEFAULT_SELECTED_COSTS = (0.0, 0.001, 0.005, 0.01, 0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_result_summary(
        freeze_path=args.freeze,
        scores_path=args.scores,
        targets_path=args.targets,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": _dict(result.get("decision")).get("status", ""),
                "json_output": str(args.json_output),
                "positive_count": _dict(result.get("summary")).get("positive_count", 0),
                "report_output": str(args.report_output),
                "selected_waste_count": _dict(result.get("summary")).get("selected_waste_count", 0),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_result_summary(
    *,
    freeze_path: Path,
    scores_path: Path,
    targets_path: Path,
) -> dict[str, object]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    rows = _list_of_dicts(targets.get("rows"))
    comparison_rows = _list_of_dicts(scores.get("comparison_rows"))
    comparison_by_task = {str(row.get("task_id", "")): row for row in comparison_rows}

    selected_rows = []
    for row in rows:
        comparison = _dict(comparison_by_task.get(str(row.get("task_id", ""))))
        if str(comparison.get("repair_selection_reason", "")).startswith("max_generated_repair_value_v1"):
            selected_repair = str(comparison.get("repair_control", ""))
            if str(row.get("repair", "")) == selected_repair:
                selected_rows.append(row)

    positives = [row for row in rows if _float(row.get("candidate_lift_vs_trajectory")) > 0.0]
    selected_positives = [
        row for row in selected_rows if _float(row.get("candidate_lift_vs_trajectory")) > 0.0
    ]
    selected_waste = [
        row for row in selected_rows if _float(row.get("candidate_lift_vs_trajectory")) <= 0.0
    ]
    selected_positive_lift = sum(_float(row.get("candidate_lift_vs_trajectory")) for row in selected_positives)
    repair_covered_count = int(_float(_dict(_dict(scores.get("arms")).get("repair_selected")).get("count")))
    gates = _dict(freeze.get("conclusive_result_gates"))
    selected_cost_sweep = _selected_cost_sweep(
        repair_task_delta=_float(scores.get("repair_task_delta_vs_evolved")),
        selected_count=len(selected_rows),
        repair_covered_count=repair_covered_count,
    )

    return {
        "candidate_name_summary": _candidate_name_summary(rows, selected_rows),
        "decision": _decision(
            positive_count=len(positives),
            selected_waste_count=len(selected_waste),
            selected_cost_sweep=selected_cost_sweep,
            gates=gates,
        ),
        "generated_by": "experiments/analyze_diffusion_candidate_diversity_v21_result.py",
        "inputs": {
            "freeze": str(freeze_path),
            "scores": str(scores_path),
            "targets": str(targets_path),
        },
        "schema": "diffusion_candidate_diversity_v21_result.v1",
        "selected_cost_sweep": selected_cost_sweep,
        "selected_rows": _slim_rows(selected_rows),
        "summary": {
            "all_generation_count": scores.get("all_generation_count"),
            "candidate_aware_promotion_error_count": _dict(targets.get("summary")).get(
                "candidate_aware_promotion_error_count", 0
            ),
            "positive_count": len(positives),
            "positive_tasks": [str(row.get("task_id", "")) for row in positives],
            "repair_task_delta_per_extra_generation_vs_evolved": _float(
                scores.get("repair_task_delta_per_extra_generation_vs_evolved")
            ),
            "repair_task_delta_vs_evolved": _float(scores.get("repair_task_delta_vs_evolved")),
            "repair_task_delta_vs_fixed": _float(scores.get("repair_task_delta_vs_fixed")),
            "repair_task_delta_vs_random": _float(scores.get("repair_task_delta_vs_random")),
            "run_id": str(scores.get("run_id", "")),
            "selected_count": len(selected_rows),
            "selected_positive_count": len(selected_positives),
            "selected_positive_lift": selected_positive_lift,
            "selected_positive_tasks": [str(row.get("task_id", "")) for row in selected_positives],
            "selected_waste_count": len(selected_waste),
            "selected_waste_tasks": [str(row.get("task_id", "")) for row in selected_waste],
            "target_count": len(rows),
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    decision = _dict(result.get("decision"))
    lines = [
        "# Diffusion Candidate Diversity V21 Result",
        "",
        "This file is generated by `experiments/analyze_diffusion_candidate_diversity_v21_result.py`.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision.get('status')}`",
        f"- Reason: {decision.get('reason')}",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Full generations: `{summary.get('all_generation_count')}`",
        "",
        "## Summary",
        "",
        f"- Target rows: `{summary.get('target_count')}`",
        f"- Generated-positive candidate rows: `{summary.get('positive_count')}`",
        f"- Positive tasks: {_join_tasks(summary.get('positive_tasks'))}",
        f"- Selected repair-pool rows: `{summary.get('selected_count')}`",
        f"- Selected-positive rows: `{summary.get('selected_positive_count')}`",
        f"- Selected-positive tasks: {_join_tasks(summary.get('selected_positive_tasks'))}",
        f"- Selected no-lift/negative rows: `{summary.get('selected_waste_count')}`",
        f"- Selected no-lift/negative tasks: {_join_tasks(summary.get('selected_waste_tasks'))}",
        f"- Repair task delta vs fixed: `{_format_float(summary.get('repair_task_delta_vs_fixed'))}`",
        f"- Repair task delta vs random: `{_format_float(summary.get('repair_task_delta_vs_random'))}`",
        f"- Repair task delta vs evolved: `{_format_float(summary.get('repair_task_delta_vs_evolved'))}`",
        f"- Repair task delta per extra generation: `{_format_float(summary.get('repair_task_delta_per_extra_generation_vs_evolved'))}`",
        "",
        "## Candidate Names",
        "",
        "| Candidate | Rows | Positives | Selected | Selected Positives | Selected Waste | Positive Tasks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in _list_of_dicts(result.get("candidate_name_summary")):
        lines.append(
            "| "
            f"`{row.get('repair')}` | "
            f"{row.get('row_count')} | "
            f"{row.get('positive_count')} | "
            f"{row.get('selected_count')} | "
            f"{row.get('selected_positive_count')} | "
            f"{row.get('selected_waste_count')} | "
            f"{_join_tasks(row.get('positive_tasks'))} |"
        )
    lines.extend(
        [
            "",
            "## Selected Rows",
            "",
            "| Task | Candidate | Lift vs Trajectory | Lift vs Source | Selected Class |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in _list_of_dicts(result.get("selected_rows")):
        lines.append(
            "| "
            f"`{row.get('task_id')}` | "
            f"`{row.get('repair')}` | "
            f"{_format_float(row.get('candidate_lift_vs_trajectory'))} | "
            f"{_format_float(row.get('candidate_lift_vs_source'))} | "
            f"`{row.get('selected_class')}` |"
        )
    lines.extend(
        [
            "",
            "## Selected-Output Cost Sweep",
            "",
            "| Selected Cost | Net Repair Delta |",
            "| ---: | ---: |",
        ]
    )
    for row in _list_of_dicts(result.get("selected_cost_sweep")):
        lines.append(
            "| "
            f"{_format_float(row.get('selected_cost'))} | "
            f"{_format_float(row.get('net_repair_delta_vs_evolved'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "V21 answers the upstream availability question positively: the two-candidate "
                "pool creates fresh generated-positive rows after v20 had none. It does not "
                "validate live broadening, because the unchanged selector also selected "
                "zero/negative-lift rows. The correct next step is selector/cost repair over "
                "this broader pool, not promotion of the wider pool as-is."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _candidate_name_summary(
    rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected_keys = {(str(row.get("task_id", "")), str(row.get("repair", ""))) for row in selected_rows}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("repair", ""))].append(row)
    summaries = []
    for repair in sorted(grouped):
        repair_rows = grouped[repair]
        positives = [row for row in repair_rows if _float(row.get("candidate_lift_vs_trajectory")) > 0.0]
        selected = [
            row for row in repair_rows if (str(row.get("task_id", "")), str(row.get("repair", ""))) in selected_keys
        ]
        selected_positives = [
            row for row in selected if _float(row.get("candidate_lift_vs_trajectory")) > 0.0
        ]
        selected_waste = [
            row for row in selected if _float(row.get("candidate_lift_vs_trajectory")) <= 0.0
        ]
        summaries.append(
            {
                "positive_count": len(positives),
                "positive_tasks": [str(row.get("task_id", "")) for row in positives],
                "repair": repair,
                "row_count": len(repair_rows),
                "selected_count": len(selected),
                "selected_positive_count": len(selected_positives),
                "selected_waste_count": len(selected_waste),
            }
        )
    return summaries


def _decision(
    *,
    positive_count: int,
    selected_waste_count: int,
    selected_cost_sweep: list[dict[str, object]],
    gates: dict[str, object],
) -> dict[str, object]:
    if positive_count < int(_float(gates.get("minimum_generated_positive_count"))):
        return {
            "reason": "No fresh generated-positive repair candidates appeared.",
            "status": "availability_negative",
        }
    if selected_waste_count > int(_float(gates.get("maximum_selected_no_lift_rows"))):
        return {
            "reason": (
                "Candidate diversity creates positives, but the unchanged hook also selects "
                "zero/negative-lift rows."
            ),
            "status": "availability_positive_selector_failed",
        }
    if min(_float(row.get("net_repair_delta_vs_evolved")) for row in selected_cost_sweep) <= 0.0:
        return {
            "reason": "Selected-output cost erases the measured repair lift.",
            "status": "availability_positive_cost_failed",
        }
    return {
        "reason": "Candidate diversity creates selected positives without selected waste after cost.",
        "status": "validated",
    }


def _selected_cost_sweep(
    *,
    repair_task_delta: float,
    selected_count: int,
    repair_covered_count: int,
) -> list[dict[str, object]]:
    denominator = max(repair_covered_count, 1)
    return [
        {
            "net_repair_delta_vs_evolved": repair_task_delta - (selected_cost * selected_count / denominator),
            "selected_cost": selected_cost,
        }
        for selected_cost in DEFAULT_SELECTED_COSTS
    ]


def _slim_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    slim = []
    for row in rows:
        lift = _float(row.get("candidate_lift_vs_trajectory"))
        if lift > 0.0:
            selected_class = "positive"
        elif lift == 0.0:
            selected_class = "zero_lift"
        else:
            selected_class = "negative_lift"
        slim.append(
            {
                "candidate_lift_vs_source": _float(row.get("candidate_lift_vs_source")),
                "candidate_lift_vs_trajectory": lift,
                "repair": str(row.get("repair", "")),
                "selected_class": selected_class,
                "task_id": str(row.get("task_id", "")),
            }
        )
    return slim


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _join_tasks(value: object) -> str:
    items = [str(item) for item in value] if isinstance(value, list) else []
    if not items:
        return "`none`"
    return ", ".join(f"`{item}`" for item in items)


if __name__ == "__main__":
    raise SystemExit(main())
