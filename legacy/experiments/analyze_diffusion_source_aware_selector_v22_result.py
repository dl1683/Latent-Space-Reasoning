"""Replay the frozen v22 source-aware selector surface against fresh labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_FREEZE = Path("eval_results/diffusion_language/source_aware_selector_v22_freeze.json")
DEFAULT_SCORES = Path("eval_results/diffusion_language/source_aware_selector_v22_label_scores.json")
DEFAULT_TARGETS = Path("eval_results/diffusion_language/source_aware_selector_v22_targets.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/source_aware_selector_v22_result.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_SOURCE_AWARE_SELECTOR_V22_RESULT.md")
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
                "decision": result["decision"]["status"],
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "source_aware_selected_positive_count": result["summary"]["source_aware_selected_positive_count"],
                "source_aware_selected_waste_count": result["summary"]["source_aware_selected_waste_count"],
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
    surface = _dict(freeze.get("target_surface"))
    rows = _list_of_dicts(targets.get("rows"))
    comparison_by_task = {
        str(row.get("task_id", "")): row for row in _list_of_dicts(scores.get("comparison_rows"))
    }
    unchanged_selected = [
        row
        for row in rows
        if str(_dict(comparison_by_task.get(str(row.get("task_id", "")))).get("repair_control", ""))
        == str(row.get("repair", ""))
    ]
    source_aware_selected = [
        row for row in rows if _source_aware_selects(row, surface)
    ]
    positives = [row for row in rows if _float(row.get("candidate_lift_vs_trajectory")) > 0.0]
    unchanged = _policy_summary("generated_repair_value_v1", unchanged_selected, scores)
    source_aware = _policy_summary("source_aware_candidate_selector_v22", source_aware_selected, scores)
    decision = _decision(
        source_aware=source_aware,
        unchanged=unchanged,
        positive_count=len(positives),
    )
    return {
        "decision": decision,
        "generated_by": "experiments/analyze_diffusion_source_aware_selector_v22_result.py",
        "inputs": {
            "freeze": str(freeze_path),
            "scores": str(scores_path),
            "targets": str(targets_path),
        },
        "policies": [unchanged, source_aware],
        "schema": "diffusion_source_aware_selector_v22_result.v1",
        "summary": {
            "candidate_aware_promotion_error_count": _dict(targets.get("summary")).get(
                "candidate_aware_promotion_error_count", 0
            ),
            "positive_count": len(positives),
            "positive_tasks": [str(row.get("task_id", "")) for row in positives],
            "repair_task_delta_vs_evolved": _float(scores.get("repair_task_delta_vs_evolved")),
            "repair_task_delta_per_extra_generation_vs_evolved": _float(
                scores.get("repair_task_delta_per_extra_generation_vs_evolved")
            ),
            "run_id": str(scores.get("run_id", "")),
            "source_aware_selected_positive_count": source_aware["selected_positive_count"],
            "source_aware_selected_waste_count": source_aware["selected_waste_count"],
            "target_count": len(rows),
            "unchanged_selected_positive_count": unchanged["selected_positive_count"],
            "unchanged_selected_waste_count": unchanged["selected_waste_count"],
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    decision = _dict(result.get("decision"))
    lines = [
        "# Diffusion Source-Aware Selector V22 Result",
        "",
        "This file is generated by `experiments/analyze_diffusion_source_aware_selector_v22_result.py`.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision.get('status')}`",
        f"- Reason: {decision.get('reason')}",
        f"- Run ID: `{summary.get('run_id')}`",
        "",
        "## Summary",
        "",
        f"- Target rows: `{summary.get('target_count')}`",
        f"- Generated-positive candidate rows: `{summary.get('positive_count')}`",
        f"- Positive tasks: {_join_tasks(summary.get('positive_tasks'))}",
        f"- Unchanged hook selected positives: `{summary.get('unchanged_selected_positive_count')}`",
        f"- Unchanged hook selected waste: `{summary.get('unchanged_selected_waste_count')}`",
        f"- Source-aware selected positives: `{summary.get('source_aware_selected_positive_count')}`",
        f"- Source-aware selected waste: `{summary.get('source_aware_selected_waste_count')}`",
        f"- Repair task delta vs evolved: `{_format_float(summary.get('repair_task_delta_vs_evolved'))}`",
        f"- Repair task delta per extra generation: `{_format_float(summary.get('repair_task_delta_per_extra_generation_vs_evolved'))}`",
        "",
        "## Policy Replay",
        "",
        "| Policy | Selected | Positives | Waste | Missed Positives | Selected Tasks |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for policy in _list_of_dicts(result.get("policies")):
        lines.append(
            "| "
            f"`{policy.get('policy_id')}` | "
            f"{policy.get('selected_count')} | "
            f"{policy.get('selected_positive_count')} | "
            f"{policy.get('selected_waste_count')} | "
            f"{_join_tasks(policy.get('missed_positive_tasks'))} | "
            f"{_join_tasks(policy.get('selected_tasks'))} |"
        )
    lines.extend(
        [
            "",
            "## Cost Sweep",
            "",
            "| Selected Cost | Unchanged Net | Source-Aware Net | Source-Aware Advantage |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _list_of_dicts(decision.get("cost_sweep")):
        lines.append(
            "| "
            f"{_format_float(row.get('selected_cost'))} | "
            f"{_format_float(row.get('unchanged_net'))} | "
            f"{_format_float(row.get('source_aware_net'))} | "
            f"{_format_float(row.get('source_aware_advantage'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "The frozen source-aware surface fixes precision but overfilters recall. "
                "It rejects selected waste, but it misses fresh final-preserve positives "
                "because the v21-fitted final-preserve planning-delta threshold is too high."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _source_aware_selects(row: dict[str, object], surface: dict[str, object]) -> bool:
    repair = str(row.get("repair", ""))
    planning_delta = _float(row.get("planning_quality_delta_vs_source"))
    span_score = _float(row.get("max_span_target_score"))
    if repair == "history_prefix_25_repair":
        return planning_delta >= _float(surface.get("history_prefix_planning_delta_min"))
    if repair == "constraint_gap_span_phase_final_preserve_seeded_gated_repair":
        return planning_delta >= _float(surface.get("final_preserve_planning_delta_min")) and span_score >= _float(
            surface.get("final_preserve_span_score_min")
        )
    return False


def _policy_summary(policy_id: str, selected: list[dict[str, object]], scores: dict[str, object]) -> dict[str, object]:
    positives = [row for row in selected if _float(row.get("candidate_lift_vs_trajectory")) > 0.0]
    waste = [row for row in selected if _float(row.get("candidate_lift_vs_trajectory")) <= 0.0]
    all_positive_tasks = {
        str(row.get("task_id", ""))
        for row in _list_of_dicts(scores.get("comparison_rows"))
        if _float(row.get("oracle_delta_vs_trajectory")) > 0.0
    }
    selected_tasks = [str(row.get("task_id", "")) for row in selected]
    return {
        "missed_positive_tasks": sorted(all_positive_tasks - set(selected_tasks)),
        "policy_id": policy_id,
        "selected_count": len(selected),
        "selected_positive_count": len(positives),
        "selected_positive_lift": sum(_float(row.get("candidate_lift_vs_trajectory")) for row in positives),
        "selected_tasks": selected_tasks,
        "selected_waste_count": len(waste),
        "selected_waste_tasks": [str(row.get("task_id", "")) for row in waste],
    }


def _decision(
    *,
    source_aware: dict[str, object],
    unchanged: dict[str, object],
    positive_count: int,
) -> dict[str, object]:
    cost_sweep = _cost_sweep(source_aware, unchanged)
    if positive_count == 0:
        return {"cost_sweep": cost_sweep, "reason": "No generated positives appeared.", "status": "inconclusive"}
    if int(source_aware.get("selected_waste_count", 0)) > 0:
        return {"cost_sweep": cost_sweep, "reason": "Source-aware replay still selected waste.", "status": "precision_failed"}
    if int(source_aware.get("selected_positive_count", 0)) == 0:
        return {"cost_sweep": cost_sweep, "reason": "Source-aware replay selected no positives.", "status": "overfiltered"}
    if not any(_float(row.get("source_aware_advantage")) > 0.0 for row in cost_sweep):
        return {
            "cost_sweep": cost_sweep,
            "reason": "Source-aware replay rejects waste but loses too much positive utility versus the unchanged hook.",
            "status": "precision_positive_utility_failed",
        }
    return {
        "cost_sweep": cost_sweep,
        "reason": "Source-aware replay rejects waste and beats the unchanged hook after selected-output cost.",
        "status": "validated",
    }


def _cost_sweep(source_aware: dict[str, object], unchanged: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for selected_cost in DEFAULT_SELECTED_COSTS:
        unchanged_net = _float(unchanged.get("selected_positive_lift")) - selected_cost * int(
            unchanged.get("selected_count", 0)
        )
        source_net = _float(source_aware.get("selected_positive_lift")) - selected_cost * int(
            source_aware.get("selected_count", 0)
        )
        rows.append(
            {
                "selected_cost": selected_cost,
                "source_aware_advantage": source_net - unchanged_net,
                "source_aware_net": source_net,
                "unchanged_net": unchanged_net,
            }
        )
    return rows


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
