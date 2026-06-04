"""Fit and replay the frozen v25 learned candidate-selector proof obligation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SCORES = Path("eval_results/diffusion_language/learned_selector_v25_label_scores.json")
DEFAULT_TARGETS = Path("eval_results/diffusion_language/learned_selector_v25_targets.json")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/learned_selector_v25_result.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_LEARNED_SELECTOR_V25_RESULT.md")
TRAINING_TARGETS = {
    "v21": Path("eval_results/diffusion_language/candidate_diversity_v21_targets.json"),
    "v22": Path("eval_results/diffusion_language/source_aware_selector_v22_targets.json"),
    "v23": Path("eval_results/diffusion_language/asymmetric_filter_v23_targets.json"),
    "v24": Path("eval_results/diffusion_language/history_guard_v24_targets.json"),
}
SELECTED_COSTS = (0.0, 0.001, 0.005, 0.01, 0.02)
HISTORY_REPAIR = "history_prefix_25_repair"
FINAL_REPAIR = "constraint_gap_span_phase_final_preserve_seeded_gated_repair"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_result_summary(scores_path=args.scores, targets_path=args.targets)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"]["status"],
                "json_output": str(args.json_output),
                "learned_selected_positive_count": result["summary"]["learned_selected_positive_count"],
                "learned_selected_waste_count": result["summary"]["learned_selected_waste_count"],
                "report_output": str(args.report_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_result_summary(*, scores_path: Path, targets_path: Path) -> dict[str, object]:
    training_rows = _load_training_rows(TRAINING_TARGETS)
    model = _fit_model(training_rows)
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    rows = _list_of_dicts(targets.get("rows"))
    unchanged_selected = _unchanged_selected(rows, _list_of_dicts(scores.get("comparison_rows")))
    learned_selected = [row for row in rows if _model_selects(row, model)]
    policies = [
        _policy_summary("generated_repair_value_v1", unchanged_selected, rows),
        _policy_summary("candidate_row_value_model_v25", learned_selected, rows),
        _policy_summary("history_prefix_all_control", [row for row in rows if row.get("repair") == HISTORY_REPAIR], rows),
        _policy_summary("final_preserve_all_control", [row for row in rows if row.get("repair") == FINAL_REPAIR], rows),
        _policy_summary("asymmetric_repair_source_filter_v23", [row for row in rows if _v23_asymmetric_selects(row)], rows),
    ]
    unchanged = policies[0]
    learned = policies[1]
    decision = _decision(learned=learned, unchanged=unchanged, positive_count=_positive_count(rows))
    return {
        "decision": decision,
        "generated_by": "experiments/analyze_diffusion_learned_selector_v25_result.py",
        "inputs": {"scores": str(scores_path), "targets": str(targets_path)},
        "learned_model": model,
        "leave_one_slice_out": _leave_one_slice_out(training_rows),
        "policies": policies,
        "schema": "diffusion_learned_selector_v25_result.v1",
        "summary": {
            "all_generation_count": scores.get("all_generation_count"),
            "candidate_aware_promotion_error_count": _dict(targets.get("summary")).get(
                "candidate_aware_promotion_error_count", 0
            ),
            "learned_selected_positive_count": learned["selected_positive_count"],
            "learned_selected_waste_count": learned["selected_waste_count"],
            "oracle_headroom_vs_repair": _float(scores.get("oracle_headroom_vs_repair")),
            "positive_count": _positive_count(rows),
            "positive_tasks": sorted(
                {str(row.get("task_id", "")) for row in rows if _float(row.get("candidate_lift_vs_trajectory")) > 0.0}
            ),
            "repair_task_delta_per_extra_generation_vs_evolved": _float(
                scores.get("repair_task_delta_per_extra_generation_vs_evolved")
            ),
            "repair_task_delta_vs_evolved": _float(scores.get("repair_task_delta_vs_evolved")),
            "run_id": str(scores.get("run_id", "")),
            "target_count": len(rows),
            "unchanged_selected_positive_count": unchanged["selected_positive_count"],
            "unchanged_selected_waste_count": unchanged["selected_waste_count"],
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    decision = _dict(result.get("decision"))
    model = _dict(result.get("learned_model"))
    lines = [
        "# Diffusion Learned Selector V25 Result",
        "",
        "This file is generated by `experiments/analyze_diffusion_learned_selector_v25_result.py`.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision.get('status')}`",
        f"- Reason: {decision.get('reason')}",
        f"- Run ID: `{summary.get('run_id')}`",
        "",
        "## Summary",
        "",
        f"- Full model generations: `{summary.get('all_generation_count')}`",
        f"- Target rows: `{summary.get('target_count')}`",
        f"- Generated-positive candidate rows: `{summary.get('positive_count')}`",
        f"- Positive tasks: {_join_tasks(summary.get('positive_tasks'))}",
        f"- Candidate-aware duplicate-row errors: `{summary.get('candidate_aware_promotion_error_count')}`",
        f"- Unchanged hook selected positives: `{summary.get('unchanged_selected_positive_count')}`",
        f"- Unchanged hook selected waste: `{summary.get('unchanged_selected_waste_count')}`",
        f"- Learned model selected positives: `{summary.get('learned_selected_positive_count')}`",
        f"- Learned model selected waste: `{summary.get('learned_selected_waste_count')}`",
        f"- Repair task delta vs evolved: `{_format_float(summary.get('repair_task_delta_vs_evolved'))}`",
        f"- Repair task delta per extra generation: `{_format_float(summary.get('repair_task_delta_per_extra_generation_vs_evolved'))}`",
        f"- Oracle headroom vs repair: `{_format_float(summary.get('oracle_headroom_vs_repair'))}`",
        "",
        "## Frozen Learned Model",
        "",
        f"- Model class: `{model.get('model_class')}`",
        f"- History planning delta min: `{_format_float(model.get('history_planning_delta_min'))}`",
        f"- Final planning delta min: `{_format_float(model.get('final_planning_delta_min'))}`",
        f"- Final span score min: `{_format_float(model.get('final_span_score_min'))}`",
        f"- Training selected positives: `{model.get('training_selected_positive_count')}`",
        f"- Training selected waste: `{model.get('training_selected_waste_count')}`",
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
    lines.extend(["", "## Leave-One-Slice-Out", "", "| Held-Out Slice | Positives | Waste | Selected |", "| --- | ---: | ---: | ---: |"])
    for row in _list_of_dicts(result.get("leave_one_slice_out")):
        lines.append(
            f"| `{row.get('heldout_slice')}` | {row.get('selected_positive_count')} | "
            f"{row.get('selected_waste_count')} | {row.get('selected_count')} |"
        )
    lines.extend(
        [
            "",
            "## Cost Sweep",
            "",
            "| Selected Cost | Unchanged Net | Learned Net | Learned Advantage |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _list_of_dicts(decision.get("cost_sweep")):
        lines.append(
            "| "
            f"{_format_float(row.get('selected_cost'))} | "
            f"{_format_float(row.get('unchanged_net'))} | "
            f"{_format_float(row.get('learned_net'))} | "
            f"{_format_float(row.get('learned_advantage'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "The learned threshold model fails the held-out gate: it is precision-clean, but it "
                "misses the low-delta history-prefix positive on `plan_194`. The unchanged hook "
                "again selects every generated positive with zero selected waste, so no runner "
                "selector should change."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _load_training_rows(paths: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slice_id, path in paths.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in _list_of_dicts(data.get("rows")):
            item = dict(row)
            item["slice_id"] = slice_id
            rows.append(item)
    return rows


def _fit_model(rows: list[dict[str, object]]) -> dict[str, object]:
    thresholds = _threshold_grid(rows)
    candidates = []
    for history_delta in thresholds:
        for final_delta in thresholds:
            for final_span in (0.0, 1.5, 1.85, 2.0, 2.1, 2.2, 2.5):
                model = {
                    "final_planning_delta_min": final_delta,
                    "final_span_score_min": final_span,
                    "history_planning_delta_min": history_delta,
                    "model_class": "source_specific_delta_span_threshold",
                }
                summary = _policy_summary("candidate_row_value_model_v25", [row for row in rows if _model_selects(row, model)], rows)
                candidates.append((summary["selected_waste_count"], -summary["selected_positive_lift"], summary["selected_count"], model, summary))
    candidates.sort(key=lambda item: item[:3])
    _, _, _, model, summary = candidates[0]
    model.update(
        {
            "training_selected_count": summary["selected_count"],
            "training_selected_positive_count": summary["selected_positive_count"],
            "training_selected_positive_lift": summary["selected_positive_lift"],
            "training_selected_waste_count": summary["selected_waste_count"],
        }
    )
    return model


def _threshold_grid(rows: list[dict[str, object]]) -> list[float]:
    values = {0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2}
    values.update(round(_float(row.get("planning_quality_delta_vs_source")), 6) for row in rows)
    return sorted(values)


def _leave_one_slice_out(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for slice_id in sorted({str(row.get("slice_id", "")) for row in rows}):
        train = [row for row in rows if row.get("slice_id") != slice_id]
        heldout = [row for row in rows if row.get("slice_id") == slice_id]
        model = _fit_model(train)
        summary = _policy_summary("candidate_row_value_model_v25", [row for row in heldout if _model_selects(row, model)], heldout)
        output.append(
            {
                "heldout_slice": slice_id,
                "selected_count": summary["selected_count"],
                "selected_positive_count": summary["selected_positive_count"],
                "selected_waste_count": summary["selected_waste_count"],
            }
        )
    return output


def _model_selects(row: dict[str, object], model: dict[str, object]) -> bool:
    repair = str(row.get("repair", ""))
    planning_delta = _float(row.get("planning_quality_delta_vs_source"))
    span_score = _float(row.get("max_span_target_score"))
    if repair == HISTORY_REPAIR:
        return planning_delta >= _float(model.get("history_planning_delta_min"))
    if repair == FINAL_REPAIR:
        return planning_delta >= _float(model.get("final_planning_delta_min")) and span_score >= _float(
            model.get("final_span_score_min")
        )
    return False


def _unchanged_selected(rows: list[dict[str, object]], comparison_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    comparison_by_task = {str(row.get("task_id", "")): row for row in comparison_rows}
    return [
        row
        for row in rows
        if str(_dict(comparison_by_task.get(str(row.get("task_id", "")))).get("repair_control", ""))
        == str(row.get("repair", ""))
    ]


def _v23_asymmetric_selects(row: dict[str, object]) -> bool:
    repair = str(row.get("repair", ""))
    planning_delta = _float(row.get("planning_quality_delta_vs_source"))
    span_score = _float(row.get("max_span_target_score"))
    if repair == HISTORY_REPAIR:
        return planning_delta >= 0.20
    if repair == FINAL_REPAIR:
        return planning_delta >= 0.005 and span_score >= 1.85
    return False


def _policy_summary(policy_id: str, selected: list[dict[str, object]], all_rows: list[dict[str, object]]) -> dict[str, object]:
    positives = [row for row in selected if _float(row.get("candidate_lift_vs_trajectory")) > 0.0]
    waste = [row for row in selected if _float(row.get("candidate_lift_vs_trajectory")) <= 0.0]
    positive_tasks = {
        str(row.get("task_id", "")) for row in all_rows if _float(row.get("candidate_lift_vs_trajectory")) > 0.0
    }
    selected_tasks = [str(row.get("task_id", "")) for row in selected]
    return {
        "missed_positive_tasks": sorted(positive_tasks - set(selected_tasks)),
        "policy_id": policy_id,
        "selected_count": len(selected),
        "selected_positive_count": len(positives),
        "selected_positive_lift": sum(_float(row.get("candidate_lift_vs_trajectory")) for row in positives),
        "selected_tasks": selected_tasks,
        "selected_waste_count": len(waste),
        "selected_waste_tasks": [str(row.get("task_id", "")) for row in waste],
    }


def _decision(*, learned: dict[str, object], unchanged: dict[str, object], positive_count: int) -> dict[str, object]:
    cost_sweep = _cost_sweep(learned, unchanged)
    if positive_count == 0:
        return {"cost_sweep": cost_sweep, "reason": "No generated positives appeared.", "status": "inconclusive"}
    if int(learned.get("selected_waste_count", 0)) > 0:
        return {"cost_sweep": cost_sweep, "reason": "Learned model selected waste.", "status": "precision_failed"}
    if int(learned.get("selected_positive_count", 0)) < int(unchanged.get("selected_positive_count", 0)):
        return {
            "cost_sweep": cost_sweep,
            "reason": "Learned model is precision-clean but misses positives selected by the unchanged hook.",
            "status": "heldout_recall_failed",
        }
    if not any(_float(row.get("learned_advantage")) > 0.0 for row in cost_sweep):
        return {
            "cost_sweep": cost_sweep,
            "reason": "Learned model does not beat the unchanged hook after selected-output cost.",
            "status": "utility_failed",
        }
    return {
        "cost_sweep": cost_sweep,
        "reason": "Learned model beats the unchanged hook on held-out selected utility.",
        "status": "validated",
    }


def _cost_sweep(learned: dict[str, object], unchanged: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for selected_cost in SELECTED_COSTS:
        unchanged_net = _net(unchanged, selected_cost)
        learned_net = _net(learned, selected_cost)
        rows.append(
            {
                "learned_advantage": learned_net - unchanged_net,
                "learned_net": learned_net,
                "selected_cost": selected_cost,
                "unchanged_net": unchanged_net,
            }
        )
    return rows


def _net(policy: dict[str, object], selected_cost: float) -> float:
    return _float(policy.get("selected_positive_lift")) - selected_cost * int(policy.get("selected_count", 0))


def _positive_count(rows: list[dict[str, object]]) -> int:
    return sum(1 for row in rows if _float(row.get("candidate_lift_vs_trajectory")) > 0.0)


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
