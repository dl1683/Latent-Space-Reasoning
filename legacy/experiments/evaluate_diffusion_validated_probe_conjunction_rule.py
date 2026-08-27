"""Evaluate one frozen validated probe conjunction on a target slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.fit_diffusion_validated_probe_stage1_gate import (
    _dict,
    _float,
    _format_float,
    fit_validated_probe_stage1_gate,
)

DEFAULT_TARGETS = Path(
    "eval_results/diffusion_language/diffusion_counterfactual_probe_spend_eval_targets_v1.json"
)
DEFAULT_SCORES = Path(
    "eval_results/diffusion_language/counterfactual_micro_probe_span_tomography_v4_transfer_v3_planning_scores.json"
)
DEFAULT_TEXT_FIDELITY = Path(
    "eval_results/diffusion_language/counterfactual_span_probe_text_fidelity_v4_transfer_v3_planning.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/counterfactual_span_gap_span_rule_v4_transfer_v3_planning.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_COUNTERFACTUAL_SPAN_GAP_SPAN_RULE_V4_TRANSFER_V3_PLANNING.md")
DEFAULT_CONDITIONS = (
    "measured_expected_gap_visibility_gain:ge:0.666667,"
    "measured_expected_span_evidence_gain:le:0.600000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--text-fidelity", type=Path, default=DEFAULT_TEXT_FIDELITY)
    parser.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--report-title", default="Diffusion Counterfactual Span Gap/Span Rule V4 Transfer V3 Planning")
    parser.add_argument("--allow-invalid-probe-selection", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation = evaluate_validated_probe_conjunction_rule(
        conditions=parse_conditions(args.conditions),
        require_valid_probe=not args.allow_invalid_probe_selection,
        scores_path=args.scores,
        targets_path=args.targets,
        text_fidelity_path=args.text_fidelity,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(evaluation, title=args.report_title), encoding="utf-8")
    print(
        json.dumps(
            {
                "error_count": evaluation["summary"]["error_count"],
                "false_negative_count": evaluation["summary"]["false_negative_count"],
                "false_positive_count": evaluation["summary"]["false_positive_count"],
                "gate_decision": evaluation["summary"]["gate_decision"],
                "selected_count": evaluation["summary"]["selected_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def evaluate_validated_probe_conjunction_rule(
    *,
    conditions: list[dict[str, object]],
    require_valid_probe: bool,
    scores_path: Path,
    targets_path: Path,
    text_fidelity_path: Path | None,
) -> dict[str, object]:
    fit = fit_validated_probe_stage1_gate(
        scores_path=scores_path,
        targets_path=targets_path,
        text_fidelity_path=text_fidelity_path,
    )
    rows = [dict(row) for row in _list_of_dicts(fit.get("rows"))]
    false_positives = []
    false_negatives = []
    selected_rows = []
    for row in rows:
        selected = _rule_selects(row, conditions)
        if require_valid_probe:
            selected = selected and bool(row.get("valid_for_stage1"))
        row["selected_by_fixed_conjunction_rule"] = selected
        label = bool(row.get("label"))
        if selected:
            selected_rows.append(row)
        if selected and not label:
            false_positives.append(row)
        if not selected and label:
            false_negatives.append(row)
    rule = {
        "conditions": conditions,
        "error_count": len(false_positives) + len(false_negatives),
        "false_negative_count": len(false_negatives),
        "false_negative_task_ids": _task_ids(false_negatives),
        "false_positive_count": len(false_positives),
        "false_positive_task_ids": _task_ids(false_positives),
        "missed_positive_lift": sum(
            _float(row.get("candidate_lift_vs_trajectory")) for row in false_negatives
        ),
        "requires_valid_probe": require_valid_probe,
        "rule_name": _rule_name(conditions, require_valid_probe=require_valid_probe),
        "selected_count": len(selected_rows),
        "selected_task_ids": _task_ids(selected_rows),
    }
    return {
        "generated_by": "experiments/evaluate_diffusion_validated_probe_conjunction_rule.py",
        "inputs": {
            "scores": str(scores_path),
            "targets": str(targets_path),
            "text_fidelity": str(text_fidelity_path) if text_fidelity_path else "",
        },
        "rows": rows,
        "rule": rule,
        "schema": "diffusion_counterfactual_validated_probe_conjunction_rule_eval.v1",
        "summary": _summary(rows, rule),
    }


def parse_conditions(value: str) -> list[dict[str, object]]:
    conditions = []
    for chunk in value.split(","):
        if not chunk.strip():
            continue
        parts = [part.strip() for part in chunk.split(":")]
        if len(parts) != 3:
            raise ValueError(f"condition must be feature:direction:threshold, got {chunk!r}")
        feature, direction, threshold = parts
        if direction not in {"ge", "le"}:
            raise ValueError(f"unsupported direction {direction!r}")
        conditions.append({"direction": direction, "feature": feature, "threshold": float(threshold)})
    if not conditions:
        raise ValueError("at least one condition is required")
    return conditions


def render_markdown(
    evaluation: dict[str, object],
    *,
    title: str = "Diffusion Counterfactual Validated Probe Conjunction Rule Evaluation",
) -> str:
    summary = _dict(evaluation.get("summary"))
    rule = _dict(evaluation.get("rule"))
    lines = [
        f"# {title}",
        "",
        "This file is generated by `experiments/evaluate_diffusion_validated_probe_conjunction_rule.py`.",
        "It applies one frozen validated probe conjunction. The rule is not refit on this slice.",
        "",
        "## Summary",
        "",
        f"- Rows: `{summary.get('row_count', 0)}`",
        f"- Valid probe rows: `{summary.get('valid_probe_count', 0)}`",
        f"- Positive labels: `{summary.get('positive_label_count', 0)}`",
        f"- Negative labels: `{summary.get('negative_label_count', 0)}`",
        f"- Fixed rule: `{rule.get('rule_name', '')}`",
        f"- Selected rows: `{summary.get('selected_count', 0)}`",
        f"- Errors: `{summary.get('error_count', 0)}`",
        f"- False positives: `{summary.get('false_positive_count', 0)}`",
        f"- False negatives: `{summary.get('false_negative_count', 0)}`",
        f"- Missed positive lift: `{_format_float(summary.get('missed_positive_lift'))}`",
        f"- Gate decision: `{summary.get('gate_decision', '')}`",
        "",
        "## Rows",
        "",
        "| Task | Label | Lift | Valid Probe | Gap Gain | Span Gain | Fixed Select | Error |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in _list_of_dicts(evaluation.get("rows")):
        features = _dict(row.get("features"))
        selected = bool(row.get("selected_by_fixed_conjunction_rule"))
        label = bool(row.get("label"))
        lines.append(
            "| "
            f"`{row.get('task_id', '')}` | "
            f"{label} | "
            f"{_format_float(row.get('candidate_lift_vs_trajectory'))} | "
            f"{bool(row.get('valid_for_stage1'))} | "
            f"{_format_float(features.get('measured_expected_gap_visibility_gain'))} | "
            f"{_format_float(features.get('measured_expected_span_evidence_gain'))} | "
            f"{selected} | "
            f"{selected != label} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "This is the strict no-retuning check for the gap/span conjunction. "
                "A failure here is useful: it means the transfer-screened challenger "
                "was still slice-shaped and should remain diagnostic."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _rule_selects(row: dict[str, object], conditions: list[dict[str, object]]) -> bool:
    features = _dict(row.get("features"))
    for condition in conditions:
        value = _float(features.get(str(condition.get("feature", ""))))
        threshold = _float(condition.get("threshold"))
        if condition.get("direction") == "ge":
            if value < threshold:
                return False
        elif value > threshold:
            return False
    return True


def _summary(rows: list[dict[str, object]], rule: dict[str, object]) -> dict[str, object]:
    positives = [row for row in rows if bool(row.get("label"))]
    negatives = [row for row in rows if not bool(row.get("label"))]
    valid_rows = [row for row in rows if bool(row.get("valid_for_stage1"))]
    return {
        "error_count": int(rule.get("error_count", 0)),
        "false_negative_count": int(rule.get("false_negative_count", 0)),
        "false_negative_task_ids": list(rule.get("false_negative_task_ids", [])),
        "false_positive_count": int(rule.get("false_positive_count", 0)),
        "false_positive_task_ids": list(rule.get("false_positive_task_ids", [])),
        "gate_decision": "diagnostic_only",
        "missed_positive_lift": _float(rule.get("missed_positive_lift")),
        "negative_label_count": len(negatives),
        "positive_label_count": len(positives),
        "row_count": len(rows),
        "selected_count": int(rule.get("selected_count", 0)),
        "valid_probe_count": len(valid_rows),
    }


def _rule_name(conditions: list[dict[str, object]], *, require_valid_probe: bool) -> str:
    prefix = "valid_" if require_valid_probe else ""
    return prefix + "and_".join(
        f"{condition.get('feature', '')}_{condition.get('direction', '')}_{_format_token(condition.get('threshold'))}"
        for condition in conditions
    )


def _format_token(value: object) -> str:
    return _format_float(value).replace("-", "neg_").replace(".", "p")


def _task_ids(rows: list[dict[str, object]]) -> list[str]:
    return [str(row.get("task_id", "")) for row in rows]


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


if __name__ == "__main__":
    raise SystemExit(main())
