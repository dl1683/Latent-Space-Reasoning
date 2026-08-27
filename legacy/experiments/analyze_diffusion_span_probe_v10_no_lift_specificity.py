"""Analyze pre-spend no-lift specificity on the v10 composite replay rows."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.fit_diffusion_span_probe_signed_value import DEFAULT_SELECTION_PENALTY, _float

DEFAULT_REPLAY = Path("eval_results/diffusion_language/span_probe_composite_v10_fixed_source_replay.json")
DEFAULT_MEASUREMENT = Path(
    "eval_results/diffusion_language/span_probe_composite_v10_fixed_source_measurement_scores.json"
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/span_probe_composite_v10_no_lift_specificity.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_V10_NO_LIFT_SPECIFICITY.md")

FEATURES = (
    "measured_probe_value_prediction",
    "prompt_gap_count",
    "prompt_coverage",
    "source_quality",
    "counterfactual_probe_remaining_gap_count",
    "counterfactual_probe_resolved_gap_count",
    "expected_gap_visibility_gain",
    "expected_realization_defect_visibility",
    "expected_retention_risk_visibility",
    "expected_span_evidence_gain",
    "first_repairable_denoise_skeleton_coverage",
    "source_task_delta_vs_trajectory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--measurement", type=Path, default=DEFAULT_MEASUREMENT)
    parser.add_argument("--selection-penalty", type=float, default=DEFAULT_SELECTION_PENALTY)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze_no_lift_specificity(
        replay_path=args.replay,
        measurement_path=args.measurement,
        selection_penalty=args.selection_penalty,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    selected = _dict(result.get("selected_rule"))
    print(
        json.dumps(
            {
                "false_negative_count": selected.get("false_negative_count"),
                "false_positive_count": selected.get("false_positive_count"),
                "json_output": str(args.json_output),
                "policy_utility": selected.get("policy_utility"),
                "report_output": str(args.report_output),
                "rule_id": selected.get("rule_id"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def analyze_no_lift_specificity(
    *,
    replay_path: Path,
    measurement_path: Path,
    selection_penalty: float = DEFAULT_SELECTION_PENALTY,
) -> dict[str, object]:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    measurement_by_task = {
        str(row.get("task_id")): row for row in _list_of_dicts(measurement.get("repair_spend_gate_rows"))
    }
    rows = [
        _joined_row(row, measurement_by_task[str(row.get("task_id"))])
        for row in _list_of_dicts(replay.get("row_diagnostics"))
        if str(row.get("task_id")) in measurement_by_task
    ]
    base_summary = _summarize(rows, selected_key="selected", selection_penalty=selection_penalty)
    rule_results = _evaluate_rules(rows, selection_penalty=selection_penalty)
    selected_rule = max(
        rule_results,
        key=lambda row: (
            _float(row.get("policy_utility")),
            -_float(row.get("false_negative_count")),
            -_float(row.get("false_positive_count")),
            _float(row.get("selected_count")),
        ),
    )
    return {
        "base_replay": base_summary,
        "generated_by": "experiments/analyze_diffusion_span_probe_v10_no_lift_specificity.py",
        "inputs": {"measurement": str(measurement_path), "replay": str(replay_path)},
        "row_diagnostics": _compact_rows(rows),
        "schema": "diffusion_span_probe_v10_no_lift_specificity.v1",
        "selected_rule": selected_rule,
        "selection_penalty": selection_penalty,
        "summary": {
            "candidate_rule_count": len(rule_results),
            "max_possible_utility_on_selected_repair_labels": _max_possible_utility(
                rows,
                selection_penalty=selection_penalty,
            ),
            "positive_count": sum(1 for row in rows if bool(row.get("label"))),
            "target_count": len(rows),
        },
        "top_rules": sorted(
            rule_results,
            key=lambda row: (
                _float(row.get("policy_utility")),
                -_float(row.get("false_negative_count")),
                -_float(row.get("false_positive_count")),
            ),
            reverse=True,
        )[:20],
    }


def render_markdown(result: dict[str, object]) -> str:
    base = _dict(result.get("base_replay"))
    selected = _dict(result.get("selected_rule"))
    summary = _dict(result.get("summary"))
    lines = [
        "# Diffusion Span Probe V10 No-Lift Specificity",
        "",
        "This file is generated by `experiments/analyze_diffusion_span_probe_v10_no_lift_specificity.py`.",
        "",
        "## Summary",
        "",
        f"- Candidate rules evaluated: `{summary.get('candidate_rule_count', 0)}`",
        f"- Base fixed-source replay utility: `{_format_float(base.get('policy_utility'))}`",
        f"- Base false positives: `{base.get('false_positive_count', 0)}`",
        f"- Base false negatives: `{base.get('false_negative_count', 0)}`",
        f"- Diagnostic best rule: `{selected.get('rule_id', '')}`",
        f"- Diagnostic rule utility: `{_format_float(selected.get('policy_utility'))}`",
        f"- Diagnostic rule FP/FN: `{selected.get('false_positive_count', 0)}` / `{selected.get('false_negative_count', 0)}`",
        f"- Max possible v10 selected-repair utility: `{_format_float(summary.get('max_possible_utility_on_selected_repair_labels'))}`",
        "",
        "## Decision",
        "",
        (
            "Do not promote this as a controller. This is a post-label v10 diagnostic "
            "frontier that identifies a plausible next frozen hypothesis: a measured "
            "probe-value floor can remove the remaining no-lift rows while preserving "
            "the selected-repair positives on this slice."
        ),
        "",
        "## Selected Diagnostic Rule",
        "",
        "| Rule | Selected | FP | FN | Positive Lift | Utility | False Positives | False Negatives |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        _rule_row(selected),
        "",
        "## Top Rules",
        "",
        "| Rule | Selected | FP | FN | Positive Lift | Utility |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _list_of_dicts(result.get("top_rules"))[:10]:
        lines.append(
            "| "
            f"`{row.get('rule_id', '')}` | "
            f"{row.get('selected_count', 0)} | "
            f"{row.get('false_positive_count', 0)} | "
            f"{row.get('false_negative_count', 0)} | "
            f"{_format_float(row.get('positive_lift_covered'))} | "
            f"{_format_float(row.get('policy_utility'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "The first no-lift specificity signal is not another text scalar in "
                "isolation; it is a cost/value floor on the measured probe path. On v10, "
                "`plan_075` and `plan_078` both sit below the selected probe-value floor, "
                "while all selected-repair positives remain above it. Because this rule "
                "was chosen after seeing v10 labels, the only valid next step is to freeze "
                "it and test it on a source-divergent fresh slice."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _evaluate_rules(rows: list[dict[str, object]], *, selection_penalty: float) -> list[dict[str, object]]:
    predicates = []
    for feature in FEATURES:
        values = sorted({_float(row.get(feature)) for row in rows})
        for value in values:
            predicates.append(
                (feature, f"{feature}_ge_{_slug(value)}", lambda row, f=feature, v=value: _float(row.get(f)) >= v)
            )
            predicates.append(
                (feature, f"{feature}_le_{_slug(value)}", lambda row, f=feature, v=value: _float(row.get(f)) <= v)
            )
    rule_results = []
    for _feature, rule_id, predicate in predicates:
        rule_results.append(
            _summarize_rule(rows, rule_id=rule_id, predicates=(predicate,), selection_penalty=selection_penalty)
        )
    for (left_feature, left_id, left), (right_feature, right_id, right) in combinations(predicates, 2):
        if left_feature == right_feature:
            continue
        rule_results.append(
            _summarize_rule(
                rows,
                rule_id=f"{left_id}__and__{right_id}",
                predicates=(left, right),
                selection_penalty=selection_penalty,
            )
        )
    return rule_results


def _summarize_rule(
    rows: list[dict[str, object]],
    *,
    rule_id: str,
    predicates: tuple[object, ...],
    selection_penalty: float,
) -> dict[str, object]:
    scored_rows = []
    for row in rows:
        selected = bool(row.get("selected")) and all(predicate(row) for predicate in predicates)
        scored_rows.append({**row, "rule_selected": selected})
    return {
        **_summarize(scored_rows, selected_key="rule_selected", selection_penalty=selection_penalty),
        "rule_id": rule_id,
    }


def _summarize(
    rows: list[dict[str, object]],
    *,
    selected_key: str,
    selection_penalty: float,
) -> dict[str, object]:
    selected_rows = [row for row in rows if bool(row.get(selected_key))]
    false_positives = [row for row in selected_rows if not bool(row.get("label"))]
    false_negatives = [row for row in rows if bool(row.get("label")) and not bool(row.get(selected_key))]
    signed_lift = sum(_float(row.get("candidate_lift_vs_trajectory")) for row in selected_rows)
    return {
        "error_count": len(false_positives) + len(false_negatives),
        "false_negative_count": len(false_negatives),
        "false_negative_task_ids": _task_ids(false_negatives),
        "false_positive_count": len(false_positives),
        "false_positive_task_ids": _task_ids(false_positives),
        "policy_utility": signed_lift - selection_penalty * len(selected_rows),
        "positive_lift_covered": sum(
            _float(row.get("candidate_lift_vs_trajectory"))
            for row in selected_rows
            if bool(row.get("label"))
        ),
        "selected_count": len(selected_rows),
    }


def _joined_row(replay_row: dict[str, object], measurement_row: dict[str, object]) -> dict[str, object]:
    feature_delta = _dict(measurement_row.get("measured_probe_feature_delta"))
    return {
        **replay_row,
        "counterfactual_probe_remaining_gap_count": _float(
            measurement_row.get("counterfactual_probe_remaining_gap_count")
        ),
        "counterfactual_probe_resolved_gap_count": _float(
            measurement_row.get("counterfactual_probe_resolved_gap_count")
        ),
        "expected_gap_visibility_gain": _float(feature_delta.get("expected_gap_visibility_gain")),
        "expected_realization_defect_visibility": _float(
            feature_delta.get("expected_realization_defect_visibility")
        ),
        "expected_retention_risk_visibility": _float(feature_delta.get("expected_retention_risk_visibility")),
        "expected_span_evidence_gain": _float(feature_delta.get("expected_span_evidence_gain")),
        "first_repairable_denoise_skeleton_coverage": _float(
            measurement_row.get("first_repairable_denoise_skeleton_coverage")
        ),
        "measured_probe_value_prediction": _float(
            measurement_row.get("measured_probe_value_prediction")
        ),
        "prompt_coverage": _float(measurement_row.get("prompt_coverage")),
        "prompt_gap_count": _float(measurement_row.get("prompt_gap_count")),
        "source_quality": _float(measurement_row.get("source_quality")),
    }


def _compact_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = (
        "task_id",
        "selected",
        "label",
        "candidate_lift_vs_trajectory",
        "measured_probe_value_prediction",
        "prompt_gap_count",
        "prompt_coverage",
        "counterfactual_probe_remaining_gap_count",
        "expected_span_evidence_gain",
        "source_task_delta_vs_trajectory",
    )
    return [{key: row.get(key) for key in keys} for row in rows]


def _max_possible_utility(rows: list[dict[str, object]], *, selection_penalty: float) -> float:
    positives = [row for row in rows if bool(row.get("label"))]
    return sum(_float(row.get("candidate_lift_vs_trajectory")) for row in positives) - selection_penalty * len(positives)


def _rule_row(row: dict[str, object]) -> str:
    return (
        "| "
        f"`{row.get('rule_id', '')}` | "
        f"{row.get('selected_count', 0)} | "
        f"{row.get('false_positive_count', 0)} | "
        f"{row.get('false_negative_count', 0)} | "
        f"{_format_float(row.get('positive_lift_covered'))} | "
        f"{_format_float(row.get('policy_utility'))} | "
        f"{_join_tasks(row.get('false_positive_task_ids'))} | "
        f"{_join_tasks(row.get('false_negative_task_ids'))} |"
    )


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _task_ids(rows: list[dict[str, object]]) -> list[str]:
    return [str(row.get("task_id", "")) for row in rows]


def _join_tasks(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return ", ".join(str(item) for item in value)


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


def _slug(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "n").replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())
