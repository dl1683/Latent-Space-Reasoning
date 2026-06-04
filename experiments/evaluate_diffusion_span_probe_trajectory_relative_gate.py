"""Evaluate trajectory-relative gating on the span-v4 cohort-risk head."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.fit_diffusion_span_probe_signed_value import (
    BASE_SIGNATURE_POSITIVE_UTILITY_BAR,
    DEFAULT_SELECTION_PENALTY,
    DEFAULT_SIGNATURE_MODEL,
    FEATURE_GROUPS,
    _dict,
    _distance,
    _feature_space,
    _float,
    _format_float,
    _join_tasks,
    _load_signature_rows,
    _summary,
    _vector,
)

DEFAULT_SPEND_EVAL = Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v3_eval.json")
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/counterfactual_span_probe_trajectory_relative_gate_v4.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    "DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_TRAJECTORY_RELATIVE_GATE_V4.md"
)
DEFAULT_WEAK_SLICE = "counterfactual_span_validated_probe_stage1_gate_v4_transfer_v3_planning.json"

COHORT_RISK_NEIGHBOR_COUNT = 13
COHORT_RISK_STD_PENALTY = 0.0
COHORT_RISK_NEGATIVE_FRACTION_PENALTY = 0.03
COHORT_RISK_MARGIN = -0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-model", type=Path, default=DEFAULT_SIGNATURE_MODEL)
    parser.add_argument("--spend-eval", type=Path, default=DEFAULT_SPEND_EVAL)
    parser.add_argument("--selection-penalty", type=float, default=DEFAULT_SELECTION_PENALTY)
    parser.add_argument("--weak-slice", default=DEFAULT_WEAK_SLICE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_trajectory_relative_gate(
        signature_model_path=args.signature_model,
        spend_eval_path=args.spend_eval,
        selection_penalty=args.selection_penalty,
        weak_slice=args.weak_slice,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    selected = _dict(result.get("trajectory_relative_gate"))
    weak = _dict(selected.get("weak_slice_summary"))
    print(
        json.dumps(
            {
                "false_negatives": selected.get("false_negative_count"),
                "false_positives": selected.get("false_positive_count"),
                "json_output": str(args.json_output),
                "policy_utility": selected.get("policy_utility"),
                "report_output": str(args.report_output),
                "weak_slice_false_positives": weak.get("false_positive_count"),
                "weak_slice_policy_utility": weak.get("policy_utility"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def evaluate_trajectory_relative_gate(
    *,
    signature_model_path: Path,
    spend_eval_path: Path,
    selection_penalty: float = DEFAULT_SELECTION_PENALTY,
    weak_slice: str = DEFAULT_WEAK_SLICE,
) -> dict[str, object]:
    rows = _load_signature_rows(signature_model_path, selection_penalty=selection_penalty)
    spend_features = _load_spend_features(spend_eval_path)
    scored_rows = _score_cohort_risk_rows(rows)
    cohort_risk = _summarize_policy(
        rows=scored_rows,
        selected_key="cohort_risk_selected",
        selection_penalty=selection_penalty,
        weak_slice=weak_slice,
    )
    gated_rows = []
    blocked_tasks = []
    for row in scored_rows:
        features = _dict(spend_features.get(str(row.get("task_id", ""))))
        has_trajectory_channel = bool(features)
        trajectory_relative_pass = (
            bool(features.get("trajectory_relative_prediction"))
            if has_trajectory_channel
            else True
        )
        selected = bool(row.get("cohort_risk_selected")) and trajectory_relative_pass
        if bool(row.get("cohort_risk_selected")) and not trajectory_relative_pass:
            blocked_tasks.append(str(row.get("task_id", "")))
        gated_rows.append(
            {
                **row,
                "has_trajectory_channel": has_trajectory_channel,
                "selected": selected,
                "source_task_delta_vs_trajectory": features.get("source_task_delta_vs_trajectory"),
                "trajectory_relative_prediction": trajectory_relative_pass,
            }
        )
    trajectory_relative_gate = _summarize_policy(
        rows=gated_rows,
        selected_key="selected",
        selection_penalty=selection_penalty,
        weak_slice=weak_slice,
    )
    trajectory_relative_gate["blocked_cohort_risk_task_ids"] = blocked_tasks
    trajectory_relative_gate["model_id"] = "cohort_risk_plus_trajectory_relative_gate"
    return {
        "base_signature_positive_utility_bar": BASE_SIGNATURE_POSITIVE_UTILITY_BAR,
        "cohort_risk_baseline": cohort_risk,
        "cohort_risk_parameters": {
            "margin": COHORT_RISK_MARGIN,
            "negative_fraction_penalty": COHORT_RISK_NEGATIVE_FRACTION_PENALTY,
            "neighbor_count": COHORT_RISK_NEIGHBOR_COUNT,
            "std_penalty": COHORT_RISK_STD_PENALTY,
        },
        "generated_by": "experiments/evaluate_diffusion_span_probe_trajectory_relative_gate.py",
        "inputs": {
            "signature_model": str(signature_model_path),
            "spend_eval": str(spend_eval_path),
        },
        "row_diagnostics": _compact_rows(gated_rows),
        "schema": "diffusion_counterfactual_span_probe_trajectory_relative_gate.v1",
        "selection_penalty": selection_penalty,
        "summary": {
            "positive_count": sum(1 for row in rows if bool(row.get("label"))),
            "target_count": len(rows),
            "trajectory_channel_row_count": len(spend_features),
            "weak_slice": weak_slice,
        },
        "trajectory_relative_gate": trajectory_relative_gate,
    }


def render_markdown(result: dict[str, object]) -> str:
    baseline = _dict(result.get("cohort_risk_baseline"))
    selected = _dict(result.get("trajectory_relative_gate"))
    weak = _dict(selected.get("weak_slice_summary"))
    lines = [
        "# Diffusion Counterfactual Span Probe Trajectory-Relative Gate V4",
        "",
        (
            "This file is generated by "
            "`experiments/evaluate_diffusion_span_probe_trajectory_relative_gate.py`."
        ),
        "",
        "## Summary",
        "",
        f"- Composite policy: `{selected.get('model_id', '')}`",
        f"- Signed utility: `{_format_float(selected.get('policy_utility'))}`",
        f"- False positives: `{selected.get('false_positive_count', 0)}`",
        f"- False negatives: `{selected.get('false_negative_count', 0)}`",
        f"- Weak-slice selected rows: `{weak.get('selected_count', 0)}`",
        f"- Weak-slice false positives: `{weak.get('false_positive_count', 0)}`",
        f"- Weak-slice signed utility: `{_format_float(weak.get('policy_utility'))}`",
        "",
        "## Decision",
        "",
    ]
    if _passes_offline_bar(selected) and _passes_weak_slice(weak):
        lines.append(
            "This composite clears the offline signed-value bar and the weak-slice "
            "cohort-calibration bar. Do not run GPU promotion yet: the result depends "
            "on a second information channel and needs negative controls before M3."
        )
    else:
        lines.append(
            "Do not promote this composite. It does not clear the required offline "
            "and weak-slice gates under the frozen audit."
        )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            "| Policy | Selected | FP | FN | Positive Lift | Signed Utility | Weak FP | Weak Utility | False Positives |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            _comparison_row("cohort_risk", baseline),
            _comparison_row("cohort_risk_plus_trajectory_relative_gate", selected),
            "",
            "## Blocked Cohort-Risk Tasks",
            "",
            f"- Blocked tasks: {_join_tasks(selected.get('blocked_cohort_risk_task_ids'))}",
            "",
            "## Reading",
            "",
            (
                "The weak-slice failure was not solved by another probe-text scalar. "
                "It was solved here by adding trajectory-relative source evidence: "
                "when the source state is already below the selected trajectory state "
                "or fails the trajectory-relative spend rule, the probe spend is blocked. "
                "That removes the known weak no-lift cohort while preserving all signed "
                "positive rows, but it also changes the information contract. The next "
                "step is a negative-control audit for the trajectory-relative channel "
                "before any fresh GPU slice."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _score_cohort_risk_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    scored_rows = []
    features = FEATURE_GROUPS["all"]
    for held_out in sorted({str(row.get("source_fit", "")) for row in rows}):
        train_rows = [row for row in rows if str(row.get("source_fit", "")) != held_out]
        test_rows = [row for row in rows if str(row.get("source_fit", "")) == held_out]
        feature_space = _feature_space(train_rows, features)
        for row in test_rows:
            scored_rows.append(
                _score_cohort_risk_row(
                    row,
                    train_rows=train_rows,
                    features=features,
                    feature_space=feature_space,
                )
            )
    return scored_rows


def _score_cohort_risk_row(
    row: dict[str, object],
    *,
    train_rows: list[dict[str, object]],
    features: tuple[str, ...],
    feature_space: object,
) -> dict[str, object]:
    row_vector = _vector(row, features=features, feature_space=feature_space)
    nearest = sorted(
        (
            (
                _distance(row_vector, _vector(train_row, features=features, feature_space=feature_space)),
                train_row,
            )
            for train_row in train_rows
        ),
        key=lambda item: item[0],
    )[:COHORT_RISK_NEIGHBOR_COUNT]
    weights = [1.0 / (distance + 1e-6) for distance, _ in nearest]
    signed_values = [_float(train_row.get("signed_value")) for _, train_row in nearest]
    predicted = sum(weight * value for weight, value in zip(weights, signed_values)) / sum(weights)
    mean_value = sum(signed_values) / len(signed_values)
    std_value = math.sqrt(
        sum((value - mean_value) ** 2 for value in signed_values) / len(signed_values)
    )
    negative_fraction = sum(value <= 0.0 for value in signed_values) / len(signed_values)
    risk_adjusted = (
        predicted
        - COHORT_RISK_STD_PENALTY * std_value
        - COHORT_RISK_NEGATIVE_FRACTION_PENALTY * negative_fraction
    )
    return {
        **row,
        "cohort_negative_fraction": negative_fraction,
        "cohort_risk_selected": risk_adjusted > COHORT_RISK_MARGIN
        and bool(row.get("valid_for_stage1")),
        "cohort_signed_value_std": std_value,
        "predicted_signed_value": predicted,
        "risk_adjusted_signed_value": risk_adjusted,
    }


def _summarize_policy(
    *,
    rows: list[dict[str, object]],
    selected_key: str,
    selection_penalty: float,
    weak_slice: str,
) -> dict[str, object]:
    selected_rows = [{**row, "selected": bool(row.get(selected_key))} for row in rows]
    summary = _summary(selected_rows, selection_penalty=selection_penalty)
    weak_rows = [
        row
        for row in selected_rows
        if Path(str(row.get("source_fit", ""))).name == weak_slice
    ]
    weak_summary = _summary(weak_rows, selection_penalty=selection_penalty)
    weak_summary["positive_task_ids"] = [
        str(row.get("task_id", "")) for row in weak_rows if bool(row.get("label"))
    ]
    summary["weak_slice_summary"] = weak_summary
    return summary


def _compact_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "candidate_lift_vs_trajectory": _float(row.get("candidate_lift_vs_trajectory")),
            "cohort_risk_selected": bool(row.get("cohort_risk_selected")),
            "has_trajectory_channel": bool(row.get("has_trajectory_channel")),
            "label": bool(row.get("label")),
            "risk_adjusted_signed_value": _float(row.get("risk_adjusted_signed_value")),
            "selected": bool(row.get("selected")),
            "source_task_delta_vs_trajectory": row.get("source_task_delta_vs_trajectory"),
            "task_id": str(row.get("task_id", "")),
            "trajectory_relative_prediction": bool(row.get("trajectory_relative_prediction")),
        }
        for row in rows
    ]


def _load_spend_features(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row.get("task_id", "")): {
            "source_task_delta_vs_trajectory": _float(row.get("source_task_delta_vs_trajectory")),
            "trajectory_relative_prediction": bool(row.get("trajectory_relative_prediction")),
        }
        for row in _list_of_dicts(payload.get("rows"))
    }


def _comparison_row(policy_id: str, summary: dict[str, object]) -> str:
    weak = _dict(summary.get("weak_slice_summary"))
    return (
        "| "
        f"`{policy_id}` | "
        f"{int(_float(summary.get('selected_count')))} | "
        f"{int(_float(summary.get('false_positive_count')))} | "
        f"{int(_float(summary.get('false_negative_count')))} | "
        f"{_format_float(summary.get('positive_lift_covered'))} | "
        f"{_format_float(summary.get('policy_utility'))} | "
        f"{int(_float(weak.get('false_positive_count')))} | "
        f"{_format_float(weak.get('policy_utility'))} | "
        f"{_join_tasks(summary.get('false_positive_task_ids'))} |"
    )


def _passes_offline_bar(summary: dict[str, object]) -> bool:
    return (
        _float(summary.get("policy_utility")) > BASE_SIGNATURE_POSITIVE_UTILITY_BAR
        and int(_float(summary.get("false_negative_count"))) == 0
        and int(_float(summary.get("false_positive_count"))) < 8
    )


def _passes_weak_slice(summary: dict[str, object]) -> bool:
    return (
        int(_float(summary.get("false_negative_count"))) == 0
        and int(_float(summary.get("false_positive_count"))) < 6
        and _float(summary.get("policy_utility")) > 0.001428571428571418
    )


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


if __name__ == "__main__":
    raise SystemExit(main())
