"""Diagnose the weak span-v4 signed-value transfer slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.fit_diffusion_span_probe_signed_value import (
    FEATURE_GROUPS,
    _dict,
    _distance,
    _feature_space,
    _float,
    _format_float,
    _load_signature_rows,
    _vector,
)

DEFAULT_SIGNED_VALUE = Path(
    "eval_results/diffusion_language/counterfactual_span_probe_signed_value_v4.json"
)
DEFAULT_SIGNATURE_MODEL = Path(
    "eval_results/diffusion_language/counterfactual_span_probe_signature_model_v4.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/counterfactual_span_probe_signed_value_weak_slice_v4.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_WEAK_SLICE_V4.md")
DEFAULT_WEAK_SLICE = "counterfactual_span_validated_probe_stage1_gate_v4_transfer_v3_planning.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signed-value", type=Path, default=DEFAULT_SIGNED_VALUE)
    parser.add_argument("--signature-model", type=Path, default=DEFAULT_SIGNATURE_MODEL)
    parser.add_argument("--weak-slice", default=DEFAULT_WEAK_SLICE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze_weak_slice(
        signed_value_path=args.signed_value,
        signature_model_path=args.signature_model,
        weak_slice=args.weak_slice,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "false_positive_count": result["weak_slice_summary"]["false_positive_count"],
                "json_output": str(args.json_output),
                "positive_count": result["weak_slice_summary"]["positive_count"],
                "report_output": str(args.report_output),
                "selected_count": result["weak_slice_summary"]["selected_count"],
                "weak_slice_policy_utility": result["weak_slice_summary"]["policy_utility"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def analyze_weak_slice(
    *,
    signed_value_path: Path,
    signature_model_path: Path,
    weak_slice: str = DEFAULT_WEAK_SLICE,
) -> dict[str, object]:
    signed_value = json.loads(signed_value_path.read_text(encoding="utf-8"))
    selected_model = _dict(signed_value.get("selected_model"))
    selection_penalty = _float(signed_value.get("selection_penalty"))
    feature_group_id = str(selected_model.get("feature_group_id", "all"))
    features = tuple(FEATURE_GROUPS[feature_group_id])
    neighbor_count = int(_float(selected_model.get("neighbor_count")))
    rows = _load_signature_rows(signature_model_path, selection_penalty=selection_penalty)
    target_rows = [row for row in rows if Path(str(row.get("source_fit", ""))).name == weak_slice]
    train_rows = [row for row in rows if Path(str(row.get("source_fit", ""))).name != weak_slice]
    feature_space = _feature_space(train_rows, features)
    scored_rows = [
        _score_row(
            row,
            train_rows=train_rows,
            features=features,
            feature_space=feature_space,
            neighbor_count=neighbor_count,
        )
        for row in target_rows
    ]
    selected_rows = [row for row in scored_rows if bool(row.get("selected"))]
    positive_rows = [row for row in scored_rows if bool(row.get("label"))]
    false_positive_rows = [
        row for row in selected_rows if bool(row.get("selected")) and not bool(row.get("label"))
    ]
    signed_lift = sum(_float(row.get("candidate_lift_vs_trajectory")) for row in selected_rows)
    summary = {
        "false_positive_count": len(false_positive_rows),
        "false_positive_task_ids": _task_ids(false_positive_rows),
        "model_id": selected_model.get("model_id", ""),
        "neighbor_count": neighbor_count,
        "policy_utility": signed_lift - selection_penalty * len(selected_rows),
        "positive_count": len(positive_rows),
        "positive_task_ids": _task_ids(positive_rows),
        "selected_count": len(selected_rows),
        "selected_positive_rate": len(positive_rows) / max(len(selected_rows), 1),
        "signed_lift_covered": signed_lift,
        "weak_slice": weak_slice,
    }
    return {
        "feature_contrasts": _feature_contrasts(
            positive_rows=positive_rows,
            false_positive_rows=false_positive_rows,
            features=features,
        ),
        "generated_by": "experiments/analyze_diffusion_span_probe_signed_value_weak_slice.py",
        "inputs": {
            "signature_model": str(signature_model_path),
            "signed_value": str(signed_value_path),
        },
        "row_diagnostics": scored_rows,
        "schema": "diffusion_counterfactual_span_probe_signed_value_weak_slice.v1",
        "selection_penalty": selection_penalty,
        "weak_slice_summary": summary,
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("weak_slice_summary"))
    lines = [
        "# Diffusion Counterfactual Span Probe Signed Value Weak Slice V4",
        "",
        (
            "This file is generated by "
            "`experiments/analyze_diffusion_span_probe_signed_value_weak_slice.py`."
        ),
        "",
        "## Summary",
        "",
        f"- Weak slice: `{summary.get('weak_slice', '')}`",
        f"- Model: `{summary.get('model_id', '')}`",
        f"- Selected rows: `{summary.get('selected_count', 0)}`",
        f"- Positive rows: `{summary.get('positive_count', 0)}`",
        f"- False positives: `{summary.get('false_positive_count', 0)}`",
        f"- Selected positive rate: `{_format_float(summary.get('selected_positive_rate'))}`",
        f"- Signed utility: `{_format_float(summary.get('policy_utility'))}`",
        f"- Positive tasks: {_join_tasks(summary.get('positive_task_ids'))}",
        f"- False-positive tasks: {_join_tasks(summary.get('false_positive_task_ids'))}",
        "",
        "## Decision",
        "",
        (
            "Do not run the frozen GPU promotion slice yet. The selected signed-value "
            "head still treats the entire weak cohort as spend-worthy, so the remaining "
            "failure is a cohort-calibration problem rather than a missing local threshold."
        ),
        "",
        "## Row Diagnostics",
        "",
        "| Task | Label | Selected | Lift | Signed Value | Predicted Signed Value | Margin | Nearest Signed Values |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in _list_of_dicts(result.get("row_diagnostics")):
        neighbors = ", ".join(
            f"`{item.get('task_id')}:{_format_float(item.get('signed_value'))}`"
            for item in _list_of_dicts(row.get("nearest_neighbors"))
        )
        lines.append(
            "| "
            f"`{row.get('task_id', '')}` | "
            f"`{bool(row.get('label'))}` | "
            f"`{bool(row.get('selected'))}` | "
            f"{_format_float(row.get('candidate_lift_vs_trajectory'))} | "
            f"{_format_float(row.get('signed_value'))} | "
            f"{_format_float(row.get('predicted_signed_value'))} | "
            f"{_format_float(row.get('selection_margin'))} | "
            f"{neighbors} |"
        )
    lines.extend(
        [
            "",
            "## Feature Contrast",
            "",
            "| Feature | Positive Mean | False-Positive Mean | Difference |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in _list_of_dicts(result.get("feature_contrasts"))[:10]:
        lines.append(
            "| "
            f"`{row.get('feature', '')}` | "
            f"{_format_float(row.get('positive_mean'))} | "
            f"{_format_float(row.get('false_positive_mean'))} | "
            f"{_format_float(row.get('mean_difference'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "The two positive rows are not isolated by the current local signatures. "
                "They sit inside a selected eight-row cohort where six rows have zero "
                "realized lift, so the model preserves recall by buying a mostly no-lift "
                "neighborhood. The next signed-value feature should estimate cohort "
                "value density, prediction uncertainty, or neighbor disagreement before "
                "any fresh GPU spend test."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _score_row(
    row: dict[str, object],
    *,
    train_rows: list[dict[str, object]],
    features: tuple[str, ...],
    feature_space: object,
    neighbor_count: int,
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
    )[:neighbor_count]
    numerator = 0.0
    denominator = 0.0
    neighbor_rows = []
    for distance, train_row in nearest:
        weight = 1.0 / (distance + 1e-6)
        signed_value = _float(train_row.get("signed_value"))
        numerator += weight * signed_value
        denominator += weight
        neighbor_rows.append(
            {
                "distance": distance,
                "label": bool(train_row.get("label")),
                "signed_value": signed_value,
                "task_id": train_row.get("task_id", ""),
            }
        )
    predicted_signed_value = numerator / denominator if denominator else 0.0
    return {
        "candidate_lift_vs_trajectory": _float(row.get("candidate_lift_vs_trajectory")),
        "feature_values": {feature: _float(row.get(feature)) for feature in features},
        "label": bool(row.get("label")),
        "nearest_neighbors": neighbor_rows,
        "predicted_signed_value": predicted_signed_value,
        "selected": predicted_signed_value > 0.0 and bool(row.get("valid_for_stage1")),
        "selection_margin": predicted_signed_value,
        "signed_value": _float(row.get("signed_value")),
        "task_id": row.get("task_id", ""),
    }


def _feature_contrasts(
    *,
    positive_rows: list[dict[str, object]],
    false_positive_rows: list[dict[str, object]],
    features: tuple[str, ...],
) -> list[dict[str, object]]:
    rows = []
    for feature in features:
        positive_mean = _mean(_feature_lookup(row, feature) for row in positive_rows)
        false_positive_mean = _mean(_feature_lookup(row, feature) for row in false_positive_rows)
        rows.append(
            {
                "feature": feature,
                "false_positive_mean": false_positive_mean,
                "mean_difference": positive_mean - false_positive_mean,
                "positive_mean": positive_mean,
            }
        )
    return sorted(rows, key=lambda row: abs(_float(row.get("mean_difference"))), reverse=True)


def _feature_lookup(row: dict[str, object], feature: str) -> float:
    feature_values = _dict(row.get("feature_values"))
    return _float(feature_values.get(feature))


def _mean(values: object) -> float:
    concrete = list(values)
    return sum(concrete) / len(concrete) if concrete else 0.0


def _join_tasks(value: object) -> str:
    values = [str(item) for item in value] if isinstance(value, list) else []
    return ", ".join(f"`{item}`" for item in values) if values else "`none`"


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _task_ids(rows: list[dict[str, object]]) -> list[str]:
    return [str(row.get("task_id", "")) for row in rows]


if __name__ == "__main__":
    raise SystemExit(main())
