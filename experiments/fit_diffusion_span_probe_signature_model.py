"""Fit a leave-slice-out value model over span-v4 probe signatures."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SLICE_FITS = (
    Path("eval_results/diffusion_language/counterfactual_span_validated_probe_stage1_gate_v4.json"),
    Path("eval_results/diffusion_language/counterfactual_span_validated_probe_stage1_gate_v4_fresh_planning.json"),
    Path("eval_results/diffusion_language/counterfactual_span_validated_probe_stage1_gate_v4_transfer_v3_planning.json"),
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/counterfactual_span_probe_signature_model_v4.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_MODEL_V4.md")

NUMERIC_FEATURES = (
    "measured_probe_value_prediction",
    "measured_expected_gap_visibility_gain",
    "measured_expected_realization_defect_visibility",
    "measured_expected_span_evidence_gain",
    "measured_expected_retention_risk_visibility",
    "measured_distinct_retention_risk_visibility",
    "counterfactual_probe_text_x0_x2_slot_overlap",
    "counterfactual_probe_text_max_slot_overlap",
    "counterfactual_probe_text_repeated_token_excess",
    "prompt_gap_count",
    "source_quality",
)
BOOLEAN_FEATURES = (
    "counterfactual_probe_text_semantic_valid_for_stage1",
    "counterfactual_probe_text_semantic_defect",
    "counterfactual_probe_text_malformed_compact_key",
    "counterfactual_probe_text_template_slot_echo",
    "counterfactual_probe_text_duplicate_authorization",
    "counterfactual_probe_text_weird_punctuation",
    "would_probe_score",
)


@dataclass(frozen=True)
class FeatureSpace:
    means: dict[str, float]
    scales: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slice-fit",
        action="append",
        dest="slice_fits",
        type=Path,
        help="Validated probe fit JSON. May be passed multiple times.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    slice_fit_paths = tuple(args.slice_fits or DEFAULT_SLICE_FITS)
    result = fit_span_probe_signature_model(slice_fit_paths=slice_fit_paths)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "in_sample_errors": result["in_sample"]["summary"]["error_count"],
                "json_output": str(args.json_output),
                "loo_errors": result["leave_one_slice_out"]["summary"]["error_count"],
                "loo_false_negatives": result["leave_one_slice_out"]["summary"]["false_negative_count"],
                "report_output": str(args.report_output),
                "target_count": result["summary"]["target_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def fit_span_probe_signature_model(*, slice_fit_paths: tuple[Path, ...]) -> dict[str, object]:
    rows = _load_rows(slice_fit_paths)
    in_sample = _fit_and_score(rows, rows)
    slice_results = []
    for held_out in sorted({str(row["source_fit"]) for row in rows}):
        train_rows = [row for row in rows if row["source_fit"] != held_out]
        test_rows = [row for row in rows if row["source_fit"] == held_out]
        scored = _fit_and_score(train_rows, test_rows)
        scored["held_out_fit"] = held_out
        slice_results.append(scored)
    loo_rows = [row for split in slice_results for row in _list_of_dicts(split.get("rows"))]
    return {
        "features": {
            "boolean": list(BOOLEAN_FEATURES),
            "numeric": list(NUMERIC_FEATURES),
            "score": "nearest_negative_distance_minus_nearest_positive_distance",
        },
        "generated_by": "experiments/fit_diffusion_span_probe_signature_model.py",
        "inputs": {"slice_fits": [str(path) for path in slice_fit_paths]},
        "in_sample": in_sample,
        "leave_one_slice_out": {
            "rows": loo_rows,
            "slice_results": slice_results,
            "summary": _summary(loo_rows),
        },
        "schema": "diffusion_counterfactual_span_probe_signature_model.v1",
        "summary": {
            "positive_count": sum(1 for row in rows if bool(row.get("label"))),
            "slice_count": len({str(row["source_fit"]) for row in rows}),
            "target_count": len(rows),
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    in_sample = _dict(_dict(result.get("in_sample")).get("summary"))
    loo = _dict(_dict(result.get("leave_one_slice_out")).get("summary"))
    lines = [
        "# Diffusion Counterfactual Span Probe Signature Model V4",
        "",
        "This file is generated by `experiments/fit_diffusion_span_probe_signature_model.py`.",
        (
            "It tests whether a richer measured-probe signature transfers better "
            "than the failed hand thresholds before any live spend gate is allowed."
        ),
        "",
        "## Summary",
        "",
        f"- Target rows: `{summary.get('target_count', 0)}`",
        f"- Slices: `{summary.get('slice_count', 0)}`",
        f"- Positive rows: `{summary.get('positive_count', 0)}`",
        f"- In-sample errors: `{in_sample.get('error_count', 0)}`",
        f"- In-sample false positives: `{in_sample.get('false_positive_count', 0)}`",
        f"- In-sample false negatives: `{in_sample.get('false_negative_count', 0)}`",
        f"- Leave-one-slice-out errors: `{loo.get('error_count', 0)}`",
        f"- Leave-one-slice-out false positives: `{loo.get('false_positive_count', 0)}`",
        f"- Leave-one-slice-out false negatives: `{loo.get('false_negative_count', 0)}`",
        f"- Leave-one-slice-out lift covered: `{_format_float(loo.get('positive_lift_covered'))}`",
        "",
        "## Decision",
        "",
    ]
    if int(loo.get("false_negative_count", 0)) == 0 and int(loo.get("false_positive_count", 0)) == 0:
        lines.append(
            "This signature model clears the current leave-slice-out audit, but it "
            "still needs a frozen fresh GPU slice before promotion."
        )
    else:
        lines.append(
            "Do not promote this signature model. It is a stronger diagnostic than "
            "single thresholds, but leave-slice-out transfer still has strict errors."
        )
    lines.extend(
        [
            "",
            "## Model",
            "",
            (
                "Rows are embedded with measured probe value, gap, realization, "
                "span, retention, text-overlap, source, and validity features. "
                "The score is distance to the nearest negative prototype minus "
                "distance to the nearest positive prototype after standardizing "
                "numeric features on training rows."
            ),
            "",
            "## In-Sample Rows",
            "",
        ]
    )
    lines.extend(_rows_table(_list_of_dicts(_dict(result.get("in_sample")).get("rows"))))
    lines.extend(["", "## Leave-One-Slice-Out Rows", ""])
    lines.extend(_rows_table(_list_of_dicts(_dict(result.get("leave_one_slice_out")).get("rows"))))
    lines.extend(["", "## Slice Transfer", ""])
    lines.append("| Held-Out Fit | Errors | FP | FN | Lift Covered | Missed Positives | False Positives |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- |")
    for split in _list_of_dicts(_dict(result.get("leave_one_slice_out")).get("slice_results")):
        split_summary = _dict(split.get("summary"))
        lines.append(
            "| "
            f"`{Path(str(split.get('held_out_fit', ''))).name}` | "
            f"{int(split_summary.get('error_count', 0))} | "
            f"{int(split_summary.get('false_positive_count', 0))} | "
            f"{int(split_summary.get('false_negative_count', 0))} | "
            f"{_format_float(split_summary.get('positive_lift_covered'))} | "
            f"{_join_tasks(split_summary.get('false_negative_task_ids'))} | "
            f"{_join_tasks(split_summary.get('false_positive_task_ids'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "This is the bridge from hand threshold chasing to probe-signature "
                "learning. In-sample separation is not evidence of a controller. "
                "The leave-slice-out rows are the gate, and any missed positive "
                "repair remains a blocker unless a cost objective explicitly accepts it."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _fit_and_score(
    train_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
) -> dict[str, object]:
    feature_space = _feature_space(train_rows)
    train_vectors = [_vector(row, feature_space) for row in train_rows]
    train_scores = [
        _prototype_score(vector, row, train_vectors, train_rows)
        for vector, row in zip(train_vectors, train_rows)
    ]
    threshold = _choose_threshold(train_rows, train_scores)
    scored_rows = []
    for row in test_rows:
        vector = _vector(row, feature_space)
        score = _prototype_score(vector, row, train_vectors, train_rows)
        prediction = score >= threshold and bool(row.get("valid_for_stage1"))
        label = bool(row.get("label"))
        enriched = {
            "candidate_lift_vs_trajectory": _float(row.get("candidate_lift_vs_trajectory")),
            "error": prediction != label,
            "label": label,
            "prediction": prediction,
            "probe_signature_score": score,
            "source_fit": row.get("source_fit", ""),
            "task_id": row.get("task_id", ""),
            "threshold": threshold,
            "valid_for_stage1": bool(row.get("valid_for_stage1")),
        }
        for feature in NUMERIC_FEATURES:
            enriched[feature] = _feature_value(row, feature)
        for feature in BOOLEAN_FEATURES:
            enriched[feature] = _boolean_feature(row, feature)
        scored_rows.append(enriched)
    return {
        "rows": scored_rows,
        "summary": _summary(scored_rows),
        "threshold": threshold,
    }


def _choose_threshold(rows: list[dict[str, object]], scores: list[float]) -> float:
    thresholds = sorted({score for score in scores})
    if thresholds:
        thresholds = [thresholds[0] - 1.0, *thresholds, thresholds[-1] + 1.0]
    else:
        thresholds = [0.0]
    best_threshold = thresholds[0]
    best_key: tuple[float, float, float, float] | None = None
    for threshold in thresholds:
        predictions = [
            score >= threshold and bool(row.get("valid_for_stage1"))
            for row, score in zip(rows, scores)
        ]
        false_positives = [
            row for row, prediction in zip(rows, predictions) if prediction and not bool(row.get("label"))
        ]
        false_negatives = [
            row for row, prediction in zip(rows, predictions) if not prediction and bool(row.get("label"))
        ]
        missed_lift = sum(_float(row.get("candidate_lift_vs_trajectory")) for row in false_negatives)
        selected = sum(1 for prediction in predictions if prediction)
        key = (missed_lift, float(len(false_negatives)), float(len(false_positives)), float(selected))
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def _prototype_score(
    vector: tuple[float, ...],
    row: dict[str, object],
    train_vectors: list[tuple[float, ...]],
    train_rows: list[dict[str, object]],
) -> float:
    positive_distances = []
    negative_distances = []
    task_id = str(row.get("task_id", ""))
    source_fit = str(row.get("source_fit", ""))
    for train_vector, train_row in zip(train_vectors, train_rows):
        if (
            str(train_row.get("task_id", "")) == task_id
            and str(train_row.get("source_fit", "")) == source_fit
        ):
            continue
        distance = _distance(vector, train_vector)
        if bool(train_row.get("label")):
            positive_distances.append(distance)
        else:
            negative_distances.append(distance)
    nearest_positive = min(positive_distances, default=10.0)
    nearest_negative = min(negative_distances, default=10.0)
    return nearest_negative - nearest_positive


def _feature_space(rows: list[dict[str, object]]) -> FeatureSpace:
    means = {}
    scales = {}
    for feature in NUMERIC_FEATURES:
        values = [_feature_value(row, feature) for row in rows]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        scale = math.sqrt(variance) or 1.0
        means[feature] = mean
        scales[feature] = scale
    return FeatureSpace(means=means, scales=scales)


def _vector(row: dict[str, object], feature_space: FeatureSpace) -> tuple[float, ...]:
    numeric = [
        (_feature_value(row, feature) - feature_space.means[feature]) / feature_space.scales[feature]
        for feature in NUMERIC_FEATURES
    ]
    boolean = [_boolean_feature(row, feature) for feature in BOOLEAN_FEATURES]
    return tuple(numeric + boolean)


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    false_positives = [row for row in rows if bool(row.get("prediction")) and not bool(row.get("label"))]
    false_negatives = [row for row in rows if not bool(row.get("prediction")) and bool(row.get("label"))]
    true_positives = [row for row in rows if bool(row.get("prediction")) and bool(row.get("label"))]
    return {
        "error_count": len(false_positives) + len(false_negatives),
        "false_negative_count": len(false_negatives),
        "false_negative_task_ids": _task_ids(false_negatives),
        "false_positive_count": len(false_positives),
        "false_positive_task_ids": _task_ids(false_positives),
        "positive_lift_covered": sum(_float(row.get("candidate_lift_vs_trajectory")) for row in true_positives),
        "row_count": len(rows),
        "selected_count": sum(1 for row in rows if bool(row.get("prediction"))),
    }


def _load_rows(paths: tuple[Path, ...]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in _list_of_dicts(payload.get("rows")):
            current = dict(row)
            current["source_fit"] = str(path)
            rows.append(current)
    rows.sort(key=lambda row: (str(row.get("source_fit", "")), str(row.get("task_id", ""))))
    return rows


def _rows_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Slice | Task | Label | Prediction | Error | Score | Threshold | Lift | Gap | Span | Retention | Source Quality |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{Path(str(row.get('source_fit', ''))).name}` | "
            f"`{row.get('task_id', '')}` | "
            f"{bool(row.get('label'))} | "
            f"{bool(row.get('prediction'))} | "
            f"{bool(row.get('error'))} | "
            f"{_format_float(row.get('probe_signature_score'))} | "
            f"{_format_float(row.get('threshold'))} | "
            f"{_format_float(row.get('candidate_lift_vs_trajectory'))} | "
            f"{_format_float(row.get('measured_expected_gap_visibility_gain'))} | "
            f"{_format_float(row.get('measured_expected_span_evidence_gain'))} | "
            f"{_format_float(row.get('measured_distinct_retention_risk_visibility'))} | "
            f"{_format_float(row.get('source_quality'))} |"
        )
    return lines


def _feature_value(row: dict[str, object], feature: str) -> float:
    return _float(_dict(row.get("features")).get(feature, row.get(feature)))


def _boolean_feature(row: dict[str, object], feature: str) -> float:
    return 1.0 if _feature_value(row, feature) > 0.0 else 0.0


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


def _join_tasks(value: object) -> str:
    values = [str(item) for item in value] if isinstance(value, list) else []
    return ", ".join(f"`{item}`" for item in values) if values else "`none`"


def _task_ids(rows: list[dict[str, object]]) -> list[str]:
    return [str(row.get("task_id", "")) for row in rows]


if __name__ == "__main__":
    raise SystemExit(main())
