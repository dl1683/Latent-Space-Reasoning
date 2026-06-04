"""Fit an offline spend-value challenger over accumulated transfer rows.

This script deliberately stays offline: it uses only pre-repair diagnostics from
the spend-transfer rows and reports leave-one-slice-out transfer before any live
GPU spend gate can be considered.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_SPEND_EVALS = (
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v5_eval.json"),
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v6_eval.json"),
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v7_eval.json"),
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v8_eval.json"),
    Path("eval_results/diffusion_language/diffusion_independent_spend_transfer_v9_eval.json"),
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/diffusion_spend_value_model_v1.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_SPEND_VALUE_MODEL_V1.md")

NUMERIC_FEATURES = (
    "prompt_gap_count",
    "source_quality",
    "first_repairable_step",
    "source_task_delta_vs_trajectory",
)
BOOLEAN_FEATURES = (
    "first_repairable_step_missing",
    "single_repairability_prediction",
    "decomposed_prediction",
    "trajectory_relative_prediction",
    "learned_availability_prediction",
    "calibrated_availability_prediction",
)
BASELINE_POLICIES = (
    ("repairable_denoise_spend", "single_repairability_prediction"),
    ("decomposed_spend", "decomposed_prediction"),
    ("trajectory_relative_spend", "trajectory_relative_prediction"),
    ("learned_availability_predictor_v1", "learned_availability_prediction"),
    ("calibrated_availability_predictor_v1", "calibrated_availability_prediction"),
)


@dataclass(frozen=True)
class FeatureSpace:
    means: dict[str, float]
    scales: dict[str, float]
    missing_first_step_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spend-eval",
        action="append",
        dest="spend_evals",
        type=Path,
        help="Spend-transfer evaluation JSON. May be passed multiple times.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spend_eval_paths = tuple(args.spend_evals or DEFAULT_SPEND_EVALS)
    result = fit_spend_value_model(spend_eval_paths=spend_eval_paths)
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
                "report_output": str(args.report_output),
                "target_count": result["summary"]["target_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def fit_spend_value_model(*, spend_eval_paths: tuple[Path, ...]) -> dict[str, object]:
    rows = _load_rows(spend_eval_paths)
    in_sample = _fit_and_score(rows, rows)
    slice_results = []
    for held_out in sorted({str(row["source_eval"]) for row in rows}):
        train_rows = [row for row in rows if row["source_eval"] != held_out]
        test_rows = [row for row in rows if row["source_eval"] == held_out]
        scored = _fit_and_score(train_rows, test_rows)
        scored["held_out_eval"] = held_out
        slice_results.append(scored)
    loo_rows = [row for split in slice_results for row in _list_of_dicts(split.get("rows"))]
    return {
        "features": {
            "boolean": list(BOOLEAN_FEATURES),
            "numeric": list(NUMERIC_FEATURES),
            "score": "nearest_negative_distance_minus_nearest_positive_distance",
        },
        "generated_by": "experiments/fit_diffusion_spend_value_model.py",
        "baseline_policies": _baseline_policy_summaries(rows),
        "in_sample": in_sample,
        "inputs": {"spend_evals": [str(path) for path in spend_eval_paths]},
        "leave_one_slice_out": {
            "rows": loo_rows,
            "slice_results": slice_results,
            "summary": _summary(loo_rows),
        },
        "schema": "diffusion_spend_value_model.v1",
        "summary": {
            "profitable_count": sum(1 for row in rows if bool(row.get("profitable"))),
            "target_count": len(rows),
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    in_sample = _dict(_dict(result.get("in_sample")).get("summary"))
    loo = _dict(_dict(result.get("leave_one_slice_out")).get("summary"))
    lines = [
        "# Diffusion Spend Value Model V1",
        "",
        "This file is generated by `experiments/fit_diffusion_spend_value_model.py`.",
        "It tests a richer offline pre-repair value model before any live GPU spend gate.",
        "",
        "## Summary",
        "",
        f"- Target rows: `{summary.get('target_count', 0)}`",
        f"- Profitable rows: `{summary.get('profitable_count', 0)}`",
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
    if int(loo.get("false_negative_count", 0)) == 0:
        lines.append(
            "This offline challenger preserves every held-out positive repair, "
            "but it is still not a live GPU policy until it is locked and run on "
            "a fresh slice."
        )
    else:
        lines.append(
            "Do not promote this value model to a live spend gate. It is useful as "
            "a diagnostic, but leave-one-slice-out transfer still drops named "
            "profitable repairs."
        )
    lines.extend(
        [
            "",
            "## Model",
            "",
            (
                "Rows are embedded with pre-repair diagnostics only: prompt gap, "
                "source quality, first repairable denoise step, source-vs-trajectory "
                "task delta, missing-step indicator, and existing frozen spend-head "
                "predictions. The score is distance to the nearest negative prototype "
                "minus distance to the nearest positive prototype after standardizing "
                "numeric features on the training rows."
            ),
            "",
            "## Baselines",
            "",
            "| Policy | Selected | Errors | FP | FN | Lift Covered | Missed Profitable | No-Lift Selected |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for policy in _list_of_dicts(result.get("baseline_policies")):
        lines.append(
            "| "
            f"`{policy.get('policy_id', '')}` | "
            f"{int(policy.get('selected_count', 0))} | "
            f"{int(policy.get('error_count', 0))} | "
            f"{int(policy.get('false_positive_count', 0))} | "
            f"{int(policy.get('false_negative_count', 0))} | "
            f"{_format_float(policy.get('positive_lift_covered'))} | "
            f"{_join_tasks(policy.get('missed_profitable_tasks'))} | "
            f"{_join_tasks(policy.get('no_lift_selected_tasks'))} |"
        )
    lines.extend(
        [
            "",
            "## In-Sample Rows",
            "",
        ]
    )
    lines.extend(_rows_table(_list_of_dicts(_dict(result.get("in_sample")).get("rows"))))
    lines.extend(["", "## Leave-One-Slice-Out Rows", ""])
    lines.extend(_rows_table(_list_of_dicts(_dict(result.get("leave_one_slice_out")).get("rows"))))
    lines.extend(["", "## Slice Transfer", ""])
    lines.append("| Held-Out Eval | Errors | FP | FN | Lift Covered | Missed Profitable | No-Lift Selected |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- |")
    for split in _list_of_dicts(_dict(result.get("leave_one_slice_out")).get("slice_results")):
        split_summary = _dict(split.get("summary"))
        lines.append(
            "| "
            f"`{Path(str(split.get('held_out_eval', ''))).name}` | "
            f"{int(split_summary.get('error_count', 0))} | "
            f"{int(split_summary.get('false_positive_count', 0))} | "
            f"{int(split_summary.get('false_negative_count', 0))} | "
            f"{_format_float(split_summary.get('positive_lift_covered'))} | "
            f"{_join_tasks(split_summary.get('missed_profitable_tasks'))} | "
            f"{_join_tasks(split_summary.get('no_lift_selected_tasks'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "The model asks whether counterexample geometry is separable by "
                "local neighborhood structure rather than a single threshold. Treat "
                "in-sample gains as hypothesis generation. The transfer rows are the "
                "gate: missed positives remain a hard blocker for live spend gating "
                "unless their exact lift is explicitly traded for cost."
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
    train_scores = [_prototype_score(vector, row, train_vectors, train_rows) for vector, row in zip(train_vectors, train_rows)]
    threshold = _choose_threshold(train_rows, train_scores)
    test_vectors = [_vector(row, feature_space) for row in test_rows]
    scored_rows = []
    for row, vector in zip(test_rows, test_vectors):
        score = _prototype_score(vector, row, train_vectors, train_rows)
        prediction = score >= threshold
        enriched = {
            "error": prediction != bool(row.get("profitable")),
            "prediction": prediction,
            "profit_lift": _float(row.get("repair_lift")),
            "profitable": bool(row.get("profitable")),
            "prototype_score": score,
            "source_eval": row.get("source_eval", ""),
            "task_id": row.get("task_id", ""),
            "threshold": threshold,
        }
        for feature in NUMERIC_FEATURES:
            enriched[feature] = _feature_value(row, feature, feature_space)
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
        predictions = [score >= threshold for score in scores]
        false_positives = [
            row for row, prediction in zip(rows, predictions) if prediction and not bool(row.get("profitable"))
        ]
        false_negatives = [
            row for row, prediction in zip(rows, predictions) if not prediction and bool(row.get("profitable"))
        ]
        missed_lift = sum(_float(row.get("repair_lift")) for row in false_negatives)
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
    source_eval = str(row.get("source_eval", ""))
    for train_vector, train_row in zip(train_vectors, train_rows):
        if (
            str(train_row.get("task_id", "")) == task_id
            and str(train_row.get("source_eval", "")) == source_eval
        ):
            continue
        distance = _distance(vector, train_vector)
        if bool(train_row.get("profitable")):
            positive_distances.append(distance)
        else:
            negative_distances.append(distance)
    positive_distance = min(positive_distances) if positive_distances else 0.0
    negative_distance = min(negative_distances) if negative_distances else 0.0
    return negative_distance - positive_distance


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _feature_space(rows: list[dict[str, object]]) -> FeatureSpace:
    observed_steps = [
        _float(row.get("first_repairable_step"))
        for row in rows
        if row.get("first_repairable_step") is not None
    ]
    missing_first_step_value = (max(observed_steps) + 1.0) if observed_steps else 999.0
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for feature in NUMERIC_FEATURES:
        values = [_feature_value(row, feature, None, missing_first_step_value) for row in rows]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        means[feature] = mean
        scales[feature] = math.sqrt(variance) or 1.0
    return FeatureSpace(
        means=means,
        scales=scales,
        missing_first_step_value=missing_first_step_value,
    )


def _vector(row: dict[str, object], feature_space: FeatureSpace) -> tuple[float, ...]:
    numeric = [
        (_feature_value(row, feature, feature_space) - feature_space.means[feature])
        / feature_space.scales[feature]
        for feature in NUMERIC_FEATURES
    ]
    boolean = [1.0 if _boolean_feature(row, feature) else 0.0 for feature in BOOLEAN_FEATURES]
    return tuple(numeric + boolean)


def _feature_value(
    row: dict[str, object],
    feature: str,
    feature_space: FeatureSpace | None,
    missing_first_step_value: float | None = None,
) -> float:
    if feature == "first_repairable_step" and row.get(feature) is None:
        if feature_space is not None:
            return feature_space.missing_first_step_value
        if missing_first_step_value is not None:
            return missing_first_step_value
    return _float(row.get(feature))


def _boolean_feature(row: dict[str, object], feature: str) -> bool:
    if feature == "first_repairable_step_missing":
        return row.get("first_repairable_step") is None
    return bool(row.get(feature))


def _load_rows(paths: tuple[Path, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in _list_of_dicts(payload.get("rows")):
            enriched = dict(row)
            enriched["source_eval"] = str(path)
            rows.append(enriched)
    return rows


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    true_positive = [row for row in rows if bool(row.get("prediction")) and bool(row.get("profitable"))]
    false_positive = [row for row in rows if bool(row.get("prediction")) and not bool(row.get("profitable"))]
    false_negative = [row for row in rows if not bool(row.get("prediction")) and bool(row.get("profitable"))]
    true_negative = [row for row in rows if not bool(row.get("prediction")) and not bool(row.get("profitable"))]
    return {
        "error_count": len(false_positive) + len(false_negative),
        "false_negative_count": len(false_negative),
        "false_positive_count": len(false_positive),
        "missed_lift": sum(_float(row.get("profit_lift")) for row in false_negative),
        "missed_profitable_tasks": _task_ids(false_negative),
        "no_lift_selected_tasks": _task_ids(false_positive),
        "positive_lift_covered": sum(_float(row.get("profit_lift")) for row in true_positive),
        "selected_count": len(true_positive) + len(false_positive),
        "target_count": len(rows),
        "true_negative_count": len(true_negative),
        "true_positive_count": len(true_positive),
    }


def _baseline_policy_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for policy_id, field in BASELINE_POLICIES:
        scored_rows = [
            {
                "prediction": bool(row.get(field)),
                "profit_lift": _float(row.get("repair_lift")),
                "profitable": bool(row.get("profitable")),
                "task_id": row.get("task_id", ""),
            }
            for row in rows
        ]
        summary = _summary(scored_rows)
        summary["policy_id"] = policy_id
        summary["prediction_field"] = field
        summaries.append(summary)
    return summaries


def _rows_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Task | Source Eval | Label | Predict | Score | Threshold | Repair Lift | Gap | Source Quality | Source Delta | Error |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('task_id', '')}` | "
            f"`{Path(str(row.get('source_eval', ''))).name}` | "
            f"{bool(row.get('profitable'))} | "
            f"{bool(row.get('prediction'))} | "
            f"{_format_float(row.get('prototype_score'))} | "
            f"{_format_float(row.get('threshold'))} | "
            f"{_format_float(row.get('profit_lift'))} | "
            f"{_format_float(row.get('prompt_gap_count'))} | "
            f"{_format_float(row.get('source_quality'))} | "
            f"{_format_float(row.get('source_task_delta_vs_trajectory'))} | "
            f"{bool(row.get('error'))} |"
        )
    return lines


def _task_ids(rows: Iterable[dict[str, object]]) -> list[str]:
    return [str(row.get("task_id", "")) for row in rows]


def _join_tasks(value: object) -> str:
    tasks = [str(item) for item in _list(value)]
    if not tasks:
        return "`none`"
    return ", ".join(f"`{task}`" for task in tasks)


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
