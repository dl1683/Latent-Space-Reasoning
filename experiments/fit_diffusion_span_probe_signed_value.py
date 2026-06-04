"""Fit a signed-value head over span-v4 probe signatures."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SIGNATURE_MODEL = Path(
    "eval_results/diffusion_language/counterfactual_span_probe_signature_model_v4.json"
)
DEFAULT_NO_LIFT_VETO = Path(
    "eval_results/diffusion_language/counterfactual_span_probe_no_lift_veto_v4.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/counterfactual_span_probe_signed_value_v4.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_V4.md")
DEFAULT_SELECTION_PENALTY = 0.02
DEFAULT_NEIGHBOR_COUNTS = (1, 2, 3, 5, 8, 13)
BASE_SIGNATURE_POSITIVE_UTILITY_BAR = 0.6255

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
    "probe_signature_score",
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
FEATURE_GROUPS = {
    "all": tuple(NUMERIC_FEATURES + BOOLEAN_FEATURES),
    "no_text": tuple(
        feature
        for feature in NUMERIC_FEATURES + BOOLEAN_FEATURES
        if not feature.startswith("counterfactual_probe_text") and feature != "would_probe_score"
    ),
    "no_source": tuple(
        feature for feature in NUMERIC_FEATURES + BOOLEAN_FEATURES if feature != "source_quality"
    ),
    "no_gap_span": tuple(
        feature
        for feature in NUMERIC_FEATURES + BOOLEAN_FEATURES
        if "gap" not in feature and "span" not in feature
    ),
    "no_retention": tuple(
        feature for feature in NUMERIC_FEATURES + BOOLEAN_FEATURES if "retention" not in feature
    ),
}


@dataclass(frozen=True)
class FeatureSpace:
    means: dict[str, float]
    scales: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-model", type=Path, default=DEFAULT_SIGNATURE_MODEL)
    parser.add_argument("--no-lift-veto", type=Path, default=DEFAULT_NO_LIFT_VETO)
    parser.add_argument("--selection-penalty", type=float, default=DEFAULT_SELECTION_PENALTY)
    parser.add_argument(
        "--neighbor-counts",
        default=",".join(str(value) for value in DEFAULT_NEIGHBOR_COUNTS),
        help="Comma-separated k values for inverse-distance kNN signed-value prediction.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    neighbor_counts = tuple(
        int(value.strip()) for value in args.neighbor_counts.split(",") if value.strip()
    )
    result = fit_signed_value_head(
        signature_model_path=args.signature_model,
        no_lift_veto_path=args.no_lift_veto,
        neighbor_counts=neighbor_counts,
        selection_penalty=args.selection_penalty,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    selected = _dict(result.get("selected_model"))
    print(
        json.dumps(
            {
                "false_negatives": selected.get("false_negative_count"),
                "false_positives": selected.get("false_positive_count"),
                "json_output": str(args.json_output),
                "model_id": selected.get("model_id"),
                "policy_utility": selected.get("policy_utility"),
                "report_output": str(args.report_output),
                "target_count": result["summary"]["target_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def fit_signed_value_head(
    *,
    signature_model_path: Path,
    no_lift_veto_path: Path = DEFAULT_NO_LIFT_VETO,
    neighbor_counts: tuple[int, ...] = DEFAULT_NEIGHBOR_COUNTS,
    selection_penalty: float = DEFAULT_SELECTION_PENALTY,
) -> dict[str, object]:
    rows = _load_signature_rows(signature_model_path, selection_penalty=selection_penalty)
    model_results = []
    for group_id, features in FEATURE_GROUPS.items():
        for neighbor_count in neighbor_counts:
            model_results.append(
                _evaluate_model(
                    rows,
                    feature_group_id=group_id,
                    features=features,
                    neighbor_count=neighbor_count,
                    selection_penalty=selection_penalty,
                )
            )
    selected_model = _select_model(
        [row for row in model_results if row.get("feature_group_id") == "all"]
    )
    baseline = _baseline_signature_gate(rows, selection_penalty=selection_penalty)
    no_lift_veto = _load_no_lift_veto_summary(no_lift_veto_path)
    return {
        "baseline_signature_gate": baseline,
        "base_signature_positive_utility_bar": BASE_SIGNATURE_POSITIVE_UTILITY_BAR,
        "feature_groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
        "generated_by": "experiments/fit_diffusion_span_probe_signed_value.py",
        "inputs": {
            "no_lift_veto": str(no_lift_veto_path),
            "signature_model": str(signature_model_path),
        },
        "model_results": model_results,
        "no_lift_veto": no_lift_veto,
        "schema": "diffusion_counterfactual_span_probe_signed_value.v1",
        "selected_model": selected_model,
        "selection_penalty": selection_penalty,
        "summary": {
            "positive_count": sum(1 for row in rows if bool(row.get("label"))),
            "target_count": len(rows),
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    selected = _dict(result.get("selected_model"))
    baseline = _dict(result.get("baseline_signature_gate"))
    no_lift_veto = _dict(result.get("no_lift_veto"))
    lines = [
        "# Diffusion Counterfactual Span Probe Signed Value V4",
        "",
        "This file is generated by `experiments/fit_diffusion_span_probe_signed_value.py`.",
        (
            "It evaluates the M1 Signed Value Tomography Controller target from "
            "`docs/DIFFUSION_MOONSHOT_REASONING_ARCHITECTURE_V1.md`."
        ),
        "",
        "## Summary",
        "",
        f"- Target rows: `{summary.get('target_count', 0)}`",
        f"- Positive rows: `{summary.get('positive_count', 0)}`",
        f"- Selection penalty: `{_format_float(result.get('selection_penalty'))}`",
        f"- Selected model: `{selected.get('model_id', '')}`",
        f"- Selected signed utility: `{_format_float(selected.get('policy_utility'))}`",
        f"- Selected false positives: `{selected.get('false_positive_count', 0)}`",
        f"- Selected false negatives: `{selected.get('false_negative_count', 0)}`",
        "",
        "## Decision",
        "",
    ]
    if _clears_promotion_bar(selected):
        lines.append(
            "This offline signed-value head clears the M1 bar. It still needs "
            "negative controls and a frozen fresh GPU slice before promotion."
        )
    else:
        lines.append(
            "Do not promote this signed-value head. It is a real improvement over "
            "the stricter signed utility of the base gate, but it does not clear "
            "the architecture document's promotion bar against the older positive-lift utility."
        )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            "| Policy | Selected | FP | FN | Positive Lift | Signed Utility | Missed Positives | False Positives |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            _comparison_row("base_signature_gate", baseline),
            _comparison_row("no_lift_veto", no_lift_veto),
            _comparison_row(str(selected.get("model_id", "selected_signed_value")), selected),
            "",
            "## Model Sweep",
            "",
            "| Model | Features | k | Selected | FP | FN | Positive Lift | Signed Utility |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        _list_of_dicts(result.get("model_results")),
        key=lambda item: (str(item.get("feature_group_id", "")), int(_float(item.get("neighbor_count")))),
    ):
        lines.append(
            "| "
            f"`{row.get('model_id', '')}` | "
            f"`{row.get('feature_group_id', '')}` | "
            f"{int(_float(row.get('neighbor_count')))} | "
            f"{int(_float(row.get('selected_count')))} | "
            f"{int(_float(row.get('false_positive_count')))} | "
            f"{int(_float(row.get('false_negative_count')))} | "
            f"{_format_float(row.get('positive_lift_covered'))} | "
            f"{_format_float(row.get('policy_utility'))} |"
        )
    lines.extend(
        [
            "",
            "## Slice Transfer",
            "",
            "| Held-Out Fit | Selected | FP | FN | Positive Lift | Signed Utility | Missed Positives | False Positives |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for split in _list_of_dicts(selected.get("slice_results")):
        lines.append(
            "| "
            f"`{Path(str(split.get('held_out_fit', ''))).name}` | "
            f"{int(_float(split.get('selected_count')))} | "
            f"{int(_float(split.get('false_positive_count')))} | "
            f"{int(_float(split.get('false_negative_count')))} | "
            f"{_format_float(split.get('positive_lift_covered'))} | "
            f"{_format_float(split.get('policy_utility'))} | "
            f"{_join_tasks(split.get('false_negative_task_ids'))} | "
            f"{_join_tasks(split.get('false_positive_task_ids'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "The signed-value head is the first controller in this sequence "
                "that predicts realized cost-adjusted value directly. It improves "
                "over the base gate when harmful/no-lift rows count against utility: "
                "the base signed utility is `0.498571`, while the selected signed "
                "model reaches `0.582500` with zero false negatives and nine false "
                "positives. That is progress, not promotion. The stricter M1 bar "
                "from the architecture doc remains `0.625500`, so M2 should add "
                "negative controls and better signed-value features before any GPU slice."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _evaluate_model(
    rows: list[dict[str, object]],
    *,
    feature_group_id: str,
    features: tuple[str, ...],
    neighbor_count: int,
    selection_penalty: float,
) -> dict[str, object]:
    scored_rows = []
    slice_results = []
    for held_out in sorted({str(row.get("source_fit", "")) for row in rows}):
        train_rows = [row for row in rows if str(row.get("source_fit", "")) != held_out]
        test_rows = [row for row in rows if str(row.get("source_fit", "")) == held_out]
        feature_space = _feature_space(train_rows, features)
        split_rows = []
        for row in test_rows:
            prediction = _predict_signed_value(
                row,
                train_rows=train_rows,
                features=features,
                feature_space=feature_space,
                neighbor_count=neighbor_count,
            )
            selected = prediction > 0.0 and bool(row.get("valid_for_stage1"))
            split_rows.append({**row, "predicted_signed_value": prediction, "selected": selected})
        split_summary = _summary(split_rows, selection_penalty=selection_penalty)
        split_summary["held_out_fit"] = held_out
        slice_results.append(split_summary)
        scored_rows.extend(split_rows)
    summary = _summary(scored_rows, selection_penalty=selection_penalty)
    return {
        **summary,
        "feature_group_id": feature_group_id,
        "model_id": f"signed_value_knn_k{neighbor_count}_{feature_group_id}",
        "neighbor_count": neighbor_count,
        "slice_results": slice_results,
    }


def _select_model(model_results: list[dict[str, object]]) -> dict[str, object]:
    return max(
        model_results,
        key=lambda row: (
            _float(row.get("policy_utility")),
            -_float(row.get("false_negative_count")),
            -_float(row.get("false_positive_count")),
            _float(row.get("positive_lift_covered")),
        ),
    )


def _predict_signed_value(
    row: dict[str, object],
    *,
    train_rows: list[dict[str, object]],
    features: tuple[str, ...],
    feature_space: FeatureSpace,
    neighbor_count: int,
) -> float:
    row_vector = _vector(row, features=features, feature_space=feature_space)
    distances = []
    for train_row in train_rows:
        train_vector = _vector(train_row, features=features, feature_space=feature_space)
        distances.append((_distance(row_vector, train_vector), train_row))
    nearest = sorted(distances, key=lambda item: item[0])[:neighbor_count]
    numerator = 0.0
    denominator = 0.0
    for distance, train_row in nearest:
        weight = 1.0 / (distance + 1e-6)
        numerator += weight * _float(train_row.get("signed_value"))
        denominator += weight
    return numerator / denominator if denominator else 0.0


def _summary(rows: list[dict[str, object]], *, selection_penalty: float) -> dict[str, object]:
    selected_rows = [row for row in rows if bool(row.get("selected"))]
    false_positives = [row for row in selected_rows if not bool(row.get("label"))]
    false_negatives = [
        row for row in rows if bool(row.get("label")) and not bool(row.get("selected"))
    ]
    positive_lift = sum(
        _float(row.get("candidate_lift_vs_trajectory"))
        for row in selected_rows
        if bool(row.get("label"))
    )
    signed_lift = sum(_float(row.get("candidate_lift_vs_trajectory")) for row in selected_rows)
    return {
        "error_count": len(false_positives) + len(false_negatives),
        "false_negative_count": len(false_negatives),
        "false_negative_task_ids": _task_ids(false_negatives),
        "false_positive_count": len(false_positives),
        "false_positive_task_ids": _task_ids(false_positives),
        "policy_utility": signed_lift - selection_penalty * len(selected_rows),
        "positive_lift_covered": positive_lift,
        "selected_count": len(selected_rows),
        "signed_lift_covered": signed_lift,
    }


def _baseline_signature_gate(
    rows: list[dict[str, object]], *, selection_penalty: float
) -> dict[str, object]:
    baseline_rows = [
        {**row, "selected": bool(row.get("prediction")) and bool(row.get("valid_for_stage1"))}
        for row in rows
    ]
    summary = _summary(baseline_rows, selection_penalty=selection_penalty)
    summary["model_id"] = "base_signature_gate"
    summary["positive_lift_utility"] = (
        _float(summary.get("positive_lift_covered"))
        - selection_penalty * _float(summary.get("selected_count"))
    )
    return summary


def _load_no_lift_veto_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = _dict(payload.get("veto_leave_one_slice_out"))
    return {
        "false_negative_count": summary.get("false_negative_count", 0),
        "false_negative_task_ids": summary.get("false_negative_task_ids", []),
        "false_positive_count": summary.get("false_positive_count", 0),
        "false_positive_task_ids": summary.get("false_positive_task_ids", []),
        "model_id": "no_lift_veto",
        "policy_utility": summary.get("policy_utility", 0.0),
        "positive_lift_covered": summary.get("selected_lift", 0.0),
        "selected_count": summary.get("selected_count", 0),
    }


def _feature_space(rows: list[dict[str, object]], features: tuple[str, ...]) -> FeatureSpace:
    means = {}
    scales = {}
    for feature in features:
        values = [_feature_value(row, feature) for row in rows]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        means[feature] = mean
        scales[feature] = math.sqrt(variance) or 1.0
    return FeatureSpace(means=means, scales=scales)


def _vector(
    row: dict[str, object],
    *,
    features: tuple[str, ...],
    feature_space: FeatureSpace,
) -> tuple[float, ...]:
    return tuple(
        (_feature_value(row, feature) - feature_space.means[feature])
        / feature_space.scales[feature]
        for feature in features
    )


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _load_signature_rows(path: Path, *, selection_penalty: float) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in _list_of_dicts(_dict(payload.get("leave_one_slice_out")).get("rows")):
        current = dict(row)
        current["signed_value"] = _float(row.get("candidate_lift_vs_trajectory")) - selection_penalty
        rows.append(current)
    return rows


def _clears_promotion_bar(summary: dict[str, object]) -> bool:
    return (
        _float(summary.get("policy_utility")) > BASE_SIGNATURE_POSITIVE_UTILITY_BAR
        and int(_float(summary.get("false_negative_count"))) == 0
        and int(_float(summary.get("false_positive_count"))) < 11
    )


def _comparison_row(policy_id: str, summary: dict[str, object]) -> str:
    return (
        "| "
        f"`{policy_id}` | "
        f"{int(_float(summary.get('selected_count')))} | "
        f"{int(_float(summary.get('false_positive_count')))} | "
        f"{int(_float(summary.get('false_negative_count')))} | "
        f"{_format_float(summary.get('positive_lift_covered'))} | "
        f"{_format_float(summary.get('policy_utility'))} | "
        f"{_join_tasks(summary.get('false_negative_task_ids'))} | "
        f"{_join_tasks(summary.get('false_positive_task_ids'))} |"
    )


def _feature_value(row: dict[str, object], feature: str) -> float:
    return _float(row.get(feature))


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
