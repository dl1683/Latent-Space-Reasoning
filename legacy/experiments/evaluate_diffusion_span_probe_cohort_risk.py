"""Evaluate risk-adjusted cohort calibration for span-v4 signed-value heads."""

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

DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/counterfactual_span_probe_cohort_risk_v4.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COHORT_RISK_V4.md")
DEFAULT_WEAK_SLICE = "counterfactual_span_validated_probe_stage1_gate_v4_transfer_v3_planning.json"
DEFAULT_NEIGHBOR_COUNTS = (8, 13)
DEFAULT_STD_PENALTIES = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_NEGATIVE_FRACTION_PENALTIES = (0.0, 0.005, 0.01, 0.02, 0.03, 0.04)
DEFAULT_MARGINS = (-0.02, -0.01, 0.0, 0.005, 0.01, 0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-model", type=Path, default=DEFAULT_SIGNATURE_MODEL)
    parser.add_argument("--selection-penalty", type=float, default=DEFAULT_SELECTION_PENALTY)
    parser.add_argument("--weak-slice", default=DEFAULT_WEAK_SLICE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_cohort_risk(
        signature_model_path=args.signature_model,
        selection_penalty=args.selection_penalty,
        weak_slice=args.weak_slice,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    selected = _dict(result.get("selected_model"))
    weak = _dict(selected.get("weak_slice_summary"))
    print(
        json.dumps(
            {
                "false_negatives": selected.get("false_negative_count"),
                "false_positives": selected.get("false_positive_count"),
                "json_output": str(args.json_output),
                "model_id": selected.get("model_id"),
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


def evaluate_cohort_risk(
    *,
    signature_model_path: Path,
    selection_penalty: float = DEFAULT_SELECTION_PENALTY,
    weak_slice: str = DEFAULT_WEAK_SLICE,
) -> dict[str, object]:
    rows = _load_signature_rows(signature_model_path, selection_penalty=selection_penalty)
    model_results = []
    for neighbor_count in DEFAULT_NEIGHBOR_COUNTS:
        for std_penalty in DEFAULT_STD_PENALTIES:
            for negative_fraction_penalty in DEFAULT_NEGATIVE_FRACTION_PENALTIES:
                for margin in DEFAULT_MARGINS:
                    model_results.append(
                        _evaluate_model(
                            rows,
                            neighbor_count=neighbor_count,
                            std_penalty=std_penalty,
                            negative_fraction_penalty=negative_fraction_penalty,
                            margin=margin,
                            selection_penalty=selection_penalty,
                            weak_slice=weak_slice,
                        )
                    )
    selected_model = max(
        model_results,
        key=lambda row: (
            _float(row.get("policy_utility")),
            -_float(row.get("false_negative_count")),
            -_float(row.get("false_positive_count")),
            _float(row.get("positive_lift_covered")),
        ),
    )
    top_model_results = sorted(
        model_results,
        key=lambda row: _float(row.get("policy_utility")),
        reverse=True,
    )[:40]
    return {
        "base_signature_positive_utility_bar": BASE_SIGNATURE_POSITIVE_UTILITY_BAR,
        "generated_by": "experiments/evaluate_diffusion_span_probe_cohort_risk.py",
        "inputs": {"signature_model": str(signature_model_path)},
        "model_results": top_model_results,
        "schema": "diffusion_counterfactual_span_probe_cohort_risk.v1",
        "selected_model": selected_model,
        "selection_penalty": selection_penalty,
        "summary": {
            "evaluated_model_count": len(model_results),
            "positive_count": sum(1 for row in rows if bool(row.get("label"))),
            "target_count": len(rows),
            "weak_slice": weak_slice,
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    selected = _dict(result.get("selected_model"))
    weak = _dict(selected.get("weak_slice_summary"))
    lines = [
        "# Diffusion Counterfactual Span Probe Cohort Risk V4",
        "",
        "This file is generated by `experiments/evaluate_diffusion_span_probe_cohort_risk.py`.",
        "",
        "## Summary",
        "",
        f"- Selected model: `{selected.get('model_id', '')}`",
        f"- Selected signed utility: `{_format_float(selected.get('policy_utility'))}`",
        f"- Selected false positives: `{selected.get('false_positive_count', 0)}`",
        f"- Selected false negatives: `{selected.get('false_negative_count', 0)}`",
        f"- Weak-slice selected rows: `{weak.get('selected_count', 0)}`",
        f"- Weak-slice false positives: `{weak.get('false_positive_count', 0)}`",
        f"- Weak-slice signed utility: `{_format_float(weak.get('policy_utility'))}`",
        "",
        "## Decision",
        "",
    ]
    if _passes_m1(selected) and _passes_weak_slice(weak):
        lines.append(
            "This risk-adjusted head clears the offline bar and reduces the weak-slice "
            "cohort failure. It still needs feature-family controls before any GPU slice."
        )
    elif _passes_m1(selected):
        lines.append(
            "Do not promote this head. It clears the old M1 utility bar, but it still "
            "selects the weak `plan_017`-`plan_024` cohort wholesale and therefore "
            "fails the M2.5 cohort-calibration requirement."
        )
    else:
        lines.append(
            "Do not promote this head. The cohort-risk sweep does not clear the M1 "
            "signed-utility bar under leave-one-slice-out evaluation."
        )
    lines.extend(
        [
            "",
            "## Selected Model",
            "",
            "| Model | k | Std Penalty | Negative-Fraction Penalty | Margin | Selected | FP | FN | Positive Lift | Signed Utility |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            _model_row(selected),
            "",
            "## Top Sweep Rows",
            "",
            "| Model | Selected | FP | FN | Positive Lift | Signed Utility | Weak FP | Weak Utility |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        _list_of_dicts(result.get("model_results")),
        key=lambda item: _float(item.get("policy_utility")),
        reverse=True,
    )[:12]:
        weak_row = _dict(row.get("weak_slice_summary"))
        lines.append(
            "| "
            f"`{row.get('model_id', '')}` | "
            f"{int(_float(row.get('selected_count')))} | "
            f"{int(_float(row.get('false_positive_count')))} | "
            f"{int(_float(row.get('false_negative_count')))} | "
            f"{_format_float(row.get('positive_lift_covered'))} | "
            f"{_format_float(row.get('policy_utility'))} | "
            f"{int(_float(weak_row.get('false_positive_count')))} | "
            f"{_format_float(weak_row.get('policy_utility'))} |"
        )
    lines.extend(
        [
            "",
            "## Weak-Slice Transfer",
            "",
            f"- False-positive tasks: {_join_tasks(weak.get('false_positive_task_ids'))}",
            f"- Positive tasks: {_join_tasks(weak.get('positive_task_ids'))}",
            "",
            "## Reading",
            "",
            (
                "Neighbor-risk calibration is useful but incomplete. The selected "
                "risk-adjusted head improves global signed utility by dropping some "
                "non-weak no-lift rows, yet it does not learn the weak-slice cohort "
                "density boundary. The next feature needs a stronger slice-local "
                "density, uncertainty, or disagreement signal before the controller "
                "earns a frozen GPU test."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _evaluate_model(
    rows: list[dict[str, object]],
    *,
    neighbor_count: int,
    std_penalty: float,
    negative_fraction_penalty: float,
    margin: float,
    selection_penalty: float,
    weak_slice: str,
) -> dict[str, object]:
    scored_rows = []
    slice_results = []
    features = FEATURE_GROUPS["all"]
    for held_out in sorted({str(row.get("source_fit", "")) for row in rows}):
        train_rows = [row for row in rows if str(row.get("source_fit", "")) != held_out]
        test_rows = [row for row in rows if str(row.get("source_fit", "")) == held_out]
        feature_space = _feature_space(train_rows, features)
        split_rows = [
            _score_row(
                row,
                train_rows=train_rows,
                features=features,
                feature_space=feature_space,
                neighbor_count=neighbor_count,
                std_penalty=std_penalty,
                negative_fraction_penalty=negative_fraction_penalty,
                margin=margin,
            )
            for row in test_rows
        ]
        split_summary = _summary(split_rows, selection_penalty=selection_penalty)
        split_summary["held_out_fit"] = held_out
        slice_results.append(split_summary)
        scored_rows.extend(split_rows)
    summary = _summary(scored_rows, selection_penalty=selection_penalty)
    weak_rows = [row for row in scored_rows if Path(str(row.get("source_fit", ""))).name == weak_slice]
    weak_summary = _summary(weak_rows, selection_penalty=selection_penalty)
    weak_summary["positive_task_ids"] = [str(row.get("task_id", "")) for row in weak_rows if bool(row.get("label"))]
    return {
        **summary,
        "margin": margin,
        "model_id": _model_id(neighbor_count, std_penalty, negative_fraction_penalty, margin),
        "negative_fraction_penalty": negative_fraction_penalty,
        "neighbor_count": neighbor_count,
        "slice_results": slice_results,
        "std_penalty": std_penalty,
        "weak_slice_summary": weak_summary,
    }


def _score_row(
    row: dict[str, object],
    *,
    train_rows: list[dict[str, object]],
    features: tuple[str, ...],
    feature_space: object,
    neighbor_count: int,
    std_penalty: float,
    negative_fraction_penalty: float,
    margin: float,
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
    weights = [1.0 / (distance + 1e-6) for distance, _ in nearest]
    signed_values = [_float(train_row.get("signed_value")) for _, train_row in nearest]
    predicted = sum(weight * value for weight, value in zip(weights, signed_values)) / sum(weights)
    mean_value = sum(signed_values) / len(signed_values)
    std_value = math.sqrt(
        sum((value - mean_value) ** 2 for value in signed_values) / len(signed_values)
    )
    negative_fraction = sum(value <= 0.0 for value in signed_values) / len(signed_values)
    risk_adjusted = predicted - std_penalty * std_value - negative_fraction_penalty * negative_fraction
    return {
        **row,
        "cohort_negative_fraction": negative_fraction,
        "cohort_signed_value_std": std_value,
        "predicted_signed_value": predicted,
        "risk_adjusted_signed_value": risk_adjusted,
        "selected": risk_adjusted > margin and bool(row.get("valid_for_stage1")),
    }


def _passes_m1(summary: dict[str, object]) -> bool:
    return (
        _float(summary.get("policy_utility")) > BASE_SIGNATURE_POSITIVE_UTILITY_BAR
        and int(_float(summary.get("false_negative_count"))) == 0
        and int(_float(summary.get("false_positive_count"))) < 11
    )


def _passes_weak_slice(summary: dict[str, object]) -> bool:
    return (
        int(_float(summary.get("false_negative_count"))) == 0
        and int(_float(summary.get("false_positive_count"))) < 6
        and _float(summary.get("policy_utility")) > 0.001428571428571418
    )


def _model_row(row: dict[str, object]) -> str:
    return (
        "| "
        f"`{row.get('model_id', '')}` | "
        f"{int(_float(row.get('neighbor_count')))} | "
        f"{_format_float(row.get('std_penalty'))} | "
        f"{_format_float(row.get('negative_fraction_penalty'))} | "
        f"{_format_float(row.get('margin'))} | "
        f"{int(_float(row.get('selected_count')))} | "
        f"{int(_float(row.get('false_positive_count')))} | "
        f"{int(_float(row.get('false_negative_count')))} | "
        f"{_format_float(row.get('positive_lift_covered'))} | "
        f"{_format_float(row.get('policy_utility'))} |"
    )


def _model_id(
    neighbor_count: int,
    std_penalty: float,
    negative_fraction_penalty: float,
    margin: float,
) -> str:
    return (
        f"cohort_risk_k{neighbor_count}"
        f"_std{_slug_float(std_penalty)}"
        f"_negfrac{_slug_float(negative_fraction_penalty)}"
        f"_margin{_slug_float(margin)}"
    )


def _slug_float(value: float) -> str:
    sign = "n" if value < 0 else ""
    return sign + f"{abs(value):.6f}".replace(".", "p").rstrip("0").rstrip("p")


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


if __name__ == "__main__":
    raise SystemExit(main())
