"""Evaluate a no-lift veto head over span-v4 probe signature selections."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

DEFAULT_SIGNATURE_MODEL = Path(
    "eval_results/diffusion_language/counterfactual_span_probe_signature_model_v4.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/counterfactual_span_probe_no_lift_veto_v4.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    "DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_NO_LIFT_VETO_V4.md"
)
DEFAULT_SELECTION_PENALTY = 0.02

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-model", type=Path, default=DEFAULT_SIGNATURE_MODEL)
    parser.add_argument("--selection-penalty", type=float, default=DEFAULT_SELECTION_PENALTY)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_no_lift_veto(
        signature_model_path=args.signature_model,
        selection_penalty=args.selection_penalty,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline_loo_errors": result["baseline_leave_one_slice_out"]["error_count"],
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "target_count": result["summary"]["target_count"],
                "veto_loo_errors": result["veto_leave_one_slice_out"]["error_count"],
                "veto_loo_false_negatives": result["veto_leave_one_slice_out"][
                    "false_negative_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def evaluate_no_lift_veto(
    *,
    signature_model_path: Path,
    selection_penalty: float = DEFAULT_SELECTION_PENALTY,
) -> dict[str, object]:
    payload = json.loads(signature_model_path.read_text(encoding="utf-8"))
    rows = _list_of_dicts(_dict(payload.get("leave_one_slice_out")).get("rows"))
    candidates = _candidate_rules(rows)
    slice_results = []
    for held_out in sorted({str(row.get("source_fit", "")) for row in rows}):
        train_rows = [row for row in rows if str(row.get("source_fit", "")) != held_out]
        test_rows = [row for row in rows if str(row.get("source_fit", "")) == held_out]
        best_rule = _fit_veto_rule(
            train_rows,
            candidates=candidates,
            selection_penalty=selection_penalty,
        )
        train_summary = _score_veto_rule(
            train_rows,
            rule=best_rule,
            selection_penalty=selection_penalty,
        )
        test_summary = _score_veto_rule(
            test_rows,
            rule=best_rule,
            selection_penalty=selection_penalty,
        )
        slice_results.append(
            {
                "held_out_fit": held_out,
                "rule": [_condition_dict(condition) for condition in best_rule],
                "test_summary": test_summary,
                "train_summary": train_summary,
            }
        )
    baseline = _score_veto_rule(rows, rule=(), selection_penalty=selection_penalty)
    veto = _combine_slices(slice_results)
    return {
        "baseline_leave_one_slice_out": baseline,
        "candidate_rule_count": len(candidates),
        "features": list(NUMERIC_FEATURES),
        "generated_by": "experiments/evaluate_diffusion_span_probe_no_lift_veto.py",
        "inputs": {"signature_model": str(signature_model_path)},
        "schema": "diffusion_counterfactual_span_probe_no_lift_veto.v1",
        "selection_penalty": selection_penalty,
        "slice_results": slice_results,
        "summary": {
            "base_selected_count": sum(1 for row in rows if bool(row.get("prediction"))),
            "positive_count": sum(1 for row in rows if bool(row.get("label"))),
            "target_count": len(rows),
        },
        "veto_leave_one_slice_out": veto,
    }


def render_markdown(result: dict[str, object]) -> str:
    summary = _dict(result.get("summary"))
    baseline = _dict(result.get("baseline_leave_one_slice_out"))
    veto = _dict(result.get("veto_leave_one_slice_out"))
    lines = [
        "# Diffusion Counterfactual Span Probe No-Lift Veto V4",
        "",
        (
            "This file is generated by "
            "`experiments/evaluate_diffusion_span_probe_no_lift_veto.py`."
        ),
        (
            "It tests whether a separate one- or two-condition veto head can "
            "remove no-lift span-probe selections after the recall-biased "
            "signature gate has already selected them."
        ),
        "",
        "## Summary",
        "",
        f"- Target rows: `{summary.get('target_count', 0)}`",
        f"- Base selected rows: `{summary.get('base_selected_count', 0)}`",
        f"- Positive rows: `{summary.get('positive_count', 0)}`",
        f"- Candidate veto rules: `{result.get('candidate_rule_count', 0)}`",
        f"- Selection penalty: `{_format_float(result.get('selection_penalty'))}`",
        "",
        "## Decision",
        "",
        (
            "Do not promote this no-lift veto. It reduces false positives on the "
            "current leave-slice-out replay, but the reduction is paid for by "
            "new false negatives and lower practical-penalty utility. The veto "
            "families are useful as failure geometry, not a controller."
        ),
        "",
        "## Aggregate Transfer",
        "",
        "| Policy | Selected | FP | FN | Lift | Utility | Missed Positives | False Positives | Vetoed Rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        _summary_row("base_signature_gate", baseline),
        _summary_row("leave_slice_out_veto", veto),
        "",
        "## Slice Rules",
        "",
        "| Held-Out Fit | Rule | Train Utility | Test Utility | Test FP | Test FN | Test Vetoed Rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for split in _list_of_dicts(result.get("slice_results")):
        train = _dict(split.get("train_summary"))
        test = _dict(split.get("test_summary"))
        lines.append(
            "| "
            f"`{Path(str(split.get('held_out_fit', ''))).name}` | "
            f"{_format_rule(_list_of_dicts(split.get('rule')))} | "
            f"{_format_float(train.get('policy_utility'))} | "
            f"{_format_float(test.get('policy_utility'))} | "
            f"{int(_float(test.get('false_positive_count')))} | "
            f"{int(_float(test.get('false_negative_count')))} | "
            f"{_join_tasks(test.get('vetoed_task_ids'))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "This is the first direct test of the obvious two-stage controller: "
                "select high-recall probe opportunities, then veto likely no-lift "
                "spends. The leave-slice-out veto improves specificity only weakly "
                "and harms recall, so the next model should predict realized lift "
                "as a signed value rather than hard-vetoing with threshold fragments."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _candidate_rules(rows: list[dict[str, object]]) -> list[tuple[tuple[str, str, float], ...]]:
    conditions = []
    selected_rows = [row for row in rows if bool(row.get("prediction"))]
    for feature in NUMERIC_FEATURES:
        values = sorted({_float(row.get(feature)) for row in selected_rows})
        for value in values:
            conditions.append((feature, "<=", value))
            conditions.append((feature, ">=", value))
    return [(), *[(condition,) for condition in conditions], *itertools.combinations(conditions, 2)]


def _fit_veto_rule(
    rows: list[dict[str, object]],
    *,
    candidates: list[tuple[tuple[str, str, float], ...]],
    selection_penalty: float,
) -> tuple[tuple[str, str, float], ...]:
    best_rule = candidates[0]
    best_summary = _score_veto_rule(rows, rule=best_rule, selection_penalty=selection_penalty)
    for rule in candidates[1:]:
        summary = _score_veto_rule(rows, rule=rule, selection_penalty=selection_penalty)
        if _score_key(summary) > _score_key(best_summary):
            best_rule = rule
            best_summary = summary
    return best_rule


def _score_key(summary: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        _float(summary.get("policy_utility")),
        -_float(summary.get("false_negative_count")),
        -_float(summary.get("false_positive_count")),
        -_float(summary.get("selected_count")),
    )


def _score_veto_rule(
    rows: list[dict[str, object]],
    *,
    rule: tuple[tuple[str, str, float], ...],
    selection_penalty: float,
) -> dict[str, object]:
    selected = []
    vetoed = []
    for row in rows:
        if not bool(row.get("prediction")):
            continue
        if _rule_matches(row, rule):
            vetoed.append(row)
        else:
            selected.append(row)
    false_positives = [row for row in selected if not bool(row.get("label"))]
    false_negatives = [
        row
        for row in rows
        if bool(row.get("label"))
        and (not bool(row.get("prediction")) or _rule_matches(row, rule))
    ]
    selected_lift = sum(
        _float(row.get("candidate_lift_vs_trajectory"))
        for row in selected
        if bool(row.get("label"))
    )
    utility = selected_lift - selection_penalty * len(selected)
    return {
        "error_count": len(false_positives) + len(false_negatives),
        "false_negative_count": len(false_negatives),
        "false_negative_task_ids": _task_ids(false_negatives),
        "false_positive_count": len(false_positives),
        "false_positive_task_ids": _task_ids(false_positives),
        "policy_utility": utility,
        "selected_count": len(selected),
        "selected_lift": selected_lift,
        "vetoed_count": len(vetoed),
        "vetoed_task_ids": _task_ids(vetoed),
    }


def _combine_slices(slice_results: list[dict[str, object]]) -> dict[str, object]:
    totals = {
        "error_count": 0,
        "false_negative_count": 0,
        "false_negative_task_ids": [],
        "false_positive_count": 0,
        "false_positive_task_ids": [],
        "policy_utility": 0.0,
        "selected_count": 0,
        "selected_lift": 0.0,
        "vetoed_count": 0,
        "vetoed_task_ids": [],
    }
    for split in slice_results:
        summary = _dict(split.get("test_summary"))
        for key in (
            "error_count",
            "false_negative_count",
            "false_positive_count",
            "selected_count",
            "vetoed_count",
        ):
            totals[key] += int(_float(summary.get(key)))
        for key in ("policy_utility", "selected_lift"):
            totals[key] += _float(summary.get(key))
        for key in ("false_negative_task_ids", "false_positive_task_ids", "vetoed_task_ids"):
            totals[key].extend(str(item) for item in summary.get(key, []) if item)
    return totals


def _rule_matches(row: dict[str, object], rule: tuple[tuple[str, str, float], ...]) -> bool:
    return bool(rule) and all(_condition_matches(row, condition) for condition in rule)


def _condition_matches(row: dict[str, object], condition: tuple[str, str, float]) -> bool:
    feature, operator, threshold = condition
    value = _float(row.get(feature))
    return value <= threshold if operator == "<=" else value >= threshold


def _condition_dict(condition: tuple[str, str, float]) -> dict[str, object]:
    feature, operator, threshold = condition
    return {"feature": feature, "operator": operator, "threshold": threshold}


def _summary_row(policy_id: str, summary: dict[str, object]) -> str:
    return (
        "| "
        f"`{policy_id}` | "
        f"{int(_float(summary.get('selected_count')))} | "
        f"{int(_float(summary.get('false_positive_count')))} | "
        f"{int(_float(summary.get('false_negative_count')))} | "
        f"{_format_float(summary.get('selected_lift'))} | "
        f"{_format_float(summary.get('policy_utility'))} | "
        f"{_join_tasks(summary.get('false_negative_task_ids'))} | "
        f"{_join_tasks(summary.get('false_positive_task_ids'))} | "
        f"{_join_tasks(summary.get('vetoed_task_ids'))} |"
    )


def _format_rule(rule: list[dict[str, object]]) -> str:
    if not rule:
        return "`none`"
    parts = [
        (
            f"`{condition.get('feature', '')} "
            f"{condition.get('operator', '')} "
            f"{_format_float(condition.get('threshold'))}`"
        )
        for condition in rule
    ]
    return " AND ".join(parts)


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
