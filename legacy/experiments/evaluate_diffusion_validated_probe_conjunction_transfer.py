"""Search small validated probe conjunctions across train and transfer slices."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.fit_diffusion_validated_probe_stage1_gate import (
    MEASURED_FEATURES,
    _dict,
    _float,
    _format_float,
    fit_validated_probe_stage1_gate,
)

DEFAULT_TRAIN_TARGETS = Path("eval_results/diffusion_language/diffusion_counterfactual_probe_targets_v1.json")
DEFAULT_TRAIN_SCORES = Path("eval_results/diffusion_language/counterfactual_micro_probe_span_tomography_v4_scores.json")
DEFAULT_TRAIN_TEXT_FIDELITY = Path("eval_results/diffusion_language/counterfactual_span_probe_text_fidelity_v4.json")
DEFAULT_TEST_TARGETS = Path(
    "eval_results/diffusion_language/diffusion_counterfactual_probe_transfer_targets_v1.json"
)
DEFAULT_TEST_SCORES = Path(
    "eval_results/diffusion_language/counterfactual_micro_probe_span_tomography_v4_fresh_planning_scores.json"
)
DEFAULT_TEST_TEXT_FIDELITY = Path(
    "eval_results/diffusion_language/counterfactual_span_probe_text_fidelity_v4_fresh_planning.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/counterfactual_span_validated_probe_conjunction_transfer_v4.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_CONJUNCTION_TRANSFER_V4.md")
FEATURES = (
    *MEASURED_FEATURES,
    "prompt_gap_count",
    "source_quality",
    "counterfactual_probe_text_x0_x2_slot_overlap",
    "counterfactual_probe_text_max_slot_overlap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-targets", type=Path, default=DEFAULT_TRAIN_TARGETS)
    parser.add_argument("--train-scores", type=Path, default=DEFAULT_TRAIN_SCORES)
    parser.add_argument("--train-text-fidelity", type=Path, default=DEFAULT_TRAIN_TEXT_FIDELITY)
    parser.add_argument("--test-targets", type=Path, default=DEFAULT_TEST_TARGETS)
    parser.add_argument("--test-scores", type=Path, default=DEFAULT_TEST_SCORES)
    parser.add_argument("--test-text-fidelity", type=Path, default=DEFAULT_TEST_TEXT_FIDELITY)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--report-title", default="Diffusion Counterfactual Span Validated Probe Conjunction Transfer V4")
    parser.add_argument("--max-conditions", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fit = evaluate_validated_probe_conjunction_transfer(
        max_conditions=args.max_conditions,
        test_scores_path=args.test_scores,
        test_targets_path=args.test_targets,
        test_text_fidelity_path=args.test_text_fidelity,
        top_n=args.top_n,
        train_scores_path=args.train_scores,
        train_targets_path=args.train_targets,
        train_text_fidelity_path=args.train_text_fidelity,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(fit, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(fit, title=args.report_title), encoding="utf-8")
    print(
        json.dumps(
            {
                "best_train_rule": fit["summary"]["best_train_rule_id"],
                "best_train_transfer_errors": fit["summary"]["best_train_transfer_error_count"],
                "best_transfer_screened_rule": fit["summary"]["best_transfer_screened_rule_id"],
                "best_transfer_screened_test_errors": fit["summary"]["best_transfer_screened_test_error_count"],
                "gate_decision": fit["summary"]["gate_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def evaluate_validated_probe_conjunction_transfer(
    *,
    max_conditions: int,
    test_scores_path: Path,
    test_targets_path: Path,
    test_text_fidelity_path: Path | None,
    top_n: int,
    train_scores_path: Path,
    train_targets_path: Path,
    train_text_fidelity_path: Path | None,
) -> dict[str, object]:
    train_fit = fit_validated_probe_stage1_gate(
        scores_path=train_scores_path,
        targets_path=train_targets_path,
        text_fidelity_path=train_text_fidelity_path,
    )
    test_fit = fit_validated_probe_stage1_gate(
        scores_path=test_scores_path,
        targets_path=test_targets_path,
        text_fidelity_path=test_text_fidelity_path,
    )
    train_rows = _list_of_dicts(train_fit.get("rows"))
    test_rows = _list_of_dicts(test_fit.get("rows"))
    rules = _rank_conjunction_rules(
        max_conditions=max_conditions,
        test_rows=test_rows,
        train_rows=train_rows,
    )
    train_ranked = sorted(rules, key=_train_rank_key)
    transfer_screened = sorted(rules, key=_transfer_screened_key)
    best_train = train_ranked[0] if train_ranked else _empty_rule()
    best_screened = transfer_screened[0] if transfer_screened else _empty_rule()
    return {
        "generated_by": "experiments/evaluate_diffusion_validated_probe_conjunction_transfer.py",
        "inputs": {
            "max_conditions": max_conditions,
            "test_scores": str(test_scores_path),
            "test_targets": str(test_targets_path),
            "test_text_fidelity": str(test_text_fidelity_path) if test_text_fidelity_path else "",
            "train_scores": str(train_scores_path),
            "train_targets": str(train_targets_path),
            "train_text_fidelity": str(train_text_fidelity_path) if train_text_fidelity_path else "",
        },
        "schema": "diffusion_counterfactual_validated_probe_conjunction_transfer.v1",
        "summary": _summary(best_screened, best_train, rules, test_rows, train_rows),
        "top_train_ranked_rules": train_ranked[: max(1, top_n)],
        "top_transfer_screened_rules": transfer_screened[: max(1, top_n)],
    }


def render_markdown(
    fit: dict[str, object],
    *,
    title: str = "Diffusion Counterfactual Span Validated Probe Conjunction Transfer",
) -> str:
    summary = _dict(fit.get("summary"))
    lines = [
        f"# {title}",
        "",
        "This file is generated by `experiments/evaluate_diffusion_validated_probe_conjunction_transfer.py`.",
        (
            "It exhaustively searches small AND-rules over validated measured-probe "
            "features. Train-ranked rules are the deployable discipline; "
            "transfer-screened rules are diagnostic challengers only because they "
            "use fresh labels for selection."
        ),
        "",
        "## Summary",
        "",
        f"- Train rows: `{summary.get('train_row_count', 0)}`",
        f"- Test rows: `{summary.get('test_row_count', 0)}`",
        f"- Candidate rules searched: `{summary.get('candidate_rule_count', 0)}`",
        f"- Max conditions: `{summary.get('max_conditions', 0)}`",
        f"- Best train-ranked rule: `{summary.get('best_train_rule_id', '')}`",
        f"- Best train-ranked train errors: `{summary.get('best_train_error_count', 0)}`",
        f"- Best train-ranked transfer errors: `{summary.get('best_train_transfer_error_count', 0)}`",
        f"- Best transfer-screened rule: `{summary.get('best_transfer_screened_rule_id', '')}`",
        f"- Best transfer-screened train errors: `{summary.get('best_transfer_screened_train_error_count', 0)}`",
        f"- Best transfer-screened test errors: `{summary.get('best_transfer_screened_test_error_count', 0)}`",
        f"- Gate decision: `{summary.get('gate_decision', '')}`",
        "",
        "## Train-Ranked Rules",
        "",
        "| Rule | Conditions | Train Errors | Test Errors | Test FP | Test FN | Test Selected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rule in _list_of_dicts(fit.get("top_train_ranked_rules")):
        lines.append(_rule_row(rule))
    lines.extend(
        [
            "",
            "## Transfer-Screened Diagnostic Challengers",
            "",
            "| Rule | Conditions | Train Errors | Test Errors | Test FP | Test FN | Test Selected |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rule in _list_of_dicts(fit.get("top_transfer_screened_rules")):
        lines.append(_rule_row(rule))
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "The first train-ranked rule still overfits the named surface. The "
                "best transfer-screened conjunction is useful only as a next-slice "
                "hypothesis: it must be frozen and tested on another independent "
                "planning slice before it can become a controller rule."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _rank_conjunction_rules(
    *,
    max_conditions: int,
    test_rows: list[dict[str, object]],
    train_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    conditions = _candidate_conditions(train_rows)
    rules = []
    for condition_count in range(1, max_conditions + 1):
        for combo in itertools.combinations(conditions, condition_count):
            condition_list = list(combo)
            train_score = _score_rule(train_rows, condition_list)
            test_score = _score_rule(test_rows, condition_list)
            rules.append(
                {
                    "condition_count": condition_count,
                    "conditions": [
                        {
                            "direction": direction,
                            "feature": feature,
                            "threshold": threshold,
                        }
                        for feature, direction, threshold in condition_list
                    ],
                    "rule_id": _rule_id(condition_list),
                    "test": test_score,
                    "train": train_score,
                }
            )
    return rules


def _candidate_conditions(rows: list[dict[str, object]]) -> list[tuple[str, str, float]]:
    conditions = []
    for feature in FEATURES:
        values = sorted({_feature_value(row, feature) for row in rows})
        for value in values:
            conditions.append((feature, "ge", value))
            conditions.append((feature, "le", value))
    return conditions


def _score_rule(
    rows: list[dict[str, object]],
    conditions: list[tuple[str, str, float]],
) -> dict[str, object]:
    false_positives = []
    false_negatives = []
    selected = []
    for row in rows:
        prediction = bool(row.get("valid_for_stage1")) and all(
            _condition_selects(row, condition) for condition in conditions
        )
        label = bool(row.get("label"))
        if prediction:
            selected.append(row)
        if prediction and not label:
            false_positives.append(row)
        if not prediction and label:
            false_negatives.append(row)
    return {
        "error_count": len(false_positives) + len(false_negatives),
        "false_negative_count": len(false_negatives),
        "false_negative_task_ids": _task_ids(false_negatives),
        "false_positive_count": len(false_positives),
        "false_positive_task_ids": _task_ids(false_positives),
        "missed_positive_lift": sum(_float(row.get("candidate_lift_vs_trajectory")) for row in false_negatives),
        "selected_count": len(selected),
        "selected_task_ids": _task_ids(selected),
    }


def _condition_selects(row: dict[str, object], condition: tuple[str, str, float]) -> bool:
    feature, direction, threshold = condition
    value = _feature_value(row, feature)
    return value >= threshold if direction == "ge" else value <= threshold


def _feature_value(row: dict[str, object], feature: str) -> float:
    return _float(_dict(row.get("features")).get(feature))


def _train_rank_key(rule: dict[str, object]) -> tuple[object, ...]:
    train = _dict(rule.get("train"))
    test = _dict(rule.get("test"))
    return (
        int(train.get("error_count", 0)),
        _float(train.get("missed_positive_lift")),
        int(train.get("false_positive_count", 0)),
        int(rule.get("condition_count", 0)),
        int(test.get("error_count", 0)),
        str(rule.get("rule_id", "")),
    )


def _transfer_screened_key(rule: dict[str, object]) -> tuple[object, ...]:
    train = _dict(rule.get("train"))
    test = _dict(rule.get("test"))
    return (
        int(test.get("error_count", 0)),
        int(train.get("error_count", 0)),
        _float(test.get("missed_positive_lift")),
        int(test.get("false_positive_count", 0)),
        int(rule.get("condition_count", 0)),
        str(rule.get("rule_id", "")),
    )


def _summary(
    best_screened: dict[str, object],
    best_train: dict[str, object],
    rules: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    train_rows: list[dict[str, object]],
) -> dict[str, object]:
    best_train_train = _dict(best_train.get("train"))
    best_train_test = _dict(best_train.get("test"))
    best_screened_train = _dict(best_screened.get("train"))
    best_screened_test = _dict(best_screened.get("test"))
    return {
        "best_train_error_count": int(best_train_train.get("error_count", 0)),
        "best_train_rule_id": str(best_train.get("rule_id", "")),
        "best_train_transfer_error_count": int(best_train_test.get("error_count", 0)),
        "best_transfer_screened_rule_id": str(best_screened.get("rule_id", "")),
        "best_transfer_screened_test_error_count": int(best_screened_test.get("error_count", 0)),
        "best_transfer_screened_train_error_count": int(best_screened_train.get("error_count", 0)),
        "candidate_rule_count": len(rules),
        "gate_decision": "diagnostic_only_transfer_screened_challenger",
        "max_conditions": max((int(rule.get("condition_count", 0)) for rule in rules), default=0),
        "test_row_count": len(test_rows),
        "train_row_count": len(train_rows),
    }


def _rule_id(conditions: list[tuple[str, str, float]]) -> str:
    return "valid_if_" + "_and_".join(
        f"{feature}_{direction}_{_format_token(threshold)}"
        for feature, direction, threshold in conditions
    )


def _rule_row(rule: dict[str, object]) -> str:
    train = _dict(rule.get("train"))
    test = _dict(rule.get("test"))
    conditions = _list_of_dicts(rule.get("conditions"))
    return (
        "| "
        f"`{rule.get('rule_id', '')}` | "
        f"{_condition_summary(conditions)} | "
        f"{int(train.get('error_count', 0))} | "
        f"{int(test.get('error_count', 0))} | "
        f"{int(test.get('false_positive_count', 0))} | "
        f"{int(test.get('false_negative_count', 0))} | "
        f"{int(test.get('selected_count', 0))} |"
    )


def _condition_summary(conditions: list[dict[str, object]]) -> str:
    return "<br>".join(
        f"`{row.get('feature', '')} {row.get('direction', '')} {_format_float(row.get('threshold'))}`"
        for row in conditions
    )


def _format_token(value: float) -> str:
    return _format_float(value).replace("-", "neg_").replace(".", "p")


def _task_ids(rows: list[dict[str, object]]) -> list[str]:
    return [str(row.get("task_id", "")) for row in rows]


def _empty_rule() -> dict[str, object]:
    return {
        "condition_count": 0,
        "conditions": [],
        "rule_id": "",
        "test": _empty_score(),
        "train": _empty_score(),
    }


def _empty_score() -> dict[str, object]:
    return {
        "error_count": 0,
        "false_negative_count": 0,
        "false_negative_task_ids": [],
        "false_positive_count": 0,
        "false_positive_task_ids": [],
        "missed_positive_lift": 0.0,
        "selected_count": 0,
        "selected_task_ids": [],
    }


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


if __name__ == "__main__":
    raise SystemExit(main())
