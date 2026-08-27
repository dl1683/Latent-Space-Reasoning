"""Evaluate validated probe rules across train and transfer slices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.fit_diffusion_validated_probe_stage1_gate import (
    MEASURED_FEATURES,
    _dict,
    _evaluate_rule,
    _float,
    _format_float,
    _rank_rules,
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
    "eval_results/diffusion_language/counterfactual_span_validated_probe_transfer_matrix_v4.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_TRANSFER_MATRIX_V4.md")


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
    parser.add_argument("--report-title", default="Diffusion Counterfactual Span Validated Probe Transfer Matrix V4")
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = evaluate_validated_probe_transfer_matrix(
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
    args.json_output.write_text(json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(matrix, title=args.report_title), encoding="utf-8")
    print(
        json.dumps(
            {
                "best_train_rule": matrix["summary"]["best_train_rule_name"],
                "best_train_transfer_errors": matrix["summary"]["best_train_transfer_error_count"],
                "best_test_rule": matrix["summary"]["best_test_rule_name"],
                "best_test_error_count": matrix["summary"]["best_test_error_count"],
                "transfer_gate_decision": matrix["summary"]["transfer_gate_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def evaluate_validated_probe_transfer_matrix(
    *,
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
    train_rules = _rank_rules(train_rows, MEASURED_FEATURES, require_valid_probe=True)
    test_rules = _rank_rules(test_rows, MEASURED_FEATURES, require_valid_probe=True)
    transfer_rows = [
        _transfer_rule_row(rule, test_rows)
        for rule in train_rules[: max(1, top_n)]
    ]
    best_train_transfer = transfer_rows[0] if transfer_rows else _empty_transfer_row()
    best_test = test_rules[0] if test_rules else _empty_rule()
    return {
        "generated_by": "experiments/evaluate_diffusion_validated_probe_transfer_matrix.py",
        "inputs": {
            "test_scores": str(test_scores_path),
            "test_targets": str(test_targets_path),
            "test_text_fidelity": str(test_text_fidelity_path) if test_text_fidelity_path else "",
            "train_scores": str(train_scores_path),
            "train_targets": str(train_targets_path),
            "train_text_fidelity": str(train_text_fidelity_path) if train_text_fidelity_path else "",
        },
        "schema": "diffusion_counterfactual_validated_probe_transfer_matrix.v1",
        "summary": _summary(
            best_test=best_test,
            best_train_transfer=best_train_transfer,
            test_fit=test_fit,
            test_rows=test_rows,
            train_fit=train_fit,
            train_rows=train_rows,
        ),
        "test_best_rules": test_rules[: max(1, top_n)],
        "train_to_test_rules": transfer_rows,
    }


def render_markdown(
    matrix: dict[str, object],
    *,
    title: str = "Diffusion Counterfactual Validated Probe Transfer Matrix",
) -> str:
    summary = _dict(matrix.get("summary"))
    lines = [
        f"# {title}",
        "",
        "This file is generated by `experiments/evaluate_diffusion_validated_probe_transfer_matrix.py`.",
        (
            "It fits validated measured-probe rules on one slice and evaluates the "
            "same frozen thresholds on an independent transfer slice. Fresh-slice "
            "best rules are listed only as diagnostic upper bounds, not promoted "
            "controller rules."
        ),
        "",
        "## Summary",
        "",
        f"- Train rows: `{summary.get('train_row_count', 0)}`",
        f"- Train valid probe rows: `{summary.get('train_valid_probe_count', 0)}`",
        f"- Test rows: `{summary.get('test_row_count', 0)}`",
        f"- Test valid probe rows: `{summary.get('test_valid_probe_count', 0)}`",
        f"- Best train rule: `{summary.get('best_train_rule_name', '')}`",
        f"- Best train errors: `{summary.get('best_train_error_count', 0)}`",
        f"- Best train-rule transfer errors: `{summary.get('best_train_transfer_error_count', 0)}`",
        f"- Best train-rule transfer FP: `{summary.get('best_train_transfer_false_positive_count', 0)}`",
        f"- Best train-rule transfer FN: `{summary.get('best_train_transfer_false_negative_count', 0)}`",
        f"- Best fresh-only rule: `{summary.get('best_test_rule_name', '')}`",
        f"- Best fresh-only errors: `{summary.get('best_test_error_count', 0)}`",
        f"- Transfer gate decision: `{summary.get('transfer_gate_decision', '')}`",
        "",
        "## Train Rules Applied To Test Slice",
        "",
        "| Train Rule | Train Errors | Test Errors | Test FP | Test FN | Test Missed Lift | Test Selected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _list_of_dicts(matrix.get("train_to_test_rules")):
        test_rule = _dict(row.get("test_rule"))
        lines.append(
            "| "
            f"`{row.get('train_rule_name', '')}` | "
            f"{int(row.get('train_error_count', 0))} | "
            f"{int(test_rule.get('error_count', 0))} | "
            f"{int(test_rule.get('false_positive_count', 0))} | "
            f"{int(test_rule.get('false_negative_count', 0))} | "
            f"{_format_float(test_rule.get('missed_positive_lift'))} | "
            f"{int(test_rule.get('selected_count', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Fresh-Only Diagnostic Upper Bound",
            "",
            "| Fresh Rule | Errors | FP | FN | Missed Lift | Selected |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rule in _list_of_dicts(matrix.get("test_best_rules")):
        lines.append(
            "| "
            f"`{rule.get('rule_name', '')}` | "
            f"{int(rule.get('error_count', 0))} | "
            f"{int(rule.get('false_positive_count', 0))} | "
            f"{int(rule.get('false_negative_count', 0))} | "
            f"{_format_float(rule.get('missed_positive_lift'))} | "
            f"{int(rule.get('selected_count', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "The moonshot discipline here is to preserve the failed transfer "
                "instead of absorbing it into another local fit. A measured "
                "tomography feature becomes a controller feature only when the same "
                "threshold survives this matrix on a non-overlapping slice."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _transfer_rule_row(rule: dict[str, object], test_rows: list[dict[str, object]]) -> dict[str, object]:
    test_rule = _evaluate_rule(
        test_rows,
        str(rule.get("feature", "")),
        str(rule.get("direction", "ge")),
        _float(rule.get("threshold")),
        require_valid_probe=bool(rule.get("requires_valid_probe")),
    )
    return {
        "test_rule": test_rule,
        "train_error_count": int(rule.get("error_count", 0)),
        "train_false_negative_count": int(rule.get("false_negative_count", 0)),
        "train_false_positive_count": int(rule.get("false_positive_count", 0)),
        "train_missed_positive_lift": _float(rule.get("missed_positive_lift")),
        "train_rule_name": str(rule.get("rule_name", "")),
        "train_selected_count": int(rule.get("selected_count", 0)),
    }


def _summary(
    *,
    best_test: dict[str, object],
    best_train_transfer: dict[str, object],
    test_fit: dict[str, object],
    test_rows: list[dict[str, object]],
    train_fit: dict[str, object],
    train_rows: list[dict[str, object]],
) -> dict[str, object]:
    best_transfer_test_rule = _dict(best_train_transfer.get("test_rule"))
    transfer_errors = int(best_transfer_test_rule.get("error_count", 0))
    return {
        "best_test_error_count": int(best_test.get("error_count", 0)),
        "best_test_rule_name": str(best_test.get("rule_name", "")),
        "best_train_error_count": int(best_train_transfer.get("train_error_count", 0)),
        "best_train_rule_name": str(best_train_transfer.get("train_rule_name", "")),
        "best_train_transfer_error_count": transfer_errors,
        "best_train_transfer_false_negative_count": int(best_transfer_test_rule.get("false_negative_count", 0)),
        "best_train_transfer_false_positive_count": int(best_transfer_test_rule.get("false_positive_count", 0)),
        "counterfactual_probe_policy_test": str(_dict(test_fit.get("summary")).get("counterfactual_probe_policy", "")),
        "counterfactual_probe_policy_train": str(_dict(train_fit.get("summary")).get("counterfactual_probe_policy", "")),
        "test_row_count": len(test_rows),
        "test_valid_probe_count": sum(1 for row in test_rows if bool(row.get("valid_for_stage1"))),
        "train_row_count": len(train_rows),
        "train_valid_probe_count": sum(1 for row in train_rows if bool(row.get("valid_for_stage1"))),
        "transfer_gate_decision": "diagnostic_only_transfer_failed"
        if transfer_errors
        else "diagnostic_only_transfer_passed",
    }


def _empty_rule() -> dict[str, object]:
    return {
        "direction": "ge",
        "error_count": 0,
        "false_negative_count": 0,
        "false_positive_count": 0,
        "feature": "",
        "missed_positive_lift": 0.0,
        "requires_valid_probe": True,
        "rule_name": "",
        "selected_count": 0,
        "threshold": 0.0,
    }


def _empty_transfer_row() -> dict[str, object]:
    return {
        "test_rule": _empty_rule(),
        "train_error_count": 0,
        "train_false_negative_count": 0,
        "train_false_positive_count": 0,
        "train_missed_positive_lift": 0.0,
        "train_rule_name": "",
        "train_selected_count": 0,
    }


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


if __name__ == "__main__":
    raise SystemExit(main())
