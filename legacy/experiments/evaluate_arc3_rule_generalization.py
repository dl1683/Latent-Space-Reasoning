"""Evaluate ARC-3 rule generalization on held-out transition traces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.check_arc3_contextual_rules import check_contextual_rules
from experiments.check_arc3_rules import check_rules
from experiments.extract_arc3_transitions import extract_traces
from experiments.infer_arc3_contextual_rules import infer_contextual_rules
from experiments.infer_arc3_objects import infer_objects
from experiments.infer_arc3_rules import infer_rules


@dataclass(frozen=True)
class RuleGeneralizationScore:
    input: str
    train_transitions: int
    test_transitions: int
    train_fraction: float
    candidate_rules: int
    contextual_rules: int
    rule_checks: int
    supported: int
    contradicted: int
    not_applicable: int
    contextual_rule_checks: int
    contextual_supported: int
    contextual_contradicted: int
    applicable_precision: float
    contextual_applicable_precision: float
    transition_coverage: float
    status: str


def _split_traces(rows: list[dict[str, Any]], train_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split_index = max(1, min(len(rows) - 1, int(len(rows) * train_fraction)))
    return rows[:split_index], rows[split_index:]


def _count_status(rows: list[Any], status: str) -> int:
    return sum(1 for row in rows if getattr(row, "status", "") == status)


def _covered_transitions(rows: list[Any]) -> set[tuple[str, int]]:
    return {
        (str(getattr(row, "level_id", "")), int(getattr(row, "step_index", 0)))
        for row in rows
        if getattr(row, "status", "") in {"supported", "contradicted"}
    }


def evaluate_rule_generalization(
    input_path: Path,
    train_fraction: float = 0.7,
    min_support: int = 2,
) -> RuleGeneralizationScore:
    traces = [asdict(trace) for trace in extract_traces([input_path])]
    train_rows, test_rows = _split_traces(traces, train_fraction)
    object_rows = [asdict(item) for item in infer_objects(train_rows)]
    rule_rows = [asdict(item) for item in infer_rules(object_rows, min_support=min_support)]
    contextual_rule_rows = [
        asdict(item) for item in infer_contextual_rules(train_rows, min_support=min_support)
    ]
    checks = check_rules(test_rows, rule_rows)
    contextual_checks = check_contextual_rules(test_rows, contextual_rule_rows)

    supported = _count_status(checks, "supported")
    contradicted = _count_status(checks, "contradicted")
    not_applicable = _count_status(checks, "not_applicable")
    contextual_supported = _count_status(contextual_checks, "supported")
    contextual_contradicted = _count_status(contextual_checks, "contradicted")
    applicable = supported + contradicted
    contextual_applicable = contextual_supported + contextual_contradicted
    covered = _covered_transitions(checks) | _covered_transitions(contextual_checks)
    transition_coverage = len(covered) / len(test_rows) if test_rows else 0.0
    applicable_precision = supported / applicable if applicable else 0.0
    contextual_applicable_precision = (
        contextual_supported / contextual_applicable if contextual_applicable else 0.0
    )

    if contradicted or contextual_contradicted:
        status = "contradicted"
    elif applicable or contextual_applicable:
        status = "predictive"
    else:
        status = "no_heldout_predictions"

    return RuleGeneralizationScore(
        input=str(input_path),
        train_transitions=len(train_rows),
        test_transitions=len(test_rows),
        train_fraction=train_fraction,
        candidate_rules=sum(1 for rule in rule_rows if rule.get("status") == "candidate"),
        contextual_rules=sum(1 for rule in contextual_rule_rows if rule.get("status") == "candidate"),
        rule_checks=len(checks),
        supported=supported,
        contradicted=contradicted,
        not_applicable=not_applicable,
        contextual_rule_checks=len(contextual_checks),
        contextual_supported=contextual_supported,
        contextual_contradicted=contextual_contradicted,
        applicable_precision=applicable_precision,
        contextual_applicable_precision=contextual_applicable_precision,
        transition_coverage=transition_coverage,
        status=status,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score = evaluate_rule_generalization(
        args.input,
        train_fraction=args.train_fraction,
        min_support=args.min_support,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(score), indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(asdict(score), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
