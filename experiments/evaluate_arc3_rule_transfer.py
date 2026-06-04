"""Evaluate learned ARC-3 rule policy transfer across traces.

This trains a compact rule library on one replay trace and tests action choice
on a different replay trace. It is a stricter signal than an in-trace temporal
split because the held-out transitions come from a separate level/run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate_arc3_rule_policy import (
    ActionChoice,
    _actions_from_library,
    _choose_action,
    _context_action_scores,
    _modeled_fields_from_library,
    _rule_library,
)
from experiments.extract_arc3_transitions import extract_traces
from experiments.infer_arc3_contextual_rules import infer_contextual_rules
from experiments.infer_arc3_objects import infer_objects
from experiments.infer_arc3_rules import infer_rules


@dataclass(frozen=True)
class RuleTransferScore:
    train_input: str
    test_input: str
    train_transitions: int
    test_transitions: int
    candidate_rules: int
    contextual_rules: int
    learned_actions: int
    decidable_transitions: int
    no_rule_applicable: int
    top1_action_matches: int
    oracle_action_matches: int
    frequency_baseline_matches: int
    modeled_transition_matches: int
    top1_action_accuracy: float
    oracle_action_accuracy: float
    frequency_baseline_accuracy: float
    top1_lift_over_frequency: float
    modeled_transition_accuracy: float
    choices: list[ActionChoice]


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    return [asdict(trace) for trace in extract_traces([path])]


def evaluate_rule_transfer(
    train_input: Path,
    test_input: Path,
    min_support: int = 2,
) -> RuleTransferScore:
    train_rows = _trace_rows(train_input)
    test_rows = _trace_rows(test_input)
    object_rows = [asdict(item) for item in infer_objects(train_rows)]
    rule_rows = [asdict(item) for item in infer_rules(object_rows, min_support=min_support)]
    contextual_rule_rows = [
        asdict(item) for item in infer_contextual_rules(train_rows, min_support=min_support)
    ]
    rule_library = _rule_library(rule_rows, contextual_rule_rows)
    actions = _actions_from_library(rule_library)
    modeled_fields = _modeled_fields_from_library(rule_library)
    choices = [
        _choose_action(rule_library, actions, modeled_fields, row, _context_action_scores(train_rows, row.get("state_before", {})))
        for row in test_rows
    ]

    decidable = [choice for choice in choices if choice.decidable]
    top1_matches = sum(1 for choice in decidable if choice.selected_action == choice.actual_action)
    oracle_matches = sum(1 for choice in decidable if choice.actual_action in choice.best_actions)
    train_action_counts = Counter(str(row.get("action", "")) for row in train_rows)
    frequency_action = train_action_counts.most_common(1)[0][0] if train_action_counts else ""
    frequency_matches = sum(1 for row in test_rows if str(row.get("action", "")) == frequency_action)
    modeled_matches = sum(1 for choice in decidable if choice.modeled_transition_match)
    top1_accuracy = top1_matches / len(decidable) if decidable else 0.0
    frequency_accuracy = frequency_matches / len(test_rows) if test_rows else 0.0

    return RuleTransferScore(
        train_input=str(train_input),
        test_input=str(test_input),
        train_transitions=len(train_rows),
        test_transitions=len(test_rows),
        candidate_rules=sum(1 for rule in rule_rows if rule.get("status") == "candidate"),
        contextual_rules=sum(1 for rule in contextual_rule_rows if rule.get("status") == "candidate"),
        learned_actions=len(actions),
        decidable_transitions=len(decidable),
        no_rule_applicable=len(choices) - len(decidable),
        top1_action_matches=top1_matches,
        oracle_action_matches=oracle_matches,
        frequency_baseline_matches=frequency_matches,
        modeled_transition_matches=modeled_matches,
        top1_action_accuracy=top1_accuracy,
        oracle_action_accuracy=oracle_matches / len(decidable) if decidable else 0.0,
        frequency_baseline_accuracy=frequency_accuracy,
        top1_lift_over_frequency=top1_accuracy - frequency_accuracy,
        modeled_transition_accuracy=modeled_matches / len(decidable) if decidable else 0.0,
        choices=choices,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_input", type=Path)
    parser.add_argument("test_input", type=Path)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score = evaluate_rule_transfer(args.train_input, args.test_input, min_support=args.min_support)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(score), indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(asdict(score), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
