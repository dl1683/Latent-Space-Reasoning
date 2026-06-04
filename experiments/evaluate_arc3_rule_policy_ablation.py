"""Ablate ARC-3 learned-rule policy mechanisms.

This evaluates the same temporal held-out policy split under a small set of
mechanism toggles. The goal is to separate direct transition learning from
contextual rules and inverse-action symmetry.
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
    _is_boundary_transition,
    _modeled_fields_from_library,
    _rule_library,
    _split_traces,
)
from experiments.extract_arc3_transitions import extract_traces
from experiments.infer_arc3_contextual_rules import infer_contextual_rules
from experiments.infer_arc3_objects import infer_objects
from experiments.infer_arc3_rules import infer_rules


@dataclass(frozen=True)
class AblationRun:
    name: str
    include_contextual: bool
    include_inverse_symmetry: bool
    learned_actions: int
    decidable_transitions: int
    top1_action_matches: int
    boundary_top1_action_matches: int
    non_boundary_top1_action_matches: int
    frequency_baseline_matches: int
    modeled_transition_matches: int
    top1_action_accuracy: float
    boundary_top1_action_accuracy: float
    non_boundary_top1_action_accuracy: float
    frequency_baseline_accuracy: float
    top1_lift_over_frequency: float
    modeled_transition_accuracy: float


@dataclass(frozen=True)
class AblationScore:
    input: str
    train_fraction: float
    train_transitions: int
    test_transitions: int
    candidate_rules: int
    contextual_rules: int
    runs: list[AblationRun]


def _score_choices(
    name: str,
    include_contextual: bool,
    include_inverse_symmetry: bool,
    actions: list[str],
    choices: list[ActionChoice],
    test_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
) -> AblationRun:
    decidable = [choice for choice in choices if choice.decidable]
    boundary_choices = [choice for choice in decidable if _is_boundary_transition(choice)]
    non_boundary_choices = [choice for choice in decidable if not _is_boundary_transition(choice)]
    top1_matches = sum(1 for choice in decidable if choice.selected_action == choice.actual_action)
    boundary_matches = sum(1 for choice in boundary_choices if choice.selected_action == choice.actual_action)
    non_boundary_matches = sum(1 for choice in non_boundary_choices if choice.selected_action == choice.actual_action)
    modeled_matches = sum(1 for choice in decidable if choice.modeled_transition_match)
    train_action_counts = Counter(str(row.get("action", "")) for row in train_rows)
    frequency_action = train_action_counts.most_common(1)[0][0] if train_action_counts else ""
    frequency_matches = sum(1 for row in test_rows if str(row.get("action", "")) == frequency_action)
    top1_accuracy = top1_matches / len(decidable) if decidable else 0.0
    frequency_accuracy = frequency_matches / len(test_rows) if test_rows else 0.0

    return AblationRun(
        name=name,
        include_contextual=include_contextual,
        include_inverse_symmetry=include_inverse_symmetry,
        learned_actions=len(actions),
        decidable_transitions=len(decidable),
        top1_action_matches=top1_matches,
        boundary_top1_action_matches=boundary_matches,
        non_boundary_top1_action_matches=non_boundary_matches,
        frequency_baseline_matches=frequency_matches,
        modeled_transition_matches=modeled_matches,
        top1_action_accuracy=top1_accuracy,
        boundary_top1_action_accuracy=boundary_matches / len(boundary_choices) if boundary_choices else 0.0,
        non_boundary_top1_action_accuracy=non_boundary_matches / len(non_boundary_choices) if non_boundary_choices else 0.0,
        frequency_baseline_accuracy=frequency_accuracy,
        top1_lift_over_frequency=top1_accuracy - frequency_accuracy,
        modeled_transition_accuracy=modeled_matches / len(decidable) if decidable else 0.0,
    )


def evaluate_policy_ablation(
    input_path: Path,
    train_fraction: float = 0.7,
    min_support: int = 2,
) -> AblationScore:
    traces = [asdict(trace) for trace in extract_traces([input_path])]
    train_rows, test_rows = _split_traces(traces, train_fraction)
    object_rows = [asdict(item) for item in infer_objects(train_rows)]
    rule_rows = [asdict(item) for item in infer_rules(object_rows, min_support=min_support)]
    contextual_rule_rows = [
        asdict(item) for item in infer_contextual_rules(train_rows, min_support=min_support)
    ]

    variants = [
        ("base_only", False, False),
        ("base_plus_contextual", True, False),
        ("base_plus_inverse", False, True),
        ("full", True, True),
    ]
    runs: list[AblationRun] = []
    for name, include_contextual, include_inverse_symmetry in variants:
        rule_library = _rule_library(
            rule_rows,
            contextual_rule_rows,
            include_contextual=include_contextual,
            include_inverse_symmetry=include_inverse_symmetry,
        )
        actions = _actions_from_library(rule_library)
        modeled_fields = _modeled_fields_from_library(rule_library)
        choices = [
            _choose_action(
                rule_library,
                actions,
                modeled_fields,
                row,
                _context_action_scores(train_rows, row.get("state_before", {})),
            )
            for row in test_rows
        ]
        runs.append(
            _score_choices(
                name,
                include_contextual,
                include_inverse_symmetry,
                actions,
                choices,
                test_rows,
                train_rows,
            )
        )

    return AblationScore(
        input=str(input_path),
        train_fraction=train_fraction,
        train_transitions=len(train_rows),
        test_transitions=len(test_rows),
        candidate_rules=sum(1 for rule in rule_rows if rule.get("status") == "candidate"),
        contextual_rules=sum(1 for rule in contextual_rule_rows if rule.get("status") == "candidate"),
        runs=runs,
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
    score = evaluate_policy_ablation(args.input, train_fraction=args.train_fraction, min_support=args.min_support)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(score), indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(asdict(score), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
