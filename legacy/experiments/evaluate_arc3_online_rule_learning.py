"""Evaluate ARC-3 rule policy improvement as transitions accumulate.

This is a prequential evaluator: before each transition, it learns rules from
all earlier transitions and asks whether those rules choose the next action.
After scoring, the transition is added to the learner's evidence.
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
)
from experiments.extract_arc3_transitions import extract_traces
from experiments.infer_arc3_contextual_rules import infer_contextual_rules
from experiments.infer_arc3_objects import infer_objects
from experiments.infer_arc3_rules import infer_rules


@dataclass(frozen=True)
class OnlineRuleStep:
    step_number: int
    train_transitions: int
    candidate_rules: int
    contextual_rules: int
    learned_actions: int
    choice: ActionChoice


@dataclass(frozen=True)
class OnlineRuleLearningScore:
    input: str
    transitions: int
    warmup: int
    evaluated_transitions: int
    decidable_transitions: int
    no_rule_applicable: int
    top1_action_matches: int
    oracle_action_matches: int
    frequency_baseline_matches: int
    modeled_transition_matches: int
    boundary_transitions: int
    non_boundary_transitions: int
    boundary_top1_action_matches: int
    non_boundary_top1_action_matches: int
    top1_action_accuracy: float
    boundary_top1_action_accuracy: float
    non_boundary_top1_action_accuracy: float
    oracle_action_accuracy: float
    frequency_baseline_accuracy: float
    top1_lift_over_frequency: float
    modeled_transition_accuracy: float
    first_half_top1_accuracy: float
    second_half_top1_accuracy: float
    improvement_over_time: float
    steps: list[OnlineRuleStep]


def _learn_library(rows: list[dict[str, Any]], min_support: int) -> tuple[dict[str, Any], int, int]:
    object_rows = [asdict(item) for item in infer_objects(rows)]
    rule_rows = [asdict(item) for item in infer_rules(object_rows, min_support=min_support)]
    contextual_rule_rows = [
        asdict(item) for item in infer_contextual_rules(rows, min_support=min_support)
    ]
    return (
        _rule_library(rule_rows, contextual_rule_rows),
        sum(1 for rule in rule_rows if rule.get("status") == "candidate"),
        sum(1 for rule in contextual_rule_rows if rule.get("status") == "candidate"),
    )


def _accuracy(choices: list[ActionChoice]) -> float:
    decidable = [choice for choice in choices if choice.decidable]
    if not decidable:
        return 0.0
    return sum(1 for choice in decidable if choice.selected_action == choice.actual_action) / len(decidable)


def evaluate_online_rule_learning(
    input_path: Path,
    warmup: int = 4,
    min_support: int = 2,
) -> OnlineRuleLearningScore:
    rows = [asdict(trace) for trace in extract_traces([input_path])]
    steps: list[OnlineRuleStep] = []

    for index in range(max(1, warmup), len(rows)):
        train_rows = rows[:index]
        rule_library, candidate_rules, contextual_rules = _learn_library(train_rows, min_support)
        actions = _actions_from_library(rule_library)
        modeled_fields = _modeled_fields_from_library(rule_library)
        choice = _choose_action(
            rule_library,
            actions,
            modeled_fields,
            rows[index],
            _context_action_scores(train_rows, rows[index].get("state_before", {})),
        )
        steps.append(
            OnlineRuleStep(
                step_number=index,
                train_transitions=len(train_rows),
                candidate_rules=candidate_rules,
                contextual_rules=contextual_rules,
                learned_actions=len(actions),
                choice=choice,
            )
        )

    choices = [step.choice for step in steps]
    decidable = [choice for choice in choices if choice.decidable]
    top1_matches = sum(1 for choice in decidable if choice.selected_action == choice.actual_action)
    boundary_choices = [choice for choice in decidable if _is_boundary_transition(choice)]
    non_boundary_choices = [choice for choice in decidable if not _is_boundary_transition(choice)]
    boundary_top1_matches = sum(1 for choice in boundary_choices if choice.selected_action == choice.actual_action)
    non_boundary_top1_matches = sum(1 for choice in non_boundary_choices if choice.selected_action == choice.actual_action)
    oracle_matches = sum(1 for choice in decidable if choice.actual_action in choice.best_actions)
    modeled_matches = sum(1 for choice in decidable if choice.modeled_transition_match)
    train_action_counts: Counter[str] = Counter()
    frequency_matches = 0
    for index in range(max(1, warmup), len(rows)):
        train_action_counts = Counter(str(row.get("action", "")) for row in rows[:index])
        frequency_action = train_action_counts.most_common(1)[0][0] if train_action_counts else ""
        if str(rows[index].get("action", "")) == frequency_action:
            frequency_matches += 1

    midpoint = len(choices) // 2
    first_half = choices[:midpoint]
    second_half = choices[midpoint:]
    first_half_accuracy = _accuracy(first_half)
    second_half_accuracy = _accuracy(second_half)
    top1_accuracy = top1_matches / len(decidable) if decidable else 0.0
    frequency_accuracy = frequency_matches / len(choices) if choices else 0.0

    return OnlineRuleLearningScore(
        input=str(input_path),
        transitions=len(rows),
        warmup=warmup,
        evaluated_transitions=len(choices),
        decidable_transitions=len(decidable),
        no_rule_applicable=len(choices) - len(decidable),
        top1_action_matches=top1_matches,
        oracle_action_matches=oracle_matches,
        frequency_baseline_matches=frequency_matches,
        modeled_transition_matches=modeled_matches,
        boundary_transitions=len(boundary_choices),
        non_boundary_transitions=len(non_boundary_choices),
        boundary_top1_action_matches=boundary_top1_matches,
        non_boundary_top1_action_matches=non_boundary_top1_matches,
        top1_action_accuracy=top1_accuracy,
        boundary_top1_action_accuracy=boundary_top1_matches / len(boundary_choices) if boundary_choices else 0.0,
        non_boundary_top1_action_accuracy=non_boundary_top1_matches / len(non_boundary_choices) if non_boundary_choices else 0.0,
        oracle_action_accuracy=oracle_matches / len(decidable) if decidable else 0.0,
        frequency_baseline_accuracy=frequency_accuracy,
        top1_lift_over_frequency=top1_accuracy - frequency_accuracy,
        modeled_transition_accuracy=modeled_matches / len(decidable) if decidable else 0.0,
        first_half_top1_accuracy=first_half_accuracy,
        second_half_top1_accuracy=second_half_accuracy,
        improvement_over_time=second_half_accuracy - first_half_accuracy,
        steps=steps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score = evaluate_online_rule_learning(args.input, warmup=args.warmup, min_support=args.min_support)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(score), indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(asdict(score), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
