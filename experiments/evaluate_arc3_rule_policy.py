"""Evaluate whether learned ARC-3 rules can choose held-out actions.

This is stricter than rule generalization: instead of asking whether a learned
rule predicts an observed transition, it asks whether the learned rule library
can rank actions from the held-out state and put the actual action on top.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import Counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.apply_arc3_validated_rules import predict_state
from experiments.extract_arc3_transitions import extract_traces
from experiments.infer_arc3_contextual_rules import infer_contextual_rules
from experiments.infer_arc3_objects import infer_objects
from experiments.infer_arc3_rules import infer_rules

INVERSE_ACTIONS = {
    "ACTION1": "ACTION2",
    "ACTION2": "ACTION1",
    "ACTION3": "ACTION4",
    "ACTION4": "ACTION3",
}

INVERSE_FIELDS = {
    "ACTION1": "y",
    "ACTION2": "y",
    "ACTION3": "x",
    "ACTION4": "x",
}


@dataclass(frozen=True)
class ActionChoice:
    level_id: str
    step_index: int
    actual_action: str
    selected_action: str | None
    best_actions: list[str]
    decidable: bool
    exact_transition_match: bool
    modeled_transition_match: bool
    changed_field_matches: int
    modeled_field_matches: int
    changed_fields: int
    modeled_fields: int
    changed_field_names: list[str]
    modeled_field_names: list[str]
    changed_missed_fields: list[str]
    modeled_missed_fields: list[str]
    side_effects: int


@dataclass(frozen=True)
class RulePolicyScore:
    input: str
    train_transitions: int
    test_transitions: int
    train_fraction: float
    candidate_rules: int
    contextual_rules: int
    learned_actions: int
    decidable_transitions: int
    no_rule_applicable: int
    top1_action_matches: int
    oracle_action_matches: int
    frequency_baseline_matches: int
    exact_transition_matches: int
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
    exact_transition_accuracy: float
    modeled_transition_accuracy: float
    choices: list[ActionChoice]


def _split_traces(rows: list[dict[str, Any]], train_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split_index = max(1, min(len(rows) - 1, int(len(rows) * train_fraction)))
    return rows[:split_index], rows[split_index:]


def _rule_library(
    rule_rows: list[dict[str, Any]],
    contextual_rule_rows: list[dict[str, Any]],
    include_contextual: bool = True,
    include_inverse_symmetry: bool = True,
) -> dict[str, Any]:
    validated_rules = [
        {"rule_id": str(rule.get("rule_id", "")), "rule": rule}
        for rule in rule_rows
        if rule.get("status") == "candidate"
    ]
    existing = {
        (str(item["rule"].get("action", "")), str(item["rule"].get("field", "")))
        for item in validated_rules
        if isinstance(item.get("rule"), dict)
    }
    derived_rules: list[dict[str, Any]] = []
    for item in validated_rules if include_inverse_symmetry else []:
        rule = item.get("rule") if isinstance(item, dict) else None
        if not isinstance(rule, dict):
            continue
        action = str(rule.get("action", ""))
        inverse_action = INVERSE_ACTIONS.get(action)
        field = str(rule.get("field", ""))
        effect = rule.get("effect") if isinstance(rule.get("effect"), dict) else {}
        delta = effect.get("delta")
        if (
            inverse_action is None
            or field != INVERSE_FIELDS.get(action)
            or not isinstance(delta, (int, float))
            or (inverse_action, field) in existing
        ):
            continue
        inverse_rule = dict(rule)
        inverse_rule["rule_id"] = f"derived_inverse:{rule.get('rule_id', item.get('rule_id', ''))}"
        inverse_rule["action"] = inverse_action
        inverse_rule["effect"] = {"delta": -delta}
        inverse_rule["support"] = 0
        inverse_rule["counterexamples"] = 0
        inverse_rule["derivation"] = {
            "type": "inverse_action_symmetry",
            "source_action": action,
            "source_rule_id": str(item.get("rule_id", rule.get("rule_id", ""))),
        }
        derived_rules.append({"rule_id": str(inverse_rule["rule_id"]), "rule": inverse_rule})
        existing.add((inverse_action, field))
    contextual_rules = [
        rule for rule in contextual_rule_rows if include_contextual and rule.get("status") == "candidate"
    ]
    contextual_existing = {
        (str(rule.get("action", "")), str(rule.get("field", "")))
        for rule in contextual_rules
        if isinstance(rule, dict)
    }
    derived_contextual_rules: list[dict[str, Any]] = []
    for rule in contextual_rules if include_inverse_symmetry else []:
        if not isinstance(rule, dict):
            continue
        action = str(rule.get("action", ""))
        inverse_action = INVERSE_ACTIONS.get(action)
        field = str(rule.get("field", ""))
        effect = rule.get("effect") if isinstance(rule.get("effect"), dict) else {}
        delta = effect.get("delta")
        if (
            inverse_action is None
            or field != INVERSE_FIELDS.get(action)
            or not isinstance(delta, (int, float))
            or (inverse_action, field) in existing
            or (inverse_action, field) in contextual_existing
        ):
            continue
        inverse_rule = dict(rule)
        inverse_rule["rule_id"] = f"derived_inverse:{rule.get('rule_id', '')}"
        inverse_rule["action"] = inverse_action
        inverse_rule["effect"] = {"delta": -delta}
        inverse_rule["support"] = 0
        inverse_rule["derivation"] = {
            "type": "inverse_action_symmetry",
            "source_action": action,
            "source_rule_id": str(rule.get("rule_id", "")),
        }
        derived_contextual_rules.append(inverse_rule)
        contextual_existing.add((inverse_action, field))

    return {
        "validated_rules": [*validated_rules, *derived_rules],
        "contextual_rules": [*contextual_rules, *derived_contextual_rules],
    }


def _actions_from_library(rule_library: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for item in rule_library.get("validated_rules", []):
        rule = item.get("rule") if isinstance(item, dict) else None
        if not isinstance(rule, dict):
            continue
        action = str(rule.get("action", ""))
        if action and action not in actions:
            actions.append(action)
    for rule in rule_library.get("contextual_rules", []):
        if not isinstance(rule, dict):
            continue
        action = str(rule.get("action", ""))
        if action and action not in actions:
            actions.append(action)
    return actions


def _modeled_fields_from_library(rule_library: dict[str, Any]) -> set[str]:
    fields: set[str] = set()
    for item in rule_library.get("validated_rules", []):
        rule = item.get("rule") if isinstance(item, dict) else None
        if isinstance(rule, dict) and rule.get("field"):
            fields.add(str(rule["field"]))
    for rule in rule_library.get("contextual_rules", []):
        if isinstance(rule, dict) and rule.get("field"):
            fields.add(str(rule["field"]))
    return fields


def _changed_fields(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    keys = set(before) | set(after)
    return sorted(key for key in keys if before.get(key) != after.get(key))


def _is_boundary_transition(choice: ActionChoice) -> bool:
    fields = set(choice.changed_field_names)
    return "level_index" in fields or "levels_completed" in fields or "state" in fields


CONTEXT_FIELDS = ("level_index", "levels_completed", "x", "y", "shape", "color", "rotation")


def _state_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    distance = 0.0
    compared = 0
    for field in CONTEXT_FIELDS:
        if field not in left or field not in right:
            continue
        compared += 1
        left_value = left[field]
        right_value = right[field]
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            distance += abs(float(left_value) - float(right_value))
        elif left_value != right_value:
            distance += 10.0
    return distance if compared else 1_000_000.0


def _context_action_scores(train_rows: list[dict[str, Any]], state: dict[str, Any], k: int = 5) -> dict[str, float]:
    neighbors: list[tuple[float, str]] = []
    for row in train_rows:
        before = row.get("state_before") if isinstance(row.get("state_before"), dict) else {}
        action = str(row.get("action", ""))
        if not action:
            continue
        neighbors.append((_state_distance(state, before), action))
    scores: dict[str, float] = {}
    for distance, action in sorted(neighbors, key=lambda item: item[0])[:k]:
        scores[action] = scores.get(action, 0.0) + (1.0 / (1.0 + distance))
    return scores


def _score_prediction(
    before: dict[str, Any],
    after: dict[str, Any],
    predicted: dict[str, Any],
    changed: list[str],
    modeled_changed: list[str],
) -> tuple[int, int, list[str], list[str], int, bool, bool]:
    changed_matches = sum(1 for field in changed if predicted.get(field) == after.get(field))
    changed_missed = [field for field in changed if predicted.get(field) != after.get(field)]
    side_effects = sum(
        1
        for field in set(before) | set(after) | set(predicted)
        if field not in changed and predicted.get(field) != after.get(field)
    )
    exact = bool(changed) and changed_matches == len(changed) and side_effects == 0
    modeled_matches = sum(1 for field in modeled_changed if predicted.get(field) == after.get(field))
    modeled_missed = [field for field in modeled_changed if predicted.get(field) != after.get(field)]
    modeled_exact = bool(modeled_changed) and modeled_matches == len(modeled_changed)
    return changed_matches, modeled_matches, changed_missed, modeled_missed, side_effects, exact, modeled_exact


def _choose_action(
    rule_library: dict[str, Any],
    actions: list[str],
    modeled_fields: set[str],
    row: dict[str, Any],
    context_scores: dict[str, float] | None = None,
) -> ActionChoice:
    before = row.get("state_before") if isinstance(row.get("state_before"), dict) else {}
    after = row.get("state_after") if isinstance(row.get("state_after"), dict) else {}
    changed = _changed_fields(before, after)
    modeled_changed = [field for field in changed if field in modeled_fields]
    if context_scores is None:
        context_scores = {}
    scored: list[tuple[tuple[int, int, int, int, float], str, bool, bool, int, int, list[str], list[str], int]] = []

    for action in actions:
        prediction = predict_state(rule_library, before, action)
        applied = sum(1 for item in prediction.applications if item.status == "applied")
        if applied == 0:
            continue
        changed_matches, modeled_matches, changed_missed, modeled_missed, side_effects, exact, modeled_exact = _score_prediction(
            before, after, prediction.predicted_state, changed, modeled_changed
        )
        scored.append(
            (
                (modeled_matches, changed_matches, -side_effects, applied, context_scores.get(action, 0.0)),
                action,
                exact,
                modeled_exact,
                changed_matches,
                modeled_matches,
                changed_missed,
                modeled_missed,
                side_effects,
            )
        )

    if not scored:
        return ActionChoice(
            level_id=str(row.get("level_id", "")),
            step_index=int(row.get("step_index", 0)),
            actual_action=str(row.get("action", "")),
            selected_action=None,
            best_actions=[],
            decidable=False,
            exact_transition_match=False,
            modeled_transition_match=False,
            changed_field_matches=0,
            modeled_field_matches=0,
            changed_fields=len(changed),
            modeled_fields=len(modeled_changed),
            changed_field_names=changed,
            modeled_field_names=modeled_changed,
            changed_missed_fields=changed,
            modeled_missed_fields=modeled_changed,
            side_effects=0,
        )

    best_score = max(item[0] for item in scored)
    best = [item for item in scored if item[0] == best_score]
    selected = sorted(item[1] for item in best)[0]
    selected_item = next(item for item in best if item[1] == selected)
    return ActionChoice(
        level_id=str(row.get("level_id", "")),
        step_index=int(row.get("step_index", 0)),
        actual_action=str(row.get("action", "")),
        selected_action=selected,
        best_actions=sorted(item[1] for item in best),
        decidable=True,
        exact_transition_match=selected_item[2],
        modeled_transition_match=selected_item[3],
        changed_field_matches=selected_item[4],
        modeled_field_matches=selected_item[5],
        changed_fields=len(changed),
        modeled_fields=len(modeled_changed),
        changed_field_names=changed,
        modeled_field_names=modeled_changed,
        changed_missed_fields=selected_item[6],
        modeled_missed_fields=selected_item[7],
        side_effects=selected_item[8],
    )


def evaluate_rule_policy(
    input_path: Path,
    train_fraction: float = 0.7,
    min_support: int = 2,
) -> RulePolicyScore:
    traces = [asdict(trace) for trace in extract_traces([input_path])]
    train_rows, test_rows = _split_traces(traces, train_fraction)
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
    boundary_choices = [choice for choice in decidable if _is_boundary_transition(choice)]
    non_boundary_choices = [choice for choice in decidable if not _is_boundary_transition(choice)]
    boundary_top1_matches = sum(1 for choice in boundary_choices if choice.selected_action == choice.actual_action)
    non_boundary_top1_matches = sum(1 for choice in non_boundary_choices if choice.selected_action == choice.actual_action)
    oracle_matches = sum(1 for choice in decidable if choice.actual_action in choice.best_actions)
    train_action_counts = Counter(str(row.get("action", "")) for row in train_rows)
    frequency_action = train_action_counts.most_common(1)[0][0] if train_action_counts else ""
    frequency_matches = sum(1 for row in test_rows if str(row.get("action", "")) == frequency_action)
    exact_matches = sum(1 for choice in decidable if choice.exact_transition_match)
    modeled_matches = sum(1 for choice in decidable if choice.modeled_transition_match)
    frequency_accuracy = frequency_matches / len(test_rows) if test_rows else 0.0
    top1_accuracy = top1_matches / len(decidable) if decidable else 0.0

    return RulePolicyScore(
        input=str(input_path),
        train_transitions=len(train_rows),
        test_transitions=len(test_rows),
        train_fraction=train_fraction,
        candidate_rules=sum(1 for rule in rule_rows if rule.get("status") == "candidate"),
        contextual_rules=sum(1 for rule in contextual_rule_rows if rule.get("status") == "candidate"),
        learned_actions=len(actions),
        decidable_transitions=len(decidable),
        no_rule_applicable=len(choices) - len(decidable),
        top1_action_matches=top1_matches,
        oracle_action_matches=oracle_matches,
        frequency_baseline_matches=frequency_matches,
        exact_transition_matches=exact_matches,
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
        exact_transition_accuracy=exact_matches / len(decidable) if decidable else 0.0,
        modeled_transition_accuracy=modeled_matches / len(decidable) if decidable else 0.0,
        choices=choices,
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
    score = evaluate_rule_policy(
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
