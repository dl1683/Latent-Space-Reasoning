"""Plan short abstract action sequences with validated ARC-3 rules.

This planner operates over compact JSON state dictionaries, not the full game
engine. It is useful for testing whether the learned rule library can compose
validated transitions toward a goal before spending time on real ARC-3 runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.apply_arc3_validated_rules import predict_state


@dataclass(frozen=True)
class AbstractPlan:
    solved: bool
    actions: list[str]
    final_state: dict[str, Any]
    expanded_states: int
    reason: str


def _read_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {label} object: {path}")
    return payload


def _freeze_state(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True)


def _actions_from_library(rule_library: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for item in rule_library.get("validated_rules", []):
        if not isinstance(item, dict):
            continue
        rule = item.get("rule")
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


def _goal_satisfied(state: dict[str, Any], goal: dict[str, Any]) -> bool:
    return all(state.get(key) == value for key, value in goal.items())


def plan_with_rules(
    rule_library: dict[str, Any],
    initial_state: dict[str, Any],
    goal_state: dict[str, Any],
    max_depth: int = 6,
) -> AbstractPlan:
    actions = _actions_from_library(rule_library)
    if _goal_satisfied(initial_state, goal_state):
        return AbstractPlan(True, [], initial_state, 0, "initial state satisfies goal")
    if not actions:
        return AbstractPlan(False, [], initial_state, 0, "no validated actions available")

    queue = deque([(initial_state, [])])
    visited = {_freeze_state(initial_state)}
    expanded = 0

    while queue:
        state, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        expanded += 1
        for action in actions:
            prediction = predict_state(rule_library, state, action)
            next_state = prediction.predicted_state
            if next_state == state:
                continue
            next_key = _freeze_state(next_state)
            if next_key in visited:
                continue
            next_path = [*path, action]
            if _goal_satisfied(next_state, goal_state):
                return AbstractPlan(True, next_path, next_state, expanded, "goal reached")
            visited.add(next_key)
            queue.append((next_state, next_path))

    return AbstractPlan(False, [], initial_state, expanded, "goal not reached within depth bound")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validated_rules_json", type=Path)
    parser.add_argument("initial_state_json", type=Path)
    parser.add_argument("goal_state_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = plan_with_rules(
        _read_object(args.validated_rules_json, "validated rule library"),
        _read_object(args.initial_state_json, "initial state"),
        _read_object(args.goal_state_json, "goal state"),
        max_depth=args.max_depth,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(plan), indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
