"""Evaluate the validated-rule planner on compact planning scenarios.

This is an offline benchmark harness for the learned rule library. Scenarios are
JSON objects with an initial state, goal state, and optional expected actions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.plan_arc3_with_validated_rules import plan_with_rules


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    solved: bool
    expected_solved: bool | None
    expected_actions: list[str] | None
    actions: list[str]
    action_match: bool | None
    final_state: dict[str, Any]
    reason: str


@dataclass(frozen=True)
class PlannerEvaluation:
    scenarios: int
    solved: int
    expected_solved_matches: int
    action_matches: int
    results: list[ScenarioResult]


def _read_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {label} object: {path}")
    return payload


def _read_scenarios(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected scenario list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _read_contextual_rules(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("contextual_rules", [])
    if not isinstance(payload, list):
        raise ValueError(f"Expected contextual rule list or library object: {path}")
    return [item for item in payload if isinstance(item, dict)]


def evaluate_planner(
    rule_library: dict[str, Any],
    scenarios: list[dict[str, Any]],
    max_depth: int = 6,
) -> PlannerEvaluation:
    results: list[ScenarioResult] = []

    for index, scenario in enumerate(scenarios):
        scenario_id = str(scenario.get("id", index))
        initial_state = scenario.get("initial_state")
        goal_state = scenario.get("goal_state")
        if not isinstance(initial_state, dict) or not isinstance(goal_state, dict):
            results.append(
                ScenarioResult(
                    scenario_id=scenario_id,
                    solved=False,
                    expected_solved=None,
                    expected_actions=None,
                    actions=[],
                    action_match=None,
                    final_state={},
                    reason="scenario missing initial_state or goal_state object",
                )
            )
            continue

        scenario_depth = int(scenario.get("max_depth", max_depth))
        plan = plan_with_rules(rule_library, initial_state, goal_state, max_depth=scenario_depth)
        expected_solved = scenario.get("expected_solved")
        if expected_solved is not None:
            expected_solved = bool(expected_solved)
        expected_actions = scenario.get("expected_actions")
        if expected_actions is not None and not isinstance(expected_actions, list):
            expected_actions = None
        action_match = None if expected_actions is None else plan.actions == [str(action) for action in expected_actions]
        results.append(
            ScenarioResult(
                scenario_id=scenario_id,
                solved=plan.solved,
                expected_solved=expected_solved,
                expected_actions=[str(action) for action in expected_actions] if expected_actions is not None else None,
                actions=plan.actions,
                action_match=action_match,
                final_state=plan.final_state,
                reason=plan.reason,
            )
        )

    solved = sum(1 for result in results if result.solved)
    expected_solved_matches = sum(
        1 for result in results if result.expected_solved is not None and result.expected_solved == result.solved
    )
    action_matches = sum(1 for result in results if result.action_match is True)
    return PlannerEvaluation(
        scenarios=len(results),
        solved=solved,
        expected_solved_matches=expected_solved_matches,
        action_matches=action_matches,
        results=results,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validated_rules_json", type=Path)
    parser.add_argument("scenarios_json", type=Path)
    parser.add_argument("--contextual-rules", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = _read_object(args.validated_rules_json, "validated rule library")
    if args.contextual_rules:
        library = {**library, "contextual_rules": _read_contextual_rules(args.contextual_rules)}
    evaluation = evaluate_planner(
        library,
        _read_scenarios(args.scenarios_json),
        max_depth=args.max_depth,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(evaluation), indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
