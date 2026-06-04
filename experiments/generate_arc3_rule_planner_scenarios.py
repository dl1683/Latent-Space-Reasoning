"""Generate compact planner scenarios from a validated ARC-3 rule library.

The generated scenarios are offline regression cases. They ask whether the
validated-rule planner can reproduce each rule's effect and compose pairs of
independent effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_library(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected validated rule library object: {path}")
    return payload


def _validated_rule_items(library: dict[str, Any]) -> list[dict[str, Any]]:
    items = library.get("validated_rules", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and isinstance(item.get("rule"), dict)]


def _contextual_rule_items(library: dict[str, Any]) -> list[dict[str, Any]]:
    items = library.get("contextual_rules", [])
    if not isinstance(items, list):
        return []
    return [
        {"rule_id": str(rule.get("rule_id", "")), "rule": rule}
        for rule in items
        if isinstance(rule, dict)
    ]


def _read_contextual_rules(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("contextual_rules", [])
    if not isinstance(payload, list):
        raise ValueError(f"Expected contextual rule list or library object: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _rule_preconditions(item: dict[str, Any]) -> dict[str, Any]:
    rule = item["rule"]
    preconditions = rule.get("preconditions")
    return dict(preconditions) if isinstance(preconditions, dict) else {}


def _rule_effect(item: dict[str, Any]) -> tuple[str, str, Any, Any] | None:
    rule = item["rule"]
    effect = rule.get("effect")
    if not isinstance(effect, dict):
        return None
    action = str(rule.get("action", ""))
    field = str(rule.get("field", ""))
    if not action or not field:
        return None
    if "delta" in effect:
        before = _rule_preconditions(item).get(field, 0)
        return action, field, before, before + effect["delta"]
    return action, field, effect.get("before"), effect.get("after")


def generate_scenarios(library: dict[str, Any], include_pairs: bool = True) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    effects: list[tuple[str, str, Any, Any, str, dict[str, Any]]] = []
    action_fields: dict[str, set[str]] = {}

    for item in [*_validated_rule_items(library), *_contextual_rule_items(library)]:
        effect = _rule_effect(item)
        if effect is None:
            continue
        action, field, before, after = effect
        rule_id = str(item.get("rule_id", item["rule"].get("rule_id", "")))
        preconditions = _rule_preconditions(item)
        initial_state = {**preconditions, field: before}
        goal_state = {field: after}
        action_fields.setdefault(action, set()).add(field)
        effects.append((action, field, before, after, rule_id, preconditions))
        scenarios.append(
            {
                "id": f"one-step:{rule_id}",
                "initial_state": initial_state,
                "goal_state": goal_state,
                "expected_solved": True,
                "expected_actions": [action],
                "max_depth": 1,
            }
        )

    if include_pairs:
        for left_index, left in enumerate(effects):
            for right in effects[left_index + 1 :]:
                left_action, left_field, left_before, left_after, left_rule_id, left_preconditions = left
                right_action, right_field, right_before, right_after, right_rule_id, right_preconditions = right
                if left_field == right_field:
                    continue
                if right_field in action_fields.get(left_action, set()):
                    continue
                if left_field in action_fields.get(right_action, set()):
                    continue
                if left_field in right_preconditions or right_field in left_preconditions:
                    continue
                scenarios.append(
                    {
                        "id": f"two-step:{left_rule_id}+{right_rule_id}",
                        "initial_state": {left_field: left_before, right_field: right_before},
                        "goal_state": {left_field: left_after, right_field: right_after},
                        "expected_solved": True,
                        "expected_actions": [left_action, right_action],
                        "max_depth": 2,
                    }
                )

    return scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validated_rules_json", type=Path)
    parser.add_argument("--contextual-rules", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-pairs", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = _read_library(args.validated_rules_json)
    if args.contextual_rules:
        library = {**library, "contextual_rules": _read_contextual_rules(args.contextual_rules)}
    scenarios = generate_scenarios(library, include_pairs=not args.no_pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(scenarios, indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
