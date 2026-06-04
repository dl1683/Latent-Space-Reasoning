"""Infer conservative rule hypotheses from ARC-3 object hypotheses.

Rules here are not prompts and not explanations. They are small, auditable
claims of the form: when action X occurs, field Y changes from A to B. A rule is
only promoted when the same effect is observed more than once and no conflicting
effect has been seen for that action/field pair.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MIN_SUPPORT = 2
DELTA_FIELDS = {"steps", "x", "y"}
BOUNDARY_DELTA_FIELDS = {"level_index", "levels_completed"}


@dataclass(frozen=True)
class RuleHypothesis:
    rule_id: str
    level_id: str
    object_id: str
    object_type: str
    action: str
    field: str
    effect: dict[str, Any]
    support: int
    counterexamples: int
    status: str
    evidence_steps: list[int]


def _read_objects(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of object hypotheses: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw(item) for key, item in value}
        return [_thaw(item) for item in value]
    return value


def _change_effect(change: dict[str, Any]) -> tuple[Any, Any]:
    return _freeze(change.get("before")), _freeze(change.get("after"))


def _delta_effect(field: str, change: dict[str, Any]) -> tuple[str, Any] | None:
    if field not in DELTA_FIELDS:
        return None
    before = change.get("before")
    after = change.get("after")
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        delta = after - before
        if field == "steps" and delta != 1:
            return None
        return "delta", delta
    return None


def infer_rules(objects: Iterable[dict[str, Any]], min_support: int = MIN_SUPPORT) -> list[RuleHypothesis]:
    rules: list[RuleHypothesis] = []

    for obj in objects:
        level_id = str(obj.get("level_id", "unknown"))
        object_id = str(obj.get("object_id", "unknown"))
        object_type = str(obj.get("object_type", "unknown"))
        transitions = obj.get("transitions", [])
        if not isinstance(transitions, list):
            continue

        by_action_field: dict[tuple[str, str], dict[tuple[Any, ...], list[int]]] = defaultdict(lambda: defaultdict(list))
        null_observations: dict[tuple[str, str], int] = defaultdict(int)

        for transition in transitions:
            if not isinstance(transition, dict):
                continue
            action = str(transition.get("action", ""))
            step = int(transition.get("step_index", 0))
            changes = transition.get("changed_keys", {})
            if not isinstance(changes, dict):
                changes = {}
            level_boundary = "level_index" in changes or "levels_completed" in changes
            for field, change in changes.items():
                if isinstance(change, dict):
                    by_action_field[(action, str(field))][_change_effect(change)].append(step)
                    delta = _delta_effect(str(field), change)
                    if delta is not None and not level_boundary:
                        by_action_field[(action, str(field))][delta].append(step)
                    if (
                        str(field) in BOUNDARY_DELTA_FIELDS
                        and level_boundary
                        and isinstance(change.get("before"), (int, float))
                        and isinstance(change.get("after"), (int, float))
                    ):
                        by_action_field[(action, str(field))][("boundary_delta", change["after"] - change["before"])].append(step)
            for action_field in list(by_action_field):
                seen_action, seen_field = action_field
                if seen_action == action and seen_field not in changes:
                    null_observations[action_field] += 1

        for (action, field), effects in sorted(by_action_field.items()):
            total_effects = sum(len(steps) for steps in effects.values())
            for effect_key, steps in sorted(effects.items(), key=lambda item: (-len(item[1]), str(item[0]))):
                if len(effect_key) == 2 and effect_key[0] == "delta":
                    delta_total = sum(
                        len(effect_steps)
                        for other_key, effect_steps in effects.items()
                        if len(other_key) == 2 and other_key[0] == "delta"
                    )
                    counterexamples = delta_total - len(steps) + null_observations[(action, field)]
                    status = "candidate" if len(steps) >= min_support and counterexamples == 0 else "unknown"
                    effect = {"delta": _thaw(effect_key[1])}
                    rule_id = f"{level_id}:{object_id}:{action}:{field}:delta:{effect['delta']}"
                elif len(effect_key) == 2 and effect_key[0] == "boundary_delta":
                    boundary_total = sum(
                        len(effect_steps)
                        for other_key, effect_steps in effects.items()
                        if len(other_key) == 2 and other_key[0] == "boundary_delta"
                    )
                    counterexamples = boundary_total - len(steps)
                    status = "candidate" if len(steps) >= min_support and counterexamples == 0 else "unknown"
                    effect = {"delta": _thaw(effect_key[1]), "scope": "level_boundary"}
                    rule_id = f"{level_id}:{object_id}:{action}:{field}:boundary_delta:{effect['delta']}"
                else:
                    absolute_total = sum(
                        len(effect_steps)
                        for other_key, effect_steps in effects.items()
                        if not (len(other_key) == 2 and other_key[0] == "delta")
                    )
                    counterexamples = absolute_total - len(steps) + null_observations[(action, field)]
                    status = "candidate" if len(steps) >= min_support and counterexamples == 0 else "unknown"
                    before_value = _thaw(effect_key[0])
                    after_value = _thaw(effect_key[1])
                    effect = {"before": before_value, "after": after_value}
                    rule_id = f"{level_id}:{object_id}:{action}:{field}:{before_value}->{after_value}"
                rules.append(
                    RuleHypothesis(
                        rule_id=rule_id,
                        level_id=level_id,
                        object_id=object_id,
                        object_type=object_type,
                        action=action,
                        field=field,
                        effect=effect,
                        support=len(steps),
                        counterexamples=counterexamples,
                        status=status,
                        evidence_steps=steps,
                    )
                )

    return rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objects_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = infer_rules(_read_objects(args.objects_json), min_support=args.min_support)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([asdict(rule) for rule in rules], indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
