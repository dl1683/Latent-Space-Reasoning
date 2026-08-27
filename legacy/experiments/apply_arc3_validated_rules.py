"""Apply a validated ARC-3 rule library to proposed local states.

This is the first downstream consumer of the mechanistic pipeline. It predicts
the next state for a single action using only exported validated rules.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuleApplication:
    rule_id: str
    action: str
    field: str
    status: str
    reason: str
    before: Any
    after: Any


@dataclass(frozen=True)
class PredictionResult:
    action: str
    input_state: dict[str, Any]
    predicted_state: dict[str, Any]
    applications: list[RuleApplication]


def _read_rule_library(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected validated rule library object: {path}")
    return payload


def _read_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected state object: {path}")
    return payload


def _matches_expected_precondition(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict) and set(expected) == {"min", "max"}:
        return (
            isinstance(actual, (int, float))
            and isinstance(expected["min"], (int, float))
            and isinstance(expected["max"], (int, float))
            and expected["min"] <= actual <= expected["max"]
        )
    return actual == expected


def _matches_preconditions(state: dict[str, Any], preconditions: dict[str, Any]) -> bool:
    return all(_matches_expected_precondition(state.get(field), expected) for field, expected in preconditions.items())


def _rule_items(rule_library: dict[str, Any]) -> list[dict[str, Any]]:
    rules = rule_library.get("validated_rules", [])
    if not isinstance(rules, list):
        rules = []
    contextual_rules = rule_library.get("contextual_rules", [])
    if not isinstance(contextual_rules, list):
        contextual_rules = []
    return [*rules, *({"rule": rule, "rule_id": rule.get("rule_id", "")} for rule in contextual_rules if isinstance(rule, dict))]


def predict_state(rule_library: dict[str, Any], state: dict[str, Any], action: str) -> PredictionResult:
    predicted = dict(state)
    applications: list[RuleApplication] = []
    applied_fields: set[str] = set()

    for item in _rule_items(rule_library):
        if not isinstance(item, dict):
            continue
        rule = item.get("rule")
        if not isinstance(rule, dict):
            continue
        if str(rule.get("action", "")) != action:
            continue
        field = str(rule.get("field", ""))
        if field in applied_fields:
            applications.append(
                RuleApplication(
                    rule_id=str(item.get("rule_id", rule.get("rule_id", ""))),
                    action=action,
                    field=field,
                    status="skipped",
                    reason="field already predicted",
                    before=predicted.get(field),
                    after=predicted.get(field),
                )
            )
            continue
        effect = rule.get("effect") if isinstance(rule.get("effect"), dict) else {}
        preconditions = rule.get("preconditions") if isinstance(rule.get("preconditions"), dict) else {}
        if preconditions and not _matches_preconditions(predicted, preconditions):
            applications.append(
                RuleApplication(
                    rule_id=str(item.get("rule_id", rule.get("rule_id", ""))),
                    action=action,
                    field=field,
                    status="skipped",
                    reason="context precondition mismatch",
                    before=predicted.get(field),
                    after=predicted.get(field),
                )
            )
            continue
        actual_before = predicted.get(field)
        if "delta" in effect:
            if not isinstance(actual_before, (int, float)):
                applications.append(
                    RuleApplication(
                        rule_id=str(item.get("rule_id", rule.get("rule_id", ""))),
                        action=action,
                        field=field,
                        status="skipped",
                        reason="delta precondition mismatch",
                        before=actual_before,
                        after=actual_before,
                    )
                )
                continue
            expected_after = actual_before + effect["delta"]
        else:
            expected_before = effect.get("before")
            expected_after = effect.get("after")
            if actual_before != expected_before:
                applications.append(
                    RuleApplication(
                        rule_id=str(item.get("rule_id", rule.get("rule_id", ""))),
                        action=action,
                        field=field,
                        status="skipped",
                        reason="precondition mismatch",
                        before=actual_before,
                        after=actual_before,
                    )
                )
                continue
        predicted[field] = expected_after
        applied_fields.add(field)
        applications.append(
            RuleApplication(
                rule_id=str(item.get("rule_id", rule.get("rule_id", ""))),
                action=action,
                field=field,
                status="applied",
                reason="matched preconditions",
                before=actual_before,
                after=expected_after,
            )
        )

    return PredictionResult(
        action=action,
        input_state=state,
        predicted_state=predicted,
        applications=applications,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validated_rules_json", type=Path)
    parser.add_argument("state_json", type=Path)
    parser.add_argument("--action", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_state(_read_rule_library(args.validated_rules_json), _read_state(args.state_json), args.action)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(result), indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
