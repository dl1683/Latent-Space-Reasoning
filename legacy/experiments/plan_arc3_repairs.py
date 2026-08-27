"""Plan targeted repairs from contradicted ARC-3 prediction checks.

The repair planner is the feedback loop for the offline mechanistic pipeline.
It does not solve a level directly. It turns failed predictions into explicit
next observations or interventions that would improve the rule set.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RepairRecord:
    repair_id: str
    rule_id: str
    level_id: str
    action: str
    field: str
    priority: str
    reason: str
    contradicted_steps: list[int]
    expected_effects: list[dict[str, Any]]
    observed_effects: list[dict[str, Any]]
    requested_trace: dict[str, Any]


def _read_checks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of prediction checks: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _unique_dicts(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def plan_repairs(checks: Iterable[dict[str, Any]]) -> list[RepairRecord]:
    contradicted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    not_applicable: dict[str, int] = defaultdict(int)

    for check in checks:
        rule_id = str(check.get("rule_id", ""))
        status = check.get("status")
        if status == "contradicted":
            contradicted[rule_id].append(check)
        elif status == "not_applicable":
            not_applicable[rule_id] += 1

    repairs: list[RepairRecord] = []
    for rule_id, failures in sorted(contradicted.items()):
        first = failures[0]
        level_id = str(first.get("level_id", "unknown"))
        action = str(first.get("action", ""))
        field = str(first.get("field", ""))
        contradicted_steps = [int(item.get("step_index", 0)) for item in failures]
        expected_effects = _unique_dicts(
            item.get("expected", {}) for item in failures if isinstance(item.get("expected"), dict)
        )
        observed_effects = _unique_dicts(
            item.get("observed", {}) for item in failures if isinstance(item.get("observed"), dict)
        )
        priority = "high" if len(failures) > 1 or not_applicable[rule_id] else "medium"
        reason = (
            "Candidate rule has direct contradictions; collect a trace around the same action and field "
            "to split hidden preconditions from false effects."
        )
        repairs.append(
            RepairRecord(
                repair_id=f"repair:{rule_id}",
                rule_id=rule_id,
                level_id=level_id,
                action=action,
                field=field,
                priority=priority,
                reason=reason,
                contradicted_steps=contradicted_steps,
                expected_effects=expected_effects,
                observed_effects=observed_effects,
                requested_trace={
                    "level_id": level_id,
                    "action": action,
                    "field": field,
                    "include_state_before": True,
                    "include_state_after": True,
                    "include_neighbor_tiles": True,
                    "include_inventory_and_attributes": True,
                    "minimum_examples": max(2, len(failures) + 1),
                },
            )
        )

    return repairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checks_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repairs = plan_repairs(_read_checks(args.checks_json))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([asdict(repair) for repair in repairs], indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
