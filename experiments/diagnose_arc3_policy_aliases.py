"""Diagnose learned-rule policy alias failures.

This reads an online rule-learning artifact and summarizes cases where the
selected action differs from the actual action. The goal is to expose whether
failures are concentrated in a few action confusions or spread across the run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AliasFailure:
    step_number: int
    step_index: int
    actual_action: str
    selected_action: str | None
    best_actions: list[str]
    changed_field_matches: int
    changed_fields: int
    modeled_field_matches: int
    modeled_fields: int
    changed_missed_fields: list[str]
    modeled_missed_fields: list[str]


@dataclass(frozen=True)
class AliasDiagnosis:
    input: str
    evaluated_transitions: int
    failures: int
    failure_rate: float
    confusion_counts: dict[str, int]
    actual_action_failures: dict[str, int]
    selected_action_failures: dict[str, int]
    modeled_missed_field_counts: dict[str, int]
    changed_missed_field_counts: dict[str, int]
    oracle_misses: int
    modeled_zero_match_failures: int
    failures_by_step: list[AliasFailure]


def diagnose_aliases(path: Path) -> AliasDiagnosis:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    steps = payload.get("steps", []) if isinstance(payload, dict) else []
    failures: list[AliasFailure] = []
    confusion_counts: Counter[str] = Counter()
    actual_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    modeled_missed_counts: Counter[str] = Counter()
    changed_missed_counts: Counter[str] = Counter()
    oracle_misses = 0
    modeled_zero = 0

    for step in steps:
        if not isinstance(step, dict):
            continue
        choice = step.get("choice") if isinstance(step.get("choice"), dict) else {}
        actual = str(choice.get("actual_action", ""))
        selected = choice.get("selected_action")
        selected_text = str(selected) if selected is not None else None
        if selected_text == actual:
            continue
        best_actions = choice.get("best_actions") if isinstance(choice.get("best_actions"), list) else []
        if actual not in best_actions:
            oracle_misses += 1
        if int(choice.get("modeled_field_matches", 0)) == 0:
            modeled_zero += 1
        changed_missed = [
            str(field)
            for field in choice.get("changed_missed_fields", [])
            if isinstance(choice.get("changed_missed_fields", []), list)
        ]
        modeled_missed = [
            str(field)
            for field in choice.get("modeled_missed_fields", [])
            if isinstance(choice.get("modeled_missed_fields", []), list)
        ]
        changed_missed_counts.update(changed_missed)
        modeled_missed_counts.update(modeled_missed)
        confusion_counts[f"{actual}->{selected_text}"] += 1
        actual_counts[actual] += 1
        if selected_text is not None:
            selected_counts[selected_text] += 1
        failures.append(
            AliasFailure(
                step_number=int(step.get("step_number", 0)),
                step_index=int(choice.get("step_index", 0)),
                actual_action=actual,
                selected_action=selected_text,
                best_actions=[str(action) for action in best_actions],
                changed_field_matches=int(choice.get("changed_field_matches", 0)),
                changed_fields=int(choice.get("changed_fields", 0)),
                modeled_field_matches=int(choice.get("modeled_field_matches", 0)),
                modeled_fields=int(choice.get("modeled_fields", 0)),
                changed_missed_fields=changed_missed,
                modeled_missed_fields=modeled_missed,
            )
        )

    evaluated = int(payload.get("evaluated_transitions", len(steps))) if isinstance(payload, dict) else len(steps)
    return AliasDiagnosis(
        input=str(path),
        evaluated_transitions=evaluated,
        failures=len(failures),
        failure_rate=len(failures) / evaluated if evaluated else 0.0,
        confusion_counts=dict(confusion_counts),
        actual_action_failures=dict(actual_counts),
        selected_action_failures=dict(selected_counts),
        modeled_missed_field_counts=dict(modeled_missed_counts),
        changed_missed_field_counts=dict(changed_missed_counts),
        oracle_misses=oracle_misses,
        modeled_zero_match_failures=modeled_zero,
        failures_by_step=failures,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("online_rule_learning_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnosis = diagnose_aliases(args.online_rule_learning_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(diagnosis), indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(asdict(diagnosis), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
