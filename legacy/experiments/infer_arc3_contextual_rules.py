"""Infer preconditioned contextual rules from transition traces.

This complements infer_arc3_rules.py. The base rule inferencer only promotes
unconditional effects when an action/field has a stable effect. This file keeps
the repeated special cases instead of discarding them entirely: when one
action/field has multiple observed numeric deltas, each repeated delta can
become a contextual rule if its evidence shares stable before-state fields.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DELTA_FIELDS = {"x", "y"}
PRECONDITION_FIELDS = ("level_index", "levels_completed", "x", "y", "shape", "color", "rotation", "delivered")
INTERVAL_PRECONDITION_FIELDS = ("x", "y")


@dataclass(frozen=True)
class ContextualRule:
    rule_id: str
    level_id: str
    action: str
    field: str
    effect: dict[str, Any]
    preconditions: dict[str, Any]
    support: int
    status: str
    evidence_steps: list[int]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


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


def _stable_preconditions(evidence: list[dict[str, Any]], effect_field: str) -> dict[str, Any]:
    stable: dict[str, Any] = {}
    for field in PRECONDITION_FIELDS:
        values = []
        for trace in evidence:
            before = trace.get("state_before") if isinstance(trace.get("state_before"), dict) else {}
            if field not in before:
                break
            values.append(_freeze(before[field]))
        if len(values) == len(evidence) and values and len(set(values)) == 1:
            stable[field] = _thaw(values[0])
    return stable


def _matches_preconditions(trace: dict[str, Any], preconditions: dict[str, Any]) -> bool:
    before = trace.get("state_before") if isinstance(trace.get("state_before"), dict) else {}
    for field, expected in preconditions.items():
        if field not in before:
            return False
        actual = _thaw(_freeze(before[field]))
        if isinstance(expected, dict) and set(expected) == {"min", "max"}:
            if not isinstance(actual, (int, float)) or not expected["min"] <= actual <= expected["max"]:
                return False
            continue
        if actual != expected:
            return False
    return True


def _interval_preconditions(
    evidence: list[dict[str, Any]],
    counterexamples: list[dict[str, Any]],
    effect_field: str,
) -> dict[str, Any]:
    best: tuple[int, float, str, dict[str, Any]] | None = None
    for field in INTERVAL_PRECONDITION_FIELDS:
        values: list[float] = []
        for trace in evidence:
            before = trace.get("state_before") if isinstance(trace.get("state_before"), dict) else {}
            value = before.get(field)
            if not isinstance(value, (int, float)):
                values = []
                break
            values.append(float(value))
        if len(values) != len(evidence) or not values:
            continue
        minimum = min(values)
        maximum = max(values)
        if minimum == maximum:
            continue
        preconditions = {field: {"min": minimum, "max": maximum}}
        counterexample_matches = sum(1 for trace in counterexamples if _matches_preconditions(trace, preconditions))
        if counterexample_matches >= len(evidence):
            continue
        candidate = (counterexample_matches, maximum - minimum, field, preconditions)
        if best is None or candidate < best:
            best = candidate
    return best[3] if best else {}


def _discriminating_preconditions(
    evidence: list[dict[str, Any]],
    counterexamples: list[dict[str, Any]],
    effect_field: str,
) -> dict[str, Any]:
    candidates = _stable_preconditions(evidence, effect_field)
    if candidates:
        counterexample_matches = sum(1 for trace in counterexamples if _matches_preconditions(trace, candidates))
        if counterexample_matches < len(evidence):
            return candidates
    return _interval_preconditions(evidence, counterexamples, effect_field)


def infer_contextual_rules(
    traces: Iterable[dict[str, Any]],
    min_support: int = 2,
) -> list[ContextualRule]:
    grouped: dict[tuple[str, str, str], dict[Any, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for trace in traces:
        before = trace.get("state_before") if isinstance(trace.get("state_before"), dict) else {}
        after = trace.get("state_after") if isinstance(trace.get("state_after"), dict) else {}
        action = str(trace.get("action", ""))
        level_id = str(trace.get("level_id", "unknown"))
        for field in DELTA_FIELDS:
            old = before.get(field)
            new = after.get(field)
            if isinstance(old, (int, float)) and isinstance(new, (int, float)) and old != new:
                grouped[(level_id, action, field)][new - old].append(trace)

    rules: list[ContextualRule] = []
    for (level_id, action, field), by_delta in sorted(grouped.items()):
        if len(by_delta) < 2:
            continue
        for delta, evidence in sorted(by_delta.items(), key=lambda item: (-len(item[1]), item[0])):
            if len(evidence) < min_support:
                continue
            counterexamples = [
                trace
                for other_delta, traces_for_delta in by_delta.items()
                if other_delta != delta
                for trace in traces_for_delta
            ]
            preconditions = _discriminating_preconditions(evidence, counterexamples, field)
            if not preconditions:
                continue
            steps = [int(item.get("step_index", 0)) for item in evidence]
            precondition_id = ",".join(f"{key}={preconditions[key]}" for key in sorted(preconditions))
            rules.append(
                ContextualRule(
                    rule_id=f"{level_id}:{action}:{field}:delta:{delta}:when:{precondition_id}",
                    level_id=level_id,
                    action=action,
                    field=field,
                    effect={"delta": delta},
                    preconditions=preconditions,
                    support=len(evidence),
                    status="candidate",
                    evidence_steps=steps,
                )
            )

    return rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = infer_contextual_rules(_read_jsonl(args.trace_jsonl), min_support=args.min_support)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([asdict(rule) for rule in rules], indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
