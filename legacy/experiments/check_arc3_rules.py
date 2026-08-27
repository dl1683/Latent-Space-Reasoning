"""Check ARC-3 rule hypotheses against transition traces.

This is the verification layer for the offline mechanistic pipeline. It does
not prove a rule globally; it records whether candidate rules predict observed
transitions in a trace file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PredictionCheck:
    rule_id: str
    level_id: str
    step_index: int
    action: str
    field: str
    expected: dict[str, Any]
    observed: dict[str, Any] | None
    status: str


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _read_rules(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of rule hypotheses: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _observed_change(trace: dict[str, Any], field: str) -> dict[str, Any] | None:
    before = trace.get("state_before") if isinstance(trace.get("state_before"), dict) else {}
    after = trace.get("state_after") if isinstance(trace.get("state_after"), dict) else {}
    if field not in before and field not in after:
        return None
    observed = {"before": before.get(field), "after": after.get(field)}
    if observed["before"] == observed["after"]:
        return None
    return observed


def _is_level_boundary(trace: dict[str, Any]) -> bool:
    before = trace.get("state_before") if isinstance(trace.get("state_before"), dict) else {}
    after = trace.get("state_after") if isinstance(trace.get("state_after"), dict) else {}
    return before.get("level_index") != after.get("level_index") or before.get("levels_completed") != after.get(
        "levels_completed"
    )


def _matches_expected(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    if "delta" in expected:
        before = observed.get("before")
        after = observed.get("after")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            return after - before == expected["delta"]
        return False
    return observed == expected


def check_rules(traces: Iterable[dict[str, Any]], rules: Iterable[dict[str, Any]]) -> list[PredictionCheck]:
    candidate_rules = [rule for rule in rules if rule.get("status") == "candidate"]
    checks: list[PredictionCheck] = []

    for trace in traces:
        action = str(trace.get("action", ""))
        level_id = str(trace.get("level_id", "unknown"))
        step_index = int(trace.get("step_index", 0))
        for rule in candidate_rules:
            if str(rule.get("level_id", "unknown")) != level_id:
                continue
            if str(rule.get("action", "")) != action:
                continue
            field = str(rule.get("field", ""))
            expected = rule.get("effect") if isinstance(rule.get("effect"), dict) else {}
            if "delta" in expected and _is_level_boundary(trace) and expected.get("scope") != "level_boundary":
                continue
            observed = _observed_change(trace, field)
            if observed is None:
                status = "not_applicable"
            elif _matches_expected(observed, expected):
                status = "supported"
            else:
                status = "contradicted"
            checks.append(
                PredictionCheck(
                    rule_id=str(rule.get("rule_id", "")),
                    level_id=level_id,
                    step_index=step_index,
                    action=action,
                    field=field,
                    expected=expected,
                    observed=observed,
                    status=status,
                )
            )

    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_jsonl", type=Path)
    parser.add_argument("rules_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = check_rules(_read_jsonl(args.trace_jsonl), _read_rules(args.rules_json))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([asdict(check) for check in checks], indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
