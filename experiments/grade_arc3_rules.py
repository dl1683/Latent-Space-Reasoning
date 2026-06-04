"""Grade ARC-3 rule hypotheses from prediction checks.

This turns local check evidence into a reusable rule status:
- validated: enough supported checks and no contradictions
- rejected: at least one contradiction
- tentative: some support, not enough to validate
- untested: no applicable checks
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GradedRule:
    rule_id: str
    status: str
    supported: int
    contradicted: int
    not_applicable: int
    validation_threshold: int
    rule: dict[str, Any]


def _read_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def grade_rules(
    rules: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    validation_threshold: int = 2,
) -> list[GradedRule]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for check in checks:
        rule_id = str(check.get("rule_id", ""))
        status = str(check.get("status", "unknown"))
        counts[rule_id][status] += 1

    graded: list[GradedRule] = []
    for rule in rules:
        rule_id = str(rule.get("rule_id", ""))
        supported = counts[rule_id]["supported"]
        contradicted = counts[rule_id]["contradicted"]
        not_applicable = counts[rule_id]["not_applicable"]
        if contradicted:
            status = "rejected"
        elif supported >= validation_threshold:
            status = "validated"
        elif supported:
            status = "tentative"
        else:
            status = "untested"
        graded.append(
            GradedRule(
                rule_id=rule_id,
                status=status,
                supported=supported,
                contradicted=contradicted,
                not_applicable=not_applicable,
                validation_threshold=validation_threshold,
                rule=rule,
            )
        )
    return graded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rules_json", type=Path)
    parser.add_argument("checks_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-threshold", type=int, default=2)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graded = grade_rules(
        _read_list(args.rules_json),
        _read_list(args.checks_json),
        validation_threshold=args.validation_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([asdict(item) for item in graded], indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
