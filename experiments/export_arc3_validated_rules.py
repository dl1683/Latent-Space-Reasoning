"""Export reusable ARC-3 rules from graded rule output.

This keeps downstream planners away from tentative or rejected hypotheses. The
output is a compact rule library plus a manifest that records what was excluded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuleLibrary:
    source: str
    validation_threshold: int | None
    validated_rules: list[dict[str, Any]]
    excluded_counts: dict[str, int]


def _read_graded_rules(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of graded rules: {path}")
    return [item for item in payload if isinstance(item, dict)]


def export_validated_rules(path: Path) -> RuleLibrary:
    graded_rules = _read_graded_rules(path)
    validated: list[dict[str, Any]] = []
    excluded_counts: dict[str, int] = {}
    threshold: int | None = None

    for graded in graded_rules:
        status = str(graded.get("status", "unknown"))
        if isinstance(graded.get("validation_threshold"), int):
            threshold = graded["validation_threshold"]
        if status == "validated":
            rule = graded.get("rule")
            if isinstance(rule, dict):
                validated.append(
                    {
                        "rule_id": str(graded.get("rule_id", rule.get("rule_id", ""))),
                        "supported": int(graded.get("supported", 0)),
                        "contradicted": int(graded.get("contradicted", 0)),
                        "not_applicable": int(graded.get("not_applicable", 0)),
                        "rule": rule,
                    }
                )
        else:
            excluded_counts[status] = excluded_counts.get(status, 0) + 1

    return RuleLibrary(
        source=str(path),
        validation_threshold=threshold,
        validated_rules=validated,
        excluded_counts=excluded_counts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graded_rules_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = export_validated_rules(args.graded_rules_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(library), indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
