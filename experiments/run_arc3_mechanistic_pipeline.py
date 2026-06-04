"""Run the offline ARC-3 mechanistic pipeline end to end.

Input is an existing replay/trace artifact. Output is a directory containing:
transitions, object hypotheses, rule hypotheses, prediction checks, repairs,
and a manifest tying the artifacts together.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.check_arc3_contextual_rules import check_contextual_rules
from experiments.check_arc3_rules import check_rules
from experiments.extract_arc3_transitions import extract_traces
from experiments.export_arc3_validated_contextual_rules import export_validated_contextual_rules
from experiments.export_arc3_validated_rules import export_validated_rules
from experiments.grade_arc3_rules import grade_rules
from experiments.infer_arc3_contextual_rules import infer_contextual_rules
from experiments.infer_arc3_objects import infer_objects
from experiments.infer_arc3_rules import infer_rules
from experiments.plan_arc3_repairs import plan_repairs


def _write_json(path: Path, payload: Any, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2 if pretty else None, sort_keys=True),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def run_pipeline(input_path: Path, output_dir: Path, min_support: int = 2, pretty: bool = False) -> dict[str, Any]:
    transitions_path = output_dir / "transitions.jsonl"
    objects_path = output_dir / "objects.json"
    rules_path = output_dir / "rules.json"
    checks_path = output_dir / "rule_checks.json"
    graded_rules_path = output_dir / "graded_rules.json"
    validated_rules_path = output_dir / "validated_rules.json"
    contextual_rules_path = output_dir / "contextual_rules.json"
    contextual_checks_path = output_dir / "contextual_rule_checks.json"
    contextual_graded_rules_path = output_dir / "contextual_graded_rules.json"
    contextual_validated_rules_path = output_dir / "contextual_validated_rules.json"
    repairs_path = output_dir / "repairs.json"
    manifest_path = output_dir / "manifest.json"

    traces = extract_traces([input_path])
    trace_rows = [asdict(trace) for trace in traces]
    objects = infer_objects(trace_rows)
    object_rows = [asdict(item) for item in objects]
    rules = infer_rules(object_rows, min_support=min_support)
    rule_rows = [asdict(item) for item in rules]
    checks = check_rules(trace_rows, rule_rows)
    check_rows = [asdict(item) for item in checks]
    graded_rules = grade_rules(rule_rows, check_rows, validation_threshold=min_support)
    graded_rule_rows = [asdict(item) for item in graded_rules]
    contextual_rules = infer_contextual_rules(trace_rows, min_support=min_support)
    contextual_rule_rows = [asdict(item) for item in contextual_rules]
    contextual_checks = check_contextual_rules(trace_rows, contextual_rule_rows)
    contextual_check_rows = [asdict(item) for item in contextual_checks]
    contextual_graded_rules = grade_rules(
        contextual_rule_rows,
        contextual_check_rows,
        validation_threshold=min_support,
    )
    contextual_graded_rule_rows = [asdict(item) for item in contextual_graded_rules]
    repairs = plan_repairs(check_rows)
    repair_rows = [asdict(item) for item in repairs]

    _write_jsonl(transitions_path, trace_rows)
    _write_json(objects_path, object_rows, pretty=pretty)
    _write_json(rules_path, rule_rows, pretty=pretty)
    _write_json(checks_path, check_rows, pretty=pretty)
    _write_json(graded_rules_path, graded_rule_rows, pretty=pretty)
    rule_library = export_validated_rules(graded_rules_path)
    _write_json(validated_rules_path, asdict(rule_library), pretty=pretty)
    _write_json(contextual_rules_path, contextual_rule_rows, pretty=pretty)
    _write_json(contextual_checks_path, contextual_check_rows, pretty=pretty)
    _write_json(contextual_graded_rules_path, contextual_graded_rule_rows, pretty=pretty)
    contextual_rule_library = export_validated_contextual_rules(contextual_graded_rules_path)
    _write_json(contextual_validated_rules_path, asdict(contextual_rule_library), pretty=pretty)
    _write_json(repairs_path, repair_rows, pretty=pretty)

    manifest = {
        "input": str(input_path.resolve()),
        "outputs": {
            "transitions": str(transitions_path.resolve()),
            "objects": str(objects_path.resolve()),
            "rules": str(rules_path.resolve()),
            "rule_checks": str(checks_path.resolve()),
            "graded_rules": str(graded_rules_path.resolve()),
            "validated_rules": str(validated_rules_path.resolve()),
            "contextual_rules": str(contextual_rules_path.resolve()),
            "contextual_rule_checks": str(contextual_checks_path.resolve()),
            "contextual_graded_rules": str(contextual_graded_rules_path.resolve()),
            "contextual_validated_rules": str(contextual_validated_rules_path.resolve()),
            "repairs": str(repairs_path.resolve()),
        },
        "counts": {
            "transitions": len(trace_rows),
            "objects": len(object_rows),
            "rules": len(rule_rows),
            "candidate_rules": sum(1 for rule in rule_rows if rule.get("status") == "candidate"),
            "rule_checks": len(check_rows),
            "graded_rules": len(graded_rule_rows),
            "validated_rules": sum(1 for rule in graded_rule_rows if rule.get("status") == "validated"),
            "contextual_rules": len(contextual_rule_rows),
            "contextual_rule_checks": len(contextual_check_rows),
            "contextual_graded_rules": len(contextual_graded_rule_rows),
            "contextual_validated_rules": sum(
                1 for rule in contextual_graded_rule_rows if rule.get("status") == "validated"
            ),
            "contextual_rejected_rules": sum(
                1 for rule in contextual_graded_rule_rows if rule.get("status") == "rejected"
            ),
            "contextual_contradictions": sum(
                1 for check in contextual_check_rows if check.get("status") == "contradicted"
            ),
            "rejected_rules": sum(1 for rule in graded_rule_rows if rule.get("status") == "rejected"),
            "contradictions": sum(1 for check in check_rows if check.get("status") == "contradicted"),
            "repairs": len(repair_rows),
        },
        "min_support": min_support,
    }
    _write_json(manifest_path, manifest, pretty=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_pipeline(args.input, args.output_dir, min_support=args.min_support, pretty=args.pretty)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
