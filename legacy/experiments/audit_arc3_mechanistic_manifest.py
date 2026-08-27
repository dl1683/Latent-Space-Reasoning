"""Audit an ARC-3 mechanistic pipeline manifest.

This checks that the offline pipeline outputs are present, parseable, and
consistent with the manifest counts. It is a lightweight gate for treating a
pipeline run as evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditFinding:
    field: str
    status: str
    detail: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row is not an object in {path}")
            rows.append(payload)
    return rows


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def audit_manifest(manifest_path: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        return [AuditFinding("manifest", "fail", "manifest root is not a JSON object")]

    outputs = manifest.get("outputs")
    counts = manifest.get("counts")
    if not isinstance(outputs, dict):
        findings.append(AuditFinding("outputs", "fail", "missing outputs object"))
        outputs = {}
    if not isinstance(counts, dict):
        findings.append(AuditFinding("counts", "fail", "missing counts object"))
        counts = {}

    loaded: dict[str, list[dict[str, Any]]] = {}
    loaded_objects: dict[str, dict[str, Any]] = {}
    for name in (
        "transitions",
        "objects",
        "rules",
        "rule_checks",
        "graded_rules",
        "validated_rules",
        "contextual_rules",
        "contextual_rule_checks",
        "contextual_graded_rules",
        "contextual_validated_rules",
        "repairs",
    ):
        value = outputs.get(name)
        if not isinstance(value, str):
            findings.append(AuditFinding(name, "fail", "missing output path"))
            continue
        path = _resolve_manifest_path(manifest_path, value)
        if not path.exists():
            findings.append(AuditFinding(name, "fail", f"missing file: {path}"))
            continue
        try:
            rows = _load_jsonl(path) if path.suffix == ".jsonl" else _load_json(path)
        except Exception as exc:  # noqa: BLE001 - report parse failures without hiding context.
            findings.append(AuditFinding(name, "fail", f"cannot parse {path}: {exc}"))
            continue
        if name in {"validated_rules", "contextual_validated_rules"}:
            if not isinstance(rows, dict):
                findings.append(AuditFinding(name, "fail", f"output is not an object: {path}"))
                continue
            rules_field = "contextual_rules" if name == "contextual_validated_rules" else "validated_rules"
            if not isinstance(rows.get(rules_field), list):
                findings.append(AuditFinding(name, "fail", f"{rules_field} field is not a list: {path}"))
                continue
            loaded_objects[name] = rows
            findings.append(AuditFinding(name, "pass", f"loaded {len(rows[rules_field])} validated rules"))
            continue
        if not isinstance(rows, list):
            findings.append(AuditFinding(name, "fail", f"output is not a list: {path}"))
            continue
        if not all(isinstance(row, dict) for row in rows):
            findings.append(AuditFinding(name, "fail", f"output rows are not all objects: {path}"))
            continue
        loaded[name] = rows
        findings.append(AuditFinding(name, "pass", f"loaded {len(rows)} rows"))

    count_fields = {
        "transitions": "transitions",
        "objects": "objects",
        "rules": "rules",
        "rule_checks": "rule_checks",
        "graded_rules": "graded_rules",
        "contextual_rules": "contextual_rules",
        "contextual_rule_checks": "contextual_rule_checks",
        "contextual_graded_rules": "contextual_graded_rules",
        "repairs": "repairs",
    }
    for count_name, output_name in count_fields.items():
        if output_name not in loaded:
            continue
        expected = counts.get(count_name)
        actual = len(loaded[output_name])
        status = "pass" if expected == actual else "fail"
        findings.append(AuditFinding(f"counts.{count_name}", status, f"expected {expected}, actual {actual}"))

    if "rules" in loaded:
        candidate_rules = sum(1 for row in loaded["rules"] if row.get("status") == "candidate")
        expected = counts.get("candidate_rules")
        status = "pass" if expected == candidate_rules else "fail"
        findings.append(
            AuditFinding("counts.candidate_rules", status, f"expected {expected}, actual {candidate_rules}")
        )

    if "rule_checks" in loaded:
        contradictions = sum(1 for row in loaded["rule_checks"] if row.get("status") == "contradicted")
        expected = counts.get("contradictions")
        status = "pass" if expected == contradictions else "fail"
        findings.append(AuditFinding("counts.contradictions", status, f"expected {expected}, actual {contradictions}"))

    if "contextual_rule_checks" in loaded:
        contradictions = sum(
            1 for row in loaded["contextual_rule_checks"] if row.get("status") == "contradicted"
        )
        expected = counts.get("contextual_contradictions")
        status = "pass" if expected == contradictions else "fail"
        findings.append(
            AuditFinding(
                "counts.contextual_contradictions",
                status,
                f"expected {expected}, actual {contradictions}",
            )
        )

    if "graded_rules" in loaded:
        for status_name, count_name in (("validated", "validated_rules"), ("rejected", "rejected_rules")):
            actual = sum(1 for row in loaded["graded_rules"] if row.get("status") == status_name)
            expected = counts.get(count_name)
            status = "pass" if expected == actual else "fail"
            findings.append(AuditFinding(f"counts.{count_name}", status, f"expected {expected}, actual {actual}"))

    if "contextual_graded_rules" in loaded:
        for status_name, count_name in (
            ("validated", "contextual_validated_rules"),
            ("rejected", "contextual_rejected_rules"),
        ):
            actual = sum(1 for row in loaded["contextual_graded_rules"] if row.get("status") == status_name)
            expected = counts.get(count_name)
            status = "pass" if expected == actual else "fail"
            findings.append(AuditFinding(f"counts.{count_name}", status, f"expected {expected}, actual {actual}"))

    if "rules" in loaded and "rule_checks" in loaded:
        candidate_rule_ids = {str(row.get("rule_id", "")) for row in loaded["rules"] if row.get("status") == "candidate"}
        checked_rule_ids = {str(row.get("rule_id", "")) for row in loaded["rule_checks"]}
        missing = sorted(rule_id for rule_id in checked_rule_ids if rule_id not in candidate_rule_ids)
        status = "pass" if not missing else "fail"
        detail = "all checked rules are candidates" if not missing else f"non-candidate or missing rule checks: {missing}"
        findings.append(AuditFinding("links.rule_checks_to_candidate_rules", status, detail))

    if "contextual_rules" in loaded and "contextual_rule_checks" in loaded:
        candidate_rule_ids = {
            str(row.get("rule_id", "")) for row in loaded["contextual_rules"] if row.get("status") == "candidate"
        }
        checked_rule_ids = {str(row.get("rule_id", "")) for row in loaded["contextual_rule_checks"]}
        missing = sorted(rule_id for rule_id in checked_rule_ids if rule_id not in candidate_rule_ids)
        status = "pass" if not missing else "fail"
        detail = (
            "all contextual checks are candidate contextual rules"
            if not missing
            else f"non-candidate or missing contextual checks: {missing}"
        )
        findings.append(AuditFinding("links.contextual_checks_to_contextual_rules", status, detail))

    if "rule_checks" in loaded and "repairs" in loaded:
        contradicted_rule_ids = {
            str(row.get("rule_id", "")) for row in loaded["rule_checks"] if row.get("status") == "contradicted"
        }
        repair_rule_ids = {str(row.get("rule_id", "")) for row in loaded["repairs"]}
        missing = sorted(rule_id for rule_id in repair_rule_ids if rule_id not in contradicted_rule_ids)
        status = "pass" if not missing else "fail"
        detail = "all repairs map to contradicted rules" if not missing else f"repairs without contradiction: {missing}"
        findings.append(AuditFinding("links.repairs_to_contradictions", status, detail))

    if "rules" in loaded and "graded_rules" in loaded:
        rule_ids = {str(row.get("rule_id", "")) for row in loaded["rules"]}
        graded_rule_ids = {str(row.get("rule_id", "")) for row in loaded["graded_rules"]}
        missing = sorted(rule_id for rule_id in graded_rule_ids if rule_id not in rule_ids)
        status = "pass" if not missing else "fail"
        detail = "all graded rules map to source rules" if not missing else f"graded rules without source rule: {missing}"
        findings.append(AuditFinding("links.graded_rules_to_rules", status, detail))

    if "contextual_rules" in loaded and "contextual_graded_rules" in loaded:
        rule_ids = {str(row.get("rule_id", "")) for row in loaded["contextual_rules"]}
        graded_rule_ids = {str(row.get("rule_id", "")) for row in loaded["contextual_graded_rules"]}
        missing = sorted(rule_id for rule_id in graded_rule_ids if rule_id not in rule_ids)
        status = "pass" if not missing else "fail"
        detail = (
            "all contextual graded rules map to source contextual rules"
            if not missing
            else f"contextual graded rules without source rule: {missing}"
        )
        findings.append(AuditFinding("links.contextual_graded_rules_to_contextual_rules", status, detail))

    if "graded_rules" in loaded and "validated_rules" in loaded_objects:
        validated_from_grades = {
            str(row.get("rule_id", "")) for row in loaded["graded_rules"] if row.get("status") == "validated"
        }
        exported = loaded_objects["validated_rules"].get("validated_rules", [])
        exported_rule_ids = {
            str(row.get("rule_id", ""))
            for row in exported
            if isinstance(row, dict)
        }
        missing = sorted(rule_id for rule_id in exported_rule_ids if rule_id not in validated_from_grades)
        omitted = sorted(rule_id for rule_id in validated_from_grades if rule_id not in exported_rule_ids)
        status = "pass" if not missing and not omitted else "fail"
        if status == "pass":
            detail = "validated rule library matches graded validated rules"
        else:
            detail = f"missing from grades: {missing}; omitted from library: {omitted}"
        findings.append(AuditFinding("links.validated_library_to_graded_rules", status, detail))

    if "contextual_graded_rules" in loaded and "contextual_validated_rules" in loaded_objects:
        validated_from_grades = {
            str(row.get("rule_id", "")) for row in loaded["contextual_graded_rules"] if row.get("status") == "validated"
        }
        exported = loaded_objects["contextual_validated_rules"].get("contextual_rules", [])
        exported_rule_ids = {
            str(row.get("rule_id", ""))
            for row in exported
            if isinstance(row, dict)
        }
        missing = sorted(rule_id for rule_id in exported_rule_ids if rule_id not in validated_from_grades)
        omitted = sorted(rule_id for rule_id in validated_from_grades if rule_id not in exported_rule_ids)
        status = "pass" if not missing and not omitted else "fail"
        if status == "pass":
            detail = "contextual validated rule library matches graded contextual validated rules"
        else:
            detail = f"missing from grades: {missing}; omitted from library: {omitted}"
        findings.append(AuditFinding("links.contextual_validated_library_to_graded_rules", status, detail))

    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    findings = audit_manifest(args.manifest)
    payload = [asdict(finding) for finding in findings]
    text = json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    if any(finding.status == "fail" for finding in findings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
