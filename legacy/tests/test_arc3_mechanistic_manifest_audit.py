import json

from experiments.audit_arc3_mechanistic_manifest import audit_manifest


def _write_pipeline_outputs(root):
    (root / "transitions.jsonl").write_text(json.dumps({"step_index": 0}) + "\n", encoding="utf-8")
    (root / "objects.json").write_text(json.dumps([{"object_id": "agent"}]), encoding="utf-8")
    (root / "rules.json").write_text(json.dumps([{"rule_id": "rule-1", "status": "candidate"}]), encoding="utf-8")
    (root / "rule_checks.json").write_text(
        json.dumps([{"rule_id": "rule-1", "status": "contradicted"}]), encoding="utf-8"
    )
    (root / "graded_rules.json").write_text(
        json.dumps([{"rule_id": "rule-1", "status": "rejected"}]), encoding="utf-8"
    )
    (root / "validated_rules.json").write_text(
        json.dumps({"validated_rules": [], "excluded_counts": {"rejected": 1}}), encoding="utf-8"
    )
    (root / "contextual_rules.json").write_text(
        json.dumps([{"rule_id": "context-rule-1", "status": "candidate"}]), encoding="utf-8"
    )
    (root / "contextual_rule_checks.json").write_text(
        json.dumps([{"rule_id": "context-rule-1", "status": "supported"}]), encoding="utf-8"
    )
    (root / "contextual_graded_rules.json").write_text(
        json.dumps([{"rule_id": "context-rule-1", "status": "validated"}]), encoding="utf-8"
    )
    (root / "contextual_validated_rules.json").write_text(
        json.dumps({"contextual_rules": [{"rule_id": "context-rule-1"}], "excluded_counts": {}}), encoding="utf-8"
    )
    (root / "repairs.json").write_text(json.dumps([{"repair_id": "r", "rule_id": "rule-1"}]), encoding="utf-8")


def test_audits_valid_manifest(tmp_path):
    _write_pipeline_outputs(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "outputs": {
                    "transitions": "transitions.jsonl",
                    "objects": "objects.json",
                    "rules": "rules.json",
                    "rule_checks": "rule_checks.json",
                    "graded_rules": "graded_rules.json",
                    "validated_rules": "validated_rules.json",
                    "contextual_rules": "contextual_rules.json",
                    "contextual_rule_checks": "contextual_rule_checks.json",
                    "contextual_graded_rules": "contextual_graded_rules.json",
                    "contextual_validated_rules": "contextual_validated_rules.json",
                    "repairs": "repairs.json",
                },
                "counts": {
                    "transitions": 1,
                    "objects": 1,
                    "rules": 1,
                    "candidate_rules": 1,
                    "rule_checks": 1,
                    "graded_rules": 1,
                    "validated_rules": 0,
                    "contextual_rules": 1,
                    "contextual_rule_checks": 1,
                    "contextual_graded_rules": 1,
                    "contextual_validated_rules": 1,
                    "contextual_rejected_rules": 0,
                    "contextual_contradictions": 0,
                    "rejected_rules": 1,
                    "contradictions": 1,
                    "repairs": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    findings = audit_manifest(manifest)

    assert findings
    assert all(finding.status == "pass" for finding in findings)


def test_fails_when_manifest_count_does_not_match_output(tmp_path):
    _write_pipeline_outputs(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "outputs": {
                    "transitions": "transitions.jsonl",
                    "objects": "objects.json",
                    "rules": "rules.json",
                    "rule_checks": "rule_checks.json",
                    "graded_rules": "graded_rules.json",
                    "validated_rules": "validated_rules.json",
                    "contextual_rules": "contextual_rules.json",
                    "contextual_rule_checks": "contextual_rule_checks.json",
                    "contextual_graded_rules": "contextual_graded_rules.json",
                    "contextual_validated_rules": "contextual_validated_rules.json",
                    "repairs": "repairs.json",
                },
                "counts": {
                    "transitions": 2,
                    "objects": 1,
                    "rules": 1,
                    "candidate_rules": 1,
                    "rule_checks": 1,
                    "graded_rules": 1,
                    "validated_rules": 0,
                    "contextual_rules": 1,
                    "contextual_rule_checks": 1,
                    "contextual_graded_rules": 1,
                    "contextual_validated_rules": 1,
                    "contextual_rejected_rules": 0,
                    "contextual_contradictions": 0,
                    "rejected_rules": 1,
                    "contradictions": 1,
                    "repairs": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    findings = audit_manifest(manifest)

    failures = [finding for finding in findings if finding.status == "fail"]
    assert len(failures) == 1
    assert failures[0].field == "counts.transitions"


def test_fails_when_rule_check_references_missing_candidate(tmp_path):
    _write_pipeline_outputs(tmp_path)
    (tmp_path / "rule_checks.json").write_text(
        json.dumps([{"rule_id": "missing-rule", "status": "contradicted"}]), encoding="utf-8"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "outputs": {
                    "transitions": "transitions.jsonl",
                    "objects": "objects.json",
                    "rules": "rules.json",
                    "rule_checks": "rule_checks.json",
                    "graded_rules": "graded_rules.json",
                    "validated_rules": "validated_rules.json",
                    "contextual_rules": "contextual_rules.json",
                    "contextual_rule_checks": "contextual_rule_checks.json",
                    "contextual_graded_rules": "contextual_graded_rules.json",
                    "contextual_validated_rules": "contextual_validated_rules.json",
                    "repairs": "repairs.json",
                },
                "counts": {
                    "transitions": 1,
                    "objects": 1,
                    "rules": 1,
                    "candidate_rules": 1,
                    "rule_checks": 1,
                    "graded_rules": 1,
                    "validated_rules": 0,
                    "contextual_rules": 1,
                    "contextual_rule_checks": 1,
                    "contextual_graded_rules": 1,
                    "contextual_validated_rules": 1,
                    "contextual_rejected_rules": 0,
                    "contextual_contradictions": 0,
                    "rejected_rules": 1,
                    "contradictions": 1,
                    "repairs": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    findings = audit_manifest(manifest)

    assert any(finding.field == "links.rule_checks_to_candidate_rules" and finding.status == "fail" for finding in findings)
