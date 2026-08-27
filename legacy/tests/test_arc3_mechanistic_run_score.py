import json

from experiments.score_arc3_mechanistic_run import score_run


def _write_outputs(root, validated_status="validated"):
    (root / "transitions.jsonl").write_text(json.dumps({"step_index": 0}) + "\n", encoding="utf-8")
    (root / "objects.json").write_text(json.dumps([{"object_id": "agent"}]), encoding="utf-8")
    (root / "rules.json").write_text(json.dumps([{"rule_id": "rule-1", "status": "candidate"}]), encoding="utf-8")
    (root / "rule_checks.json").write_text(
        json.dumps([{"rule_id": "rule-1", "status": "supported"}]), encoding="utf-8"
    )
    (root / "graded_rules.json").write_text(
        json.dumps([{"rule_id": "rule-1", "status": validated_status}]), encoding="utf-8"
    )
    (root / "validated_rules.json").write_text(
        json.dumps({"validated_rules": [{"rule_id": "rule-1"}], "excluded_counts": {}}), encoding="utf-8"
    )
    (root / "contextual_rules.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "contextual_rule_checks.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "contextual_graded_rules.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "contextual_validated_rules.json").write_text(
        json.dumps({"contextual_rules": [], "excluded_counts": {}}), encoding="utf-8"
    )
    (root / "repairs.json").write_text(json.dumps([]), encoding="utf-8")


def test_scores_reusable_run(tmp_path):
    _write_outputs(tmp_path)
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
                    "validated_rules": 1,
                    "contextual_rules": 0,
                    "contextual_rule_checks": 0,
                    "contextual_graded_rules": 0,
                    "contextual_validated_rules": 0,
                    "contextual_rejected_rules": 0,
                    "contextual_contradictions": 0,
                    "rejected_rules": 0,
                    "contradictions": 0,
                    "repairs": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    score = score_run(manifest)

    assert score.audit_passed is True
    assert score.status == "reusable"
    assert score.reuse_ratio == 1.0
    assert score.contradiction_rate == 0.0
    assert score.contextual_contradiction_rate == 0.0
    assert score.contextual_validated_rules == 0


def test_scores_invalid_run_when_audit_fails(tmp_path):
    _write_outputs(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "outputs": {
                    "transitions": "missing.jsonl",
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
                    "validated_rules": 1,
                    "contextual_rules": 0,
                    "contextual_rule_checks": 0,
                    "contextual_graded_rules": 0,
                    "contextual_validated_rules": 0,
                    "contextual_rejected_rules": 0,
                    "contextual_contradictions": 0,
                    "rejected_rules": 0,
                    "contradictions": 0,
                    "repairs": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    score = score_run(manifest)

    assert score.audit_passed is False
    assert score.status == "invalid"
    assert score.blockers
