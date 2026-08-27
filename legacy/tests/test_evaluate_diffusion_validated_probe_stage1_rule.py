import json

from experiments.evaluate_diffusion_validated_probe_stage1_rule import (
    evaluate_validated_probe_stage1_rule,
    render_markdown,
)


def test_validated_probe_stage1_rule_evaluates_frozen_threshold(tmp_path):
    targets = tmp_path / "targets.json"
    scores = tmp_path / "scores.json"
    text_fidelity = tmp_path / "text_fidelity.json"
    targets.write_text(
        json.dumps(
            {
                "rows": [
                    _target_row("plan_a", label=True, lift=0.20),
                    _target_row("plan_b", label=False, lift=-0.05),
                    _target_row("plan_c", label=True, lift=0.10),
                ]
            }
        ),
        encoding="utf-8",
    )
    scores.write_text(
        json.dumps(
            {
                "counterfactual_probe_policy": "span_tomography_probe_v4",
                "repair_spend_gate_rows": [
                    _gate_row("plan_a", retention_risk=0.95, valid=True),
                    _gate_row("plan_b", retention_risk=0.80, valid=True),
                    _gate_row("plan_c", retention_risk=0.99, valid=False),
                ],
            }
        ),
        encoding="utf-8",
    )
    text_fidelity.write_text(
        json.dumps(
            {
                "rows": [
                    _text_fidelity_row("plan_a", semantic_valid=True, x0_x2_overlap=0.0),
                    _text_fidelity_row("plan_b", semantic_valid=True, x0_x2_overlap=0.0),
                    _text_fidelity_row("plan_c", semantic_valid=True, x0_x2_overlap=0.0),
                ]
            }
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_validated_probe_stage1_rule(
        direction="ge",
        feature="measured_distinct_retention_risk_visibility",
        require_valid_probe=True,
        scores_path=scores,
        targets_path=targets,
        text_fidelity_path=text_fidelity,
        threshold=0.90,
    )
    markdown = render_markdown(evaluation)

    assert evaluation["schema"] == "diffusion_counterfactual_validated_probe_stage1_rule_eval.v1"
    assert evaluation["summary"]["row_count"] == 3
    assert evaluation["summary"]["selected_count"] == 1
    assert evaluation["summary"]["error_count"] == 1
    assert evaluation["summary"]["false_negative_task_ids"] == ["plan_c"]
    assert evaluation["rows"][0]["selected_by_fixed_rule"] is True
    assert evaluation["rows"][2]["selected_by_fixed_rule"] is False
    assert "threshold is not refit" in markdown


def _target_row(task_id, *, label, lift):
    return {
        "labels": {
            "candidate_lift_vs_trajectory": lift,
            "promote_vs_trajectory": label,
        },
        "task_id": task_id,
    }


def _gate_row(task_id, *, retention_risk, valid):
    return {
        "counterfactual_probe_observation": "measured_generation",
        "counterfactual_probe_text_slot_count": 3,
        "counterfactual_probe_text_valid_for_stage1": valid,
        "measured_probe_feature_delta": {
            "expected_gap_visibility_gain": 0.5,
            "expected_realization_defect_visibility": 0.5,
            "expected_retention_risk_visibility": retention_risk,
            "expected_span_evidence_gain": 0.5,
        },
        "measured_probe_value_prediction": retention_risk,
        "prompt_gap_count": 4,
        "source_quality": 0.2,
        "task_id": task_id,
        "would_probe": True,
    }


def _text_fidelity_row(task_id, *, semantic_valid, x0_x2_overlap):
    return {
        "features": {
            "duplicate_authorization": 0.0,
            "duplicate_slot_key": 0.0,
            "malformed_compact_key": 0.0,
            "max_slot_overlap": x0_x2_overlap,
            "repeated_token_excess": 0.0,
            "semantic_defect": 0.0 if semantic_valid else 1.0,
            "semantic_valid_for_stage1": 1.0 if semantic_valid else 0.0,
            "template_slot_echo": 0.0,
            "x0_x2_slot_overlap": x0_x2_overlap,
        },
        "task_id": task_id,
    }
