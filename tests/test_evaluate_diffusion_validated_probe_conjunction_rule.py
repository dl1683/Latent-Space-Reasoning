import json

from experiments.evaluate_diffusion_validated_probe_conjunction_rule import (
    evaluate_validated_probe_conjunction_rule,
    parse_conditions,
    render_markdown,
)


def test_validated_probe_conjunction_rule_applies_frozen_conditions(tmp_path):
    targets = tmp_path / "targets.json"
    scores = tmp_path / "scores.json"
    targets.write_text(
        json.dumps(
            {
                "rows": [
                    _target_row("plan_a", label=True, lift=0.2),
                    _target_row("plan_b", label=False, lift=-0.1),
                    _target_row("plan_c", label=True, lift=0.1),
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
                    _gate_row("plan_a", gap=0.8, span=0.4, valid=True),
                    _gate_row("plan_b", gap=0.8, span=0.8, valid=True),
                    _gate_row("plan_c", gap=0.8, span=0.4, valid=False),
                ],
            }
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_validated_probe_conjunction_rule(
        conditions=parse_conditions(
            "measured_expected_gap_visibility_gain:ge:0.666667,"
            "measured_expected_span_evidence_gain:le:0.600000"
        ),
        require_valid_probe=True,
        scores_path=scores,
        targets_path=targets,
        text_fidelity_path=None,
    )
    markdown = render_markdown(evaluation)

    assert evaluation["schema"] == "diffusion_counterfactual_validated_probe_conjunction_rule_eval.v1"
    assert evaluation["summary"]["selected_count"] == 1
    assert evaluation["summary"]["false_negative_task_ids"] == ["plan_c"]
    assert evaluation["summary"]["false_positive_count"] == 0
    assert "not refit" in markdown


def _target_row(task_id, *, label, lift):
    return {
        "labels": {
            "candidate_lift_vs_trajectory": lift,
            "promote_vs_trajectory": label,
        },
        "task_id": task_id,
    }


def _gate_row(task_id, *, gap, span, valid):
    return {
        "counterfactual_probe_observation": "measured_generation",
        "counterfactual_probe_text_slot_count": 3,
        "counterfactual_probe_text_valid_for_stage1": valid,
        "measured_probe_feature_delta": {
            "expected_gap_visibility_gain": gap,
            "expected_realization_defect_visibility": 0.0,
            "expected_retention_risk_visibility": 0.0,
            "expected_span_evidence_gain": span,
        },
        "measured_probe_value_prediction": gap,
        "prompt_gap_count": 4,
        "source_quality": 0.2,
        "task_id": task_id,
        "would_probe": True,
    }
