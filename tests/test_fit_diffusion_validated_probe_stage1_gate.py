import json

from experiments.fit_diffusion_validated_probe_stage1_gate import (
    fit_validated_probe_stage1_gate,
    render_markdown,
)


def test_validated_probe_stage1_gate_counts_invalid_positive_misses(tmp_path):
    targets = tmp_path / "targets.json"
    scores = tmp_path / "scores.json"
    targets.write_text(
        json.dumps(
            {
                "rows": [
                    _target_row("plan_a", label=True, lift=0.20),
                    _target_row("plan_b", label=True, lift=0.10),
                    _target_row("plan_c", label=False, lift=-0.05),
                    _target_row("plan_d", label=False, lift=-0.03),
                ]
            }
        ),
        encoding="utf-8",
    )
    scores.write_text(
        json.dumps(
            {
                "counterfactual_probe_policy": "strict_tomography_probe_v1",
                "repair_spend_gate_rows": [
                    _gate_row("plan_a", measured_value=0.9, valid=False),
                    _gate_row("plan_b", measured_value=0.8, valid=True),
                    _gate_row("plan_c", measured_value=0.7, valid=True),
                    _gate_row("plan_d", measured_value=0.1, valid=True),
                ],
            }
        ),
        encoding="utf-8",
    )

    fit = fit_validated_probe_stage1_gate(scores_path=scores, targets_path=targets)
    markdown = render_markdown(fit)

    assert fit["schema"] == "diffusion_counterfactual_validated_probe_stage1_gate.v1"
    assert fit["summary"]["row_count"] == 4
    assert fit["summary"]["valid_probe_count"] == 3
    assert fit["summary"]["invalid_positive_count"] == 1
    assert fit["summary"]["invalid_positive_lift"] == 0.20
    assert fit["summary"]["best_validated_error_count"] == 1
    assert fit["summary"]["gate_decision"] == "diagnostic_only"
    assert fit["rows"][0]["selected_by_validated_rule"] is False
    assert "Validated Probe Stage 1 Gate" in markdown


def _target_row(task_id, *, label, lift):
    return {
        "labels": {
            "candidate_lift_vs_trajectory": lift,
            "promote_vs_trajectory": label,
        },
        "task_id": task_id,
    }


def _gate_row(task_id, *, measured_value, valid):
    return {
        "counterfactual_probe_observation": "measured_generation",
        "counterfactual_probe_text_slot_count": 3,
        "counterfactual_probe_text_valid_for_stage1": valid,
        "measured_probe_feature_delta": {
            "expected_gap_visibility_gain": measured_value,
            "expected_realization_defect_visibility": measured_value / 2,
            "expected_retention_risk_visibility": 1.0 - measured_value / 10,
            "expected_span_evidence_gain": measured_value / 3,
        },
        "measured_probe_value_prediction": measured_value,
        "prompt_gap_count": 4,
        "source_quality": 0.2,
        "task_id": task_id,
        "would_probe": True,
    }
