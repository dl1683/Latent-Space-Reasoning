import json

from experiments.analyze_diffusion_span_probe_v10_no_lift_specificity import analyze_no_lift_specificity


def test_no_lift_specificity_finds_probe_value_floor(tmp_path):
    replay = tmp_path / "replay.json"
    measurement = tmp_path / "measurement.json"
    replay.write_text(
        json.dumps(
            {
                "row_diagnostics": [
                    _replay_row("pos_low", lift=0.03, label=True),
                    _replay_row("pos_high", lift=0.10, label=True),
                    _replay_row("fp", lift=0.0, label=False),
                ]
            }
        ),
        encoding="utf-8",
    )
    measurement.write_text(
        json.dumps(
            {
                "repair_spend_gate_rows": [
                    _measurement_row("pos_low", value=0.03),
                    _measurement_row("pos_high", value=0.05),
                    _measurement_row("fp", value=0.01),
                ]
            }
        ),
        encoding="utf-8",
    )

    result = analyze_no_lift_specificity(
        replay_path=replay,
        measurement_path=measurement,
        selection_penalty=0.02,
    )

    selected = result["selected_rule"]
    assert selected["false_positive_count"] == 0
    assert selected["false_negative_count"] == 0
    assert selected["selected_count"] == 2
    assert "measured_probe_value_prediction_ge_" in selected["rule_id"]


def _replay_row(task_id, *, lift, label):
    return {
        "candidate_lift_vs_trajectory": lift,
        "label": label,
        "selected": True,
        "source_task_delta_vs_trajectory": 0.0,
        "task_id": task_id,
    }


def _measurement_row(task_id, *, value):
    return {
        "counterfactual_probe_remaining_gap_count": 0,
        "counterfactual_probe_resolved_gap_count": 1,
        "first_repairable_denoise_skeleton_coverage": 0.5,
        "measured_probe_feature_delta": {
            "expected_gap_visibility_gain": 1.0,
            "expected_realization_defect_visibility": 0.2,
            "expected_retention_risk_visibility": 0.9,
            "expected_span_evidence_gain": 0.4,
        },
        "measured_probe_value_prediction": value,
        "prompt_coverage": 0.5,
        "prompt_gap_count": 1,
        "source_quality": 0.2,
        "task_id": task_id,
    }
