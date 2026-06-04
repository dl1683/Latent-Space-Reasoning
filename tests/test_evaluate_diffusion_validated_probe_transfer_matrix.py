import json

from experiments.evaluate_diffusion_validated_probe_transfer_matrix import (
    evaluate_validated_probe_transfer_matrix,
    render_markdown,
)


def test_validated_probe_transfer_matrix_keeps_train_fit_and_transfer_apart(tmp_path):
    train_targets = tmp_path / "train_targets.json"
    train_scores = tmp_path / "train_scores.json"
    test_targets = tmp_path / "test_targets.json"
    test_scores = tmp_path / "test_scores.json"
    train_targets.write_text(
        json.dumps(
            {
                "rows": [
                    _target_row("train_a", label=True, lift=0.20),
                    _target_row("train_b", label=False, lift=-0.05),
                ]
            }
        ),
        encoding="utf-8",
    )
    train_scores.write_text(
        json.dumps(
            {
                "counterfactual_probe_policy": "span_tomography_probe_v4",
                "repair_spend_gate_rows": [
                    _gate_row("train_a", retention_risk=0.95, valid=True),
                    _gate_row("train_b", retention_risk=0.70, valid=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    test_targets.write_text(
        json.dumps(
            {
                "rows": [
                    _target_row("test_a", label=True, lift=0.15),
                    _target_row("test_b", label=False, lift=-0.02),
                ]
            }
        ),
        encoding="utf-8",
    )
    test_scores.write_text(
        json.dumps(
            {
                "counterfactual_probe_policy": "span_tomography_probe_v4",
                "repair_spend_gate_rows": [
                    _gate_row("test_a", retention_risk=0.60, valid=True),
                    _gate_row("test_b", retention_risk=0.99, valid=True),
                ],
            }
        ),
        encoding="utf-8",
    )

    matrix = evaluate_validated_probe_transfer_matrix(
        test_scores_path=test_scores,
        test_targets_path=test_targets,
        test_text_fidelity_path=None,
        top_n=3,
        train_scores_path=train_scores,
        train_targets_path=train_targets,
        train_text_fidelity_path=None,
    )
    markdown = render_markdown(matrix)

    assert matrix["schema"] == "diffusion_counterfactual_validated_probe_transfer_matrix.v1"
    assert matrix["summary"]["best_train_error_count"] == 0
    assert matrix["summary"]["best_train_transfer_error_count"] == 2
    assert matrix["summary"]["transfer_gate_decision"] == "diagnostic_only_transfer_failed"
    assert matrix["summary"]["best_test_error_count"] == 0
    assert "Fresh-Only Diagnostic Upper Bound" in markdown


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
            "expected_gap_visibility_gain": retention_risk,
            "expected_realization_defect_visibility": retention_risk / 2,
            "expected_retention_risk_visibility": retention_risk,
            "expected_span_evidence_gain": retention_risk / 3,
        },
        "measured_probe_value_prediction": retention_risk,
        "prompt_gap_count": 4,
        "source_quality": 0.2,
        "task_id": task_id,
        "would_probe": True,
    }
