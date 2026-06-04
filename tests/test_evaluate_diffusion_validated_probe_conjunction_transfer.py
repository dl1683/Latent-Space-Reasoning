import json

from experiments.evaluate_diffusion_validated_probe_conjunction_transfer import (
    evaluate_validated_probe_conjunction_transfer,
    render_markdown,
)


def test_probe_conjunction_transfer_separates_train_rank_from_screened_challenger(tmp_path):
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
                    _target_row("train_c", label=True, lift=0.10),
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
                    _gate_row("train_a", gap=0.90, span=0.40, retention=0.90),
                    _gate_row("train_b", gap=0.90, span=0.80, retention=0.10),
                    _gate_row("train_c", gap=0.20, span=0.20, retention=0.80),
                ],
            }
        ),
        encoding="utf-8",
    )
    test_targets.write_text(
        json.dumps(
            {
                "rows": [
                    _target_row("test_a", label=True, lift=0.20),
                    _target_row("test_b", label=False, lift=-0.05),
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
                    _gate_row("test_a", gap=0.90, span=0.40, retention=0.10),
                    _gate_row("test_b", gap=0.90, span=0.80, retention=0.90),
                ],
            }
        ),
        encoding="utf-8",
    )

    fit = evaluate_validated_probe_conjunction_transfer(
        max_conditions=2,
        test_scores_path=test_scores,
        test_targets_path=test_targets,
        test_text_fidelity_path=None,
        top_n=5,
        train_scores_path=train_scores,
        train_targets_path=train_targets,
        train_text_fidelity_path=None,
    )
    markdown = render_markdown(fit)

    assert fit["schema"] == "diffusion_counterfactual_validated_probe_conjunction_transfer.v1"
    assert fit["summary"]["best_train_error_count"] == 0
    assert fit["summary"]["best_transfer_screened_test_error_count"] == 0
    assert fit["summary"]["best_transfer_screened_rule_id"]
    assert fit["summary"]["gate_decision"] == "diagnostic_only_transfer_screened_challenger"
    assert "transfer-screened rules are diagnostic challengers only" in markdown


def _target_row(task_id, *, label, lift):
    return {
        "labels": {
            "candidate_lift_vs_trajectory": lift,
            "promote_vs_trajectory": label,
        },
        "task_id": task_id,
    }


def _gate_row(task_id, *, gap, span, retention):
    return {
        "counterfactual_probe_observation": "measured_generation",
        "counterfactual_probe_text_slot_count": 3,
        "counterfactual_probe_text_valid_for_stage1": True,
        "measured_probe_feature_delta": {
            "expected_gap_visibility_gain": gap,
            "expected_realization_defect_visibility": 0.0,
            "expected_retention_risk_visibility": retention,
            "expected_span_evidence_gain": span,
        },
        "measured_probe_value_prediction": gap,
        "prompt_gap_count": 4,
        "source_quality": 0.2,
        "task_id": task_id,
        "would_probe": True,
    }
