import json

from experiments.fit_diffusion_measured_probe_value_policy import (
    fit_measured_probe_value_policy,
    render_markdown,
)


def test_measured_probe_value_policy_keeps_stage0_features_separate(tmp_path):
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
                "counterfactual_probe_generation_count": 4,
                "repair_spend_gate_rows": [
                    _gate_row("plan_a", measured_value=0.9, prompt_gap=4, would_probe=True),
                    _gate_row("plan_b", measured_value=0.7, prompt_gap=4, would_probe=True),
                    _gate_row("plan_c", measured_value=0.8, prompt_gap=10, would_probe=False),
                    _gate_row("plan_d", measured_value=0.6, prompt_gap=10, would_probe=False),
                ],
            }
        ),
        encoding="utf-8",
    )

    fit = fit_measured_probe_value_policy(scores_path=scores, targets_path=targets)
    markdown = render_markdown(fit)

    assert fit["schema"] == "diffusion_counterfactual_measured_probe_value_policy.v1"
    assert fit["summary"]["row_count"] == 4
    assert fit["summary"]["counterfactual_probe_generation_count"] == 4
    assert fit["summary"]["best_measured_only_error_count"] == 1
    assert fit["summary"]["best_all_error_count"] == 0
    assert fit["summary"]["gate_decision"] == "diagnostic_only"
    assert fit["summary"]["best_all_rule_name"] in {
        "prompt_gap_count_le_4p000000",
        "would_probe_score_ge_1p000000",
    }
    assert "Measured Probe Value Policy" in markdown


def _target_row(task_id, *, label, lift):
    return {
        "labels": {
            "candidate_lift_vs_trajectory": lift,
            "promote_vs_trajectory": label,
        },
        "task_id": task_id,
    }


def _gate_row(task_id, *, measured_value, prompt_gap, would_probe):
    return {
        "counterfactual_probe_observation": "measured_generation",
        "measured_probe_feature_delta": {
            "expected_gap_visibility_gain": measured_value / 2,
            "expected_realization_defect_visibility": measured_value / 3,
            "expected_retention_risk_visibility": measured_value / 4,
            "expected_span_evidence_gain": measured_value / 5,
        },
        "measured_probe_value_prediction": measured_value,
        "prompt_gap_count": prompt_gap,
        "source_quality": 0.2,
        "task_id": task_id,
        "would_probe": would_probe,
    }
