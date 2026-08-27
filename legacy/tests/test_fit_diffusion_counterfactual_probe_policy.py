import json

from experiments.fit_diffusion_counterfactual_probe_policy import (
    fit_counterfactual_probe_policy,
    render_markdown,
)


def test_counterfactual_probe_policy_fit_ranks_offline_threshold_rules(tmp_path):
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "rows": [
                    _row("plan_a", label=True, lift=0.20, score=0.9),
                    _row("plan_b", label=True, lift=0.10, score=0.8),
                    _row("plan_c", label=False, lift=-0.05, score=0.2),
                    _row("plan_d", label=False, lift=-0.03, score=0.1),
                ]
            }
        ),
        encoding="utf-8",
    )

    fit = fit_counterfactual_probe_policy(targets)
    markdown = render_markdown(fit)

    assert fit["schema"] == "diffusion_counterfactual_probe_policy_fit.v1"
    assert fit["summary"]["row_count"] == 4
    assert fit["summary"]["best_error_count"] == 0
    assert fit["summary"]["gate_decision"] == "diagnostic_only"
    assert fit["summary"]["best_rule_name"].startswith("probe_value_prediction_ge_")
    assert [row["selected_by_best_rule"] for row in fit["rows"]] == [True, True, False, False]
    assert "Counterfactual Probe Policy Fit" in markdown


def _row(task_id, *, label, lift, score):
    return {
        "counterexample_type": "false_negative" if label else "false_positive",
        "labels": {
            "candidate_lift_vs_trajectory": lift,
            "promote_vs_trajectory": label,
        },
        "pre_probe_features": {
            "degeneracy_score": 0.2,
            "first_repairable_step": 4,
            "max_span_target_score": 2.0,
            "min_span_source_relative_preservation": 0.8,
            "prompt_gap_count": 4,
            "source_quality": 0.2,
            "source_task_delta_vs_trajectory": 0.0,
        },
        "probe_feature_delta": {
            "expected_gap_visibility_gain": 4 / 12,
            "expected_realization_defect_visibility": 0.2,
            "expected_retention_risk_visibility": 0.2,
            "expected_span_evidence_gain": 0.8,
        },
        "probe_value_prediction": score,
        "task_id": task_id,
    }
