import json

from experiments.evaluate_diffusion_span_probe_cohort_risk import (
    evaluate_cohort_risk,
    render_markdown,
)


def test_cohort_risk_can_pass_m1_while_failing_weak_slice(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "experiments.evaluate_diffusion_span_probe_cohort_risk.DEFAULT_NEIGHBOR_COUNTS",
        (1,),
    )
    monkeypatch.setattr(
        "experiments.evaluate_diffusion_span_probe_cohort_risk.DEFAULT_STD_PENALTIES",
        (0.0,),
    )
    monkeypatch.setattr(
        "experiments.evaluate_diffusion_span_probe_cohort_risk.DEFAULT_NEGATIVE_FRACTION_PENALTIES",
        (0.0,),
    )
    monkeypatch.setattr(
        "experiments.evaluate_diffusion_span_probe_cohort_risk.DEFAULT_MARGINS",
        (0.0,),
    )
    monkeypatch.setattr(
        "experiments.evaluate_diffusion_span_probe_cohort_risk.BASE_SIGNATURE_POSITIVE_UTILITY_BAR",
        0.1,
    )
    signature_model = tmp_path / "signature.json"
    signature_model.write_text(
        json.dumps(
            {
                "leave_one_slice_out": {
                    "rows": [
                        _row("train_pos", "other.json", True, lift=0.20),
                        _row("weak_pos", "weak.json", True, lift=0.12),
                        _row("weak_zero_1", "weak.json", False, lift=0.0),
                        _row("weak_zero_2", "weak.json", False, lift=0.0),
                        _row("weak_zero_3", "weak.json", False, lift=0.0),
                        _row("weak_zero_4", "weak.json", False, lift=0.0),
                        _row("weak_zero_5", "weak.json", False, lift=0.0),
                        _row("weak_zero_6", "weak.json", False, lift=0.0),
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_cohort_risk(
        signature_model_path=signature_model,
        selection_penalty=0.02,
        weak_slice="weak.json",
    )
    markdown = render_markdown(result)

    assert result["selected_model"]["policy_utility"] > 0.0
    assert result["selected_model"]["false_negative_count"] == 0
    assert result["selected_model"]["weak_slice_summary"]["false_positive_count"] == 6
    assert "fails the M2.5 cohort-calibration requirement" in markdown


def _row(task_id, source_fit, label, *, lift):
    return {
        "candidate_lift_vs_trajectory": lift,
        "counterfactual_probe_text_semantic_valid_for_stage1": 1.0,
        "label": label,
        "measured_expected_gap_visibility_gain": 1.0,
        "prediction": True,
        "source_fit": source_fit,
        "task_id": task_id,
        "valid_for_stage1": True,
    }
