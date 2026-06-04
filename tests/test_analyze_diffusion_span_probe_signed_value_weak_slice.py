import json

from experiments.analyze_diffusion_span_probe_signed_value_weak_slice import (
    analyze_weak_slice,
    render_markdown,
)


def test_weak_slice_diagnostic_reports_cohort_failure(tmp_path):
    signed_value = tmp_path / "signed_value.json"
    signature_model = tmp_path / "signature.json"
    signed_value.write_text(
        json.dumps(
            {
                "selected_model": {
                    "feature_group_id": "all",
                    "model_id": "signed_value_knn_k1_all",
                    "neighbor_count": 1,
                },
                "selection_penalty": 0.02,
            }
        ),
        encoding="utf-8",
    )
    signature_model.write_text(
        json.dumps(
            {
                "leave_one_slice_out": {
                    "rows": [
                        _row("train_pos", "other.json", True, gap=1.0, lift=0.12),
                        _row("weak_pos", "weak.json", True, gap=1.0, lift=0.10),
                        _row("weak_zero", "weak.json", False, gap=3.0, lift=0.0),
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    result = analyze_weak_slice(
        signed_value_path=signed_value,
        signature_model_path=signature_model,
        weak_slice="weak.json",
    )
    markdown = render_markdown(result)

    assert result["weak_slice_summary"]["selected_count"] == 2
    assert result["weak_slice_summary"]["false_positive_count"] == 1
    assert result["weak_slice_summary"]["selected_positive_rate"] == 0.5
    prompt_gap = next(
        row for row in result["feature_contrasts"] if row["feature"] == "prompt_gap_count"
    )
    assert prompt_gap["mean_difference"] == -2.0
    assert "cohort-calibration problem" in markdown


def _row(task_id, source_fit, label, *, gap, lift):
    return {
        "candidate_lift_vs_trajectory": lift,
        "counterfactual_probe_text_semantic_valid_for_stage1": 1.0,
        "label": label,
        "measured_expected_gap_visibility_gain": 1.0,
        "prediction": True,
        "prompt_gap_count": gap,
        "source_fit": source_fit,
        "task_id": task_id,
        "valid_for_stage1": True,
    }
