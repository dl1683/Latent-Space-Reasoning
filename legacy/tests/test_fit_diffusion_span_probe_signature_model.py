import json

from experiments.fit_diffusion_span_probe_signature_model import (
    fit_span_probe_signature_model,
    render_markdown,
)


def test_span_probe_signature_model_reports_leave_slice_out_failure(tmp_path):
    train_slice = tmp_path / "slice_a.json"
    held_slice = tmp_path / "slice_b.json"
    train_slice.write_text(
        json.dumps(
            {
                "rows": [
                    _row("train_pos", True, gap=1.0, span=0.4, lift=0.2),
                    _row("train_neg", False, gap=0.0, span=0.8, lift=0.0),
                ]
            }
        ),
        encoding="utf-8",
    )
    held_slice.write_text(
        json.dumps(
            {
                "rows": [
                    _row("held_pos", True, gap=0.0, span=0.8, lift=0.1),
                    _row("held_neg", False, gap=1.0, span=0.4, lift=0.0),
                ]
            }
        ),
        encoding="utf-8",
    )

    result = fit_span_probe_signature_model(slice_fit_paths=(train_slice, held_slice))
    markdown = render_markdown(result)

    assert result["summary"]["target_count"] == 4
    assert result["leave_one_slice_out"]["summary"]["error_count"] >= 1
    assert "Do not promote this signature model" in markdown
    assert "Leave-One-Slice-Out Rows" in markdown


def _row(task_id, label, *, gap, span, lift):
    return {
        "candidate_lift_vs_trajectory": lift,
        "features": {
            "counterfactual_probe_text_duplicate_authorization": 0.0,
            "counterfactual_probe_text_malformed_compact_key": 0.0,
            "counterfactual_probe_text_max_slot_overlap": 0.0,
            "counterfactual_probe_text_repeated_token_excess": 0.0,
            "counterfactual_probe_text_semantic_defect": 0.0,
            "counterfactual_probe_text_semantic_valid_for_stage1": 1.0,
            "counterfactual_probe_text_template_slot_echo": 0.0,
            "counterfactual_probe_text_weird_punctuation": 0.0,
            "counterfactual_probe_text_x0_x2_slot_overlap": 0.0,
            "measured_distinct_retention_risk_visibility": 0.5,
            "measured_expected_gap_visibility_gain": gap,
            "measured_expected_realization_defect_visibility": 0.0,
            "measured_expected_retention_risk_visibility": 0.5,
            "measured_expected_span_evidence_gain": span,
            "measured_probe_value_prediction": gap,
            "prompt_gap_count": 4.0,
            "source_quality": 0.2,
            "would_probe_score": 1.0,
        },
        "label": label,
        "task_id": task_id,
        "valid_for_stage1": True,
    }
