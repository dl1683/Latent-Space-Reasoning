import json

from experiments.fit_diffusion_span_probe_signed_value import (
    fit_signed_value_head,
    render_markdown,
)


def test_signed_value_head_reports_direct_utility_gate(tmp_path):
    signature_model = tmp_path / "signature.json"
    veto = tmp_path / "veto.json"
    signature_model.write_text(
        json.dumps(
            {
                "leave_one_slice_out": {
                    "rows": [
                        _row("a_pos", "slice_a", True, gap=1.0, lift=0.20),
                        _row("a_neg", "slice_a", False, gap=0.0, lift=-0.02),
                        _row("b_pos", "slice_b", True, gap=1.0, lift=0.18),
                        _row("b_neg", "slice_b", False, gap=0.0, lift=-0.01),
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    veto.write_text(
        json.dumps(
            {
                "veto_leave_one_slice_out": {
                    "false_negative_count": 0,
                    "false_positive_count": 1,
                    "policy_utility": 0.1,
                    "selected_count": 2,
                    "selected_lift": 0.2,
                }
            }
        ),
        encoding="utf-8",
    )

    result = fit_signed_value_head(
        signature_model_path=signature_model,
        no_lift_veto_path=veto,
        neighbor_counts=(1,),
        selection_penalty=0.02,
    )
    markdown = render_markdown(result)

    assert result["summary"]["target_count"] == 4
    assert result["selected_model"]["policy_utility"] > 0.0
    assert "Signed Value Tomography Controller" in markdown
    assert "Do not promote this signed-value head" in markdown


def _row(task_id, source_fit, label, *, gap, lift):
    return {
        "candidate_lift_vs_trajectory": lift,
        "counterfactual_probe_text_semantic_valid_for_stage1": 1.0,
        "label": label,
        "measured_expected_gap_visibility_gain": gap,
        "measured_expected_span_evidence_gain": gap,
        "prediction": True,
        "probe_signature_score": gap,
        "source_fit": source_fit,
        "task_id": task_id,
        "valid_for_stage1": True,
    }
