import json

from experiments.evaluate_diffusion_span_probe_no_lift_veto import (
    evaluate_no_lift_veto,
    render_markdown,
)


def test_no_lift_veto_reports_leave_slice_out_tradeoff(tmp_path):
    signature_model = tmp_path / "signature.json"
    signature_model.write_text(
        json.dumps(
            {
                "leave_one_slice_out": {
                    "rows": [
                        _row("train_pos", "slice_a", True, prediction=True, gap=1.0, lift=0.2),
                        _row("train_neg", "slice_a", False, prediction=True, gap=0.0, lift=0.0),
                        _row("held_pos", "slice_b", True, prediction=True, gap=0.0, lift=0.2),
                        _row("held_neg", "slice_b", False, prediction=True, gap=1.0, lift=0.0),
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_no_lift_veto(
        signature_model_path=signature_model,
        selection_penalty=0.02,
    )
    markdown = render_markdown(result)

    assert result["summary"]["target_count"] == 4
    assert result["candidate_rule_count"] > 1
    assert result["veto_leave_one_slice_out"]["false_negative_count"] >= 1
    assert "Do not promote this no-lift veto" in markdown
    assert "Slice Rules" in markdown


def _row(task_id, source_fit, label, *, prediction, gap, lift):
    return {
        "candidate_lift_vs_trajectory": lift,
        "label": label,
        "measured_expected_gap_visibility_gain": gap,
        "prediction": prediction,
        "probe_signature_score": gap,
        "source_fit": source_fit,
        "task_id": task_id,
    }
