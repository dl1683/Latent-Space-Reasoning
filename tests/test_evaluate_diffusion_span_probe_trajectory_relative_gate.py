import json

from experiments.evaluate_diffusion_span_probe_trajectory_relative_gate import (
    evaluate_trajectory_relative_gate,
    render_markdown,
)


def test_trajectory_relative_gate_blocks_weak_no_lift_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "experiments.evaluate_diffusion_span_probe_trajectory_relative_gate.BASE_SIGNATURE_POSITIVE_UTILITY_BAR",
        0.1,
    )
    signature_model = tmp_path / "signature.json"
    spend_eval = tmp_path / "spend_eval.json"
    signature_model.write_text(
        json.dumps(
            {
                "leave_one_slice_out": {
                    "rows": [
                        _signature_row("train_pos", "other.json", True, lift=0.20),
                        _signature_row("weak_pos", "weak.json", True, lift=0.12),
                        _signature_row("weak_zero", "weak.json", False, lift=0.0),
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    spend_eval.write_text(
        json.dumps(
            {
                "rows": [
                    _spend_row("weak_pos", trajectory_relative=True),
                    _spend_row("weak_zero", trajectory_relative=False),
                ]
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_trajectory_relative_gate(
        signature_model_path=signature_model,
        spend_eval_path=spend_eval,
        selection_penalty=0.02,
        weak_slice="weak.json",
    )
    markdown = render_markdown(result)

    selected = result["trajectory_relative_gate"]
    assert selected["false_negative_count"] == 0
    assert selected["false_positive_count"] == 0
    assert selected["blocked_cohort_risk_task_ids"] == ["weak_zero"]
    assert selected["weak_slice_summary"]["selected_count"] == 1
    assert "depends on a second information channel" in markdown


def _signature_row(task_id, source_fit, label, *, lift):
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


def _spend_row(task_id, *, trajectory_relative):
    return {
        "source_task_delta_vs_trajectory": 0.0 if trajectory_relative else -0.05,
        "task_id": task_id,
        "trajectory_relative_prediction": trajectory_relative,
    }
