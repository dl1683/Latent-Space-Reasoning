import json

from experiments.analyze_diffusion_span_probe_trajectory_relative_controls import (
    analyze_trajectory_relative_controls,
    render_markdown,
)


def test_trajectory_relative_controls_degrade_without_true_channel(tmp_path):
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

    result = analyze_trajectory_relative_controls(
        signature_model_path=signature_model,
        spend_eval_path=spend_eval,
        selection_penalty=0.02,
        weak_slice="weak.json",
    )
    markdown = render_markdown(result)

    assert result["summary"]["strictly_degraded_controls"] >= 3
    assert result["selected_policy"]["weak_slice_summary"]["false_positive_count"] == 0
    controls = {row["control_id"]: row for row in result["controls"]}
    assert controls["no_trajectory_channel"]["weak_slice_summary"]["false_positive_count"] == 1
    assert controls["inverted_trajectory_relative"]["false_negative_count"] >= 1
    assert controls["delta_nonnegative_only"]["weak_slice_summary"]["false_positive_count"] == 1
    assert "passes this negative-control audit" in markdown


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
        "source_task_delta_vs_trajectory": 0.0,
        "task_id": task_id,
        "trajectory_relative_prediction": trajectory_relative,
    }
