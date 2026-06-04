import json

from experiments.replay_diffusion_denoise_phase_realization_v13 import replay_v13, render_markdown


def test_v13_replay_scores_frozen_surface_against_selected_and_oracle_labels(tmp_path):
    measurement = tmp_path / "measurement.json"
    labels = tmp_path / "labels.json"
    measurement.write_text(json.dumps(_measurement_payload()), encoding="utf-8")
    labels.write_text(json.dumps(_label_payload()), encoding="utf-8")

    result = replay_v13(measurement_boundary_path=measurement, label_scores_path=labels)
    markdown = render_markdown(result)

    selected = result["selected_repair_hypotheses"]["frozen_denoise_realization_surface"]
    oracle = result["oracle_hypotheses"]["frozen_denoise_realization_surface"]
    skeleton = result["selected_repair_hypotheses"]["skeleton_only"]

    assert result["summary"]["selected_repair_positive_task_ids"] == ["plan_a"]
    assert result["summary"]["oracle_positive_task_ids"] == ["plan_a", "plan_c"]
    assert selected["false_positive_task_ids"] == []
    assert selected["false_negative_task_ids"] == []
    assert selected["selected_count"] == 1
    assert oracle["false_negative_task_ids"] == ["plan_c"]
    assert skeleton["false_positive_task_ids"] == ["plan_b", "plan_c"]
    assert "clears selected-repair replay" in markdown


def _measurement_payload():
    return {
        "row_diagnostics": [
            {
                "first_repairable_denoise_skeleton_step_fraction": 0.2,
                "has_repairable_denoise_skeleton": True,
                "measured_probe_value_prediction": 0.03,
                "peak_denoise_prompt_coverage": 0.5,
                "prompt_coverage": 0.8,
                "prompt_gap_count": 2,
                "source_task_delta_vs_trajectory": 0.0,
                "surface_selected": True,
                "task_id": "plan_a",
            },
            {
                "first_repairable_denoise_skeleton_step_fraction": 0.6,
                "has_repairable_denoise_skeleton": True,
                "measured_probe_value_prediction": 0.04,
                "peak_denoise_prompt_coverage": 0.5,
                "prompt_coverage": 0.8,
                "prompt_gap_count": 2,
                "source_task_delta_vs_trajectory": -0.1,
                "surface_selected": False,
                "task_id": "plan_b",
            },
            {
                "first_repairable_denoise_skeleton_step_fraction": 0.2,
                "has_repairable_denoise_skeleton": True,
                "measured_probe_value_prediction": 0.01,
                "peak_denoise_prompt_coverage": 0.2,
                "prompt_coverage": 1.0,
                "prompt_gap_count": 0,
                "source_task_delta_vs_trajectory": 0.0,
                "surface_selected": False,
                "task_id": "plan_c",
            },
        ]
    }


def _label_payload():
    return {
        "all_generation_count": 9,
        "comparison_rows": [
            {
                "oracle_task_score": 0.5,
                "repair_task_score": 0.5,
                "task_id": "plan_a",
                "trajectory_task_score": 0.4,
            },
            {
                "oracle_task_score": 0.4,
                "repair_task_score": 0.4,
                "task_id": "plan_b",
                "trajectory_task_score": 0.4,
            },
            {
                "oracle_task_score": 0.45,
                "repair_task_score": 0.4,
                "task_id": "plan_c",
                "trajectory_task_score": 0.4,
            },
        ],
        "repair_spend_gate_rows": [
            {"should_run": True, "task_id": "plan_a"},
            {"should_run": True, "task_id": "plan_b"},
            {"should_run": False, "task_id": "plan_c"},
        ],
        "run_id": "diffusion-label-test",
    }
