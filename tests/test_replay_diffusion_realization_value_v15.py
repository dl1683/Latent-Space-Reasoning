import json

from experiments.replay_diffusion_realization_value_v15 import replay_v15, render_markdown


def test_v15_replay_rejects_probe_surface_when_static_disagreement_is_positive(tmp_path):
    measurement = tmp_path / "measurement.json"
    labels = tmp_path / "labels.json"
    measurement.write_text(json.dumps(_measurement_payload()), encoding="utf-8")
    labels.write_text(json.dumps(_label_payload()), encoding="utf-8")

    result = replay_v15(measurement_boundary_path=measurement, label_scores_path=labels)
    markdown = render_markdown(result)

    static = result["selected_repair_hypotheses"]["static_source_gap_coverage_v15"]
    probe = result["selected_repair_hypotheses"]["probe_conditioned_realization_value_v15"]
    trigger = result["selected_repair_hypotheses"]["label_pass_denoise_trigger"]

    assert result["summary"]["selected_repair_positive_task_ids"] == ["plan_static_only"]
    assert result["summary"]["oracle_positive_task_ids"] == ["plan_static_only", "plan_oracle_only"]
    assert static["selected_task_ids"] == ["plan_static_only"]
    assert static["false_positive_task_ids"] == []
    assert static["false_negative_task_ids"] == []
    assert probe["selected_task_ids"] == []
    assert probe["false_negative_task_ids"] == ["plan_static_only"]
    assert trigger["false_positive_task_ids"] == ["plan_oracle_only", "plan_trigger_only"]
    assert "rejects the probe-conditioned surface" in markdown


def _measurement_payload():
    return {
        "row_diagnostics": [
            {
                "has_repairable_denoise_skeleton": True,
                "measured_probe_value_prediction": 0.04,
                "peak_denoise_prompt_coverage": 0.4,
                "probe_surface_selected": False,
                "prompt_coverage": 0.7,
                "prompt_gap_count": 4,
                "source_task_delta_vs_trajectory": 0.0,
                "static_surface_selected": True,
                "task_id": "plan_static_only",
            },
            {
                "has_repairable_denoise_skeleton": True,
                "measured_probe_value_prediction": 0.02,
                "peak_denoise_prompt_coverage": 0.3,
                "probe_surface_selected": False,
                "prompt_coverage": 0.5,
                "prompt_gap_count": 8,
                "source_task_delta_vs_trajectory": 0.0,
                "static_surface_selected": False,
                "task_id": "plan_oracle_only",
            },
            {
                "has_repairable_denoise_skeleton": True,
                "measured_probe_value_prediction": 0.05,
                "peak_denoise_prompt_coverage": 0.2,
                "probe_surface_selected": False,
                "prompt_coverage": 0.5,
                "prompt_gap_count": 8,
                "source_task_delta_vs_trajectory": -0.1,
                "static_surface_selected": False,
                "task_id": "plan_trigger_only",
            },
        ],
        "summary": {
            "disagreement_task_ids": ["plan_static_only"],
            "probe_generation_count": 3,
        },
    }


def _label_payload():
    return {
        "all_generation_count": 9,
        "comparison_rows": [
            {
                "oracle_task_score": 0.50,
                "repair_task_score": 0.50,
                "task_id": "plan_static_only",
                "trajectory_task_score": 0.40,
            },
            {
                "oracle_task_score": 0.42,
                "repair_task_score": 0.40,
                "task_id": "plan_oracle_only",
                "trajectory_task_score": 0.40,
            },
            {
                "oracle_task_score": 0.40,
                "repair_task_score": 0.40,
                "task_id": "plan_trigger_only",
                "trajectory_task_score": 0.40,
            },
        ],
        "repair_spend_gate_rows": [
            {"should_run": True, "task_id": "plan_static_only"},
            {"should_run": True, "task_id": "plan_oracle_only"},
            {"should_run": True, "task_id": "plan_trigger_only"},
        ],
        "run_id": "diffusion-label-test",
    }
