import json

from experiments.replay_diffusion_realization_value_v14b import replay_v14b, render_markdown


def test_v14b_replay_scores_addendum_against_labels(tmp_path):
    addendum = tmp_path / "addendum.json"
    labels = tmp_path / "labels.json"
    addendum.write_text(json.dumps(_addendum_payload()), encoding="utf-8")
    labels.write_text(json.dumps(_label_payload()), encoding="utf-8")

    result = replay_v14b(addendum_path=addendum, label_scores_path=labels)
    markdown = render_markdown(result)

    v14 = result["selected_repair_hypotheses"]["realization_value_probe_banded_v14"]
    v14b = result["selected_repair_hypotheses"]["realization_value_probe_banded_v14b"]
    trigger = result["selected_repair_hypotheses"]["label_pass_denoise_trigger"]

    assert result["summary"]["selected_repair_positive_task_ids"] == ["plan_109", "plan_112"]
    assert v14["selected_task_ids"] == []
    assert v14["false_negative_task_ids"] == ["plan_109", "plan_112"]
    assert v14b["selected_task_ids"] == ["plan_109", "plan_112"]
    assert v14b["false_positive_task_ids"] == []
    assert v14b["false_negative_task_ids"] == []
    assert trigger["false_positive_task_ids"] == ["plan_108"]
    assert "rescues the empty v14 target" in markdown


def _addendum_payload():
    return {
        "measurement_replay": {
            "row_diagnostics": [
                {
                    "has_repairable_denoise_skeleton": True,
                    "measured_probe_value_prediction": 0.048,
                    "prompt_coverage": 0.77,
                    "prompt_gap_count": 4,
                    "source_task_delta_vs_trajectory": -0.1,
                    "surface_selected": False,
                    "surface_selected_v14b": False,
                    "task_id": "plan_108",
                },
                {
                    "has_repairable_denoise_skeleton": True,
                    "measured_probe_value_prediction": 0.0324,
                    "prompt_coverage": 0.64,
                    "prompt_gap_count": 5,
                    "source_task_delta_vs_trajectory": 0.0,
                    "surface_selected": False,
                    "surface_selected_v14b": True,
                    "task_id": "plan_109",
                },
                {
                    "has_repairable_denoise_skeleton": True,
                    "measured_probe_value_prediction": 0.0323,
                    "prompt_coverage": 0.57,
                    "prompt_gap_count": 6,
                    "source_task_delta_vs_trajectory": 0.063,
                    "surface_selected": False,
                    "surface_selected_v14b": True,
                    "task_id": "plan_112",
                },
            ]
        },
        "target_surface": {"surface_id": "realization_value_probe_banded_v14b"},
    }


def _label_payload():
    return {
        "all_generation_count": 9,
        "comparison_rows": [
            {
                "oracle_task_score": 0.29,
                "repair_task_score": 0.29,
                "task_id": "plan_108",
                "trajectory_task_score": 0.29,
            },
            {
                "oracle_task_score": 0.44,
                "repair_task_score": 0.44,
                "task_id": "plan_109",
                "trajectory_task_score": 0.30,
            },
            {
                "oracle_task_score": 0.52,
                "repair_task_score": 0.52,
                "task_id": "plan_112",
                "trajectory_task_score": 0.46,
            },
        ],
        "repair_spend_gate_rows": [
            {"should_run": True, "task_id": "plan_108"},
            {"should_run": True, "task_id": "plan_109"},
            {"should_run": True, "task_id": "plan_112"},
        ],
        "run_id": "diffusion-label-test",
    }
