import json

from experiments.analyze_latent_aggregation_gain_failure import analyze_gain_failure, render_markdown


def test_gain_failure_flags_score_lift_without_new_components(tmp_path):
    replay = tmp_path / "replay.json"
    components = tmp_path / "components.jsonl"
    replay.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "best_single_score": 0.4,
                        "best_single_trajectory_id": "task_a:best",
                        "component_gain": 0,
                        "decision": {"status": "blocked_no_component_gain"},
                        "realized_aggregate_score": 0.6,
                        "source_diversity": 2,
                        "task_id": "task_a",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    components.write_text(
        "\n".join(
            [
                json.dumps(_component("task_a", "task_a:best", "a", 0.9)),
                json.dumps(_component("task_a", "task_a:other", "a", 0.95)),
            ]
        ),
        encoding="utf-8",
    )

    result = analyze_gain_failure(replay_path=replay, components_path=components)
    task = result["tasks"][0]
    markdown = render_markdown(result)

    assert task["diagnosis"] == "score_lift_from_multi_source_but_no_new_components"
    assert task["score_lift_without_component_gain"] is True
    assert task["selected_outside_best_count"] == 0
    assert task["outside_best_source_count"] == 1
    assert result["summary"]["score_lift_without_component_gain_task_count"] == 1
    assert "complement-aware selector" in markdown


def _component(task_id, trajectory_id, component_id, support_score):
    return {
        "component_id": component_id,
        "component_type": "planning",
        "component_weight": 1.0,
        "source_span": component_id,
        "support_score": support_score,
        "supported": True,
        "task_id": task_id,
        "trajectory_family": "test",
        "trajectory_id": trajectory_id,
    }
