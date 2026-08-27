import json

from experiments.analyze_latent_aggregation_multi_aspect_v2_failure import (
    analyze_failure,
    render_markdown,
)


def test_multi_aspect_failure_separates_coverage_from_conditional_lift(tmp_path):
    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            {
                "summary": {"mean_non_rubric_lift": 0.02},
                "tasks": [
                    _task("a", complements=1, non_rubric_lift=0.08, score_lift=0.09),
                    _task("b", complements=0, non_rubric_lift=0.0, score_lift=0.0),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = analyze_failure(replay_path=replay)
    markdown = render_markdown(result)

    assert result["summary"]["complement_task_count"] == 1
    assert result["summary"]["no_complement_task_count"] == 1
    assert result["summary"]["complement_task_mean_non_rubric_lift"] == 0.08
    assert abs(result["summary"]["missing_global_lift"] - 0.01) < 1e-9
    assert "complement discovery coverage" in markdown


def _task(task_id, *, complements, non_rubric_lift, score_lift):
    return {
        "decision": {"status": "online_promoted_local" if complements else "blocked_no_complement_material"},
        "dimension_gain_count": complements,
        "non_rubric_lift": non_rubric_lift,
        "rubric_gain_count": 0,
        "score_lift": score_lift,
        "selected_complement_count": complements,
        "task_id": task_id,
    }
