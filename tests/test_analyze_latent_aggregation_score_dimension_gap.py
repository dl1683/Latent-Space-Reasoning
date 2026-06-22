import json

from experiments.analyze_latent_aggregation_score_dimension_gap import (
    analyze_score_dimension_gap,
    render_markdown,
)


def test_score_dimension_gap_detects_hidden_non_rubric_lift(tmp_path):
    replay = tmp_path / "replay.json"
    raw = tmp_path / "raw.jsonl"
    realized = tmp_path / "realized.jsonl"
    replay.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "component_gain": 0,
                        "decision": {"status": "blocked_no_component_gain"},
                        "task_id": "plan_a",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    raw.write_text(
        json.dumps(
            {
                "candidate_key": "base",
                "generation_stage": "candidate_generation",
                "schedule": {"name": "x"},
                "task": {"task_id": "plan_a"},
                "task_score": {
                    "details": _details(rubric_coverage=1.0, specificity=0.1, constraint_handling=0.0),
                    "score": 0.30,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    realized.write_text(
        json.dumps(
            {
                "task_id": "plan_a",
                "task_score": {
                    "details": _details(rubric_coverage=1.0, specificity=0.8, constraint_handling=0.5),
                    "score": 0.60,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = analyze_score_dimension_gap(replay_path=replay, raw_path=raw, realized_path=realized)
    task = result["tasks"][0]
    markdown = render_markdown(result)

    assert task["diagnosis"] == "score_lift_hidden_by_rubric_saturation"
    assert task["best_full_rubric"] is True
    assert task["score_lift_without_component_gain"] is True
    assert task["largest_dimension_lift"] == "specificity"
    assert result["summary"]["best_full_rubric_score_lift_without_gain_task_count"] == 1
    assert "multi-aspect component fusion" in markdown


def _details(*, rubric_coverage, specificity, constraint_handling):
    return {
        "causal_diagnosis": 0.0,
        "completion": 1.0,
        "constraint_handling": constraint_handling,
        "risk_awareness": 0.0,
        "rubric_coverage": rubric_coverage,
        "specificity": specificity,
    }
