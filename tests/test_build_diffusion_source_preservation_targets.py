import json

from experiments.build_diffusion_source_preservation_targets import (
    build_source_preservation_targets,
    render_markdown,
)


def test_source_preservation_targets_split_source_degradation_from_repair_positive(tmp_path):
    raw = tmp_path / "raw.jsonl"
    scores = tmp_path / "scores.json"
    raw.write_text(
        "\n".join(
            [
                json.dumps(_repair_record("plan_201", 0.48)),
                json.dumps(_repair_record("plan_202", 0.62)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scores.write_text(json.dumps(_scores_payload()), encoding="utf-8")

    targets = build_source_preservation_targets(raw_path=raw, scores_path=scores)
    markdown = render_markdown(targets)
    summary = targets["summary"]

    assert summary["source_positive_repair_degradation_tasks"] == ["plan_201"]
    assert summary["generated_repair_positive_tasks"] == ["plan_202"]
    assert summary["source_preservation_gate_inconclusive"] is False
    assert "source-positive repair-degradation rows" in markdown


def _repair_record(task_id, score):
    return {
        "generation_stage": "repair_candidate",
        "task": {"task_id": task_id},
        "task_score": {"score": score},
    }


def _scores_payload():
    return {
        "comparison_rows": [
            {
                "oracle_delta_vs_trajectory": 0.05,
                "repair_delta_vs_evolved": 0.0,
                "repair_task_score": 0.50,
                "task_id": "plan_201",
            },
            {
                "oracle_delta_vs_trajectory": 0.10,
                "repair_delta_vs_evolved": 0.12,
                "repair_task_score": 0.62,
                "task_id": "plan_202",
            },
        ],
        "repair_spend_gate_rows": [
            {
                "prompt_coverage": 0.7,
                "prompt_gap_count": 4,
                "should_run": True,
                "source_control": "random_32",
                "source_task_score": 0.55,
                "task_id": "plan_201",
                "trajectory_task_score": 0.50,
            },
            {
                "prompt_coverage": 0.8,
                "prompt_gap_count": 3,
                "should_run": True,
                "source_control": "low_confidence_32",
                "source_task_score": 0.50,
                "task_id": "plan_202",
                "trajectory_task_score": 0.50,
            },
        ],
    }
