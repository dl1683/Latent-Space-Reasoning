import json

from experiments.build_diffusion_counterfactual_probe_transfer_targets import (
    build_counterfactual_probe_transfer_targets,
    render_markdown,
)


def test_counterfactual_probe_transfer_targets_label_fresh_planning_rows(tmp_path):
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            {
                "content_hash": "abc123",
                "comparison_rows": [
                    _comparison("plan_a", repair=0.55, trajectory=0.50),
                    _comparison("plan_b", repair=0.45, trajectory=0.50),
                    {"task_id": "math_a", "repair_task_score": 0.99, "trajectory_task_score": 0.01},
                ],
            }
        ),
        encoding="utf-8",
    )

    targets = build_counterfactual_probe_transfer_targets(
        lift_min=1e-9,
        probe_policy="span_tomography_probe_v4",
        scores_path=scores,
    )
    markdown = render_markdown(targets)

    assert targets["schema"] == "diffusion_counterfactual_probe_transfer_targets.v1"
    assert targets["summary"]["target_count"] == 2
    assert targets["summary"]["positive_task_ids"] == ["plan_a"]
    assert targets["summary"]["negative_task_ids"] == ["plan_b"]
    assert targets["rows"][0]["labels"]["candidate_lift_vs_trajectory"] == 0.050000000000000044
    assert targets["rows"][0]["probe_policy"] == "span_tomography_probe_v4"
    assert "abc123" in markdown


def _comparison(task_id, *, repair, trajectory):
    return {
        "repair_control": "constraint_gap_span_repair",
        "repair_selector_edge": repair - trajectory,
        "repair_source_control": "low_confidence_32",
        "repair_task_score": repair,
        "task_id": task_id,
        "trajectory_task_score": trajectory,
    }
