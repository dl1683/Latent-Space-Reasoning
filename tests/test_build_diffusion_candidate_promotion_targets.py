import json

from experiments.build_diffusion_candidate_promotion_targets import (
    build_candidate_promotion_targets,
    render_markdown,
)


def test_candidate_promotion_targets_label_generated_repair_not_oracle(tmp_path):
    raw = tmp_path / "raw.jsonl"
    scores = tmp_path / "scores.json"
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    _repair_raw(
                        "plan_a",
                        repair_task=0.40,
                        source_task=0.30,
                        repair_quality=0.35,
                        source_quality=0.25,
                    )
                ),
                json.dumps(
                    _repair_raw(
                        "plan_b",
                        repair_task=0.28,
                        source_task=0.30,
                        repair_quality=0.22,
                        source_quality=0.25,
                    )
                ),
            ]
        ),
        encoding="utf-8",
    )
    scores.write_text(
        json.dumps(
            {
                "comparison_rows": [
                    _comparison("plan_a", trajectory=0.32, selected_repair_lift=0.08),
                    _comparison(
                        "plan_b",
                        trajectory=0.32,
                        selected_repair_lift=0.0,
                        oracle_lift=0.05,
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )

    targets = build_candidate_promotion_targets(raw_path=raw, scores_path=scores)

    assert targets["summary"]["positive_tasks"] == ["plan_a"]
    assert targets["summary"]["negative_tasks"] == ["plan_b"]
    assert targets["summary"]["candidate_aware_promotion_error_count"] == 0
    assert targets["rows"][1]["candidate_lift_vs_trajectory"] == -0.03999999999999998


def test_candidate_promotion_targets_markdown_exposes_boundary(tmp_path):
    raw = tmp_path / "raw.jsonl"
    scores = tmp_path / "scores.json"
    raw.write_text(
        json.dumps(
            _repair_raw(
                "plan_a",
                repair_task=0.40,
                source_task=0.30,
                repair_quality=0.35,
                source_quality=0.25,
            )
        ),
        encoding="utf-8",
    )
    scores.write_text(
        json.dumps({"comparison_rows": [_comparison("plan_a", trajectory=0.32)]}),
        encoding="utf-8",
    )

    targets = build_candidate_promotion_targets(raw_path=raw, scores_path=scores)
    markdown = render_markdown(targets)

    assert "# Diffusion Candidate Promotion Targets" in markdown
    assert "post-repair promotion" in markdown
    assert "candidate_aware_promotion_v1" in markdown


def _repair_raw(
    task_id,
    *,
    repair_task,
    source_task,
    repair_quality,
    source_quality,
):
    return {
        "generation_stage": "repair_candidate",
        "planning_quality_score": repair_quality,
        "repair": {
            "name": "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
            "planning_span_target_scores": [
                {
                    "score": 2.0,
                    "source_relative_preservation": 0.7,
                }
            ],
            "prompt_constraint_gap_terms": ["measure", "baseline"],
            "source_planning_quality_score": source_quality,
            "source_task_score": source_task,
        },
        "task": {"task_id": task_id},
        "task_score": {"score": repair_task},
    }


def _comparison(task_id, *, trajectory, selected_repair_lift=0.0, oracle_lift=0.0):
    return {
        "oracle_delta_vs_trajectory": oracle_lift,
        "repair_delta_vs_evolved": selected_repair_lift,
        "repair_selector_edge": selected_repair_lift,
        "repair_selector_score": selected_repair_lift + 0.3,
        "task_id": task_id,
        "trajectory_task_score": trajectory,
    }
