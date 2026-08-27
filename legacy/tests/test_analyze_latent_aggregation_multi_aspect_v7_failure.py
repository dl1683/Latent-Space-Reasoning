import json

from experiments.analyze_latent_aggregation_multi_aspect_v7_failure import (
    analyze_v7_failure,
    render_markdown,
)


def test_v7_failure_analysis_defines_next_experiment_floor(tmp_path):
    replay = tmp_path / "replay.json"
    aspects = tmp_path / "aspects.jsonl"
    realized = tmp_path / "realized.jsonl"

    replay.write_text(
        json.dumps(
            {
                "gate_evaluation": {
                    "gates": [
                        {
                            "name": "minimum_complement_coverage_count",
                            "observed": "1",
                            "threshold": "3",
                            "status": "fail",
                        },
                        {
                            "name": "minimum_aggregate_win_count",
                            "observed": "1",
                            "threshold": "2",
                            "status": "fail",
                        },
                        {
                            "name": "minimum_wilson_lower_bound",
                            "observed": "0.045586",
                            "threshold": "0.100000",
                            "status": "fail",
                        },
                        {
                            "name": "must_report_probe_cost",
                            "observed": "reported",
                            "threshold": "reported",
                            "status": "pass",
                        },
                    ],
                    "overall_status": "failed",
                },
                "summary": {
                    "complement_coverage_count": 1,
                    "decision_status_counts": {
                        "blocked_no_complement_material": 3,
                        "online_promoted_local": 1,
                    },
                    "hard_contradiction_count": 0,
                    "label_leakage_check": "passed_label_free_view_only",
                    "online_promoted_task_count": 1,
                    "online_promoted_wilson95": [0.045586, 0.699363],
                    "task_count": 4,
                    "unsupported_addition_count": 0,
                },
                "tasks": [
                    _task("plan_a", selected=1, status="online_promoted_local", lift=0.1),
                    _task("plan_b", selected=0, status="blocked_no_complement_material"),
                    _task("plan_c", selected=0, status="blocked_no_complement_material"),
                    _task("plan_d", selected=0, status="blocked_no_complement_material"),
                ],
            }
        ),
        encoding="utf-8",
    )
    aspects.write_text(
        json.dumps(
            {
                "aspect_class": "expanded",
                "aspect_type": "scope_boundary",
                "selected": True,
                "source_family": "cross_latent_perturbation",
                "task_id": "plan_a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    realized.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "plan_a", "task_score": {"score": 0.6}}),
                json.dumps({"task_id": "plan_b", "task_score": {"score": 0.2}}),
                json.dumps({"task_id": "plan_c", "task_score": {"score": 0.2}}),
                json.dumps({"task_id": "plan_d", "task_score": {"score": 0.2}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = analyze_v7_failure(replay_path=replay, aspects_path=aspects, realized_path=realized)
    markdown = render_markdown(result)

    assert result["summary"]["coverage_shortfall_to_gate"] == 2
    assert result["summary"]["promotion_shortfall_to_gate"] == 1
    assert result["summary"]["minimum_successes_for_wilson_gate"] == 2
    assert result["summary"]["next_source_minimum_new_promoted_coverage_floor"] == 2
    assert result["summary"]["selected_source_family_counts"] == {
        "cross_latent_perturbation": 1
    }
    assert result["summary"]["uncovered_task_ids"] == ["plan_b", "plan_c", "plan_d"]
    assert "Next Experiment Contract" in markdown
    assert "must_report_probe_cost" not in result["summary"]


def test_v7_failure_analysis_tracks_multi_family_task_coverage(tmp_path):
    replay = tmp_path / "replay.json"
    aspects = tmp_path / "aspects.jsonl"
    realized = tmp_path / "realized.jsonl"

    replay.write_text(
        json.dumps(
            {
                "gate_evaluation": {
                    "gates": [
                        {
                            "name": "minimum_complement_coverage_count",
                            "threshold": "1",
                        },
                        {
                            "name": "minimum_aggregate_win_count",
                            "threshold": "1",
                        },
                        {
                            "name": "minimum_wilson_lower_bound",
                            "threshold": "0.0",
                        },
                    ]
                },
                "summary": {
                    "complement_coverage_count": 1,
                    "online_promoted_task_count": 1,
                    "online_promoted_wilson95": [0.2, 1.0],
                    "task_count": 1,
                },
                "tasks": [_task("plan_a", selected=2, status="online_promoted_local", lift=0.2)],
            }
        ),
        encoding="utf-8",
    )
    aspects.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "aspect_class": "expanded",
                        "aspect_type": "scope_boundary",
                        "source_family": "label",
                        "task_id": "plan_a",
                    }
                ),
                json.dumps(
                    {
                        "aspect_class": "expanded",
                        "aspect_type": "owner_assignment",
                        "source_family": "cross_latent_perturbation",
                        "task_id": "plan_a",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    realized.write_text(
        json.dumps({"task_id": "plan_a", "task_score": {"score": 0.7}}) + "\n",
        encoding="utf-8",
    )

    result = analyze_v7_failure(replay_path=replay, aspects_path=aspects, realized_path=realized)

    assert result["summary"]["multi_family_covered_task_count"] == 1
    assert result["summary"]["task_coverage_by_source_family"] == {
        "cross_latent_perturbation": 1,
        "label": 1,
    }
    assert result["summary"]["unique_task_coverage_by_source_family"] == {}


def _task(task_id, *, selected, status, lift=0.0):
    return {
        "anchor_score": 0.2,
        "decision": {"status": status},
        "non_rubric_lift": lift,
        "realized_score": 0.2 + lift,
        "score_lift": lift,
        "selected_complement_count": selected,
        "task_id": task_id,
    }
