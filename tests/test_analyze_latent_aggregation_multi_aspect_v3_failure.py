import json

from experiments.analyze_latent_aggregation_multi_aspect_v3_failure import analyze_failure


def test_v3_failure_analysis_computes_coverage_shortfalls(tmp_path):
    replay = tmp_path / "replay.json"
    freeze = tmp_path / "freeze.json"
    replay.write_text(
        json.dumps(
            {
                "gate_evaluation": {
                    "gates": [
                        {"name": "minimum_complement_coverage_count", "status": "fail"},
                        {"name": "minimum_all_task_mean_non_rubric_lift", "status": "fail"},
                    ]
                },
                "summary": {
                    "all_task_mean_non_rubric_lift": 0.02,
                    "complement_coverage_count": 6,
                    "complement_coverage_fraction": 0.25,
                    "conditional_mean_non_rubric_lift": 0.08,
                    "conditional_promoted_fraction": 1.0,
                    "online_promoted_task_count": 6,
                    "task_count": 24,
                },
            }
        ),
        encoding="utf-8",
    )
    freeze.write_text(
        json.dumps(
            {
                "statistical_gates": {
                    "minimum_aggregate_win_count": 8,
                    "minimum_all_task_mean_non_rubric_lift": 0.03,
                    "minimum_complement_coverage_count": 12,
                    "minimum_complement_coverage_fraction": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_failure(replay_path=replay, freeze_path=freeze)
    summary = analysis["summary"]

    assert summary["coverage_shortfall_to_aggregate_win_gate"] == 2
    assert summary["coverage_shortfall_to_global_non_rubric_gate"] == 3
    assert summary["coverage_shortfall_to_frozen_gate"] == 6
    assert summary["coverage_needed_for_global_non_rubric_gate"] == 9
    assert "minimum_complement_coverage_count" in summary["failed_gates"]
