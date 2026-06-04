import json

from experiments.analyze_diffusion_span_probe_value_floor_v11_counterexamples import (
    analyze_counterexamples,
    render_markdown,
)


def test_counterexample_analysis_tracks_source_aware_failures(tmp_path):
    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            {
                "row_diagnostics": [
                    {
                        "candidate_lift_vs_trajectory": 0.0,
                        "label": False,
                        "measured_probe_value_prediction": 0.04,
                        "oracle_label": False,
                        "oracle_lift_vs_trajectory": 0.0,
                        "prompt_coverage": 0.0,
                        "prompt_gap_count": 12,
                        "selected": True,
                        "source_task_delta_vs_trajectory": -0.2,
                        "task_id": "plan_fp",
                    },
                    {
                        "candidate_lift_vs_trajectory": 0.12,
                        "label": True,
                        "measured_probe_value_prediction": 0.02,
                        "oracle_label": True,
                        "oracle_lift_vs_trajectory": 0.12,
                        "prompt_coverage": 0.75,
                        "prompt_gap_count": 4,
                        "selected": False,
                        "source_task_delta_vs_trajectory": 0.0,
                        "task_id": "plan_fn",
                    },
                    {
                        "candidate_lift_vs_trajectory": 0.0,
                        "label": False,
                        "measured_probe_value_prediction": 0.027,
                        "oracle_label": True,
                        "oracle_lift_vs_trajectory": 0.04,
                        "prompt_coverage": 0.8,
                        "prompt_gap_count": 3,
                        "selected": False,
                        "source_task_delta_vs_trajectory": 0.0,
                        "task_id": "plan_oracle",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_counterexamples(replay_path=replay)
    markdown = render_markdown(analysis)

    assert analysis["summary"]["counterexample_count"] == 3
    assert analysis["summary"]["source_divergent_count"] == 1
    assert [row["task_id"] for row in analysis["counterexample_rows"]] == [
        "plan_fp",
        "plan_fn",
        "plan_oracle",
    ]
    assert analysis["counterexample_rows"][0]["counterexample_type"] == "value_floor_false_positive"
    assert analysis["counterexample_rows"][1]["counterexample_type"] == "value_floor_false_negative"
    assert analysis["counterexample_rows"][2]["counterexample_type"] == "oracle_positive_selector_miss"

    selected_best = analysis["selected_repair_hypotheses"][0]
    oracle_best = analysis["oracle_hypotheses"][0]
    assert selected_best["hypothesis_id"] == "source_nonnegative_and_prompt_gap_le_4"
    assert selected_best["false_positive_task_ids"] == ["plan_oracle"]
    assert oracle_best["false_negative_task_ids"] == []
    assert "Do not promote a replacement controller" in markdown
    assert "source-aware counterexample map" in markdown
