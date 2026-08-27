import json

from experiments.analyze_latent_aggregation_multi_aspect_v3_probe import (
    analyze_v3_probe_run,
    render_markdown,
)


def test_v3_probe_analysis_counts_valid_diagnostic_probe_rows(tmp_path):
    raw = tmp_path / "raw.jsonl"
    scores = tmp_path / "scores.json"
    raw.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _probe_record("plan_201", 0.35, 0.1875),
                _probe_record("plan_202", 0.10, 0.1875),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scores.write_text(
        json.dumps(
            {
                "all_generation_count": 10,
                "content_hash": "abc123",
                "counterfactual_probe_generation_count": 2,
                "repair_spend_gate_rows": [
                    _gate_row("plan_201", 0.30, 2, 0, 0.04, True),
                    _gate_row("plan_202", 0.20, 1, 1, 0.02, False),
                ],
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_v3_probe_run(raw_path=raw, scores_path=scores)
    markdown = render_markdown(analysis)

    assert analysis["schema"] == "latent_aggregation_multi_aspect_v3_probe_analysis.v1"
    assert analysis["summary"]["measured_probe_count"] == 2
    assert analysis["summary"]["stage1_valid_probe_count"] == 1
    assert analysis["summary"]["should_run_count"] == 0
    assert analysis["summary"]["probe_task_wins"] == 1
    assert analysis["summary"]["probe_task_losses"] == 1
    assert analysis["summary"]["max_remaining_gap_count"] == 1
    assert "diagnostic observations" in markdown


def _probe_record(task_id, score, cost):
    return {
        "counterfactual_probe": {"probe_cost_relative": cost},
        "generation_stage": "counterfactual_probe",
        "task": {"task_id": task_id},
        "task_score": {"score": score},
    }


def _gate_row(task_id, source_score, resolved, remaining, value, valid):
    return {
        "counterfactual_probe_remaining_gap_count": remaining,
        "counterfactual_probe_resolved_gap_count": resolved,
        "counterfactual_probe_text_valid_for_stage1": valid,
        "measured_probe_value_prediction": value,
        "should_run": False,
        "source_control": "low_confidence_32",
        "source_quality": source_score,
        "source_task_score": source_score,
        "task_id": task_id,
    }
