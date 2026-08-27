import json

from experiments.analyze_diffusion_counterfactual_micro_probe_run import (
    analyze_counterfactual_micro_probe_run,
    render_markdown,
)


def test_micro_probe_run_summary_joins_targets_and_gate_rows(tmp_path):
    targets = tmp_path / "targets.json"
    scores = tmp_path / "scores.json"
    targets.write_text(
        json.dumps(
            {
                "rows": [
                    _target("plan_a", label=True, lift=0.2),
                    _target("plan_b", label=False, lift=-0.1),
                ]
            }
        ),
        encoding="utf-8",
    )
    scores.write_text(
        json.dumps(
            {
                "all_generation_count": 2,
                "content_hash": "abc123",
                "counterfactual_probe_generation_count": 1,
                "repair_spend_gate_rows": [
                    _gate("plan_a", would_probe=True, observation="measured_generation"),
                    _gate("plan_b", would_probe=False, observation="deterministic_scaffold"),
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = analyze_counterfactual_micro_probe_run(
        scores_path=scores,
        targets_path=targets,
    )
    markdown = render_markdown(summary)

    assert summary["schema"] == "diffusion_counterfactual_micro_probe_run_summary.v1"
    assert summary["summary"]["row_count"] == 2
    assert summary["summary"]["measured_probe_count"] == 1
    assert summary["summary"]["probe_triage_error_count"] == 0
    assert summary["summary"]["should_run_count"] == 0
    assert summary["summary"]["gate_decision"] == "diagnostic_only"
    assert "Micro-Probe Counterexamples" in markdown


def _target(task_id, *, label, lift):
    return {
        "counterexample_type": "false_negative" if label else "false_positive",
        "labels": {
            "candidate_lift_vs_trajectory": lift,
            "promote_vs_trajectory": label,
        },
        "task_id": task_id,
    }


def _gate(task_id, *, would_probe, observation):
    return {
        "counterfactual_probe_generated_token_count": 32 if would_probe else None,
        "counterfactual_probe_observation": observation,
        "measured_probe_value_prediction": 0.1 if would_probe else None,
        "prompt_gap_count": 4 if would_probe else 10,
        "should_run": False,
        "source_quality": 0.2,
        "task_id": task_id,
        "would_probe": would_probe,
    }
