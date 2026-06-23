import json

import pytest

from experiments.build_latent_aggregation_multi_aspect_v8_targeted_source import (
    build_targeted_source_contract,
    render_markdown,
)


def test_v8_targeted_source_contract_freezes_uncovered_tasks_and_command(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    failure = tmp_path / "failure.json"
    raw = tmp_path / "target_raw.jsonl"
    scores = tmp_path / "target_scores.json"
    report = tmp_path / "target_report.md"
    tasks.write_text(
        "\n".join(
            [
                json.dumps(_task("plan_a")),
                json.dumps(_task("plan_b")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    failure.write_text(json.dumps(_failure(["plan_a", "plan_b"])), encoding="utf-8")

    manifest = build_targeted_source_contract(
        tasks_path=tasks,
        failure_analysis_path=failure,
        raw_output_path=raw,
        scores_output_path=scores,
        source_report_output_path=report,
    )
    markdown = render_markdown(manifest)

    assert manifest["schema"] == "latent_aggregation_multi_aspect_v8_targeted_source_contract.v1"
    assert manifest["task_ids"] == ["plan_a", "plan_b"]
    assert manifest["success_contract"]["minimum_new_promoted_coverage_floor"] == 13
    assert "--task-ids plan_a,plan_b" in manifest["source_family_contract"]["command"]
    assert "--repair-pack constraint_span_history_contrast" in manifest["source_family_contract"]["command"]
    assert "--repair-source-policy non_revision_plus_gap_trajectory" in manifest["source_family_contract"]["command"]
    assert "does not promote v8" in markdown


def test_v8_targeted_source_contract_refuses_existing_outputs(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    failure = tmp_path / "failure.json"
    raw = tmp_path / "target_raw.jsonl"
    scores = tmp_path / "target_scores.json"
    report = tmp_path / "target_report.md"
    tasks.write_text(json.dumps(_task("plan_a")) + "\n", encoding="utf-8")
    failure.write_text(json.dumps(_failure(["plan_a"])), encoding="utf-8")
    raw.write_text("already here\n", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing targeted source contract"):
        build_targeted_source_contract(
            tasks_path=tasks,
            failure_analysis_path=failure,
            raw_output_path=raw,
            scores_output_path=scores,
            source_report_output_path=report,
        )


def _task(task_id):
    return {
        "answer": None,
        "answer_type": "rubric",
        "family": "planning",
        "max_new_tokens": 64,
        "prompt": f"Prompt for {task_id}",
        "rubric_items": ["name evidence", "define rollback"],
        "scorer": "planning_rubric_v1",
        "task_id": task_id,
    }


def _failure(task_ids):
    return {
        "evidence_boundary": {"status": "fresh_v7_failed_replay_failure_analysis"},
        "summary": {
            "coverage_shortfall_to_gate": 12,
            "next_source_minimum_new_promoted_coverage_floor": 13,
            "promotion_shortfall_to_gate": 7,
            "uncovered_task_count": len(task_ids),
            "uncovered_task_ids": task_ids,
            "wilson_success_shortfall_to_gate": 13,
        },
    }
