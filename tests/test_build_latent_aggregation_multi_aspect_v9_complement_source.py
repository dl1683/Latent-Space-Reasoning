import json

import pytest

from experiments.build_latent_aggregation_multi_aspect_v9_complement_source import (
    build_complement_source_contract,
    render_markdown,
)


def test_v9_complement_source_contract_emits_complement_packet_prompts(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    failure = tmp_path / "v7_failure.json"
    gap = tmp_path / "v8_gap.json"
    raw = tmp_path / "v7_raw.jsonl"
    ontology = tmp_path / "ontology.jsonl"
    cross = tmp_path / "cross.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    source_raw = tmp_path / "source_raw.jsonl"
    scores = tmp_path / "scores.json"
    source_report = tmp_path / "source_report.md"
    replay = tmp_path / "replay.json"
    aspects = tmp_path / "aspects.jsonl"
    realized = tmp_path / "realized.jsonl"
    replay_report = tmp_path / "replay.md"

    tasks.write_text(
        "\n".join([json.dumps(_task("plan_a")), json.dumps(_task("plan_b"))]) + "\n",
        encoding="utf-8",
    )
    failure.write_text(json.dumps(_failure(["plan_a", "plan_b"])), encoding="utf-8")
    gap.write_text(json.dumps(_source_gap(["plan_a", "plan_b"])), encoding="utf-8")
    raw.write_text(
        "\n".join(
            [
                json.dumps(_row("plan_a", score=0.6, text="Measure baseline reliability.")),
                json.dumps(_row("plan_b", score=0.7, text="Assign the on-call owner.")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ontology.write_text("", encoding="utf-8")
    cross.write_text("", encoding="utf-8")

    manifest, prompt_rows = build_complement_source_contract(
        tasks_path=tasks,
        v7_failure_path=failure,
        v8_source_gap_path=gap,
        v7_raw_path=raw,
        v7_ontology_raw_path=ontology,
        v7_cross_raw_path=cross,
        prompts_output_path=prompts,
        raw_output_path=source_raw,
        scores_output_path=scores,
        source_report_output_path=source_report,
        replay_output_path=replay,
        aspects_output_path=aspects,
        realized_output_path=realized,
        replay_report_output_path=replay_report,
    )
    markdown = render_markdown(manifest)

    assert manifest["schema"] == "latent_aggregation_multi_aspect_v9_complement_source_contract.v1"
    assert manifest["source_family_contract"]["family"] == "complement_packet"
    assert manifest["success_contract"]["minimum_new_promoted_coverage_floor"] == 13
    assert manifest["task_ids"] == ["plan_a", "plan_b"]
    assert len(prompt_rows) == 2
    assert "Generate a complement packet, not a replacement final answer" in prompt_rows[0]["prompt"]
    assert "Return raw JSON only; do not wrap it in markdown fences" in prompt_rows[0]["prompt"]
    assert "Return exactly 3 complement clauses" in prompt_rows[0]["prompt"]
    assert "Every `why_not_in_anchor` value must be non-empty" in prompt_rows[0]["prompt"]
    assert "Do not omit any object key" in prompt_rows[0]["prompt"]
    assert "Example clause object" in prompt_rows[0]["prompt"]
    assert prompt_rows[0]["missing_anchor_aspects"]
    assert "--extra-raw" in manifest["replay_contract"]["command"]
    assert "does not promote v9" in markdown


def test_v9_complement_source_contract_refuses_existing_source_outputs(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    failure = tmp_path / "v7_failure.json"
    gap = tmp_path / "v8_gap.json"
    raw = tmp_path / "v7_raw.jsonl"
    empty = tmp_path / "empty.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    source_raw = tmp_path / "source_raw.jsonl"
    scores = tmp_path / "scores.json"
    source_report = tmp_path / "source_report.md"
    replay = tmp_path / "replay.json"
    aspects = tmp_path / "aspects.jsonl"
    realized = tmp_path / "realized.jsonl"
    replay_report = tmp_path / "replay.md"

    tasks.write_text(json.dumps(_task("plan_a")) + "\n", encoding="utf-8")
    failure.write_text(json.dumps(_failure(["plan_a"])), encoding="utf-8")
    gap.write_text(json.dumps(_source_gap(["plan_a"])), encoding="utf-8")
    raw.write_text(json.dumps(_row("plan_a", score=0.6, text="Measure baseline reliability.")) + "\n", encoding="utf-8")
    empty.write_text("", encoding="utf-8")
    source_raw.write_text("already generated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing complement source contract"):
        build_complement_source_contract(
            tasks_path=tasks,
            v7_failure_path=failure,
            v8_source_gap_path=gap,
            v7_raw_path=raw,
            v7_ontology_raw_path=empty,
            v7_cross_raw_path=empty,
            prompts_output_path=prompts,
            raw_output_path=source_raw,
            scores_output_path=scores,
            source_report_output_path=source_report,
            replay_output_path=replay,
            aspects_output_path=aspects,
            realized_output_path=realized,
            replay_report_output_path=replay_report,
        )


def _task(task_id):
    return {
        "answer": None,
        "answer_type": "rubric",
        "family": "planning",
        "max_new_tokens": 64,
        "prompt": f"Plan the reliability decision for {task_id}.",
        "rubric_items": ["measure reliability", "assign owner", "define rollback"],
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


def _source_gap(task_ids):
    return {
        "evidence_boundary": {"status": "v8_targeted_history_contrast_source_gap_diagnostic"},
        "summary": {
            "anchor_shift_suppression_count": 1,
            "mean_delta_vs_original_anchor": -0.05,
            "repair_not_stronger_no_new_aspect_count": len(task_ids),
            "targeted_complement_vs_augmented_anchor_count": 0,
            "targeted_repair_count": len(task_ids),
        },
        "tasks": [
            {
                "failure_class": "repair_not_stronger_no_new_expanded_aspect",
                "original_anchor_score": 0.6 + index / 10,
                "original_anchor_trajectory_id": f"{task_id}:llada:schedule:repair_candidate",
                "targeted_delta_vs_original_anchor": -0.1,
                "targeted_score": 0.5,
                "task_id": task_id,
            }
            for index, task_id in enumerate(task_ids)
        ],
    }


def _row(task_id, *, score, text):
    return {
        "candidate_key": "llada",
        "generation_stage": "repair_candidate",
        "task_id": task_id,
        "task_score": {"score": score},
        "text": text,
    }
