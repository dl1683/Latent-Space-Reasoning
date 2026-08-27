import json

import pytest

from experiments.build_latent_aggregation_multi_aspect_v10_complement_prompts import (
    build_complement_prompt_contract,
    render_markdown,
)


def test_v10_complement_prompt_contract_emits_label_free_source_prompts(tmp_path):
    freeze = tmp_path / "freeze.json"
    tasks = tmp_path / "tasks.jsonl"
    raw = tmp_path / "raw.jsonl"
    scores = tmp_path / "scores.json"
    prompts = tmp_path / "prompts.jsonl"
    packet_raw = tmp_path / "packet_raw.jsonl"
    packet_scores = tmp_path / "packet_scores.json"
    packet_report = tmp_path / "packet_report.md"
    replay = tmp_path / "replay.json"

    freeze.write_text(json.dumps(_freeze(["plan_393", "plan_394"])), encoding="utf-8")
    tasks.write_text(
        "\n".join([json.dumps(_task("plan_393")), json.dumps(_task("plan_394"))]) + "\n",
        encoding="utf-8",
    )
    raw.write_text(
        "\n".join(
            [
                json.dumps(_row("plan_393", "low_confidence_32", "candidate_generation", "Measure the baseline.")),
                json.dumps(
                    _row(
                        "plan_393",
                        "random_32",
                        "candidate_generation",
                        "Assign the owner, then verify telemetry, and define rollback criteria.",
                    )
                ),
                json.dumps(_row("plan_394", "low_confidence_32", "candidate_generation", "Name the scope.")),
                json.dumps(
                    _row(
                        "plan_394",
                        "entropy_64",
                        "repair_candidate",
                        "The maintainer should first validate logs, then stop if the threshold fails.",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scores.write_text(json.dumps(_scores(["plan_393", "plan_394"])), encoding="utf-8")

    manifest, prompt_rows = build_complement_prompt_contract(
        freeze_path=freeze,
        tasks_path=tasks,
        label_raw_path=raw,
        label_scores_path=scores,
        prompts_output_path=prompts,
        packet_raw_path=packet_raw,
        packet_scores_path=packet_scores,
        packet_report_path=packet_report,
        replay_output_path=replay,
    )
    markdown = render_markdown(manifest)

    assert manifest["schema"] == "latent_aggregation_multi_aspect_v10_complement_prompt_contract.v1"
    assert manifest["source_inputs"]["label_free_derivation"] is True
    assert "post-packet aggregation outcomes" in manifest["source_inputs"]["forbidden_inputs"]
    assert len(prompt_rows) == 2
    assert prompt_rows[0]["label_free_derivation"] is True
    assert prompt_rows[0]["anchor_schedule"] == "low_confidence_32"
    assert prompt_rows[0]["source_candidates"]
    assert "Auxiliary source text" in prompt_rows[0]["prompt"]
    assert "Return raw JSON only; do not wrap it in markdown fences" in prompt_rows[0]["prompt"]
    assert "Return exactly 3 complement clauses" in prompt_rows[0]["prompt"]
    assert "Every `why_not_in_anchor` value must be non-empty" in prompt_rows[0]["prompt"]
    assert "Every clause must be grounded in the task or one of the auxiliary source texts" in prompt_rows[0]["prompt"]
    assert "Current anchor answer" in prompt_rows[0]["prompt"]
    assert "not a result claim" in markdown


def test_v10_complement_prompt_contract_refuses_packet_outputs(tmp_path):
    freeze = tmp_path / "freeze.json"
    tasks = tmp_path / "tasks.jsonl"
    raw = tmp_path / "raw.jsonl"
    scores = tmp_path / "scores.json"
    packet_raw = tmp_path / "packet_raw.jsonl"
    freeze.write_text(json.dumps(_freeze(["plan_393"])), encoding="utf-8")
    tasks.write_text(json.dumps(_task("plan_393")) + "\n", encoding="utf-8")
    raw.write_text(
        "\n".join(
            [
                json.dumps(_row("plan_393", "low_confidence_32", "candidate_generation", "Measure baseline.")),
                json.dumps(_row("plan_393", "random_32", "candidate_generation", "Assign owner.")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scores.write_text(json.dumps(_scores(["plan_393"])), encoding="utf-8")
    packet_raw.write_text("already generated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing v10 prompt build"):
        build_complement_prompt_contract(
            freeze_path=freeze,
            tasks_path=tasks,
            label_raw_path=raw,
            label_scores_path=scores,
            prompts_output_path=tmp_path / "prompts.jsonl",
            packet_raw_path=packet_raw,
            packet_scores_path=tmp_path / "packet_scores.json",
            packet_report_path=tmp_path / "packet_report.md",
            replay_output_path=tmp_path / "replay.json",
        )


def test_v10_complement_prompt_contract_requires_nonempty_llada_rows(tmp_path):
    freeze = tmp_path / "freeze.json"
    tasks = tmp_path / "tasks.jsonl"
    raw = tmp_path / "raw.jsonl"
    scores = tmp_path / "scores.json"
    freeze.write_text(json.dumps(_freeze(["plan_393"])), encoding="utf-8")
    tasks.write_text(json.dumps(_task("plan_393")) + "\n", encoding="utf-8")
    raw.write_text(json.dumps({**_row("plan_393", "low_confidence_32", "candidate_generation", ""), "text": ""}) + "\n")
    scores.write_text(json.dumps(_scores(["plan_393"])), encoding="utf-8")

    with pytest.raises(ValueError, match="no non-empty LLaDA rows"):
        build_complement_prompt_contract(
            freeze_path=freeze,
            tasks_path=tasks,
            label_raw_path=raw,
            label_scores_path=scores,
            prompts_output_path=tmp_path / "prompts.jsonl",
            packet_raw_path=tmp_path / "packet_raw.jsonl",
            packet_scores_path=tmp_path / "packet_scores.json",
            packet_report_path=tmp_path / "packet_report.md",
            replay_output_path=tmp_path / "replay.json",
        )


def _freeze(task_ids):
    return {
        "schema": "latent_aggregation_multi_aspect_v10_complement_freeze.v1",
        "task_ids": task_ids,
        "task_preset": "test",
        "transfer_contract": {"policy": "v9_complement_packet_policy_fixed_before_v10_labels"},
    }


def _task(task_id):
    return {
        "answer": None,
        "answer_type": "rubric",
        "family": "planning",
        "max_new_tokens": 64,
        "prompt": f"Plan the label-free transfer for {task_id}.",
        "rubric_items": ["measure baseline", "assign owner", "define rollback"],
        "scorer": "planning_rubric_v1",
        "task_id": task_id,
    }


def _row(task_id, schedule, stage, text):
    return {
        "candidate_key": "llada-8b-instruct-hf",
        "generation_stage": stage,
        "schedule": {"name": schedule},
        "task": {"task_id": task_id},
        "text": text,
    }


def _scores(task_ids):
    return {
        "all_generation_count": len(task_ids) * 2,
        "comparison_rows": [{"task_id": task_id} for task_id in task_ids],
        "content_hash": "abc123",
        "run_id": "diffusion-test",
    }
