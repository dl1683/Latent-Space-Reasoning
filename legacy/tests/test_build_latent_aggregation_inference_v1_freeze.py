import json

from experiments.build_latent_aggregation_inference_v1_freeze import (
    FROZEN_TASK_IDS,
    build_freeze_manifest,
    render_markdown,
)


def test_inference_v1_freeze_locks_tasks_and_online_contracts(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    prior = tmp_path / "prior.json"
    raw = tmp_path / "labels.jsonl"
    scores = tmp_path / "scores.json"
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    prior.write_text(
        json.dumps(
            {
                "summary": {
                    "promoted_task_count": 2,
                    "promoted_task_fraction": 0.25,
                    "promoted_task_wilson95": [0.071, 0.591],
                    "task_count": 8,
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        prior_replay_path=prior,
        label_raw_path=raw,
        label_scores_path=scores,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "latent_aggregation_inference_v1_plan009_024"
    assert manifest["task_count"] == 16
    assert manifest["prior_oracle_replay"]["boundary"] == "oracle_replay_not_inference_time"
    assert "rubric hit labels" in manifest["extractor_contract"]["forbidden_inputs"]
    assert "task_score.score" in manifest["extractor_contract"]["forbidden_inputs"]
    assert manifest["statistical_gates"]["minimum_aggregate_win_count"] == 3
    assert manifest["realizer_contract"]["status"] == "must_emit_final_answer_before_rescoring"
    assert "--task-ids plan_009,plan_010" in manifest["trajectory_generation_contract"]["gpu_command"]
    assert "--device cuda" in manifest["trajectory_generation_contract"]["gpu_command"]
    assert "first inference-time aggregation validation" in markdown


def test_inference_v1_freeze_refuses_existing_label_outputs(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    prior = tmp_path / "prior.json"
    raw = tmp_path / "labels.jsonl"
    scores = tmp_path / "scores.json"
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    prior.write_text(json.dumps({"summary": {"promoted_task_fraction": 0.25}}), encoding="utf-8")
    raw.write_text("", encoding="utf-8")

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            prior_replay_path=prior,
            label_raw_path=raw,
            label_scores_path=scores,
        )
    except ValueError as exc:
        assert "label outputs exist" in str(exc)
    else:
        raise AssertionError("expected existing labels to block inference freeze")


def _task(task_id):
    return {
        "answer": None,
        "answer_type": "rubric",
        "family": "planning",
        "max_new_tokens": 64,
        "prompt": f"Prompt for {task_id}",
        "rubric_items": [f"rubric {index}" for index in range(5)],
        "scorer": "planning_rubric_v1",
        "task_id": task_id,
    }
