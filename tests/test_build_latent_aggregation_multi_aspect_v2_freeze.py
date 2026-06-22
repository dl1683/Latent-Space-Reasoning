import json

from experiments.build_latent_aggregation_multi_aspect_v2_freeze import (
    FROZEN_TASK_IDS,
    build_freeze_manifest,
    render_markdown,
)


def test_multi_aspect_v2_freeze_locks_held_out_tasks_and_aspect_contract(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    gain = tmp_path / "gain.json"
    dimension = tmp_path / "dimension.json"
    raw = tmp_path / "labels.jsonl"
    scores = tmp_path / "scores.json"
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    gain.write_text(
        json.dumps({"summary": {"score_lift_without_component_gain_task_count": 12}}),
        encoding="utf-8",
    )
    dimension.write_text(
        json.dumps({"summary": {"best_full_rubric_score_lift_without_gain_task_count": 4}}),
        encoding="utf-8",
    )

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        gain_diagnostic_path=gain,
        dimension_diagnostic_path=dimension,
        label_raw_path=raw,
        label_scores_path=scores,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "latent_aggregation_multi_aspect_v2_plan025_048"
    assert manifest["task_count"] == 24
    assert manifest["aspect_ontology"]["rubric_support_threshold"] == 0.1
    assert "risk_awareness" in manifest["aspect_ontology"]["aspect_types"]
    assert manifest["selector_contract"]["name"] == "best_anchor_plus_complement_aspect_selector_v2"
    assert manifest["statistical_gates"]["minimum_aggregate_win_count"] == 5
    assert "--task-ids plan_025,plan_026" in manifest["trajectory_generation_contract"]["gpu_command"]
    assert "--device cuda" in manifest["trajectory_generation_contract"]["gpu_command"]
    assert "multi-aspect" in markdown.lower()


def test_multi_aspect_v2_freeze_refuses_existing_label_outputs(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    gain = tmp_path / "gain.json"
    dimension = tmp_path / "dimension.json"
    raw = tmp_path / "labels.jsonl"
    scores = tmp_path / "scores.json"
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    gain.write_text(
        json.dumps({"summary": {"score_lift_without_component_gain_task_count": 1}}),
        encoding="utf-8",
    )
    dimension.write_text(
        json.dumps({"summary": {"best_full_rubric_score_lift_without_gain_task_count": 1}}),
        encoding="utf-8",
    )
    raw.write_text("", encoding="utf-8")

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            gain_diagnostic_path=gain,
            dimension_diagnostic_path=dimension,
            label_raw_path=raw,
            label_scores_path=scores,
        )
    except ValueError as exc:
        assert "label outputs exist" in str(exc)
    else:
        raise AssertionError("expected existing labels to block multi-aspect freeze")


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
