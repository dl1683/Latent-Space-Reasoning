import json

from experiments.build_latent_aggregation_multi_aspect_v3_freeze import (
    FROZEN_TASK_IDS,
    build_freeze_manifest,
    render_markdown,
)


def test_multi_aspect_v3_freeze_locks_fresh_tasks_and_split_gates(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    failure = tmp_path / "failure.json"
    coverage = tmp_path / "coverage.json"
    raw = tmp_path / "labels.jsonl"
    scores = tmp_path / "scores.json"
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    failure.write_text(
        json.dumps({"summary": {"complement_task_count": 9}}),
        encoding="utf-8",
    )
    coverage.write_text(
        json.dumps({"summary": {"tasks_without_selected_complement": 15}}),
        encoding="utf-8",
    )

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        failure_diagnostic_path=failure,
        coverage_diagnostic_path=coverage,
        label_raw_path=raw,
        label_scores_path=scores,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "latent_aggregation_multi_aspect_v3_plan201_224"
    assert manifest["task_count"] == 24
    assert manifest["freshness_contract"]["prior_planning_task_max"] == 200
    assert manifest["task_ids"][0] == "plan_201"
    assert manifest["task_ids"][-1] == "plan_224"
    assert manifest["statistical_gates"]["minimum_complement_coverage_count"] == 12
    assert manifest["statistical_gates"]["minimum_conditional_promoted_fraction"] == 0.5
    assert manifest["aspect_deficit_probe_contract"]["maximum_probes_per_task"] == 2
    assert "targeted_aspect_deficit_probe_v1" in manifest["trajectory_generation_contract"]["families"]
    assert "--task-ids plan_201,plan_202" in manifest["trajectory_generation_contract"]["gpu_command"]
    assert "coverage" in markdown.lower()


def test_multi_aspect_v3_freeze_refuses_existing_label_outputs(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    failure = tmp_path / "failure.json"
    coverage = tmp_path / "coverage.json"
    raw = tmp_path / "labels.jsonl"
    scores = tmp_path / "scores.json"
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    failure.write_text(json.dumps({"summary": {"complement_task_count": 1}}), encoding="utf-8")
    coverage.write_text(
        json.dumps({"summary": {"tasks_without_selected_complement": 1}}),
        encoding="utf-8",
    )
    raw.write_text("", encoding="utf-8")

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            failure_diagnostic_path=failure,
            coverage_diagnostic_path=coverage,
            label_raw_path=raw,
            label_scores_path=scores,
        )
    except ValueError as exc:
        assert "label outputs exist" in str(exc)
    else:
        raise AssertionError("expected existing labels to block v3 freeze")


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
