import json

from experiments.build_latent_aggregation_multi_aspect_v4_freeze import (
    FROZEN_TASK_IDS,
    build_freeze_manifest,
    render_markdown,
)


def test_multi_aspect_v4_freeze_locks_fresh_diversity_source_mix(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "v3_replay.json"
    coverage = tmp_path / "v3_coverage.json"
    raw = tmp_path / "labels.jsonl"
    scores = tmp_path / "scores.json"
    probe_raw = tmp_path / "probe.jsonl"
    probe_scores = tmp_path / "probe_scores.json"
    diversity_raw = tmp_path / "diversity.jsonl"
    diversity_scores = tmp_path / "diversity_scores.json"
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    replay.write_text(
        json.dumps({"summary": {"online_promoted_task_count": 13}}),
        encoding="utf-8",
    )
    coverage.write_text(
        json.dumps({"summary": {"tasks_with_selected_complement": 13}}),
        encoding="utf-8",
    )

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v3_replay_path=replay,
        v3_coverage_path=coverage,
        label_raw_path=raw,
        label_scores_path=scores,
        probe_raw_path=probe_raw,
        probe_scores_path=probe_scores,
        diversity_raw_path=diversity_raw,
        diversity_scores_path=diversity_scores,
    )
    markdown = render_markdown(manifest)
    generation = manifest["trajectory_generation_contract"]

    assert manifest["task_preset"] == "latent_aggregation_multi_aspect_v4_plan225_248"
    assert manifest["task_ids"][0] == "plan_225"
    assert manifest["task_ids"][-1] == "plan_248"
    assert manifest["freshness_contract"]["prior_planning_task_max"] == 224
    assert manifest["statistical_gates"]["minimum_complement_coverage_count"] == 12
    assert manifest["statistical_gates"]["must_report_diversity_generation_cost"] is True
    assert "--include-revision-schedules" in generation["diversity_extension_command"]
    assert "--extra-raw" in generation["replay_command"]
    assert str(diversity_raw) in generation["replay_command"]
    assert "Diversity Extension" in markdown
    assert "fresh 24-task replication" in markdown


def test_multi_aspect_v4_freeze_refuses_existing_outputs(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "v3_replay.json"
    coverage = tmp_path / "v3_coverage.json"
    raw = tmp_path / "labels.jsonl"
    scores = tmp_path / "scores.json"
    probe_raw = tmp_path / "probe.jsonl"
    probe_scores = tmp_path / "probe_scores.json"
    diversity_raw = tmp_path / "diversity.jsonl"
    diversity_scores = tmp_path / "diversity_scores.json"
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    replay.write_text(
        json.dumps({"summary": {"online_promoted_task_count": 13}}),
        encoding="utf-8",
    )
    coverage.write_text(
        json.dumps({"summary": {"tasks_with_selected_complement": 13}}),
        encoding="utf-8",
    )
    diversity_raw.write_text("", encoding="utf-8")

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            v3_replay_path=replay,
            v3_coverage_path=coverage,
            label_raw_path=raw,
            label_scores_path=scores,
            probe_raw_path=probe_raw,
            probe_scores_path=probe_scores,
            diversity_raw_path=diversity_raw,
            diversity_scores_path=diversity_scores,
        )
    except ValueError as exc:
        assert "output artifacts exist" in str(exc)
    else:
        raise AssertionError("expected existing v4 outputs to block freeze")


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
