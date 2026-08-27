import json

from experiments.build_latent_aggregation_multi_aspect_v5_freeze import (
    FROZEN_TASK_IDS,
    build_freeze_manifest,
    render_markdown,
)


def test_multi_aspect_v5_freeze_locks_larger_robustness_replication(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "v4_replay.json"
    coverage = tmp_path / "v4_coverage.json"
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
        json.dumps(
            {
                "gate_evaluation": {"overall_status": "passed"},
                "summary": {"complement_coverage_count": 14},
            }
        ),
        encoding="utf-8",
    )
    coverage.write_text(
        json.dumps({"summary": {"tasks_without_selected_complement": 10}}),
        encoding="utf-8",
    )

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v4_replay_path=replay,
        v4_coverage_path=coverage,
        label_raw_path=raw,
        label_scores_path=scores,
        probe_raw_path=probe_raw,
        probe_scores_path=probe_scores,
        diversity_raw_path=diversity_raw,
        diversity_scores_path=diversity_scores,
    )
    markdown = render_markdown(manifest)
    gates = manifest["statistical_gates"]
    robustness = manifest["robustness_gates"]
    generation = manifest["trajectory_generation_contract"]

    assert manifest["task_preset"] == "latent_aggregation_multi_aspect_v5_plan249_296"
    assert manifest["task_count"] == 48
    assert manifest["task_ids"][0] == "plan_249"
    assert manifest["task_ids"][-1] == "plan_296"
    assert manifest["freshness_contract"]["prior_planning_task_max"] == 248
    assert gates["minimum_complement_coverage_count"] == 26
    assert gates["minimum_wilson_lower_bound"] == 0.35
    assert robustness["maximum_single_task_share_of_total_lift"] == 0.25
    assert robustness["must_report_leave_one_out_mean_lift_range"] is True
    assert robustness["must_report_source_family_ablation"] is True
    assert manifest["task_mix_contract"]["must_report_theme_bucket_results"] is True
    assert manifest["task_mix_contract"]["task_theme_by_id"]["plan_249"] == "research_program_design"
    assert manifest["task_mix_contract"]["task_theme_by_id"]["plan_296"] == "failure_forensics"
    assert "--task-ids plan_249,plan_250" in generation["label_command"]
    assert "--include-revision-schedules" in generation["diversity_extension_command"]
    assert "--extra-raw" in generation["replay_command"]
    assert "larger 48-task replication" in markdown
    assert "Robustness Gates" in markdown


def test_multi_aspect_v5_freeze_requires_passing_v4_replay(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "v4_replay.json"
    coverage = tmp_path / "v4_coverage.json"
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
        json.dumps(
            {
                "gate_evaluation": {"overall_status": "failed"},
                "summary": {"complement_coverage_count": 14},
            }
        ),
        encoding="utf-8",
    )
    coverage.write_text(
        json.dumps({"summary": {"tasks_without_selected_complement": 10}}),
        encoding="utf-8",
    )

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            v4_replay_path=replay,
            v4_coverage_path=coverage,
            label_raw_path=raw,
            label_scores_path=scores,
            probe_raw_path=probe_raw,
            probe_scores_path=probe_scores,
            diversity_raw_path=diversity_raw,
            diversity_scores_path=diversity_scores,
        )
    except ValueError as exc:
        assert "passing v4 replay" in str(exc)
    else:
        raise AssertionError("expected failed v4 replay to block v5 freeze")


def test_multi_aspect_v5_freeze_refuses_existing_outputs(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "v4_replay.json"
    coverage = tmp_path / "v4_coverage.json"
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
        json.dumps(
            {
                "gate_evaluation": {"overall_status": "passed"},
                "summary": {"complement_coverage_count": 14},
            }
        ),
        encoding="utf-8",
    )
    coverage.write_text(
        json.dumps({"summary": {"tasks_without_selected_complement": 10}}),
        encoding="utf-8",
    )
    raw.write_text("", encoding="utf-8")

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            v4_replay_path=replay,
            v4_coverage_path=coverage,
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
        raise AssertionError("expected existing v5 outputs to block freeze")


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
