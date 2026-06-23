import json

from experiments.build_latent_aggregation_multi_aspect_v6_freeze import (
    FROZEN_TASK_IDS,
    build_freeze_manifest,
    render_markdown,
)


def test_multi_aspect_v6_freeze_targets_anchor_deficit_coverage(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "v5_replay.json"
    coverage = tmp_path / "v5_coverage.json"
    raw = tmp_path / "labels.jsonl"
    scores = tmp_path / "scores.json"
    probe_raw = tmp_path / "probe.jsonl"
    probe_scores = tmp_path / "probe_scores.json"
    diversity_raw = tmp_path / "diversity.jsonl"
    diversity_scores = tmp_path / "diversity_scores.json"
    anchor_deficit_raw = tmp_path / "anchor_deficit.jsonl"
    anchor_deficit_scores = tmp_path / "anchor_deficit_scores.json"
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    replay.write_text(
        json.dumps(
            {
                "gate_evaluation": {"overall_status": "passed"},
                "summary": {"complement_coverage_count": 34},
            }
        ),
        encoding="utf-8",
    )
    coverage.write_text(
        json.dumps(
            {
                "summary": {
                    "no_complement_blockers": {
                        "anchor_dominates_candidate_aspects": 13,
                        "positive_but_below_threshold": 1,
                    },
                    "tasks_without_selected_complement": 14,
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v5_replay_path=replay,
        v5_coverage_path=coverage,
        label_raw_path=raw,
        label_scores_path=scores,
        probe_raw_path=probe_raw,
        probe_scores_path=probe_scores,
        diversity_raw_path=diversity_raw,
        diversity_scores_path=diversity_scores,
        anchor_deficit_raw_path=anchor_deficit_raw,
        anchor_deficit_scores_path=anchor_deficit_scores,
    )
    markdown = render_markdown(manifest)
    generation = manifest["trajectory_generation_contract"]
    gates = manifest["statistical_gates"]
    robustness = manifest["robustness_gates"]

    assert manifest["task_preset"] == "latent_aggregation_multi_aspect_v6_plan297_344"
    assert manifest["task_count"] == 48
    assert manifest["task_ids"][0] == "plan_297"
    assert manifest["task_ids"][-1] == "plan_344"
    assert manifest["freshness_contract"]["prior_planning_task_max"] == 296
    assert "llada_anchor_deficit_constraint_gap_rescue" in generation["families"]
    assert "--constraint-gap-rescue-trigger prompt_gap" in generation["anchor_deficit_command"]
    assert "--extra-raw" in generation["replay_command"]
    assert generation["anchor_deficit_raw_output"] == str(anchor_deficit_raw)
    assert gates["minimum_complement_coverage_count"] == 36
    assert gates["must_report_anchor_deficit_generation_cost"] is True
    assert robustness["must_report_anchor_deficit_incremental_coverage"] is True
    assert manifest["task_mix_contract"]["task_theme_by_id"]["plan_297"] == "coverage_gap_targeting"
    assert manifest["task_mix_contract"]["task_theme_by_id"]["plan_344"] == "reproducibility_governance"
    assert "anchor-deficit constraint-gap rescue rows" in markdown


def test_multi_aspect_v6_freeze_requires_v5_anchor_dominance_bottleneck(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "v5_replay.json"
    coverage = tmp_path / "v5_coverage.json"
    outputs = [tmp_path / name for name in [
        "labels.jsonl",
        "scores.json",
        "probe.jsonl",
        "probe_scores.json",
        "diversity.jsonl",
        "diversity_scores.json",
        "anchor_deficit.jsonl",
        "anchor_deficit_scores.json",
    ]]
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    replay.write_text(
        json.dumps(
            {
                "gate_evaluation": {"overall_status": "passed"},
                "summary": {"complement_coverage_count": 34},
            }
        ),
        encoding="utf-8",
    )
    coverage.write_text(
        json.dumps(
            {
                "summary": {
                    "no_complement_blockers": {"positive_but_below_threshold": 14},
                    "tasks_without_selected_complement": 14,
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            v5_replay_path=replay,
            v5_coverage_path=coverage,
            label_raw_path=outputs[0],
            label_scores_path=outputs[1],
            probe_raw_path=outputs[2],
            probe_scores_path=outputs[3],
            diversity_raw_path=outputs[4],
            diversity_scores_path=outputs[5],
            anchor_deficit_raw_path=outputs[6],
            anchor_deficit_scores_path=outputs[7],
        )
    except ValueError as exc:
        assert "anchor-dominance coverage bottleneck" in str(exc)
    else:
        raise AssertionError("expected missing anchor-dominance blocker to fail v6 freeze")


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
