import json

from experiments.analyze_latent_aggregation_v8_targeted_source_gap import (
    analyze_targeted_source_gap,
    render_markdown,
)


def test_v8_targeted_source_gap_matches_colliding_anchor_ids_by_score(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    v7_replay = tmp_path / "v7_replay.json"
    v8_replay = tmp_path / "v8_replay.json"
    v7_raw = tmp_path / "v7_raw.jsonl"
    ontology_raw = tmp_path / "ontology_raw.jsonl"
    cross_raw = tmp_path / "cross_raw.jsonl"
    targeted_raw = tmp_path / "targeted_raw.jsonl"

    tasks.write_text(json.dumps(_task("plan_a")) + "\n", encoding="utf-8")
    anchor_id = "plan_a:llada:schedule:repair_candidate"
    v7_replay.write_text(json.dumps({"tasks": [_replay_task("plan_a", anchor_id, 0.9)]}), encoding="utf-8")
    v8_replay.write_text(json.dumps({"tasks": [_replay_task("plan_a", anchor_id, 0.9)]}), encoding="utf-8")
    v7_raw.write_text(
        "\n".join(
            [
                json.dumps(_row("plan_a", score=0.9, text="Run the baseline measurement.")),
                json.dumps(_row("plan_a", score=0.4, text="Only run the customer demo scope.")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ontology_raw.write_text("", encoding="utf-8")
    cross_raw.write_text("", encoding="utf-8")
    targeted_raw.write_text(
        json.dumps(_row("plan_a", score=0.4, text="Only run the customer demo scope.")) + "\n",
        encoding="utf-8",
    )

    result = analyze_targeted_source_gap(
        tasks_path=tasks,
        v7_replay_path=v7_replay,
        v8_replay_path=v8_replay,
        v7_raw_path=v7_raw,
        v7_ontology_raw_path=ontology_raw,
        v7_cross_raw_path=cross_raw,
        targeted_raw_path=targeted_raw,
    )

    task = result["tasks"][0]
    assert task["targeted_is_augmented_anchor"] is False
    assert task["original_anchor_trajectory_collision_count"] == 2
    assert result["summary"]["tasks_with_original_anchor_id_collisions"] == 1
    assert result["summary"]["anchor_shift_suppression_count"] == 0


def test_v8_targeted_source_gap_renders_non_promotion_boundary(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    v7_replay = tmp_path / "v7_replay.json"
    v8_replay = tmp_path / "v8_replay.json"
    v7_raw = tmp_path / "v7_raw.jsonl"
    ontology_raw = tmp_path / "ontology_raw.jsonl"
    cross_raw = tmp_path / "cross_raw.jsonl"
    targeted_raw = tmp_path / "targeted_raw.jsonl"

    tasks.write_text(json.dumps(_task("plan_a")) + "\n", encoding="utf-8")
    anchor_id = "plan_a:llada:schedule:repair_candidate"
    v7_replay.write_text(json.dumps({"tasks": [_replay_task("plan_a", anchor_id, 0.6)]}), encoding="utf-8")
    v8_replay.write_text(json.dumps({"tasks": [_replay_task("plan_a", anchor_id, 0.6)]}), encoding="utf-8")
    v7_raw.write_text(json.dumps(_row("plan_a", score=0.6, text="Run baseline measurement.")) + "\n", encoding="utf-8")
    ontology_raw.write_text("", encoding="utf-8")
    cross_raw.write_text("", encoding="utf-8")
    targeted_raw.write_text(json.dumps(_row("plan_a", score=0.3, text="Run baseline measurement.")) + "\n", encoding="utf-8")

    result = analyze_targeted_source_gap(
        tasks_path=tasks,
        v7_replay_path=v7_replay,
        v8_replay_path=v8_replay,
        v7_raw_path=v7_raw,
        v7_ontology_raw_path=ontology_raw,
        v7_cross_raw_path=cross_raw,
        targeted_raw_path=targeted_raw,
    )
    markdown = render_markdown(result)

    assert result["summary"]["repair_not_stronger_no_new_aspect_count"] == 1
    assert "does not generate new model outputs and does not promote v8" in markdown
    assert "Tasks whose original anchor ID maps to multiple source rows" in markdown


def test_v8_targeted_source_gap_uses_targeted_text_when_it_is_augmented_anchor(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    v7_replay = tmp_path / "v7_replay.json"
    v8_replay = tmp_path / "v8_replay.json"
    v7_raw = tmp_path / "v7_raw.jsonl"
    ontology_raw = tmp_path / "ontology_raw.jsonl"
    cross_raw = tmp_path / "cross_raw.jsonl"
    targeted_raw = tmp_path / "targeted_raw.jsonl"

    tasks.write_text(json.dumps(_task("plan_a")) + "\n", encoding="utf-8")
    anchor_id = "plan_a:llada:schedule:repair_candidate"
    v7_replay.write_text(json.dumps({"tasks": [_replay_task("plan_a", anchor_id, 0.4)]}), encoding="utf-8")
    v8_replay.write_text(json.dumps({"tasks": [_replay_task("plan_a", anchor_id, 0.5)]}), encoding="utf-8")
    v7_raw.write_text(
        "\n".join(
            [
                json.dumps(_row("plan_a", score=0.4, text="Run the baseline measurement.")),
                json.dumps(_row("plan_a", score=0.5, text="Run the baseline measurement.")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ontology_raw.write_text("", encoding="utf-8")
    cross_raw.write_text("", encoding="utf-8")
    targeted_raw.write_text(
        json.dumps(_row("plan_a", score=0.5, text="Only run the customer demo scope.")) + "\n",
        encoding="utf-8",
    )

    result = analyze_targeted_source_gap(
        tasks_path=tasks,
        v7_replay_path=v7_replay,
        v8_replay_path=v8_replay,
        v7_raw_path=v7_raw,
        v7_ontology_raw_path=ontology_raw,
        v7_cross_raw_path=cross_raw,
        targeted_raw_path=targeted_raw,
    )

    task = result["tasks"][0]
    assert task["targeted_is_augmented_anchor"] is True
    assert task["targeted_complement_count_vs_original_anchor"] > 0
    assert task["targeted_complement_count_vs_augmented_anchor"] == 0
    assert task["failure_class"] == "anchor_shift_suppression"


def _task(task_id):
    return {
        "answer": None,
        "answer_type": "rubric",
        "family": "planning",
        "max_new_tokens": 64,
        "prompt": f"Decide the customer demo measurement scope for {task_id}.",
        "rubric_items": ["run baseline measurement", "limit demo scope"],
        "scorer": "planning_rubric_v1",
        "task_id": task_id,
    }


def _replay_task(task_id, anchor_id, anchor_score):
    return {
        "anchor_score": anchor_score,
        "anchor_trajectory_id": anchor_id,
        "task_id": task_id,
    }


def _row(task_id, *, score, text):
    return {
        "candidate_key": "llada",
        "generation_stage": "repair_candidate",
        "task_id": task_id,
        "task_score": {"score": score},
        "text": text,
    }
