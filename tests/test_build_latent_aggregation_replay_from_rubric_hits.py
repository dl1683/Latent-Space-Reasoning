import json

from experiments.build_latent_aggregation_replay_from_rubric_hits import build_replay, render_markdown


def test_rubric_replay_finds_oracle_component_union_headroom(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    _record(
                        "plan_a",
                        "fixed",
                        [True, True, False, False],
                        score=0.5,
                    )
                ),
                json.dumps(
                    _record(
                        "plan_a",
                        "random",
                        [False, False, True, True],
                        score=0.5,
                    )
                ),
            ]
        ),
        encoding="utf-8",
    )

    result = build_replay(raw_path=raw)
    task = result["tasks"][0]
    markdown = render_markdown(result)

    assert result["summary"]["component_row_count"] == 8
    assert task["best_single_score"] == 0.5
    assert task["aggregate_score"] == 1.0
    assert task["decision"]["status"] == "promoted_local_scout"
    assert "oracle_replay_not_inference_time" in markdown


def test_rubric_replay_blocks_when_union_does_not_beat_best_single(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps(_record("plan_b", "fixed", [True, True], score=1.0)),
                json.dumps(_record("plan_b", "random", [True, False], score=0.5)),
            ]
        ),
        encoding="utf-8",
    )

    result = build_replay(raw_path=raw)
    task = result["tasks"][0]

    assert task["aggregate_score"] == task["best_single_score"]
    assert task["decision"]["status"] == "blocked_no_score_lift"
    assert result["summary"]["promoted_task_count"] == 0


def _record(task_id, schedule, hits, *, score):
    return {
        "candidate_key": "model",
        "generation_stage": "candidate_generation",
        "schedule": {"name": schedule},
        "task": {"family": "planning", "task_id": task_id},
        "task_score": {
            "details": {
                "rubric_hits": [
                    {"hit": hit, "item": f"rubric item {index}"}
                    for index, hit in enumerate(hits)
                ]
            },
            "score": score,
        },
        "text": f"{schedule} answer",
    }
