import json

from experiments.analyze_latent_aggregation_multi_aspect_v2_headroom import (
    analyze_headroom,
    render_markdown,
)


def test_multi_aspect_headroom_counts_rubric_and_dimension_complements(tmp_path):
    raw = tmp_path / "raw.jsonl"
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"task_ids": ["plan_a"]}), encoding="utf-8")
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    _record(
                        "plan_a",
                        "anchor",
                        score=0.5,
                        rubric_hits=[True, False],
                        specificity=0.2,
                        risk_awareness=0.0,
                    )
                ),
                json.dumps(
                    _record(
                        "plan_a",
                        "candidate",
                        score=0.45,
                        rubric_hits=[True, True],
                        specificity=0.4,
                        risk_awareness=0.3,
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = analyze_headroom(raw_path=raw, freeze_path=freeze)
    task = result["tasks"][0]
    markdown = render_markdown(result)

    assert task["rubric_complement_count"] == 1
    assert task["dimension_complement_count"] == 2
    assert task["complement_source_count"] == 1
    assert result["summary"]["tasks_with_dimension_complement"] == 1
    assert "candidate-level material" in markdown


def _record(task_id, candidate_key, *, score, rubric_hits, specificity, risk_awareness):
    items = ["preserve baseline", "measure risk"]
    return {
        "candidate_key": candidate_key,
        "generation_stage": "candidate_generation",
        "schedule": {"name": "fixed"},
        "task": {"task_id": task_id},
        "task_score": {
            "details": {
                "causal_diagnosis": 0.0,
                "constraint_handling": 0.0,
                "risk_awareness": risk_awareness,
                "rubric_hits": [
                    {"hit": hit, "item": item}
                    for hit, item in zip(rubric_hits, items, strict=True)
                ],
                "specificity": specificity,
            },
            "score": score,
        },
    }
