import json

from experiments.analyze_latent_aggregation_multi_aspect_v2_coverage_gap import (
    analyze_coverage_gap,
    render_markdown,
)


def test_multi_aspect_coverage_gap_classifies_blockers(tmp_path):
    raw = tmp_path / "raw.jsonl"
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"task_ids": ["blank", "near", "selected"]}), encoding="utf-8")
    rows = [
        _record("blank", "anchor", score=0.5, text="anchor", specificity=0.4),
        _record("blank", "candidate", score=0.0, text="   ", specificity=0.0),
        _record("near", "anchor", score=0.5, text="anchor", specificity=0.20),
        _record("near", "candidate", score=0.4, text="candidate", specificity=0.23),
        _record("selected", "anchor", score=0.5, text="anchor", specificity=0.20),
        _record("selected", "candidate", score=0.4, text="candidate", specificity=0.31),
    ]
    raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = analyze_coverage_gap(raw_path=raw, freeze_path=freeze)
    markdown = render_markdown(result)
    blockers = {task["task_id"]: task["coverage_blocker"] for task in result["tasks"]}

    assert blockers["blank"] == "all_non_anchor_candidates_blank"
    assert blockers["near"] == "positive_but_below_threshold"
    assert blockers["selected"] == "has_selected_complement"
    assert result["summary"]["tasks_with_selected_complement"] == 1
    assert result["summary"]["no_complement_tasks_with_near_miss"] == 1
    assert "targeted aspect-deficit probes" in markdown


def _record(task_id, candidate_key, *, score, text, specificity):
    return {
        "candidate_key": candidate_key,
        "generation_stage": "candidate_generation",
        "schedule": {"name": "fixed"},
        "task": {"task_id": task_id},
        "task_score": {
            "details": {
                "causal_diagnosis": 0.0,
                "constraint_handling": 0.0,
                "risk_awareness": 0.0,
                "rubric_hits": [{"hit": False, "item": "measure risk"}],
                "specificity": specificity,
            },
            "score": score,
        },
        "text": text,
    }
