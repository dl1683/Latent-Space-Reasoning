import json

from experiments.analyze_latent_aggregation_multi_aspect_v3_coverage_gap import (
    analyze_coverage_gap,
    render_markdown,
)


def test_v3_coverage_gap_uses_extra_raw_sources(tmp_path):
    raw = tmp_path / "raw.jsonl"
    extra = tmp_path / "extra.jsonl"
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"task_ids": ["plan_a"]}), encoding="utf-8")
    raw.write_text(
        json.dumps(_record("plan_a", "anchor", score=0.5, text="anchor", specificity=0.2))
        + "\n",
        encoding="utf-8",
    )
    extra.write_text(
        json.dumps(
            _record(
                "plan_a",
                "counterfactual_probe",
                score=0.2,
                text="candidate",
                specificity=0.4,
                generation_stage="counterfactual_probe",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = analyze_coverage_gap(raw_path=raw, freeze_path=freeze, extra_raw_paths=[extra])
    markdown = render_markdown(result)

    assert result["summary"]["tasks_with_selected_complement"] == 1
    assert result["summary"]["complement_source_counts"]["counterfactual_probe"] == 1
    assert result["inputs"]["source_record_counts"][str(extra)] == 1
    assert "Complement source counts" in markdown


def test_v4_coverage_gap_marks_fresh_predeclared_boundary(tmp_path):
    raw = tmp_path / "raw.jsonl"
    extra = tmp_path / "extra.jsonl"
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "schema": "latent_aggregation_multi_aspect_v4_freeze.v1",
                "task_ids": ["plan_b"],
            }
        ),
        encoding="utf-8",
    )
    raw.write_text(
        json.dumps(_record("plan_b", "anchor", score=0.5, text="anchor", specificity=0.2))
        + "\n",
        encoding="utf-8",
    )
    extra.write_text(
        json.dumps(
            _record(
                "plan_b",
                "counterfactual_probe",
                score=0.2,
                text="candidate",
                specificity=0.4,
                generation_stage="counterfactual_probe",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = analyze_coverage_gap(raw_path=raw, freeze_path=freeze, extra_raw_paths=[extra])
    markdown = render_markdown(result)

    assert result["evidence_boundary"]["status"] == "fresh_predeclared_multi_source_v4_coverage_gap"
    assert "# Latent Aggregation Multi-Aspect V4 Coverage Gap" in markdown


def test_v5_coverage_gap_marks_fresh_predeclared_boundary(tmp_path):
    raw = tmp_path / "raw.jsonl"
    extra = tmp_path / "extra.jsonl"
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "schema": "latent_aggregation_multi_aspect_v5_freeze.v1",
                "task_ids": ["plan_c"],
            }
        ),
        encoding="utf-8",
    )
    raw.write_text(
        json.dumps(_record("plan_c", "anchor", score=0.5, text="anchor", specificity=0.2))
        + "\n",
        encoding="utf-8",
    )
    extra.write_text(
        json.dumps(
            _record(
                "plan_c",
                "counterfactual_probe",
                score=0.2,
                text="candidate",
                specificity=0.4,
                generation_stage="counterfactual_probe",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = analyze_coverage_gap(raw_path=raw, freeze_path=freeze, extra_raw_paths=[extra])
    markdown = render_markdown(result)

    assert result["evidence_boundary"]["status"] == "fresh_predeclared_multi_source_v5_coverage_gap"
    assert "# Latent Aggregation Multi-Aspect V5 Coverage Gap" in markdown
    assert "fresh v5 48-task replay" in markdown


def _record(
    task_id,
    candidate_key,
    *,
    score,
    text,
    specificity,
    generation_stage="candidate_generation",
):
    return {
        "candidate_key": candidate_key,
        "generation_stage": generation_stage,
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
