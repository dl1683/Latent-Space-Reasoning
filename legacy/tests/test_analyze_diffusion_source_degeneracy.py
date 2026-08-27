import json

from experiments.analyze_diffusion_source_degeneracy import (
    build_source_degeneracy_audit,
    render_markdown,
    source_degeneracy_features,
)


def test_source_degeneracy_features_detect_repetition_and_meta_leakage():
    clean = source_degeneracy_features(
        "Measure latency, compare the baseline, and record rollback criteria."
    )
    degenerate = source_degeneracy_features(
        "Prompt ID,, Prompt ID,, repair repair repair seed seed run run run."
    )

    assert degenerate["adjacent_repeat_count"] > clean["adjacent_repeat_count"]
    assert degenerate["max_adjacent_repeat_run"] >= 3
    assert degenerate["comma_density"] > clean["comma_density"]
    assert degenerate["meta_leakage_count"] > clean["meta_leakage_count"]
    assert degenerate["degeneracy_score"] > clean["degeneracy_score"]


def test_source_degeneracy_audit_joins_value_rows_and_reports_clusters(tmp_path):
    scores = tmp_path / "slice_scores.json"
    raw = tmp_path / "slice_raw.jsonl"
    eval_path = tmp_path / "slice_eval.json"
    scores.write_text(
        json.dumps(
            {
                "repair_spend_gate_rows": [
                    {"source_control": "low_confidence_32", "task_id": "plan_a"},
                    {"source_control": "low_confidence_32", "task_id": "plan_b"},
                ]
            }
        ),
        encoding="utf-8",
    )
    raw.write_text(
        "\n".join(
            [
                json.dumps(_raw("plan_a", "Prompt ID,, repair repair repair seed seed.")),
                json.dumps(_raw("plan_b", "Measure latency and compare rollback criteria.")),
            ]
        ),
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps(
            {
                "inputs": {"all_repairable_scores": str(scores)},
                "rows": [
                    {"profitable": True, "repair_lift": 0.2, "task_id": "plan_a"},
                    {"profitable": False, "repair_lift": 0.0, "task_id": "plan_b"},
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = build_source_degeneracy_audit(spend_eval_paths=(eval_path,))
    markdown = render_markdown(audit)

    assert audit["schema"] == "diffusion_source_degeneracy.v1"
    assert audit["summary"]["row_count"] == 2
    assert audit["clusters"]["high_degeneracy"]["task_ids"] == ["plan_a"]
    assert audit["signal_summaries"]
    assert "Degeneracy Clusters" in markdown


def _raw(task_id, text):
    return {
        "generation_stage": "candidate_generation",
        "prompt": "Plan a validation with risk and rollback.",
        "schedule": {"name": "low_confidence_32"},
        "task": {"task_id": task_id},
        "text": text,
    }
