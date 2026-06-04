import json

from experiments.analyze_diffusion_source_value_signals import (
    build_source_value_signal_audit,
    render_markdown,
)


def test_source_value_signal_audit_joins_eval_scores_and_raw_source(tmp_path):
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
                json.dumps(_raw("plan_a", "Measure risk, compare options, and define rollback criteria.")),
                json.dumps(_raw("plan_b", "No.")),
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

    audit = build_source_value_signal_audit(spend_eval_paths=(eval_path,))
    markdown = render_markdown(audit)

    assert audit["summary"]["row_count"] == 2
    assert audit["rows"][0]["word_count"] > audit["rows"][1]["word_count"]
    assert audit["signal_summaries"]
    assert "Signal Separation" in markdown


def _raw(task_id, text):
    return {
        "generation_stage": "candidate_generation",
        "prompt": "Plan a validation with risk and rollback.",
        "schedule": {"name": "low_confidence_32"},
        "task": {"task_id": task_id},
        "text": text,
    }
