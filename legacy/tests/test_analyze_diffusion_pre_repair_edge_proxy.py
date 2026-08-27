import json

from experiments.analyze_diffusion_pre_repair_edge_proxy import (
    build_pre_repair_edge_proxy_audit,
    render_markdown,
)


def test_pre_repair_edge_proxy_joins_promotion_source_and_spend_rows(tmp_path):
    scores = tmp_path / "slice_scores.json"
    raw = tmp_path / "slice_raw.jsonl"
    spend_eval = tmp_path / "slice_eval.json"
    promotion = tmp_path / "promotion.json"
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
                json.dumps(_raw("plan_a", "Measure risk and define rollback criteria.")),
                json.dumps(_raw("plan_b", "Prompt ID,, repair repair repair seed seed.")),
            ]
        ),
        encoding="utf-8",
    )
    spend_eval.write_text(
        json.dumps(
            {
                "inputs": {"all_repairable_scores": str(scores)},
                "rows": [
                    _spend("plan_a", profitable=True, source_quality=0.2, gap=4, first_step=3),
                    _spend("plan_b", profitable=False, source_quality=0.4, gap=12, first_step=4),
                ],
            }
        ),
        encoding="utf-8",
    )
    promotion.write_text(
        json.dumps(
            {
                "rows": [
                    _promotion("plan_a", promote=True, lift=0.2, span_score=2.0),
                    _promotion("plan_b", promote=False, lift=0.0, span_score=0.0),
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = build_pre_repair_edge_proxy_audit(
        promotion_target_paths=(promotion,),
        spend_eval_paths=(spend_eval,),
    )
    markdown = render_markdown(audit)

    assert audit["schema"] == "diffusion_pre_repair_edge_proxy.v1"
    assert audit["summary"]["row_count"] == 2
    assert audit["rows"][1]["degeneracy_score"] > audit["rows"][0]["degeneracy_score"]
    assert audit["single_feature_rules"]
    assert "Pre-Repair Edge Proxy" in markdown


def _raw(task_id, text):
    return {
        "generation_stage": "candidate_generation",
        "prompt": "Plan a validation with risk and rollback.",
        "schedule": {"name": "low_confidence_32"},
        "task": {"task_id": task_id},
        "text": text,
    }


def _spend(task_id, *, profitable, source_quality, gap, first_step):
    return {
        "first_repairable_step": first_step,
        "profitable": profitable,
        "prompt_gap_count": gap,
        "repair_lift": 0.1 if profitable else 0.0,
        "source_quality": source_quality,
        "source_task_delta_vs_trajectory": 0.0,
        "task_id": task_id,
    }


def _promotion(task_id, *, promote, lift, span_score):
    return {
        "candidate_lift_vs_trajectory": lift,
        "max_span_target_score": span_score,
        "min_span_source_relative_preservation": 0.8 if promote else 0.1,
        "planning_quality_delta_vs_source": lift,
        "promote_vs_trajectory": promote,
        "prompt_gap_term_count": 4 if promote else 12,
        "repair_selector_edge": lift,
        "task_id": task_id,
    }
