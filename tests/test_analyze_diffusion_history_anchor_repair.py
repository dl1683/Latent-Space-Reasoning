"""Tests for history-anchor repair audit generation."""

import json

import pytest

from experiments.analyze_diffusion_history_anchor_repair import (
    build_history_anchor_audit,
    render_markdown,
)


def test_history_anchor_audit_compares_final_and_history_sources(tmp_path):
    final_scores = tmp_path / "final_scores.json"
    history_scores = tmp_path / "history_scores.json"
    final_raw = tmp_path / "final_raw.jsonl"
    final_scores.write_text(
        json.dumps(
            {
                "by_family_arm": {
                    "planning": {
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2.5,
                            "mean_task_score": 0.60,
                        }
                    }
                },
                "comparison_rows": [
                    {
                        "fixed_task_score": 0.40,
                        "random_task_score": 0.30,
                        "repair_control": "constraint_gap_span_repair",
                        "repair_selector_score": 0.60,
                        "repair_task_score": 0.70,
                        "task_id": "plan_001",
                    },
                    {
                        "fixed_task_score": 0.50,
                        "random_task_score": 0.35,
                        "repair_control": "constraint_gap_span_repair",
                        "repair_selector_score": 0.10,
                        "repair_task_score": 0.50,
                        "task_id": "plan_002",
                    },
                ],
                "repair_candidate_summary": {
                    "constraint_gap_span_repair": {
                        "count": 1,
                        "mean_span_fallback_used": 0.0,
                        "mean_span_literal_target_found": 1.0,
                        "source_states": "final",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    history_scores.write_text(
        json.dumps(
            {
                "by_family_arm": {
                    "planning": {
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2.5,
                            "mean_task_score": 0.55,
                        }
                    }
                },
                "comparison_rows": [
                    {
                        "fixed_task_score": 0.40,
                        "random_task_score": 0.30,
                        "repair_control": "constraint_gap_span_history_repair",
                        "repair_selector_score": 0.50,
                        "repair_source_history_step": 12,
                        "repair_source_state": "history",
                        "repair_task_score": 0.65,
                        "task_id": "plan_001",
                    },
                    {
                        "fixed_task_score": 0.50,
                        "random_task_score": 0.35,
                        "repair_control": "constraint_gap_span_history_repair",
                        "repair_selector_score": 0.10,
                        "repair_source_history_step": "",
                        "repair_source_state": "",
                        "repair_task_score": 0.50,
                        "task_id": "plan_002",
                    },
                ],
                "repair_candidate_summary": {
                    "constraint_gap_span_history_repair": {
                        "count": 1,
                        "mean_span_fallback_used": 0.0,
                        "mean_span_literal_target_found": 1.0,
                        "source_states": "history",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention. Decide which measurements to "
        "collect so tomorrow's result is publishable even if the intervention fails."
    )
    final_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful result (either the baseline or the "
        "intervention) can be published tomorrow, ensuring a publishable result even "
        "if the intervention fails."
    )
    history_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful ( the baseline or the intervention) "
        "can be published tomorrow, ensuring a publishable result even if the "
        "intervention fails."
    )
    final_raw.write_text(
        json.dumps(
            {
                "generation_stage": "candidate_generation",
                "history_samples": [{"generated_token_ids": [1, 2, 3], "step": 31, "text": history_text}],
                "prompt": prompt,
                "schedule": {"name": "low_confidence_32"},
                "task": {"task_id": "plan_001"},
                "text": final_text,
                "trajectory_summary": {
                    "samples": [
                        {
                            "mask_count": 1,
                            "step": 31,
                            "visible_chars": len(history_text.strip()),
                            "visible_text": history_text,
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = build_history_anchor_audit(
        final_source_scores_path=final_scores,
        history_anchor_scores_path=history_scores,
        final_source_raw_path=final_raw,
    )
    rendered = render_markdown(audit)

    assert audit["schema"] == "diffusion_history_anchor_repair_audit.v1"
    assert audit["summary"]["score_delta_history_vs_final"] == pytest.approx(-0.05)
    assert audit["summary"]["dual_anchor_selector_score"] == pytest.approx(0.60)
    assert audit["summary"]["dual_anchor_selector_relative_cost"] == pytest.approx(3.0)
    assert audit["summary"]["anchor_choice_counts"] == {"final": 2}
    assert audit["summary"]["pre_generation_anchor_choice_counts"] == {"history": 1, "final": 1}
    assert audit["summary"]["pre_generation_anchor_selector_score"] == pytest.approx(0.575)
    assert audit["summary"]["pre_generation_anchor_selector_relative_cost"] == pytest.approx(2.5)
    assert audit["summary"]["history_span_localized"] == 1.0
    assert audit["summary"]["classification_counts"] == {
        "history_positive_but_loses_final_context": 1,
        "history_matches_or_beats_final": 1,
    }
    assert "Diffusion History-Anchor Repair Audit" in rendered
    assert "Pre-generation anchor selector score" in rendered
    assert "plan_001" in rendered
