import json

from experiments.analyze_diffusion_spend_counterexamples import (
    build_counterexample_workbench,
    render_markdown,
)


def test_counterexample_workbench_splits_false_positive_and_negative(tmp_path):
    spend_fit = tmp_path / "spend.json"
    promotion = tmp_path / "promotion.json"
    spend_fit.write_text(json.dumps(_spend_fit()), encoding="utf-8")
    promotion.write_text(json.dumps(_promotion_targets()), encoding="utf-8")

    workbench = build_counterexample_workbench(
        promotion_targets_path=promotion,
        spend_gate_fit_path=spend_fit,
    )

    assert workbench["summary"]["counterexample_count"] == 2
    lessons = {row["task_id"]: row["lesson"] for row in workbench["counterexamples"]}
    assert "prompt-gap floor is too brittle" in lessons["missed_low_gap"]
    assert "high-gap no-lift repair" in lessons["wasted_high_gap"]
    assert {q["question_id"] for q in workbench["active_questions"]} >= {
        "high_gap_waste_probe",
        "low_gap_value_probe",
    }


def test_counterexample_workbench_markdown_includes_contract(tmp_path):
    spend_fit = tmp_path / "spend.json"
    promotion = tmp_path / "promotion.json"
    spend_fit.write_text(json.dumps(_spend_fit()), encoding="utf-8")
    promotion.write_text(json.dumps(_promotion_targets()), encoding="utf-8")

    markdown = render_markdown(
        build_counterexample_workbench(
            promotion_targets_path=promotion,
            spend_gate_fit_path=spend_fit,
        )
    )

    assert "# Diffusion Spend Counterexample Workbench" in markdown
    assert "Next Controller Contract" in markdown
    assert "candidate_aware_promotion_v1" in markdown


def _spend_fit():
    return {
        "rows": [
            _row("missed_low_gap", profitable=True, gap=3, lift=0.2),
            _row("wasted_high_gap", profitable=False, gap=12, lift=0.0),
            _row("true_positive", profitable=True, gap=8, lift=0.1),
            _row("true_negative", profitable=False, gap=4, lift=0.0),
        ],
        "summary": {
            "best_rule": {
                "conditions": ["prompt_gap_count >= 8.000000"],
                "error_count": 2,
                "false_negative_count": 1,
                "false_positive_count": 1,
                "missed_profitable_tasks": ["missed_low_gap"],
                "no_lift_selected_tasks": ["wasted_high_gap"],
                "positive_lift_covered": 0.1,
                "predictions": {
                    "missed_low_gap": False,
                    "true_negative": False,
                    "true_positive": True,
                    "wasted_high_gap": True,
                },
                "rule_id": "repairable_if_prompt_gap_count_ge_8p000000",
            },
            "profitable_count": 2,
            "target_count": 4,
        },
    }


def _promotion_targets():
    return {
        "rows": [
            {"repair_selector_edge": 0.0, "task_id": "missed_low_gap"},
            {"repair_selector_edge": 0.0, "task_id": "wasted_high_gap"},
            {"repair_selector_edge": 0.1, "task_id": "true_positive"},
            {"repair_selector_edge": 0.0, "task_id": "true_negative"},
        ],
        "summary": {
            "candidate_aware_promotion_error_count": 0,
            "positive_tasks": ["missed_low_gap", "true_positive"],
        },
    }


def _row(task_id, *, profitable, gap, lift):
    return {
        "first_repairable_step": 4,
        "profitable": profitable,
        "prompt_gap_count": gap,
        "repair_lift": lift,
        "source_quality": 0.25,
        "source_task_delta_vs_trajectory": 0.0,
        "task_id": task_id,
    }
