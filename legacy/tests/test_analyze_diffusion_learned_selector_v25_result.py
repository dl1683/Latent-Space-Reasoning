import json

import experiments.analyze_diffusion_learned_selector_v25_result as analyze


def test_learned_selector_result_reports_heldout_recall_failure(tmp_path, monkeypatch):
    train_paths = {}
    for slice_id in ("v21", "v22", "v23", "v24"):
        target = tmp_path / f"{slice_id}.json"
        target.write_text(json.dumps(_training_targets(slice_id)), encoding="utf-8")
        train_paths[slice_id] = target
    monkeypatch.setattr(analyze, "TRAINING_TARGETS", train_paths)
    scores = tmp_path / "scores.json"
    targets = tmp_path / "targets.json"
    scores.write_text(json.dumps(_scores()), encoding="utf-8")
    targets.write_text(json.dumps(_targets()), encoding="utf-8")

    result = analyze.build_result_summary(scores_path=scores, targets_path=targets)

    assert result["summary"]["learned_selected_positive_count"] == 1
    assert result["summary"]["learned_selected_waste_count"] == 0
    assert result["summary"]["unchanged_selected_positive_count"] == 2
    assert result["decision"]["status"] == "heldout_recall_failed"


def _training_targets(slice_id):
    return {
        "rows": [
            {
                "candidate_lift_vs_trajectory": 0.03,
                "max_span_target_score": 0.0,
                "planning_quality_delta_vs_source": 0.2,
                "repair": "history_prefix_25_repair",
                "task_id": f"{slice_id}_a",
            },
            {
                "candidate_lift_vs_trajectory": 0.04,
                "max_span_target_score": 2.0,
                "planning_quality_delta_vs_source": 0.02,
                "repair": "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                "task_id": f"{slice_id}_b",
            },
            {
                "candidate_lift_vs_trajectory": -0.01,
                "max_span_target_score": 0.0,
                "planning_quality_delta_vs_source": 0.05,
                "repair": "history_prefix_25_repair",
                "task_id": f"{slice_id}_c",
            },
        ]
    }


def _scores():
    return {
        "all_generation_count": 8,
        "comparison_rows": [
            {"repair_control": "history_prefix_25_repair", "task_id": "plan_low"},
            {
                "repair_control": "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                "task_id": "plan_final",
            },
        ],
        "oracle_headroom_vs_repair": 0.0,
        "repair_task_delta_per_extra_generation_vs_evolved": 0.01,
        "repair_task_delta_vs_evolved": 0.02,
        "run_id": "diffusion-test",
    }


def _targets():
    return {
        "rows": [
            {
                "candidate_lift_vs_trajectory": 0.01,
                "max_span_target_score": 0.0,
                "planning_quality_delta_vs_source": 0.01,
                "repair": "history_prefix_25_repair",
                "task_id": "plan_low",
            },
            {
                "candidate_lift_vs_trajectory": 0.04,
                "max_span_target_score": 2.0,
                "planning_quality_delta_vs_source": 0.04,
                "repair": "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                "task_id": "plan_final",
            },
        ],
        "summary": {"candidate_aware_promotion_error_count": 0},
    }
