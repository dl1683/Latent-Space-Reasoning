import json

from experiments.summarize_diffusion_spend_policy_decision import (
    build_spend_policy_decision,
    render_markdown,
)


def test_spend_policy_decision_prefers_repairable_candidate_aware(tmp_path):
    spend_eval = tmp_path / "spend.json"
    repairable_scores = tmp_path / "repairable.json"
    calibrated_scores = tmp_path / "calibrated.json"
    spend_eval.write_text(json.dumps({"rows": _rows()}), encoding="utf-8")
    repairable_scores.write_text(json.dumps(_scores("repairable", 1.0, 0.08)), encoding="utf-8")
    calibrated_scores.write_text(json.dumps(_scores("calibrated", 0.5, 0.03)), encoding="utf-8")

    decision = build_spend_policy_decision(
        calibrated_candidate_aware_scores_path=calibrated_scores,
        repairable_candidate_aware_scores_path=repairable_scores,
        spend_eval_paths=(spend_eval,),
    )

    assert (
        decision["summary"]["incumbent_policy_id"]
        == "denoise_phase_repairability_plus_candidate_aware_promotion_v1"
    )
    assert decision["summary"]["repairable_false_negative_count"] == 0
    assert decision["summary"]["calibrated_missed_profitable_count"] == 1
    assert (
        decision["live_v6_policy_scores"]["repairable_minus_calibrated"][
            "incremental_lift_per_extra_generation"
        ]
        == 0.1
    )


def test_render_markdown_includes_policy_and_cost_tables(tmp_path):
    spend_eval = tmp_path / "spend.json"
    repairable_scores = tmp_path / "repairable.json"
    calibrated_scores = tmp_path / "calibrated.json"
    spend_eval.write_text(json.dumps({"rows": _rows()}), encoding="utf-8")
    repairable_scores.write_text(json.dumps(_scores("repairable", 1.0, 0.08)), encoding="utf-8")
    calibrated_scores.write_text(json.dumps(_scores("calibrated", 0.5, 0.03)), encoding="utf-8")
    decision = build_spend_policy_decision(
        calibrated_candidate_aware_scores_path=calibrated_scores,
        repairable_candidate_aware_scores_path=repairable_scores,
        spend_eval_paths=(spend_eval,),
    )

    markdown = render_markdown(decision)

    assert "# Diffusion Spend Policy Decision" in markdown
    assert "repairable_denoise_spend" in markdown
    assert "Live V6 Cost Comparison" in markdown
    assert "candidate_aware_promotion_v1" in markdown


def _rows():
    return [
        {
            "calibrated_availability_prediction": True,
            "decomposed_prediction": True,
            "learned_availability_prediction": True,
            "profitable": True,
            "repair_lift": 0.2,
            "single_repairability_prediction": True,
            "task_id": "plan_a",
            "trajectory_relative_prediction": True,
        },
        {
            "calibrated_availability_prediction": False,
            "decomposed_prediction": True,
            "learned_availability_prediction": False,
            "profitable": True,
            "repair_lift": 0.1,
            "single_repairability_prediction": True,
            "task_id": "plan_b",
            "trajectory_relative_prediction": False,
        },
        {
            "calibrated_availability_prediction": True,
            "decomposed_prediction": True,
            "learned_availability_prediction": False,
            "profitable": False,
            "repair_lift": 0.0,
            "single_repairability_prediction": True,
            "task_id": "plan_c",
            "trajectory_relative_prediction": True,
        },
    ]


def _scores(run_id, extra_generation, lift_vs_fixed):
    return {
        "all_generation_count": 3,
        "oracle_headroom_vs_repair": 0.0,
        "repair_generation_budget_delta_vs_evolved": extra_generation,
        "repair_selector": "candidate_aware_promotion_v1",
        "repair_spend_trigger": run_id,
        "repair_task_delta_per_extra_generation_vs_evolved": lift_vs_fixed / extra_generation,
        "repair_task_delta_vs_fixed": lift_vs_fixed,
        "repair_task_delta_vs_random": lift_vs_fixed + 0.02,
        "repair_task_delta_vs_trajectory": lift_vs_fixed - 0.01,
        "run_id": run_id,
    }
