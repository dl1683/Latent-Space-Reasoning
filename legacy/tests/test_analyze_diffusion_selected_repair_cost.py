import json

from experiments.analyze_diffusion_selected_repair_cost import (
    build_selected_repair_cost_audit,
    render_markdown,
)


def test_selected_repair_cost_audit_favors_precision_when_aggregate_ties(tmp_path):
    hook = tmp_path / "hook.json"
    control = tmp_path / "control.json"
    targets = tmp_path / "targets.json"
    hook.write_text(json.dumps(_scores("hook-run", "generated_repair_value_v1", ["plan_a"])), encoding="utf-8")
    control.write_text(
        json.dumps(_scores("control-run", "planning_quality", ["plan_a", "plan_b", "plan_c"])),
        encoding="utf-8",
    )
    targets.write_text(json.dumps(_targets()), encoding="utf-8")

    audit = build_selected_repair_cost_audit(
        hook_scores_path=hook,
        control_scores_path=control,
        targets_path=targets,
        substantial_lift=0.02,
    )
    markdown = render_markdown(audit)

    assert audit["selected_policy"]["policy_id"] == "generated_repair_value_v1"
    assert audit["summary"]["aggregate_utility_parity"] is True
    assert audit["summary"]["control_only_waste_tasks"] == ["plan_b", "plan_c"]
    assert audit["policies"][1]["waste_selected_count"] == 2
    assert "any positive selected-output cost favors the hook" in markdown


def _scores(run_id, selector, selected_tasks):
    rows = []
    for task_id in ["plan_a", "plan_b", "plan_c"]:
        rows.append(
            {
                "repair_selection_reason": (
                    f"max_{selector}_score_repair_pool"
                    if task_id in selected_tasks
                    else "repair_margin_guard_kept_evolved_0.020"
                ),
                "task_id": task_id,
            }
        )
    return {
        "arms": {"repair_selected": {"count": 3}},
        "comparison_rows": rows,
        "repair_promotion_margin": 0.02,
        "repair_selector": selector,
        "repair_task_delta_per_extra_generation_vs_evolved": 0.1,
        "repair_task_delta_vs_evolved": 0.03,
        "run_id": run_id,
    }


def _targets():
    return {
        "rows": [
            {"candidate_lift_vs_trajectory": 0.05, "task_id": "plan_a"},
            {"candidate_lift_vs_trajectory": 0.0, "task_id": "plan_b"},
            {"candidate_lift_vs_trajectory": 0.01, "task_id": "plan_c"},
        ]
    }
