import json

import pytest

from experiments.analyze_diffusion_budget_policy_loss import (
    build_budget_policy_loss,
    render_markdown,
)


def test_budget_policy_loss_builds_cost_aware_task_targets(tmp_path):
    budget_map_path = tmp_path / "budget_map.json"
    budget_map_path.write_text(
        json.dumps(
            {
                "repair_generation_budget": 1.0,
                "runner_mode_rows": [
                    {"cap": 1, "mode": "floor"},
                    {"cap": 2, "mode": "cheap"},
                    {"cap": 4, "mode": "frontier"},
                ],
                "task_rows": [
                    _task("plan_001", first_step=2, task_lift=0.40, selected=True),
                    _task("plan_002", first_step=4, task_lift=0.20, selected=True),
                    _task("plan_003", first_step=None, task_lift=0.0, selected=False),
                    _task("plan_004", first_step=3, task_lift=0.0, selected=False),
                ],
                "transition_rows": [
                    {
                        "active_repair_count": 0,
                        "active_repair_tasks": [],
                        "cap": 1,
                        "cap_range": "1-1",
                        "newly_active_tasks": [],
                        "predicted_relative_cost": 2.0,
                        "predicted_score": 0.30,
                    },
                    {
                        "active_repair_count": 1,
                        "active_repair_tasks": ["plan_001"],
                        "cap": 2,
                        "cap_range": "2-3",
                        "newly_active_tasks": ["plan_001"],
                        "predicted_relative_cost": 2.25,
                        "predicted_score": 0.40,
                    },
                    {
                        "active_repair_count": 2,
                        "active_repair_tasks": ["plan_001", "plan_002"],
                        "cap": 4,
                        "cap_range": "4+",
                        "newly_active_tasks": ["plan_002"],
                        "predicted_relative_cost": 2.50,
                        "predicted_score": 0.45,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = build_budget_policy_loss(
        budget_map_path=budget_map_path,
        cost_penalties=(0.0, 0.35, 0.45),
    )
    rendered = render_markdown(audit)

    assert audit["schema"] == "diffusion_budget_policy_loss.v1"
    assert audit["planning_task_count"] == 4
    assert audit["marginal_relative_cost_per_repair"] == pytest.approx(0.25)
    plan_001 = _target(audit, "plan_001")
    plan_002 = _target(audit, "plan_002")
    plan_003 = _target(audit, "plan_003")
    assert plan_001["break_even_lambda"] == pytest.approx(0.40)
    assert plan_001["minimum_phase_budget"] == "cheap"
    assert plan_002["break_even_lambda"] == pytest.approx(0.20)
    assert plan_002["minimum_phase_budget"] == "frontier"
    assert plan_003["target"] == "skip_repair"
    assert audit["transition_value_rows"][1]["marginal_roi"] == pytest.approx(0.40)
    assert audit["cap_policy_rows"][1]["cap_range"] == "2-3"
    assert audit["task_policy_rows"][1]["selected_tasks"] == ["plan_001"]
    assert audit["task_policy_rows"][1]["gain_vs_best_cap_policy"] == pytest.approx(0.0)
    assert audit["cap_policy_rows"][2]["cap_range"] == "1-1"
    assert audit["task_policy_rows"][2]["selected_tasks"] == []
    assert "utility(task, lambda)" in rendered
    assert "Diffusion Budget Policy Loss" in rendered


def _target(audit, task_id):
    for row in audit["target_rows"]:
        if row["task_id"] == task_id:
            return row
    raise AssertionError(f"missing target {task_id}")


def _task(task_id, *, first_step, task_lift, selected):
    return {
        "first_repairable_coverage": 0.4 if first_step is not None else None,
        "first_repairable_step": first_step,
        "first_repairable_step_fraction": 0.5 if first_step is not None else None,
        "gate_reason": "denoise_phase_repairable" if selected else "source_quality_ok",
        "prompt_coverage": 0.5,
        "prompt_gap_count": 4,
        "repair_control": "constraint_gap_span_phase_final_preserve_seeded_gated_repair" if selected else "fixed",
        "repair_lift_vs_trajectory": task_lift,
        "repair_score": 0.5 + task_lift,
        "selected_repair": selected,
        "source_needs_repair": selected,
        "task_id": task_id,
        "trajectory_score": 0.5,
    }
