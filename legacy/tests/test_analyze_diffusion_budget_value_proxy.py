import json

import pytest

from experiments.analyze_diffusion_budget_value_proxy import (
    build_budget_value_proxy_audit,
    render_markdown,
)


def test_budget_value_proxy_audit_selects_label_free_quality_gate(tmp_path):
    targets_path = tmp_path / "targets.jsonl"
    targets_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _target(
                    "plan_001",
                    aggregate_lift=0.10,
                    marginal_cost=0.25,
                    trajectory_score=0.30,
                    first_step=2,
                    prompt_gap=5,
                    source_needs_repair=True,
                ),
                _target(
                    "plan_002",
                    aggregate_lift=0.05,
                    marginal_cost=0.25,
                    trajectory_score=0.55,
                    first_step=4,
                    prompt_gap=4,
                    source_needs_repair=True,
                ),
                _target(
                    "plan_003",
                    aggregate_lift=0.0,
                    marginal_cost=0.25,
                    trajectory_score=0.20,
                    first_step=5,
                    prompt_gap=12,
                    source_needs_repair=True,
                ),
                _target(
                    "plan_004",
                    aggregate_lift=0.0,
                    marginal_cost=0.25,
                    trajectory_score=0.10,
                    first_step=None,
                    prompt_gap=1,
                    source_needs_repair=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    audit = build_budget_value_proxy_audit(
        loss_targets_path=targets_path,
        confirmation_score_path=None,
        cost_penalty_lambda=0.35,
    )
    rendered = render_markdown(audit)

    assert audit["schema"] == "diffusion_budget_value_proxy_audit.v1"
    assert audit["summary"]["oracle_profitable_tasks"] == ["plan_001"]
    assert audit["selected_policy"]["policy_id"] in {
        "calibrated_quality_gate",
        "low_quality_repairable_040",
    }
    assert audit["selected_policy"]["regret_vs_oracle"] == pytest.approx(0.0)
    assert audit["selected_policy"]["selected_tasks"] == ["plan_001"]
    all_repairable = _policy_row(audit, "all_repairable_phase")
    assert all_repairable["false_positive_count"] == 2
    assert all_repairable["regret_vs_oracle"] > 0.0
    assert "Diffusion Budget-Value Proxy Audit" in rendered
    assert "trajectory_score <=" in rendered


def _policy_row(audit, policy_id):
    for row in audit["policy_rows"]:
        if row["policy_id"] == policy_id:
            return row
    raise AssertionError(f"missing policy row {policy_id}")


def _target(
    task_id,
    *,
    aggregate_lift,
    marginal_cost,
    trajectory_score,
    first_step,
    prompt_gap,
    source_needs_repair,
):
    return {
        "aggregate_score_lift": aggregate_lift,
        "break_even_lambda": aggregate_lift / marginal_cost if marginal_cost else 0.0,
        "first_repairable_step": first_step,
        "marginal_relative_cost": marginal_cost,
        "prompt_coverage": 0.5,
        "prompt_gap_count": prompt_gap,
        "source_needs_repair": source_needs_repair,
        "task_id": task_id,
        "trajectory_score": trajectory_score,
    }
