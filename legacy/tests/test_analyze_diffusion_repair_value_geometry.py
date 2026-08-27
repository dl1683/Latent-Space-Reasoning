import json

import pytest

from experiments.analyze_diffusion_repair_value_geometry import (
    build_repair_value_geometry,
    render_markdown,
)


def test_repair_value_geometry_tracks_runner_value_boundary(tmp_path):
    targets_path = tmp_path / "targets.jsonl"
    targets_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _target(
                    "plan_001",
                    aggregate_lift=0.05,
                    marginal_cost=0.10,
                    first_step=10,
                    prompt_gap=2,
                    source_quality=0.20,
                    source_needs_repair=True,
                ),
                _target(
                    "plan_002",
                    aggregate_lift=0.01,
                    marginal_cost=0.10,
                    first_step=10,
                    prompt_gap=4,
                    source_quality=0.45,
                    source_needs_repair=True,
                ),
                _target(
                    "plan_003",
                    aggregate_lift=0.00,
                    marginal_cost=0.10,
                    first_step=20,
                    prompt_gap=12,
                    source_quality=0.25,
                    source_needs_repair=True,
                ),
                _target(
                    "plan_004",
                    aggregate_lift=0.04,
                    marginal_cost=0.10,
                    first_step=30,
                    prompt_gap=8,
                    source_quality=0.30,
                    source_needs_repair=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    audit = build_repair_value_geometry(
        loss_targets_path=targets_path,
        cost_penalty_lambda=0.20,
        runner_source_quality_max=0.31,
        runner_prompt_gap_max=9,
    )
    rendered = render_markdown(audit)

    assert audit["schema"] == "diffusion_repair_value_geometry.v1"
    assert audit["summary"]["profitable_tasks"] == ["plan_001", "plan_004"]
    assert audit["runner_policy"]["selected_tasks"] == ["plan_001", "plan_004"]
    assert audit["runner_policy"]["regret_vs_oracle"] == pytest.approx(0.0)
    assert audit["summary"]["source_quality_band_gap"] == pytest.approx(0.15)
    assert audit["summary"]["lowest_low_quality_excluded_gap"] == pytest.approx(12.0)
    assert _policy(audit, "runner_source_quality_gap")["false_positive_count"] == 0
    assert "Diffusion Repair-Value Geometry" in rendered
    assert "source_quality <=" in rendered


def _policy(audit, policy_id):
    for row in audit["separability_rows"]:
        if row["policy_id"] == policy_id:
            return row
    raise AssertionError(f"missing policy {policy_id}")


def _target(
    task_id,
    *,
    aggregate_lift,
    marginal_cost,
    first_step,
    prompt_gap,
    source_quality,
    source_needs_repair,
):
    return {
        "aggregate_score_lift": aggregate_lift,
        "break_even_lambda": aggregate_lift / marginal_cost if marginal_cost else 0.0,
        "first_repairable_step": first_step,
        "first_repairable_step_fraction": first_step / 32 if first_step is not None else None,
        "marginal_relative_cost": marginal_cost,
        "prompt_coverage": 0.5,
        "prompt_gap_count": prompt_gap,
        "source_needs_repair": source_needs_repair,
        "source_quality": source_quality,
        "task_id": task_id,
        "trajectory_score": source_quality + 0.1,
    }
