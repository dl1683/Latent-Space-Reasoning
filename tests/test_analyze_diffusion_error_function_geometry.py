import json

from experiments.analyze_diffusion_error_function_geometry import (
    build_error_function_geometry,
    render_markdown,
)


def test_build_error_function_geometry_derives_cost_and_source_assertions(tmp_path):
    repair_value_path = tmp_path / "repair_value.json"
    source_targets_path = tmp_path / "source_targets.jsonl"
    repair_value_path.write_text(
        json.dumps(
            {
                "coordinate_rows": [
                    _value_row("plan_a", step=2, lift=0.01, profitable=False),
                    _value_row("plan_b", step=2, lift=0.04, profitable=True),
                    _value_row("plan_c", step=5, lift=0.03, profitable=True),
                ],
                "cost_penalty_lambda": 0.2,
                "runner_policy": {
                    "regret_vs_oracle": 0.0,
                    "selected_tasks": ["plan_b", "plan_c"],
                },
                "summary": {"source_quality_band_gap": 0.05},
            }
        ),
        encoding="utf-8",
    )
    source_targets_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "phase_repairable_count": 2,
                        "phase_retention_safety_lag": 3,
                        "phase_safe_repairable_count": 1,
                        "target_action": "trust_history_source",
                        "task_id": "plan_b",
                    }
                ),
                json.dumps(
                    {
                        "phase_repairable_count": 2,
                        "phase_retention_safety_lag": 4,
                        "phase_safe_repairable_count": 1,
                        "target_action": "preserve_final_source",
                        "task_id": "plan_c",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    audit = build_error_function_geometry(
        repair_value_geometry_path=repair_value_path,
        phase_source_targets_path=source_targets_path,
    )

    repair = audit["repair_value_geometry"]
    source = audit["source_trust_geometry"]
    assert repair["raw_positive_repair_target_count"] == 3
    assert repair["cost_profitable_repair_count"] == 2
    assert repair["cost_flip_tasks"] == ["plan_a"]
    assert repair["early_step_has_mixed_value"] is True
    assert source["naive_repairable_history_false_positive_count"] == 1
    assert source["any_safe_history_false_positive_count"] == 1
    assert [row["id"] for row in audit["assertions"]] == ["E1", "E2", "E3", "E4"]


def test_render_markdown_includes_error_function_names(tmp_path):
    repair_value_path = tmp_path / "repair_value.json"
    source_targets_path = tmp_path / "source_targets.jsonl"
    repair_value_path.write_text(
        json.dumps(
            {
                "coordinate_rows": [_value_row("plan_a", step=1, lift=0.02, profitable=True)],
                "cost_penalty_lambda": 0.1,
                "runner_policy": {"regret_vs_oracle": 0.0, "selected_tasks": ["plan_a"]},
                "summary": {"source_quality_band_gap": 0.1},
            }
        ),
        encoding="utf-8",
    )
    source_targets_path.write_text(
        json.dumps(
            {
                "phase_repairable_count": 1,
                "phase_retention_safety_lag": 0,
                "phase_safe_repairable_count": 1,
                "target_action": "trust_history_source",
                "task_id": "plan_a",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = build_error_function_geometry(
        repair_value_geometry_path=repair_value_path,
        phase_source_targets_path=source_targets_path,
    )
    markdown = render_markdown(audit)

    assert "# Diffusion Error-Function Geometry" in markdown
    assert "Cost-Aware Repair-Value Loss" in markdown
    assert "Retention-Gated Source-Trust Loss" in markdown
    assert "Composite Denoise Reasoning Loss" in markdown


def _value_row(task_id, *, step, lift, profitable):
    return {
        "aggregate_score_lift": lift,
        "first_repairable_step": step,
        "profitable": profitable,
        "task_id": task_id,
    }
