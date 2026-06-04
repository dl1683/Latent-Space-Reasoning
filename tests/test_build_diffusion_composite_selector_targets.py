import json

from experiments.build_diffusion_composite_selector_targets import (
    build_composite_selector_targets,
    render_markdown,
)


def test_build_composite_selector_targets_merges_four_term_labels(tmp_path):
    repair_value = tmp_path / "repair_value.json"
    source_targets = tmp_path / "source_targets.jsonl"
    retention = tmp_path / "retention.json"
    realization = tmp_path / "realization.json"
    decomposed = tmp_path / "decomposed.json"
    repair_value.write_text(
        json.dumps(
            {
                "coordinate_rows": [
                    _value_row("plan_a", profitable=True, utility=0.1),
                    _value_row("plan_b", profitable=False, utility=-0.2),
                ]
            }
        ),
        encoding="utf-8",
    )
    source_targets.write_text(
        "\n".join(
            [
                json.dumps(_source_row("plan_a", label=1)),
                json.dumps(_source_row("plan_b", label=0)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    retention.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "classification": "safe_history_anchor",
                        "constraint_retention_loss": 0.1,
                        "task_id": "plan_a",
                    },
                    {
                        "classification": "span_advantage_blocks_history",
                        "constraint_retention_loss": 0.8,
                        "task_id": "plan_b",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    realization.write_text(
        json.dumps(
            {
                "policy_summaries": [
                    _realization_policy("policy_good", loss=0.1, task_score=0.8),
                    _realization_policy("policy_bad", loss=0.4, task_score=0.6),
                ]
            }
        ),
        encoding="utf-8",
    )
    decomposed.write_text(
        json.dumps(
            {
                "selected_selector": {
                    "realization_policy": "policy_good",
                    "selector_id": "decomposed",
                    "source_history_tasks": ["plan_a"],
                    "value_selected_tasks": ["plan_a"],
                }
            }
        ),
        encoding="utf-8",
    )

    dataset = build_composite_selector_targets(
        repair_value_geometry_path=repair_value,
        phase_source_targets_path=source_targets,
        retention_audit_path=retention,
        realization_audit_path=realization,
        decomposed_audit_path=decomposed,
    )

    rows = {row["task_id"]: row for row in dataset["task_targets"]}
    assert rows["plan_a"]["spend_repair_label"] is True
    assert rows["plan_a"]["source_trust_history_label"] is True
    assert rows["plan_a"]["retention_safe_history_label"] is True
    assert rows["plan_a"]["selected_spend_decision"] == "spend_repair"
    assert rows["plan_b"]["spend_repair_label"] is False
    assert rows["plan_b"]["selected_source_decision"] == "preserve_final_source"
    policies = {row["policy_id"]: row for row in dataset["realization_policy_targets"]}
    assert policies["policy_good"]["selected"] is True
    assert policies["policy_bad"]["realization_policy_error"] == 0.6000000000000001


def test_render_markdown_names_training_use(tmp_path):
    repair_value = tmp_path / "repair_value.json"
    source_targets = tmp_path / "source_targets.jsonl"
    retention = tmp_path / "retention.json"
    realization = tmp_path / "realization.json"
    decomposed = tmp_path / "decomposed.json"
    repair_value.write_text(
        json.dumps({"coordinate_rows": [_value_row("plan_a", profitable=True, utility=0.1)]}),
        encoding="utf-8",
    )
    source_targets.write_text(json.dumps(_source_row("plan_a", label=1)) + "\n", encoding="utf-8")
    retention.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "classification": "safe_history_anchor",
                        "constraint_retention_loss": 0.1,
                        "task_id": "plan_a",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    realization.write_text(
        json.dumps({"policy_summaries": [_realization_policy("policy_good", loss=0.1, task_score=0.8)]}),
        encoding="utf-8",
    )
    decomposed.write_text(
        json.dumps(
            {
                "selected_selector": {
                    "realization_policy": "policy_good",
                    "selector_id": "decomposed",
                    "source_history_tasks": ["plan_a"],
                    "value_selected_tasks": ["plan_a"],
                }
            }
        ),
        encoding="utf-8",
    )

    dataset = build_composite_selector_targets(
        repair_value_geometry_path=repair_value,
        phase_source_targets_path=source_targets,
        retention_audit_path=retention,
        realization_audit_path=realization,
        decomposed_audit_path=decomposed,
    )
    markdown = render_markdown(dataset)

    assert "# Diffusion Composite Selector Targets" in markdown
    assert "Training Use" in markdown
    assert "plan_a" in markdown


def _value_row(task_id, *, profitable, utility):
    return {
        "first_repairable_step": 10,
        "first_repairable_step_fraction": 0.5,
        "profitable": profitable,
        "prompt_coverage": 0.5,
        "prompt_gap_count": 2,
        "source_quality": 0.2,
        "task_id": task_id,
        "trajectory_score": 0.3,
        "utility": utility,
    }


def _source_row(task_id, *, label):
    return {
        "label": label,
        "loss_weight": 0.2,
        "target_action": "trust_history_source" if label else "preserve_final_source",
        "target_similarity": 0.97,
        "task_id": task_id,
        "text_similarity": 0.98,
    }


def _realization_policy(policy_id, *, loss, task_score):
    return {
        "mean_meta_penalty": 0.0,
        "mean_realization_quality_loss": loss,
        "mean_seed_objective_score": 0.7,
        "mean_task_score": task_score,
        "policy_id": policy_id,
    }
