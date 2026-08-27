import json

from experiments.analyze_diffusion_decomposed_selector import (
    build_decomposed_selector_audit,
    render_markdown,
)


def test_decomposed_selector_beats_single_repairability_label(tmp_path):
    repair_value = tmp_path / "repair_value.json"
    source_policy = tmp_path / "source_policy.json"
    retention_audit = tmp_path / "retention.json"
    realization_audit = tmp_path / "realization.json"
    repair_value.write_text(
        json.dumps(
            {
                "coordinate_rows": [
                    _value_row("plan_a", utility=-0.1, repairable=True, source_quality=0.5),
                    _value_row("plan_b", utility=0.2, repairable=True, source_quality=0.2),
                    _value_row("plan_c", utility=0.0, repairable=False, source_quality=0.9),
                ]
            }
        ),
        encoding="utf-8",
    )
    source_policy.write_text(
        json.dumps(
            {
                "targets": [
                    _source_row("plan_a", label=0, repairable=1, safe=1, weight=0.3),
                    _source_row("plan_b", label=1, repairable=1, safe=1, weight=0.2),
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_retention_and_realization_inputs(retention_audit, realization_audit)

    audit = build_decomposed_selector_audit(
        repair_value_geometry_path=repair_value,
        phase_source_policy_path=source_policy,
        retention_audit_path=retention_audit,
        realization_audit_path=realization_audit,
    )

    rows = {row["selector_id"]: row for row in audit["selector_rows"]}
    assert rows["single_repairability_label"]["composite_shortfall"] > rows[
        "decomposed_value_source"
    ]["composite_shortfall"]
    assert rows["single_repairability_label"]["value_false_positive_count"] == 1
    assert rows["single_repairability_label"]["source_false_positive_count"] == 1
    assert rows["single_repairability_label"]["retention_constraint_error"] == 0.5
    assert rows["decomposed_value_source"]["retention_constraint_error"] == 0.0
    assert rows["decomposed_value_source"]["realization_policy_error"] < rows[
        "single_repairability_label"
    ]["realization_policy_error"]
    assert audit["selected_selector"]["selector_id"] in {
        "decomposed_value_source",
        "oracle_targets",
    }


def test_render_markdown_names_single_and_decomposed_selectors(tmp_path):
    repair_value = tmp_path / "repair_value.json"
    source_policy = tmp_path / "source_policy.json"
    retention_audit = tmp_path / "retention.json"
    realization_audit = tmp_path / "realization.json"
    repair_value.write_text(
        json.dumps({"coordinate_rows": [_value_row("plan_a", utility=0.1, repairable=True)]}),
        encoding="utf-8",
    )
    source_policy.write_text(
        json.dumps({"targets": [_source_row("plan_a", label=1, repairable=1, safe=1)]}),
        encoding="utf-8",
    )
    _write_retention_and_realization_inputs(retention_audit, realization_audit)

    audit = build_decomposed_selector_audit(
        repair_value_geometry_path=repair_value,
        phase_source_policy_path=source_policy,
        retention_audit_path=retention_audit,
        realization_audit_path=realization_audit,
    )
    markdown = render_markdown(audit)

    assert "# Diffusion Decomposed Selector Audit" in markdown
    assert "single_repairability_label" in markdown
    assert "decomposed_value_source" in markdown
    assert "Retention Error" in markdown
    assert "Realization Error" in markdown


def _value_row(task_id, *, utility, repairable, source_quality=0.2):
    return {
        "first_repairable_step": 10 if repairable else None,
        "prompt_gap_count": 2,
        "source_needs_repair": repairable,
        "source_quality": source_quality,
        "task_id": task_id,
        "utility": utility,
    }


def _source_row(task_id, *, label, repairable, safe, weight=0.1):
    return {
        "label": label,
        "loss_weight": weight,
        "phase_repairable_count": repairable,
        "phase_safe_repairable_count": safe,
        "target_similarity": 0.97 if label else 0.93,
        "task_id": task_id,
        "text_similarity": 0.97 if label else 0.93,
    }


def _write_retention_and_realization_inputs(retention_audit, realization_audit):
    retention_audit.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "classification": "span_advantage_blocks_history",
                        "constraint_retention_loss": 0.5,
                        "task_id": "plan_a",
                    },
                    {
                        "classification": "safe_history_anchor",
                        "constraint_retention_loss": 0.2,
                        "task_id": "plan_b",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    realization_audit.write_text(
        json.dumps(
            {
                "policy_summaries": [
                    _realization_policy("auto_seeded", task_score=0.5, loss=0.4),
                    _realization_policy("auto_compat_seeded", task_score=0.6, loss=0.3),
                    _realization_policy("auto_compat_preserve_seeded", task_score=0.7, loss=0.1),
                ]
            }
        ),
        encoding="utf-8",
    )


def _realization_policy(policy_id, *, task_score, loss):
    return {
        "mean_realization_quality_loss": loss,
        "mean_task_score": task_score,
        "policy_id": policy_id,
    }
