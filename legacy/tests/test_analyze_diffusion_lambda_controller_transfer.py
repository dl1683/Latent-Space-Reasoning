import json

from experiments.analyze_diffusion_lambda_controller_transfer import (
    build_active_target_manifest,
    build_lambda_controller_transfer_audit,
    render_markdown,
)


def test_lambda_controller_transfer_audit_reports_named_generalization_failures(tmp_path):
    spend_eval = tmp_path / "spend_eval.json"
    spend_eval.write_text(
        json.dumps(
            {
                "rows": [
                    _row("plan_low_positive", lift=0.02, quality=0.34, gap=9, step=10),
                    _row("plan_high_waste", lift=0.0, quality=0.29, gap=9, step=10),
                    _row("plan_gap_hidden_positive", lift=0.20, quality=0.20, gap=12, step=10),
                    _row("plan_high_quality_negative", lift=0.0, quality=0.50, gap=6, step=10),
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = build_lambda_controller_transfer_audit(
        spend_eval_paths=(spend_eval,),
        cost_penalties=(0.05, 0.25),
        marginal_relative_cost=0.125,
    )
    manifest = build_active_target_manifest(audit)
    markdown = render_markdown(audit)
    rows = {row["cost_penalty_lambda"]: row for row in audit["lambda_rows"]}
    targets = {row["task_id"]: row for row in audit["active_target_rows"]}

    assert audit["schema"] == "diffusion_lambda_controller_transfer.v1"
    assert audit["summary"]["controller_transfer_safe"] is False
    assert audit["summary"]["active_target_count"] == 2
    assert rows[0.05]["false_positive_tasks"] == ["plan_high_waste"]
    assert rows[0.05]["false_negative_tasks"] == ["plan_gap_hidden_positive"]
    assert rows[0.25]["false_positive_tasks"] == ["plan_high_waste"]
    assert rows[0.25]["false_negative_tasks"] == ["plan_gap_hidden_positive"]
    assert targets["plan_gap_hidden_positive"]["probe_type"] == "hidden_value_probe"
    assert targets["plan_gap_hidden_positive"]["failing_lambdas"] == [0.05, 0.25]
    assert targets["plan_high_waste"]["probe_type"] == "waste_probe"
    assert targets["plan_gap_hidden_positive"]["priority_score"] > targets["plan_high_waste"]["priority_score"]
    assert manifest["schema"] == "diffusion_lambda_repair_active_targets.v1"
    assert manifest["top_hidden_value_task_ids"] == ["plan_gap_hidden_positive"]
    assert manifest["top_waste_probe_task_ids"] == ["plan_high_waste"]
    assert manifest["hidden_value_collection_command"][:2] == [
        "python",
        "experiments/run_diffusion_three_arm_benchmark.py",
    ]
    assert "--task-ids" in manifest["hidden_value_collection_command"]
    assert "Controller transfer safe: `False`" in markdown
    assert "Active Data Targets" in markdown
    assert "Runner Bridge" in markdown
    assert "plan_gap_hidden_positive" in markdown


def _row(task_id, *, lift, quality, gap, step):
    return {
        "first_repairable_step": step,
        "prompt_gap_count": gap,
        "repair_lift": lift,
        "source_quality": quality,
        "task_id": task_id,
    }
