import json

from experiments.fit_diffusion_spend_transfer_rule import fit_spend_transfer_rule, render_markdown


def test_fit_spend_transfer_rule_uses_source_task_floor_to_drop_false_positive(tmp_path):
    targets = tmp_path / "targets.json"
    independent = tmp_path / "independent.json"
    targets.write_text(
        json.dumps(
            {
                "task_targets": [
                    _target("profitable_a", label=True, source_quality=0.20, source_task=0.40),
                    _target("profitable_b", label=True, source_quality=0.25, source_task=0.35),
                    _target("false_positive", label=False, source_quality=0.20, source_task=0.20),
                    _target("high_quality_skip", label=False, source_quality=0.50, source_task=0.60),
                ]
            }
        ),
        encoding="utf-8",
    )
    independent.write_text(
        json.dumps(
            {
                "rows": [
                    _independent("transfer_false_positive", label=False, source_quality=0.20, source_task=0.30)
                ]
            }
        ),
        encoding="utf-8",
    )

    fit = fit_spend_transfer_rule(targets_path=targets, independent_eval_path=independent)

    assert fit["best_rule"]["error_count"] == 0
    assert fit["best_rule"]["rule_id"].startswith("current_decomposed_spend_source_task_ge_")
    assert fit["best_rule"]["selected_tasks"] == ["profitable_a", "profitable_b"]


def test_render_markdown_includes_rule_comparison(tmp_path):
    targets = tmp_path / "targets.json"
    independent = tmp_path / "independent.json"
    targets.write_text(
        json.dumps({"task_targets": [_target("a", label=True, source_quality=0.20, source_task=0.40)]}),
        encoding="utf-8",
    )
    independent.write_text(json.dumps({"rows": []}), encoding="utf-8")

    fit = fit_spend_transfer_rule(targets_path=targets, independent_eval_path=independent)
    markdown = render_markdown(fit)

    assert "# Diffusion Spend Transfer Rule Fit" in markdown
    assert "Rule Comparison" in markdown
    assert "Training Rows" in markdown


def _target(task_id, *, label, source_quality, source_task):
    return {
        "first_repairable_step": 10,
        "prompt_gap_count": 4,
        "source_quality": source_quality,
        "spend_repair_label": label,
        "task_id": task_id,
        "trajectory_score": source_task,
    }


def _independent(task_id, *, label, source_quality, source_task):
    return {
        "first_repairable_step": 10,
        "profitable": label,
        "prompt_gap_count": 4,
        "source_quality": source_quality,
        "source_task_score": source_task,
        "task_id": task_id,
    }
