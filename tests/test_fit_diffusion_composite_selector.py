import json

from experiments.fit_diffusion_composite_selector import fit_composite_selector, render_markdown


def test_fit_composite_selector_learns_four_heads(tmp_path):
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "realization_policy_targets": [
                    _realization("policy_good", selected=True, error=0.1, loss=0.1, task_score=0.9),
                    _realization("policy_bad", selected=False, error=0.4, loss=0.4, task_score=0.7),
                ],
                "task_targets": [
                    _task("a", spend=True, source=True, retention=True, gap=2, source_quality=0.2),
                    _task("b", spend=False, source=False, retention=False, gap=9, source_quality=0.5),
                    _task("c", spend=True, source=False, retention=False, gap=3, source_quality=0.25),
                ],
            }
        ),
        encoding="utf-8",
    )

    fit = fit_composite_selector(targets_path=targets)

    assert fit["summary"]["total_training_error"] == 0
    assert fit["spend_head"]["error_count"] == 0
    assert fit["source_head"]["error_count"] == 0
    assert fit["retention_head"]["rule_id"] == "classification_safe_history_anchor"
    assert fit["realization_head"]["selected_policy_id"] == "policy_good"


def test_render_markdown_includes_head_names(tmp_path):
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "realization_policy_targets": [
                    _realization("policy_good", selected=True, error=0.1, loss=0.1, task_score=0.9)
                ],
                "task_targets": [_task("a", spend=True, source=True, retention=True)],
            }
        ),
        encoding="utf-8",
    )

    fit = fit_composite_selector(targets_path=targets)
    markdown = render_markdown(fit)

    assert "# Diffusion Composite Selector Fit" in markdown
    assert "Spend head" in markdown
    assert "Realization Head" in markdown


def _task(
    task_id,
    *,
    spend,
    source,
    retention,
    gap=2,
    source_quality=0.2,
):
    return {
        "constraint_retention_loss": 0.1 if retention else 0.8,
        "first_repairable_step": 10,
        "prompt_gap_count": gap,
        "retention_classification": "safe_history_anchor" if retention else "blocked",
        "retention_safe_history_label": retention,
        "source_quality": source_quality,
        "source_trust_history_label": source,
        "spend_repair_label": spend,
        "target_similarity": 0.97 if source else 0.90,
        "task_id": task_id,
        "text_similarity": 0.98 if source else 0.91,
    }


def _realization(policy_id, *, selected, error, loss, task_score):
    return {
        "mean_meta_penalty": 0.0,
        "mean_realization_quality_loss": loss,
        "mean_seed_objective_score": 0.8,
        "mean_task_score": task_score,
        "policy_id": policy_id,
        "realization_policy_error": error,
        "selected": selected,
    }
