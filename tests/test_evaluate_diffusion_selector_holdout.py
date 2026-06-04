import json

from experiments.evaluate_diffusion_selector_holdout import (
    evaluate_selector_holdout,
    render_markdown,
)


def test_evaluate_selector_holdout_beats_single_label_baseline(tmp_path):
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "task_targets": [
                    _task("a", repair=True, spend=True, source=False, retention=False),
                    _task("b", repair=True, spend=True, source=False, retention=False),
                    _task("c", repair=True, spend=False, source=False, retention=False),
                    _task("d", repair=False, spend=False, source=False, retention=False),
                ]
            }
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_selector_holdout(targets_path=targets)
    summary = evaluation["summary"]

    assert summary["holdout_task_count"] == 4
    assert summary["single_label_error_count"] > summary["decomposed_error_count"]
    assert summary["absolute_error_reduction"] > 0


def test_render_markdown_includes_holdout_summary(tmp_path):
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "task_targets": [
                    _task("a", repair=True, spend=True, source=False, retention=False),
                    _task("b", repair=False, spend=False, source=False, retention=False),
                ]
            }
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_selector_holdout(targets_path=targets)
    markdown = render_markdown(evaluation)

    assert "# Diffusion Selector Holdout Evaluation" in markdown
    assert "Single-label repairability errors" in markdown
    assert "Held-Out Task" in markdown


def _task(task_id, *, repair, spend, source, retention):
    return {
        "constraint_retention_loss": 0.1 if retention else 0.8,
        "first_repairable_step": 10 if repair else None,
        "prompt_gap_count": 2 if spend else 10,
        "retention_classification": "safe_history_anchor" if retention else "blocked",
        "retention_safe_history_label": retention,
        "source_quality": 0.2 if spend else 0.6,
        "source_trust_history_label": source,
        "spend_repair_label": spend,
        "target_similarity": 0.97 if source else 0.90,
        "task_id": task_id,
        "text_similarity": 0.98 if source else 0.91,
    }
