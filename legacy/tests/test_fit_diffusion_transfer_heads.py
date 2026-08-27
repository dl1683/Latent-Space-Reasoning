import json

from experiments.fit_diffusion_transfer_heads import fit_transfer_heads, render_markdown


def test_fit_transfer_heads_separates_availability_from_promotion(tmp_path):
    original = tmp_path / "targets.json"
    transfer = tmp_path / "transfer.json"
    promotion = tmp_path / "promotion.json"
    original.write_text(
        json.dumps(
            {
                "task_targets": [
                    _original("orig_positive", label=True, source_quality=0.20, gap=4),
                    _original("orig_skip", label=False, source_quality=0.50, gap=4),
                ]
            }
        ),
        encoding="utf-8",
    )
    transfer.write_text(
        json.dumps(
            {
                "rows": [
                    _transfer("plan_012", label=True, source_quality=0.23, gap=8),
                    _transfer("plan_010", label=False, source_quality=0.33, gap=7),
                ]
            }
        ),
        encoding="utf-8",
    )
    promotion.write_text(
        json.dumps(
            {
                "policies": [
                    _policy("planning_quality_seed_realization_guarded", selected=False),
                    _policy("inherit", selected=True),
                ]
            }
        ),
        encoding="utf-8",
    )

    fit = fit_transfer_heads(
        original_targets_path=original,
        promotion_eval_path=promotion,
        transfer_availability_path=transfer,
    )

    assert fit["availability_head"]["error_count"] == 0
    assert fit["promotion_head"]["head_id"] == "transfer_promotion_value"
    assert fit["promotion_head"]["error_count"] == 0
    assert fit["summary"]["availability_positive_baseline_missed_tasks"] == ["plan_012"]


def test_render_markdown_includes_head_sections(tmp_path):
    original = tmp_path / "targets.json"
    transfer = tmp_path / "transfer.json"
    promotion = tmp_path / "promotion.json"
    original.write_text(
        json.dumps({"task_targets": [_original("orig_positive", label=True)]}),
        encoding="utf-8",
    )
    transfer.write_text(
        json.dumps({"rows": [_transfer("plan_012", label=True)]}),
        encoding="utf-8",
    )
    promotion.write_text(
        json.dumps({"policies": [_policy("inherit", selected=True)]}),
        encoding="utf-8",
    )

    fit = fit_transfer_heads(
        original_targets_path=original,
        promotion_eval_path=promotion,
        transfer_availability_path=transfer,
    )
    markdown = render_markdown(fit)

    assert "# Diffusion Transfer Head Fit" in markdown
    assert "Availability Head" in markdown
    assert "Promotion Head" in markdown
    assert "transfer_promotion_value" in markdown


def _original(task_id, *, label, source_quality=0.20, gap=4):
    return {
        "first_repairable_step": 10,
        "prompt_gap_count": gap,
        "source_quality": source_quality,
        "spend_repair_label": label,
        "task_id": task_id,
        "trajectory_score": 0.30,
    }


def _transfer(task_id, *, label, source_quality=0.20, gap=4):
    return {
        "first_repairable_step": 10,
        "profitable": label,
        "prompt_gap_count": gap,
        "source_quality": source_quality,
        "source_task_score": 0.30,
        "task_id": task_id,
    }


def _policy(policy_id, *, selected):
    return {
        "policy_id": policy_id,
        "rows": [
            {
                "available": True,
                "selected_lift": 0.02 if selected else 0.0,
                "selected_positive_lift": selected,
                "task_id": "plan_012",
            }
        ],
        "run_id": f"{policy_id}-run",
    }
