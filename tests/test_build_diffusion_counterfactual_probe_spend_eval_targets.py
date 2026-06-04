import json

from experiments.build_diffusion_counterfactual_probe_spend_eval_targets import (
    build_counterfactual_probe_spend_eval_targets,
    render_markdown,
)


def test_probe_spend_eval_targets_convert_profitable_rows(tmp_path):
    spend_eval = tmp_path / "spend_eval.json"
    spend_eval.write_text(
        json.dumps(
            {
                "rows": [
                    _row("plan_a", profitable=True, lift=0.1),
                    _row("plan_b", profitable=False, lift=0.0),
                    _row("math_a", profitable=True, lift=1.0),
                ]
            }
        ),
        encoding="utf-8",
    )

    targets = build_counterfactual_probe_spend_eval_targets(
        probe_policy="span_tomography_probe_v4",
        spend_eval_path=spend_eval,
        task_ids={"plan_a", "plan_b"},
    )
    markdown = render_markdown(targets)

    assert targets["schema"] == "diffusion_counterfactual_probe_spend_eval_targets.v1"
    assert targets["summary"]["target_count"] == 2
    assert targets["summary"]["positive_task_ids"] == ["plan_a"]
    assert targets["rows"][0]["labels"]["promote_vs_trajectory"] is True
    assert "span_tomography_probe_v4" in markdown


def _row(task_id, *, profitable, lift):
    return {
        "decomposed_prediction": True,
        "first_repairable_step": 8,
        "profitable": profitable,
        "prompt_gap_count": 4,
        "repair_lift": lift,
        "selected_repair_lift": lift,
        "single_repairability_prediction": True,
        "source_quality": 0.2,
        "source_task_delta_vs_trajectory": 0.1,
        "task_id": task_id,
        "trajectory_relative_prediction": True,
    }
