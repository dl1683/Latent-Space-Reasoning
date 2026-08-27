import json

from experiments.fit_diffusion_spend_gate_v7 import fit_spend_gate, render_markdown


def test_fit_spend_gate_prefers_threshold_rule_over_repairable_spend(tmp_path):
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "rows": [
                    _row("pos_a", True, gap=8, delta=0.0),
                    _row("pos_b", True, gap=9, delta=-0.1),
                    _row("neg_a", False, gap=4, delta=0.0),
                    _row("neg_b", False, gap=6, delta=0.2),
                ]
            }
        ),
        encoding="utf-8",
    )

    fit = fit_spend_gate(eval_paths=(eval_path,), max_conditions=2)

    assert fit["summary"]["best_error_count"] == 0
    assert fit["summary"]["best_rule_id"] != "repairable_denoise_spend"
    assert fit["summary"]["best_rule"]["selected_tasks"] == ["pos_a", "pos_b"]


def test_render_markdown_includes_decision(tmp_path):
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps({"rows": [_row("pos", True, gap=8, delta=0.0)]}), encoding="utf-8")

    markdown = render_markdown(fit_spend_gate(eval_paths=(eval_path,), max_conditions=1))

    assert "# Diffusion Spend Gate V7 Fit" in markdown
    assert "Do not promote this gate" in markdown
    assert "Top Rules" in markdown


def _row(task_id, profitable, *, gap, delta):
    return {
        "first_repairable_step": 3,
        "profitable": profitable,
        "prompt_gap_count": gap,
        "repair_lift": 0.1 if profitable else 0.0,
        "single_repairability_prediction": True,
        "source_quality": 0.2,
        "source_task_delta_vs_trajectory": delta,
        "task_id": task_id,
    }
