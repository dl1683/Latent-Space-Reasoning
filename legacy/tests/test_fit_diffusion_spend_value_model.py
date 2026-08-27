import json

from experiments.fit_diffusion_spend_value_model import (
    fit_spend_value_model,
    render_markdown,
)


def test_spend_value_model_preserves_training_positive_when_neighbor_is_clear(tmp_path):
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "rows": [
                    _row("pos_a", True, gap=3, quality=0.20, step=4, delta=0.0),
                    _row("pos_b", True, gap=4, quality=0.21, step=4, delta=0.0),
                    _row("neg_a", False, gap=12, quality=0.31, step=4, delta=0.0),
                    _row("neg_b", False, gap=12, quality=0.32, step=4, delta=0.0),
                ]
            }
        ),
        encoding="utf-8",
    )

    result = fit_spend_value_model(spend_eval_paths=(eval_path,))

    assert result["summary"]["target_count"] == 4
    assert result["in_sample"]["summary"]["false_negative_count"] == 0
    assert result["in_sample"]["summary"]["false_positive_count"] == 0


def test_spend_value_model_leave_one_slice_out_reports_missed_positive(tmp_path):
    train_eval = tmp_path / "v1.json"
    held_eval = tmp_path / "v2.json"
    train_eval.write_text(
        json.dumps(
            {
                "rows": [
                    _row("train_pos", True, gap=4, quality=0.20, step=4, delta=0.0),
                    _row("train_neg", False, gap=12, quality=0.30, step=4, delta=0.0),
                ]
            }
        ),
        encoding="utf-8",
    )
    held_eval.write_text(
        json.dumps(
            {
                "rows": [
                    _row("held_pos", True, gap=12, quality=0.30, step=4, delta=0.0),
                    _row("held_neg", False, gap=4, quality=0.20, step=4, delta=0.0),
                ]
            }
        ),
        encoding="utf-8",
    )

    result = fit_spend_value_model(spend_eval_paths=(train_eval, held_eval))
    markdown = render_markdown(result)

    assert result["leave_one_slice_out"]["summary"]["false_negative_count"] >= 1
    assert "Do not promote this value model" in markdown
    assert "Leave-One-Slice-Out Rows" in markdown


def _row(task_id, profitable, *, gap, quality, step, delta):
    return {
        "calibrated_availability_prediction": True,
        "decomposed_prediction": True,
        "first_repairable_step": step,
        "learned_availability_prediction": False,
        "profitable": profitable,
        "prompt_gap_count": gap,
        "repair_lift": 0.1 if profitable else 0.0,
        "single_repairability_prediction": True,
        "source_quality": quality,
        "source_task_delta_vs_trajectory": delta,
        "task_id": task_id,
        "trajectory_relative_prediction": True,
    }
