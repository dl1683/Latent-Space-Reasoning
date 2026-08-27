import json

from experiments.analyze_diffusion_repair_value_tomography import (
    build_repair_value_tomography,
    render_markdown,
)


def test_repair_value_tomography_scores_probe_surface(tmp_path):
    targets = tmp_path / "targets.jsonl"
    rows = [
        _target("plan_001", lift=0.008, quality=0.348, gap=9, step=10, needs_repair=True),
        _target("plan_003", lift=0.015, quality=0.324, gap=6, step=10, needs_repair=True),
        _target("plan_004", lift=0.035, quality=0.278, gap=2, step=10, needs_repair=True),
        _target("plan_005", lift=0.0, quality=0.299, gap=10, step=30, needs_repair=True, label=0),
        _target("plan_006", lift=0.024, quality=0.301, gap=9, step=20, needs_repair=True),
        _target("plan_007", lift=0.035, quality=0.248, gap=8, step=31, needs_repair=True),
        _target("plan_008", lift=0.0, quality=0.244, gap=12, step=None, needs_repair=True, label=0),
    ]
    targets.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    audit = build_repair_value_tomography(loss_targets_path=targets)
    markdown = render_markdown(audit)

    assert audit["schema"] == "diffusion_repair_value_tomography.v1"
    probes = {row["probe_id"]: row for row in audit["probe_rows"]}
    controller = {row["probe_id"]: row for row in audit["implemented_controller_rows"]}
    assert probes["incumbent_lambda_018"]["selected_tasks"] == ["plan_004", "plan_006", "plan_007"]
    assert probes["incumbent_lambda_018"]["oracle_tasks"] == ["plan_004", "plan_006", "plan_007"]
    assert probes["quality_relaxed_035"]["false_positive_tasks"] == ["plan_001", "plan_003"]
    assert probes["gap_relaxed_12"]["false_positive_tasks"] == ["plan_005"]
    assert probes["gap_tight_8"]["false_negative_tasks"] == ["plan_006"]
    assert probes["lambda_low_005"]["false_negative_tasks"] == ["plan_001", "plan_003"]
    assert probes["lambda_high_025"]["false_positive_tasks"] == ["plan_006"]
    assert audit["summary"]["implemented_controller_zero_regret"] is True
    assert controller["implemented_lambda_low_005"]["selected_tasks"] == [
        "plan_001",
        "plan_003",
        "plan_004",
        "plan_006",
        "plan_007",
    ]
    assert controller["implemented_lambda_low_005"]["oracle_tasks"] == [
        "plan_001",
        "plan_003",
        "plan_004",
        "plan_006",
        "plan_007",
    ]
    assert controller["implemented_lambda_neutral_018"]["selected_tasks"] == [
        "plan_004",
        "plan_006",
        "plan_007",
    ]
    assert controller["implemented_lambda_high_025"]["selected_tasks"] == ["plan_004", "plan_007"]
    assert controller["implemented_lambda_high_025"]["oracle_tasks"] == ["plan_004", "plan_007"]
    assert all(row["regret_vs_oracle"] == 0.0 for row in controller.values())
    assert "behavior surface" in markdown
    assert "Implemented Controller Checks" in markdown
    assert "lambda-aware" in markdown


def _target(task_id, *, lift, quality, gap, step, needs_repair, label=1):
    return {
        "aggregate_score_lift": lift,
        "break_even_lambda": lift / 0.125 if lift else 0.0,
        "first_repairable_step": step,
        "marginal_relative_cost": 0.125,
        "prompt_gap_count": gap,
        "source_needs_repair": needs_repair,
        "source_quality": quality,
        "task_id": task_id,
        "target": "spend_repair" if label else "skip_repair",
    }
