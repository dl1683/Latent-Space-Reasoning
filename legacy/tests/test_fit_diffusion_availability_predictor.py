import json

from experiments.fit_diffusion_availability_predictor import (
    fit_availability_predictor,
    render_markdown,
)


def test_fit_availability_predictor_learns_trajectory_relative_rule(tmp_path):
    availability = tmp_path / "availability.json"
    availability.write_text(json.dumps({"rows": _rows()}), encoding="utf-8")

    fit = fit_availability_predictor(availability)

    assert fit["best_rule"]["error_count"] == 0
    assert fit["baselines"][0]["rule_id"] == "single_repairability_prediction"
    assert fit["baselines"][0]["error_count"] == 2
    assert fit["baselines"][1]["rule_id"] == "decomposed_prediction"
    assert fit["baselines"][1]["error_count"] == 1
    assert fit["baselines"][2]["rule_id"] == "trajectory_relative_prediction"
    assert fit["baselines"][2]["error_count"] == 0
    assert fit["best_rule"]["selected_tasks"] == ["plan_pos"]
    assert any(
        predicate["feature"] == "source_task_delta_vs_trajectory"
        for predicate in fit["best_rule"]["predicates"]
    )


def test_render_markdown_includes_predictor_summary(tmp_path):
    availability = tmp_path / "availability.json"
    availability.write_text(json.dumps({"rows": _rows()}), encoding="utf-8")

    markdown = render_markdown(fit_availability_predictor(availability))

    assert "# Diffusion Availability Predictor Fit" in markdown
    assert "Leave-one-out errors" in markdown
    assert "Source-Selected Delta" in markdown


def _rows():
    return [
        _row(
            "plan_pos",
            profitable=True,
            first_step=14,
            gap=6,
            quality=0.25,
            delta=0.0,
            single=True,
            decomposed=True,
            trajectory_relative=True,
        ),
        _row(
            "plan_high_quality",
            profitable=False,
            first_step=15,
            gap=7,
            quality=0.33,
            delta=0.0,
            single=True,
            decomposed=False,
            trajectory_relative=False,
        ),
        _row(
            "plan_below_trajectory",
            profitable=False,
            first_step=11,
            gap=6,
            quality=0.25,
            delta=-0.06,
            single=True,
            decomposed=True,
            trajectory_relative=False,
        ),
        _row(
            "plan_outside_gap",
            profitable=False,
            first_step=17,
            gap=10,
            quality=0.25,
            delta=0.0,
            single=False,
            decomposed=False,
            trajectory_relative=False,
        ),
    ]


def _row(
    task_id,
    *,
    profitable,
    first_step,
    gap,
    quality,
    delta,
    single,
    decomposed,
    trajectory_relative,
):
    return {
        "decomposed_prediction": decomposed,
        "first_repairable_step": first_step,
        "profitable": profitable,
        "prompt_gap_count": gap,
        "single_repairability_prediction": single,
        "source_quality": quality,
        "source_task_delta_vs_trajectory": delta,
        "task_id": task_id,
        "trajectory_relative_prediction": trajectory_relative,
    }
