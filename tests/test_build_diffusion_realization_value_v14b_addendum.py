import json

from experiments.build_diffusion_realization_value_v14b_addendum import (
    build_addendum_manifest,
    render_markdown,
)


def test_realization_value_v14b_addendum_selects_near_misses_before_labels(tmp_path):
    freeze = tmp_path / "freeze.json"
    boundary = tmp_path / "boundary.json"
    label_scores = tmp_path / "labels.json"
    v14b_label_scores = tmp_path / "v14b_labels.json"
    freeze.write_text(json.dumps({"task_preset": "lean_gpu_mixed_transfer_v14"}), encoding="utf-8")
    boundary.write_text(json.dumps(_boundary_payload()), encoding="utf-8")

    manifest = build_addendum_manifest(
        freeze_path=freeze,
        measurement_boundary_path=boundary,
        label_scores_path=label_scores,
        v14b_label_scores_path=v14b_label_scores,
    )
    markdown = render_markdown(manifest)

    assert manifest["target_surface"]["measured_probe_value_prediction_max"] == 0.033
    assert manifest["measurement_replay"]["surface_selected_task_ids"] == ["plan_109", "plan_112"]
    assert manifest["measurement_replay"]["near_miss_task_ids"] == ["plan_109", "plan_112"]
    assert "realization_value_v14b_label_scores.json" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "No v14 labels exist" in markdown


def test_realization_value_v14b_addendum_refuses_existing_labels(tmp_path):
    freeze = tmp_path / "freeze.json"
    boundary = tmp_path / "boundary.json"
    label_scores = tmp_path / "labels.json"
    v14b_label_scores = tmp_path / "v14b_labels.json"
    freeze.write_text(json.dumps({"task_preset": "lean_gpu_mixed_transfer_v14"}), encoding="utf-8")
    boundary.write_text(json.dumps(_boundary_payload()), encoding="utf-8")
    label_scores.write_text("{}", encoding="utf-8")

    try:
        build_addendum_manifest(
            freeze_path=freeze,
            measurement_boundary_path=boundary,
            label_scores_path=label_scores,
            v14b_label_scores_path=v14b_label_scores,
        )
    except ValueError as exc:
        assert "labels exist" in str(exc)
    else:
        raise AssertionError("expected existing labels to block addendum")


def _boundary_payload():
    return {
        "row_diagnostics": [
            {
                "measured_probe_value_prediction": 0.0324,
                "near_miss_probe_cap_only": True,
                "prompt_gap_count": 5,
                "source_task_delta_vs_trajectory": 0.0,
                "task_id": "plan_109",
            },
            {
                "measured_probe_value_prediction": 0.0323,
                "near_miss_probe_cap_only": True,
                "prompt_gap_count": 6,
                "source_task_delta_vs_trajectory": 0.06,
                "task_id": "plan_112",
            },
            {
                "measured_probe_value_prediction": 0.02,
                "near_miss_probe_cap_only": False,
                "prompt_gap_count": 9,
                "source_task_delta_vs_trajectory": -0.1,
                "task_id": "plan_blocked",
            },
        ],
        "summary": {
            "run_id": "diffusion-test",
            "source_divergent_task_ids": ["plan_112", "plan_blocked"],
            "surface_selected_task_ids": [],
        },
    }
