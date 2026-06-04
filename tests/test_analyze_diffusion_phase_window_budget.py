import json

from experiments.analyze_diffusion_phase_window_budget import (
    build_phase_window_budget_map,
    render_markdown,
)


def test_phase_window_budget_map_derives_minimal_caps_and_confirms_runs(tmp_path):
    reference_path = tmp_path / "reference_scores.json"
    cap1_path = tmp_path / "cap1_scores.json"
    cap4_path = tmp_path / "cap4_scores.json"
    reference_path.write_text(
        json.dumps(
            {
                "all_generation_count": 11,
                "arms": {"fixed": {"count": 4}},
                "by_family_arm": {
                    "planning": {
                        "fixed": {"mean_task_score": 0.30},
                        "random": {"mean_task_score": 0.20},
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2.6666666667,
                            "mean_task_score": 0.70,
                        },
                    }
                },
                "comparison_rows": [
                    {
                        "task_id": "plan_001",
                        "repair_control": "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                        "repair_task_score": 0.80,
                        "trajectory_task_score": 0.20,
                    },
                    {
                        "task_id": "plan_002",
                        "repair_control": "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                        "repair_task_score": 0.70,
                        "trajectory_task_score": 0.30,
                    },
                    {
                        "task_id": "plan_003",
                        "repair_control": "low_confidence_32",
                        "repair_task_score": 0.60,
                        "trajectory_task_score": 0.60,
                    },
                    {
                        "task_id": "math_001",
                        "repair_control": "",
                        "repair_task_score": 1.0,
                        "trajectory_task_score": 1.0,
                    },
                ],
                "repair_pack": "constraint_span_phase_final_preserve_seeded_gated",
                "repair_spend_gate_rows": [
                    {
                        "first_repairable_denoise_skeleton_coverage": 0.4,
                        "first_repairable_denoise_skeleton_step": 2,
                        "first_repairable_denoise_skeleton_step_fraction": 0.25,
                        "prompt_coverage": 0.5,
                        "prompt_gap_count": 4,
                        "reason": "denoise_phase_repairable",
                        "source_needs_repair": True,
                        "task_id": "plan_001",
                    },
                    {
                        "first_repairable_denoise_skeleton_coverage": 0.5,
                        "first_repairable_denoise_skeleton_step": 4,
                        "first_repairable_denoise_skeleton_step_fraction": 0.50,
                        "prompt_coverage": 0.5,
                        "prompt_gap_count": 6,
                        "reason": "denoise_phase_repairable",
                        "source_needs_repair": True,
                        "task_id": "plan_002",
                    },
                    {
                        "first_repairable_denoise_skeleton_step": None,
                        "prompt_coverage": 0.8,
                        "prompt_gap_count": 1,
                        "reason": "source_quality_ok",
                        "source_needs_repair": False,
                        "task_id": "plan_003",
                    },
                ],
                "run_id": "diffusion-reference",
            }
        ),
        encoding="utf-8",
    )
    cap1_path.write_text(
        json.dumps(
            {
                "all_generation_count": 8,
                "by_family_arm": {
                    "planning": {
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2.0,
                            "mean_task_score": (0.20 + 0.30 + 0.60) / 3,
                        }
                    }
                },
                "comparison_rows": [
                    {"task_id": "plan_001", "repair_control": "low_confidence_32"},
                    {"task_id": "plan_002", "repair_control": "low_confidence_32"},
                    {"task_id": "plan_003", "repair_control": "low_confidence_32"},
                ],
                "repair_denoise_skeleton_max_step": 1,
                "run_id": "diffusion-cap1",
            }
        ),
        encoding="utf-8",
    )
    cap4_path.write_text(
        json.dumps(
            {
                "all_generation_count": 10,
                "by_family_arm": {
                    "planning": {
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2 + 2 / 3,
                            "mean_task_score": (0.80 + 0.70 + 0.60) / 3,
                        }
                    }
                },
                "comparison_rows": [
                    {
                        "task_id": "plan_001",
                        "repair_control": "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                    },
                    {
                        "task_id": "plan_002",
                        "repair_control": "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                    },
                    {"task_id": "plan_003", "repair_control": "low_confidence_32"},
                ],
                "repair_denoise_skeleton_max_step": 4,
                "repair_phase_budget": "frontier",
                "run_id": "diffusion-cap4",
            }
        ),
        encoding="utf-8",
    )

    budget_map = build_phase_window_budget_map(
        reference_score_path=reference_path,
        confirmation_score_paths=[cap1_path, cap4_path],
        promotion_margin=0.02,
    )
    rendered = render_markdown(budget_map)

    assert budget_map["schema"] == "diffusion_phase_window_budget_map.v1"
    assert budget_map["summary"]["floor_cap"] == 1
    assert budget_map["summary"]["first_repair_cap"] == 2
    assert budget_map["summary"]["full_frontier_cap"] == 4
    assert budget_map["summary"]["confirmation_mismatch_count"] == 0
    assert [row["cap_range"] for row in budget_map["transition_rows"]] == ["1-1", "2-3", "4+"]
    assert budget_map["transition_rows"][1]["newly_active_tasks"] == ["plan_001"]
    assert budget_map["transition_rows"][2]["newly_active_tasks"] == ["plan_002"]
    assert [row["mode"] for row in budget_map["runner_mode_rows"]] == [
        "floor",
        "cheap",
        "mid",
        "frontier",
    ]
    assert "--repair-phase-budget cheap" in rendered
    assert all(row["matches_budget_model"] for row in budget_map["confirmation_rows"])
    assert budget_map["confirmation_rows"][1]["repair_phase_budget"] == "frontier"
    assert "Diffusion Phase-Window Budget Map" in rendered
    assert "| 4 | `frontier` | `diffusion-cap4`" in rendered
    assert "diffusion-cap4" in rendered
    assert "plan_001@2" in budget_map["transition_rows"][1]["onset_explanation"]
