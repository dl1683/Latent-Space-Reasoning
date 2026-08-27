"""Tests for diffusion-native trajectory control scoring."""

from latent_reasoning.diffusion.control import (
    DiffusionScheduleCandidate,
    default_dream_schedules,
    score_trajectory_summary,
)


def test_schedule_candidate_converts_to_generation_config():
    schedule = DiffusionScheduleCandidate(
        name="entropy_test",
        steps=16,
        max_new_tokens=32,
        algorithm="entropy",
        temperature=0.2,
        top_p=0.95,
    )
    config = schedule.to_config()
    assert config.steps == 16
    assert config.max_new_tokens == 32
    assert config.algorithm == "entropy"
    assert config.output_history is True


def test_early_stable_trajectory_scores_above_late_trajectory():
    early = {
        "first_final_text_step": 16,
        "first_mask_free_step": 64,
        "samples": [
            {"step": 16, "eos_count": 0, "mask_count": 48},
            {"step": 64, "eos_count": 8, "mask_count": 0},
        ],
    }
    late = {
        "first_final_text_step": 64,
        "first_mask_free_step": 64,
        "samples": [
            {"step": 16, "eos_count": 12, "mask_count": 48},
            {"step": 64, "eos_count": 8, "mask_count": 0},
        ],
    }

    early_score = score_trajectory_summary(early, history_steps=64, final_text="A useful plan." * 20)
    late_score = score_trajectory_summary(late, history_steps=64, final_text="A useful plan." * 20)
    assert early_score.overall > late_score.overall


def test_default_dream_schedule_pack_has_entropy_and_origin_controls():
    names = [schedule.name for schedule in default_dream_schedules()]
    assert "entropy_64" in names
    assert "origin_64" in names
