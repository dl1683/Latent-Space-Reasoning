"""Tests for diffusion trajectory summaries."""

from latent_reasoning.diffusion.trajectory import summarize_history_samples


def test_summarize_history_samples_tracks_mask_and_stability_steps():
    samples = [
        {"step": 1, "generated_token_ids": [3, 3], "text": "<|mask|><|mask|>"},
        {"step": 4, "generated_token_ids": [10, 3], "text": "Plan<|mask|>"},
        {"step": 8, "generated_token_ids": [10, 11], "text": "Plan done"},
    ]

    summary = summarize_history_samples(
        samples,
        final_text="Plan done",
        mask_token_id=3,
        eos_token_id=2,
    )

    assert summary is not None
    assert summary["sample_count"] == 3
    assert summary["first_visible_step"] == 4
    assert summary["first_final_text_step"] == 8
    assert summary["first_mask_free_step"] == 8
    assert summary["samples"][0]["mask_count"] == 2
    assert summary["newly_visible_token_count"] == 2
    assert summary["committed_token_change_count"] == 0
    assert summary["committed_token_remask_count"] == 0
    assert summary["remasked_token_rewrite_count"] == 0
    assert summary["mask_count_increase_count"] == 0
    assert summary["sampled_history_is_monotonic_fill"] is True


def test_summarize_history_samples_allows_empty_history():
    summary = summarize_history_samples([], final_text="")
    assert summary == {
        "sample_count": 0,
        "first_visible_step": None,
        "first_final_text_step": None,
        "first_mask_free_step": None,
        "final_visible_chars": 0,
        "final_has_visible_text": False,
        "newly_visible_token_count": 0,
        "committed_token_change_count": 0,
        "committed_token_remask_count": 0,
        "remasked_token_rewrite_count": 0,
        "mask_count_increase_count": 0,
        "sampled_history_is_monotonic_fill": True,
        "samples": [],
    }


def test_summarize_history_samples_removes_special_token_text():
    samples = [
        {
            "step": 1,
            "generated_token_ids": [10, 2],
            "text": "Done.<|eot_id|>",
        }
    ]

    summary = summarize_history_samples(
        samples,
        final_text="Done.",
        mask_token_id=3,
        eos_token_id=2,
    )

    assert summary is not None
    assert summary["first_final_text_step"] == 1
    assert summary["samples"][0]["visible_text"] == "Done."


def test_summarize_history_samples_tracks_visible_token_revisions():
    samples = [
        {"step": 1, "generated_token_ids": [3, 3], "text": "<|mask|><|mask|>"},
        {"step": 2, "generated_token_ids": [10, 3], "text": "A<|mask|>"},
        {"step": 3, "generated_token_ids": [11, 3], "text": "B<|mask|>"},
        {"step": 4, "generated_token_ids": [3, 3], "text": "<|mask|><|mask|>"},
    ]

    summary = summarize_history_samples(
        samples,
        final_text="",
        mask_token_id=3,
        eos_token_id=2,
    )

    assert summary is not None
    assert summary["newly_visible_token_count"] == 1
    assert summary["committed_token_change_count"] == 1
    assert summary["committed_token_remask_count"] == 1
    assert summary["remasked_token_rewrite_count"] == 0
    assert summary["mask_count_increase_count"] == 1
    assert summary["sampled_history_is_monotonic_fill"] is False


def test_summarize_history_samples_tracks_remask_mediated_rewrites():
    samples = [
        {"step": 1, "generated_token_ids": [3], "text": "<|mask|>"},
        {"step": 2, "generated_token_ids": [10], "text": "A"},
        {"step": 3, "generated_token_ids": [3], "text": "<|mask|>"},
        {"step": 4, "generated_token_ids": [11], "text": "B"},
    ]

    summary = summarize_history_samples(
        samples,
        final_text="B",
        mask_token_id=3,
        eos_token_id=2,
    )

    assert summary is not None
    assert summary["committed_token_remask_count"] == 1
    assert summary["remasked_token_rewrite_count"] == 1
    assert summary["sampled_history_is_monotonic_fill"] is False
