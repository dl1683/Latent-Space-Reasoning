from experiments.latent_aggregation_expanded_aspects import (
    assert_label_free_aspect_view,
    expanded_aspect_scores,
    expanded_complement_aspects,
    label_free_aspect_view,
)


def test_expanded_aspect_scores_detect_supported_planning_aspects():
    prompt = "Plan a rollout for a risky migration with rollback and monitoring."
    text = (
        "Assign the platform owner to approve each phase. "
        "First migrate the canary, then expand after error metrics stay below threshold. "
        "Rollback if logs show elevated failures outside the scoped canary."
    )

    scores = expanded_aspect_scores(text, prompt=prompt)

    assert scores["expanded::owner_assignment"]["support_score"] == 1.0
    assert scores["expanded::timeline_or_sequence"]["support_score"] == 1.0
    assert scores["expanded::rollback_or_exit_criteria"]["support_score"] == 1.0
    assert scores["expanded::evidence_or_measurement"]["support_score"] == 1.0
    assert scores["expanded::scope_boundary"]["support_score"] == 1.0
    assert scores["expanded::polarity_or_action_direction"]["support_score"] == 1.0


def test_expanded_aspect_scores_reject_generic_process_mentions():
    scores = expanded_aspect_scores(
        "Create a better process and improve the overall workflow.",
        prompt="Plan a migration with rollback and owners.",
    )

    assert scores["expanded::owner_assignment"]["support_score"] == 0.0
    assert scores["expanded::rollback_or_exit_criteria"]["support_score"] == 0.0
    assert scores["expanded::evidence_or_measurement"]["support_score"] == 0.0


def test_expanded_complement_aspects_select_candidate_only_support():
    complements = expanded_complement_aspects(
        anchor_text="Use a careful migration plan.",
        candidate_text="Assign the platform owner and rollback if canary error metrics rise.",
        prompt="Plan a risky platform migration with rollback and metrics.",
        trajectory_id="candidate-1",
    )

    aspect_types = {row["aspect_type"] for row in complements}
    assert "owner_assignment" in aspect_types
    assert "rollback_or_exit_criteria" in aspect_types
    assert "evidence_or_measurement" in aspect_types
    assert all(row["trajectory_id"] == "candidate-1" for row in complements)


def test_label_free_aspect_view_strips_scores_and_rejects_label_fields():
    record = {
        "task_score": {"score": 1.0, "details": {"rubric_hits": [{"hit": True}]}},
        "text": "Assign the platform owner.",
        "trajectory_id": "candidate-1",
    }

    view = label_free_aspect_view(record, prompt="Plan owner assignment.", source_family="probe")

    assert set(view) == {"prompt", "source_family", "text", "trajectory_id"}
    assert "task_score" not in view

    try:
        assert_label_free_aspect_view({"prompt": "x", "text": "x", "task_score": {"score": 1.0}})
    except ValueError as exc:
        assert "forbidden label fields" in str(exc)
    else:
        raise AssertionError("expected task_score to be rejected")
