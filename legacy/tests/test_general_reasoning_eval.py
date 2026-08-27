"""Tests for the general reasoning scout scorers."""

from latent_reasoning.eval.general_reasoning import (
    GeneralReasoningTask,
    extract_choice,
    score_integer_output,
    score_planning_output,
    score_short_text_output,
)


def test_integer_scorer_uses_final_integer():
    task = GeneralReasoningTask(
        task_id="m",
        family="math",
        prompt="",
        answer_type="integer",
        scorer="exact_integer",
        answer=42,
        max_new_tokens=32,
    )
    score = score_integer_output(task, "Compute 20 + 22. Answer: 42")
    assert score.score == 1.0
    assert score.extracted_answer == 42


def test_choice_extractor_uses_explicit_answer():
    assert extract_choice("Reasoning here. Answer: C", {"C": "increases"}) == "C"


def test_planning_scorer_rewards_specific_risk_aware_answer():
    task = GeneralReasoningTask(
        task_id="p",
        family="planning",
        prompt="",
        answer_type="rubric",
        scorer="planning_rubric_v1",
        max_new_tokens=64,
        rubric_items=(
            "preserve baseline measurement",
            "compare intervention against baseline",
            "name rollback or failure risk",
        ),
    )
    score = score_planning_output(
        task,
        (
            "First preserve the baseline measurement and log the key metric. "
            "Then compare the intervention against that baseline because the "
            "risky job may fail. Add a rollback threshold and validate the result."
        ),
    )
    assert score.score > 0.7


def test_short_text_scorer_uses_token_boundaries():
    task = GeneralReasoningTask(
        task_id="s",
        family="symbolic",
        prompt="",
        answer_type="short_text",
        scorer="exact_short_text",
        answer="on",
        max_new_tokens=16,
    )
    assert score_short_text_output(task, "The lamp is on.").score == 1.0
    wrong = score_short_text_output(task, "The lamp is off.")
    assert wrong.score == 0.0
    assert wrong.extracted_answer == "off"
    assert score_short_text_output(task, "The option is unclear.").score == 0.0


def test_short_text_scorer_allows_punctuation_between_order_tokens():
    task = GeneralReasoningTask(
        task_id="s",
        family="symbolic",
        prompt="",
        answer_type="short_text",
        scorer="exact_short_text",
        answer="green red blue",
        max_new_tokens=16,
    )
    assert score_short_text_output(task, "Final: green, red, blue.").score == 1.0
