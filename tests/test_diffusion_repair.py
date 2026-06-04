"""Tests for diffusion suffix repair scaffolding."""

import torch

from experiments.run_diffusion_repair_scout import (
    _proposal_only_record,
    _select_best_record,
    summarize_scores,
)
from latent_reasoning.diffusion import (
    DiffusionRepairCandidate,
    DiffusionVerifierRepairCandidate,
    build_answer_span_repair_seed,
    build_history_instability_repair_seed,
    build_history_state_repair_seed,
    build_low_confidence_repair_seed,
    build_suffix_repair_seed,
    build_tail_window_repair_seed,
    build_text_policy_repair_seed,
    build_text_span_repair_seed,
    build_text_span_repair_seed_diagnostics,
    build_text_span_repair_seed_with_diagnostics,
    build_token_position_repair_seed,
    default_llada_constraint_gap_repair_candidates,
    default_llada_constraint_span_anchor_instability_claim_auto_action_seeded_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_claim_auto_compat_seeded_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_claim_auto_joint_seeded_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_claim_auto_seeded_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_claim_auto_seeded_realization_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_claim_compatible_seeded_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_claim_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_claim_oracle_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_claim_seeded_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_claim_strict_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_prompt_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_prompt_only_gated_repair_candidates,
    default_llada_constraint_span_anchor_instability_repair_candidates,
    default_llada_constraint_span_history_instability_repair_candidates,
    default_llada_constraint_span_phase_anchor_repair_candidates,
    default_llada_constraint_span_phase_hybrid_preserve_seeded_gated_repair_candidates,
    default_llada_constraint_span_repair_candidates,
    default_llada_history_repair_candidates,
    default_llada_history_visible_repair_candidates,
    default_llada_prompt_guided_repair_candidates,
    default_llada_repair_candidates,
    default_llada_replay_consistency_repair_candidates,
    default_llada_state_adaptive_repair_candidates,
    default_llada_targeted_content_repair_candidates,
    default_llada_verifier_repair_candidates,
)
from latent_reasoning.diffusion.backends import _apply_initial_suffix_tokens
from latent_reasoning.eval.answer_proposals import (
    AnswerProposal,
    counterfactual_answer_candidates,
    counterfactual_answer_proposals,
)
from latent_reasoning.eval.general_reasoning import GeneralReasoningTask


def test_suffix_repair_seed_keeps_prefix_and_remasks_tail():
    seed = build_suffix_repair_seed(
        [11, 12, 13, 14],
        max_new_tokens=8,
        keep_prefix_fraction=0.5,
    )

    assert seed == (11, 12, None, None, None, None, None, None)


def test_repair_candidate_builds_llada_inpainting_config():
    repair = DiffusionRepairCandidate(name="prefix_test", keep_prefix_fraction=0.25)
    config = repair.to_config([21, 22, 23, 24], max_new_tokens=4)

    assert config.algorithm == "low_confidence"
    assert config.block_length == 4
    assert config.output_history is True
    assert config.initial_suffix_token_ids == (21, None, None, None)


def test_default_llada_repair_pack_is_small():
    names = [repair.name for repair in default_llada_repair_candidates()]
    assert names == [
        "prefix_25_repair",
        "prefix_50_repair",
        "low_confidence_25_repair",
        "low_confidence_40_repair",
    ]


def test_low_confidence_repair_seed_remasks_weakest_positions():
    seed = build_low_confidence_repair_seed(
        [11, 12, 13, 14],
        token_confidences=[0.9, 0.1, 0.8, 0.2],
        max_new_tokens=4,
        remask_fraction=0.5,
    )

    assert seed == (11, None, 13, None)


def test_low_confidence_repair_candidate_uses_confidence_seed():
    repair = DiffusionRepairCandidate(
        name="low_confidence_test",
        remask_low_confidence_fraction=0.25,
    )
    config = repair.to_config(
        [31, 32, 33, 34],
        token_confidences=[0.8, 0.7, 0.1, 0.9],
        max_new_tokens=4,
    )

    assert config.initial_suffix_token_ids == (31, 32, None, 34)


def test_history_instability_repair_seed_masks_replay_unstable_positions():
    seed = build_history_instability_repair_seed(
        [10, 20, 30, 40],
        history_samples_token_ids=[
            [10, 126336, 30, 41],
            [10, 21, 31, 42],
            [10, 22, 30, 40],
        ],
        max_new_tokens=4,
        remask_fraction=0.5,
    )

    assert seed == (10, None, 30, None)


def test_replay_consistency_candidate_uses_history_instability_seed():
    repair = default_llada_replay_consistency_repair_candidates()[0]

    config = repair.to_config(
        [10, 20, 30, 40],
        history_samples_token_ids=[
            [10, 126336, 30, 41],
            [10, 21, 31, 42],
            [10, 22, 30, 40],
        ],
        max_new_tokens=4,
    )

    assert repair.name == "replay_unstable_25_repair"
    assert config.initial_suffix_token_ids == (10, None, 30, 40)


def test_quality_scaled_low_confidence_repair_masks_more_for_weak_source():
    repair = DiffusionRepairCandidate(
        name="state_confidence_test",
        quality_scaled_low_confidence=True,
        quality_scaled_min_fraction=0.25,
        quality_scaled_max_fraction=0.50,
        quality_scaled_floor=0.25,
        quality_scaled_ceiling=0.55,
    )

    weak_config = repair.to_config(
        [31, 32, 33, 34],
        token_confidences=[0.8, 0.7, 0.1, 0.9],
        max_new_tokens=4,
        source_quality_score=0.25,
    )
    strong_config = repair.to_config(
        [31, 32, 33, 34],
        token_confidences=[0.8, 0.7, 0.1, 0.9],
        max_new_tokens=4,
        source_quality_score=0.55,
    )

    assert weak_config.initial_suffix_token_ids == (31, None, None, 34)
    assert strong_config.initial_suffix_token_ids == (31, 32, None, 34)


def test_text_policy_repair_seed_masks_repeated_filler_span():
    vocab = {
        11: "Compare",
        12: " the",
        13: " the",
        14: " baseline",
        15: ".",
    }

    seed = build_text_policy_repair_seed(
        [11, 12, 13, 14, 15],
        source_text="Compare the the baseline.",
        token_decoder=lambda token_ids: "".join(vocab[token_id] for token_id in token_ids),
        max_new_tokens=5,
        policy="generic_filler",
        context_window=0,
    )

    assert seed == (11, None, None, 14, 15)


def test_targeted_content_repair_candidate_masks_bad_surface_span():
    vocab = {
        21: "Use",
        22: " checkpoint",
        23: " checkpointing",
        24: " safely",
    }
    repair = default_llada_targeted_content_repair_candidates()[0]

    config = repair.to_config(
        [21, 22, 23, 24],
        source_text="Use checkpoint checkpointing safely",
        token_decoder=lambda token_ids: "".join(vocab[token_id] for token_id in token_ids),
        max_new_tokens=4,
    )

    assert repair.name == "targeted_filler_repair"
    assert config.initial_suffix_token_ids == (21, None, None, 24)


def test_prompt_guided_repair_pack_starts_with_full_revision_candidate():
    repairs = default_llada_prompt_guided_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "prompt_guided_revision_repair",
        "prompt_guided_revision_anchor25_repair",
        "targeted_filler_repair",
    ]
    assert repairs[0].prompt_repair_instruction is not None
    assert repairs[0].keep_prefix_fraction == 0.0


def test_constraint_gap_repair_pack_preserves_canonical_first_candidates():
    repairs = default_llada_constraint_gap_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "state_adaptive_history_repair",
        "prefix_25_repair",
        "constraint_gap_revision_repair",
        "constraint_gap_revision_anchor25_repair",
        "constraint_gap_span_repair",
    ]
    assert repairs[2].prompt_repair_policy == "constraint_gap"
    assert repairs[2].prompt_repair_instruction is not None
    assert repairs[4].remask_text_policy == "generic_filler"


def test_constraint_span_repair_pack_only_spends_span_branch():
    repairs = default_llada_constraint_span_repair_candidates()

    assert [repair.name for repair in repairs] == ["constraint_gap_span_repair"]
    assert repairs[0].prompt_repair_policy == "constraint_gap"
    assert repairs[0].remask_text_policy == "generic_filler"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_history_state_repair_seed_keeps_visible_tokens_and_remasks_unknowns():
    seed = build_history_state_repair_seed(
        [11, 126336, 12, 126081],
        max_new_tokens=6,
    )

    assert seed == (11, None, 12, None, None, None)


def test_history_state_repair_seed_can_remask_after_prefix_anchor():
    seed = build_history_state_repair_seed(
        [11, 12, 13, 14, 15, 16],
        max_new_tokens=6,
        keep_prefix_fraction=0.5,
    )

    assert seed == (11, 12, 13, None, None, None)


def test_history_state_repair_candidate_uses_history_tokens():
    repair = DiffusionRepairCandidate(
        name="history_test",
        source_state="history",
        keep_prefix_fraction=0.5,
    )
    config = repair.to_config(
        [31, 32, 33, 34],
        history_token_ids=[41, 126336, 43, 126081],
        max_new_tokens=4,
    )

    assert config.initial_suffix_token_ids == (41, None, None, None)


def test_state_adaptive_history_repair_uses_long_anchor_for_weak_state():
    repair = DiffusionRepairCandidate(
        name="state_history_test",
        source_state="history",
        adaptive_history_prefix=True,
        adaptive_history_default_fraction=0.25,
        adaptive_history_weak_fraction=0.50,
    )

    weak_config = repair.to_config(
        [31, 32, 33, 34],
        history_token_ids=[41, 42, 43, 44, 45, 126336, 126336, 126336],
        max_new_tokens=8,
        source_quality_score=0.28,
        history_selection_score=0.30,
        history_mask_count=1,
    )
    stable_config = repair.to_config(
        [31, 32, 33, 34],
        history_token_ids=[41, 42, 43, 44, 45, 126336, 126336, 126336],
        max_new_tokens=8,
        source_quality_score=0.28,
        history_selection_score=0.36,
        history_mask_count=1,
    )

    assert weak_config.initial_suffix_token_ids == (41, 42, 43, 44, None, None, None, None)
    assert stable_config.initial_suffix_token_ids == (41, 42, None, None, None, None, None, None)


def test_history_visible_repair_candidate_preserves_nonprefix_visible_tokens():
    repair = default_llada_history_visible_repair_candidates()[0]
    config = repair.to_config(
        [31, 32, 33, 34],
        history_token_ids=[41, 126336, 43, 126081],
        max_new_tokens=4,
    )

    assert repair.name == "history_visible_repair"
    assert config.initial_suffix_token_ids == (41, None, 43, None)


def test_default_llada_history_repair_pack_is_explicit():
    names = [repair.name for repair in default_llada_history_repair_candidates()]
    assert names == ["history_prefix_25_repair"]


def test_default_llada_history_repair_pack_accepts_multiple_fractions():
    repairs = default_llada_history_repair_candidates((0.25, 0.5))

    assert [repair.name for repair in repairs] == [
        "history_prefix_25_repair",
        "history_prefix_50_repair",
    ]
    assert [repair.keep_prefix_fraction for repair in repairs] == [0.25, 0.5]


def test_default_llada_state_adaptive_pack_contains_conditional_repairs():
    repairs = default_llada_state_adaptive_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "state_adaptive_history_repair",
        "prefix_25_repair",
        "state_adaptive_confidence_repair",
    ]
    assert repairs[0].adaptive_history_prefix is True
    assert repairs[2].quality_scaled_low_confidence is True


def test_default_llada_replay_consistency_pack_starts_with_replay_repair():
    repairs = default_llada_replay_consistency_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "replay_unstable_25_repair",
        "state_adaptive_history_repair",
        "prefix_25_repair",
    ]
    assert repairs[0].remask_history_unstable_fraction == 0.25


def test_llada_initial_suffix_tokens_are_applied_to_masked_suffix():
    tokens = torch.full((1, 6), 99, dtype=torch.long)
    tokens[:, :2] = torch.tensor([[1, 2]])

    _apply_initial_suffix_tokens(
        tokens,
        prompt_length=2,
        gen_length=4,
        initial_suffix_token_ids=(10, None, 12),
    )

    assert tokens.tolist() == [[1, 2, 10, 99, 12, 99]]


def test_token_position_repair_seed_masks_answer_span_with_context():
    seed = build_token_position_repair_seed(
        [10, 11, 12, 13, 14],
        mask_positions=[2],
        max_new_tokens=5,
        context_window=1,
    )

    assert seed == (10, None, None, None, 14)


def test_answer_span_repair_seed_masks_decoded_answer_text():
    pieces = {
        10: "Scratch. ",
        11: "Answer: ",
        12: "off",
        13: ".",
    }

    def decode(token_ids):
        return "".join(pieces[token_id] for token_id in token_ids)

    seed = build_answer_span_repair_seed(
        [10, 11, 12, 13],
        answer_text="off",
        max_new_tokens=4,
        token_decoder=decode,
    )

    assert seed == (10, 11, None, 13)


def test_answer_span_repair_seed_falls_back_to_tail_window():
    seed = build_answer_span_repair_seed(
        [10, 11, 12, 126081],
        answer_text="missing",
        max_new_tokens=4,
        token_decoder=lambda token_ids: "no answer here",
        fallback_tail_window=1,
    )

    assert seed == (10, 11, None, 126081)


def test_text_span_repair_seed_masks_multiple_verifier_targets():
    pieces = {
        10: "3*14 + ",
        11: "2*9 = ",
        12: "54",
        13: ". Answer: 12",
    }

    def decode(token_ids):
        return "".join(pieces[token_id] for token_id in token_ids)

    seed = build_text_span_repair_seed(
        [10, 11, 12, 13],
        target_texts=["3*14 + 2*9 = 54"],
        max_new_tokens=4,
        token_decoder=decode,
        context_window=0,
    )

    assert seed == (None, None, None, 13)


def test_text_span_repair_seed_reports_literal_localization_diagnostics():
    pieces = {
        10: "Keep. ",
        11: "Bad span",
        12: ". Done.",
    }

    def decode(token_ids):
        return "".join(pieces[token_id] for token_id in token_ids)

    seed, diagnostics = build_text_span_repair_seed_with_diagnostics(
        [10, 11, 12],
        target_texts=["Bad span"],
        max_new_tokens=3,
        token_decoder=decode,
    )

    assert seed == (10, None, 12)
    assert diagnostics["mode"] == "literal_span"
    assert diagnostics["literal_target_found"] is True
    assert diagnostics["used_fallback"] is False
    assert diagnostics["token_positions"] == [1]
    assert diagnostics["masked_positions"] == [1]
    assert diagnostics["matched_target_count"] == 1


def test_text_span_repair_seed_can_add_history_instability_positions():
    pieces = {
        10: "Keep. ",
        11: "Bad span",
        12: ". Stable ",
        13: "Risky.",
    }

    def decode(token_ids):
        return "".join(pieces[token_id] for token_id in token_ids)

    seed, diagnostics = build_text_span_repair_seed_with_diagnostics(
        [10, 11, 12, 13],
        target_texts=["Bad span"],
        max_new_tokens=4,
        history_instability_remask_fraction=0.25,
        history_samples_token_ids=[
            [10, 11, 12, 999],
            [10, 11, 12, 998],
            [10, 126336, 12, 997],
        ],
        token_decoder=decode,
    )

    assert seed == (10, None, 12, None)
    assert diagnostics["mode"] == "literal_span"
    assert diagnostics["history_instability_positions"] == [3]
    assert diagnostics["token_positions"] == [1, 3]
    assert diagnostics["masked_positions"] == [1, 3]


def test_text_span_repair_seed_reports_tail_fallback_diagnostics():
    diagnostics = build_text_span_repair_seed_diagnostics(
        [10, 11, 12, 126081],
        target_texts=["missing"],
        max_new_tokens=4,
        token_decoder=lambda token_ids: "no matching target",
        fallback_tail_window=1,
    )

    assert diagnostics["mode"] == "tail_window_fallback"
    assert diagnostics["literal_target_found"] is False
    assert diagnostics["used_fallback"] is True
    assert diagnostics["masked_positions"] == [2]
    assert diagnostics["seed_masked_positions"] == 1


def test_tail_window_repair_seed_ignores_likely_special_tokens():
    seed = build_tail_window_repair_seed(
        [10, 11, 126081, 126348],
        max_new_tokens=4,
        tail_window=1,
    )

    assert seed == (10, None, 126081, 126348)


def test_verifier_repair_candidate_masks_positions():
    repair = DiffusionVerifierRepairCandidate(name="answer_test", context_window=0)
    config = repair.to_config([41, 42, 43], max_new_tokens=3, mask_positions=[1])

    assert config.initial_suffix_token_ids == (41, None, 43)


def test_default_verifier_repair_pack_has_answer_span_and_random_context():
    names = [repair.name for repair in default_llada_verifier_repair_candidates()]
    assert names == ["answer_span_repair", "answer_context_random_repair"]


def test_default_constraint_span_history_instability_pack_masks_unstable_positions():
    repairs = default_llada_constraint_span_history_instability_repair_candidates()

    assert [repair.name for repair in repairs] == ["constraint_gap_span_history_instability_repair"]
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_default_constraint_span_phase_anchor_pack_selects_phase_anchor():
    repairs = default_llada_constraint_span_phase_anchor_repair_candidates()

    assert [repair.name for repair in repairs] == ["constraint_gap_span_phase_anchor_repair"]
    assert repairs[0].source_state == "pre_generation_phase_anchor"
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"
    assert "first safe repairable denoise skeleton" in str(repairs[0].prompt_repair_instruction)


def test_default_constraint_span_anchor_instability_pack_selects_anchor_and_masks_instability():
    repairs = default_llada_constraint_span_anchor_instability_repair_candidates()

    assert [repair.name for repair in repairs] == ["constraint_gap_span_anchor_instability_repair"]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_default_constraint_span_anchor_instability_gated_pack_has_gate_policy():
    repairs = default_llada_constraint_span_anchor_instability_gated_repair_candidates()

    assert [repair.name for repair in repairs] == ["constraint_gap_span_anchor_instability_gated_repair"]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_default_constraint_span_anchor_instability_prompt_gated_pack_has_prompt_gate_policy():
    repairs = default_llada_constraint_span_anchor_instability_prompt_gated_repair_candidates()

    assert [repair.name for repair in repairs] == ["constraint_gap_span_anchor_instability_prompt_gated_repair"]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].history_instability_gate_prompt_policy == "active_instability_instruction"
    assert "unstable across sampled denoise history" in str(repairs[0].prompt_repair_instruction)
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_default_constraint_span_anchor_instability_claim_gated_pack_has_claim_gate_policy():
    repairs = default_llada_constraint_span_anchor_instability_claim_gated_repair_candidates()

    assert [repair.name for repair in repairs] == ["constraint_gap_span_anchor_instability_claim_gated_repair"]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].history_instability_gate_prompt_policy == "active_instability_instruction"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert "Equalize token budget and prompt format" in str(repairs[0].planning_prompt_gate_instruction)
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_default_constraint_span_anchor_instability_claim_strict_gated_pack_forces_oracle_split():
    repairs = default_llada_constraint_span_anchor_instability_claim_strict_gated_repair_candidates()

    assert [repair.name for repair in repairs] == ["constraint_gap_span_anchor_instability_claim_strict_gated_repair"]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].history_instability_gate_prompt_policy == "active_instability_instruction"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert "separate oracle best-of results from selected results" in str(
        repairs[0].planning_prompt_gate_instruction
    )


def test_default_constraint_span_anchor_instability_claim_oracle_gated_pack_keeps_compact_oracle_split():
    repairs = default_llada_constraint_span_anchor_instability_claim_oracle_gated_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_oracle_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].history_instability_gate_prompt_policy == "active_instability_instruction"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    instruction = str(repairs[0].planning_prompt_gate_instruction)
    assert "Because extra tokens and a different prompt format are confounds" in instruction
    assert "separately report oracle best-of results and selected results" in instruction


def test_default_constraint_span_anchor_instability_claim_seeded_gated_pack_has_seed_anchor():
    repairs = default_llada_constraint_span_anchor_instability_claim_seeded_gated_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text == (
        " separate oracle best-of results from selected results."
    )
    assert "fixed oracle/selected-results seed anchor" in str(
        repairs[0].planning_prompt_gate_instruction
    )


def test_default_constraint_span_anchor_instability_claim_compatible_seeded_gated_pack_has_dual_seed_anchor():
    repairs = default_llada_constraint_span_anchor_instability_claim_compatible_seeded_gated_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_compatible_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text == (
        " oracle selected results; claim survives if disappears."
    )
    instruction = str(repairs[0].planning_prompt_gate_instruction)
    assert "oracle/selected results" in instruction
    assert "claim if the effect disappears" in instruction


def test_default_constraint_span_anchor_instability_claim_auto_seeded_gated_pack_uses_seed_policy():
    repairs = default_llada_constraint_span_anchor_instability_claim_auto_seeded_gated_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_auto_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text is None
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_control_terms"
    assert "compact fixed seed anchor" in str(repairs[0].planning_prompt_gate_instruction)


def test_default_constraint_span_anchor_instability_claim_auto_action_seeded_gated_pack_uses_seed_policy():
    repairs = default_llada_constraint_span_anchor_instability_claim_auto_action_seeded_gated_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_auto_action_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text is None
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_action_control_terms"
    assert "compact generated action" in str(repairs[0].planning_prompt_gate_instruction)


def test_default_constraint_span_anchor_instability_claim_auto_compat_seeded_gated_pack_uses_seed_policy():
    repairs = default_llada_constraint_span_anchor_instability_claim_auto_compat_seeded_gated_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_auto_compat_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text is None
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_compatibility_control_terms"
    instruction = str(repairs[0].planning_prompt_gate_instruction)
    assert "selected compact seed anchor" in instruction
    assert "oracle/selected results" in instruction


def test_default_constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated_pack_avoids_meta_seed_language():
    repairs = default_llada_constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_auto_compat_realized_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text is None
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_compatibility_control_terms"
    instruction = str(repairs[0].planning_prompt_gate_instruction)
    assert "separates oracle/best-of results from selected-run results" in instruction
    assert "Do not mention seeds, anchors, masks, or repair instructions" in instruction
    assert "seed anchor" not in instruction


def test_default_constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_pack_uses_direct_claim_preservation():
    repairs = default_llada_constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text is None
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_preservation_control_terms"
    instruction = str(repairs[0].planning_prompt_gate_instruction)
    assert "preserve only the public claim" in instruction
    assert "selected-run results" in instruction
    assert "Do not mention seeds, anchors, masks, or repair instructions" in instruction
    assert "seed anchor" not in instruction


def test_default_constraint_span_phase_hybrid_preserve_seeded_gated_pack_uses_phase_hybrid_source():
    repairs = default_llada_constraint_span_phase_hybrid_preserve_seeded_gated_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_phase_hybrid_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_preservation_control_terms"
    instruction = str(repairs[0].planning_prompt_gate_instruction)
    assert "preserve only the public claim" in instruction
    assert "selected-run results" in instruction
    assert "Do not mention seeds, anchors, masks, or repair instructions" in instruction


def test_default_constraint_span_anchor_instability_claim_auto_joint_seeded_gated_pack_uses_joint_seed_policy():
    repairs = default_llada_constraint_span_anchor_instability_claim_auto_joint_seeded_gated_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_auto_joint_seeded_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text is None
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_joint_control_terms"
    instruction = str(repairs[0].planning_prompt_gate_instruction)
    assert "selected-run results" in instruction
    assert "Do not mention seeds, anchors, masks, or repair instructions" in instruction


def test_default_constraint_span_anchor_instability_claim_auto_seeded_realization_gated_pack_uses_seed_policy():
    repairs = default_llada_constraint_span_anchor_instability_claim_auto_seeded_realization_gated_repair_candidates()

    assert [repair.name for repair in repairs] == [
        "constraint_gap_span_anchor_instability_claim_auto_seeded_realization_gated_repair"
    ]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction == 0.08
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].planning_prompt_gate_policy == "public_claim_confound_control"
    assert repairs[0].planning_prompt_gate_seed_suffix_text is None
    assert repairs[0].planning_prompt_gate_seed_suffix_policy == "compact_control_terms"
    instruction = str(repairs[0].planning_prompt_gate_instruction)
    assert "token budget" in instruction
    assert "Do not say compare to the anchor" in instruction


def test_default_constraint_span_anchor_instability_prompt_only_gated_pack_has_prompt_gate_without_remask():
    repairs = default_llada_constraint_span_anchor_instability_prompt_only_gated_repair_candidates()

    assert [repair.name for repair in repairs] == ["constraint_gap_span_anchor_instability_prompt_only_gated_repair"]
    assert repairs[0].source_state == "pre_generation_anchor"
    assert repairs[0].remask_history_unstable_fraction is None
    assert repairs[0].history_instability_gate_policy == "multi_span_low_quality"
    assert repairs[0].history_instability_gate_prompt_policy == "active_instability_instruction"
    assert "denoise history shows instability" in str(repairs[0].prompt_repair_instruction)
    assert repairs[0].planning_span_chunk_mode == "adaptive"
    assert repairs[0].planning_span_selection_policy == "compact"


def test_short_text_counterfactual_candidates_are_read_from_prompt_surface():
    task = GeneralReasoningTask(
        task_id="sym",
        family="symbolic",
        prompt="Answer only on or off.",
        answer_type="short_text",
        scorer="exact_short_text",
        answer="on",
        max_new_tokens=16,
    )

    assert counterfactual_answer_candidates(task, None) == ["on", "off"]


def test_counterfactual_candidates_exclude_extracted_short_text_answer():
    task = GeneralReasoningTask(
        task_id="sym",
        family="symbolic",
        prompt="Answer only on or off.",
        answer_type="short_text",
        scorer="exact_short_text",
        answer="on",
        max_new_tokens=16,
    )

    assert counterfactual_answer_candidates(task, "off") == ["on"]


def test_counterfactual_candidates_use_multiple_choice_letters():
    task = GeneralReasoningTask(
        task_id="sci",
        family="science",
        prompt="A) red B) blue",
        answer_type="multiple_choice",
        scorer="multiple_choice",
        answer="B",
        max_new_tokens=16,
        choices={"A": "red", "B": "blue"},
    )

    assert counterfactual_answer_candidates(task, "A") == ["B"]


def test_counterfactual_candidates_include_integer_prompt_solver():
    task = GeneralReasoningTask(
        task_id="math",
        family="math",
        prompt=(
            "A notebook starts with 96 pages. Four sections use 17, 23, 8, and 19 pages. "
            "How many blank pages remain? Answer with one integer."
        ),
        answer_type="integer",
        scorer="exact_integer",
        answer=29,
        max_new_tokens=16,
    )

    proposals = counterfactual_answer_proposals(task, 30)

    assert [(proposal.value, proposal.source) for proposal in proposals] == [
        ("29", "arithmetic_prompt_solver")
    ]


def test_integer_prompt_solver_handles_number_words():
    task = GeneralReasoningTask(
        task_id="math",
        family="math",
        prompt=(
            "Five identical machines make 1200 parts in 8 hours. At the same rate, "
            "how many parts do 3 machines make in 6 hours? Answer with one integer."
        ),
        answer_type="integer",
        scorer="exact_integer",
        answer=540,
        max_new_tokens=16,
    )

    assert counterfactual_answer_candidates(task, 360) == ["540"]


def test_counterfactual_candidates_include_symbolic_order_solver():
    task = GeneralReasoningTask(
        task_id="sym",
        family="symbolic",
        prompt=(
            "If D is before A, A is before B, and B is before C, "
            "what is the full order from first to last? Answer with the four letters separated by spaces."
        ),
        answer_type="short_text",
        scorer="exact_short_text",
        answer="D A B C",
        max_new_tokens=16,
    )

    assert counterfactual_answer_candidates(task, "A B C D") == ["D A B C"]


def test_counterfactual_candidates_include_symbolic_list_swap_solver():
    task = GeneralReasoningTask(
        task_id="sym",
        family="symbolic",
        prompt=(
            "Start with the list red, blue, green. Swap the first and third items, "
            "then swap the second and third items. What is the final list?"
        ),
        answer_type="short_text",
        scorer="exact_short_text",
        answer="green red blue",
        max_new_tokens=16,
    )

    assert counterfactual_answer_candidates(task, "green blue red") == ["green red blue"]


def test_counterfactual_candidates_include_symbolic_syllogism_solver():
    task = GeneralReasoningTask(
        task_id="sym",
        family="symbolic",
        prompt="All zargs are blicks. No blicks are morts. Can a zarg be a mort? Answer yes or no.",
        answer_type="short_text",
        scorer="exact_short_text",
        answer="no",
        max_new_tokens=16,
    )

    assert counterfactual_answer_candidates(task, "yes") == ["no"]


def test_counterfactual_candidates_include_letter_code_transform_solver():
    task = GeneralReasoningTask(
        task_id="sym",
        family="symbolic",
        prompt=(
            "A display starts with the code K L M. Rotate the code one step left, "
            "then swap the final two letters. What code should be displayed? "
            "Answer with the three letters separated by spaces."
        ),
        answer_type="short_text",
        scorer="exact_short_text",
        answer="L K M",
        max_new_tokens=16,
    )

    assert counterfactual_answer_candidates(task, "M L K") == ["L K M"]


def test_proposal_only_record_scores_without_counting_as_model_generation():
    task = GeneralReasoningTask(
        task_id="math",
        family="math",
        prompt="Answer with one integer.",
        answer_type="integer",
        scorer="exact_integer",
        answer=42,
        max_new_tokens=16,
    )
    baseline = {
        "candidate_key": "dream-7b-instruct-hf",
        "generation_stage": "baseline",
        "text": "41",
        "task": {"task_id": "math"},
        "schedule": {"name": "entropy_32"},
        "task_score": {"score": 0.0, "extracted_answer": 41},
        "trajectory_control_score": {"overall": 0.2},
        "combined_selection_score": 0.05,
    }

    proposal_record = _proposal_only_record(
        "dream-7b-instruct-hf",
        "Dream 7B",
        task,
        baseline,
        AnswerProposal(value="42", source="arithmetic_prompt_solver"),
    )
    proposal_selected = _select_best_record(task, baseline, [baseline, proposal_record])
    scores = summarize_scores(
        [baseline],
        [baseline],
        [baseline],
        proposal_only_records=[proposal_record],
        proposal_only_selected_records=[proposal_selected],
    )

    assert proposal_record["generation_stage"] == "proposal_only"
    assert proposal_record["is_model_generation"] is False
    assert proposal_record["task_score"]["score"] == 1.0
    assert scores["all_generation_count"] == 1
    assert scores["proposal_only_candidate_count"] == 1
    assert scores["proposal_only_selected_mean_task_score"] == 1.0
    assert scores["selected_task_delta_vs_proposal_only"] == -1.0
