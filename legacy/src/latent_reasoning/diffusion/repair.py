"""Diffusion-native repair candidates.

The repair surface is deliberately narrow: take a generated suffix, keep a
small prefix as a scaffold, remask the rest, and let a diffusion model denoise
the missing suffix again. This gives the project a real branch-and-repair loop
without adding an expensive model judge dependency.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import ceil

from latent_reasoning.diffusion.backends import DiffusionGenerationConfig

DEFAULT_LLADA_MASK_TOKEN_IDS = (126336,)
TokenDecoder = Callable[[list[int]], str]


@dataclass(frozen=True)
class DiffusionRepairCandidate:
    """A suffix-inpainting branch derived from an earlier denoise output."""

    name: str
    source_state: str = "final"
    keep_prefix_fraction: float = 0.0
    remask_low_confidence_fraction: float | None = None
    remask_history_unstable_fraction: float | None = None
    remask_text_policy: str | None = None
    text_context_window: int = 1
    fallback_remask_low_confidence_fraction: float | None = None
    prompt_repair_instruction: str | None = None
    prompt_repair_policy: str | None = None
    prompt_history_contrast: bool = False
    history_instability_gate_policy: str | None = None
    history_instability_gate_prompt_policy: str | None = None
    planning_prompt_gate_policy: str | None = None
    planning_prompt_gate_instruction: str | None = None
    planning_prompt_gate_seed_suffix_text: str | None = None
    planning_prompt_gate_seed_suffix_policy: str | None = None
    planning_span_chunk_mode: str = "sentence"
    planning_span_selection_policy: str = "top_ranked"
    quality_scaled_low_confidence: bool = False
    quality_scaled_min_fraction: float = 0.15
    quality_scaled_max_fraction: float = 0.40
    quality_scaled_floor: float = 0.25
    quality_scaled_ceiling: float = 0.55
    adaptive_history_prefix: bool = False
    adaptive_history_default_fraction: float = 0.25
    adaptive_history_weak_fraction: float = 0.50
    adaptive_history_source_quality_threshold: float = 0.30
    adaptive_history_score_threshold: float = 0.31
    adaptive_history_max_mask_fraction: float = 0.20
    steps: int = 32
    temperature: float = 0.0
    block_length: int | None = None
    remasking: str = "low_confidence"
    history_sample_count: int = 6

    def to_config(
        self,
        source_token_ids: list[int],
        *,
        max_new_tokens: int,
        token_confidences: list[float | None] | None = None,
        history_token_ids: list[int] | None = None,
        history_samples_token_ids: list[list[int]] | None = None,
        mask_token_ids: tuple[int, ...] = DEFAULT_LLADA_MASK_TOKEN_IDS,
        source_text: str = "",
        token_decoder: TokenDecoder | None = None,
        source_quality_score: float | None = None,
        history_selection_score: float | None = None,
        history_mask_count: int | None = None,
    ) -> DiffusionGenerationConfig:
        """Build an LLaDA-compatible inpainting config."""
        block_length = self.block_length or max_new_tokens
        if self.source_state == "final":
            if self.remask_text_policy:
                initial_suffix_token_ids = build_text_policy_repair_seed(
                    source_token_ids,
                    source_text=source_text,
                    token_decoder=token_decoder,
                    max_new_tokens=max_new_tokens,
                    policy=self.remask_text_policy,
                    context_window=self.text_context_window,
                    fallback_remask_low_confidence_fraction=self.fallback_remask_low_confidence_fraction,
                    token_confidences=token_confidences,
                )
            elif self.remask_history_unstable_fraction is not None:
                initial_suffix_token_ids = build_history_instability_repair_seed(
                    source_token_ids,
                    history_samples_token_ids=history_samples_token_ids,
                    max_new_tokens=max_new_tokens,
                    remask_fraction=self.remask_history_unstable_fraction,
                    mask_token_ids=mask_token_ids,
                    token_confidences=token_confidences,
                    fallback_remask_low_confidence_fraction=self.fallback_remask_low_confidence_fraction,
                )
            elif self.quality_scaled_low_confidence:
                initial_suffix_token_ids = build_low_confidence_repair_seed(
                    source_token_ids,
                    token_confidences=token_confidences,
                    max_new_tokens=max_new_tokens,
                    remask_fraction=_quality_scaled_remask_fraction(
                        source_quality_score,
                        min_fraction=self.quality_scaled_min_fraction,
                        max_fraction=self.quality_scaled_max_fraction,
                        quality_floor=self.quality_scaled_floor,
                        quality_ceiling=self.quality_scaled_ceiling,
                    ),
                )
            else:
                initial_suffix_token_ids = build_repair_seed(
                    source_token_ids,
                    token_confidences=token_confidences,
                    max_new_tokens=max_new_tokens,
                    keep_prefix_fraction=self.keep_prefix_fraction,
                    remask_low_confidence_fraction=self.remask_low_confidence_fraction,
                )
        elif self.source_state == "history":
            keep_prefix_fraction = self.keep_prefix_fraction or None
            if self.adaptive_history_prefix:
                keep_prefix_fraction = _adaptive_history_prefix_fraction(
                    source_quality_score=source_quality_score,
                    history_selection_score=history_selection_score,
                    history_mask_count=history_mask_count,
                    max_new_tokens=max_new_tokens,
                    default_fraction=self.adaptive_history_default_fraction,
                    weak_fraction=self.adaptive_history_weak_fraction,
                    source_quality_threshold=self.adaptive_history_source_quality_threshold,
                    history_score_threshold=self.adaptive_history_score_threshold,
                    max_mask_fraction=self.adaptive_history_max_mask_fraction,
                )
            initial_suffix_token_ids = build_history_state_repair_seed(
                history_token_ids or source_token_ids,
                max_new_tokens=max_new_tokens,
                mask_token_ids=mask_token_ids,
                keep_prefix_fraction=keep_prefix_fraction,
            )
        else:
            raise ValueError("source_state must be 'final' or 'history'")
        return DiffusionGenerationConfig(
            max_new_tokens=max_new_tokens,
            steps=self.steps,
            temperature=self.temperature,
            algorithm="low_confidence",
            block_length=block_length,
            remasking=self.remasking,
            output_history=True,
            history_sample_count=self.history_sample_count,
            initial_suffix_token_ids=initial_suffix_token_ids,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DiffusionVerifierRepairCandidate:
    """A repair branch that remasks verifier-identified answer positions."""

    name: str
    context_window: int = 0
    history_instability_remask_fraction: float | None = None
    steps: int = 32
    temperature: float = 0.0
    block_length: int | None = None
    remasking: str = "low_confidence"
    history_sample_count: int = 6

    def to_config(
        self,
        source_token_ids: list[int],
        *,
        max_new_tokens: int,
        mask_positions: list[int],
    ) -> DiffusionGenerationConfig:
        block_length = self.block_length or max_new_tokens
        return DiffusionGenerationConfig(
            max_new_tokens=max_new_tokens,
            steps=self.steps,
            temperature=self.temperature,
            algorithm="low_confidence",
            block_length=block_length,
            remasking=self.remasking,
            output_history=True,
            history_sample_count=self.history_sample_count,
            initial_suffix_token_ids=build_token_position_repair_seed(
                source_token_ids,
                mask_positions=mask_positions,
                max_new_tokens=max_new_tokens,
                context_window=self.context_window,
            ),
        )

    def to_answer_span_config(
        self,
        source_token_ids: list[int],
        *,
        answer_text: object | None,
        max_new_tokens: int,
        source_text: str = "",
        token_decoder: TokenDecoder | None = None,
        fallback_tail_window: int = 2,
    ) -> DiffusionGenerationConfig:
        block_length = self.block_length or max_new_tokens
        return DiffusionGenerationConfig(
            max_new_tokens=max_new_tokens,
            steps=self.steps,
            temperature=self.temperature,
            algorithm="low_confidence",
            block_length=block_length,
            remasking=self.remasking,
            output_history=True,
            history_sample_count=self.history_sample_count,
            initial_suffix_token_ids=build_answer_span_repair_seed(
                source_token_ids,
                answer_text=answer_text,
                max_new_tokens=max_new_tokens,
                source_text=source_text,
                token_decoder=token_decoder,
                context_window=self.context_window,
                fallback_tail_window=fallback_tail_window,
            ),
        )

    def to_text_span_config(
        self,
        source_token_ids: list[int],
        *,
        target_texts: list[object],
        max_new_tokens: int,
        history_samples_token_ids: list[list[int]] | None = None,
        source_text: str = "",
        token_decoder: TokenDecoder | None = None,
        fallback_tail_window: int = 2,
    ) -> DiffusionGenerationConfig:
        block_length = self.block_length or max_new_tokens
        return DiffusionGenerationConfig(
            max_new_tokens=max_new_tokens,
            steps=self.steps,
            temperature=self.temperature,
            algorithm="low_confidence",
            block_length=block_length,
            remasking=self.remasking,
            output_history=True,
            history_sample_count=self.history_sample_count,
            initial_suffix_token_ids=build_text_span_repair_seed(
                source_token_ids,
                target_texts=target_texts,
                max_new_tokens=max_new_tokens,
                source_text=source_text,
                token_decoder=token_decoder,
                context_window=self.context_window,
                fallback_tail_window=fallback_tail_window,
                history_instability_remask_fraction=self.history_instability_remask_fraction,
                history_samples_token_ids=history_samples_token_ids,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_llada_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Cheap first repair pack for local GPU scout runs."""
    return (
        DiffusionRepairCandidate(name="prefix_25_repair", keep_prefix_fraction=0.25),
        DiffusionRepairCandidate(name="prefix_50_repair", keep_prefix_fraction=0.50),
        DiffusionRepairCandidate(
            name="low_confidence_25_repair",
            remask_low_confidence_fraction=0.25,
        ),
        DiffusionRepairCandidate(
            name="low_confidence_40_repair",
            remask_low_confidence_fraction=0.40,
        ),
    )


def default_llada_source_relative_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Minimal-edit repair pack for source-relative improvement tests."""
    return (
        DiffusionRepairCandidate(
            name="low_confidence_15_repair",
            remask_low_confidence_fraction=0.15,
        ),
        DiffusionRepairCandidate(
            name="low_confidence_25_repair",
            remask_low_confidence_fraction=0.25,
        ),
        DiffusionRepairCandidate(name="prefix_25_repair", keep_prefix_fraction=0.25),
        DiffusionRepairCandidate(
            name="low_confidence_40_repair",
            remask_low_confidence_fraction=0.40,
        ),
    )


def default_llada_targeted_content_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Repair pack that remasks low-value generated content spans."""
    return (
        DiffusionRepairCandidate(
            name="targeted_filler_repair",
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
        ),
        DiffusionRepairCandidate(
            name="targeted_filler_wide_repair",
            remask_text_policy="generic_filler",
            text_context_window=2,
            fallback_remask_low_confidence_fraction=0.40,
        ),
        DiffusionRepairCandidate(name="prefix_25_repair", keep_prefix_fraction=0.25),
        DiffusionRepairCandidate(
            name="low_confidence_25_repair",
            remask_low_confidence_fraction=0.25,
        ),
    )


def default_llada_prompt_guided_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Repair pack that lets diffusion rewrite a source draft under a generic critique."""
    return (
        DiffusionRepairCandidate(
            name="prompt_guided_revision_repair",
            keep_prefix_fraction=0.0,
            prompt_repair_instruction=(
                "Rewrite the draft answer directly. Keep useful concrete steps, "
                "remove repetition or filler, and add any missing causal checks, "
                "measurements, constraints, risk controls, rollback or fallback "
                "steps, and decision thresholds that are implied by the task. "
                "Do not mention that you are revising a draft."
            ),
        ),
        DiffusionRepairCandidate(
            name="prompt_guided_revision_anchor25_repair",
            keep_prefix_fraction=0.25,
            prompt_repair_instruction=(
                "Continue and repair the draft answer directly. Preserve useful "
                "specific setup, remove repetition or filler, and complete the "
                "answer with causal checks, measurements, constraints, risks, "
                "rollback or fallback steps, and decision thresholds implied by "
                "the task."
            ),
        ),
        DiffusionRepairCandidate(
            name="targeted_filler_repair",
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
        ),
    )


def default_llada_constraint_gap_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Repair pack that combines canonical history repair with prompt-gap revision."""
    return (
        DiffusionRepairCandidate(
            name="state_adaptive_history_repair",
            source_state="history",
            adaptive_history_prefix=True,
        ),
        DiffusionRepairCandidate(name="prefix_25_repair", keep_prefix_fraction=0.25),
        DiffusionRepairCandidate(
            name="constraint_gap_revision_repair",
            keep_prefix_fraction=0.0,
            prompt_repair_policy="constraint_gap",
            prompt_repair_instruction=(
                "Rewrite the draft answer directly. Preserve useful concrete "
                "steps, but add missing task-specific measurements, decision "
                "rules, constraints, risk controls, rollback or fallback paths, "
                "and stop conditions. Tie every added step to the original task."
            ),
        ),
        DiffusionRepairCandidate(
            name="constraint_gap_revision_anchor25_repair",
            keep_prefix_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            prompt_repair_instruction=(
                "Continue and repair the draft answer directly. Keep the useful "
                "opening, then add the missing task-specific constraints, "
                "measurements, decision rules, risks, fallback paths, and stop "
                "conditions from the original task."
            ),
        ),
        DiffusionRepairCandidate(
            name="constraint_gap_span_repair",
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly. Preserve useful "
                "source details, but use the missing task terms to add the "
                "specific constraints, measurements, decision rules, risks, "
                "fallback paths, and stop conditions implied by the task."
            ),
        ),
    )


def default_llada_constraint_span_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Low-budget prompt-gap span repair pack for models where full revision is a no-op."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_repair",
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly. Preserve useful "
                "source details, but use the missing task terms to add the "
                "specific constraints, measurements, decision rules, risks, "
                "fallback paths, and stop conditions implied by the task."
            ),
        ),
    )


def default_llada_constraint_span_history_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Prompt-gap span repair seeded from a sampled denoise-history state."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_history_repair",
            source_state="history",
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from this partial "
                "denoise state. Preserve useful visible constraints, but use "
                "the missing task terms to add measurements, decision rules, "
                "risks, fallback paths, and stop conditions implied by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_select_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Prompt-gap span repair with a runner-selected final/history anchor."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_select_repair",
            source_state="pre_generation_anchor",
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. Preserve useful visible constraints, but use "
                "the missing task terms to add measurements, decision rules, "
                "risks, fallback paths, and stop conditions implied by the task."
            ),
        ),
    )


def default_llada_constraint_span_phase_anchor_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Prompt-gap span repair seeded from the first safe repairable denoise phase."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_phase_anchor_repair",
            source_state="pre_generation_phase_anchor",
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the first safe "
                "repairable denoise skeleton. Preserve stable visible constraints, "
                "and use missing task terms to add measurements, decision rules, "
                "risks, fallback paths, and stop conditions implied by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Runner-selected final/history anchor plus denoise-instability remasking."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_gated_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Anchor-selected span repair with conditional denoise-instability remasking."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_gated_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. Preserve useful visible constraints, but use "
                "the missing task terms to add measurements, decision rules, "
                "risks, fallback paths, and stop conditions implied by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_prompt_gated_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Anchor-selected span repair with conditional instability masks and prompt."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_prompt_gated_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_claim_gated_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Prompt-gated instability repair plus public-claim confound control."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_claim_gated_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_prompt_gate_policy="public_claim_confound_control",
            planning_prompt_gate_instruction=(
                "Repair the masked continuation as a falsification/control plan. "
                "Equalize token budget and prompt format before comparing, rerun "
                "baseline and intervention on locked tasks, separate selected-run "
                "results from extra-sampling or best-of results, record regressions "
                "as well as wins, and state what public claim survives if the effect "
                "disappears. Preserve stable source details and answer directly."
            ),
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_claim_strict_gated_repair_candidates() -> tuple[
    DiffusionRepairCandidate, ...
]:
    """Claim-gated repair that explicitly forces oracle/best-of separation."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_claim_strict_gated_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_prompt_gate_policy="public_claim_confound_control",
            planning_prompt_gate_instruction=(
                "Repair the masked continuation as a falsification/control plan. "
                "Include these controls explicitly: equal token budget, identical "
                "prompt format, locked-task reruns, separate oracle best-of results "
                "from selected results, record regressions and wins, then state the "
                "public claim that survives if the effect disappears. Explain the "
                "confound briefly and answer directly."
            ),
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_claim_oracle_gated_repair_candidates() -> tuple[
    DiffusionRepairCandidate, ...
]:
    """Claim-gated repair with compact oracle-vs-selected result separation."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_claim_oracle_gated_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_prompt_gate_policy="public_claim_confound_control",
            planning_prompt_gate_instruction=(
                "Repair the masked continuation as a compact falsification/control "
                "plan. Because extra tokens and a different prompt format are "
                "confounds, equalize token budget and prompt format, rerun baseline "
                "and intervention on locked tasks, separately report oracle best-of "
                "results and selected results, record regressions and wins, validate "
                "failure modes, and state what public claim survives if the effect "
                "disappears. Preserve stable source details and answer directly."
            ),
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_claim_seeded_gated_repair_candidates() -> tuple[
    DiffusionRepairCandidate, ...
]:
    """Oracle-aware claim gate with a fixed denoise seed anchor for the missing control."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_claim_seeded_gated_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_prompt_gate_policy="public_claim_confound_control",
            planning_prompt_gate_seed_suffix_text=" separate oracle best-of results from selected results.",
            planning_prompt_gate_instruction=(
                "Repair the masked continuation as a compact falsification/control "
                "plan. Because extra tokens and a different prompt format are "
                "confounds, equalize token budget and prompt format, rerun baseline "
                "and intervention on locked tasks, record regressions and wins, "
                "validate failure modes, and state what public claim survives if "
                "the effect disappears. Preserve the fixed oracle/selected-results "
                "seed anchor as a natural control step and answer directly."
            ),
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_claim_compatible_seeded_gated_repair_candidates() -> tuple[
    DiffusionRepairCandidate, ...
]:
    """Claim gate with a compact seed anchor that carries both required controls."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_claim_compatible_seeded_gated_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_prompt_gate_policy="public_claim_confound_control",
            planning_prompt_gate_seed_suffix_text=" oracle selected results; claim survives if disappears.",
            planning_prompt_gate_instruction=(
                "Repair the masked continuation as a compact falsification/control "
                "plan. Because extra tokens and a different prompt format are "
                "confounds, equalize token budget and prompt format, rerun baseline "
                "and intervention on locked tasks, record regressions and wins, "
                "validate failure modes, and preserve the fixed seed anchor as one "
                "natural sentence about oracle/selected results and the surviving "
                "claim if the effect disappears. Answer directly."
            ),
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_claim_auto_seeded_gated_repair_candidates() -> tuple[
    DiffusionRepairCandidate, ...
]:
    """Claim gate with an automatically synthesized compact control seed anchor."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_claim_auto_seeded_gated_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_prompt_gate_policy="public_claim_confound_control",
            planning_prompt_gate_seed_suffix_policy="compact_control_terms",
            planning_prompt_gate_instruction=(
                "Repair the masked continuation as a compact falsification/control "
                "plan. Because extra tokens and a different prompt format are "
                "confounds, equalize token budget and prompt format, rerun baseline "
                "and intervention on locked tasks, record regressions and wins, "
                "validate failure modes, and preserve the compact fixed seed anchor "
                "as one natural control sentence. Answer directly."
            ),
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_claim_auto_action_seeded_gated_repair_candidates() -> tuple[
    DiffusionRepairCandidate, ...
]:
    """Claim gate with an automatic compact seed that includes direct action verbs."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_claim_auto_action_seeded_gated_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_prompt_gate_policy="public_claim_confound_control",
            planning_prompt_gate_seed_suffix_policy="compact_action_control_terms",
            planning_prompt_gate_instruction=(
                "Repair the masked continuation as a compact falsification/control "
                "plan. Because extra tokens and a different prompt format are "
                "confounds, equalize token budget and prompt format, rerun baseline "
                "and intervention on locked tasks, record regressions and wins, "
                "validate failure modes, and preserve the compact generated action "
                "seed as a natural control clause. Answer directly."
            ),
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_claim_auto_compat_seeded_gated_repair_candidates() -> tuple[
    DiffusionRepairCandidate, ...
]:
    """Claim gate with a scored automatic seed chosen for control compatibility."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_claim_auto_compat_seeded_gated_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_prompt_gate_policy="public_claim_confound_control",
            planning_prompt_gate_seed_suffix_policy="compact_compatibility_control_terms",
            planning_prompt_gate_instruction=(
                "Repair the masked continuation as a compact falsification/control "
                "plan. Because extra tokens and a different prompt format are "
                "confounds, equalize token budget and prompt format, rerun baseline "
                "and intervention on locked tasks, record regressions and wins, "
                "validate failure modes, and preserve the selected compact seed "
                "anchor as one natural sentence about oracle/selected results and "
                "the surviving claim if the effect disappears. Answer directly."
            ),
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated_repair_candidates() -> tuple[
    DiffusionRepairCandidate, ...
]:
    """Auto compatibility seed with prompt language optimized for direct realization."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_claim_auto_compat_realized_seeded_gated_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_prompt_gate_policy="public_claim_confound_control",
            planning_prompt_gate_seed_suffix_policy="compact_compatibility_control_terms",
            planning_prompt_gate_instruction=(
                "Repair the masked continuation as a compact falsification/control "
                "plan. Because extra tokens and a different prompt format are "
                "confounds, equalize token budget and prompt format, rerun baseline "
                "and intervention on locked tasks, record regressions and wins, "
                "validate failure modes, and finish with a direct clause that "
                "separates oracle/best-of results from selected-run results and "
                "states what claim survives if the effect disappears. Do not "
                "mention seeds, anchors, masks, or repair instructions. Answer directly."
            ),
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_repair_candidates() -> tuple[
    DiffusionRepairCandidate, ...
]:
    """Auto compatibility seed with direct non-meta claim-preservation wording."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_prompt_gate_policy="public_claim_confound_control",
            planning_prompt_gate_seed_suffix_policy="compact_preservation_control_terms",
            planning_prompt_gate_instruction=(
                "Repair the masked continuation as a compact falsification/control "
                "plan. Because extra tokens and a different prompt format are "
                "confounds, equalize token budget and prompt format, rerun baseline "
                "and intervention on locked tasks, record regressions and wins, "
                "validate failure modes, separate oracle/best-of results from "
                "selected-run results, and preserve only the public claim that "
                "survives if the effect disappears. Do not mention seeds, anchors, "
                "masks, or repair instructions. Answer directly."
            ),
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_phase_hybrid_preserve_seeded_gated_repair_candidates() -> tuple[
    DiffusionRepairCandidate, ...
]:
    """Preservation-seeded repair with phase history used only when source-advantaged."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair",
            source_state="pre_generation_phase_hybrid_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_prompt_gate_policy="public_claim_confound_control",
            planning_prompt_gate_seed_suffix_policy="compact_preservation_control_terms",
            planning_prompt_gate_instruction=(
                "Repair the masked continuation as a compact falsification/control "
                "plan. Because extra tokens and a different prompt format are "
                "confounds, equalize token budget and prompt format, rerun baseline "
                "and intervention on locked tasks, record regressions and wins, "
                "validate failure modes, separate oracle/best-of results from "
                "selected-run results, and preserve only the public claim that "
                "survives if the effect disappears. Do not mention seeds, anchors, "
                "masks, or repair instructions. Answer directly."
            ),
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_phase_final_preserve_seeded_gated_repair_candidates() -> tuple[
    DiffusionRepairCandidate, ...
]:
    """Preservation-seeded repair that uses phase evidence for spend gating, not source replacement."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_phase_final_preserve_seeded_gated_repair",
            source_state="final",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_prompt_gate_policy="public_claim_confound_control",
            planning_prompt_gate_seed_suffix_policy="compact_preservation_control_terms",
            planning_prompt_gate_instruction=(
                "Repair the masked continuation as a compact falsification/control "
                "plan. Because extra tokens and a different prompt format are "
                "confounds, equalize token budget and prompt format, rerun baseline "
                "and intervention on locked tasks, record regressions and wins, "
                "validate failure modes, separate oracle/best-of results from "
                "selected-run results, and preserve only the public claim that "
                "survives if the effect disappears. Do not mention seeds, anchors, "
                "masks, or repair instructions. Answer directly."
            ),
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_claim_auto_joint_seeded_gated_repair_candidates() -> tuple[
    DiffusionRepairCandidate, ...
]:
    """Claim gate with a compact seed chosen by compatibility plus realization objective."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_claim_auto_joint_seeded_gated_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_prompt_gate_policy="public_claim_confound_control",
            planning_prompt_gate_seed_suffix_policy="compact_joint_control_terms",
            planning_prompt_gate_instruction=(
                "Repair the masked continuation as a compact falsification/control "
                "plan. Because extra tokens and a different prompt format are "
                "confounds, equalize token budget and prompt format, rerun baseline "
                "and intervention on locked tasks, record regressions and wins, "
                "validate failure modes, and finish with a direct clause that "
                "separates oracle/best-of results from selected-run results and "
                "states what claim survives if the effect disappears. Do not "
                "mention seeds, anchors, masks, or repair instructions. Answer directly."
            ),
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_claim_auto_seeded_realization_gated_repair_candidates() -> tuple[
    DiffusionRepairCandidate, ...
]:
    """Automatic compact control seed with explicit realization constraints."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_claim_auto_seeded_realization_gated_repair",
            source_state="pre_generation_anchor",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_prompt_gate_policy="public_claim_confound_control",
            planning_prompt_gate_seed_suffix_policy="compact_control_terms",
            planning_prompt_gate_instruction=(
                "Repair the masked continuation as one direct falsification/control "
                "plan sentence. Keep the explicit control words: token budget, prompt "
                "format, locked tasks, regressions, wins, and failure modes. Then "
                "integrate the generated compact seed anchor as the final "
                "oracle/selected-results and surviving-claim clause. Do not say "
                "compare to the anchor or discuss the seed. Answer directly."
            ),
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The mask also includes a small number of positions "
                "that were unstable across sampled denoise history. Preserve stable "
                "source details, and use missing task terms to add measurements, "
                "decision rules, risks, fallback paths, and stop conditions implied "
                "by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_instability_prompt_only_gated_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Anchor-selected span repair with conditional instability prompt only."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_instability_prompt_only_gated_repair",
            source_state="pre_generation_anchor",
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            history_instability_gate_policy="multi_span_low_quality",
            history_instability_gate_prompt_policy="active_instability_instruction",
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise anchor. The denoise history shows instability around "
                "the targeted reasoning spans, so rewrite the masked continuation "
                "as a corrected causal test rather than copying the unstable draft. "
                "Preserve stable source details, and use missing task terms to add "
                "measurements, decision rules, risks, fallback paths, and stop "
                "conditions implied by the task."
            ),
        ),
    )


def default_llada_constraint_span_anchor_search_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Prompt-gap span repair with runner search over sampled history anchors."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_anchor_search_repair",
            source_state="pre_generation_anchor_search",
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly from the selected "
                "denoise-history search anchor. Preserve useful visible constraints, "
                "but use the missing task terms to add measurements, decision rules, "
                "risks, fallback paths, and stop conditions implied by the task."
            ),
        ),
    )


def default_llada_constraint_span_history_contrast_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Final-source span repair with a compact denoise-history contrast in the prompt."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_history_contrast_repair",
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            prompt_history_contrast=True,
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly. Preserve useful final-source "
                "details, use the denoise-history contrast only as supporting evidence, "
                "and add the specific constraints, measurements, decision rules, risks, "
                "fallback paths, and stop conditions implied by the task."
            ),
        ),
    )


def default_llada_constraint_span_history_instability_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Final-source span repair plus a small denoise-history instability mask."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_history_instability_repair",
            remask_history_unstable_fraction=0.08,
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            planning_span_chunk_mode="adaptive",
            planning_span_selection_policy="compact",
            prompt_repair_instruction=(
                "Repair the masked or weak spans directly. The mask also includes "
                "a small number of final-source positions that were unstable across "
                "denoise history. Preserve stable source details, and use missing "
                "task terms to add measurements, decision rules, risks, fallback "
                "paths, and stop conditions implied by the task."
            ),
        ),
    )


def default_llada_constraint_span_clause_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Diagnostic span repair pack that targets clauses inside long planning sentences."""
    return (
        DiffusionRepairCandidate(
            name="constraint_gap_span_clause_repair",
            remask_text_policy="generic_filler",
            text_context_window=0,
            fallback_remask_low_confidence_fraction=0.25,
            prompt_repair_policy="constraint_gap",
            planning_span_chunk_mode="clause",
            prompt_repair_instruction=(
                "Repair the masked or weak clauses directly. Preserve useful "
                "source details, but use the missing task terms to add the "
                "specific constraints, measurements, decision rules, risks, "
                "fallback paths, and stop conditions implied by the task."
            ),
        ),
    )


def default_llada_state_adaptive_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Repair pack whose mask shape changes with source and history-state quality."""
    return (
        DiffusionRepairCandidate(
            name="state_adaptive_history_repair",
            source_state="history",
            adaptive_history_prefix=True,
        ),
        DiffusionRepairCandidate(name="prefix_25_repair", keep_prefix_fraction=0.25),
        DiffusionRepairCandidate(
            name="state_adaptive_confidence_repair",
            quality_scaled_low_confidence=True,
        ),
    )


def default_llada_replay_consistency_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Repair pack that remasks positions unstable across the denoise replay."""
    return (
        DiffusionRepairCandidate(
            name="replay_unstable_25_repair",
            remask_history_unstable_fraction=0.25,
            fallback_remask_low_confidence_fraction=0.25,
        ),
        DiffusionRepairCandidate(
            name="state_adaptive_history_repair",
            source_state="history",
            adaptive_history_prefix=True,
        ),
        DiffusionRepairCandidate(name="prefix_25_repair", keep_prefix_fraction=0.25),
    )


def default_llada_history_repair_candidates(
    keep_prefix_fractions: tuple[float, ...] = (0.25,),
) -> tuple[DiffusionRepairCandidate, ...]:
    """Cheap repair pack that branches from a sampled denoise-history state."""
    return tuple(
        DiffusionRepairCandidate(
            name=f"history_prefix_{_fraction_label(fraction)}_repair",
            source_state="history",
            keep_prefix_fraction=fraction,
        )
        for fraction in keep_prefix_fractions
    )


def default_llada_history_visible_repair_candidates() -> tuple[DiffusionRepairCandidate, ...]:
    """Repair from all visible tokens in a sampled denoise-history state."""
    return (
        DiffusionRepairCandidate(
            name="history_visible_repair",
            source_state="history",
        ),
    )


def default_llada_verifier_repair_candidates() -> tuple[DiffusionVerifierRepairCandidate, ...]:
    """Verifier-guided repair pack for exact-answer failures."""
    return (
        DiffusionVerifierRepairCandidate(name="answer_span_repair"),
        DiffusionVerifierRepairCandidate(
            name="answer_context_random_repair",
            context_window=2,
            temperature=0.5,
            remasking="random",
        ),
    )


def build_repair_seed(
    source_token_ids: list[int],
    *,
    token_confidences: list[float | None] | None,
    max_new_tokens: int,
    keep_prefix_fraction: float,
    remask_low_confidence_fraction: float | None,
) -> tuple[int | None, ...]:
    """Build an inpainting seed from either prefix or confidence repair policy."""
    if remask_low_confidence_fraction is None:
        return build_suffix_repair_seed(
            source_token_ids,
            max_new_tokens=max_new_tokens,
            keep_prefix_fraction=keep_prefix_fraction,
        )
    return build_low_confidence_repair_seed(
        source_token_ids,
        token_confidences=token_confidences,
        max_new_tokens=max_new_tokens,
        remask_fraction=remask_low_confidence_fraction,
    )


def build_suffix_repair_seed(
    source_token_ids: list[int],
    *,
    max_new_tokens: int,
    keep_prefix_fraction: float,
) -> tuple[int | None, ...]:
    """Keep a prefix from an old suffix and remask the remaining positions."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if keep_prefix_fraction < 0.0 or keep_prefix_fraction > 1.0:
        raise ValueError("keep_prefix_fraction must be between 0 and 1")

    usable_token_ids = source_token_ids[:max_new_tokens]
    keep_count = _prefix_keep_count(
        token_count=len(usable_token_ids),
        keep_prefix_fraction=keep_prefix_fraction,
    )
    kept = tuple(usable_token_ids[:keep_count])
    remasked_count = max_new_tokens - len(kept)
    return (*kept, *(None for _ in range(remasked_count)))


def build_low_confidence_repair_seed(
    source_token_ids: list[int],
    *,
    token_confidences: list[float | None] | None,
    max_new_tokens: int,
    remask_fraction: float,
) -> tuple[int | None, ...]:
    """Keep high-confidence suffix tokens and remask the weakest positions."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if remask_fraction <= 0.0 or remask_fraction > 1.0:
        raise ValueError("remask_fraction must be greater than 0 and at most 1")
    if not token_confidences:
        return build_suffix_repair_seed(
            source_token_ids,
            max_new_tokens=max_new_tokens,
            keep_prefix_fraction=max(0.0, 1.0 - remask_fraction),
        )

    usable_token_ids = source_token_ids[:max_new_tokens]
    seed: list[int | None] = [*usable_token_ids, *(None for _ in range(max_new_tokens - len(usable_token_ids)))]
    scored_positions = [
        (index, float(confidence))
        for index, confidence in enumerate(token_confidences[: len(usable_token_ids)])
        if isinstance(confidence, int | float)
    ]
    if not scored_positions:
        return tuple(seed)

    remask_count = max(1, min(len(scored_positions), ceil(len(scored_positions) * remask_fraction)))
    low_confidence_positions = {
        index for index, _confidence in sorted(scored_positions, key=lambda item: item[1])[:remask_count]
    }
    for index in low_confidence_positions:
        seed[index] = None
    return tuple(seed)


def build_history_instability_repair_seed(
    source_token_ids: list[int],
    *,
    history_samples_token_ids: list[list[int]] | None,
    max_new_tokens: int,
    remask_fraction: float,
    mask_token_ids: tuple[int, ...] = DEFAULT_LLADA_MASK_TOKEN_IDS,
    token_confidences: list[float | None] | None = None,
    fallback_remask_low_confidence_fraction: float | None = None,
) -> tuple[int | None, ...]:
    """Remask final positions that were unstable across sampled denoise states."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if remask_fraction <= 0.0 or remask_fraction > 1.0:
        raise ValueError("remask_fraction must be greater than 0 and at most 1")
    usable_token_ids = source_token_ids[:max_new_tokens]
    seed: list[int | None] = [*usable_token_ids, *(None for _ in range(max_new_tokens - len(usable_token_ids)))]
    scored_positions = _history_instability_scores(
        usable_token_ids,
        history_samples_token_ids=history_samples_token_ids,
        max_new_tokens=max_new_tokens,
        mask_token_ids=mask_token_ids,
    )
    if not scored_positions:
        if fallback_remask_low_confidence_fraction is not None:
            return build_low_confidence_repair_seed(
                source_token_ids,
                token_confidences=token_confidences,
                max_new_tokens=max_new_tokens,
                remask_fraction=fallback_remask_low_confidence_fraction,
            )
        return tuple(seed)
    remask_count = max(1, min(len(scored_positions), ceil(len(usable_token_ids) * remask_fraction)))
    for index, _score in sorted(scored_positions, key=lambda item: item[1], reverse=True)[:remask_count]:
        seed[index] = None
    return tuple(seed)


def _history_instability_scores(
    source_token_ids: list[int],
    *,
    history_samples_token_ids: list[list[int]] | None,
    max_new_tokens: int,
    mask_token_ids: tuple[int, ...],
) -> list[tuple[int, float]]:
    if not history_samples_token_ids:
        return []
    mask_ids = set(mask_token_ids)
    scores: list[tuple[int, float]] = []
    usable_count = min(len(source_token_ids), max_new_tokens)
    sample_count = len(history_samples_token_ids)
    for index in range(usable_count):
        sample_tokens = [
            sample[index] if index < len(sample) else None
            for sample in history_samples_token_ids
        ]
        unresolved_count = sum(
            1
            for token_id in sample_tokens
            if token_id is None or token_id in mask_ids or (isinstance(token_id, int) and _looks_like_special_token(token_id))
        )
        visible_tokens = [
            token_id
            for token_id in sample_tokens
            if isinstance(token_id, int) and token_id not in mask_ids and not _looks_like_special_token(token_id)
        ]
        unique_visible = len(set(visible_tokens))
        if unresolved_count == 0 and unique_visible <= 1:
            continue
        final_token = source_token_ids[index]
        final_disagreement = 0.0 if final_token in visible_tokens or not visible_tokens else 0.25
        score = (unresolved_count / sample_count) + max(0, unique_visible - 1) * 0.50 + final_disagreement
        scores.append((index, score))
    return scores


def _quality_scaled_remask_fraction(
    source_quality_score: float | None,
    *,
    min_fraction: float,
    max_fraction: float,
    quality_floor: float,
    quality_ceiling: float,
) -> float:
    if min_fraction <= 0.0 or max_fraction > 1.0 or min_fraction > max_fraction:
        raise ValueError("quality-scaled remask fractions must satisfy 0 < min <= max <= 1")
    if quality_floor >= quality_ceiling:
        raise ValueError("quality_floor must be less than quality_ceiling")
    if source_quality_score is None:
        return max_fraction
    quality = max(quality_floor, min(quality_ceiling, float(source_quality_score)))
    normalized = (quality - quality_floor) / (quality_ceiling - quality_floor)
    return max_fraction - normalized * (max_fraction - min_fraction)


def build_text_policy_repair_seed(
    source_token_ids: list[int],
    *,
    source_text: str,
    token_decoder: TokenDecoder | None,
    max_new_tokens: int,
    policy: str,
    context_window: int = 1,
    fallback_remask_low_confidence_fraction: float | None = None,
    token_confidences: list[float | None] | None = None,
) -> tuple[int | None, ...]:
    """Remask token positions selected by a cheap text-surface policy."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if context_window < 0:
        raise ValueError("context_window must be non-negative")
    if policy != "generic_filler":
        raise ValueError(f"Unsupported text repair policy: {policy}")

    usable_token_ids = source_token_ids[:max_new_tokens]
    decoded_text = _decode_tokens(usable_token_ids, token_decoder) or source_text
    spans = _generic_filler_char_spans(decoded_text)
    positions = _token_positions_for_char_spans(
        usable_token_ids,
        spans,
        token_decoder=token_decoder,
    )
    if positions:
        return build_token_position_repair_seed(
            source_token_ids,
            mask_positions=positions,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
        )
    if fallback_remask_low_confidence_fraction is not None:
        return build_low_confidence_repair_seed(
            source_token_ids,
            token_confidences=token_confidences,
            max_new_tokens=max_new_tokens,
            remask_fraction=fallback_remask_low_confidence_fraction,
        )
    return build_suffix_repair_seed(
        source_token_ids,
        max_new_tokens=max_new_tokens,
        keep_prefix_fraction=1.0,
    )


def build_token_position_repair_seed(
    source_token_ids: list[int],
    *,
    mask_positions: list[int],
    max_new_tokens: int,
    context_window: int = 0,
) -> tuple[int | None, ...]:
    """Keep a suffix but remask verifier-identified token positions."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if context_window < 0:
        raise ValueError("context_window must be non-negative")
    usable_token_ids = source_token_ids[:max_new_tokens]
    seed: list[int | None] = [*usable_token_ids, *(None for _ in range(max_new_tokens - len(usable_token_ids)))]
    positions = _expanded_mask_positions(
        mask_positions,
        token_count=len(usable_token_ids),
        context_window=context_window,
    )
    for position in positions:
        seed[position] = None
    return tuple(seed)


def build_answer_span_repair_seed(
    source_token_ids: list[int],
    *,
    answer_text: object | None,
    max_new_tokens: int,
    source_text: str = "",
    token_decoder: TokenDecoder | None = None,
    context_window: int = 0,
    fallback_tail_window: int = 2,
) -> tuple[int | None, ...]:
    """Remask the decoded answer span, falling back to a short final-token window."""
    answer = _normalize_answer_span_text(answer_text)
    return build_text_span_repair_seed(
        source_token_ids,
        target_texts=[answer] if answer else [],
        max_new_tokens=max_new_tokens,
        source_text=source_text,
        token_decoder=token_decoder,
        context_window=context_window,
        fallback_tail_window=fallback_tail_window,
    )


def build_text_span_repair_seed(
    source_token_ids: list[int],
    *,
    target_texts: list[object],
    max_new_tokens: int,
    history_instability_remask_fraction: float | None = None,
    history_samples_token_ids: list[list[int]] | None = None,
    source_text: str = "",
    token_decoder: TokenDecoder | None = None,
    context_window: int = 0,
    fallback_tail_window: int = 2,
) -> tuple[int | None, ...]:
    """Remask decoded spans named by a verifier, falling back to final content."""
    seed, _diagnostics = build_text_span_repair_seed_with_diagnostics(
        source_token_ids,
        target_texts=target_texts,
        max_new_tokens=max_new_tokens,
        history_instability_remask_fraction=history_instability_remask_fraction,
        history_samples_token_ids=history_samples_token_ids,
        source_text=source_text,
        token_decoder=token_decoder,
        context_window=context_window,
        fallback_tail_window=fallback_tail_window,
    )
    return seed


def build_text_span_repair_seed_with_diagnostics(
    source_token_ids: list[int],
    *,
    target_texts: list[object],
    max_new_tokens: int,
    history_instability_remask_fraction: float | None = None,
    history_samples_token_ids: list[list[int]] | None = None,
    source_text: str = "",
    token_decoder: TokenDecoder | None = None,
    context_window: int = 0,
    fallback_tail_window: int = 2,
) -> tuple[tuple[int | None, ...], dict[str, object]]:
    """Build a verifier-span seed and report whether the target was localized."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if context_window < 0:
        raise ValueError("context_window must be non-negative")
    if fallback_tail_window < 0:
        raise ValueError("fallback_tail_window must be non-negative")
    usable_token_ids = source_token_ids[:max_new_tokens]
    normalized_targets = [
        _normalize_answer_span_text(target)
        for target in target_texts
        if _normalize_answer_span_text(target)
    ]
    decoded_text = _decode_tokens(usable_token_ids, token_decoder) or source_text
    target_rows: list[dict[str, object]] = []
    spans = []
    for target in normalized_targets:
        target_spans = _literal_char_spans(decoded_text, target)
        spans.extend(target_spans)
        target_rows.append(
            {
                "target": target,
                "matched": bool(target_spans),
                "char_spans": _json_spans(target_spans),
            }
        )
    spans = _merge_char_spans(spans)
    positions = _token_positions_for_char_spans(
        usable_token_ids,
        spans,
        token_decoder=token_decoder,
    )
    history_instability_positions = _history_instability_mask_positions(
        usable_token_ids,
        history_instability_remask_fraction=history_instability_remask_fraction,
        history_samples_token_ids=history_samples_token_ids,
        mask_token_ids=DEFAULT_LLADA_MASK_TOKEN_IDS,
        max_new_tokens=max_new_tokens,
    )
    if positions:
        positions = sorted(set(positions).union(history_instability_positions))
        seed = build_token_position_repair_seed(
            source_token_ids,
            mask_positions=positions,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
        )
        return seed, _text_span_repair_seed_diagnostics(
            mode="literal_span",
            normalized_targets=normalized_targets,
            target_rows=target_rows,
            char_spans=spans,
            token_positions=positions,
            masked_positions=_expanded_mask_positions(
                positions,
                token_count=len(usable_token_ids),
                context_window=context_window,
            ),
            decoded_text=decoded_text,
            context_window=context_window,
            fallback_tail_window=fallback_tail_window,
            seed=seed,
            history_instability_positions=history_instability_positions,
            history_instability_remask_fraction=history_instability_remask_fraction,
        )
    if fallback_tail_window:
        seed = build_tail_window_repair_seed(
            source_token_ids,
            max_new_tokens=max_new_tokens,
            tail_window=fallback_tail_window,
        )
        return seed, _text_span_repair_seed_diagnostics(
            mode="tail_window_fallback",
            normalized_targets=normalized_targets,
            target_rows=target_rows,
            char_spans=spans,
            token_positions=[],
            masked_positions=_tail_window_mask_positions(
                usable_token_ids,
                tail_window=fallback_tail_window,
            ),
            decoded_text=decoded_text,
            context_window=context_window,
            fallback_tail_window=fallback_tail_window,
            seed=seed,
        )
    seed = build_suffix_repair_seed(
        source_token_ids,
        max_new_tokens=max_new_tokens,
        keep_prefix_fraction=1.0,
    )
    return seed, _text_span_repair_seed_diagnostics(
        mode="no_match_keep_source",
        normalized_targets=normalized_targets,
        target_rows=target_rows,
        char_spans=spans,
        token_positions=[],
        masked_positions=[],
        decoded_text=decoded_text,
        context_window=context_window,
        fallback_tail_window=fallback_tail_window,
        seed=seed,
    )


def build_text_span_repair_seed_diagnostics(
    source_token_ids: list[int],
    *,
    target_texts: list[object],
    max_new_tokens: int,
    history_instability_remask_fraction: float | None = None,
    history_samples_token_ids: list[list[int]] | None = None,
    source_text: str = "",
    token_decoder: TokenDecoder | None = None,
    context_window: int = 0,
    fallback_tail_window: int = 2,
) -> dict[str, object]:
    """Report localization diagnostics for a verifier-guided text-span seed."""
    _seed, diagnostics = build_text_span_repair_seed_with_diagnostics(
        source_token_ids,
        target_texts=target_texts,
        max_new_tokens=max_new_tokens,
        history_instability_remask_fraction=history_instability_remask_fraction,
        history_samples_token_ids=history_samples_token_ids,
        source_text=source_text,
        token_decoder=token_decoder,
        context_window=context_window,
        fallback_tail_window=fallback_tail_window,
    )
    return diagnostics


def _history_instability_mask_positions(
    source_token_ids: list[int],
    *,
    history_instability_remask_fraction: float | None,
    history_samples_token_ids: list[list[int]] | None,
    mask_token_ids: tuple[int, ...],
    max_new_tokens: int,
) -> list[int]:
    if history_instability_remask_fraction is None:
        return []
    if history_instability_remask_fraction <= 0.0 or history_instability_remask_fraction > 1.0:
        raise ValueError("history_instability_remask_fraction must be greater than 0 and at most 1")
    scored_positions = _history_instability_scores(
        source_token_ids,
        history_samples_token_ids=history_samples_token_ids,
        max_new_tokens=max_new_tokens,
        mask_token_ids=mask_token_ids,
    )
    if not scored_positions:
        return []
    usable_count = min(len(source_token_ids), max_new_tokens)
    remask_count = max(
        1,
        min(
            len(scored_positions),
            ceil(usable_count * history_instability_remask_fraction),
        ),
    )
    return sorted(
        index
        for index, _score in sorted(
            scored_positions,
            key=lambda item: item[1],
            reverse=True,
        )[:remask_count]
    )


def _normalize_answer_span_text(answer_text: object | None) -> str:
    if answer_text is None:
        return ""
    return " ".join(str(answer_text).strip().split())


def _literal_char_spans(text: str, needle: str) -> list[tuple[int, int]]:
    if not text or not needle:
        return []
    lowered_text = text.lower()
    lowered_needle = needle.lower()
    spans = [
        (match.start(), match.end())
        for match in re.finditer(re.escape(lowered_needle), lowered_text)
    ]
    return _merge_char_spans(spans)


def _text_span_repair_seed_diagnostics(
    *,
    mode: str,
    normalized_targets: list[str],
    target_rows: list[dict[str, object]],
    char_spans: list[tuple[int, int]],
    token_positions: list[int],
    masked_positions: list[int],
    decoded_text: str,
    context_window: int,
    fallback_tail_window: int,
    seed: tuple[int | None, ...],
    history_instability_positions: list[int] | None = None,
    history_instability_remask_fraction: float | None = None,
) -> dict[str, object]:
    return {
        "char_spans": _json_spans(char_spans),
        "context_window": context_window,
        "decoded_text_char_count": len(decoded_text),
        "fallback_tail_window": fallback_tail_window,
        "history_instability_positions": history_instability_positions or [],
        "history_instability_remask_fraction": history_instability_remask_fraction,
        "literal_target_found": bool(token_positions),
        "masked_positions": sorted(masked_positions),
        "matched_target_count": sum(1 for row in target_rows if row.get("matched")),
        "mode": mode,
        "seed_masked_positions": sum(1 for token_id in seed if token_id is None),
        "target_count": len(normalized_targets),
        "targets": target_rows,
        "token_positions": sorted(token_positions),
        "used_fallback": mode != "literal_span",
    }


def _json_spans(spans: list[tuple[int, int]]) -> list[list[int]]:
    return [[start, end] for start, end in spans]


def _expanded_mask_positions(
    mask_positions: list[int],
    *,
    token_count: int,
    context_window: int,
) -> list[int]:
    positions = set()
    for position in mask_positions:
        if position < 0 or position >= token_count:
            continue
        start = max(0, position - context_window)
        end = min(token_count, position + context_window + 1)
        positions.update(range(start, end))
    return sorted(positions)


def _generic_filler_char_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    lower = text.lower()
    target_phrases = (
        "complex nuanced tasks",
        "deep understanding",
        "actual reasoning ability",
        "quality of the qualitative outputs",
        "presentister",
        "checkpoint checkpointing",
        "the the",
    )
    for phrase in target_phrases:
        spans.extend((match.start(), match.end()) for match in re.finditer(re.escape(phrase), lower))

    spans.extend(
        (match.start(), match.end())
        for match in re.finditer(r"\b([a-z0-9]{2,})\s+\1\b", lower)
    )
    word_matches = list(re.finditer(r"\b[a-z0-9]{5,}\b", lower))
    for left, right in zip(word_matches, word_matches[1:], strict=False):
        left_word = left.group(0)
        right_word = right.group(0)
        if left_word == right_word:
            continue
        if left_word.startswith(right_word) or right_word.startswith(left_word):
            spans.append((left.start(), right.end()))
    return _merge_char_spans(spans)


def _token_positions_for_char_spans(
    token_ids: list[int],
    char_spans: list[tuple[int, int]],
    *,
    token_decoder: TokenDecoder | None,
) -> list[int]:
    if not token_ids or not char_spans or token_decoder is None:
        return []
    token_spans = _token_char_spans(token_ids, token_decoder)
    positions = []
    for index, (token_start, token_end) in enumerate(token_spans):
        if token_end <= token_start:
            continue
        if any(token_start < span_end and token_end > span_start for span_start, span_end in char_spans):
            positions.append(index)
    return positions


def _token_char_spans(
    token_ids: list[int],
    token_decoder: TokenDecoder,
) -> list[tuple[int, int]]:
    spans = []
    previous_text = ""
    for index in range(len(token_ids)):
        current_text = _decode_tokens(token_ids[: index + 1], token_decoder)
        spans.append((len(previous_text), len(current_text)))
        previous_text = current_text
    return spans


def _decode_tokens(token_ids: list[int], token_decoder: TokenDecoder | None) -> str:
    if token_decoder is None:
        return ""
    return token_decoder(token_ids)


def _merge_char_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def build_history_state_repair_seed(
    history_token_ids: list[int],
    *,
    max_new_tokens: int,
    mask_token_ids: tuple[int, ...] = DEFAULT_LLADA_MASK_TOKEN_IDS,
    keep_prefix_fraction: float | None = None,
) -> tuple[int | None, ...]:
    """Keep visible mid-trajectory tokens and remask unresolved/special slots."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if keep_prefix_fraction is not None and (keep_prefix_fraction < 0.0 or keep_prefix_fraction > 1.0):
        raise ValueError("keep_prefix_fraction must be between 0 and 1")
    mask_ids = set(mask_token_ids)
    seed: list[int | None] = []
    for token_id in history_token_ids[:max_new_tokens]:
        if token_id in mask_ids or _looks_like_special_token(token_id):
            seed.append(None)
        else:
            seed.append(token_id)
    seed.extend(None for _ in range(max_new_tokens - len(seed)))
    if keep_prefix_fraction is not None:
        keep_count = _prefix_keep_count(
            token_count=max_new_tokens,
            keep_prefix_fraction=keep_prefix_fraction,
        )
        for index in range(keep_count, len(seed)):
            seed[index] = None
    return tuple(seed)


def _adaptive_history_prefix_fraction(
    *,
    source_quality_score: float | None,
    history_selection_score: float | None,
    history_mask_count: int | None,
    max_new_tokens: int,
    default_fraction: float,
    weak_fraction: float,
    source_quality_threshold: float,
    history_score_threshold: float,
    max_mask_fraction: float,
) -> float:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if default_fraction < 0.0 or default_fraction > 1.0 or weak_fraction < 0.0 or weak_fraction > 1.0:
        raise ValueError("history prefix fractions must be between 0 and 1")
    if max_mask_fraction < 0.0 or max_mask_fraction > 1.0:
        raise ValueError("max_mask_fraction must be between 0 and 1")
    source_quality = 1.0 if source_quality_score is None else float(source_quality_score)
    history_score = 1.0 if history_selection_score is None else float(history_selection_score)
    mask_count = max_new_tokens if history_mask_count is None else max(0, int(history_mask_count))
    mask_fraction = mask_count / max_new_tokens
    if (
        source_quality < source_quality_threshold
        and history_score < history_score_threshold
        and mask_fraction <= max_mask_fraction
    ):
        return weak_fraction
    return default_fraction


def build_tail_window_repair_seed(
    source_token_ids: list[int],
    *,
    max_new_tokens: int,
    tail_window: int,
) -> tuple[int | None, ...]:
    """Fallback exact-answer repair seed when no answer span can be found."""
    if tail_window <= 0:
        raise ValueError("tail_window must be positive")
    usable_token_ids = source_token_ids[:max_new_tokens]
    mask_positions = _tail_window_mask_positions(usable_token_ids, tail_window=tail_window)
    return build_token_position_repair_seed(
        source_token_ids,
        mask_positions=mask_positions,
        max_new_tokens=max_new_tokens,
    )


def _tail_window_mask_positions(token_ids: list[int], *, tail_window: int) -> list[int]:
    content_positions = [
        index for index, token_id in enumerate(token_ids) if not _looks_like_special_token(token_id)
    ]
    return content_positions[-tail_window:]


def _prefix_keep_count(*, token_count: int, keep_prefix_fraction: float) -> int:
    if token_count == 0 or keep_prefix_fraction == 0.0:
        return 0
    return max(1, min(token_count, ceil(token_count * keep_prefix_fraction)))


def _looks_like_special_token(token_id: int) -> bool:
    return token_id >= 126000


def _fraction_label(value: float) -> str:
    return f"{round(value * 100):02d}"
