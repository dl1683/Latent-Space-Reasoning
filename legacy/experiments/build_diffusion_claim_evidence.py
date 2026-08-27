"""Build a repo-level evidence map for current diffusion reasoning claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_OUTPUT = Path("CLAIM_EVIDENCE_MAP.md")
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/diffusion_claim_evidence_map.json")
DEFAULT_INDEX_OUTPUT = Path("eval_results/diffusion_language/diffusion_ground_truth_index.json")
DEFAULT_INDEX_MARKDOWN_OUTPUT = Path("DIFFUSION_GROUND_TRUTH_INDEX.md")
DEFAULT_PUBLIC_BENCHMARK_OUTPUT = Path("DIFFUSION_PUBLIC_BENCHMARK.md")
DEFAULT_PUBLIC_BENCHMARK_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/diffusion_public_benchmark.json"
)
DEFAULT_PUBLIC_BENCHMARK_CLAIM_ID = "moe_mixed_anchor_instability_claim_auto_compat_preserve_seeded_gated_budget"
DEFAULT_PUBLIC_TOP_SCORE_CLAIM_ID = "moe_mixed_anchor_instability_claim_auto_compat_preserve_seeded_gated_budget"
DEFAULT_PUBLIC_BUDGET_CLAIM_ID = "moe_mixed_decomposed_four_head_selector_budget"
PUBLIC_GUARD_FAMILIES = ("math", "symbolic", "science")


class EvidenceArtifactMissingError(FileNotFoundError):
    """Raised when a promoted claim lacks a required evidence artifact."""


@dataclass(frozen=True)
class RepairDiagnosticRequirement:
    repair_name: str
    metric: str
    min_value: float | None = None
    max_value: float | None = None
    note: str = ""


@dataclass(frozen=True)
class ClaimSpec:
    claim_id: str
    title: str
    scores_path: Path
    status: str
    note: str
    raw_path: Path | None = None
    required_repair_diagnostics: tuple[RepairDiagnosticRequirement, ...] = ()


@dataclass(frozen=True)
class ClaimEvidence:
    claim_id: str
    title: str
    status: str
    note: str
    scores_path: str
    report_path: str
    raw_path: str
    all_generation_count: int
    repair_count: int
    full_count: int
    repair_eligible_count: int
    fixed_generation_budget: float
    random_generation_budget: float
    repair_generation_budget: float
    repair_relative_gpu_cost: float
    fixed_repair_slice_score: float
    random_repair_slice_score: float
    repair_score: float
    repair_delta_vs_fixed: float
    repair_delta_vs_random: float
    repair_delta_vs_evolved: float | None
    repair_budget_delta_vs_evolved: float | None
    repair_gain_per_extra_generation: float | None
    repair_wins_vs_fixed: str
    repair_wins_vs_random: str
    repair_wins_vs_evolved: str
    oracle_headroom_vs_repair: float | None
    repair_pack: str
    repair_source_policy: str
    adaptive_source_gate_mode: str
    exact_task_trajectory_policy: str
    result_run_id: str
    result_content_hash: str
    required_repair_diagnostics: tuple[RepairDiagnosticRequirement, ...]


@dataclass(frozen=True)
class CanonicalClaimSlot:
    slot_id: str
    label: str
    claim_id: str
    selection_reason: str


DEFAULT_CLAIMS = (
    ClaimSpec(
        claim_id="dense_llada_lean_mixed_guarded_repair",
        title="Dense LLaDA compact mixed guarded repair remains the strongest lean repair line",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_mixed_gated_ranked_span_guarded_exact_identity_v1_scores.json"
        ),
        status="supported scout result",
        note=(
            "Dense LLaDA gives the strongest compact repair-covered score today; exact checks "
            "include proposal attribution and guarded symbolic repair. The current canonical "
            "artifact includes deterministic run identity and artifact hashes."
        ),
    ),
    ClaimSpec(
        claim_id="dense_llada_hard_exact_no_proposal_span_repair",
        title="Dense LLaDA hard exact repair solves no-proposal failures with verifier-localized span inpainting",
        scores_path=Path(
            "eval_results/diffusion_language/llada_hard_exact_verifier_span_early_stop_v1_scores.json"
        ),
        status="supported hard-exact scout result",
        note=(
            "No deterministic answer proposals are available on this exact slice; the win comes "
            "from label-free self-repair plus verifier-localized arithmetic span inpainting, "
            "with feedback skipped after span repair passes exact-answer guards."
        ),
    ),
    ClaimSpec(
        claim_id="dense_llada_planning_adaptive_span_repair",
        title="Dense LLaDA adaptive planning span repair improves all eight short planning tasks",
        scores_path=Path(
            "eval_results/diffusion_language/llada_planning_constraint_gap_span_adaptive_8task_v1_scores.json"
        ),
        status="supported planning scout result",
        note=(
            "Default planning repair now uses verifier-ranked adaptive spans: sentence spans when "
            "specific, clause spans when sentence targeting collapses to whole-draft masking. "
            "The 8-task CUDA scout improves fixed/random/evolved with 6 wins and 2 ties."
        ),
    ),
    ClaimSpec(
        claim_id="dense_llada_mixed_adaptive_span_budget",
        title="Dense LLaDA adaptive constraint-span repair is the budget-favored mixed policy",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_mixed_adaptive_constraint_span_identity_v1_scores.json"
        ),
        status="supported budget scout result",
        note=(
            "This lean mixed run trades some absolute score for lower repair spend: "
            "the adaptive span-only planning path plus exact verifier repair uses fewer "
            "full generations than the strongest guarded mixed run and has higher gain "
            "per extra generation versus evolved. The current canonical artifact includes "
            "deterministic run identity and artifact hashes."
        ),
    ),
    ClaimSpec(
        claim_id="moe_local_mixed_transfer_baseline",
        title="LLaDA-MoE is locally runnable but dense-LLaDA repair policy transfer was weak",
        scores_path=Path(
            "eval_results/diffusion_language/llada_moe_mixed_gated_ranked_span_guarded_exact_v1_scores.json"
        ),
        status="baseline and negative-transfer evidence",
        note=(
            "The sparse MoE target runs cheaply enough for iteration, but the inherited "
            "state-adaptive dense-LLaDA repair pack produced only a small planning gain."
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_adaptive_score_max",
        title="Compact MoE revision plus constraint-span repair improves lean mixed planning",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_score_max_v1_scores.json"
        ),
        status="supported scout result",
        note=(
            "Current MoE lean mixed score-max line: revision schedules, adaptive second-source "
            "spending, and compact verifier-localized span targeting. The compact policy keeps "
            "decision-rule context and near-tie failure chains while improving the prior "
            "source-ranked mixed score at the same generation count."
        ),
    ),
    ClaimSpec(
        claim_id="moe_planning_span_localized_repair",
        title="MoE compact planning span repair localizes verifier targets before inpainting",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_planning_compact_span_score_max_v2_scores.json"
        ),
        status="supported planning localization scout result",
        note=(
            "Fresh planning-only CUDA confirmation of the MoE compact constraint-span "
            "policy. Compact target selection keeps decision-rule context and near-tie "
            "failure chains while reducing average masked span size versus the older "
            "source-ranked line. The repair candidate summary reports literal span "
            "localization on every constraint-gap repair and zero tail-window fallback."
        ),
        required_repair_diagnostics=(
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_repair",
                metric="mean_span_literal_target_found",
                min_value=1.0,
                note="verifier target must be found as decoded text, not inferred post hoc",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_repair",
                metric="mean_span_fallback_used",
                max_value=0.0,
                note="tail-window fallback must not support the promoted localization claim",
            ),
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_adaptive_efficiency",
        title="Score-efficient compact MoE gate preserves the top score with one fewer generation",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_score_efficient_fresh_v1_scores.json"
        ),
        status="fresh GPU score-efficient scout result",
        note=(
            "Fresh CUDA confirmation of the score_efficient gate. It adds a quality ceiling, "
            "skips the high-quality no-op plan_002 second source, keeps the useful plan_006 "
            "source, and preserves the top repair score with one fewer generation."
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_single_source_budget",
        title="Compact single-source MoE repair is dominated by direct fixed-source repair",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_single_source_fresh_v1_scores.json"
        ),
        status="fresh GPU dominated budget scout result",
        note=(
            "Fresh CUDA confirmation of the cheapest revision-enabled compact MoE policy. "
            "It branches span repair only from the best non-revision evolved source, uses "
            "one fewer generation than score_efficient, and preserves the expected tradeoff. "
            "The newer direct fixed-source run dominates it on both score and cost, so this "
            "claim is retained as historical frontier evidence rather than the public budget point."
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_fixed_source_budget",
        title="Direct fixed-source MoE repair is dominated by the quality-gated budget policy",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_fixed_source_fresh_v1_scores.json"
        ),
        status="fresh GPU dominated direct-repair scout result",
        note=(
            "Fresh CUDA confirmation that the compact span repair can branch directly from "
            "the greedy fixed denoise output, without evolved schedules. This cuts the "
            "repair-covered relative cost from 7.125x to 3.000x while preserving nearly "
            "the same planning repair score. The newer quality-gated fixed-source run "
            "preserves the exact same score and wins at 2.875x, so this claim is retained "
            "as historical evidence rather than the public budget point."
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_fixed_source_quality_gate_budget",
        title="Quality-gated fixed-source MoE repair is dominated by repairability geometry gating",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_fixed_source_quality_gate_fresh_v1_scores.json"
        ),
        status="fresh GPU dominated quality-gated direct-repair scout result",
        note=(
            "Fresh CUDA confirmation that source-quality gating can skip the high-quality "
            "plan_002 repair pass while keeping the direct fixed-source score unchanged. "
            "It preserves repair-selected 0.489911, the same win/tie/loss profile, and "
            "zero oracle headroom while cutting relative repair cost from 3.000x to 2.875x. "
            "The newer repairability-geometry gate preserves the same score and wins at "
            "2.625x, so this claim is retained as historical evidence."
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_fixed_source_repairability_gate_budget",
        title="Repairability-geometry gated MoE repair keeps the budget score at lower cost",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_fixed_source_repairability_gate_fresh_v1_scores.json"
        ),
        status="fresh GPU repairability-geometry budget scout result",
        note=(
            "Fresh CUDA confirmation that a source-quality plus prompt-gap/coverage geometry "
            "gate can skip the high-quality no-op plan_002 repair and the under-grounded "
            "plan_005/plan_008 span repairs while keeping the direct fixed-source score "
            "unchanged. It preserves repair-selected 0.489911, the same win/tie/loss "
            "profile, and zero oracle headroom while cutting relative repair cost from "
            "2.875x to 2.625x."
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_fixed_source_denoise_phase_gate_budget",
        title="Denoise-phase gated MoE repair preserves the budget frontier with a trajectory skeleton trigger",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_fixed_source_denoise_phase_gate_fresh_v1_scores.json"
        ),
        status="fresh GPU denoise-phase geometry budget scout result",
        note=(
            "Fresh CUDA confirmation that the repairability gate can be executed from sampled "
            "denoise-history structure, not only final-text geometry. The trigger requires the "
            "same source-quality plus prompt-gap/coverage repairable band and a visible denoise "
            "constraint skeleton before spending repair compute. It preserves repair-selected "
            "0.489911, the same win/tie/loss profile, and zero oracle headroom at 2.625x "
            "relative cost while giving the public budget point a trajectory-level mechanism."
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_anchor_select_denoise_phase_gate_budget",
        title="Pre-generation anchor-selected MoE repair uses denoise history while preserving the budget frontier",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_anchor_select_denoise_phase_gate_dense_history_fresh_v1_scores.json"
        ),
        status="fresh GPU pre-generation anchor-select budget scout result",
        note=(
            "Fresh CUDA confirmation that the runner can choose between final and sampled "
            "history anchors before spending repair compute. Dense history sampling exposes "
            "near-final denoise states without adding model generations; the selector chooses "
            "a history anchor on plan_001 and final anchors elsewhere, preserving repair-selected "
            "0.489911, zero oracle headroom, and 2.625x relative cost."
        ),
        required_repair_diagnostics=(
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_select_repair",
                metric="mean_span_literal_target_found",
                min_value=1.0,
                note="anchor-selected span repair must localize decoded verifier targets",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_select_repair",
                metric="mean_span_fallback_used",
                max_value=0.0,
                note="tail-window fallback must not support the promoted anchor-select claim",
            ),
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_anchor_instability_prompt_gated_budget",
        title="Prompt-gated anchor-instability MoE repair improves the budget frontier",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_anchor_instability_prompt_gated_denoise_phase_gate_dense_history_fresh_v1_scores.json"
        ),
        status="fresh GPU prompt-gated denoise-geometry budget scout result",
        note=(
            "Fresh CUDA confirmation that sampled denoise instability can help when it gates "
            "both the remask positions and the instability-specific repair instruction. The "
            "runner preserves exact anchor-select identity on gate-off tasks, then activates "
            "the instability prompt only on the low-quality multi-span plan_007 branch. That "
            "lifts repair-selected planning score to 0.498304 at the same 2.625x relative "
            "cost, with zero oracle headroom."
        ),
        required_repair_diagnostics=(
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_prompt_gated_repair",
                metric="mean_span_literal_target_found",
                min_value=1.0,
                note="prompt-gated instability repair must still localize decoded verifier targets",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_prompt_gated_repair",
                metric="mean_span_fallback_used",
                max_value=0.0,
                note="tail-window fallback must not support the prompt-gated claim",
            ),
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_anchor_instability_claim_gated_budget",
        title="Claim-gated anchor-instability MoE repair improves the budget frontier",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_anchor_instability_claim_gated_compact_prompt_denoise_phase_gate_dense_history_fresh_v1_scores.json"
        ),
        status="fresh GPU claim-gated denoise-geometry budget scout result",
        note=(
            "Fresh CUDA confirmation of the composite denoise-geometry repair policy. It "
            "preserves prompt-gated anchor-instability behavior on gate-off tasks, keeps "
            "the active plan_007 denoise-instability branch, and adds a public-claim "
            "confound-control prompt gate for plan_004 without increasing relative cost. "
            "The resulting repair-selected planning score is 0.513438 at 2.625x relative "
            "cost, with zero oracle headroom."
        ),
        required_repair_diagnostics=(
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_gated_repair",
                metric="mean_span_literal_target_found",
                min_value=1.0,
                note="claim-gated instability repair must still localize decoded verifier targets",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_gated_repair",
                metric="mean_span_fallback_used",
                max_value=0.0,
                note="tail-window fallback must not support the claim-gated frontier",
            ),
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_anchor_instability_claim_oracle_gated_budget",
        title="Oracle-aware claim-gated MoE repair improves the budget frontier",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_anchor_instability_claim_oracle_gated_denoise_phase_gate_dense_history_fresh_v1_scores.json"
        ),
        status="fresh GPU oracle-aware claim-gated denoise-geometry budget scout result",
        note=(
            "Fresh CUDA confirmation of the compact oracle-aware public-claim gate. It "
            "keeps the same denoise-anchor and instability-mask geometry, preserves the "
            "active plan_007 branch, and improves the plan_004 public-claim repair without "
            "increasing relative cost. The resulting repair-selected planning score is "
            "0.523304 at 2.625x relative cost, with zero oracle headroom."
        ),
        required_repair_diagnostics=(
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_oracle_gated_repair",
                metric="mean_span_literal_target_found",
                min_value=1.0,
                note="oracle-aware claim gate must still localize decoded verifier targets",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_oracle_gated_repair",
                metric="mean_span_fallback_used",
                max_value=0.0,
                note="tail-window fallback must not support the oracle-aware frontier",
            ),
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_anchor_instability_claim_compatible_seeded_gated_budget",
        title="Realization-guarded compatible seeded claim-gated MoE repair improves the budget frontier",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_anchor_instability_claim_compatible_seeded_gated_realization_guard_fresh_v1_scores.json"
        ),
        status="fresh GPU realization-guarded compatible-seeded denoise-geometry budget scout result",
        note=(
            "Fresh CUDA confirmation that a compact semantic seed can carry both the "
            "oracle/selected-results split and the claim-survival control without adding "
            "repair candidates. It keeps the same denoise-anchor and instability-mask "
            "geometry, uses the realization-quality selector to reject low-specificity seed "
            "meta text, raises plan_004 to full rubric coverage, and preserves the "
            "repair-selected planning score at 0.531116 with 2.625x relative cost and "
            "zero oracle headroom."
        ),
        required_repair_diagnostics=(
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_compatible_seeded_gated_repair",
                metric="mean_span_literal_target_found",
                min_value=1.0,
                note="compatible seeded claim gate must still localize decoded verifier targets",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_compatible_seeded_gated_repair",
                metric="mean_span_fallback_used",
                max_value=0.0,
                note="tail-window fallback must not support the compatible-seeded frontier",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_compatible_seeded_gated_repair",
                metric="mean_seed_realization_meta_penalty",
                max_value=0.01,
                note="realization guard must keep compact seed repairs low-meta on average",
            ),
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_anchor_instability_claim_auto_compat_seeded_gated_budget",
        title="Automatic compatibility-scored claim seed recovers the MoE budget frontier",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_seeded_gated_realization_guard_v1_scores.json"
        ),
        status="fresh GPU automatic compatibility-scored denoise-geometry budget scout result",
        note=(
            "Fresh CUDA confirmation that compact semantic seed selection can be automated "
            "with a label-free compatibility score over required controls. The policy scores "
            "candidate 9-token anchors, selects the oracle/selected-results plus "
            "claim-survival anchor for the public-claim repair, matches the prior "
            "hand-compatible repair-selected planning score at 0.531116 with 2.625x "
            "relative cost, and keeps zero repair-oracle headroom."
        ),
        required_repair_diagnostics=(
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_auto_compat_seeded_gated_repair",
                metric="mean_span_literal_target_found",
                min_value=1.0,
                note="auto-compatible seeded claim gate must still localize decoded verifier targets",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_auto_compat_seeded_gated_repair",
                metric="mean_span_fallback_used",
                max_value=0.0,
                note="tail-window fallback must not support the auto-compatible seeded frontier",
            ),
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_anchor_instability_claim_auto_compat_preserve_seeded_gated_budget",
        title="Automatic preservation-seeded claim repair recovers the MoE budget frontier without seed-meta wording",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_"
            "preservation_seed_fresh_v1_scores.json"
        ),
        status="fresh GPU automatic preservation-seeded denoise-geometry budget scout result",
        note=(
            "Fresh CUDA confirmation that the automatic compatibility seed can be refined into "
            "a public-claim preservation seed. The policy selects a compact 9-token denoise "
            "tail for plan_004, recovers the same 0.531116 repair-selected planning score at "
            "2.625x relative cost, keeps zero repair-oracle headroom, and removes explicit "
            "seed/anchor meta wording from the frontier task."
        ),
        required_repair_diagnostics=(
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_repair",
                metric="mean_span_literal_target_found",
                min_value=1.0,
                note="auto-preserve seeded claim gate must still localize decoded verifier targets",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_repair",
                metric="mean_span_fallback_used",
                max_value=0.0,
                note="tail-window fallback must not support the auto-preserve seeded frontier",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_repair",
                metric="mean_seed_realization_meta_penalty",
                max_value=0.0,
                note="preservation seed must avoid explicit seed/anchor meta text on the promoted run",
            ),
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_phase_hybrid_preserve_seeded_equivalent_frontier",
        title="Strict phase-hybrid MoE repair matches the public budget frontier with explicit phase evidence",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_phase_hybrid_preserve_seeded_gated_fresh_v2_scores.json"
        ),
        status="fresh GPU phase-conditioned hybrid mechanism confirmation",
        note=(
            "Fresh CUDA confirmation that denoise-history source replacement must be "
            "conditional. The strict phase-hybrid policy keeps the promoted "
            "preservation-seeded controls, records repairable and retention-safe "
            "denoise phase timing, switches to history only under strict "
            "retention/source-advantage checks, and matches the public frontier at "
            "0.531116 repair-selected planning score, 2.625x relative cost, and zero "
            "repair-oracle headroom. This is mechanism-equivalent frontier evidence, "
            "not a replacement for the simpler public headline claim."
        ),
        required_repair_diagnostics=(
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair",
                metric="mean_span_literal_target_found",
                min_value=1.0,
                note="strict phase-hybrid repair must still localize decoded verifier targets",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair",
                metric="mean_span_fallback_used",
                max_value=0.0,
                note="tail-window fallback must not support the phase-hybrid mechanism claim",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair",
                metric="mean_seed_realization_meta_penalty",
                max_value=0.0,
                note="strict phase-hybrid frontier must avoid seed/anchor meta text",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair",
                metric="mean_task_delta_vs_source",
                min_value=0.18,
                note="phase-hybrid repairs must be real improvements over their selected sources",
            ),
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_phase_final_preserve_seeded_value_proxy_budget",
        title="Cost-aware value-proxy MoE repair beats the cheap tier at the same GPU cost",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_phase_final_preserve_seeded_gated_value_proxy_fresh_v1_scores.json"
        ),
        status="fresh GPU value-proxy budget frontier result",
        note=(
            "Fresh CUDA confirmation that the cost-aware marginal-value proxy can be run "
            "as a primary repair spend trigger. The rule keeps the phase/final "
            "preservation-seeded operator, requires a repairable denoise skeleton, and "
            "spends only when source quality is low enough to justify the extra GPU "
            "generation. It repairs plan_004, plan_006, and plan_007, reaches 0.508705 "
            "at 2.375x relative cost, and dominates the cheap and mid phase-budget tiers "
            "on score/cost."
        ),
        required_repair_diagnostics=(
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                metric="mean_span_literal_target_found",
                min_value=1.0,
                note="value-proxy repair must still localize decoded verifier targets",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                metric="mean_span_fallback_used",
                max_value=0.0,
                note="tail-window fallback must not support the value-proxy budget claim",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                metric="mean_task_delta_vs_source",
                min_value=0.17,
                note="value-proxy repairs must be real improvements over their selected sources",
            ),
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_decomposed_four_head_selector_budget",
        title="Decomposed four-head MoE selector reproduces the lower-cost budget point with runner provenance",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_decomposed_four_head_selector_frontier_v1_scores.json"
        ),
        status="fresh GPU decomposed-selector budget confirmation",
        note=(
            "Fresh CUDA confirmation that the fitted four-head selector is executable in "
            "the benchmark runner. The trigger records spend/source/retention/realization "
            "head provenance on every repair-spend gate row, repairs plan_004, plan_006, "
            "and plan_007, and reproduces the value-proxy budget point: 0.508705 repair "
            "score at 2.375x relative cost with zero repair-oracle headroom."
        ),
        required_repair_diagnostics=(
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                metric="mean_span_literal_target_found",
                min_value=1.0,
                note="decomposed-selector repair must still localize decoded verifier targets",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                metric="mean_span_fallback_used",
                max_value=0.0,
                note="tail-window fallback must not support the decomposed-selector budget claim",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                metric="mean_task_delta_vs_source",
                min_value=0.17,
                note="decomposed-selector repairs must be real improvements over their selected sources",
            ),
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_anchor_instability_claim_auto_seeded_gated_boundary",
        title="Automatic compact-control seeding works mechanically but trails the fixed compatible seed",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_anchor_instability_claim_auto_seeded_gated_denoise_phase_gate_dense_history_fresh_v1_scores.json"
        ),
        status="fresh GPU automatic-seed boundary scout result",
        note=(
            "Fresh CUDA boundary result for synthesizing a compact semantic seed from the "
            "active task/rubric surface. The policy generates and applies the expected "
            "oracle/selected plus claim-survival anchor, and plan_004 hits all five rubric "
            "controls, but the aggregate repair-selected planning score is 0.520536 at "
            "2.625x relative cost. This trails the fixed compatible seed, so the next policy "
            "needs a realization-quality term in addition to control-term extraction."
        ),
        required_repair_diagnostics=(
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_auto_seeded_gated_repair",
                metric="mean_span_literal_target_found",
                min_value=1.0,
                note="auto-seeded claim gate must still localize decoded verifier targets",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_auto_seeded_gated_repair",
                metric="mean_span_fallback_used",
                max_value=0.0,
                note="tail-window fallback must not support the auto-seeded boundary result",
            ),
        ),
    ),
    ClaimSpec(
        claim_id="moe_mixed_anchor_instability_claim_auto_seeded_realization_gated_boundary",
        title="Explicit realization constraints over-compress automatic compact-control seeding",
        scores_path=Path(
            "eval_results/diffusion_language/"
            "llada_moe_mixed_compact_span_anchor_instability_claim_auto_seeded_realization_gated_denoise_phase_gate_dense_history_fresh_v1_scores.json"
        ),
        status="fresh GPU realization-constraint boundary scout result",
        note=(
            "Fresh CUDA boundary result for adding explicit realization constraints to "
            "automatic compact-control seeding. The generated seed still applies and plan_004 "
            "still hits all rubric controls, but the answer collapses toward a labeled control "
            "sentence and the aggregate repair-selected planning score falls to 0.515759 at "
            "2.625x relative cost. This rules out stronger prompt constraints as the next path; "
            "the policy needs a scored realization-quality loss."
        ),
        required_repair_diagnostics=(
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_auto_seeded_realization_gated_repair",
                metric="mean_span_literal_target_found",
                min_value=1.0,
                note="realization-gated auto seed must still localize decoded verifier targets",
            ),
            RepairDiagnosticRequirement(
                repair_name="constraint_gap_span_anchor_instability_claim_auto_seeded_realization_gated_repair",
                metric="mean_span_fallback_used",
                max_value=0.0,
                note="tail-window fallback must not support the realization boundary result",
            ),
        ),
    ),
)

DEFAULT_CANONICAL_SLOTS = (
    CanonicalClaimSlot(
        slot_id="dense_compact_top_score",
        label="Dense compact top-score policy",
        claim_id="dense_llada_lean_mixed_guarded_repair",
        selection_reason="Highest current compact mixed dense-LLaDA repair score.",
    ),
    CanonicalClaimSlot(
        slot_id="dense_compact_budget",
        label="Dense compact budget policy",
        claim_id="dense_llada_mixed_adaptive_span_budget",
        selection_reason="Best dense-LLaDA mixed gain per extra generation in the lean stack.",
    ),
    CanonicalClaimSlot(
        slot_id="dense_planning_span_repair",
        label="Dense planning span-repair policy",
        claim_id="dense_llada_planning_adaptive_span_repair",
        selection_reason="Current canonical short-planning adaptive span repair result.",
    ),
    CanonicalClaimSlot(
        slot_id="dense_hard_exact_no_proposal",
        label="Dense hard-exact no-proposal policy",
        claim_id="dense_llada_hard_exact_no_proposal_span_repair",
        selection_reason="Verifier-localized span inpainting result without deterministic proposals.",
    ),
    CanonicalClaimSlot(
        slot_id="moe_transfer_baseline",
        label="MoE dense-policy transfer baseline",
        claim_id="moe_local_mixed_transfer_baseline",
        selection_reason="Baseline showing local MoE viability and weak direct dense-policy transfer.",
    ),
    CanonicalClaimSlot(
        slot_id="moe_compact_score_max",
        label="MoE compact score-max policy",
        claim_id="moe_mixed_adaptive_score_max",
        selection_reason="Historical score-max compact revision policy before prompt-gated fixed-source repair.",
    ),
    CanonicalClaimSlot(
        slot_id="moe_planning_span_localization",
        label="MoE planning span-localized policy",
        claim_id="moe_planning_span_localized_repair",
        selection_reason=(
            "Fresh planning-only confirmation that compact span repair improves the MoE "
            "planning score line while still using literal target localization rather "
            "than tail-window fallback."
        ),
    ),
    CanonicalClaimSlot(
        slot_id="moe_compact_efficiency",
        label="MoE compact score-efficient policy",
        claim_id="moe_mixed_adaptive_efficiency",
        selection_reason=(
            "Pareto improvement over score_max: same repair score with one fewer generation."
        ),
    ),
    CanonicalClaimSlot(
        slot_id="moe_claim_auto_compat_preserve_seeded_instability_budget",
        label="MoE auto-compatible preservation-seeded claim-gated instability budget policy",
        claim_id="moe_mixed_anchor_instability_claim_auto_compat_preserve_seeded_gated_budget",
        selection_reason=(
            "Current public MoE budget frontier: preserves the denoise-anchor/instability "
            "geometry and automatically scores compact seed anchors so oracle/selected "
            "result separation remains compatible with public-claim preservation "
            "without seed/anchor meta text or extra cost."
        ),
    ),
    CanonicalClaimSlot(
        slot_id="moe_phase_final_value_proxy_budget",
        label="MoE phase-final value-proxy budget policy",
        claim_id="moe_mixed_phase_final_preserve_seeded_value_proxy_budget",
        selection_reason=(
            "Current lower-cost MoE budget frontier: same 2.375x relative GPU cost as "
            "the cheap phase-budget tier, but higher score by using a source-quality "
            "marginal-value proxy to skip low-value early repairs and keep high-value "
            "late repairs."
        ),
    ),
    CanonicalClaimSlot(
        slot_id="moe_decomposed_four_head_selector_budget",
        label="MoE decomposed four-head selector budget policy",
        claim_id="moe_mixed_decomposed_four_head_selector_budget",
        selection_reason=(
            "Executable bridge from the theory/audit layer to the runner: same lower-cost "
            "2.375x budget point as the value proxy, now with fitted spend/source/"
            "retention/realization head provenance in repair-spend diagnostics."
        ),
    ),
    CanonicalClaimSlot(
        slot_id="moe_phase_hybrid_preserve_seeded_mechanism_equivalent",
        label="MoE strict phase-hybrid mechanism-equivalent policy",
        claim_id="moe_mixed_phase_hybrid_preserve_seeded_equivalent_frontier",
        selection_reason=(
            "Mechanism-equivalent confirmation of the public frontier: same score/cost "
            "point as the promoted preservation-seeded claim, but with explicit "
            "denoise phase repairability, retention-safety timing, and strict "
            "history-source gating."
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--index-output", default=str(DEFAULT_INDEX_OUTPUT))
    parser.add_argument("--index-markdown-output", default=str(DEFAULT_INDEX_MARKDOWN_OUTPUT))
    parser.add_argument("--public-output", default=str(DEFAULT_PUBLIC_BENCHMARK_OUTPUT))
    parser.add_argument(
        "--public-json-output",
        default=str(DEFAULT_PUBLIC_BENCHMARK_JSON_OUTPUT),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    claims = [build_claim_evidence(spec) for spec in DEFAULT_CLAIMS]
    output = Path(args.output)
    json_output = Path(args.json_output)
    index_output = Path(args.index_output)
    index_markdown_output = Path(args.index_markdown_output)
    public_output = Path(args.public_output)
    public_json_output = Path(args.public_json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    index_output.parent.mkdir(parents=True, exist_ok=True)
    index_markdown_output.parent.mkdir(parents=True, exist_ok=True)
    public_output.parent.mkdir(parents=True, exist_ok=True)
    public_json_output.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_index = build_ground_truth_index(claims)
    public_benchmark = build_public_benchmark_summary(claims)
    output.write_text(render_markdown(claims), encoding="utf-8")
    json_output.write_text(
        json.dumps([asdict(claim) for claim in claims], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    index_output.write_text(
        json.dumps(ground_truth_index, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    index_markdown_output.write_text(
        render_ground_truth_index_markdown(ground_truth_index),
        encoding="utf-8",
    )
    public_output.write_text(
        render_public_benchmark_markdown(public_benchmark),
        encoding="utf-8",
    )
    public_json_output.write_text(
        json.dumps(public_benchmark, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "claims": len(claims),
                "index_markdown_output": str(index_markdown_output),
                "index_output": str(index_output),
                "json_output": str(json_output),
                "output": str(output),
                "public_json_output": str(public_json_output),
                "public_output": str(public_output),
            },
            sort_keys=True,
        )
    )
    return 0


def build_claim_evidence(spec: ClaimSpec) -> ClaimEvidence:
    _require_existing_path(spec.scores_path, "scores")
    scores = json.loads(spec.scores_path.read_text(encoding="utf-8"))
    arms = scores["arms"]
    fixed = _dict_value(arms.get("fixed"))
    random = _dict_value(arms.get("random"))
    repair = arms["repair_selected"]
    repair_score = float(repair["mean_task_score"])
    fixed_slice = repair_score - float(scores["repair_task_delta_vs_fixed"])
    random_slice = repair_score - float(scores["repair_task_delta_vs_random"])
    trajectory = arms.get("trajectory_selected") or arms.get("fixed") or {}
    fixed_budget = _mean_generation_budget(fixed, default=1.0)
    random_budget = _mean_generation_budget(random, default=fixed_budget)
    repair_budget = _mean_generation_budget(_dict_value(repair), default=fixed_budget)
    report_path = _report_path_for_scores(spec.scores_path)
    raw_path = spec.raw_path or _raw_path_for_scores(spec.scores_path)
    _require_existing_path(report_path, "report")
    _require_existing_path(raw_path, "raw")
    return ClaimEvidence(
        claim_id=spec.claim_id,
        title=spec.title,
        status=spec.status,
        note=spec.note,
        scores_path=_path_text(spec.scores_path),
        report_path=_path_text(report_path),
        raw_path=_path_text(raw_path),
        all_generation_count=int(scores["all_generation_count"]),
        repair_count=int(repair.get("count", 0)),
        full_count=int(trajectory.get("count", 0)),
        repair_eligible_count=int(scores.get("repair_eligible_task_count", 0)),
        fixed_generation_budget=fixed_budget,
        random_generation_budget=random_budget,
        repair_generation_budget=repair_budget,
        repair_relative_gpu_cost=repair_budget / fixed_budget if fixed_budget else 0.0,
        fixed_repair_slice_score=fixed_slice,
        random_repair_slice_score=random_slice,
        repair_score=repair_score,
        repair_delta_vs_fixed=float(scores["repair_task_delta_vs_fixed"]),
        repair_delta_vs_random=float(scores["repair_task_delta_vs_random"]),
        repair_delta_vs_evolved=_optional_float(scores.get("repair_task_delta_vs_evolved")),
        repair_budget_delta_vs_evolved=_optional_float(
            scores.get("repair_generation_budget_delta_vs_evolved")
        ),
        repair_gain_per_extra_generation=_optional_float(
            scores.get("repair_task_delta_per_extra_generation_vs_evolved")
        ),
        repair_wins_vs_fixed=_format_wins(scores.get("repair_wins_vs_fixed")),
        repair_wins_vs_random=_format_wins(scores.get("repair_wins_vs_random")),
        repair_wins_vs_evolved=_format_wins(scores.get("repair_wins_vs_evolved")),
        oracle_headroom_vs_repair=_optional_float(scores.get("oracle_headroom_vs_repair")),
        repair_pack=str(scores.get("repair_pack", "")),
        repair_source_policy=str(scores.get("repair_source_policy", "")),
        adaptive_source_gate_mode=str(scores.get("adaptive_source_gate_mode", "")),
        exact_task_trajectory_policy=str(scores.get("exact_task_trajectory_policy", "")),
        result_run_id=str(scores.get("run_id", "")),
        result_content_hash=str(scores.get("content_hash", "")),
        required_repair_diagnostics=spec.required_repair_diagnostics,
    )


def render_markdown(claims: list[ClaimEvidence]) -> str:
    lines = [
        "# Claim Evidence Map",
        "",
        "This file is generated by `experiments/build_diffusion_claim_evidence.py`.",
        "It maps current diffusion-reasoning claims to concrete score/report artifacts.",
        "",
        "## Lean Three-Arm Evidence",
        "",
        (
            "| Claim | Status | Records | Coverage | Fixed | Random | Repair | "
            "Repair Cost | Repair Delta vs Fixed | Repair Delta vs Random | Repair Delta vs Evolved | "
            "Gain/Extra Gen | Oracle Headroom | Evidence |"
        ),
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for claim in claims:
        evidence = f"`{claim.scores_path}`<br>`{claim.report_path}`"
        if claim.raw_path:
            evidence += f"<br>`{claim.raw_path}`"
        if claim.result_run_id:
            evidence += f"<br>run `{claim.result_run_id}`"
        lines.append(
            "| "
            f"{claim.title} | "
            f"{claim.status} | "
            f"{claim.all_generation_count} | "
            f"{claim.repair_count}/{claim.full_count} overall; "
            f"{claim.repair_count}/{claim.repair_eligible_count} eligible | "
            f"{claim.fixed_repair_slice_score:.6f} | "
            f"{claim.random_repair_slice_score:.6f} | "
            f"{claim.repair_score:.6f} | "
            f"{claim.repair_relative_gpu_cost:.6f}x | "
            f"{claim.repair_delta_vs_fixed:.6f} | "
            f"{claim.repair_delta_vs_random:.6f} | "
            f"{_format_optional(claim.repair_delta_vs_evolved)} | "
            f"{_format_optional(claim.repair_gain_per_extra_generation)} | "
            f"{_format_optional(claim.oracle_headroom_vs_repair)} | "
            f"{evidence} |"
        )
    ledger = build_moe_mixed_cost_ledger(claims)
    if ledger:
        lines.extend(
            [
                "",
                "## MoE Lean Mixed Cost Ledger",
                "",
                (
                    "Comparable LLaDA-MoE mixed claims only: 11-task lean mixed runs with "
                    "8 repair-covered planning tasks. This table is for score/cost accounting; "
                    "it is not a cross-family ranking of dense-LLaDA or exact-only slices."
                ),
                "",
                "| Frontier | Claim | Repair | Relative Cost | Delta vs Fixed | Delta vs Random | Run |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in ledger:
            lines.append(
                "| "
                f"{'yes' if row['on_frontier'] else 'no'} | "
                f"`{row['claim_id']}` | "
                f"{row['repair_score']:.6f} | "
                f"{row['relative_gpu_cost']:.6f}x | "
                f"{row['repair_delta_vs_fixed']:.6f} | "
                f"{row['repair_delta_vs_random']:.6f} | "
                f"`{row['run_id']}` |"
            )
    lines.extend(
        [
            "",
            "## Claim Notes",
            "",
        ]
    )
    for claim in claims:
        lines.extend(
            [
                f"### {claim.claim_id}",
                "",
                f"- Status: `{claim.status}`",
                f"- Note: {claim.note}",
                f"- Repair wins vs fixed/random/evolved: `{claim.repair_wins_vs_fixed}` / "
                f"`{claim.repair_wins_vs_random}` / `{claim.repair_wins_vs_evolved}`",
                f"- Repair policy: pack `{claim.repair_pack}`, source policy "
                f"`{claim.repair_source_policy}`, adaptive gate `{claim.adaptive_source_gate_mode}`",
                f"- Exact-task trajectory policy: `{claim.exact_task_trajectory_policy}`",
                f"- Result identity: {_format_result_identity(claim)}",
                f"- Required repair diagnostics: {_format_repair_diagnostic_requirements(claim)}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def build_moe_mixed_cost_ledger(claims: list[ClaimEvidence]) -> list[dict[str, object]]:
    comparable = [claim for claim in claims if _is_comparable_moe_mixed_claim(claim)]
    frontier_ids = {claim.claim_id for claim in _pareto_frontier_by_score_and_cost(comparable)}
    rows = [
        {
            "claim_id": claim.claim_id,
            "on_frontier": claim.claim_id in frontier_ids,
            "repair_delta_vs_fixed": claim.repair_delta_vs_fixed,
            "repair_delta_vs_random": claim.repair_delta_vs_random,
            "repair_score": claim.repair_score,
            "relative_gpu_cost": claim.repair_relative_gpu_cost,
            "run_id": claim.result_run_id,
        }
        for claim in comparable
    ]
    return sorted(
        rows,
        key=lambda row: (
            float(row["relative_gpu_cost"]),
            -float(row["repair_score"]),
            str(row["claim_id"]),
        ),
    )


def build_ground_truth_index(
    claims: list[ClaimEvidence],
    *,
    canonical_slots: tuple[CanonicalClaimSlot, ...] = DEFAULT_CANONICAL_SLOTS,
) -> dict[str, object]:
    by_claim_id = {claim.claim_id: claim for claim in claims}
    missing = [slot.claim_id for slot in canonical_slots if slot.claim_id not in by_claim_id]
    if missing:
        raise ValueError(f"Canonical slot references missing claim IDs: {', '.join(missing)}")
    return {
        "claim_count": len(claims),
        "claims": [_claim_index_record(claim) for claim in claims],
        "canonical_slots": [
            _canonical_slot_record(slot, by_claim_id[slot.claim_id]) for slot in canonical_slots
        ],
        "generated_by": "experiments/build_diffusion_claim_evidence.py",
        "schema": "diffusion_ground_truth_index.v1",
    }


def render_ground_truth_index_markdown(index: dict[str, object]) -> str:
    canonical_slots = index.get("canonical_slots", [])
    claims = index.get("claims", [])
    lines = [
        "# Diffusion Ground Truth Index",
        "",
        "This file is generated by `experiments/build_diffusion_claim_evidence.py`.",
        "It names the canonical score, report, and raw artifacts for each promoted claim.",
        "",
        "## Canonical Slots",
        "",
        (
            "| Slot | Claim | Repair | Delta vs Fixed | Delta vs Random | "
            "Delta vs Evolved | Gain/Extra Gen | Evidence |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for slot in canonical_slots:
        if not isinstance(slot, dict):
            continue
        headline = slot.get("headline", {})
        files = slot.get("canonical_files", {})
        if not isinstance(headline, dict) or not isinstance(files, dict):
            continue
        lines.append(
            "| "
            f"{slot.get('label', '')} | "
            f"`{slot.get('claim_id', '')}` | "
            f"{_format_optional(_optional_float(headline.get('repair_score')))} | "
            f"{_format_optional(_optional_float(headline.get('repair_delta_vs_fixed')))} | "
            f"{_format_optional(_optional_float(headline.get('repair_delta_vs_random')))} | "
            f"{_format_optional(_optional_float(headline.get('repair_delta_vs_evolved')))} | "
            f"{_format_optional(_optional_float(headline.get('repair_gain_per_extra_generation')))} | "
            f"`{files.get('scores', '')}`<br>`{files.get('report', '')}`<br>"
            f"`{files.get('raw', '')}` |"
        )

    lines.extend(
        [
            "",
            "## Promoted Claim Files",
            "",
            "| Claim ID | Status | Run ID | Scores | Report | Raw | Hashes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        files = claim.get("canonical_files", {})
        hashes = claim.get("canonical_file_hashes", {})
        if not isinstance(files, dict):
            continue
        lines.append(
            "| "
            f"`{claim.get('claim_id', '')}` | "
            f"{claim.get('status', '')} | "
            f"{_format_index_result_identity(claim)} | "
            f"`{files.get('scores', '')}` | "
            f"`{files.get('report', '')}` | "
            f"`{files.get('raw', '')}` | "
            f"{_format_artifact_hashes(hashes)} |"
        )
    lines.extend(["", "## Selection Notes", ""])
    for slot in canonical_slots:
        if not isinstance(slot, dict):
            continue
        lines.append(
            f"- `{slot.get('slot_id', '')}` -> `{slot.get('claim_id', '')}`: "
            f"{slot.get('selection_reason', '')}"
        )
    return "\n".join(lines) + "\n"


def build_public_benchmark_summary(
    claims: list[ClaimEvidence],
    *,
    claim_id: str = DEFAULT_PUBLIC_BENCHMARK_CLAIM_ID,
    top_score_claim_id: str = DEFAULT_PUBLIC_TOP_SCORE_CLAIM_ID,
    budget_claim_id: str = DEFAULT_PUBLIC_BUDGET_CLAIM_ID,
) -> dict[str, object]:
    claim = _claim_by_id(claims, claim_id)
    top_score_claim = _claim_by_id(claims, top_score_claim_id)
    budget_claim = _claim_by_id(claims, budget_claim_id)
    scores = json.loads(Path(claim.scores_path).read_text(encoding="utf-8"))
    arms = _dict_value(scores.get("arms"))
    fixed_arm = _dict_value(arms.get("fixed"))
    random_arm = _dict_value(arms.get("random"))
    repair_arm = _dict_value(arms.get("repair_selected"))
    fixed_cost = _mean_generation_budget(fixed_arm, default=1.0)
    random_cost = _mean_generation_budget(random_arm, default=1.0)
    repair_cost = _mean_generation_budget(repair_arm, default=fixed_cost)
    guard_checks = _public_guard_checks(scores)
    return {
        "artifact_paths": {
            "raw": claim.raw_path,
            "report": claim.report_path,
            "scores": claim.scores_path,
        },
        "benchmark": "lean_gpu_mixed",
        "claim_id": claim.claim_id,
        "coverage": {
            "full_task_count": claim.full_count,
            "repair_eligible_task_count": claim.repair_eligible_count,
            "repair_task_count": claim.repair_count,
        },
        "generated_by": "experiments/build_diffusion_claim_evidence.py",
        "guard_checks": guard_checks,
        "latent_repair_frontier": _latent_repair_frontier_records(top_score_claim, budget_claim),
        "primary_slice": {
            "family": "planning",
            "task_count": claim.repair_count,
            "description": "short-planning repair-eligible tasks",
        },
        "public_arms": [
            _public_arm_record(
                arm_id="greedy",
                label="Greedy",
                score=claim.fixed_repair_slice_score,
                generation_budget=fixed_cost,
                baseline_budget=fixed_cost,
                random_budget=random_cost,
                delta_vs_greedy=0.0,
                delta_vs_random=claim.fixed_repair_slice_score - claim.random_repair_slice_score,
            ),
            _public_arm_record(
                arm_id="random_perturbation",
                label="Random perturbation",
                score=claim.random_repair_slice_score,
                generation_budget=random_cost,
                baseline_budget=fixed_cost,
                random_budget=random_cost,
                delta_vs_greedy=claim.random_repair_slice_score - claim.fixed_repair_slice_score,
                delta_vs_random=0.0,
            ),
            _public_arm_record(
                arm_id="latent_repair",
                label="Latent repair",
                score=claim.repair_score,
                generation_budget=repair_cost,
                baseline_budget=fixed_cost,
                random_budget=random_cost,
                delta_vs_greedy=claim.repair_delta_vs_fixed,
                delta_vs_random=claim.repair_delta_vs_random,
            ),
        ],
        "result_identity": _result_identity_record(claim),
        "schema": "diffusion_public_benchmark.v1",
        "total_generation_records": claim.all_generation_count,
    }


def render_public_benchmark_markdown(summary: dict[str, object]) -> str:
    coverage = _dict_value(summary.get("coverage"))
    identity = _dict_value(summary.get("result_identity"))
    lines = [
        "# Diffusion Public Benchmark",
        "",
        "This file is generated by `experiments/build_diffusion_claim_evidence.py`.",
        "It keeps the public comparison to the three allowed GPU arms.",
        "",
        "## Current Result",
        "",
        f"- Benchmark: `{summary.get('benchmark', '')}`",
        "- Primary slice: short-planning repair-eligible tasks",
        (
            f"- Coverage: {coverage.get('repair_task_count', '')}/"
            f"{coverage.get('full_task_count', '')} tasks, "
            f"{coverage.get('repair_eligible_task_count', '')} repair-eligible"
        ),
        f"- Total generation records: {summary.get('total_generation_records', '')}",
        f"- Run: `{identity.get('run_id', '') or 'unrecorded'}`",
        "- Evidence: canonical score, report, and raw artifacts are recorded in "
        "`DIFFUSION_GROUND_TRUTH_INDEX.md`.",
        "",
        "## Public Arms",
        "",
        (
            "| Arm | Score | Relative GPU Cost | Delta vs Greedy | Delta vs Random | "
            "Lift per Extra Cost vs Greedy | Lift per Extra Cost vs Random |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in _list_of_dicts(summary.get("public_arms")):
        lines.append(
            "| "
            f"{arm.get('label', '')} | "
            f"{_format_optional(_optional_float(arm.get('score')))} | "
            f"{_format_optional(_optional_float(arm.get('relative_gpu_cost')))}x | "
            f"{_format_optional(_optional_float(arm.get('delta_vs_greedy')))} | "
            f"{_format_optional(_optional_float(arm.get('delta_vs_random')))} | "
            f"{_format_optional(_optional_float(arm.get('lift_per_extra_cost_vs_greedy')))} |"
            f" {_format_optional(_optional_float(arm.get('lift_per_extra_cost_vs_random')))} |"
        )
    lines.extend(
        [
            "",
            "## Latent Repair Cost Frontier",
            "",
            (
                "| Policy | Score | Relative GPU Cost | Delta vs Greedy | Delta vs Random | "
                "Lift per Extra Cost vs Greedy | Lift per Extra Cost vs Random | Records |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy in _list_of_dicts(summary.get("latent_repair_frontier")):
        lines.append(
            "| "
            f"{policy.get('label', '')} | "
            f"{_format_optional(_optional_float(policy.get('score')))} | "
            f"{_format_optional(_optional_float(policy.get('relative_gpu_cost')))}x | "
            f"{_format_optional(_optional_float(policy.get('delta_vs_greedy')))} | "
            f"{_format_optional(_optional_float(policy.get('delta_vs_random')))} | "
            f"{_format_optional(_optional_float(policy.get('lift_per_extra_cost_vs_greedy')))} | "
            f"{_format_optional(_optional_float(policy.get('lift_per_extra_cost_vs_random')))} | "
            f"{policy.get('total_generation_records', '')} |"
        )
    lines.extend(
        [
            "",
            "## Guard Checks",
            "",
            "| Family | Tasks | Greedy | Random Perturbation | Latent Repair Spend |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for guard in _list_of_dicts(summary.get("guard_checks")):
        lines.append(
            "| "
            f"{guard.get('family', '')} | "
            f"{guard.get('task_count', '')} | "
            f"{_format_optional(_optional_float(guard.get('greedy_score')))} | "
            f"{_format_optional(_optional_float(guard.get('random_perturbation_score')))} | "
            f"{guard.get('latent_repair_spend', '')} |"
        )
    return "\n".join(lines) + "\n"


def _claim_by_id(claims: list[ClaimEvidence], claim_id: str) -> ClaimEvidence:
    for claim in claims:
        if claim.claim_id == claim_id:
            return claim
    if len(claims) == 1:
        return claims[0]
    raise ValueError(f"Public benchmark claim is missing: {claim_id}")


def _latent_repair_frontier_records(
    top_score_claim: ClaimEvidence,
    budget_claim: ClaimEvidence,
) -> list[dict[str, object]]:
    records = [
        _latent_repair_policy_record("Top-score latent repair", top_score_claim),
    ]
    if budget_claim.claim_id != top_score_claim.claim_id:
        records.append(_latent_repair_policy_record("Budget-favored latent repair", budget_claim))
    return records


def _latent_repair_policy_record(label: str, claim: ClaimEvidence) -> dict[str, object]:
    scores = json.loads(Path(claim.scores_path).read_text(encoding="utf-8"))
    arms = _dict_value(scores.get("arms"))
    repair = _dict_value(arms.get("repair_selected"))
    fixed = _dict_value(arms.get("fixed"))
    random = _dict_value(arms.get("random"))
    repair_cost = _mean_generation_budget(repair, default=1.0)
    fixed_cost = _mean_generation_budget(fixed, default=1.0)
    random_cost = _mean_generation_budget(random, default=fixed_cost)
    extra_cost = repair_cost - fixed_cost
    extra_cost_vs_random = repair_cost - random_cost
    return {
        "claim_id": claim.claim_id,
        "delta_vs_greedy": claim.repair_delta_vs_fixed,
        "delta_vs_random": claim.repair_delta_vs_random,
        "generation_budget_per_task": repair_cost,
        "label": label,
        "lift_per_extra_cost_vs_greedy": (
            claim.repair_delta_vs_fixed / extra_cost if extra_cost > 0.0 else None
        ),
        "lift_per_extra_cost_vs_random": (
            claim.repair_delta_vs_random / extra_cost_vs_random
            if extra_cost_vs_random > 0.0
            else None
        ),
        "relative_gpu_cost": repair_cost / fixed_cost if fixed_cost else None,
        "score": claim.repair_score,
        "scores_path": claim.scores_path,
        "total_generation_records": claim.all_generation_count,
    }


def _public_arm_record(
    *,
    arm_id: str,
    label: str,
    score: float,
    generation_budget: float,
    baseline_budget: float,
    random_budget: float,
    delta_vs_greedy: float,
    delta_vs_random: float,
) -> dict[str, object]:
    extra_cost = generation_budget - baseline_budget
    extra_cost_vs_random = generation_budget - random_budget
    lift_per_extra = delta_vs_greedy / extra_cost if extra_cost > 0.0 else None
    lift_per_extra_vs_random = (
        delta_vs_random / extra_cost_vs_random if extra_cost_vs_random > 0.0 else None
    )
    return {
        "arm_id": arm_id,
        "delta_vs_greedy": delta_vs_greedy,
        "delta_vs_random": delta_vs_random,
        "generation_budget_per_task": generation_budget,
        "label": label,
        "lift_per_extra_cost_vs_greedy": lift_per_extra,
        "lift_per_extra_cost_vs_random": lift_per_extra_vs_random,
        "relative_gpu_cost": generation_budget / baseline_budget if baseline_budget else None,
        "score": score,
    }


def _public_guard_checks(scores: dict[str, object]) -> list[dict[str, object]]:
    by_family = _dict_value(scores.get("by_family_arm"))
    rows: list[dict[str, object]] = []
    for family in PUBLIC_GUARD_FAMILIES:
        family_scores = _dict_value(by_family.get(family))
        fixed = _dict_value(family_scores.get("fixed"))
        random = _dict_value(family_scores.get("random"))
        rows.append(
            {
                "family": family,
                "greedy_score": _optional_float(fixed.get("mean_task_score")),
                "latent_repair_spend": "none",
                "random_perturbation_score": _optional_float(random.get("mean_task_score")),
                "task_count": int(fixed.get("count", random.get("count", 0)) or 0),
            }
        )
    return rows


def _dict_value(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _mean_generation_budget(value: dict[str, object], *, default: float) -> float:
    if "mean_generation_budget_per_task" not in value:
        return default
    return float(value["mean_generation_budget_per_task"])


def _is_comparable_moe_mixed_claim(claim: ClaimEvidence) -> bool:
    return (
        claim.claim_id.startswith("moe_mixed")
        and claim.full_count == 11
        and claim.repair_count == 8
        and claim.repair_eligible_count == 8
    )


def _pareto_frontier_by_score_and_cost(claims: list[ClaimEvidence]) -> list[ClaimEvidence]:
    frontier: list[ClaimEvidence] = []
    for candidate in claims:
        dominated = False
        for challenger in claims:
            if challenger.claim_id == candidate.claim_id:
                continue
            score_at_least = challenger.repair_score >= candidate.repair_score - 1e-12
            cost_at_most = (
                challenger.repair_relative_gpu_cost <= candidate.repair_relative_gpu_cost + 1e-12
            )
            strictly_better = (
                challenger.repair_score > candidate.repair_score + 1e-12
                or challenger.repair_relative_gpu_cost
                < candidate.repair_relative_gpu_cost - 1e-12
            )
            if score_at_least and cost_at_most and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def _claim_index_record(claim: ClaimEvidence) -> dict[str, object]:
    return {
        "all_generation_count": claim.all_generation_count,
        "canonical_files": {
            "raw": claim.raw_path,
            "report": claim.report_path,
            "scores": claim.scores_path,
        },
        "canonical_file_hashes": _artifact_hashes(claim),
        "claim_id": claim.claim_id,
        "coverage": {
            "full_count": claim.full_count,
            "repair_count": claim.repair_count,
            "repair_eligible_count": claim.repair_eligible_count,
        },
        "headline": _headline_record(claim),
        "note": claim.note,
        "result_identity": _result_identity_record(claim),
        "required_repair_diagnostics": [
            _repair_diagnostic_requirement_record(requirement)
            for requirement in claim.required_repair_diagnostics
        ],
        "settings": {
            "adaptive_source_gate_mode": claim.adaptive_source_gate_mode,
            "exact_task_trajectory_policy": claim.exact_task_trajectory_policy,
            "repair_pack": claim.repair_pack,
            "repair_source_policy": claim.repair_source_policy,
        },
        "status": claim.status,
        "title": claim.title,
    }


def _canonical_slot_record(slot: CanonicalClaimSlot, claim: ClaimEvidence) -> dict[str, object]:
    return {
        "canonical_files": {
            "raw": claim.raw_path,
            "report": claim.report_path,
            "scores": claim.scores_path,
        },
        "canonical_file_hashes": _artifact_hashes(claim),
        "claim_id": claim.claim_id,
        "headline": _headline_record(claim),
        "label": slot.label,
        "result_identity": _result_identity_record(claim),
        "required_repair_diagnostics": [
            _repair_diagnostic_requirement_record(requirement)
            for requirement in claim.required_repair_diagnostics
        ],
        "selection_reason": slot.selection_reason,
        "slot_id": slot.slot_id,
        "status": claim.status,
        "title": claim.title,
    }


def _headline_record(claim: ClaimEvidence) -> dict[str, object]:
    return {
        "fixed_repair_slice_score": claim.fixed_repair_slice_score,
        "fixed_generation_budget": claim.fixed_generation_budget,
        "oracle_headroom_vs_repair": claim.oracle_headroom_vs_repair,
        "random_generation_budget": claim.random_generation_budget,
        "random_repair_slice_score": claim.random_repair_slice_score,
        "repair_generation_budget": claim.repair_generation_budget,
        "repair_relative_gpu_cost": claim.repair_relative_gpu_cost,
        "repair_budget_delta_vs_evolved": claim.repair_budget_delta_vs_evolved,
        "repair_delta_vs_evolved": claim.repair_delta_vs_evolved,
        "repair_delta_vs_fixed": claim.repair_delta_vs_fixed,
        "repair_delta_vs_random": claim.repair_delta_vs_random,
        "repair_gain_per_extra_generation": claim.repair_gain_per_extra_generation,
        "repair_score": claim.repair_score,
        "repair_wins_vs_evolved": claim.repair_wins_vs_evolved,
        "repair_wins_vs_fixed": claim.repair_wins_vs_fixed,
        "repair_wins_vs_random": claim.repair_wins_vs_random,
    }


def _result_identity_record(claim: ClaimEvidence) -> dict[str, str]:
    return {
        "content_hash": claim.result_content_hash,
        "run_id": claim.result_run_id,
    }


def _artifact_hashes(claim: ClaimEvidence) -> dict[str, str]:
    return {
        "raw_sha256": _file_sha256(Path(claim.raw_path)),
        "report_sha256": _file_sha256(Path(claim.report_path)),
        "scores_sha256": _file_sha256(Path(claim.scores_path)),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_path_for_scores(scores_path: Path) -> Path:
    name = scores_path.name
    if name.endswith("_scores.json"):
        return scores_path.with_name(name.removesuffix("_scores.json") + "_report.md")
    return scores_path.with_suffix(".md")


def _raw_path_for_scores(scores_path: Path) -> Path:
    name = scores_path.name
    if name.endswith("_scores.json"):
        return scores_path.with_name(name.removesuffix("_scores.json") + "_raw.jsonl")
    return Path("")


def _require_existing_path(path: Path, label: str) -> None:
    if not str(path) or not path.exists():
        raise EvidenceArtifactMissingError(f"Missing {label} evidence artifact: {path}")


def _path_text(path: Path | None) -> str:
    if path is None or not str(path):
        return ""
    return path.as_posix()


def _format_wins(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    return f"{int(value.get('wins', 0))}/{int(value.get('ties', 0))}/{int(value.get('losses', 0))}"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _format_artifact_hashes(hashes: object) -> str:
    if not isinstance(hashes, dict):
        return ""
    return (
        f"scores `{_short_hash(hashes.get('scores_sha256'))}`<br>"
        f"report `{_short_hash(hashes.get('report_sha256'))}`<br>"
        f"raw `{_short_hash(hashes.get('raw_sha256'))}`"
    )


def _format_result_identity(claim: ClaimEvidence) -> str:
    if claim.result_run_id and claim.result_content_hash:
        return f"`{claim.result_run_id}` / `{claim.result_content_hash}`"
    if claim.result_run_id:
        return f"`{claim.result_run_id}`"
    if claim.result_content_hash:
        return f"`{claim.result_content_hash}`"
    return "`unrecorded`"


def _format_repair_diagnostic_requirements(claim: ClaimEvidence) -> str:
    if not claim.required_repair_diagnostics:
        return "`none`"
    return "; ".join(
        f"`{requirement.repair_name}.{requirement.metric} "
        f"{_format_requirement_bounds(requirement)}`"
        for requirement in claim.required_repair_diagnostics
    )


def _format_requirement_bounds(requirement: RepairDiagnosticRequirement) -> str:
    parts: list[str] = []
    if requirement.min_value is not None:
        parts.append(f">= {requirement.min_value:.6f}")
    if requirement.max_value is not None:
        parts.append(f"<= {requirement.max_value:.6f}")
    return " and ".join(parts) if parts else "present"


def _repair_diagnostic_requirement_record(
    requirement: RepairDiagnosticRequirement,
) -> dict[str, object]:
    return {
        "max_value": requirement.max_value,
        "metric": requirement.metric,
        "min_value": requirement.min_value,
        "note": requirement.note,
        "repair_name": requirement.repair_name,
    }


def _format_index_result_identity(claim: dict[str, object]) -> str:
    identity = claim.get("result_identity", {})
    if not isinstance(identity, dict):
        return "`unrecorded`"
    run_id = str(identity.get("run_id", "") or "")
    content_hash = str(identity.get("content_hash", "") or "")
    if run_id and content_hash:
        return f"`{run_id}`<br>`{content_hash[:12]}`"
    if run_id:
        return f"`{run_id}`"
    if content_hash:
        return f"`{content_hash[:12]}`"
    return "`unrecorded`"


def _short_hash(value: object) -> str:
    return str(value or "")[:12]


if __name__ == "__main__":
    raise SystemExit(main())
