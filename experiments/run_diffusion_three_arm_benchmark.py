"""Run a GPU-bounded diffusion trajectory benchmark.

The arms are deliberately narrow:

- fixed: first default denoising schedule for the model
- random: deterministic per-task random schedule choice
- trajectory_selected: select the schedule with the best trajectory-control
  score from the base schedule pool, without using task scores
- evolved: select from the base pool plus a small mutated schedule pool,
  still without using task scores. A conservative promotion margin can keep the
  base trajectory-selected schedule when the evolved selector edge is tiny.
- repair_selected: optional LLaDA suffix-inpainting repair branches from the
  configured repair source, again promoted only by selector margin.

Task scores are still computed after generation so the arms can be compared.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.diffusion import (  # noqa: E402
    DiffusionGenerationConfig,
    DiffusionVerifierRepairCandidate,
    HFDiffusionBackend,
    attach_control_score,
    build_text_span_repair_seed_diagnostics,
    default_dream_schedules,
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
    default_llada_constraint_span_anchor_search_repair_candidates,
    default_llada_constraint_span_anchor_select_repair_candidates,
    default_llada_constraint_span_clause_repair_candidates,
    default_llada_constraint_span_history_contrast_repair_candidates,
    default_llada_constraint_span_history_instability_repair_candidates,
    default_llada_constraint_span_history_repair_candidates,
    default_llada_constraint_span_phase_anchor_repair_candidates,
    default_llada_constraint_span_phase_final_preserve_seeded_gated_repair_candidates,
    default_llada_constraint_span_phase_hybrid_preserve_seeded_gated_repair_candidates,
    default_llada_constraint_span_repair_candidates,
    default_llada_history_repair_candidates,
    default_llada_history_visible_repair_candidates,
    default_llada_prompt_guided_repair_candidates,
    default_llada_repair_candidates,
    default_llada_replay_consistency_repair_candidates,
    default_llada_schedules,
    default_llada_source_relative_repair_candidates,
    default_llada_state_adaptive_repair_candidates,
    default_llada_targeted_content_repair_candidates,
    default_llada_verifier_repair_candidates,
    get_candidate,
    is_llada_family,
)
from latent_reasoning.eval.answer_proposals import (  # noqa: E402
    counterfactual_answer_proposals,
    symbolic_short_text_candidate_from_prompt,
)
from latent_reasoning.eval.general_reasoning import (  # noqa: E402
    GeneralReasoningTask,
    extract_choice,
    load_tasks,
    score_planning_output,
    score_task_output,
)

MODEL_PATHS = {
    "dream-7b-instruct-hf": "external/diffusion_models/Dream-v0-Instruct-7B",
    "llada-8b-instruct-hf": "external/diffusion_models/LLaDA-8B-Instruct",
    "llada-moe-7b-a1b-instruct-hf": "external/diffusion_models/LLaDA-MoE-7B-A1B-Instruct",
}


def _local_model_path(candidate_key: str) -> str | None:
    path = MODEL_PATHS.get(candidate_key)
    if path and Path(path).exists():
        return path
    return None

BASE_ARMS = ("fixed", "random", "trajectory_selected")
ARMS = (*BASE_ARMS, "evolved", "repair_selected")
ARM_ORDER = {arm: index for index, arm in enumerate(ARMS)}
DEFAULT_EVOLVED_SELECTOR = "inherit"
DEFAULT_EVOLVED_QUALITY_MARGIN = 0.01
DEFAULT_EVOLVED_SELECTOR_TOLERANCE = 0.015
DEFAULT_REVISION_PROMOTION_MARGIN = 0.05
DEFAULT_REPAIR_PROMOTION_MARGIN = 0.02
DEFAULT_REPAIR_SELECTOR = "planning_quality"
DECOMPOSED_FOUR_HEAD_SELECTOR_ID = "decomposed_four_head_selector"
DECOMPOSED_SELECTOR_SPEND_RULE_ID = "first_repairable_gap_le_9_source_quality_le_0p301429"
DECOMPOSED_SELECTOR_SOURCE_RULE_ID = "retention_safe_history"
DECOMPOSED_SELECTOR_RETENTION_RULE_ID = "classification_safe_history_anchor"
DECOMPOSED_SELECTOR_REALIZATION_RULE_ID = "min_realization_policy_error"
DECOMPOSED_SPEND_TRANSFER_SELECTOR_ID = "decomposed_spend_transfer_rule"
DECOMPOSED_SPEND_TRANSFER_RULE_ID = "current_decomposed_spend_source_task_ge_0p295357"
DECOMPOSED_SPEND_TRANSFER_SOURCE_TASK_MIN = 0.295357
TRAJECTORY_RELATIVE_DECOMPOSED_SPEND_SELECTOR_ID = "trajectory_relative_decomposed_spend"
TRAJECTORY_RELATIVE_DECOMPOSED_SPEND_RULE_ID = (
    "current_decomposed_spend_source_task_ge_0p295357_source_ge_trajectory"
)
LEARNED_AVAILABILITY_PREDICTOR_SELECTOR_ID = "learned_availability_predictor_v1"
LEARNED_AVAILABILITY_PREDICTOR_RULE_ID = (
    "learned_gap_le_8_source_quality_le_0p256429_source_ge_trajectory"
)
LEARNED_AVAILABILITY_SOURCE_QUALITY_MAX = 0.256429
LEARNED_AVAILABILITY_PROMPT_GAP_MAX = 8
CALIBRATED_AVAILABILITY_PREDICTOR_SELECTOR_ID = "calibrated_availability_predictor_v1"
CALIBRATED_AVAILABILITY_PREDICTOR_RULE_ID = "calibrated_gap_not_7_source_ge_trajectory"
CALIBRATED_AVAILABILITY_BLOCKED_PROMPT_GAP = 7
COUNTERFACTUAL_MICRO_PROBE_TRIGGER_ID = "counterfactual_micro_probe_v1"
COUNTERFACTUAL_MICRO_PROBE_POLICY_ID = "deterministic_missing_constraint_probe_v1"
COUNTERFACTUAL_MICRO_PROBE_TOMOGRAPHY_POLICY_ID = "strict_tomography_probe_v1"
COUNTERFACTUAL_MICRO_PROBE_KEY_VALUE_POLICY_ID = "key_value_tomography_probe_v2"
COUNTERFACTUAL_MICRO_PROBE_COMPACT_POLICY_ID = "compact_tomography_probe_v3"
COUNTERFACTUAL_MICRO_PROBE_SPAN_POLICY_ID = "span_tomography_probe_v4"
COUNTERFACTUAL_MICRO_PROBE_POLICIES = (
    COUNTERFACTUAL_MICRO_PROBE_POLICY_ID,
    COUNTERFACTUAL_MICRO_PROBE_TOMOGRAPHY_POLICY_ID,
    COUNTERFACTUAL_MICRO_PROBE_KEY_VALUE_POLICY_ID,
    COUNTERFACTUAL_MICRO_PROBE_COMPACT_POLICY_ID,
    COUNTERFACTUAL_MICRO_PROBE_SPAN_POLICY_ID,
)
COUNTERFACTUAL_MICRO_PROBE_COST_RELATIVE = 0.125
COUNTERFACTUAL_MICRO_PROBE_GAP_VISIBILITY_MAX = 2.0 / 3.0
COUNTERFACTUAL_MICRO_PROBE_MAX_NEW_TOKENS = 32
COUNTERFACTUAL_MICRO_PROBE_STEPS = 16
COUNTERFACTUAL_MICRO_PROBE_TOMOGRAPHY_MAX_NEW_TOKENS = 48
COUNTERFACTUAL_MICRO_PROBE_TOMOGRAPHY_STEPS = 24
COUNTERFACTUAL_MICRO_PROBE_KEY_VALUE_MAX_NEW_TOKENS = 64
COUNTERFACTUAL_MICRO_PROBE_KEY_VALUE_STEPS = 32
COUNTERFACTUAL_MICRO_PROBE_COMPACT_MAX_NEW_TOKENS = 48
COUNTERFACTUAL_MICRO_PROBE_COMPACT_STEPS = 24
COUNTERFACTUAL_MICRO_PROBE_SPAN_MAX_NEW_TOKENS = 48
COUNTERFACTUAL_MICRO_PROBE_SPAN_STEPS = 24
COUNTERFACTUAL_MICRO_PROBE_MODES = ("triage", "all")
DEFAULT_ADAPTIVE_SOURCE_GAP_MIN_TERMS = 6
DEFAULT_ADAPTIVE_SOURCE_QUALITY_FLOOR = 0.25
DEFAULT_ADAPTIVE_SOURCE_QUALITY_CEILING: float | None = None
ANCHOR_HISTORY_CHAR_RATIO_MIN = 0.95
ANCHOR_TARGET_SIMILARITY_MIN = 0.96
PHASE_ANCHOR_HISTORY_CHAR_RATIO_MIN = 0.90
PHASE_ANCHOR_TARGET_SIMILARITY_MIN = 0.90
PHASE_SOURCE_TARGET_SIMILARITY_MIN = 0.96
PHASE_SOURCE_TEXT_SIMILARITY_MIN = 0.96
PHASE_SOURCE_HISTORY_CHAR_RATIO_MIN = 0.95
ADAPTIVE_SOURCE_GATE_MODES = ("custom", "score_max", "efficiency", "score_efficient")
ADAPTIVE_SOURCE_GATE_MODE_DEFAULTS = {
    "score_max": (6, 0.25, None),
    "efficiency": (10, 0.25, None),
    "score_efficient": (6, 0.25, 0.50),
}
REPAIR_SOURCE_POLICIES = (
    "evolved",
    "trajectory",
    "fixed",
    "random",
    "non_revision_evolved",
    "evolved_and_trajectory",
    "non_revision_plus_gap_trajectory",
)
PRE_GENERATION_ANCHOR_SOURCE_STATE = "pre_generation_anchor"
PRE_GENERATION_ANCHOR_SEARCH_SOURCE_STATE = "pre_generation_anchor_search"
PRE_GENERATION_PHASE_ANCHOR_SOURCE_STATE = "pre_generation_phase_anchor"
PRE_GENERATION_PHASE_HYBRID_SOURCE_STATE = "pre_generation_phase_hybrid_anchor"
PHASE_FINAL_PRESERVE_REPAIR_NAME = "constraint_gap_span_phase_final_preserve_seeded_gated_repair"
PRE_GENERATION_ANCHOR_SOURCE_STATES = frozenset(
    {
        PRE_GENERATION_ANCHOR_SOURCE_STATE,
        PRE_GENERATION_ANCHOR_SEARCH_SOURCE_STATE,
        PRE_GENERATION_PHASE_ANCHOR_SOURCE_STATE,
        PRE_GENERATION_PHASE_HYBRID_SOURCE_STATE,
    }
)
REPAIR_SELECTORS = (
    "inherit",
    "transfer_promotion_value",
    "planning_quality",
    "planning_quality_guarded",
    "planning_quality_risk_guarded",
    "planning_quality_delta",
    "planning_quality_delta_guarded",
    "planning_quality_delta_risk_guarded",
    "planning_quality_prompt_coverage_guarded",
    "planning_quality_seed_objective_guarded",
    "planning_quality_seed_realization_guarded",
    "candidate_aware_promotion_v1",
)
EXACT_ANSWER_REPAIR_NAMES = frozenset(
    {
        "counterfactual_answer_proposal",
        "answer_span_repair",
        "answer_context_random_repair",
        "arithmetic_contradiction_span_repair",
        "self_check_answer_repair",
        "arithmetic_feedback_repair",
        "arithmetic_evidence_repair",
    }
)
SELF_REPAIR_EVIDENCE_NAMES = frozenset(
    {
        "answer_span_repair",
        "self_check_answer_repair",
        "arithmetic_contradiction_span_repair",
        "arithmetic_feedback_repair",
        "arithmetic_evidence_repair",
    }
)
NO_PROPOSAL_VERIFIER_SPAN_ANSWER_TYPES = frozenset({"short_text", "multiple_choice"})
EXACT_REPAIR_SELECTION_PRIORITIES = {
    "answer_span_repair": 0.030,
    "arithmetic_contradiction_span_repair": 0.030,
    "arithmetic_feedback_repair": 0.015,
    "arithmetic_evidence_repair": 0.010,
    "self_check_answer_repair": 0.000,
}

LEAN_GPU_MIXED_TASK_IDS = (
    "plan_001",
    "plan_002",
    "plan_003",
    "plan_004",
    "plan_005",
    "plan_006",
    "plan_007",
    "plan_008",
    "math_001",
    "sym_002",
    "sci_001",
)
LEAN_GPU_MIXED_TRANSFER_TASK_IDS = (
    "plan_009",
    "plan_010",
    "plan_011",
    "plan_012",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V2_TASK_IDS = (
    "plan_009",
    "plan_010",
    "plan_011",
    "plan_012",
    "plan_013",
    "plan_014",
    "plan_015",
    "plan_016",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V3_TASK_IDS = (
    "plan_009",
    "plan_010",
    "plan_011",
    "plan_012",
    "plan_013",
    "plan_014",
    "plan_015",
    "plan_016",
    "plan_017",
    "plan_018",
    "plan_019",
    "plan_020",
    "plan_021",
    "plan_022",
    "plan_023",
    "plan_024",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V4_TASK_IDS = (
    "plan_025",
    "plan_026",
    "plan_027",
    "plan_028",
    "plan_029",
    "plan_030",
    "plan_031",
    "plan_032",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V5_TASK_IDS = (
    "plan_033",
    "plan_034",
    "plan_035",
    "plan_036",
    "plan_037",
    "plan_038",
    "plan_039",
    "plan_040",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V6_TASK_IDS = (
    "plan_041",
    "plan_042",
    "plan_043",
    "plan_044",
    "plan_045",
    "plan_046",
    "plan_047",
    "plan_048",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V7_TASK_IDS = (
    "plan_049",
    "plan_050",
    "plan_051",
    "plan_052",
    "plan_053",
    "plan_054",
    "plan_055",
    "plan_056",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V8_TASK_IDS = (
    "plan_057",
    "plan_058",
    "plan_059",
    "plan_060",
    "plan_061",
    "plan_062",
    "plan_063",
    "plan_064",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V9_TASK_IDS = (
    "plan_065",
    "plan_066",
    "plan_067",
    "plan_068",
    "plan_069",
    "plan_070",
    "plan_071",
    "plan_072",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V10_TASK_IDS = (
    "plan_073",
    "plan_074",
    "plan_075",
    "plan_076",
    "plan_077",
    "plan_078",
    "plan_079",
    "plan_080",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V11_TASK_IDS = (
    "plan_081",
    "plan_082",
    "plan_083",
    "plan_084",
    "plan_085",
    "plan_086",
    "plan_087",
    "plan_088",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V12_TASK_IDS = (
    "plan_089",
    "plan_090",
    "plan_091",
    "plan_092",
    "plan_093",
    "plan_094",
    "plan_095",
    "plan_096",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V13_TASK_IDS = (
    "plan_097",
    "plan_098",
    "plan_099",
    "plan_100",
    "plan_101",
    "plan_102",
    "plan_103",
    "plan_104",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V14_TASK_IDS = (
    "plan_105",
    "plan_106",
    "plan_107",
    "plan_108",
    "plan_109",
    "plan_110",
    "plan_111",
    "plan_112",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V15_TASK_IDS = (
    "plan_113",
    "plan_114",
    "plan_115",
    "plan_116",
    "plan_117",
    "plan_118",
    "plan_119",
    "plan_120",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V16_TASK_IDS = (
    "plan_121",
    "plan_122",
    "plan_123",
    "plan_124",
    "plan_125",
    "plan_126",
    "plan_127",
    "plan_128",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V17_TASK_IDS = (
    "plan_129",
    "plan_130",
    "plan_131",
    "plan_132",
    "plan_133",
    "plan_134",
    "plan_135",
    "plan_136",
    "math_009",
    "sym_007",
    "sci_002",
)
LEAN_GPU_MIXED_TRANSFER_V18_TASK_IDS = (
    "plan_137",
    "plan_138",
    "plan_139",
    "plan_140",
    "plan_141",
    "plan_142",
    "plan_143",
    "plan_144",
    "math_009",
    "sym_007",
    "sci_002",
)

TASK_PRESETS = {
    "lean_gpu_mixed": LEAN_GPU_MIXED_TASK_IDS,
    "lean-gpu-mixed": LEAN_GPU_MIXED_TASK_IDS,
    "lean_gpu_mixed_transfer": LEAN_GPU_MIXED_TRANSFER_TASK_IDS,
    "lean-gpu-mixed-transfer": LEAN_GPU_MIXED_TRANSFER_TASK_IDS,
    "lean_gpu_mixed_transfer_v2": LEAN_GPU_MIXED_TRANSFER_V2_TASK_IDS,
    "lean-gpu-mixed-transfer-v2": LEAN_GPU_MIXED_TRANSFER_V2_TASK_IDS,
    "lean_gpu_mixed_transfer_v3": LEAN_GPU_MIXED_TRANSFER_V3_TASK_IDS,
    "lean-gpu-mixed-transfer-v3": LEAN_GPU_MIXED_TRANSFER_V3_TASK_IDS,
    "lean_gpu_mixed_transfer_v4": LEAN_GPU_MIXED_TRANSFER_V4_TASK_IDS,
    "lean-gpu-mixed-transfer-v4": LEAN_GPU_MIXED_TRANSFER_V4_TASK_IDS,
    "lean_gpu_mixed_transfer_v5": LEAN_GPU_MIXED_TRANSFER_V5_TASK_IDS,
    "lean-gpu-mixed-transfer-v5": LEAN_GPU_MIXED_TRANSFER_V5_TASK_IDS,
    "lean_gpu_mixed_transfer_v6": LEAN_GPU_MIXED_TRANSFER_V6_TASK_IDS,
    "lean-gpu-mixed-transfer-v6": LEAN_GPU_MIXED_TRANSFER_V6_TASK_IDS,
    "lean_gpu_mixed_transfer_v7": LEAN_GPU_MIXED_TRANSFER_V7_TASK_IDS,
    "lean-gpu-mixed-transfer-v7": LEAN_GPU_MIXED_TRANSFER_V7_TASK_IDS,
    "lean_gpu_mixed_transfer_v8": LEAN_GPU_MIXED_TRANSFER_V8_TASK_IDS,
    "lean-gpu-mixed-transfer-v8": LEAN_GPU_MIXED_TRANSFER_V8_TASK_IDS,
    "lean_gpu_mixed_transfer_v9": LEAN_GPU_MIXED_TRANSFER_V9_TASK_IDS,
    "lean-gpu-mixed-transfer-v9": LEAN_GPU_MIXED_TRANSFER_V9_TASK_IDS,
    "lean_gpu_mixed_transfer_v10": LEAN_GPU_MIXED_TRANSFER_V10_TASK_IDS,
    "lean-gpu-mixed-transfer-v10": LEAN_GPU_MIXED_TRANSFER_V10_TASK_IDS,
    "lean_gpu_mixed_transfer_v11": LEAN_GPU_MIXED_TRANSFER_V11_TASK_IDS,
    "lean-gpu-mixed-transfer-v11": LEAN_GPU_MIXED_TRANSFER_V11_TASK_IDS,
    "lean_gpu_mixed_transfer_v12": LEAN_GPU_MIXED_TRANSFER_V12_TASK_IDS,
    "lean-gpu-mixed-transfer-v12": LEAN_GPU_MIXED_TRANSFER_V12_TASK_IDS,
    "lean_gpu_mixed_transfer_v13": LEAN_GPU_MIXED_TRANSFER_V13_TASK_IDS,
    "lean-gpu-mixed-transfer-v13": LEAN_GPU_MIXED_TRANSFER_V13_TASK_IDS,
    "lean_gpu_mixed_transfer_v14": LEAN_GPU_MIXED_TRANSFER_V14_TASK_IDS,
    "lean-gpu-mixed-transfer-v14": LEAN_GPU_MIXED_TRANSFER_V14_TASK_IDS,
    "lean_gpu_mixed_transfer_v15": LEAN_GPU_MIXED_TRANSFER_V15_TASK_IDS,
    "lean-gpu-mixed-transfer-v15": LEAN_GPU_MIXED_TRANSFER_V15_TASK_IDS,
    "lean_gpu_mixed_transfer_v16": LEAN_GPU_MIXED_TRANSFER_V16_TASK_IDS,
    "lean-gpu-mixed-transfer-v16": LEAN_GPU_MIXED_TRANSFER_V16_TASK_IDS,
    "lean_gpu_mixed_transfer_v17": LEAN_GPU_MIXED_TRANSFER_V17_TASK_IDS,
    "lean-gpu-mixed-transfer-v17": LEAN_GPU_MIXED_TRANSFER_V17_TASK_IDS,
    "lean_gpu_mixed_transfer_v18": LEAN_GPU_MIXED_TRANSFER_V18_TASK_IDS,
    "lean-gpu-mixed-transfer-v18": LEAN_GPU_MIXED_TRANSFER_V18_TASK_IDS,
}
REPAIR_PHASE_BUDGET_CAPS = {
    "floor": 9,
    "cheap": 10,
    "mid": 20,
    "frontier": 31,
}
REPAIR_PHASE_BUDGET_MODES = ("custom", *REPAIR_PHASE_BUDGET_CAPS)
RESULT_IDENTITY_VOLATILE_KEYS = frozenset(
    {
        "created_at",
        "repair_denoise_skeleton_max_step",
        "repair_phase_budget",
        "repair_spend_gate_rows",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="experiments/general_reasoning_tasks_scout.jsonl")
    parser.add_argument("--families", default="planning", help="Comma-separated families or 'all'.")
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids to run.")
    parser.add_argument(
        "--task-preset",
        choices=sorted(TASK_PRESETS),
        default=None,
        help="Named task suite. Current compact GPU preset: lean_gpu_mixed.",
    )
    parser.add_argument(
        "--candidates",
        default="dream-7b-instruct-hf,llada-8b-instruct-hf",
        help="Comma-separated candidate keys.",
    )
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--limit-schedules", type=int, default=None)
    parser.add_argument(
        "--history-sample-count",
        type=int,
        default=None,
        help=(
            "Override schedule history sampling for fresh generations. Use a "
            "value near the denoise step count for transient-answer probes."
        ),
    )
    parser.add_argument(
        "--limit-evolved-schedules",
        type=int,
        default=2,
        help="Number of mutated schedules added for the evolved arm. Use 0 to disable.",
    )
    parser.add_argument(
        "--include-revision-schedules",
        action="store_true",
        help=(
            "Add non-monotonic LLaDA revision schedules that remask weak "
            "committed tokens and continue denoising inside the same trajectory."
        ),
    )
    parser.add_argument(
        "--revision-remask-fraction",
        type=float,
        default=0.25,
        help="Fraction of committed suffix tokens remasked by revision schedules.",
    )
    parser.add_argument(
        "--revision-steps",
        type=int,
        default=16,
        help="Additional denoising steps after revision remasking.",
    )
    parser.add_argument(
        "--evolved-promotion-margin",
        type=float,
        default=0.015,
        help=(
            "Minimum selector-score edge required for the evolved arm to "
            "replace the base trajectory-selected schedule."
        ),
    )
    parser.add_argument(
        "--revision-promotion-margin",
        type=float,
        default=DEFAULT_REVISION_PROMOTION_MARGIN,
        help=(
            "Minimum selector/planning-quality edge required before an evolved "
            "non-monotonic revision schedule can replace the base trajectory."
        ),
    )
    parser.add_argument(
        "--evolved-selector",
        choices=("inherit", "planning_quality_fallback"),
        default=DEFAULT_EVOLVED_SELECTOR,
        help=(
            "Selector used only for the evolved arm. 'inherit' uses the main "
            "trajectory selector; 'planning_quality_fallback' can promote a "
            "near-tie evolved schedule when final planning quality also improves."
        ),
    )
    parser.add_argument(
        "--evolved-quality-margin",
        type=float,
        default=DEFAULT_EVOLVED_QUALITY_MARGIN,
        help="Final planning-quality edge required by the evolved planning-quality fallback.",
    )
    parser.add_argument(
        "--evolved-selector-tolerance",
        type=float,
        default=DEFAULT_EVOLVED_SELECTOR_TOLERANCE,
        help="How far below the selected trajectory score an evolved fallback candidate may be.",
    )
    parser.add_argument(
        "--limit-repair-candidates",
        type=int,
        default=0,
        help=(
            "Number of LLaDA suffix-inpainting repairs to generate from the "
            "configured repair source. Use 0 to disable repair arm."
        ),
    )
    parser.add_argument(
        "--repair-source-policy",
        choices=REPAIR_SOURCE_POLICIES,
        default="evolved",
        help=(
            "Which selected source should seed LLaDA repair candidates. "
            "'evolved' preserves old behavior; 'non_revision_evolved' lets "
            "non-monotonic revision schedules win the evolved arm without "
            "forcing repairs to branch from the revised text; "
            "'random' forces the stable random perturbation source, useful for "
            "source-divergence stress tests against a separate trajectory; "
            "'evolved_and_trajectory' spends repairs from both the evolved "
            "winner and the base trajectory source; "
            "'non_revision_plus_gap_trajectory' starts from the non-revision "
            "source and adds a low-confidence trajectory source only when it "
            "still has a prompt-gap repair surface."
        ),
    )
    parser.add_argument(
        "--adaptive-source-gate-mode",
        choices=ADAPTIVE_SOURCE_GATE_MODES,
        default="custom",
        help=(
            "Named threshold preset for non_revision_plus_gap_trajectory. "
            "'score_max' uses the fresh-confirmed best-score gate; "
            "'efficiency' uses the fresh-confirmed plan_002-only budget gate; "
            "'score_efficient' keeps score-max's useful low-quality second source "
            "while skipping high-quality no-op trajectory sources; "
            "'custom' uses the explicit threshold arguments."
        ),
    )
    parser.add_argument(
        "--adaptive-source-gap-min-terms",
        type=int,
        default=DEFAULT_ADAPTIVE_SOURCE_GAP_MIN_TERMS,
        help=(
            "Minimum number of prompt terms missing from the trajectory source "
            "before non_revision_plus_gap_trajectory spends the extra source."
        ),
    )
    parser.add_argument(
        "--adaptive-source-quality-ceiling",
        type=float,
        default=DEFAULT_ADAPTIVE_SOURCE_QUALITY_CEILING,
        help=(
            "Optional maximum label-free planning quality for the trajectory source. "
            "This prevents spending an extra repair source when the trajectory output "
            "is already high-quality enough that the branch is usually a no-op."
        ),
    )
    parser.add_argument(
        "--adaptive-source-quality-floor",
        type=float,
        default=DEFAULT_ADAPTIVE_SOURCE_QUALITY_FLOOR,
        help=(
            "Minimum label-free planning quality for the trajectory source "
            "before non_revision_plus_gap_trajectory spends the extra source."
        ),
    )
    parser.add_argument(
        "--repair-pack",
        choices=(
            "prefix",
            "source_relative",
            "targeted_content",
            "prompt_guided",
            "constraint_gap",
            "constraint_span",
            "constraint_span_anchor_select",
            "constraint_span_phase_anchor",
            "constraint_span_phase_hybrid_preserve_seeded_gated",
            "constraint_span_phase_final_preserve_seeded_gated",
            "constraint_span_anchor_instability",
            "constraint_span_anchor_instability_gated",
            "constraint_span_anchor_instability_claim_gated",
            "constraint_span_anchor_instability_claim_oracle_gated",
            "constraint_span_anchor_instability_claim_seeded_gated",
            "constraint_span_anchor_instability_claim_compatible_seeded_gated",
            "constraint_span_anchor_instability_claim_auto_seeded_gated",
            "constraint_span_anchor_instability_claim_auto_action_seeded_gated",
            "constraint_span_anchor_instability_claim_auto_compat_seeded_gated",
            "constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated",
            "constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated",
            "constraint_span_anchor_instability_claim_auto_joint_seeded_gated",
            "constraint_span_anchor_instability_claim_auto_seeded_realization_gated",
            "constraint_span_anchor_instability_claim_strict_gated",
            "constraint_span_anchor_instability_prompt_only_gated",
            "constraint_span_anchor_instability_prompt_gated",
            "constraint_span_anchor_search",
            "constraint_span_history",
            "constraint_span_history_contrast",
            "constraint_span_history_instability",
            "constraint_span_clause",
            "state_adaptive",
            "replay_consistency",
        ),
        default="prefix",
        help=(
            "Repair candidate ordering. 'prefix' preserves the original broad "
            "suffix-inpainting pack; 'source_relative' prioritizes minimal "
            "low-confidence remasks before broader rewrites; "
            "'targeted_content' remasks low-value filler/repetition spans; "
            "'prompt_guided' rewrites a source draft under a generic critique; "
            "'constraint_gap' rewrites a source draft against missing prompt terms; "
            "'constraint_span' spends only the prompt-gap span repair branch; "
            "'constraint_span_anchor_select' chooses final or history span anchoring before repair; "
            "'constraint_span_phase_anchor' repairs from the first safe denoise skeleton "
            "that already covers enough task constraints; "
            "'constraint_span_phase_hybrid_preserve_seeded_gated' keeps the promoted "
            "preservation-seeded repair but only switches to phase history when source "
            "geometry predicts an advantage; "
            "'constraint_span_phase_final_preserve_seeded_gated' keeps the same phase-gated "
            "repair controls but always preserves the final denoise state as the repair source; "
            "'constraint_span_anchor_instability' chooses final/history anchoring and also masks "
            "positions unstable across denoise history; "
            "'constraint_span_anchor_instability_gated' only adds instability masks for "
            "low-quality multi-span planning anchors; "
            "'constraint_span_anchor_instability_claim_gated' keeps that instability gate and also "
            "adds a public-claim confound-control repair prompt on matching planning tasks; "
            "'constraint_span_anchor_instability_claim_oracle_gated' keeps that claim gate but "
            "uses a compact oracle-vs-selected result separation instruction; "
            "'constraint_span_anchor_instability_claim_seeded_gated' also fixes the missing "
            "oracle-vs-selected control phrase inside the denoise seed; "
            "'constraint_span_anchor_instability_claim_compatible_seeded_gated' fixes a compact "
            "oracle/selected plus claim-survival phrase inside the same denoise seed; "
            "'constraint_span_anchor_instability_claim_auto_seeded_gated' synthesizes that compact "
            "control seed from the active task/rubric surface; "
            "'constraint_span_anchor_instability_claim_auto_action_seeded_gated' synthesizes an "
            "action-bearing compact seed from the same surface; "
            "'constraint_span_anchor_instability_claim_auto_compat_seeded_gated' scores compact "
            "seed candidates for required-control compatibility; "
            "'constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated' keeps that "
            "compatibility scorer but asks for direct public-claim preservation without seed/anchor "
            "wording; "
            "'constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated' keeps that "
            "compatibility scorer but removes seed/anchor meta language from the repair prompt; "
            "'constraint_span_anchor_instability_claim_auto_joint_seeded_gated' scores compact seeds for "
            "compatibility, expected realization, and selected/oracle semantic preservation; "
            "'constraint_span_anchor_instability_claim_auto_seeded_realization_gated' also constrains "
            "the generated sentence so explicit control words survive; "
            "'constraint_span_anchor_instability_claim_strict_gated' also forces explicit "
            "oracle/best-of separation in that public-claim gate; "
            "'constraint_span_anchor_instability_prompt_only_gated' only adds the instability-specific "
            "repair instruction when that gate is active; "
            "'constraint_span_anchor_instability_prompt_gated' also adds the instability-specific "
            "repair instruction only when that gate is active; "
            "'constraint_span_anchor_search' searches sampled denoise states for a retention-safe anchor; "
            "'constraint_span_history' spends the prompt-gap span branch from a sampled denoise state; "
            "'constraint_span_history_contrast' keeps final-source span repair but adds denoise-history evidence; "
            "'constraint_span_history_instability' keeps final-source span repair but also masks "
            "positions unstable across denoise history; "
            "'constraint_span_clause' diagnoses clause-level masks inside long planning sentences; "
            "'state_adaptive' scales final/history masks from source-state signals; "
            "'replay_consistency' remasks positions unstable across denoise history."
        ),
    )
    parser.add_argument(
        "--include-history-repairs",
        action="store_true",
        help=(
            "Prepend a repair candidate seeded from the best sampled denoise-history "
            "state instead of only from the final output."
        ),
    )
    parser.add_argument(
        "--history-repair-fractions",
        default="0.25",
        help=(
            "Comma-separated prefix fractions for history-seeded repairs. "
            "Used only when --include-history-repairs is set."
        ),
    )
    parser.add_argument(
        "--include-history-visible-repair",
        action="store_true",
        help=(
            "Add a repair candidate seeded from all visible tokens in the "
            "selected denoise-history state."
        ),
    )
    parser.add_argument(
        "--repair-spend-trigger",
        choices=(
            "always",
            "source_quality_or_short",
            "source_repairability_geometry",
            "denoise_phase_repairability",
            "denoise_phase_value_proxy",
            DECOMPOSED_FOUR_HEAD_SELECTOR_ID,
            DECOMPOSED_SPEND_TRANSFER_SELECTOR_ID,
            TRAJECTORY_RELATIVE_DECOMPOSED_SPEND_SELECTOR_ID,
            LEARNED_AVAILABILITY_PREDICTOR_SELECTOR_ID,
            CALIBRATED_AVAILABILITY_PREDICTOR_SELECTOR_ID,
            COUNTERFACTUAL_MICRO_PROBE_TRIGGER_ID,
        ),
        default="always",
        help=(
            "When to spend the primary repair pass. 'always' preserves the "
            "original behavior; 'source_quality_or_short' skips primary repairs "
            "when the selected source already has high label-free planning "
            "quality and enough visible text to look complete; "
            "'source_repairability_geometry' additionally requires the source "
            "prompt-gap count and prompt keyword coverage to sit inside a "
            "repairable geometry band; 'denoise_phase_repairability' also "
            "requires the sampled denoise history to expose a repairable "
            "constraint skeleton; 'denoise_phase_value_proxy' further applies "
            "the calibrated source-quality proxy from the budget-value audit; "
            f"'{DECOMPOSED_FOUR_HEAD_SELECTOR_ID}' exposes the fitted four-head "
            "selector policy as a runner trigger; "
            f"'{DECOMPOSED_SPEND_TRANSFER_SELECTOR_ID}' adds the fitted independent "
            "transfer source-task floor to the decomposed spend head; "
            f"'{TRAJECTORY_RELATIVE_DECOMPOSED_SPEND_SELECTOR_ID}' also blocks "
            "repairs whose source state is below the already selected trajectory; "
            f"'{LEARNED_AVAILABILITY_PREDICTOR_SELECTOR_ID}' uses the fitted v3 "
            "availability rule from DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md; "
            f"'{CALIBRATED_AVAILABILITY_PREDICTOR_SELECTOR_ID}' uses the v3/v4 "
            "calibrated availability boundary without the failed absolute "
            "source-quality ceiling; "
            f"'{COUNTERFACTUAL_MICRO_PROBE_TRIGGER_ID}' records a bounded "
            "counterfactual-probe observation in gate diagnostics but never "
            "authorizes full repair spend."
        ),
    )
    parser.add_argument(
        "--repair-source-quality-threshold",
        type=float,
        default=0.50,
        help="Source-quality threshold used by --repair-spend-trigger source_quality_or_short.",
    )
    parser.add_argument(
        "--repair-source-min-chars",
        type=int,
        default=320,
        help="Minimum source text length treated as complete by source_quality_or_short.",
    )
    parser.add_argument(
        "--repair-source-prompt-gap-min",
        type=int,
        default=0,
        help="Minimum missing prompt terms required by source_repairability_geometry.",
    )
    parser.add_argument(
        "--repair-source-prompt-gap-max",
        type=int,
        default=999,
        help="Maximum missing prompt terms allowed by source_repairability_geometry.",
    )
    parser.add_argument(
        "--repair-source-prompt-coverage-min",
        type=float,
        default=0.0,
        help="Minimum prompt keyword coverage required by source_repairability_geometry.",
    )
    parser.add_argument(
        "--repair-source-prompt-coverage-max",
        type=float,
        default=1.0,
        help="Maximum prompt keyword coverage allowed by source_repairability_geometry.",
    )
    parser.add_argument(
        "--repair-denoise-skeleton-max-step",
        type=int,
        default=0,
        help=(
            "Optional maximum sampled denoise step for the first repairable "
            "constraint skeleton when using denoise_phase_repairability. "
            "Use 0 to allow any step."
        ),
    )
    parser.add_argument(
        "--repair-value-proxy-source-quality-max",
        type=float,
        default=0.31,
        help=(
            "Maximum label-free source quality for --repair-spend-trigger "
            "denoise_phase_value_proxy. The current budget-value audit calibrates "
            "the public MoE mixed proxy near 0.301429; 0.31 is the stable CLI tier."
        ),
    )
    parser.add_argument(
        "--repair-transfer-source-task-min",
        type=float,
        default=DECOMPOSED_SPEND_TRANSFER_SOURCE_TASK_MIN,
        help=(
            "Minimum source task score for --repair-spend-trigger "
            f"{DECOMPOSED_SPEND_TRANSFER_SELECTOR_ID}. The current fitted transfer "
            "floor uses 0.295357 to preserve the low-margin plan_012 repair."
        ),
    )
    parser.add_argument(
        "--counterfactual-probe-mode",
        choices=COUNTERFACTUAL_MICRO_PROBE_MODES,
        default="triage",
        help=(
            "Diagnostic generation mode for --repair-spend-trigger "
            f"{COUNTERFACTUAL_MICRO_PROBE_TRIGGER_ID}. 'triage' generates measured "
            "probe records only when would_probe is true; 'all' generates shadow "
            "probe records for every selected repair source. Both modes keep "
            "should_run=false."
        ),
    )
    parser.add_argument(
        "--counterfactual-probe-policy",
        choices=COUNTERFACTUAL_MICRO_PROBE_POLICIES,
        default=COUNTERFACTUAL_MICRO_PROBE_POLICY_ID,
        help=(
            "Diagnostic prompt policy for --repair-spend-trigger "
            f"{COUNTERFACTUAL_MICRO_PROBE_TRIGGER_ID}. The default preserves the "
            "legacy prose micro-probe; strict_tomography_probe_v1 asks for fixed "
            "diagnostic slots plus an exact no-repair sentinel; "
            "key_value_tomography_probe_v2 removes placeholder exemplars and "
            "forbids generic slot values."
        ),
    )
    parser.add_argument(
        "--repair-phase-budget",
        choices=REPAIR_PHASE_BUDGET_MODES,
        default="custom",
        help=(
            "Named denoise phase-window budget for the current public MoE stack. "
            "floor=cap9, cheap=cap10, mid=cap20, frontier=cap31. "
            "Use custom with --repair-denoise-skeleton-max-step for manual caps."
        ),
    )
    parser.add_argument(
        "--phase-source-target-similarity-min",
        type=float,
        default=PHASE_SOURCE_TARGET_SIMILARITY_MIN,
        help=(
            "Minimum compact-span target similarity required before a safe "
            "denoise-history phase can become the phase-hybrid repair source."
        ),
    )
    parser.add_argument(
        "--phase-source-text-similarity-min",
        type=float,
        default=PHASE_SOURCE_TEXT_SIMILARITY_MIN,
        help=(
            "Minimum whole-text similarity required before a safe denoise-history "
            "phase can become the phase-hybrid repair source."
        ),
    )
    parser.add_argument(
        "--phase-source-history-char-ratio-min",
        type=float,
        default=PHASE_SOURCE_HISTORY_CHAR_RATIO_MIN,
        help=(
            "Minimum history/final visible-character ratio required before a "
            "safe denoise-history phase can become the phase-hybrid repair source."
        ),
    )
    parser.add_argument(
        "--repair-source-controls",
        default="",
        help="Optional comma-separated source controls eligible for the primary repair pass.",
    )
    parser.add_argument(
        "--history-rescue-fractions",
        default="",
        help=(
            "Comma-separated history-prefix fractions generated only after the "
            "primary repair pass keeps the evolved baseline."
        ),
    )
    parser.add_argument(
        "--history-rescue-visible",
        action="store_true",
        help=(
            "Add the all-visible history repair only during adaptive history rescue."
        ),
    )
    parser.add_argument(
        "--history-rescue-trigger",
        choices=("baseline", "selector_disagreement", "baseline_or_selector_disagreement"),
        default="baseline",
        help=(
            "When to spend adaptive history-rescue repairs. 'baseline' preserves "
            "the original behavior; 'selector_disagreement' spends rescue when "
            "the repair selector and trajectory selector prefer different "
            "generated repair candidates."
        ),
    )
    parser.add_argument(
        "--history-rescue-source-controls",
        default="",
        help=(
            "Optional comma-separated evolved/source controls eligible for "
            "history-rescue repairs."
        ),
    )
    parser.add_argument(
        "--prompt-guided-rescue-trigger",
        choices=(
            "off",
            "baseline",
            "source_quality",
            "baseline_or_source_quality",
            "selector_disagreement",
            "baseline_or_selector_disagreement",
        ),
        default="off",
        help=(
            "When to spend prompt-guided repair as a late adaptive rescue. "
            "The source-quality trigger uses label-free planning quality on "
            "the evolved source."
        ),
    )
    parser.add_argument(
        "--prompt-guided-rescue-limit",
        type=int,
        default=1,
        help="Maximum prompt-guided rescue candidates to generate per eligible task.",
    )
    parser.add_argument(
        "--prompt-guided-rescue-source-quality-threshold",
        type=float,
        default=0.45,
        help="Run source-quality prompt rescue when evolved-source planning quality is below this value.",
    )
    parser.add_argument(
        "--prompt-guided-rescue-source-controls",
        default="",
        help=(
            "Optional comma-separated evolved/source controls eligible for "
            "prompt-guided rescue repairs."
        ),
    )
    parser.add_argument(
        "--constraint-gap-rescue-trigger",
        choices=("off", "prompt_gap", "baseline_or_prompt_gap"),
        default="off",
        help=(
            "When to spend prompt-grounded constraint-gap repair as a late "
            "adaptive rescue. The prompt-gap trigger uses only source quality "
            "and missing prompt terms."
        ),
    )
    parser.add_argument(
        "--constraint-gap-rescue-limit",
        type=int,
        default=1,
        help="Maximum constraint-gap rescue candidates to generate per eligible task.",
    )
    parser.add_argument(
        "--constraint-gap-rescue-min-terms",
        type=int,
        default=6,
        help="Minimum missing prompt terms required by the prompt-gap rescue trigger.",
    )
    parser.add_argument(
        "--constraint-gap-rescue-source-quality-floor",
        type=float,
        default=0.40,
        help="Minimum evolved-source planning quality for prompt-gap rescue.",
    )
    parser.add_argument(
        "--constraint-gap-rescue-source-quality-ceiling",
        type=float,
        default=0.50,
        help="Maximum evolved-source planning quality for prompt-gap rescue.",
    )
    parser.add_argument(
        "--constraint-gap-rescue-source-controls",
        default="",
        help=(
            "Optional comma-separated evolved/source controls eligible for "
            "constraint-gap rescue repairs."
        ),
    )
    parser.add_argument(
        "--repair-promotion-margin",
        type=float,
        default=DEFAULT_REPAIR_PROMOTION_MARGIN,
        help=(
            "Minimum selector-score edge required for a repair candidate to "
            "replace the evolved schedule-selected output."
        ),
    )
    parser.add_argument(
        "--repair-selector",
        choices=REPAIR_SELECTORS,
        default=DEFAULT_REPAIR_SELECTOR,
        help=(
            "Selector used only for the repair arm. 'inherit' reuses the main "
            "trajectory selector; 'transfer_promotion_value' is the current "
            "named transfer-promotion policy alias for inherited planning-state "
            "selection; 'planning_quality' scores final planning quality without "
            "hidden rubric items; 'planning_quality_guarded' penalizes history "
            "repairs that preserve too much already-visible mid-state content; "
            "risk_guarded variants also penalize plans that contradict explicit "
            "prompt constraints; planning_quality_delta variants require "
            "label-free improvement over the source output; "
            "planning_quality_seed_realization_guarded also rewards natural "
            "integration of compact seed anchors while penalizing meta labels; "
            "candidate_aware_promotion_v1 is the named post-repair promotion "
            "policy currently backed by that seed-realization score; "
            "planning_quality_seed_objective_guarded also rewards preservation "
            "of selected/oracle and claim-survival seed semantics."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--generation-seed", type=int, default=20260521)
    parser.add_argument(
        "--exact-task-trajectory-policy",
        choices=("fixed", "trajectory", "proposal_history"),
        default="fixed",
        help=(
            "How trajectory/evolved arms handle exact-answer tasks. fixed keeps "
            "the first schedule; trajectory uses raw trajectory score; "
            "proposal_history may select a final or denoise-history state only "
            "when it matches a label-free prompt-derived proposal."
        ),
    )
    parser.add_argument(
        "--exact-self-repair",
        action="store_true",
        help=(
            "For failed LLaDA exact-answer tasks with no prompt-derived proposals, "
            "spend one label-free solve-again repair and promote only parseable "
            "answers that differ from the failed source."
        ),
    )
    parser.add_argument(
        "--exact-verifier-revision",
        action="store_true",
        help=(
            "For failed LLaDA exact-answer tasks, spend one verifier-guided "
            "answer-span inpainting repair before counterfactual prompt "
            "repairs. When paired with --exact-self-repair, constrained "
            "non-integer label-free tasks without prompt-derived proposals "
            "can also use the rejected answer span as the remask target."
        ),
    )
    parser.add_argument(
        "--trajectory-selector",
        choices=("generic", "planning_prompt", "planning_state", "planning_state_v2"),
        default="planning_state",
        help=(
            "Selector used for the trajectory arm. planning_state variants use "
            "sampled denoise states plus prompt-visible planning signals."
        ),
    )
    parser.add_argument(
        "--raw-output",
        default="eval_results/diffusion_language/three_arm_raw.jsonl",
    )
    parser.add_argument(
        "--reuse-raw-input",
        default=None,
        help="Reuse an existing raw JSONL generation file and only rerun arm selection/reporting.",
    )
    parser.add_argument(
        "--scores-output",
        default="eval_results/diffusion_language/three_arm_scores.json",
    )
    parser.add_argument(
        "--report-output",
        default="eval_results/diffusion_language/three_arm_report.md",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repair_denoise_skeleton_max_step = _resolve_repair_phase_budget(
        args.repair_phase_budget,
        args.repair_denoise_skeleton_max_step,
    )
    if args.history_sample_count is not None and args.history_sample_count < 0:
        raise SystemExit("--history-sample-count must be non-negative")
    if args.history_sample_count is None and _repair_pack_needs_dense_history(args.repair_pack):
        args.history_sample_count = 32
    if args.revision_remask_fraction <= 0.0 or args.revision_remask_fraction > 1.0:
        raise SystemExit("--revision-remask-fraction must be greater than 0 and at most 1")
    if args.revision_steps < 0:
        raise SystemExit("--revision-steps must be non-negative")
    if args.revision_promotion_margin < 0.0:
        raise SystemExit("--revision-promotion-margin must be non-negative")
    (
        args.adaptive_source_gap_min_terms,
        args.adaptive_source_quality_floor,
        args.adaptive_source_quality_ceiling,
    ) = (
        _resolve_adaptive_source_gate_mode(
            args.adaptive_source_gate_mode,
            gap_min_terms=args.adaptive_source_gap_min_terms,
            quality_floor=args.adaptive_source_quality_floor,
            quality_ceiling=args.adaptive_source_quality_ceiling,
        )
    )
    tasks = _select_tasks(args)
    if args.reuse_raw_input:
        return _rescore_raw_main(args, tasks)

    raw_output = Path(args.raw_output)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    if raw_output.exists():
        raw_output.unlink()

    all_records: list[dict[str, object]] = []
    arm_records: list[dict[str, object]] = []
    repair_spend_gate_rows: list[dict[str, object]] = []

    for candidate_key in _split_csv(args.candidates):
        candidate = get_candidate(candidate_key)
        backend = HFDiffusionBackend(
            candidate_key,
            device=args.device,
            dtype=args.dtype,
            model_path=_local_model_path(candidate_key),
        )
        base_schedules = tuple(_schedules_for_candidate(candidate.family))
        base_schedules = _with_history_sample_count(
            base_schedules,
            args.history_sample_count,
        )
        if args.limit_schedules is not None:
            base_schedules = base_schedules[: args.limit_schedules]
        if not base_schedules:
            raise SystemExit(f"No schedules selected for {candidate_key}.")
        evolved_schedules = _evolved_schedules_for_candidate(
            candidate.family,
            base_schedules,
            limit=max(0, args.limit_evolved_schedules),
        )
        if args.include_revision_schedules:
            evolved_schedules = (
                *evolved_schedules,
                *_revision_schedules_for_candidate(
                    candidate.family,
                    base_schedules,
                    revision_remask_fraction=args.revision_remask_fraction,
                    revision_steps=args.revision_steps,
                ),
            )
        schedules = (*base_schedules, *evolved_schedules)

        for task in tasks:
            task_records = []
            base_task_records = []
            for schedule in schedules:
                generation_seed = _stable_generation_seed(
                    args.generation_seed,
                    candidate_key,
                    task.task_id,
                    schedule.name,
                )
                _set_generation_seed(generation_seed)
                config = schedule.to_config()
                if task.max_new_tokens:
                    config = _replace_max_tokens(config, task.max_new_tokens)
                record = _generate_record(
                    backend,
                    task,
                    config=config,
                    schedule=schedule.to_dict(),
                    stage="candidate_generation",
                    generation_seed=generation_seed,
                )
                task_records.append(record)
                if schedule in base_schedules:
                    base_task_records.append(record)
                all_records.append(record)
                _append_jsonl(raw_output, record)
                _print_generation(record)

            selected = select_three_arm_records(
                base_task_records,
                seed=args.random_seed,
                candidate_key=candidate_key,
                task_id=task.task_id,
                task_prompt=task.prompt,
                task_answer_type=task.answer_type,
                exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                trajectory_selector=args.trajectory_selector,
            )
            for arm, record in selected.items():
                budget = len(base_task_records) if arm == "trajectory_selected" else 1
                arm_records.append(
                    _with_arm_metadata(
                        arm,
                        record,
                        budget,
                        _selection_reason(
                            arm,
                            task.answer_type,
                            args.exact_task_trajectory_policy,
                            args.trajectory_selector,
                            evolved_record=record,
                            baseline_record=selected["fixed"],
                        ),
                        _selection_score(record, task.prompt, task.answer_type, args.trajectory_selector),
                    )
                )
            evolved_record = selected["trajectory_selected"]
            if evolved_schedules:
                evolved_record = select_evolved_record(
                    task_records,
                    baseline_record=selected["trajectory_selected"],
                    task_prompt=task.prompt,
                    task_answer_type=task.answer_type,
                    exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                    trajectory_selector=args.trajectory_selector,
                    evolved_selector=args.evolved_selector,
                    evolved_quality_margin=args.evolved_quality_margin,
                    evolved_selector_tolerance=args.evolved_selector_tolerance,
                    promotion_margin=args.evolved_promotion_margin,
                    revision_promotion_margin=args.revision_promotion_margin,
                )
                arm_records.append(
                    _with_arm_metadata(
                        "evolved",
                        evolved_record,
                        len(task_records),
                        _selection_reason(
                            "evolved",
                            task.answer_type,
                            args.exact_task_trajectory_policy,
                            args.trajectory_selector,
                            evolved_record=evolved_record,
                            baseline_record=selected["trajectory_selected"],
                            promotion_margin=args.evolved_promotion_margin,
                            revision_promotion_margin=args.revision_promotion_margin,
                            evolved_selector=args.evolved_selector,
                        ),
                        _selection_score(
                            evolved_record,
                            task.prompt,
                            task.answer_type,
                            args.trajectory_selector,
                        ),
                    )
                )
            repair_records = []
            if _should_run_repairs(candidate.family, task, args.limit_repair_candidates):
                repair_source_records = _select_repair_source_records(
                    args.repair_source_policy,
                    selected_records=selected,
                    evolved_record=evolved_record,
                    candidate_records=task_records,
                    task_prompt=task.prompt,
                    task_answer_type=task.answer_type,
                    exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                    trajectory_selector=args.trajectory_selector,
                    evolved_selector=args.evolved_selector,
                    evolved_quality_margin=args.evolved_quality_margin,
                    evolved_selector_tolerance=args.evolved_selector_tolerance,
                    evolved_promotion_margin=args.evolved_promotion_margin,
                    revision_promotion_margin=args.revision_promotion_margin,
                    adaptive_source_gap_min_terms=args.adaptive_source_gap_min_terms,
                    adaptive_source_quality_floor=args.adaptive_source_quality_floor,
                    adaptive_source_quality_ceiling=args.adaptive_source_quality_ceiling,
                )
                repairs = _repair_candidates(
                    repair_pack=args.repair_pack,
                    include_history_repairs=args.include_history_repairs,
                    history_repair_fractions=_float_csv(args.history_repair_fractions),
                    include_history_visible_repair=args.include_history_visible_repair,
                    limit=args.limit_repair_candidates,
                )
                repair_gate_pairs = [
                    (
                        source_record,
                        _primary_repair_gate_diagnostics(
                            trigger=args.repair_spend_trigger,
                            source_record=source_record,
                            source_controls=_split_csv(args.repair_source_controls),
                            task_prompt=task.prompt,
                            task_answer_type=task.answer_type,
                            source_quality_threshold=args.repair_source_quality_threshold,
                            source_min_chars=args.repair_source_min_chars,
                            source_prompt_gap_min=args.repair_source_prompt_gap_min,
                            source_prompt_gap_max=args.repair_source_prompt_gap_max,
                            source_prompt_coverage_min=args.repair_source_prompt_coverage_min,
                            source_prompt_coverage_max=args.repair_source_prompt_coverage_max,
                            denoise_skeleton_max_step=_positive_int_or_none(
                                args.repair_denoise_skeleton_max_step
                            ),
                            value_proxy_source_quality_max=args.repair_value_proxy_source_quality_max,
                            transfer_source_task_min=args.repair_transfer_source_task_min,
                            trajectory_record=selected["trajectory_selected"],
                        ),
                    )
                    for source_record in repair_source_records
                ]
                if args.repair_spend_trigger == COUNTERFACTUAL_MICRO_PROBE_TRIGGER_ID:
                    for source_record, diagnostics in repair_gate_pairs:
                        if not _should_generate_counterfactual_probe_record(
                            diagnostics,
                            mode=args.counterfactual_probe_mode,
                        ):
                            continue
                        probe_record = _generate_counterfactual_micro_probe_record(
                            backend,
                            task,
                            source_record=source_record,
                            diagnostics=diagnostics,
                            generation_seed_base=args.generation_seed,
                            probe_policy=args.counterfactual_probe_policy,
                        )
                        diagnostics.update(
                            _measured_counterfactual_micro_probe_diagnostics(
                                probe_record,
                                source_record=source_record,
                                task_prompt=task.prompt,
                                diagnostics=diagnostics,
                            )
                        )
                        _append_jsonl(raw_output, probe_record)
                        _print_generation(probe_record)
                repair_spend_gate_rows.extend(
                    _repair_spend_gate_row(source_record, diagnostics)
                    for source_record, diagnostics in repair_gate_pairs
                )
                primary_repair_source_records = [
                    source_record
                    for source_record, diagnostics in repair_gate_pairs
                    if bool(diagnostics["should_run"])
                ]
                primary_repair_enabled = bool(primary_repair_source_records)
                for source_record in primary_repair_source_records:
                    repair_records.extend(
                        _generate_repair_records(
                            backend,
                            task,
                            source_record=source_record,
                            repairs=repairs,
                            generation_seed_base=args.generation_seed,
                            phase_source_history_char_ratio_min=args.phase_source_history_char_ratio_min,
                            phase_source_target_similarity_min=args.phase_source_target_similarity_min,
                            phase_source_text_similarity_min=args.phase_source_text_similarity_min,
                            raw_output=raw_output,
                            all_records=all_records,
                        )
                    )
                repair_pool = [evolved_record, *repair_records]
                repair_record = select_repair_record(
                    repair_pool,
                    baseline_record=evolved_record,
                    task_prompt=task.prompt,
                    task_answer_type=task.answer_type,
                    exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                    trajectory_selector=args.trajectory_selector,
                    repair_selector=args.repair_selector,
                    promotion_margin=args.repair_promotion_margin,
                )
                rescue_repairs = _history_rescue_candidates(
                    history_rescue_fractions=_float_csv(args.history_rescue_fractions),
                    include_history_rescue_visible=args.history_rescue_visible,
                    existing_repairs=repairs,
                )
                generated_rescue_repairs: tuple[Any, ...] = ()
                if primary_repair_enabled and _should_run_adaptive_history_rescue(
                    trigger=args.history_rescue_trigger,
                    selected_repair=repair_record,
                    baseline_record=evolved_record,
                    repair_pool=repair_pool,
                    source_controls=_split_csv(args.history_rescue_source_controls),
                    task_prompt=task.prompt,
                    task_answer_type=task.answer_type,
                    exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                    trajectory_selector=args.trajectory_selector,
                ):
                    rescue_records = _generate_repair_records(
                        backend,
                        task,
                        source_record=primary_repair_source_records[0],
                        repairs=rescue_repairs,
                        generation_seed_base=args.generation_seed,
                        phase_source_history_char_ratio_min=args.phase_source_history_char_ratio_min,
                        phase_source_target_similarity_min=args.phase_source_target_similarity_min,
                        phase_source_text_similarity_min=args.phase_source_text_similarity_min,
                        raw_output=raw_output,
                        all_records=all_records,
                    )
                    if rescue_records:
                        generated_rescue_repairs = rescue_repairs
                        repair_records.extend(rescue_records)
                        repair_pool = [evolved_record, *repair_records]
                        repair_record = select_repair_record(
                            repair_pool,
                            baseline_record=evolved_record,
                            task_prompt=task.prompt,
                            task_answer_type=task.answer_type,
                            exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                            trajectory_selector=args.trajectory_selector,
                            repair_selector=args.repair_selector,
                            promotion_margin=args.repair_promotion_margin,
                        )
                prompt_guided_rescue_repairs = _prompt_guided_rescue_candidates(
                    existing_repairs=(*repairs, *generated_rescue_repairs),
                    limit=args.prompt_guided_rescue_limit,
                )
                generated_prompt_guided_rescue_repairs: tuple[Any, ...] = ()
                if primary_repair_enabled and _should_run_prompt_guided_rescue(
                    trigger=args.prompt_guided_rescue_trigger,
                    selected_repair=repair_record,
                    baseline_record=evolved_record,
                    repair_pool=repair_pool,
                    source_controls=_split_csv(args.prompt_guided_rescue_source_controls),
                    task_prompt=task.prompt,
                    task_answer_type=task.answer_type,
                    exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                    trajectory_selector=args.trajectory_selector,
                    source_quality_threshold=args.prompt_guided_rescue_source_quality_threshold,
                ):
                    prompt_guided_rescue_records = _generate_repair_records(
                        backend,
                        task,
                        source_record=primary_repair_source_records[0],
                        repairs=prompt_guided_rescue_repairs,
                        generation_seed_base=args.generation_seed,
                        phase_source_history_char_ratio_min=args.phase_source_history_char_ratio_min,
                        phase_source_target_similarity_min=args.phase_source_target_similarity_min,
                        phase_source_text_similarity_min=args.phase_source_text_similarity_min,
                        raw_output=raw_output,
                        all_records=all_records,
                    )
                    if prompt_guided_rescue_records:
                        generated_prompt_guided_rescue_repairs = prompt_guided_rescue_repairs
                        repair_records.extend(prompt_guided_rescue_records)
                        repair_pool = [evolved_record, *repair_records]
                        repair_record = select_repair_record(
                            repair_pool,
                            baseline_record=evolved_record,
                            task_prompt=task.prompt,
                            task_answer_type=task.answer_type,
                            exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                            trajectory_selector=args.trajectory_selector,
                            repair_selector=args.repair_selector,
                            promotion_margin=args.repair_promotion_margin,
                        )
                constraint_gap_rescue_repairs = _constraint_gap_rescue_candidates(
                    existing_repairs=(
                        *repairs,
                        *generated_rescue_repairs,
                        *generated_prompt_guided_rescue_repairs,
                    ),
                    limit=args.constraint_gap_rescue_limit,
                )
                if primary_repair_enabled and _should_run_constraint_gap_rescue(
                    trigger=args.constraint_gap_rescue_trigger,
                    selected_repair=repair_record,
                    baseline_record=evolved_record,
                    source_controls=_split_csv(args.constraint_gap_rescue_source_controls),
                    task_prompt=task.prompt,
                    task_answer_type=task.answer_type,
                    min_terms=args.constraint_gap_rescue_min_terms,
                    source_quality_floor=args.constraint_gap_rescue_source_quality_floor,
                    source_quality_ceiling=args.constraint_gap_rescue_source_quality_ceiling,
                ):
                    constraint_gap_rescue_records = _generate_repair_records(
                        backend,
                        task,
                        source_record=primary_repair_source_records[0],
                        repairs=constraint_gap_rescue_repairs,
                        generation_seed_base=args.generation_seed,
                        phase_source_history_char_ratio_min=args.phase_source_history_char_ratio_min,
                        phase_source_target_similarity_min=args.phase_source_target_similarity_min,
                        phase_source_text_similarity_min=args.phase_source_text_similarity_min,
                        raw_output=raw_output,
                        all_records=all_records,
                    )
                    if constraint_gap_rescue_records:
                        repair_records.extend(constraint_gap_rescue_records)
                        repair_pool = [evolved_record, *repair_records]
                        repair_record = select_repair_record(
                            repair_pool,
                            baseline_record=evolved_record,
                            task_prompt=task.prompt,
                            task_answer_type=task.answer_type,
                            exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                            trajectory_selector=args.trajectory_selector,
                            repair_selector=args.repair_selector,
                            promotion_margin=args.repair_promotion_margin,
                        )
                arm_records.append(
                    _with_arm_metadata(
                        "repair_selected",
                        repair_record,
                        len(task_records) + len(repair_records),
                        _primary_repair_selection_reason(args.repair_spend_trigger)
                        if not primary_repair_enabled
                        else _selection_reason(
                            "repair_selected",
                            task.answer_type,
                            args.exact_task_trajectory_policy,
                            args.trajectory_selector,
                            evolved_record=repair_record,
                            baseline_record=evolved_record,
                            promotion_margin=args.repair_promotion_margin,
                            repair_selector=args.repair_selector,
                        ),
                        _repair_selection_score(
                            repair_record,
                            baseline_record=evolved_record,
                            task_prompt=task.prompt,
                            task_answer_type=task.answer_type,
                            trajectory_selector=args.trajectory_selector,
                            repair_selector=args.repair_selector,
                        ),
                        _repair_selection_score(
                            evolved_record,
                            baseline_record=evolved_record,
                            task_prompt=task.prompt,
                            task_answer_type=task.answer_type,
                            trajectory_selector=args.trajectory_selector,
                            repair_selector=args.repair_selector,
                        ),
                    )
                )
            elif _should_run_exact_answer_repairs(
                candidate.family,
                task,
                args.limit_repair_candidates,
                evolved_record,
                exact_self_repair=args.exact_self_repair,
                exact_verifier_revision=args.exact_verifier_revision,
            ):
                exact_repair_records = _generate_exact_answer_repair_records(
                    backend,
                    task,
                    source_record=evolved_record,
                    limit=args.limit_repair_candidates,
                    exact_self_repair=args.exact_self_repair,
                    exact_verifier_revision=args.exact_verifier_revision,
                    generation_seed_base=args.generation_seed,
                    raw_output=raw_output,
                    all_records=all_records,
                )
                if exact_repair_records:
                    repair_pool = [evolved_record, *exact_repair_records]
                    repair_record = select_repair_record(
                        repair_pool,
                        baseline_record=evolved_record,
                        task_prompt=task.prompt,
                        task_answer_type=task.answer_type,
                        exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                        trajectory_selector=args.trajectory_selector,
                        repair_selector=args.repair_selector,
                        promotion_margin=args.repair_promotion_margin,
                    )
                    arm_records.append(
                        _with_arm_metadata(
                            "repair_selected",
                            repair_record,
                            len(task_records) + len(exact_repair_records),
                            _selection_reason(
                                "repair_selected",
                                task.answer_type,
                                args.exact_task_trajectory_policy,
                                args.trajectory_selector,
                                evolved_record=repair_record,
                                baseline_record=evolved_record,
                                promotion_margin=args.repair_promotion_margin,
                                repair_selector=args.repair_selector,
                            ),
                            _repair_selection_score(
                                repair_record,
                                baseline_record=evolved_record,
                                task_prompt=task.prompt,
                                task_answer_type=task.answer_type,
                                trajectory_selector=args.trajectory_selector,
                                repair_selector=args.repair_selector,
                            ),
                            _repair_selection_score(
                                evolved_record,
                                baseline_record=evolved_record,
                                task_prompt=task.prompt,
                                task_answer_type=task.answer_type,
                                trajectory_selector=args.trajectory_selector,
                                repair_selector=args.repair_selector,
                            ),
                        )
                    )

        _release_backend(backend)

    scores = summarize_three_arm_scores(
        all_records,
        arm_records,
        exact_task_trajectory_policy=args.exact_task_trajectory_policy,
        trajectory_selector=args.trajectory_selector,
        evolved_selector=args.evolved_selector,
        evolved_quality_margin=args.evolved_quality_margin,
        evolved_selector_tolerance=args.evolved_selector_tolerance,
        evolved_promotion_margin=args.evolved_promotion_margin,
        revision_promotion_margin=args.revision_promotion_margin,
        adaptive_source_gate_mode=args.adaptive_source_gate_mode,
        adaptive_source_gap_min_terms=args.adaptive_source_gap_min_terms,
        adaptive_source_quality_floor=args.adaptive_source_quality_floor,
        adaptive_source_quality_ceiling=args.adaptive_source_quality_ceiling,
        include_revision_schedules=args.include_revision_schedules,
        revision_remask_fraction=args.revision_remask_fraction,
        revision_steps=args.revision_steps,
        include_history_repairs=args.include_history_repairs,
        repair_pack=args.repair_pack,
        repair_source_policy=args.repair_source_policy,
        history_repair_fractions=_float_csv(args.history_repair_fractions),
        include_history_visible_repair=args.include_history_visible_repair,
        repair_spend_trigger=args.repair_spend_trigger,
        repair_source_quality_threshold=args.repair_source_quality_threshold,
        repair_source_min_chars=args.repair_source_min_chars,
        repair_source_prompt_gap_min=args.repair_source_prompt_gap_min,
        repair_source_prompt_gap_max=args.repair_source_prompt_gap_max,
        repair_source_prompt_coverage_min=args.repair_source_prompt_coverage_min,
        repair_source_prompt_coverage_max=args.repair_source_prompt_coverage_max,
        counterfactual_probe_mode=args.counterfactual_probe_mode,
        counterfactual_probe_policy=args.counterfactual_probe_policy,
        repair_value_proxy_source_quality_max=args.repair_value_proxy_source_quality_max,
        repair_transfer_source_task_min=args.repair_transfer_source_task_min,
        repair_phase_budget=args.repair_phase_budget,
        repair_denoise_skeleton_max_step=_positive_int_or_none(
            args.repair_denoise_skeleton_max_step
        ),
        phase_source_history_char_ratio_min=args.phase_source_history_char_ratio_min,
        phase_source_target_similarity_min=args.phase_source_target_similarity_min,
        phase_source_text_similarity_min=args.phase_source_text_similarity_min,
        repair_source_controls=_split_csv(args.repair_source_controls),
        history_rescue_fractions=_float_csv(args.history_rescue_fractions),
        history_rescue_visible=args.history_rescue_visible,
        history_rescue_trigger=args.history_rescue_trigger,
        history_rescue_source_controls=_split_csv(args.history_rescue_source_controls),
        prompt_guided_rescue_trigger=args.prompt_guided_rescue_trigger,
        prompt_guided_rescue_limit=args.prompt_guided_rescue_limit,
        prompt_guided_rescue_source_quality_threshold=args.prompt_guided_rescue_source_quality_threshold,
        prompt_guided_rescue_source_controls=_split_csv(args.prompt_guided_rescue_source_controls),
        constraint_gap_rescue_trigger=args.constraint_gap_rescue_trigger,
        constraint_gap_rescue_limit=args.constraint_gap_rescue_limit,
        constraint_gap_rescue_min_terms=args.constraint_gap_rescue_min_terms,
        constraint_gap_rescue_source_quality_floor=args.constraint_gap_rescue_source_quality_floor,
        constraint_gap_rescue_source_quality_ceiling=args.constraint_gap_rescue_source_quality_ceiling,
        constraint_gap_rescue_source_controls=_split_csv(args.constraint_gap_rescue_source_controls),
        repair_promotion_margin=args.repair_promotion_margin,
        repair_selector=args.repair_selector,
        exact_verifier_revision=args.exact_verifier_revision,
        repair_spend_gate_rows=repair_spend_gate_rows,
    )
    Path(args.scores_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scores_output).write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(render_report(scores), encoding="utf-8")
    print(json.dumps({"raw": args.raw_output, "scores": args.scores_output, "report": args.report_output}, indent=2))
    return 0


def _resolve_adaptive_source_gate_mode(
    mode: str,
    *,
    gap_min_terms: int,
    quality_floor: float,
    quality_ceiling: float | None = DEFAULT_ADAPTIVE_SOURCE_QUALITY_CEILING,
) -> tuple[int, float, float | None]:
    if mode not in ADAPTIVE_SOURCE_GATE_MODES:
        raise SystemExit(f"Unsupported adaptive source gate mode: {mode}")
    if mode != "custom":
        gap_min_terms, quality_floor, quality_ceiling = ADAPTIVE_SOURCE_GATE_MODE_DEFAULTS[mode]
    if gap_min_terms < 0:
        raise SystemExit("--adaptive-source-gap-min-terms must be non-negative")
    if quality_floor < 0.0:
        raise SystemExit("--adaptive-source-quality-floor must be non-negative")
    if quality_ceiling is not None:
        if quality_ceiling < 0.0:
            raise SystemExit("--adaptive-source-quality-ceiling must be non-negative")
        if quality_ceiling < quality_floor:
            raise SystemExit("--adaptive-source-quality-ceiling must be >= quality floor")
    return gap_min_terms, quality_floor, quality_ceiling


def _resolve_repair_phase_budget(mode: str, repair_denoise_skeleton_max_step: int) -> int:
    if mode == "custom":
        return repair_denoise_skeleton_max_step
    if repair_denoise_skeleton_max_step > 0:
        raise SystemExit(
            "--repair-phase-budget cannot be combined with --repair-denoise-skeleton-max-step"
        )
    try:
        return REPAIR_PHASE_BUDGET_CAPS[mode]
    except KeyError as exc:
        raise SystemExit(f"Unsupported repair phase budget: {mode}") from exc


def _rescore_raw_main(args: argparse.Namespace, tasks: list[GeneralReasoningTask]) -> int:
    raw_input = Path(args.reuse_raw_input)
    if not raw_input.exists():
        raise SystemExit(f"Raw input does not exist: {raw_input}")
    selected_candidates = set(_split_csv(args.candidates))
    selected_task_ids = {task.task_id for task in tasks}
    task_by_id = {task.task_id: task for task in tasks}
    all_records = [
        record
        for record in _read_jsonl(raw_input)
        if str(record.get("candidate_key")) in selected_candidates
        and _task_id(record) in selected_task_ids
    ]
    for record in all_records:
        task = task_by_id.get(_task_id(record))
        if task is not None:
            _attach_planning_quality_score(record, task)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in all_records:
        grouped[(str(record["candidate_key"]), _task_id(record))].append(record)

    arm_records: list[dict[str, object]] = []
    budgeted_records: list[dict[str, object]] = []
    repair_spend_gate_rows: list[dict[str, object]] = []
    for (candidate_key, task_id), task_records in sorted(grouped.items()):
        task = task_by_id[task_id]
        base_task_records = _limit_records_by_schedule(
            [record for record in task_records if _is_base_schedule_record(record)],
            args.limit_schedules,
        )
        evolved_task_records = _selected_evolved_records_for_rescore(
            task_records,
            limit_evolved_schedules=max(0, args.limit_evolved_schedules),
            include_revision_schedules=args.include_revision_schedules,
        )
        repair_task_records = [record for record in task_records if _is_repair_record(record)]
        if not base_task_records:
            continue
        budgeted_records.extend(base_task_records)
        budgeted_records.extend(evolved_task_records)
        selected = select_three_arm_records(
            base_task_records,
            seed=args.random_seed,
            candidate_key=candidate_key,
            task_id=task.task_id,
            task_prompt=task.prompt,
            task_answer_type=task.answer_type,
            exact_task_trajectory_policy=args.exact_task_trajectory_policy,
            trajectory_selector=args.trajectory_selector,
        )
        for arm, record in selected.items():
            budget = len(base_task_records) if arm == "trajectory_selected" else 1
            arm_records.append(
                _with_arm_metadata(
                    arm,
                    record,
                    budget,
                    _selection_reason(
                        arm,
                        task.answer_type,
                        args.exact_task_trajectory_policy,
                        args.trajectory_selector,
                        evolved_record=record,
                        baseline_record=selected["fixed"],
                    ),
                    _selection_score(record, task.prompt, task.answer_type, args.trajectory_selector),
                )
            )
        if evolved_task_records:
            evolved_pool = [*base_task_records, *evolved_task_records]
            evolved_record = select_evolved_record(
                evolved_pool,
                baseline_record=selected["trajectory_selected"],
                task_prompt=task.prompt,
                task_answer_type=task.answer_type,
                exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                trajectory_selector=args.trajectory_selector,
                evolved_selector=args.evolved_selector,
                evolved_quality_margin=args.evolved_quality_margin,
                evolved_selector_tolerance=args.evolved_selector_tolerance,
                promotion_margin=args.evolved_promotion_margin,
                revision_promotion_margin=args.revision_promotion_margin,
            )
            arm_records.append(
                _with_arm_metadata(
                    "evolved",
                    evolved_record,
                    len(evolved_pool),
                    _selection_reason(
                        "evolved",
                        task.answer_type,
                        args.exact_task_trajectory_policy,
                        args.trajectory_selector,
                        evolved_record=evolved_record,
                        baseline_record=selected["trajectory_selected"],
                        promotion_margin=args.evolved_promotion_margin,
                        revision_promotion_margin=args.revision_promotion_margin,
                        evolved_selector=args.evolved_selector,
                    ),
                    _selection_score(evolved_record, task.prompt, task.answer_type, args.trajectory_selector),
                )
            )
        else:
            evolved_record = selected["trajectory_selected"]
        repair_source_records = _select_repair_source_records(
            args.repair_source_policy,
            selected_records=selected,
            evolved_record=evolved_record,
            candidate_records=[*base_task_records, *evolved_task_records],
            task_prompt=task.prompt,
            task_answer_type=task.answer_type,
            exact_task_trajectory_policy=args.exact_task_trajectory_policy,
            trajectory_selector=args.trajectory_selector,
            evolved_selector=args.evolved_selector,
            evolved_quality_margin=args.evolved_quality_margin,
            evolved_selector_tolerance=args.evolved_selector_tolerance,
            evolved_promotion_margin=args.evolved_promotion_margin,
            revision_promotion_margin=args.revision_promotion_margin,
            adaptive_source_gap_min_terms=args.adaptive_source_gap_min_terms,
            adaptive_source_quality_floor=args.adaptive_source_quality_floor,
            adaptive_source_quality_ceiling=args.adaptive_source_quality_ceiling,
        )
        if _should_run_exact_answer_repairs(
            get_candidate(candidate_key).family,
            task,
            args.limit_repair_candidates,
            evolved_record,
            exact_self_repair=args.exact_self_repair,
            exact_verifier_revision=args.exact_verifier_revision,
        ):
            compatible_exact_repair_records = _exact_answer_repair_records_for_source(
                repair_task_records,
                evolved_record,
                limit=args.limit_repair_candidates,
                exact_verifier_revision=args.exact_verifier_revision,
            )
            if compatible_exact_repair_records:
                budgeted_records.extend(compatible_exact_repair_records)
                repair_pool = [evolved_record, *compatible_exact_repair_records]
                repair_record = select_repair_record(
                    repair_pool,
                    baseline_record=evolved_record,
                    task_prompt=task.prompt,
                    task_answer_type=task.answer_type,
                    exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                    trajectory_selector=args.trajectory_selector,
                    repair_selector=args.repair_selector,
                    promotion_margin=args.repair_promotion_margin,
                )
                arm_records.append(
                    _with_arm_metadata(
                        "repair_selected",
                        repair_record,
                        len(base_task_records) + len(evolved_task_records) + len(compatible_exact_repair_records),
                        _selection_reason(
                            "repair_selected",
                            task.answer_type,
                            args.exact_task_trajectory_policy,
                            args.trajectory_selector,
                            evolved_record=repair_record,
                            baseline_record=evolved_record,
                            promotion_margin=args.repair_promotion_margin,
                            repair_selector=args.repair_selector,
                        ),
                        _repair_selection_score(
                            repair_record,
                            baseline_record=evolved_record,
                            task_prompt=task.prompt,
                            task_answer_type=task.answer_type,
                            trajectory_selector=args.trajectory_selector,
                            repair_selector=args.repair_selector,
                        ),
                        _repair_selection_score(
                            evolved_record,
                            baseline_record=evolved_record,
                            task_prompt=task.prompt,
                            task_answer_type=task.answer_type,
                            trajectory_selector=args.trajectory_selector,
                            repair_selector=args.repair_selector,
                        ),
                    )
                )
            continue
        if not _should_run_repairs(get_candidate(candidate_key).family, task, args.limit_repair_candidates):
            continue
        repair_candidates = _repair_candidates(
            repair_pack=args.repair_pack,
            include_history_repairs=args.include_history_repairs,
            history_repair_fractions=_float_csv(args.history_repair_fractions),
            include_history_visible_repair=args.include_history_visible_repair,
            limit=args.limit_repair_candidates,
        )
        repair_gate_pairs = [
            (
                source_record,
                _primary_repair_gate_diagnostics(
                    trigger=args.repair_spend_trigger,
                    source_record=source_record,
                    source_controls=_split_csv(args.repair_source_controls),
                    task_prompt=task.prompt,
                    task_answer_type=task.answer_type,
                    source_quality_threshold=args.repair_source_quality_threshold,
                    source_min_chars=args.repair_source_min_chars,
                    source_prompt_gap_min=args.repair_source_prompt_gap_min,
                    source_prompt_gap_max=args.repair_source_prompt_gap_max,
                    source_prompt_coverage_min=args.repair_source_prompt_coverage_min,
                    source_prompt_coverage_max=args.repair_source_prompt_coverage_max,
                    denoise_skeleton_max_step=_positive_int_or_none(
                        args.repair_denoise_skeleton_max_step
                    ),
                    value_proxy_source_quality_max=args.repair_value_proxy_source_quality_max,
                    transfer_source_task_min=args.repair_transfer_source_task_min,
                    trajectory_record=selected["trajectory_selected"],
                ),
            )
            for source_record in repair_source_records
        ]
        repair_spend_gate_rows.extend(
            _repair_spend_gate_row(source_record, diagnostics)
            for source_record, diagnostics in repair_gate_pairs
        )
        primary_repair_source_records = [
            source_record for source_record, diagnostics in repair_gate_pairs if bool(diagnostics["should_run"])
        ]
        primary_repair_enabled = bool(primary_repair_source_records)
        compatible_repair_task_records = []
        if primary_repair_enabled:
            compatible_repair_task_records = _repair_records_for_sources(
                _repair_records_for_candidates(repair_task_records, repair_candidates),
                primary_repair_source_records,
            )
        if not primary_repair_enabled or compatible_repair_task_records:
            repair_pool = [evolved_record, *compatible_repair_task_records]
            repair_record = select_repair_record(
                repair_pool,
                baseline_record=evolved_record,
                task_prompt=task.prompt,
                task_answer_type=task.answer_type,
                exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                trajectory_selector=args.trajectory_selector,
                repair_selector=args.repair_selector,
                promotion_margin=args.repair_promotion_margin,
            )
            rescue_candidates = _history_rescue_candidates(
                history_rescue_fractions=_float_csv(args.history_rescue_fractions),
                include_history_rescue_visible=args.history_rescue_visible,
                existing_repairs=repair_candidates,
            )
            generated_rescue_candidates: tuple[Any, ...] = ()
            if primary_repair_enabled and _should_run_adaptive_history_rescue(
                trigger=args.history_rescue_trigger,
                selected_repair=repair_record,
                baseline_record=evolved_record,
                repair_pool=repair_pool,
                source_controls=_split_csv(args.history_rescue_source_controls),
                task_prompt=task.prompt,
                task_answer_type=task.answer_type,
                exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                trajectory_selector=args.trajectory_selector,
            ):
                rescue_records = _repair_records_for_source(
                    _repair_records_for_candidates(repair_task_records, rescue_candidates),
                    primary_repair_source_records[0],
                )
                if rescue_records:
                    generated_rescue_candidates = rescue_candidates
                    compatible_repair_task_records = [
                        *compatible_repair_task_records,
                        *rescue_records,
                    ]
                    repair_pool = [evolved_record, *compatible_repair_task_records]
                    repair_record = select_repair_record(
                        repair_pool,
                        baseline_record=evolved_record,
                        task_prompt=task.prompt,
                        task_answer_type=task.answer_type,
                        exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                        trajectory_selector=args.trajectory_selector,
                        repair_selector=args.repair_selector,
                        promotion_margin=args.repair_promotion_margin,
                )
            prompt_guided_rescue_candidates = _prompt_guided_rescue_candidates(
                existing_repairs=(*repair_candidates, *generated_rescue_candidates),
                limit=args.prompt_guided_rescue_limit,
            )
            generated_prompt_guided_rescue_candidates: tuple[Any, ...] = ()
            if primary_repair_enabled and _should_run_prompt_guided_rescue(
                trigger=args.prompt_guided_rescue_trigger,
                selected_repair=repair_record,
                baseline_record=evolved_record,
                repair_pool=repair_pool,
                source_controls=_split_csv(args.prompt_guided_rescue_source_controls),
                task_prompt=task.prompt,
                task_answer_type=task.answer_type,
                exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                trajectory_selector=args.trajectory_selector,
                source_quality_threshold=args.prompt_guided_rescue_source_quality_threshold,
            ):
                prompt_guided_rescue_records = _repair_records_for_source(
                    _repair_records_for_candidates(repair_task_records, prompt_guided_rescue_candidates),
                    primary_repair_source_records[0],
                )
                if prompt_guided_rescue_records:
                    generated_prompt_guided_rescue_candidates = prompt_guided_rescue_candidates
                    compatible_repair_task_records = [
                        *compatible_repair_task_records,
                        *prompt_guided_rescue_records,
                    ]
                    repair_pool = [evolved_record, *compatible_repair_task_records]
                    repair_record = select_repair_record(
                        repair_pool,
                        baseline_record=evolved_record,
                        task_prompt=task.prompt,
                        task_answer_type=task.answer_type,
                        exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                        trajectory_selector=args.trajectory_selector,
                        repair_selector=args.repair_selector,
                        promotion_margin=args.repair_promotion_margin,
                    )
            constraint_gap_rescue_candidates = _constraint_gap_rescue_candidates(
                existing_repairs=(
                    *repair_candidates,
                    *generated_rescue_candidates,
                    *generated_prompt_guided_rescue_candidates,
                ),
                limit=args.constraint_gap_rescue_limit,
            )
            if primary_repair_enabled and _should_run_constraint_gap_rescue(
                trigger=args.constraint_gap_rescue_trigger,
                selected_repair=repair_record,
                baseline_record=evolved_record,
                source_controls=_split_csv(args.constraint_gap_rescue_source_controls),
                task_prompt=task.prompt,
                task_answer_type=task.answer_type,
                min_terms=args.constraint_gap_rescue_min_terms,
                source_quality_floor=args.constraint_gap_rescue_source_quality_floor,
                source_quality_ceiling=args.constraint_gap_rescue_source_quality_ceiling,
            ):
                constraint_gap_rescue_records = _repair_records_for_source(
                    _repair_records_for_candidates(repair_task_records, constraint_gap_rescue_candidates),
                    primary_repair_source_records[0],
                )
                if constraint_gap_rescue_records:
                    compatible_repair_task_records = [
                        *compatible_repair_task_records,
                        *constraint_gap_rescue_records,
                    ]
                    repair_pool = [evolved_record, *compatible_repair_task_records]
                    repair_record = select_repair_record(
                        repair_pool,
                        baseline_record=evolved_record,
                        task_prompt=task.prompt,
                        task_answer_type=task.answer_type,
                        exact_task_trajectory_policy=args.exact_task_trajectory_policy,
                        trajectory_selector=args.trajectory_selector,
                        repair_selector=args.repair_selector,
                        promotion_margin=args.repair_promotion_margin,
                    )
            budgeted_records.extend(compatible_repair_task_records)
            arm_records.append(
                _with_arm_metadata(
                    "repair_selected",
                    repair_record,
                    len(base_task_records) + len(evolved_task_records) + len(compatible_repair_task_records),
                    _primary_repair_selection_reason(args.repair_spend_trigger)
                    if not primary_repair_enabled
                    else _selection_reason(
                        "repair_selected",
                        task.answer_type,
                        args.exact_task_trajectory_policy,
                        args.trajectory_selector,
                        evolved_record=repair_record,
                        baseline_record=evolved_record,
                        promotion_margin=args.repair_promotion_margin,
                        repair_selector=args.repair_selector,
                    ),
                    _repair_selection_score(
                        repair_record,
                        baseline_record=evolved_record,
                        task_prompt=task.prompt,
                        task_answer_type=task.answer_type,
                        trajectory_selector=args.trajectory_selector,
                        repair_selector=args.repair_selector,
                    ),
                    _repair_selection_score(
                        evolved_record,
                        baseline_record=evolved_record,
                        task_prompt=task.prompt,
                        task_answer_type=task.answer_type,
                        trajectory_selector=args.trajectory_selector,
                        repair_selector=args.repair_selector,
                    ),
                )
            )

    if not arm_records:
        raise SystemExit("No matching raw records were available for rescoring.")
    scores = summarize_three_arm_scores(
        budgeted_records,
        arm_records,
        exact_task_trajectory_policy=args.exact_task_trajectory_policy,
        trajectory_selector=args.trajectory_selector,
        evolved_selector=args.evolved_selector,
        evolved_quality_margin=args.evolved_quality_margin,
        evolved_selector_tolerance=args.evolved_selector_tolerance,
        evolved_promotion_margin=args.evolved_promotion_margin,
        revision_promotion_margin=args.revision_promotion_margin,
        adaptive_source_gate_mode=args.adaptive_source_gate_mode,
        adaptive_source_gap_min_terms=args.adaptive_source_gap_min_terms,
        adaptive_source_quality_floor=args.adaptive_source_quality_floor,
        adaptive_source_quality_ceiling=args.adaptive_source_quality_ceiling,
        include_revision_schedules=args.include_revision_schedules,
        revision_remask_fraction=args.revision_remask_fraction,
        revision_steps=args.revision_steps,
        include_history_repairs=args.include_history_repairs,
        repair_pack=args.repair_pack,
        repair_source_policy=args.repair_source_policy,
        history_repair_fractions=_float_csv(args.history_repair_fractions),
        include_history_visible_repair=args.include_history_visible_repair,
        repair_spend_trigger=args.repair_spend_trigger,
        repair_source_quality_threshold=args.repair_source_quality_threshold,
        repair_source_min_chars=args.repair_source_min_chars,
        repair_source_prompt_gap_min=args.repair_source_prompt_gap_min,
        repair_source_prompt_gap_max=args.repair_source_prompt_gap_max,
        repair_source_prompt_coverage_min=args.repair_source_prompt_coverage_min,
        repair_source_prompt_coverage_max=args.repair_source_prompt_coverage_max,
        counterfactual_probe_mode=args.counterfactual_probe_mode,
        counterfactual_probe_policy=args.counterfactual_probe_policy,
        repair_value_proxy_source_quality_max=args.repair_value_proxy_source_quality_max,
        repair_transfer_source_task_min=args.repair_transfer_source_task_min,
        repair_phase_budget=args.repair_phase_budget,
        repair_denoise_skeleton_max_step=_positive_int_or_none(
            args.repair_denoise_skeleton_max_step
        ),
        phase_source_history_char_ratio_min=args.phase_source_history_char_ratio_min,
        phase_source_target_similarity_min=args.phase_source_target_similarity_min,
        phase_source_text_similarity_min=args.phase_source_text_similarity_min,
        repair_source_controls=_split_csv(args.repair_source_controls),
        history_rescue_fractions=_float_csv(args.history_rescue_fractions),
        history_rescue_visible=args.history_rescue_visible,
        history_rescue_trigger=args.history_rescue_trigger,
        history_rescue_source_controls=_split_csv(args.history_rescue_source_controls),
        prompt_guided_rescue_trigger=args.prompt_guided_rescue_trigger,
        prompt_guided_rescue_limit=args.prompt_guided_rescue_limit,
        prompt_guided_rescue_source_quality_threshold=args.prompt_guided_rescue_source_quality_threshold,
        prompt_guided_rescue_source_controls=_split_csv(args.prompt_guided_rescue_source_controls),
        constraint_gap_rescue_trigger=args.constraint_gap_rescue_trigger,
        constraint_gap_rescue_limit=args.constraint_gap_rescue_limit,
        constraint_gap_rescue_min_terms=args.constraint_gap_rescue_min_terms,
        constraint_gap_rescue_source_quality_floor=args.constraint_gap_rescue_source_quality_floor,
        constraint_gap_rescue_source_quality_ceiling=args.constraint_gap_rescue_source_quality_ceiling,
        constraint_gap_rescue_source_controls=_split_csv(args.constraint_gap_rescue_source_controls),
        repair_promotion_margin=args.repair_promotion_margin,
        repair_selector=args.repair_selector,
        exact_verifier_revision=args.exact_verifier_revision,
        repair_spend_gate_rows=repair_spend_gate_rows,
    )
    Path(args.scores_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scores_output).write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(render_report(scores), encoding="utf-8")
    print(
        json.dumps(
            {"raw_input": args.reuse_raw_input, "scores": args.scores_output, "report": args.report_output},
            indent=2,
        )
    )
    return 0


def _select_tasks(args: argparse.Namespace) -> list[GeneralReasoningTask]:
    tasks = load_tasks(args.tasks)
    if args.task_preset:
        if args.task_ids:
            raise SystemExit("--task-preset and --task-ids cannot be used together")
        task_by_id = {task.task_id: task for task in tasks}
        preset_ids = TASK_PRESETS[args.task_preset]
        missing_ids = [task_id for task_id in preset_ids if task_id not in task_by_id]
        if missing_ids:
            missing = ",".join(missing_ids)
            raise SystemExit(f"Task preset {args.task_preset!r} missing task ids: {missing}")
        tasks = [task_by_id[task_id] for task_id in preset_ids]
        if args.limit_tasks is not None:
            tasks = tasks[: args.limit_tasks]
        return tasks
    families = None if args.families == "all" else set(_split_csv(args.families))
    if families is not None:
        tasks = [task for task in tasks if task.family in families]
    if args.task_ids:
        selected_ids = set(_split_csv(args.task_ids))
        tasks = [task for task in tasks if task.task_id in selected_ids]
    if args.limit_tasks is not None:
        tasks = tasks[: args.limit_tasks]
    if not tasks:
        raise SystemExit("No tasks selected.")
    return tasks


def _schedules_for_candidate(family: str):
    if is_llada_family(family):
        return default_llada_schedules(max_new_tokens=64)
    return default_dream_schedules(max_new_tokens=64)


def _with_history_sample_count(schedules: tuple[Any, ...], sample_count: int | None) -> tuple[Any, ...]:
    if sample_count is None:
        return schedules
    return tuple(replace(schedule, history_sample_count=sample_count) for schedule in schedules)


def _repair_pack_needs_dense_history(repair_pack: str) -> bool:
    return repair_pack in {
        "constraint_span_anchor_search",
        "constraint_span_anchor_select",
        "constraint_span_phase_anchor",
        "constraint_span_phase_hybrid_preserve_seeded_gated",
        "constraint_span_phase_final_preserve_seeded_gated",
        "constraint_span_anchor_instability",
        "constraint_span_anchor_instability_gated",
        "constraint_span_anchor_instability_claim_gated",
        "constraint_span_anchor_instability_claim_oracle_gated",
        "constraint_span_anchor_instability_claim_seeded_gated",
        "constraint_span_anchor_instability_claim_compatible_seeded_gated",
        "constraint_span_anchor_instability_claim_auto_seeded_gated",
        "constraint_span_anchor_instability_claim_auto_action_seeded_gated",
        "constraint_span_anchor_instability_claim_auto_compat_seeded_gated",
        "constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated",
        "constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated",
        "constraint_span_anchor_instability_claim_auto_joint_seeded_gated",
        "constraint_span_anchor_instability_claim_auto_seeded_realization_gated",
        "constraint_span_anchor_instability_claim_strict_gated",
        "constraint_span_anchor_instability_prompt_only_gated",
        "constraint_span_anchor_instability_prompt_gated",
        "constraint_span_history_contrast",
        "constraint_span_history_instability",
        "constraint_span_history",
    }


def _evolved_schedules_for_candidate(
    family: str,
    base_schedules: tuple[Any, ...],
    *,
    limit: int,
) -> tuple[Any, ...]:
    if limit <= 0 or not base_schedules:
        return ()
    by_name = {str(schedule.name): schedule for schedule in base_schedules}
    if is_llada_family(family):
        low_confidence = by_name.get("low_confidence_32", base_schedules[0])
        random_remask = by_name.get("random_32", low_confidence)
        mutations = (
            replace(low_confidence, name="evolved_low_confidence_48", steps=48),
            replace(random_remask, name="evolved_random_48", steps=48),
            replace(low_confidence, name="evolved_low_confidence_64", steps=64),
        )
    else:
        entropy_fast = by_name.get("entropy_32", base_schedules[0])
        entropy_deep = by_name.get("entropy_64", entropy_fast)
        origin = by_name.get("origin_64", entropy_deep)
        mutations = (
            replace(entropy_fast, name="evolved_entropy_48", steps=48),
            replace(entropy_deep, name="evolved_entropy_96", steps=96),
            replace(origin, name="evolved_origin_32", steps=32),
        )
    base_names = {str(schedule.name) for schedule in base_schedules}
    deduped = tuple(schedule for schedule in mutations if str(schedule.name) not in base_names)
    return deduped[:limit]


def _revision_schedules_for_candidate(
    family: str,
    base_schedules: tuple[Any, ...],
    *,
    revision_remask_fraction: float = 0.25,
    revision_steps: int = 16,
) -> tuple[Any, ...]:
    if not is_llada_family(family) or not base_schedules:
        return ()
    by_name = {str(schedule.name): schedule for schedule in base_schedules}
    low_confidence = by_name.get("low_confidence_32", base_schedules[0])
    random_remask = by_name.get("random_32", low_confidence)
    return (
        replace(
            low_confidence,
            name="evolved_revision_low_confidence_32",
            revision_remask_fraction=revision_remask_fraction,
            revision_steps=revision_steps,
        ),
        replace(
            random_remask,
            name="evolved_revision_random_32",
            revision_remask_fraction=revision_remask_fraction,
            revision_steps=revision_steps,
        ),
    )


def _replace_max_tokens(config: Any, max_new_tokens: int):
    block_length = config.block_length
    if config.algorithm == "low_confidence":
        block_length = max_new_tokens
    return replace(config, max_new_tokens=max_new_tokens, block_length=block_length)


def _should_run_repairs(family: str, task: GeneralReasoningTask, limit: int) -> bool:
    return limit > 0 and is_llada_family(family) and task.answer_type == "rubric"


def _should_run_exact_answer_repairs(
    family: str,
    task: GeneralReasoningTask,
    limit: int,
    source_record: dict[str, object],
    *,
    exact_self_repair: bool = False,
    exact_verifier_revision: bool = False,
) -> bool:
    source_answer = _nested_value(source_record, ("task_score", "extracted_answer"))
    proposals_available = bool(counterfactual_answer_proposals(task, source_answer, limit=limit))
    verifier_span_available = (
        exact_verifier_revision
        and exact_self_repair
        and bool(_int_list(source_record.get("generated_token_ids")))
        and source_answer is not None
        and _supports_no_proposal_answer_span_revision(task)
    )
    return (
        limit > 0
        and is_llada_family(family)
        and task.answer_type != "rubric"
        and _task_score(source_record) < 0.999
        and (
            proposals_available
            or verifier_span_available
            or (
                exact_self_repair
                and _label_free_exact_answer_supported(task)
            )
        )
    )


def _should_run_primary_repair_pass(
    *,
    trigger: str,
    source_record: dict[str, object],
    source_controls: list[str],
    task_prompt: str,
    task_answer_type: str,
    source_quality_threshold: float,
    source_min_chars: int,
    source_prompt_gap_min: int = 0,
    source_prompt_gap_max: int = 999,
    source_prompt_coverage_min: float = 0.0,
    source_prompt_coverage_max: float = 1.0,
    denoise_skeleton_max_step: int | None = None,
    value_proxy_source_quality_max: float = 0.31,
    transfer_source_task_min: float = DECOMPOSED_SPEND_TRANSFER_SOURCE_TASK_MIN,
    trajectory_record: dict[str, object] | None = None,
) -> bool:
    return bool(
        _primary_repair_gate_diagnostics(
            trigger=trigger,
            source_record=source_record,
            source_controls=source_controls,
            task_prompt=task_prompt,
            task_answer_type=task_answer_type,
            source_quality_threshold=source_quality_threshold,
            source_min_chars=source_min_chars,
            source_prompt_gap_min=source_prompt_gap_min,
            source_prompt_gap_max=source_prompt_gap_max,
            source_prompt_coverage_min=source_prompt_coverage_min,
            source_prompt_coverage_max=source_prompt_coverage_max,
            denoise_skeleton_max_step=denoise_skeleton_max_step,
            value_proxy_source_quality_max=value_proxy_source_quality_max,
            transfer_source_task_min=transfer_source_task_min,
            trajectory_record=trajectory_record,
        )["should_run"]
    )


def _primary_repair_gate_diagnostics(
    *,
    trigger: str,
    source_record: dict[str, object],
    source_controls: list[str],
    task_prompt: str,
    task_answer_type: str,
    source_quality_threshold: float,
    source_min_chars: int,
    source_prompt_gap_min: int = 0,
    source_prompt_gap_max: int = 999,
    source_prompt_coverage_min: float = 0.0,
    source_prompt_coverage_max: float = 1.0,
    denoise_skeleton_max_step: int | None = None,
    value_proxy_source_quality_max: float = 0.31,
    transfer_source_task_min: float = DECOMPOSED_SPEND_TRANSFER_SOURCE_TASK_MIN,
    trajectory_record: dict[str, object] | None = None,
) -> dict[str, object]:
    allowed_source_control = _source_control_allowed(source_record, source_controls)
    source_text = str(source_record.get("text", ""))
    source_quality = _planning_quality_score(source_record, task_prompt) if task_answer_type == "rubric" else 0.0
    source_task_score = _task_score(source_record)
    trajectory_task_score = _task_score(trajectory_record or source_record)
    source_chars = len(source_text.strip())
    source_needs_repair = source_quality < source_quality_threshold or source_chars < source_min_chars
    prompt_gap_count = (
        len(_prompt_constraint_gap_terms(task_prompt, source_text))
        if task_answer_type == "rubric"
        else 0
    )
    prompt_coverage = (
        _prompt_keyword_coverage(task_prompt, _normalize(source_text))
        if task_answer_type == "rubric"
        else 0.0
    )
    denoise_skeleton = _repairable_denoise_skeleton_features(
        source_record,
        task_prompt=task_prompt,
        prompt_coverage_min=source_prompt_coverage_min,
    )
    diagnostics: dict[str, object] = {
        "trigger": trigger,
        "allowed_source_control": allowed_source_control,
        "task_answer_type": task_answer_type,
        "source_quality": source_quality,
        "source_quality_threshold": source_quality_threshold,
        "source_chars": source_chars,
        "source_min_chars": source_min_chars,
        "source_needs_repair": source_needs_repair,
        "prompt_gap_count": prompt_gap_count,
        "source_prompt_gap_min": source_prompt_gap_min,
        "source_prompt_gap_max": source_prompt_gap_max,
        "prompt_coverage": prompt_coverage,
        "source_prompt_coverage_min": source_prompt_coverage_min,
        "source_prompt_coverage_max": source_prompt_coverage_max,
        "denoise_skeleton_max_step": denoise_skeleton_max_step,
        "value_proxy_source_quality_max": value_proxy_source_quality_max,
        "source_task_score": source_task_score,
        "trajectory_task_score": trajectory_task_score,
        "source_task_delta_vs_trajectory": source_task_score - trajectory_task_score,
        "transfer_source_task_min": transfer_source_task_min,
        "in_repairable_band": False,
        **denoise_skeleton,
        "denoise_skeleton_within_max_step": _denoise_skeleton_within_max_step(
            denoise_skeleton,
            max_step=denoise_skeleton_max_step,
        ),
        "should_run": False,
        "reason": "",
    }
    decomposed_selector_triggers = {
        DECOMPOSED_FOUR_HEAD_SELECTOR_ID,
        DECOMPOSED_SPEND_TRANSFER_SELECTOR_ID,
        TRAJECTORY_RELATIVE_DECOMPOSED_SPEND_SELECTOR_ID,
        LEARNED_AVAILABILITY_PREDICTOR_SELECTOR_ID,
        CALIBRATED_AVAILABILITY_PREDICTOR_SELECTOR_ID,
    }
    if trigger in decomposed_selector_triggers:
        _apply_decomposed_selector_diagnostics(
            diagnostics,
            selector_id=trigger,
            spend_head_prediction=False,
            transfer_source_task_min=transfer_source_task_min,
        )
    if not allowed_source_control:
        diagnostics["reason"] = "source_control_blocked"
        return diagnostics
    if task_answer_type != "rubric":
        diagnostics["reason"] = "unsupported_answer_type"
        return diagnostics
    if trigger == "always":
        diagnostics["should_run"] = True
        diagnostics["reason"] = "trigger_always"
        return diagnostics
    if trigger == "source_quality_or_short":
        diagnostics["should_run"] = source_needs_repair
        diagnostics["reason"] = "source_needs_repair" if source_needs_repair else "source_quality_ok"
        return diagnostics
    if trigger == COUNTERFACTUAL_MICRO_PROBE_TRIGGER_ID:
        diagnostics.update(
            _counterfactual_micro_probe_diagnostics(
                source_record=source_record,
                task_prompt=task_prompt,
                source_quality=source_quality,
                source_task_score=source_task_score,
                trajectory_task_score=trajectory_task_score,
                prompt_gap_count=prompt_gap_count,
                denoise_skeleton=denoise_skeleton,
                source_needs_repair=source_needs_repair,
                source_prompt_gap_min=source_prompt_gap_min,
                source_prompt_gap_max=source_prompt_gap_max,
                prompt_coverage=prompt_coverage,
                source_prompt_coverage_min=source_prompt_coverage_min,
                source_prompt_coverage_max=source_prompt_coverage_max,
            )
        )
        diagnostics["reason"] = (
            "counterfactual_probe_recorded_no_repair"
            if bool(diagnostics["would_probe"])
            else "counterfactual_probe_triage_skip_no_repair"
        )
        return diagnostics
    if trigger in {
        "source_repairability_geometry",
        "denoise_phase_repairability",
        "denoise_phase_value_proxy",
        DECOMPOSED_FOUR_HEAD_SELECTOR_ID,
        DECOMPOSED_SPEND_TRANSFER_SELECTOR_ID,
        TRAJECTORY_RELATIVE_DECOMPOSED_SPEND_SELECTOR_ID,
        LEARNED_AVAILABILITY_PREDICTOR_SELECTOR_ID,
        CALIBRATED_AVAILABILITY_PREDICTOR_SELECTOR_ID,
    }:
        if source_prompt_gap_min < 0:
            raise ValueError("source_prompt_gap_min must be non-negative")
        if source_prompt_gap_max < source_prompt_gap_min:
            raise ValueError("source_prompt_gap_max must be >= source_prompt_gap_min")
        if source_prompt_coverage_max < source_prompt_coverage_min:
            raise ValueError("source_prompt_coverage_max must be >= source_prompt_coverage_min")
        if not source_needs_repair:
            diagnostics["reason"] = "source_quality_ok"
            return diagnostics
        in_repairable_band = (
            source_prompt_gap_min <= prompt_gap_count <= source_prompt_gap_max
            and source_prompt_coverage_min <= prompt_coverage <= source_prompt_coverage_max
        )
        diagnostics["in_repairable_band"] = in_repairable_band
        if not in_repairable_band:
            diagnostics["reason"] = "outside_repairable_band"
            return diagnostics
        if trigger == "source_repairability_geometry":
            diagnostics["should_run"] = True
            diagnostics["reason"] = "repairable_geometry"
            return diagnostics
        has_repairable_denoise_skeleton = bool(
            diagnostics["has_repairable_denoise_skeleton"]
        )
        skeleton_within_max_step = bool(diagnostics["denoise_skeleton_within_max_step"])
        diagnostics["should_run"] = has_repairable_denoise_skeleton and skeleton_within_max_step
        if has_repairable_denoise_skeleton and not skeleton_within_max_step:
            diagnostics["reason"] = "late_repairable_denoise_skeleton"
            return diagnostics
        if trigger in decomposed_selector_triggers:
            _apply_decomposed_selector_diagnostics(
                diagnostics,
                selector_id=trigger,
                spend_head_prediction=False,
                transfer_source_task_min=transfer_source_task_min,
            )
        if (
            trigger
            in {
                "denoise_phase_value_proxy",
                DECOMPOSED_FOUR_HEAD_SELECTOR_ID,
                DECOMPOSED_SPEND_TRANSFER_SELECTOR_ID,
                TRAJECTORY_RELATIVE_DECOMPOSED_SPEND_SELECTOR_ID,
            }
            and has_repairable_denoise_skeleton
            and skeleton_within_max_step
            and source_quality > value_proxy_source_quality_max
        ):
            diagnostics["should_run"] = False
            diagnostics["reason"] = "value_proxy_source_quality_high"
            if trigger in decomposed_selector_triggers:
                diagnostics["spend_head_prediction"] = False
            return diagnostics
        if (
            trigger
            in {
                DECOMPOSED_SPEND_TRANSFER_SELECTOR_ID,
                TRAJECTORY_RELATIVE_DECOMPOSED_SPEND_SELECTOR_ID,
            }
            and has_repairable_denoise_skeleton
            and skeleton_within_max_step
            and source_task_score < transfer_source_task_min
        ):
            diagnostics["should_run"] = False
            diagnostics["reason"] = "transfer_source_task_score_low"
            diagnostics["spend_head_prediction"] = False
            return diagnostics
        if (
            trigger == TRAJECTORY_RELATIVE_DECOMPOSED_SPEND_SELECTOR_ID
            and has_repairable_denoise_skeleton
            and skeleton_within_max_step
            and source_task_score < trajectory_task_score
        ):
            diagnostics["should_run"] = False
            diagnostics["reason"] = "source_below_trajectory_selected"
            diagnostics["spend_head_prediction"] = False
            return diagnostics
        if (
            trigger == LEARNED_AVAILABILITY_PREDICTOR_SELECTOR_ID
            and has_repairable_denoise_skeleton
            and skeleton_within_max_step
            and prompt_gap_count > LEARNED_AVAILABILITY_PROMPT_GAP_MAX
        ):
            diagnostics["should_run"] = False
            diagnostics["reason"] = "learned_availability_prompt_gap_high"
            diagnostics["spend_head_prediction"] = False
            return diagnostics
        if (
            trigger == LEARNED_AVAILABILITY_PREDICTOR_SELECTOR_ID
            and has_repairable_denoise_skeleton
            and skeleton_within_max_step
            and source_quality > LEARNED_AVAILABILITY_SOURCE_QUALITY_MAX
        ):
            diagnostics["should_run"] = False
            diagnostics["reason"] = "learned_availability_source_quality_high"
            diagnostics["spend_head_prediction"] = False
            return diagnostics
        if (
            trigger == LEARNED_AVAILABILITY_PREDICTOR_SELECTOR_ID
            and has_repairable_denoise_skeleton
            and skeleton_within_max_step
            and source_task_score < trajectory_task_score
        ):
            diagnostics["should_run"] = False
            diagnostics["reason"] = "learned_availability_source_below_trajectory"
            diagnostics["spend_head_prediction"] = False
            return diagnostics
        if (
            trigger == CALIBRATED_AVAILABILITY_PREDICTOR_SELECTOR_ID
            and has_repairable_denoise_skeleton
            and skeleton_within_max_step
            and prompt_gap_count == CALIBRATED_AVAILABILITY_BLOCKED_PROMPT_GAP
        ):
            diagnostics["should_run"] = False
            diagnostics["reason"] = "calibrated_availability_prompt_gap_ambiguous"
            diagnostics["spend_head_prediction"] = False
            return diagnostics
        if (
            trigger == CALIBRATED_AVAILABILITY_PREDICTOR_SELECTOR_ID
            and has_repairable_denoise_skeleton
            and skeleton_within_max_step
            and source_task_score < trajectory_task_score
        ):
            diagnostics["should_run"] = False
            diagnostics["reason"] = "calibrated_availability_source_below_trajectory"
            diagnostics["spend_head_prediction"] = False
            return diagnostics
        if trigger in decomposed_selector_triggers:
            _apply_decomposed_selector_diagnostics(
                diagnostics,
                selector_id=trigger,
                spend_head_prediction=bool(diagnostics["should_run"]),
                transfer_source_task_min=transfer_source_task_min,
            )
        diagnostics["reason"] = (
            (
                trigger
                if trigger
                in {
                    "denoise_phase_value_proxy",
                    DECOMPOSED_FOUR_HEAD_SELECTOR_ID,
                    DECOMPOSED_SPEND_TRANSFER_SELECTOR_ID,
                    TRAJECTORY_RELATIVE_DECOMPOSED_SPEND_SELECTOR_ID,
                    LEARNED_AVAILABILITY_PREDICTOR_SELECTOR_ID,
                    CALIBRATED_AVAILABILITY_PREDICTOR_SELECTOR_ID,
                }
                else "denoise_phase_repairable"
            )
            if has_repairable_denoise_skeleton
            else "no_repairable_denoise_skeleton"
        )
        return diagnostics
    raise ValueError(f"Unsupported repair spend trigger: {trigger}")


def _counterfactual_micro_probe_diagnostics(
    *,
    source_record: dict[str, object],
    task_prompt: str,
    source_quality: float,
    source_task_score: float,
    trajectory_task_score: float,
    prompt_gap_count: int,
    denoise_skeleton: dict[str, object],
    source_needs_repair: bool,
    source_prompt_gap_min: int,
    source_prompt_gap_max: int,
    prompt_coverage: float,
    source_prompt_coverage_min: float,
    source_prompt_coverage_max: float,
) -> dict[str, object]:
    probe_feature_delta = _counterfactual_probe_feature_delta(
        prompt_gap_count=prompt_gap_count,
        denoise_skeleton=denoise_skeleton,
    )
    probe_value_prediction = _counterfactual_probe_value_prediction(
        source_quality=source_quality,
        source_task_delta_vs_trajectory=source_task_score - trajectory_task_score,
        probe_feature_delta=probe_feature_delta,
    )
    in_probe_band = (
        source_needs_repair
        and source_prompt_gap_min <= prompt_gap_count <= source_prompt_gap_max
        and source_prompt_coverage_min <= prompt_coverage <= source_prompt_coverage_max
    )
    would_probe = (
        in_probe_band
        and _number(probe_feature_delta["expected_gap_visibility_gain"])
        <= COUNTERFACTUAL_MICRO_PROBE_GAP_VISIBILITY_MAX
    )
    return {
        "counterfactual_probe_gate": "diagnostic_only",
        "counterfactual_probe_policy": COUNTERFACTUAL_MICRO_PROBE_POLICY_ID,
        "counterfactual_probe_cost_relative": COUNTERFACTUAL_MICRO_PROBE_COST_RELATIVE,
        "counterfactual_probe_observation": "deterministic_scaffold",
        "counterfactual_probe_text": _counterfactual_probe_text(
            source_record=source_record,
            task_prompt=task_prompt,
            prompt_gap_count=prompt_gap_count,
            source_quality=source_quality,
            denoise_skeleton=denoise_skeleton,
        ),
        "probe_feature_delta": probe_feature_delta,
        "probe_value_prediction": probe_value_prediction,
        "in_repairable_band": in_probe_band,
        "would_probe": would_probe,
        "should_run": False,
    }


def _should_generate_counterfactual_probe_record(
    diagnostics: dict[str, object],
    *,
    mode: str,
) -> bool:
    if mode == "triage":
        return bool(diagnostics.get("would_probe"))
    if mode == "all":
        return bool(diagnostics.get("counterfactual_probe_gate") == "diagnostic_only")
    raise ValueError(f"Unsupported counterfactual probe mode: {mode}")


def _counterfactual_probe_feature_delta(
    *,
    prompt_gap_count: int,
    denoise_skeleton: dict[str, object],
) -> dict[str, float]:
    first_coverage = _number(denoise_skeleton.get("first_repairable_denoise_skeleton_coverage"))
    peak_coverage = _number(denoise_skeleton.get("peak_denoise_prompt_coverage"))
    first_step_fraction = _number(denoise_skeleton.get("first_repairable_denoise_skeleton_step_fraction"))
    has_skeleton = bool(denoise_skeleton.get("has_repairable_denoise_skeleton"))
    return {
        "expected_gap_visibility_gain": min(1.0, max(0.0, prompt_gap_count / 12.0)),
        "expected_realization_defect_visibility": max(0.0, 1.0 - min(1.0, peak_coverage)),
        "expected_span_evidence_gain": min(1.0, max(first_coverage, peak_coverage)),
        "expected_retention_risk_visibility": (
            min(1.0, max(0.0, first_step_fraction))
            if has_skeleton
            else 1.0
        ),
    }


def _counterfactual_probe_value_prediction(
    *,
    source_quality: float,
    source_task_delta_vs_trajectory: float,
    probe_feature_delta: dict[str, float],
) -> float:
    gap_visibility = _number(probe_feature_delta.get("expected_gap_visibility_gain"))
    span_gain = _number(probe_feature_delta.get("expected_span_evidence_gain"))
    defect_visibility = _number(probe_feature_delta.get("expected_realization_defect_visibility"))
    retention_risk = _number(probe_feature_delta.get("expected_retention_risk_visibility"))
    raw = (
        0.035 * gap_visibility
        + 0.030 * span_gain
        + 0.020 * defect_visibility
        - 0.020 * retention_risk
        - 0.015 * max(0.0, source_quality - 0.30)
        + 0.010 * max(0.0, source_task_delta_vs_trajectory)
    )
    return max(0.0, raw)


def _generate_counterfactual_micro_probe_record(
    backend: HFDiffusionBackend,
    task: GeneralReasoningTask,
    *,
    source_record: dict[str, object],
    diagnostics: dict[str, object],
    generation_seed_base: int,
    probe_policy: str = COUNTERFACTUAL_MICRO_PROBE_POLICY_ID,
) -> dict[str, object]:
    max_new_tokens = _counterfactual_micro_probe_max_new_tokens(probe_policy)
    steps = _counterfactual_micro_probe_steps(probe_policy)
    generation_seed = _stable_generation_seed(
        generation_seed_base,
        str(source_record.get("candidate_key", "")),
        task.task_id,
        f"{COUNTERFACTUAL_MICRO_PROBE_TRIGGER_ID}:{probe_policy}:{_control_name(source_record)}",
    )
    _set_generation_seed(generation_seed)
    config = DiffusionGenerationConfig(
        max_new_tokens=max_new_tokens,
        steps=steps,
        algorithm="low_confidence",
        block_length=max_new_tokens,
        remasking="low_confidence",
        temperature=0.0,
        output_history=False,
        history_sample_count=0,
    )
    probe_metadata = {
        "probe_trigger": COUNTERFACTUAL_MICRO_PROBE_TRIGGER_ID,
        "probe_policy": probe_policy,
        "probe_cost_relative": _counterfactual_micro_probe_cost_relative(probe_policy),
        "probe_budget_max_new_tokens": max_new_tokens,
        "probe_budget_steps": steps,
        "probe_observation": "measured_generation",
        "source_control": _control_name(source_record),
        "source_task_score": _task_score(source_record),
        "source_planning_quality_score": diagnostics.get("source_quality"),
        "source_prompt_gap_count": diagnostics.get("prompt_gap_count"),
        "source_prompt_coverage": diagnostics.get("prompt_coverage"),
        "source_first_repairable_denoise_skeleton_step": diagnostics.get(
            "first_repairable_denoise_skeleton_step"
        ),
        "full_repair_authorized": False,
    }
    return _generate_record(
        backend,
        task,
        config=config,
        schedule={"name": COUNTERFACTUAL_MICRO_PROBE_TRIGGER_ID},
        stage="counterfactual_probe",
        generation_seed=generation_seed,
        prompt_override=_counterfactual_micro_probe_prompt(
            task_prompt=task.prompt,
            source_text=str(source_record.get("text", "")),
            diagnostics=diagnostics,
            probe_policy=probe_policy,
        ),
        counterfactual_probe=probe_metadata,
    )


def _counterfactual_micro_probe_max_new_tokens(probe_policy: str) -> int:
    if probe_policy == COUNTERFACTUAL_MICRO_PROBE_SPAN_POLICY_ID:
        return COUNTERFACTUAL_MICRO_PROBE_SPAN_MAX_NEW_TOKENS
    if probe_policy == COUNTERFACTUAL_MICRO_PROBE_COMPACT_POLICY_ID:
        return COUNTERFACTUAL_MICRO_PROBE_COMPACT_MAX_NEW_TOKENS
    if probe_policy == COUNTERFACTUAL_MICRO_PROBE_KEY_VALUE_POLICY_ID:
        return COUNTERFACTUAL_MICRO_PROBE_KEY_VALUE_MAX_NEW_TOKENS
    if probe_policy == COUNTERFACTUAL_MICRO_PROBE_TOMOGRAPHY_POLICY_ID:
        return COUNTERFACTUAL_MICRO_PROBE_TOMOGRAPHY_MAX_NEW_TOKENS
    return COUNTERFACTUAL_MICRO_PROBE_MAX_NEW_TOKENS


def _counterfactual_micro_probe_steps(probe_policy: str) -> int:
    if probe_policy == COUNTERFACTUAL_MICRO_PROBE_SPAN_POLICY_ID:
        return COUNTERFACTUAL_MICRO_PROBE_SPAN_STEPS
    if probe_policy == COUNTERFACTUAL_MICRO_PROBE_COMPACT_POLICY_ID:
        return COUNTERFACTUAL_MICRO_PROBE_COMPACT_STEPS
    if probe_policy == COUNTERFACTUAL_MICRO_PROBE_KEY_VALUE_POLICY_ID:
        return COUNTERFACTUAL_MICRO_PROBE_KEY_VALUE_STEPS
    if probe_policy == COUNTERFACTUAL_MICRO_PROBE_TOMOGRAPHY_POLICY_ID:
        return COUNTERFACTUAL_MICRO_PROBE_TOMOGRAPHY_STEPS
    return COUNTERFACTUAL_MICRO_PROBE_STEPS


def _counterfactual_micro_probe_cost_relative(probe_policy: str) -> float:
    return _safe_ratio(
        _counterfactual_micro_probe_max_new_tokens(probe_policy),
        256,
    )


def _measured_counterfactual_micro_probe_diagnostics(
    probe_record: dict[str, object],
    *,
    source_record: dict[str, object],
    task_prompt: str,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    measured_text = str(probe_record.get("text", ""))
    probe_metadata = _dict(probe_record.get("counterfactual_probe"))
    probe_policy = str(
        probe_metadata.get("probe_policy", COUNTERFACTUAL_MICRO_PROBE_POLICY_ID)
    )
    source_text = str(source_record.get("text", ""))
    text_validity = _counterfactual_micro_probe_text_validity(measured_text)
    source_gap_terms = _prompt_constraint_gap_terms(task_prompt, source_text, limit=12)
    remaining_gap_terms = [
        term
        for term in source_gap_terms
        if _normalize(term) not in _normalize(measured_text)
    ]
    source_prompt_gap_count = int(_number(diagnostics.get("prompt_gap_count")))
    measured_delta = {
        "expected_gap_visibility_gain": _safe_ratio(
            max(0, len(source_gap_terms) - len(remaining_gap_terms)),
            max(1, len(source_gap_terms)),
        ),
        "expected_realization_defect_visibility": _counterfactual_probe_realization_signal(measured_text),
        "expected_span_evidence_gain": _prompt_keyword_coverage(task_prompt, _normalize(measured_text)),
        "expected_retention_risk_visibility": max(
            0.0,
            1.0 - _text_similarity(measured_text, source_text),
        ),
    }
    measured_value = _counterfactual_probe_value_prediction(
        source_quality=_number(diagnostics.get("source_quality")),
        source_task_delta_vs_trajectory=_number(diagnostics.get("source_task_delta_vs_trajectory")),
        probe_feature_delta=measured_delta,
    )
    return {
        "counterfactual_probe_observation": "measured_generation",
        "counterfactual_probe_policy": probe_policy,
        "counterfactual_probe_cost_relative": _counterfactual_micro_probe_cost_relative(
            probe_policy
        ),
        "counterfactual_probe_record_id": _record_identity(probe_record),
        "counterfactual_probe_generated_token_count": int(
            _number(probe_record.get("generated_token_count"))
        ),
        "counterfactual_probe_measured_text": _compact_text(measured_text, max_chars=360),
        "counterfactual_probe_remaining_gap_count": len(remaining_gap_terms),
        "counterfactual_probe_resolved_gap_count": max(
            0,
            min(source_prompt_gap_count, len(source_gap_terms) - len(remaining_gap_terms)),
        ),
        "counterfactual_probe_source_gap_terms": source_gap_terms,
        "counterfactual_probe_remaining_gap_terms": remaining_gap_terms,
        "counterfactual_probe_text_exact_authorization_false": text_validity[
            "exact_authorization_false"
        ],
        "counterfactual_probe_text_generic_slot": text_validity["generic_slot"],
        "counterfactual_probe_text_malformed_authorization": text_validity[
            "malformed_authorization"
        ],
        "counterfactual_probe_text_placeholder_slot": text_validity["placeholder_slot"],
        "counterfactual_probe_text_slot_count": text_validity["slot_count"],
        "counterfactual_probe_text_valid_for_stage1": text_validity["valid_for_stage1"],
        "counterfactual_probe_text_weird_punctuation": text_validity["weird_punctuation"],
        "measured_probe_feature_delta": measured_delta,
        "measured_probe_value_prediction": measured_value,
        "probe_feature_delta": measured_delta,
        "probe_value_prediction": measured_value,
        "should_run": False,
    }


def _counterfactual_micro_probe_text_validity(text: str) -> dict[str, object]:
    compact_authorization = bool(re.search(r"(^|\n)\s*Z\s*=\s*false\s*(\n|$)", text))
    span_authorization = bool(re.search(r"(^|\n)\s*N\s*=\s*0\s*(\n|$)", text))
    exact_authorization = (
        "FULL_REPAIR_AUTHORIZED=false" in text
        or compact_authorization
        or span_authorization
    )
    has_full_repair = "FULL_REPAIR" in text
    has_compact_authorization_key = bool(re.search(r"(^|\n)\s*Z\s*=", text))
    has_span_authorization_key = bool(re.search(r"(^|\n)\s*N\s*=", text))
    malformed_authorization = (
        has_full_repair
        or has_compact_authorization_key
        or has_span_authorization_key
    ) and not exact_authorization
    strict_slot_patterns = (
        r"(^|\n)\s*MISSING_CONSTRAINT\s*=",
        r"(^|\n)\s*EVIDENCE_NEEDED\s*=",
        r"(^|\n)\s*RETENTION_RISK\s*=",
    )
    strict_slot_count = sum(int(bool(re.search(pattern, text))) for pattern in strict_slot_patterns)
    compact_slot_count = sum(
        int(bool(re.search(pattern, text)))
        for pattern in (r"(^|\n)\s*A\s*=", r"(^|\n)\s*B\s*=", r"(^|\n)\s*C\s*=")
    )
    span_slot_count = sum(
        int(bool(re.search(pattern, text)))
        for pattern in (r"(^|\n)\s*X0\s*=", r"(^|\n)\s*X1\s*=", r"(^|\n)\s*X2\s*=")
    )
    legacy_slot_count = sum(
        int(bool(re.search(pattern, text)))
        for pattern in (r"(^|[ ;])1\)", r"(^|[ ;])2\)", r"(^|[ ;])3\)")
    )
    slot_count = max(strict_slot_count, compact_slot_count, span_slot_count, legacy_slot_count)
    weird_punctuation = any(
        marker in text
        for marker in (
            "::",
            "..",
            ",AUTHORIZED",
            "AUTH_AUTHORIZED",
            "AUTHUTHORIZED",
            "AUTHORORIZED",
            "RETION_RISK",
            "RETENTION_RISK_Risk",
            "Z==",
            "N==",
        )
    )
    placeholder_slot = "<" in text or ">" in text
    generic_slot = bool(
        re.search(
            r"(MISSING_CONSTRAINT|EVIDENCE_NEEDED|RETENTION_RISK)\s*=\s*(none|true|false|unknown)?\s*(\n|$)",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"(^|\n)\s*[ABC]\s*=\s*(none|true|false|unknown)?\s*(\n|$)",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"(^|\n)\s*X[0-2]\s*=\s*(none|true|false|unknown)?\s*(\n|$)",
            text,
            flags=re.IGNORECASE,
        )
    )
    return {
        "exact_authorization_false": exact_authorization,
        "generic_slot": generic_slot,
        "malformed_authorization": malformed_authorization,
        "placeholder_slot": placeholder_slot,
        "slot_count": slot_count,
        "valid_for_stage1": (
            exact_authorization
            and slot_count >= 3
            and not generic_slot
            and not placeholder_slot
            and not weird_punctuation
        ),
        "weird_punctuation": weird_punctuation,
    }


def _counterfactual_micro_probe_prompt(
    *,
    task_prompt: str,
    source_text: str,
    diagnostics: dict[str, object],
    probe_policy: str = COUNTERFACTUAL_MICRO_PROBE_POLICY_ID,
) -> str:
    gap_terms = _prompt_constraint_gap_terms(task_prompt, source_text, limit=8)
    gap_text = ", ".join(gap_terms) if gap_terms else "none"
    draft = _compact_text(source_text, max_chars=700)
    if probe_policy == COUNTERFACTUAL_MICRO_PROBE_SPAN_POLICY_ID:
        return (
            f"{task_prompt}\n\n"
            f"Draft answer under inspection:\n{draft}\n\n"
            f"Weak task words: {gap_text}\n\n"
            "Probe only. Do not rewrite or solve. Return four newline-separated "
            "key=value lines with labels X0, X1, X2, N. After X0, copy 2-8 "
            "consecutive words from the task that are absent or weak in the draft. "
            "After X1, copy 2-8 consecutive words from the draft that a verifier can "
            "check. After X2, copy 2-8 consecutive words from the draft that should "
            "be preserved. Use copied words only after equals signs: no label words, "
            "no explanations, no angle brackets. The fourth line must be exactly N=0.\n"
            "X0=\n"
            "X1=\n"
            "X2=\n"
            "N=0"
        )
    if probe_policy == COUNTERFACTUAL_MICRO_PROBE_COMPACT_POLICY_ID:
        return (
            f"{task_prompt}\n\n"
            f"Draft answer under inspection:\n{draft}\n\n"
            f"Missing or weak task terms: {gap_text}\n\n"
            "Counterfactual compact tomography probe only. Do not rewrite the answer. "
            "Return exactly four newline-separated key=value lines using these keys only: "
            "A, B, C, Z. A is the missing or weak constraint. B is verifier-visible "
            "evidence needed before buying repair. C is source detail repair might delete "
            "or distort. Z is the repair authorization sentinel. Use concrete task words "
            "for A, B, and C. Do not use angle brackets. Do not write true, false, none, "
            "unknown, or generic template text for A, B, or C. The fourth line must be "
            "exactly Z=false.\n"
            "A=\n"
            "B=\n"
            "C=\n"
            "Z=false"
        )
    if probe_policy == COUNTERFACTUAL_MICRO_PROBE_KEY_VALUE_POLICY_ID:
        return (
            f"{task_prompt}\n\n"
            f"Draft answer under inspection:\n{draft}\n\n"
            f"Missing or weak task terms: {gap_text}\n\n"
            "Counterfactual key-value tomography probe only. Do not rewrite the answer. "
            "Return exactly four newline-separated key=value lines. Copy the four keys "
            "exactly. After each equals sign, write concrete plain words from this task. "
            "Do not use angle brackets. Do not write true, false, none, unknown, or generic "
            "template text as a value.\n"
            "MISSING_CONSTRAINT=\n"
            "EVIDENCE_NEEDED=\n"
            "RETENTION_RISK=\n"
            "FULL_REPAIR_AUTHORIZED=false"
        )
    if probe_policy == COUNTERFACTUAL_MICRO_PROBE_TOMOGRAPHY_POLICY_ID:
        return (
            f"{task_prompt}\n\n"
            f"Draft answer under inspection:\n{draft}\n\n"
            f"Missing or weak task terms: {gap_text}\n\n"
            "Counterfactual tomography micro-probe only. Do not rewrite the answer. "
            "Return exactly these four lines and no other text:\n"
            "MISSING_CONSTRAINT=<one missing or weak constraint that changes repair value>\n"
            "EVIDENCE_NEEDED=<verifier-visible evidence needed before buying repair>\n"
            "RETENTION_RISK=<source detail that repair might delete or distort>\n"
            "FULL_REPAIR_AUTHORIZED=false"
        )
    return (
        f"{task_prompt}\n\n"
        f"Draft answer under inspection:\n{draft}\n\n"
        f"Missing or weak task terms: {gap_text}\n\n"
        "Counterfactual micro-probe only. Do not rewrite the full answer and do not solve "
        "the whole task. In at most 80 words, list: "
        "1) the missing constraint that most changes repair value; "
        "2) the verifier-visible evidence needed to decide whether repair is worth buying; "
        "3) any retention risk in the draft. End with FULL_REPAIR_AUTHORIZED=false."
    )


def _counterfactual_probe_realization_signal(text: str) -> float:
    normalized = _normalize(text)
    if not normalized:
        return 0.0
    evidence_terms = (
        "missing",
        "constraint",
        "evidence",
        "verifier",
        "measure",
        "risk",
        "fallback",
        "threshold",
        "repair",
    )
    return min(1.0, sum(1 for term in evidence_terms if term in normalized) / 5.0)


def _counterfactual_probe_text(
    *,
    source_record: dict[str, object],
    task_prompt: str,
    prompt_gap_count: int,
    source_quality: float,
    denoise_skeleton: dict[str, object],
) -> str:
    gap_terms = _prompt_constraint_gap_terms(
        task_prompt,
        str(source_record.get("text", "")),
        limit=6,
    )
    gap_text = ", ".join(gap_terms) if gap_terms else "none"
    return (
        f"Probe {_task_id(source_record)}: missing_terms={gap_text}; "
        f"gap_count={prompt_gap_count}; "
        f"source_quality={source_quality:.6f}; "
        f"first_skeleton_step={denoise_skeleton.get('first_repairable_denoise_skeleton_step')}; "
        "full_repair_authorized=false."
    )


def _apply_decomposed_selector_diagnostics(
    diagnostics: dict[str, object],
    *,
    selector_id: str = DECOMPOSED_FOUR_HEAD_SELECTOR_ID,
    spend_head_prediction: bool,
    transfer_source_task_min: float = DECOMPOSED_SPEND_TRANSFER_SOURCE_TASK_MIN,
) -> None:
    spend_rule_id = (
        CALIBRATED_AVAILABILITY_PREDICTOR_RULE_ID
        if selector_id == CALIBRATED_AVAILABILITY_PREDICTOR_SELECTOR_ID
        else (
            LEARNED_AVAILABILITY_PREDICTOR_RULE_ID
            if selector_id == LEARNED_AVAILABILITY_PREDICTOR_SELECTOR_ID
            else (
                TRAJECTORY_RELATIVE_DECOMPOSED_SPEND_RULE_ID
                if selector_id == TRAJECTORY_RELATIVE_DECOMPOSED_SPEND_SELECTOR_ID
                else (
                    DECOMPOSED_SPEND_TRANSFER_RULE_ID
                    if selector_id == DECOMPOSED_SPEND_TRANSFER_SELECTOR_ID
                    else DECOMPOSED_SELECTOR_SPEND_RULE_ID
                )
            )
        )
    )
    diagnostics.update(
        {
            "composite_selector_id": selector_id,
            "spend_head_prediction": spend_head_prediction,
            "spend_head_rule_id": spend_rule_id,
            "spend_head_source_task_min": (
                transfer_source_task_min
                if selector_id == DECOMPOSED_SPEND_TRANSFER_SELECTOR_ID
                else None
            ),
            "source_head_prediction": False,
            "source_head_rule_id": DECOMPOSED_SELECTOR_SOURCE_RULE_ID,
            "retention_head_prediction": False,
            "retention_head_rule_id": DECOMPOSED_SELECTOR_RETENTION_RULE_ID,
            "realization_head_policy": "auto_compat_preserve_seeded",
            "realization_head_rule_id": DECOMPOSED_SELECTOR_REALIZATION_RULE_ID,
        }
    )


def _repair_spend_gate_row(
    source_record: dict[str, object],
    diagnostics: dict[str, object],
) -> dict[str, object]:
    row = dict(diagnostics)
    row.update(
        {
            "candidate_key": str(source_record.get("candidate_key", "")),
            "task_id": _task_id(source_record),
            "source_control": _control_name(source_record),
            "source_task_score": _task_score(source_record),
        }
    )
    return row


def _repairable_denoise_skeleton_features(
    source_record: dict[str, object],
    *,
    task_prompt: str,
    prompt_coverage_min: float,
    min_visible_chars: int = 20,
) -> dict[str, object]:
    trajectory_summary = source_record.get("trajectory_summary")
    if not isinstance(trajectory_summary, dict):
        return _empty_repairable_denoise_skeleton_features()
    samples = trajectory_summary.get("samples")
    if not isinstance(samples, list):
        return _empty_repairable_denoise_skeleton_features()
    sample_count = 0
    peak_coverage = 0.0
    peak_coverage_step: int | None = None
    history_steps = _denoise_history_step_count(source_record, samples)
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_count += 1
        visible_text = str(sample.get("visible_text", "") or "")
        coverage = _prompt_keyword_coverage(task_prompt, _normalize(visible_text)) if visible_text else 0.0
        step = _optional_int(sample.get("step"))
        if coverage > peak_coverage:
            peak_coverage = coverage
            peak_coverage_step = step
        visible_chars = len(visible_text.strip())
        if visible_chars < min_visible_chars:
            continue
        if coverage >= prompt_coverage_min:
            return {
                "has_repairable_denoise_skeleton": True,
                "denoise_history_steps": history_steps,
                "denoise_skeleton_sample_count": sample_count,
                "first_repairable_denoise_skeleton_step": step,
                "first_repairable_denoise_skeleton_step_fraction": (
                    step / history_steps
                    if step is not None and isinstance(history_steps, int) and history_steps > 0
                    else None
                ),
                "first_repairable_denoise_skeleton_coverage": coverage,
                "first_repairable_denoise_skeleton_visible_chars": visible_chars,
                "peak_denoise_prompt_coverage": peak_coverage,
                "peak_denoise_prompt_coverage_step": peak_coverage_step,
            }
    return {
        **_empty_repairable_denoise_skeleton_features(),
        "denoise_history_steps": history_steps,
        "denoise_skeleton_sample_count": sample_count,
        "peak_denoise_prompt_coverage": peak_coverage,
        "peak_denoise_prompt_coverage_step": peak_coverage_step,
    }


def _source_has_repairable_denoise_skeleton(
    source_record: dict[str, object],
    *,
    task_prompt: str,
    prompt_coverage_min: float,
    min_visible_chars: int = 20,
) -> bool:
    return bool(
        _repairable_denoise_skeleton_features(
            source_record,
            task_prompt=task_prompt,
            prompt_coverage_min=prompt_coverage_min,
            min_visible_chars=min_visible_chars,
        )["has_repairable_denoise_skeleton"]
    )


def _empty_repairable_denoise_skeleton_features() -> dict[str, object]:
    return {
        "has_repairable_denoise_skeleton": False,
        "denoise_history_steps": None,
        "denoise_skeleton_sample_count": 0,
        "first_repairable_denoise_skeleton_step": None,
        "first_repairable_denoise_skeleton_step_fraction": None,
        "first_repairable_denoise_skeleton_coverage": None,
        "first_repairable_denoise_skeleton_visible_chars": None,
        "peak_denoise_prompt_coverage": None,
        "peak_denoise_prompt_coverage_step": None,
    }


def _denoise_history_step_count(
    source_record: dict[str, object],
    samples: list[object],
) -> int | None:
    history_steps = _optional_int(source_record.get("history_steps"))
    if history_steps is not None and history_steps > 0:
        return history_steps
    sample_steps = [
        step
        for sample in samples
        if isinstance(sample, dict)
        for step in [_optional_int(sample.get("step"))]
        if step is not None
    ]
    if sample_steps:
        return max(sample_steps)
    return None


def _denoise_skeleton_within_max_step(
    features: dict[str, object],
    *,
    max_step: int | None,
) -> bool:
    if not bool(features.get("has_repairable_denoise_skeleton")):
        return False
    if max_step is None:
        return True
    first_step = _optional_int(features.get("first_repairable_denoise_skeleton_step"))
    return first_step is not None and first_step <= max_step


def _optional_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and not isinstance(value, bool) and value.is_integer():
        return int(value)
    return None


def _primary_repair_selection_reason(trigger: str) -> str:
    return f"repair_spend_gate_kept_evolved_{trigger}"


def _repair_candidates(
    *,
    repair_pack: str = "prefix",
    include_history_repairs: bool,
    history_repair_fractions: tuple[float, ...],
    include_history_visible_repair: bool,
    limit: int,
) -> tuple[Any, ...]:
    if limit <= 0:
        return ()
    if repair_pack == "prefix":
        repairs = list(default_llada_repair_candidates())
    elif repair_pack == "source_relative":
        repairs = list(default_llada_source_relative_repair_candidates())
    elif repair_pack == "targeted_content":
        repairs = list(default_llada_targeted_content_repair_candidates())
    elif repair_pack == "prompt_guided":
        repairs = list(default_llada_prompt_guided_repair_candidates())
    elif repair_pack == "constraint_gap":
        repairs = list(default_llada_constraint_gap_repair_candidates())
    elif repair_pack == "constraint_span":
        repairs = list(default_llada_constraint_span_repair_candidates())
    elif repair_pack == "constraint_span_anchor_select":
        repairs = list(default_llada_constraint_span_anchor_select_repair_candidates())
    elif repair_pack == "constraint_span_phase_anchor":
        repairs = list(default_llada_constraint_span_phase_anchor_repair_candidates())
    elif repair_pack == "constraint_span_phase_hybrid_preserve_seeded_gated":
        repairs = list(default_llada_constraint_span_phase_hybrid_preserve_seeded_gated_repair_candidates())
    elif repair_pack == "constraint_span_phase_final_preserve_seeded_gated":
        repairs = list(default_llada_constraint_span_phase_final_preserve_seeded_gated_repair_candidates())
    elif repair_pack == "constraint_span_anchor_instability":
        repairs = list(default_llada_constraint_span_anchor_instability_repair_candidates())
    elif repair_pack == "constraint_span_anchor_instability_gated":
        repairs = list(default_llada_constraint_span_anchor_instability_gated_repair_candidates())
    elif repair_pack == "constraint_span_anchor_instability_claim_gated":
        repairs = list(default_llada_constraint_span_anchor_instability_claim_gated_repair_candidates())
    elif repair_pack == "constraint_span_anchor_instability_claim_oracle_gated":
        repairs = list(default_llada_constraint_span_anchor_instability_claim_oracle_gated_repair_candidates())
    elif repair_pack == "constraint_span_anchor_instability_claim_seeded_gated":
        repairs = list(default_llada_constraint_span_anchor_instability_claim_seeded_gated_repair_candidates())
    elif repair_pack == "constraint_span_anchor_instability_claim_compatible_seeded_gated":
        repairs = list(
            default_llada_constraint_span_anchor_instability_claim_compatible_seeded_gated_repair_candidates()
        )
    elif repair_pack == "constraint_span_anchor_instability_claim_auto_seeded_gated":
        repairs = list(default_llada_constraint_span_anchor_instability_claim_auto_seeded_gated_repair_candidates())
    elif repair_pack == "constraint_span_anchor_instability_claim_auto_action_seeded_gated":
        repairs = list(
            default_llada_constraint_span_anchor_instability_claim_auto_action_seeded_gated_repair_candidates()
        )
    elif repair_pack == "constraint_span_anchor_instability_claim_auto_compat_seeded_gated":
        repairs = list(
            default_llada_constraint_span_anchor_instability_claim_auto_compat_seeded_gated_repair_candidates()
        )
    elif repair_pack == "constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated":
        repairs = list(
            default_llada_constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_repair_candidates()
        )
    elif repair_pack == "constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated":
        repairs = list(
            default_llada_constraint_span_anchor_instability_claim_auto_compat_realized_seeded_gated_repair_candidates()
        )
    elif repair_pack == "constraint_span_anchor_instability_claim_auto_joint_seeded_gated":
        repairs = list(default_llada_constraint_span_anchor_instability_claim_auto_joint_seeded_gated_repair_candidates())
    elif repair_pack == "constraint_span_anchor_instability_claim_auto_seeded_realization_gated":
        repairs = list(
            default_llada_constraint_span_anchor_instability_claim_auto_seeded_realization_gated_repair_candidates()
        )
    elif repair_pack == "constraint_span_anchor_instability_claim_strict_gated":
        repairs = list(default_llada_constraint_span_anchor_instability_claim_strict_gated_repair_candidates())
    elif repair_pack == "constraint_span_anchor_instability_prompt_only_gated":
        repairs = list(default_llada_constraint_span_anchor_instability_prompt_only_gated_repair_candidates())
    elif repair_pack == "constraint_span_anchor_instability_prompt_gated":
        repairs = list(default_llada_constraint_span_anchor_instability_prompt_gated_repair_candidates())
    elif repair_pack == "constraint_span_anchor_search":
        repairs = list(default_llada_constraint_span_anchor_search_repair_candidates())
    elif repair_pack == "constraint_span_history":
        repairs = list(default_llada_constraint_span_history_repair_candidates())
    elif repair_pack == "constraint_span_history_contrast":
        repairs = list(default_llada_constraint_span_history_contrast_repair_candidates())
    elif repair_pack == "constraint_span_history_instability":
        repairs = list(default_llada_constraint_span_history_instability_repair_candidates())
    elif repair_pack == "constraint_span_clause":
        repairs = list(default_llada_constraint_span_clause_repair_candidates())
    elif repair_pack == "state_adaptive":
        repairs = list(default_llada_state_adaptive_repair_candidates())
    elif repair_pack == "replay_consistency":
        repairs = list(default_llada_replay_consistency_repair_candidates())
    else:
        raise ValueError(f"Unsupported repair pack: {repair_pack}")
    if include_history_repairs:
        history_repairs = list(default_llada_history_repair_candidates(history_repair_fractions))
        if include_history_visible_repair:
            history_repairs.extend(default_llada_history_visible_repair_candidates())
        repairs = [
            *history_repairs,
            *repairs,
        ]
    return tuple(repairs[:limit])


def _history_rescue_candidates(
    *,
    history_rescue_fractions: tuple[float, ...],
    include_history_rescue_visible: bool,
    existing_repairs: tuple[Any, ...],
) -> tuple[Any, ...]:
    if not history_rescue_fractions and not include_history_rescue_visible:
        return ()
    existing_names = {str(repair.name) for repair in existing_repairs}
    rescue_candidates = [
        *default_llada_history_repair_candidates(history_rescue_fractions),
    ]
    if include_history_rescue_visible:
        rescue_candidates.extend(default_llada_history_visible_repair_candidates())
    return tuple(
        repair
        for repair in rescue_candidates
        if str(repair.name) not in existing_names
    )


def _prompt_guided_rescue_candidates(
    *,
    existing_repairs: tuple[Any, ...],
    limit: int,
) -> tuple[Any, ...]:
    if limit <= 0:
        return ()
    existing_names = {str(repair.name) for repair in existing_repairs}
    rescue_candidates = [
        repair
        for repair in default_llada_prompt_guided_repair_candidates()
        if str(repair.name) not in existing_names
    ]
    return tuple(rescue_candidates[:limit])


def _constraint_gap_rescue_candidates(
    *,
    existing_repairs: tuple[Any, ...],
    limit: int,
) -> tuple[Any, ...]:
    if limit <= 0:
        return ()
    existing_names = {str(repair.name) for repair in existing_repairs}
    rescue_candidates = [
        repair
        for repair in default_llada_constraint_gap_repair_candidates()
        if getattr(repair, "prompt_repair_policy", None) == "constraint_gap"
        and str(repair.name) not in existing_names
    ]
    return tuple(rescue_candidates[:limit])


def _is_planning_span_repair(repair: Any) -> bool:
    return str(getattr(repair, "name", "")) in {
        "constraint_gap_span_anchor_search_repair",
        "constraint_gap_span_anchor_instability_gated_repair",
        "constraint_gap_span_anchor_instability_claim_gated_repair",
        "constraint_gap_span_anchor_instability_claim_oracle_gated_repair",
        "constraint_gap_span_anchor_instability_claim_seeded_gated_repair",
        "constraint_gap_span_anchor_instability_claim_compatible_seeded_gated_repair",
        "constraint_gap_span_anchor_instability_claim_auto_seeded_gated_repair",
        "constraint_gap_span_anchor_instability_claim_auto_action_seeded_gated_repair",
        "constraint_gap_span_anchor_instability_claim_auto_compat_seeded_gated_repair",
        "constraint_gap_span_anchor_instability_claim_auto_seeded_realization_gated_repair",
        "constraint_gap_span_anchor_instability_claim_strict_gated_repair",
        PHASE_FINAL_PRESERVE_REPAIR_NAME,
        "constraint_gap_span_anchor_instability_prompt_only_gated_repair",
        "constraint_gap_span_anchor_instability_prompt_gated_repair",
        "constraint_gap_span_anchor_instability_repair",
        "constraint_gap_span_anchor_select_repair",
        "constraint_gap_span_history_contrast_repair",
        "constraint_gap_span_history_instability_repair",
        "constraint_gap_span_repair",
        "constraint_gap_span_history_repair",
        "constraint_gap_span_clause_repair",
    }


def _history_instability_gate_decision(
    repair: Any,
    *,
    planning_span_targets: list[str],
    source_quality_score: float,
    source_state: str,
) -> dict[str, object]:
    policy = getattr(repair, "history_instability_gate_policy", None)
    remask_fraction = getattr(repair, "remask_history_unstable_fraction", None)
    prompt_policy = getattr(repair, "history_instability_gate_prompt_policy", None)
    if remask_fraction is None and prompt_policy is None:
        return {"active": False, "policy": policy, "reason": "no_instability_fraction"}
    if not policy:
        return {"active": True, "policy": policy, "reason": "ungated_instability"}
    if policy != "multi_span_low_quality":
        raise ValueError(f"Unsupported history instability gate policy: {policy}")
    if source_state != "final":
        return {"active": False, "policy": policy, "reason": "history_anchor_skip"}
    if len(planning_span_targets) < 3:
        return {"active": False, "policy": policy, "reason": "too_few_planning_spans"}
    if source_quality_score > 0.27:
        return {"active": False, "policy": policy, "reason": "source_quality_above_gate"}
    return {"active": True, "policy": policy, "reason": "multi_span_low_quality"}


def _planning_prompt_gate_decision(
    repair: Any,
    *,
    task_prompt: str,
    prompt_constraint_gap_terms: list[str],
    planning_span_targets: list[str],
    source_quality_score: float,
    source_state: str,
) -> dict[str, object]:
    policy = getattr(repair, "planning_prompt_gate_policy", None)
    if policy is None:
        return {"active": False, "policy": policy, "reason": "disabled"}
    if policy != "public_claim_confound_control":
        raise ValueError(f"Unsupported planning prompt gate policy: {policy}")
    normalized_prompt = _normalize(task_prompt)
    normalized_terms = " ".join(_normalize(term) for term in prompt_constraint_gap_terms)
    if source_state != "final":
        return {"active": False, "policy": policy, "reason": "history_anchor_skip"}
    if source_quality_score > 0.32:
        return {"active": False, "policy": policy, "reason": "source_quality_above_gate"}
    if not planning_span_targets:
        return {"active": False, "policy": policy, "reason": "no_planning_span_targets"}
    if "public claim" not in normalized_prompt:
        return {"active": False, "policy": policy, "reason": "no_public_claim"}
    if "baseline" not in normalized_prompt:
        return {"active": False, "policy": policy, "reason": "no_baseline"}
    confound_surface = " ".join([normalized_prompt, normalized_terms])
    if not _contains_any(
        confound_surface,
        (
            "more tokens",
            "token budget",
            "different prompt",
            "prompt format",
            "best of",
            "best-of",
            "oracle",
        ),
    ):
        return {"active": False, "policy": policy, "reason": "no_claim_confound"}
    return {"active": True, "policy": policy, "reason": "public_claim_confound_control"}


def _should_run_history_rescue(
    *,
    selected_repair: dict[str, object],
    baseline_record: dict[str, object],
    source_controls: list[str],
) -> bool:
    if _record_identity(selected_repair) != _record_identity(baseline_record):
        return False
    return _source_control_allowed(baseline_record, source_controls)


def _should_run_adaptive_history_rescue(
    *,
    trigger: str,
    selected_repair: dict[str, object],
    baseline_record: dict[str, object],
    repair_pool: list[dict[str, object]],
    source_controls: list[str],
    task_prompt: str,
    task_answer_type: str,
    exact_task_trajectory_policy: str,
    trajectory_selector: str,
) -> bool:
    baseline_triggered = _should_run_history_rescue(
        selected_repair=selected_repair,
        baseline_record=baseline_record,
        source_controls=source_controls,
    )
    disagreement_triggered = _should_run_selector_disagreement_rescue(
        selected_repair=selected_repair,
        baseline_record=baseline_record,
        repair_pool=repair_pool,
        source_controls=source_controls,
        task_prompt=task_prompt,
        task_answer_type=task_answer_type,
        exact_task_trajectory_policy=exact_task_trajectory_policy,
        trajectory_selector=trajectory_selector,
    )
    if trigger == "baseline":
        return baseline_triggered
    if trigger == "selector_disagreement":
        return disagreement_triggered
    if trigger == "baseline_or_selector_disagreement":
        return baseline_triggered or disagreement_triggered
    raise ValueError(f"Unsupported history rescue trigger: {trigger}")


def _should_run_prompt_guided_rescue(
    *,
    trigger: str,
    selected_repair: dict[str, object],
    baseline_record: dict[str, object],
    repair_pool: list[dict[str, object]],
    source_controls: list[str],
    task_prompt: str,
    task_answer_type: str,
    exact_task_trajectory_policy: str,
    trajectory_selector: str,
    source_quality_threshold: float,
) -> bool:
    if trigger == "off":
        return False
    baseline_triggered = _should_run_history_rescue(
        selected_repair=selected_repair,
        baseline_record=baseline_record,
        source_controls=source_controls,
    )
    disagreement_triggered = _should_run_selector_disagreement_rescue(
        selected_repair=selected_repair,
        baseline_record=baseline_record,
        repair_pool=repair_pool,
        source_controls=source_controls,
        task_prompt=task_prompt,
        task_answer_type=task_answer_type,
        exact_task_trajectory_policy=exact_task_trajectory_policy,
        trajectory_selector=trajectory_selector,
    )
    source_quality_triggered = (
        _source_control_allowed(baseline_record, source_controls)
        and task_answer_type == "rubric"
        and _planning_quality_score(baseline_record, task_prompt) < source_quality_threshold
    )
    if trigger == "baseline":
        return baseline_triggered
    if trigger == "source_quality":
        return source_quality_triggered
    if trigger == "baseline_or_source_quality":
        return baseline_triggered or source_quality_triggered
    if trigger == "selector_disagreement":
        return disagreement_triggered
    if trigger == "baseline_or_selector_disagreement":
        return baseline_triggered or disagreement_triggered
    raise ValueError(f"Unsupported prompt-guided rescue trigger: {trigger}")


def _should_run_constraint_gap_rescue(
    *,
    trigger: str,
    selected_repair: dict[str, object],
    baseline_record: dict[str, object],
    source_controls: list[str],
    task_prompt: str,
    task_answer_type: str,
    min_terms: int,
    source_quality_floor: float,
    source_quality_ceiling: float,
) -> bool:
    if trigger == "off":
        return False
    if min_terms < 0:
        raise ValueError("min_terms must be non-negative")
    if source_quality_floor > source_quality_ceiling:
        raise ValueError("source_quality_floor must be <= source_quality_ceiling")
    baseline_triggered = _should_run_history_rescue(
        selected_repair=selected_repair,
        baseline_record=baseline_record,
        source_controls=source_controls,
    )
    prompt_gap_triggered = (
        _source_control_allowed(baseline_record, source_controls)
        and task_answer_type == "rubric"
        and source_quality_floor
        <= _planning_quality_score(baseline_record, task_prompt)
        <= source_quality_ceiling
        and len(_prompt_constraint_gap_terms(task_prompt, str(baseline_record.get("text", "")))) >= min_terms
    )
    if trigger == "prompt_gap":
        return prompt_gap_triggered
    if trigger == "baseline_or_prompt_gap":
        return baseline_triggered or prompt_gap_triggered
    raise ValueError(f"Unsupported constraint-gap rescue trigger: {trigger}")


def _should_run_selector_disagreement_rescue(
    *,
    selected_repair: dict[str, object],
    baseline_record: dict[str, object],
    repair_pool: list[dict[str, object]],
    source_controls: list[str],
    task_prompt: str,
    task_answer_type: str,
    exact_task_trajectory_policy: str,
    trajectory_selector: str,
) -> bool:
    if not _source_control_allowed(baseline_record, source_controls):
        return False
    if task_answer_type != "rubric" and exact_task_trajectory_policy != "trajectory":
        return False
    repair_candidates = [record for record in repair_pool if _repair_name(record)]
    if not repair_candidates:
        return False
    trajectory_choice = max(
        repair_pool,
        key=lambda record: _selection_score(
            record,
            task_prompt,
            task_answer_type,
            trajectory_selector,
        ),
    )
    if not _repair_name(trajectory_choice):
        return False
    return _record_identity(trajectory_choice) != _record_identity(selected_repair)


def _source_control_allowed(
    baseline_record: dict[str, object],
    source_controls: list[str],
) -> bool:
    if not source_controls:
        return True
    return _control_name(baseline_record) in set(source_controls)


def _generate_repair_records(
    backend: HFDiffusionBackend,
    task: GeneralReasoningTask,
    *,
    source_record: dict[str, object],
    repairs: tuple[Any, ...],
    generation_seed_base: int,
    raw_output: Path,
    all_records: list[dict[str, object]],
    phase_source_history_char_ratio_min: float = PHASE_SOURCE_HISTORY_CHAR_RATIO_MIN,
    phase_source_target_similarity_min: float = PHASE_SOURCE_TARGET_SIMILARITY_MIN,
    phase_source_text_similarity_min: float = PHASE_SOURCE_TEXT_SIMILARITY_MIN,
) -> list[dict[str, object]]:
    source_token_ids = _int_list(source_record.get("generated_token_ids"))
    if not source_token_ids:
        return []
    source_confidences = _float_or_none_list(source_record.get("generated_token_confidences"))
    source_text = str(source_record.get("text", ""))
    source_quality_score = _planning_quality_score(source_record, task.prompt)
    token_decoder = _backend_token_decoder(backend)
    history_samples_token_ids = _history_samples_token_ids(source_record)
    default_history_source = _selected_history_repair_sample(source_record, task.prompt)
    max_new_tokens = task.max_new_tokens or len(source_token_ids)
    records = []
    for repair in repairs:
        configured_source_state = str(getattr(repair, "source_state", "final"))
        anchor_selection: dict[str, object] = {"anchor_choice": "final", "features": {}, "reason": ""}
        if configured_source_state in PRE_GENERATION_ANCHOR_SOURCE_STATES:
            anchor_selection = _choose_pre_generation_repair_anchor(
                source_record,
                task.prompt,
                search_history=configured_source_state == PRE_GENERATION_ANCHOR_SEARCH_SOURCE_STATE,
                phase_anchor=configured_source_state == PRE_GENERATION_PHASE_ANCHOR_SOURCE_STATE,
                phase_hybrid=configured_source_state == PRE_GENERATION_PHASE_HYBRID_SOURCE_STATE,
                phase_source_history_char_ratio_min=phase_source_history_char_ratio_min,
                phase_source_target_similarity_min=phase_source_target_similarity_min,
                phase_source_text_similarity_min=phase_source_text_similarity_min,
            )
        source_state = (
            str(anchor_selection.get("anchor_choice", "final"))
            if configured_source_state in PRE_GENERATION_ANCHOR_SOURCE_STATES
            else configured_source_state
        )
        execution_repair = _anchor_selected_execution_repair(
            repair,
            configured_source_state=configured_source_state,
            resolved_source_state=source_state,
        )
        if _is_phase_final_preserve_repair(repair):
            execution_repair = _phase_final_preserve_execution_repair(repair)
        history_token_ids = None
        history_source: dict[str, object] | None = None
        repair_source_token_ids = source_token_ids
        repair_source_text = source_text
        if source_state == "history":
            history_source = _dict(anchor_selection.get("history_sample")) or default_history_source
            if history_source is None:
                continue
            history_token_ids = _int_list(history_source.get("generated_token_ids"))
            if not history_token_ids:
                continue
            repair_source_token_ids = history_token_ids
            repair_source_text = str(history_source.get("visible_text") or "")
            if not repair_source_text.strip() and token_decoder is not None:
                repair_source_text = token_decoder(history_token_ids)
        repair_seed_name = str(getattr(execution_repair, "name", ""))
        generation_seed = _stable_generation_seed(
            generation_seed_base,
            str(source_record.get("candidate_key", "")),
            task.task_id,
            f"{_control_name(source_record)}:{repair_seed_name}",
        )
        _set_generation_seed(generation_seed)
        prompt_constraint_gap_terms = _prompt_constraint_gap_terms(
            task.prompt,
            repair_source_text,
            execution_repair,
        )
        planning_span_targets: list[str] = []
        planning_span_target_scores: list[dict[str, object]] = []
        span_seed_diagnostics: dict[str, object] = {}
        history_instability_gate: dict[str, object] = {
            "active": getattr(execution_repair, "remask_history_unstable_fraction", None) is not None,
            "policy": getattr(execution_repair, "history_instability_gate_policy", None),
            "reason": "ungated_instability"
            if getattr(execution_repair, "remask_history_unstable_fraction", None) is not None
            else "disabled",
        }
        effective_history_instability_remask_fraction = getattr(
            execution_repair,
            "remask_history_unstable_fraction",
            None,
        )
        configured_history_instability_gate_prompt_policy = getattr(
            repair,
            "history_instability_gate_prompt_policy",
            None,
        )
        configured_planning_prompt_gate_policy = getattr(
            repair,
            "planning_prompt_gate_policy",
            None,
        )
        effective_history_instability_gate_prompt_policy = None
        effective_planning_prompt_gate_policy = None
        planning_seed_suffix_anchor: dict[str, object] = {
            "active": False,
            "reason": "disabled",
        }
        planning_prompt_gate: dict[str, object] = {
            "active": False,
            "policy": configured_planning_prompt_gate_policy,
            "reason": "disabled" if configured_planning_prompt_gate_policy is None else "not_evaluated",
        }
        prompt_execution_repair = execution_repair
        planning_span_chunk_mode = str(getattr(execution_repair, "planning_span_chunk_mode", "sentence"))
        planning_span_selection_policy = str(getattr(execution_repair, "planning_span_selection_policy", "top_ranked"))
        if _is_planning_span_repair(execution_repair):
            planning_span_targets = _planning_constraint_gap_span_targets(
                task.prompt,
                repair_source_text,
                prompt_constraint_gap_terms,
                chunk_mode=planning_span_chunk_mode,
                selection_policy=planning_span_selection_policy,
            )
            planning_span_target_scores = _planning_constraint_gap_span_target_scores(
                task.prompt,
                repair_source_text,
                prompt_constraint_gap_terms,
                chunk_mode=planning_span_chunk_mode,
                selection_policy=planning_span_selection_policy,
            )
            history_instability_gate = _history_instability_gate_decision(
                execution_repair,
                planning_span_targets=planning_span_targets,
                source_quality_score=source_quality_score,
                source_state=source_state,
            )
            planning_prompt_gate = _planning_prompt_gate_decision(
                execution_repair,
                task_prompt=task.prompt,
                prompt_constraint_gap_terms=prompt_constraint_gap_terms,
                planning_span_targets=planning_span_targets,
                source_quality_score=source_quality_score,
                source_state=source_state,
            )
            if not bool(history_instability_gate.get("active", False)):
                effective_history_instability_remask_fraction = None
            elif configured_history_instability_gate_prompt_policy == "active_instability_instruction":
                effective_history_instability_gate_prompt_policy = configured_history_instability_gate_prompt_policy
                prompt_execution_repair = replace(
                    execution_repair,
                    prompt_repair_instruction=getattr(
                        repair,
                        "prompt_repair_instruction",
                        getattr(execution_repair, "prompt_repair_instruction", None),
                    ),
                )
            elif configured_history_instability_gate_prompt_policy is not None:
                raise ValueError(
                    "Unsupported history instability gate prompt policy: "
                    f"{configured_history_instability_gate_prompt_policy}"
                )
            if (
                not bool(history_instability_gate.get("active", False))
                and bool(planning_prompt_gate.get("active", False))
            ):
                planning_gate_instruction = getattr(repair, "planning_prompt_gate_instruction", None)
                if not planning_gate_instruction:
                    raise ValueError(
                        "Planning prompt gate is active but no planning_prompt_gate_instruction is configured"
                    )
                effective_planning_prompt_gate_policy = configured_planning_prompt_gate_policy
                prompt_execution_repair = replace(
                    execution_repair,
                    prompt_repair_instruction=planning_gate_instruction,
                )
            span_repair = DiffusionVerifierRepairCandidate(
                name=execution_repair.name,
                context_window=getattr(execution_repair, "text_context_window", 0),
                steps=getattr(execution_repair, "steps", 32),
                temperature=getattr(execution_repair, "temperature", 0.0),
                block_length=getattr(execution_repair, "block_length", None),
                remasking=getattr(execution_repair, "remasking", "low_confidence"),
                history_sample_count=getattr(execution_repair, "history_sample_count", 6),
                history_instability_remask_fraction=effective_history_instability_remask_fraction,
            )
            config = span_repair.to_text_span_config(
                repair_source_token_ids,
                target_texts=planning_span_targets,
                max_new_tokens=max_new_tokens,
                history_samples_token_ids=history_samples_token_ids,
                source_text=repair_source_text,
                token_decoder=token_decoder,
                fallback_tail_window=12,
            )
            span_seed_diagnostics = build_text_span_repair_seed_diagnostics(
                repair_source_token_ids,
                target_texts=planning_span_targets,
                max_new_tokens=max_new_tokens,
                history_instability_remask_fraction=effective_history_instability_remask_fraction,
                history_samples_token_ids=history_samples_token_ids,
                source_text=repair_source_text,
                token_decoder=token_decoder,
                context_window=getattr(execution_repair, "text_context_window", 0),
                fallback_tail_window=12,
            )
            seed_suffix_text, seed_suffix_diagnostics = _planning_prompt_gate_seed_suffix_text(
                prompt_execution_repair,
                task_prompt=task.prompt,
                prompt_constraint_gap_terms=prompt_constraint_gap_terms,
                rubric_items=task.rubric_items,
            )
            config, planning_seed_suffix_anchor = _apply_planning_seed_suffix_anchor(
                config,
                seed_suffix_text=seed_suffix_text,
                token_encoder=_backend_token_encoder(backend),
                active=bool(planning_prompt_gate.get("active", False)),
            )
            planning_seed_suffix_anchor = {
                **seed_suffix_diagnostics,
                **planning_seed_suffix_anchor,
            }
        else:
            config = execution_repair.to_config(
                source_token_ids,
                max_new_tokens=max_new_tokens,
                token_confidences=source_confidences,
                history_token_ids=history_token_ids,
                history_samples_token_ids=history_samples_token_ids,
                source_text=repair_source_text,
                token_decoder=token_decoder,
                source_quality_score=source_quality_score,
                history_selection_score=_history_float(history_source, "selection_score"),
                history_mask_count=_history_int(history_source, "mask_count"),
        )
        if _is_planning_span_repair(execution_repair):
            history_contrast = (
                _planning_span_history_contrast(
                    source_record,
                    task_prompt=task.prompt,
                    source_text=repair_source_text,
                    span_targets=planning_span_targets,
                )
                if getattr(execution_repair, "prompt_history_contrast", False)
                else ""
            )
            prompt_override = _planning_span_repair_prompt_override(
                task.prompt,
                repair_source_text,
                prompt_constraint_gap_terms,
                planning_span_targets,
                prompt_execution_repair,
                history_contrast=history_contrast,
                planning_prompt_gate_active=bool(planning_prompt_gate.get("active", False)),
            )
        else:
            history_contrast = ""
            prompt_override = _repair_prompt_override(task.prompt, repair_source_text, prompt_execution_repair)
        repair_metadata = {
            **repair.to_dict(),
            "source_control": _control_name(source_record),
            "source_state": source_state,
            "configured_source_state": configured_source_state,
            "execution_repair_name": str(getattr(execution_repair, "name", "")),
            "generation_seed_repair_name": repair_seed_name,
            "source_generation_stage": source_record.get("generation_stage"),
            "source_task_score": _task_score(source_record),
            "source_planning_quality_score": source_quality_score,
            "source_selector_score": _selector_score(source_record),
            "source_trajectory_score": _trajectory_score(source_record),
            "repair_source_text_chars": len(repair_source_text.strip()),
            "source_history_step": history_source.get("step") if source_state == "history" and history_source else None,
            "source_history_score": history_source.get("selection_score")
            if source_state == "history" and history_source
            else None,
            "source_history_visible_chars": history_source.get("visible_chars")
            if source_state == "history" and history_source
            else None,
            "source_history_mask_count": history_source.get("mask_count")
            if source_state == "history" and history_source
            else None,
            "seed_masked_positions": _masked_seed_count(config.initial_suffix_token_ids),
            "uses_prompt_repair_instruction": prompt_override is not None,
            "configured_prompt_repair_instruction": getattr(repair, "prompt_repair_instruction", None),
            "prompt_repair_instruction": getattr(prompt_execution_repair, "prompt_repair_instruction", None),
            "prompt_repair_policy": getattr(prompt_execution_repair, "prompt_repair_policy", None),
            "prompt_constraint_gap_terms": prompt_constraint_gap_terms,
            "planning_prompt_gate_policy": configured_planning_prompt_gate_policy,
            "planning_prompt_gate_active": bool(planning_prompt_gate.get("active", False)),
            "planning_prompt_gate_reason": str(planning_prompt_gate.get("reason", "")),
            "effective_planning_prompt_gate_policy": effective_planning_prompt_gate_policy,
            "planning_seed_suffix_anchor": planning_seed_suffix_anchor,
        }
        if configured_source_state in PRE_GENERATION_ANCHOR_SOURCE_STATES:
            anchor_features = anchor_selection.get("features")
            repair_metadata["anchor_selection_policy"] = configured_source_state
            repair_metadata["anchor_selection_reason"] = str(anchor_selection.get("reason", ""))
            repair_metadata["anchor_selection_features"] = (
                anchor_features if isinstance(anchor_features, dict) else {}
            )
        if _is_planning_span_repair(execution_repair):
            repair_metadata["uses_planning_span_revision"] = True
            repair_metadata["planning_span_targets"] = planning_span_targets
            repair_metadata["planning_span_chunk_mode"] = planning_span_chunk_mode
            repair_metadata["planning_span_selection_policy"] = planning_span_selection_policy
            repair_metadata["history_instability_gate_policy"] = history_instability_gate.get("policy")
            repair_metadata["history_instability_gate_active"] = bool(
                history_instability_gate.get("active", False)
            )
            repair_metadata["history_instability_gate_reason"] = str(
                history_instability_gate.get("reason", "")
            )
            repair_metadata["history_instability_gate_prompt_policy"] = (
                configured_history_instability_gate_prompt_policy
            )
            repair_metadata["effective_history_instability_gate_prompt_policy"] = (
                effective_history_instability_gate_prompt_policy
            )
            repair_metadata["effective_history_instability_remask_fraction"] = (
                effective_history_instability_remask_fraction
            )
            if getattr(execution_repair, "prompt_history_contrast", False):
                repair_metadata["uses_history_contrast_prompt"] = True
                repair_metadata["history_contrast_text"] = history_contrast
            repair_metadata["planning_span_target_scores"] = planning_span_target_scores
            repair_metadata["span_seed_diagnostics"] = span_seed_diagnostics
            repair_metadata["span_localization_mode"] = str(span_seed_diagnostics.get("mode", ""))
            repair_metadata["span_literal_target_found"] = bool(
                span_seed_diagnostics.get("literal_target_found", False)
            )
            repair_metadata["span_fallback_used"] = bool(span_seed_diagnostics.get("used_fallback", False))
        record = _generate_record(
            backend,
            task,
            config=config,
            schedule=None,
            stage="repair_candidate",
            generation_seed=generation_seed,
            prompt_override=prompt_override,
            repair=repair_metadata,
        )
        records.append(record)
        all_records.append(record)
        _append_jsonl(raw_output, record)
        _print_generation(record)
    return records


def _anchor_selected_execution_repair(
    repair: Any,
    *,
    configured_source_state: str,
    resolved_source_state: str,
) -> Any:
    if configured_source_state in PRE_GENERATION_ANCHOR_SOURCE_STATES:
        selected_repair = (
            default_llada_constraint_span_history_repair_candidates()[0]
            if resolved_source_state == "history"
            else default_llada_constraint_span_repair_candidates()[0]
        )
        if (
            getattr(repair, "remask_history_unstable_fraction", None) is None
            and getattr(repair, "history_instability_gate_prompt_policy", None) is None
            and getattr(repair, "planning_prompt_gate_policy", None) is None
        ):
            return selected_repair
        gate_policy = getattr(repair, "history_instability_gate_policy", None)
        return replace(
            selected_repair,
            name=selected_repair.name if gate_policy else str(getattr(repair, "name", selected_repair.name)),
            remask_history_unstable_fraction=getattr(repair, "remask_history_unstable_fraction", None),
            remask_text_policy=getattr(repair, "remask_text_policy", selected_repair.remask_text_policy),
            text_context_window=getattr(repair, "text_context_window", selected_repair.text_context_window),
            fallback_remask_low_confidence_fraction=getattr(
                repair,
                "fallback_remask_low_confidence_fraction",
                selected_repair.fallback_remask_low_confidence_fraction,
            ),
            prompt_repair_policy=getattr(repair, "prompt_repair_policy", selected_repair.prompt_repair_policy),
            prompt_repair_instruction=selected_repair.prompt_repair_instruction
            if gate_policy
            else getattr(repair, "prompt_repair_instruction", selected_repair.prompt_repair_instruction),
            history_instability_gate_policy=gate_policy,
            history_instability_gate_prompt_policy=getattr(
                repair,
                "history_instability_gate_prompt_policy",
                selected_repair.history_instability_gate_prompt_policy,
            ),
            planning_prompt_gate_policy=getattr(
                repair,
                "planning_prompt_gate_policy",
                selected_repair.planning_prompt_gate_policy,
            ),
            planning_prompt_gate_instruction=getattr(
                repair,
                "planning_prompt_gate_instruction",
                selected_repair.planning_prompt_gate_instruction,
            ),
            planning_prompt_gate_seed_suffix_text=getattr(
                repair,
                "planning_prompt_gate_seed_suffix_text",
                selected_repair.planning_prompt_gate_seed_suffix_text,
            ),
            planning_prompt_gate_seed_suffix_policy=getattr(
                repair,
                "planning_prompt_gate_seed_suffix_policy",
                selected_repair.planning_prompt_gate_seed_suffix_policy,
            ),
            planning_span_chunk_mode=getattr(
                repair,
                "planning_span_chunk_mode",
                selected_repair.planning_span_chunk_mode,
            ),
            planning_span_selection_policy=getattr(
                repair,
                "planning_span_selection_policy",
                selected_repair.planning_span_selection_policy,
            ),
            steps=getattr(repair, "steps", selected_repair.steps),
            temperature=getattr(repair, "temperature", selected_repair.temperature),
            block_length=getattr(repair, "block_length", selected_repair.block_length),
            remasking=getattr(repair, "remasking", selected_repair.remasking),
            history_sample_count=getattr(repair, "history_sample_count", selected_repair.history_sample_count),
        )
    return repair


def _is_phase_final_preserve_repair(repair: Any) -> bool:
    return str(getattr(repair, "name", "")) == PHASE_FINAL_PRESERVE_REPAIR_NAME


def _phase_final_preserve_execution_repair(repair: Any) -> Any:
    return _anchor_selected_execution_repair(
        repair,
        configured_source_state=PRE_GENERATION_ANCHOR_SOURCE_STATE,
        resolved_source_state="final",
    )


def _generate_exact_answer_repair_records(
    backend: HFDiffusionBackend,
    task: GeneralReasoningTask,
    *,
    source_record: dict[str, object],
    limit: int,
    exact_self_repair: bool,
    generation_seed_base: int,
    raw_output: Path,
    all_records: list[dict[str, object]],
    exact_verifier_revision: bool = False,
) -> list[dict[str, object]]:
    extracted_answer = _nested_value(source_record, ("task_score", "extracted_answer"))
    proposals = counterfactual_answer_proposals(task, extracted_answer, limit=limit)
    if not proposals and not (exact_self_repair and _label_free_exact_answer_supported(task)):
        return []
    base_schedule = default_llada_schedules(max_new_tokens=task.max_new_tokens or 32)[0]
    base_config = _replace_max_tokens(base_schedule.to_config(), task.max_new_tokens or 32)
    records = []
    source_token_ids = _int_list(source_record.get("generated_token_ids"))
    should_run_answer_span_revision = (
        exact_verifier_revision
        and source_token_ids
        and extracted_answer is not None
        and (
            (proposals and limit > 1)
            or (not proposals and exact_self_repair and _supports_no_proposal_answer_span_revision(task))
        )
    )
    if should_run_answer_span_revision:
        repair = default_llada_verifier_repair_candidates()[0]
        proposal = proposals[0] if proposals else None
        proposal_label = f"{proposal.source}:{proposal.value}" if proposal is not None else "label-free"
        generation_seed = _stable_generation_seed(
            generation_seed_base,
            str(source_record.get("candidate_key", "")),
            task.task_id,
            f"{_control_name(source_record)}:verifier-answer-span:{proposal_label}",
        )
        _set_generation_seed(generation_seed)
        max_new_tokens = task.max_new_tokens or len(source_token_ids) or 32
        config = repair.to_answer_span_config(
            source_token_ids,
            answer_text=extracted_answer,
            max_new_tokens=max_new_tokens,
            source_text=str(source_record.get("text", "")),
            token_decoder=_backend_token_decoder(backend),
        )
        span_seed_diagnostics = build_text_span_repair_seed_diagnostics(
            source_token_ids,
            target_texts=[extracted_answer],
            max_new_tokens=max_new_tokens,
            source_text=str(source_record.get("text", "")),
            token_decoder=_backend_token_decoder(backend),
        )
        record = _generate_record(
            backend,
            task,
            config=config,
            schedule=None,
            stage="exact_answer_repair_candidate",
            generation_seed=generation_seed,
            prompt_override=None,
            repair={
                **repair.to_dict(),
                "source_control": _control_name(source_record),
                "source_state": "final",
                "source_generation_stage": source_record.get("generation_stage"),
                "source_task_score": _task_score(source_record),
                "source_selector_score": _selector_score(source_record),
                "source_trajectory_score": _trajectory_score(source_record),
                "source_extracted_answer": extracted_answer,
                **(
                    {
                        "proposal": proposal.value,
                        "proposal_source": proposal.source,
                        "proposal_task_score": score_task_output(task, proposal.value).score,
                    }
                    if proposal is not None
                    else {
                        "proposal": None,
                        "proposal_source": None,
                        "proposal_task_score": None,
                    }
                ),
                "seed_masked_positions": _masked_seed_count(config.initial_suffix_token_ids),
                "span_seed_diagnostics": span_seed_diagnostics,
                "span_localization_mode": str(span_seed_diagnostics.get("mode", "")),
                "span_literal_target_found": bool(span_seed_diagnostics.get("literal_target_found", False)),
                "span_fallback_used": bool(span_seed_diagnostics.get("used_fallback", False)),
                "uses_verifier_answer_span_revision": True,
                "uses_label_free_verifier_span_revision": proposal is None,
                "uses_counterfactual_prompt": False,
            },
        )
        _attach_exact_self_repair_metadata(task, record, extracted_answer)
        records.append(record)
        all_records.append(record)
        _append_jsonl(raw_output, record)
        _print_generation(record)
    for proposal in proposals[: max(0, limit - len(records))]:
        generation_seed = _stable_generation_seed(
            generation_seed_base,
            str(source_record.get("candidate_key", "")),
            task.task_id,
            f"{_control_name(source_record)}:counterfactual:{proposal.source}:{proposal.value}",
        )
        _set_generation_seed(generation_seed)
        record = _generate_record(
            backend,
            task,
            config=base_config,
            schedule=None,
            stage="exact_answer_repair_candidate",
            generation_seed=generation_seed,
            prompt_override=_counterfactual_prompt(task, extracted_answer, proposal.value),
            repair={
                "name": "counterfactual_answer_proposal",
                "source_control": _control_name(source_record),
                "source_state": "final",
                "source_generation_stage": source_record.get("generation_stage"),
                "source_task_score": _task_score(source_record),
                "source_selector_score": _selector_score(source_record),
                "source_trajectory_score": _trajectory_score(source_record),
                "source_extracted_answer": extracted_answer,
                "proposal": proposal.value,
                "proposal_source": proposal.source,
                "proposal_task_score": score_task_output(task, proposal.value).score,
                "seed_masked_positions": 0,
                "uses_counterfactual_prompt": True,
            },
        )
        records.append(record)
        all_records.append(record)
        _append_jsonl(raw_output, record)
        _print_generation(record)
    if not proposals and exact_self_repair and _label_free_exact_answer_supported(task) and len(records) < limit:
        self_config = _exact_self_repair_config(base_config)
        generation_seed = _stable_generation_seed(
            generation_seed_base,
            str(source_record.get("candidate_key", "")),
            task.task_id,
            f"{_control_name(source_record)}:exact-self-repair",
        )
        _set_generation_seed(generation_seed)
        record = _generate_record(
            backend,
            task,
            config=self_config,
            schedule=None,
            stage="exact_answer_repair_candidate",
            generation_seed=generation_seed,
            prompt_override=_exact_self_repair_prompt(task, extracted_answer),
            repair={
                "name": "self_check_answer_repair",
                "source_control": _control_name(source_record),
                "source_state": "final",
                "source_generation_stage": source_record.get("generation_stage"),
                "source_task_score": _task_score(source_record),
                "source_selector_score": _selector_score(source_record),
                "source_trajectory_score": _trajectory_score(source_record),
                "source_extracted_answer": extracted_answer,
                "seed_masked_positions": 0,
                "uses_self_repair_prompt": True,
            },
        )
        _attach_exact_self_repair_metadata(task, record, extracted_answer)
        records.append(record)
        all_records.append(record)
        _append_jsonl(raw_output, record)
        _print_generation(record)
        inconsistencies = _arithmetic_claim_inconsistencies(str(record.get("text", "")))
        if inconsistencies and len(records) < limit:
            if exact_verifier_revision:
                span_record = _generate_arithmetic_contradiction_span_repair_record(
                    backend,
                    task,
                    source_record=source_record,
                    failed_repair_record=record,
                    inconsistencies=inconsistencies,
                    extracted_answer=extracted_answer,
                    generation_seed_base=generation_seed_base,
                    raw_output=raw_output,
                    all_records=all_records,
                )
                if span_record is not None:
                    records.append(span_record)
                    if _exact_answer_repair_selection_score(span_record, task.answer_type, task.prompt) > 0.0:
                        return records
            if len(records) >= limit:
                return records
            feedback_seed = _stable_generation_seed(
                generation_seed_base,
                str(source_record.get("candidate_key", "")),
                task.task_id,
                f"{_control_name(source_record)}:exact-arithmetic-feedback",
            )
            _set_generation_seed(feedback_seed)
            feedback_record = _generate_record(
                backend,
                task,
                config=self_config,
                schedule=None,
                stage="exact_answer_repair_candidate",
                generation_seed=feedback_seed,
                prompt_override=_exact_arithmetic_feedback_prompt(task, record, inconsistencies),
                repair={
                    "name": "arithmetic_feedback_repair",
                    "source_control": _control_name(source_record),
                    "source_state": "final",
                    "source_generation_stage": source_record.get("generation_stage"),
                    "source_task_score": _task_score(source_record),
                    "source_selector_score": _selector_score(source_record),
                    "source_trajectory_score": _trajectory_score(source_record),
                    "source_extracted_answer": extracted_answer,
                    "feedback_from": "self_check_answer_repair",
                    "arithmetic_feedback_claims": inconsistencies,
                    "seed_masked_positions": 0,
                    "uses_arithmetic_feedback_prompt": True,
                },
            )
            _attach_exact_self_repair_metadata(task, feedback_record, extracted_answer)
            records.append(feedback_record)
            all_records.append(feedback_record)
            _append_jsonl(raw_output, feedback_record)
            _print_generation(feedback_record)
        elif (
            task.answer_type == "integer"
            and _arithmetic_claim_count(str(record.get("text", ""))) == 0
            and len(records) < limit
        ):
            evidence_seed = _stable_generation_seed(
                generation_seed_base,
                str(source_record.get("candidate_key", "")),
                task.task_id,
                f"{_control_name(source_record)}:exact-arithmetic-evidence",
            )
            _set_generation_seed(evidence_seed)
            evidence_record = _generate_record(
                backend,
                task,
                config=self_config,
                schedule=None,
                stage="exact_answer_repair_candidate",
                generation_seed=evidence_seed,
                prompt_override=_exact_arithmetic_evidence_prompt(task, record),
                repair={
                    "name": "arithmetic_evidence_repair",
                    "source_control": _control_name(source_record),
                    "source_state": "final",
                    "source_generation_stage": source_record.get("generation_stage"),
                    "source_task_score": _task_score(source_record),
                    "source_selector_score": _selector_score(source_record),
                    "source_trajectory_score": _trajectory_score(source_record),
                    "source_extracted_answer": extracted_answer,
                    "feedback_from": "self_check_answer_repair",
                    "seed_masked_positions": 0,
                    "uses_arithmetic_evidence_prompt": True,
                },
            )
            _attach_exact_self_repair_metadata(task, evidence_record, extracted_answer)
            records.append(evidence_record)
            all_records.append(evidence_record)
            _append_jsonl(raw_output, evidence_record)
            _print_generation(evidence_record)
    return records


def _generate_arithmetic_contradiction_span_repair_record(
    backend: HFDiffusionBackend,
    task: GeneralReasoningTask,
    *,
    source_record: dict[str, object],
    failed_repair_record: dict[str, object],
    inconsistencies: list[dict[str, object]],
    extracted_answer: object | None,
    generation_seed_base: int,
    raw_output: Path,
    all_records: list[dict[str, object]],
) -> dict[str, object] | None:
    source_token_ids = _int_list(failed_repair_record.get("generated_token_ids"))
    if not source_token_ids:
        return None
    repair = DiffusionVerifierRepairCandidate(
        name="arithmetic_contradiction_span_repair",
        context_window=1,
        steps=64,
    )
    max_new_tokens = max(task.max_new_tokens or 0, len(source_token_ids))
    span_targets = _arithmetic_inconsistency_span_targets(
        inconsistencies,
        str(failed_repair_record.get("text", "")),
    )
    config = repair.to_text_span_config(
        source_token_ids,
        target_texts=span_targets,
        max_new_tokens=max_new_tokens,
        source_text=str(failed_repair_record.get("text", "")),
        token_decoder=_backend_token_decoder(backend),
        fallback_tail_window=4,
    )
    span_seed_diagnostics = build_text_span_repair_seed_diagnostics(
        source_token_ids,
        target_texts=span_targets,
        max_new_tokens=max_new_tokens,
        source_text=str(failed_repair_record.get("text", "")),
        token_decoder=_backend_token_decoder(backend),
        context_window=repair.context_window,
        fallback_tail_window=4,
    )
    generation_seed = _stable_generation_seed(
        generation_seed_base,
        str(source_record.get("candidate_key", "")),
        task.task_id,
        f"{_control_name(source_record)}:arithmetic-contradiction-span",
    )
    _set_generation_seed(generation_seed)
    record = _generate_record(
        backend,
        task,
        config=config,
        schedule=None,
        stage="exact_answer_repair_candidate",
        generation_seed=generation_seed,
        prompt_override=_exact_arithmetic_span_repair_prompt(task, failed_repair_record, inconsistencies),
        repair={
            **repair.to_dict(),
            "source_control": _control_name(source_record),
            "source_state": "final",
            "source_generation_stage": source_record.get("generation_stage"),
            "source_task_score": _task_score(source_record),
            "source_selector_score": _selector_score(source_record),
            "source_trajectory_score": _trajectory_score(source_record),
            "source_extracted_answer": extracted_answer,
            "feedback_from": "self_check_answer_repair",
            "arithmetic_feedback_claims": inconsistencies,
            "arithmetic_span_targets": span_targets,
            "seed_masked_positions": _masked_seed_count(config.initial_suffix_token_ids),
            "span_seed_diagnostics": span_seed_diagnostics,
            "span_localization_mode": str(span_seed_diagnostics.get("mode", "")),
            "span_literal_target_found": bool(span_seed_diagnostics.get("literal_target_found", False)),
            "span_fallback_used": bool(span_seed_diagnostics.get("used_fallback", False)),
            "uses_arithmetic_span_revision": True,
        },
    )
    _attach_exact_self_repair_metadata(task, record, extracted_answer)
    all_records.append(record)
    _append_jsonl(raw_output, record)
    _print_generation(record)
    return record


def _arithmetic_inconsistency_span_targets(
    inconsistencies: list[dict[str, object]],
    text: str = "",
) -> list[object]:
    targets: list[object] = []
    first_bad_claim_index = _first_inconsistent_arithmetic_claim_index(
        text,
        inconsistencies,
    )
    if first_bad_claim_index is not None:
        for expression, claimed_text in _arithmetic_claims(text)[first_bad_claim_index:]:
            expression = expression.strip()
            claimed_text = str(claimed_text).strip()
            if expression:
                targets.append(expression)
                if claimed_text:
                    targets.append(f"{expression} = {claimed_text}")
        final_answer = _last_integer_text(_final_answer_context(text))
        if final_answer:
            targets.append(f"Answer: {final_answer}")
    for item in inconsistencies:
        expression = str(item.get("expression", "")).strip()
        claimed = item.get("claimed")
        if expression:
            targets.append(expression)
            if claimed is not None:
                targets.append(f"{expression} = {_format_arithmetic_value(float(claimed))}")
    return _dedupe([str(target) for target in targets if str(target).strip()])


def _first_inconsistent_arithmetic_claim_index(
    text: str,
    inconsistencies: list[dict[str, object]],
) -> int | None:
    if not text or not inconsistencies:
        return None
    bad_expressions = {
        str(item.get("expression", "")).strip()
        for item in inconsistencies
        if str(item.get("expression", "")).strip()
    }
    for index, (expression, _claimed_text) in enumerate(_arithmetic_claims(text)):
        if expression.strip() in bad_expressions:
            return index
    return None


def _counterfactual_prompt(
    task: GeneralReasoningTask,
    extracted_answer: object | None,
    proposal: str,
) -> str:
    failed = "" if extracted_answer is None else f" The previous extracted answer was {extracted_answer!r}."
    return (
        f"{task.prompt}\n\n"
        f"A verifier rejected the previous answer.{failed} "
        f"Evaluate the alternative candidate {proposal!r}. "
        "If it is consistent with the problem, answer with only that candidate. "
        "Do not explain."
    )


def _exact_self_repair_prompt(
    task: GeneralReasoningTask,
    extracted_answer: object | None,
) -> str:
    return (
        f"{task.prompt}\n\n"
        "A verifier rejected a previous answer. "
        "Solve the problem again from scratch with brief scratch work. "
        "End with one final line in the form 'Answer: <final exact answer>'."
    )


def _exact_arithmetic_feedback_prompt(
    task: GeneralReasoningTask,
    failed_repair_record: dict[str, object],
    inconsistencies: list[dict[str, object]],
) -> str:
    feedback = "; ".join(
        (
            f"{item['expression']} equals {_format_arithmetic_value(float(item['computed']))}, "
            f"not {_format_arithmetic_value(float(item['claimed']))}"
        )
        for item in inconsistencies
    )
    failed_text = " ".join(str(failed_repair_record.get("text", "")).split())
    return (
        f"{task.prompt}\n\n"
        "A previous scratchpad made an arithmetic error. "
        f"Correction: {feedback}. "
        f"Previous scratchpad: {failed_text}\n\n"
        "Redo the calculation using the correction. "
        "End with one final line in the form 'Answer: <final exact answer>'."
    )


def _exact_arithmetic_span_repair_prompt(
    task: GeneralReasoningTask,
    failed_repair_record: dict[str, object],
    inconsistencies: list[dict[str, object]],
) -> str:
    feedback = "; ".join(
        (
            f"{item['expression']} equals {_format_arithmetic_value(float(item['computed']))}, "
            f"not {_format_arithmetic_value(float(item['claimed']))}"
        )
        for item in inconsistencies
    )
    return (
        f"{task.prompt}\n\n"
        "A verifier found an arithmetic contradiction in the masked draft span. "
        f"Correction: {feedback}. "
        "Denoise the masked span, preserve useful correct work, and update the "
        "final answer if needed. "
        "End with one final line in the form 'Answer: <final exact answer>'."
    )


def _exact_arithmetic_evidence_prompt(
    task: GeneralReasoningTask,
    failed_repair_record: dict[str, object],
) -> str:
    failed_text = " ".join(str(failed_repair_record.get("text", "")).split())
    return (
        f"{task.prompt}\n\n"
        "A previous repair gave an exact number but did not show checkable arithmetic. "
        f"Previous response: {failed_text}\n\n"
        "Solve again and show each calculation as an explicit equation using digits "
        "and +, -, *, or /. Ignore quantities that are irrelevant to the question. "
        "End with one final line in the form 'Answer: <final exact answer>'."
    )


def _attach_exact_self_repair_metadata(
    task: GeneralReasoningTask,
    record: dict[str, object],
    source_extracted_answer: object | None,
) -> None:
    repair = record.get("repair")
    if not isinstance(repair, dict):
        return
    text = str(record.get("text", ""))
    repair_answer = _label_free_exact_answer_from_text(task, text)
    source_answer = _normalize_exact_value(source_extracted_answer)
    normalized_repair_answer = _normalize_exact_value(repair_answer)
    repair["self_repair_extracted_answer"] = repair_answer
    repair["self_repair_changed_answer"] = bool(
        repair_answer is not None and normalized_repair_answer != source_answer
    )
    if task.answer_type == "integer":
        repair["self_repair_arithmetic_consistent"] = _arithmetic_claims_consistent(text)
        repair["self_repair_arithmetic_claim_count"] = _arithmetic_claim_count(text)
        repair["self_repair_irrelevant_number_used"] = _repair_irrelevant_prompt_number_used(
            record,
            task.prompt,
        )
        repair["self_repair_irrelevant_numbers"] = sorted(_prompt_irrelevant_numbers(task.prompt))
        repair["self_repair_required_operators"] = sorted(_prompt_required_arithmetic_operators(task.prompt))
        repair["self_repair_missing_required_operators"] = sorted(
            _repair_missing_required_operators(record, task.prompt)
        )
        repair["self_repair_quantity_role_gaps"] = sorted(_repair_quantity_role_gaps(record, task.prompt))
        repair["self_repair_arithmetic_provenance_gaps"] = sorted(
            _repair_arithmetic_provenance_gaps(record, task.prompt)
        )
        repair["self_repair_final_answer_role_gaps"] = sorted(
            _repair_final_answer_role_gaps(record, task.prompt)
        )
        repair["self_repair_final_answer_object_gaps"] = sorted(
            _repair_final_answer_object_gaps(record, task.prompt)
        )
        repair["self_repair_final_answer_target_gaps"] = sorted(
            _repair_final_answer_target_gaps(record, task.prompt)
        )
    else:
        repair["self_repair_arithmetic_consistent"] = True
        repair["self_repair_arithmetic_claim_count"] = 0
        repair["self_repair_irrelevant_number_used"] = False
        repair["self_repair_irrelevant_numbers"] = []
        repair["self_repair_required_operators"] = []
        repair["self_repair_missing_required_operators"] = []
        repair["self_repair_quantity_role_gaps"] = []
        repair["self_repair_arithmetic_provenance_gaps"] = []
        repair["self_repair_final_answer_role_gaps"] = []
        repair["self_repair_final_answer_object_gaps"] = []
        repair["self_repair_final_answer_target_gaps"] = []
    repair["self_repair_short_text_symbolic_gaps"] = sorted(
        _repair_short_text_symbolic_gaps(record, task.prompt)
    )
    repair["self_repair_short_text_trace_gaps"] = sorted(
        _repair_short_text_trace_gaps(record, task.prompt)
    )


def _exact_self_repair_config(config: Any) -> Any:
    max_new_tokens = max(64, int(getattr(config, "max_new_tokens", 32) or 32))
    block_length = max_new_tokens if getattr(config, "algorithm", "") == "low_confidence" else config.block_length
    return replace(
        config,
        max_new_tokens=max_new_tokens,
        steps=max(64, int(getattr(config, "steps", 32) or 32)),
        block_length=block_length,
    )


def _generate_record(
    backend: HFDiffusionBackend,
    task: GeneralReasoningTask,
    *,
    config: Any,
    schedule: dict[str, object] | None,
    stage: str,
    generation_seed: int,
    repair: dict[str, object] | None = None,
    counterfactual_probe: dict[str, object] | None = None,
    prompt_override: str | None = None,
) -> dict[str, object]:
    result = backend.generate(prompt_override or task.prompt, config=config)
    task_score = score_task_output(task, result.text)
    record = result.to_dict()
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    record["generation_stage"] = stage
    record["generation_seed"] = generation_seed
    record["task"] = {
        "task_id": task.task_id,
        "family": task.family,
        "answer_type": task.answer_type,
        "scorer": task.scorer,
        "answer": task.answer,
    }
    _attach_planning_quality_score(record, task)
    record["schedule"] = schedule
    if repair is not None:
        record["repair"] = repair
    if counterfactual_probe is not None:
        record["counterfactual_probe"] = counterfactual_probe
    record["task_score"] = task_score.to_dict()
    record = attach_control_score(record)
    record["combined_selection_score"] = _combined_score(record)
    return record


def _repair_prompt_override(
    task_prompt: str,
    source_text: str,
    repair: Any,
) -> str | None:
    instruction = getattr(repair, "prompt_repair_instruction", None)
    policy = getattr(repair, "prompt_repair_policy", None)
    if not instruction and policy != "constraint_gap":
        return None
    draft = _compact_text(source_text, max_chars=900)
    if policy == "constraint_gap":
        gap_terms = _prompt_constraint_gap_terms(task_prompt, source_text, repair)
        gap_text = ", ".join(gap_terms) if gap_terms else "none obvious from keyword coverage"
        instruction = instruction or (
            "Rewrite the draft answer directly. Add missing task-specific "
            "constraints, measurements, decision rules, risk controls, fallback "
            "paths, and stop conditions from the original task."
        )
        return (
            f"{task_prompt}\n\n"
            f"Draft answer to repair:\n{draft}\n\n"
            f"Missing or weak task terms to cover: {gap_text}\n\n"
            f"{instruction}\n"
            "Do not mention this checklist unless the terms belong naturally in the final answer."
        )
    return (
        f"{task_prompt}\n\n"
        f"Draft answer to repair:\n{draft}\n\n"
        f"{instruction}"
    )


def _prompt_constraint_gap_terms(
    task_prompt: str,
    source_text: str,
    repair: Any | None = None,
    *,
    limit: int = 12,
) -> list[str]:
    policy = getattr(repair, "prompt_repair_policy", None) if repair is not None else "constraint_gap"
    if policy != "constraint_gap":
        return []
    normalized_source = _normalize(source_text)
    priority_terms = [
        term.lower()
        for term in re.findall(
            r"\b(?:[A-Z]{2,}|\d+(?:[.,]\d+)?[a-zA-Z%$]*|[A-Za-z]+-[A-Za-z0-9-]+)\b",
            task_prompt,
        )
    ]
    missing_priority_terms = [
        term
        for term in priority_terms
        if term and _normalize(term) not in normalized_source
    ]
    missing_keywords = [
        keyword
        for keyword in _keywords(task_prompt)
        if keyword not in normalized_source
    ]
    return _dedupe([*missing_priority_terms, *missing_keywords])[:limit]


def _planning_constraint_gap_span_targets(
    task_prompt: str,
    source_text: str,
    gap_terms: list[str],
    *,
    limit: int = 3,
    chunk_mode: str = "sentence",
    selection_policy: str = "top_ranked",
) -> list[str]:
    """Pick decoded planning spans whose preservation blocks prompt-gap repair."""
    ranked_targets = _planning_constraint_gap_span_target_scores(
        task_prompt,
        source_text,
        gap_terms,
        limit=limit,
        chunk_mode=chunk_mode,
        selection_policy=selection_policy,
    )
    return [str(target["span"]) for target in ranked_targets]


def _planning_constraint_gap_span_target_scores(
    task_prompt: str,
    source_text: str,
    gap_terms: list[str],
    *,
    limit: int = 3,
    chunk_mode: str = "sentence",
    selection_policy: str = "top_ranked",
) -> list[dict[str, object]]:
    """Rank source spans for verifier-guided planning repair.

    The score is intentionally source-relative: a span is a better mask target
    when removing it does not damage the draft, when it relieves an explicit
    prompt-constraint violation, and when the span fails to cover the prompt
    terms that triggered the repair.
    """
    if chunk_mode == "adaptive":
        sentence_targets = _planning_constraint_gap_span_target_scores(
            task_prompt,
            source_text,
            gap_terms,
            limit=limit,
            chunk_mode="sentence",
            selection_policy=selection_policy,
        )
        retry_as_clauses = _should_retry_planning_span_targets_as_clauses(sentence_targets, source_text)
        refine_as_clauses = (
            _is_compact_planning_span_selection_policy(selection_policy)
            and _should_refine_planning_span_targets_as_clauses(sentence_targets, source_text)
        )
        if retry_as_clauses or refine_as_clauses:
            clause_targets = _planning_constraint_gap_span_target_scores(
                task_prompt,
                source_text,
                gap_terms,
                limit=limit,
                chunk_mode="clause",
                selection_policy=selection_policy,
            )
            if clause_targets and (
                retry_as_clauses
                or _compact_clause_targets_are_preferable(sentence_targets, clause_targets)
            ):
                return clause_targets
        return sentence_targets
    sentences = _planning_repair_chunks(source_text, chunk_mode=chunk_mode)
    if not sentences:
        return []
    source_surface = _planning_surface_v2_score(
        source_text,
        task_prompt,
        task_id="planning_span_source",
    )
    source_penalty = _planning_contradiction_penalty({"text": source_text}, task_prompt)
    normalized_prompt = _normalize(task_prompt)
    normalized_gap_terms = {_normalize(term) for term in gap_terms if _normalize(term)}
    prompt_keywords = set(_keywords(task_prompt))
    measurement_pressure = _contains_any(
        " ".join([normalized_prompt, *normalized_gap_terms]),
        (
            "measure",
            "measurement",
            "measurements",
            "metric",
            "metrics",
            "collect",
            "record",
            "compare",
        ),
    )
    risk_pressure = _contains_any(
        " ".join([normalized_prompt, *normalized_gap_terms]),
        ("risk", "risky", "fallback", "rollback", "fail", "failure", "stop", "threshold"),
    )
    weak_or_contradictory_phrases = (
        "valid comparison",
        "will not be available",
        "not be available",
        "still run the intervention",
        "still run",
        "ensuring you have a publishable result",
        "publishable result even if",
        "can then run",
        "skip the baseline",
        "without a baseline",
        "no baseline",
        "ship immediately",
        "delete anything",
        "remove anything",
    )
    scored: list[dict[str, object]] = []
    for index, sentence in enumerate(sentences):
        normalized_sentence = _normalize(sentence)
        if len(normalized_sentence.split()) < 3:
            continue
        without_sentence = _without_sentence(sentences, index)
        sentence_surface = _planning_surface_v2_score(
            sentence,
            task_prompt,
            task_id="planning_span_candidate",
        )
        without_surface = _planning_surface_v2_score(
            without_sentence,
            task_prompt,
            task_id="planning_span_without_candidate",
        )
        sentence_penalty = _planning_contradiction_penalty({"text": sentence}, task_prompt)
        without_penalty = _planning_contradiction_penalty({"text": without_sentence}, task_prompt)
        source_relative_preservation = max(
            0.0,
            min(1.0, 1.0 - max(0.0, source_surface - without_surface) / 0.20),
        )
        contradiction_relief = max(0.0, source_penalty - without_penalty)
        keyword_coverage = _prompt_keyword_coverage(task_prompt, normalized_sentence)
        if normalized_gap_terms:
            gap_hit_count = sum(1 for term in normalized_gap_terms if term in normalized_sentence)
            prompt_gap_miss = 1.0 - (gap_hit_count / len(normalized_gap_terms))
        else:
            prompt_gap_miss = 1.0 - keyword_coverage
        low_sentence_surface = max(0.0, 0.55 - sentence_surface)
        score = 0.0
        if index > 0:
            score += 0.40
        has_weak_phrase = _contains_any(normalized_sentence, weak_or_contradictory_phrases)
        if has_weak_phrase:
            score += 3.50
        score += 0.45 * source_relative_preservation
        score += 0.70 * contradiction_relief
        score += 0.35 * sentence_penalty
        score += 0.45 * prompt_gap_miss
        score += 0.35 * low_sentence_surface
        if measurement_pressure and not _contains_any(
            normalized_sentence,
            ("measure", "measurement", "metric", "collect", "record", "compare", "failure mode"),
        ):
            score += 1.10
        if risk_pressure and not _contains_any(
            normalized_sentence,
            ("risk", "risky", "fallback", "rollback", "fail", "failure", "stop", "threshold"),
        ):
            score += 0.50
        if prompt_keywords:
            keyword_hits = sum(1 for keyword in prompt_keywords if keyword in normalized_sentence)
            if keyword_hits <= 1:
                score += 0.65
        if index == len(sentences) - 1 and index > 0:
            score += 0.75
        if index == 0 and "baseline" in normalized_sentence and _contains_any(normalized_sentence, ("first", "reliable")):
            score -= 2.00
        if index == 0 and len(sentences) > 1 and not has_weak_phrase and sentence_penalty <= 0.0:
            score -= 2.00
        if score >= 1.25:
            scored.append(
                {
                    "span": sentence,
                    "index": index,
                    "score": round(score, 6),
                    "sentence_surface": round(sentence_surface, 6),
                    "without_surface": round(without_surface, 6),
                    "source_relative_preservation": round(source_relative_preservation, 6),
                    "contradiction_relief": round(contradiction_relief, 6),
                    "prompt_gap_miss": round(prompt_gap_miss, 6),
                    "keyword_coverage": round(keyword_coverage, 6),
                }
            )

    if _is_compact_planning_span_selection_policy(selection_policy):
        selected_indexes = _compact_planning_span_target_indexes(scored, sentences, limit=limit)
    else:
        selected_indexes = {
            int(target["index"])
            for target in sorted(
                scored,
                key=lambda item: (-float(item["score"]), int(item["index"])),
            )[:limit]
        }
        if len(gap_terms) >= 6 and len(sentences) > 1:
            for index in range(1, len(sentences)):
                if len(selected_indexes) >= min(limit, len(sentences) - 1):
                    break
                selected_indexes.add(index)
    if not selected_indexes and len(sentences) > 1:
        selected_indexes.add(len(sentences) - 1)
    elif not selected_indexes:
        selected_indexes.add(0)
    by_index = {int(target["index"]): target for target in scored}
    targets = []
    for index in sorted(selected_indexes):
        if not 0 <= index < len(sentences):
            continue
        target = dict(by_index.get(index) or {})
        if not target:
            target = {
                "span": sentences[index],
                "index": index,
                "score": 0.0,
                "sentence_surface": 0.0,
                "without_surface": 0.0,
                "source_relative_preservation": 0.0,
                "contradiction_relief": 0.0,
                "prompt_gap_miss": 0.0,
                "keyword_coverage": 0.0,
                "fallback": True,
            }
        targets.append(target)
    deduped = []
    seen_spans = set()
    for target in targets:
        span = str(target["span"])
        if span in seen_spans:
            continue
        seen_spans.add(span)
        deduped.append(target)
    return deduped[:limit]


def _is_compact_planning_span_selection_policy(selection_policy: str) -> bool:
    return selection_policy in {"compact", "compact_density"}


def _compact_planning_span_target_indexes(
    scored: list[dict[str, object]],
    sentences: list[str],
    *,
    limit: int,
) -> set[int]:
    """Select the smallest high-value denoise region instead of saturating the span limit."""
    ranked = sorted(
        scored,
        key=lambda item: (
            -_planning_span_score_density(item, sentences),
            -float(item.get("score", 0.0)),
            int(item.get("index", 0)),
        ),
    )
    if not ranked:
        return set()

    source_word_count = max(1, sum(_planning_span_word_count(sentence) for sentence in sentences))
    word_budget = max(14, min(56, math.ceil(source_word_count * 0.70)))
    top_score = float(ranked[0].get("score", 0.0))
    selected_indexes: set[int] = set()
    selected_words = 0
    for target in ranked:
        if len(selected_indexes) >= limit:
            break
        index = int(target.get("index", 0))
        if not 0 <= index < len(sentences):
            continue
        score = float(target.get("score", 0.0))
        word_count = _planning_span_word_count(sentences[index])
        if selected_indexes and score < max(1.25, top_score * 0.55):
            continue
        if selected_indexes and selected_words + word_count > word_budget + 4 and score < 3.25:
            continue
        selected_indexes.add(index)
        selected_words += word_count
    return selected_indexes


def _planning_span_score_density(target: dict[str, object], sentences: list[str]) -> float:
    index = int(target.get("index", 0))
    if not 0 <= index < len(sentences):
        return 0.0
    word_count = max(1, _planning_span_word_count(sentences[index]))
    return float(target.get("score", 0.0)) / math.sqrt(word_count)


def _planning_span_word_count(text: str) -> int:
    return len(re.findall(r"[a-z0-9]+", text.lower()))


def _should_refine_planning_span_targets_as_clauses(
    targets: list[dict[str, object]],
    source_text: str,
) -> bool:
    if not targets:
        return False
    clause_chunks = _planning_repair_chunks(source_text, chunk_mode="clause")
    sentence_chunks = _planning_repair_chunks(source_text, chunk_mode="sentence")
    if len(clause_chunks) <= len(sentence_chunks):
        return False
    for target in targets:
        span = str(target.get("span", ""))
        if _planning_span_should_stay_as_sentence(target, span):
            continue
        if _planning_span_word_count(span) >= 18 and len(_split_long_planning_chunk(span)) > 1:
            return True
    return False


def _planning_span_should_stay_as_sentence(target: dict[str, object], span: str) -> bool:
    """Keep structural planning units intact when clause repair would lose context."""
    normalized_span = _normalize(span)
    keyword_coverage = _float_value(target.get("keyword_coverage"), default=0.0)
    if keyword_coverage >= 0.45:
        return True
    if "decision rule" in normalized_span:
        return True
    return _contains_any(normalized_span, ("if accuracy", "if latency")) and ";" in span


def _compact_clause_targets_are_preferable(
    sentence_targets: list[dict[str, object]],
    clause_targets: list[dict[str, object]],
) -> bool:
    if not sentence_targets:
        return True
    sentence_words = sum(_planning_span_word_count(str(target.get("span", ""))) for target in sentence_targets)
    clause_words = sum(_planning_span_word_count(str(target.get("span", ""))) for target in clause_targets)
    if clause_words <= 0 or clause_words >= sentence_words:
        return False
    sentence_best = max(float(target.get("score", 0.0)) for target in sentence_targets)
    clause_best = max(float(target.get("score", 0.0)) for target in clause_targets)
    return clause_best >= max(1.25, sentence_best * 0.45)


def _should_retry_planning_span_targets_as_clauses(
    targets: list[dict[str, object]],
    source_text: str,
) -> bool:
    clause_chunks = _planning_repair_chunks(source_text, chunk_mode="clause")
    if len(clause_chunks) <= 1:
        return False
    if not targets:
        return True
    if len(targets) != 1:
        return False
    target = targets[0]
    target_span = _normalize(str(target.get("span", "")))
    source_span = _normalize(source_text)
    return bool(target.get("fallback")) or target_span == source_span


def _without_sentence(sentences: list[str], index: int) -> str:
    return " ".join(sentence for item_index, sentence in enumerate(sentences) if item_index != index)


def _planning_repair_chunks(text: str, *, chunk_mode: str = "sentence") -> list[str]:
    """Return literal source chunks suitable for span repair masking."""
    if chunk_mode == "sentence":
        return _planning_sentence_chunks(text)
    if chunk_mode != "clause":
        raise ValueError(f"Unsupported planning span chunk mode: {chunk_mode}")
    chunks: list[str] = []
    for sentence in _planning_sentence_chunks(text):
        chunks.extend(_split_long_planning_chunk(sentence))
    return chunks


def _split_long_planning_chunk(chunk: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", chunk.lower())
    if len(words) < 24:
        return [chunk]
    split_markers = {
        "also",
        "analyze",
        "check",
        "collect",
        "compare",
        "define",
        "document",
        "ensure",
        "evaluate",
        "if",
        "initiate",
        "prioritize",
        "record",
        "run",
        "then",
        "use",
        "verify",
    }
    split_after_markers = {
        "and",
        "but",
        "so",
        "then",
    }
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[,;]\s+", chunk):
        next_word_match = re.match(r"([A-Za-z][A-Za-z-]*)", chunk[match.end() :])
        next_word = next_word_match.group(1).lower() if next_word_match else ""
        previous_text = chunk[max(start, match.start() - 28) : match.start()].lower()
        should_split = (
            chunk[match.start()] == ";"
            or next_word in split_markers
            or any(previous_text.rstrip().endswith(f" {marker}") for marker in split_after_markers)
        )
        if not should_split:
            continue
        end = match.end()
        segment = chunk[start:end].strip()
        if _planning_chunk_is_useful(segment):
            spans.append((start, end))
        start = end
    tail = chunk[start:].strip()
    if spans and _planning_chunk_is_useful(tail):
        spans.append((start, len(chunk)))
    if len(spans) < 2:
        return [chunk]
    return [chunk[start:end].strip() for start, end in spans]


def _planning_chunk_is_useful(chunk: str) -> bool:
    return len(re.findall(r"[a-z0-9]+", chunk.lower())) >= 4


def _planning_sentence_chunks(text: str) -> list[str]:
    chunks = []
    start = 0
    for index, char in enumerate(text):
        if char not in ".!?":
            continue
        if not _is_planning_sentence_boundary(text, index):
            continue
        chunk = text[start : index + 1].strip()
        if chunk and not re.fullmatch(r"\d+[.)]", chunk):
            chunks.append(chunk)
        start = index + 1
    tail = text[start:].strip()
    if tail and not re.fullmatch(r"\d+[.)]", tail):
        chunks.append(tail)
    if chunks:
        return chunks
    stripped = text.strip()
    return [stripped] if stripped else []


def _is_planning_sentence_boundary(text: str, index: int) -> bool:
    char = text[index]
    if char in "!?":
        return True
    if char != ".":
        return False
    previous_char = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""
    if previous_char.isdigit() and next_char.isdigit():
        return False
    if next_char and not next_char.isspace():
        return False
    lookahead = index + 1
    while lookahead < len(text) and text[lookahead].isspace():
        if text[lookahead] == "\n":
            return True
        lookahead += 1
    if lookahead >= len(text):
        return True
    return text[lookahead].isupper() or text[lookahead].isdigit()


def _planning_span_repair_prompt_override(
    task_prompt: str,
    source_text: str,
    gap_terms: list[str],
    span_targets: list[str],
    repair: Any,
    *,
    history_contrast: str = "",
    planning_prompt_gate_active: bool = False,
) -> str:
    opening = _planning_span_preserved_opening(source_text, span_targets)
    gap_text = ", ".join(gap_terms) if gap_terms else "none obvious from keyword coverage"
    avoid_instructions = _planning_span_avoid_instructions(task_prompt, span_targets)
    avoid_text = f"\n{avoid_instructions}" if avoid_instructions else ""
    contrast_text = (
        f"\nDenoise-history contrast to use only if consistent with the task:\n{history_contrast}\n"
        if history_contrast.strip()
        else ""
    )
    instruction = getattr(repair, "prompt_repair_instruction", None) or (
        "Rewrite the masked continuation with concrete task-specific measurements, "
        "decision rules, risks, fallback paths, and stop conditions."
    )
    if (
        planning_prompt_gate_active
        and getattr(repair, "planning_prompt_gate_policy", None) == "public_claim_confound_control"
    ):
        return _planning_public_claim_control_prompt_override(
            task_prompt,
            opening=opening,
            gap_text=gap_text,
            instruction=instruction,
        )
    return (
        f"{task_prompt}\n\n"
        f"Preserved opening from the previous draft:\n{opening}\n\n"
        "The continuation after that opening is masked in the seed. Rewrite that "
        "continuation instead of copying the weak downstream draft.\n\n"
        f"Missing or weak task terms to cover: {gap_text}\n\n"
        f"{contrast_text}"
        f"{instruction}{avoid_text}\n"
        "Answer directly from the preserved opening. Do not mention this checklist."
    )


def _planning_public_claim_control_prompt_override(
    task_prompt: str,
    *,
    opening: str,
    gap_text: str,
    instruction: str,
) -> str:
    return (
        f"{task_prompt}\n\n"
        f"Start with this answer prefix and continue it directly:\n{opening}\n\n"
        "Write only the completed falsification plan. Do not mention drafts, masks, "
        "checklists, or instructions.\n\n"
        f"Task terms still weak or missing: {gap_text}\n\n"
        f"{instruction}"
    )


def _planning_prompt_gate_seed_suffix_text(
    repair: Any,
    *,
    task_prompt: str,
    prompt_constraint_gap_terms: list[str],
    rubric_items: tuple[str, ...] = (),
) -> tuple[str | None, dict[str, object]]:
    fixed_text = getattr(repair, "planning_prompt_gate_seed_suffix_text", None)
    if fixed_text:
        return str(fixed_text), {
            "seed_suffix_policy": "fixed_text",
            "seed_suffix_policy_reason": "fixed_text_configured",
        }
    policy = getattr(repair, "planning_prompt_gate_seed_suffix_policy", None)
    if policy is None:
        return None, {
            "seed_suffix_policy": None,
            "seed_suffix_policy_reason": "disabled",
        }
    if policy not in {
        "compact_action_control_terms",
        "compact_compatibility_control_terms",
        "compact_control_terms",
        "compact_joint_control_terms",
        "compact_preservation_control_terms",
    }:
        raise ValueError(f"Unsupported planning prompt seed suffix policy: {policy}")
    if policy == "compact_preservation_control_terms":
        seed_text, reason, candidates = _compact_preservation_control_seed_suffix_text(
            task_prompt=task_prompt,
            prompt_constraint_gap_terms=prompt_constraint_gap_terms,
            rubric_items=rubric_items,
        )
        return seed_text, {
            "seed_suffix_policy": policy,
            "seed_suffix_policy_reason": reason,
            "generated_seed_suffix_text": seed_text or "",
            "seed_suffix_candidate_scores": candidates,
        }
    if policy == "compact_joint_control_terms":
        seed_text, reason, candidates = _compact_joint_control_seed_suffix_text(
            task_prompt=task_prompt,
            prompt_constraint_gap_terms=prompt_constraint_gap_terms,
            rubric_items=rubric_items,
        )
        return seed_text, {
            "seed_suffix_policy": policy,
            "seed_suffix_policy_reason": reason,
            "generated_seed_suffix_text": seed_text or "",
            "seed_suffix_candidate_scores": candidates,
        }
    if policy == "compact_compatibility_control_terms":
        seed_text, reason, candidates = _compact_compatibility_control_seed_suffix_text(
            task_prompt=task_prompt,
            prompt_constraint_gap_terms=prompt_constraint_gap_terms,
            rubric_items=rubric_items,
        )
        return seed_text, {
            "seed_suffix_policy": policy,
            "seed_suffix_policy_reason": reason,
            "generated_seed_suffix_text": seed_text or "",
            "seed_suffix_candidate_scores": candidates,
        }
    seed_fn = (
        _compact_action_control_seed_suffix_text
        if policy == "compact_action_control_terms"
        else _compact_control_seed_suffix_text
    )
    seed_text, reason = seed_fn(
        task_prompt=task_prompt,
        prompt_constraint_gap_terms=prompt_constraint_gap_terms,
        rubric_items=rubric_items,
    )
    return seed_text, {
        "seed_suffix_policy": policy,
        "seed_suffix_policy_reason": reason,
        "generated_seed_suffix_text": seed_text or "",
    }


def _compact_control_seed_suffix_text(
    *,
    task_prompt: str,
    prompt_constraint_gap_terms: list[str],
    rubric_items: tuple[str, ...] = (),
) -> tuple[str | None, str]:
    surface = _normalize(
        " ".join(
            [
                task_prompt,
                " ".join(prompt_constraint_gap_terms),
                " ".join(rubric_items),
            ]
        )
    )
    if not surface:
        return None, "empty_control_surface"
    has_oracle_selected = _contains_any(
        surface,
        (
            "oracle",
            "best-of",
            "best of",
            "selected result",
            "selected results",
        ),
    )
    has_claim_survival = "claim" in surface and _contains_any(
        surface,
        (
            "survive",
            "survives",
            "disappear",
            "disappears",
            "effect disappears",
        ),
    )
    if has_oracle_selected and has_claim_survival:
        return " oracle selected results; claim survives if disappears.", "oracle_selected_claim_survival"
    if has_oracle_selected:
        return " oracle selected results.", "oracle_selected"
    if has_claim_survival:
        return " claim survives if disappears.", "claim_survival"
    if "baseline" in surface and _contains_any(surface, ("regression", "regressions")):
        return " baseline regressions recorded.", "baseline_regressions"
    if "baseline" in surface and "intervention" in surface:
        return " baseline intervention locked tasks.", "baseline_intervention"
    return None, "no_compact_control_anchor"


def _compact_action_control_seed_suffix_text(
    *,
    task_prompt: str,
    prompt_constraint_gap_terms: list[str],
    rubric_items: tuple[str, ...] = (),
) -> tuple[str | None, str]:
    surface = _normalize(
        " ".join(
            [
                task_prompt,
                " ".join(prompt_constraint_gap_terms),
                " ".join(rubric_items),
            ]
        )
    )
    if not surface:
        return None, "empty_control_surface"
    has_oracle_selected = _contains_any(
        surface,
        (
            "oracle",
            "best-of",
            "best of",
            "selected result",
            "selected results",
        ),
    )
    has_claim_survival = "claim" in surface and _contains_any(
        surface,
        (
            "survive",
            "survives",
            "disappear",
            "disappears",
            "effect disappears",
        ),
    )
    has_locked_rerun = _contains_any(
        surface,
        (
            "locked task",
            "locked tasks",
            "rerun",
            "baseline",
            "intervention",
        ),
    )
    if has_oracle_selected and has_claim_survival and has_locked_rerun:
        return (
            " rerun; oracle selected; claim survives.",
            "action_oracle_selected_claim_survival",
        )
    if has_oracle_selected and has_claim_survival:
        return (
            " report oracle selected results; claim survives.",
            "action_oracle_selected_claim_survival_no_rerun",
        )
    if has_locked_rerun:
        return " rerun tasks; record wins.", "action_locked_rerun"
    return _compact_control_seed_suffix_text(
        task_prompt=task_prompt,
        prompt_constraint_gap_terms=prompt_constraint_gap_terms,
        rubric_items=rubric_items,
    )


def _compact_compatibility_control_seed_suffix_text(
    *,
    task_prompt: str,
    prompt_constraint_gap_terms: list[str],
    rubric_items: tuple[str, ...] = (),
) -> tuple[str | None, str, list[dict[str, object]]]:
    surface = _normalize(
        " ".join(
            [
                task_prompt,
                " ".join(prompt_constraint_gap_terms),
                " ".join(rubric_items),
            ]
        )
    )
    if not surface:
        return None, "empty_control_surface", []
    needs_oracle_selected = _contains_any(
        surface,
        (
            "oracle",
            "best-of",
            "best of",
            "selected result",
            "selected results",
        ),
    )
    needs_claim_survival = "claim" in surface and _contains_any(
        surface,
        (
            "survive",
            "survives",
            "disappear",
            "disappears",
            "effect disappears",
        ),
    )
    needs_locked_rerun = _contains_any(
        surface,
        (
            "locked task",
            "locked tasks",
            "rerun",
            "baseline",
            "intervention",
        ),
    )
    candidate_reasons: list[tuple[str, str]] = []
    if needs_oracle_selected and needs_claim_survival:
        candidate_reasons.extend(
            [
                (
                    " oracle selected results; claim survives if disappears.",
                    "oracle_selected_claim_survival",
                ),
                (
                    " report oracle selected results; claim survives.",
                    "report_oracle_selected_claim_survival",
                ),
            ]
        )
        if needs_locked_rerun:
            candidate_reasons.append(
                (
                    " rerun; oracle selected; claim survives.",
                    "action_oracle_selected_claim_survival",
                )
            )
    elif needs_oracle_selected:
        candidate_reasons.append((" oracle selected results.", "oracle_selected"))
    elif needs_claim_survival:
        candidate_reasons.append((" claim survives if disappears.", "claim_survival"))
    if needs_locked_rerun:
        candidate_reasons.append((" rerun tasks; record wins.", "action_locked_rerun"))

    seen: set[str] = set()
    candidates: list[dict[str, object]] = []
    for seed_text, reason in candidate_reasons:
        if seed_text in seen:
            continue
        seen.add(seed_text)
        score, components = _compact_seed_compatibility_score(
            seed_text,
            needs_oracle_selected=needs_oracle_selected,
            needs_claim_survival=needs_claim_survival,
            needs_locked_rerun=needs_locked_rerun,
        )
        candidates.append(
            {
                "seed_suffix_text": seed_text,
                "reason": reason,
                "score": round(score, 6),
                **components,
            }
        )
    candidates.sort(
        key=lambda row: (
            -float(row["score"]),
            len(str(row["seed_suffix_text"])),
            str(row["seed_suffix_text"]),
        )
    )
    if candidates and float(candidates[0]["score"]) > 0.0:
        return (
            str(candidates[0]["seed_suffix_text"]),
            f"compatibility_{candidates[0]['reason']}",
            candidates,
        )
    seed_text, reason = _compact_control_seed_suffix_text(
        task_prompt=task_prompt,
        prompt_constraint_gap_terms=prompt_constraint_gap_terms,
        rubric_items=rubric_items,
    )
    return seed_text, reason, candidates


def _compact_preservation_control_seed_suffix_text(
    *,
    task_prompt: str,
    prompt_constraint_gap_terms: list[str],
    rubric_items: tuple[str, ...] = (),
) -> tuple[str | None, str, list[dict[str, object]]]:
    surface = _normalize(
        " ".join(
            [
                task_prompt,
                " ".join(prompt_constraint_gap_terms),
                " ".join(rubric_items),
            ]
        )
    )
    if not surface:
        return None, "empty_control_surface", []
    needs_oracle_selected = _contains_any(
        surface,
        (
            "oracle",
            "best-of",
            "best of",
            "selected result",
            "selected results",
        ),
    )
    needs_claim_survival = "claim" in surface and _contains_any(
        surface,
        (
            "survive",
            "survives",
            "disappear",
            "disappears",
            "effect disappears",
        ),
    )
    needs_locked_rerun = _contains_any(
        surface,
        (
            "locked task",
            "locked tasks",
            "rerun",
            "baseline",
            "intervention",
        ),
    )
    candidate_reasons: list[tuple[str, str]] = []
    if needs_oracle_selected and needs_claim_survival:
        candidate_reasons.extend(
            [
                (
                    " separate oracle selected; preserve claim if disappears.",
                    "separate_oracle_selected_preserve_claim",
                ),
                (
                    " oracle selected results; preserve claim if disappears.",
                    "oracle_selected_results_preserve_claim",
                ),
                (
                    " selected results; preserve claim if disappears.",
                    "selected_results_preserve_claim",
                ),
                (
                    " oracle selected results; claim survives if disappears.",
                    "oracle_selected_claim_survival",
                ),
            ]
        )
        if needs_locked_rerun:
            candidate_reasons.append(
                (
                    " rerun; preserve claim if disappears.",
                    "action_preserve_claim",
                )
            )
    elif needs_oracle_selected:
        candidate_reasons.append((" oracle selected results.", "oracle_selected"))
    elif needs_claim_survival:
        candidate_reasons.append((" preserve claim if disappears.", "preserve_claim"))
    if needs_locked_rerun:
        candidate_reasons.append((" rerun tasks; record wins.", "action_locked_rerun"))

    max_compatibility = _compact_seed_max_compatibility_score(
        needs_oracle_selected=needs_oracle_selected,
        needs_claim_survival=needs_claim_survival,
        needs_locked_rerun=needs_locked_rerun,
    )
    seen: set[str] = set()
    candidates: list[dict[str, object]] = []
    for seed_text, reason in candidate_reasons:
        if seed_text in seen:
            continue
        seen.add(seed_text)
        compatibility_score, components = _compact_seed_compatibility_score(
            seed_text,
            needs_oracle_selected=needs_oracle_selected,
            needs_claim_survival=needs_claim_survival,
            needs_locked_rerun=needs_locked_rerun,
        )
        compatibility_norm = max(0.0, min(1.0, compatibility_score / max(1.0, max_compatibility)))
        expected_realization = _compact_seed_expected_realization_score(seed_text)
        semantic_intent = _seed_semantic_preservation_score(_normalize(seed_text), _normalize(seed_text))
        preservation_action = 1.0 if "preserv" in _normalize(seed_text) and "claim" in _normalize(seed_text) else 0.0
        joint_score = (
            0.35 * compatibility_norm
            + 0.25 * semantic_intent
            + 0.25 * expected_realization
            + 0.15 * preservation_action
        )
        candidates.append(
            {
                "seed_suffix_text": seed_text,
                "reason": reason,
                "score": round(joint_score, 6),
                "compatibility_score": round(compatibility_score, 6),
                "compatibility_norm": round(compatibility_norm, 6),
                "expected_realization_score": round(expected_realization, 6),
                "preservation_action_score": preservation_action,
                "semantic_intent_score": round(semantic_intent, 6),
                **components,
            }
        )
    candidates.sort(
        key=lambda row: (
            -float(row["score"]),
            len(str(row["seed_suffix_text"])),
            str(row["seed_suffix_text"]),
        )
    )
    if candidates and float(candidates[0]["score"]) > 0.0:
        return (
            str(candidates[0]["seed_suffix_text"]),
            f"preservation_{candidates[0]['reason']}",
            candidates,
        )
    seed_text, reason = _compact_control_seed_suffix_text(
        task_prompt=task_prompt,
        prompt_constraint_gap_terms=prompt_constraint_gap_terms,
        rubric_items=rubric_items,
    )
    return seed_text, reason, candidates


def _compact_joint_control_seed_suffix_text(
    *,
    task_prompt: str,
    prompt_constraint_gap_terms: list[str],
    rubric_items: tuple[str, ...] = (),
) -> tuple[str | None, str, list[dict[str, object]]]:
    surface = _normalize(
        " ".join(
            [
                task_prompt,
                " ".join(prompt_constraint_gap_terms),
                " ".join(rubric_items),
            ]
        )
    )
    if not surface:
        return None, "empty_control_surface", []
    needs_oracle_selected = _contains_any(
        surface,
        (
            "oracle",
            "best-of",
            "best of",
            "selected result",
            "selected results",
        ),
    )
    needs_claim_survival = "claim" in surface and _contains_any(
        surface,
        (
            "survive",
            "survives",
            "disappear",
            "disappears",
            "effect disappears",
        ),
    )
    needs_locked_rerun = _contains_any(
        surface,
        (
            "locked task",
            "locked tasks",
            "rerun",
            "baseline",
            "intervention",
        ),
    )
    candidate_reasons: list[tuple[str, str]] = []
    if needs_oracle_selected and needs_claim_survival:
        candidate_reasons.extend(
            [
                (
                    " oracle selected results; claim survives if disappears.",
                    "oracle_selected_claim_survival",
                ),
                (
                    " separate oracle selected; claim survives if disappears.",
                    "separate_oracle_selected_claim_survival",
                ),
                (
                    " distinguish oracle selected; claim survives if disappears.",
                    "distinguish_oracle_selected_claim_survival",
                ),
                (
                    " report oracle selected results; claim survives.",
                    "report_oracle_selected_claim_survival",
                ),
            ]
        )
        if needs_locked_rerun:
            candidate_reasons.append(
                (
                    " rerun; oracle selected; claim survives.",
                    "action_oracle_selected_claim_survival",
                )
            )
    elif needs_oracle_selected:
        candidate_reasons.append((" oracle selected results.", "oracle_selected"))
    elif needs_claim_survival:
        candidate_reasons.append((" claim survives if disappears.", "claim_survival"))
    if needs_locked_rerun:
        candidate_reasons.append((" rerun tasks; record wins.", "action_locked_rerun"))

    max_compatibility = _compact_seed_max_compatibility_score(
        needs_oracle_selected=needs_oracle_selected,
        needs_claim_survival=needs_claim_survival,
        needs_locked_rerun=needs_locked_rerun,
    )
    seen: set[str] = set()
    candidates: list[dict[str, object]] = []
    for seed_text, reason in candidate_reasons:
        if seed_text in seen:
            continue
        seen.add(seed_text)
        compatibility_score, components = _compact_seed_compatibility_score(
            seed_text,
            needs_oracle_selected=needs_oracle_selected,
            needs_claim_survival=needs_claim_survival,
            needs_locked_rerun=needs_locked_rerun,
        )
        compatibility_norm = max(0.0, min(1.0, compatibility_score / max(1.0, max_compatibility)))
        expected_realization = _compact_seed_expected_realization_score(seed_text)
        semantic_intent = _seed_semantic_preservation_score(_normalize(seed_text), _normalize(seed_text))
        joint_score = 0.35 * compatibility_norm + 0.35 * semantic_intent + 0.30 * expected_realization
        candidates.append(
            {
                "seed_suffix_text": seed_text,
                "reason": reason,
                "score": round(joint_score, 6),
                "compatibility_score": round(compatibility_score, 6),
                "compatibility_norm": round(compatibility_norm, 6),
                "expected_realization_score": round(expected_realization, 6),
                "semantic_intent_score": round(semantic_intent, 6),
                **components,
            }
        )
    candidates.sort(
        key=lambda row: (
            -float(row["score"]),
            len(str(row["seed_suffix_text"])),
            str(row["seed_suffix_text"]),
        )
    )
    if candidates and float(candidates[0]["score"]) > 0.0:
        return (
            str(candidates[0]["seed_suffix_text"]),
            f"joint_{candidates[0]['reason']}",
            candidates,
        )
    seed_text, reason = _compact_control_seed_suffix_text(
        task_prompt=task_prompt,
        prompt_constraint_gap_terms=prompt_constraint_gap_terms,
        rubric_items=rubric_items,
    )
    return seed_text, reason, candidates


def _compact_seed_compatibility_score(
    seed_text: str,
    *,
    needs_oracle_selected: bool,
    needs_claim_survival: bool,
    needs_locked_rerun: bool,
) -> tuple[float, dict[str, object]]:
    seed_surface = _normalize(seed_text)
    has_oracle_selected = "oracle" in seed_surface and "selected" in seed_surface
    has_results = "result" in seed_surface
    has_claim_survival = "claim" in seed_surface and _contains_any(seed_surface, ("surviv", "preserv"))
    has_disappear_condition = "disappear" in seed_surface
    has_action = _contains_any(seed_surface, ("preserve", "rerun", "report", "record", "test"))

    score = 0.0
    if needs_oracle_selected:
        score += 2.0 if has_oracle_selected else -1.0
        score += 0.8 if has_results else -0.35
    if needs_claim_survival:
        score += 2.0 if has_claim_survival else -1.0
        score += 0.8 if has_disappear_condition else -0.2
    if needs_locked_rerun:
        score += 0.35 if has_action else -0.1
    if needs_oracle_selected and needs_claim_survival and has_action:
        if not has_results:
            score -= 0.35
        if not has_disappear_condition:
            score -= 0.25

    return score, {
        "has_action": has_action,
        "has_claim_survival": has_claim_survival,
        "has_disappear_condition": has_disappear_condition,
        "has_oracle_selected": has_oracle_selected,
        "has_results": has_results,
    }


def _compact_seed_max_compatibility_score(
    *,
    needs_oracle_selected: bool,
    needs_claim_survival: bool,
    needs_locked_rerun: bool,
) -> float:
    score = 0.0
    if needs_oracle_selected:
        score += 2.0 + 0.8
    if needs_claim_survival:
        score += 2.0 + 0.8
    if needs_locked_rerun:
        score += 0.35
    return max(1.0, score)


def _compact_seed_expected_realization_score(seed_text: str) -> float:
    seed_surface = _normalize(seed_text)
    if not seed_surface:
        return 0.0
    has_oracle_selected = "oracle" in seed_surface and "selected" in seed_surface
    has_claim_survival = "claim" in seed_surface and _contains_any(seed_surface, ("surviv", "preserv"))
    has_disappear_condition = "disappear" in seed_surface
    has_action = _contains_any(
        seed_surface,
        ("preserve", "separate", "distinguish", "split", "rerun", "report", "record"),
    )
    has_relation_action = _contains_any(seed_surface, ("separate", "distinguish", "split"))
    word_count = len(seed_surface.split())
    concision = 1.0 if word_count <= 7 else max(0.0, 1.0 - 0.12 * (word_count - 7))
    score = (
        0.30 * float(has_oracle_selected)
        + 0.25 * float(has_claim_survival and has_disappear_condition)
        + 0.20 * float(has_action)
        + 0.15 * float(has_relation_action)
        + 0.10 * concision
    )
    return max(0.0, min(1.0, score))


def _apply_planning_seed_suffix_anchor(
    config: Any,
    *,
    seed_suffix_text: str | None,
    token_encoder: Any,
    active: bool,
) -> tuple[Any, dict[str, object]]:
    """Fix a short semantic anchor into masked denoise positions."""
    if not active:
        return config, {"active": False, "reason": "gate_inactive"}
    if not seed_suffix_text:
        return config, {"active": False, "reason": "no_seed_suffix_text"}
    if token_encoder is None:
        return config, {"active": False, "reason": "no_token_encoder"}
    initial_suffix = getattr(config, "initial_suffix_token_ids", None)
    if initial_suffix is None:
        return config, {"active": False, "reason": "no_initial_suffix_seed"}
    seed = list(initial_suffix)
    masked_positions = [index for index, token_id in enumerate(seed) if token_id is None]
    if not masked_positions:
        return config, {"active": False, "reason": "no_masked_positions"}
    token_ids = [
        token_id
        for token_id in token_encoder(seed_suffix_text)
        if isinstance(token_id, int) and not isinstance(token_id, bool)
    ]
    if not token_ids:
        return config, {"active": False, "reason": "empty_encoded_anchor"}
    truncated = len(token_ids) > len(masked_positions)
    if truncated:
        token_ids = token_ids[-len(masked_positions) :]
    anchor_positions = masked_positions[-len(token_ids) :]
    for position, token_id in zip(anchor_positions, token_ids, strict=True):
        seed[position] = token_id
    return replace(config, initial_suffix_token_ids=tuple(seed)), {
        "active": True,
        "anchor_token_count": len(token_ids),
        "anchor_positions": anchor_positions,
        "reason": "seed_suffix_text_applied",
        "seed_suffix_text": seed_suffix_text,
        "truncated": truncated,
    }


def _planning_span_history_contrast(
    record: dict[str, object],
    *,
    task_prompt: str,
    source_text: str,
    span_targets: list[str],
) -> str:
    history_samples = _history_repair_sample_candidates(record, task_prompt)
    if not history_samples:
        return ""
    source_normalized = _normalize(source_text)
    ranked = []
    for sample in history_samples:
        visible_text = str(sample.get("visible_text", ""))
        if not visible_text.strip():
            continue
        normalized = _normalize(visible_text)
        if normalized == source_normalized:
            continue
        target_overlap = max(
            (_text_similarity(target, visible_text) for target in span_targets),
            default=0.0,
        )
        prompt_coverage = _prompt_keyword_coverage(task_prompt, normalized)
        text_similarity = _text_similarity(source_text, visible_text)
        char_ratio = len(visible_text.strip()) / max(1, len(source_text.strip()))
        if text_similarity < 0.90 or char_ratio < 0.90:
            continue
        ranked.append(
            (
                prompt_coverage,
                -target_overlap,
                text_similarity,
                int(sample.get("step", 0)),
                visible_text,
            )
        )
    if not ranked:
        return ""
    _coverage, _target_overlap, _similarity, step, visible_text = max(ranked)
    return f"history step {step}: {_compact_text(visible_text, max_chars=420)}"


def _planning_span_preserved_opening(source_text: str, span_targets: list[str]) -> str:
    lowered_source = source_text.lower()
    first_target_start = len(source_text)
    for target in span_targets:
        target_start = lowered_source.find(target.lower())
        if target_start >= 0:
            first_target_start = min(first_target_start, target_start)
    if first_target_start < len(source_text):
        opening = source_text[:first_target_start].strip()
    else:
        opening = _planning_sentence_chunks(source_text)[0] if source_text.strip() else ""
    return _compact_text(opening or source_text, max_chars=320)


def _planning_span_avoid_instructions(task_prompt: str, span_targets: list[str]) -> str:
    normalized_prompt = _normalize(task_prompt)
    normalized_targets = _normalize(" ".join(span_targets))
    instructions = []
    if "baseline" in normalized_prompt and _contains_any(
        normalized_targets,
        ("baseline data will not be available", "valid comparison", "without a baseline"),
    ):
        instructions.append(
            "Do not claim the comparison is valid when baseline data is unavailable; "
            "preserve a reliable baseline before comparing the intervention."
        )
    if _contains_any(normalized_targets, ("publishable result even if", "ensuring you have a publishable result")):
        instructions.append(
            "Do not rely on a generic publishable-result claim; name the measurements "
            "and failure evidence that remain publishable."
        )
    return " ".join(instructions)


def select_three_arm_records(
    records: list[dict[str, object]],
    *,
    seed: int,
    candidate_key: str,
    task_id: str,
    task_prompt: str = "",
    task_answer_type: str = "rubric",
    exact_task_trajectory_policy: str = "fixed",
    trajectory_selector: str = "planning_prompt",
) -> dict[str, dict[str, object]]:
    """Select arm records without using task-score labels."""
    if not records:
        raise ValueError("records must not be empty")
    random_index = _stable_random_index(seed, candidate_key, task_id, len(records))
    trajectory_record = max(
        records,
        key=lambda record: _selection_score(record, task_prompt, task_answer_type, trajectory_selector),
    )
    if task_answer_type != "rubric" and exact_task_trajectory_policy == "fixed":
        trajectory_record = records[0]
    elif task_answer_type != "rubric" and exact_task_trajectory_policy == "proposal_history":
        trajectory_record = _select_exact_answer_proposal_history_record(
            records,
            task_prompt=task_prompt,
            task_answer_type=task_answer_type,
        )
    return {
        "fixed": records[0],
        "random": records[random_index],
        "trajectory_selected": trajectory_record,
    }


def select_evolved_record(
    records: list[dict[str, object]],
    *,
    baseline_record: dict[str, object] | None = None,
    task_prompt: str = "",
    task_answer_type: str = "rubric",
    exact_task_trajectory_policy: str = "fixed",
    trajectory_selector: str = "planning_prompt",
    evolved_selector: str = DEFAULT_EVOLVED_SELECTOR,
    evolved_quality_margin: float = DEFAULT_EVOLVED_QUALITY_MARGIN,
    evolved_selector_tolerance: float = DEFAULT_EVOLVED_SELECTOR_TOLERANCE,
    promotion_margin: float = 0.0,
    revision_promotion_margin: float = DEFAULT_REVISION_PROMOTION_MARGIN,
) -> dict[str, object]:
    """Select from the base plus mutated pool, with optional conservative gating."""
    if not records:
        raise ValueError("records must not be empty")
    if task_answer_type != "rubric" and exact_task_trajectory_policy == "fixed":
        return records[0]
    if task_answer_type != "rubric" and exact_task_trajectory_policy == "proposal_history":
        return _select_exact_answer_proposal_history_record(
            records,
            task_prompt=task_prompt,
            task_answer_type=task_answer_type,
        )
    state_best = max(
        records,
        key=lambda record: _selection_score(record, task_prompt, task_answer_type, trajectory_selector),
    )
    if baseline_record is None:
        return state_best
    best_score = _selection_score(state_best, task_prompt, task_answer_type, trajectory_selector)
    baseline_score = _selection_score(baseline_record, task_prompt, task_answer_type, trajectory_selector)
    selected = baseline_record
    required_margin = _evolved_promotion_margin_for_record(
        state_best,
        promotion_margin=promotion_margin,
        revision_promotion_margin=revision_promotion_margin,
    )
    if best_score >= baseline_score + required_margin:
        selected = state_best
    if evolved_selector == "inherit" or task_answer_type != "rubric":
        return selected
    if evolved_selector == "planning_quality_fallback":
        return _planning_quality_evolved_fallback(
            records,
            selected_record=selected,
            task_prompt=task_prompt,
            task_answer_type=task_answer_type,
            trajectory_selector=trajectory_selector,
            quality_margin=evolved_quality_margin,
            selector_tolerance=evolved_selector_tolerance,
            revision_promotion_margin=revision_promotion_margin,
        )
    raise ValueError(f"Unsupported evolved selector: {evolved_selector}")


def _select_repair_source_record(
    policy: str,
    *,
    selected_records: dict[str, dict[str, object]],
    evolved_record: dict[str, object],
    candidate_records: list[dict[str, object]],
    task_prompt: str = "",
    task_answer_type: str = "rubric",
    exact_task_trajectory_policy: str = "fixed",
    trajectory_selector: str = "planning_prompt",
    evolved_selector: str = DEFAULT_EVOLVED_SELECTOR,
    evolved_quality_margin: float = DEFAULT_EVOLVED_QUALITY_MARGIN,
    evolved_selector_tolerance: float = DEFAULT_EVOLVED_SELECTOR_TOLERANCE,
    evolved_promotion_margin: float = 0.0,
    revision_promotion_margin: float = DEFAULT_REVISION_PROMOTION_MARGIN,
) -> dict[str, object]:
    """Choose the record whose latent state seeds repair candidates."""
    if policy == "evolved":
        return evolved_record
    if policy == "trajectory":
        return selected_records["trajectory_selected"]
    if policy == "fixed":
        return selected_records["fixed"]
    if policy == "random":
        return selected_records["random"]
    if policy == "non_revision_evolved":
        if not _is_revision_record(evolved_record):
            return evolved_record
        non_revision_records = [
            record
            for record in candidate_records
            if not _is_revision_record(record) and not _is_repair_record(record)
        ]
        if not non_revision_records:
            return selected_records["trajectory_selected"]
        return select_evolved_record(
            non_revision_records,
            baseline_record=selected_records["trajectory_selected"],
            task_prompt=task_prompt,
            task_answer_type=task_answer_type,
            exact_task_trajectory_policy=exact_task_trajectory_policy,
            trajectory_selector=trajectory_selector,
            evolved_selector=evolved_selector,
            evolved_quality_margin=evolved_quality_margin,
            evolved_selector_tolerance=evolved_selector_tolerance,
            promotion_margin=evolved_promotion_margin,
            revision_promotion_margin=revision_promotion_margin,
        )
    raise ValueError(f"Unsupported repair source policy: {policy}")


def _select_repair_source_records(
    policy: str,
    *,
    selected_records: dict[str, dict[str, object]],
    evolved_record: dict[str, object],
    candidate_records: list[dict[str, object]],
    task_prompt: str = "",
    task_answer_type: str = "rubric",
    exact_task_trajectory_policy: str = "fixed",
    trajectory_selector: str = "planning_prompt",
    evolved_selector: str = DEFAULT_EVOLVED_SELECTOR,
    evolved_quality_margin: float = DEFAULT_EVOLVED_QUALITY_MARGIN,
    evolved_selector_tolerance: float = DEFAULT_EVOLVED_SELECTOR_TOLERANCE,
    evolved_promotion_margin: float = 0.0,
    revision_promotion_margin: float = DEFAULT_REVISION_PROMOTION_MARGIN,
    adaptive_source_gate_mode: str = "custom",
    adaptive_source_gap_min_terms: int = DEFAULT_ADAPTIVE_SOURCE_GAP_MIN_TERMS,
    adaptive_source_quality_floor: float = DEFAULT_ADAPTIVE_SOURCE_QUALITY_FLOOR,
    adaptive_source_quality_ceiling: float | None = DEFAULT_ADAPTIVE_SOURCE_QUALITY_CEILING,
) -> list[dict[str, object]]:
    if policy == "evolved_and_trajectory":
        return _dedupe_records_by_identity(
            [evolved_record, selected_records["trajectory_selected"]]
        )
    if policy == "non_revision_plus_gap_trajectory":
        primary = _select_repair_source_record(
            "non_revision_evolved",
            selected_records=selected_records,
            evolved_record=evolved_record,
            candidate_records=candidate_records,
            task_prompt=task_prompt,
            task_answer_type=task_answer_type,
            exact_task_trajectory_policy=exact_task_trajectory_policy,
            trajectory_selector=trajectory_selector,
            evolved_selector=evolved_selector,
            evolved_quality_margin=evolved_quality_margin,
            evolved_selector_tolerance=evolved_selector_tolerance,
            evolved_promotion_margin=evolved_promotion_margin,
            revision_promotion_margin=revision_promotion_margin,
        )
        trajectory_source = selected_records["trajectory_selected"]
        sources = [primary]
        if _should_add_gap_trajectory_repair_source(
            trajectory_source,
            primary,
            task_prompt,
            gap_min_terms=adaptive_source_gap_min_terms,
            quality_floor=adaptive_source_quality_floor,
            quality_ceiling=adaptive_source_quality_ceiling,
        ):
            sources.append(trajectory_source)
        return _dedupe_records_by_identity(sources)
    return [
        _select_repair_source_record(
            policy,
            selected_records=selected_records,
            evolved_record=evolved_record,
            candidate_records=candidate_records,
            task_prompt=task_prompt,
            task_answer_type=task_answer_type,
            exact_task_trajectory_policy=exact_task_trajectory_policy,
            trajectory_selector=trajectory_selector,
            evolved_selector=evolved_selector,
            evolved_quality_margin=evolved_quality_margin,
            evolved_selector_tolerance=evolved_selector_tolerance,
            evolved_promotion_margin=evolved_promotion_margin,
            revision_promotion_margin=revision_promotion_margin,
        )
    ]


def _should_add_gap_trajectory_repair_source(
    trajectory_source: dict[str, object],
    primary_source: dict[str, object],
    task_prompt: str,
    *,
    gap_min_terms: int = DEFAULT_ADAPTIVE_SOURCE_GAP_MIN_TERMS,
    quality_floor: float = DEFAULT_ADAPTIVE_SOURCE_QUALITY_FLOOR,
    quality_ceiling: float | None = DEFAULT_ADAPTIVE_SOURCE_QUALITY_CEILING,
) -> bool:
    return bool(
        _gap_trajectory_repair_source_gate(
            trajectory_source,
            primary_source,
            task_prompt,
            gap_min_terms=gap_min_terms,
            quality_floor=quality_floor,
            quality_ceiling=quality_ceiling,
        )["add"]
    )


def _gap_trajectory_repair_source_gate(
    trajectory_source: dict[str, object],
    primary_source: dict[str, object],
    task_prompt: str,
    *,
    gap_min_terms: int = DEFAULT_ADAPTIVE_SOURCE_GAP_MIN_TERMS,
    quality_floor: float = DEFAULT_ADAPTIVE_SOURCE_QUALITY_FLOOR,
    quality_ceiling: float | None = DEFAULT_ADAPTIVE_SOURCE_QUALITY_CEILING,
) -> dict[str, object]:
    schedule_name = _schedule_name(trajectory_source)
    gap_terms = _prompt_constraint_gap_terms(task_prompt, str(trajectory_source.get("text", "")))
    planning_quality = _planning_quality_score(trajectory_source, task_prompt)
    failures = []
    if _record_identity(trajectory_source) == _record_identity(primary_source):
        failures.append("same_as_primary")
    if not schedule_name.startswith("low_confidence"):
        failures.append("not_low_confidence")
    if len(gap_terms) < gap_min_terms:
        failures.append("prompt_gap_below_floor")
    if planning_quality < quality_floor:
        failures.append("planning_quality_below_floor")
    if quality_ceiling is not None and planning_quality > quality_ceiling:
        failures.append("planning_quality_above_ceiling")
    return {
        "add": not failures,
        "reason": "add" if not failures else ",".join(failures),
        "primary_control": _control_name(primary_source),
        "trajectory_control": _control_name(trajectory_source),
        "trajectory_schedule": schedule_name,
        "trajectory_planning_quality": planning_quality,
        "prompt_gap_term_count": len(gap_terms),
        "prompt_gap_terms": gap_terms,
        "gap_min_terms": gap_min_terms,
        "quality_floor": quality_floor,
        "quality_ceiling": quality_ceiling,
    }


def _evolved_promotion_margin_for_record(
    record: dict[str, object],
    *,
    promotion_margin: float,
    revision_promotion_margin: float,
) -> float:
    if _is_revision_record(record):
        return max(promotion_margin, revision_promotion_margin)
    return promotion_margin


def _select_exact_answer_proposal_history_record(
    records: list[dict[str, object]],
    *,
    task_prompt: str,
    task_answer_type: str,
) -> dict[str, object]:
    """Select an exact-answer final/history state only through prompt-derived proposals."""
    candidates: list[dict[str, object]] = []
    for record in records:
        task = _exact_task_from_record(record, task_prompt, task_answer_type)
        if task is None:
            continue
        proposals = counterfactual_answer_proposals(task, None)
        for proposal in proposals:
            if _answer_text_matches_proposal(str(record.get("text", "")), proposal.value, task_answer_type):
                candidates.append(
                    _with_exact_trajectory_metadata(
                        dict(record),
                        proposal=proposal.value,
                        proposal_source=proposal.source,
                        source="final",
                        source_record=record,
                    )
                )
            candidates.extend(
                _exact_answer_history_candidate_records(
                    record,
                    task=task,
                    proposal=proposal.value,
                    proposal_source=proposal.source,
                )
            )
    if not candidates:
        return records[0]
    return max(candidates, key=_exact_answer_trajectory_rank)


def _exact_task_from_record(
    record: dict[str, object],
    task_prompt: str,
    task_answer_type: str,
) -> GeneralReasoningTask | None:
    task = record.get("task")
    if not isinstance(task, dict):
        return None
    return GeneralReasoningTask(
        task_id=str(task.get("task_id", _task_id(record))),
        family=str(task.get("family", _task_family(record))),
        prompt=task_prompt,
        answer_type=task_answer_type,
        scorer=str(task.get("scorer", "")),
        max_new_tokens=_int_value(record.get("generated_token_count"), default=64),
        answer=task.get("answer"),
    )


def _exact_answer_history_candidate_records(
    record: dict[str, object],
    *,
    task: GeneralReasoningTask,
    proposal: str,
    proposal_source: str,
) -> list[dict[str, object]]:
    trajectory_summary = record.get("trajectory_summary")
    if not isinstance(trajectory_summary, dict):
        return []
    samples = trajectory_summary.get("samples")
    if not isinstance(samples, list):
        return []
    candidates = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        visible_text = str(sample.get("visible_text", ""))
        if not visible_text.strip():
            continue
        if not _answer_text_matches_proposal(visible_text, proposal, task.answer_type):
            continue
        candidates.append(
            _exact_answer_history_state_record(
                record,
                task=task,
                sample=sample,
                proposal=proposal,
                proposal_source=proposal_source,
                visible_text=visible_text,
            )
        )
    return candidates


def _exact_answer_history_state_record(
    record: dict[str, object],
    *,
    task: GeneralReasoningTask,
    sample: dict[str, object],
    proposal: str,
    proposal_source: str,
    visible_text: str,
) -> dict[str, object]:
    state_record = dict(record)
    step = _int_value(sample.get("step"), default=0)
    mask_count = _int_value(sample.get("mask_count"), default=0)
    visible_chars = _int_value(sample.get("visible_chars"), default=len(visible_text.strip()))
    source_control = _control_name(record)
    state_record["text"] = visible_text
    state_record["generation_stage"] = "denoise_history_state_selection"
    schedule = dict(record.get("schedule")) if isinstance(record.get("schedule"), dict) else {}
    schedule["name"] = f"{source_control}:history_step_{step}"
    schedule["source_control"] = source_control
    schedule["source_history_step"] = step
    state_record["schedule"] = schedule
    state_record["task_score"] = score_task_output(task, visible_text).to_dict()
    state_record = _with_exact_trajectory_metadata(
        state_record,
        proposal=proposal,
        proposal_source=proposal_source,
        source="history",
        source_record=record,
        source_history_step=step,
        source_history_mask_count=mask_count,
        source_history_visible_chars=visible_chars,
    )
    state_record["combined_selection_score"] = _combined_score(state_record)
    return state_record


def _with_exact_trajectory_metadata(
    record: dict[str, object],
    *,
    proposal: str,
    proposal_source: str,
    source: str,
    source_record: dict[str, object],
    source_history_step: int | None = None,
    source_history_mask_count: int | None = None,
    source_history_visible_chars: int | None = None,
) -> dict[str, object]:
    record["exact_trajectory_selection"] = {
        "proposal": proposal,
        "proposal_source": proposal_source,
        "source": source,
        "source_control": _control_name(source_record),
        "source_task_score": _task_score(source_record),
        "source_trajectory_score": _trajectory_score(source_record),
    }
    if source_history_step is not None:
        record["exact_trajectory_selection"]["source_history_step"] = source_history_step
    if source_history_mask_count is not None:
        record["exact_trajectory_selection"]["source_history_mask_count"] = source_history_mask_count
    if source_history_visible_chars is not None:
        record["exact_trajectory_selection"]["source_history_visible_chars"] = source_history_visible_chars
    return record


def _exact_answer_trajectory_rank(record: dict[str, object]) -> tuple[float, float, float, float]:
    selection = record.get("exact_trajectory_selection")
    source = selection.get("source") if isinstance(selection, dict) else ""
    source_rank = 1.0 if source == "final" else 0.0
    step = int(selection.get("source_history_step", 10**9)) if isinstance(selection, dict) else 10**9
    mask_count = int(selection.get("source_history_mask_count", 0)) if isinstance(selection, dict) else 0
    return (
        -mask_count,
        step,
        source_rank,
        _trajectory_score(record),
    )


def _planning_quality_evolved_fallback(
    records: list[dict[str, object]],
    *,
    selected_record: dict[str, object],
    task_prompt: str,
    task_answer_type: str,
    trajectory_selector: str,
    quality_margin: float,
    selector_tolerance: float,
    revision_promotion_margin: float,
) -> dict[str, object]:
    selected_state_score = _selection_score(
        selected_record,
        task_prompt,
        task_answer_type,
        trajectory_selector,
    )
    selected_quality = _planning_quality_score(selected_record, task_prompt)
    eligible = [
        record
        for record in records
        if _is_evolved_record(record)
        and _selection_score(record, task_prompt, task_answer_type, trajectory_selector)
        >= selected_state_score - selector_tolerance
        and _planning_quality_score(record, task_prompt)
        >= selected_quality
        + _evolved_promotion_margin_for_record(
            record,
            promotion_margin=quality_margin,
            revision_promotion_margin=revision_promotion_margin,
        )
    ]
    if not eligible:
        return selected_record
    return max(
        eligible,
        key=lambda record: (
            _planning_quality_score(record, task_prompt),
            _selection_score(record, task_prompt, task_answer_type, trajectory_selector),
        ),
    )


def select_repair_record(
    records: list[dict[str, object]],
    *,
    baseline_record: dict[str, object],
    task_prompt: str = "",
    task_answer_type: str = "rubric",
    exact_task_trajectory_policy: str = "fixed",
    trajectory_selector: str = "planning_prompt",
    repair_selector: str = DEFAULT_REPAIR_SELECTOR,
    promotion_margin: float = 0.0,
) -> dict[str, object]:
    """Select a repair candidate only when it clears the baseline margin."""
    if not records:
        raise ValueError("records must not be empty")
    if task_answer_type != "rubric" and exact_task_trajectory_policy != "trajectory":
        best = max(
            records,
            key=lambda record: _exact_answer_repair_selection_score(
                record,
                task_answer_type,
                task_prompt,
            ),
        )
        best_score = _exact_answer_repair_selection_score(best, task_answer_type, task_prompt)
        baseline_score = _exact_answer_repair_selection_score(
            baseline_record,
            task_answer_type,
            task_prompt,
        )
        if best_score >= baseline_score + promotion_margin:
            return best
        return baseline_record
    best = max(
        records,
        key=lambda record: _repair_selection_score(
            record,
            baseline_record=baseline_record,
            task_prompt=task_prompt,
            task_answer_type=task_answer_type,
            trajectory_selector=trajectory_selector,
            repair_selector=repair_selector,
        ),
    )
    best_score = _repair_selection_score(
        best,
        baseline_record=baseline_record,
        task_prompt=task_prompt,
        task_answer_type=task_answer_type,
        trajectory_selector=trajectory_selector,
        repair_selector=repair_selector,
    )
    baseline_score = _repair_selection_score(
        baseline_record,
        baseline_record=baseline_record,
        task_prompt=task_prompt,
        task_answer_type=task_answer_type,
        trajectory_selector=trajectory_selector,
        repair_selector=repair_selector,
    )
    if best_score >= baseline_score + promotion_margin:
        return best
    return baseline_record


def summarize_three_arm_scores(
    all_records: list[dict[str, object]],
    arm_records: list[dict[str, object]],
    *,
    exact_task_trajectory_policy: str = "fixed",
    trajectory_selector: str = "planning_prompt",
    evolved_selector: str = DEFAULT_EVOLVED_SELECTOR,
    evolved_quality_margin: float = DEFAULT_EVOLVED_QUALITY_MARGIN,
    evolved_selector_tolerance: float = DEFAULT_EVOLVED_SELECTOR_TOLERANCE,
    evolved_promotion_margin: float = 0.0,
    revision_promotion_margin: float = DEFAULT_REVISION_PROMOTION_MARGIN,
    adaptive_source_gate_mode: str = "custom",
    adaptive_source_gap_min_terms: int = DEFAULT_ADAPTIVE_SOURCE_GAP_MIN_TERMS,
    adaptive_source_quality_floor: float = DEFAULT_ADAPTIVE_SOURCE_QUALITY_FLOOR,
    adaptive_source_quality_ceiling: float | None = DEFAULT_ADAPTIVE_SOURCE_QUALITY_CEILING,
    include_revision_schedules: bool = False,
    revision_remask_fraction: float = 0.25,
    revision_steps: int = 16,
    include_history_repairs: bool = False,
    repair_pack: str = "prefix",
    repair_source_policy: str = "evolved",
    history_repair_fractions: tuple[float, ...] = (0.25,),
    include_history_visible_repair: bool = False,
    repair_spend_trigger: str = "always",
    repair_source_quality_threshold: float = 0.50,
    repair_source_min_chars: int = 320,
    repair_source_prompt_gap_min: int = 0,
    repair_source_prompt_gap_max: int = 999,
    repair_source_prompt_coverage_min: float = 0.0,
    repair_source_prompt_coverage_max: float = 1.0,
    counterfactual_probe_mode: str = "triage",
    counterfactual_probe_policy: str = COUNTERFACTUAL_MICRO_PROBE_POLICY_ID,
    repair_value_proxy_source_quality_max: float = 0.31,
    repair_transfer_source_task_min: float = DECOMPOSED_SPEND_TRANSFER_SOURCE_TASK_MIN,
    repair_phase_budget: str = "custom",
    repair_denoise_skeleton_max_step: int | None = None,
    phase_source_history_char_ratio_min: float = PHASE_SOURCE_HISTORY_CHAR_RATIO_MIN,
    phase_source_target_similarity_min: float = PHASE_SOURCE_TARGET_SIMILARITY_MIN,
    phase_source_text_similarity_min: float = PHASE_SOURCE_TEXT_SIMILARITY_MIN,
    repair_source_controls: list[str] | None = None,
    history_rescue_fractions: tuple[float, ...] = (),
    history_rescue_visible: bool = False,
    history_rescue_trigger: str = "baseline",
    history_rescue_source_controls: list[str] | None = None,
    prompt_guided_rescue_trigger: str = "off",
    prompt_guided_rescue_limit: int = 1,
    prompt_guided_rescue_source_quality_threshold: float = 0.45,
    prompt_guided_rescue_source_controls: list[str] | None = None,
    constraint_gap_rescue_trigger: str = "off",
    constraint_gap_rescue_limit: int = 1,
    constraint_gap_rescue_min_terms: int = 6,
    constraint_gap_rescue_source_quality_floor: float = 0.40,
    constraint_gap_rescue_source_quality_ceiling: float = 0.50,
    constraint_gap_rescue_source_controls: list[str] | None = None,
    repair_promotion_margin: float = 0.0,
    repair_selector: str = DEFAULT_REPAIR_SELECTOR,
    exact_verifier_revision: bool = False,
    repair_spend_gate_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_candidate_arm: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    by_family_arm: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in arm_records:
        arm = str(record["arm"])
        by_arm[arm].append(record)
        by_candidate_arm[(str(record["candidate_key"]), arm)].append(record)
        by_family_arm[(_task_family(record), arm)].append(record)

    fixed_records = by_arm.get("fixed", [])
    random_records = by_arm.get("random", [])
    trajectory_records = by_arm.get("trajectory_selected", [])
    evolved_records = by_arm.get("evolved", [])
    repair_records = by_arm.get("repair_selected", [])
    repair_eligible_records = [
        record for record in trajectory_records if _is_repair_eligible_arm_record(record)
    ]
    task_rows = _comparison_rows(arm_records, all_records)
    oracle_records = _oracle_records(all_records)
    gate_rows = repair_spend_gate_rows or []
    counterfactual_probe_generation_count = sum(
        1
        for row in gate_rows
        if isinstance(row, dict)
        and row.get("counterfactual_probe_observation") == "measured_generation"
    )
    scores = {
        "all_generation_count": len(all_records),
        "arm_selection_count": len(arm_records),
        "counterfactual_probe_generation_count": counterfactual_probe_generation_count,
        "exact_task_trajectory_policy": exact_task_trajectory_policy,
        "exact_trajectory_selection_source_counts": _exact_trajectory_selection_source_counts(arm_records),
        "history_mutability": _history_mutability_summary(all_records),
        "trajectory_selector": trajectory_selector,
        "evolved_selector": evolved_selector,
        "evolved_quality_margin": evolved_quality_margin,
        "evolved_selector_tolerance": evolved_selector_tolerance,
        "evolved_promotion_margin": evolved_promotion_margin,
        "revision_promotion_margin": revision_promotion_margin,
        "adaptive_source_gate_mode": adaptive_source_gate_mode,
        "adaptive_source_gap_min_terms": adaptive_source_gap_min_terms,
        "adaptive_source_quality_floor": adaptive_source_quality_floor,
        "adaptive_source_quality_ceiling": adaptive_source_quality_ceiling,
        "include_revision_schedules": include_revision_schedules,
        "revision_remask_fraction": revision_remask_fraction,
        "revision_steps": revision_steps,
        "exact_verifier_revision": exact_verifier_revision,
        "include_history_repairs": include_history_repairs,
        "repair_pack": repair_pack,
        "repair_source_policy": repair_source_policy,
        "history_repair_fractions": list(history_repair_fractions),
        "include_history_visible_repair": include_history_visible_repair,
        "repair_spend_trigger": repair_spend_trigger,
        "repair_source_quality_threshold": repair_source_quality_threshold,
        "repair_source_min_chars": repair_source_min_chars,
        "repair_source_prompt_gap_min": repair_source_prompt_gap_min,
        "repair_source_prompt_gap_max": repair_source_prompt_gap_max,
        "repair_source_prompt_coverage_min": repair_source_prompt_coverage_min,
        "repair_source_prompt_coverage_max": repair_source_prompt_coverage_max,
        "counterfactual_probe_mode": counterfactual_probe_mode,
        "counterfactual_probe_policy": counterfactual_probe_policy,
        "repair_value_proxy_source_quality_max": repair_value_proxy_source_quality_max,
        "repair_transfer_source_task_min": repair_transfer_source_task_min,
        "repair_phase_budget": repair_phase_budget,
        "repair_denoise_skeleton_max_step": repair_denoise_skeleton_max_step,
        "phase_source_history_char_ratio_min": phase_source_history_char_ratio_min,
        "phase_source_target_similarity_min": phase_source_target_similarity_min,
        "phase_source_text_similarity_min": phase_source_text_similarity_min,
        "repair_source_controls": repair_source_controls or [],
        "history_rescue_fractions": list(history_rescue_fractions),
        "history_rescue_visible": history_rescue_visible,
        "history_rescue_trigger": history_rescue_trigger,
        "history_rescue_source_controls": history_rescue_source_controls or [],
        "prompt_guided_rescue_trigger": prompt_guided_rescue_trigger,
        "prompt_guided_rescue_limit": prompt_guided_rescue_limit,
        "prompt_guided_rescue_source_quality_threshold": prompt_guided_rescue_source_quality_threshold,
        "prompt_guided_rescue_source_controls": prompt_guided_rescue_source_controls or [],
        "constraint_gap_rescue_trigger": constraint_gap_rescue_trigger,
        "constraint_gap_rescue_limit": constraint_gap_rescue_limit,
        "constraint_gap_rescue_min_terms": constraint_gap_rescue_min_terms,
        "constraint_gap_rescue_source_quality_floor": constraint_gap_rescue_source_quality_floor,
        "constraint_gap_rescue_source_quality_ceiling": constraint_gap_rescue_source_quality_ceiling,
        "constraint_gap_rescue_source_controls": constraint_gap_rescue_source_controls or [],
        "repair_promotion_margin": repair_promotion_margin,
        "repair_selector": repair_selector,
        "arms": {
            arm: _arm_summary(records)
            for arm, records in sorted(by_arm.items(), key=_arm_sort_key)
        },
        "by_candidate_arm": {
            f"{candidate}:{arm}": _arm_summary(records)
            for (candidate, arm), records in sorted(by_candidate_arm.items())
        },
        "by_family_arm": _nested_arm_summary(by_family_arm),
        "trajectory_task_delta_vs_fixed": _mean_delta(trajectory_records, fixed_records),
        "trajectory_task_delta_vs_random": _mean_delta(trajectory_records, random_records),
        "trajectory_wins_vs_fixed": _win_count(trajectory_records, fixed_records),
        "trajectory_wins_vs_random": _win_count(trajectory_records, random_records),
        "repair_eligible_task_count": len(repair_eligible_records),
        "oracle_generation_budget_per_task": _oracle_generation_budget(all_records),
        "oracle_task_score": _mean(_task_score(record) for record in oracle_records),
        "oracle_headroom_vs_trajectory": _mean_delta(oracle_records, trajectory_records),
        "oracle_wins_vs_trajectory": _win_count(oracle_records, trajectory_records),
        "selector_regret_vs_trajectory": _selector_regret_summary(
            oracle_records,
            trajectory_records,
        ),
        "repair_candidate_summary": _repair_candidate_summary(all_records, repair_records),
        "planning_span_target_rows": _planning_span_target_rows(all_records, repair_records),
        "repair_spend_gate_rows": gate_rows,
        "adaptive_source_gate_rows": _adaptive_source_gate_rows(
            all_records,
            arm_records,
            repair_source_policy=repair_source_policy,
            exact_task_trajectory_policy=exact_task_trajectory_policy,
            trajectory_selector=trajectory_selector,
            evolved_selector=evolved_selector,
            evolved_quality_margin=evolved_quality_margin,
            evolved_selector_tolerance=evolved_selector_tolerance,
            evolved_promotion_margin=evolved_promotion_margin,
            revision_promotion_margin=revision_promotion_margin,
            adaptive_source_gap_min_terms=adaptive_source_gap_min_terms,
            adaptive_source_quality_floor=adaptive_source_quality_floor,
            adaptive_source_quality_ceiling=adaptive_source_quality_ceiling,
        ),
        "comparison_rows": task_rows,
    }
    if evolved_records:
        scores.update(
            {
                "evolved_task_delta_vs_fixed": _mean_delta(evolved_records, fixed_records),
                "evolved_task_delta_vs_random": _mean_delta(evolved_records, random_records),
                "evolved_task_delta_vs_trajectory": _mean_delta(evolved_records, trajectory_records),
                "evolved_wins_vs_fixed": _win_count(evolved_records, fixed_records),
                "evolved_wins_vs_random": _win_count(evolved_records, random_records),
                "evolved_wins_vs_trajectory": _win_count(evolved_records, trajectory_records),
                "oracle_headroom_vs_evolved": _mean_delta(oracle_records, evolved_records),
                "oracle_wins_vs_evolved": _win_count(oracle_records, evolved_records),
                "selector_regret_vs_evolved": _selector_regret_summary(
                    oracle_records,
                    evolved_records,
                ),
            }
        )
    if repair_records:
        baseline_for_repair = evolved_records or trajectory_records
        repair_budget_delta = _mean_budget_delta(repair_records, baseline_for_repair)
        repair_delta_vs_evolved = _mean_delta(repair_records, baseline_for_repair)
        scores.update(
            {
                "repair_task_delta_vs_fixed": _mean_delta(repair_records, fixed_records),
                "repair_task_delta_vs_random": _mean_delta(repair_records, random_records),
                "repair_task_delta_vs_trajectory": _mean_delta(repair_records, trajectory_records),
                "repair_task_delta_vs_evolved": repair_delta_vs_evolved,
                "repair_generation_budget_delta_vs_evolved": repair_budget_delta,
                "repair_task_delta_per_extra_generation_vs_evolved": _safe_ratio(
                    repair_delta_vs_evolved,
                    repair_budget_delta,
                ),
                "repair_wins_vs_fixed": _win_count(repair_records, fixed_records),
                "repair_wins_vs_random": _win_count(repair_records, random_records),
                "repair_wins_vs_trajectory": _win_count(repair_records, trajectory_records),
                "repair_wins_vs_evolved": _win_count(repair_records, baseline_for_repair),
                "oracle_headroom_vs_repair": _mean_delta(oracle_records, repair_records),
                "oracle_wins_vs_repair": _win_count(oracle_records, repair_records),
                "selector_regret_vs_repair": _selector_regret_summary(
                    oracle_records,
                    repair_records,
                ),
            }
        )
    _attach_result_identity(scores, all_records, arm_records)
    return scores


def _attach_result_identity(
    scores: dict[str, object],
    all_records: list[dict[str, object]],
    arm_records: list[dict[str, object]],
) -> None:
    payload = {
        "all_records": _stable_hash_payload(all_records),
        "arm_records": _stable_hash_payload(arm_records),
        "score_summary": _stable_hash_payload(scores),
    }
    content_hash = _content_hash(payload)
    scores["content_hash"] = content_hash
    scores["run_id"] = f"diffusion-{content_hash[:16]}"


def _content_hash(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _stable_hash_payload(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _stable_hash_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in RESULT_IDENTITY_VOLATILE_KEYS
        }
    if isinstance(value, list | tuple):
        return [_stable_hash_payload(item) for item in value]
    return value


def _repair_candidate_summary(
    all_records: list[dict[str, object]],
    selected_repair_records: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    selected_identities = {
        _record_identity(record) for record in selected_repair_records if _repair_name(record)
    }
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in all_records:
        repair_name = _repair_name(record)
        if repair_name:
            groups[repair_name].append(record)
    source_records = _source_records_by_control(all_records)

    summaries: dict[str, dict[str, object]] = {}
    for repair_name, records in sorted(groups.items()):
        source_controls = sorted(
            {
                _repair_metadata_value(record, "source_control")
                for record in records
                if _repair_metadata_value(record, "source_control")
            }
        )
        source_states = sorted(
            {
                _repair_metadata_value(record, "source_state")
                for record in records
                if _repair_metadata_value(record, "source_state")
            }
        )
        summaries[repair_name] = {
            "count": len(records),
            "selected_count": sum(1 for record in records if _record_identity(record) in selected_identities),
            "source_controls": ",".join(source_controls),
            "source_states": ",".join(source_states),
            "mean_seed_masked_positions": _mean(
                _nested_float(record, ("repair", "seed_masked_positions")) for record in records
            ),
            "mean_span_literal_target_found": _mean(
                _repair_span_literal_target_found(record) for record in records
            ),
            "mean_span_fallback_used": _mean(_repair_span_fallback_used(record) for record in records),
            "mean_overpreservation_penalty": _mean(_repair_overpreservation_penalty(record) for record in records),
            "mean_contradiction_penalty": _mean(
                _planning_contradiction_penalty(record, str(record.get("prompt", ""))) for record in records
            ),
            "mean_planning_span_residue_penalty": _mean(
                _planning_span_residue_penalty(record) for record in records
            ),
            "mean_seed_realization_quality": _mean(
                _seed_realization_quality_score(record, str(record.get("prompt", ""))) for record in records
            ),
            "mean_seed_objective_score": _mean(
                _seed_objective_score(record, str(record.get("prompt", ""))) for record in records
            ),
            "mean_seed_realization_meta_penalty": _mean(
                float(
                    _seed_realization_quality_components(
                        record,
                        str(record.get("prompt", "")),
                    )["meta_penalty"]
                )
                for record in records
            ),
            "mean_seed_realization_control_coverage": _mean(
                float(
                    _seed_realization_quality_components(
                        record,
                        str(record.get("prompt", "")),
                    )["control_coverage"]
                )
                for record in records
            ),
            "mean_seed_semantic_preservation": _mean(
                float(
                    _seed_realization_quality_components(
                        record,
                        str(record.get("prompt", "")),
                    )["semantic_preservation_score"]
                )
                for record in records
            ),
            "mean_planning_quality_delta_vs_source": _mean(
                _repair_planning_quality_delta_vs_source(record, source_records) for record in records
            ),
            "mean_task_delta_vs_source": _mean(_repair_task_delta_vs_source(record) for record in records),
            "mean_proposal_task_score": _mean(_repair_proposal_task_score(record) for record in records),
            "mean_task_delta_vs_proposal": _mean(_repair_task_delta_vs_proposal(record) for record in records),
            "mean_self_repair_changed_answer": _mean(
                _repair_self_repair_changed_answer(record) for record in records
            ),
            "mean_self_repair_arithmetic_consistent": _mean(
                _repair_self_repair_arithmetic_consistent(record) for record in records
            ),
            "mean_self_repair_arithmetic_claim_count": _mean(
                _repair_self_repair_arithmetic_claim_count(record) for record in records
            ),
            "mean_self_repair_irrelevant_number_used": _mean(
                _repair_self_repair_irrelevant_number_used(record) for record in records
            ),
            "mean_self_repair_missing_required_operator_count": _mean(
                _repair_self_repair_missing_required_operator_count(record) for record in records
            ),
            "mean_self_repair_quantity_role_gap_count": _mean(
                _repair_self_repair_quantity_role_gap_count(record) for record in records
            ),
            "mean_self_repair_arithmetic_provenance_gap_count": _mean(
                _repair_self_repair_arithmetic_provenance_gap_count(record) for record in records
            ),
            "mean_self_repair_final_answer_role_gap_count": _mean(
                _repair_self_repair_final_answer_role_gap_count(record) for record in records
            ),
            "mean_self_repair_final_answer_object_gap_count": _mean(
                _repair_self_repair_final_answer_object_gap_count(record) for record in records
            ),
            "mean_self_repair_final_answer_target_gap_count": _mean(
                _repair_self_repair_final_answer_target_gap_count(record) for record in records
            ),
            "mean_self_repair_short_text_symbolic_gap_count": _mean(
                _repair_self_repair_short_text_symbolic_gap_count(record) for record in records
            ),
            "mean_self_repair_short_text_trace_gap_count": _mean(
                _repair_self_repair_short_text_trace_gap_count(record) for record in records
            ),
            "wins_vs_source": _repair_wins_vs_source(records),
            "mean_task_score": _mean(_task_score(record) for record in records),
            "mean_trajectory_score": _mean(_trajectory_score(record) for record in records),
            "mean_combined_score": _mean(_combined_score(record) for record in records),
        }
    return summaries


def _planning_span_target_rows(
    all_records: list[dict[str, object]],
    selected_repair_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected_identities = {
        _record_identity(record) for record in selected_repair_records if _repair_name(record)
    }
    rows = []
    for record in all_records:
        repair = record.get("repair")
        if not isinstance(repair, dict):
            continue
        target_scores = repair.get("planning_span_target_scores")
        if not isinstance(target_scores, list):
            continue
        for target in target_scores:
            if not isinstance(target, dict):
                continue
            rows.append(
                {
                    "candidate_key": str(record.get("candidate_key", "")),
                    "task_id": _task_id(record),
                    "repair": _repair_name(record),
                    "source_control": str(repair.get("source_control", "")),
                    "selected": _record_identity(record) in selected_identities,
                    "span": str(target.get("span", "")),
                    "score": _float_value(target.get("score"), default=0.0),
                    "source_relative_preservation": _float_value(
                        target.get("source_relative_preservation"),
                        default=0.0,
                    ),
                    "prompt_gap_miss": _float_value(target.get("prompt_gap_miss"), default=0.0),
                    "contradiction_relief": _float_value(
                        target.get("contradiction_relief"),
                        default=0.0,
                    ),
                    "keyword_coverage": _float_value(target.get("keyword_coverage"), default=0.0),
                    "fallback": bool(target.get("fallback", False)),
                }
            )
    return rows


def _adaptive_source_gate_rows(
    all_records: list[dict[str, object]],
    arm_records: list[dict[str, object]],
    *,
    repair_source_policy: str,
    exact_task_trajectory_policy: str,
    trajectory_selector: str,
    evolved_selector: str,
    evolved_quality_margin: float,
    evolved_selector_tolerance: float,
    evolved_promotion_margin: float,
    revision_promotion_margin: float,
    adaptive_source_gap_min_terms: int,
    adaptive_source_quality_floor: float,
    adaptive_source_quality_ceiling: float | None,
) -> list[dict[str, object]]:
    if repair_source_policy != "non_revision_plus_gap_trajectory":
        return []
    records_by_task: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    arms_by_task: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for record in all_records:
        records_by_task[(str(record.get("candidate_key", "")), _task_id(record))].append(record)
    for record in arm_records:
        arms_by_task[(str(record.get("candidate_key", "")), _task_id(record))][str(record.get("arm", ""))] = record

    rows: list[dict[str, object]] = []
    for (candidate_key, task_id), selected in sorted(arms_by_task.items()):
        trajectory_source = selected.get("trajectory_selected")
        evolved_record = selected.get("evolved") or trajectory_source
        fixed_record = selected.get("fixed")
        random_record = selected.get("random")
        if trajectory_source is None or evolved_record is None or fixed_record is None or random_record is None:
            continue
        task = trajectory_source.get("task")
        task_answer_type = str(task.get("answer_type", "rubric")) if isinstance(task, dict) else "rubric"
        if task_answer_type != "rubric":
            continue
        task_prompt = str(trajectory_source.get("prompt", ""))
        primary = _select_repair_source_record(
            "non_revision_evolved",
            selected_records={
                "fixed": fixed_record,
                "random": random_record,
                "trajectory_selected": trajectory_source,
            },
            evolved_record=evolved_record,
            candidate_records=records_by_task[(candidate_key, task_id)],
            task_prompt=task_prompt,
            task_answer_type=task_answer_type,
            exact_task_trajectory_policy=exact_task_trajectory_policy,
            trajectory_selector=trajectory_selector,
            evolved_selector=evolved_selector,
            evolved_quality_margin=evolved_quality_margin,
            evolved_selector_tolerance=evolved_selector_tolerance,
            evolved_promotion_margin=evolved_promotion_margin,
            revision_promotion_margin=revision_promotion_margin,
        )
        gate = _gap_trajectory_repair_source_gate(
            trajectory_source,
            primary,
            task_prompt,
            gap_min_terms=adaptive_source_gap_min_terms,
            quality_floor=adaptive_source_quality_floor,
            quality_ceiling=adaptive_source_quality_ceiling,
        )
        trajectory_control = str(gate["trajectory_control"])
        source_repair_records = [
            record
            for record in records_by_task[(candidate_key, task_id)]
            if _repair_name(record)
            and _repair_metadata_value(record, "source_control") == trajectory_control
        ]
        selected_source_repairs = [
            record
            for record in arm_records
            if str(record.get("candidate_key", "")) == candidate_key
            and _task_id(record) == task_id
            and str(record.get("arm", "")) == "repair_selected"
            and _repair_metadata_value(record, "source_control") == trajectory_control
        ]
        added_source = bool(gate["add"])
        rows.append(
            {
                "candidate_key": candidate_key,
                "task_id": task_id,
                "add": added_source,
                "reason": str(gate["reason"]),
                "primary_control": str(gate["primary_control"]),
                "trajectory_control": trajectory_control,
                "trajectory_planning_quality": float(gate["trajectory_planning_quality"]),
                "prompt_gap_term_count": int(gate["prompt_gap_term_count"]),
                "prompt_gap_terms": list(gate["prompt_gap_terms"])[:8],
                "quality_ceiling": gate["quality_ceiling"],
                "generated_repair_count": len(source_repair_records) if added_source else 0,
                "selected_repair_count": len(selected_source_repairs) if added_source else 0,
            }
        )
    return rows


def _source_records_by_control(
    records: list[dict[str, object]],
) -> dict[tuple[str, str, str], dict[str, object]]:
    sources = {}
    for record in records:
        if _is_repair_record(record):
            continue
        control_name = _control_name(record)
        if not control_name:
            continue
        sources[(str(record.get("candidate_key", "")), _task_id(record), control_name)] = record
    return sources


def _repair_planning_quality_delta_vs_source(
    record: dict[str, object],
    source_records: dict[tuple[str, str, str], dict[str, object]],
) -> float:
    repair_score = _record_planning_quality_score(record)
    source_score = _repair_source_planning_quality_score(record)
    if source_score is None:
        source_record = source_records.get(
            (
                str(record.get("candidate_key", "")),
                _task_id(record),
                _repair_metadata_value(record, "source_control"),
            )
        )
        if source_record is not None:
            source_score = _record_planning_quality_score(source_record)
    if repair_score is None or source_score is None:
        return 0.0
    return repair_score - source_score


def _repair_task_delta_vs_source(record: dict[str, object]) -> float:
    source_score = _repair_source_task_score(record)
    if source_score is None:
        return 0.0
    return _task_score(record) - source_score


def _repair_proposal_task_score(record: dict[str, object]) -> float:
    return _nested_float(record, ("repair", "proposal_task_score"))


def _repair_task_delta_vs_proposal(record: dict[str, object]) -> float:
    proposal_score = _repair_proposal_task_score(record)
    if proposal_score <= 0.0 and not _repair_metadata_value(record, "proposal"):
        return 0.0
    return _task_score(record) - proposal_score


def _repair_self_repair_changed_answer(record: dict[str, object]) -> float:
    if _repair_name(record) not in SELF_REPAIR_EVIDENCE_NAMES:
        return 0.0
    return 1.0 if _repair_metadata_bool(record, "self_repair_changed_answer") else 0.0


def _repair_self_repair_arithmetic_consistent(record: dict[str, object]) -> float:
    if _repair_name(record) not in SELF_REPAIR_EVIDENCE_NAMES:
        return 0.0
    if _repair_metadata_bool(
        record,
        "self_repair_arithmetic_consistent",
        default=_arithmetic_claims_consistent(str(record.get("text", ""))),
    ):
        return 1.0
    return 0.0


def _repair_self_repair_arithmetic_claim_count(record: dict[str, object]) -> float:
    if _repair_name(record) not in SELF_REPAIR_EVIDENCE_NAMES:
        return 0.0
    value = _repair_metadata_value(record, "self_repair_arithmetic_claim_count")
    if value:
        return float(value)
    return float(_arithmetic_claim_count(str(record.get("text", ""))))


def _repair_self_repair_irrelevant_number_used(record: dict[str, object]) -> float:
    if _repair_name(record) not in SELF_REPAIR_EVIDENCE_NAMES:
        return 0.0
    repair = record.get("repair")
    if isinstance(repair, dict) and "self_repair_irrelevant_number_used" in repair:
        return 1.0 if _repair_metadata_bool(record, "self_repair_irrelevant_number_used") else 0.0
    return 1.0 if _repair_irrelevant_prompt_number_used(record, str(record.get("prompt", ""))) else 0.0


def _repair_self_repair_missing_required_operator_count(record: dict[str, object]) -> float:
    if _repair_name(record) not in SELF_REPAIR_EVIDENCE_NAMES:
        return 0.0
    repair = record.get("repair")
    if isinstance(repair, dict) and "self_repair_missing_required_operators" in repair:
        value = repair.get("self_repair_missing_required_operators")
        return float(len(value)) if isinstance(value, list | tuple | set) else 0.0
    return float(len(_repair_missing_required_operators(record, str(record.get("prompt", "")))))


def _repair_self_repair_quantity_role_gap_count(record: dict[str, object]) -> float:
    if _repair_name(record) not in SELF_REPAIR_EVIDENCE_NAMES:
        return 0.0
    repair = record.get("repair")
    if isinstance(repair, dict) and "self_repair_quantity_role_gaps" in repair:
        value = repair.get("self_repair_quantity_role_gaps")
        return float(len(value)) if isinstance(value, list | tuple | set) else 0.0
    return float(len(_repair_quantity_role_gaps(record, str(record.get("prompt", "")))))


def _repair_self_repair_arithmetic_provenance_gap_count(record: dict[str, object]) -> float:
    if _repair_name(record) not in SELF_REPAIR_EVIDENCE_NAMES:
        return 0.0
    repair = record.get("repair")
    if isinstance(repair, dict) and "self_repair_arithmetic_provenance_gaps" in repair:
        value = repair.get("self_repair_arithmetic_provenance_gaps")
        return float(len(value)) if isinstance(value, list | tuple | set) else 0.0
    return float(len(_repair_arithmetic_provenance_gaps(record, str(record.get("prompt", "")))))


def _repair_self_repair_final_answer_role_gap_count(record: dict[str, object]) -> float:
    if _repair_name(record) not in SELF_REPAIR_EVIDENCE_NAMES:
        return 0.0
    repair = record.get("repair")
    if isinstance(repair, dict) and "self_repair_final_answer_role_gaps" in repair:
        value = repair.get("self_repair_final_answer_role_gaps")
        return float(len(value)) if isinstance(value, list | tuple | set) else 0.0
    return float(len(_repair_final_answer_role_gaps(record, str(record.get("prompt", "")))))


def _repair_self_repair_final_answer_object_gap_count(record: dict[str, object]) -> float:
    if _repair_name(record) not in SELF_REPAIR_EVIDENCE_NAMES:
        return 0.0
    repair = record.get("repair")
    if isinstance(repair, dict) and "self_repair_final_answer_object_gaps" in repair:
        value = repair.get("self_repair_final_answer_object_gaps")
        return float(len(value)) if isinstance(value, list | tuple | set) else 0.0
    return float(len(_repair_final_answer_object_gaps(record, str(record.get("prompt", "")))))


def _repair_self_repair_final_answer_target_gap_count(record: dict[str, object]) -> float:
    if _repair_name(record) not in SELF_REPAIR_EVIDENCE_NAMES:
        return 0.0
    repair = record.get("repair")
    if isinstance(repair, dict) and "self_repair_final_answer_target_gaps" in repair:
        value = repair.get("self_repair_final_answer_target_gaps")
        return float(len(value)) if isinstance(value, list | tuple | set) else 0.0
    return float(len(_repair_final_answer_target_gaps(record, str(record.get("prompt", "")))))


def _repair_self_repair_short_text_symbolic_gap_count(record: dict[str, object]) -> float:
    if _repair_name(record) not in SELF_REPAIR_EVIDENCE_NAMES:
        return 0.0
    repair = record.get("repair")
    if isinstance(repair, dict) and "self_repair_short_text_symbolic_gaps" in repair:
        value = repair.get("self_repair_short_text_symbolic_gaps")
        return float(len(value)) if isinstance(value, list | tuple | set) else 0.0
    return float(len(_repair_short_text_symbolic_gaps(record, str(record.get("prompt", "")))))


def _repair_self_repair_short_text_trace_gap_count(record: dict[str, object]) -> float:
    if _repair_name(record) not in SELF_REPAIR_EVIDENCE_NAMES:
        return 0.0
    repair = record.get("repair")
    if isinstance(repair, dict) and "self_repair_short_text_trace_gaps" in repair:
        value = repair.get("self_repair_short_text_trace_gaps")
        return float(len(value)) if isinstance(value, list | tuple | set) else 0.0
    return float(len(_repair_short_text_trace_gaps(record, str(record.get("prompt", "")))))


def _repair_wins_vs_source(records: list[dict[str, object]]) -> dict[str, int]:
    counts = {"wins": 0, "ties": 0, "losses": 0}
    for record in records:
        source_score = _repair_source_task_score(record)
        if source_score is None:
            continue
        delta = _task_score(record) - source_score
        if delta > 1e-9:
            counts["wins"] += 1
        elif delta < -1e-9:
            counts["losses"] += 1
        else:
            counts["ties"] += 1
    return counts


def _exact_trajectory_selection_source_counts(records: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        arm = str(record.get("arm", ""))
        if arm not in {"trajectory_selected", "evolved"}:
            continue
        selection = record.get("exact_trajectory_selection")
        source = ""
        if isinstance(selection, dict):
            source = str(selection.get("source", "unknown") or "unknown")
        elif str(record.get("arm_selection_reason", "")) == "exact_answer_proposal_history_no_match_kept_fixed":
            source = "fallback"
        if source:
            counts[f"{arm}:{source}"] += 1
    return dict(sorted(counts.items()))


def _history_mutability_summary(records: list[dict[str, object]]) -> dict[str, object]:
    history_records = []
    for record in records:
        summary = record.get("trajectory_summary")
        if isinstance(summary, dict) and "sampled_history_is_monotonic_fill" in summary:
            history_records.append(summary)
    count = len(history_records)
    monotonic_count = sum(
        1 for summary in history_records if bool(summary.get("sampled_history_is_monotonic_fill", False))
    )
    return {
        "count": count,
        "monotonic_fill_count": monotonic_count,
        "committed_token_change_count": int(
            sum(_int_value(summary.get("committed_token_change_count"), default=0) for summary in history_records)
        ),
        "committed_token_remask_count": int(
            sum(_int_value(summary.get("committed_token_remask_count"), default=0) for summary in history_records)
        ),
        "remasked_token_rewrite_count": int(
            sum(_int_value(summary.get("remasked_token_rewrite_count"), default=0) for summary in history_records)
        ),
        "mask_count_increase_count": int(
            sum(_int_value(summary.get("mask_count_increase_count"), default=0) for summary in history_records)
        ),
    }


def _repair_source_task_score(record: dict[str, object]) -> float | None:
    repair = record.get("repair")
    if not isinstance(repair, dict):
        return None
    source_score = repair.get("source_task_score")
    if isinstance(source_score, int | float) and not isinstance(source_score, bool):
        return float(source_score)
    return None


def _repair_source_planning_quality_score(record: dict[str, object]) -> float | None:
    repair = record.get("repair")
    if not isinstance(repair, dict):
        return None
    source_score = repair.get("source_planning_quality_score")
    if isinstance(source_score, int | float) and not isinstance(source_score, bool):
        return float(source_score)
    return None


def render_report(scores: dict[str, object]) -> str:
    arms = scores.get("arms", {})
    has_evolved = isinstance(arms, dict) and "evolved" in arms
    has_repair = isinstance(arms, dict) and "repair_selected" in arms
    lines = [
        "# Diffusion Schedule-Selection Benchmark Report",
        "",
        f"Full model generations: `{scores['all_generation_count']}`",
        f"Counterfactual probe generations: `{int(scores.get('counterfactual_probe_generation_count', 0))}`",
        f"Arm selections: `{scores['arm_selection_count']}`",
        f"Run ID: `{scores.get('run_id', '')}`",
        f"Content hash: `{scores.get('content_hash', '')}`",
        f"Exact-task trajectory policy: `{scores['exact_task_trajectory_policy']}`",
        f"Trajectory selector: `{scores['trajectory_selector']}`",
        f"Evolved selector: `{scores.get('evolved_selector', DEFAULT_EVOLVED_SELECTOR)}`",
        f"Evolved quality margin: `{scores.get('evolved_quality_margin', DEFAULT_EVOLVED_QUALITY_MARGIN):.3f}`",
        f"Evolved selector tolerance: `{scores.get('evolved_selector_tolerance', DEFAULT_EVOLVED_SELECTOR_TOLERANCE):.3f}`",
        f"Evolved promotion margin: `{scores.get('evolved_promotion_margin', 0.0):.3f}`",
        f"Revision promotion margin: `{scores.get('revision_promotion_margin', DEFAULT_REVISION_PROMOTION_MARGIN):.3f}`",
        f"Revision schedules included: `{bool(scores.get('include_revision_schedules', False))}`",
        f"Revision remask fraction: `{scores.get('revision_remask_fraction', 0.25):.3f}`",
        f"Revision steps: `{int(scores.get('revision_steps', 16))}`",
        f"Exact verifier revision: `{bool(scores.get('exact_verifier_revision', False))}`",
        f"History mutability: `{_format_history_mutability(scores.get('history_mutability', {}))}`",
        f"History repairs included: `{bool(scores.get('include_history_repairs', False))}`",
        f"Repair pack: `{scores.get('repair_pack', 'prefix')}`",
        f"Repair source policy: `{scores.get('repair_source_policy', 'evolved')}`",
        f"Adaptive source gate mode: `{scores.get('adaptive_source_gate_mode', 'custom')}`",
        f"Adaptive source gap min terms: `{int(scores.get('adaptive_source_gap_min_terms', DEFAULT_ADAPTIVE_SOURCE_GAP_MIN_TERMS))}`",
        f"Adaptive source quality floor: `{scores.get('adaptive_source_quality_floor', DEFAULT_ADAPTIVE_SOURCE_QUALITY_FLOOR):.3f}`",
        "Adaptive source quality ceiling: "
        f"`{_format_optional_float(scores.get('adaptive_source_quality_ceiling'))}`",
        f"History repair fractions: `{_format_fraction_list(scores.get('history_repair_fractions', []))}`",
        f"History visible repair included: `{bool(scores.get('include_history_visible_repair', False))}`",
        f"Repair spend trigger: `{scores.get('repair_spend_trigger', 'always')}`",
        f"Counterfactual probe mode: `{scores.get('counterfactual_probe_mode', 'triage')}`",
        f"Counterfactual probe policy: `{scores.get('counterfactual_probe_policy', COUNTERFACTUAL_MICRO_PROBE_POLICY_ID)}`",
        f"Repair source-quality threshold: `{scores.get('repair_source_quality_threshold', 0.0):.3f}`",
        f"Repair source min chars: `{int(scores.get('repair_source_min_chars', 0))}`",
        f"Repair source prompt-gap min: `{int(scores.get('repair_source_prompt_gap_min', 0))}`",
        f"Repair source prompt-gap max: `{int(scores.get('repair_source_prompt_gap_max', 999))}`",
        (
            "Repair source prompt coverage band: "
            f"`{scores.get('repair_source_prompt_coverage_min', 0.0):.3f}-"
            f"{scores.get('repair_source_prompt_coverage_max', 1.0):.3f}`"
        ),
        (
            "Repair value-proxy source-quality max: "
            f"`{scores.get('repair_value_proxy_source_quality_max', 0.31):.3f}`"
        ),
        (
            "Repair transfer source-task min: "
            f"`{scores.get('repair_transfer_source_task_min', DECOMPOSED_SPEND_TRANSFER_SOURCE_TASK_MIN):.4f}`"
        ),
        f"Repair phase budget: `{scores.get('repair_phase_budget', 'custom')}`",
        f"Repair denoise skeleton max step: `{_format_optional_float(scores.get('repair_denoise_skeleton_max_step'))}`",
        (
            "Phase-source threshold band: "
            f"`target>={scores.get('phase_source_target_similarity_min', PHASE_SOURCE_TARGET_SIMILARITY_MIN):.3f}, "
            f"text>={scores.get('phase_source_text_similarity_min', PHASE_SOURCE_TEXT_SIMILARITY_MIN):.3f}, "
            "chars>="
            f"{scores.get('phase_source_history_char_ratio_min', PHASE_SOURCE_HISTORY_CHAR_RATIO_MIN):.3f}`"
        ),
        f"Repair source controls: `{_format_string_list(scores.get('repair_source_controls', []))}`",
        f"History rescue fractions: `{_format_fraction_list(scores.get('history_rescue_fractions', []))}`",
        f"History rescue visible: `{bool(scores.get('history_rescue_visible', False))}`",
        f"History rescue trigger: `{scores.get('history_rescue_trigger', 'baseline')}`",
        f"History rescue source controls: `{_format_string_list(scores.get('history_rescue_source_controls', []))}`",
        f"Prompt-guided rescue trigger: `{scores.get('prompt_guided_rescue_trigger', 'off')}`",
        f"Prompt-guided rescue limit: `{int(scores.get('prompt_guided_rescue_limit', 0))}`",
        f"Prompt-guided rescue source-quality threshold: `{scores.get('prompt_guided_rescue_source_quality_threshold', 0.0):.3f}`",
        f"Prompt-guided rescue source controls: `{_format_string_list(scores.get('prompt_guided_rescue_source_controls', []))}`",
        f"Constraint-gap rescue trigger: `{scores.get('constraint_gap_rescue_trigger', 'off')}`",
        f"Constraint-gap rescue limit: `{int(scores.get('constraint_gap_rescue_limit', 0))}`",
        f"Constraint-gap rescue min terms: `{int(scores.get('constraint_gap_rescue_min_terms', 0))}`",
        (
            "Constraint-gap rescue source-quality band: "
            f"`{scores.get('constraint_gap_rescue_source_quality_floor', 0.0):.3f}-"
            f"{scores.get('constraint_gap_rescue_source_quality_ceiling', 0.0):.3f}`"
        ),
        f"Constraint-gap rescue source controls: `{_format_string_list(scores.get('constraint_gap_rescue_source_controls', []))}`",
        f"Repair selector: `{scores.get('repair_selector', DEFAULT_REPAIR_SELECTOR)}`",
        f"Repair promotion margin: `{scores.get('repair_promotion_margin', 0.0):.3f}`",
        f"Trajectory task delta vs fixed: `{scores['trajectory_task_delta_vs_fixed']:.3f}`",
        f"Trajectory task delta vs random: `{scores['trajectory_task_delta_vs_random']:.3f}`",
        f"Trajectory wins/ties/losses vs fixed: `{_format_wins(scores['trajectory_wins_vs_fixed'])}`",
        f"Trajectory wins/ties/losses vs random: `{_format_wins(scores['trajectory_wins_vs_random'])}`",
        f"Oracle generation budget/task: `{scores['oracle_generation_budget_per_task']:.2f}`",
        f"Oracle task score: `{scores['oracle_task_score']:.3f}`",
        f"Oracle headroom vs trajectory: `{scores['oracle_headroom_vs_trajectory']:.3f}`",
        f"Oracle wins/ties/losses vs trajectory: `{_format_wins(scores['oracle_wins_vs_trajectory'])}`",
        f"Selector regret vs trajectory: `{_format_selector_regret(scores['selector_regret_vs_trajectory'])}`",
    ]
    if scores.get("exact_task_trajectory_policy") == "proposal_history":
        lines.append(
            "Exact proposal-history sources: "
            f"`{_format_count_map(scores.get('exact_trajectory_selection_source_counts', {}))}`"
        )
    if has_evolved:
        lines.extend(
            [
                f"Evolved task delta vs fixed: `{scores['evolved_task_delta_vs_fixed']:.3f}`",
                f"Evolved task delta vs random: `{scores['evolved_task_delta_vs_random']:.3f}`",
                f"Evolved task delta vs trajectory: `{scores['evolved_task_delta_vs_trajectory']:.3f}`",
                f"Evolved wins/ties/losses vs fixed: `{_format_wins(scores['evolved_wins_vs_fixed'])}`",
                f"Evolved wins/ties/losses vs random: `{_format_wins(scores['evolved_wins_vs_random'])}`",
                f"Evolved wins/ties/losses vs trajectory: `{_format_wins(scores['evolved_wins_vs_trajectory'])}`",
                f"Oracle headroom vs evolved: `{scores['oracle_headroom_vs_evolved']:.3f}`",
                f"Oracle wins/ties/losses vs evolved: `{_format_wins(scores['oracle_wins_vs_evolved'])}`",
                f"Selector regret vs evolved: `{_format_selector_regret(scores['selector_regret_vs_evolved'])}`",
            ]
        )
    if has_repair:
        repair_count = int(arms["repair_selected"].get("count", 0)) if isinstance(arms, dict) else 0
        trajectory_count = int(arms["trajectory_selected"].get("count", 0)) if isinstance(arms, dict) else 0
        repair_eligible_count = int(scores.get("repair_eligible_task_count", 0))
        lines.extend(
            [
                f"Repair arm coverage: `{repair_count}/{trajectory_count}` overall",
                f"Repair eligible coverage: `{repair_count}/{repair_eligible_count}`",
                f"Repair task delta vs fixed: `{scores['repair_task_delta_vs_fixed']:.3f}`",
                f"Repair task delta vs random: `{scores['repair_task_delta_vs_random']:.3f}`",
                f"Repair task delta vs trajectory: `{scores['repair_task_delta_vs_trajectory']:.3f}`",
                f"Repair task delta vs evolved: `{scores['repair_task_delta_vs_evolved']:.3f}`",
                f"Repair generation budget delta vs evolved: `{scores['repair_generation_budget_delta_vs_evolved']:.2f}`",
                (
                    "Repair task delta per extra generation vs evolved: "
                    f"`{scores['repair_task_delta_per_extra_generation_vs_evolved']:.3f}`"
                ),
                f"Repair wins/ties/losses vs evolved: `{_format_wins(scores['repair_wins_vs_evolved'])}`",
                f"Oracle headroom vs repair: `{scores['oracle_headroom_vs_repair']:.3f}`",
                f"Oracle wins/ties/losses vs repair: `{_format_wins(scores['oracle_wins_vs_repair'])}`",
                f"Selector regret vs repair: `{_format_selector_regret(scores['selector_regret_vs_repair'])}`",
            ]
        )
        lines.extend(_lean_three_arm_headline(scores, arms))
    lines.extend(
        [
            "",
            "## Arm Summary",
            "",
            "| Arm | Count | Budget/Task | Task | Trajectory | Combined |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if isinstance(arms, dict):
        for arm in ARMS:
            summary = arms.get(arm)
            if not isinstance(summary, dict):
                continue
            lines.append(
                "| "
                f"{arm} | "
                f"{summary['count']} | "
                f"{summary['mean_generation_budget_per_task']:.2f} | "
                f"{summary['mean_task_score']:.3f} | "
                f"{summary['mean_trajectory_score']:.3f} | "
                f"{summary['mean_combined_score']:.3f} |"
            )

    family_arms = scores.get("by_family_arm", {})
    if isinstance(family_arms, dict) and family_arms:
        lines.extend(
            [
                "",
                "## Family Arm Summary",
                "",
                "| Family | Arm | Count | Budget/Task | Task | Trajectory | Combined |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for family, family_summary in sorted(family_arms.items()):
            if not isinstance(family_summary, dict):
                continue
            for arm in ARMS:
                summary = family_summary.get(arm)
                if not isinstance(summary, dict):
                    continue
                lines.append(
                    "| "
                    f"{family} | "
                    f"{arm} | "
                    f"{summary['count']} | "
                    f"{summary['mean_generation_budget_per_task']:.2f} | "
                    f"{summary['mean_task_score']:.3f} | "
                    f"{summary['mean_trajectory_score']:.3f} | "
                    f"{summary['mean_combined_score']:.3f} |"
                )

    adaptive_source_rows = scores.get("adaptive_source_gate_rows", [])
    if isinstance(adaptive_source_rows, list) and adaptive_source_rows:
        lines.extend(
            [
                "",
                "## Adaptive Source Gate",
                "",
                "| Candidate | Task | Add Source | Reason | Primary | Trajectory | Gap Terms | Traj PQ | Quality Ceiling | Generated | Selected | Gap Term Sample |",
                "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in adaptive_source_rows:
            if not isinstance(row, dict):
                continue
            gap_terms = row.get("prompt_gap_terms", [])
            gap_sample = ",".join(str(term) for term in gap_terms) if isinstance(gap_terms, list) else ""
            lines.append(
                "| "
                f"{row.get('candidate_key', '')} | "
                f"{row.get('task_id', '')} | "
                f"{bool(row.get('add', False))} | "
                f"{row.get('reason', '')} | "
                f"{row.get('primary_control', '')} | "
                f"{row.get('trajectory_control', '')} | "
                f"{int(row.get('prompt_gap_term_count', 0))} | "
                f"{float(row.get('trajectory_planning_quality', 0.0)):.3f} | "
                f"{_format_optional_float(row.get('quality_ceiling'))} | "
                f"{int(row.get('generated_repair_count', 0))} | "
                f"{int(row.get('selected_repair_count', 0))} | "
                f"{gap_sample} |"
            )

    repair_spend_gate_rows = scores.get("repair_spend_gate_rows", [])
    if isinstance(repair_spend_gate_rows, list) and repair_spend_gate_rows:
        lines.extend(
            [
                "",
                "## Repair Spend Gate Diagnostics",
                "",
                "| Candidate | Task | Source | Run | Reason | Probe | Probe Observation | Source Task | Source PQ | Chars | Needs Repair | Gap Terms | Coverage | Repairable Band | Denoise Skeleton | Skeleton Step | Step Frac | Skeleton Coverage | Peak Coverage |",
                "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in repair_spend_gate_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                f"{row.get('candidate_key', '')} | "
                f"{row.get('task_id', '')} | "
                f"{row.get('source_control', '')} | "
                f"{bool(row.get('should_run', False))} | "
                f"{row.get('reason', '')} | "
                f"{bool(row.get('would_probe', False))} | "
                f"{row.get('counterfactual_probe_observation', '')} | "
                f"{float(row.get('source_task_score', 0.0)):.3f} | "
                f"{float(row.get('source_quality', 0.0)):.3f} | "
                f"{int(row.get('source_chars', 0))} | "
                f"{bool(row.get('source_needs_repair', False))} | "
                f"{int(row.get('prompt_gap_count', 0))} | "
                f"{float(row.get('prompt_coverage', 0.0)):.3f} | "
                f"{bool(row.get('in_repairable_band', False))} | "
                f"{bool(row.get('has_repairable_denoise_skeleton', False))} | "
                f"{_format_optional_float(row.get('first_repairable_denoise_skeleton_step'))} | "
                f"{_format_optional_float(row.get('first_repairable_denoise_skeleton_step_fraction'))} | "
                f"{_format_optional_float(row.get('first_repairable_denoise_skeleton_coverage'))} | "
                f"{_format_optional_float(row.get('peak_denoise_prompt_coverage'))} |"
            )

    repair_candidate_summary = scores.get("repair_candidate_summary", {})
    if isinstance(repair_candidate_summary, dict) and repair_candidate_summary:
        lines.extend(
            [
                "",
                "## Repair Candidate Diagnostics",
                "",
                "| Candidate | Count | Selected | Source Controls | Source States | Masked/Run | Span Localized | Span Fallback | Guard Penalty | Risk Penalty | Span Residue | PQ Delta | Task Delta | Proposal Task | Task-vs-Proposal | Self Changed | Arithmetic OK | Arithmetic Claims | Irrelevant # Used | Missing Ops | Role Gaps | Provenance Gaps | Final Role Gaps | Object Gaps | Target Gaps | Symbolic Gaps | Trace Gaps | W/T/L vs Source | Task | Trajectory | Combined |",
                "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for name, summary in sorted(repair_candidate_summary.items()):
            if not isinstance(summary, dict):
                continue
            lines.append(
                "| "
                f"{name} | "
                f"{summary['count']} | "
                f"{summary['selected_count']} | "
                f"{summary.get('source_controls', '')} | "
                f"{summary['source_states']} | "
                f"{summary['mean_seed_masked_positions']:.1f} | "
                f"{summary['mean_span_literal_target_found']:.3f} | "
                f"{summary['mean_span_fallback_used']:.3f} | "
                f"{summary['mean_overpreservation_penalty']:.3f} | "
                f"{summary['mean_contradiction_penalty']:.3f} | "
                f"{summary['mean_planning_span_residue_penalty']:.3f} | "
                f"{summary['mean_planning_quality_delta_vs_source']:.3f} | "
                f"{summary['mean_task_delta_vs_source']:.3f} | "
                f"{summary['mean_proposal_task_score']:.3f} | "
                f"{summary['mean_task_delta_vs_proposal']:.3f} | "
                f"{summary['mean_self_repair_changed_answer']:.3f} | "
                f"{summary['mean_self_repair_arithmetic_consistent']:.3f} | "
                f"{summary['mean_self_repair_arithmetic_claim_count']:.1f} | "
                f"{summary['mean_self_repair_irrelevant_number_used']:.3f} | "
                f"{summary['mean_self_repair_missing_required_operator_count']:.1f} | "
                f"{summary['mean_self_repair_quantity_role_gap_count']:.1f} | "
                f"{summary['mean_self_repair_arithmetic_provenance_gap_count']:.1f} | "
                f"{summary['mean_self_repair_final_answer_role_gap_count']:.1f} | "
                f"{summary['mean_self_repair_final_answer_object_gap_count']:.1f} | "
                f"{summary['mean_self_repair_final_answer_target_gap_count']:.1f} | "
                f"{summary['mean_self_repair_short_text_symbolic_gap_count']:.1f} | "
                f"{summary['mean_self_repair_short_text_trace_gap_count']:.1f} | "
                f"{_format_wins(summary['wins_vs_source'])} | "
                f"{summary['mean_task_score']:.3f} | "
                f"{summary['mean_trajectory_score']:.3f} | "
                f"{summary['mean_combined_score']:.3f} |"
            )

    span_target_rows = scores.get("planning_span_target_rows", [])
    if isinstance(span_target_rows, list) and span_target_rows:
        lines.extend(
            [
                "",
                "## Planning Span Target Diagnostics",
                "",
                "| Candidate | Task | Selected | Source | Score | Preserve | Gap Miss | Contradiction Relief | Keyword Coverage | Fallback | Span |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in span_target_rows:
            if not isinstance(row, dict):
                continue
            span = _compact_text(str(row.get("span", "")).replace("|", "/"), max_chars=90)
            lines.append(
                "| "
                f"{row.get('candidate_key', '')} | "
                f"{row.get('task_id', '')} | "
                f"{bool(row.get('selected', False))} | "
                f"{row.get('source_control', '')} | "
                f"{float(row.get('score', 0.0)):.3f} | "
                f"{float(row.get('source_relative_preservation', 0.0)):.3f} | "
                f"{float(row.get('prompt_gap_miss', 0.0)):.3f} | "
                f"{float(row.get('contradiction_relief', 0.0)):.3f} | "
                f"{float(row.get('keyword_coverage', 0.0)):.3f} | "
                f"{bool(row.get('fallback', False))} | "
                f"{span} |"
            )

    if has_repair:
        comparison_header = (
            "| Candidate | Task | Fixed | Random | Trajectory | Evolved | Repair | Oracle | "
            "Trajectory Reason | Evolved Reason | Repair Reason | Repair Source Control | Repair Source State | Repair Source Step | "
            "Traj Selector | Evolved Selector | Repair Selector | "
            "Repair Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Repair Task | "
            "Repair Delta vs Evolved | Oracle Task | Oracle Delta vs Repair |"
        )
        comparison_rule = (
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
    elif has_evolved:
        comparison_header = (
            "| Candidate | Task | Fixed | Random | Trajectory | Evolved | "
            "Oracle | Trajectory Reason | Evolved Reason | Traj Selector | Evolved Selector | "
            "Selector Edge | Fixed Task | Random Task | Trajectory Task | Evolved Task | Trajectory Delta vs Fixed | "
            "Evolved Delta vs Fixed | Evolved Delta vs Trajectory | Oracle Task | Oracle Delta vs Evolved |"
        )
        comparison_rule = (
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
    else:
        comparison_header = (
            "| Candidate | Task | Fixed | Random | Trajectory | Reason | Selector | "
            "Fixed Task | Random Task | Trajectory Task | Delta vs Fixed | Delta vs Random | Oracle | Oracle Task | Oracle Delta vs Trajectory |"
        )
        comparison_rule = "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"
    lines.extend(
        [
            "",
            "## Task Comparisons",
            "",
            comparison_header,
            comparison_rule,
        ]
    )
    rows = scores.get("comparison_rows", [])
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            if has_repair:
                lines.append(
                    "| "
                    f"{row['candidate']} | "
                    f"{row['task_id']} | "
                    f"{row['fixed_schedule']} | "
                    f"{row['random_schedule']} | "
                    f"{row['trajectory_schedule']} | "
                    f"{row['evolved_schedule']} | "
                    f"{row['repair_control']} | "
                    f"{row['oracle_schedule']} | "
                    f"{row['trajectory_selection_reason']} | "
                    f"{row['evolved_selection_reason']} | "
                    f"{row['repair_selection_reason']} | "
                    f"{row.get('repair_source_control', '')} | "
                    f"{row['repair_source_state']} | "
                    f"{row['repair_source_history_step']} | "
                    f"{row['trajectory_selector_score']:.3f} | "
                    f"{row['evolved_selector_score']:.3f} | "
                    f"{row['repair_selector_score']:.3f} | "
                    f"{row['repair_selector_edge']:.3f} | "
                    f"{row['fixed_task_score']:.3f} | "
                    f"{row['random_task_score']:.3f} | "
                    f"{row['trajectory_task_score']:.3f} | "
                    f"{row['evolved_task_score']:.3f} | "
                    f"{row['repair_task_score']:.3f} | "
                    f"{row['repair_delta_vs_evolved']:.3f} | "
                    f"{row['oracle_task_score']:.3f} | "
                    f"{row['oracle_delta_vs_repair']:.3f} |"
                )
            elif has_evolved:
                lines.append(
                    "| "
                    f"{row['candidate']} | "
                    f"{row['task_id']} | "
                    f"{row['fixed_schedule']} | "
                    f"{row['random_schedule']} | "
                    f"{row['trajectory_schedule']} | "
                    f"{row['evolved_schedule']} | "
                    f"{row['oracle_schedule']} | "
                    f"{row['trajectory_selection_reason']} | "
                    f"{row['evolved_selection_reason']} | "
                    f"{row['trajectory_selector_score']:.3f} | "
                    f"{row['evolved_selector_score']:.3f} | "
                    f"{row['evolved_selector_edge']:.3f} | "
                    f"{row['fixed_task_score']:.3f} | "
                    f"{row['random_task_score']:.3f} | "
                    f"{row['trajectory_task_score']:.3f} | "
                    f"{row['evolved_task_score']:.3f} | "
                    f"{row['trajectory_delta_vs_fixed']:.3f} | "
                    f"{row['evolved_delta_vs_fixed']:.3f} | "
                    f"{row['evolved_delta_vs_trajectory']:.3f} | "
                    f"{row['oracle_task_score']:.3f} | "
                    f"{row['oracle_delta_vs_evolved']:.3f} |"
                )
            else:
                lines.append(
                    "| "
                    f"{row['candidate']} | "
                    f"{row['task_id']} | "
                    f"{row['fixed_schedule']} | "
                    f"{row['random_schedule']} | "
                    f"{row['trajectory_schedule']} | "
                    f"{row['trajectory_selection_reason']} | "
                    f"{row['trajectory_selector_score']:.3f} | "
                    f"{row['fixed_task_score']:.3f} | "
                    f"{row['random_task_score']:.3f} | "
                    f"{row['trajectory_task_score']:.3f} | "
                    f"{row['trajectory_delta_vs_fixed']:.3f} | "
                    f"{row['trajectory_delta_vs_random']:.3f} | "
                    f"{row['oracle_schedule']} | "
                    f"{row['oracle_task_score']:.3f} | "
                    f"{row['oracle_delta_vs_trajectory']:.3f} |"
                )
    return "\n".join(lines) + "\n"


def _lean_three_arm_headline(scores: dict[str, object], arms: dict[str, object]) -> list[str]:
    repair_summary = arms.get("repair_selected") if isinstance(arms, dict) else None
    if not isinstance(repair_summary, dict):
        return []
    repair_score = float(repair_summary.get("mean_task_score", 0.0))
    fixed_score = repair_score - float(scores.get("repair_task_delta_vs_fixed", 0.0))
    random_score = repair_score - float(scores.get("repair_task_delta_vs_random", 0.0))
    trajectory_summary = arms.get("trajectory_selected") if isinstance(arms, dict) else {}
    full_count = int(trajectory_summary.get("count", 0)) if isinstance(trajectory_summary, dict) else 0
    repair_count = int(repair_summary.get("count", 0))
    repair_eligible_count = int(scores.get("repair_eligible_task_count", 0))
    return [
        "",
        "## Lean Three-Arm Headline",
        "",
        (
            "This is the public-facing comparison: fixed baseline, random perturbation, "
            "and selected latent repair. Trajectory/evolved/oracle rows below are diagnostics."
        ),
        "",
        f"Repair coverage: `{repair_count}/{full_count}` overall, `{repair_count}/{repair_eligible_count}` eligible.",
        "",
        "| Arm | Scope | Task Score | Delta vs Fixed | Delta vs Random | W/T/L vs Fixed | W/T/L vs Random |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
        (
            f"| fixed baseline | repair-covered tasks | {fixed_score:.6f} | "
            f"0.000000 | {fixed_score - random_score:.6f} | - | - |"
        ),
        (
            f"| random perturbation | repair-covered tasks | {random_score:.6f} | "
            f"{random_score - fixed_score:.6f} | 0.000000 | - | - |"
        ),
        (
            f"| selected latent repair | repair-covered tasks | {repair_score:.6f} | "
            f"{float(scores.get('repair_task_delta_vs_fixed', 0.0)):.6f} | "
            f"{float(scores.get('repair_task_delta_vs_random', 0.0)):.6f} | "
            f"{_format_wins(scores.get('repair_wins_vs_fixed', {}))} | "
            f"{_format_wins(scores.get('repair_wins_vs_random', {}))} |"
        ),
    ]


def _comparison_rows(
    arm_records: list[dict[str, object]],
    all_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for record in arm_records:
        grouped[(str(record["candidate_key"]), _task_id(record))][str(record["arm"])] = record
    oracle_by_key = {
        (str(record["candidate_key"]), _task_id(record)): record
        for record in _oracle_records(all_records)
    }

    rows = []
    for (candidate, task_id), records in sorted(grouped.items()):
        if not all(arm in records for arm in BASE_ARMS):
            continue
        fixed = records["fixed"]
        random = records["random"]
        trajectory = records["trajectory_selected"]
        evolved = records.get("evolved")
        repair = records.get("repair_selected")
        oracle = oracle_by_key.get((candidate, task_id), trajectory)
        repair_baseline = evolved or trajectory
        rows.append(
            {
                "candidate": candidate,
                "task_id": task_id,
                "fixed_schedule": _schedule_name(fixed),
                "random_schedule": _schedule_name(random),
                "trajectory_schedule": _schedule_name(trajectory),
                "trajectory_selection_reason": str(trajectory.get("arm_selection_reason", "")),
                "trajectory_selector_score": _selector_score(trajectory),
                "fixed_task_score": _task_score(fixed),
                "random_task_score": _task_score(random),
                "trajectory_task_score": _task_score(trajectory),
                "evolved_schedule": _schedule_name(evolved) if evolved is not None else "",
                "evolved_selection_reason": str(evolved.get("arm_selection_reason", ""))
                if evolved is not None
                else "",
                "evolved_selector_score": _selector_score(evolved) if evolved is not None else 0.0,
                "evolved_selector_edge": (_selector_score(evolved) - _selector_score(trajectory))
                if evolved is not None
                else 0.0,
                "evolved_task_score": _task_score(evolved) if evolved is not None else 0.0,
                "evolved_trajectory_score": _trajectory_score(evolved) if evolved is not None else 0.0,
                "fixed_trajectory_score": _trajectory_score(fixed),
                "random_trajectory_score": _trajectory_score(random),
                "trajectory_trajectory_score": _trajectory_score(trajectory),
                "trajectory_delta_vs_fixed": _task_score(trajectory) - _task_score(fixed),
                "trajectory_delta_vs_random": _task_score(trajectory) - _task_score(random),
                "evolved_delta_vs_fixed": (_task_score(evolved) - _task_score(fixed))
                if evolved is not None
                else 0.0,
                "evolved_delta_vs_random": (_task_score(evolved) - _task_score(random))
                if evolved is not None
                else 0.0,
                "evolved_delta_vs_trajectory": (_task_score(evolved) - _task_score(trajectory))
                if evolved is not None
                else 0.0,
                "repair_control": _control_name(repair) if repair is not None else "",
                "repair_selection_reason": str(repair.get("arm_selection_reason", ""))
                if repair is not None
                else "",
                "repair_source_control": _repair_metadata_value(repair, "source_control"),
                "repair_source_state": _repair_metadata_value(repair, "source_state"),
                "repair_source_history_step": _repair_metadata_value(repair, "source_history_step"),
                "repair_selector_score": _selector_score(repair) if repair is not None else 0.0,
                "repair_selector_edge": (
                    _selector_score(repair)
                    - _repair_baseline_selector_score(repair, repair_baseline)
                )
                if repair is not None
                else 0.0,
                "repair_task_score": _task_score(repair) if repair is not None else 0.0,
                "repair_delta_vs_evolved": (_task_score(repair) - _task_score(repair_baseline))
                if repair is not None
                else 0.0,
                "oracle_schedule": _control_name(oracle),
                "oracle_task_score": _task_score(oracle),
                "oracle_delta_vs_trajectory": _task_score(oracle) - _task_score(trajectory),
                "oracle_delta_vs_evolved": (_task_score(oracle) - _task_score(evolved))
                if evolved is not None
                else 0.0,
                "oracle_delta_vs_repair": (_task_score(oracle) - _task_score(repair))
                if repair is not None
                else 0.0,
            }
        )
    return rows


def _oracle_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record.get("candidate_key", "")), _task_id(record))].append(record)
    return [
        max(task_records, key=lambda record: (_task_score(record), _trajectory_score(record)))
        for _key, task_records in sorted(grouped.items())
        if task_records
    ]


def _oracle_generation_budget(records: list[dict[str, object]]) -> float:
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        grouped[(str(record.get("candidate_key", "")), _task_id(record))] += 1
    return _mean(grouped.values())


def _nested_arm_summary(
    grouped_records: dict[tuple[str, str], list[dict[str, object]]],
) -> dict[str, dict[str, dict[str, object]]]:
    summaries: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for (family, arm), records in sorted(grouped_records.items()):
        summaries[family][arm] = _arm_summary(records)
    return dict(summaries)


def _arm_summary(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "count": len(records),
        "mean_generation_budget_per_task": _mean(
            _generation_budget(record) for record in records
        ),
        "mean_task_score": _mean(_task_score(record) for record in records),
        "mean_trajectory_score": _mean(_trajectory_score(record) for record in records),
        "mean_combined_score": _mean(float(record["combined_selection_score"]) for record in records),
    }


def _with_arm_metadata(
    arm: str,
    record: dict[str, object],
    generation_budget_per_task: int,
    selection_reason: str,
    selector_score: float,
    selector_baseline_score: float | None = None,
) -> dict[str, object]:
    arm_record = dict(record)
    arm_record["arm"] = arm
    arm_record["arm_generation_budget_per_task"] = generation_budget_per_task
    arm_record["arm_selection_reason"] = selection_reason
    arm_record["arm_selector_score"] = selector_score
    if selector_baseline_score is not None:
        arm_record["arm_selector_baseline_score"] = selector_baseline_score
    return arm_record


def _selection_reason(
    arm: str,
    task_answer_type: str,
    exact_task_trajectory_policy: str,
    trajectory_selector: str,
    *,
    evolved_record: dict[str, object] | None = None,
    baseline_record: dict[str, object] | None = None,
    promotion_margin: float = 0.0,
    revision_promotion_margin: float = DEFAULT_REVISION_PROMOTION_MARGIN,
    evolved_selector: str = DEFAULT_EVOLVED_SELECTOR,
    repair_selector: str = DEFAULT_REPAIR_SELECTOR,
) -> str:
    if arm == "fixed":
        return "first_default_schedule"
    if arm == "random":
        return "stable_random_schedule"
    if task_answer_type != "rubric" and exact_task_trajectory_policy == "fixed" and arm != "repair_selected":
        return "fixed_exact_answer_guard"
    if task_answer_type != "rubric" and exact_task_trajectory_policy == "proposal_history" and arm in {
        "trajectory_selected",
        "evolved",
    }:
        source = _exact_trajectory_selection_source(evolved_record)
        if source == "history":
            return "exact_answer_proposal_history_match"
        if source == "final":
            return "exact_answer_proposal_final_match"
        return "exact_answer_proposal_history_no_match_kept_fixed"
    if arm == "trajectory_selected":
        return f"max_{trajectory_selector}_score_base_pool"
    if arm == "evolved":
        if (
            baseline_record is not None
            and evolved_record is not None
            and _record_identity(evolved_record) == _record_identity(baseline_record)
            and promotion_margin > 0
        ):
            if revision_promotion_margin > promotion_margin:
                return (
                    f"evolved_margin_guard_kept_base_pool_{promotion_margin:.3f}"
                    f"_revision_{revision_promotion_margin:.3f}"
                )
            return f"evolved_margin_guard_kept_base_pool_{promotion_margin:.3f}"
        if evolved_selector != "inherit":
            return f"max_{evolved_selector}_evolved_pool"
        return f"max_{trajectory_selector}_score_evolved_pool"
    if arm == "repair_selected":
        if task_answer_type != "rubric" and exact_task_trajectory_policy != "trajectory":
            if (
                baseline_record is not None
                and evolved_record is not None
                and _record_identity(evolved_record) == _record_identity(baseline_record)
            ):
                return "exact_answer_repair_kept_source"
            if _repair_name(evolved_record) == "self_check_answer_repair":
                return "exact_answer_self_repair_format_change"
            if _repair_name(evolved_record) == "arithmetic_contradiction_span_repair":
                return "exact_answer_arithmetic_span_revision"
            if _repair_name(evolved_record) == "arithmetic_feedback_repair":
                return "exact_answer_arithmetic_feedback"
            if _repair_name(evolved_record) == "arithmetic_evidence_repair":
                return "exact_answer_arithmetic_evidence"
            if _repair_name(evolved_record) in {"answer_span_repair", "answer_context_random_repair"}:
                return "exact_answer_verifier_span_revision"
            return "exact_answer_counterfactual_proposal_match"
        if (
            baseline_record is not None
            and evolved_record is not None
            and _record_identity(evolved_record) == _record_identity(baseline_record)
            and promotion_margin > 0
        ):
            return f"repair_margin_guard_kept_evolved_{promotion_margin:.3f}"
        return f"max_{repair_selector}_score_repair_pool"
    return f"max_{trajectory_selector}_score"


def _exact_trajectory_selection_source(record: dict[str, object] | None) -> str:
    if record is None:
        return ""
    selection = record.get("exact_trajectory_selection")
    if not isinstance(selection, dict):
        return ""
    return str(selection.get("source", ""))


def _selection_score(
    record: dict[str, object],
    task_prompt: str,
    task_answer_type: str,
    trajectory_selector: str,
) -> float:
    if trajectory_selector == "generic" or task_answer_type != "rubric":
        return _trajectory_score(record)
    if trajectory_selector == "planning_state_v2":
        return _planning_state_v2_selector_score(record, task_prompt)
    if trajectory_selector == "planning_state":
        return _planning_state_selector_score(record, task_prompt)
    return _planning_prompt_selector_score(record, task_prompt)


def _repair_selection_score(
    record: dict[str, object],
    *,
    baseline_record: dict[str, object],
    task_prompt: str,
    task_answer_type: str,
    trajectory_selector: str,
    repair_selector: str,
) -> float:
    if task_answer_type != "rubric":
        return _exact_answer_repair_selection_score(record, task_answer_type, task_prompt)
    if repair_selector in {"inherit", "transfer_promotion_value"}:
        return _selection_score(record, task_prompt, task_answer_type, trajectory_selector)
    if repair_selector == "planning_quality":
        return _planning_quality_score(record, task_prompt)
    if repair_selector == "planning_quality_guarded":
        return max(
            0.0,
            _planning_quality_score(record, task_prompt)
            - _repair_overpreservation_penalty(record),
        )
    if repair_selector == "planning_quality_risk_guarded":
        return max(
            0.0,
            _planning_quality_score(record, task_prompt)
            - _repair_overpreservation_penalty(record)
            - _planning_contradiction_penalty(record, task_prompt),
        )
    if repair_selector == "planning_quality_delta":
        return _source_relative_planning_quality_score(
            record,
            baseline_record=baseline_record,
            task_prompt=task_prompt,
            guarded=False,
            risk_guarded=False,
        )
    if repair_selector == "planning_quality_delta_guarded":
        return _source_relative_planning_quality_score(
            record,
            baseline_record=baseline_record,
            task_prompt=task_prompt,
            guarded=True,
            risk_guarded=False,
        )
    if repair_selector == "planning_quality_delta_risk_guarded":
        return _source_relative_planning_quality_score(
            record,
            baseline_record=baseline_record,
            task_prompt=task_prompt,
            guarded=True,
            risk_guarded=True,
        )
    if repair_selector == "planning_quality_prompt_coverage_guarded":
        return _planning_quality_prompt_coverage_guarded_score(record, task_prompt)
    if repair_selector == "planning_quality_seed_objective_guarded":
        return _planning_quality_seed_objective_guarded_score(record, task_prompt)
    if repair_selector in {"planning_quality_seed_realization_guarded", "candidate_aware_promotion_v1"}:
        return _planning_quality_seed_realization_guarded_score(record, task_prompt)
    raise ValueError(f"Unsupported repair selector: {repair_selector}")


def _exact_answer_repair_selection_score(
    record: dict[str, object],
    task_answer_type: str,
    task_prompt: str = "",
) -> float:
    proposal = _repair_metadata_value(record, "proposal")
    if proposal:
        if not _answer_text_matches_proposal(str(record.get("text", "")), proposal, task_answer_type):
            return 0.0
        return 1.0 + _exact_repair_selection_priority(record) + 0.01 * _trajectory_score(record)
    if _repair_name(record) in SELF_REPAIR_EVIDENCE_NAMES:
        self_answer = _repair_metadata_value(record, "self_repair_extracted_answer")
        source_answer = _repair_metadata_value(record, "source_extracted_answer")
        if not self_answer:
            return 0.0
        if source_answer and _normalize_exact_value(self_answer) == _normalize_exact_value(source_answer):
            return 0.0
        if not _repair_metadata_bool(
            record,
            "self_repair_arithmetic_consistent",
            default=_arithmetic_claims_consistent(str(record.get("text", ""))),
        ):
            return 0.0
        if task_answer_type == "integer" and _repair_self_repair_arithmetic_claim_count(record) <= 0.0:
            return 0.0
        if task_answer_type == "integer" and _repair_irrelevant_prompt_number_used(record, task_prompt):
            return 0.0
        if task_answer_type == "integer" and _repair_missing_required_operators(record, task_prompt):
            return 0.0
        if task_answer_type == "integer" and _repair_quantity_role_gaps(record, task_prompt):
            return 0.0
        if task_answer_type == "integer" and _repair_arithmetic_provenance_gaps(record, task_prompt):
            return 0.0
        if task_answer_type == "integer" and _repair_final_answer_role_gaps(record, task_prompt):
            return 0.0
        if task_answer_type == "integer" and _repair_final_answer_object_gaps(record, task_prompt):
            return 0.0
        if task_answer_type == "integer" and _repair_final_answer_target_gaps(record, task_prompt):
            return 0.0
        if task_answer_type == "short_text" and _repair_short_text_symbolic_gaps(record, task_prompt):
            return 0.0
        if task_answer_type == "short_text" and _repair_short_text_trace_gaps(record, task_prompt):
            return 0.0
        return 0.95 + _exact_repair_selection_priority(record) + 0.01 * _trajectory_score(record)
    return 0.0


def _exact_repair_selection_priority(record: dict[str, object]) -> float:
    return EXACT_REPAIR_SELECTION_PRIORITIES.get(_repair_name(record), 0.0)


def _answer_text_matches_proposal(text: str, proposal: str, task_answer_type: str) -> bool:
    if task_answer_type == "integer":
        numbers = re.findall(r"-?\d+", text.replace(",", ""))
        try:
            return bool(numbers) and int(numbers[-1]) == int(proposal)
        except ValueError:
            return False
    if task_answer_type == "multiple_choice":
        proposal_letter = proposal.strip().upper()
        letters = re.findall(r"(?:^|\b)(?:ANSWER\s*[:\-]?\s*)?\(?([A-D])\)?(?:\.|\b)", text.upper())
        return bool(letters) and letters[-1] == proposal_letter
    proposal_tokens = re.findall(r"[a-z0-9]+", proposal.lower())
    text_tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not proposal_tokens:
        return False
    width = len(proposal_tokens)
    return any(text_tokens[index : index + width] == proposal_tokens for index in range(len(text_tokens) - width + 1))


def _label_free_exact_answer_supported(task: GeneralReasoningTask) -> bool:
    if task.answer_type in {"integer", "multiple_choice"}:
        return True
    if task.answer_type == "short_text":
        return bool(_short_text_answer_schema(task.prompt))
    return False


def _supports_no_proposal_answer_span_revision(task: GeneralReasoningTask) -> bool:
    return (
        task.answer_type in NO_PROPOSAL_VERIFIER_SPAN_ANSWER_TYPES
        and _label_free_exact_answer_supported(task)
    )


def _label_free_exact_answer_from_text(task: GeneralReasoningTask, text: str) -> str | None:
    if task.answer_type == "integer":
        numbers = re.findall(r"(?<![\w.])-?\d+(?!\.\d)(?!\w)", text.replace(",", ""))
        return str(int(numbers[-1])) if numbers else None
    if task.answer_type == "multiple_choice":
        choice = extract_choice(text, task.choices or {})
        return choice.upper() if choice else None
    if task.answer_type == "short_text":
        return _label_free_short_text_answer_from_text(task.prompt, text)
    return None


def _label_free_short_text_answer_from_text(prompt: str, text: str) -> str | None:
    schema = _short_text_answer_schema(prompt)
    if not schema:
        return None
    context = _final_answer_context(text)
    tokens = _object_tokens(context)
    if not tokens:
        return None
    kind = schema.get("kind")
    if kind == "choice":
        choices = schema.get("choices", [])
        if not isinstance(choices, list):
            return None
        choice_set = set(choices)
        matches = [token for token in tokens if token in choice_set]
        return matches[-1] if matches else None
    if kind == "letters":
        count = int(schema.get("count", 0) or 0)
        allowed = set(schema.get("allowed", []))
        letters = [token.upper() for token in re.findall(r"\b[a-zA-Z]\b", context)]
        if count <= 0 or len(letters) < count:
            return None
        candidate = letters[-count:]
        if allowed and not set(candidate).issubset(allowed):
            return None
        if len(set(candidate)) != len(candidate):
            return None
        return " ".join(candidate)
    if kind == "list":
        items = schema.get("items", [])
        if not isinstance(items, list) or not items:
            return None
        item_set = set(items)
        width = len(items)
        candidates = []
        for index in range(len(tokens) - width + 1):
            window = tokens[index : index + width]
            if set(window) == item_set:
                candidates.append(window)
        return " ".join(candidates[-1]) if candidates else None
    return None


def _repair_short_text_symbolic_gaps(record: dict[str, object], task_prompt: str) -> set[str]:
    expected = symbolic_short_text_candidate_from_prompt(task_prompt)
    if expected is None:
        return set()
    answer = _repair_metadata_value(record, "self_repair_extracted_answer")
    if not answer:
        answer = _label_free_short_text_answer_from_text(task_prompt, str(record.get("text", "")))
    if not answer:
        return {"symbolic:missing_answer"}
    if _normalize_exact_value(answer) != _normalize_exact_value(expected):
        return {f"symbolic:expected_{_gap_safe_token(expected)}"}
    return set()


def _repair_short_text_trace_gaps(record: dict[str, object], task_prompt: str) -> set[str]:
    expected = symbolic_short_text_candidate_from_prompt(task_prompt)
    if expected is None:
        return set()
    trace_kind = _short_text_trace_kind(task_prompt)
    if not trace_kind:
        return set()
    trace_text = _pre_final_answer_text(str(record.get("text", "")))
    if not trace_text.strip():
        return {f"{trace_kind}:missing_trace"}
    normalized = _normalize_object_text(trace_text)
    if trace_kind == "order" and not _order_trace_present(normalized, str(expected)):
        return {"order:missing_before_trace"}
    if trace_kind == "list" and not _list_trace_present(normalized):
        return {"list:missing_swap_trace"}
    if trace_kind == "letter_transform" and not _letter_transform_trace_present(normalized, task_prompt):
        return {"letter_transform:missing_operation_trace"}
    if trace_kind == "toggle" and not _toggle_trace_present(normalized):
        return {"toggle:missing_toggle_trace"}
    if trace_kind == "syllogism" and not _syllogism_trace_present(normalized):
        return {"syllogism:missing_relation_trace"}
    return set()


def _short_text_trace_kind(prompt: str) -> str:
    normalized = _normalize_object_text(prompt)
    if "full order" in normalized and "before" in normalized:
        return "order"
    if "final list" in normalized and "swap" in normalized:
        return "list"
    if (
        "starts with the code" in normalized
        and "letters separated by spaces" in normalized
        and ("rotate" in normalized or "swap" in normalized)
    ):
        return "letter_transform"
    if "toggled" in normalized and re.search(r"\banswer\s+(?:only\s+)?on\s+or\s+off\b", normalized):
        return "toggle"
    if re.search(r"\bcan\s+(?:a|an)\s+\w+\s+be\s+(?:a|an)\s+\w+\b", normalized) and (
        "all " in normalized or "no " in normalized
    ):
        return "syllogism"
    return ""


def _pre_final_answer_text(text: str) -> str:
    flattened = " ".join(text.replace("\n", ". ").split())
    answer_cues = list(
        re.finditer(r"\b(?:answer|therefore|so|final(?:\s+answer)?)\b\s*[:\-]?", flattened, re.IGNORECASE)
    )
    if not answer_cues:
        return flattened
    return flattened[: answer_cues[-1].start()]


def _order_trace_present(normalized_trace: str, expected: str) -> bool:
    letters = [token.lower() for token in re.findall(r"\b[A-Za-z]\b", expected)]
    if len(letters) < 2:
        return False
    for left, right in zip(letters, letters[1:], strict=False):
        if not re.search(rf"\b{re.escape(left)}\b(?:\s+\w+){{0,2}}\s+before\s+\b{re.escape(right)}\b", normalized_trace):
            return False
    return True


def _list_trace_present(normalized_trace: str) -> bool:
    return "swap" in normalized_trace or "after first" in normalized_trace or "after second" in normalized_trace


def _letter_transform_trace_present(normalized_trace: str, prompt: str) -> bool:
    normalized_prompt = _normalize_object_text(prompt)
    if "rotate" in normalized_prompt and "rotate" not in normalized_trace:
        return False
    return "swap" not in normalized_prompt or "swap" in normalized_trace


def _toggle_trace_present(normalized_trace: str) -> bool:
    return "toggle" in normalized_trace or "odd" in normalized_trace or "even" in normalized_trace


def _syllogism_trace_present(normalized_trace: str) -> bool:
    return (
        ("all" in normalized_trace or "must be" in normalized_trace or "is a" in normalized_trace)
        and ("no" in normalized_trace or "cannot" in normalized_trace or "not" in normalized_trace)
    )


def _gap_safe_token(value: str) -> str:
    token = "_".join(re.findall(r"[a-z0-9]+", value.lower()))
    return token or "answer"


def _short_text_answer_schema(prompt: str) -> dict[str, object]:
    normalized = _normalize_object_text(prompt)
    if re.search(r"\banswer\s+(?:only\s+)?on\s+or\s+off\b", normalized):
        return {"kind": "choice", "choices": ["on", "off"]}
    if re.search(r"\banswer\s+(?:only\s+)?yes\s+or\s+no\b", normalized):
        return {"kind": "choice", "choices": ["yes", "no"]}
    if "letters separated by spaces" in normalized:
        count = _prompt_requested_letter_count(normalized)
        allowed = _prompt_single_letter_symbols(prompt)
        if count > 0 and len(allowed) >= count:
            return {"kind": "letters", "count": count, "allowed": sorted(allowed)}
    if "final list" in normalized:
        items = _prompt_initial_list_items(prompt.lower())
        if items:
            return {"kind": "list", "items": items}
    return {}


def _prompt_requested_letter_count(normalized_prompt: str) -> int:
    match = re.search(r"\b(?:the\s+)?([a-z0-9]+)\s+letters?\s+separated\s+by\s+spaces\b", normalized_prompt)
    if not match:
        return 0
    value = match.group(1)
    if value.isdigit():
        return int(value)
    return {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }.get(value, 0)


def _prompt_single_letter_symbols(prompt: str) -> set[str]:
    symbols = {item.upper() for item in re.findall(r"\b([A-Z])\b", prompt)}
    if symbols:
        return symbols
    pairs = re.findall(r"\b([a-z])\s+is\s+before\s+([a-z])\b", prompt, flags=re.IGNORECASE)
    return {item.upper() for pair in pairs for item in pair}


def _prompt_initial_list_items(normalized_prompt: str) -> list[str]:
    match = re.search(r"\bstart\s+with\s+the\s+list\s+(.+?)\.", normalized_prompt)
    if not match:
        return []
    return [
        item.strip()
        for item in re.split(r",|\band\b", match.group(1))
        if item.strip()
    ]


def _normalize_exact_value(value: object | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _arithmetic_claims_consistent(text: str) -> bool:
    return not _arithmetic_claim_inconsistencies(text)


def _arithmetic_claim_count(text: str) -> int:
    return len(_arithmetic_claims(text))


def _repair_missing_required_operators(record: dict[str, object], task_prompt: str) -> set[str]:
    required = _prompt_required_arithmetic_operators(task_prompt)
    if not required:
        return set()
    present = _arithmetic_claim_operators(str(record.get("text", "")))
    return required - present


def _repair_quantity_role_gaps(record: dict[str, object], task_prompt: str) -> set[str]:
    requirements = _prompt_quantity_role_requirements(task_prompt)
    if not any(requirements.values()):
        return set()
    claims = [expression for expression, _claimed in _arithmetic_claims(str(record.get("text", "")))]
    multiplication_pairs: set[tuple[str, str]] = set()
    division_right_values: set[str] = set()
    subtraction_right_values: set[str] = set()
    for expression in claims:
        multiplication_pairs.update(_arithmetic_expression_operator_number_pairs(expression, "*"))
        division_right_values.update(_arithmetic_expression_operator_right_values(expression, "/"))
        subtraction_right_values.update(_arithmetic_expression_operator_right_values(expression, "-"))

    gaps: set[str] = set()
    for left, right in requirements.get("multiply_pairs", set()):
        if tuple(sorted((left, right), key=float)) not in multiplication_pairs:
            gaps.add(f"mul:{left}*{right}")
    for divisor in requirements.get("division_right_values", set()):
        if divisor not in division_right_values:
            gaps.add(f"div:{divisor}")
    for removed in requirements.get("subtraction_right_values", set()):
        if removed not in subtraction_right_values:
            gaps.add(f"sub:{removed}")
    return gaps


def _repair_arithmetic_provenance_gaps(record: dict[str, object], task_prompt: str) -> set[str]:
    grounded_numbers = _prompt_grounded_arithmetic_numbers(task_prompt)
    if not grounded_numbers:
        return set()
    gaps: set[str] = set()
    for expression, claimed_text in _arithmetic_claims(str(record.get("text", ""))):
        expression_numbers = _arithmetic_expression_numbers(expression)
        missing = expression_numbers - grounded_numbers
        if missing:
            gaps.add(f"{expression.strip()}:{','.join(sorted(missing, key=float))}")
            continue
        computed = _safe_arithmetic_value(expression)
        if computed is None:
            continue
        try:
            claimed = float(claimed_text)
        except ValueError:
            continue
        if abs(computed - claimed) <= 1e-9:
            grounded_numbers.add(_format_arithmetic_value(claimed))
    return gaps


def _repair_final_answer_role_gaps(record: dict[str, object], task_prompt: str) -> set[str]:
    role = _prompt_final_answer_role(task_prompt)
    if not role:
        return set()
    answer_text = _repair_metadata_value(record, "self_repair_extracted_answer")
    if not answer_text:
        answer_text = _last_integer_text(str(record.get("text", "")))
    if not answer_text:
        return {f"{role}:missing_answer"}
    try:
        final_answer = int(str(answer_text))
    except ValueError:
        return {f"{role}:non_integer_answer"}

    text = str(record.get("text", ""))
    role_values = _arithmetic_claim_values_for_final_roles(text)
    if role == "sum" and final_answer not in role_values["sum"]:
        return {"sum:final_not_sum"}
    if role == "division" and final_answer not in role_values["division"]:
        return {"division:final_not_division"}
    if role == "floor_division" and final_answer not in role_values["floor_division"]:
        return {"floor_division:final_not_floor_division"}
    if role == "remainder" and final_answer not in (_remainder_answer_values(text) | role_values["remainder"]):
        return {"remainder:final_not_remainder"}
    return set()


def _repair_final_answer_object_gaps(record: dict[str, object], task_prompt: str) -> set[str]:
    excluded_terms = _prompt_excluded_final_answer_terms(task_prompt)
    if not excluded_terms:
        return set()
    context = _final_answer_context(str(record.get("text", "")))
    if not context:
        return set()
    gaps: set[str] = set()
    for term in excluded_terms:
        if _final_answer_term_present(context, term):
            gaps.add(f"excluded:{term.replace(' ', '_')}")
    return gaps


def _repair_final_answer_target_gaps(record: dict[str, object], task_prompt: str) -> set[str]:
    spec = _prompt_final_answer_target_spec(task_prompt)
    target_heads = spec["heads"]
    if not target_heads:
        return set()
    context = _final_answer_context(str(record.get("text", "")))
    answer_text = _repair_metadata_value(record, "self_repair_extracted_answer") or _last_integer_text(context)
    unit_tokens = _final_answer_unit_tokens(context, str(answer_text))
    if not unit_tokens:
        return set()

    target_modifiers = spec["modifiers"]
    excluded_heads = _term_heads(_prompt_excluded_final_answer_terms(task_prompt))
    target_modifier_terms = set().union(
        *(_final_answer_term_variants(modifier) for modifier in target_modifiers)
    ) if target_modifiers else set()
    known_non_target_heads = _prompt_known_object_heads(task_prompt) - target_heads - excluded_heads - target_modifier_terms
    gaps: set[str] = set()
    for token in unit_tokens:
        token_variants = _final_answer_term_variants(token)
        if token_variants & known_non_target_heads:
            gaps.add(f"wrong_target:{token}")

    for index, token in enumerate(unit_tokens):
        if not (_final_answer_term_variants(token) & target_heads):
            continue
        modifier = unit_tokens[index - 1] if index > 0 else ""
        if (
            modifier
            and modifier not in target_modifiers
            and modifier not in _FINAL_ANSWER_NEUTRAL_MODIFIERS
            and not modifier.isdigit()
        ):
            gaps.add(f"conflicting_modifier:{modifier}_{token}")
    return gaps


_FINAL_ANSWER_NEUTRAL_MODIFIERS = {
    "answer",
    "final",
    "full",
    "many",
    "number",
    "of",
    "remaining",
    "the",
    "total",
}


def _prompt_final_answer_target_spec(prompt: str) -> dict[str, set[str]]:
    phrase = _prompt_final_answer_target_phrase(prompt)
    if not phrase:
        return {"terms": set(), "heads": set(), "modifiers": set()}
    terms = _final_answer_term_variants(phrase)
    tokens = _object_tokens(phrase)
    if not tokens:
        return {"terms": set(), "heads": set(), "modifiers": set()}
    head = tokens[-1]
    heads = _final_answer_term_variants(head)
    modifiers = set(tokens[:-1])
    return {
        "terms": terms | heads,
        "heads": heads,
        "modifiers": modifiers,
    }


def _prompt_final_answer_target_phrase(prompt: str) -> str:
    normalized = _normalize_object_text(
        _normalize_prompt_number_words(_normalize_arithmetic_expression(prompt).lower())
    )
    patterns = (
        r"\bhow many\s+(.+?)\s+(?:can|does|do|did|is|are|was|were|came|pass|remain|get)\b",
        r"\bwhat is the maximum number of\s+([a-z]+(?:\s+[a-z]+){0,3})\b",
        r"\bwhat is the full capacity\b.*?\bin\s+([a-z]+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            phrase = _clean_final_answer_target_phrase(match.group(1))
            if phrase:
                return phrase
    return ""


def _clean_final_answer_target_phrase(phrase: str) -> str:
    tokens = [
        token
        for token in _object_tokens(phrase)
        if token not in {"each", "many", "number", "of", "the", "what"}
    ]
    return " ".join(tokens[-4:])


def _prompt_known_object_heads(prompt: str) -> set[str]:
    normalized = _normalize_object_text(
        _normalize_prompt_number_words(_normalize_arithmetic_expression(prompt).lower())
    )
    heads: set[str] = set()
    object_phrases: list[str] = []
    object_phrases.extend(
        match.group(1)
        for match in re.finditer(r"\b\d+\s+([a-z]+(?:\s+[a-z]+){0,3})\b", normalized)
    )
    object_phrases.extend(
        match.group(1)
        for match in re.finditer(r"\b(?:by|among|per|each)\s+([a-z]+)\b", normalized)
    )
    target_phrase = _prompt_final_answer_target_phrase(prompt)
    if target_phrase:
        object_phrases.append(target_phrase)
    for phrase in object_phrases:
        tokens = _object_tokens(phrase)
        if not tokens:
            continue
        heads.update(_final_answer_term_variants(tokens[-1]))
    return heads


def _term_heads(terms: set[str]) -> set[str]:
    heads: set[str] = set()
    for term in terms:
        tokens = _object_tokens(term)
        if tokens:
            heads.update(_final_answer_term_variants(tokens[-1]))
    return heads


def _final_answer_unit_tokens(context: str, answer_text: str) -> list[str]:
    if not context or not answer_text:
        return []
    answer = re.escape(str(answer_text).strip())
    matches = list(re.finditer(rf"(?<![\w.]){answer}(?!\.\d)(?!\w)", context))
    if not matches:
        return []
    after = context[matches[-1].end() :]
    after = re.split(r"[,;.!?]|\bnot\b", after, maxsplit=1)[0]
    tokens = _object_tokens(after)
    while tokens and tokens[0] in {"answer", "is", "equals", "therefore", "so", "the", "final"}:
        tokens.pop(0)
    return tokens[:6]


def _prompt_excluded_final_answer_terms(prompt: str) -> set[str]:
    normalized = _normalize_object_text(
        _normalize_prompt_number_words(_normalize_arithmetic_expression(prompt).lower())
    )
    terms: set[str] = set()
    for sentence in re.split(r"[.!?]\s*", normalized):
        sentence = sentence.strip()
        if not sentence:
            continue
        for match in re.finditer(r"\b(?:the\s+)?([a-z]+)\s+are\s+not\s+being\s+[a-z]+\b", sentence):
            terms.update(_final_answer_term_variants(match.group(1)))
        for match in re.finditer(
            r"\b(?:that\s+)?([a-z]+)\s+is\s+not\s+(?:[a-z]+\s+){0,2}"
            r"(?:revenue|counted|included|used|packed)\b",
            sentence,
        ):
            terms.update(_final_answer_term_variants(match.group(1)))
        for match in re.finditer(r"\b(?:a|an|the)\s+([a-z]+)\s+also\s+donated\b", sentence):
            terms.update(_final_answer_term_variants(match.group(1)))
        if "but only count" in sentence:
            for match in re.finditer(r"\b(?:the|a|an)\s+([a-z]+)\s+has\s+\d+\b", sentence):
                terms.update(_final_answer_term_variants(match.group(1)))
        if "question asks about all" in sentence or "asks about all" in sentence:
            terms.update(_prompt_modifier_terms_before_numbered_object(sentence))
    return {term for term in terms if term}


def _prompt_modifier_terms_before_numbered_object(sentence: str) -> set[str]:
    terms: set[str] = set()
    for match in re.finditer(r"\bmentions\s+\d+\s+([a-z]+(?:\s+[a-z]+){0,4}?)(?=\s+but\b|$)", sentence):
        words = match.group(1).split()
        if len(words) <= 1:
            continue
        modifier = " ".join(words[:-1])
        terms.update(_final_answer_term_variants(modifier))
    return terms


def _final_answer_term_variants(term: str) -> set[str]:
    normalized = _normalize_object_text(term)
    if not normalized or normalized in {"that", "the", "a", "an"}:
        return set()
    variants = {normalized}
    words = normalized.split()
    if len(words) == 1:
        word = words[0]
        if word.endswith("ies") and len(word) > 3:
            variants.add(word[:-1] if word[:-1].endswith("ie") else f"{word[:-3]}y")
        elif word.endswith("s") and len(word) > 3:
            variants.add(word[:-1])
        else:
            variants.add(f"{word}s")
    else:
        last = words[-1]
        if last.endswith("ies") and len(last) > 3:
            singular = last[:-1] if last[:-1].endswith("ie") else f"{last[:-3]}y"
            variants.add(" ".join([*words[:-1], singular]))
        elif last.endswith("s") and len(last) > 3:
            variants.add(" ".join([*words[:-1], last[:-1]]))
        else:
            variants.add(" ".join([*words[:-1], f"{last}s"]))
    return variants


def _final_answer_context(text: str) -> str:
    flattened = " ".join(text.replace("\n", ". ").split())
    if not flattened:
        return ""
    answer_cues = list(
        re.finditer(r"\b(?:answer|therefore|so|final(?:\s+answer)?)\b\s*[:\-]?", flattened, re.IGNORECASE)
    )
    if answer_cues:
        return flattened[answer_cues[-1].start() :]
    final_integer = _last_integer_text(flattened)
    if final_integer:
        sentences = re.split(r"(?<=[.!?])\s+", flattened)
        integer_pattern = rf"(?<![\w.]){re.escape(final_integer)}(?!\.\d)(?!\w)"
        for sentence in reversed(sentences):
            if re.search(integer_pattern, sentence):
                return sentence
    return flattened[-240:]


def _final_answer_term_present(context: str, term: str) -> bool:
    context_tokens = _object_tokens(context)
    term_tokens = _object_tokens(term)
    if not context_tokens or not term_tokens:
        return False
    width = len(term_tokens)
    for index in range(len(context_tokens) - width + 1):
        if context_tokens[index : index + width] != term_tokens:
            continue
        before = context_tokens[max(0, index - 3) : index]
        after = context_tokens[index + width : index + width + 3]
        if set(before) & {"not", "no", "without", "ignore", "ignored", "excluding"}:
            continue
        if set(after) & {"ignored", "excluded", "irrelevant"}:
            continue
        return True
    return False


def _object_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_object_text(text))


def _normalize_object_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("-", " ").replace(",", " ").lower()).strip()


def _prompt_final_answer_role(prompt: str) -> str:
    normalized = _normalize_prompt_number_words(_normalize_arithmetic_expression(prompt).lower().replace(",", ""))
    if _contains_any(normalized, ("left over", "leftover", "remainder", "remain after making equal")):
        return "remainder"
    if re.search(r"\bfull\s+\w+\s+bags?\b", normalized) or "maximum number" in normalized:
        return "floor_division"
    if _contains_any(
        normalized,
        (
            "shared equally",
            "split evenly",
            "split equally",
            "per bag",
            "each student get",
            "how many minutes is each",
            "how many does each",
            "how many cookies does each",
        ),
    ):
        return "division"
    if _contains_any(
        normalized,
        (
            "came from ticket sales",
            "ticket sales",
            "across those",
            "across the",
            "total pages",
            "how many pages did",
            "total revenue",
        ),
    ):
        return "sum"
    return ""


def _arithmetic_claim_values_for_final_roles(text: str) -> dict[str, set[int]]:
    values = {
        "sum": set(),
        "division": set(),
        "floor_division": set(),
        "remainder": set(),
    }
    for expression, claimed_text in _arithmetic_claims(text):
        computed = _safe_arithmetic_value(expression)
        if computed is None:
            continue
        try:
            claimed = float(claimed_text)
        except ValueError:
            continue
        if abs(computed - claimed) > 1e-9:
            continue
        claimed_int = _integer_if_close(claimed)
        operators = _arithmetic_expression_operators(expression)
        if "+" in operators and claimed_int is not None:
            values["sum"].add(claimed_int)
        if "/" in operators:
            if claimed_int is not None:
                values["division"].add(claimed_int)
            values["floor_division"].add(int(computed // 1))
        if "%" in operators and claimed_int is not None:
            values["remainder"].add(claimed_int)
    return values


def _remainder_answer_values(text: str) -> set[int]:
    normalized = _normalize_arithmetic_expression(text).lower().replace(",", "")
    values: set[int] = set()
    for match in re.finditer(r"\bremainder\s*(?:is|=|:)?\s*(-?\d+)\b", normalized):
        values.add(int(match.group(1)))
    for match in re.finditer(r"\b(-?\d+)\s+(?:left\s+over|leftover)\b", normalized):
        values.add(int(match.group(1)))
    return values


def _last_integer_text(text: str) -> str:
    numbers = re.findall(r"(?<![\w.])-?\d+(?!\.\d)(?!\w)", text.replace(",", ""))
    return numbers[-1] if numbers else ""


def _integer_if_close(value: float) -> int | None:
    rounded = round(value)
    return rounded if abs(value - rounded) <= 1e-9 else None


def _prompt_grounded_arithmetic_numbers(prompt: str) -> set[str]:
    normalized = _normalize_prompt_number_words(_normalize_arithmetic_expression(prompt).lower())
    numbers = _arithmetic_expression_numbers(normalized)
    if _contains_any(normalized, ("twice", "double", "half", "halve", "halved")):
        numbers.add("2")
    if _contains_any(normalized, ("triple", "tripled")):
        numbers.add("3")
    return numbers


def _prompt_quantity_role_requirements(prompt: str) -> dict[str, set[Any]]:
    normalized = _normalize_prompt_number_words(_normalize_arithmetic_expression(prompt).lower().replace(",", ""))
    requirements: dict[str, set[Any]] = {
        "multiply_pairs": set(),
        "division_right_values": set(),
        "subtraction_right_values": set(),
    }
    if not normalized.strip():
        return requirements

    for count, price in re.findall(
        r"\b(\d+)\s+\w+\s+tickets?\s+for\s+(\d+)\s+dollars?\s+each\b",
        normalized,
    ):
        requirements["multiply_pairs"].add(_ordered_number_pair(count, price))
    for count, unit in re.findall(
        r"\b(\d+)\s+\w+\s+with\s+(\d+)\s+\w+(?:\s+on\s+each\s+\w+|\s+each)\b",
        normalized,
    ):
        requirements["multiply_pairs"].add(_ordered_number_pair(count, unit))
    for count, unit in re.findall(
        r"\b(\d+)\s+(?:talks?|breaks?)\s+of\s+(\d+)\s+\w+\s+each\b",
        normalized,
    ):
        requirements["multiply_pairs"].add(_ordered_number_pair(count, unit))
    monday_pages = re.search(r"\b(\d+)\s+pages?\s+on\s+monday\b", normalized)
    if monday_pages and "twice as many" in normalized and "monday" in normalized:
        requirements["multiply_pairs"].add(_ordered_number_pair("2", monday_pages.group(1)))
    pack_match = re.search(r"\bpacks?\s+of\s+(\d+)\b.*?\bbuys?\s+(\d+)\s+packs?\b", normalized)
    if pack_match:
        requirements["multiply_pairs"].add(_ordered_number_pair(pack_match.group(1), pack_match.group(2)))

    for divisor in re.findall(
        r"\b(\d+)\s+\w+\s+per\s+bag\b|shared\s+equally\s+by\s+(\d+)\b|split\s+evenly\s+into\s+(\d+)\b|"
        r"split\s+equally\s+into\s+(\d+)\b|among\s+(\d+)\s+\w+",
        normalized,
    ):
        value = next((item for item in divisor if item), "")
        if value:
            requirements["division_right_values"].add(_normalize_number_text(value))

    for value in re.findall(
        r"sets?\s+aside\s+(\d+)\b|(\d+)\s+\w+\s+are\s+eaten\b|(\d+)\s+fewer\b|gives?\s+(\d+)\s+\w+\s+away\b",
        normalized,
    ):
        number = next((item for item in value if item), "")
        if number:
            requirements["subtraction_right_values"].add(_normalize_number_text(number))
    return requirements


def _prompt_required_arithmetic_operators(prompt: str) -> set[str]:
    normalized = _normalize_arithmetic_expression(prompt).lower().replace(",", "")
    operators: set[str] = set()
    if not normalized.strip():
        return operators

    if _contains_any(
        normalized,
        (
            "sets aside",
            "set aside",
            "remaining",
            "left",
            "fewer",
            "less than",
            "minus",
            "after",
            "eaten",
            "used",
            "spent",
            "removed",
            "gives away",
            "gave away",
        ),
    ):
        operators.add("-")

    if _contains_any(
        normalized,
        (
            "shared equally",
            "split evenly",
            "split equally",
            "divided equally",
            "divided evenly",
            "divided by",
        ),
    ) or re.search(r"\bbags?\s+with\s+\d+\b|\bper\s+bag\b|\bfull\s+\w+\s+bags?\b", normalized):
        operators.add("/")

    if _contains_any(
        normalized,
        (
            "twice as many",
            "times as many",
            "multiplied by",
            "dollars each",
            "on each tray",
            "on each plate",
            "on each shelf",
            "on each row",
        ),
    ) or re.search(r"\b\d+\s+\w+\s+with\s+\d+\s+\w+\s+on\s+each\b", normalized):
        operators.add("*")

    ticket_price_groups = re.findall(
        r"\b\d+\s+\w+\s+tickets?\s+for\s+\d+\s+dollars?\s+each\b",
        normalized,
    )
    if len(ticket_price_groups) >= 2:
        operators.update({"+", "*"})
    elif ticket_price_groups:
        operators.add("*")

    if _contains_any(normalized, ("altogether", "in all", "combined", "across those", "across the")):
        operators.add("+")
    if "came from ticket sales" in normalized or "ticket sales" in normalized and len(ticket_price_groups) >= 2:
        operators.add("+")
    return operators


def _arithmetic_claim_operators(text: str) -> set[str]:
    operators: set[str] = set()
    for expression, _claimed in _arithmetic_claims(text):
        operators.update(_arithmetic_expression_operators(expression))
    return operators


def _arithmetic_expression_operators(text: str) -> set[str]:
    normalized = _normalize_arithmetic_expression(text).lower()
    operators = {operator for operator in "+-*/%" if operator in normalized}
    word_operators = {
        "plus": "+",
        "minus": "-",
        "times": "*",
        "multiplied by": "*",
        "divided by": "/",
    }
    for words, operator in word_operators.items():
        if words in normalized:
            operators.add(operator)
    return operators


def _arithmetic_expression_operator_number_pairs(expression: str, operator: str) -> set[tuple[str, str]]:
    node = _safe_arithmetic_expression_node(expression)
    if node is None:
        return set()
    pairs: set[tuple[str, str]] = set()
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.BinOp) or not _ast_operator_matches(candidate.op, operator):
            continue
        left_values = _arithmetic_ast_number_values(candidate.left)
        right_values = _arithmetic_ast_number_values(candidate.right)
        for left in left_values:
            for right in right_values:
                pairs.add(_ordered_number_pair(left, right))
    return pairs


def _arithmetic_expression_operator_right_values(expression: str, operator: str) -> set[str]:
    node = _safe_arithmetic_expression_node(expression)
    if node is None:
        return set()
    values: set[str] = set()
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.BinOp) and _ast_operator_matches(candidate.op, operator):
            values.update(_arithmetic_ast_number_values(candidate.right))
    return values


def _safe_arithmetic_expression_node(expression: str) -> ast.AST | None:
    try:
        return ast.parse(_normalize_arithmetic_expression(expression), mode="eval").body
    except SyntaxError:
        return None


def _ast_operator_matches(operator_node: ast.operator, operator: str) -> bool:
    return (
        (operator == "+" and isinstance(operator_node, ast.Add))
        or (operator == "-" and isinstance(operator_node, ast.Sub))
        or (operator == "*" and isinstance(operator_node, ast.Mult))
        or (operator == "/" and isinstance(operator_node, ast.Div))
        or (operator == "%" and isinstance(operator_node, ast.Mod))
    )


def _arithmetic_ast_number_values(node: ast.AST) -> set[str]:
    values: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, int | float):
            number = float(child.value)
            values.add(str(int(number)) if number.is_integer() else f"{number:g}")
        elif isinstance(child, ast.UnaryOp) and isinstance(child.op, ast.USub):
            operand_values = _arithmetic_ast_number_values(child.operand)
            for value in operand_values:
                if value.startswith("-"):
                    values.add(value[1:])
                else:
                    values.add(f"-{value}")
    return values


def _ordered_number_pair(left: str, right: str) -> tuple[str, str]:
    first = _normalize_number_text(left)
    second = _normalize_number_text(right)
    return tuple(sorted((first, second), key=float))


def _normalize_number_text(value: str) -> str:
    number = float(str(value).strip())
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _normalize_prompt_number_words(text: str) -> str:
    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    pattern = r"\b(" + "|".join(re.escape(word) for word in number_words) + r")\b"
    return re.sub(pattern, lambda match: number_words[match.group(1)], text)


def _repair_irrelevant_prompt_number_used(record: dict[str, object], task_prompt: str) -> bool:
    irrelevant_numbers = _prompt_irrelevant_numbers(task_prompt)
    if not irrelevant_numbers:
        return False
    for expression, _claimed in _arithmetic_claims(str(record.get("text", ""))):
        if _arithmetic_expression_numbers(expression) & irrelevant_numbers:
            return True
    return False


def _prompt_irrelevant_numbers(prompt: str) -> set[str]:
    numbers: set[str] = set()
    for sentence in re.split(r"[.!?]\s*", prompt.lower().replace(",", "")):
        if not sentence.strip():
            continue
        sentence_numbers = _arithmetic_expression_numbers(sentence)
        if not sentence_numbers:
            continue
        if "ignore" in sentence or "irrelevant" in sentence:
            numbers.update(sentence_numbers)
            continue
        if re.search(r"\bnot\s+(?:being|be|used|counted|included|ticket|revenue|packed)\b", sentence):
            numbers.update(sentence_numbers)
            continue
        if " but " not in sentence:
            continue
        before, after = sentence.split(" but ", 1)
        before_numbers = _arithmetic_expression_numbers(before)
        if not before_numbers:
            continue
        after_marks_irrelevant = (
            " not " in f" {after} "
            or "only count" in after
            or "question asks" in after
            or "asks about" in after
            or "do not" in after
        )
        if after_marks_irrelevant:
            numbers.update(before_numbers)
    return numbers


def _arithmetic_expression_numbers(text: str) -> set[str]:
    values = set()
    normalized = _normalize_arithmetic_expression(text).replace(",", "")
    for value in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\.\d)(?!\w)", normalized):
        number = float(value)
        values.add(str(int(number)) if number.is_integer() else f"{number:g}")
    return values


def _arithmetic_claim_inconsistencies(text: str) -> list[dict[str, object]]:
    inconsistencies = []
    for expression, expected_text in _arithmetic_claims(text):
        value = _safe_arithmetic_value(expression)
        if value is None:
            continue
        expected = float(expected_text)
        if abs(value - expected) > 1e-9:
            inconsistencies.append(
                {
                    "expression": expression.strip(),
                    "claimed": expected,
                    "computed": value,
                }
            )
    return inconsistencies


def _arithmetic_claims(text: str) -> list[tuple[str, str]]:
    normalized = _normalize_arithmetic_expression(text)
    symbolic_claims = re.findall(
        r"([0-9][0-9\s()+\-*/%.]*[+\-*/%][0-9\s()+\-*/%.]*)\s*=\s*(-?\d+(?:\.\d+)?)",
        normalized,
    )
    return symbolic_claims + _word_arithmetic_claims(normalized)


def _normalize_arithmetic_expression(text: str) -> str:
    return (
        text.replace("\u00d7", "*")
        .replace("\u2715", "*")
        .replace("\u00f7", "/")
        .replace("\u2212", "-")
    )


def _word_arithmetic_claims(text: str) -> list[tuple[str, str]]:
    claims: list[tuple[str, str]] = []
    normalized = text.replace(",", "").lower()
    number = r"(-?\d+(?:\.\d+)?)"
    compound_spans: list[tuple[int, int]] = []
    compound_times_plus = (
        rf"\b{number}\s+times\s+{number}\s+plus\s+{number}\s+times\s+{number}"
        rf"\s+(?:is|equals)\s+{number}\b"
    )
    for match in re.finditer(compound_times_plus, normalized):
        left_a, left_b, right_a, right_b, claimed = match.groups()
        claims.append((f"{left_a} * {left_b} + {right_a} * {right_b}", claimed))
        compound_spans.append(match.span())

    binary_ops = {
        "plus": "+",
        "minus": "-",
        "times": "*",
        "multiplied by": "*",
        "divided by": "/",
    }
    for words, operator in binary_ops.items():
        pattern = rf"\b{number}\s+{words}\s+{number}\s+(?:is|equals)\s+{number}\b"
        for match in re.finditer(pattern, normalized):
            if any(_spans_overlap(match.span(), span) for span in compound_spans):
                continue
            left, right, claimed = match.groups()
            claims.append((f"{left} {operator} {right}", claimed))
    return claims


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _format_arithmetic_value(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _safe_arithmetic_value(expression: str) -> float | None:
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_arithmetic_node(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None


def _eval_arithmetic_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_arithmetic_node(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_arithmetic_node(node.left)
        right = _eval_arithmetic_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
    raise ValueError(f"Unsupported arithmetic expression: {ast.dump(node)}")


def _source_relative_planning_quality_score(
    record: dict[str, object],
    *,
    baseline_record: dict[str, object],
    task_prompt: str,
    guarded: bool,
    risk_guarded: bool,
) -> float:
    score = _planning_quality_score(record, task_prompt) - _planning_quality_score(
        baseline_record,
        task_prompt,
    )
    if guarded:
        score -= _repair_overpreservation_penalty(record)
    if risk_guarded:
        score -= _planning_contradiction_penalty(record, task_prompt)
        score += _planning_contradiction_penalty(baseline_record, task_prompt)
    return score


def _planning_quality_prompt_coverage_guarded_score(
    record: dict[str, object],
    task_prompt: str,
) -> float:
    planning_quality = _planning_quality_score(record, task_prompt)
    prompt_bonus = 0.0
    if planning_quality >= 0.30:
        prompt_bonus = 0.10 * _prompt_keyword_coverage(
            task_prompt,
            _normalize(str(record.get("text", ""))),
        )
    score = (
        planning_quality
        + prompt_bonus
        - _repair_overpreservation_penalty(record)
        - _planning_contradiction_penalty(record, task_prompt)
    )
    return max(0.0, min(1.0, score))


def _planning_quality_seed_realization_guarded_score(
    record: dict[str, object],
    task_prompt: str,
) -> float:
    """Repair selector that treats compact seed integration as a first-class signal."""
    base_score = _planning_quality_prompt_coverage_guarded_score(record, task_prompt)
    components = _seed_realization_quality_components(record, task_prompt)
    if not components["active_seed_anchor"]:
        return base_score
    realization_quality = float(components["realization_quality_score"])
    low_realization_penalty = max(0.0, 0.35 - realization_quality) * 0.75
    score = 0.55 * base_score + 0.45 * realization_quality - low_realization_penalty
    return max(0.0, min(1.0, score))


def _planning_quality_seed_objective_guarded_score(
    record: dict[str, object],
    task_prompt: str,
) -> float:
    """Repair selector that rewards direct realization and seed-semantic preservation."""
    base_score = _planning_quality_prompt_coverage_guarded_score(record, task_prompt)
    components = _seed_realization_quality_components(record, task_prompt)
    if not components["active_seed_anchor"]:
        return base_score
    seed_objective = float(components["seed_objective_score"])
    semantic_score = float(components["semantic_preservation_score"])
    low_objective_penalty = max(0.0, 0.55 - seed_objective) * 0.50
    low_semantic_penalty = max(0.0, 0.60 - semantic_score) * 0.35
    score = 0.48 * base_score + 0.32 * seed_objective + 0.20 * semantic_score
    score -= low_objective_penalty + low_semantic_penalty
    return max(0.0, min(1.0, score))


def _seed_realization_quality_score(record: dict[str, object], task_prompt: str = "") -> float:
    """Label-free score for whether a compact seed became a direct useful plan."""
    return float(_seed_realization_quality_components(record, task_prompt)["realization_quality_score"])


def _seed_objective_score(record: dict[str, object], task_prompt: str = "") -> float:
    """Joint score for direct compact-seed realization plus semantic preservation."""
    return float(_seed_realization_quality_components(record, task_prompt)["seed_objective_score"])


def _seed_realization_quality_components(
    record: dict[str, object],
    task_prompt: str = "",
) -> dict[str, object]:
    """Score compact semantic anchors as realized control plans, not raw term hits.

    The target is the current diffusion failure mode: a generated seed can bind
    every required control token and still be a weak repair if the denoised text
    surfaces those controls as meta labels, seed chatter, or checklist residue.
    """
    text = str(record.get("text", ""))
    normalized = _normalize(text)
    anchor = _active_seed_suffix_anchor(record)
    if not anchor:
        return {
            "active_seed_anchor": False,
            "action_coverage": 0.0,
            "control_coverage": 0.0,
            "expected_seed_text": "",
            "meta_penalty": 0.0,
            "prompt_coverage": 0.0,
            "realization_quality_score": 0.0,
            "seed_objective_score": 0.0,
            "seed_term_coverage": 0.0,
            "semantic_preservation_score": 0.0,
            "sentence_shape_score": 0.0,
            "specificity_score": 0.0,
            "word_count": len(normalized.split()),
        }
    expected_seed = _normalize(str(anchor.get("seed_suffix_text", "")))
    if not expected_seed:
        expected_seed = _normalize(str(anchor.get("generated_seed_suffix_text", "")))

    control_terms = (
        "token budget",
        "prompt format",
        "locked tasks",
        "regressions",
        "wins",
        "failure modes",
        "oracle selected results",
        "claim survives",
        "preserve claim",
    )
    action_terms = (
        "equalize",
        "rerun",
        "record",
        "validate",
        "state",
        "separate",
        "report",
    )
    control_coverage = _phrase_coverage(normalized, control_terms)
    action_coverage = _phrase_coverage(normalized, action_terms)
    seed_term_coverage = _seed_term_coverage(normalized, expected_seed)
    prompt_coverage = _prompt_keyword_coverage(task_prompt, normalized) if task_prompt else 0.0
    word_count = len(normalized.split())
    specificity_score = max(0.0, min(1.0, (word_count - 14.0) / 42.0))
    sentence_shape_score = _direct_sentence_shape_score(text)
    meta_penalty = _seed_realization_meta_penalty(normalized)
    semantic_preservation_score = _seed_semantic_preservation_score(normalized, expected_seed)
    realization_score = (
        0.26 * control_coverage
        + 0.24 * action_coverage
        + 0.18 * seed_term_coverage
        + 0.14 * prompt_coverage
        + 0.10 * specificity_score
        + 0.08 * sentence_shape_score
        - meta_penalty
    )
    bounded_realization_score = max(0.0, min(1.0, realization_score))
    seed_objective_score = 0.62 * bounded_realization_score + 0.38 * semantic_preservation_score
    return {
        "active_seed_anchor": bool(anchor),
        "action_coverage": action_coverage,
        "control_coverage": control_coverage,
        "expected_seed_text": expected_seed,
        "meta_penalty": meta_penalty,
        "prompt_coverage": prompt_coverage,
        "realization_quality_score": bounded_realization_score,
        "seed_objective_score": max(0.0, min(1.0, seed_objective_score)),
        "seed_term_coverage": seed_term_coverage,
        "semantic_preservation_score": semantic_preservation_score,
        "sentence_shape_score": sentence_shape_score,
        "specificity_score": specificity_score,
        "word_count": word_count,
    }


def _active_seed_suffix_anchor(record: dict[str, object]) -> dict[str, object]:
    repair = record.get("repair")
    if not isinstance(repair, dict):
        return {}
    anchor = repair.get("planning_seed_suffix_anchor")
    if not isinstance(anchor, dict) or not anchor.get("active"):
        return {}
    return anchor


def _phrase_coverage(text: str, phrases: tuple[str, ...]) -> float:
    if not phrases:
        return 0.0
    return sum(1 for phrase in phrases if phrase in text) / len(phrases)


def _seed_term_coverage(text: str, seed_text: str) -> float:
    terms = tuple(
        term
        for term in ("oracle", "selected", "results", "claim", "survives", "preserve", "disappears")
        if term in seed_text
    )
    if not terms:
        return 0.0
    return _phrase_coverage(text, terms)


def _seed_semantic_preservation_score(text: str, seed_text: str) -> float:
    """Score whether realized text preserves the compact seed's relation semantics."""
    expected = _normalize(seed_text)
    scores: list[float] = []
    if "oracle" in expected and "selected" in expected:
        has_oracle = "oracle" in text
        has_selected = "selected" in text
        has_results = "result" in text
        has_separation = _contains_any(text, ("separate", "distinguish", "split"))
        if has_oracle and has_selected and has_separation:
            scores.append(1.0 if has_results else 0.92)
        elif has_oracle and has_selected and "compare" in text:
            scores.append(0.55)
        elif "oracle selected results" in text:
            scores.append(0.78)
        elif has_oracle and has_selected:
            scores.append(0.68)
        elif has_oracle or has_selected:
            scores.append(0.30)
        else:
            scores.append(0.0)
    if "claim" in expected and _contains_any(expected, ("surviv", "preserv")):
        has_claim = "claim" in text
        has_survival = _contains_any(text, ("surviv", "preserv"))
        has_disappear_condition = "disappear" in text
        if has_claim and has_survival and has_disappear_condition:
            scores.append(1.0)
        elif has_claim and has_survival:
            scores.append(0.78)
        elif has_claim:
            scores.append(0.35)
        else:
            scores.append(0.0)
    if not scores:
        return 0.0
    return max(0.0, min(1.0, sum(scores) / len(scores)))


def _direct_sentence_shape_score(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    normalized = _normalize(stripped)
    score = 1.0
    if re.search(r"\b(control|checklist|seed|anchor|draft|instruction|repair)\s*:", stripped, re.IGNORECASE):
        score -= 0.55
    if "\n" in stripped and len([line for line in stripped.splitlines() if line.strip()]) > 3:
        score -= 0.15
    if _contains_any(normalized, ("generated compact seed", "generated seed", "seed anchor")):
        score -= 0.25
    return max(0.0, min(1.0, score))


def _seed_realization_meta_penalty(normalized_text: str) -> float:
    penalty = 0.0
    weighted_phrases = (
        ("control:", 0.20),
        ("generated compact seed", 0.18),
        ("generated seed", 0.14),
        ("seed anchor", 0.14),
        ("compare to the anchor", 0.16),
        ("discuss the seed", 0.14),
        ("masked", 0.10),
        ("mask", 0.10),
        ("draft", 0.10),
        ("checklist", 0.10),
        ("instruction", 0.10),
        ("repair", 0.08),
        ("denoise", 0.08),
        ("compare to oracle selected", 0.08),
        ("same fixed seed", 0.05),
        ("fixed seed", 0.04),
    )
    for phrase, value in weighted_phrases:
        if phrase in normalized_text:
            penalty += value
    return min(0.45, penalty)


def _repair_overpreservation_penalty(record: dict[str, object]) -> float:
    repair = record.get("repair")
    if not isinstance(repair, dict):
        return 0.0
    if str(repair.get("source_state", "")) != "history":
        return 0.0
    masked_positions = _nested_float(record, ("repair", "seed_masked_positions"))
    max_new_tokens = _config_number(record, "max_new_tokens")
    if masked_positions <= 0 or max_new_tokens <= 0:
        return 0.0
    masked_fraction = masked_positions / max_new_tokens
    if masked_fraction >= 0.35:
        return 0.0
    visible_chars = _nested_float(record, ("repair", "source_history_visible_chars"))
    visible_pressure = max(0.0, min(1.0, (visible_chars - 120.0) / 160.0))
    if visible_pressure <= 0.0:
        return 0.0
    return min(0.18, (0.35 - masked_fraction) * visible_pressure * 0.40)


def _planning_contradiction_penalty(record: dict[str, object], task_prompt: str) -> float:
    """Label-free penalty for plans that violate explicit prompt constraints."""
    normalized = _normalize(str(record.get("text", "")))
    prompt = _normalize(task_prompt)
    if not normalized or not prompt:
        return 0.0
    penalty = _prompt_checklist_leakage_penalty(record)
    penalty += _planning_span_residue_penalty(record)
    if "baseline" in prompt:
        if _contains_any(
            normalized,
            (
                "skip the baseline",
                "skip baseline",
                "ignore the baseline",
                "ignore baseline",
                "discard the baseline",
                "discard baseline",
                "without a baseline",
                "without baseline",
                "no baseline",
            ),
        ):
            penalty += 0.16
        if _contains_any(normalized, ("baseline data will not be available", "baseline data is not available")) and (
            "valid comparison" in normalized or "publishable result" in normalized
        ):
            penalty += 0.16
        if (
            ("two gpu" in prompt or "only two" in prompt)
            and "risky" in prompt
            and _contains_any(
                normalized,
                (
                    "both jobs on risky",
                    "both jobs for risky",
                    "both jobs to risky",
                    "spend both jobs on the intervention",
                    "spend both jobs on risky",
                    "run only risky variants",
                    "only risky variants",
                ),
            )
        ):
            penalty += 0.16
    if (
        "checkpoint" in prompt
        and _contains_any(normalized, ("delete", "remove", "purge", "clean"))
        and not _contains_any(normalized, ("verify", "validated", "latest good", "copy", "backup", "preserve"))
    ):
        penalty += 0.14
    if (
        ("ship" in prompt and "rollback" in prompt and "gate" in prompt)
        and _contains_any(
            normalized,
            ("ship immediately", "ship to all", "ship everything", "full rollout"),
        )
        and not _contains_any(normalized, ("gate", "canary", "rollback", "threshold"))
    ):
        penalty += 0.12
    if "demo" in prompt and _contains_any(
        normalized,
        ("risk the demo", "break the demo", "skip the demo", "disable the demo"),
    ):
        penalty += 0.12
    if ("root-cause" in prompt or "root cause" in prompt) and _contains_any(
        normalized,
        ("skip root cause", "ignore root cause", "ignore timezone", "not investigate"),
    ):
        penalty += 0.12
    if "optimizer" in prompt and "one gpu" in prompt and _contains_any(
        normalized,
        ("change multiple variables", "change several variables", "increase learning rate"),
    ):
        penalty += 0.10
    if (
        "gaming the scorer" in prompt
        and _contains_any(
            normalized,
            ("trust the score", "use the score alone", "score is enough", "validate the score improvement"),
        )
        and not _contains_any(normalized, ("qualitative", "examples", "specificity", "held-out", "adversarial"))
        ):
            penalty += 0.10
    return min(0.24, penalty)


def _planning_span_residue_penalty(record: dict[str, object]) -> float:
    """Penalize span repairs that regenerate verifier-targeted weak spans."""
    repair = record.get("repair")
    if not isinstance(repair, dict):
        return 0.0
    targets_value = repair.get("planning_span_targets")
    if not isinstance(targets_value, list) or not targets_value:
        return 0.0
    normalized_text = _normalize(str(record.get("text", "")))
    if not normalized_text:
        return 0.0
    residue_count = 0
    for target in targets_value:
        normalized_target = _normalize(str(target))
        if len(normalized_target.split()) < 4:
            continue
        if normalized_target in normalized_text:
            residue_count += 1
    if residue_count >= 2:
        return 0.18
    if residue_count == 1:
        return 0.12
    return 0.0


def _repair_span_literal_target_found(record: dict[str, object]) -> float:
    repair = record.get("repair")
    if not isinstance(repair, dict):
        return 0.0
    if not isinstance(repair.get("span_seed_diagnostics"), dict):
        return 0.0
    return 1.0 if repair.get("span_literal_target_found") else 0.0


def _repair_span_fallback_used(record: dict[str, object]) -> float:
    repair = record.get("repair")
    if not isinstance(repair, dict):
        return 0.0
    if not isinstance(repair.get("span_seed_diagnostics"), dict):
        return 0.0
    return 1.0 if repair.get("span_fallback_used") else 0.0


def _prompt_checklist_leakage_penalty(record: dict[str, object]) -> float:
    repair = record.get("repair")
    if not isinstance(repair, dict):
        return 0.0
    terms_value = repair.get("prompt_constraint_gap_terms")
    if not isinstance(terms_value, list) or not terms_value:
        return 0.0
    gap_terms = {
        _normalize(str(term))
        for term in terms_value
        if _normalize(str(term))
    }
    if not gap_terms:
        return 0.0
    text = str(record.get("text", ""))
    normalized_text = _normalize(text)
    if "missing or weak task terms" in normalized_text:
        return 0.18
    highest_leak_count = 0
    for fragment in re.split(r"[.;\n]", text):
        parts = [
            _normalize(part)
            for part in fragment.split(",")
            if _normalize(part)
        ]
        if len(parts) < 5:
            continue
        leak_count = sum(
            1
            for part in parts
            if part in gap_terms or any(term in part.split() for term in gap_terms)
        )
        highest_leak_count = max(highest_leak_count, leak_count)
    if highest_leak_count >= 7:
        return 0.18
    if highest_leak_count >= 5:
        return 0.12
    return 0.0


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _planning_quality_score(record: dict[str, object], task_prompt: str) -> float:
    text = str(record.get("text", ""))
    generic_task = GeneralReasoningTask(
        task_id=_task_id(record),
        family="planning",
        prompt=task_prompt,
        answer_type="rubric",
        scorer="generic_planning_surface",
        max_new_tokens=64,
    )
    return max(0.0, min(1.0, score_planning_output(generic_task, text).score))


def _attach_planning_quality_score(record: dict[str, object], task: GeneralReasoningTask) -> None:
    if task.answer_type != "rubric":
        return
    record["planning_quality_score"] = _planning_quality_score(record, task.prompt)


def _record_planning_quality_score(record: dict[str, object]) -> float | None:
    value = record.get("planning_quality_score")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _planning_prompt_selector_score(record: dict[str, object], task_prompt: str) -> float:
    text = str(record.get("text", ""))
    final_surface = _planning_surface_score(text, task_prompt, task_id=_task_id(record))
    score = 0.30 * _trajectory_score(record) + 0.70 * final_surface
    return max(0.0, min(1.0, score))


def _planning_state_selector_score(record: dict[str, object], task_prompt: str) -> float:
    final_surface = _planning_surface_score(
        str(record.get("text", "")),
        task_prompt,
        task_id=_task_id(record),
    )
    sample_scores = _planning_history_surface_scores(record, task_prompt)
    if not sample_scores:
        return _planning_prompt_selector_score(record, task_prompt)
    peak_surface = max(sample_scores)
    mean_surface = _mean(sample_scores)
    early_surface = max(sample_scores[: max(1, len(sample_scores) // 2)])
    stability_score = max(0.0, 1.0 - min(1.0, abs(peak_surface - final_surface)))
    score = (
        0.35 * final_surface
        + 0.25 * peak_surface
        + 0.20 * mean_surface
        + 0.10 * early_surface
        + 0.10 * stability_score
    )
    return max(0.0, min(1.0, score))


def _planning_state_v2_selector_score(record: dict[str, object], task_prompt: str) -> float:
    final_surface = _planning_surface_v2_score(
        str(record.get("text", "")),
        task_prompt,
        task_id=_task_id(record),
    )
    sample_scores = _planning_history_surface_v2_scores(record, task_prompt)
    if not sample_scores:
        prompt_surface = _planning_prompt_selector_score(record, task_prompt)
        return max(0.0, min(1.0, 0.45 * final_surface + 0.55 * prompt_surface))
    peak_surface = max(sample_scores)
    mean_surface = _mean(sample_scores)
    late_surface = max(sample_scores[len(sample_scores) // 2 :])
    stability_score = max(0.0, 1.0 - min(1.0, abs(peak_surface - final_surface)))
    score = (
        0.45 * final_surface
        + 0.20 * peak_surface
        + 0.15 * mean_surface
        + 0.10 * late_surface
        + 0.10 * stability_score
    )
    return max(0.0, min(1.0, score))


def _planning_history_surface_scores(record: dict[str, object], task_prompt: str) -> list[float]:
    trajectory_summary = record.get("trajectory_summary")
    if not isinstance(trajectory_summary, dict):
        return []
    samples = trajectory_summary.get("samples")
    if not isinstance(samples, list):
        return []
    scores = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        visible_text = str(sample.get("visible_text", ""))
        if visible_text.strip():
            scores.append(
                _planning_surface_score(
                    visible_text,
                    task_prompt,
                    task_id=_task_id(record),
                )
            )
    return scores


def _planning_history_surface_v2_scores(record: dict[str, object], task_prompt: str) -> list[float]:
    trajectory_summary = record.get("trajectory_summary")
    if not isinstance(trajectory_summary, dict):
        return []
    samples = trajectory_summary.get("samples")
    if not isinstance(samples, list):
        return []
    scores = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        visible_text = str(sample.get("visible_text", ""))
        if visible_text.strip():
            scores.append(
                _planning_surface_v2_score(
                    visible_text,
                    task_prompt,
                    task_id=_task_id(record),
                )
            )
    return scores


def _selected_history_repair_sample(
    record: dict[str, object],
    task_prompt: str,
) -> dict[str, object] | None:
    candidates = _history_repair_sample_candidates(record, task_prompt)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda sample: (
            float(sample["selection_score"]),
            int(sample["visible_chars"]),
            int(sample["step"]),
        ),
    )


def _history_repair_sample_candidates(
    record: dict[str, object],
    task_prompt: str,
) -> list[dict[str, object]]:
    history_samples = record.get("history_samples")
    if not isinstance(history_samples, list):
        return []
    summary_by_step = _trajectory_summary_samples_by_step(record)
    candidates = []
    for sample in history_samples:
        if not isinstance(sample, dict):
            continue
        generated_token_ids = _int_list(sample.get("generated_token_ids"))
        if not generated_token_ids:
            continue
        step = _int_value(sample.get("step"), default=0)
        summary_sample = summary_by_step.get(step, {})
        mask_count = _int_value(
            summary_sample.get("mask_count"),
            default=_llada_mask_token_count(generated_token_ids),
        )
        if mask_count <= 0:
            continue
        visible_text = str(summary_sample.get("visible_text") or sample.get("text") or "")
        visible_chars = _int_value(summary_sample.get("visible_chars"), default=len(visible_text.strip()))
        if visible_chars < 8:
            continue
        selection_score = _planning_surface_score(
            visible_text,
            task_prompt,
            task_id=_task_id(record),
        )
        candidates.append(
            {
                "generated_token_ids": generated_token_ids,
                "mask_count": mask_count,
                "selection_score": selection_score,
                "step": step,
                "visible_chars": visible_chars,
                "visible_text": visible_text,
            }
        )
    return candidates


def _choose_pre_generation_repair_anchor(
    record: dict[str, object],
    task_prompt: str,
    *,
    search_history: bool = False,
    phase_anchor: bool = False,
    phase_hybrid: bool = False,
    phase_prompt_coverage_min: float = 0.40,
    phase_source_history_char_ratio_min: float = PHASE_SOURCE_HISTORY_CHAR_RATIO_MIN,
    phase_source_target_similarity_min: float = PHASE_SOURCE_TARGET_SIMILARITY_MIN,
    phase_source_text_similarity_min: float = PHASE_SOURCE_TEXT_SIMILARITY_MIN,
) -> dict[str, object]:
    """Choose one repair anchor from label-free source/history geometry."""
    final_text = str(record.get("text", ""))
    history_samples = (
        _history_repair_sample_candidates(record, task_prompt)
        if search_history or phase_anchor or phase_hybrid
        else [sample for sample in [_selected_history_repair_sample(record, task_prompt)] if sample]
    )
    if not task_prompt.strip() or not final_text.strip() or not history_samples:
        return _anchor_choice("final", "missing_prompt_final_or_history")

    final_gaps = _prompt_constraint_gap_terms(task_prompt, final_text)
    final_targets = _planning_constraint_gap_span_target_scores(
        task_prompt,
        final_text,
        final_gaps,
        chunk_mode="adaptive",
        selection_policy="compact",
    )
    scored_samples = []
    for history_sample in history_samples:
        history_text = str(history_sample.get("visible_text", ""))
        if not history_text.strip():
            continue
        history_gaps = _prompt_constraint_gap_terms(task_prompt, history_text)
        history_targets = _planning_constraint_gap_span_target_scores(
            task_prompt,
            history_text,
            history_gaps,
            chunk_mode="adaptive",
            selection_policy="compact",
        )
        features = _pre_generation_anchor_features(
            prompt=task_prompt,
            final_text=final_text,
            history_text=history_text,
            final_targets=final_targets,
            history_targets=history_targets,
        )
        features["history_mask_count"] = int(history_sample.get("mask_count", 0))
        features["history_retention_loss"] = _anchor_retention_loss(features)
        features["history_safety_char_ratio_min"] = ANCHOR_HISTORY_CHAR_RATIO_MIN
        features["history_safety_target_similarity_min"] = ANCHOR_TARGET_SIMILARITY_MIN
        features["phase_history_safety_char_ratio_min"] = PHASE_ANCHOR_HISTORY_CHAR_RATIO_MIN
        features["phase_history_safety_target_similarity_min"] = PHASE_ANCHOR_TARGET_SIMILARITY_MIN
        features["phase_source_history_char_ratio_min"] = phase_source_history_char_ratio_min
        features["phase_source_target_similarity_min"] = phase_source_target_similarity_min
        features["phase_source_text_similarity_min"] = phase_source_text_similarity_min
        history_prompt_coverage = _prompt_keyword_coverage(task_prompt, _normalize(history_text))
        features["history_prompt_coverage"] = round(history_prompt_coverage, 6)
        features["history_phase_prompt_coverage_min"] = phase_prompt_coverage_min
        features["history_repairable_denoise_skeleton"] = (
            int(history_sample.get("visible_chars", 0)) >= 20
            and history_prompt_coverage >= phase_prompt_coverage_min
        )
        features["history_selection_score"] = round(_number(history_sample.get("selection_score")), 6)
        features["history_step"] = int(history_sample.get("step", 0))
        features["history_visible_chars"] = int(history_sample.get("visible_chars", 0))
        scored_samples.append((features, history_sample))
    if not scored_samples:
        return _anchor_choice("final", "missing_history_text")
    if phase_anchor or phase_hybrid:
        repairable_samples = [
            (features, history_sample)
            for features, history_sample in scored_samples
            if bool(features.get("history_repairable_denoise_skeleton", False))
        ]
        repairable_safe_samples = [
            (features, history_sample)
            for features, history_sample in repairable_samples
            if _phase_history_anchor_is_retention_safe(features)
        ]
        phase_summary_features = _phase_anchor_summary_features(
            repairable_samples,
            repairable_safe_samples,
        )
        if phase_hybrid:
            source_advantage_samples = [
                (_with_phase_anchor_summary(features, phase_summary_features), history_sample)
                for features, history_sample in repairable_safe_samples
                if _phase_history_anchor_has_source_advantage(features)
            ]
            if source_advantage_samples:
                features, history_sample = min(
                    source_advantage_samples,
                    key=lambda item: (
                        -_number(item[0].get("history_span_score_delta")),
                        _number(item[0].get("history_retention_loss")),
                        _number(item[0].get("history_step")),
                    ),
                )
                return _anchor_choice(
                    "history",
                    "phase_hybrid_history_source_advantage",
                    features,
                    history_sample=history_sample,
                )
            features, _history_sample = min(
                repairable_safe_samples or repairable_samples or scored_samples,
                key=lambda item: (
                    _number(item[0].get("history_step")),
                    _number(item[0].get("history_retention_loss")),
                ),
            )
            if repairable_safe_samples:
                reason = "phase_hybrid_final_no_source_advantage"
            elif repairable_samples:
                reason = "phase_hybrid_final_no_safe_repairable_skeleton"
            else:
                reason = "phase_hybrid_final_no_repairable_skeleton"
            return _anchor_choice(
                "final",
                reason,
                _with_phase_anchor_summary(features, phase_summary_features),
            )
        if repairable_safe_samples:
            features, history_sample = min(
                repairable_safe_samples,
                key=lambda item: (
                    _number(item[0].get("history_step")),
                    _number(item[0].get("history_retention_loss")),
                    -_number(item[0].get("history_span_score_delta")),
                ),
            )
            return _anchor_choice(
                "history",
                "history_phase_first_repairable_skeleton",
                _with_phase_anchor_summary(features, phase_summary_features),
                history_sample=history_sample,
            )
        features, _history_sample = min(
            repairable_samples or scored_samples,
            key=lambda item: (
                _number(item[0].get("history_step")),
                _number(item[0].get("history_retention_loss")),
            ),
        )
        reason = (
            "phase_anchor_not_retention_safe"
            if repairable_samples
                else "phase_anchor_no_repairable_skeleton"
        )
        return _anchor_choice(
            "final",
            reason,
            _with_phase_anchor_summary(features, phase_summary_features),
        )
    eligible = [
        (features, history_sample)
        for features, history_sample in scored_samples
        if _history_anchor_is_retention_safe(features)
    ]
    if eligible:
        features, history_sample = min(
            eligible,
            key=lambda item: (
                _number(item[0].get("history_retention_loss")),
                -_number(item[0].get("history_span_score_delta")),
                -_number(item[0].get("history_step")),
            ),
        )
        reason = (
            "history_search_retention_loss_minimum"
            if search_history
            else "history_single_span_score_advantage"
        )
        return _anchor_choice("history", reason, features, history_sample=history_sample)
    features, _history_sample = min(
        scored_samples,
        key=lambda item: (
            _number(item[0].get("history_retention_loss")),
            -_number(item[0].get("history_span_score_delta")),
            -_number(item[0].get("history_step")),
        ),
    )
    return _anchor_choice("final", "final_source_preserves_more_context", features)


def _history_anchor_is_retention_safe(features: dict[str, object]) -> bool:
    return (
        features["history_target_count"] == 1
        and features["final_target_count"] == 1
        and features["text_similarity"] >= 0.93
        and features["target_similarity"] >= ANCHOR_TARGET_SIMILARITY_MIN
        and features["history_to_final_char_ratio"] >= ANCHOR_HISTORY_CHAR_RATIO_MIN
        and features["lost_digit_token_count"] == 0
        and features["lost_prompt_keyword_count"] == 0
        and features["history_span_score_delta"] > 1e-6
    )


def _phase_history_anchor_is_retention_safe(features: dict[str, object]) -> bool:
    return (
        bool(features.get("history_repairable_denoise_skeleton", False))
        and features["history_target_count"] == features["final_target_count"]
        and features["history_target_count"] >= 1
        and features["text_similarity"] >= PHASE_ANCHOR_HISTORY_CHAR_RATIO_MIN
        and features["target_similarity"] >= PHASE_ANCHOR_TARGET_SIMILARITY_MIN
        and features["history_to_final_char_ratio"] >= PHASE_ANCHOR_HISTORY_CHAR_RATIO_MIN
        and features["lost_digit_token_count"] == 0
        and features["lost_prompt_keyword_count"] == 0
    )


def _phase_history_anchor_has_source_advantage(features: dict[str, object]) -> bool:
    return (
        _phase_history_anchor_is_retention_safe(features)
        and _phase_history_anchor_passes_source_policy(features)
        and features["history_target_count"] == 1
        and features["final_target_count"] == 1
        and features["lost_digit_token_count"] == 0
        and features["lost_prompt_keyword_count"] == 0
        and features["history_span_score_delta"] > 1e-6
    )


def _phase_history_anchor_passes_source_policy(features: dict[str, object]) -> bool:
    return (
        bool(features.get("history_repairable_denoise_skeleton", False))
        and features["text_similarity"]
        >= _number(features.get("phase_source_text_similarity_min", PHASE_SOURCE_TEXT_SIMILARITY_MIN))
        and features["target_similarity"]
        >= _number(
            features.get("phase_source_target_similarity_min", PHASE_SOURCE_TARGET_SIMILARITY_MIN)
        )
        and features["history_to_final_char_ratio"]
        >= _number(
            features.get("phase_source_history_char_ratio_min", PHASE_SOURCE_HISTORY_CHAR_RATIO_MIN)
        )
    )


def _phase_anchor_summary_features(
    repairable_samples: list[tuple[dict[str, object], dict[str, object]]],
    repairable_safe_samples: list[tuple[dict[str, object], dict[str, object]]],
) -> dict[str, object]:
    first_repairable_step = _min_history_step(repairable_samples)
    first_safe_step = _min_history_step(repairable_safe_samples)
    retention_safety_lag = None
    if first_repairable_step is not None and first_safe_step is not None:
        retention_safety_lag = first_safe_step - first_repairable_step
    return {
        "phase_repairable_sample_count": len(repairable_samples),
        "phase_safe_repairable_sample_count": len(repairable_safe_samples),
        "phase_first_repairable_step": first_repairable_step,
        "phase_first_safe_repairable_step": first_safe_step,
        "phase_retention_safety_lag": retention_safety_lag,
    }


def _with_phase_anchor_summary(
    features: dict[str, object],
    phase_summary_features: dict[str, object],
) -> dict[str, object]:
    return {**features, **phase_summary_features}


def _min_history_step(
    samples: list[tuple[dict[str, object], dict[str, object]]],
) -> int | None:
    steps = [int(_number(features.get("history_step"))) for features, _sample in samples]
    return min(steps) if steps else None


def _anchor_retention_loss(features: dict[str, object]) -> float:
    target_count_gap = abs(
        int(_number(features.get("history_target_count")))
        - int(_number(features.get("final_target_count")))
    )
    loss = (
        (1.0 - _number(features.get("target_similarity")))
        + 0.25 * len(_list(features.get("lost_target_tokens")))
        + 0.75 * _number(features.get("lost_prompt_keyword_count"))
        + _number(features.get("lost_digit_token_count"))
        + max(0.0, 0.90 - _number(features.get("history_to_final_char_ratio")))
        + 0.25 * target_count_gap
    )
    return round(loss, 6)


def _pre_generation_anchor_features(
    *,
    prompt: str,
    final_text: str,
    history_text: str,
    final_targets: list[dict[str, object]],
    history_targets: list[dict[str, object]],
) -> dict[str, object]:
    final_target_text = " ".join(str(target.get("span", "")) for target in final_targets)
    history_target_text = " ".join(str(target.get("span", "")) for target in history_targets)
    lost_tokens = _target_tokens_missing_from_history(final_target_text, history_target_text)
    prompt_keywords = set(_keywords(prompt))
    history_span_score = _number(history_targets[0].get("score")) if history_targets else 0.0
    final_span_score = _number(final_targets[0].get("score")) if final_targets else 0.0
    return {
        "final_span_score": round(final_span_score, 6),
        "final_target_count": len(final_targets),
        "history_span_score": round(history_span_score, 6),
        "history_span_score_delta": round(history_span_score - final_span_score, 6),
        "history_target_count": len(history_targets),
        "history_to_final_char_ratio": round(len(history_text.strip()) / max(1, len(final_text.strip())), 6),
        "lost_digit_token_count": sum(1 for token in lost_tokens if any(char.isdigit() for char in token)),
        "lost_prompt_keyword_count": sum(1 for token in lost_tokens if token in prompt_keywords),
        "lost_target_tokens": lost_tokens[:8],
        "target_similarity": round(_text_similarity(final_target_text, history_target_text), 6),
        "text_similarity": round(_text_similarity(final_text, history_text), 6),
    }


def _target_tokens_missing_from_history(final_target_text: str, history_target_text: str) -> list[str]:
    history_tokens = set(_word_tokens(history_target_text))
    missing = []
    for token in _word_tokens(final_target_text):
        if token in history_tokens or token in missing:
            continue
        missing.append(token)
    return missing


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize(text))


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()


def _anchor_choice(
    anchor_choice: str,
    reason: str,
    features: dict[str, object] | None = None,
    *,
    history_sample: dict[str, object] | None = None,
) -> dict[str, object]:
    choice = {"anchor_choice": anchor_choice, "features": features or {}, "reason": reason}
    if history_sample is not None:
        choice["history_sample"] = history_sample
    return choice


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _trajectory_summary_samples_by_step(record: dict[str, object]) -> dict[int, dict[str, object]]:
    trajectory_summary = record.get("trajectory_summary")
    if not isinstance(trajectory_summary, dict):
        return {}
    samples = trajectory_summary.get("samples")
    if not isinstance(samples, list):
        return {}
    by_step = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        step = _int_value(sample.get("step"), default=-1)
        if step >= 0:
            by_step[step] = sample
    return by_step


def _planning_surface_score(text: str, task_prompt: str, *, task_id: str) -> float:
    normalized = _normalize(text)
    generic_task = GeneralReasoningTask(
        task_id=task_id,
        family="planning",
        prompt=task_prompt,
        answer_type="rubric",
        scorer="generic_planning_surface",
        max_new_tokens=64,
    )
    generic_score = score_planning_output(generic_task, text).score
    prompt_score = _prompt_keyword_coverage(task_prompt, normalized)
    repetition_penalty = _repetition_penalty(normalized)
    filler_penalty = _filler_penalty(normalized)
    score = (
        0.70 * generic_score
        + 0.30 * prompt_score
        - repetition_penalty
        - filler_penalty
    )
    return max(0.0, min(1.0, score))


def _planning_surface_v2_score(text: str, task_prompt: str, *, task_id: str) -> float:
    normalized = _normalize(text)
    generic_task = GeneralReasoningTask(
        task_id=task_id,
        family="planning",
        prompt=task_prompt,
        answer_type="rubric",
        scorer="generic_planning_surface",
        max_new_tokens=64,
    )
    generic_score = score_planning_output(generic_task, text).score
    prompt_score = _prompt_keyword_coverage(task_prompt, normalized)
    phrase_score = _prompt_phrase_coverage(task_prompt, normalized)
    action_score = _action_structure_score(normalized)
    specificity_score = _specificity_score(normalized, task_prompt)
    score = (
        0.30 * generic_score
        + 0.30 * prompt_score
        + 0.15 * phrase_score
        + 0.15 * action_score
        + 0.10 * specificity_score
        - _repetition_penalty(normalized)
        - _filler_penalty(normalized)
        - _destructive_action_penalty(normalized, task_prompt)
    )
    return max(0.0, min(1.0, score))


def _prompt_keyword_coverage(prompt: str, normalized_text: str) -> float:
    keywords = _keywords(prompt)
    if not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in normalized_text)
    return hits / len(keywords)


def _prompt_phrase_coverage(prompt: str, normalized_text: str) -> float:
    keywords = _keywords(prompt)
    if len(keywords) < 2:
        return 0.0
    phrases = [f"{first} {second}" for first, second in zip(keywords, keywords[1:], strict=False)]
    if not phrases:
        return 0.0
    hits = sum(1 for phrase in phrases if phrase in normalized_text)
    return hits / len(phrases)


def _action_structure_score(normalized_text: str) -> float:
    action_markers = (
        "identify",
        "isolate",
        "collect",
        "compare",
        "measure",
        "verify",
        "validate",
        "preserve",
        "copy",
        "resume",
        "recover",
        "repeat",
        "document",
        "monitor",
        "rollback",
        "falsify",
        "fix",
        "triage",
        "analyze",
        "investigate",
        "root cause",
    )
    marker_hits = sum(1 for marker in action_markers if marker in normalized_text)
    numbered_steps = len(re.findall(r"(?:^|\s)\d+[.)]", normalized_text))
    return min(1.0, (marker_hits * 0.10) + min(0.35, numbered_steps * 0.06))


def _specificity_score(normalized_text: str, task_prompt: str) -> float:
    prompt_keywords = _keywords(task_prompt)
    prompt_hits = sum(1 for keyword in prompt_keywords if keyword in normalized_text)
    concrete_markers = (
        "baseline",
        "intervention",
        "logs",
        "customer",
        "demo",
        "tokens",
        "prompt format",
        "checkpoint",
        "disk",
        "timezone",
        "dashboard",
        "qualitative outputs",
        "scorer",
        "root cause",
        "temporary fix",
        "same tasks",
    )
    marker_hits = sum(1 for marker in concrete_markers if marker in normalized_text)
    word_count = len(re.findall(r"[a-z0-9]+", normalized_text))
    length_score = 0.0
    if 25 <= word_count <= 95:
        length_score = 0.20
    elif 12 <= word_count < 25:
        length_score = 0.10
    return min(1.0, length_score + prompt_hits * 0.05 + marker_hits * 0.08)


def _destructive_action_penalty(normalized_text: str, task_prompt: str) -> float:
    penalty = 0.0
    if "baseline" in task_prompt.lower() and "discard the baseline" in normalized_text:
        penalty += 0.16
    if "checkpoint" in task_prompt.lower() and "restart" in normalized_text and "copy" not in normalized_text:
        penalty += 0.08
    prompt = task_prompt.lower()
    if (
        ("root-cause" in prompt or "root cause" in prompt)
        and "root cause" not in normalized_text
        and "cause" not in normalized_text
    ):
        penalty += 0.08
    return min(0.20, penalty)


def _keywords(text: str) -> list[str]:
    stopwords = {
        "about",
        "after",
        "again",
        "against",
        "before",
        "being",
        "could",
        "decide",
        "design",
        "does",
        "even",
        "first",
        "from",
        "give",
        "have",
        "into",
        "only",
        "rather",
        "result",
        "short",
        "should",
        "state",
        "team",
        "test",
        "that",
        "their",
        "there",
        "they",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "without",
    }
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    keywords = [token for token in tokens if len(token) >= 4 and token not in stopwords]
    return _dedupe(keywords)


def _repetition_penalty(normalized_text: str) -> float:
    words = re.findall(r"[a-z0-9]+", normalized_text)
    if len(words) < 8:
        return 0.0
    repeated_bigrams = 0
    for first, second in zip(words, words[1:], strict=False):
        if first == second:
            repeated_bigrams += 1
    return min(0.12, repeated_bigrams * 0.04)


def _filler_penalty(normalized_text: str) -> float:
    filler_phrases = (
        "complex nuanced tasks",
        "deep understanding",
        "actual reasoning ability",
        "quality of the qualitative outputs",
        "presentister",
    )
    return min(0.12, sum(1 for phrase in filler_phrases if phrase in normalized_text) * 0.04)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _dedupe_records_by_identity(records: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = set()
    deduped = []
    for record in records:
        identity = _record_identity(record)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(record)
    return deduped


def _stable_random_index(seed: int, candidate_key: str, task_id: str, count: int) -> int:
    digest = hashlib.sha256(f"{seed}:{candidate_key}:{task_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % count


def _stable_generation_seed(seed: int, candidate_key: str, task_id: str, schedule_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{candidate_key}:{task_id}:{schedule_name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _record_identity(record: dict[str, object]) -> tuple[str, str, int]:
    seed = record.get("generation_seed")
    generation_seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else -1
    return (str(record.get("candidate_key", "")), _control_name(record), generation_seed)


def _set_generation_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _mean_delta(left_records: list[dict[str, object]], right_records: list[dict[str, object]]) -> float:
    right_by_key = {(str(record["candidate_key"]), _task_id(record)): record for record in right_records}
    deltas = []
    for left in left_records:
        right = right_by_key.get((str(left["candidate_key"]), _task_id(left)))
        if right is not None:
            deltas.append(_task_score(left) - _task_score(right))
    return _mean(deltas)


def _mean_budget_delta(left_records: list[dict[str, object]], right_records: list[dict[str, object]]) -> float:
    right_by_key = {(str(record["candidate_key"]), _task_id(record)): record for record in right_records}
    deltas = []
    for left in left_records:
        right = right_by_key.get((str(left["candidate_key"]), _task_id(left)))
        if right is not None:
            deltas.append(_generation_budget(left) - _generation_budget(right))
    return _mean(deltas)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def _selector_regret_summary(
    oracle_records: list[dict[str, object]],
    selected_records: list[dict[str, object]],
) -> dict[str, object]:
    selected_by_key = {
        (str(record["candidate_key"]), _task_id(record)): record for record in selected_records
    }
    regrets = []
    for oracle in oracle_records:
        selected = selected_by_key.get((str(oracle["candidate_key"]), _task_id(oracle)))
        if selected is None:
            continue
        regrets.append(max(0.0, _task_score(oracle) - _task_score(selected)))
    improvable_count = sum(1 for regret in regrets if regret > 1e-9)
    count = len(regrets)
    return {
        "count": count,
        "mean_task_regret": _mean(regrets),
        "improvable_count": improvable_count,
        "improvable_fraction": _safe_ratio(float(improvable_count), float(count)),
        "wins_vs_selected": _win_count(oracle_records, selected_records),
    }


def _win_count(left_records: list[dict[str, object]], right_records: list[dict[str, object]]) -> dict[str, int]:
    right_by_key = {(str(record["candidate_key"]), _task_id(record)): record for record in right_records}
    counts = {"wins": 0, "ties": 0, "losses": 0}
    for left in left_records:
        right = right_by_key.get((str(left["candidate_key"]), _task_id(left)))
        if right is None:
            continue
        delta = _task_score(left) - _task_score(right)
        if delta > 1e-9:
            counts["wins"] += 1
        elif delta < -1e-9:
            counts["losses"] += 1
        else:
            counts["ties"] += 1
    return counts


def _format_wins(value: object) -> str:
    if not isinstance(value, dict):
        return "0/0/0"
    return f"{value.get('wins', 0)}/{value.get('ties', 0)}/{value.get('losses', 0)}"


def _format_selector_regret(value: object) -> str:
    if not isinstance(value, dict):
        return "0.000 over 0/0 improvable"
    count = int(value.get("count", 0))
    improvable = int(value.get("improvable_count", 0))
    mean_regret = float(value.get("mean_task_regret", 0.0))
    return f"{mean_regret:.3f} over {improvable}/{count} improvable"


def _format_count_map(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    return ", ".join(f"{key}={value[key]}" for key in sorted(value))


def _format_history_mutability(value: object) -> str:
    if not isinstance(value, dict):
        return "monotonic 0/0, changes 0, remasks 0, mask increases 0"
    return (
        f"monotonic {int(value.get('monotonic_fill_count', 0))}/"
        f"{int(value.get('count', 0))}, "
        f"changes {int(value.get('committed_token_change_count', 0))}, "
        f"remasks {int(value.get('committed_token_remask_count', 0))}, "
        f"rewrites {int(value.get('remasked_token_rewrite_count', 0))}, "
        f"mask increases {int(value.get('mask_count_increase_count', 0))}"
    )


def _arm_sort_key(item: tuple[str, list[dict[str, object]]]) -> tuple[int, str]:
    arm = item[0]
    return (ARM_ORDER.get(arm, len(ARM_ORDER)), arm)


def _combined_score(record: dict[str, object]) -> float:
    return 0.75 * _task_score(record) + 0.25 * _trajectory_score(record)


def _task_score(record: dict[str, object]) -> float:
    return _nested_float(record, ("task_score", "score"))


def _trajectory_score(record: dict[str, object]) -> float:
    return _nested_float(record, ("trajectory_control_score", "overall"))


def _generation_budget(record: dict[str, object]) -> float:
    value = record.get("arm_generation_budget_per_task")
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 1.0


def _selector_score(record: dict[str, object]) -> float:
    value = record.get("arm_selector_score")
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _repair_baseline_selector_score(
    repair: dict[str, object] | None,
    fallback_baseline: dict[str, object] | None,
) -> float:
    if repair is not None:
        value = repair.get("arm_selector_baseline_score")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return _selector_score(fallback_baseline) if fallback_baseline is not None else 0.0


def _repair_metadata_value(record: dict[str, object] | None, key: str) -> str:
    if record is None:
        return ""
    repair = record.get("repair")
    if not isinstance(repair, dict):
        return ""
    value = repair.get(key)
    return "" if value is None else str(value)


def _repair_metadata_bool(record: dict[str, object] | None, key: str, *, default: bool = False) -> bool:
    if record is None:
        return default
    repair = record.get("repair")
    if not isinstance(repair, dict):
        return default
    value = repair.get(key)
    return value if isinstance(value, bool) else default


def _nested_float(record: dict[str, object], path: tuple[str, str]) -> float:
    outer = record.get(path[0])
    if not isinstance(outer, dict):
        return 0.0
    value = outer.get(path[1])
    return float(value) if isinstance(value, int | float) else 0.0


def _float_value(value: object, *, default: float) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default


def _nested_value(record: dict[str, object], path: tuple[str, str]) -> object | None:
    outer = record.get(path[0])
    if not isinstance(outer, dict):
        return None
    return outer.get(path[1])


def _config_number(record: dict[str, object], key: str) -> float:
    config = record.get("config")
    if not isinstance(config, dict):
        return 0.0
    value = config.get(key)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _task_id(record: dict[str, object]) -> str:
    task = record.get("task")
    return str(task.get("task_id")) if isinstance(task, dict) else ""


def _task_family(record: dict[str, object]) -> str:
    task = record.get("task")
    return str(task.get("family")) if isinstance(task, dict) else ""


def _schedule_name(record: dict[str, object]) -> str:
    if record is None:
        return ""
    schedule = record.get("schedule")
    return str(schedule.get("name")) if isinstance(schedule, dict) else ""


def _repair_name(record: dict[str, object] | None) -> str:
    if record is None:
        return ""
    repair = record.get("repair")
    return str(repair.get("name")) if isinstance(repair, dict) else ""


def _control_name(record: dict[str, object] | None) -> str:
    return _repair_name(record) or _schedule_name(record)


def _is_base_schedule_record(record: dict[str, object]) -> bool:
    return bool(_schedule_name(record)) and not _is_evolved_record(record)


def _is_evolved_record(record: dict[str, object]) -> bool:
    return _schedule_name(record).startswith("evolved_")


def _is_revision_record(record: dict[str, object]) -> bool:
    return _schedule_name(record).startswith("evolved_revision_")


def _is_repair_record(record: dict[str, object]) -> bool:
    return bool(_repair_name(record))


def _is_repair_eligible_arm_record(record: dict[str, object]) -> bool:
    task = record.get("task")
    answer_type = task.get("answer_type") if isinstance(task, dict) else None
    candidate_key = str(record.get("candidate_key", "")).lower()
    model_id = str(record.get("model_id", "")).lower()
    if "llada" not in candidate_key and "llada" not in model_id:
        return False
    if answer_type == "rubric":
        return True
    return answer_type in {"integer", "multiple_choice", "short_text"} and _task_score(record) < 0.999


def _repair_records_for_source(
    records: list[dict[str, object]],
    source_record: dict[str, object],
) -> list[dict[str, object]]:
    source_control = _control_name(source_record)
    compatible = []
    for record in records:
        repair = record.get("repair")
        if not isinstance(repair, dict):
            continue
        repair_source = repair.get("source_control")
        if repair_source is None or str(repair_source) == source_control:
            compatible.append(record)
    return compatible


def _repair_records_for_sources(
    records: list[dict[str, object]],
    source_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    compatible = []
    seen = set()
    for source_record in source_records:
        for record in _repair_records_for_source(records, source_record):
            identity = _record_identity(record)
            if identity in seen:
                continue
            seen.add(identity)
            compatible.append(record)
    return compatible


def _exact_answer_repair_records_for_source(
    records: list[dict[str, object]],
    source_record: dict[str, object],
    *,
    limit: int,
    exact_verifier_revision: bool = False,
) -> list[dict[str, object]]:
    allowed_names = set(EXACT_ANSWER_REPAIR_NAMES)
    if not exact_verifier_revision:
        allowed_names.difference_update({"answer_span_repair", "answer_context_random_repair"})
    compatible = [
        record
        for record in _repair_records_for_source(records, source_record)
        if _repair_name(record) in allowed_names
    ]
    return compatible[:limit] if limit >= 0 else compatible


def _repair_records_for_candidates(
    records: list[dict[str, object]],
    candidates: tuple[Any, ...],
) -> list[dict[str, object]]:
    candidate_names = {str(candidate.name) for candidate in candidates}
    if not candidate_names:
        return []
    return [record for record in records if _repair_name(record) in candidate_names]


def _selected_evolved_records_for_rescore(
    records: list[dict[str, object]],
    *,
    limit_evolved_schedules: int,
    include_revision_schedules: bool,
) -> list[dict[str, object]]:
    evolved_records = [record for record in records if _is_evolved_record(record)]
    mutation_records = [record for record in evolved_records if not _is_revision_record(record)]
    revision_records = [record for record in evolved_records if _is_revision_record(record)]
    selected = _limit_records_by_schedule(mutation_records, limit_evolved_schedules)
    if include_revision_schedules:
        selected.extend(_limit_records_by_schedule(revision_records, None))
    return selected


def _limit_records_by_schedule(
    records: list[dict[str, object]],
    limit: int | None,
) -> list[dict[str, object]]:
    if limit is None:
        return records
    if limit <= 0:
        return []
    selected_names = []
    seen = set()
    for record in records:
        schedule_name = _schedule_name(record)
        if schedule_name in seen:
            continue
        seen.add(schedule_name)
        selected_names.append(schedule_name)
        if len(selected_names) >= limit:
            break
    allowed = set(selected_names)
    return [record for record in records if _schedule_name(record) in allowed]


def _limit_records_by_control(
    records: list[dict[str, object]],
    limit: int | None,
) -> list[dict[str, object]]:
    if limit is None:
        return records
    if limit <= 0:
        return []
    selected_names = []
    seen = set()
    for record in records:
        control_name = _control_name(record)
        if control_name in seen:
            continue
        seen.add(control_name)
        selected_names.append(control_name)
        if len(selected_names) >= limit:
            break
    allowed = set(selected_names)
    return [record for record in records if _control_name(record) in allowed]


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int) and not isinstance(item, bool)]


def _int_value(value: object, *, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _history_float(sample: dict[str, object] | None, key: str) -> float | None:
    if sample is None:
        return None
    value = sample.get(key)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _history_int(sample: dict[str, object] | None, key: str) -> int | None:
    if sample is None:
        return None
    value = sample.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _history_samples_token_ids(record: dict[str, object]) -> list[list[int]]:
    history_samples = record.get("history_samples")
    if not isinstance(history_samples, list):
        return []
    samples = []
    for sample in history_samples:
        if not isinstance(sample, dict):
            continue
        token_ids = _int_list(sample.get("generated_token_ids"))
        if token_ids:
            samples.append(token_ids)
    return samples


def _llada_mask_token_count(token_ids: list[int]) -> int:
    return sum(1 for token_id in token_ids if token_id == 126336)


def _float_or_none_list(value: object) -> list[float | None]:
    if not isinstance(value, list):
        return []
    items: list[float | None] = []
    for item in value:
        if item is None:
            items.append(None)
        elif isinstance(item, int | float) and not isinstance(item, bool):
            items.append(float(item))
    return items


def _masked_seed_count(value: object) -> int:
    if not isinstance(value, tuple | list):
        return 0
    return sum(1 for item in value if item is None)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object on {path}:{line_number}")
            records.append(record)
    return records


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _backend_token_decoder(backend: HFDiffusionBackend):
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        return None

    def decode(token_ids: list[int]) -> str:
        try:
            return tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode(token_ids, skip_special_tokens=True)

    return decode


def _backend_token_encoder(backend: HFDiffusionBackend):
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        return None

    def encode(text: str) -> list[int]:
        encode_fn = getattr(tokenizer, "encode", None)
        if callable(encode_fn):
            try:
                return list(encode_fn(text, add_special_tokens=False))
            except TypeError:
                return list(encode_fn(text))
        encoded = tokenizer(text, add_special_tokens=False)
        return list(encoded.get("input_ids", []))

    return encode


def _release_backend(backend: HFDiffusionBackend) -> None:
    del backend
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _compact_text(text: str, *, max_chars: int) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def _print_generation(record: dict[str, object]) -> None:
    print(
        f"{record['candidate_key']} {_task_id(record)} {_control_name(record)}: "
        f"task={_task_score(record):.3f} trajectory={_trajectory_score(record):.3f} "
        f"combined={record['combined_selection_score']:.3f}"
    )


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _float_csv(value: str) -> tuple[float, ...]:
    fractions = tuple(float(item) for item in _split_csv(value))
    for fraction in fractions:
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError("history repair fractions must be between 0 and 1")
    return fractions


def _positive_int_or_none(value: object) -> int | None:
    integer = _optional_int(value)
    if integer is None or integer <= 0:
        return None
    return integer


def _format_fraction_list(value: object) -> str:
    if not isinstance(value, list | tuple):
        return ""
    return ",".join(f"{float(item):.2f}" for item in value if isinstance(item, int | float))


def _format_optional_float(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float(value):.3f}"
    return str(value)


def _format_string_list(value: object) -> str:
    if not isinstance(value, list | tuple):
        return ""
    return ",".join(str(item) for item in value)


if __name__ == "__main__":
    raise SystemExit(main())
