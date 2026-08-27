"""Audit constraint-retention loss for diffusion denoise-history anchors."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_diffusion_three_arm_benchmark import (  # noqa: E402
    _choose_pre_generation_repair_anchor,
    _planning_constraint_gap_span_target_scores,
    _prompt_constraint_gap_terms,
    _selected_history_repair_sample,
    _target_tokens_missing_from_history,
)

DEFAULT_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_select_denoise_phase_gate_dense_history_fresh_v1_raw.jsonl"
)
DEFAULT_FINAL_SOURCE_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_fixed_source_denoise_phase_gate_fresh_v1_scores.json"
)
DEFAULT_HISTORY_ANCHOR_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_history_anchor_denoise_phase_gate_fresh_v1_scores.json"
)
DEFAULT_LOOSE_SEARCH_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_search_denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_GUARDED_SEARCH_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_search_guarded_denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_HISTORY_CONTRAST_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_history_contrast_denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_HISTORY_INSTABILITY_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_history_instability_denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_ANCHOR_INSTABILITY_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_ANCHOR_INSTABILITY_GATED_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_gated_identity_denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_ANCHOR_INSTABILITY_GATED_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_gated_identity_denoise_phase_gate_dense_history_fresh_v1_raw.jsonl"
)
DEFAULT_ANCHOR_INSTABILITY_PROMPT_GATED_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_prompt_gated_denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_ANCHOR_INSTABILITY_PROMPT_GATED_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_prompt_gated_denoise_phase_gate_dense_history_fresh_v1_raw.jsonl"
)
DEFAULT_ANCHOR_INSTABILITY_PROMPT_ONLY_GATED_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_prompt_only_gated_denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_ANCHOR_INSTABILITY_PROMPT_ONLY_GATED_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_prompt_only_gated_denoise_phase_gate_dense_history_fresh_v1_raw.jsonl"
)
DEFAULT_ANCHOR_INSTABILITY_CLAIM_GATED_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_gated_compact_prompt_denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_ANCHOR_INSTABILITY_CLAIM_GATED_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_gated_compact_prompt_denoise_phase_gate_dense_history_fresh_v1_raw.jsonl"
)
DEFAULT_ANCHOR_INSTABILITY_CLAIM_ORACLE_GATED_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_oracle_gated_denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_ANCHOR_INSTABILITY_CLAIM_ORACLE_GATED_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_oracle_gated_denoise_phase_gate_dense_history_fresh_v1_raw.jsonl"
)
DEFAULT_ANCHOR_INSTABILITY_CLAIM_SEEDED_GATED_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_seeded_gated_denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_ANCHOR_INSTABILITY_CLAIM_SEEDED_GATED_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_seeded_gated_denoise_phase_gate_dense_history_fresh_v1_raw.jsonl"
)
DEFAULT_ANCHOR_INSTABILITY_CLAIM_COMPATIBLE_SEEDED_GATED_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_compatible_seeded_gated_realization_guard_fresh_v1_scores.json"
)
DEFAULT_ANCHOR_INSTABILITY_CLAIM_COMPATIBLE_SEEDED_GATED_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_compatible_seeded_gated_realization_guard_fresh_v1_raw.jsonl"
)
DEFAULT_ANCHOR_INSTABILITY_CLAIM_AUTO_SEEDED_GATED_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_auto_seeded_gated_denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_ANCHOR_INSTABILITY_CLAIM_AUTO_SEEDED_GATED_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_auto_seeded_gated_denoise_phase_gate_dense_history_fresh_v1_raw.jsonl"
)
DEFAULT_ANCHOR_INSTABILITY_CLAIM_AUTO_SEEDED_REALIZATION_GATED_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_auto_seeded_realization_gated_denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_ANCHOR_INSTABILITY_CLAIM_AUTO_SEEDED_REALIZATION_GATED_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_auto_seeded_realization_gated_denoise_phase_gate_dense_history_fresh_v1_raw.jsonl"
)
DEFAULT_ANCHOR_INSTABILITY_CLAIM_STRICT_GATED_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_strict_gated_denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_ANCHOR_INSTABILITY_CLAIM_STRICT_GATED_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_strict_gated_denoise_phase_gate_dense_history_fresh_v1_raw.jsonl"
)
DEFAULT_JSON_OUTPUT = Path(
    "eval_results/diffusion_language/diffusion_anchor_retention_loss_audit.json"
)
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_ANCHOR_RETENTION_LOSS.md")
LOSS_FORMULA = (
    "(1 - target_similarity) + 0.25 * lost_target_token_count + "
    "0.75 * lost_prompt_keyword_count + 1.0 * lost_digit_token_count + "
    "max(0, 0.90 - history_to_final_char_ratio) + 0.25 * target_count_gap"
)
THEORY_STATEMENT = (
    "A denoise-history state is a useful latent repair anchor only when it keeps the "
    "constraints that survived in the final state while exposing a cleaner compact span "
    "to rewrite. The loss below turns that into an executable closed-loop criterion."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--final-source-scores", type=Path, default=DEFAULT_FINAL_SOURCE_SCORES)
    parser.add_argument("--history-anchor-scores", type=Path, default=DEFAULT_HISTORY_ANCHOR_SCORES)
    parser.add_argument("--loose-search-scores", type=Path, default=DEFAULT_LOOSE_SEARCH_SCORES)
    parser.add_argument("--guarded-search-scores", type=Path, default=DEFAULT_GUARDED_SEARCH_SCORES)
    parser.add_argument("--history-contrast-scores", type=Path, default=DEFAULT_HISTORY_CONTRAST_SCORES)
    parser.add_argument("--history-instability-scores", type=Path, default=DEFAULT_HISTORY_INSTABILITY_SCORES)
    parser.add_argument("--anchor-instability-scores", type=Path, default=DEFAULT_ANCHOR_INSTABILITY_SCORES)
    parser.add_argument(
        "--anchor-instability-gated-scores",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_GATED_SCORES,
    )
    parser.add_argument(
        "--anchor-instability-gated-raw",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_GATED_RAW,
    )
    parser.add_argument(
        "--anchor-instability-prompt-gated-scores",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_PROMPT_GATED_SCORES,
    )
    parser.add_argument(
        "--anchor-instability-prompt-gated-raw",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_PROMPT_GATED_RAW,
    )
    parser.add_argument(
        "--anchor-instability-prompt-only-gated-scores",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_PROMPT_ONLY_GATED_SCORES,
    )
    parser.add_argument(
        "--anchor-instability-prompt-only-gated-raw",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_PROMPT_ONLY_GATED_RAW,
    )
    parser.add_argument(
        "--anchor-instability-claim-gated-scores",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_CLAIM_GATED_SCORES,
    )
    parser.add_argument(
        "--anchor-instability-claim-gated-raw",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_CLAIM_GATED_RAW,
    )
    parser.add_argument(
        "--anchor-instability-claim-oracle-gated-scores",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_CLAIM_ORACLE_GATED_SCORES,
    )
    parser.add_argument(
        "--anchor-instability-claim-oracle-gated-raw",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_CLAIM_ORACLE_GATED_RAW,
    )
    parser.add_argument(
        "--anchor-instability-claim-seeded-gated-scores",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_CLAIM_SEEDED_GATED_SCORES,
    )
    parser.add_argument(
        "--anchor-instability-claim-seeded-gated-raw",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_CLAIM_SEEDED_GATED_RAW,
    )
    parser.add_argument(
        "--anchor-instability-claim-compatible-seeded-gated-scores",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_CLAIM_COMPATIBLE_SEEDED_GATED_SCORES,
    )
    parser.add_argument(
        "--anchor-instability-claim-compatible-seeded-gated-raw",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_CLAIM_COMPATIBLE_SEEDED_GATED_RAW,
    )
    parser.add_argument(
        "--anchor-instability-claim-auto-seeded-gated-scores",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_CLAIM_AUTO_SEEDED_GATED_SCORES,
    )
    parser.add_argument(
        "--anchor-instability-claim-auto-seeded-gated-raw",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_CLAIM_AUTO_SEEDED_GATED_RAW,
    )
    parser.add_argument(
        "--anchor-instability-claim-auto-seeded-realization-gated-scores",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_CLAIM_AUTO_SEEDED_REALIZATION_GATED_SCORES,
    )
    parser.add_argument(
        "--anchor-instability-claim-auto-seeded-realization-gated-raw",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_CLAIM_AUTO_SEEDED_REALIZATION_GATED_RAW,
    )
    parser.add_argument(
        "--anchor-instability-claim-strict-gated-scores",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_CLAIM_STRICT_GATED_SCORES,
    )
    parser.add_argument(
        "--anchor-instability-claim-strict-gated-raw",
        type=Path,
        default=DEFAULT_ANCHOR_INSTABILITY_CLAIM_STRICT_GATED_RAW,
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def build_anchor_retention_loss_audit(
    *,
    raw_path: Path,
    final_source_scores_path: Path | None = DEFAULT_FINAL_SOURCE_SCORES,
    history_anchor_scores_path: Path | None = DEFAULT_HISTORY_ANCHOR_SCORES,
    loose_search_scores_path: Path | None = DEFAULT_LOOSE_SEARCH_SCORES,
    guarded_search_scores_path: Path | None = DEFAULT_GUARDED_SEARCH_SCORES,
    history_contrast_scores_path: Path | None = DEFAULT_HISTORY_CONTRAST_SCORES,
    history_instability_scores_path: Path | None = DEFAULT_HISTORY_INSTABILITY_SCORES,
    anchor_instability_scores_path: Path | None = DEFAULT_ANCHOR_INSTABILITY_SCORES,
    anchor_instability_gated_scores_path: Path | None = DEFAULT_ANCHOR_INSTABILITY_GATED_SCORES,
    anchor_instability_gated_raw_path: Path | None = DEFAULT_ANCHOR_INSTABILITY_GATED_RAW,
    anchor_instability_prompt_gated_scores_path: Path | None = DEFAULT_ANCHOR_INSTABILITY_PROMPT_GATED_SCORES,
    anchor_instability_prompt_gated_raw_path: Path | None = DEFAULT_ANCHOR_INSTABILITY_PROMPT_GATED_RAW,
    anchor_instability_prompt_only_gated_scores_path: Path
    | None = DEFAULT_ANCHOR_INSTABILITY_PROMPT_ONLY_GATED_SCORES,
    anchor_instability_prompt_only_gated_raw_path: Path
    | None = DEFAULT_ANCHOR_INSTABILITY_PROMPT_ONLY_GATED_RAW,
    anchor_instability_claim_gated_scores_path: Path | None = DEFAULT_ANCHOR_INSTABILITY_CLAIM_GATED_SCORES,
    anchor_instability_claim_gated_raw_path: Path | None = DEFAULT_ANCHOR_INSTABILITY_CLAIM_GATED_RAW,
    anchor_instability_claim_oracle_gated_scores_path: Path
    | None = DEFAULT_ANCHOR_INSTABILITY_CLAIM_ORACLE_GATED_SCORES,
    anchor_instability_claim_oracle_gated_raw_path: Path
    | None = DEFAULT_ANCHOR_INSTABILITY_CLAIM_ORACLE_GATED_RAW,
    anchor_instability_claim_seeded_gated_scores_path: Path
    | None = DEFAULT_ANCHOR_INSTABILITY_CLAIM_SEEDED_GATED_SCORES,
    anchor_instability_claim_seeded_gated_raw_path: Path
    | None = DEFAULT_ANCHOR_INSTABILITY_CLAIM_SEEDED_GATED_RAW,
    anchor_instability_claim_compatible_seeded_gated_scores_path: Path
    | None = DEFAULT_ANCHOR_INSTABILITY_CLAIM_COMPATIBLE_SEEDED_GATED_SCORES,
    anchor_instability_claim_compatible_seeded_gated_raw_path: Path
    | None = DEFAULT_ANCHOR_INSTABILITY_CLAIM_COMPATIBLE_SEEDED_GATED_RAW,
    anchor_instability_claim_auto_seeded_gated_scores_path: Path
    | None = DEFAULT_ANCHOR_INSTABILITY_CLAIM_AUTO_SEEDED_GATED_SCORES,
    anchor_instability_claim_auto_seeded_gated_raw_path: Path
    | None = DEFAULT_ANCHOR_INSTABILITY_CLAIM_AUTO_SEEDED_GATED_RAW,
    anchor_instability_claim_auto_seeded_realization_gated_scores_path: Path
    | None = DEFAULT_ANCHOR_INSTABILITY_CLAIM_AUTO_SEEDED_REALIZATION_GATED_SCORES,
    anchor_instability_claim_auto_seeded_realization_gated_raw_path: Path
    | None = DEFAULT_ANCHOR_INSTABILITY_CLAIM_AUTO_SEEDED_REALIZATION_GATED_RAW,
    anchor_instability_claim_strict_gated_scores_path: Path
    | None = DEFAULT_ANCHOR_INSTABILITY_CLAIM_STRICT_GATED_SCORES,
    anchor_instability_claim_strict_gated_raw_path: Path
    | None = DEFAULT_ANCHOR_INSTABILITY_CLAIM_STRICT_GATED_RAW,
) -> dict[str, object]:
    final_rows = _score_rows_by_task(final_source_scores_path)
    history_rows = _score_rows_by_task(history_anchor_scores_path)
    rows = [_retention_row(record, final_rows, history_rows) for record in _candidate_rows(raw_path)]
    return {
        "final_source_scores_path": str(final_source_scores_path) if final_source_scores_path else "",
        "generated_by": "experiments/analyze_diffusion_anchor_retention_loss.py",
        "guarded_search_scores_path": str(guarded_search_scores_path) if guarded_search_scores_path else "",
        "history_anchor_scores_path": str(history_anchor_scores_path) if history_anchor_scores_path else "",
        "history_contrast_scores_path": str(history_contrast_scores_path) if history_contrast_scores_path else "",
        "history_instability_scores_path": str(history_instability_scores_path)
        if history_instability_scores_path
        else "",
        "anchor_instability_scores_path": str(anchor_instability_scores_path)
        if anchor_instability_scores_path
        else "",
        "anchor_instability_gated_scores_path": str(anchor_instability_gated_scores_path)
        if anchor_instability_gated_scores_path
        else "",
        "anchor_instability_gated_raw_path": str(anchor_instability_gated_raw_path)
        if anchor_instability_gated_raw_path
        else "",
        "anchor_instability_prompt_gated_scores_path": str(anchor_instability_prompt_gated_scores_path)
        if anchor_instability_prompt_gated_scores_path
        else "",
        "anchor_instability_prompt_gated_raw_path": str(anchor_instability_prompt_gated_raw_path)
        if anchor_instability_prompt_gated_raw_path
        else "",
        "anchor_instability_prompt_only_gated_scores_path": str(
            anchor_instability_prompt_only_gated_scores_path
        )
        if anchor_instability_prompt_only_gated_scores_path
        else "",
        "anchor_instability_prompt_only_gated_raw_path": str(
            anchor_instability_prompt_only_gated_raw_path
        )
        if anchor_instability_prompt_only_gated_raw_path
        else "",
        "anchor_instability_claim_gated_scores_path": str(anchor_instability_claim_gated_scores_path)
        if anchor_instability_claim_gated_scores_path
        else "",
        "anchor_instability_claim_gated_raw_path": str(anchor_instability_claim_gated_raw_path)
        if anchor_instability_claim_gated_raw_path
        else "",
        "anchor_instability_claim_oracle_gated_scores_path": str(
            anchor_instability_claim_oracle_gated_scores_path
        )
        if anchor_instability_claim_oracle_gated_scores_path
        else "",
        "anchor_instability_claim_oracle_gated_raw_path": str(anchor_instability_claim_oracle_gated_raw_path)
        if anchor_instability_claim_oracle_gated_raw_path
        else "",
        "anchor_instability_claim_seeded_gated_scores_path": str(
            anchor_instability_claim_seeded_gated_scores_path
        )
        if anchor_instability_claim_seeded_gated_scores_path
        else "",
        "anchor_instability_claim_seeded_gated_raw_path": str(anchor_instability_claim_seeded_gated_raw_path)
        if anchor_instability_claim_seeded_gated_raw_path
        else "",
        "anchor_instability_claim_compatible_seeded_gated_scores_path": str(
            anchor_instability_claim_compatible_seeded_gated_scores_path
        )
        if anchor_instability_claim_compatible_seeded_gated_scores_path
        else "",
        "anchor_instability_claim_compatible_seeded_gated_raw_path": str(
            anchor_instability_claim_compatible_seeded_gated_raw_path
        )
        if anchor_instability_claim_compatible_seeded_gated_raw_path
        else "",
        "anchor_instability_claim_auto_seeded_gated_scores_path": str(
            anchor_instability_claim_auto_seeded_gated_scores_path
        )
        if anchor_instability_claim_auto_seeded_gated_scores_path
        else "",
        "anchor_instability_claim_auto_seeded_gated_raw_path": str(
            anchor_instability_claim_auto_seeded_gated_raw_path
        )
        if anchor_instability_claim_auto_seeded_gated_raw_path
        else "",
        "anchor_instability_claim_auto_seeded_realization_gated_scores_path": str(
            anchor_instability_claim_auto_seeded_realization_gated_scores_path
        )
        if anchor_instability_claim_auto_seeded_realization_gated_scores_path
        else "",
        "anchor_instability_claim_auto_seeded_realization_gated_raw_path": str(
            anchor_instability_claim_auto_seeded_realization_gated_raw_path
        )
        if anchor_instability_claim_auto_seeded_realization_gated_raw_path
        else "",
        "anchor_instability_claim_strict_gated_scores_path": str(
            anchor_instability_claim_strict_gated_scores_path
        )
        if anchor_instability_claim_strict_gated_scores_path
        else "",
        "anchor_instability_claim_strict_gated_raw_path": str(anchor_instability_claim_strict_gated_raw_path)
        if anchor_instability_claim_strict_gated_raw_path
        else "",
        "loose_search_scores_path": str(loose_search_scores_path) if loose_search_scores_path else "",
        "loss_formula": LOSS_FORMULA,
        "raw_path": str(raw_path),
        "rows": rows,
        "schema": "diffusion_anchor_retention_loss_audit.v1",
        "summary": _summary(rows),
        "whole_history_search": {
            "guarded": _benchmark_summary(guarded_search_scores_path),
            "loose": _benchmark_summary(loose_search_scores_path),
        },
        "trajectory_contrast": _benchmark_summary(history_contrast_scores_path),
        "trajectory_instability": _benchmark_summary(history_instability_scores_path),
        "anchor_instability": _benchmark_summary(anchor_instability_scores_path),
        "anchor_instability_gated": _benchmark_summary(anchor_instability_gated_scores_path),
        "anchor_instability_gated_identity": _gated_identity_summary(raw_path, anchor_instability_gated_raw_path),
        "anchor_instability_prompt_gated": _benchmark_summary(anchor_instability_prompt_gated_scores_path),
        "anchor_instability_prompt_gated_identity": _gated_identity_summary(
            raw_path,
            anchor_instability_prompt_gated_raw_path,
        ),
        "anchor_instability_prompt_only_gated": _benchmark_summary(
            anchor_instability_prompt_only_gated_scores_path
        ),
        "anchor_instability_prompt_only_gated_identity": _gated_identity_summary(
            raw_path,
            anchor_instability_prompt_only_gated_raw_path,
        ),
        "anchor_instability_claim_gated": _benchmark_summary(anchor_instability_claim_gated_scores_path),
        "anchor_instability_claim_gated_identity": _composite_gate_identity_summary(
            anchor_instability_prompt_gated_raw_path,
            anchor_instability_claim_gated_raw_path,
        ),
        "anchor_instability_claim_oracle_gated": _benchmark_summary(
            anchor_instability_claim_oracle_gated_scores_path
        ),
        "anchor_instability_claim_oracle_gated_identity": _composite_gate_identity_summary(
            anchor_instability_claim_gated_raw_path,
            anchor_instability_claim_oracle_gated_raw_path,
        ),
        "anchor_instability_claim_seeded_gated": _benchmark_summary(
            anchor_instability_claim_seeded_gated_scores_path
        ),
        "anchor_instability_claim_seeded_gated_identity": _composite_gate_identity_summary(
            anchor_instability_claim_oracle_gated_raw_path,
            anchor_instability_claim_seeded_gated_raw_path,
        ),
        "anchor_instability_claim_compatible_seeded_gated": _benchmark_summary(
            anchor_instability_claim_compatible_seeded_gated_scores_path
        ),
        "anchor_instability_claim_compatible_seeded_gated_identity": _composite_gate_identity_summary(
            anchor_instability_claim_seeded_gated_raw_path,
            anchor_instability_claim_compatible_seeded_gated_raw_path,
        ),
        "anchor_instability_claim_auto_seeded_gated": _benchmark_summary(
            anchor_instability_claim_auto_seeded_gated_scores_path
        ),
        "anchor_instability_claim_auto_seeded_gated_identity": _composite_gate_identity_summary(
            anchor_instability_claim_compatible_seeded_gated_raw_path,
            anchor_instability_claim_auto_seeded_gated_raw_path,
        ),
        "anchor_instability_claim_auto_seeded_realization_gated": _benchmark_summary(
            anchor_instability_claim_auto_seeded_realization_gated_scores_path
        ),
        "anchor_instability_claim_auto_seeded_realization_gated_identity": _composite_gate_identity_summary(
            anchor_instability_claim_auto_seeded_gated_raw_path,
            anchor_instability_claim_auto_seeded_realization_gated_raw_path,
        ),
        "anchor_instability_claim_strict_gated": _benchmark_summary(
            anchor_instability_claim_strict_gated_scores_path
        ),
        "anchor_instability_claim_strict_gated_identity": _composite_gate_identity_summary(
            anchor_instability_claim_gated_raw_path,
            anchor_instability_claim_strict_gated_raw_path,
        ),
        "theory_statement": THEORY_STATEMENT,
    }


def render_markdown(audit: dict[str, object]) -> str:
    summary = _dict(audit.get("summary"))
    rows = _list_of_dicts(audit.get("rows"))
    lines = [
        "# Diffusion Anchor Retention Loss",
        "",
        "This file is generated by `experiments/analyze_diffusion_anchor_retention_loss.py`.",
        THEORY_STATEMENT,
        "",
        "## Summary",
        "",
        f"- Raw trace: `{audit.get('raw_path', '')}`",
        f"- Final-source scores: `{audit.get('final_source_scores_path', '')}`",
        f"- History-anchor scores: `{audit.get('history_anchor_scores_path', '')}`",
        f"- Loose search scores: `{audit.get('loose_search_scores_path', '')}`",
        f"- Guarded search scores: `{audit.get('guarded_search_scores_path', '')}`",
        f"- History-contrast scores: `{audit.get('history_contrast_scores_path', '')}`",
        f"- History-instability scores: `{audit.get('history_instability_scores_path', '')}`",
        f"- Anchor-instability scores: `{audit.get('anchor_instability_scores_path', '')}`",
        f"- Gated anchor-instability scores: `{audit.get('anchor_instability_gated_scores_path', '')}`",
        f"- Gated anchor-instability raw: `{audit.get('anchor_instability_gated_raw_path', '')}`",
        (
            "- Prompt-gated anchor-instability scores: "
            f"`{audit.get('anchor_instability_prompt_gated_scores_path', '')}`"
        ),
        f"- Prompt-gated anchor-instability raw: `{audit.get('anchor_instability_prompt_gated_raw_path', '')}`",
        (
            "- Prompt-only gated anchor-instability scores: "
            f"`{audit.get('anchor_instability_prompt_only_gated_scores_path', '')}`"
        ),
        (
            "- Prompt-only gated anchor-instability raw: "
            f"`{audit.get('anchor_instability_prompt_only_gated_raw_path', '')}`"
        ),
        (
            "- Claim-gated anchor-instability scores: "
            f"`{audit.get('anchor_instability_claim_gated_scores_path', '')}`"
        ),
        f"- Claim-gated anchor-instability raw: `{audit.get('anchor_instability_claim_gated_raw_path', '')}`",
        (
            "- Claim-oracle gated anchor-instability scores: "
            f"`{audit.get('anchor_instability_claim_oracle_gated_scores_path', '')}`"
        ),
        (
            "- Claim-oracle gated anchor-instability raw: "
            f"`{audit.get('anchor_instability_claim_oracle_gated_raw_path', '')}`"
        ),
        (
            "- Claim-seeded gated anchor-instability scores: "
            f"`{audit.get('anchor_instability_claim_seeded_gated_scores_path', '')}`"
        ),
        (
            "- Claim-seeded gated anchor-instability raw: "
            f"`{audit.get('anchor_instability_claim_seeded_gated_raw_path', '')}`"
        ),
        (
            "- Claim-compatible-seeded gated anchor-instability scores: "
            f"`{audit.get('anchor_instability_claim_compatible_seeded_gated_scores_path', '')}`"
        ),
        (
            "- Claim-compatible-seeded gated anchor-instability raw: "
            f"`{audit.get('anchor_instability_claim_compatible_seeded_gated_raw_path', '')}`"
        ),
        (
            "- Claim-auto-seeded gated anchor-instability scores: "
            f"`{audit.get('anchor_instability_claim_auto_seeded_gated_scores_path', '')}`"
        ),
        (
            "- Claim-auto-seeded gated anchor-instability raw: "
            f"`{audit.get('anchor_instability_claim_auto_seeded_gated_raw_path', '')}`"
        ),
        (
            "- Claim-auto-seeded realization-gated anchor-instability scores: "
            f"`{audit.get('anchor_instability_claim_auto_seeded_realization_gated_scores_path', '')}`"
        ),
        (
            "- Claim-auto-seeded realization-gated anchor-instability raw: "
            f"`{audit.get('anchor_instability_claim_auto_seeded_realization_gated_raw_path', '')}`"
        ),
        (
            "- Claim-strict gated anchor-instability scores: "
            f"`{audit.get('anchor_instability_claim_strict_gated_scores_path', '')}`"
        ),
        (
            "- Claim-strict gated anchor-instability raw: "
            f"`{audit.get('anchor_instability_claim_strict_gated_raw_path', '')}`"
        ),
        f"- Planning rows: `{summary.get('row_count', 0)}`",
        f"- Classifications: `{summary.get('classification_counts', {})}`",
        f"- Anchor choices: `{summary.get('anchor_choice_counts', {})}`",
        f"- Mean retention loss: `{_format_float(summary.get('mean_retention_loss'))}`",
        f"- Mean loss, safe history anchors: `{_format_float(summary.get('mean_loss_safe_history_anchor'))}`",
        f"- Mean loss, blocked history anchors: `{_format_float(summary.get('mean_loss_blocked_history_anchor'))}`",
        f"- Mean history-minus-final repair score: `{_format_float(summary.get('mean_history_minus_final_repair_score'))}`",
        f"- Loss formula: `{audit.get('loss_formula', '')}`",
        "",
        "## Whole-History Search Check",
        "",
        (
            "| Policy | Run | Repair Score | Delta vs Greedy | Delta vs Random | "
            "Relative Cost | Source States |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        *_search_summary_rows(_dict(audit.get("whole_history_search"))),
        "",
        "## Prompt-Only Contrast Check",
        "",
        (
            "| Policy | Run | Repair Score | Delta vs Greedy | Delta vs Random | "
            "Relative Cost | Selected Repairs |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        *_contrast_summary_rows(_dict(audit.get("trajectory_contrast"))),
        "",
        "## Seed/Remask Geometry Check",
        "",
        (
            "| Policy | Run | Repair Score | Delta vs Greedy | Delta vs Random | "
            "Relative Cost | Selected Repairs |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        *_single_policy_summary_rows(
            _dict(audit.get("trajectory_instability")),
            label="history instability remask",
        ),
        *_single_policy_summary_rows(
            _dict(audit.get("anchor_instability")),
            label="anchor-selected instability remask",
        ),
        *_single_policy_summary_rows(
            _dict(audit.get("anchor_instability_gated")),
            label="gated anchor instability remask",
        ),
        *_single_policy_summary_rows(
            _dict(audit.get("anchor_instability_prompt_gated")),
            label="prompt-gated anchor instability remask",
        ),
        *_single_policy_summary_rows(
            _dict(audit.get("anchor_instability_prompt_only_gated")),
            label="prompt-only gated anchor instability",
        ),
        *_single_policy_summary_rows(
            _dict(audit.get("anchor_instability_claim_gated")),
            label="claim-gated anchor instability remask",
        ),
        *_single_policy_summary_rows(
            _dict(audit.get("anchor_instability_claim_oracle_gated")),
            label="claim-oracle gated anchor instability",
        ),
        *_single_policy_summary_rows(
            _dict(audit.get("anchor_instability_claim_seeded_gated")),
            label="claim-seeded gated anchor instability",
        ),
        *_single_policy_summary_rows(
            _dict(audit.get("anchor_instability_claim_compatible_seeded_gated")),
            label="claim-compatible-seeded gated anchor instability",
        ),
        *_single_policy_summary_rows(
            _dict(audit.get("anchor_instability_claim_auto_seeded_gated")),
            label="claim-auto-seeded gated anchor instability",
        ),
        *_single_policy_summary_rows(
            _dict(audit.get("anchor_instability_claim_auto_seeded_realization_gated")),
            label="claim-auto-seeded realization-gated anchor instability",
        ),
        *_single_policy_summary_rows(
            _dict(audit.get("anchor_instability_claim_strict_gated")),
            label="claim-strict gated anchor instability",
        ),
        "",
        "## Gated Identity Check",
        "",
        (
            "| Policy | Compared Tasks | Gate-Off Tasks | Gate-Off Identity Matches | Gate-On Tasks | "
            "Gate-On Mean Score Delta |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *_gated_identity_summary_rows(
            _dict(audit.get("anchor_instability_gated_identity")),
            label="mask-only gate",
        ),
        *_gated_identity_summary_rows(
            _dict(audit.get("anchor_instability_prompt_gated_identity")),
            label="mask+prompt gate",
        ),
        *_gated_identity_summary_rows(
            _dict(audit.get("anchor_instability_prompt_only_gated_identity")),
            label="prompt-only gate",
        ),
        *_gated_identity_summary_rows(
            _dict(audit.get("anchor_instability_claim_gated_identity")),
            label="claim+instability gate",
        ),
        *_gated_identity_summary_rows(
            _dict(audit.get("anchor_instability_claim_oracle_gated_identity")),
            label="claim-oracle gate",
        ),
        *_gated_identity_summary_rows(
            _dict(audit.get("anchor_instability_claim_seeded_gated_identity")),
            label="claim-seeded gate",
        ),
        *_gated_identity_summary_rows(
            _dict(audit.get("anchor_instability_claim_compatible_seeded_gated_identity")),
            label="claim-compatible-seeded gate",
        ),
        *_gated_identity_summary_rows(
            _dict(audit.get("anchor_instability_claim_auto_seeded_gated_identity")),
            label="claim-auto-seeded gate",
        ),
        *_gated_identity_summary_rows(
            _dict(audit.get("anchor_instability_claim_auto_seeded_realization_gated_identity")),
            label="claim-auto-seeded realization gate",
        ),
        *_gated_identity_summary_rows(
            _dict(audit.get("anchor_instability_claim_strict_gated_identity")),
            label="claim-strict gate",
        ),
        "",
        "## Interpretation",
        "",
        (
            "This is the first executable theory artifact that treats diffusion denoising "
            "as an error-correction loop rather than a bag of text samples. The loss is "
            "label-free: it looks at constraint retention, target overlap, digit retention, "
            "prompt-keyword retention, and compact-span consistency before any repair "
            "generation is spent."
        ),
        (
            "`plan_001` is allowed because its history anchor keeps the prompt-critical "
            "constraints and has a small positive compact-span score advantage. The other "
            "planning histories are blocked because they drop too much target/context "
            "information or fail to beat the final source on pre-generation span geometry."
        ),
        (
            "The loose whole-history search was an important failure mode: it selected an "
            "earlier `plan_003` history state that looked good under the older loose "
            "thresholds, but the fresh GPU run dropped the repair average. The guarded "
            "search therefore requires near-final target similarity and length retention "
            "before a history state can replace the final source."
        ),
        (
            "Prompt-only trajectory contrast is also a boundary result: adding a compact "
            "near-final history snippet to the final-source span repair prompt did not "
            "produce selected improvements. The useful trajectory signal has to alter "
            "the seed/remask geometry or anchor selection, not merely append evidence to "
            "the prompt."
        ),
        (
            "History-instability remasking is a stronger boundary than prompt-only "
            "contrast because it changes the actual repair seed. It still trails the "
            "anchor-select policy, so instability should be treated as a secondary "
            "mask feature inside anchor-selected repair, not as a replacement for "
            "constraint-retention anchor choice."
        ),
        (
            "The first anchor-plus-instability run confirms that blindly combining "
            "the two mechanisms is still weaker than anchor choice alone. It improves "
            "over standalone instability, but most instability masks reduce the "
            "anchor-selected repair; this points to a conditional instability gate "
            "rather than unconditional union masks."
        ),
        (
            "The fixed conditional gate is now an identity-stable A/B harness: "
            "gate-off repairs match anchor-select in generation seed, prompt, masked "
            "seed, output text, and score. The only active gate changes the seed and "
            "text on `plan_007`, but ties the anchor-select score, so instability "
            "masking is isolated and currently non-improving rather than a promoted "
            "mechanism."
        ),
        (
            "The prompt-gated version keeps that gate-off identity but adds the "
            "instability-specific instruction only when the gate is active. That "
            "single active branch lifts `plan_007` by `0.067143`, moving the public "
            "three-arm planning line to `0.498304` at the same `2.625000x` cost."
        ),
        (
            "The prompt-only gated control preserves the same gate-off identity but "
            "removes the active instability remask. Fresh run "
            "`diffusion-4b5fc2b7604c28a5` falls to `0.479911`, and the active "
            "`plan_007` branch drops by `-0.080000` versus anchor-select. That "
            "makes it a negative control: the positive result needs the denoise "
            "instability mask plus the prompt, not the prompt route alone."
        ),
        (
            "The claim-gated composite adds a second active gate for public-claim "
            "confound tasks without disturbing the existing `plan_007` instability "
            "branch. Fresh run `diffusion-0fc7f067a7d87799` lifts `plan_004` by "
            "`0.121071` versus the prompt-gated frontier and moves the public "
            "three-arm planning line to `0.513438` at the same `2.625000x` cost."
        ),
        (
            "The claim-oracle gated composite keeps the same denoise-anchor and "
            "instability-mask geometry but uses a compact public-claim control "
            "instruction that emphasizes failure-mode validation and selected/oracle "
            "result separation. Fresh run `diffusion-692592da063daa60` moves the "
            "public three-arm planning line to `0.523304` at the same `2.625000x` "
            "cost, with zero repair-oracle headroom. The active `plan_004` branch "
            "rises to `0.559286` task score, although it still misses the explicit "
            "oracle-result rubric phrase; the lift currently comes from better "
            "risk/failure-mode coverage, not from fully solving that wording gap."
        ),
        (
            "The claim-seeded control fixes a short oracle/selected-results phrase "
            "directly into the masked denoise seed. Fresh run "
            "`diffusion-6ae167dc85d5e6ac` proves the mechanism can bind that literal "
            "rubric item: `plan_004` includes `separate oracle best-of results from "
            "selected results`. But the run falls to `0.521295` at `2.625000x` "
            "because the fixed anchor displaces the public-claim survival control. "
            "The boundary is useful: hard semantic anchors are stronger than prompts "
            "for phrase binding, but they need a compatibility loss so one fixed "
            "control does not crowd out another."
        ),
        (
            "The compatible-seeded control is the first positive version of that "
            "compatibility idea. It fixes a 9-token seed that carries both "
            "`oracle selected results` and `claim survives` in the same masked tail. "
            "The fresh realization-guarded run `diffusion-a9ae901393235364` "
            "preserves the public three-arm planning line at `0.531116` and "
            "`2.625000x` cost under the seed-realization selector; `plan_004` "
            "hits all five rubric controls and reaches `0.621786` task score. "
            "The theory update is concrete: semantic seed anchors can be useful "
            "when the anchor is compact enough to preserve compatibility between "
            "the required controls and direct enough to pass realization quality."
        ),
        (
            "The auto-seeded control is a partial generalization and a useful "
            "negative boundary. It synthesizes the same compact seed from the "
            "active task/rubric surface, and fresh run "
            "`diffusion-7b74493b8c5ca15a` confirms the generated anchor is applied "
            "without truncation and `plan_004` hits all five rubric controls. The "
            "aggregate line is only `0.520536` at `2.625000x`, however, because "
            "the automatically generated continuation is less direct than the fixed "
            "compatible seed. Mechanically extracting the right control terms is "
            "not enough; the next learned policy needs a realization-quality term "
            "for how the anchor integrates into the denoised sentence."
        ),
        (
            "The auto-seeded realization-gated follow-up tested whether stronger "
            "explicit realization constraints would fix that integration gap. Fresh "
            "run `diffusion-2a310ed45712a36b` fell further to `0.515759` at "
            "`2.625000x`: `plan_004` still hits all rubric controls, but the model "
            "turns the answer into a `Control:` label with low specificity. This "
            "rules out simply adding more prompt constraints. The needed loss is a "
            "scored realization-quality term that rewards natural, direct integration "
            "of the compact seed into the falsification plan."
        ),
        (
            "The claim-strict gated control is a negative boundary: forcing an "
            "explicit oracle/best-of separation into the repair instruction did "
            "not improve the frontier. Fresh run `diffusion-df4149f37f6b21bf` "
            "scored `0.495625` at `2.625000x`, below the compact claim-gated "
            "`0.513438`; `plan_004` fell to `0.355000` task score because the "
            "repair over-compressed the plan instead of adding the missing "
            "rerun/regression/oracle controls. The useful theory signal is "
            "therefore selective geometry-conditioned control, not simply more "
            "explicit prompt obligations."
        ),
        "",
        "## Task Table",
        "",
        (
            "| Task | Class | Anchor | Loss | Target Sim | Text Sim | Lost Target | "
            "Lost Prompt | Lost Digits | Span Delta | History-Final Repair | Reason |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('task_id', '')} | "
            f"{row.get('classification', '')} | "
            f"{row.get('anchor_choice', '')} | "
            f"{_format_float(row.get('constraint_retention_loss'))} | "
            f"{_format_float(row.get('target_similarity'))} | "
            f"{_format_float(row.get('text_similarity'))} | "
            f"{row.get('lost_target_token_count', 0)} | "
            f"{row.get('lost_prompt_keyword_count', 0)} | "
            f"{row.get('lost_digit_token_count', 0)} | "
            f"{_format_float(row.get('history_span_score_delta'))} | "
            f"{_format_float(row.get('history_minus_final_repair_score'))} | "
            f"{row.get('anchor_selection_reason', '')} |"
        )
    return "\n".join(lines) + "\n"


def _retention_row(
    record: dict[str, object],
    final_rows: dict[str, dict[str, object]],
    history_rows: dict[str, dict[str, object]],
) -> dict[str, object]:
    task_id = _task_id(record)
    prompt = str(record.get("prompt", ""))
    final_text = str(record.get("text", ""))
    history_sample = _selected_history_repair_sample(record, prompt)
    anchor = _choose_pre_generation_repair_anchor(record, prompt)
    anchor_choice = str(anchor.get("anchor_choice", "final"))
    features = _dict(anchor.get("features"))
    full_lost_target_tokens: list[str] = []
    if history_sample and final_text.strip() and prompt.strip():
        history_text = str(history_sample.get("visible_text", ""))
        final_targets = _compact_targets(prompt, final_text)
        history_targets = _compact_targets(prompt, history_text)
        full_lost_target_tokens = _target_tokens_missing_from_history(
            " ".join(str(target.get("span", "")) for target in final_targets),
            " ".join(str(target.get("span", "")) for target in history_targets),
        )
    loss_parts = _constraint_retention_loss_parts(features, full_lost_target_tokens)
    classification = _classification(
        features=features,
        has_history=history_sample is not None,
        anchor_choice=anchor_choice,
        reason=str(anchor.get("reason", "")),
    )
    final_score = _float(final_rows.get(task_id, {}).get("repair_task_score"))
    history_score = _float(history_rows.get(task_id, {}).get("repair_task_score"))
    return {
        "anchor_choice": anchor_choice,
        "anchor_selection_features": features,
        "anchor_selection_reason": str(anchor.get("reason", "")),
        "classification": classification,
        "constraint_retention_loss": loss_parts["loss"],
        "final_repair_score": final_score,
        "history_minus_final_repair_score": history_score - final_score
        if task_id in final_rows and task_id in history_rows
        else 0.0,
        "history_repair_score": history_score,
        "history_span_score_delta": _float(features.get("history_span_score_delta")),
        "history_to_final_char_ratio": _float(features.get("history_to_final_char_ratio")),
        "lost_digit_token_count": int(loss_parts["lost_digit_token_count"]),
        "lost_prompt_keyword_count": int(loss_parts["lost_prompt_keyword_count"]),
        "lost_target_token_count": int(loss_parts["lost_target_token_count"]),
        "lost_target_tokens": full_lost_target_tokens[:12],
        "target_count_gap": int(loss_parts["target_count_gap"]),
        "target_similarity": _float(features.get("target_similarity")),
        "task_id": task_id,
        "text_similarity": _float(features.get("text_similarity")),
    }


def _compact_targets(prompt: str, text: str) -> list[dict[str, object]]:
    gaps = _prompt_constraint_gap_terms(prompt, text)
    return _planning_constraint_gap_span_target_scores(
        prompt,
        text,
        gaps,
        chunk_mode="adaptive",
        selection_policy="compact",
    )


def _constraint_retention_loss_parts(
    features: dict[str, object],
    full_lost_target_tokens: list[str],
) -> dict[str, float]:
    target_similarity = _float(features.get("target_similarity"))
    char_ratio = _float(features.get("history_to_final_char_ratio"))
    lost_target_token_count = len(full_lost_target_tokens or _list(features.get("lost_target_tokens")))
    lost_prompt_keyword_count = _float(features.get("lost_prompt_keyword_count"))
    lost_digit_token_count = _float(features.get("lost_digit_token_count"))
    target_count_gap = abs(
        int(_float(features.get("history_target_count")))
        - int(_float(features.get("final_target_count")))
    )
    loss = (
        (1.0 - target_similarity)
        + 0.25 * lost_target_token_count
        + 0.75 * lost_prompt_keyword_count
        + lost_digit_token_count
        + max(0.0, 0.90 - char_ratio)
        + 0.25 * target_count_gap
    )
    return {
        "loss": round(loss, 6),
        "lost_digit_token_count": lost_digit_token_count,
        "lost_prompt_keyword_count": lost_prompt_keyword_count,
        "lost_target_token_count": float(lost_target_token_count),
        "target_count_gap": float(target_count_gap),
    }


def _classification(
    *,
    features: dict[str, object],
    has_history: bool,
    anchor_choice: str,
    reason: str,
) -> str:
    if not has_history:
        return "no_history_anchor"
    if anchor_choice == "history":
        return "safe_history_anchor"
    if reason == "final_source_preserves_more_context":
        if _float(features.get("history_span_score_delta")) <= 1e-6:
            return "span_advantage_blocks_history"
        if int(_float(features.get("history_target_count"))) != 1:
            return "compact_target_blocks_history"
        return "retention_loss_blocks_history"
    return "history_anchor_blocked"


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    safe_rows = [row for row in rows if row.get("classification") == "safe_history_anchor"]
    blocked_rows = [
        row
        for row in rows
        if row.get("classification") not in {"safe_history_anchor", "no_history_anchor"}
    ]
    score_delta_rows = [
        row
        for row in rows
        if row.get("final_repair_score", 0.0) or row.get("history_repair_score", 0.0)
    ]
    return {
        "anchor_choice_counts": dict(Counter(str(row.get("anchor_choice", "")) for row in rows)),
        "blocked_history_anchor_count": len(blocked_rows),
        "classification_counts": dict(Counter(str(row.get("classification", "")) for row in rows)),
        "loss_formula": LOSS_FORMULA,
        "mean_history_minus_final_repair_score": _mean(
            _float(row.get("history_minus_final_repair_score")) for row in score_delta_rows
        ),
        "mean_loss_blocked_history_anchor": _mean(
            _float(row.get("constraint_retention_loss")) for row in blocked_rows
        ),
        "mean_loss_safe_history_anchor": _mean(
            _float(row.get("constraint_retention_loss")) for row in safe_rows
        ),
        "mean_retention_loss": _mean(
            _float(row.get("constraint_retention_loss")) for row in rows
        ),
        "row_count": len(rows),
        "safe_history_anchor_count": len(safe_rows),
    }


def _benchmark_summary(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    scores = _read_json(path)
    planning = _dict(_dict(scores.get("by_family_arm")).get("planning"))
    repair = _dict(planning.get("repair_selected"))
    fixed = _dict(planning.get("fixed"))
    random_arm = _dict(planning.get("random"))
    candidate_summary = _first_candidate_summary(scores)
    return {
        "delta_vs_fixed": _float(repair.get("mean_task_score")) - _float(fixed.get("mean_task_score")),
        "delta_vs_random": _float(repair.get("mean_task_score")) - _float(random_arm.get("mean_task_score")),
        "relative_cost": _float(repair.get("mean_generation_budget_per_task")),
        "repair_score": _float(repair.get("mean_task_score")),
        "run_id": str(scores.get("run_id", "")),
        "selected_count": _float(candidate_summary.get("selected_count")),
        "source_states": str(candidate_summary.get("source_states", "")),
    }


def _search_summary_rows(search: dict[str, object]) -> list[str]:
    rows = []
    for label in ("loose", "guarded"):
        summary = _dict(search.get(label))
        if not summary:
            continue
        rows.append(
            "| "
            f"{label} | "
            f"`{summary.get('run_id', '')}` | "
            f"{_format_float(summary.get('repair_score'))} | "
            f"{_format_float(summary.get('delta_vs_fixed'))} | "
            f"{_format_float(summary.get('delta_vs_random'))} | "
            f"{_format_float(summary.get('relative_cost'))} | "
            f"`{summary.get('source_states', '')}` |"
        )
    return rows


def _contrast_summary_rows(summary: dict[str, object]) -> list[str]:
    return _single_policy_summary_rows(summary, label="history contrast")


def _single_policy_summary_rows(summary: dict[str, object], *, label: str) -> list[str]:
    if not summary:
        return []
    return [
        "| "
        f"{label} | "
        f"`{summary.get('run_id', '')}` | "
        f"{_format_float(summary.get('repair_score'))} | "
        f"{_format_float(summary.get('delta_vs_fixed'))} | "
        f"{_format_float(summary.get('delta_vs_random'))} | "
        f"{_format_float(summary.get('relative_cost'))} | "
        f"{int(_float(summary.get('selected_count')))} |"
    ]


def _gated_identity_summary_rows(summary: dict[str, object], *, label: str) -> list[str]:
    if not summary:
        return []
    return [
        "| "
        f"{label} | "
        f"{int(_float(summary.get('compared_task_count')))} | "
        f"{int(_float(summary.get('gate_inactive_count')))} | "
        f"{int(_float(summary.get('gate_inactive_identity_match_count')))} | "
        f"{int(_float(summary.get('gate_active_count')))} | "
        f"{_format_float(summary.get('gate_active_mean_task_score_delta'))} |"
    ]


def _gated_identity_summary(anchor_raw_path: Path, gated_raw_path: Path | None) -> dict[str, object]:
    if gated_raw_path is None or not gated_raw_path.exists() or not anchor_raw_path.exists():
        return {}
    anchor_rows = _repair_rows_by_task(anchor_raw_path)
    gated_rows = _repair_rows_by_task(gated_raw_path)
    rows = []
    for task_id in sorted(set(anchor_rows) & set(gated_rows)):
        anchor = anchor_rows[task_id]
        gated = gated_rows[task_id]
        repair = _dict(gated.get("repair"))
        seed_same = gated.get("generation_seed") == anchor.get("generation_seed")
        prompt_same = gated.get("prompt") == anchor.get("prompt")
        seed_tokens_same = _dict(gated.get("config")).get("initial_suffix_token_ids") == _dict(
            anchor.get("config")
        ).get("initial_suffix_token_ids")
        text_same = gated.get("text") == anchor.get("text")
        task_score_delta = _float(_dict(gated.get("task_score")).get("score")) - _float(
            _dict(anchor.get("task_score")).get("score")
        )
        rows.append(
            {
                "gate_active": bool(repair.get("history_instability_gate_active", False)),
                "prompt_same": prompt_same,
                "seed_same": seed_same,
                "seed_tokens_same": seed_tokens_same,
                "task_id": task_id,
                "task_score_delta": task_score_delta,
                "text_same": text_same,
            }
        )
    gate_inactive_rows = [row for row in rows if not bool(row.get("gate_active", False))]
    gate_active_rows = [row for row in rows if bool(row.get("gate_active", False))]
    identity_rows = [
        row
        for row in gate_inactive_rows
        if row.get("seed_same")
        and row.get("prompt_same")
        and row.get("seed_tokens_same")
        and row.get("text_same")
        and abs(_float(row.get("task_score_delta"))) <= 1e-12
    ]
    return {
        "anchor_raw_path": str(anchor_raw_path),
        "compared_task_count": len(rows),
        "gate_active_count": len(gate_active_rows),
        "gate_active_mean_task_score_delta": _mean(
            _float(row.get("task_score_delta")) for row in gate_active_rows
        ),
        "gate_inactive_count": len(gate_inactive_rows),
        "gate_inactive_identity_match_count": len(identity_rows),
        "gated_raw_path": str(gated_raw_path),
        "rows": rows,
    }


def _composite_gate_identity_summary(base_raw_path: Path | None, gated_raw_path: Path | None) -> dict[str, object]:
    if base_raw_path is None or gated_raw_path is None or not base_raw_path.exists() or not gated_raw_path.exists():
        return {}
    base_rows = _repair_rows_by_task(base_raw_path)
    gated_rows = _repair_rows_by_task(gated_raw_path)
    rows = []
    for task_id in sorted(set(base_rows) & set(gated_rows)):
        base = base_rows[task_id]
        gated = gated_rows[task_id]
        repair = _dict(gated.get("repair"))
        gate_active = bool(repair.get("history_instability_gate_active", False)) or bool(
            repair.get("planning_prompt_gate_active", False)
        )
        seed_same = gated.get("generation_seed") == base.get("generation_seed")
        prompt_same = gated.get("prompt") == base.get("prompt")
        seed_tokens_same = _dict(gated.get("config")).get("initial_suffix_token_ids") == _dict(
            base.get("config")
        ).get("initial_suffix_token_ids")
        text_same = gated.get("text") == base.get("text")
        task_score_delta = _float(_dict(gated.get("task_score")).get("score")) - _float(
            _dict(base.get("task_score")).get("score")
        )
        rows.append(
            {
                "gate_active": gate_active,
                "history_instability_gate_active": bool(repair.get("history_instability_gate_active", False)),
                "planning_prompt_gate_active": bool(repair.get("planning_prompt_gate_active", False)),
                "prompt_same": prompt_same,
                "seed_same": seed_same,
                "seed_tokens_same": seed_tokens_same,
                "task_id": task_id,
                "task_score_delta": task_score_delta,
                "text_same": text_same,
            }
        )
    gate_inactive_rows = [row for row in rows if not bool(row.get("gate_active", False))]
    gate_active_rows = [row for row in rows if bool(row.get("gate_active", False))]
    identity_rows = [
        row
        for row in gate_inactive_rows
        if row.get("seed_same")
        and row.get("prompt_same")
        and row.get("seed_tokens_same")
        and row.get("text_same")
        and abs(_float(row.get("task_score_delta"))) <= 1e-12
    ]
    return {
        "anchor_raw_path": str(base_raw_path),
        "compared_task_count": len(rows),
        "gate_active_count": len(gate_active_rows),
        "gate_active_mean_task_score_delta": _mean(
            _float(row.get("task_score_delta")) for row in gate_active_rows
        ),
        "gate_inactive_count": len(gate_inactive_rows),
        "gate_inactive_identity_match_count": len(identity_rows),
        "gated_raw_path": str(gated_raw_path),
        "rows": rows,
    }


def _repair_rows_by_task(path: Path) -> dict[str, dict[str, object]]:
    return {
        _task_id(record): record
        for record in _read_jsonl(path)
        if record.get("generation_stage") == "repair_candidate"
    }


def _candidate_rows(path: Path) -> list[dict[str, object]]:
    return [
        record
        for record in _read_jsonl(path)
        if record.get("generation_stage") == "candidate_generation"
        and _schedule_name(record) == "low_confidence_32"
        and _task_id(record).startswith("plan_")
    ]


def _score_rows_by_task(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    scores = _read_json(path)
    rows = _list_of_dicts(scores.get("comparison_rows"))
    return {str(row.get("task_id", "")): row for row in rows}


def _first_candidate_summary(scores: dict[str, object]) -> dict[str, object]:
    summary = _dict(scores.get("repair_candidate_summary"))
    for value in summary.values():
        if isinstance(value, dict):
            return value
    return {}


def _schedule_name(record: dict[str, object]) -> str:
    schedule = record.get("schedule")
    return str(schedule.get("name", "")) if isinstance(schedule, dict) else ""


def _task_id(record: dict[str, object]) -> str:
    task = record.get("task")
    if isinstance(task, dict):
        return str(task.get("task_id", ""))
    return str(record.get("task_id", ""))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _float(value: object) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _mean(values: Any) -> float:
    numbers = list(values)
    return mean(numbers) if numbers else 0.0


def _format_float(value: object) -> str:
    return f"{_float(value):.6f}"


def main() -> int:
    args = parse_args()
    audit = build_anchor_retention_loss_audit(
        raw_path=args.raw,
        final_source_scores_path=args.final_source_scores,
        history_anchor_scores_path=args.history_anchor_scores,
        guarded_search_scores_path=args.guarded_search_scores,
        history_contrast_scores_path=args.history_contrast_scores,
        history_instability_scores_path=args.history_instability_scores,
        anchor_instability_scores_path=args.anchor_instability_scores,
        anchor_instability_gated_scores_path=args.anchor_instability_gated_scores,
        anchor_instability_gated_raw_path=args.anchor_instability_gated_raw,
        anchor_instability_prompt_gated_scores_path=args.anchor_instability_prompt_gated_scores,
        anchor_instability_prompt_gated_raw_path=args.anchor_instability_prompt_gated_raw,
        anchor_instability_prompt_only_gated_scores_path=args.anchor_instability_prompt_only_gated_scores,
        anchor_instability_prompt_only_gated_raw_path=args.anchor_instability_prompt_only_gated_raw,
        anchor_instability_claim_gated_scores_path=args.anchor_instability_claim_gated_scores,
        anchor_instability_claim_gated_raw_path=args.anchor_instability_claim_gated_raw,
        anchor_instability_claim_oracle_gated_scores_path=args.anchor_instability_claim_oracle_gated_scores,
        anchor_instability_claim_oracle_gated_raw_path=args.anchor_instability_claim_oracle_gated_raw,
        anchor_instability_claim_seeded_gated_scores_path=args.anchor_instability_claim_seeded_gated_scores,
        anchor_instability_claim_seeded_gated_raw_path=args.anchor_instability_claim_seeded_gated_raw,
        anchor_instability_claim_compatible_seeded_gated_scores_path=args.anchor_instability_claim_compatible_seeded_gated_scores,
        anchor_instability_claim_compatible_seeded_gated_raw_path=args.anchor_instability_claim_compatible_seeded_gated_raw,
        anchor_instability_claim_auto_seeded_gated_scores_path=args.anchor_instability_claim_auto_seeded_gated_scores,
        anchor_instability_claim_auto_seeded_gated_raw_path=args.anchor_instability_claim_auto_seeded_gated_raw,
        anchor_instability_claim_auto_seeded_realization_gated_scores_path=args.anchor_instability_claim_auto_seeded_realization_gated_scores,
        anchor_instability_claim_auto_seeded_realization_gated_raw_path=args.anchor_instability_claim_auto_seeded_realization_gated_raw,
        anchor_instability_claim_strict_gated_scores_path=args.anchor_instability_claim_strict_gated_scores,
        anchor_instability_claim_strict_gated_raw_path=args.anchor_instability_claim_strict_gated_raw,
        loose_search_scores_path=args.loose_search_scores,
    )
    _write_json(args.json_output, audit)
    args.report_output.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps({"json": str(args.json_output), "report": str(args.report_output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
