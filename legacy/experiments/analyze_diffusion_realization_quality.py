"""Audit compact-seed realization quality for diffusion planning repairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_diffusion_three_arm_benchmark import (  # noqa: E402
    _seed_realization_quality_components,
)

DEFAULT_COMPATIBLE_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_compatible_seeded_gated_"
    "realization_guard_fresh_v1_scores.json"
)
DEFAULT_COMPATIBLE_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_compatible_seeded_gated_"
    "realization_guard_fresh_v1_raw.jsonl"
)
DEFAULT_AUTO_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_auto_seeded_gated_"
    "denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_AUTO_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_auto_seeded_gated_"
    "denoise_phase_gate_dense_history_fresh_v1_raw.jsonl"
)
DEFAULT_AUTO_COMPAT_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_seeded_gated_"
    "realization_guard_v1_scores.json"
)
DEFAULT_AUTO_COMPAT_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_seeded_gated_"
    "realization_guard_v1_raw.jsonl"
)
DEFAULT_AUTO_COMPAT_REALIZED_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_plan004_anchor_instability_claim_auto_compat_realized_seeded_gated_"
    "realization_guard_smoke_v2_scores.json"
)
DEFAULT_AUTO_COMPAT_REALIZED_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_plan004_anchor_instability_claim_auto_compat_realized_seeded_gated_"
    "realization_guard_smoke_v2_raw.jsonl"
)
DEFAULT_AUTO_COMPAT_PRESERVE_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_plan004_anchor_instability_claim_auto_compat_preserve_seeded_gated_"
    "preservation_seed_smoke_v2_scores.json"
)
DEFAULT_AUTO_COMPAT_PRESERVE_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_plan004_anchor_instability_claim_auto_compat_preserve_seeded_gated_"
    "preservation_seed_smoke_v2_raw.jsonl"
)
DEFAULT_AUTO_JOINT_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_plan004_anchor_instability_claim_auto_joint_seeded_gated_"
    "seed_objective_smoke_v1_scores.json"
)
DEFAULT_AUTO_JOINT_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_plan004_anchor_instability_claim_auto_joint_seeded_gated_"
    "seed_objective_smoke_v1_raw.jsonl"
)
DEFAULT_REALIZATION_SCORES = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_auto_seeded_realization_gated_"
    "denoise_phase_gate_dense_history_fresh_v1_scores.json"
)
DEFAULT_REALIZATION_RAW = Path(
    "eval_results/diffusion_language/"
    "llada_moe_mixed_compact_span_anchor_instability_claim_auto_seeded_realization_gated_"
    "denoise_phase_gate_dense_history_fresh_v1_raw.jsonl"
)
DEFAULT_JSON_OUTPUT = Path("eval_results/diffusion_language/diffusion_realization_quality_audit.json")
DEFAULT_REPORT_OUTPUT = Path("DIFFUSION_REALIZATION_QUALITY.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compatible-scores", type=Path, default=DEFAULT_COMPATIBLE_SCORES)
    parser.add_argument("--compatible-raw", type=Path, default=DEFAULT_COMPATIBLE_RAW)
    parser.add_argument("--auto-scores", type=Path, default=DEFAULT_AUTO_SCORES)
    parser.add_argument("--auto-raw", type=Path, default=DEFAULT_AUTO_RAW)
    parser.add_argument("--auto-compat-scores", type=Path, default=DEFAULT_AUTO_COMPAT_SCORES)
    parser.add_argument("--auto-compat-raw", type=Path, default=DEFAULT_AUTO_COMPAT_RAW)
    parser.add_argument("--auto-compat-realized-scores", type=Path, default=DEFAULT_AUTO_COMPAT_REALIZED_SCORES)
    parser.add_argument("--auto-compat-realized-raw", type=Path, default=DEFAULT_AUTO_COMPAT_REALIZED_RAW)
    parser.add_argument("--auto-compat-preserve-scores", type=Path, default=DEFAULT_AUTO_COMPAT_PRESERVE_SCORES)
    parser.add_argument("--auto-compat-preserve-raw", type=Path, default=DEFAULT_AUTO_COMPAT_PRESERVE_RAW)
    parser.add_argument("--auto-joint-scores", type=Path, default=DEFAULT_AUTO_JOINT_SCORES)
    parser.add_argument("--auto-joint-raw", type=Path, default=DEFAULT_AUTO_JOINT_RAW)
    parser.add_argument("--realization-scores", type=Path, default=DEFAULT_REALIZATION_SCORES)
    parser.add_argument("--realization-raw", type=Path, default=DEFAULT_REALIZATION_RAW)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def build_realization_quality_audit(
    *,
    policy_specs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    specs = policy_specs or _default_policy_specs()
    rows = []
    for spec in specs:
        policy_id = str(spec["policy_id"])
        raw_path = Path(str(spec["raw_path"]))
        scores_path = Path(str(spec["scores_path"]))
        score_summary = _read_json(scores_path)
        for record in _load_jsonl(raw_path):
            if record.get("generation_stage") != "repair_candidate":
                continue
            repair = record.get("repair")
            if not isinstance(repair, dict):
                continue
            anchor = repair.get("planning_seed_suffix_anchor")
            if not isinstance(anchor, dict) or not anchor.get("active"):
                continue
            components = _seed_realization_quality_components(record, str(record.get("prompt", "")))
            task = record.get("task")
            rows.append(
                {
                    "action_coverage": components["action_coverage"],
                    "control_coverage": components["control_coverage"],
                    "expected_seed_text": components["expected_seed_text"],
                    "meta_penalty": components["meta_penalty"],
                    "policy_id": policy_id,
                    "policy_label": str(spec["policy_label"]),
                    "realization_quality_loss": 1.0 - float(components["realization_quality_score"]),
                    "realization_quality_score": components["realization_quality_score"],
                    "repair_name": str(repair.get("name", "")),
                    "run_id": str(score_summary.get("run_id", "")),
                    "seed_objective_score": components["seed_objective_score"],
                    "seed_term_coverage": components["seed_term_coverage"],
                    "semantic_preservation_score": components["semantic_preservation_score"],
                    "sentence_shape_score": components["sentence_shape_score"],
                    "specificity_score": components["specificity_score"],
                    "task_id": _task_id(task),
                    "task_score": _task_score(record),
                    "text": str(record.get("text", "")),
                    "word_count": components["word_count"],
                }
            )
    policy_summaries = _policy_summaries(rows)
    return {
        "schema": "diffusion_realization_quality_audit.v1",
        "summary": {
            "row_count": len(rows),
            "policy_count": len(policy_summaries),
            "best_policy_by_realization_quality": _best_policy(policy_summaries, "mean_realization_quality_score"),
            "best_policy_by_seed_objective": _best_policy(policy_summaries, "mean_seed_objective_score"),
            "best_policy_by_task_score": _best_policy(policy_summaries, "mean_task_score"),
        },
        "policy_summaries": policy_summaries,
        "rows": rows,
    }


def render_markdown(audit: dict[str, object]) -> str:
    summary = _dict(audit.get("summary"))
    lines = [
        "# Diffusion Realization Quality",
        "",
        "This file is generated by `experiments/analyze_diffusion_realization_quality.py`.",
        "It turns the current compact-seed boundary into an executable label-free loss.",
        "",
        "## Summary",
        "",
        f"- Rows: `{summary.get('row_count', 0)}`",
        f"- Policies: `{summary.get('policy_count', 0)}`",
        "- Realization-quality score: weighted control coverage, action coverage, seed-term coverage, "
        "prompt coverage, specificity, and direct sentence shape minus meta-text penalties.",
        "- Seed-objective score: realization quality plus semantic preservation of selected/oracle "
        "and claim-survival relations.",
        f"- Best policy by realization quality: `{summary.get('best_policy_by_realization_quality', '')}`",
        f"- Best policy by seed objective: `{summary.get('best_policy_by_seed_objective', '')}`",
        f"- Best policy by task score: `{summary.get('best_policy_by_task_score', '')}`",
        "",
        "## Policy Table",
        "",
        "| Policy | Run | Active Seeds | Mean Task Score | Mean Realization Score | Mean Seed Objective | Mean Semantic Preservation | Mean Realization Loss | Mean Meta Penalty | Mean Control Coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _list_of_dicts(audit.get("policy_summaries")):
        lines.append(
            "| {policy_label} | `{run_id}` | {count} | {mean_task_score:.6f} | "
            "{mean_realization_quality_score:.6f} | {mean_seed_objective_score:.6f} | "
            "{mean_semantic_preservation_score:.6f} | {mean_realization_quality_loss:.6f} | "
            "{mean_meta_penalty:.6f} | {mean_control_coverage:.6f} |".format(
                policy_label=row.get("policy_label", ""),
                run_id=row.get("run_id", ""),
                count=int(row.get("active_seed_count", 0)),
                mean_task_score=float(row.get("mean_task_score", 0.0)),
                mean_realization_quality_score=float(row.get("mean_realization_quality_score", 0.0)),
                mean_seed_objective_score=float(row.get("mean_seed_objective_score", 0.0)),
                mean_semantic_preservation_score=float(row.get("mean_semantic_preservation_score", 0.0)),
                mean_realization_quality_loss=float(row.get("mean_realization_quality_loss", 0.0)),
                mean_meta_penalty=float(row.get("mean_meta_penalty", 0.0)),
                mean_control_coverage=float(row.get("mean_control_coverage", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Active Seed Rows",
            "",
            "| Policy | Task | Task Score | Realization Score | Seed Objective | Semantic Preservation | Loss | Action Coverage | Meta Penalty | Text |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in _list_of_dicts(audit.get("rows")):
        lines.append(
            "| {policy_label} | `{task_id}` | {task_score:.6f} | {realization_quality_score:.6f} | "
            "{seed_objective_score:.6f} | {semantic_preservation_score:.6f} | "
            "{realization_quality_loss:.6f} | {action_coverage:.6f} | {meta_penalty:.6f} | {text} |".format(
                policy_label=row.get("policy_label", ""),
                task_id=row.get("task_id", ""),
                task_score=float(row.get("task_score", 0.0)),
                realization_quality_score=float(row.get("realization_quality_score", 0.0)),
                seed_objective_score=float(row.get("seed_objective_score", 0.0)),
                semantic_preservation_score=float(row.get("semantic_preservation_score", 0.0)),
                realization_quality_loss=float(row.get("realization_quality_loss", 0.0)),
                action_coverage=float(row.get("action_coverage", 0.0)),
                meta_penalty=float(row.get("meta_penalty", 0.0)),
                text=_markdown_cell(str(row.get("text", ""))),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The compact compatible seed remains the positive mechanism result: it scores highest "
            "because the denoised text turns the anchor into direct actions rather than only naming controls.",
            "The automatic compatibility-scored seed recovers that same budget frontier by selecting the "
            "compatible compact anchor without exposing seed-scoring language to the model prompt.",
            "The automatic compatibility-realized seed is the new boundary: it removes seed/anchor meta "
            "language and improves the joint seed objective on `plan_004`, but its task score remains below "
            "the current public frontier.",
            "The automatic compatibility-preserve seed is the cleaner frontier tie: it moves the useful "
            "`preserve` pressure into the denoise seed as public-claim preservation, recovers the `plan_004` "
            "task frontier, and avoids explicit seed/anchor meta text.",
            "The automatic seed applies the same anchor and preserves rubric controls, but its realization "
            "loss is higher because it is less explicit about token-budget and prompt-format equalization.",
            "The realization-gated boundary is correctly penalized: it compresses the answer into a `Control:` "
            "label and mentions the generated seed anchor, so stronger prompt wording is not the right fix.",
            "The next GPU-facing selector should use this as a cheap pre/post-generation term: promote compact "
            "seed repairs that are direct, action-bearing, low-meta, and preserve the selected/oracle plus "
            "claim-survival relations.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    audit = build_realization_quality_audit(
        policy_specs=[
            _policy_spec("compatible_seeded", "compatible seeded", args.compatible_scores, args.compatible_raw),
            _policy_spec(
                "auto_compat_seeded",
                "auto compatibility-scored seeded",
                args.auto_compat_scores,
                args.auto_compat_raw,
            ),
            _policy_spec(
                "auto_compat_realized_seeded",
                "auto compatibility-realized seeded",
                args.auto_compat_realized_scores,
                args.auto_compat_realized_raw,
            ),
            _policy_spec(
                "auto_compat_preserve_seeded",
                "auto compatibility-preserve seeded",
                args.auto_compat_preserve_scores,
                args.auto_compat_preserve_raw,
            ),
            _policy_spec("auto_joint_seeded", "auto joint-objective seeded", args.auto_joint_scores, args.auto_joint_raw),
            _policy_spec("auto_seeded", "auto seeded", args.auto_scores, args.auto_raw),
            _policy_spec(
                "auto_seeded_realization_gated",
                "auto seeded realization-gated",
                args.realization_scores,
                args.realization_raw,
            ),
        ]
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_output.write_text(render_markdown(audit), encoding="utf-8")


def _default_policy_specs() -> list[dict[str, object]]:
    return [
        _policy_spec("compatible_seeded", "compatible seeded", DEFAULT_COMPATIBLE_SCORES, DEFAULT_COMPATIBLE_RAW),
        _policy_spec(
            "auto_compat_seeded",
            "auto compatibility-scored seeded",
            DEFAULT_AUTO_COMPAT_SCORES,
            DEFAULT_AUTO_COMPAT_RAW,
        ),
        _policy_spec(
            "auto_compat_realized_seeded",
            "auto compatibility-realized seeded",
            DEFAULT_AUTO_COMPAT_REALIZED_SCORES,
            DEFAULT_AUTO_COMPAT_REALIZED_RAW,
        ),
        _policy_spec(
            "auto_compat_preserve_seeded",
            "auto compatibility-preserve seeded",
            DEFAULT_AUTO_COMPAT_PRESERVE_SCORES,
            DEFAULT_AUTO_COMPAT_PRESERVE_RAW,
        ),
        _policy_spec("auto_joint_seeded", "auto joint-objective seeded", DEFAULT_AUTO_JOINT_SCORES, DEFAULT_AUTO_JOINT_RAW),
        _policy_spec("auto_seeded", "auto seeded", DEFAULT_AUTO_SCORES, DEFAULT_AUTO_RAW),
        _policy_spec(
            "auto_seeded_realization_gated",
            "auto seeded realization-gated",
            DEFAULT_REALIZATION_SCORES,
            DEFAULT_REALIZATION_RAW,
        ),
    ]


def _policy_spec(policy_id: str, policy_label: str, scores_path: Path, raw_path: Path) -> dict[str, object]:
    return {
        "policy_id": policy_id,
        "policy_label": policy_label,
        "raw_path": raw_path,
        "scores_path": scores_path,
    }


def _policy_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_policy: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy_id"]), []).append(row)
    summaries = []
    for policy_id, policy_rows in sorted(by_policy.items()):
        summaries.append(
            {
                "active_seed_count": len(policy_rows),
                "mean_control_coverage": _mean(row["control_coverage"] for row in policy_rows),
                "mean_meta_penalty": _mean(row["meta_penalty"] for row in policy_rows),
                "mean_realization_quality_loss": _mean(row["realization_quality_loss"] for row in policy_rows),
                "mean_realization_quality_score": _mean(row["realization_quality_score"] for row in policy_rows),
                "mean_seed_objective_score": _mean(row["seed_objective_score"] for row in policy_rows),
                "mean_semantic_preservation_score": _mean(
                    row["semantic_preservation_score"] for row in policy_rows
                ),
                "mean_task_score": _mean(row["task_score"] for row in policy_rows),
                "policy_id": policy_id,
                "policy_label": str(policy_rows[0].get("policy_label", policy_id)),
                "run_id": str(policy_rows[0].get("run_id", "")),
            }
        )
    return sorted(summaries, key=lambda row: float(row["mean_realization_quality_score"]), reverse=True)


def _best_policy(policy_summaries: list[dict[str, object]], metric: str) -> str:
    if not policy_summaries:
        return ""
    return str(max(policy_summaries, key=lambda row: float(row.get(metric, 0.0))).get("policy_id", ""))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _task_id(task: object) -> str:
    if isinstance(task, dict):
        return str(task.get("task_id", ""))
    return ""


def _task_score(record: dict[str, object]) -> float:
    task_score = record.get("task_score")
    if isinstance(task_score, dict):
        value = task_score.get("score")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return 0.0


def _mean(values: Any) -> float:
    numeric = [float(value) for value in values]
    if not numeric:
        return 0.0
    return float(mean(numeric))


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _markdown_cell(text: str, max_chars: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) > max_chars:
        compact = compact[: max_chars - 3].rstrip() + "..."
    return compact.replace("|", "\\|")


if __name__ == "__main__":
    main()
