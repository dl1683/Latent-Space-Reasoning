"""Tests for compact-seed realization-quality audit generation."""

from __future__ import annotations

import json

from experiments.analyze_diffusion_realization_quality import (
    build_realization_quality_audit,
    render_markdown,
)


def test_realization_quality_audit_ranks_direct_seed_integration(tmp_path):
    compatible_raw = tmp_path / "compatible.jsonl"
    auto_raw = tmp_path / "auto.jsonl"
    realization_raw = tmp_path / "realization.jsonl"
    compatible_scores = tmp_path / "compatible_scores.json"
    auto_scores = tmp_path / "auto_scores.json"
    realization_scores = tmp_path / "realization_scores.json"
    prompt = (
        "A research result looks impressive, but the baseline used more tokens "
        "and a different prompt format. Design a quick falsification plan before "
        "anyone writes a public claim."
    )
    _write_jsonl(
        compatible_raw,
        [
            _seeded_record(
                prompt=prompt,
                score=0.62,
                text=(
                    "Equalize token budget and prompt format, rerun baseline and intervention "
                    "on locked tasks, record regressions and wins, validate failure modes, "
                    "report oracle selected results, and state the claim survives if it disappears."
                ),
            )
        ],
    )
    _write_jsonl(
        auto_raw,
        [
            _seeded_record(
                prompt=prompt,
                score=0.54,
                text=(
                    "Rework baseline and intervention on locked tasks with same fixed seed, "
                    "record regressions and wins, validate failure modes, and compare to "
                    "oracle selected results; claim survives if disappears."
                ),
            )
        ],
    )
    _write_jsonl(
        realization_raw,
        [
            _seeded_record(
                prompt=prompt,
                score=0.50,
                text=(
                    "Control: token budget, prompt format, locked tasks, regressions, wins "
                    "and failure modes; use generated compact seed anchor as oracle selected "
                    "results; claim survives if disappears."
                ),
            )
        ],
    )
    compatible_scores.write_text(json.dumps({"run_id": "diffusion-compatible"}), encoding="utf-8")
    auto_scores.write_text(json.dumps({"run_id": "diffusion-auto"}), encoding="utf-8")
    realization_scores.write_text(json.dumps({"run_id": "diffusion-realization"}), encoding="utf-8")

    audit = build_realization_quality_audit(
        policy_specs=[
            {
                "policy_id": "compatible",
                "policy_label": "compatible",
                "raw_path": compatible_raw,
                "scores_path": compatible_scores,
            },
            {
                "policy_id": "auto",
                "policy_label": "auto",
                "raw_path": auto_raw,
                "scores_path": auto_scores,
            },
            {
                "policy_id": "realization",
                "policy_label": "realization",
                "raw_path": realization_raw,
                "scores_path": realization_scores,
            },
        ]
    )
    rendered = render_markdown(audit)

    assert audit["schema"] == "diffusion_realization_quality_audit.v1"
    assert audit["summary"]["row_count"] == 3
    assert audit["summary"]["best_policy_by_realization_quality"] == "compatible"
    assert audit["summary"]["best_policy_by_seed_objective"] == "compatible"
    assert audit["summary"]["best_policy_by_task_score"] == "compatible"
    assert [row["policy_id"] for row in audit["policy_summaries"]] == [
        "compatible",
        "auto",
        "realization",
    ]
    realization_row = next(row for row in audit["rows"] if row["policy_id"] == "realization")
    compatible_row = next(row for row in audit["rows"] if row["policy_id"] == "compatible")
    assert realization_row["meta_penalty"] > compatible_row["meta_penalty"]
    assert compatible_row["seed_objective_score"] > realization_row["seed_objective_score"]
    assert "Mean Seed Objective" in rendered
    assert "Diffusion Realization Quality" in rendered
    assert "compatible" in rendered
    assert "Control:" in rendered


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _seeded_record(*, prompt: str, score: float, text: str) -> dict[str, object]:
    return {
        "generation_stage": "repair_candidate",
        "prompt": prompt,
        "repair": {
            "name": "constraint_gap_span_anchor_instability_claim_auto_seeded_gated_repair",
            "planning_seed_suffix_anchor": {
                "active": True,
                "generated_seed_suffix_text": " oracle selected results; claim survives if disappears.",
                "seed_suffix_text": " oracle selected results; claim survives if disappears.",
            },
        },
        "task": {"task_id": "plan_004"},
        "task_score": {"score": score},
        "text": text,
    }
