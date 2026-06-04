"""Tests for promoted diffusion claim evidence validation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from experiments.build_diffusion_claim_evidence import (
    CanonicalClaimSlot,
    ClaimSpec,
    RepairDiagnosticRequirement,
    build_claim_evidence,
    build_ground_truth_index,
    build_public_benchmark_summary,
    render_ground_truth_index_markdown,
    render_markdown,
    render_public_benchmark_markdown,
)
from experiments.validate_diffusion_claim_evidence import validate_claim_evidence

TOY_CANONICAL_SLOTS = (
    CanonicalClaimSlot(
        slot_id="toy_slot",
        label="Toy slot",
        claim_id="toy",
        selection_reason="unit test selection",
    ),
)


def _write_scores(
    path: Path,
    *,
    repair_count: int = 2,
    wins: int = 2,
    span_localized: float | None = None,
    span_fallback: float | None = None,
) -> None:
    scores = {
        "all_generation_count": 12,
        "arms": {
            "fixed": {
                "count": 3,
                "mean_generation_budget_per_task": 1.0,
                "mean_task_score": 0.2,
            },
            "random": {
                "count": 3,
                "mean_generation_budget_per_task": 1.0,
                "mean_task_score": 0.1,
            },
            "trajectory_selected": {"count": 3, "mean_task_score": 0.21},
            "repair_selected": {
                "count": repair_count,
                "mean_generation_budget_per_task": 4.0,
                "mean_task_score": 0.55,
            },
        },
        "by_family_arm": {
            "math": {
                "fixed": {"count": 1, "mean_task_score": 1.0},
                "random": {"count": 1, "mean_task_score": 1.0},
            },
            "planning": {
                "fixed": {"count": repair_count, "mean_task_score": 0.30},
                "random": {"count": repair_count, "mean_task_score": 0.20},
                "repair_selected": {"count": repair_count, "mean_task_score": 0.55},
            },
            "science": {
                "fixed": {"count": 1, "mean_task_score": 1.0},
                "random": {"count": 1, "mean_task_score": 1.0},
            },
            "symbolic": {
                "fixed": {"count": 1, "mean_task_score": 1.0},
                "random": {"count": 1, "mean_task_score": 1.0},
            },
        },
        "exact_task_trajectory_policy": "proposal_history",
        "oracle_headroom_vs_repair": 0.01,
        "oracle_task_score": 0.56,
        "repair_eligible_task_count": repair_count,
        "repair_generation_budget_delta_vs_evolved": 1.5,
        "repair_pack": "constraint_span",
        "repair_source_policy": "non_revision_plus_gap_trajectory",
        "repair_task_delta_per_extra_generation_vs_evolved": 0.02,
        "repair_task_delta_vs_evolved": 0.03,
        "repair_task_delta_vs_fixed": 0.25,
        "repair_task_delta_vs_random": 0.35,
        "repair_wins_vs_evolved": {"wins": wins, "ties": 0, "losses": 0},
        "repair_wins_vs_fixed": {"wins": repair_count, "ties": 0, "losses": 0},
        "repair_wins_vs_random": {"wins": repair_count, "ties": 0, "losses": 0},
        "adaptive_source_gate_mode": "score_max",
    }
    if span_localized is not None or span_fallback is not None:
        scores["repair_candidate_summary"] = {
            "constraint_gap_span_repair": {
                "mean_span_literal_target_found": span_localized,
                "mean_span_fallback_used": span_fallback,
            }
        }
    path.write_text(json.dumps(scores), encoding="utf-8")


def _write_bundle(
    tmp_path: Path,
    *,
    repair_count: int = 2,
    wins: int = 2,
    required_repair_diagnostics: tuple[RepairDiagnosticRequirement, ...] = (),
    span_localized: float | None = None,
    span_fallback: float | None = None,
):
    scores = tmp_path / "toy_scores.json"
    _write_scores(
        scores,
        repair_count=repair_count,
        wins=wins,
        span_localized=span_localized,
        span_fallback=span_fallback,
    )
    (tmp_path / "toy_report.md").write_text("# report", encoding="utf-8")
    (tmp_path / "toy_raw.jsonl").write_text("\n".join("{}" for _ in range(12)), encoding="utf-8")
    spec = ClaimSpec(
        claim_id="toy",
        title="Toy claim",
        scores_path=scores,
        status="test",
        note="unit test",
        required_repair_diagnostics=required_repair_diagnostics,
    )
    evidence = build_claim_evidence(spec)
    output = tmp_path / "CLAIM_EVIDENCE_MAP.md"
    json_output = tmp_path / "diffusion_claim_evidence_map.json"
    index_output = tmp_path / "diffusion_ground_truth_index.json"
    index_markdown_output = tmp_path / "DIFFUSION_GROUND_TRUTH_INDEX.md"
    index = build_ground_truth_index([evidence], canonical_slots=TOY_CANONICAL_SLOTS)
    public_benchmark = build_public_benchmark_summary([evidence])
    output.write_text(render_markdown([evidence]), encoding="utf-8")
    json_output.write_text(
        json.dumps([asdict(evidence)], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    index_output.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    index_markdown_output.write_text(render_ground_truth_index_markdown(index), encoding="utf-8")
    (tmp_path / "DIFFUSION_PUBLIC_BENCHMARK.md").write_text(
        render_public_benchmark_markdown(public_benchmark),
        encoding="utf-8",
    )
    (tmp_path / "diffusion_public_benchmark.json").write_text(
        json.dumps(public_benchmark, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return spec, output, json_output, index_output, index_markdown_output


def _validate(
    spec: ClaimSpec,
    output: Path,
    json_output: Path,
    index_output: Path,
    index_markdown_output: Path,
    stale_doc_paths: tuple[Path, ...] = (),
):
    return validate_claim_evidence(
        output=output,
        json_output=json_output,
        index_output=index_output,
        index_markdown_output=index_markdown_output,
        claim_specs=(spec,),
        canonical_slots=TOY_CANONICAL_SLOTS,
        stale_doc_paths=stale_doc_paths,
    )


def test_validate_claim_evidence_accepts_current_generated_bundle(tmp_path):
    spec, output, json_output, index_output, index_markdown_output = _write_bundle(tmp_path)

    issues = _validate(spec, output, json_output, index_output, index_markdown_output)

    assert issues == []


def test_validate_claim_evidence_rejects_stale_json_map(tmp_path):
    spec, output, json_output, index_output, index_markdown_output = _write_bundle(tmp_path)
    stale = json.loads(json_output.read_text(encoding="utf-8"))
    stale[0]["repair_score"] = 999
    json_output.write_text(json.dumps(stale, indent=2, sort_keys=True), encoding="utf-8")

    issues = _validate(spec, output, json_output, index_output, index_markdown_output)

    assert any("Generated JSON claim map is stale" in issue for issue in issues)


def test_validate_claim_evidence_rejects_stale_ground_truth_index(tmp_path):
    spec, output, json_output, index_output, index_markdown_output = _write_bundle(tmp_path)
    stale = json.loads(index_output.read_text(encoding="utf-8"))
    stale["claims"][0]["canonical_files"]["scores"] = "stale_scores.json"
    index_output.write_text(json.dumps(stale, indent=2, sort_keys=True), encoding="utf-8")

    issues = _validate(spec, output, json_output, index_output, index_markdown_output)

    assert any("Generated ground truth index JSON is stale" in issue for issue in issues)


def test_validate_claim_evidence_rejects_stale_public_benchmark(tmp_path):
    spec, output, json_output, index_output, index_markdown_output = _write_bundle(tmp_path)
    public_output = tmp_path / "DIFFUSION_PUBLIC_BENCHMARK.md"
    public_output.write_text(public_output.read_text(encoding="utf-8") + "\nstale\n", encoding="utf-8")

    issues = _validate(spec, output, json_output, index_output, index_markdown_output)

    assert any("Generated public benchmark Markdown is stale" in issue for issue in issues)


def test_validate_claim_evidence_rejects_inconsistent_win_counts(tmp_path):
    spec, output, json_output, index_output, index_markdown_output = _write_bundle(
        tmp_path,
        repair_count=2,
        wins=3,
    )

    issues = _validate(spec, output, json_output, index_output, index_markdown_output)

    assert any("repair_wins_vs_evolved total 3 != repair count 2" in issue for issue in issues)


def test_validate_claim_evidence_rejects_missing_score_settings(tmp_path):
    spec, output, json_output, index_output, index_markdown_output = _write_bundle(tmp_path)
    scores = json.loads(spec.scores_path.read_text(encoding="utf-8"))
    scores.pop("exact_task_trajectory_policy")
    spec.scores_path.write_text(json.dumps(scores), encoding="utf-8")

    issues = _validate(spec, output, json_output, index_output, index_markdown_output)

    assert any("missing required key exact_task_trajectory_policy" in issue for issue in issues)


def test_validate_claim_evidence_accepts_required_repair_diagnostics(tmp_path):
    requirement = RepairDiagnosticRequirement(
        repair_name="constraint_gap_span_repair",
        metric="mean_span_literal_target_found",
        min_value=1.0,
    )
    spec, output, json_output, index_output, index_markdown_output = _write_bundle(
        tmp_path,
        required_repair_diagnostics=(requirement,),
        span_localized=1.0,
        span_fallback=0.0,
    )

    issues = _validate(spec, output, json_output, index_output, index_markdown_output)

    assert issues == []


def test_validate_claim_evidence_rejects_failed_repair_diagnostic_requirement(tmp_path):
    requirement = RepairDiagnosticRequirement(
        repair_name="constraint_gap_span_repair",
        metric="mean_span_fallback_used",
        max_value=0.0,
    )
    spec, output, json_output, index_output, index_markdown_output = _write_bundle(
        tmp_path,
        required_repair_diagnostics=(requirement,),
        span_localized=1.0,
        span_fallback=0.25,
    )

    issues = _validate(spec, output, json_output, index_output, index_markdown_output)

    assert any(
        "constraint_gap_span_repair.mean_span_fallback_used=0.25 exceeds required maximum 0.0"
        in issue
        for issue in issues
    )


def test_validate_claim_evidence_rejects_missing_repair_diagnostic_requirement(tmp_path):
    requirement = RepairDiagnosticRequirement(
        repair_name="constraint_gap_span_repair",
        metric="mean_span_literal_target_found",
        min_value=1.0,
    )
    spec, output, json_output, index_output, index_markdown_output = _write_bundle(
        tmp_path,
        required_repair_diagnostics=(requirement,),
    )

    issues = _validate(spec, output, json_output, index_output, index_markdown_output)

    assert any("scores file is missing repair_candidate_summary" in issue for issue in issues)


def test_validate_claim_evidence_rejects_stale_public_doc_artifacts(tmp_path):
    spec, output, json_output, index_output, index_markdown_output = _write_bundle(tmp_path)
    public_doc = tmp_path / "README.md"
    public_doc.write_text(
        "Current public evidence: eval_results/diffusion_language/stale_report.md\n",
        encoding="utf-8",
    )

    issues = _validate(
        spec,
        output,
        json_output,
        index_output,
        index_markdown_output,
        stale_doc_paths=(public_doc,),
    )

    assert any("stale_report.md" in issue for issue in issues)
