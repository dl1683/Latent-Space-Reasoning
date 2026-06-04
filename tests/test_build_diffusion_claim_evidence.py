"""Tests for diffusion claim evidence map generation."""

import json

import pytest

from experiments.build_diffusion_claim_evidence import (
    DEFAULT_CANONICAL_SLOTS,
    DEFAULT_CLAIMS,
    DEFAULT_PUBLIC_BUDGET_CLAIM_ID,
    CanonicalClaimSlot,
    ClaimSpec,
    EvidenceArtifactMissingError,
    build_claim_evidence,
    build_ground_truth_index,
    build_moe_mixed_cost_ledger,
    build_public_benchmark_summary,
    render_ground_truth_index_markdown,
    render_markdown,
    render_public_benchmark_markdown,
)


def _write_scores(
    path,
    *,
    repair_score=0.55,
    repair_budget=4.0,
    repair_delta_vs_fixed=0.25,
    repair_delta_vs_random=0.35,
):
    path.write_text(
        json.dumps(
            {
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
                        "count": 2,
                        "mean_generation_budget_per_task": repair_budget,
                        "mean_task_score": repair_score,
                    },
                },
                "by_family_arm": {
                    "math": {
                        "fixed": {"count": 1, "mean_task_score": 1.0},
                        "random": {"count": 1, "mean_task_score": 1.0},
                    },
                    "planning": {
                        "fixed": {"count": 2, "mean_task_score": 0.30},
                        "random": {"count": 2, "mean_task_score": 0.20},
                        "repair_selected": {"count": 2, "mean_task_score": repair_score},
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
                "repair_eligible_task_count": 2,
                "repair_generation_budget_delta_vs_evolved": 1.5,
                "repair_pack": "constraint_span",
                "repair_source_policy": "non_revision_plus_gap_trajectory",
                "repair_task_delta_per_extra_generation_vs_evolved": 0.02,
                "repair_task_delta_vs_evolved": 0.03,
                "repair_task_delta_vs_fixed": repair_delta_vs_fixed,
                "repair_task_delta_vs_random": repair_delta_vs_random,
                "repair_wins_vs_evolved": {"wins": 2, "ties": 0, "losses": 0},
                "repair_wins_vs_fixed": {"wins": 2, "ties": 0, "losses": 0},
                "repair_wins_vs_random": {"wins": 2, "ties": 0, "losses": 0},
                "adaptive_source_gate_mode": "score_max",
                "content_hash": "a" * 64,
                "run_id": "diffusion-aaaaaaaaaaaaaaaa",
            }
        ),
        encoding="utf-8",
    )


def _write_moe_mixed_scores(path, *, repair_score, repair_budget):
    _write_scores(
        path,
        repair_score=repair_score,
        repair_budget=repair_budget,
        repair_delta_vs_fixed=repair_score - 0.30,
        repair_delta_vs_random=repair_score - 0.20,
    )
    scores = json.loads(path.read_text(encoding="utf-8"))
    scores["all_generation_count"] = 27
    scores["arms"]["fixed"]["count"] = 11
    scores["arms"]["random"]["count"] = 11
    scores["arms"]["trajectory_selected"]["count"] = 11
    scores["arms"]["repair_selected"]["count"] = 8
    scores["by_family_arm"]["planning"]["fixed"]["count"] = 8
    scores["by_family_arm"]["planning"]["random"]["count"] = 8
    scores["by_family_arm"]["planning"]["repair_selected"]["count"] = 8
    scores["repair_eligible_task_count"] = 8
    scores["repair_wins_vs_evolved"] = {"wins": 8, "ties": 0, "losses": 0}
    scores["repair_wins_vs_fixed"] = {"wins": 8, "ties": 0, "losses": 0}
    scores["repair_wins_vs_random"] = {"wins": 8, "ties": 0, "losses": 0}
    path.write_text(json.dumps(scores), encoding="utf-8")


def _build_tmp_claim(tmp_path, claim_id, *, repair_score, repair_budget):
    scores = tmp_path / f"{claim_id}_scores.json"
    _write_moe_mixed_scores(scores, repair_score=repair_score, repair_budget=repair_budget)
    (tmp_path / f"{claim_id}_report.md").write_text("# report", encoding="utf-8")
    (tmp_path / f"{claim_id}_raw.jsonl").write_text("{}", encoding="utf-8")
    return build_claim_evidence(
        ClaimSpec(
            claim_id=claim_id,
            title=claim_id,
            scores_path=scores,
            status="test",
            note="unit test",
        )
    )


def test_build_claim_evidence_computes_repair_covered_baselines(tmp_path):
    scores = tmp_path / "toy_scores.json"
    _write_scores(scores)
    (tmp_path / "toy_report.md").write_text("# report", encoding="utf-8")
    raw = tmp_path / "toy_raw.jsonl"
    raw.write_text("{}", encoding="utf-8")

    evidence = build_claim_evidence(
        ClaimSpec(
            claim_id="toy",
            title="Toy claim",
            scores_path=scores,
            status="test",
            note="unit test",
            raw_path=raw,
        )
    )

    assert evidence.fixed_repair_slice_score == pytest.approx(0.30)
    assert evidence.random_repair_slice_score == pytest.approx(0.20)
    assert evidence.repair_score == pytest.approx(0.55)
    assert evidence.repair_count == 2
    assert evidence.full_count == 3
    assert evidence.repair_generation_budget == pytest.approx(4.0)
    assert evidence.repair_relative_gpu_cost == pytest.approx(4.0)
    assert evidence.repair_wins_vs_fixed == "2/0/0"
    assert evidence.raw_path.endswith("toy_raw.jsonl")
    assert evidence.result_run_id == "diffusion-aaaaaaaaaaaaaaaa"
    assert evidence.result_content_hash == "a" * 64


def test_render_markdown_links_evidence_artifacts(tmp_path):
    scores = tmp_path / "toy_scores.json"
    _write_scores(scores)
    (tmp_path / "toy_report.md").write_text("# report", encoding="utf-8")
    (tmp_path / "toy_raw.jsonl").write_text("{}", encoding="utf-8")
    evidence = build_claim_evidence(
        ClaimSpec(
            claim_id="toy",
            title="Toy claim",
            scores_path=scores,
            status="test",
            note="unit test",
        )
    )

    rendered = render_markdown([evidence])

    assert "# Claim Evidence Map" in rendered
    assert "Toy claim" in rendered
    assert "0.300000" in rendered
    assert "0.550000" in rendered
    assert "toy_scores.json" in rendered
    assert "diffusion-aaaaaaaaaaaaaaaa" in rendered


def test_moe_mixed_cost_ledger_marks_score_cost_frontier(tmp_path):
    claims = [
        _build_tmp_claim(tmp_path, "moe_mixed_low", repair_score=0.50, repair_budget=4.0),
        _build_tmp_claim(tmp_path, "moe_mixed_frontier", repair_score=0.60, repair_budget=4.0),
        _build_tmp_claim(tmp_path, "moe_mixed_tie", repair_score=0.60, repair_budget=4.0),
        _build_tmp_claim(tmp_path, "moe_mixed_cheap", repair_score=0.55, repair_budget=3.0),
        _build_tmp_claim(tmp_path, "moe_mixed_expensive", repair_score=0.60, repair_budget=5.0),
    ]

    ledger = build_moe_mixed_cost_ledger(claims)
    frontier = {row["claim_id"] for row in ledger if row["on_frontier"]}

    assert frontier == {"moe_mixed_cheap", "moe_mixed_frontier", "moe_mixed_tie"}
    assert ledger[0]["claim_id"] == "moe_mixed_cheap"
    assert ledger[0]["relative_gpu_cost"] == pytest.approx(3.0)


def test_render_markdown_includes_moe_mixed_cost_ledger(tmp_path):
    evidence = _build_tmp_claim(tmp_path, "moe_mixed_frontier", repair_score=0.60, repair_budget=4.0)

    rendered = render_markdown([evidence])

    assert "## MoE Lean Mixed Cost Ledger" in rendered
    assert "`moe_mixed_frontier`" in rendered
    assert "4.000000x" in rendered


def test_public_benchmark_summary_uses_only_allowed_public_arms(tmp_path):
    scores = tmp_path / "toy_scores.json"
    _write_scores(scores)
    (tmp_path / "toy_report.md").write_text("# report", encoding="utf-8")
    (tmp_path / "toy_raw.jsonl").write_text("\n".join("{}" for _ in range(12)), encoding="utf-8")
    evidence = build_claim_evidence(
        ClaimSpec(
            claim_id="toy",
            title="Toy claim",
            scores_path=scores,
            status="test",
            note="unit test",
        )
    )

    summary = build_public_benchmark_summary([evidence], claim_id="toy")
    rendered = render_public_benchmark_markdown(summary)
    arms = {arm["arm_id"]: arm for arm in summary["public_arms"]}

    assert list(arms) == ["greedy", "random_perturbation", "latent_repair"]
    assert arms["greedy"]["relative_gpu_cost"] == pytest.approx(1.0)
    assert arms["latent_repair"]["relative_gpu_cost"] == pytest.approx(4.0)
    assert arms["latent_repair"]["lift_per_extra_cost_vs_greedy"] == pytest.approx(0.25 / 3.0)
    assert arms["latent_repair"]["lift_per_extra_cost_vs_random"] == pytest.approx(0.35 / 3.0)
    assert summary["latent_repair_frontier"][0]["relative_gpu_cost"] == pytest.approx(4.0)
    assert summary["latent_repair_frontier"][0]["lift_per_extra_cost_vs_greedy"] == pytest.approx(
        0.25 / 3.0
    )
    assert summary["latent_repair_frontier"][0]["lift_per_extra_cost_vs_random"] == pytest.approx(
        0.35 / 3.0
    )
    assert summary["primary_slice"]["task_count"] == 2
    assert {guard["family"] for guard in summary["guard_checks"]} == {
        "math",
        "science",
        "symbolic",
    }
    assert "Greedy" in rendered
    assert "Random perturbation" in rendered
    assert "Latent repair" in rendered
    for internal_term in ("evolved", "oracle", "trajectory"):
        assert internal_term not in rendered.lower()


def test_build_ground_truth_index_points_to_canonical_artifacts(tmp_path):
    scores = tmp_path / "toy_scores.json"
    _write_scores(scores)
    (tmp_path / "toy_report.md").write_text("# report", encoding="utf-8")
    (tmp_path / "toy_raw.jsonl").write_text("{}", encoding="utf-8")
    evidence = build_claim_evidence(
        ClaimSpec(
            claim_id="toy",
            title="Toy claim",
            scores_path=scores,
            status="test",
            note="unit test",
        )
    )

    index = build_ground_truth_index(
        [evidence],
        canonical_slots=(
            CanonicalClaimSlot(
                slot_id="toy_slot",
                label="Toy slot",
                claim_id="toy",
                selection_reason="unit test selection",
            ),
        ),
    )
    rendered = render_ground_truth_index_markdown(index)

    assert index["schema"] == "diffusion_ground_truth_index.v1"
    assert index["claims"][0]["canonical_files"]["scores"].endswith("toy_scores.json")
    assert len(index["claims"][0]["canonical_file_hashes"]["scores_sha256"]) == 64
    assert index["canonical_slots"][0]["slot_id"] == "toy_slot"
    assert index["canonical_slots"][0]["result_identity"]["run_id"] == "diffusion-aaaaaaaaaaaaaaaa"
    assert "toy_raw.jsonl" in rendered
    assert "diffusion-aaaaaaaaaaaaaaaa" in rendered
    assert "scores `" in rendered
    assert "unit test selection" in rendered


def test_default_canonical_slots_reference_default_claims():
    claim_ids = {claim.claim_id for claim in DEFAULT_CLAIMS}

    assert {slot.claim_id for slot in DEFAULT_CANONICAL_SLOTS} <= claim_ids


def test_default_claims_include_hard_exact_no_proposal_span_repair():
    claim_ids = {claim.claim_id for claim in DEFAULT_CLAIMS}

    assert "dense_llada_hard_exact_no_proposal_span_repair" in claim_ids


def test_default_claims_include_dense_planning_adaptive_span_repair():
    claim_ids = {claim.claim_id for claim in DEFAULT_CLAIMS}

    assert "dense_llada_planning_adaptive_span_repair" in claim_ids


def test_default_claims_include_dense_mixed_adaptive_span_budget():
    claim_ids = {claim.claim_id for claim in DEFAULT_CLAIMS}

    assert "dense_llada_mixed_adaptive_span_budget" in claim_ids


def test_default_claims_include_moe_planning_span_localization():
    claim_ids = {claim.claim_id for claim in DEFAULT_CLAIMS}

    assert "moe_planning_span_localized_repair" in claim_ids


def test_default_claims_include_oracle_aware_claim_gated_moe_frontier():
    claim_ids = {claim.claim_id for claim in DEFAULT_CLAIMS}

    assert "moe_mixed_anchor_instability_claim_oracle_gated_budget" in claim_ids


def test_default_claims_include_compatible_seeded_claim_gated_moe_frontier():
    claim_ids = {claim.claim_id for claim in DEFAULT_CLAIMS}

    assert "moe_mixed_anchor_instability_claim_compatible_seeded_gated_budget" in claim_ids


def test_default_claims_include_auto_seeded_claim_gated_boundary():
    claim_ids = {claim.claim_id for claim in DEFAULT_CLAIMS}

    assert "moe_mixed_anchor_instability_claim_auto_seeded_gated_boundary" in claim_ids


def test_default_claims_include_auto_compat_preserve_seeded_claim_gated_moe_frontier():
    claim_ids = {claim.claim_id for claim in DEFAULT_CLAIMS}

    assert "moe_mixed_anchor_instability_claim_auto_compat_preserve_seeded_gated_budget" in claim_ids


def test_default_claims_include_phase_hybrid_mechanism_equivalent_frontier():
    claim_ids = {claim.claim_id for claim in DEFAULT_CLAIMS}

    assert "moe_mixed_phase_hybrid_preserve_seeded_equivalent_frontier" in claim_ids


def test_default_claims_include_value_proxy_budget_frontier():
    claim_ids = {claim.claim_id for claim in DEFAULT_CLAIMS}

    assert "moe_mixed_phase_final_preserve_seeded_value_proxy_budget" in claim_ids
    assert DEFAULT_PUBLIC_BUDGET_CLAIM_ID == "moe_mixed_phase_final_preserve_seeded_value_proxy_budget"


def test_default_canonical_slots_include_value_proxy_budget_frontier():
    slot_ids = {slot.slot_id for slot in DEFAULT_CANONICAL_SLOTS}

    assert "moe_phase_final_value_proxy_budget" in slot_ids


def test_default_canonical_slots_include_phase_hybrid_mechanism_equivalent_frontier():
    slot_ids = {slot.slot_id for slot in DEFAULT_CANONICAL_SLOTS}

    assert "moe_phase_hybrid_preserve_seeded_mechanism_equivalent" in slot_ids


def test_default_claims_include_auto_seeded_realization_claim_gated_boundary():
    claim_ids = {claim.claim_id for claim in DEFAULT_CLAIMS}

    assert "moe_mixed_anchor_instability_claim_auto_seeded_realization_gated_boundary" in claim_ids


def test_build_claim_evidence_requires_report_and_raw_artifacts(tmp_path):
    scores = tmp_path / "toy_scores.json"
    _write_scores(scores)

    with pytest.raises(EvidenceArtifactMissingError, match="Missing report evidence artifact"):
        build_claim_evidence(
            ClaimSpec(
                claim_id="toy",
                title="Toy claim",
                scores_path=scores,
                status="test",
                note="unit test",
            )
        )

    (tmp_path / "toy_report.md").write_text("# report", encoding="utf-8")

    with pytest.raises(EvidenceArtifactMissingError, match="Missing raw evidence artifact"):
        build_claim_evidence(
            ClaimSpec(
                claim_id="toy",
                title="Toy claim",
                scores_path=scores,
                status="test",
                note="unit test",
            )
        )
