"""Tests for diffusion anchor retention-loss audit generation."""

import json

import pytest

from experiments.analyze_diffusion_anchor_retention_loss import (
    build_anchor_retention_loss_audit,
    render_markdown,
)


def test_anchor_retention_loss_audit_scores_safe_and_blocked_history_states(tmp_path):
    raw = tmp_path / "raw.jsonl"
    final_scores = tmp_path / "final_scores.json"
    gated_scores = tmp_path / "gated_scores.json"
    gated_raw = tmp_path / "gated_raw.jsonl"
    prompt_only_gated_scores = tmp_path / "prompt_only_gated_scores.json"
    prompt_only_gated_raw = tmp_path / "prompt_only_gated_raw.jsonl"
    claim_oracle_gated_scores = tmp_path / "claim_oracle_gated_scores.json"
    claim_oracle_gated_raw = tmp_path / "claim_oracle_gated_raw.jsonl"
    claim_seeded_gated_scores = tmp_path / "claim_seeded_gated_scores.json"
    claim_seeded_gated_raw = tmp_path / "claim_seeded_gated_raw.jsonl"
    claim_compatible_seeded_gated_scores = tmp_path / "claim_compatible_seeded_gated_scores.json"
    claim_compatible_seeded_gated_raw = tmp_path / "claim_compatible_seeded_gated_raw.jsonl"
    claim_auto_seeded_gated_scores = tmp_path / "claim_auto_seeded_gated_scores.json"
    claim_auto_seeded_gated_raw = tmp_path / "claim_auto_seeded_gated_raw.jsonl"
    claim_auto_seeded_realization_gated_scores = tmp_path / "claim_auto_seeded_realization_gated_scores.json"
    claim_auto_seeded_realization_gated_raw = tmp_path / "claim_auto_seeded_realization_gated_raw.jsonl"
    claim_strict_gated_scores = tmp_path / "claim_strict_gated_scores.json"
    claim_strict_gated_raw = tmp_path / "claim_strict_gated_raw.jsonl"
    history_scores = tmp_path / "history_scores.json"
    prompt = (
        "A lab can run only two GPU jobs overnight. One job gives a reliable baseline, "
        "the other tests a risky reasoning intervention. Decide which measurements to "
        "collect so tomorrow's result is publishable even if the intervention fails."
    )
    final_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful result (either the baseline or the "
        "intervention) can be published tomorrow, ensuring a publishable result even "
        "if the intervention fails."
    )
    safe_history_text = (
        "Collect the baseline measurement first. If it is successful, proceed to run "
        "the risky intervention job. If the baseline fails, do the intervention job "
        "instead. This way, at least one successful ( the baseline or the intervention) "
        "can be published tomorrow, ensuring a publishable result even if the "
        "intervention fails."
    )
    blocked_history_text = "Collect the baseline measurement first, then publish any result."
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    _candidate_record(
                        "plan_001",
                        prompt=prompt,
                        final_text=final_text,
                        history_text=safe_history_text,
                    )
                ),
                json.dumps(
                    _candidate_record(
                        "plan_002",
                        prompt=prompt,
                        final_text=final_text,
                        history_text=blocked_history_text,
                    )
                ),
                json.dumps(_repair_record("plan_001", gate_active=False)),
                json.dumps(_repair_record("plan_002", gate_active=False, score=0.3)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    gated_raw.write_text(
        "\n".join(
            [
                json.dumps(_repair_record("plan_001", gate_active=False)),
                json.dumps(
                    _repair_record(
                        "plan_002",
                        gate_active=True,
                        initial_suffix_token_ids=[1, None, 3],
                        score=0.5,
                        seed=8,
                        text="changed repair text",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_only_gated_raw.write_text(
        "\n".join(
            [
                json.dumps(_repair_record("plan_001", gate_active=False)),
                json.dumps(
                    _repair_record(
                        "plan_002",
                        gate_active=True,
                        score=0.2,
                        text="prompt-only changed repair text",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    claim_oracle_gated_raw.write_text(
        "\n".join(
            [
                json.dumps(_repair_record("plan_001", gate_active=False)),
                json.dumps(
                    _repair_record(
                        "plan_002",
                        gate_active=True,
                        score=0.55,
                        text="oracle-aware changed repair text",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    claim_seeded_gated_raw.write_text(
        "\n".join(
            [
                json.dumps(_repair_record("plan_001", gate_active=False)),
                json.dumps(
                    _repair_record(
                        "plan_002",
                        gate_active=True,
                        score=0.48,
                        text="seeded changed repair text",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    claim_compatible_seeded_gated_raw.write_text(
        "\n".join(
            [
                json.dumps(_repair_record("plan_001", gate_active=False)),
                json.dumps(
                    _repair_record(
                        "plan_002",
                        gate_active=True,
                        score=0.61,
                        text="compatible seeded changed repair text",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    claim_auto_seeded_gated_raw.write_text(
        "\n".join(
            [
                json.dumps(_repair_record("plan_001", gate_active=False)),
                json.dumps(
                    _repair_record(
                        "plan_002",
                        gate_active=True,
                        score=0.58,
                        text="auto seeded changed repair text",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    claim_auto_seeded_realization_gated_raw.write_text(
        "\n".join(
            [
                json.dumps(_repair_record("plan_001", gate_active=False)),
                json.dumps(
                    _repair_record(
                        "plan_002",
                        gate_active=True,
                        score=0.52,
                        text="auto realization changed repair text",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    claim_strict_gated_raw.write_text(
        "\n".join(
            [
                json.dumps(_repair_record("plan_001", gate_active=False)),
                json.dumps(
                    _repair_record(
                        "plan_002",
                        gate_active=True,
                        score=0.45,
                        text="strict changed repair text",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    final_scores.write_text(
        json.dumps(
            {
                "comparison_rows": [
                    {"repair_task_score": 0.70, "task_id": "plan_001"},
                    {"repair_task_score": 0.60, "task_id": "plan_002"},
                ]
            }
        ),
        encoding="utf-8",
    )
    history_scores.write_text(
        json.dumps(
            {
                "comparison_rows": [
                    {"repair_task_score": 0.65, "task_id": "plan_001"},
                    {"repair_task_score": 0.35, "task_id": "plan_002"},
                ]
            }
        ),
        encoding="utf-8",
    )
    gated_scores.write_text(
        json.dumps(
            {
                "by_family_arm": {
                    "planning": {
                        "fixed": {"mean_task_score": 0.41},
                        "random": {"mean_task_score": 0.37},
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2.625,
                            "mean_task_score": 0.45,
                        },
                    }
                },
                "repair_candidate_summary": {
                    "constraint_gap_span_anchor_instability_gated_repair": {
                        "selected_count": 1,
                        "source_states": "final",
                    }
                },
                "run_id": "diffusion-test-gated",
            }
        ),
        encoding="utf-8",
    )
    prompt_only_gated_scores.write_text(
        json.dumps(
            {
                "by_family_arm": {
                    "planning": {
                        "fixed": {"mean_task_score": 0.41},
                        "random": {"mean_task_score": 0.37},
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2.625,
                            "mean_task_score": 0.39,
                        },
                    }
                },
                "repair_candidate_summary": {
                    "constraint_gap_span_anchor_instability_prompt_only_gated_repair": {
                        "selected_count": 1,
                        "source_states": "final",
                    }
                },
                "run_id": "diffusion-test-prompt-only-gated",
            }
        ),
        encoding="utf-8",
    )
    claim_oracle_gated_scores.write_text(
        json.dumps(
            {
                "by_family_arm": {
                    "planning": {
                        "fixed": {"mean_task_score": 0.41},
                        "random": {"mean_task_score": 0.37},
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2.625,
                            "mean_task_score": 0.50,
                        },
                    }
                },
                "repair_candidate_summary": {
                    "constraint_gap_span_anchor_instability_claim_oracle_gated_repair": {
                        "selected_count": 1,
                        "source_states": "final",
                    }
                },
                "run_id": "diffusion-test-claim-oracle-gated",
            }
        ),
        encoding="utf-8",
    )
    claim_seeded_gated_scores.write_text(
        json.dumps(
            {
                "by_family_arm": {
                    "planning": {
                        "fixed": {"mean_task_score": 0.41},
                        "random": {"mean_task_score": 0.37},
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2.625,
                            "mean_task_score": 0.47,
                        },
                    }
                },
                "repair_candidate_summary": {
                    "constraint_gap_span_anchor_instability_claim_seeded_gated_repair": {
                        "selected_count": 1,
                        "source_states": "final",
                    }
                },
                "run_id": "diffusion-test-claim-seeded-gated",
            }
        ),
        encoding="utf-8",
    )
    claim_compatible_seeded_gated_scores.write_text(
        json.dumps(
            {
                "by_family_arm": {
                    "planning": {
                        "fixed": {"mean_task_score": 0.41},
                        "random": {"mean_task_score": 0.37},
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2.625,
                            "mean_task_score": 0.53,
                        },
                    }
                },
                "repair_candidate_summary": {
                    "constraint_gap_span_anchor_instability_claim_compatible_seeded_gated_repair": {
                        "selected_count": 1,
                        "source_states": "final",
                    }
                },
                "run_id": "diffusion-test-claim-compatible-seeded-gated",
            }
        ),
        encoding="utf-8",
    )
    claim_auto_seeded_gated_scores.write_text(
        json.dumps(
            {
                "by_family_arm": {
                    "planning": {
                        "fixed": {"mean_task_score": 0.41},
                        "random": {"mean_task_score": 0.37},
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2.625,
                            "mean_task_score": 0.51,
                        },
                    }
                },
                "repair_candidate_summary": {
                    "constraint_gap_span_anchor_instability_claim_auto_seeded_gated_repair": {
                        "selected_count": 1,
                        "source_states": "final",
                    }
                },
                "run_id": "diffusion-test-claim-auto-seeded-gated",
            }
        ),
        encoding="utf-8",
    )
    claim_auto_seeded_realization_gated_scores.write_text(
        json.dumps(
            {
                "by_family_arm": {
                    "planning": {
                        "fixed": {"mean_task_score": 0.41},
                        "random": {"mean_task_score": 0.37},
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2.625,
                            "mean_task_score": 0.49,
                        },
                    }
                },
                "repair_candidate_summary": {
                    "constraint_gap_span_anchor_instability_claim_auto_seeded_realization_gated_repair": {
                        "selected_count": 1,
                        "source_states": "final",
                    }
                },
                "run_id": "diffusion-test-claim-auto-seeded-realization-gated",
            }
        ),
        encoding="utf-8",
    )
    claim_strict_gated_scores.write_text(
        json.dumps(
            {
                "by_family_arm": {
                    "planning": {
                        "fixed": {"mean_task_score": 0.41},
                        "random": {"mean_task_score": 0.37},
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2.625,
                            "mean_task_score": 0.44,
                        },
                    }
                },
                "repair_candidate_summary": {
                    "constraint_gap_span_anchor_instability_claim_strict_gated_repair": {
                        "selected_count": 1,
                        "source_states": "final",
                    }
                },
                "run_id": "diffusion-test-claim-strict-gated",
            }
        ),
        encoding="utf-8",
    )

    audit = build_anchor_retention_loss_audit(
        raw_path=raw,
        final_source_scores_path=final_scores,
        history_anchor_scores_path=history_scores,
        guarded_search_scores_path=None,
        history_contrast_scores_path=None,
        history_instability_scores_path=None,
        anchor_instability_scores_path=None,
        anchor_instability_gated_scores_path=gated_scores,
        anchor_instability_gated_raw_path=gated_raw,
        anchor_instability_prompt_gated_scores_path=None,
        anchor_instability_prompt_gated_raw_path=None,
        anchor_instability_prompt_only_gated_scores_path=prompt_only_gated_scores,
        anchor_instability_prompt_only_gated_raw_path=prompt_only_gated_raw,
        anchor_instability_claim_gated_scores_path=None,
        anchor_instability_claim_gated_raw_path=gated_raw,
        anchor_instability_claim_oracle_gated_scores_path=claim_oracle_gated_scores,
        anchor_instability_claim_oracle_gated_raw_path=claim_oracle_gated_raw,
        anchor_instability_claim_seeded_gated_scores_path=claim_seeded_gated_scores,
        anchor_instability_claim_seeded_gated_raw_path=claim_seeded_gated_raw,
        anchor_instability_claim_compatible_seeded_gated_scores_path=claim_compatible_seeded_gated_scores,
        anchor_instability_claim_compatible_seeded_gated_raw_path=claim_compatible_seeded_gated_raw,
        anchor_instability_claim_auto_seeded_gated_scores_path=claim_auto_seeded_gated_scores,
        anchor_instability_claim_auto_seeded_gated_raw_path=claim_auto_seeded_gated_raw,
        anchor_instability_claim_auto_seeded_realization_gated_scores_path=claim_auto_seeded_realization_gated_scores,
        anchor_instability_claim_auto_seeded_realization_gated_raw_path=claim_auto_seeded_realization_gated_raw,
        anchor_instability_claim_strict_gated_scores_path=claim_strict_gated_scores,
        anchor_instability_claim_strict_gated_raw_path=claim_strict_gated_raw,
        loose_search_scores_path=None,
    )
    rendered = render_markdown(audit)

    assert audit["schema"] == "diffusion_anchor_retention_loss_audit.v1"
    assert audit["summary"]["row_count"] == 2
    assert audit["summary"]["safe_history_anchor_count"] == 1
    assert audit["summary"]["blocked_history_anchor_count"] == 1
    assert audit["summary"]["classification_counts"] == {
        "safe_history_anchor": 1,
        "span_advantage_blocks_history": 1,
    }
    safe_row = audit["rows"][0]
    blocked_row = audit["rows"][1]
    assert safe_row["anchor_choice"] == "history"
    assert safe_row["constraint_retention_loss"] == pytest.approx(0.289514)
    assert blocked_row["anchor_choice"] == "final"
    assert blocked_row["constraint_retention_loss"] > safe_row["constraint_retention_loss"]
    assert audit["summary"]["mean_history_minus_final_repair_score"] == pytest.approx(-0.15)
    assert audit["anchor_instability_gated"]["run_id"] == "diffusion-test-gated"
    assert audit["anchor_instability_gated_identity"]["gate_inactive_identity_match_count"] == 1
    assert audit["anchor_instability_gated_identity"]["gate_active_mean_task_score_delta"] == pytest.approx(0.2)
    assert audit["anchor_instability_prompt_only_gated"]["run_id"] == "diffusion-test-prompt-only-gated"
    assert audit["anchor_instability_prompt_only_gated_identity"]["gate_inactive_identity_match_count"] == 1
    assert audit["anchor_instability_prompt_only_gated_identity"][
        "gate_active_mean_task_score_delta"
    ] == pytest.approx(-0.1)
    assert audit["anchor_instability_claim_oracle_gated"]["run_id"] == (
        "diffusion-test-claim-oracle-gated"
    )
    assert audit["anchor_instability_claim_oracle_gated_identity"][
        "gate_active_mean_task_score_delta"
    ] == pytest.approx(0.05)
    assert audit["anchor_instability_claim_seeded_gated"]["run_id"] == (
        "diffusion-test-claim-seeded-gated"
    )
    assert audit["anchor_instability_claim_seeded_gated_identity"][
        "gate_active_mean_task_score_delta"
    ] == pytest.approx(-0.07)
    assert audit["anchor_instability_claim_compatible_seeded_gated"]["run_id"] == (
        "diffusion-test-claim-compatible-seeded-gated"
    )
    assert audit["anchor_instability_claim_compatible_seeded_gated_identity"][
        "gate_active_mean_task_score_delta"
    ] == pytest.approx(0.13)
    assert audit["anchor_instability_claim_auto_seeded_gated"]["run_id"] == (
        "diffusion-test-claim-auto-seeded-gated"
    )
    assert audit["anchor_instability_claim_auto_seeded_gated_identity"][
        "gate_active_mean_task_score_delta"
    ] == pytest.approx(-0.03)
    assert audit["anchor_instability_claim_auto_seeded_realization_gated"]["run_id"] == (
        "diffusion-test-claim-auto-seeded-realization-gated"
    )
    assert audit["anchor_instability_claim_auto_seeded_realization_gated_identity"][
        "gate_active_mean_task_score_delta"
    ] == pytest.approx(-0.06)
    assert audit["anchor_instability_claim_strict_gated"]["run_id"] == (
        "diffusion-test-claim-strict-gated"
    )
    assert audit["anchor_instability_claim_strict_gated_identity"][
        "gate_active_mean_task_score_delta"
    ] == pytest.approx(-0.05)
    assert "Diffusion Anchor Retention Loss" in rendered
    assert "gated anchor instability remask" in rendered
    assert "prompt-only gated anchor instability" in rendered
    assert "claim-oracle gated anchor instability" in rendered
    assert "claim-seeded gated anchor instability" in rendered
    assert "claim-compatible-seeded gated anchor instability" in rendered
    assert "claim-auto-seeded gated anchor instability" in rendered
    assert "claim-auto-seeded realization-gated anchor instability" in rendered
    assert "claim-strict gated anchor instability" in rendered
    assert "Gated Identity Check" in rendered
    assert "plan_001" in rendered
    assert "Loss formula" in rendered


def _candidate_record(
    task_id: str,
    *,
    prompt: str,
    final_text: str,
    history_text: str,
) -> dict[str, object]:
    return {
        "generation_stage": "candidate_generation",
        "generated_token_ids": [1, 2, 3],
        "history_samples": [{"generated_token_ids": [1, 2, 3], "step": 31, "text": history_text}],
        "prompt": prompt,
        "schedule": {"name": "low_confidence_32"},
        "task": {"task_id": task_id},
        "text": final_text,
        "trajectory_summary": {
            "samples": [
                {
                    "mask_count": 1,
                    "step": 31,
                    "visible_chars": len(history_text.strip()),
                    "visible_text": history_text,
                }
            ]
        },
    }


def _repair_record(
    task_id: str,
    *,
    gate_active: bool,
    initial_suffix_token_ids: list[int | None] | None = None,
    score: float = 0.4,
    seed: int = 7,
    text: str = "stable repair text",
) -> dict[str, object]:
    return {
        "candidate_key": "llada",
        "config": {"initial_suffix_token_ids": initial_suffix_token_ids or [1, 2, 3]},
        "generation_seed": seed,
        "generation_stage": "repair_candidate",
        "prompt": "repair prompt",
        "repair": {"history_instability_gate_active": gate_active},
        "task": {"task_id": task_id},
        "task_score": {"score": score},
        "text": text,
    }
