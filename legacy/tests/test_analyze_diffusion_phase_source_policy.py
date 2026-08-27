"""Tests for phase-source selector policy audits."""

import json

import pytest

from experiments.analyze_diffusion_phase_source_policy import (
    build_phase_source_policy_audit,
    render_markdown,
)


def test_phase_source_policy_audit_calibrates_strict_similarity_gate(tmp_path):
    loss_targets = tmp_path / "loss_targets.jsonl"
    loss_targets.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _target(
                    "plan_001",
                    label=1,
                    weight=0.06,
                    target_similarity=0.960486,
                    text_similarity=0.979969,
                    safe_count=2,
                    first_safe=30,
                ),
                _target(
                    "plan_003",
                    label=0,
                    weight=0.12,
                    target_similarity=0.943503,
                    text_similarity=0.934498,
                    safe_count=2,
                    first_safe=30,
                ),
                _target(
                    "plan_004",
                    label=0,
                    weight=0.28,
                    target_similarity=0.297872,
                    text_similarity=0.432773,
                    safe_count=0,
                    first_safe=None,
                ),
                _target(
                    "plan_006",
                    label=0,
                    weight=0.19,
                    target_similarity=0.953405,
                    text_similarity=0.981132,
                    safe_count=1,
                    first_safe=31,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    audit = build_phase_source_policy_audit(loss_targets_path=loss_targets)
    rendered = render_markdown(audit)

    assert audit["schema"] == "diffusion_phase_source_policy_audit.v1"
    assert audit["summary"]["target_count"] == 4
    assert audit["selected_policy"]["policy_id"] in {
        "calibrated_similarity_gate",
        "similarity_096",
    }
    assert audit["selected_policy"]["weighted_error"] == pytest.approx(0.0)
    assert audit["selected_policy"]["false_positive_count"] == 0
    assert audit["selected_policy"]["false_negative_count"] == 0
    assert audit["selected_policy"]["predictions"]["plan_001"] == "trust_history_source"
    assert audit["selected_policy"]["predictions"]["plan_003"] == "preserve_final_source"
    naive_safe = _policy_row(audit, "any_safe_phase")
    assert naive_safe["false_positive_count"] == 2
    assert naive_safe["weighted_error"] == pytest.approx(0.31)
    assert "Diffusion Phase-Source Policy Audit" in rendered
    assert "phase_safe_repairable_count > 0" in rendered


def _policy_row(audit, policy_id):
    for row in audit["policy_rows"]:
        if row["policy_id"] == policy_id:
            return row
    raise AssertionError(f"missing policy row {policy_id}")


def _target(
    task_id,
    *,
    label,
    weight,
    target_similarity,
    text_similarity,
    safe_count,
    first_safe,
):
    return {
        "first_repairable_step": 10,
        "first_safe_repairable_step": first_safe,
        "label": label,
        "loss_weight": weight,
        "phase_repairable_count": 22,
        "phase_retention_safety_lag": 20 if first_safe is not None else None,
        "phase_safe_repairable_count": safe_count,
        "target_similarity": target_similarity,
        "task_id": task_id,
        "text_similarity": text_similarity,
    }
