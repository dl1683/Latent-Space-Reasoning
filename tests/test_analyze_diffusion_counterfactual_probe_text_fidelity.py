import json

from experiments.analyze_diffusion_counterfactual_probe_text_fidelity import (
    analyze_probe_text_fidelity,
    render_markdown,
)


def test_probe_text_fidelity_audit_scores_slots_and_authorization(tmp_path):
    raw = tmp_path / "raw.jsonl"
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "rows": [
                    _target_row("plan_a", label=True, lift=0.20),
                    _target_row("plan_b", label=True, lift=0.10),
                    _target_row("plan_c", label=False, lift=-0.05),
                    _target_row("plan_d", label=False, lift=-0.03),
                ]
            }
        ),
        encoding="utf-8",
    )
    raw.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _probe_row(
                    "plan_a",
                    "1) Missing constraint; 2) Evidence metric; 3) Retention risk. FULL_REPAIR_AUTHORIZED=false",
                    "alpha, beta",
                    score=0.30,
                ),
                _probe_row(
                    "plan_b",
                    "Missing constraint: beta. Evidence: metric. Retention risk: none. FULL_REPAIR_AUTHORIZED=false",
                    "gamma, delta",
                    score=0.25,
                ),
                _probe_row(
                    "plan_c",
                    "1) Missing constraint; 2) Evidence metric; 3) Retention risk. FULL_REPAIR_AUTHORIZED=false",
                    "gamma, delta",
                    score=0.30,
                ),
                _probe_row(
                    "plan_d",
                    "Missing constraint: delta. Evidence: metric.. Retention risk. FULL_REPAIR_AUTHUTHORIZED=false",
                    "delta, epsilon",
                    score=0.24,
                ),
            ]
        ),
        encoding="utf-8",
    )

    audit = analyze_probe_text_fidelity(raw_path=raw, targets_path=targets)
    markdown = render_markdown(audit)

    assert audit["schema"] == "diffusion_counterfactual_probe_text_fidelity.v1"
    assert audit["summary"]["row_count"] == 4
    assert audit["summary"]["malformed_authorization_count"] == 1
    assert audit["summary"]["weird_punctuation_count"] == 1
    assert audit["summary"]["best_post_probe_error_count"] == 1
    assert audit["summary"]["gate_decision"] == "diagnostic_only"
    assert audit["rows"][0]["features"]["diagnostic_slot_count"] == 3.0
    assert "Probe Text Fidelity" in markdown


def _target_row(task_id, *, label, lift):
    return {
        "labels": {
            "candidate_lift_vs_trajectory": lift,
            "promote_vs_trajectory": label,
        },
        "task_id": task_id,
    }


def _probe_row(task_id, text, weak_terms, *, score):
    return {
        "generation_stage": "counterfactual_probe",
        "prompt": f"Task prompt\n\nMissing or weak task terms: {weak_terms}\n\nCounterfactual micro-probe only.",
        "task": {
            "task_id": task_id,
        },
        "task_score": {
            "score": score,
        },
        "text": text,
    }
