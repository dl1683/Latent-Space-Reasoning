import json

from experiments.build_diffusion_proof_object import build_proof_object, render_markdown


def test_build_proof_object_collects_all_heads(tmp_path):
    transfer = tmp_path / "transfer.json"
    composite_fit = tmp_path / "composite_fit.json"
    composite_targets = tmp_path / "targets.json"
    budget = tmp_path / "budget.json"
    transfer.write_text(json.dumps(_transfer_fit()), encoding="utf-8")
    composite_fit.write_text(json.dumps(_composite_fit()), encoding="utf-8")
    composite_targets.write_text(json.dumps(_targets()), encoding="utf-8")
    budget.write_text(json.dumps(_budget()), encoding="utf-8")

    proof = build_proof_object(
        budget_loss_path=budget,
        composite_fit_path=composite_fit,
        composite_targets_path=composite_targets,
        transfer_head_fit_path=transfer,
    )

    assert proof["summary"]["head_count"] == 6
    assert proof["summary"]["unresolved_head_count"] == 0
    assert {head["head_id"] for head in proof["heads"]} == {
        "availability",
        "cost",
        "promotion_value",
        "realization",
        "retention",
        "source_trust",
    }
    assert all(head["falsifier"] for head in proof["heads"])


def test_render_markdown_includes_falsifiers(tmp_path):
    transfer = tmp_path / "transfer.json"
    composite_fit = tmp_path / "composite_fit.json"
    composite_targets = tmp_path / "targets.json"
    budget = tmp_path / "budget.json"
    transfer.write_text(json.dumps(_transfer_fit()), encoding="utf-8")
    composite_fit.write_text(json.dumps(_composite_fit()), encoding="utf-8")
    composite_targets.write_text(json.dumps(_targets()), encoding="utf-8")
    budget.write_text(json.dumps(_budget()), encoding="utf-8")

    proof = build_proof_object(
        budget_loss_path=budget,
        composite_fit_path=composite_fit,
        composite_targets_path=composite_targets,
        transfer_head_fit_path=transfer,
    )
    markdown = render_markdown(proof)

    assert "# Diffusion Reasoning Proof Object" in markdown
    assert "## Falsifiers" in markdown
    assert "promotion_value" in markdown
    assert "Next GPU validation" in markdown


def test_build_proof_object_marks_fresh_availability_boundary(tmp_path):
    transfer = tmp_path / "transfer.json"
    composite_fit = tmp_path / "composite_fit.json"
    composite_targets = tmp_path / "targets.json"
    budget = tmp_path / "budget.json"
    predictor = tmp_path / "predictor.json"
    fresh = tmp_path / "fresh.json"
    transfer.write_text(json.dumps(_transfer_fit()), encoding="utf-8")
    composite_fit.write_text(json.dumps(_composite_fit()), encoding="utf-8")
    composite_targets.write_text(json.dumps(_targets()), encoding="utf-8")
    budget.write_text(json.dumps(_budget()), encoding="utf-8")
    predictor.write_text(json.dumps(_availability_predictor()), encoding="utf-8")
    fresh.write_text(json.dumps(_fresh_availability()), encoding="utf-8")

    proof = build_proof_object(
        availability_predictor_fit_path=predictor,
        budget_loss_path=budget,
        composite_fit_path=composite_fit,
        composite_targets_path=composite_targets,
        fresh_availability_eval_path=fresh,
        transfer_head_fit_path=transfer,
    )

    availability = next(head for head in proof["heads"] if head["head_id"] == "availability")
    assert availability["status"] == "boundary"
    assert availability["error_count"] == 2
    assert availability["target_row_count"] == 24
    assert availability["rule_id"] == "calibrated_availability_predictor_v1"
    assert "pre-repair geometry alone" in availability["assertion"]
    assert proof["summary"]["unresolved_head_count"] == 1


def _transfer_fit():
    return {
        "availability_head": {
            "error_count": 0,
            "head_id": "availability_current_decomposed_spend",
            "row_count": 2,
        },
        "promotion_head": {
            "error_count": 0,
            "head_id": "transfer_promotion_value",
        },
        "promotion_policies": [
            {
                "false_negative_count": 0,
                "false_positive_count": 0,
                "true_negative_count": 1,
                "true_positive_count": 1,
            }
        ],
    }


def _composite_fit():
    return {
        "realization_head": {"error_count": 0, "rule_id": "min_realization_policy_error"},
        "retention_head": {"error_count": 0, "rule_id": "classification_safe_history_anchor"},
        "source_head": {"error_count": 0, "rule_id": "retention_safe_history"},
    }


def _targets():
    return {
        "realization_policy_targets": [{"policy_id": "good"}],
        "task_targets": [
            {
                "retention_safe_history_label": True,
                "source_trust_history_label": True,
                "task_id": "a",
            },
            {
                "retention_safe_history_label": False,
                "source_trust_history_label": None,
                "task_id": "b",
            },
        ],
    }


def _budget():
    return {
        "marginal_relative_cost_per_repair": 0.125,
        "planning_task_count": 2,
        "summary": {
            "max_task_policy_gain_lambda": 0.18,
            "max_task_policy_gain_vs_cap": 0.01,
        },
    }


def _availability_predictor():
    return {
        "cuda_policy": {"run_id": "diffusion-test"},
        "summary": {
            "best_rule_error_count": 0,
            "best_rule_id": "prompt_gap_count_le_8",
            "row_count": 16,
        },
    }


def _fresh_availability():
    return {
        "summary": {
            "calibrated_availability_error_count": 2,
            "learned_availability_error_count": 3,
            "target_count": 8,
        }
    }
