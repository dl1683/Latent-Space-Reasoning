import json

from experiments.analyze_diffusion_span_probe_signature_utility import (
    analyze_signature_utility_frontier,
    render_markdown,
)


def test_signature_utility_frontier_reports_cost_recall_tradeoff(tmp_path):
    signature_model = tmp_path / "signature.json"
    signature_model.write_text(
        json.dumps(
            {
                "leave_one_slice_out": {
                    "rows": [
                        _row("pos_hi", True, score=2.0, lift=0.2),
                        _row("pos_lo", True, score=0.4, lift=0.1),
                        _row("neg_mid", False, score=1.0, lift=0.0),
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    result = analyze_signature_utility_frontier(
        signature_model_path=signature_model,
        selection_penalties=(0.0, 0.15),
    )
    markdown = render_markdown(result)

    low_penalty = result["selected_penalty_results"]["0.000000"]
    high_penalty = result["selected_penalty_results"]["0.150000"]

    assert low_penalty["false_positive_count"] == 1
    assert high_penalty["false_negative_count"] >= 1
    assert "Do not promote this utility frontier" in markdown
    assert "Selected Frontier" in markdown


def _row(task_id, label, *, score, lift):
    return {
        "candidate_lift_vs_trajectory": lift,
        "label": label,
        "probe_signature_score": score,
        "task_id": task_id,
        "valid_for_stage1": True,
    }
