import json

from experiments.analyze_diffusion_span_probe_signed_value_controls import (
    analyze_signed_value_controls,
    render_markdown,
)


def test_signed_value_controls_report_degraded_feature_removals(tmp_path):
    signed_value = tmp_path / "signed_value.json"
    signed_value.write_text(
        json.dumps(
            {
                "model_results": [
                    _model("signed_value_knn_k2_all", "all", 2, 0.5),
                    _model("signed_value_knn_k2_no_text", "no_text", 2, 0.3),
                    _model("signed_value_knn_k1_no_text", "no_text", 1, 0.4),
                    _model("signed_value_knn_k2_no_source", "no_source", 2, 0.2),
                    _model("signed_value_knn_k2_no_gap_span", "no_gap_span", 2, 0.1),
                    _model("signed_value_knn_k2_no_retention", "no_retention", 2, 0.0),
                ],
                "selected_model": _model("signed_value_knn_k2_all", "all", 2, 0.5),
            }
        ),
        encoding="utf-8",
    )

    result = analyze_signed_value_controls(signed_value_path=signed_value)
    markdown = render_markdown(result)

    assert result["summary"]["matched_k_degraded_count"] == 4
    assert result["summary"]["best_withheld_degraded_count"] == 4
    assert "M2 passes as a negative-control audit" in markdown
    assert "Best-Withheld Controls" in markdown


def _model(model_id, feature_group_id, neighbor_count, utility):
    return {
        "false_negative_count": 0,
        "false_positive_count": 1,
        "feature_group_id": feature_group_id,
        "model_id": model_id,
        "neighbor_count": neighbor_count,
        "policy_utility": utility,
        "selected_count": 2,
    }
