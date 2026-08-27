import json

from experiments.analyze_latent_aggregation_extractor_failure import analyze_extractor_failure


def test_extractor_failure_diagnostic_sweeps_thresholds_and_examples(tmp_path):
    components = tmp_path / "components.jsonl"
    replay = tmp_path / "replay.json"
    components.write_text(
        "\n".join(
            [
                json.dumps(_row("plan_a", "keep audit trail", 0.6, True, "audit trail")),
                json.dumps(_row("plan_a", "define rollback", 0.2, True, "rollback plan")),
                json.dumps(_row("plan_a", "measure regressions", 0.4, False, "measure something else")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    replay.write_text(
        json.dumps(
            {
                "summary": {
                    "component_precision": 1.0,
                    "component_recall": 0.5,
                    "online_promoted_task_count": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    result = analyze_extractor_failure(components_path=components, replay_path=replay)

    assert result["component_count"] == 3
    assert result["best_threshold_by_f1"]["threshold"] == 0.2
    assert result["best_threshold_by_f1"]["recall"] == 1.0
    assert result["false_negative_examples"][0]["rubric_item"] == "define rollback"


def _row(task_id, item, support_score, oracle_supported, source_span):
    return {
        "oracle_supported": oracle_supported,
        "rubric_item": item,
        "source_span": source_span,
        "support_score": support_score,
        "task_id": task_id,
    }
