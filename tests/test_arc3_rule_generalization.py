import json

from experiments.evaluate_arc3_rule_generalization import evaluate_rule_generalization


def test_evaluates_heldout_rule_generalization(tmp_path):
    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            {
                "level": "demo",
                "trace": [
                    {
                        "step": 0,
                        "action": "RIGHT",
                        "state_before": {"x": 0, "y": 0},
                        "state_after": {"x": 1, "y": 0},
                    },
                    {
                        "step": 1,
                        "action": "RIGHT",
                        "state_before": {"x": 1, "y": 0},
                        "state_after": {"x": 2, "y": 0},
                    },
                    {
                        "step": 2,
                        "action": "RIGHT",
                        "state_before": {"x": 2, "y": 0},
                        "state_after": {"x": 3, "y": 0},
                    },
                    {
                        "step": 3,
                        "action": "RIGHT",
                        "state_before": {"x": 3, "y": 0},
                        "state_after": {"x": 4, "y": 0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    score = evaluate_rule_generalization(replay, train_fraction=0.5)

    assert score.train_transitions == 2
    assert score.test_transitions == 2
    assert score.candidate_rules == 1
    assert score.supported == 2
    assert score.contradicted == 0
    assert score.applicable_precision == 1.0
    assert score.transition_coverage == 1.0
    assert score.status == "predictive"
