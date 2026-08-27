import json

from experiments.evaluate_arc3_rule_transfer import evaluate_rule_transfer


def test_evaluates_rule_policy_transfer_between_traces(tmp_path):
    train = tmp_path / "train.json"
    test = tmp_path / "test.json"
    train.write_text(
        json.dumps(
            {
                "level": "train",
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
                        "action": "UP",
                        "state_before": {"x": 2, "y": 0},
                        "state_after": {"x": 2, "y": 1},
                    },
                    {
                        "step": 3,
                        "action": "UP",
                        "state_before": {"x": 2, "y": 1},
                        "state_after": {"x": 2, "y": 2},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    test.write_text(
        json.dumps(
            {
                "level": "test",
                "trace": [
                    {
                        "step": 0,
                        "action": "RIGHT",
                        "state_before": {"x": 10, "y": 10},
                        "state_after": {"x": 11, "y": 10},
                    },
                    {
                        "step": 1,
                        "action": "UP",
                        "state_before": {"x": 11, "y": 10},
                        "state_after": {"x": 11, "y": 11},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    score = evaluate_rule_transfer(train, test)

    assert score.train_transitions == 4
    assert score.test_transitions == 2
    assert score.learned_actions == 2
    assert score.decidable_transitions == 2
    assert score.top1_action_matches == 2
    assert score.top1_action_accuracy == 1.0
