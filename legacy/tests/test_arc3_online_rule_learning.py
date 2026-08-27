import json

from experiments.evaluate_arc3_online_rule_learning import evaluate_online_rule_learning


def test_evaluates_online_rule_learning_improvement(tmp_path):
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
                    {
                        "step": 4,
                        "action": "RIGHT",
                        "state_before": {"x": 2, "y": 2},
                        "state_after": {"x": 3, "y": 2},
                    },
                    {
                        "step": 5,
                        "action": "UP",
                        "state_before": {"x": 3, "y": 2},
                        "state_after": {"x": 3, "y": 3},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    score = evaluate_online_rule_learning(replay, warmup=4)

    assert score.transitions == 6
    assert score.evaluated_transitions == 2
    assert score.decidable_transitions == 2
    assert score.top1_action_accuracy == 1.0
    assert score.modeled_transition_accuracy == 1.0
