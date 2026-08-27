import json

from experiments.sweep_arc3_rule_policy import sweep_rule_policy


def test_sweeps_rule_policy_across_train_fractions(tmp_path):
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

    score = sweep_rule_policy(replay, fractions=[0.67, 0.8])

    assert score.fractions == [0.67, 0.8]
    assert len(score.runs) == 2
    assert score.mean_top1_action_accuracy == 1.0
    assert score.min_top1_action_accuracy == 1.0
