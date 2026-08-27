import json

from experiments.evaluate_arc3_rule_policy_ablation import evaluate_policy_ablation


def test_evaluates_rule_policy_ablation_variants(tmp_path):
    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            {
                "level": "demo",
                "trace": [
                    {"step": 0, "action": "ACTION3", "state_before": {"x": 20, "y": 0}, "state_after": {"x": 15, "y": 0}},
                    {"step": 1, "action": "ACTION3", "state_before": {"x": 15, "y": 0}, "state_after": {"x": 10, "y": 0}},
                    {"step": 2, "action": "ACTION1", "state_before": {"x": 10, "y": 20}, "state_after": {"x": 10, "y": 15}},
                    {"step": 3, "action": "ACTION1", "state_before": {"x": 10, "y": 15}, "state_after": {"x": 10, "y": 10}},
                    {"step": 4, "action": "ACTION4", "state_before": {"x": 10, "y": 10}, "state_after": {"x": 15, "y": 10}},
                    {"step": 5, "action": "ACTION2", "state_before": {"x": 15, "y": 10}, "state_after": {"x": 15, "y": 15}},
                ],
            }
        ),
        encoding="utf-8",
    )

    score = evaluate_policy_ablation(replay, train_fraction=0.67)
    runs = {run.name: run for run in score.runs}

    assert set(runs) == {"base_only", "base_plus_contextual", "base_plus_inverse", "full"}
    assert runs["base_only"].top1_action_accuracy == 0.0
    assert runs["base_plus_inverse"].top1_action_accuracy == 1.0
    assert runs["full"].top1_action_accuracy == 1.0
