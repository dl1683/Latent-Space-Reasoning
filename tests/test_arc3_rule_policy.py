import json

from experiments.evaluate_arc3_rule_policy import _rule_library, evaluate_rule_policy


def test_rule_library_adds_inverse_action_symmetry():
    library = _rule_library(
        [
            {
                "rule_id": "move-left",
                "level_id": "1",
                "object_id": "agent",
                "object_type": "agent",
                "action": "ACTION3",
                "field": "x",
                "effect": {"delta": -5},
                "support": 2,
                "counterexamples": 0,
                "status": "candidate",
                "evidence_steps": [1, 2],
            }
        ],
        [],
    )

    derived = [
        item["rule"]
        for item in library["validated_rules"]
        if item["rule"].get("derivation", {}).get("type") == "inverse_action_symmetry"
    ]

    assert len(derived) == 1
    assert derived[0]["action"] == "ACTION4"
    assert derived[0]["field"] == "x"
    assert derived[0]["effect"] == {"delta": 5}


def test_rule_library_adds_contextual_inverse_action_symmetry():
    library = _rule_library(
        [],
        [
            {
                "rule_id": "move-up-context",
                "level_id": "1",
                "action": "ACTION1",
                "field": "y",
                "effect": {"delta": -5},
                "preconditions": {"shape": 0},
                "support": 2,
                "status": "candidate",
                "evidence_steps": [1, 2],
            }
        ],
    )

    derived = [
        rule
        for rule in library["contextual_rules"]
        if rule.get("derivation", {}).get("type") == "inverse_action_symmetry"
    ]

    assert len(derived) == 1
    assert derived[0]["action"] == "ACTION2"
    assert derived[0]["field"] == "y"
    assert derived[0]["effect"] == {"delta": 5}
    assert derived[0]["preconditions"] == {"shape": 0}


def test_evaluates_heldout_rule_action_selection(tmp_path):
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

    score = evaluate_rule_policy(replay, train_fraction=0.67)

    assert score.train_transitions == 4
    assert score.test_transitions == 2
    assert score.learned_actions == 2
    assert score.decidable_transitions == 2
    assert score.no_rule_applicable == 0
    assert score.top1_action_matches == 2
    assert score.oracle_action_matches == 2
    assert score.frequency_baseline_matches == 1
    assert score.exact_transition_matches == 2
    assert score.modeled_transition_matches == 2
    assert score.boundary_transitions == 0
    assert score.non_boundary_transitions == 2
    assert score.boundary_top1_action_matches == 0
    assert score.non_boundary_top1_action_matches == 2
    assert score.top1_action_accuracy == 1.0
    assert score.boundary_top1_action_accuracy == 0.0
    assert score.non_boundary_top1_action_accuracy == 1.0
    assert score.top1_lift_over_frequency == 0.5
    assert score.modeled_transition_accuracy == 1.0
