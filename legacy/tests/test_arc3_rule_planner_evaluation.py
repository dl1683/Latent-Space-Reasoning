from experiments.evaluate_arc3_rule_planner import evaluate_planner


def test_evaluates_compact_planning_scenarios():
    library = {
        "validated_rules": [
            {
                "rule_id": "shape",
                "rule": {
                    "rule_id": "shape",
                    "action": "enter_shape_pad",
                    "field": "shape",
                    "effect": {"before": 0, "after": 5},
                },
            },
            {
                "rule_id": "color",
                "rule": {
                    "rule_id": "color",
                    "action": "enter_color_pad",
                    "field": "color",
                    "effect": {"before": 0, "after": 3},
                },
            },
        ]
    }
    scenarios = [
        {
            "id": "compose",
            "initial_state": {"shape": 0, "color": 0},
            "goal_state": {"shape": 5, "color": 3},
            "expected_solved": True,
            "expected_actions": ["enter_shape_pad", "enter_color_pad"],
        },
        {
            "id": "too-shallow",
            "initial_state": {"shape": 0, "color": 0},
            "goal_state": {"shape": 5, "color": 3},
            "max_depth": 1,
            "expected_solved": False,
        },
    ]

    evaluation = evaluate_planner(library, scenarios, max_depth=3)

    assert evaluation.scenarios == 2
    assert evaluation.solved == 1
    assert evaluation.expected_solved_matches == 2
    assert evaluation.action_matches == 1
    assert evaluation.results[0].action_match is True


def test_invalid_scenario_is_reported_as_unsolved():
    evaluation = evaluate_planner({"validated_rules": []}, [{"id": "bad"}])

    assert evaluation.scenarios == 1
    assert evaluation.solved == 0
    assert evaluation.results[0].reason == "scenario missing initial_state or goal_state object"
