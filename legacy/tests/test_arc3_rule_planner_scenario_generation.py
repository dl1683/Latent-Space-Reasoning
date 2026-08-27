from experiments.generate_arc3_rule_planner_scenarios import generate_scenarios


def test_generates_one_step_and_pair_scenarios():
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

    scenarios = generate_scenarios(library)

    assert [scenario["id"] for scenario in scenarios] == [
        "one-step:shape",
        "one-step:color",
        "two-step:shape+color",
    ]
    assert scenarios[0]["initial_state"] == {"shape": 0}
    assert scenarios[0]["goal_state"] == {"shape": 5}
    assert scenarios[2]["expected_actions"] == ["enter_shape_pad", "enter_color_pad"]


def test_does_not_pair_rules_for_same_field():
    library = {
        "validated_rules": [
            {
                "rule_id": "shape-a",
                "rule": {
                    "rule_id": "shape-a",
                    "action": "shape_a",
                    "field": "shape",
                    "effect": {"before": 0, "after": 1},
                },
            },
            {
                "rule_id": "shape-b",
                "rule": {
                    "rule_id": "shape-b",
                    "action": "shape_b",
                    "field": "shape",
                    "effect": {"before": 1, "after": 2},
                },
            },
        ]
    }

    scenarios = generate_scenarios(library)

    assert [scenario["id"] for scenario in scenarios] == ["one-step:shape-a", "one-step:shape-b"]


def test_can_disable_pair_generation():
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

    scenarios = generate_scenarios(library, include_pairs=False)

    assert len(scenarios) == 2
    assert all(scenario["id"].startswith("one-step:") for scenario in scenarios)


def test_generates_scenario_for_delta_rule():
    library = {
        "validated_rules": [
            {
                "rule_id": "move-left",
                "rule": {
                    "rule_id": "move-left",
                    "action": "ACTION3",
                    "field": "x",
                    "effect": {"delta": -5},
                },
            }
        ]
    }

    scenarios = generate_scenarios(library)

    assert scenarios[0]["initial_state"] == {"x": 0}
    assert scenarios[0]["goal_state"] == {"x": -5}


def test_does_not_pair_actions_with_side_effect_on_other_goal_field():
    library = {
        "validated_rules": [
            {
                "rule_id": "left",
                "rule": {"rule_id": "left", "action": "left", "field": "x", "effect": {"delta": -5}},
            },
            {
                "rule_id": "right-x",
                "rule": {"rule_id": "right-x", "action": "right", "field": "x", "effect": {"delta": 5}},
            },
            {
                "rule_id": "right-steps",
                "rule": {"rule_id": "right-steps", "action": "right", "field": "steps", "effect": {"delta": -2}},
            },
        ]
    }

    scenarios = generate_scenarios(library)

    assert "two-step:left+right-steps" not in [scenario["id"] for scenario in scenarios]


def test_generates_scenario_for_contextual_rule():
    library = {
        "contextual_rules": [
            {
                "rule_id": "pusher",
                "action": "ACTION1",
                "field": "y",
                "effect": {"delta": 15},
                "preconditions": {"x": 49, "level_index": 5},
            }
        ]
    }

    scenarios = generate_scenarios(library)

    assert scenarios == [
        {
            "id": "one-step:pusher",
            "initial_state": {"x": 49, "level_index": 5, "y": 0},
            "goal_state": {"y": 15},
            "expected_solved": True,
            "expected_actions": ["ACTION1"],
            "max_depth": 1,
        }
    ]


def test_does_not_pair_rule_that_changes_contextual_precondition():
    library = {
        "validated_rules": [
            {
                "rule_id": "move-right",
                "rule": {"rule_id": "move-right", "action": "ACTION4", "field": "x", "effect": {"delta": 5}},
            }
        ],
        "contextual_rules": [
            {
                "rule_id": "pusher",
                "action": "ACTION1",
                "field": "y",
                "effect": {"delta": 15},
                "preconditions": {"x": 49},
            }
        ],
    }

    scenarios = generate_scenarios(library)

    assert [scenario["id"] for scenario in scenarios] == ["one-step:move-right", "one-step:pusher"]
