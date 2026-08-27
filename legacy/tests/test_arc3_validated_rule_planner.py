from experiments.plan_arc3_with_validated_rules import plan_with_rules


def test_plans_with_composed_validated_rules():
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

    plan = plan_with_rules(library, {"shape": 0, "color": 0}, {"shape": 5, "color": 3}, max_depth=3)

    assert plan.solved is True
    assert plan.actions == ["enter_shape_pad", "enter_color_pad"]
    assert plan.final_state == {"shape": 5, "color": 3}


def test_respects_depth_bound():
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

    plan = plan_with_rules(library, {"shape": 0, "color": 0}, {"shape": 5, "color": 3}, max_depth=1)

    assert plan.solved is False
    assert plan.reason == "goal not reached within depth bound"


def test_reports_no_validated_actions():
    plan = plan_with_rules({"validated_rules": []}, {"shape": 0}, {"shape": 5}, max_depth=2)

    assert plan.solved is False
    assert plan.reason == "no validated actions available"


def test_plans_with_contextual_rule():
    library = {
        "contextual_rules": [
            {
                "rule_id": "pusher",
                "action": "ACTION1",
                "field": "y",
                "effect": {"delta": 15},
                "preconditions": {"x": 49},
            }
        ]
    }

    plan = plan_with_rules(library, {"x": 49, "y": 10}, {"y": 25}, max_depth=1)

    assert plan.solved is True
    assert plan.actions == ["ACTION1"]
    assert plan.final_state == {"x": 49, "y": 25}
