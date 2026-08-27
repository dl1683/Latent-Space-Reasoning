from experiments.apply_arc3_validated_rules import predict_state


def test_applies_validated_rule_when_precondition_matches():
    library = {
        "validated_rules": [
            {
                "rule_id": "r1",
                "rule": {
                    "rule_id": "r1",
                    "action": "enter_shape_pad",
                    "field": "shape",
                    "effect": {"before": 0, "after": 5},
                },
            }
        ]
    }

    result = predict_state(library, {"shape": 0, "color": 1}, "enter_shape_pad")

    assert result.predicted_state == {"shape": 5, "color": 1}
    assert len(result.applications) == 1
    assert result.applications[0].status == "applied"


def test_skips_validated_rule_when_precondition_mismatches():
    library = {
        "validated_rules": [
            {
                "rule_id": "r1",
                "rule": {
                    "rule_id": "r1",
                    "action": "enter_shape_pad",
                    "field": "shape",
                    "effect": {"before": 0, "after": 5},
                },
            }
        ]
    }

    result = predict_state(library, {"shape": 2, "color": 1}, "enter_shape_pad")

    assert result.predicted_state == {"shape": 2, "color": 1}
    assert result.applications[0].status == "skipped"
    assert result.applications[0].reason == "precondition mismatch"


def test_ignores_rules_for_other_actions():
    library = {
        "validated_rules": [
            {
                "rule_id": "r1",
                "rule": {
                    "rule_id": "r1",
                    "action": "enter_color_pad",
                    "field": "color",
                    "effect": {"before": 0, "after": 3},
                },
            }
        ]
    }

    result = predict_state(library, {"color": 0}, "enter_shape_pad")

    assert result.predicted_state == {"color": 0}
    assert result.applications == []


def test_applies_delta_rule_to_numeric_field():
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

    result = predict_state(library, {"x": 49, "y": 35}, "ACTION3")

    assert result.predicted_state == {"x": 44, "y": 35}
    assert result.applications[0].status == "applied"


def test_applies_contextual_rule_when_preconditions_match():
    library = {
        "contextual_rules": [
            {
                "rule_id": "pusher",
                "action": "ACTION1",
                "field": "y",
                "effect": {"delta": 15},
                "preconditions": {"level_index": 5, "levels_completed": 5, "x": 49},
            }
        ]
    }

    result = predict_state(library, {"level_index": 5, "levels_completed": 5, "x": 49, "y": 10}, "ACTION1")

    assert result.predicted_state["y"] == 25
    assert result.applications[0].status == "applied"


def test_does_not_stack_multiple_rules_for_same_field():
    library = {
        "validated_rules": [
            {
                "rule_id": "base",
                "rule": {
                    "rule_id": "base",
                    "action": "ACTION1",
                    "field": "y",
                    "effect": {"delta": -5},
                },
            },
            {
                "rule_id": "duplicate",
                "rule": {
                    "rule_id": "duplicate",
                    "action": "ACTION1",
                    "field": "y",
                    "effect": {"delta": -5},
                },
            },
        ]
    }

    result = predict_state(library, {"y": 45}, "ACTION1")

    assert result.predicted_state["y"] == 40
    assert result.applications[0].status == "applied"
    assert result.applications[1].status == "skipped"
    assert result.applications[1].reason == "field already predicted"


def test_skips_contextual_rule_when_preconditions_mismatch():
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

    result = predict_state(library, {"x": 24, "y": 50}, "ACTION1")

    assert result.predicted_state == {"x": 24, "y": 50}
    assert result.applications[0].status == "skipped"
    assert result.applications[0].reason == "context precondition mismatch"
