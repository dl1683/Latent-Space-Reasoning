from experiments.check_arc3_contextual_rules import check_contextual_rules


def test_contextual_rule_supported_when_preconditions_match():
    checks = check_contextual_rules(
        [
            {
                "level_id": "6",
                "step_index": 20,
                "action": "ACTION1",
                "state_before": {"x": 49, "y": 10},
                "state_after": {"x": 49, "y": 25},
            }
        ],
        [
            {
                "rule_id": "pusher",
                "level_id": "6",
                "action": "ACTION1",
                "field": "y",
                "effect": {"delta": 15},
                "preconditions": {"x": 49},
                "status": "candidate",
            }
        ],
    )

    assert len(checks) == 1
    assert checks[0].status == "supported"
    assert checks[0].observed == {"before": 10, "after": 25}


def test_contextual_rule_ignored_when_preconditions_do_not_match():
    checks = check_contextual_rules(
        [
            {
                "level_id": "6",
                "step_index": 1,
                "action": "ACTION1",
                "state_before": {"x": 24, "y": 50},
                "state_after": {"x": 24, "y": 45},
            }
        ],
        [
            {
                "rule_id": "pusher",
                "level_id": "6",
                "action": "ACTION1",
                "field": "y",
                "effect": {"delta": 15},
                "preconditions": {"x": 49},
                "status": "candidate",
            }
        ],
    )

    assert checks == []


def test_contextual_rule_contradicted_when_matching_context_has_wrong_effect():
    checks = check_contextual_rules(
        [
            {
                "level_id": "6",
                "step_index": 20,
                "action": "ACTION1",
                "state_before": {"x": 49, "y": 10},
                "state_after": {"x": 49, "y": 15},
            }
        ],
        [
            {
                "rule_id": "pusher",
                "level_id": "6",
                "action": "ACTION1",
                "field": "y",
                "effect": {"delta": 15},
                "preconditions": {"x": 49},
                "status": "candidate",
            }
        ],
    )

    assert checks[0].status == "contradicted"


def test_contextual_rule_interval_precondition_matches_range():
    checks = check_contextual_rules(
        [
            {
                "level_id": "6",
                "step_index": 20,
                "action": "ACTION4",
                "state_before": {"x": 11, "y": 10},
                "state_after": {"x": 16, "y": 10},
            }
        ],
        [
            {
                "rule_id": "range-move",
                "level_id": "6",
                "action": "ACTION4",
                "field": "x",
                "effect": {"delta": 5},
                "preconditions": {"x": {"min": 10, "max": 12}},
                "status": "candidate",
            }
        ],
    )

    assert len(checks) == 1
    assert checks[0].status == "supported"
