from experiments.check_arc3_rules import check_rules


def test_candidate_rule_is_supported_by_matching_trace():
    traces = [
        {
            "level_id": "6",
            "step_index": 3,
            "action": "enter_shape_pad",
            "state_before": {"shape": 0},
            "state_after": {"shape": 5},
        }
    ]
    rules = [
        {
            "rule_id": "r1",
            "level_id": "6",
            "action": "enter_shape_pad",
            "field": "shape",
            "effect": {"before": 0, "after": 5},
            "status": "candidate",
        }
    ]

    checks = check_rules(traces, rules)

    assert len(checks) == 1
    assert checks[0].status == "supported"
    assert checks[0].observed == {"before": 0, "after": 5}


def test_candidate_rule_is_contradicted_by_different_effect():
    traces = [
        {
            "level_id": "6",
            "step_index": 4,
            "action": "enter_shape_pad",
            "state_before": {"shape": 0},
            "state_after": {"shape": 2},
        }
    ]
    rules = [
        {
            "rule_id": "r1",
            "level_id": "6",
            "action": "enter_shape_pad",
            "field": "shape",
            "effect": {"before": 0, "after": 5},
            "status": "candidate",
        }
    ]

    checks = check_rules(traces, rules)

    assert checks[0].status == "contradicted"
    assert checks[0].observed == {"before": 0, "after": 2}


def test_candidate_rule_can_be_not_applicable_for_same_action():
    traces = [
        {
            "level_id": "6",
            "step_index": 5,
            "action": "enter_shape_pad",
            "state_before": {"position": [1, 1], "shape": 5},
            "state_after": {"position": [1, 2], "shape": 5},
        }
    ]
    rules = [
        {
            "rule_id": "r1",
            "level_id": "6",
            "action": "enter_shape_pad",
            "field": "shape",
            "effect": {"before": 0, "after": 5},
            "status": "candidate",
        }
    ]

    checks = check_rules(traces, rules)

    assert checks[0].status == "not_applicable"
    assert checks[0].observed is None


def test_ignores_unknown_rules():
    checks = check_rules(
        [
            {
                "level_id": "6",
                "step_index": 1,
                "action": "move",
                "state_before": {"shape": 0},
                "state_after": {"shape": 1},
            }
        ],
        [
            {
                "rule_id": "r1",
                "level_id": "6",
                "action": "move",
                "field": "shape",
                "effect": {"before": 0, "after": 1},
                "status": "unknown",
            }
        ],
    )

    assert checks == []


def test_candidate_delta_rule_is_supported_by_matching_delta():
    checks = check_rules(
        [
            {
                "level_id": "5",
                "step_index": 1,
                "action": "ACTION3",
                "state_before": {"x": 49},
                "state_after": {"x": 44},
            }
        ],
        [
            {
                "rule_id": "r1",
                "level_id": "5",
                "action": "ACTION3",
                "field": "x",
                "effect": {"delta": -5},
                "status": "candidate",
            }
        ],
    )

    assert checks[0].status == "supported"


def test_delta_rules_ignore_level_boundary_transitions():
    checks = check_rules(
        [
            {
                "level_id": "5",
                "step_index": 48,
                "action": "ACTION1",
                "state_before": {"level_index": 4, "levels_completed": 4, "y": 10},
                "state_after": {"level_index": 5, "levels_completed": 5, "y": 50},
            }
        ],
        [
            {
                "rule_id": "r1",
                "level_id": "5",
                "action": "ACTION1",
                "field": "y",
                "effect": {"delta": -5},
                "status": "candidate",
            }
        ],
    )

    assert checks == []


def test_boundary_scoped_delta_rule_checks_level_boundary_transitions():
    checks = check_rules(
        [
            {
                "level_id": "5",
                "step_index": 48,
                "action": "ACTION1",
                "state_before": {"level_index": 4, "levels_completed": 4, "y": 10},
                "state_after": {"level_index": 5, "levels_completed": 5, "y": 50},
            }
        ],
        [
            {
                "rule_id": "r1",
                "level_id": "5",
                "action": "ACTION1",
                "field": "level_index",
                "effect": {"delta": 1, "scope": "level_boundary"},
                "status": "candidate",
            }
        ],
    )

    assert len(checks) == 1
    assert checks[0].status == "supported"
