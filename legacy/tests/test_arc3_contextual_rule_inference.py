from experiments.infer_arc3_contextual_rules import infer_contextual_rules


def test_infers_contextual_rule_for_repeated_special_delta():
    traces = [
        {
            "level_id": "6",
            "step_index": 1,
            "action": "ACTION1",
            "state_before": {"x": 49, "y": 10, "shape": 0},
            "state_after": {"x": 49, "y": 25, "shape": 0},
        },
        {
            "level_id": "6",
            "step_index": 2,
            "action": "ACTION1",
            "state_before": {"x": 49, "y": 10, "shape": 1},
            "state_after": {"x": 49, "y": 25, "shape": 1},
        },
        {
            "level_id": "6",
            "step_index": 3,
            "action": "ACTION1",
            "state_before": {"x": 24, "y": 50, "shape": 0},
            "state_after": {"x": 24, "y": 45, "shape": 0},
        },
        {
            "level_id": "6",
            "step_index": 4,
            "action": "ACTION1",
            "state_before": {"x": 24, "y": 45, "shape": 0},
            "state_after": {"x": 24, "y": 40, "shape": 0},
        },
        {
            "level_id": "6",
            "step_index": 5,
            "action": "ACTION1",
            "state_before": {"x": 24, "y": 40, "shape": 0},
            "state_after": {"x": 24, "y": 35, "shape": 0},
        },
    ]

    rules = infer_contextual_rules(traces, min_support=2)
    special = [rule for rule in rules if rule.effect == {"delta": 15}]
    dominant = [rule for rule in rules if rule.effect == {"delta": -5}]

    assert len(special) == 1
    assert special[0].preconditions == {"x": 49, "y": 10}
    assert special[0].support == 2
    assert special[0].evidence_steps == [1, 2]
    assert len(dominant) == 1
    assert dominant[0].preconditions == {"x": 24, "shape": 0}
    assert dominant[0].support == 3


def test_does_not_emit_contextual_rules_without_conflicting_deltas():
    traces = [
        {
            "level_id": "5",
            "step_index": 1,
            "action": "ACTION4",
            "state_before": {"x": 10, "y": 20},
            "state_after": {"x": 15, "y": 20},
        },
        {
            "level_id": "5",
            "step_index": 2,
            "action": "ACTION4",
            "state_before": {"x": 15, "y": 20},
            "state_after": {"x": 20, "y": 20},
        },
    ]

    assert infer_contextual_rules(traces, min_support=2) == []


def test_contextual_rule_preconditions_exclude_counterexamples():
    traces = [
        {
            "level_id": "6",
            "step_index": 1,
            "action": "ACTION1",
            "state_before": {"x": 24, "y": 50, "shape": 0, "color": 3},
            "state_after": {"x": 24, "y": 45, "shape": 0, "color": 3},
        },
        {
            "level_id": "6",
            "step_index": 2,
            "action": "ACTION1",
            "state_before": {"x": 24, "y": 40, "shape": 0, "color": 3},
            "state_after": {"x": 24, "y": 35, "shape": 0, "color": 3},
        },
        {
            "level_id": "6",
            "step_index": 3,
            "action": "ACTION1",
            "state_before": {"x": 24, "y": 20, "shape": 1, "color": 3},
            "state_after": {"x": 24, "y": 30, "shape": 1, "color": 3},
        },
        {
            "level_id": "6",
            "step_index": 4,
            "action": "ACTION1",
            "state_before": {"x": 24, "y": 10, "shape": 1, "color": 3},
            "state_after": {"x": 24, "y": 20, "shape": 1, "color": 3},
        },
    ]

    rules = infer_contextual_rules(traces, min_support=2)
    negative = [rule for rule in rules if rule.effect == {"delta": -5}]

    assert len(negative) == 1
    assert negative[0].preconditions == {"x": 24, "shape": 0, "color": 3}


def test_rejects_contextual_rule_when_counterexamples_match_support():
    traces = [
        {
            "level_id": "6",
            "step_index": 1,
            "action": "ACTION1",
            "state_before": {"x": 24, "y": 50, "shape": 0},
            "state_after": {"x": 24, "y": 45, "shape": 0},
        },
        {
            "level_id": "6",
            "step_index": 2,
            "action": "ACTION1",
            "state_before": {"x": 24, "y": 50, "shape": 0},
            "state_after": {"x": 24, "y": 45, "shape": 0},
        },
        {
            "level_id": "6",
            "step_index": 3,
            "action": "ACTION1",
            "state_before": {"x": 24, "y": 50, "shape": 0},
            "state_after": {"x": 24, "y": 55, "shape": 0},
        },
        {
            "level_id": "6",
            "step_index": 4,
            "action": "ACTION1",
            "state_before": {"x": 24, "y": 50, "shape": 0},
            "state_after": {"x": 24, "y": 55, "shape": 0},
        },
    ]

    assert infer_contextual_rules(traces, min_support=2) == []


def test_infers_interval_contextual_rule_without_exact_stable_context():
    traces = [
        {
            "level_id": "6",
            "step_index": 1,
            "action": "ACTION4",
            "state_before": {"x": 10, "y": 5, "shape": 0},
            "state_after": {"x": 15, "y": 5, "shape": 0},
        },
        {
            "level_id": "6",
            "step_index": 2,
            "action": "ACTION4",
            "state_before": {"x": 12, "y": 7, "shape": 1},
            "state_after": {"x": 17, "y": 7, "shape": 1},
        },
        {
            "level_id": "6",
            "step_index": 3,
            "action": "ACTION4",
            "state_before": {"x": 30, "y": 7, "shape": 0},
            "state_after": {"x": 20, "y": 7, "shape": 0},
        },
    ]

    rules = infer_contextual_rules(traces, min_support=2)

    assert len(rules) == 1
    assert rules[0].effect == {"delta": 5}
    assert rules[0].preconditions == {"x": {"min": 10.0, "max": 12.0}}
