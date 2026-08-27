from experiments.infer_arc3_rules import infer_rules


def test_promotes_repeated_effect_without_counterexamples():
    objects = [
        {
            "level_id": "6",
            "object_id": "agent",
            "object_type": "agent",
            "transitions": [
                {
                    "step_index": 0,
                    "action": "right",
                    "changed_keys": {"position": {"before": [1, 2], "after": [2, 2]}},
                },
                {
                    "step_index": 1,
                    "action": "right",
                    "changed_keys": {"position": {"before": [2, 2], "after": [3, 2]}},
                },
            ],
        }
    ]

    rules = infer_rules(objects, min_support=2)

    assert len(rules) == 2
    assert all(rule.status == "unknown" for rule in rules)


def test_promotes_exact_repeated_attribute_effect():
    objects = [
        {
            "level_id": "6",
            "object_id": "agent",
            "object_type": "agent",
            "transitions": [
                {
                    "step_index": 3,
                    "action": "enter_shape_pad",
                    "changed_keys": {"shape": {"before": 0, "after": 5}},
                },
                {
                    "step_index": 9,
                    "action": "enter_shape_pad",
                    "changed_keys": {"shape": {"before": 0, "after": 5}},
                },
            ],
        }
    ]

    rules = infer_rules(objects, min_support=2)

    assert len(rules) == 1
    assert rules[0].status == "candidate"
    assert rules[0].support == 2
    assert rules[0].counterexamples == 0
    assert rules[0].evidence_steps == [3, 9]


def test_conflicting_effect_remains_unknown():
    objects = [
        {
            "level_id": "6",
            "object_id": "agent",
            "object_type": "agent",
            "transitions": [
                {
                    "step_index": 3,
                    "action": "enter_color_pad",
                    "changed_keys": {"color": {"before": 0, "after": 1}},
                },
                {
                    "step_index": 9,
                    "action": "enter_color_pad",
                    "changed_keys": {"color": {"before": 0, "after": 3}},
                },
            ],
        }
    ]

    rules = infer_rules(objects, min_support=1)

    assert len(rules) == 2
    assert all(rule.status == "unknown" for rule in rules)
    assert all(rule.counterexamples == 1 for rule in rules)


def test_promotes_repeated_numeric_delta_effect():
    objects = [
        {
            "level_id": "5",
            "object_id": "agent",
            "object_type": "agent",
            "transitions": [
                {"step_index": 1, "action": "ACTION3", "changed_keys": {"x": {"before": 49, "after": 44}}},
                {"step_index": 2, "action": "ACTION3", "changed_keys": {"x": {"before": 44, "after": 39}}},
            ],
        }
    ]

    rules = infer_rules(objects, min_support=2)
    delta_rules = [rule for rule in rules if rule.effect == {"delta": -5}]

    assert len(delta_rules) == 1
    assert delta_rules[0].status == "candidate"
    assert delta_rules[0].support == 2


def test_does_not_create_delta_rule_for_steps_counter():
    objects = [
        {
            "level_id": "5",
            "object_id": "agent",
            "object_type": "agent",
            "transitions": [
                {"step_index": 1, "action": "ACTION1", "changed_keys": {"steps": {"before": 42, "after": 40}}},
                {"step_index": 2, "action": "ACTION1", "changed_keys": {"steps": {"before": 40, "after": 38}}},
            ],
        }
    ]

    rules = infer_rules(objects, min_support=2)

    assert not [rule for rule in rules if rule.effect == {"delta": -2}]


def test_promotes_forward_step_counter_delta():
    objects = [
        {
            "level_id": "5",
            "object_id": "agent",
            "object_type": "agent",
            "transitions": [
                {"step_index": 1, "action": "ACTION1", "changed_keys": {"steps": {"before": 40, "after": 41}}},
                {"step_index": 2, "action": "ACTION1", "changed_keys": {"steps": {"before": 41, "after": 42}}},
            ],
        }
    ]

    rules = infer_rules(objects, min_support=2)
    step_rules = [rule for rule in rules if rule.field == "steps" and rule.effect == {"delta": 1}]

    assert len(step_rules) == 1
    assert step_rules[0].status == "candidate"


def test_promotes_level_boundary_counter_delta():
    objects = [
        {
            "level_id": "5",
            "object_id": "agent",
            "object_type": "agent",
            "transitions": [
                {
                    "step_index": 10,
                    "action": "ACTION2",
                    "changed_keys": {
                        "level_index": {"before": 4, "after": 5},
                        "levels_completed": {"before": 4, "after": 5},
                    },
                },
                {
                    "step_index": 20,
                    "action": "ACTION2",
                    "changed_keys": {
                        "level_index": {"before": 5, "after": 6},
                        "levels_completed": {"before": 5, "after": 6},
                    },
                },
            ],
        }
    ]

    rules = infer_rules(objects, min_support=2)
    boundary_rules = [rule for rule in rules if rule.effect == {"delta": 1, "scope": "level_boundary"}]

    assert len(boundary_rules) == 2
    assert all(rule.status == "candidate" for rule in boundary_rules)


def test_conflicting_numeric_delta_effects_remain_unknown():
    objects = [
        {
            "level_id": "6",
            "object_id": "agent",
            "object_type": "agent",
            "transitions": [
                {"step_index": 1, "action": "ACTION1", "changed_keys": {"y": {"before": 50, "after": 45}}},
                {"step_index": 2, "action": "ACTION1", "changed_keys": {"y": {"before": 45, "after": 40}}},
                {"step_index": 3, "action": "ACTION1", "changed_keys": {"y": {"before": 10, "after": 25}}},
            ],
        }
    ]

    rules = infer_rules(objects, min_support=2)
    y_delta_rules = [rule for rule in rules if rule.field == "y" and "delta" in rule.effect]

    assert y_delta_rules
    assert all(rule.status == "unknown" for rule in y_delta_rules)
