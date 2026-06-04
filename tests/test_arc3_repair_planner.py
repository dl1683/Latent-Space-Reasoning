from experiments.plan_arc3_repairs import plan_repairs


def test_plans_repair_for_contradicted_rule():
    checks = [
        {
            "rule_id": "r1",
            "level_id": "6",
            "step_index": 10,
            "action": "enter_shape_pad",
            "field": "shape",
            "expected": {"before": 0, "after": 5},
            "observed": {"before": 0, "after": 2},
            "status": "contradicted",
        }
    ]

    repairs = plan_repairs(checks)

    assert len(repairs) == 1
    assert repairs[0].rule_id == "r1"
    assert repairs[0].priority == "medium"
    assert repairs[0].contradicted_steps == [10]
    assert repairs[0].expected_effects == [{"before": 0, "after": 5}]
    assert repairs[0].observed_effects == [{"before": 0, "after": 2}]
    assert repairs[0].requested_trace["include_neighbor_tiles"] is True
    assert repairs[0].requested_trace["minimum_examples"] == 2


def test_escalates_priority_when_rule_has_multiple_failures():
    checks = [
        {
            "rule_id": "r1",
            "level_id": "6",
            "step_index": 10,
            "action": "enter_color_pad",
            "field": "color",
            "expected": {"before": 0, "after": 1},
            "observed": {"before": 0, "after": 3},
            "status": "contradicted",
        },
        {
            "rule_id": "r1",
            "level_id": "6",
            "step_index": 14,
            "action": "enter_color_pad",
            "field": "color",
            "expected": {"before": 0, "after": 1},
            "observed": {"before": 0, "after": 2},
            "status": "contradicted",
        },
    ]

    repairs = plan_repairs(checks)

    assert repairs[0].priority == "high"
    assert repairs[0].contradicted_steps == [10, 14]
    assert repairs[0].requested_trace["minimum_examples"] == 3


def test_ignores_supported_and_unknown_checks():
    repairs = plan_repairs(
        [
            {
                "rule_id": "r1",
                "level_id": "6",
                "step_index": 1,
                "action": "move",
                "field": "position",
                "status": "supported",
            },
            {
                "rule_id": "r2",
                "level_id": "6",
                "step_index": 2,
                "action": "move",
                "field": "position",
                "status": "not_applicable",
            },
        ]
    )

    assert repairs == []
