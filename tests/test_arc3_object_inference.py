from experiments.infer_arc3_objects import infer_objects


def test_infers_agent_object_history_from_transition_traces():
    traces = [
        {
            "level_id": "6",
            "step_index": 0,
            "action": "right",
            "state_before": {"position": [1, 2], "shape": 0, "color": 0},
            "state_after": {"position": [2, 2], "shape": 0, "color": 0},
            "solved": False,
        },
        {
            "level_id": "6",
            "step_index": 1,
            "action": "up",
            "state_before": {"position": [2, 2], "shape": 0, "color": 0},
            "state_after": {"position": [2, 1], "shape": 1, "color": 0},
            "solved": False,
        },
    ]

    objects = infer_objects(traces)

    assert len(objects) == 1
    agent = objects[0]
    assert agent.object_id == "agent"
    assert agent.object_type == "agent"
    assert agent.level_id == "6"
    assert agent.first_step == 0
    assert agent.last_step == 1
    assert agent.observations == 2
    assert agent.positions == [[1, 2], [2, 2], [2, 1]]
    assert agent.attributes == {"shape": [0, 1], "color": [0]}
    assert agent.transitions[1]["changed_keys"]["shape"] == {"before": 0, "after": 1}


def test_keeps_explicit_object_identity_when_present():
    traces = [
        {
            "level_id": "demo",
            "step_index": 5,
            "action": "wait",
            "state_before": {"object_id": "pad-1", "object_type": "rotation_pad", "position": [4, 4]},
            "state_after": {"object_id": "pad-1", "object_type": "rotation_pad", "position": [4, 5]},
        }
    ]

    objects = infer_objects(traces)

    assert objects[0].object_id == "pad-1"
    assert objects[0].object_type == "rotation_pad"
    assert objects[0].positions == [[4, 4], [4, 5]]
