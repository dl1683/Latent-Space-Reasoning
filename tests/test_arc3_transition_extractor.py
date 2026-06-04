import json

from experiments.extract_arc3_transitions import extract_traces


def test_extracts_nested_replay_steps(tmp_path):
    source = tmp_path / "replay.json"
    source.write_text(
        json.dumps(
            {
                "level": 6,
                "trace": [
                    {
                        "step": 0,
                        "action": "right",
                        "state_before": {"position": [1, 2]},
                        "state_after": {"position": [2, 2]},
                        "solved": False,
                        "note": "moved",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    traces = extract_traces([source])

    assert len(traces) == 1
    assert traces[0].level_id == "6"
    assert traces[0].step_index == 0
    assert traces[0].action == "right"
    assert traces[0].state_before == {"position": [1, 2]}
    assert traces[0].state_after == {"position": [2, 2]}
    assert traces[0].observations == {"note": "moved"}
    assert traces[0].solved is False


def test_extracts_jsonl_records(tmp_path):
    source = tmp_path / "trace.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"level_id": "ls20-1", "t": 3, "move": "up", "before": {}, "after": {}}),
                json.dumps({"level_id": "ls20-1", "t": 4, "move": "down", "before": {}, "after": {}}),
            ]
        ),
        encoding="utf-8",
    )

    traces = extract_traces([source])

    assert [trace.step_index for trace in traces] == [3, 4]
    assert [trace.action for trace in traces] == ["up", "down"]
    assert all(trace.level_id == "ls20-1" for trace in traces)


def test_reconstructs_before_after_from_replay_snapshots(tmp_path):
    source = tmp_path / "replay.json"
    source.write_text(
        json.dumps(
            {
                "level": 5,
                "start": {"x": 49, "y": 40, "shape": 4, "color": 0},
                "trace": [
                    {"action_index": 1, "action": "ACTION1", "x": 49, "y": 35, "shape": 4, "color": 0},
                    {"action_index": 2, "action": "ACTION3", "x": 44, "y": 35, "shape": 4, "color": 0},
                ],
            }
        ),
        encoding="utf-8",
    )

    traces = extract_traces([source])

    assert len(traces) == 2
    assert traces[0].state_before == {"x": 49, "y": 40, "shape": 4, "color": 0}
    assert traces[0].state_after == {"x": 49, "y": 35, "shape": 4, "color": 0}
    assert traces[1].state_before == {"x": 49, "y": 35, "shape": 4, "color": 0}
    assert traces[1].state_after == {"x": 44, "y": 35, "shape": 4, "color": 0}


def test_scalar_state_field_does_not_block_replay_snapshot_reconstruction(tmp_path):
    source = tmp_path / "replay.json"
    source.write_text(
        json.dumps(
            {
                "level": 5,
                "start": {"state": "GameState.NOT_FINISHED", "x": 49},
                "trace": [
                    {"action_index": 1, "action": "ACTION1", "state": "GameState.NOT_FINISHED", "x": 44}
                ],
            }
        ),
        encoding="utf-8",
    )

    traces = extract_traces([source])

    assert traces[0].state_before == {"state": "GameState.NOT_FINISHED", "x": 49}
    assert traces[0].state_after == {"state": "GameState.NOT_FINISHED", "x": 44}
