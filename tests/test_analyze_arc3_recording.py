import json

from experiments.analyze_arc3_recording import analyze_recording


def test_analyze_recording_summarizes_action_deltas(tmp_path):
    recording = tmp_path / "recording.jsonl"
    records = [
        {
            "data": {
                "frame": [[[4, 4], [4, 9]]],
                "action_input": {"id": 1},
            }
        },
        {
            "data": {
                "frame": [[[4, 3], [4, 9]]],
                "action_input": {"id": 2},
            }
        },
        {
            "data": {
                "frame": [[[4, 3], [5, 9]]],
                "action_input": {"id": 2},
            }
        },
    ]
    recording.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    summary = analyze_recording(recording)

    assert summary["frame_records"] == 3
    assert summary["actions"]["ACTION2"]["count"] == 2
    assert summary["actions"]["ACTION2"]["changed_cells_mean"] == 1.0
    assert summary["actions"]["ACTION2"]["top_color_transitions"] == {
        "4->3": 1,
        "4->5": 1,
    }


def test_analyze_recording_estimates_action_direction(tmp_path):
    recording = tmp_path / "recording.jsonl"
    records = [
        {
            "data": {
                "frame": [[[4, 4, 4], [4, 9, 4], [4, 4, 4]]],
                "action_input": {"id": 4},
            }
        },
        {
            "data": {
                "frame": [[[4, 4, 4], [4, 4, 9], [4, 4, 4]]],
                "action_input": {"id": 4},
            }
        },
        {
            "data": {
                "frame": [[[4, 4, 4], [4, 4, 4], [4, 4, 9]]],
                "action_input": {"id": 4},
            }
        },
    ]
    recording.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    summary = analyze_recording(recording)

    assert summary["actions"]["ACTION4"]["centroid_step_estimate"]["direction"] == "down"


def test_action_direction_uses_consecutive_runs_only(tmp_path):
    recording = tmp_path / "recording.jsonl"
    records = [
        {
            "data": {
                "frame": [[[4, 9, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]]],
                "action_input": {"id": 1},
            }
        },
        {
            "data": {
                "frame": [[[4, 4, 9, 4], [4, 4, 4, 4], [4, 4, 4, 4]]],
                "action_input": {"id": 1},
            }
        },
        {
            "data": {
                "frame": [[[4, 4, 4, 9], [4, 4, 4, 4], [4, 4, 4, 4]]],
                "action_input": {"id": 1},
            }
        },
        {
            "data": {
                "frame": [[[9, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]]],
                "action_input": {"id": 2},
            }
        },
        {
            "data": {
                "frame": [[[4, 4, 4, 4], [9, 4, 4, 4], [4, 4, 4, 4]]],
                "action_input": {"id": 1},
            }
        },
    ]
    recording.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    summary = analyze_recording(recording)

    assert summary["actions"]["ACTION1"]["centroid_step_estimate"]["direction"] == "right"
    assert summary["actions"]["ACTION1"]["centroid_step_estimate"]["samples"] == 1
