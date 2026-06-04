import json

from experiments.compare_arc3_level_traces import compare_level_traces


def test_compare_level_traces_finds_first_divergence(tmp_path):
    oracle = tmp_path / "oracle.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    oracle.write_text(
        "\n".join(
            json.dumps(
                {
                    "normalized_action": action,
                    "policy_metadata": {"visual_state": {"levels_completed": 2}},
                }
            )
            for action in ["ACTION1", "ACTION2", "ACTION3"]
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        "\n".join(
            json.dumps(
                {
                    "normalized_action": action,
                    "policy_metadata": {
                        "policy": "learned_visual",
                        "levels_completed": 2,
                        "visual_state": {
                            "levels_completed": 2,
                            "bbox_y0": 1,
                            "bbox_y1": 3,
                            "bbox_x0": 5,
                            "bbox_x1": 7,
                            "foreground_components": [{"size": 4, "y0": 1}],
                        },
                        "neighbors": [{"distance": 3.5, "action": "ACTION4"}],
                    },
                }
            )
            for action in ["ACTION1", "ACTION4", "ACTION3"]
        ),
        encoding="utf-8",
    )

    summary = compare_level_traces(oracle, candidate, level=2, context=1)

    assert summary["oracle_records"] == 3
    assert summary["candidate_records"] == 3
    assert summary["first_divergence"] == 1
    assert summary["prefix_action_matches"] == 2
    assert summary["divergence_window"][1]["candidate_neighbors"] == [
        {"distance": 3.5, "action": "ACTION4"}
    ]
    assert summary["divergence_window"][1]["candidate_state"]["bbox"]["x1"] == 7
    assert summary["divergence_window"][1]["candidate_state"]["foreground_components"][0]["size"] == 4
