import json
from pathlib import Path

from experiments.run_arc3_mechanistic_pipeline import run_pipeline


def test_runs_offline_mechanistic_pipeline(tmp_path):
    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            {
                "level": "demo",
                "trace": [
                    {
                        "step": 0,
                        "action": "enter_shape_pad",
                        "state_before": {"shape": 0, "position": [1, 1]},
                        "state_after": {"shape": 5, "position": [1, 1]},
                    },
                    {
                        "step": 1,
                        "action": "enter_shape_pad",
                        "state_before": {"shape": 0, "position": [2, 1]},
                        "state_after": {"shape": 5, "position": [2, 1]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = run_pipeline(replay, tmp_path / "out", min_support=2, pretty=True)

    assert manifest["counts"]["transitions"] == 2
    assert manifest["counts"]["objects"] == 1
    assert manifest["counts"]["candidate_rules"] == 1
    assert manifest["counts"]["rule_checks"] == 2
    assert manifest["counts"]["graded_rules"] == 1
    assert manifest["counts"]["validated_rules"] == 1
    assert manifest["counts"]["contextual_rules"] == 0
    assert manifest["counts"]["contextual_rule_checks"] == 0
    assert manifest["counts"]["contextual_graded_rules"] == 0
    assert manifest["counts"]["contextual_validated_rules"] == 0
    assert manifest["counts"]["contextual_rejected_rules"] == 0
    assert manifest["counts"]["contextual_contradictions"] == 0
    assert manifest["counts"]["rejected_rules"] == 0
    assert manifest["counts"]["contradictions"] == 0
    assert manifest["counts"]["repairs"] == 0

    for output_value in manifest["outputs"].values():
        output_path = Path(output_value)
        assert tmp_path in output_path.parents
        assert output_path.exists()

    rules = json.loads(Path(manifest["outputs"]["rules"]).read_text(encoding="utf-8"))
    assert rules[0]["status"] == "candidate"
    graded_rules = json.loads(Path(manifest["outputs"]["graded_rules"]).read_text(encoding="utf-8"))
    assert graded_rules[0]["status"] == "validated"
    validated_library = json.loads(Path(manifest["outputs"]["validated_rules"]).read_text(encoding="utf-8"))
    assert validated_library["validated_rules"][0]["rule_id"] == graded_rules[0]["rule_id"]
    contextual_rules = json.loads(Path(manifest["outputs"]["contextual_rules"]).read_text(encoding="utf-8"))
    assert contextual_rules == []
    contextual_checks = json.loads(Path(manifest["outputs"]["contextual_rule_checks"]).read_text(encoding="utf-8"))
    assert contextual_checks == []
    contextual_graded_rules = json.loads(Path(manifest["outputs"]["contextual_graded_rules"]).read_text(encoding="utf-8"))
    assert contextual_graded_rules == []
    contextual_library = json.loads(Path(manifest["outputs"]["contextual_validated_rules"]).read_text(encoding="utf-8"))
    assert contextual_library["contextual_rules"] == []

    transition_lines = Path(manifest["outputs"]["transitions"]).read_text(encoding="utf-8").splitlines()
    assert len(transition_lines) == 2
    assert all(json.loads(line) for line in transition_lines)
