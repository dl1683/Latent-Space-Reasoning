import json

from experiments.diagnose_arc3_policy_aliases import diagnose_aliases


def test_diagnoses_policy_alias_failures(tmp_path):
    artifact = tmp_path / "online.json"
    artifact.write_text(
        json.dumps(
            {
                "evaluated_transitions": 2,
                "steps": [
                    {
                        "step_number": 1,
                        "choice": {
                            "step_index": 10,
                            "actual_action": "A",
                            "selected_action": "B",
                            "best_actions": ["B"],
                            "changed_field_matches": 0,
                            "changed_fields": 2,
                            "modeled_field_matches": 0,
                            "modeled_fields": 1,
                            "changed_missed_fields": ["x", "y"],
                            "modeled_missed_fields": ["y"],
                        },
                    },
                    {
                        "step_number": 2,
                        "choice": {
                            "step_index": 11,
                            "actual_action": "C",
                            "selected_action": "C",
                            "best_actions": ["C"],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    diagnosis = diagnose_aliases(artifact)

    assert diagnosis.failures == 1
    assert diagnosis.confusion_counts == {"A->B": 1}
    assert diagnosis.changed_missed_field_counts == {"x": 1, "y": 1}
    assert diagnosis.modeled_missed_field_counts == {"y": 1}
    assert diagnosis.oracle_misses == 1
    assert diagnosis.modeled_zero_match_failures == 1
