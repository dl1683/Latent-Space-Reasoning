import json

from experiments.export_arc3_validated_contextual_rules import export_validated_contextual_rules


def test_exports_only_validated_contextual_rules(tmp_path):
    graded = tmp_path / "graded_contextual_rules.json"
    graded.write_text(
        json.dumps(
            [
                {
                    "rule_id": "pusher",
                    "status": "validated",
                    "supported": 2,
                    "contradicted": 0,
                    "not_applicable": 0,
                    "validation_threshold": 2,
                    "rule": {
                        "rule_id": "pusher",
                        "action": "ACTION1",
                        "field": "y",
                        "effect": {"delta": 15},
                        "preconditions": {"x": 49, "y": 10},
                    },
                },
                {
                    "rule_id": "weak",
                    "status": "tentative",
                    "supported": 1,
                    "contradicted": 0,
                    "validation_threshold": 2,
                    "rule": {"rule_id": "weak"},
                },
            ]
        ),
        encoding="utf-8",
    )

    library = export_validated_contextual_rules(graded)

    assert library.validation_threshold == 2
    assert len(library.contextual_rules) == 1
    assert library.contextual_rules[0]["rule_id"] == "pusher"
    assert library.contextual_rules[0]["supported"] == 2
    assert library.excluded_counts == {"tentative": 1}
