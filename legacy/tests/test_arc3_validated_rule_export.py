import json

from experiments.export_arc3_validated_rules import export_validated_rules


def test_exports_only_validated_rules(tmp_path):
    graded = tmp_path / "graded_rules.json"
    graded.write_text(
        json.dumps(
            [
                {
                    "rule_id": "r1",
                    "status": "validated",
                    "supported": 2,
                    "contradicted": 0,
                    "not_applicable": 1,
                    "validation_threshold": 2,
                    "rule": {"rule_id": "r1", "action": "enter_shape_pad"},
                },
                {
                    "rule_id": "r2",
                    "status": "rejected",
                    "supported": 1,
                    "contradicted": 1,
                    "validation_threshold": 2,
                    "rule": {"rule_id": "r2", "action": "enter_color_pad"},
                },
                {
                    "rule_id": "r3",
                    "status": "tentative",
                    "supported": 1,
                    "contradicted": 0,
                    "validation_threshold": 2,
                    "rule": {"rule_id": "r3", "action": "move"},
                },
            ]
        ),
        encoding="utf-8",
    )

    library = export_validated_rules(graded)

    assert library.validation_threshold == 2
    assert len(library.validated_rules) == 1
    assert library.validated_rules[0]["rule_id"] == "r1"
    assert library.validated_rules[0]["supported"] == 2
    assert library.excluded_counts == {"rejected": 1, "tentative": 1}
