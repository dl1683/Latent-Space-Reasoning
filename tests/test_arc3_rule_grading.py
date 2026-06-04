from experiments.grade_arc3_rules import grade_rules


def test_grades_validated_rule_from_supported_checks():
    graded = grade_rules(
        [{"rule_id": "r1"}],
        [
            {"rule_id": "r1", "status": "supported"},
            {"rule_id": "r1", "status": "supported"},
        ],
        validation_threshold=2,
    )

    assert graded[0].status == "validated"
    assert graded[0].supported == 2
    assert graded[0].contradicted == 0


def test_contradiction_rejects_rule_even_with_support():
    graded = grade_rules(
        [{"rule_id": "r1"}],
        [
            {"rule_id": "r1", "status": "supported"},
            {"rule_id": "r1", "status": "contradicted"},
        ],
        validation_threshold=2,
    )

    assert graded[0].status == "rejected"
    assert graded[0].supported == 1
    assert graded[0].contradicted == 1


def test_partial_support_is_tentative():
    graded = grade_rules(
        [{"rule_id": "r1"}],
        [{"rule_id": "r1", "status": "supported"}],
        validation_threshold=2,
    )

    assert graded[0].status == "tentative"


def test_no_applicable_checks_is_untested():
    graded = grade_rules(
        [{"rule_id": "r1"}],
        [{"rule_id": "r1", "status": "not_applicable"}],
        validation_threshold=2,
    )

    assert graded[0].status == "untested"
    assert graded[0].not_applicable == 1
