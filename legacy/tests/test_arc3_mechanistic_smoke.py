from experiments.run_arc3_mechanistic_smoke import run_smoke


def test_runs_mechanistic_smoke(tmp_path):
    result = run_smoke(tmp_path)

    assert result.passed is True
    assert result.audit_failures == []
    assert result.score["status"] == "reusable"
    assert result.planner_evaluation["solved"] == 1
    assert result.planner_evaluation["action_matches"] == 1
