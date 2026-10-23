"""Synthetic regressions for exact and incomplete usage reporting."""
import copy
import json

import pytest


from experiments import dogfood_report as REPORT


def response(input_tokens=11, output_tokens=5):
    return {"response": {"prompt_eval_count": input_tokens, "eval_count": output_tokens}}


def row(**changes):
    value = dict(benchmark="synthetic", question_type="fixture", reader="none",
                 cap=1, control="fixture", arm="fixture", protocol="fixture",
                 judge={"judge": "synthetic_only", "score": 1, "failure_classes": []},
                 status="JUDGED_AUDITED", terminated=True, interactions_used=0,
                 seconds=0, input_tokens=11, output_tokens=5,
                 token_usage_complete=True, model_calls_attempted=1,
                 model_calls_completed=1, trace=[response()])
    value.update(changes)
    return value


def summary(*rows):
    return REPORT.summarize(rows)["table"][0]


def assert_count(actual, name, exact, lower, complete):
    assert actual[name] == exact
    assert actual[name + "_lower_bound"] == lower
    assert type(actual[name + "_lower_bound"]) is int
    assert actual[name + "_lower_bound"] >= 0
    assert actual[name + "_complete"] is complete


def test_complete_and_measured_zero():
    actual = summary(row())
    for name, number in (("input_tokens", 11), ("output_tokens", 5),
                         ("model_calls_attempted", 1), ("model_calls_completed", 1)):
        assert_count(actual, name, number, number, True)
    assert actual["token_usage_complete"] is True
    assert actual["model_calls_complete"] is True
    actual = summary(row(input_tokens=0, output_tokens=0, trace=[],
                         model_calls_attempted=0, model_calls_completed=0))
    for name in ("input_tokens", "output_tokens", "model_calls_attempted", "model_calls_completed"):
        assert_count(actual, name, 0, 0, True)


def test_incomplete_group_keeps_subtotals_not_exact():
    actual = summary(row(), row(input_tokens=7, output_tokens=2,
                                trace=[response(7, 2)], token_usage_complete=False))
    assert_count(actual, "input_tokens", None, 18, False)
    assert_count(actual, "output_tokens", None, 7, False)
    assert_count(actual, "model_calls_attempted", 2, 2, True)
    assert actual["token_usage_complete"] is False


@pytest.mark.parametrize("bad", [None, -1, True, False, "11", 11.0, {}, []])
@pytest.mark.parametrize("name,key,other,other_number", [
    ("input_tokens", "prompt_eval_count", "output_tokens", 5),
    ("output_tokens", "eval_count", "input_tokens", 11),
])
def test_bad_trace_count_preserves_other_side(bad, name, key, other, other_number):
    event = response()
    event["response"][key] = bad
    actual = summary(row(trace=[event]))
    assert_count(actual, name, None, 0, False)
    assert_count(actual, other, other_number, other_number, True)
    assert actual["token_usage_complete"] is False


@pytest.mark.parametrize("bad", [None, -1, True, False, "11", 11.0])
def test_bad_declared_count_does_not_erase_valid_trace(bad):
    actual = summary(row(input_tokens=bad))
    assert_count(actual, "input_tokens", None, 11, False)
    assert_count(actual, "output_tokens", 5, 5, True)


@pytest.mark.parametrize("flag", [None, False, 1, "true"])
def test_only_literal_true_certifies_tokens(flag):
    actual = summary(row(token_usage_complete=flag))
    assert_count(actual, "input_tokens", None, 11, False)
    assert_count(actual, "output_tokens", None, 5, False)


def test_missing_legacy_completeness_with_trace():
    value = row()
    del value["token_usage_complete"]
    del value["model_calls_attempted"]
    del value["model_calls_completed"]
    actual = summary(value)
    assert_count(actual, "input_tokens", None, 11, False)
    assert_count(actual, "model_calls_attempted", None, 1, False)
    assert_count(actual, "model_calls_completed", None, 1, False)


def test_absent_trace_declared_exact_and_legacy_lower_bounds():
    value = row()
    del value["trace"]
    assert_count(summary(value), "input_tokens", 11, 11, True)
    del value["token_usage_complete"]
    assert_count(summary(value), "input_tokens", None, 11, False)
    del value["input_tokens"]
    actual = summary(value)
    assert_count(actual, "input_tokens", None, 0, False)
    assert_count(actual, "output_tokens", None, 5, False)


@pytest.mark.parametrize("trace", [None, {}, "invalid", [None], [{}]])
def test_malformed_trace_cannot_be_overridden_by_aggregate(trace):
    actual = summary(row(trace=trace))
    assert_count(actual, "input_tokens", None, 0, False)
    assert_count(actual, "output_tokens", None, 0, False)
    assert_count(actual, "model_calls_attempted", None, 0, False)


def test_empty_trace_contradicts_positive_declared_costs():
    actual = summary(row(trace=[]))
    assert_count(actual, "input_tokens", None, 0, False)
    assert_count(actual, "output_tokens", None, 0, False)
    assert_count(actual, "model_calls_attempted", None, 0, False)
    assert_count(actual, "model_calls_completed", None, 0, False)


@pytest.mark.parametrize("declared", [0, 9, 99])
def test_contradictory_aggregate_does_not_replace_observed_tokens(declared):
    actual = summary(row(input_tokens=declared))
    assert_count(actual, "input_tokens", None, 11, False)
    assert_count(actual, "output_tokens", 5, 5, True)


def test_one_side_missing_per_call_retains_other_known_counts():
    second = response(7, 2)
    del second["response"]["eval_count"]
    actual = summary(row(input_tokens=18, output_tokens=5,
                         trace=[response(), second],
                         model_calls_attempted=2, model_calls_completed=2))
    assert_count(actual, "input_tokens", 18, 18, True)
    assert_count(actual, "output_tokens", None, 5, False)


def test_interrupted_attempt_is_unknown_not_free_even_with_partial_counts():
    error = {"error": "synthetic interrupted", "partial_response":
             {"prompt_eval_count": 100, "eval_count": 50}}
    actual = summary(row(trace=[response(), error], token_usage_complete=True,
                         model_calls_attempted=2, model_calls_completed=1))
    assert_count(actual, "input_tokens", None, 11, False)
    assert_count(actual, "output_tokens", None, 5, False)
    assert_count(actual, "model_calls_attempted", 2, 2, True)
    assert_count(actual, "model_calls_completed", 1, 1, True)
    assert actual["token_usage_complete"] is False
    assert actual["model_calls_complete"] is True


@pytest.mark.parametrize("event", [
    {"error": "synthetic failure"},
    {"partial_response": {"prompt_eval_count": 12, "eval_count": 3}},
    {"response": {"error": "synthetic server failure"}},
])
def test_only_failed_call_never_becomes_exact_zero_token_cost(event):
    actual = summary(row(input_tokens=0, output_tokens=0, trace=[event],
                         model_calls_attempted=1, model_calls_completed=0))
    assert_count(actual, "input_tokens", None, 0, False)
    assert_count(actual, "output_tokens", None, 0, False)
    assert_count(actual, "model_calls_attempted", 1, 1, True)
    assert_count(actual, "model_calls_completed", 0, 0, True)


@pytest.mark.parametrize("name", ["model_calls_attempted", "model_calls_completed"])
@pytest.mark.parametrize("bad", [None, -1, True, False, "1", 1.0, 5])
def test_bad_or_conflicting_call_declaration_keeps_observation(name, bad):
    actual = summary(row(**{name: bad}))
    assert_count(actual, name, None, 1, False)
    assert actual["model_calls_complete"] is False


def test_unlogged_attempt_blocks_token_exactness():
    actual = summary(row(model_calls_attempted=2, model_calls_completed=2))
    assert_count(actual, "input_tokens", None, 11, False)
    assert_count(actual, "model_calls_attempted", None, 1, False)


@pytest.mark.parametrize("bad_response", [None, True, "invalid", []])
def test_malformed_response_does_not_certify_zero_completed_calls(bad_response):
    actual = summary(row(trace=[{"response": bad_response}],
                         model_calls_completed=0, input_tokens=0, output_tokens=0))
    assert_count(actual, "model_calls_attempted", 1, 1, True)
    assert_count(actual, "model_calls_completed", None, 0, False)
    assert_count(actual, "input_tokens", None, 0, False)


def test_unknown_row_does_not_hide_known_group_costs():
    unknown = row()
    for name in ("trace", "input_tokens", "output_tokens", "token_usage_complete",
                 "model_calls_attempted", "model_calls_completed"):
        del unknown[name]
    actual = summary(row(), unknown)
    assert_count(actual, "input_tokens", None, 11, False)
    assert_count(actual, "output_tokens", None, 5, False)
    assert_count(actual, "model_calls_attempted", None, 1, False)


def test_no_trace_inconsistent_calls_and_unknown_calls():
    value = row(model_calls_attempted=1, model_calls_completed=2)
    del value["trace"]
    actual = summary(value)
    assert_count(actual, "model_calls_attempted", None, 2, False)
    assert_count(actual, "model_calls_completed", None, 2, False)
    assert_count(actual, "input_tokens", None, 11, False)
    value = row()
    for name in ("trace", "model_calls_attempted", "model_calls_completed"):
        del value[name]
    actual = summary(value)
    assert_count(actual, "model_calls_attempted", None, 0, False)
    assert_count(actual, "model_calls_completed", None, 0, False)


def test_summary_is_json_safe_and_does_not_mutate_or_change_score_policy():
    value = row(status="JUDGED_WITHHELD_AUDIT", input_tokens=True)
    before = copy.deepcopy(value)
    actual = REPORT.summarize([value])
    assert value == before
    assert actual["table"][0]["score"] is None
    assert actual["table"][0]["score_status"] == "WITHHELD_AUDIT"
    assert json.loads(json.dumps(actual, allow_nan=False)) == actual
    assert REPORT.summarize([])["table"] == []
