from dataclasses import replace
from json import loads

import pytest

from experiments.memory.iterative_text_competence import (
    MAX_INPUT_TOKENS, read_iterative_text_competence, run_iterative_text_competence,
)
from experiments.memory.source_revision_feed import write_source_delta
from experiments.memory.task_episode import PolicyReply
from experiments.memory.task_evidence import TaskBudget, TaskWorld


class _Tokenizer:
    def apply_chat_template(self, messages, **_options):
        return list("\n".join(row["content"] for row in messages).encode())


def _world(path, changes, *, previous=None, previous_world=None):
    delta = write_source_delta(path, changes, previous=previous, max_bytes=65536)
    world = TaskWorld.from_source_epoch(path, delta.seal, max_feed_bytes=65536, max_source_bytes=65536,
                                        max_passages=16, previous=previous_world)
    return world, delta


def _run(task, world, calls, path, *, tokenizer=None, budget=None):
    artifact = run_iterative_text_competence(
        task, world, _Tokenizer() if tokenizer is None else tokenizer, calls, path,
        backend_id="scripted-development-adapter", tokenizer_id="development-byte-tokenizer",
        budget=TaskBudget(candidates_per_search=1) if budget is None else budget,
    )
    return artifact, read_iterative_text_competence(artifact)


def test_four_calls_follow_three_new_aliases_and_keep_withdrawal_visible(tmp_path):
    a, b, c, d = (character * 32 for character in "abcd")
    documents = {
        a: b"Parcel Wren uses consignment Neven.",
        b: b"Manifest Neven delegates to docket Fenlock.",
        c: b"Docket Fenlock assigns the Lantern vault.",
        d: b"Lantern vault is closed. Use Saffron annex instead.",
    }
    world, delta = _world(tmp_path / "opening", documents)
    withdrawn, _ = _world(tmp_path / "withdrawn", {d: None}, previous=delta, previous_world=world)
    task = "Where should parcel Wren go?"

    def check(world, name, last_answer, last_span):
        clicks = []

        def generate(call):
            frame = loads(call.messages[-1][1])
            clicks.append(frame)
            assert frame["task"] == task
            opened = tuple(row["text"] for row in frame["passages"])
            prefixes = (a, b, c)
            turn = len(clicks)
            assert call.max_new_tokens == 512 and len(call.input_ids) <= MAX_INPUT_TOKENS
            if turn <= 3:
                assert opened == tuple(documents[key].decode() for key in prefixes[:turn])
                assert frame["searches"][-1]["new_passages"] == 1
                return PolicyReply(("Neven", "Fenlock", "Lantern")[turn - 1], 3, True)
            assert opened == tuple(documents[key].decode() for key in (*prefixes, *last_span))
            assert set(frame) == {"task", "passages", "searches"}
            assert frame["searches"][-1]["new_passages"] == len(last_span)
            return PolicyReply(last_answer, 32, True)

        artifact, rows = _run(task, world, generate, tmp_path / name)
        assert artifact.status == "submitted"
        assert tuple(row["data"]["query"] for row in rows if row["kind"] == "search") == (
            task, "Neven", "Fenlock", "Lantern",
        )
        assert rows[-1]["data"]["calls"] == 4
        assert rows[-1]["data"]["lexical_rows_scored"] == 4 * (3 + len(last_span))
        assert rows[-1]["data"]["source_bytes_read"] == sum(len(documents[key]) for key in (a, b, c, *last_span))
        assert rows[-2]["kind"] == "response" and rows[-2]["data"]["text"] == last_answer
        assert rows[0]["data"]["source_epoch_seal"] == world.source_seal
        return artifact

    check(world, "opening.jsonl", "Use Saffron [r1] [r2] [r3] [r4].", (d,))
    check(withdrawn, "withdrawn.jsonl",
          "The docket names Lantern [r1] [r2] [r3]. Current availability is unresolved.", ())


def test_stop_preserves_first_answer_and_rejects_a_forged_handle(tmp_path):
    world, _ = _world(tmp_path / "opening", {"a" * 32: b"The Ember gate issues amber badges."})
    for handle, status in (("r1", "submitted"), ("r9", "invalid_citations")):
        replies = iter((PolicyReply("STOP", 2, True), PolicyReply(f"Amber [{handle}].", 6, True)))
        artifact, rows = _run("What badge does the Ember gate issue?", world, lambda _: next(replies),
                              tmp_path / f"{handle}.jsonl")
        assert artifact.status == status
        assert rows[-1]["data"]["calls"] == 2
        assert tuple(row["data"]["phase"] for row in rows if row["kind"] == "request") == ("search", "answer")
        assert rows[-2]["data"]["text"] == f"Amber [{handle}]."
        assert sum(row["kind"] == "search" for row in rows) == 1
        with pytest.raises(FileExistsError):
            _run("What badge does the Ember gate issue?", world, lambda _: next(replies), artifact.path)
    altered = tmp_path / "altered.jsonl"
    altered.write_bytes(artifact.path.read_bytes().replace(b"Amber [r9].", b"Green [r9]."))
    with pytest.raises(ValueError, match="digest differs"):
        read_iterative_text_competence(replace(artifact, path=altered))


def test_limits_refuse_before_dispatch_and_keep_incomplete_backend_usage(tmp_path):
    world, _ = _world(tmp_path / "opening", {"a" * 32: b"The Ember gate issues amber badges."})

    class LongTokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return [1] * (MAX_INPUT_TOKENS + 1)

    artifact, rows = _run("What badge does the Ember gate issue?", world,
                          lambda call: pytest.fail("oversize prompt dispatched"), tmp_path / "over.jsonl",
                          tokenizer=LongTokenizer())
    assert artifact.status == "input_limit" and rows[-1]["data"]["calls"] == 0
    assert rows[-2]["data"]["input_tokens"] == MAX_INPUT_TOKENS + 1

    def broken_backend(_call):
        raise RuntimeError("simulated interruption")

    artifact, rows = _run("What badge does the Ember gate issue?", world, broken_backend,
                          tmp_path / "broken.jsonl")
    assert artifact.status == "backend_error" and rows[-1]["data"]["calls"] == 1
    assert rows[-1]["data"]["input_tokens"] > 0
    assert rows[-1]["data"]["output_usage_complete"] is False


def test_invalid_query_keeps_its_first_reply_and_answers_without_retry(tmp_path):
    world, _ = _world(tmp_path / "opening", {"a" * 32: b"The Ember gate issues amber badges."})
    first_query = "SEARCH\nAGAIN"
    answer = "There is no verified evidence about Cedar's badge."
    replies = iter((PolicyReply(first_query, 3, True), PolicyReply(answer, 13, True)))
    artifact, rows = _run("Where is Cedar?", world, lambda _: next(replies), tmp_path / "unresolved.jsonl")
    assert artifact.status == "submitted"
    assert [row["data"]["text"] for row in rows if row["kind"] == "response"] == [first_query, answer]
    assert sum(row["kind"] == "query_refusal" for row in rows) == 1
    assert rows[-1]["data"]["calls"] == 2 and rows[-1]["data"]["verified_passages"] == 0

    artifact, rows = _run("Where is Cedar?", world, lambda _: PolicyReply("STOP", 0, True),
                          tmp_path / "bad-usage.jsonl")
    assert artifact.status == "output_contract_violation" and rows[-1]["data"]["calls"] == 1
    assert rows[-1]["data"]["output_usage_complete"] is False


def test_answer_sees_when_a_matched_passage_could_not_be_opened(tmp_path):
    world, _ = _world(tmp_path / "opening", {
        "a" * 32: b"Ember gate issues amber badges, according to the old calendar.",
        "b" * 32: b"Ember gate issues green badges, according to the new calendar.",
    })

    def generate(call):
        frame = loads(call.messages[-1][1])
        assert len(frame["passages"]) == 1
        assert frame["searches"][0]["matched"] == 2
        assert frame["searches"][0]["read_limits_or_errors"] == {"step_limit": 1}
        return PolicyReply("One record [r1] was read. The second remains unresolved.", 16, True)

    artifact, rows = _run("Ember gate badge", world, generate, tmp_path / "bounded.jsonl",
                          budget=TaskBudget(candidates_per_search=2, steps=2))
    assert artifact.status == "submitted"
    assert rows[-1]["data"]["calls"] == 1 and rows[-1]["data"]["steps"] == 2
