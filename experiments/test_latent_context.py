"""Software contracts. Real corpus/browser journeys are recorded separately."""
import json
import os

import pytest

from experiments.latent_context import ContextMemory, MAX_PACK_BYTES, render
from experiments.latent_context_app import MemoryHost, call, resolve_corpus
from experiments.latent_mcp import serialized
from experiments.test_latent_mcp import corpus, sha


@pytest.fixture(autouse=True)
def cpu_floor(monkeypatch):
    monkeypatch.setenv("LSR_CPU_RAM_FLOOR_GIB", "10")


def spans(pack):
    return [span for group in pack["facets"] for span in group["spans"]]


def test_pack_has_verified_exact_quotes_and_real_reference_walk(corpus):
    memory = ContextMemory(corpus[0])
    pack = memory.ask("Alpha", dict(k=2))
    assert pack["model_calls"] == pack["model_tokens"] == 0
    assert any(step["operation"] == "reference_hop" for step in pack["trace"])
    assert {p["file"] for p in spans(pack)} == {"origin.md", "target.md"}
    for p in spans(pack):
        raw = (corpus[1] / p["file"]).read_bytes()
        assert raw[p["start_byte"]:p["end_byte"]].decode("utf-8") == p["text"]
        assert sha(p["text"].encode()) == p["sha256"] and p["verified"]
    assert pack["coverage"]["not_in_memory"] == "UNKNOWN"
    assert isinstance(pack["ambiguity"]["note"], str)
    assert isinstance(pack["related_topics"], list)
    assert isinstance(pack["next_steps"], list)
    assert "generated answer" in render(pack)


def test_neighbors_are_source_verified_not_invented(corpus):
    memory = ContextMemory(corpus[0])
    result = memory.neighbors(memory.tool.corpus_id + ":S0")
    assert result["passages"][0]["file"] == "target.md"
    assert result["passages"][0]["relation"] == "vector_similarity"
    assert result["passages"][0]["verified"]


def test_working_context_reverified_on_followup_and_restart(corpus):
    first = ContextMemory(corpus[0])
    id = spans(first.ask("Alpha", dict(k=1)))[0]["id"]
    second = ContextMemory(corpus[0])
    pack = second.ask("value", dict(working_ids=[id], previous_question="Alpha", k=2))
    assert spans(pack)[0]["retained_working_context"]
    assert pack["trace"][0]["operation"] == "reverify_working_context"
    (corpus[1] / "origin.md").write_text("Changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        second.ask("value", dict(working_ids=[id]))


@pytest.mark.parametrize("question,options", [
    ("", {}), ("x" * 4097, {}), ("Alpha", {"k": True}), ("Alpha", {"k": 0}),
    ("Alpha", {"k": 17}), ("Alpha", {"hops": 3}), ("Alpha", {"unknown": 3}),
    ("Alpha", {"working_ids": ["x"] * 17}), ("Alpha", {"brief": True}),
    ("Alpha", {"working_ids": [4]}),
])
def test_invalid_pack_inputs_refuse(corpus, question, options):
    with pytest.raises(ValueError):
        ContextMemory(corpus[0]).ask(question, options)


def test_duplicate_proposals_removed_and_bounded(corpus):
    pack = ContextMemory(corpus[0]).ask("Alpha and target", dict(k=16))
    assert len({p["sha256"] for p in spans(pack)}) == len(spans(pack))
    assert pack["coverage"]["duplicate_proposals_removed"] > 0
    assert len(serialized(pack).encode("utf-8")) <= MAX_PACK_BYTES


def test_no_match_never_becomes_semantic_absence(corpus):
    pack = ContextMemory(corpus[0]).ask("zzzxylophone absentcanary", dict(k=1))
    assert pack["coverage"]["not_in_memory"] == "UNKNOWN"
    assert "absentcanary" in pack["coverage"]["thin_or_unseen_query_terms"]
    assert "absentcanary" not in spans(pack)[0]["text"]


def test_host_reuses_backend_and_refuses_changed_config(corpus, tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(dict(test=dict(
        config=str(corpus[0]), source_root=str(corpus[1]), description="fixture"))))
    host = MemoryHost(registry)
    first = host.invoke(dict(method="ask", corpus="test", question="Alpha"))
    second = host.invoke(dict(method="ask", corpus="test", question="target"))
    assert first["corpus_id"] == second["corpus_id"] and len(host.memories) == 1
    corpus[0].write_text(corpus[0].read_text() + " ")
    with pytest.raises(ValueError, match="configuration changed"):
        host.invoke(dict(method="warm", corpus="test"))


def test_custom_paths_require_explicit_approved_config(corpus, tmp_path):
    assert resolve_corpus(str(corpus[0]), {})["source_root"] == str(corpus[1])
    with pytest.raises(ValueError, match="approved"):
        resolve_corpus(str(tmp_path), {})


@pytest.mark.parametrize("url", ["https://example.com", "http://localhost:1",
                                "http://127.0.0.1:1/remote", "http://user@127.0.0.1:1"])
def test_client_never_sends_question_to_remote_endpoint(url):
    with pytest.raises(ValueError, match="loopback"):
        call(dict(url=url), "ask", question="private")
