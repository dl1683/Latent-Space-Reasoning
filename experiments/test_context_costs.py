"""Independent boundary probes for both pilot strategies, including failed work."""
import asyncio
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.context_costs import CURRENT, MeteredRows, counters
from experiments.latent_context import ContextMemory
from experiments.latent_context_app import MemoryHost, mcp_server
from experiments.latent_mcp import serialized
from experiments.test_expand_context import bridge_corpus
from experiments.test_latent_mcp import corpus, put


@pytest.mark.parametrize("strategy", ["one_shot", "expand"])
@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("bridge_corpus", [True], indirect=True)
def test_actual_rows_opens_hashes_and_failed_work(bridge_corpus, monkeypatch, strategy, corrupt):
    memory, source, index = bridge_corpus
    actual = dict(rows=0, iterations=0, source=0, leaf=0, hash_checks=0, hash_bytes=0)

    class Rows(MeteredRows):
        def __iter__(self):
            for row in super().__iter__():
                actual["rows"] += 1
                actual["iterations"] += 1
                yield row

        def __getitem__(self, key):
            result = super().__getitem__(key)
            actual["rows"] += len(result) if isinstance(key, slice) else 1
            return result

    memory.tool.memory.rows = Rows(memory.tool.memory.rows)
    if corrupt:
        # The highest-ranked source's leaf fails after an actual open/hash.
        leaf = next((index / "leaves").glob("*"))
        leaf.write_bytes(b"!" * leaf.stat().st_size)
    original_open = Path.open
    def opened(path, *args, **kwargs):
        if path.parent == source:
            actual["source"] += 1
        elif path.parent == index / "leaves":
            actual["leaf"] += 1
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", opened)
    import experiments.dogfood as dogfood
    original_digest = dogfood.digest
    def hashed(raw):
        attempt = memory.context_last_attempt
        if attempt is not None and CURRENT.get() is attempt["access"]:
            actual["hash_checks"] += 1
            actual["hash_bytes"] += len(raw)
        return original_digest(raw)
    for name in ("experiments.dogfood", "experiments.latent_context",
                 "experiments.context_spans", "experiments.memory.explicit_reference_memory",
                 "experiments.memory.ppr_context", "experiments.memory.evidence_map"):
        monkeypatch.setattr(name + ".digest", hashed)
    def assert_measured(access):
        assert access["source_row_records_read"] == actual["rows"]
        assert access["row_iteration_visits"] == actual["iterations"]
        assert access["source_opens"] == actual["source"]
        assert access["leaf_opens"] == actual["leaf"]
        assert access["verification_hash_checks"] == actual["hash_checks"]
        assert access["verification_hash_bytes"] == actual["hash_bytes"]
    if corrupt:
        with pytest.raises(ValueError, match="changed|verification"):
            memory.context("Beacon registry", strategy=strategy)
        access = memory.context_last_attempt["access"]
        assert memory.context_last_attempt["status"] == "REFUSED"
    else:
        pack = memory.context("Beacon registry", strategy=strategy, budget={"max_spans": 1})
        access = pack["costs"]["access"]
        assert set(access) == set(counters())
        assert access["edge_reads"] > 0
        assert pack["continuation"]
        assert_measured(access)
        initial = dict(actual)
        page = memory.context(**pack["continuation"]["arguments"])
        # Continuations retain proposals, not routing work; their verification is charged anew.
        assert page["costs"]["continuation_calls"] == 1
        assert page["costs"]["access"]["query_encoder_calls"] == 0
        assert page["costs"]["access"]["exact_vector_rows_read"] == 0
        for key in actual:
            actual[key] -= initial[key]
        access = page["costs"]["access"]
    assert_measured(access)


@pytest.mark.parametrize("strategy", ["one_shot", "expand"])
def test_dense_payloads_and_encoding_match_actual_operations(bridge_corpus, strategy):
    memory, _, _ = bridge_corpus
    actual = dict(rows=0, bytes=0, encodes=0)
    class Bank(np.ndarray):
        def __matmul__(self, other):
            actual["rows"] += len(self)
            actual["bytes"] += self.nbytes
            return super().__matmul__(other)

        def __getitem__(self, key):
            value = super().__getitem__(key)
            if isinstance(key, np.ndarray):
                actual["rows"] += len(key)
                actual["bytes"] += value.nbytes
            return value

    class Encoder:
        max_seq_length = 100
        @staticmethod
        def tokenizer(text, **kwargs):
            return {"input_ids": [1]}

        @staticmethod
        def encode(texts, **kwargs):
            actual["encodes"] += 1
            return np.ones((len(texts), 3), dtype=np.float32)

    backend = memory.tool.memory
    backend.encoder = Encoder()
    backend.vectors = backend.vectors.view(Bank)
    memory._evidence_map.bank = backend.vectors
    memory.tool.routing = "hybrid"
    pack = memory.context("Beacon registry", strategy=strategy)
    access = pack["costs"]["access"]
    assert access["exact_vector_rows_read"] == actual["rows"] > 0
    assert access["exact_vector_bytes_read"] == actual["bytes"] > 0
    assert access["query_encoder_calls"] == actual["encodes"] == 1
    assert access["lexical_text_bytes_read"] > 0


@pytest.mark.parametrize("strategy", ["one_shot", "expand"])
def test_setup_retention_and_full_reader_visible_receipt(bridge_corpus, strategy):
    fixture, _, _ = bridge_corpus
    memory = ContextMemory(fixture.config)
    first = memory.context("Beacon registry", strategy=strategy)
    second = memory.context("Beacon registry", strategy=strategy)
    assert first["costs"]["setup_this_call"]["startup"]
    assert second["costs"]["setup_this_call"] == dict(startup=None, adapter=None)
    if strategy == "expand":
        assert first["costs"]["setup_this_call"]["adapter"]
        assert first["costs"]["setup_this_call"]["adapter"]["exact_vector_rows_read"] == 12
        assert second["costs"]["expand"]["setup_this_call"] is None
    assert first["costs"]["retained_setup"] == second["costs"]["retained_setup"]
    for pack in (first, second):
        receipt = pack["costs"]["reader_visible"]
        metadata = copy.deepcopy(pack)
        for group in metadata["evidence_groups"]:
            for span in group["spans"]:
                span["text"] = ""
        assert receipt["serialized_bytes"] == len(serialized(pack).encode())
        assert receipt["metadata_bytes"] == len(serialized(metadata).encode())
        assert receipt["local_tokens_estimate"] >= memory.meter.tokens(serialized(pack))
        assert receipt["quotation_bytes"] > 0 and receipt["metadata_bytes"] > 0


def test_configured_strategy_is_symmetric_for_host_and_mcp(corpus, tmp_path, monkeypatch):
    config, source, _ = corpus
    settings = json.loads(config.read_bytes())
    registry = tmp_path / "registry.json"
    put(registry, dict(test=dict(config=str(config), source_root=str(source), description="fixture")))
    for default in ("expand", "one_shot"):
        put(config, dict(settings, default_strategy=default))
        host = MemoryHost(registry)
        assert host.invoke(dict(method="warm", corpus="test"))["strategy"] == default
        server = mcp_server({}, "test")
        monkeypatch.setattr("experiments.latent_context_app.call",
            lambda endpoint, method, **args: host.invoke(dict(method=method, **args)))
        result = asyncio.run(server.call_tool("context", {"question": "Alpha"}))
        pack = json.loads(result.content[0].text)
        assert not result.is_error and pack["strategy"] == default
        assert pack["strategy_selection"] == dict(default_strategy=default, effective_strategy=default)
        ablation = host.invoke(dict(method="context", corpus="test", question="Alpha", strategy="one_shot"))
        assert ablation["strategy"] == "one_shot"
        legacy = host.invoke(dict(method="ask", corpus="test", question="Alpha", options={"hops": 0}))
        assert legacy["strategy"] == "one_shot" and not any("hop" in t["operation"] for t in legacy["trace"])
    put(config, dict(settings, default_strategy="unsupported"))
    with pytest.raises(ValueError, match="default_strategy"):
        ContextMemory(config)


def test_failed_startup_retains_attempted_access(corpus):
    config, source, _ = corpus
    (source / "origin.md").write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed") as refusal:
        ContextMemory(config)
    attempt = refusal.value.attempt_costs
    assert attempt["phase"] == "startup" and attempt["status"] == "REFUSED"
    assert attempt["access"]["source_opens"] == 1
    assert attempt["access"]["verification_hash_checks"] > 0
