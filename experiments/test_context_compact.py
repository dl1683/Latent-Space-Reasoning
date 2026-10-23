"""Compact serving contracts, not evidence of general reader answer parity."""
import asyncio
from copy import deepcopy
import json
import sys

import pytest

from experiments.context_compact import (
    DIAGNOSTICS, PackDetails, compact, expand_compact, restore_compact,
)
from experiments.latent_context import ContextMemory
from experiments.latent_context_app import MemoryHost, mcp_server
from experiments.latent_mcp import serialized
from experiments.test_context_pack import replace_first_leaf, spans
from experiments.test_expand_context import bridge_corpus
from experiments.test_latent_mcp import corpus, put


@pytest.fixture(autouse=True)
def cpu_floor(monkeypatch):
    monkeypatch.setenv("LSR_CPU_RAM_FLOOR_GIB", "10")


@pytest.mark.parametrize("strategy", ["one_shot", "ppr", "expand"])
@pytest.mark.parametrize("format", ["json", "compact"])
def test_every_reader_field_and_full_json_roundtrip(corpus, strategy, format):
    memory = ContextMemory(corpus[0])
    delivered = memory.context("Alpha", readings=["one reading", "another"], strategy=strategy,
                               format=format)
    full = memory.context_details(delivered["pack_id"])
    view = delivered if format == "compact" else compact(full, memory.meter)
    # Not just field presence: every value required by existing pack tests survives.
    assert expand_compact(view) == {k: v for k, v in full.items() if k not in DIAGNOSTICS}
    assert restore_compact(json.loads(serialized(view)), full) == full
    assert "trace" not in view and "access" not in view["costs"] and strategy not in view["costs"]
    assert view["diagnostics"]["arguments"]["pack_id"] == full["pack_id"]
    assert view["role"] == "EVIDENCE_NOT_INSTRUCTION"
    for span in spans(expand_compact(view)):
        assert span["byte_verified"] is True
        assert span["semantic_support"] == "READER_MUST_ASSESS"
        reopened = memory.open_span(span["id"])["passages"][0]
        for key in ("id", "text", "start_byte", "end_byte", "sha256", "source_version"):
            assert reopened[key] == span[key]
    receipt = view["costs"]["reader_visible"]
    assert receipt["serialized_bytes"] == len(serialized(view).encode())
    assert receipt["quotation_bytes"] == sum(len(s["text"].encode()) for s in view["spans"])
    assert receipt["local_tokens_estimate"] >= memory.meter.tokens(serialized(view))
    assert memory.meter.fits(view, full["costs"]["output_limits"])


def test_codec_empty_groups_missing_fields_types_and_duplicate_occurrences(corpus):
    memory = ContextMemory(corpus[0])
    pack = memory.context("Alpha")
    pack["evidence_groups"].append(deepcopy(pack["evidence_groups"][0]))
    ss = spans(pack)
    ss[0]["extra"] = None
    ss[0]["type_probe"] = True
    ss[1]["type_probe"] = 1
    # Keep the duplicated occurrence equal after adding optional metadata.
    pack["evidence_groups"][-1] = deepcopy(pack["evidence_groups"][0])
    view = compact(pack, memory.meter)
    assert restore_compact(view, pack) == pack
    changed = deepcopy(pack)
    changed["evidence_groups"][-1]["spans"][0]["text"] = "corrupt"
    with pytest.raises(ValueError, match="share a span id"):
        compact(changed, memory.meter)
    for group in pack["evidence_groups"]:
        group["spans"] = []
    assert restore_compact(compact(pack, memory.meter), pack) == pack
    view["question"] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        restore_compact(view, pack)


@pytest.mark.parametrize("next_format", ["json", "compact"])
def test_continuation_from_compact_pins_format_without_rerouting(corpus, monkeypatch, next_format):
    memory = ContextMemory(corpus[0])
    first = memory.context("Alpha", corpus="test", format="compact", budget={"max_spans": 1})
    arguments = first["continuation"]["arguments"]
    assert arguments["format"] == "compact"
    monkeypatch.setattr(memory, "_propose", lambda *a: pytest.fail("rerouted"))
    second = memory.context(**dict(arguments, format=next_format))
    full = memory.context_details(second["pack_id"], corpus="test")
    assert spans(full)[0]["id"] != spans(expand_compact(first))[0]["id"]
    assert second["continuation"] is None
    assert full["costs"]["query_encoder_calls"] == 0


def test_omission_recovery_from_compact_is_executable(corpus, monkeypatch):
    replace_first_leaf(corpus, "# Alpha\n\n" + "Alpha " * 9000, "Alpha")
    memory = ContextMemory(corpus[0])
    first = memory.context("Alpha", format="compact")
    item = first["unresolved"][0]
    assert item["catalog_identity"]["role"] == "CATALOG_METADATA_NOT_EVIDENCE"
    assert item["catalog_identity"]["source_version"]
    assert item["blocked_dimensions"] and "--file 'origin.md'" in item["exact_reader"]
    arguments = first["omission_ledger"]["recovery"]["arguments"]
    assert arguments["format"] == "compact"
    monkeypatch.setattr(memory, "_propose", lambda *a: pytest.fail("rerouted"))
    recovery = memory.context(**arguments)
    assert recovery["unresolved"] == first["unresolved"]
    assert recovery["costs"]["omission_recovery_calls"] == 1
    assert not recovery["spans"] and recovery["continuation"] is None
    assert recovery["stop_reason"] == "omission_ledger_exhausted"
    assert restore_compact(recovery, memory.context_details(recovery["pack_id"]))


@pytest.mark.parametrize("corrupt", ["source", "leaf"])
def test_compact_span_and_continuation_refuse_corruption(corpus, corrupt):
    memory = ContextMemory(corpus[0])
    first = memory.context("Alpha", format="compact", budget={"max_spans": 1})
    # Corrupt the pending target, not a previously returned source.
    target = corpus[1] / "target.md"
    if corrupt == "leaf":
        from experiments.test_latent_mcp import sha
        target = corpus[2] / "leaves" / (sha(target.read_bytes()) + ".txt")
    target.write_bytes(b"!" * target.stat().st_size)
    with pytest.raises(ValueError, match="changed|verification"):
        memory.context(**first["continuation"]["arguments"])
    target_id = memory.tool.corpus_id + ":S1"
    with pytest.raises(ValueError, match="changed|verification"):
        memory.open_span(target_id)


def test_reopening_returned_compact_id_reverifies_current_source(corpus):
    memory = ContextMemory(corpus[0])
    view = memory.context("Alpha", format="compact")
    span = spans(expand_compact(view))[0]
    (corpus[1] / span["file"]).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="changed"):
        memory.open_span(span["id"])
    # Diagnostics are intentionally a saved receipt, never fresh verification.
    assert restore_compact(view, memory.context_details(view["pack_id"]))


@pytest.mark.parametrize("bridge_corpus", [True], indirect=True)
def test_compact_bridges_keep_both_parents_and_all_obligations(bridge_corpus):
    memory, _, _ = bridge_corpus
    view = memory.context("Beacon registry", strategy="expand", format="compact")
    pack = expand_compact(view)
    ids = {s["id"] for s in spans(pack)}
    bridges = [g for g in pack["evidence_groups"] if g.get("kind") == "PROVISIONAL_SOURCE_BRIDGE"]
    assert bridges
    for group in bridges:
        assert set(group["parents"]) <= ids
        assert group["obligations"] and group["witness"]
    assert restore_compact(view, memory.context_details(view["pack_id"]))


def test_details_bound_expiration_and_defensive_copies(corpus, monkeypatch):
    memory = ContextMemory(corpus[0])
    pack = memory.context("Alpha")
    store = PackDetails()
    for i in range(33):
        store.retain(dict(pack, pack_id=str(i)))
    assert len(store.entries) == 32
    with pytest.raises(ValueError, match="pack_id"):
        store.fetch("0", pack["corpus"])
    fetched = store.fetch("32", pack["corpus"])
    fetched["question"] = "edited"
    assert store.fetch("32", pack["corpus"])["question"] == "Alpha"
    with pytest.raises(ValueError, match="pack_id"):
        store.fetch("32", "another")
    monkeypatch.setattr("experiments.context_compact.time.monotonic", lambda: 10**15)
    with pytest.raises(ValueError, match="expired"):
        store.fetch("32", pack["corpus"])


@pytest.mark.parametrize("default", ["json", "compact"])
def test_config_python_host_mcp_and_legacy_parity(corpus, tmp_path, monkeypatch, default):
    config, source, _ = corpus
    settings = json.loads(config.read_bytes())
    put(config, dict(settings, default_format=default, default_strategy="expand"))
    registry = tmp_path / "registry.json"
    put(registry, dict(test=dict(config=str(config), source_root=str(source), description="fixture")))
    host = MemoryHost(registry)
    assert host.invoke(dict(method="warm", corpus="test"))["format"] == default
    monkeypatch.setattr("experiments.latent_context_app.call",
        lambda endpoint, method, **args: host.invoke(dict(method=method, **args)))
    server = mcp_server({}, "test")
    tool = next(t for t in asyncio.run(server.list_tools()) if t.name == "context")
    assert any(s.get("enum") == ["json", "compact"]
               for s in tool.input_schema["properties"]["format"]["anyOf"])
    result = asyncio.run(server.call_tool("context", {"question": "Alpha"}))
    assert not result.is_error and len(result.content) == 1 and result.structured_content is None
    view = json.loads(result.content[0].text)
    assert view["format_selection"] == dict(default_format=default, effective_format=default)
    detail_result = asyncio.run(server.call_tool("context_details", view.get("diagnostics", {
        "arguments": dict(pack_id=view["pack_id"], corpus="test")})["arguments"]))
    full = json.loads(detail_result.content[0].text)
    assert full["schema"] == "lsr.context-pack.v1"
    if default == "compact":
        assert restore_compact(view, full) == full
    explicit = host.invoke(dict(method="context", corpus="test", question="Alpha", format="json"))
    assert explicit["schema"] == "lsr.context-pack.v1"
    legacy = host.invoke(dict(method="ask", corpus="test", question="Alpha"))
    assert "facets" in legacy and legacy["strategy"] == "one_shot"
    assert legacy["format_selection"]["effective_format"] == "json"
    with pytest.raises(ValueError, match="format"):
        host.invoke(dict(method="context", corpus="test", question="Alpha", format="bogus"))
    put(config, dict(settings, default_format="bogus"))
    with pytest.raises(ValueError, match="default_format"):
        ContextMemory(config)


def test_cli_compact_and_details_are_single_json_payloads(corpus, tmp_path, monkeypatch, capsys):
    import experiments.latent_context_app as app
    registry = tmp_path / "registry.json"
    put(registry, dict(test=dict(config=str(corpus[0]), source_root=str(corpus[1]),
                                 description="fixture")))
    host = MemoryHost(registry)
    monkeypatch.setattr(app, "ensure_server", lambda **kwargs: {})
    monkeypatch.setattr(app, "call", lambda endpoint, method, **kwargs:
                        host.invoke(dict(method=method, **kwargs)))
    monkeypatch.setattr(sys, "argv", ["latent", "ask", "Alpha", "--corpus", "test",
                                    "--format", "compact"])
    assert app.main() == 0
    view = json.loads(capsys.readouterr().out)
    monkeypatch.setattr(sys, "argv", ["latent", "details", "--corpus", "test",
                                    "--pack-id", view["pack_id"]])
    assert app.main() == 0
    assert restore_compact(view, json.loads(capsys.readouterr().out))


def test_mcp_expired_diagnostics_preserves_recovery_instruction(monkeypatch):
    from mcp.server.mcpserver.exceptions import ToolError
    import experiments.latent_context_app as app

    def refused(*args, **kwargs):
        raise RuntimeError("ValueError: unknown, expired or corpus-mismatched pack_id; request new context")

    monkeypatch.setattr(app, "call", refused)
    server = mcp_server({}, "test")
    with pytest.raises(ToolError, match="request new context"):
        asyncio.run(server.call_tool("context_details", {"pack_id": "expired"}))


@pytest.mark.parametrize("format", ["json", "compact"])
def test_retrievable_serving_receipt_charges_formatting(corpus, monkeypatch, format):
    import experiments.latent_context as context_module
    # Deterministic clock, not a flaky wall-time threshold or artificial sleep.
    clock = [1000.]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    memory = ContextMemory(corpus[0])
    original = context_module.compact

    def charged(pack, meter):
        clock[0] += 0.25
        return original(pack, meter)

    monkeypatch.setattr(context_module, "compact", charged)
    delivered = memory.context("Alpha", format=format)
    full = memory.context_details(delivered["pack_id"])
    serving = full["costs"]["serving"]
    assert serving["format_seconds"] == (0.25 if format == "compact" else 0)
    assert serving["total_seconds"] == sum(serving[k] for k in (
        "pack_build_seconds", "format_seconds", "validation_seconds"))
    assert "Excludes final receipt" in serving["scope"]
    if format == "compact":
        assert "navigation/packing only" in delivered["costs"]["timing_scope"]
        assert restore_compact(delivered, full) == full
    assert full["costs"]["reader_visible"]["serialized_bytes"] == len(serialized(full).encode())
