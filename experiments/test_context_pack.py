"""Step-one operation contracts, not a quality benchmark or simulation evaluation."""
import asyncio
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.context_pack import OutputMeter
from experiments.latent_context import ContextMemory, render
from experiments.latent_context_app import MemoryHost, mcp_server, startup_lock
from experiments.latent_mcp import serialized
from experiments.test_latent_mcp import corpus, put, sha


@pytest.fixture(autouse=True)
def cpu_floor(monkeypatch):
    monkeypatch.setenv("LSR_CPU_RAM_FLOOR_GIB", "10")


def spans(pack):
    return [p for g in pack["evidence_groups"] for p in g["spans"]]


def replace_first_leaf(corpus, text, chunk):
    config, source, index = corpus
    raw = text.encode()
    (source / "origin.md").write_bytes(raw)
    (index / "leaves" / (sha(raw) + ".txt")).write_bytes(raw)
    a = raw.index(chunk.encode())
    rows = json.loads((index / "rows.json").read_text())
    rows[0].update(source_sha256=sha(raw), leaf_sha256=sha(raw), start_byte=a,
                   end_byte=a + len(chunk.encode()), span_sha256=sha(chunk.encode()), text=chunk)
    put(index / "rows.json", rows)
    manifest = json.loads((index / "manifest.json").read_text())
    manifest["files"][0].update(sha256=sha(raw), bytes=len(raw))
    manifest["rows_sha256"] = sha((index / "rows.json").read_bytes())
    put(index / "manifest.json", manifest)
    cfg = json.loads(config.read_text())
    cfg["manifest_sha256"] = sha((index / "manifest.json").read_bytes())
    put(config, cfg)


def test_context_pack_contract_and_expanded_span_adapters(corpus):
    memory = ContextMemory(corpus[0])
    pack = memory.context("Alpha", readings=["named source", "alternative scope"])
    assert pack["strategy"] == "one_shot" and len(pack["snapshot"]) == 64
    assert pack["proposed_joins"] == pack["conflicts"] == []
    assert pack["analysis_status"]["conflicts"] == "not computed"
    assert pack["coverage"]["semantic_coverage"] == "not computed"
    assert pack["coverage"]["not_in_memory"] == "UNKNOWN"
    assert pack["costs"]["verification_bytes_read"] > 0
    assert pack["costs"]["corpus_encodes"] == pack["costs"]["reader_model_calls"] == 0
    for p in spans(pack):
        assert p["byte_verified"] and p["semantic_support"] == "READER_MUST_ASSESS"
        assert p["snapshot"] == pack["snapshot"]
        assert p["source_version"] == p["source_sha256"]
        assert p["coordinate_space"] == "EXTRACTED_LEAF_UTF8"
        assert p["text"].encode() == (corpus[1] / p["file"]).read_bytes()
        assert sha(p["text"].encode()) == p["sha256"]
        assert memory.open_span(p["id"])["passages"][0]["text"] == p["text"]
    origin = next(p for p in spans(pack) if p["file"] == "origin.md")
    assert memory.follow_reference(origin["id"])["passages"][0]["file"] == "target.md"


def test_executable_cursor_drains_without_rerouting_and_is_replayable(corpus, monkeypatch):
    memory = ContextMemory(corpus[0])
    first = memory.context("Alpha", corpus="test", budget={"max_spans": 1})
    args = first["continuation"]["arguments"]
    assert args["snapshot"] == first["snapshot"]
    monkeypatch.setattr(memory, "_propose", lambda *a: pytest.fail("resume rerouted"))
    second = memory.context(**args)
    assert len(spans(second)) == 1
    assert spans(second)[0]["id"] != spans(first)[0]["id"]
    assert second["continuation"] is None
    assert spans(memory.context(**args)) == spans(second)
    assert second["costs"]["query_encoder_calls"] == 0


@pytest.mark.parametrize("change", [
    {"snapshot": "0" * 64}, {"question": "Different"}, {"corpus": "another"},
    {"readings": ["another"]}, {"intent": "explore"}, {"continuation": "ctx1_forged"},
])
def test_cursor_binding_refuses(corpus, change):
    memory = ContextMemory(corpus[0])
    first = memory.context("Alpha", budget={"max_spans": 1})
    args = dict(first["continuation"]["arguments"], **change)
    with pytest.raises(ValueError, match="mismatch|continuation"):
        memory.context(**args)


def test_cursor_rejects_changed_source_and_expired_server_state(corpus):
    memory = ContextMemory(corpus[0])
    first = memory.context("Alpha", budget={"max_spans": 1})
    args = first["continuation"]["arguments"]
    with pytest.raises(ValueError, match="continuation"):
        ContextMemory(corpus[0]).context(**args)
    (corpus[1] / "target.md").write_text("Changed")
    with pytest.raises(ValueError, match="changed"):
        memory.context(**args)


@pytest.mark.parametrize("strategy", ["one_shot", "ppr", "expand"])
def test_complete_table_including_headers_units_and_final_row(corpus, strategy):
    table = "# Alpha revenue\nUSD thousands; fiscal 2025\n\n| Product | Value |\n|---|---|\n| Alpha | 29 |\n| Final | 42 |\n\nFootnote: gross amounts.\n"
    replace_first_leaf(corpus, table, "| Alpha | 29 |")
    memory = ContextMemory(corpus[0])
    pack = memory.context("Alpha", strategy=strategy)
    p = next(p for p in spans(pack) if p["file"] == "origin.md")
    assert p["text"] == table
    assert p["start_byte"] == 0 and p["end_byte"] == len(table.encode())
    assert p["unit"] == "COMPLETE_EXTRACTED_LEAF"
    assert p["sha256"] == sha(table.encode())


def test_oversized_table_is_explicitly_withheld_not_clipped(corpus):
    table = "# Alpha\nUSD thousands\n| Alpha | 29 |\n" + "| Final | 42 |\n" * 20000
    replace_first_leaf(corpus, table, "| Alpha | 29 |")
    pack = ContextMemory(corpus[0]).context("Alpha")
    assert not any(p["file"] == "origin.md" for p in spans(pack))
    assert pack["unresolved"][0]["reason"] == "atomic_leaf_exceeds_output_budget"
    assert pack["coverage"]["completeness"] == "UNKNOWN"
    assert pack["continuation"] is None  # no endless cursor returning the same blocked leaf


def test_cross_leaf_table_pages_travel_atomically(corpus):
    config, source, index = corpus
    pages = ["Alpha table; USD thousands\n| Product | Revenue |\n| A | 29 |\n",
             "Alpha table continued\n| B | 42 |\nEnd of table\n"]
    raw = "".join(pages).encode()
    (source / "origin.md").write_bytes(raw)
    rows = json.loads((index / "rows.json").read_text())
    prototype = rows.pop(0)
    for page, text in enumerate(pages, 1):
        leaf = text.encode()
        (index / "leaves" / (sha(leaf) + ".txt")).write_bytes(leaf)
        rows.insert(page - 1, dict(prototype, source_sha256=sha(raw), page=page,
                                  leaf_sha256=sha(leaf), start_byte=0, end_byte=len(leaf),
                                  span_sha256=sha(leaf), text=text))
    put(index / "rows.json", rows)
    np.save(index / "vectors.npy", np.ones((len(rows), 3), dtype=np.float32))
    manifest = json.loads((index / "manifest.json").read_text())
    manifest["files"][0].update(sha256=sha(raw), bytes=len(raw))
    manifest.update(rows_sha256=sha((index / "rows.json").read_bytes()),
                    vector_sha256=sha((index / "vectors.npy").read_bytes()))
    put(index / "manifest.json", manifest)
    cfg = json.loads(config.read_text())
    cfg["manifest_sha256"] = sha((index / "manifest.json").read_bytes())
    put(config, cfg)
    memory = ContextMemory(config)
    pack = memory.context("Alpha")
    assert [p["text"] for p in spans(pack) if p["file"] == "origin.md"] == pages
    tiny = memory.context("Alpha", budget={"max_spans": 1})
    assert not any(p["file"] == "origin.md" for p in spans(tiny))
    assert tiny["unresolved"][0]["source_leaves"] == 2


@pytest.mark.parametrize("budget", [
    {"bytes": 0}, {"tokens": 6001}, {"max_spans": True}, {"unknown": 2}, [],
])
def test_invalid_budgets(corpus, budget):
    with pytest.raises(ValueError):
        ContextMemory(corpus[0]).context("Alpha", budget=budget)


def test_wire_and_human_output_caps_and_tiny_metadata_refusal(corpus):
    memory = ContextMemory(corpus[0])
    pack = memory.context("Alpha", budget=dict(bytes=12000, tokens=4000, max_spans=1))
    assert len(serialized(pack).encode()) <= 12000
    assert memory.meter.tokens(serialized(pack)) <= 4000
    assert len(render(pack).encode()) <= 12000
    assert memory.meter.tokens(render(pack)) <= 4000
    with pytest.raises(ValueError, match="budget"):
        memory.context("Alpha " * 600, budget=dict(bytes=2048, tokens=512))


def test_no_cached_tokenizer_uses_byte_upper_bound(monkeypatch, tmp_path):
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    meter = OutputMeter()
    assert meter.name == "utf8_bytes_upper_bound"
    assert meter.tokens("界\\\n") == len("界\\\n".encode())


def test_punctuation_only_question_terminates_without_semantic_absence(corpus):
    pack = ContextMemory(corpus[0]).context("???")
    assert pack["coverage"]["not_in_memory"] == "UNKNOWN"
    assert pack["stop_reason"] == "proposal_list_exhausted"


def test_warm_startup_lock_serializes_and_releases_after_error(tmp_path):
    path = tmp_path / "startup.lock"
    with pytest.raises(RuntimeError):
        with startup_lock(path):
            with pytest.raises(TimeoutError):
                with startup_lock(path, timeout=0.05):
                    pytest.fail("competing startup acquired the same lock")
            raise RuntimeError("failed cold load")
    with startup_lock(path, timeout=0.05):
        pass


def test_host_and_mcp_primary_schema(corpus, tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    put(registry, dict(test=dict(config=str(corpus[0]), source_root=str(corpus[1]),
                                  description="operation fixture")))
    host = MemoryHost(registry)
    warm = host.invoke(dict(method="warm", corpus="test"))
    pack = host.invoke(dict(method="context", corpus="test", snapshot=warm["snapshot"],
                            question="Alpha"))
    assert pack["snapshot"] == warm["snapshot"] and len(host.memories) == 1
    import experiments.latent_context_app as app
    monkeypatch.setattr(app, "call", lambda endpoint, method, **args: host.invoke(
        dict(method=method, **args)))
    server = mcp_server(dict(url="http://127.0.0.1:1"), "test")
    tools = asyncio.run(server.list_tools())
    primary = next(t for t in tools if t.name == "context")
    assert "ONCE" in primary.description
    schema = primary.input_schema
    assert "continuation" in schema["properties"] and "budget" in schema["properties"]
    assert {"latent_ask", "open_span", "follow_reference"} <= {t.name for t in tools}
    assert primary.output_schema is None
    result = asyncio.run(server.call_tool("context", {"question": "Alpha"}))
    assert not result.is_error and result.structured_content is None
    assert len(result.content) == 1
    returned = json.loads(result.content[0].text)
    assert returned["snapshot"] == warm["snapshot"]
    assert spans(returned)


def test_legacy_ask_cursor_executes_through_context(corpus):
    memory = ContextMemory(corpus[0])
    first = memory.ask("Alpha", {"k": 1})
    assert spans(memory.context(**first["continuation"]["arguments"]))


@pytest.mark.parametrize("strategy", ["one_shot", "expand"])
@pytest.mark.parametrize("format", ["json", "compact"])
def test_primary_context_stdio_has_no_duplicate_structured_copy(corpus, tmp_path, strategy, format):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    registry = tmp_path / "registry.json"
    put(registry, dict(test=dict(config=str(corpus[0]), source_root=str(corpus[1]),
                                  description="operation fixture")))
    code = (
        "import sys; from experiments import latent_context_app as app; "
        "host=app.MemoryHost(sys.argv[1]); "
        "app.call=lambda endpoint,method,**args:host.invoke(dict(method=method,**args)); "
        "app.mcp_server({},'test').run(transport='stdio')"
    )

    async def journey():
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                   PYTHONUTF8="1", LSR_CPU_RAM_FLOOR_GIB="10")
        params = StdioServerParameters(command=sys.executable,
                                       args=["-B", "-c", code, str(registry)], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("context", {"question": "Alpha", "strategy": strategy,
                                                            "format": format})
                assert not result.is_error
                assert result.structured_content is None
                assert len(result.content) == 1
                pack = json.loads(result.content[0].text)
                if format == "compact":
                    from experiments.context_compact import expand_compact
                    pack = expand_compact(pack)
                assert spans(pack) and len(result.content[0].text.encode()) <= 48000
                assert OutputMeter().tokens(result.content[0].text) <= 6000
    asyncio.run(journey())


@pytest.mark.parametrize("strategy", ["one_shot", "ppr", "expand"])
def test_structural_table_keeps_cross_page_headers_final_rows_and_footnotes(corpus, strategy):
    table = (
        "CONSOLIDATED STATEMENTS OF OPERATIONS\n(in thousands)\n2025     2024\n"
        "Alpha revenue       29       21\n\n=== PDF PAGE 2 ===\n\n"
        "CONSOLIDATED STATEMENTS OF OPERATIONS\n(in thousands; continued)\n"
        "Final row           42       32\n\n"
        "See accompanying notes to consolidated financial statements.\n"
        "(1) Alpha includes discontinued products.\n\n"
    )
    text = table + "Other Section\n\n" + "Other prose. " * 2000
    replace_first_leaf(corpus, text, "Alpha revenue       29       21")
    memory = ContextMemory(corpus[0])
    pack = memory.context("Alpha revenue table", strategy=strategy)
    found = next(s for s in spans(pack) if s.get("unit") == "COMPLETE_TABLE")
    assert found["text"] == table
    assert found["text"].count("CONSOLIDATED STATEMENTS OF OPERATIONS") == 2
    assert memory.open_span(found["id"])["passages"][0]["text"] == table
    assert sha(found["text"].encode()) == found["sha256"]
    forged = found["id"].rsplit("_", 1)[0] + "_20"
    with pytest.raises(ValueError, match="boundary"):
        memory.open_span(forged)


def test_unclosed_numeric_table_is_not_a_complete_section(corpus):
    text = ("Alpha Data\n\nUSD thousands\nAlpha revenue        29        21\n\n"
            "Other Section\n\n" + "Other prose. " * 2000)
    replace_first_leaf(corpus, text, "Alpha revenue        29        21")
    pack = ContextMemory(corpus[0]).context("Alpha Data revenue table", budget={"bytes": 12000})
    assert not any("Alpha revenue" in s["text"] for s in spans(pack))
    assert pack["unresolved"]
    item = pack["unresolved"][0]
    assert item["catalog_identity"]["role"] == "CATALOG_METADATA_NOT_EVIDENCE"
    assert item["catalog_identity"]["file"] == "origin.md"
    assert "bytes" in item["blocked_dimensions"]
    assert "--file 'origin.md'" in item["exact_reader"]
    assert "Catalog metadata, NOT evidence: origin.md" in render(pack)


def test_omission_ledger_survives_finished_cursor(corpus):
    text = ("Alpha Overflow\n\n" + "Alpha " * 5000 +
            "\n\nAlpha One\n\nAlpha first result.\n\nAlpha Two\n\nAlpha second result.\n")
    replace_first_leaf(corpus, text, "Alpha first result.")
    memory = ContextMemory(corpus[0])
    first = memory.context("Alpha", budget={"bytes": 12000, "max_spans": 1})
    assert first["unresolved"] and first["continuation"]
    pack = first
    for _ in range(6):
        if not pack["continuation"]:
            break
        pack = memory.context(**pack["continuation"]["arguments"])
    assert pack["continuation"] is None
    recovered = memory.context(**pack["omission_ledger"]["recovery"]["arguments"])
    assert recovered["unresolved"] == first["unresolved"]
    assert "origin.md" in render(recovered)
    assert "Omission ledger" in render(pack)


FINANCE_QUESTION = (
    "Compare fiscal 2025 and 2024 revenue, net income and operating cash flow; "
    "include the complete financial table and units."
)
FINANCE_BUDGET = dict(bytes=12000, tokens=4000, max_spans=1)


def drain(memory, pack):
    pages = []
    for _ in range(2000):
        assert memory.meter.fits(pack, FINANCE_BUDGET)
        pages.append(pack)
        if pack["continuation"] is None:
            return pages
        assert pack["continuation"]["arguments"]["budget"] == FINANCE_BUDGET
        pack = memory.context(**pack["continuation"]["arguments"])
    pytest.fail("nonprogressing continuation")


def test_fixed_budget_finance_chain_and_stable_omission_recovery(corpus, monkeypatch):
    # More pages than the page-cursor retention cap, and enough omission records
    # to overflow the old cumulative inline ledger.
    text = ("Revenue\n\n" + "revenue " * 2000 + "\n\n") * 35
    text += ("Revenue\n\nFiscal 2025 and 2024 revenue, net income and operating cash flow; "
             "complete financial table and units.\n\n") * 35
    replace_first_leaf(corpus, text, "Fiscal 2025 and 2024 revenue")
    memory = ContextMemory(corpus[0])
    first = memory.context(FINANCE_QUESTION, budget=FINANCE_BUDGET)
    root = first["omission_ledger"]["recovery"]["arguments"]
    monkeypatch.setattr(memory, "_propose", lambda *a: pytest.fail("continuation rerouted"))
    pages = drain(memory, first)
    assert len(pages) > 32
    assert all(p["omission_ledger"]["recovery"]["arguments"] == root for p in pages)
    inline = [item for p in pages for item in p["unresolved"]]
    assert len(inline) == 35
    assert pages[-1]["omission_ledger"]["total"] == 35
    assert pages[-1]["truncated"]
    returned = [s["id"] for p in pages for s in spans(p)]
    assert len(returned) == len(set(returned)) == 36  # 35 units plus the second fixture file
    replay = memory.context(**pages[-2]["continuation"]["arguments"])
    assert replay["unresolved"] == pages[-1]["unresolved"]
    assert replay["omission_ledger"]["total"] == 35
    recovered = drain(memory, memory.context(**pages[-1]["omission_ledger"]["recovery"]["arguments"]))
    assert [item for p in recovered for item in p["unresolved"]] == inline
    assert len(recovered) > 1
    assert all(not spans(p) and p["costs"]["query_encoder_calls"] == 0 for p in recovered)
    assert recovered[-1]["stop_reason"] == "omission_ledger_exhausted"
    assert memory.context(**root)["unresolved"] == recovered[0]["unresolved"]
    for change in (dict(question="another"), dict(snapshot="0" * 64),
                   dict(corpus="another"), dict(intent="explore"),
                   dict(strategy="ppr"), dict(readings=["another"])):
        with pytest.raises(ValueError, match="mismatch"):
            memory.context(**dict(root, **change))
    with pytest.raises(ValueError, match="continuation"):
        ContextMemory(corpus[0]).context(**root)
    token = root["continuation"]
    _, binding, ledger = memory.continuations.ledgers[token]
    memory.continuations.ledgers[token] = (0, binding, ledger)
    with pytest.raises(ValueError, match="expired"):
        memory.context(**root)


@pytest.mark.skipif(not os.environ.get("LSR_CONTEXT_INTEGRATION_CONFIG"),
                    reason="requires an explicitly approved prebuilt CPU corpus")
def test_registered_finance_fixed_budget_chain():
    memory = ContextMemory(Path(os.environ["LSR_CONTEXT_INTEGRATION_CONFIG"]))
    pages = drain(memory, memory.context(FINANCE_QUESTION, budget=FINANCE_BUDGET))
    inline = [item for p in pages for item in p["unresolved"]]
    assert len(pages) > 14 and len(inline) > 9
    assert pages[-1]["omission_ledger"]["total"] == len(inline)
    recovered = drain(memory, memory.context(**pages[-1]["omission_ledger"]["recovery"]["arguments"]))
    assert [item for p in recovered for item in p["unresolved"]] == inline


@pytest.mark.parametrize("heading", ["Notes", "Footnotes", "NOTES", "Notes to the table:"])
def test_headed_local_footnote_stays_with_complete_table(corpus, heading):
    table = (
        "CONSOLIDATED STATEMENTS OF OPERATIONS\n(in thousands)\n"
        "2025     2024\nRevenue      100     90\n\n"
        "See accompanying notes to consolidated financial statements.\n\n"
        + heading + "\n\n(1) Revenue includes discontinued operations.\n\n"
    )
    text = table + "Other Section\n\n" + "Other prose. " * 2000
    replace_first_leaf(corpus, text, "Revenue      100     90")
    memory = ContextMemory(corpus[0])
    pack = memory.context("Revenue table", budget=FINANCE_BUDGET)
    found = next(s for s in spans(pack) if s.get("unit") == "COMPLETE_TABLE")
    assert found["text"] == table
    assert memory.open_span(found["id"])["passages"][0]["text"] == table
    assert found["sha256"] == sha(table.encode())
    assert not any(s.get("section_title") == heading for s in spans(pack))


def test_full_financial_notes_chapter_is_not_a_local_table_note(corpus):
    from experiments.context_spans import section_ranges
    table = (
        "CONSOLIDATED STATEMENTS OF OPERATIONS\n(in thousands)\n"
        "2025     2024\nRevenue      100     90\n\n"
        "See accompanying notes to consolidated financial statements.\n\n"
    )
    chapter = "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS\n\nAccounting policies.\n\n"
    ranges = section_ranges(table + chapter)
    assert ranges[0][:3] == (0, len(table.encode()), "COMPLETE_TABLE")
    assert ranges[1][3] == "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS"


def test_mcp_advertises_before_loading_encoder(monkeypatch):
    import experiments.latent_context_app as app
    def no_start(**kwargs):
        pytest.fail("listing tools started the corpus/encoder")
    monkeypatch.setattr(app, "ensure_server", no_start)
    server = mcp_server(None, "datadog")
    tools = asyncio.run(server.list_tools())
    assert "Datadog" in next(t for t in tools if t.name == "context").description


def test_mcp_budget_schema_and_actionable_refusal(monkeypatch):
    from mcp.server.mcpserver.exceptions import ToolError
    import experiments.latent_context_app as app

    server = mcp_server({}, "test")
    primary = next(t for t in asyncio.run(server.list_tools()) if t.name == "context")
    budget = primary.input_schema["$defs"]["ContextBudget"]["properties"]
    assert (budget["tokens"]["minimum"], budget["tokens"]["maximum"]) == (512, 6000)
    assert (budget["bytes"]["minimum"], budget["bytes"]["maximum"]) == (2048, 48000)
    assert (budget["max_spans"]["minimum"], budget["max_spans"]["maximum"]) == (1, 16)
    with pytest.raises(ToolError, match="6000"):
        asyncio.run(server.call_tool("context", {"question": "Alpha", "budget": {"tokens": 6500}}))

    def refuse(*args, **kwargs):
        raise RuntimeError("ValueError: budget too small to report oversized leaves")

    monkeypatch.setattr(app, "call", refuse)
    with pytest.raises(ToolError, match="Omit budget"):
        asyncio.run(server.call_tool("context", {"question": "Alpha", "budget": {"tokens": 2200}}))


def test_windows_host_denied_breakaway_retries_inside_host_job(monkeypatch):
    import experiments.latent_context_app as app
    flag = 0x01000000
    monkeypatch.setattr(app.subprocess, "CREATE_BREAKAWAY_FROM_JOB", flag, raising=False)
    attempts = []
    process = object()
    def spawn(command, **options):
        attempts.append(options)
        if len(attempts) == 1:
            error = PermissionError("host prohibits breakaway")
            error.winerror = 5
            raise error
        return process
    monkeypatch.setattr(app.subprocess, "Popen", spawn)
    result, lifetime = app.launch_service(["python"], creationflags=flag | 512, cwd="approved")
    assert result is process and lifetime == "client_job_fallback"
    assert attempts[0]["creationflags"] == flag | 512
    assert attempts[1] == dict(creationflags=512, cwd="approved")


def test_service_spawn_does_not_retry_unrelated_permission_errors(monkeypatch):
    import experiments.latent_context_app as app
    def spawn(*args, **options):
        raise PermissionError("not a Windows breakaway denial")
    monkeypatch.setattr(app.subprocess, "Popen", spawn)
    with pytest.raises(PermissionError):
        app.launch_service(["python"], creationflags=0)
