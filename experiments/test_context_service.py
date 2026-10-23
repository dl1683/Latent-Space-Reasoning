"""Direct MCP surface and cost boundaries, without reader/model calls."""
import asyncio
import io
import json
from pathlib import Path
import tomllib

import pytest

from experiments import context_costs
from experiments.latent_context import ContextMemory
from experiments.latent_context_app import MemoryHost, client_receipt, mcp_server
from experiments.latent_mcp import serialized
from experiments.test_latent_mcp import corpus, put


def test_exclusive_stages_include_failed_work(monkeypatch):
    now = [0.0]
    with monkeypatch.context() as patch:
        patch.setattr(context_costs.time, "perf_counter", lambda: now[0])
        with pytest.raises(RuntimeError, match="refused"):
            with context_costs.timing() as seconds:
                now[0] = 1
                with context_costs.stage("navigation"):
                    now[0] = 3
                    with context_costs.stage("verification_reads"):
                        now[0] = 6
                    now[0] = 8
                with context_costs.stage("serialization"):
                    now[0] = 12
                now[0] = 13
                raise RuntimeError("refused")
    assert seconds == dict(navigation_seconds=4.0, verification_reads_seconds=3.0,
                           serialization_seconds=4.0, packing_seconds=2.0)


@pytest.mark.parametrize("format", ["json", "compact"])
def test_pack_stages_and_client_wall_are_bounded_and_honest(corpus, format):
    memory = ContextMemory(corpus[0])
    pack = memory.context("Alpha", format=format, _waiting_seconds=0.125)
    stages = pack["costs"]["stages"]
    assert all(stages[name] > 0 for name in context_costs.stage_seconds())
    assert sum(stages[name] for name in context_costs.stage_seconds()) > 0
    assert pack["costs"]["client"]["wall_seconds"] is None  # No HTTP client observed.
    assert pack["costs"]["client"]["waiting_seconds"] == 0.125
    assert pack["costs"]["client"]["transport_seconds"] is None
    saved = memory.context_details(pack["pack_id"])
    client_receipt(pack, wall_seconds=0.8, connection_seconds=0.1, transport_seconds=0.05)
    assert pack["costs"]["client"]["wall_seconds"] == 0.8
    assert pack["costs"]["reader_visible"]["serialized_bytes"] == len(serialized(pack).encode())
    assert memory.meter.fits(pack, pack["costs"]["output_limits"])
    assert saved["costs"]["client"]["wall_seconds"] is None  # No retroactive server claim.
    with pytest.raises(ValueError, match="budget"):
        memory.context("Alpha", budget={"tokens": 512, "bytes": 2048})
    attempt = memory.context_last_attempt
    assert attempt["status"] == "REFUSED" and sum(attempt["stages"].values()) > 0


def test_initial_listing_allows_direct_call_without_discovery_and_attaches_once(corpus, tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    put(registry, dict(test=dict(config=str(corpus[0]), source_root=str(corpus[1]), description="fixture")))
    host = MemoryHost(registry)
    import experiments.latent_context_app as app
    operations, connections = [], []

    def connect(**kwargs):
        connections.append(kwargs)
        return {}

    def call(endpoint, method, **arguments):
        operations.append(method)
        return host.invoke(dict(method=method, **arguments))

    monkeypatch.setattr(app, "ensure_server", connect)
    monkeypatch.setattr(app, "call", call)
    server = mcp_server(None, "test", registry)
    tools = asyncio.run(server.list_tools())
    assert not connections and not operations and not host.memories
    assert tools[0].name == "context"
    assert "question" in tools[0].input_schema["required"]
    assert not any("discover" in t.name or "search_tools" in t.name for t in tools)
    for _ in range(2):
        reply = asyncio.run(server.call_tool("context", {"question": "Alpha"}))
        pack = json.loads(reply.content[0].text)
        assert not reply.is_error and pack["coverage"]["returned_spans"] > 0
        assert pack["costs"]["client"]["wall_seconds"] > 0
    assert operations == ["context", "context"] and len(connections) == len(host.memories) == 1


def test_failed_lazy_attachment_reports_client_wait(monkeypatch):
    import experiments.latent_context_app as app
    from mcp.server.mcpserver.exceptions import ToolError

    def fail(**kwargs):
        raise RuntimeError("isolated backend unavailable")

    monkeypatch.setattr(app, "ensure_server", fail)
    server = mcp_server(None, "test")
    with pytest.raises(ToolError, match="Attempt costs:") as refusal:
        asyncio.run(server.call_tool("context", {"question": "Alpha"}))
    attempt = json.loads(str(refusal.value).split("Attempt costs: ", 1)[1])
    assert attempt["phase"] == "client"
    assert attempt["client"]["wall_seconds"] >= attempt["client"]["connection_seconds"] >= 0


@pytest.mark.parametrize("fail", [False, True])
def test_http_client_meters_transport_and_failures(corpus, monkeypatch, fail):
    import experiments.latent_context_app as app
    from urllib.error import HTTPError
    memory = ContextMemory(corpus[0])
    pack = memory.context("Alpha")

    class Response(io.BytesIO):
        headers = {"X-Latent-Service-Seconds": "0"}

    class Opener:
        def open(self, request, timeout):
            if fail:
                raise HTTPError(request.full_url, 400, "refused", Response.headers,
                    io.BytesIO(serialized(dict(error="ValueError: intentional refusal",
                        attempt_costs=dict(status="REFUSED", access={"source_opens": 1}))).encode()))
            return Response(serialized(pack).encode())

    monkeypatch.setattr(app.urllib.request, "build_opener", lambda *a: Opener())
    if fail:
        with pytest.raises(RuntimeError, match="Attempt costs:") as refusal:
            app.call(dict(url="http://127.0.0.1:1"), "context", question="Alpha")
        receipt = refusal.value.attempt_costs
        assert receipt["access"]["source_opens"] == 1
        assert receipt["client"]["wall_seconds"] == receipt["client"]["transport_seconds"] > 0
    else:
        result = app.call(dict(url="http://127.0.0.1:1"), "context", question="Alpha")
        receipt = result["costs"]["client"]
        assert receipt["wall_seconds"] == receipt["transport_seconds"] > 0
        assert result["costs"]["reader_visible"]["serialized_bytes"] == len(serialized(result).encode())


def test_recommended_registration_is_optional_and_direct():
    doc = (Path(__file__).resolve().parents[1] / "docs/CONTEXT_TOOL.md").read_text(encoding="utf8")
    recipe = doc.split("```toml\n", 1)[1].split("```", 1)[0]
    config = tomllib.loads(recipe)
    assert config["mcp_optional_startup_grace_ms"] == 0
    server = config["mcp_servers"]["latent-context"]
    assert server["required"] is False
    assert server["startup_readiness"] == "connection"
    assert server["omit_tools_from"] == ["deferred", "code_mode"]
    assert server["env"]["LSR_CPU_RAM_FLOOR_GIB"] == "10"
    assert server["env"]["HF_HUB_OFFLINE"] == "1"

def test_health_probes_do_not_wait_behind_context(corpus, tmp_path):
    import threading
    registry = tmp_path / 'registry.json'
    put(registry, dict(test=dict(config=str(corpus[0]), source_root=str(corpus[1]), description='fixture')))
    host = MemoryHost(registry)
    expected = host.invoke(dict(method='warm', corpus='test'))
    locked, release, finished = threading.Event(), threading.Event(), threading.Event()
    results = []

    def busy():
        with host.lock:
            locked.set()
            release.wait(10)

    def probe():
        try:
            results.append((host.invoke(dict(method='status'))['loaded'],
                            host.invoke(dict(method='warm', corpus='test'))))
        finally:
            finished.set()

    worker = threading.Thread(target=busy)
    worker.start()
    assert locked.wait(2)
    checker = threading.Thread(target=probe)
    checker.start()
    try:
        assert finished.wait(2), 'health probe queued behind navigation'
    finally:
        release.set()
        worker.join(5)
        checker.join(5)
    assert results == [(1, expected)]
    cfg = json.loads(corpus[0].read_text())
    cfg['default_strategy'] = 'expand'
    put(corpus[0], cfg)
    with pytest.raises(ValueError, match='configuration changed'):
        host.invoke(dict(method='warm', corpus='test'))
