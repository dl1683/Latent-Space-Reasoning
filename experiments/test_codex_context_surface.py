"""Capture real Codex requests with a local, deterministic Responses provider.

No credentials, model inference, encoder or corpus reads are needed. The stdio
proxy timestamps the actual context app's initialization and tool-list exchange.
"""
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]


def stdio_proxy(directory, delay, registry=None, corpus="test"):
    directory = Path(directory)
    lock = threading.Lock()

    def record(event, **fields):
        with lock, (directory / "mcp.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(dict(event=event, at=time.time(), **fields)) + "\n")

    record("proxy_started")
    time.sleep(delay)
    with (directory / "mcp.stderr").open("wb") as err:
        child = subprocess.Popen(
            [sys.executable, "-B", "-m", "experiments.latent_context_app", "mcp",
             "--corpus", corpus, "--registry", str(registry or directory / "registry.json")],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)
        pending = {}

        def incoming():
            try:
                for line in sys.stdin.buffer:
                    message = json.loads(line)
                    if "method" in message:
                        pending[message.get("id")] = message["method"]
                        record("request", method=message["method"])
                    child.stdin.write(line)
                    child.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                if child.poll() is None:
                    child.terminate()

        threading.Thread(target=incoming, daemon=True).start()
        try:
            for line in child.stdout:
                message = json.loads(line)
                method = pending.get(message.get("id"))
                record("response", method=method,
                       tools=[tool["name"] for tool in message.get("result", {}).get("tools", [])])
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)


def latent_definitions(request):
    """Include direct, additional and nested definitions, not just top-level tools."""
    found = []

    def visit(value, namespace=""):
        if isinstance(value, dict):
            name = value.get("name", "")
            scope = namespace + "/" + name if isinstance(name, str) else namespace
            if ("latent" in scope and value.get("type") in ("function", "custom")):
                found.append(scope)
            for item in value.values():
                visit(item, scope)
        elif isinstance(value, list):
            for item in value:
                visit(item, namespace)

    visit(request.get("tools", []))
    for item in request.get("input", []):
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            visit(item)
    return sorted(set(found))


def capture_surface(directory, *, required=False, direct=True, delay=0, timeout=60,
                    server_env=None, registry=None, corpus="test", optional_grace_ms=None,
                    wait_for_listing=False):
    codex = shutil.which("codex.cmd" if os.name == "nt" else "codex")
    if not codex:
        pytest.skip("codex binary unavailable")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True)
    home = directory / "home"
    home.mkdir()
    # Listing is lazy: no valid corpus/config is necessary and no service is started.
    if registry is None:
        registry = directory / "registry.json"
        registry.write_text("{}", encoding="utf8")
    requests, timestamps = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            encoding = self.headers.get("Content-Encoding")
            if encoding == "gzip":
                raw = gzip.decompress(raw)
            elif encoding == "zstd":
                import zstandard
                raw = zstandard.ZstdDecompressor().decompress(raw)
            requests.append(json.loads(raw))
            timestamps.append(time.time())
            # Keep Codex alive for listing, but only AFTER capturing the FIRST request.
            # Poll the receipt instead of assuming Python/SDK startup takes four seconds.
            deadline = time.monotonic() + 60
            mcp_log = directory / "mcp.jsonl"
            while wait_for_listing and time.monotonic() < deadline:
                # Ignore an incomplete final record while the proxy appends it.
                mcp = [json.loads(line) for line in
                       (mcp_log.read_text(encoding="utf8").splitlines(keepends=True)
                        if mcp_log.exists() else []) if line.endswith("\n")]
                if any(item["event"] == "response" and item["method"] == "tools/list"
                       for item in mcp):
                    break
                time.sleep(0.05)
            message = dict(id="msg_probe", type="message", role="assistant",
                           status="completed", content=[dict(type="output_text",
                                                            text="Probe complete.", annotations=[])])
            response = dict(id="resp_probe", object="response", status="completed",
                            output=[message], usage=dict(input_tokens=1, output_tokens=1,
                                                        total_tokens=2))
            events = [
                ("response.created", dict(response=dict(response, status="in_progress", output=[]))),
                ("response.output_item.added", dict(output_index=0, item=message)),
                ("response.output_text.delta", dict(item_id=message["id"], output_index=0,
                                                    content_index=0, delta="Probe complete.")),
                ("response.output_item.done", dict(output_index=0, item=message)),
                ("response.completed", dict(response=response)),
            ]
            body = "".join(f"event: {kind}\ndata: {json.dumps(dict(type=kind, **data))}\n\n"
                           for kind, data in events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    quote = json.dumps
    config = [
        'model = "gpt-6.1-sol"', 'model_provider = "probe"',
        'model_reasoning_effort = "low"', 'cli_auth_credentials_store = "file"',
        *([f"mcp_optional_startup_grace_ms = {optional_grace_ms}"]
          if optional_grace_ms is not None else []),
        '[model_providers.probe]', 'name = "Loopback Responses probe"',
        f'base_url = "http://127.0.0.1:{server.server_port}/v1"',
        'wire_api = "responses"', 'requires_openai_auth = false',
        '[mcp_servers.latent-context]', f'command = {quote(sys.executable)}',
        f'args = {quote(["-B", str(Path(__file__).resolve()), "--stdio-proxy", str(directory), str(delay), str(registry), corpus])}',
        f'required = {str(required).lower()}', 'startup_readiness = "connection"',
        f'startup_timeout_sec = {timeout}', 'tool_timeout_sec = 180',
        f'omit_tools_from = {quote(["deferred", "code_mode"] if direct else [])}',
        '[mcp_servers.latent-context.env]',
    ]
    env = dict(PYTHONPATH=str(ROOT), PYTHONUTF8="1", HF_HUB_OFFLINE="1",
               LSR_CPU_RAM_FLOOR_GIB="10", CUDA_VISIBLE_DEVICES="",
               TOKENIZERS_PARALLELISM="false", **(server_env or {}))
    config.extend(f"{key} = {quote(value)}" for key, value in env.items())
    (home / "config.toml").write_text("\n".join(config) + "\n", encoding="utf8")
    child_env = dict(os.environ, CODEX_HOME=str(home), **env)
    started = time.time()
    try:
        result = subprocess.run(
            [codex, "exec", "--strict-config", "--dangerously-bypass-approvals-and-sandbox",
             "--skip-git-repo-check", "-C", str(directory), "--json",
             "Reply briefly without using any tools."],
            cwd=directory, env=child_env, capture_output=True, timeout=timeout + delay + 45)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
    (directory / "events.jsonl").write_bytes(result.stdout)
    (directory / "codex.stderr").write_bytes(result.stderr)
    (directory / "requests.json").write_text(json.dumps(requests, indent=2), encoding="utf8")
    mcp_log = directory / "mcp.jsonl"
    mcp = [json.loads(line) for line in mcp_log.read_text(encoding="utf8").splitlines()] if mcp_log.exists() else []
    receipt = dict(required=required, direct=direct, delay=delay, startup_timeout=timeout,
                   optional_grace_ms=optional_grace_ms,
                   exit=result.returncode, wall_seconds=time.time() - started,
                   first_request_seconds=timestamps[0] - started if timestamps else None,
                   first_request_at=timestamps[0] if timestamps else None,
                   latent_definitions=latent_definitions(requests[0]) if requests else [],
                   requests=len(requests), mcp=mcp)
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
    return receipt


@pytest.mark.parametrize("required,direct,grace", [
    (False, True, None), (True, True, None), (False, False, None), (False, True, 0)])
def test_codex_first_request_matches_readiness_recipe(tmp_path, required, direct, grace):
    # A deliberately pending server distinguishes readiness from fast-start luck.
    receipt = capture_surface(tmp_path / "probe", required=required, direct=direct,
                              optional_grace_ms=grace, delay=12, wait_for_listing=True)
    assert receipt["exit"] == 0, (tmp_path / "probe/codex.stderr").read_text(encoding="utf8")
    assert receipt["requests"] == 1
    listed = [item for item in receipt["mcp"]
              if item["event"] == "response" and item["method"] == "tools/list"]
    assert listed and set(listed[0]["tools"]) == {
        "context", "context_details", "latent_ask", "open_span", "follow_reference", "neighbors"}
    if required or grace == 0:
        assert {name.rsplit("/", 1)[-1] for name in receipt["latent_definitions"]} == set(listed[0]["tools"])
        assert listed[0]["at"] <= receipt["first_request_at"]
    else:
        assert not receipt["latent_definitions"]
        assert receipt["first_request_at"] < listed[0]["at"]


@pytest.mark.parametrize("required", [False, True])
def test_codex_failed_startup_only_blocks_required_session(tmp_path, required):
    # Startup intentionally times out, so no completed listing is expected here.
    receipt = capture_surface(tmp_path / "probe", required=required,
                              optional_grace_ms=0, timeout=1, delay=3)
    assert not receipt["latent_definitions"]
    if required:
        assert receipt["exit"] != 0 and receipt["requests"] == 0
        assert "required MCP servers failed to initialize" in (
            tmp_path / "probe/codex.stderr").read_text(encoding="utf8")
    else:
        assert receipt["exit"] == 0 and receipt["requests"] == 1


if __name__ == "__main__" and sys.argv[1:2] == ["--stdio-proxy"]:
    stdio_proxy(sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5])
