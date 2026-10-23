"""Real HTTP disconnect and MCP cancellation must release the same worker."""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time

import pytest
from experiments.test_latent_mcp import corpus, put
from experiments import latent_context_app as app


@pytest.fixture
def service(corpus, tmp_path):
    registry = tmp_path / 'registry.json'
    put(registry, dict(test=dict(config=str(corpus[0]), source_root=str(corpus[1]), description='fixture')))
    marker = tmp_path / 'started'
    code = '''
import sys,time
from pathlib import Path
from experiments import latent_context_app as app
from experiments.request_control import checkpoint
original=app.ContextMemory.context
def context(self, question, *args, **kwargs):
    if question == 'slow':
        Path(sys.argv[2]).write_text('started')
        for _ in range(1000):
            time.sleep(.01)
            checkpoint()
        Path(sys.argv[2]).write_text('COMPLETED')
    return original(self, question, *args, **kwargs)
app.ContextMemory.context=context
app.serve(sys.argv[1], 'test')
'''
    process = subprocess.Popen([sys.executable, '-u', '-c', code, str(registry), str(marker)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=dict(os.environ, LSR_CPU_RAM_FLOOR_GIB='12', CUDA_VISIBLE_DEVICES=''))
    try:
        endpoint = json.loads(process.stdout.readline())
        yield endpoint, registry, marker
    finally:
        process.terminate()
        process.communicate(timeout=10)


def wait_started(marker):
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert marker.exists()


def assert_released(endpoint, marker):
    started = time.monotonic()
    result = app.call(endpoint, 'context', corpus='test', question='Alpha')
    assert time.monotonic() - started < 3
    assert result['coverage']['returned_spans'] > 0
    assert marker.read_text() == 'started'


def test_disconnect_releases_worker(service):
    endpoint, _, marker = service
    from urllib.parse import urlsplit
    port = urlsplit(endpoint['url']).port
    connection = socket.create_connection(('127.0.0.1', port))
    payload = json.dumps(dict(method='context', corpus='test', question='slow')).encode()
    headers = (f'POST /api HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n'
               'Content-Type: application/json\r\nX-Latent-Client: context-pack\r\n'
               f'Content-Length: {len(payload)}\r\n\r\n').encode()
    connection.sendall(headers + payload)
    wait_started(marker)
    connection.close()
    assert_released(endpoint, marker)


def test_mcp_cancel_releases_worker(service):
    endpoint, registry, marker = service
    async def run():
        server = app.mcp_server(endpoint, 'test', registry)
        task = asyncio.create_task(server.call_tool('context', dict(question='slow')))
        await asyncio.to_thread(wait_started, marker)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.to_thread(assert_released, endpoint, marker)
    asyncio.run(run())


def test_ram_floor_accepts_12_and_rejects_invalid(monkeypatch):
    from experiments.dogfood import ram_guard
    monkeypatch.setenv('LSR_CPU_RAM_FLOOR_GIB', '12')
    assert ram_guard() > 12 * 2**30
    monkeypatch.setenv('LSR_CPU_RAM_FLOOR_GIB', '9')
    with pytest.raises(ValueError):
        ram_guard()


def test_stdio_mcp_cancel_notification_releases_worker(service):
    endpoint, registry, marker = service
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from pathlib import Path
    code = "from experiments import latent_context_app as a; import json,sys; a.mcp_server(json.loads(sys.argv[1]), 'test').run()"
    async def run():
        params = StdioServerParameters(command=sys.executable,
            args=['-u', '-c', code, json.dumps(endpoint)],
            cwd=str(Path(app.__file__).resolve().parents[1]),
            env=dict(os.environ, PYTHONUTF8='1'))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                task = asyncio.create_task(session.call_tool('context', dict(question='slow')))
                await asyncio.to_thread(wait_started, marker)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                await asyncio.sleep(.1)
                await asyncio.to_thread(assert_released, endpoint, marker)
    asyncio.run(run())
