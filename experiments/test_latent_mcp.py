"""Software/transport checks, not a source-answer benchmark."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.latent_mcp import LatentMemoryTool
from experiments.latent_mcp_user import launch_command


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def corpus(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    index = tmp_path / "index"
    index.mkdir()
    leaves = index / "leaves"
    leaves.mkdir()
    rows, files = [], []
    for name, text in (("origin.md", "Alpha source [target](target.md).\n"),
                       ("target.md", "Target value is 29.\n")):
        path = source / name
        raw = text.encode("utf-8")
        path.write_bytes(raw)
        (leaves / (sha(raw) + ".txt")).write_bytes(raw)
        files.append(dict(path=str(path), sha256=sha(raw), bytes=len(raw), extraction="utf8_text"))
        rows.append(dict(source=str(path), source_sha256=sha(raw), page=None,
                         leaf_sha256=sha(raw), start_byte=0, end_byte=len(raw),
                         span_sha256=sha(raw), text=text))
    put(index / "rows.json", rows)
    np.save(index / "vectors.npy", np.ones((2, 3), dtype=np.float32))
    put(index / "manifest.json", dict(files=files, allowed_paths=[f["path"] for f in files],
        source_roots=[str(source)], encoder="unused-bm25", rows_sha256=sha((index / "rows.json").read_bytes()),
        vector_sha256=sha((index / "vectors.npy").read_bytes())))
    config = tmp_path / "config.json"
    put(config, dict(index_dir=str(index), source_root=str(source),
        manifest_sha256=sha((index / "manifest.json").read_bytes()),
        allowed_paths=[f["path"] for f in files], routing="bm25", max_results=2,
        max_output_bytes=12000))
    return config, source, index


def test_real_source_search_reopen_restart_and_reference(corpus):
    config, source, _ = corpus
    first = LatentMemoryTool(config)
    result = json.loads(first.latent_memory("Alpha", 1))
    passage = result["passages"][0]
    assert passage["verified"] is True
    assert passage["sha256"] == sha(passage["text"].encode())
    assert passage["offsets"] == "EXTRACTED_LEAF_UTF8"
    second = LatentMemoryTool(config)
    reopened = json.loads(second.open_span(passage["id"]))["passages"][0]
    assert reopened["text"] == passage["text"]
    followed = json.loads(second.follow_reference(passage["id"]))
    assert followed["passages"][0]["file"] == "target.md"
    assert followed["passages"][0]["text"] == (source / "target.md").read_text()


@pytest.mark.parametrize("query,k", [("", 1), ("x", True), ("x", 0), ("x", 3), ("x" * 4097, 1)])
def test_invalid_inputs_refuse(corpus, query, k):
    tool = LatentMemoryTool(corpus[0])
    with pytest.raises(ValueError):
        tool.latent_memory(query, k)


def test_configured_default_and_wrong_id(corpus):
    tool = LatentMemoryTool(corpus[0])
    assert json.loads(tool.latent_memory("Alpha"))["returned"] == 2
    with pytest.raises(ValueError):
        tool.open_span("0000000000000000:S0")


@pytest.mark.parametrize("change", ["source", "leaf", "allowlist", "manifest"])
def test_changed_inputs_refuse(corpus, change):
    config, source, index = corpus
    tool = LatentMemoryTool(config)
    if change == "source":
        (source / "origin.md").write_text("Changed", encoding="utf-8")
        with pytest.raises(ValueError):
            tool.open_span(tool.corpus_id + ":S0")
    elif change == "leaf":
        row = tool.memory.rows[0]
        (index / "leaves" / (row["leaf_sha256"] + ".txt")).write_text("Changed")
        with pytest.raises(ValueError):
            tool.open_span(tool.corpus_id + ":S0")
    elif change == "allowlist":
        cfg = json.loads(config.read_text())
        cfg["allowed_paths"].pop()
        put(config, cfg)
        with pytest.raises(ValueError):
            LatentMemoryTool(config)
    else:
        with (index / "manifest.json").open("a") as stream:
            stream.write(" ")
        with pytest.raises(ValueError):
            LatentMemoryTool(config)


def test_file_tools_are_source_bounded_and_paginated(corpus):
    config, source, index = corpus
    tool = LatentMemoryTool(config, files_only=True)
    assert json.loads(tool.list_files())["total"] == 2
    with pytest.raises(ValueError):
        tool.read_file(str(index / "manifest.json"))
    with pytest.raises(ValueError):
        tool.read_file("../config.json")
    result = json.loads(tool.read_file("origin.md"))
    assert result["end_byte"] > result["start_byte"]
    assert json.loads(tool.search_files("Alpha"))["matches"][0]["file"] == "origin.md"


def test_file_output_serialized_budget_and_long_line_progress(corpus):
    config, source, index = corpus
    raw = ("界\\" * 20000 + "\n").encode()
    path = source / "origin.md"
    path.write_bytes(raw)
    manifest = json.loads((index / "manifest.json").read_text())
    manifest["files"][0].update(sha256=sha(raw), bytes=len(raw))
    put(index / "manifest.json", manifest)
    cfg = json.loads(config.read_text())
    cfg.update(manifest_sha256=sha((index / "manifest.json").read_bytes()), max_output_bytes=4096)
    put(config, cfg)
    tool = LatentMemoryTool(config, files_only=True)
    first = json.loads(tool.read_file("origin.md"))
    second = json.loads(tool.read_file("origin.md", start_byte=first["next_byte"]))
    assert second["end_byte"] > first["end_byte"] > 0
    for result in (first, second, json.loads(tool.search_files("界"))):
        assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()) <= 4096


def test_unowned_project_refuses(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        launch_command(tmp_path, "unused", False, [])


def test_real_stdio_sdk_tools(corpus):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def journey():
        root = str(Path(__file__).resolve().parents[1])
        env = os.environ.copy()
        env["PYTHONPATH"] = root
        params = StdioServerParameters(command=sys.executable, args=[
            "-u", "-m", "experiments.latent_mcp", "--config", str(corpus[0])], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                names = {t.name for t in tools}
                assert names == {"list_files", "read_file", "search_files",
                                 "latent_memory", "open_span", "follow_reference"}
                assert all(t.annotations.read_only_hint for t in tools)
                result = await session.call_tool("latent_memory", {"query": "Alpha", "k": 1})
                assert not result.is_error
                passage = json.loads(result.content[0].text)["passages"][0]
                for name, args in (("open_span", {"id": passage["id"]}),
                                   ("follow_reference", {"span_id": passage["id"]})):
                    result = await session.call_tool(name, args)
                    assert not result.is_error
                    assert json.loads(result.content[0].text)["passages"]
    asyncio.run(journey())


def test_initializer_rejects_mislabeled_corpus_before_seeding(corpus, tmp_path):
    from types import SimpleNamespace
    from experiments.latent_mcp_user import init
    args = SimpleNamespace(template=str(tmp_path / "copy"), project=str(tmp_path / "project"),
        corpus=str(tmp_path / "wrong"), mcp_config=str(corpus[0]))
    with pytest.raises(ValueError, match="corpus"):
        init(args)
    assert not Path(args.project).exists() and not Path(args.template).exists()


@pytest.mark.parametrize("changed", ["guide", "active_config", "mcp_config"])
def test_successor_pins_active_inputs(corpus, tmp_path, changed):
    project = tmp_path / "project"
    home = project / ".stateful"
    home.mkdir(parents=True)
    (home / "config.toml").write_text("", encoding="utf-8")
    (home / "auth.json").write_text("{}", encoding="utf-8")
    put(home / "source_only_models.json", {"models": []})
    (project / "AGENTS.md").write_text("Generic source guidance\n", encoding="utf-8")
    pin = lambda path: sha(path.read_bytes())
    put(project / "journey.json", dict(schema="latent-mcp-owned-project-v2",
        project=str(project.resolve()), home=str(home.resolve()), corpus=str(corpus[1].resolve()),
        catalog_sha256=pin(home / "source_only_models.json"),
        mcp_config=str(corpus[0]), mcp_config_sha256=pin(corpus[0]),
        active_config_sha256=pin(home / "config.toml"), guide_sha256=pin(project / "AGENTS.md")))
    launch_command(project, "unused", False, [])
    path = {"guide": project / "AGENTS.md", "active_config": home / "config.toml",
            "mcp_config": corpus[0]}[changed]
    with path.open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="changed"):
        launch_command(project, "unused", False, [])


@pytest.mark.parametrize("code,turns,answer,status", [
    (1, 0, "", "CLI_FAILED"), (0, 1, "", "EMPTY_ANSWER"),
    (0, 0, "partial", "INCOMPLETE"), (0, 1, "answer", "COMPLETED")])
def test_runner_completion_status_and_exit_code(tmp_path, monkeypatch, code, turns, answer, status):
    """Process-boundary unit fixture, not an experimental reader answer."""
    from types import SimpleNamespace
    import experiments.latent_mcp_user as runner
    output = tmp_path / "output"
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Unit fixture", encoding="utf-8")
    monkeypatch.setattr(runner, "launch_command",
                        lambda *a: (["fixture", "exec", "-a", "never"], os.environ.copy()))
    class Process:
        pid = 1
        def wait(self, timeout):
            return code
    def start(*args, **kwargs):
        (output / "answer.txt").write_text(answer, encoding="utf-8")
        for _ in range(turns):
            kwargs["stdout"].write(json.dumps(dict(type="turn.completed",
                usage={}, trajectory={"completed_model_responses": 1})) + "\n")
        return Process()
    monkeypatch.setattr(runner.subprocess, "Popen", start)
    args = SimpleNamespace(project=str(tmp_path), output=str(output), prompt_file=str(prompt),
                           binary=sys.executable, plain=True, timeout=1)
    result = runner.run(args)
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == status and result == (0 if status == "COMPLETED" else 1)
    assert receipt["answer_present"] == bool(answer)
    assert receipt["binary_sha256"] == sha(Path(sys.executable).read_bytes())


def test_cpu_memory_profile_is_explicit_and_bounded(monkeypatch):
    from types import SimpleNamespace
    import psutil
    from experiments.dogfood import ram_guard
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=15 * 2**30))
    monkeypatch.delenv("LSR_CPU_RAM_FLOOR_GIB", raising=False)
    with pytest.raises(MemoryError):
        ram_guard()
    monkeypatch.setenv("LSR_CPU_RAM_FLOOR_GIB", "10")
    assert ram_guard() == 15 * 2**30
    monkeypatch.setenv("LSR_CPU_RAM_FLOOR_GIB", "0")
    with pytest.raises(ValueError):
        ram_guard()


def test_relative_live_home_rejected_before_seed_copy(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.latent_mcp_user import init
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LSR_CLIENT_HOME", "live")
    args = SimpleNamespace(template=str(tmp_path / "live/copy"),
                           project=str(tmp_path / "project"))
    with pytest.raises(ValueError, match="live home"):
        init(args)
    assert not (tmp_path / "live").exists()


def test_initialized_reader_override_cannot_change_comparison_model(corpus, tmp_path, monkeypatch):
    project = tmp_path / "project"
    home = project / ".stateful"
    home.mkdir(parents=True)
    (home / "config.toml").write_text("", encoding="utf-8")
    (home / "auth.json").write_text("{}", encoding="utf-8")
    put(home / "source_only_models.json", {"models": []})
    put(project / "journey.json", dict(schema="latent-mcp-owned-project-v1",
        project=str(project.resolve()), home=str(home.resolve()), reader_model="fixture-reader",
        catalog_sha256=sha((home / "source_only_models.json").read_bytes())))
    monkeypatch.setenv("LSR_READER_MODEL", "different-reader")
    with pytest.raises(ValueError, match="reader selection changed"):
        launch_command(project, "unused", False, [])
