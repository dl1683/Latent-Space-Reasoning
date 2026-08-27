"""CPU-only tests for the ARC-AGI-3 local smoke runner."""

import json
from argparse import Namespace
from subprocess import CompletedProcess
from unittest.mock import Mock, patch

from experiments.run_arc3_local_latent_smoke import (
    _harness_completed,
    _server_command,
    _write_manifest,
)


def _args(tmp_path):
    return Namespace(
        host="127.0.0.1",
        port=8013,
        encoder="Qwen/Qwen3-0.6B",
        server_backend="latent",
        chains=1,
        generations=1,
        max_tokens=128,
        max_latent_calls=-1,
        scripted_plan="eval_results/ls20_static_plans.json",
        state_probe_repeat_cap=8,
        fallback_policy="state_probe",
        wait_for_gpu=True,
        max_gpu_utilization=35.0,
        max_gpu_memory_used_mb=12000.0,
        gpu_wait_timeout_s=900.0,
        gpu_wait_poll_s=20.0,
        trace_jsonl=str(tmp_path / "trace.jsonl"),
        game_id="ls20",
        harness_output=str(tmp_path / "harness.json"),
        output=str(tmp_path / "smoke.json"),
    )


def test_server_command_runs_child_unbuffered_and_waits_for_gpu(tmp_path):
    command = _server_command(_args(tmp_path))

    assert "-u" in command
    assert "arc3_latent_openai_server.py" in command[2]
    assert "--wait-for-gpu" in command
    assert "--trace-jsonl" in command
    assert str(tmp_path / "trace.jsonl") in command
    assert "--backend" in command
    assert "latent" in command
    assert "--max-latent-calls" in command
    assert "--scripted-plan" in command
    assert "--fallback-policy" in command
    assert "state_probe" in command
    assert "--state-probe-repeat-cap" in command
    assert "--max-gpu-utilization" in command
    assert "35.0" in command


def test_write_manifest_records_server_log_path(tmp_path):
    args = _args(tmp_path)
    server_log = tmp_path / "server.log"

    _write_manifest(
        args=args,
        server_command=["python", "-u", "server.py"],
        server_ready=False,
        server_returncode=1,
        server_log=server_log,
        trace_jsonl=tmp_path / "trace.jsonl",
        harness_result=None,
    )

    manifest = json.loads((tmp_path / "smoke.json").read_text(encoding="utf-8"))

    assert manifest["server_ready"] is False
    assert manifest["server_log"] == str(server_log)
    assert manifest["trace_jsonl"] == str(tmp_path / "trace.jsonl")
    assert manifest["game_id"] == "ls20"


def test_harness_completed_detects_final_scorecard_report():
    result = CompletedProcess(
        args=["uv", "run", "main.py"],
        returncode=2,
        stdout="agent exhausted budget\n--- FINAL SCORECARD REPORT ---\nscore: 0.0",
        stderr="",
    )

    assert _harness_completed(result) is True


def test_harness_completed_detects_final_scorecard_in_manifest(tmp_path):
    manifest = tmp_path / "harness.json"
    manifest.write_text('{"stdout": "--- FINAL SCORECARD REPORT ---"}', encoding="utf-8")
    result = CompletedProcess(
        args=["python", "wrapper.py"],
        returncode=2,
        stdout="Return code: 2",
        stderr="",
    )

    assert _harness_completed(result, str(manifest)) is True


def test_harness_completed_rejects_thread_crash_without_scorecard():
    result = CompletedProcess(
        args=["python", "wrapper.py"],
        returncode=0,
        stdout="created new recording",
        stderr="Exception in thread Thread-1: AttributeError",
    )

    assert _harness_completed(result) is False


def test_smoke_main_clears_trace_file_before_server_start(tmp_path, monkeypatch):
    from experiments import run_arc3_local_latent_smoke as smoke

    trace = tmp_path / "trace.jsonl"
    trace.write_text("stale\n", encoding="utf-8")
    args = _args(tmp_path)
    args.trace_jsonl = str(trace)
    args.server_log = str(tmp_path / "server.log")
    args.output = str(tmp_path / "smoke.json")
    args.server_ready_timeout_s = 1.0

    monkeypatch.setattr(smoke, "parse_args", lambda: args)
    monkeypatch.setattr(smoke, "_wait_for_server", lambda base_url, timeout_s: False)
    process = Mock()
    process.poll.return_value = 1
    process.terminate.return_value = None

    with patch("experiments.run_arc3_local_latent_smoke.subprocess.Popen", return_value=process):
        try:
            smoke.main()
        except RuntimeError:
            pass

    assert trace.read_text(encoding="utf-8") == ""
