"""Run one ARC-AGI-3 game through the local latent reasoning bridge.

This script starts ``arc3_latent_openai_server.py` as a child process, waits
for the OpenAI-compatible endpoint, runs the official ARC-AGI-3 harness against
``local-latent-reasoning``, writes a manifest, and stops the child server.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _server_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(REPO_ROOT / "experiments" / "arc3_latent_openai_server.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--encoder",
        args.encoder,
        "--backend",
        args.server_backend,
        "--game-id",
        args.game_id,
        "--chains",
        str(args.chains),
        "--generations",
        str(args.generations),
        "--max-tokens",
        str(args.max_tokens),
        "--max-latent-calls",
        str(args.max_latent_calls),
        "--scripted-plan",
        args.scripted_plan,
        "--learned-trace",
        getattr(args, "learned_trace", "eval_results/arc3_scripted_astar_l7_trace.jsonl"),
        "--learned-policy-k",
        str(getattr(args, "learned_policy_k", 7)),
        "--learned-max-train-level",
        str(getattr(args, "learned_max_train_level", -1)),
        *(["--learned-sequence-backoff"] if getattr(args, "learned_sequence_backoff", False) else []),
        *(["--learned-phase-switch"] if getattr(args, "learned_phase_switch", False) else []),
        *(["--learned-goal-seek"] if getattr(args, "learned_goal_seek", False) else []),
        "--mechanistic-guard",
        getattr(args, "mechanistic_guard", "off"),
        "--fallback-policy",
        args.fallback_policy,
        "--state-probe-repeat-cap",
        str(args.state_probe_repeat_cap),
        *(["--executable-search-plan"] if getattr(args, "executable_search_plan", False) else []),
        "--executable-search-max-levels",
        str(getattr(args, "executable_search_max_levels", 2)),
        "--ollama-model",
        getattr(args, "ollama_model", "mistral:7b"),
        "--ollama-url",
        getattr(args, "ollama_url", "http://127.0.0.1:11434"),
        "--ollama-timeout-s",
        str(getattr(args, "ollama_timeout_s", 12.0)),
        "--trace-jsonl",
        args.trace_jsonl,
        "--max-gpu-utilization",
        str(args.max_gpu_utilization),
        "--max-gpu-memory-used-mb",
        str(args.max_gpu_memory_used_mb),
        "--gpu-wait-timeout-s",
        str(args.gpu_wait_timeout_s),
        "--gpu-wait-poll-s",
        str(args.gpu_wait_poll_s),
        *(["--wait-for-gpu"] if args.wait_for_gpu else []),
    ]


def _wait_for_server(base_url: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(1.0, timeout_s)
    models_url = f"{base_url.rstrip('/')}/models"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=5) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    return False


def _run_harness(args: argparse.Namespace, base_url: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "run_arc3_official_harness.py"),
        "--install-local-latent-config",
        "--local-latent-base-url",
        base_url,
        "--game-id",
        args.game_id,
        "--config",
        "local-latent-reasoning",
        "--tags",
        args.tags,
        "--output",
        args.harness_output,
    ]
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def _harness_completed(
    result: subprocess.CompletedProcess[str] | None,
    harness_output: str | None = None,
) -> bool:
    if result is None:
        return False
    if "FINAL SCORECARD REPORT" in result.stdout:
        return True
    if not harness_output:
        return False
    output = Path(harness_output)
    if not output.exists():
        return False
    return "FINAL SCORECARD REPORT" in output.read_text(encoding="utf-8", errors="replace")


def _write_manifest(
    args: argparse.Namespace,
    server_command: list[str],
    server_ready: bool,
    server_returncode: int | None,
    server_log: Path,
    trace_jsonl: Path,
    harness_result: subprocess.CompletedProcess[str] | None,
) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_command": server_command,
        "server_ready": server_ready,
        "server_returncode": server_returncode,
        "server_log": str(server_log),
        "trace_jsonl": str(trace_jsonl),
        "game_id": args.game_id,
        "harness_output": args.harness_output,
    }
    if harness_result is not None:
        payload["harness"] = {
            "returncode": harness_result.returncode,
            "completed": _harness_completed(harness_result, args.harness_output),
            "stdout": harness_result.stdout,
            "stderr": harness_result.stderr,
        }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--encoder", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--server-backend",
        choices=["latent", "first_legal", "state_probe", "frontier_probe", "graph_probe", "scripted_plan", "learned_visual", "transition_goal", "gemini_advisor", "ollama_advisor"],
        default="latent",
    )
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-latent-calls", type=int, default=-1)
    parser.add_argument("--scripted-plan", default="eval_results/ls20_static_astar_plans_through_l7.json")
    parser.add_argument("--learned-trace", default="eval_results/arc3_scripted_astar_l7_trace.jsonl")
    parser.add_argument("--learned-policy-k", type=int, default=7)
    parser.add_argument("--learned-max-train-level", type=int, default=-1)
    parser.add_argument("--learned-sequence-backoff", action="store_true")
    parser.add_argument("--learned-phase-switch", action="store_true")
    parser.add_argument("--learned-goal-seek", action="store_true")
    parser.add_argument(
        "--mechanistic-guard",
        choices=["off", "scripted_plan", "learned_visual"],
        default="off",
    )
    parser.add_argument("--state-probe-repeat-cap", type=int, default=8)
    parser.add_argument("--executable-search-plan", action="store_true")
    parser.add_argument("--executable-search-max-levels", type=int, default=2)
    parser.add_argument("--ollama-model", default="mistral:7b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout-s", type=float, default=12.0)
    parser.add_argument(
        "--fallback-policy",
        choices=["first_legal", "round_robin", "state_probe", "frontier_probe", "graph_probe", "scripted_plan", "learned_visual", "transition_goal"],
        default="state_probe",
    )
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--max-gpu-utilization", type=float, default=35.0)
    parser.add_argument("--max-gpu-memory-used-mb", type=float, default=12000.0)
    parser.add_argument("--gpu-wait-timeout-s", type=float, default=900.0)
    parser.add_argument("--gpu-wait-poll-s", type=float, default=15.0)
    parser.add_argument("--server-ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--game-id", default="ls20")
    parser.add_argument("--tags", default="latent-local,geometry-feedback,smoke")
    parser.add_argument("--harness-output", default="eval_results/arc3_local_latent_harness.json")
    parser.add_argument("--server-log", default="eval_results/arc3_local_latent_server.log")
    parser.add_argument("--trace-jsonl", default="eval_results/arc3_local_latent_trace.jsonl")
    parser.add_argument("--output", default="eval_results/arc3_local_latent_smoke.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}/v1"
    command = _server_command(args)
    server_log = Path(args.server_log)
    trace_jsonl = Path(args.trace_jsonl)
    server_log.parent.mkdir(parents=True, exist_ok=True)
    trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
    trace_jsonl.write_text("", encoding="utf-8")
    harness_result: subprocess.CompletedProcess[str] | None = None
    server_ready = False

    with server_log.open("w", encoding="utf-8") as log_fp:
        log_fp.write(f"generated_at_utc={datetime.now(timezone.utc).isoformat()}\n")
        log_fp.write(f"base_url={base_url}\n")
        log_fp.write("server_command=" + " ".join(command) + "\n\n")
        log_fp.flush()
        server = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )
        try:
            server_ready = _wait_for_server(base_url, args.server_ready_timeout_s)
            if not server_ready:
                raise RuntimeError(f"Local latent server did not become ready at {base_url}")
            harness_result = _run_harness(args, base_url)
            if not _harness_completed(
                harness_result,
                args.harness_output,
            ):
                raise SystemExit(harness_result.returncode or 1)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=30)
            _write_manifest(
                args=args,
                server_command=command,
                server_ready=server_ready,
                server_returncode=server.poll(),
                server_log=server_log,
                trace_jsonl=trace_jsonl,
                harness_result=harness_result,
            )
            print("Smoke manifest:", args.output)


if __name__ == "__main__":
    main()
