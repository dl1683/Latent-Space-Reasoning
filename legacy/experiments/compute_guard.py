"""Lightweight compute preflight checks for local benchmark scripts."""

from __future__ import annotations

import argparse
import subprocess
import time


def add_gpu_guard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-gpu-utilization",
        type=float,
        default=35.0,
        help="Refuse to start when nvidia-smi reports higher GPU utilization. Use -1 to disable.",
    )
    parser.add_argument(
        "--max-gpu-memory-used-mb",
        type=float,
        default=12000.0,
        help="Refuse to start when nvidia-smi reports more used VRAM. Use -1 to disable.",
    )
    parser.add_argument(
        "--wait-for-gpu",
        action="store_true",
        help="Wait until GPU load is under the configured limits instead of failing immediately.",
    )
    parser.add_argument(
        "--gpu-wait-timeout-s",
        type=float,
        default=900.0,
        help="Maximum seconds to wait for GPU load to fall under limits.",
    )
    parser.add_argument(
        "--gpu-wait-poll-s",
        type=float,
        default=15.0,
        help="Seconds between GPU load checks while --wait-for-gpu is active.",
    )


def query_gpu_load() -> tuple[float, float] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None

    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _gpu_guard_error(
    *,
    util: float,
    mem_used: float,
    max_util: float,
    max_mem: float,
) -> str | None:
    if max_util >= 0 and util > max_util:
        return (
            f"GPU utilization is {util:.0f}%, above --max-gpu-utilization={max_util:.0f}. "
            "Start later or pass a higher limit."
        )
    if max_mem >= 0 and mem_used > max_mem:
        return (
            f"GPU memory used is {mem_used:.0f}MB, above --max-gpu-memory-used-mb={max_mem:.0f}. "
            "Start later or pass a higher limit."
        )
    return None


def enforce_gpu_guard(args: argparse.Namespace) -> None:
    max_util = float(args.max_gpu_utilization)
    max_mem = float(args.max_gpu_memory_used_mb)
    if max_util < 0 and max_mem < 0:
        return

    wait_for_gpu = bool(getattr(args, "wait_for_gpu", False))
    timeout_s = max(0.0, float(getattr(args, "gpu_wait_timeout_s", 900.0)))
    poll_s = max(0.1, float(getattr(args, "gpu_wait_poll_s", 15.0)))
    started = time.monotonic()
    last_error = ""

    while True:
        load = query_gpu_load()
        if load is None:
            return

        util, mem_used = load
        error = _gpu_guard_error(
            util=util,
            mem_used=mem_used,
            max_util=max_util,
            max_mem=max_mem,
        )
        if error is None:
            return

        last_error = error
        if not wait_for_gpu:
            raise RuntimeError(error)
        if time.monotonic() - started >= timeout_s:
            raise RuntimeError(f"Timed out waiting for GPU availability. Last check: {last_error}")
        time.sleep(poll_s)
