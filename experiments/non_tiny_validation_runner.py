"""Run non-tiny benchmark validation with hard timeout and audit report."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run non-tiny benchmark with timeout and emit an audit report."
    )
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--queries-file", default="experiments/queries_non_tiny_pair.txt")
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument(
        "--score-cache",
        action="store_true",
        help="Enable evolution score-cache during the benchmark run.",
    )
    parser.add_argument(
        "--benchmark-json",
        default="experiments/aim_v1_non_tiny_benchmark_distilgpt2_pair.json",
    )
    parser.add_argument(
        "--benchmark-md",
        default="experiments/aim_v1_non_tiny_benchmark_distilgpt2_pair.md",
    )
    parser.add_argument(
        "--report-json",
        default="experiments/aim_v1_non_tiny_validation_report.json",
    )
    args = parser.parse_args()

    cmd = [
        "python",
        "experiments/benchmark_adaptive_survivors.py",
        "--model",
        args.model,
        "--queries-file",
        args.queries_file,
        "--chains",
        str(args.chains),
        "--generations",
        str(args.generations),
        "--max-tokens",
        str(args.max_tokens),
        "--repeats",
        str(args.repeats),
        "--output-json",
        args.benchmark_json,
        "--output-md",
        args.benchmark_md,
    ]
    if args.score_cache:
        cmd.append("--score-cache")

    started = perf_counter()
    status = "completed"
    returncode = None
    stdout = ""
    stderr = ""
    timeout_hit = False

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout_s,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        if returncode != 0:
            status = "failed"
    except subprocess.TimeoutExpired as err:
        status = "timed_out"
        timeout_hit = True
        stdout = err.stdout or ""
        stderr = err.stderr or ""

    elapsed_s = perf_counter() - started

    benchmark_json_path = Path(args.benchmark_json)
    benchmark_payload = _load_json_if_exists(benchmark_json_path)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "timeout_hit": timeout_hit,
        "elapsed_s": elapsed_s,
        "returncode": returncode,
        "timeout_s": args.timeout_s,
        "command": cmd,
        "benchmark_json": args.benchmark_json,
        "benchmark_md": args.benchmark_md,
        "comparison": benchmark_payload.get("comparison") if benchmark_payload else None,
        "summary_fixed": benchmark_payload.get("fixed", {}).get("summary") if benchmark_payload else None,
        "summary_adaptive": benchmark_payload.get("adaptive", {}).get("summary") if benchmark_payload else None,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Validation status: {status}")
    print(f"Elapsed seconds: {elapsed_s:.2f}")
    print(f"Wrote report: {report_path}")
    if benchmark_payload and isinstance(benchmark_payload.get("comparison"), dict):
        print(f"Comparison: {benchmark_payload['comparison']}")

    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
