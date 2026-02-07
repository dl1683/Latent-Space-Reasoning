"""
Automated experiment pipeline: V10 -> V11 -> V12.

Monitors V10 completion, then chains V11 diagnostic -> V11 full -> V12 diagnostic -> V12 full.
Uses results JSON file as completion signal (not log files, due to Windows buffering).

Usage:
    python experiments/run_pipeline.py                    # Monitor V10 + run all
    python experiments/run_pipeline.py --skip-v10-wait    # Start from V11 immediately
    python experiments/run_pipeline.py --start-from v12   # Start from V12
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).parent
PROJECT_DIR = EXP_DIR.parent

# Expected output files
V10_RESULTS = EXP_DIR / "v10_results.json"
V11_DIAG = EXP_DIR / "v11_results_diagnostic.json"
V11_RESULTS = EXP_DIR / "v11_results.json"
V12_DIAG = EXP_DIR / "v12_results_diagnostic.json"
V12_RESULTS = EXP_DIR / "v12_results.json"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def is_complete(results_path, min_seeds=None):
    """Check if a results file exists and has the expected number of seeds."""
    if not results_path.exists():
        return False
    try:
        with open(results_path) as f:
            data = json.load(f)
        seeds = data.get("config", {}).get("seeds", 0)
        if min_seeds and seeds < min_seeds:
            return False
        return True
    except (json.JSONDecodeError, KeyError):
        return False


def wait_for_completion(results_path, min_seeds=5, poll_interval=60, description=""):
    """Poll until results file appears with expected seeds."""
    log(f"Waiting for {description} ({results_path.name}, {min_seeds} seeds)...")
    waited = 0
    while not is_complete(results_path, min_seeds):
        time.sleep(poll_interval)
        waited += poll_interval
        if waited % 300 == 0:  # Log every 5 min
            log(f"  Still waiting... ({waited//60} min elapsed)")
    log(f"{description} complete! ({waited//60} min waited)")


def run_experiment(script, args_list, log_file, description):
    """Run an experiment script and wait for completion."""
    cmd = [sys.executable, "-u", str(script)] + args_list
    log(f"Starting {description}: {' '.join(cmd)}")
    log(f"  Log: {log_file}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_DIR),
        )

    log(f"  PID: {proc.pid}")
    return proc


def run_and_wait(script, args_list, log_file, results_path, description, min_seeds=None):
    """Run experiment and wait for it to finish."""
    proc = run_experiment(script, args_list, log_file, description)
    proc.wait()
    rc = proc.returncode
    if rc != 0:
        log(f"  WARNING: {description} exited with code {rc}")
    if results_path.exists():
        log(f"  Results saved: {results_path.name}")
    else:
        log(f"  ERROR: Expected {results_path.name} not found!")
    return rc


def analyze_results(results_path, description):
    """Run analysis script on results."""
    if not results_path.exists():
        log(f"  Skipping analysis for {description} - no results file")
        return
    log(f"Analyzing {description}...")
    cmd = [sys.executable, str(EXP_DIR / "analyze_results.py"), str(results_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_DIR))
    if result.stdout:
        print(result.stdout, flush=True)
    if result.returncode != 0:
        log(f"  Analysis warning (rc={result.returncode})")


def main():
    parser = argparse.ArgumentParser(description="Experiment pipeline: V10 -> V11 -> V12")
    parser.add_argument("--skip-v10-wait", action="store_true",
                        help="Skip waiting for V10 and start V11 immediately")
    parser.add_argument("--start-from", choices=["v10", "v11", "v12"], default="v10",
                        help="Which experiment to start from")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds for full runs")
    parser.add_argument("--skip-diagnostic", action="store_true",
                        help="Skip diagnostic runs and go straight to full")
    args = parser.parse_args()

    log("=" * 60)
    log("EXPERIMENT PIPELINE: V10 -> V11 -> V12")
    log(f"Seeds: {args.seeds}, Start from: {args.start_from}")
    log("=" * 60)

    v11_script = EXP_DIR / "run_verifiable_evolution_v11.py"
    v12_script = EXP_DIR / "run_verifiable_evolution_v12.py"

    # Step 1: Wait for V10 (if needed)
    if args.start_from == "v10" and not args.skip_v10_wait:
        if is_complete(V10_RESULTS, min_seeds=args.seeds):
            log("V10 already complete!")
        else:
            wait_for_completion(V10_RESULTS, min_seeds=args.seeds,
                                description="V10 full run")
        analyze_results(V10_RESULTS, "V10")

    # Step 2: V11
    if args.start_from in ("v10", "v11"):
        log("")
        log("=" * 60)
        log("STAGE: V11 (All 10 Codex fixes)")
        log("=" * 60)

        # Diagnostic first
        if not args.skip_diagnostic:
            run_and_wait(
                v11_script, ["--diagnostic"],
                EXP_DIR / "v11_diagnostic.log",
                V11_DIAG,
                "V11 diagnostic",
            )
            analyze_results(V11_DIAG, "V11 diagnostic")

        # Full run
        if not is_complete(V11_RESULTS, min_seeds=args.seeds):
            run_and_wait(
                v11_script, ["--seeds", str(args.seeds)],
                EXP_DIR / "v11_full.log",
                V11_RESULTS,
                "V11 full run",
                min_seeds=args.seeds,
            )
        else:
            log("V11 full run already complete!")
        analyze_results(V11_RESULTS, "V11")

    # Step 3: V12
    log("")
    log("=" * 60)
    log("STAGE: V12 (Mobius mutations + operator ablation)")
    log("=" * 60)

    # Diagnostic first
    if not args.skip_diagnostic:
        run_and_wait(
            v12_script, ["--diagnostic"],
            EXP_DIR / "v12_diagnostic.log",
            V12_DIAG,
            "V12 diagnostic",
        )
        analyze_results(V12_DIAG, "V12 diagnostic")

    # Full run
    if not is_complete(V12_RESULTS, min_seeds=args.seeds):
        run_and_wait(
            v12_script, ["--seeds", str(args.seeds)],
            EXP_DIR / "v12_full.log",
            V12_RESULTS,
            "V12 full run",
            min_seeds=args.seeds,
        )
    else:
        log("V12 full run already complete!")
    analyze_results(V12_RESULTS, "V12")

    log("")
    log("=" * 60)
    log("PIPELINE COMPLETE")
    log("=" * 60)
    log(f"Results:")
    for name, path in [("V10", V10_RESULTS), ("V11", V11_RESULTS), ("V12", V12_RESULTS)]:
        if path.exists():
            size = path.stat().st_size
            log(f"  {name}: {path.name} ({size} bytes)")
        else:
            log(f"  {name}: NOT FOUND")


if __name__ == "__main__":
    main()
