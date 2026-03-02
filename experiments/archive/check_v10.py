"""Quick V10 status checker. Run: python experiments/check_v10.py"""
import os
import json
from pathlib import Path

results_path = Path(__file__).parent / "v10_results.json"
diagnostic_path = Path(__file__).parent / "v10_results_diagnostic.json"

if results_path.exists():
    size = results_path.stat().st_size
    mtime = results_path.stat().st_mtime

    with open(results_path) as f:
        data = json.load(f)

    seeds = data.get("config", {}).get("seeds", "?")
    verdict = data.get("verdict", "unknown")

    print(f"v10_results.json: {size} bytes, seeds={seeds}")
    print(f"Verdict: {verdict}")

    if seeds == 5 and size > 5000:
        print("\n*** V10 FULL RUN COMPLETE ***")
        print("Next: python experiments/analyze_results.py experiments/v10_results.json")
    else:
        print(f"\nStill running (diagnostic data, {seeds} seed(s))")
        # Check if process is alive
        try:
            import subprocess
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command",
                 "Get-Process -Id 59748 -ErrorAction SilentlyContinue | Select-Object Id, CPU | Format-Table -AutoSize"],
                capture_output=True, text=True, timeout=10,
            )
            if result.stdout.strip():
                print(f"Process 59748: {result.stdout.strip()}")
            else:
                print("Process 59748 NOT FOUND - may have completed or crashed")
        except Exception:
            pass
else:
    print("v10_results.json not found")
