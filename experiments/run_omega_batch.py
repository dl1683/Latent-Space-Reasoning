"""Run all short omega experiments in sequence.

Chains: controls, composition, depth, values, queries.
Total: ~4,344 readouts, ~22 minutes.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_all():
    t0 = time.time()
    results = {}

    experiments = [
        ("controls", "run_omega_controls", "run_controls"),
        ("composition", "run_omega_composition", "run_composition_sweep"),
        ("depth", "run_omega_depth", "run_depth_sweep"),
        ("values", "run_omega_values", "run_value_sweep"),
        ("queries", "run_omega_queries", "run_query_sweep"),
    ]

    for name, module_name, func_name in experiments:
        print(f"\n{'='*60}")
        print(f"  STARTING: {name}")
        print(f"{'='*60}\n")
        t_start = time.time()

        try:
            mod = __import__(module_name)
            func = getattr(mod, func_name)
            obs = func()
            elapsed = time.time() - t_start
            results[name] = {
                "status": "ok",
                "n_observations": len(obs),
                "elapsed_s": elapsed,
            }
            print(f"\n  {name}: {len(obs)} observations in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t_start
            results[name] = {
                "status": "error",
                "error": str(e),
                "elapsed_s": elapsed,
            }
            print(f"\n  {name}: ERROR after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE: {total:.1f}s total")
    print(f"{'='*60}")
    for name, r in results.items():
        if r["status"] == "ok":
            print(f"  {name:15s}: {r['n_observations']:5d} obs, {r['elapsed_s']:.1f}s")
        else:
            print(f"  {name:15s}: ERROR — {r['error']}")
    return results


if __name__ == "__main__":
    run_all()
