"""Run full Gate 1b analysis pipeline."""
import subprocess
import sys

RESULT = sys.argv[1] if len(sys.argv) > 1 else \
    "experiments/results/svb_qwen3_gate1b/result.json"

scripts = [
    ("2x2 ANOVA", ["python", "experiments/analyze_gate1b_2x2.py", RESULT]),
    ("Fiber Square", ["python", "experiments/analyze_fiber_square.py", RESULT]),
    ("Multiplicative Character", ["python", "experiments/analyze_multiplicative_character.py", RESULT]),
    ("F6 Baseline", ["python", "experiments/f6_baseline_analysis.py", RESULT]),
    ("F8 Lumpability", ["python", "experiments/analyze_lumpability.py", RESULT]),
]

for name, cmd in scripts:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")
    subprocess.run(cmd)

print(f"\n{'='*60}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*60}")
