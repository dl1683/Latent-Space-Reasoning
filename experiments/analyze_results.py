"""
Analyze V10/V11 experiment results for Codex review.

Reads results JSON, formats for Codex, computes additional statistics,
and generates a concise summary suitable for review.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def format_for_codex(results: dict, version: str = "V11") -> str:
    """Format results into a concise summary for Codex review."""
    cfg = results["config"]
    stats = results["statistics"]
    curves = results.get("fitness_curves", {})

    lines = []
    lines.append(f"=== {version} EXPERIMENT RESULTS ===")
    lines.append(f"Model: {cfg['model']}")
    lines.append(f"Seeds: {cfg['seeds']}, Test tasks: {cfg.get('test_tasks_per_depth', 20)*2}")
    lines.append(f"Curvature: {cfg['curvature']}, Ball radius: {cfg['ball_radius']:.3f}")
    lines.append(f"Evolution: {cfg['evo_gens']} gens, pop={cfg['evo_pop']}, tasks/gen={cfg['evo_tasks']}")

    if "codex_fixes" in results:
        lines.append(f"\nCodex fixes applied ({len(results['codex_fixes'])}):")
        for fix in results["codex_fixes"]:
            lines.append(f"  - {fix}")

    # Overall accuracy
    lines.append("\n--- OVERALL ACCURACY ---")
    conditions = cfg["conditions"]
    for cond in conditions:
        s = stats["per_condition"][cond]
        seeds_str = ", ".join(f"{v*100:.1f}%" for v in s["per_seed"])
        lines.append(f"  {cond:22s}: {s['mean']*100:.1f}% +/- {s['std']*100:.1f}%  seeds=[{seeds_str}]")

    # Per-depth
    if "per_depth" in stats:
        lines.append("\n--- PER-DEPTH ---")
        for depth in sorted(stats["per_depth"].keys(), key=lambda x: int(x)):
            lines.append(f"  Depth {depth}:")
            for cond in conditions:
                ds = stats["per_depth"][str(depth) if isinstance(depth, int) else depth][cond]
                lines.append(f"    {cond:22s}: {ds['mean']*100:.1f}% +/- {ds['std']*100:.1f}%")

    # Pairwise comparisons
    lines.append("\n--- PAIRWISE COMPARISONS ---")
    primary = cfg.get("primary_comparison", "euc_constrained_vs_hyperbolic")
    for pair_key, ps in stats["pairwise"].items():
        is_primary = (pair_key == primary)
        tag = " [PRIMARY, PRE-REGISTERED]" if is_primary else ""
        lines.append(f"\n  {pair_key}{tag}:")
        lines.append(f"    Diff: {ps['diff_mean']*100:+.1f}%")
        ci = ps.get("diff_ci_95", [float("nan"), float("nan")])
        if not (math.isnan(ci[0]) if isinstance(ci[0], float) else False):
            lines.append(f"    95% CI: [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
        p_raw = ps.get("p_value_raw", float("nan"))
        p_bonf = ps.get("p_value_bonferroni", float("nan"))
        t_stat = ps.get("t_stat", float("nan"))
        lines.append(f"    Paired t: t={t_stat:.3f}, p_raw={p_raw:.4f}, p_bonf={p_bonf:.4f}")

        if "per_seed_mcnemar" in ps:
            mc_ps = [f"{m['p']:.3f}" for m in ps["per_seed_mcnemar"]]
            lines.append(f"    Per-seed McNemar p: [{', '.join(mc_ps)}]")

    # Fitness curves
    if curves:
        lines.append("\n--- FITNESS CURVES ---")
        for cond, seed_curves in curves.items():
            lines.append(f"  {cond}:")
            for si, curve in enumerate(seed_curves):
                gens = " -> ".join(f"{e['best']:.3f}" for e in curve)
                lines.append(f"    Seed {si+1}: {gens}")

    # Verdict
    if "verdict" in results:
        lines.append(f"\n--- VERDICT ---")
        lines.append(f"  {results['verdict']}")

    # Effect size (Cohen's d) for primary comparison
    if primary in stats["pairwise"]:
        ps = stats["pairwise"][primary]
        diff_mean = ps["diff_mean"]
        diff_std = ps.get("diff_std", 0)
        if diff_std and not math.isnan(diff_std) and diff_std > 0:
            cohens_d = diff_mean / diff_std
            lines.append(f"\n--- EFFECT SIZE ---")
            lines.append(f"  Cohen's d (primary): {cohens_d:.3f}")
            if abs(cohens_d) < 0.2:
                lines.append(f"  Interpretation: Negligible effect")
            elif abs(cohens_d) < 0.5:
                lines.append(f"  Interpretation: Small effect")
            elif abs(cohens_d) < 0.8:
                lines.append(f"  Interpretation: Medium effect")
            else:
                lines.append(f"  Interpretation: Large effect")

    return "\n".join(lines)


def compute_win_rate(results: dict) -> str:
    """Compute per-task win/loss/tie for each condition pair."""
    stats = results["statistics"]
    conditions = results["config"]["conditions"]
    n_seeds = results["config"]["seeds"]

    lines = ["--- WIN/LOSS/TIE PER TASK (across seeds) ---"]

    # This requires raw per-task results which may not be in the JSON
    # If available, compute them
    lines.append("  (Requires raw per-task results - check JSON structure)")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        # Auto-detect latest results
        exp_dir = Path(__file__).parent
        candidates = sorted(exp_dir.glob("v1*_results*.json"), reverse=True)
        if not candidates:
            print("No results files found. Usage: python analyze_results.py <results.json>")
            return
        path = candidates[0]
        print(f"Auto-detected: {path.name}")
    else:
        path = Path(sys.argv[1])

    if not path.exists():
        print(f"File not found: {path}")
        return

    results = load_results(str(path))
    version = "V11" if "v11" in path.name else "V10" if "v10" in path.name else "Unknown"
    summary = format_for_codex(results, version)
    print(summary)

    # Save formatted summary
    summary_path = path.parent / f"{path.stem}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
