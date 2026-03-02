"""
Analyze V10/V11/V12 experiment results for Codex review.

Reads results JSON, formats for Codex, computes additional statistics,
and generates a concise summary suitable for review.

Usage:
    python analyze_results.py                        # Auto-detect latest
    python analyze_results.py experiments/v12_results.json  # Specific file
    python analyze_results.py --compare              # Cross-version comparison
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
        for depth_key in sorted(stats["per_depth"].keys(), key=lambda x: int(x)):
            lines.append(f"  Depth {depth_key}:")
            depth_data = stats["per_depth"][depth_key]
            for cond in conditions:
                if cond not in depth_data:
                    continue
                ds = depth_data[cond]
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


def format_radius_diagnostics(results: dict) -> str:
    """Format V12 radius diagnostics showing norm evolution during mutations."""
    diag = results.get("radius_diagnostics", {})
    if not diag:
        return ""

    lines = ["\n--- RADIUS DIAGNOSTICS ---"]
    for cond, seed_diags in diag.items():
        if not seed_diags:
            continue
        lines.append(f"  {cond}:")
        for si, gens in enumerate(seed_diags):
            if not gens:
                continue
            last_gen = gens[-1]
            frac = last_gen.get("norm_as_fraction_of_ball", 0)
            lines.append(f"    Seed {si+1}: final norm/ball = {frac:.3f}")
    return "\n".join(lines)


def cross_version_compare(exp_dir: Path) -> str:
    """Compare results across V9, V10, V11, V12."""
    lines = ["=" * 60]
    lines.append("CROSS-VERSION COMPARISON")
    lines.append("=" * 60)

    versions = {}
    # V9
    v9_path = exp_dir / "v9_rigorous_results.json"
    if v9_path.exists():
        versions["V9"] = load_results(str(v9_path))
    # V10
    v10_path = exp_dir / "v10_results.json"
    if v10_path.exists():
        v10 = load_results(str(v10_path))
        if v10.get("config", {}).get("seeds", 0) >= 5:
            versions["V10"] = v10
        else:
            v10d_path = exp_dir / "v10_results_diagnostic.json"
            if v10d_path.exists():
                versions["V10-diag"] = load_results(str(v10d_path))
    # V11
    for suffix in ["", "_diagnostic"]:
        path = exp_dir / f"v11_results{suffix}.json"
        if path.exists():
            label = "V11" if not suffix else "V11-diag"
            versions[label] = load_results(str(path))
    # V12
    for suffix in ["", "_diagnostic"]:
        path = exp_dir / f"v12_results{suffix}.json"
        if path.exists():
            label = "V12" if not suffix else "V12-diag"
            versions[label] = load_results(str(path))

    if not versions:
        return "No results files found."

    lines.append(f"\nVersions found: {', '.join(versions.keys())}")

    # Overall accuracy table
    lines.append("\n--- OVERALL ACCURACY ---")
    lines.append(f"  {'Version':<12} {'Condition':<24} {'Mean':>8} {'Std':>8} {'Seeds':>6}")
    lines.append("  " + "-" * 60)

    for ver_name, ver_data in versions.items():
        cfg = ver_data.get("config", {})
        stats = ver_data.get("statistics", {})

        # Handle V9 format (different structure)
        if "hyp_mean" in stats:
            lines.append(f"  {ver_name:<12} {'hyperbolic':<24} {stats['hyp_mean']*100:>7.1f}% {stats.get('hyp_std', 0)*100:>7.1f}% {cfg.get('seeds', '?'):>6}")
            lines.append(f"  {'':<12} {'euclidean':<24} {stats['euc_mean']*100:>7.1f}% {stats.get('euc_std', 0)*100:>7.1f}%")
            continue

        # V10+ format
        per_cond = stats.get("per_condition", {})
        first = True
        for cond, s in per_cond.items():
            label = ver_name if first else ""
            seeds = cfg.get("seeds", "?")
            lines.append(f"  {label:<12} {cond:<24} {s['mean']*100:>7.1f}% {s['std']*100:>7.1f}% {seeds if first else '':>6}")
            first = False

    # Primary comparison across versions
    lines.append("\n--- PRIMARY COMPARISON (hyp vs euc_constrained) ---")
    lines.append(f"  {'Version':<12} {'Diff':>8} {'p-value':>10} {'Verdict':<30}")
    lines.append("  " + "-" * 60)

    for ver_name, ver_data in versions.items():
        stats = ver_data.get("statistics", {})

        # V9 format
        if "diff_mean" in stats:
            diff = stats["diff_mean"]
            p = stats.get("p_value", float("nan"))
            verdict = ver_data.get("verdict", "")
            lines.append(f"  {ver_name:<12} {diff*100:>+7.1f}% {p:>10.4f} {verdict:<30}")
            continue

        # V10+ format - find primary comparison
        pairwise = stats.get("pairwise", {})
        primary = ver_data.get("config", {}).get("primary_comparison", "")
        # Try common keys
        for key in [primary, "euc_constrained_vs_hyperbolic", "euc_constrained_vs_hyp_mobius"]:
            if key in pairwise:
                ps = pairwise[key]
                diff = ps.get("diff_mean", float("nan"))
                p_raw = ps.get("p_value_raw", ps.get("p_value", float("nan")))
                p_bonf = ps.get("p_value_bonferroni", float("nan"))
                p_display = p_bonf if not (isinstance(p_bonf, float) and math.isnan(p_bonf)) else p_raw
                verdict = ver_data.get("verdict", "")[:30]
                p_str = f"{p_display:>10.4f}" if not (isinstance(p_display, float) and math.isnan(p_display)) else "       n/a"
                lines.append(f"  {ver_name:<12} {diff*100:>+7.1f}% {p_str} {verdict:<30}")
                break

    return "\n".join(lines)


def detect_version(path: Path) -> str:
    name = path.name.lower()
    if "v12" in name:
        return "V12"
    elif "v11" in name:
        return "V11"
    elif "v10" in name:
        return "V10"
    elif "v9" in name:
        return "V9"
    return "Unknown"


def main():
    exp_dir = Path(__file__).parent

    if "--compare" in sys.argv:
        comparison = cross_version_compare(exp_dir)
        print(comparison)
        summary_path = exp_dir / "cross_version_comparison.txt"
        with open(summary_path, "w") as f:
            f.write(comparison)
        print(f"\nSaved to: {summary_path}")
        return

    if len(sys.argv) < 2 or sys.argv[1] == "--compare":
        # Auto-detect latest results
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
    version = detect_version(path)
    summary = format_for_codex(results, version)
    print(summary)

    # V12 radius diagnostics
    radius_info = format_radius_diagnostics(results)
    if radius_info:
        print(radius_info)
        summary += radius_info

    # Save formatted summary
    summary_path = path.parent / f"{path.stem}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
