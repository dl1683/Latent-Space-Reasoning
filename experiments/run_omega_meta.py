"""Omega Meta-Analysis (Experiment #49).

Cross-experiment analysis: which continuations are consistently top
amplifiers vs preservers across all experiments? This aggregates
omega data from experiments 38, 43, 44, and 47 to find universal
patterns.

ANALYSIS ONLY — no new model runs.
"""

import json
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from omega import tv


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def run_meta():
    print("=== Omega Meta-Analysis (Exp #49) ===\n")

    # Load all available summaries
    datasets = {}

    # Exp #43 authority
    auth_path = Path("experiments/results/omega_authority/summary.json")
    if auth_path.exists():
        datasets["exp43_authority"] = load_summary(auth_path)

    # Exp #44 taxonomy
    tax_path = Path("experiments/results/omega_comment_taxonomy/summary.json")
    if tax_path.exists():
        datasets["exp44_taxonomy"] = load_summary(tax_path)

    # Exp #47 chain
    chain_path = Path("experiments/results/omega_chain/summary.json")
    if chain_path.exists():
        datasets["exp47_chain"] = load_summary(chain_path)

    print(f"Loaded {len(datasets)} experiment datasets\n")

    # Collect all continuation omega values across experiments and pairs
    all_omegas = {}  # cont_name -> list of (experiment, pair, tau, omega)

    for exp_name, data in datasets.items():
        pairs = data.get("pairs", {})
        for pair_name, pair_data in pairs.items():
            if isinstance(pair_data, dict):
                for tau_key, tau_data in pair_data.items():
                    if not isinstance(tau_data, dict):
                        continue
                    for cont_name, cont_data in tau_data.items():
                        if isinstance(cont_data, dict) and "omega" in cont_data:
                            if cont_name not in all_omegas:
                                all_omegas[cont_name] = []
                            all_omegas[cont_name].append({
                                "exp": exp_name,
                                "pair": pair_name,
                                "tau": tau_key,
                                "omega": cont_data["omega"],
                            })

    print("=== Continuation Frequency and Mean Omega ===\n")
    cont_stats = {}
    for cont, records in all_omegas.items():
        omegas = [r["omega"] for r in records]
        cont_stats[cont] = {
            "n_observations": len(records),
            "mean_omega": float(np.mean(omegas)),
            "median_omega": float(np.median(omegas)),
            "max_omega": float(np.max(omegas)),
            "min_omega": float(np.min(omegas)),
            "std_omega": float(np.std(omegas)),
            "n_amplifying_005": sum(1 for o in omegas if o > 0.05),
            "amp_rate_005": sum(1 for o in omegas if o > 0.05) / len(omegas),
        }

    for cont, stats in sorted(cont_stats.items(),
                               key=lambda x: x[1]["mean_omega"],
                               reverse=True):
        print(f"  {cont:25s}: mean={stats['mean_omega']:.4f} "
              f"median={stats['median_omega']:.4f} "
              f"max={stats['max_omega']:.4f} "
              f"amp_rate={stats['amp_rate_005']:.0%} "
              f"(n={stats['n_observations']})")

    # Identify universal amplifiers and preservers
    print("\n=== Universal Amplifiers (amp_rate > 80%) ===")
    for cont, stats in sorted(cont_stats.items(),
                               key=lambda x: x[1]["amp_rate_005"],
                               reverse=True):
        if stats["amp_rate_005"] >= 0.8 and stats["n_observations"] >= 3:
            print(f"  {cont:25s}: {stats['amp_rate_005']:.0%} "
                  f"(mean={stats['mean_omega']:.4f}, n={stats['n_observations']})")

    print("\n=== Universal Preservers (amp_rate < 20%) ===")
    for cont, stats in sorted(cont_stats.items(),
                               key=lambda x: x[1]["amp_rate_005"]):
        if stats["amp_rate_005"] <= 0.2 and stats["n_observations"] >= 3:
            print(f"  {cont:25s}: {stats['amp_rate_005']:.0%} "
                  f"(mean={stats['mean_omega']:.4f}, n={stats['n_observations']})")

    # Consistency analysis: which continuations have lowest variance?
    print("\n=== Most Consistent (lowest relative std) ===")
    consistent = [(c, s) for c, s in cont_stats.items()
                  if s["mean_omega"] > 0.01 and s["n_observations"] >= 3]
    consistent.sort(key=lambda x: x[1]["std_omega"] / max(x[1]["mean_omega"], 0.001))
    for cont, stats in consistent[:10]:
        cv = stats["std_omega"] / max(stats["mean_omega"], 0.001)
        print(f"  {cont:25s}: CV={cv:.2f} "
              f"(mean={stats['mean_omega']:.4f}, std={stats['std_omega']:.4f})")

    print("\n=== Most Variable (highest relative std) ===")
    consistent.sort(key=lambda x: x[1]["std_omega"] / max(x[1]["mean_omega"], 0.001),
                    reverse=True)
    for cont, stats in consistent[:10]:
        cv = stats["std_omega"] / max(stats["mean_omega"], 0.001)
        print(f"  {cont:25s}: CV={cv:.2f} "
              f"(mean={stats['mean_omega']:.4f}, std={stats['std_omega']:.4f})")

    # Summary
    summary = {
        "type": "meta_analysis",
        "source_experiments": list(datasets.keys()),
        "n_continuations": len(cont_stats),
        "n_total_observations": sum(s["n_observations"] for s in cont_stats.values()),
        "continuation_stats": cont_stats,
    }

    result_dir = Path("experiments/results/omega_meta")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {result_dir}/")
    return summary


if __name__ == "__main__":
    run_meta()
