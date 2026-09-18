"""Omega Rank Transfer: Is omega's ORDERING of continuations stable across pairs?

Even if absolute omega values differ by pair, the ranking might be stable.
If identity always ranks above neutral across all pairs, that's a robust
signal even without cross-pair magnitude matching.

Tests Spearman rank correlation of continuation omega values across pairs.
Uses data from pair_transfer experiment. No model runs needed.
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))


def run_rank_transfer():
    print("=== Omega Rank Transfer Analysis ===")

    result_dir = Path("experiments/results/omega_pair_transfer")
    with open(result_dir / "observations.json") as f:
        observations = json.load(f)
    print(f"Loaded {len(observations)} observations")

    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["pair"], obs["side"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    pairs = sorted(set(obs["pair"] for obs in observations))
    conts = sorted(set(obs["continuation"] for obs in observations))
    conts_no_none = [c for c in conts if c != "none"]
    cells = sorted(set((obs["var"], obs["val"]) for obs in observations))

    for tau in [0.01, 0.02, 0.05]:
        print(f"\n=== tau={tau} ===")

        # For each pair, compute mean omega per continuation (across eligible cells)
        pair_omega = {}
        for pair_name in pairs:
            cont_omega = {}
            for cont_name in conts_no_none:
                tvs = []
                for var, val in cells:
                    ka_base = (var, val, pair_name, "a", "none")
                    kb_base = (var, val, pair_name, "b", "none")
                    if ka_base not in obs_idx or kb_base not in obs_idx:
                        continue
                    d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
                    if d_pre > tau:
                        continue
                    ka = (var, val, pair_name, "a", cont_name)
                    kb = (var, val, pair_name, "b", cont_name)
                    if ka in obs_idx and kb in obs_idx:
                        d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                        tvs.append(d)
                if tvs:
                    cont_omega[cont_name] = float(np.mean(tvs))
            pair_omega[pair_name] = cont_omega

        # Show per-pair omega table
        print(f"\n  Mean omega per pair x continuation:")
        header = f"  {'Continuation':<15s}"
        for p in pairs:
            header += f" {p[:12]:>12s}"
        print(header)
        for cont in conts_no_none:
            row = f"  {cont:<15s}"
            for p in pairs:
                if cont in pair_omega[p]:
                    row += f" {pair_omega[p][cont]:12.5f}"
                else:
                    row += f" {'---':>12s}"
            print(row)

        # Compute Spearman rank correlation between all pair combinations
        print(f"\n  Spearman rank correlation matrix:")
        common_conts = [c for c in conts_no_none
                       if all(c in pair_omega[p] for p in pairs)]
        if len(common_conts) < 3:
            print(f"  Too few common continuations ({len(common_conts)}) for rank correlation")
            continue

        print(f"  Common continuations: {common_conts}")

        def spearman(x, y):
            n = len(x)
            if n < 3:
                return float('nan')
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(y)).astype(float)
            d = rx - ry
            return 1.0 - 6.0 * np.sum(d**2) / (n * (n**2 - 1))

        header = f"  {'':>15s}"
        for p in pairs:
            header += f" {p[:12]:>12s}"
        print(header)

        corrs = []
        for p1 in pairs:
            row = f"  {p1[:15]:<15s}"
            for p2 in pairs:
                v1 = [pair_omega[p1][c] for c in common_conts]
                v2 = [pair_omega[p2][c] for c in common_conts]
                rho = spearman(v1, v2)
                row += f" {rho:12.3f}"
                if p1 != p2:
                    corrs.append(rho)
            print(row)

        if corrs:
            print(f"\n  Mean off-diagonal Spearman rho: {np.mean(corrs):.3f}")
            print(f"  Min: {np.min(corrs):.3f}, Max: {np.max(corrs):.3f}")

        # Rank-based predictor: for held-out pair, use median rank from training pairs
        print(f"\n  Rank-based leave-one-pair-out prediction:")
        for held_out in pairs:
            train = [p for p in pairs if p != held_out]
            # For each continuation, get median rank across training pairs
            cont_ranks = {}
            for c in common_conts:
                ranks = []
                for p in train:
                    # rank within this pair's continuations
                    vals = [(pair_omega[p][cc], cc) for cc in common_conts]
                    vals.sort()
                    for rank, (_, cc) in enumerate(vals):
                        if cc == c:
                            ranks.append(rank)
                            break
                cont_ranks[c] = np.median(ranks) if ranks else 0

            # Use rank threshold: top half = separate, bottom half = preserve
            sorted_conts = sorted(common_conts, key=lambda c: cont_ranks[c])
            mid = len(sorted_conts) // 2
            pred_preserve = set(sorted_conts[:mid])
            pred_separate = set(sorted_conts[mid:])

            correct = 0
            total = 0
            for c in common_conts:
                if c not in pair_omega[held_out]:
                    continue
                actual = "preserve" if pair_omega[held_out][c] <= tau else "separate"
                pred = "preserve" if c in pred_preserve else "separate"
                correct += int(pred == actual)
                total += 1

            if total > 0:
                print(f"    {held_out:<20s}: {100*correct/total:.0f}% ({correct}/{total})")

    # Continuous prediction: correlation between omega magnitude and actual d_post
    print("\n\n=== Omega as Continuous Predictor ===")
    print("Pearson correlation between calibration omega and held-out d_post")
    for tau in [0.01, 0.02, 0.05]:
        print(f"\n--- tau={tau} ---")
        for held_out in pairs:
            train = [p for p in pairs if p != held_out]
            common = [c for c in conts_no_none
                     if c in pair_omega[held_out]
                     and all(c in pair_omega[p] for p in train)]
            if len(common) < 3:
                continue
            # Mean omega across training pairs
            train_omega = [np.mean([pair_omega[p][c] for p in train]) for c in common]
            held_omega = [pair_omega[held_out][c] for c in common]
            if np.std(train_omega) > 0 and np.std(held_omega) > 0:
                r = np.corrcoef(train_omega, held_omega)[0, 1]
                print(f"  {held_out:<20s}: r={r:.3f} (n={len(common)} conts)")

    out_dir = Path("experiments/results/omega_rank_transfer")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"method": "rank_transfer", "pairs": pairs, "conts": conts_no_none}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    return summary


if __name__ == "__main__":
    run_rank_transfer()
