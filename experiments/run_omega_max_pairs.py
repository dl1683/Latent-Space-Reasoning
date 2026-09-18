"""Max-Over-Pairs Omega Predictor: Fix for multi-pair mixing weakness.

The multi-pair random splits showed omega loses to code-comment baseline
at tau=0.01-0.02 when pairs are mixed randomly. Hypothesis: taking max
omega across ALL calibration pairs (instead of averaging within-pair)
rescues the predictor because any pair detecting amplification is enough.

Uses data from pair_transfer experiment. No model runs needed.
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))


HEAVY_KW = ["heavy", "reassign", "updat", "chang"]
CODE_KW = ["identity", "type_check", "observe", "augment", "comparison", "repr"]


def run_max_pairs_analysis():
    print("=== Max-Over-Pairs Omega Analysis ===")

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
    cells = sorted(set((obs["var"], obs["val"]) for obs in observations))

    print(f"Pairs: {pairs}")
    print(f"Continuations: {conts}")
    print(f"Cells: {len(cells)}")

    rng = np.random.RandomState(42)
    n_splits = 100

    for tau in [0.01, 0.02, 0.05]:
        print(f"\n=== tau={tau} ===")

        per_pair_items = defaultdict(list)
        for pair_name in pairs:
            for cont_name in conts:
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
                    if ka not in obs_idx or kb not in obs_idx:
                        continue
                    d_post = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                    actual = "preserve" if d_post <= tau else "separate"

                    cn_lower = cont_name.lower()
                    is_heavy = any(k in cn_lower for k in HEAVY_KW)
                    is_code = any(k in cn_lower for k in CODE_KW)
                    keyword_pred = "separate" if is_heavy else "preserve"
                    code_comment_pred = "separate" if (is_heavy or is_code) else "preserve"

                    per_pair_items[pair_name].append({
                        "var": var, "val": val, "pair": pair_name,
                        "cont": cont_name, "d_post": d_post,
                        "actual": actual,
                        "keyword_pred": keyword_pred,
                        "code_comment_pred": code_comment_pred,
                    })

        all_items = []
        for pair_name in pairs:
            all_items.extend(per_pair_items[pair_name])

        print(f"  Total eligible items: {len(all_items)}")
        if len(all_items) < 10:
            print("  Too few eligible items")
            continue

        # Method 1: Original (within-pair omega, random split)
        orig_accs = []
        # Method 2: Max-over-pairs omega (for eval item, check max omega across ALL cal pairs)
        max_pair_accs = []
        # Method 3: Any-pair omega (if ANY cal pair for this cont shows amplification, predict separate)
        any_pair_accs = []
        # Baselines
        keyword_accs = []
        code_comment_accs = []

        for split_i in range(n_splits):
            idx = rng.permutation(len(all_items))
            mid = len(idx) // 2
            cal_idx = set(idx[:mid].tolist())
            eval_idx = idx[mid:]

            # Build calibration omega tables
            # Original: per-continuation max d_post
            cal_omega_orig = defaultdict(list)
            # Max-over-pairs: per-continuation, per-pair max d_post
            cal_omega_per_pair = defaultdict(lambda: defaultdict(list))

            for i in cal_idx:
                item = all_items[i]
                cal_omega_orig[item["cont"]].append(item["d_post"])
                cal_omega_per_pair[item["cont"]][item["pair"]].append(item["d_post"])

            orig_correct = 0
            max_pair_correct = 0
            any_pair_correct = 0
            keyword_correct = 0
            code_comment_correct = 0
            total = 0

            for i in eval_idx:
                item = all_items[i]
                total += 1

                # Original omega: max within all cal items for this cont
                if item["cont"] in cal_omega_orig and cal_omega_orig[item["cont"]]:
                    omega_val = max(cal_omega_orig[item["cont"]])
                    orig_pred = "preserve" if omega_val <= tau else "separate"
                else:
                    orig_pred = "preserve"

                # Max-over-pairs: for each cal pair, compute per-pair max, then take max across pairs
                if item["cont"] in cal_omega_per_pair:
                    pair_maxes = []
                    for p, vals in cal_omega_per_pair[item["cont"]].items():
                        pair_maxes.append(max(vals))
                    max_pair_omega = max(pair_maxes) if pair_maxes else 0.0
                    max_pair_pred = "preserve" if max_pair_omega <= tau else "separate"
                else:
                    max_pair_pred = "preserve"

                # Any-pair: if ANY pair shows mean d_post > tau, predict separate
                if item["cont"] in cal_omega_per_pair:
                    any_amplify = False
                    for p, vals in cal_omega_per_pair[item["cont"]].items():
                        if np.mean(vals) > tau:
                            any_amplify = True
                            break
                    any_pair_pred = "separate" if any_amplify else "preserve"
                else:
                    any_pair_pred = "preserve"

                orig_correct += int(orig_pred == item["actual"])
                max_pair_correct += int(max_pair_pred == item["actual"])
                any_pair_correct += int(any_pair_pred == item["actual"])
                keyword_correct += int(item["keyword_pred"] == item["actual"])
                code_comment_correct += int(item["code_comment_pred"] == item["actual"])

            if total > 0:
                orig_accs.append(orig_correct / total)
                max_pair_accs.append(max_pair_correct / total)
                any_pair_accs.append(any_pair_correct / total)
                keyword_accs.append(keyword_correct / total)
                code_comment_accs.append(code_comment_correct / total)

        if orig_accs:
            print(f"  Original omega:     {100*np.mean(orig_accs):.1f}% "
                  f"+/- {100*np.std(orig_accs):.1f}%")
            print(f"  Max-over-pairs:     {100*np.mean(max_pair_accs):.1f}% "
                  f"+/- {100*np.std(max_pair_accs):.1f}%")
            print(f"  Any-pair omega:     {100*np.mean(any_pair_accs):.1f}% "
                  f"+/- {100*np.std(any_pair_accs):.1f}%")
            print(f"  Keyword:            {100*np.mean(keyword_accs):.1f}% "
                  f"+/- {100*np.std(keyword_accs):.1f}%")
            print(f"  Code-comment:       {100*np.mean(code_comment_accs):.1f}% "
                  f"+/- {100*np.std(code_comment_accs):.1f}%")
            print(f"  Max-pair vs code-comment: "
                  f"+{100*(np.mean(max_pair_accs)-np.mean(code_comment_accs)):.1f}pp")
            print(f"  Any-pair vs code-comment: "
                  f"+{100*(np.mean(any_pair_accs)-np.mean(code_comment_accs)):.1f}pp")

    # Now test leave-one-pair-out with max-over-pairs
    print("\n\n=== Leave-One-Pair-Out with Max-Over-Pairs ===")
    for tau in [0.01, 0.02, 0.05]:
        print(f"\n--- tau={tau} ---")
        for held_out_pair in pairs:
            train_pairs = [p for p in pairs if p != held_out_pair]

            # Build omega from train pairs
            omega_per_cont = defaultdict(list)
            for pair_name in train_pairs:
                for item in per_pair_items.get(pair_name, []):
                    omega_per_cont[item["cont"]].append(item["d_post"])

            # Predict on held-out pair
            correct = 0
            code_comment_correct = 0
            total = 0
            for item in per_pair_items.get(held_out_pair, []):
                total += 1
                # Max omega from training pairs
                if item["cont"] in omega_per_cont and omega_per_cont[item["cont"]]:
                    omega_val = max(omega_per_cont[item["cont"]])
                    pred = "preserve" if omega_val <= tau else "separate"
                else:
                    pred = "preserve"
                correct += int(pred == item["actual"])
                code_comment_correct += int(item["code_comment_pred"] == item["actual"])

            if total > 0:
                print(f"  Held out: {held_out_pair:<20s}  "
                      f"MaxOmega: {100*correct/total:.0f}%  "
                      f"CodeComment: {100*code_comment_correct/total:.0f}%  "
                      f"(n={total})")

    out_dir = Path("experiments/results/omega_max_pairs")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "max_over_pairs_omega",
        "n_splits": n_splits,
        "pairs": pairs,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone.")
    return summary


if __name__ == "__main__":
    run_max_pairs_analysis()
