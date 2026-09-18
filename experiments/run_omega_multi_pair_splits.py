"""Multi-Pair Randomized Splits: The definitive predictive test.

Uses all 5 good surface pairs and 6 cells for 100 randomized cal/eval
splits. This is the culminating test: omega vs baselines with proper
statistical confidence, using data from the pair transfer experiment.

Requires: experiments/results/omega_pair_transfer/observations.json
(run run_omega_pair_transfer.py first)

No model runs needed — pure analysis of existing data.
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))


def load_data():
    result_dir = Path("experiments/results/omega_pair_transfer")
    with open(result_dir / "observations.json") as f:
        observations = json.load(f)
    return observations


HEAVY_KW = ["heavy", "reassign", "updat", "chang"]
CODE_KW = ["identity", "type_check", "observe", "augment", "comparison", "repr"]


def run_multi_pair_splits():
    print("=== Multi-Pair Randomized Splits ===")
    observations = load_data()
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

        eligible_items = []
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

                    eligible_items.append({
                        "var": var, "val": val, "pair": pair_name,
                        "cont": cont_name, "d_post": d_post,
                        "actual": actual,
                        "keyword_pred": keyword_pred,
                        "code_comment_pred": code_comment_pred,
                    })

        print(f"  Eligible items: {len(eligible_items)}")
        if len(eligible_items) < 10:
            print("  Too few eligible items for split test")
            continue

        omega_accs = []
        keyword_accs = []
        code_comment_accs = []
        calibration_accs = []

        for split_i in range(n_splits):
            idx = rng.permutation(len(eligible_items))
            mid = len(idx) // 2
            cal_idx = idx[:mid]
            eval_idx = idx[mid:]

            cal_omega = defaultdict(list)
            for i in cal_idx:
                item = eligible_items[i]
                cal_omega[item["cont"]].append(item["d_post"])

            cal_mean_d = np.mean([item["d_post"] for i in cal_idx
                                  for item in [eligible_items[i]]])

            omega_correct = 0
            keyword_correct = 0
            code_comment_correct = 0
            calibration_correct = 0
            total = 0

            for i in eval_idx:
                item = eligible_items[i]
                total += 1

                if item["cont"] in cal_omega and cal_omega[item["cont"]]:
                    omega_val = max(cal_omega[item["cont"]])
                    omega_pred = "preserve" if omega_val <= tau else "separate"
                else:
                    omega_pred = "preserve"

                omega_correct += int(omega_pred == item["actual"])
                keyword_correct += int(item["keyword_pred"] == item["actual"])
                code_comment_correct += int(item["code_comment_pred"] == item["actual"])

                cal_pred = "separate" if cal_mean_d > tau else "preserve"
                calibration_correct += int(cal_pred == item["actual"])

            if total > 0:
                omega_accs.append(omega_correct / total)
                keyword_accs.append(keyword_correct / total)
                code_comment_accs.append(code_comment_correct / total)
                calibration_accs.append(calibration_correct / total)

        if omega_accs:
            print(f"  Omega:        {100*np.mean(omega_accs):.1f}% "
                  f"+/- {100*np.std(omega_accs):.1f}%")
            print(f"  Keyword:      {100*np.mean(keyword_accs):.1f}% "
                  f"+/- {100*np.std(keyword_accs):.1f}%")
            print(f"  Code-comment: {100*np.mean(code_comment_accs):.1f}% "
                  f"+/- {100*np.std(code_comment_accs):.1f}%")
            print(f"  Calibration:  {100*np.mean(calibration_accs):.1f}% "
                  f"+/- {100*np.std(calibration_accs):.1f}%")
            print(f"  Omega advantage over code-comment: "
                  f"+{100*(np.mean(omega_accs)-np.mean(code_comment_accs)):.1f}pp")

    result_dir = Path("experiments/results/omega_pair_transfer")
    summary = {
        "method": "multi_pair_randomized_splits",
        "n_splits": n_splits,
        "n_items": len(eligible_items),
        "pairs": pairs,
    }
    with open(result_dir / "multi_pair_splits.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone.")
    return summary


if __name__ == "__main__":
    run_multi_pair_splits()
