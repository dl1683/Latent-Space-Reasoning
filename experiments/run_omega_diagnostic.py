"""Omega Minimal Diagnostic Set (Experiment #45).

Can a small set of 3 continuation probes predict the full omega profile
of novel continuations? This tests the practical question: do you need
to test every possible continuation, or can a diagnostic set serve as
a proxy?

Design:
  - Use the full dataset from exp #38 (partial identity, 720 readouts)
  - Train: greedy forward selection of best 3 continuations
  - Test: held-out continuations
  - Metric: Spearman rank correlation between predicted and actual omega

This experiment is ANALYSIS ONLY — no new model runs.
"""

import json
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from omega import tv


def load_experiment_data(result_dir):
    obs_path = Path(result_dir) / "observations.json"
    with open(obs_path) as f:
        observations = json.load(f)
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)
    return obs_idx


def rank_correlation(a, b):
    """Spearman rank correlation."""
    n = len(a)
    if n < 3:
        return 0.0
    a_ranks = np.argsort(np.argsort(a)).astype(float)
    b_ranks = np.argsort(np.argsort(b)).astype(float)
    d = a_ranks - b_ranks
    return float(1.0 - 6.0 * np.sum(d**2) / (n * (n**2 - 1)))


EQUIV_PAIRS = [
    ("assert_novar_1", "assert_var_1", "strong_assert"),
    ("assert_novar_2", "assert_var_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CONTS = [
    "baseline_none", "bare_newline", "pass_stmt", "var_only",
    "var_equals", "identity_no_nl", "identity_full", "dangling_assign",
    "no_lhs", "cross_var_assign", "literal_assign", "none_assign",
    "comment_var", "type_check", "observe",
]

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]


def compute_omega_per_cont(obs_idx, pair, cells, conts, tau):
    """Compute omega for each continuation on one pair."""
    surf_a, surf_b, pair_name = pair
    eligible = set()
    for depth, var, val in cells:
        key_a = (depth, var, val, surf_a, "baseline_none")
        key_b = (depth, var, val, surf_b, "baseline_none")
        if key_a in obs_idx and key_b in obs_idx:
            d = tv(obs_idx[key_a], obs_idx[key_b])
            if d <= tau:
                eligible.add((depth, var, val))

    if not eligible:
        return None, 0

    result = {}
    for cont in conts:
        tvs = []
        for cell in eligible:
            depth, var, val = cell
            key_a = (depth, var, val, surf_a, cont)
            key_b = (depth, var, val, surf_b, cont)
            if key_a in obs_idx and key_b in obs_idx:
                tvs.append(tv(obs_idx[key_a], obs_idx[key_b]))
        if tvs:
            result[cont] = float(np.max(tvs))
    return result, len(eligible)


def greedy_select(profiles, conts, k=3):
    """Greedy forward selection of k diagnostic continuations."""
    pair_names = list(profiles.keys())

    def predict_accuracy(selected, held_out):
        total_rho = 0.0
        count = 0
        for pair in pair_names:
            pp = profiles[pair]
            if not pp:
                continue
            sel_vals = [pp.get(s, 0) for s in selected]
            pred = np.mean(sel_vals)
            actual_vec = []
            pred_vec = []
            for c in held_out:
                if c in pp:
                    actual_vec.append(pp[c])
                    pred_vec.append(pred)
            if len(actual_vec) >= 3:
                total_rho += rank_correlation(actual_vec, pred_vec)
                count += 1
        return total_rho / count if count > 0 else 0

    selected = []
    remaining = set(conts)

    for step in range(k):
        best_c = None
        best_score = -999
        for cand in remaining:
            trial = selected + [cand]
            held = [c for c in conts if c not in trial]
            s = predict_accuracy(trial, held)
            if s > best_score:
                best_score = s
                best_c = cand
        if best_c:
            selected.append(best_c)
            remaining.discard(best_c)
            print(f"  Step {step+1}: '{best_c}' (avg rho={best_score:.3f})")

    return selected


def loo_evaluate(profiles, diagnostic, conts):
    """Leave-one-out: for each held-out continuation, predict its ranking."""
    pair_names = list(profiles.keys())
    results = {}

    for pair in pair_names:
        pp = profiles[pair]
        if not pp or len(pp) < 5:
            continue

        diag_omegas = [pp.get(d, 0) for d in diagnostic if d in pp]
        if not diag_omegas:
            continue

        held_out = [c for c in conts if c not in diagnostic and c in pp]
        if len(held_out) < 3:
            continue

        actual_vals = [pp[c] for c in held_out]
        actual_ranking = np.argsort(np.argsort(actual_vals))

        diag_mean = np.mean(diag_omegas)
        diag_max = np.max(diag_omegas)

        amplify_actual = sum(1 for v in actual_vals if v > 0.05)
        amplify_pred_high = len(actual_vals) if diag_max > 0.05 else 0

        classification_correct = 0
        for i, c in enumerate(held_out):
            actual_amp = pp[c] > 0.05
            pred_amp = diag_max > 0.05
            if actual_amp == pred_amp:
                classification_correct += 1

        top_actual = set(np.argsort(actual_vals)[-3:])
        diag_ratios = []
        for d in diagnostic:
            if d in pp:
                diag_ratios.append(pp[d])
        top_predicted = set(np.argsort([
            abs(v - np.mean(diag_ratios)) for v in actual_vals
        ])[-3:])

        results[pair] = {
            "n_held_out": len(held_out),
            "n_amplify_actual": amplify_actual,
            "diagnostic_max_omega": float(diag_max),
            "diagnostic_mean_omega": float(diag_mean),
            "held_out_omegas": {c: pp[c] for c in held_out},
        }

    return results


def run_diagnostic():
    print("=== Omega Minimal Diagnostic Set (Exp #45) ===")
    print("Analysis-only: uses existing data from exp #38\n")

    obs_idx = load_experiment_data("experiments/results/omega_partial")
    print(f"Loaded {len(obs_idx)} observations\n")

    summary = {"type": "analysis_only", "source": "exp_38_partial_identity"}

    for tau in [0.01, 0.05]:
        print(f"\n{'='*60}")
        print(f"  TAU = {tau}")
        print(f"{'='*60}")

        profiles = {}
        for pair in EQUIV_PAIRS:
            pp, n_elig = compute_omega_per_cont(
                obs_idx, pair, CELLS, CONTS, tau)
            if pp:
                profiles[pair[2]] = pp
                print(f"\n  {pair[2]} ({n_elig} eligible cells):")
                for c, o in sorted(pp.items(), key=lambda x: x[1],
                                   reverse=True)[:5]:
                    print(f"    {c:22s}: {o:.5f}")
                print(f"    ... ({len(pp)} total)")

        if not profiles:
            print("  No eligible pairs.")
            continue

        print(f"\n  Greedy diagnostic selection:")
        diagnostic = greedy_select(profiles, CONTS, k=3)
        print(f"  Selected: {diagnostic}")

        print(f"\n  Leave-one-out evaluation:")
        eval_res = loo_evaluate(profiles, diagnostic, CONTS)
        for pair, er in eval_res.items():
            print(f"\n  {pair}:")
            print(f"    Diagnostic max omega: {er['diagnostic_max_omega']:.5f}")
            print(f"    Held-out amplifiers: {er['n_amplify_actual']}/{er['n_held_out']}")

        summary[f"tau_{tau}"] = {
            "diagnostic_set": diagnostic,
            "profiles": profiles,
            "evaluation": eval_res,
        }

    result_dir = Path("experiments/results/omega_diagnostic")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {result_dir}/")
    return summary


if __name__ == "__main__":
    run_diagnostic()
