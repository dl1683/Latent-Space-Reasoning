"""Unified analysis of all omega experiment results.

Reads all results dirs and produces:
1. Cross-experiment summary of which continuations preserve vs amplify
2. Omega robustness check across depth/value/query variations
3. Key disagreements between omega prediction and action-only baseline
4. Data for prospective prediction test (cal/eval split)
"""

import json
import numpy as np
from pathlib import Path


def load_observations(result_dir):
    path = Path(result_dir) / "observations.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def tv(a, b):
    return 0.5 * np.abs(np.array(a) - np.array(b)).sum()


def analyze_sweep(result_dir):
    obs = load_observations(result_dir)
    if obs is None:
        return None
    omega_path = Path(result_dir) / "omega_table.json"
    if omega_path.exists():
        with open(omega_path) as f:
            return json.load(f)
    return {"n_obs": len(obs)}


def analyze_controls(result_dir):
    path = Path(result_dir) / "control_stats.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def analyze_composition(result_dir):
    path = Path(result_dir) / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def analyze_depth(result_dir):
    path = Path(result_dir) / "depth_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def analyze_values(result_dir):
    path = Path(result_dir) / "value_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def analyze_queries(result_dir):
    path = Path(result_dir) / "query_stats.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def prospective_test(sweep_dir, tau=0.01):
    """Run a prospective prediction test on sweep data.

    Split: use half the equiv pairs for calibration, half for evaluation.
    Predict from calibration omega whether eval pairs preserve or separate.
    Compare against action-only baseline.
    """
    omega_path = Path(sweep_dir) / "omega_table.json"
    pair_path = Path(sweep_dir) / "pair_distances.json"
    if not omega_path.exists() or not pair_path.exists():
        return None

    with open(omega_path) as f:
        omega_table = json.load(f)
    with open(pair_path) as f:
        pair_dists = json.load(f)

    pair_names = list(omega_table.keys())
    if len(pair_names) < 2:
        return {"error": "Not enough pairs for cal/eval split"}

    mid = len(pair_names) // 2
    cal_pairs = pair_names[:mid]
    eval_pairs = pair_names[mid:]

    all_conts = set()
    for pn in pair_names:
        all_conts.update(omega_table[pn]["continuations"].keys())

    cal_omega = {}
    for cn in all_conts:
        omegas = []
        for pn in cal_pairs:
            cr = omega_table[pn]["continuations"].get(cn, {})
            if cr.get("assessable") and cr.get("omega") is not None:
                omegas.append(cr["omega"])
        if omegas:
            cal_omega[cn] = max(omegas)

    omega_predictions = {}
    baseline_predictions = {}
    actuals = {}

    HEAVY_KEYWORDS = ["heavy", "reassign", "updat", "chang", "mislead",
                       "neutral_then_heavy"]
    NEUTRAL_KEYWORDS = ["neutral", "pass", "none", "bare", "ellipsis",
                         "identity", "observe", "verified", "ok",
                         "comment", "assert", "type_check", "print",
                         "double_neutral", "heavy_then_neutral"]

    for cn in all_conts:
        if cn in cal_omega:
            omega_predictions[cn] = "preserve" if cal_omega[cn] <= tau else "separate"

        cn_lower = cn.lower()
        is_heavy = any(k in cn_lower for k in HEAVY_KEYWORDS)
        if cn_lower.startswith("heavy_then_"):
            is_heavy = True
        if cn_lower == "neutral_then_heavy":
            is_heavy = True
        baseline_predictions[cn] = "separate" if is_heavy else "preserve"

        eval_tvs = []
        for pn in eval_pairs:
            cr = omega_table[pn]["continuations"].get(cn, {})
            if cr.get("assessable") and cr.get("omega") is not None:
                eval_tvs.append(cr["omega"])
        if eval_tvs:
            actual_max = max(eval_tvs)
            actuals[cn] = "preserve" if actual_max <= tau else "separate"

    omega_correct = 0
    baseline_correct = 0
    total = 0
    details = []

    for cn in sorted(actuals.keys()):
        if cn not in omega_predictions:
            continue
        total += 1
        o_correct = omega_predictions[cn] == actuals[cn]
        b_correct = baseline_predictions[cn] == actuals[cn]
        omega_correct += int(o_correct)
        baseline_correct += int(b_correct)
        details.append({
            "continuation": cn,
            "omega_pred": omega_predictions[cn],
            "baseline_pred": baseline_predictions[cn],
            "actual": actuals[cn],
            "omega_correct": o_correct,
            "baseline_correct": b_correct,
            "disagree": omega_predictions[cn] != baseline_predictions[cn],
        })

    return {
        "cal_pairs": cal_pairs,
        "eval_pairs": eval_pairs,
        "n_continuations": total,
        "omega_accuracy": omega_correct / total if total > 0 else None,
        "baseline_accuracy": baseline_correct / total if total > 0 else None,
        "omega_correct": omega_correct,
        "baseline_correct": baseline_correct,
        "disagreements": [d for d in details if d["disagree"]],
        "details": details,
    }


def main():
    base = Path("experiments/results")

    print("=" * 70)
    print("  UNIFIED OMEGA ANALYSIS")
    print("=" * 70)

    sweep_fast = analyze_sweep(base / "omega_sweep_fast")
    if sweep_fast:
        print("\n--- Fast Sweep Results ---")
        for pair_name, pt in sweep_fast.items():
            if not isinstance(pt, dict) or "continuations" not in pt:
                continue
            n_elig = pt.get("n_eligible_cells", 0)
            print(f"\n  {pair_name} ({n_elig} eligible cells):")
            if n_elig == 0:
                continue
            conts = pt["continuations"]
            preserving = [cn for cn, cr in conts.items()
                         if cr.get("assessable") and cr.get("preserves")]
            amplifying = [cn for cn, cr in conts.items()
                         if cr.get("assessable") and not cr.get("preserves", True)]
            print(f"    Preserving ({len(preserving)}): {', '.join(sorted(preserving))}")
            print(f"    Amplifying ({len(amplifying)}): {', '.join(sorted(amplifying))}")

    controls = analyze_controls(base / "omega_controls")
    if controls:
        print("\n--- Controls Results ---")
        preserving = [k for k, v in controls.items()
                     if v.get("n", 0) > 0 and v.get("preserves", False)]
        amplifying = [k for k, v in controls.items()
                     if v.get("n", 0) > 0 and not v.get("preserves", True)]
        print(f"  Preserving ({len(preserving)}): {', '.join(sorted(preserving))}")
        print(f"  Amplifying ({len(amplifying)}): {', '.join(sorted(amplifying))}")

    composition = analyze_composition(base / "omega_composition")
    if composition:
        print("\n--- Composition Results ---")
        if "singles" in composition:
            print("  Singles:")
            for cn, v in sorted(composition["singles"].items(),
                               key=lambda x: x[1]["mean"], reverse=True):
                status = "AMP" if v["max"] > 0.01 else "pres"
                print(f"    {cn:25s} mean={v['mean']:.6f} max={v['max']:.6f} {status}")
        if "chains" in composition:
            print(f"  Top 10 chains by mean TV:")
            for cn, v in sorted(composition["chains"].items(),
                               key=lambda x: x[1]["mean"], reverse=True)[:10]:
                status = "AMP" if v["max"] > 0.01 else "pres"
                print(f"    {cn:35s} mean={v['mean']:.6f} max={v['max']:.6f} {status}")

    depth = analyze_depth(base / "omega_depth")
    if depth:
        print("\n--- Depth Sensitivity ---")
        for d_key in ["depth2", "depth3", "depth4"]:
            if d_key not in depth:
                continue
            conts = depth[d_key]
            print(f"  {d_key}:")
            for cn, v in sorted(conts.items(), key=lambda x: x[1].get("max", 0),
                               reverse=True):
                if v.get("n", 0) == 0:
                    continue
                status = "AMP" if v["max"] > 0.01 else "pres"
                print(f"    {cn:25s} mean={v['mean']:.6f} max={v['max']:.6f} {status}")

    values = analyze_values(base / "omega_values")
    if values:
        print("\n--- Value Sensitivity ---")
        for val_key in sorted(values.keys(), key=lambda x: int(x)):
            conts = values[val_key]
            heavy = conts.get("heavy", {})
            neutral = conts.get("neutral", {})
            h_mean = f"{heavy['mean']:.6f}" if heavy.get("n", 0) > 0 else "n/a"
            n_mean = f"{neutral['mean']:.6f}" if neutral.get("n", 0) > 0 else "n/a"
            print(f"  val={val_key:>3s}: neutral_mean={n_mean}, heavy_mean={h_mean}")

    queries = analyze_queries(base / "omega_queries")
    if queries:
        print("\n--- Query Format Sensitivity ---")
        for qname, conts in sorted(queries.items()):
            heavy = conts.get("heavy", {})
            neutral = conts.get("neutral", {})
            h_mean = f"{heavy['mean']:.6f}" if heavy.get("mean") is not None else "n/a"
            n_mean = f"{neutral['mean']:.6f}" if neutral.get("mean") is not None else "n/a"
            n_elig = conts.get("none", {}).get("n", 0)
            print(f"  {qname:20s}: neutral={n_mean}, heavy={h_mean}, n={n_elig}")

    print("\n--- Prospective Prediction Test ---")
    pred_result = prospective_test(base / "omega_sweep_fast")
    if pred_result:
        if "error" in pred_result:
            print(f"  Error: {pred_result['error']}")
        else:
            print(f"  Cal pairs: {pred_result['cal_pairs']}")
            print(f"  Eval pairs: {pred_result['eval_pairs']}")
            print(f"  Continuations evaluated: {pred_result['n_continuations']}")
            print(f"  Omega accuracy: {pred_result['omega_correct']}/{pred_result['n_continuations']} "
                  f"= {pred_result['omega_accuracy']:.1%}")
            print(f"  Baseline accuracy: {pred_result['baseline_correct']}/{pred_result['n_continuations']} "
                  f"= {pred_result['baseline_accuracy']:.1%}")
            if pred_result["disagreements"]:
                print(f"  Disagreements ({len(pred_result['disagreements'])}):")
                for d in pred_result["disagreements"]:
                    marker = "omega_right" if d["omega_correct"] else "baseline_right"
                    print(f"    {d['continuation']:25s} omega={d['omega_pred']:8s} "
                          f"baseline={d['baseline_pred']:8s} actual={d['actual']:8s} "
                          f"[{marker}]")

        with open(base / "omega_sweep_fast" / "prospective_test.json", "w") as f:
            json.dump(pred_result, f, indent=2)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    datasets = {
        "sweep_fast": base / "omega_sweep_fast" / "observations.json",
        "controls": base / "omega_controls" / "observations.json",
        "composition": base / "omega_composition" / "observations.json",
        "depth": base / "omega_depth" / "observations.json",
        "values": base / "omega_values" / "observations.json",
        "queries": base / "omega_queries" / "observations.json",
    }

    total_obs = 0
    for name, path in datasets.items():
        if path.exists():
            with open(path) as f:
                obs = json.load(f)
            print(f"  {name:20s}: {len(obs):6d} observations")
            total_obs += len(obs)
        else:
            print(f"  {name:20s}: not yet available")
    print(f"  {'TOTAL':20s}: {total_obs:6d} observations")


if __name__ == "__main__":
    main()
