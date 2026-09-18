"""Cross-Model Replication: Does omega generalize to a different model?

Tests the omega pattern on Qwen3-0.6B-Base (different scale, same family)
and/or GPT-2 (different architecture entirely).

If the same continuations preserve/amplify on both models, omega measures
a structural property of code semantics, not a model-specific artifact.

Core test: 6 cells x 3 pairs x 2 sides x 8 continuations = 288 readouts per model
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix


TEMPLATE = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            def k():\n"
    "                {var} = 99999\n                return {var}\n"
    "            k()\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

QUERY = "f()\nprint({var})  # Output: "

SURFACE_PAIRS = [
    ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ("# ok\n", "# okay\n", "terse_ok_okay"),
    ("# OK.\n", "# Good.\n", "minimal_ok_good"),
]

CONTINUATIONS = {
    "none": "",
    "neutral": "# State unchanged.\n",
    "heavy": "# Reassigning {var} now.\n",
    "pass": "pass\n",
    "identity": "{var} = {var}\n",
    "type_check": "type({var})\n",
    "observe": "_ = {var}\n",
    "bare_nl": "\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 6), (4, "y", 8),
]

MODELS = [
    ("Qwen/Qwen3-0.6B-Base", "qwen3_0.6b"),
    ("openai-community/gpt2", "gpt2"),
]

HEAVY_KW = ["heavy", "reassign", "updat", "chang"]


def run_one_model(model_id, model_label):
    print(f"\n=== Running {model_label} ({model_id}) ===")
    cfg = {
        "model_id": model_id,
        "query_template": QUERY,
        "templates": {"depth4_single": TEMPLATE},
    }

    n_total = len(CELLS) * len(SURFACE_PAIRS) * 2 * len(CONTINUATIONS)
    print(f"Total readouts: {n_total}")

    adapter = ModelAdapter(cfg)
    observations = []
    t0 = time.time()
    n = 0

    for depth, var, val in CELLS:
        prefix_text = build_prefix(TEMPLATE, var=var, outer_val=val)
        query = QUERY.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_a_tmpl, surf_b_tmpl, pair_name in SURFACE_PAIRS:
            for side, surf_tmpl in [("a", surf_a_tmpl), ("b", surf_b_tmpl)]:
                surface_text = surf_tmpl.replace("{var}", var)

                for cont_name, cont_tmpl in CONTINUATIONS.items():
                    cont_text = cont_tmpl.replace("{var}", var)
                    full_suffix = surface_text + cont_text + query

                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    n += 1

                    observations.append({
                        "model": model_label,
                        "depth": depth, "var": var, "val": val,
                        "pair": pair_name, "side": side,
                        "continuation": cont_name,
                        "dist": dist.tolist(),
                    })

                    if n % 100 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n}/{n_total} readouts, {elapsed:.0f}s, "
                              f"{n/elapsed:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")
    del adapter
    return observations


def analyze_model(observations, model_label):
    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["pair"], obs["side"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    results = {}
    for tau in [0.01, 0.02, 0.05]:
        print(f"\n--- {model_label} Omega (tau={tau}) ---")
        print(f"{'Continuation':<20s} {'pair':<20s} {'mean':>8s} {'max':>8s} "
              f"{'n':>3s} {'status':>8s}")

        tau_results = {}
        for pair_name in [p[2] for p in SURFACE_PAIRS]:
            for cont_name in sorted(CONTINUATIONS.keys()):
                tvs = []
                for _, var, val in CELLS:
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
                    mean_tv = float(np.mean(tvs))
                    max_tv = float(np.max(tvs))
                    status = "PRESERVE" if max_tv <= tau else "AMPLIFY"
                    print(f"  {cont_name:<18s} {pair_name:<20s} "
                          f"{mean_tv:8.5f} {max_tv:8.5f} {len(tvs):3d} "
                          f"{status:>8s}")
                    tau_results.setdefault(pair_name, {})[cont_name] = {
                        "mean": mean_tv, "max": max_tv, "n": len(tvs),
                        "preserves": bool(max_tv <= tau),
                    }
        results[str(tau)] = tau_results

    omega_correct = 0
    baseline_correct = 0
    total = 0
    tau = 0.02
    for pair_name in [p[2] for p in SURFACE_PAIRS]:
        for cont_name in CONTINUATIONS:
            tvs = []
            for _, var, val in CELLS:
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

            if not tvs:
                continue

            actual = "preserve" if max(tvs) <= tau else "separate"
            omega_pred = "separate"
            cn_lower = cont_name.lower()
            is_heavy = any(k in cn_lower for k in HEAVY_KW)
            baseline_pred = "separate" if is_heavy else "preserve"

            total += 1
            baseline_correct += int(baseline_pred == actual)

    print(f"\n{model_label} baseline accuracy at tau={tau}: "
          f"{baseline_correct}/{total} ({100*baseline_correct/total:.0f}%)")

    return results


def run_crossmodel():
    print("=== Cross-Model Omega Replication ===")
    result_dir = Path("experiments/results/omega_crossmodel")
    result_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model_id, model_label in MODELS:
        try:
            obs = run_one_model(model_id, model_label)
            with open(result_dir / f"observations_{model_label}.json", "w") as f:
                json.dump(obs, f)
            results = analyze_model(obs, model_label)
            all_results[model_label] = results
        except Exception as e:
            print(f"  ERROR on {model_label}: {e}")
            all_results[model_label] = {"error": str(e)}

    with open(result_dir / "crossmodel_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to {result_dir}/")
    return all_results


if __name__ == "__main__":
    run_crossmodel()
