"""Omega Architecture Replication: Does the effect exist in non-Qwen models?

Tests two non-Qwen architectures (all already cached):
- EleutherAI/pythia-1.4b (GPT-NeoX, 1.4B)
- stabilityai/stablelm-2-1_6b (StableLM, 1.6B)

Both are ~1.5B params (comparable to Qwen3-1.7B reference).
If even ONE shows the omega effect, the claim is architecture-general.

6 cells x 2 pairs x 2 sides x 6 continuations = 144 readouts per model
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
    ("# OK.\n", "# Good.\n", "minimal_ok_good"),
]

CONTINUATIONS = {
    "none": "",
    "neutral": "# State unchanged.\n",
    "heavy": "# Reassigning {var} now.\n",
    "identity": "{var} = {var}\n",
    "type_check": "type({var})\n",
    "pass": "pass\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 6), (4, "y", 8),
]

MODELS = [
    ("EleutherAI/pythia-1.4b", "pythia_1.4b"),
    ("stabilityai/stablelm-2-1_6b", "stablelm_1.6b"),
]


def run_one_model(model_id, model_label):
    print(f"\n--- {model_label} ({model_id}) ---")
    cfg = {
        "model_id": model_id,
        "query_template": QUERY,
        "templates": {"depth4_single": TEMPLATE},
    }

    n_total = len(CELLS) * len(SURFACE_PAIRS) * 2 * len(CONTINUATIONS)
    print(f"Readouts: {n_total}")

    try:
        adapter = ModelAdapter(cfg)
    except AssertionError as e:
        print(f"  SKIP {model_label}: digit tokenization issue: {e}")
        return []

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

    elapsed = time.time() - t0
    print(f"  Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")
    del adapter
    return observations


def analyze_architecture(all_obs):
    print("\n=== Architecture Replication Summary ===")

    models = sorted(set(o["model"] for o in all_obs))

    for tau in [0.01, 0.02, 0.05]:
        print(f"\n--- tau={tau} ---")
        print(f"{'Model':<18s} {'Pair':<20s} {'Cont':<12s} "
              f"{'mean':>8s} {'max':>8s} {'n':>3s} {'status':>8s}")

        for model_label in models:
            obs = [o for o in all_obs if o["model"] == model_label]
            obs_idx = {}
            for o in obs:
                key = (o["var"], o["val"], o["pair"], o["side"], o["continuation"])
                obs_idx[key] = np.array(o["dist"], dtype=np.float64)

            for pair_name in [p[2] for p in SURFACE_PAIRS]:
                for cont_name in sorted(CONTINUATIONS.keys()):
                    if cont_name == "none":
                        continue
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
                        print(f"  {model_label:<16s} {pair_name:<20s} {cont_name:<12s} "
                              f"{mean_tv:8.5f} {max_tv:8.5f} {len(tvs):3d} "
                              f"{status:>8s}")


def run_architecture():
    print("=== Omega Architecture Replication ===")
    result_dir = Path("experiments/results/omega_architecture")
    result_dir.mkdir(parents=True, exist_ok=True)

    all_obs = []
    for model_id, model_label in MODELS:
        try:
            obs = run_one_model(model_id, model_label)
            all_obs.extend(obs)
            if obs:
                with open(result_dir / f"observations_{model_label}.json", "w") as f:
                    json.dump(obs, f)
        except Exception as e:
            print(f"  ERROR on {model_label}: {e}")
            import traceback
            traceback.print_exc()

    if all_obs:
        with open(result_dir / "observations_all.json", "w") as f:
            json.dump(all_obs, f)
        analyze_architecture(all_obs)

    print(f"\nSaved {len(all_obs)} observations to {result_dir}/")
    return all_obs


if __name__ == "__main__":
    run_architecture()
