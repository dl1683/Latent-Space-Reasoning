"""Omega Temperature Bifurcation (Experiment #61).

Does the grow/collapse bimodality from exp #56 survive under temperature>0?
Greedy decoding (T=0) is deterministic — maybe the bimodality is a brittle
artifact of the argmax path. Under sampling, do the basins of attraction
persist or blur?

Tests T ∈ {0, 0.3, 0.7, 1.0} with 3 seeds each for T>0.
Uses a subset of conditions from #56 for speed.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv

import torch
import copy


MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "query_template": "f()\nprint({var})  # Output: ",
    "templates": {
        "depth4_single": (
            "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
            "    def g():\n        {var} = 999\n        def h():\n"
            "            {var} = 9999\n            def k():\n"
            "                {var} = 99999\n                return {var}\n"
            "            k()\n            return {var}\n"
            "        h()\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
    },
}

SURFACES = {
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "minimal_ok": "# OK.\n",
    "minimal_good": "# Good.\n",
}

EQUIV_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert_2"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CONTINUATIONS = {
    "empty": "",
    "identity": "{var} = {var}\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
]

TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
SEEDS_PER_TEMP = 3
N_GENERATE = 20
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def generate_trajectory_temp(adapter, prefix_state, suffix_text, n_tokens,
                              temperature=0.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    ids = adapter.tok.encode(suffix_text, add_special_tokens=False,
                              return_tensors="pt")
    st = copy.deepcopy(prefix_state)

    with torch.no_grad():
        out = adapter._forward_with_cache(ids, st)
    st = adapter._get_cache(out)

    trajectory = []
    logits = out.logits[0, -1, :]
    dist = adapter._extract_11bin(logits)
    trajectory.append(dist)

    for step in range(1, n_tokens + 1):
        if temperature == 0.0:
            next_token_id = int(torch.argmax(logits))
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token_id = int(torch.multinomial(probs, 1).item())

        next_ids = torch.tensor([[next_token_id]])
        with torch.no_grad():
            out = adapter._forward_with_cache(next_ids, st)
        st = adapter._get_cache(out)
        logits = out.logits[0, -1, :]
        dist = adapter._extract_11bin(logits)
        trajectory.append(dist)

    adapter.call_count += 1
    return trajectory


def run_temp_bifurcation():
    print("=== Temperature Bifurcation ===")
    n_cond = (len(CELLS) * len(EQUIV_PAIRS) * len(CONTINUATIONS) *
              len(TEMPERATURES) * SEEDS_PER_TEMP)
    print(f"Target: ~{n_cond} trajectory pairs")

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    t0 = time.time()
    records = []
    n_done = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                suffix_a = SURFACES[surf_a].replace("{var}", var) + cont_text + query
                suffix_b = SURFACES[surf_b].replace("{var}", var) + cont_text + query

                for temp in TEMPERATURES:
                    n_seeds = 1 if temp == 0.0 else SEEDS_PER_TEMP
                    for seed_idx in range(n_seeds):
                        seed = seed_idx * 1000 + int(val) if temp > 0 else None

                        traj_a = generate_trajectory_temp(
                            adapter, prefix_state, suffix_a, N_GENERATE,
                            temperature=temp, seed=seed)
                        traj_b = generate_trajectory_temp(
                            adapter, prefix_state, suffix_b, N_GENERATE,
                            temperature=temp, seed=(seed + 500 if seed else None))

                        step_tvs = [float(tv(traj_a[s], traj_b[s]))
                                    for s in range(N_GENERATE + 1)]

                        final_tv = step_tvs[-1]
                        fate = "grow" if final_tv > 0.5 else "collapse"

                        records.append({
                            "cell": f"{var}_{val}",
                            "pair": pair_name,
                            "continuation": cont_name,
                            "temperature": temp,
                            "seed": seed,
                            "initial_tv": step_tvs[0],
                            "final_tv": final_tv,
                            "max_tv": max(step_tvs),
                            "fate": fate,
                            "step_tvs": step_tvs,
                        })
                        n_done += 1

                        if n_done % 10 == 0:
                            elapsed = time.time() - t0
                            print(f"  {n_done} pairs, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n_done} pairs in {elapsed:.1f}s")

    # Analysis: grow rate by temperature
    by_temp = defaultdict(list)
    for r in records:
        by_temp[r["temperature"]].append(r)

    temp_summary = {}
    for temp, items in sorted(by_temp.items()):
        n_grow = sum(1 for i in items if i["fate"] == "grow")
        n_total = len(items)
        mean_final = float(np.mean([i["final_tv"] for i in items]))
        temp_summary[str(temp)] = {
            "n": n_total,
            "n_grow": n_grow,
            "grow_rate": round(n_grow / n_total, 3),
            "mean_final_tv": round(mean_final, 4),
        }

    # By temperature × continuation
    by_temp_cont = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_temp_cont[r["temperature"]][r["continuation"]].append(r)

    temp_cont_summary = {}
    for temp in TEMPERATURES:
        for cont in CONTINUATIONS:
            items = by_temp_cont[temp][cont]
            if not items:
                continue
            n_grow = sum(1 for i in items if i["fate"] == "grow")
            key = f"T={temp}/{cont}"
            temp_cont_summary[key] = {
                "n": len(items),
                "grow_rate": round(n_grow / len(items), 3),
                "mean_final_tv": round(float(np.mean([i["final_tv"] for i in items])), 4),
            }

    # Seed consistency at each temperature (do seeds agree on fate?)
    consistency = defaultdict(list)
    for r in records:
        if r["temperature"] > 0:
            key = (r["cell"], r["pair"], r["continuation"], r["temperature"])
            consistency[key].append(r["fate"])

    consistency_rates = []
    for key, fates in consistency.items():
        if len(fates) > 1:
            most_common = max(set(fates), key=fates.count)
            agree = sum(1 for f in fates if f == most_common) / len(fates)
            consistency_rates.append(agree)

    mean_consistency = float(np.mean(consistency_rates)) if consistency_rates else 1.0

    summary = {
        "n_pairs": len(records),
        "temperatures": TEMPERATURES,
        "seeds_per_temp": SEEDS_PER_TEMP,
        "n_generate": N_GENERATE,
        "temp_summary": temp_summary,
        "temp_cont_summary": temp_cont_summary,
        "seed_consistency": round(mean_consistency, 3),
    }

    result_dir = Path("experiments/results/omega_temperature_bifurcation")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print(f"\n=== TEMPERATURE BIFURCATION RESULTS ===")
    print(f"\nGrow rate by temperature:")
    print(f"{'Temp':>6s} {'N':>4s} {'Grow':>5s} {'Rate':>6s} {'MeanFinal':>10s}")
    for temp, ts in sorted(temp_summary.items(), key=lambda x: float(x[0])):
        print(f"{temp:>6s} {ts['n']:4d} {ts['n_grow']:5d} "
              f"{ts['grow_rate']:6.1%} {ts['mean_final_tv']:10.4f}")

    print(f"\nSeed consistency (T>0): {mean_consistency:.1%}")
    if mean_consistency > 0.8:
        print("=> Bifurcation fate is ROBUST to sampling noise")
    elif mean_consistency > 0.6:
        print("=> Bifurcation is partially stochastic")
    else:
        print("=> Fate is dominated by sampling randomness")

    print(f"\nBy temperature × continuation:")
    for key, s in sorted(temp_cont_summary.items()):
        print(f"  {key:25s}: grow={s['grow_rate']:.0%} (n={s['n']})")


if __name__ == "__main__":
    run_temp_bifurcation()
