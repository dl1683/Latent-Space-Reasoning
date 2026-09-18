"""Omega Temperature Sensitivity (Experiment #51).

All prior experiments use greedy decoding (raw softmax = T=1.0).
Does omega survive temperature scaling? If omega vanishes at high T,
it's a softmax sharpness artifact. If it persists or grows nonlinearly,
it reveals geometry in logit space.

Tests T = 0.25, 0.5, 1.0, 2.0, 5.0 on the standard setup.
~450 readouts (5 temps x 5 continuations x 6 cells x 3 surface variants).
"""

import json
import sys
import time
import copy
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv


TEMPERATURES = [0.25, 0.5, 1.0, 2.0, 5.0]

CFG = {
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
}

EQUIV_PAIRS = [
    ("assert_novar_2", "assert_var_2", "strong_assert_2"),
    ("minimal_ok", "assert_novar_2", "ok_vs_novar"),
]

CONTINUATIONS = {
    "identity_bare": "{var} = {var}\n",
    "heavy_bare": "{var} = {var} + 1\n",
    "neutral_bare": "pass\n",
    "safety_donot": "# SAFETY: do not modify\n{var} = {var}\n",
    "depr_deprecated": "# DEPRECATED: will be removed\n{var} = {var}\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]


def extract_11bin_at_temp(logits, digit_ids, temperature):
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=0).numpy().astype(np.float64)
    bins = np.zeros(11, dtype=np.float64)
    for d in range(10):
        bins[d] = probs[digit_ids[d]]
    bins[10] = 1.0 - bins[:10].sum()
    return bins


def run_temperature():
    print("=== Omega Temperature Sensitivity (Exp #51) ===\n")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(
        CFG["model_id"], trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        CFG["model_id"], trust_remote_code=True, torch_dtype=torch.float32)
    mdl.eval()

    digit_ids = {}
    for d in range(10):
        toks = tok.encode(str(d), add_special_tokens=False)
        assert len(toks) == 1
        digit_ids[d] = toks[0]

    observations = []
    t0 = time.time()
    n = 0
    query_tmpl = CFG["query_template"]

    for depth, var, val in CELLS:
        tmpl = CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        ids_prefix = tok.encode(prefix_text, add_special_tokens=False,
                                return_tensors="pt")
        with torch.no_grad():
            out_prefix = mdl(ids_prefix, use_cache=True)
        prefix_cache = copy.deepcopy(out_prefix.past_key_values)

        for surf_name, surf_tmpl in SURFACES.items():
            surface_text = surf_tmpl.replace("{var}", var)

            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                full_suffix = surface_text + cont_text + query
                ids_suffix = tok.encode(full_suffix, add_special_tokens=False,
                                        return_tensors="pt")
                cache_copy = copy.deepcopy(prefix_cache)
                with torch.no_grad():
                    out = mdl(ids_suffix, past_key_values=cache_copy,
                              use_cache=True)
                raw_logits = out.logits[0, -1, :]

                for temp in TEMPERATURES:
                    dist = extract_11bin_at_temp(raw_logits, digit_ids, temp)
                    n += 1
                    observations.append({
                        "depth": depth,
                        "var": var,
                        "val": val,
                        "surface": surf_name,
                        "continuation": cont_name,
                        "temperature": temp,
                        "dist": dist.tolist(),
                    })

                if n % 100 == 0:
                    print(f"  {n} readouts, {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n} readouts in {elapsed:.1f}s")

    result_dir = Path("experiments/results/omega_temperature")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    # Analysis: omega at each temperature
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"], obs["temperature"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    summary = {"n_readouts": n, "elapsed_s": elapsed, "temps": {}}

    print("\n=== Omega by Temperature ===")
    for temp in TEMPERATURES:
        print(f"\n  T={temp}:")
        temp_results = {}
        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            for tau in [0.01, 0.05]:
                eligible = set()
                for depth, var, val in CELLS:
                    ka = (depth, var, val, surf_a, "identity_bare", temp)
                    kb = (depth, var, val, surf_b, "identity_bare", temp)
                    if ka in obs_idx and kb in obs_idx:
                        d = tv(obs_idx[ka], obs_idx[kb])
                        if d <= tau:
                            eligible.add((depth, var, val))

                if not eligible:
                    continue

                pair_omega = {}
                for cont in CONTINUATIONS:
                    tvs = []
                    for cell in eligible:
                        d, v, vl = cell
                        ka = (d, v, vl, surf_a, cont, temp)
                        kb = (d, v, vl, surf_b, cont, temp)
                        if ka in obs_idx and kb in obs_idx:
                            tvs.append(tv(obs_idx[ka], obs_idx[kb]))
                    if tvs:
                        pair_omega[cont] = {
                            "omega": float(np.max(tvs)),
                            "mean": float(np.mean(tvs)),
                        }

                key = f"{pair_name}_tau{tau}"
                temp_results[key] = pair_omega
                print(f"    {key} ({len(eligible)} elig):")
                for cn, cd in sorted(pair_omega.items(),
                                     key=lambda x: x[1]["omega"],
                                     reverse=True):
                    print(f"      {cn:22s}: omega={cd['omega']:.5f}")

        summary["temps"][str(temp)] = temp_results

    # Cross-temperature comparison for key continuations
    print("\n=== Temperature Scaling of heavy_bare Omega ===")
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for tau in [0.05]:
            print(f"\n  {pair_name} tau={tau}:")
            for cont in ["identity_bare", "heavy_bare", "neutral_bare",
                         "safety_donot"]:
                omegas_by_temp = []
                for temp in TEMPERATURES:
                    eligible = set()
                    for depth, var, val in CELLS:
                        ka = (depth, var, val, surf_a, "identity_bare", temp)
                        kb = (depth, var, val, surf_b, "identity_bare", temp)
                        if ka in obs_idx and kb in obs_idx:
                            d = tv(obs_idx[ka], obs_idx[kb])
                            if d <= tau:
                                eligible.add((depth, var, val))
                    tvs = []
                    for cell in eligible:
                        d, v, vl = cell
                        ka = (d, v, vl, surf_a, cont, temp)
                        kb = (d, v, vl, surf_b, cont, temp)
                        if ka in obs_idx and kb in obs_idx:
                            tvs.append(tv(obs_idx[ka], obs_idx[kb]))
                    omega = float(np.max(tvs)) if tvs else 0.0
                    omegas_by_temp.append(omega)
                print(f"    {cont:22s}: " +
                      " | ".join(f"T={t}:{o:.4f}"
                                 for t, o in zip(TEMPERATURES, omegas_by_temp)))

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir}/")
    return summary


if __name__ == "__main__":
    run_temperature()
