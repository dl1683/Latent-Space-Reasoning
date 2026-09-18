"""Omega Pair Property Analysis: What measurable property of a surface pair
predicts whether it falls in the additive (strong) or suppressive (tight) regime?

12 pairs spanning known-strong to known-tight, each measured under 3 continuations.
~216 readouts. Target: <10 min CPU.
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

CELLS = [
    (4, "x", 1), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 6), (4, "y", 8),
]

SURFACE_PAIRS = [
    # Known strong (value/semantic changes in second line)
    ("ok = {var}\nresult = \"good\"\n", "ok = {var}\nresult = \"bad\"\n", "strong_assert", "value_change"),
    ("val = {var}\nstatus = \"valid\"\n", "val = {var}\nstatus = \"invalid\"\n", "strong_assert_2", "value_change"),
    ("ok = {var}\nassert ok == True\n", "ok = {var}\nassert ok == False\n", "assert_pair", "value_change"),

    # Known tight (minimal variable name or value differences)
    ("# OK.\n", "# Good.\n", "minimal_ok_good", "name_change"),
    ("{var} = 1\n", "{var} = 2\n", "value_pair", "value_change"),

    # Probing the spectrum
    ("result = {var}\nprint(result)\n", "result = {var}\nreturn result\n", "action_pair", "structure_change"),
    ("data = {var}\ndata.append(1)\n", "data = {var}\ndata.pop()\n", "mutation_pair", "structure_change"),
    ("flag = {var}\nif flag:\n", "flag = {var}\nif not flag:\n", "negation_pair", "structure_change"),
    ("count = {var}\ncount += 1\n", "count = {var}\ncount -= 1\n", "arithmetic_pair", "value_change"),
    ("name = {var}\n", "names = {var}\n", "plural_pair", "name_change"),
    ("{var} = 1\nz = 2\n", "{var} = 2\nz = 1\n", "swap_pair", "value_change"),
    ("temp = {var}\nresult = temp\n", "temp = {var}\nresult = None\n", "alias_pair", "value_change"),
]

CONTINUATIONS = {
    "identity": "{var} = {var}\n",
    "neutral": "pass\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
}


def tv(a, b):
    return float(0.5 * np.abs(a - b).sum())


def count_token_diff(tok, text_a, text_b):
    ids_a = tok.encode(text_a, add_special_tokens=False)
    ids_b = tok.encode(text_b, add_special_tokens=False)
    max_len = max(len(ids_a), len(ids_b))
    min_len = min(len(ids_a), len(ids_b))
    diff = max_len - min_len
    for i in range(min_len):
        if ids_a[i] != ids_b[i]:
            diff += 1
    return diff, abs(len(ids_a) - len(ids_b)), len(ids_a), len(ids_b)


def run_pair_property():
    print("=== Omega Pair Property Analysis ===")
    result_dir = Path("experiments/results/omega_pair_property")
    result_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "model_id": "Qwen/Qwen3-1.7B",
        "query_template": QUERY,
        "templates": {"depth4_single": TEMPLATE},
    }
    adapter = ModelAdapter(cfg)

    observations = []
    t0 = time.time()
    n = 0

    for depth, var, val in CELLS:
        prefix_text = build_prefix(TEMPLATE, var=var, outer_val=val)
        query = QUERY.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_a_tmpl, surf_b_tmpl, pair_name, category in SURFACE_PAIRS:
            surf_a = surf_a_tmpl.replace("{var}", var)
            surf_b = surf_b_tmpl.replace("{var}", var)

            # Baseline (no continuation)
            for side, surf in [("a", surf_a), ("b", surf_b)]:
                full_suffix = surf + query
                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                n += 1
                observations.append({
                    "depth": depth, "var": var, "val": val,
                    "pair": pair_name, "category": category,
                    "side": side, "continuation": "baseline",
                    "dist": dist.tolist(),
                })

            # Continuations
            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                for side, surf in [("a", surf_a), ("b", surf_b)]:
                    full_suffix = surf + cont_text + query
                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    n += 1
                    observations.append({
                        "depth": depth, "var": var, "val": val,
                        "pair": pair_name, "category": category,
                        "side": side, "continuation": cont_name,
                        "dist": dist.tolist(),
                    })

    elapsed = time.time() - t0
    print(f"Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    summary = analyze_pair_property(observations, adapter.tok)

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {len(observations)} observations to {result_dir}/")
    return observations


def analyze_pair_property(all_obs, tokenizer):
    print("\n=== Pair Property Analysis ===")

    obs_idx = {}
    for o in all_obs:
        key = (o["var"], o["val"], o["pair"], o["side"], o["continuation"])
        obs_idx[key] = np.array(o["dist"], dtype=np.float64)

    pair_results = []

    for surf_a_tmpl, surf_b_tmpl, pair_name, category in SURFACE_PAIRS:
        # Measure token properties using first var
        surf_a_sample = surf_a_tmpl.replace("{var}", "x")
        surf_b_sample = surf_b_tmpl.replace("{var}", "x")
        token_diff, len_diff, len_a, len_b = count_token_diff(
            tokenizer, surf_a_sample, surf_b_sample)

        baseline_tvs = []
        for _, var, val in CELLS:
            ka = (var, val, pair_name, "a", "baseline")
            kb = (var, val, pair_name, "b", "baseline")
            if ka in obs_idx and kb in obs_idx:
                baseline_tvs.append(tv(obs_idx[ka], obs_idx[kb]))

        if not baseline_tvs:
            continue

        baseline_max = float(np.max(baseline_tvs))
        baseline_mean = float(np.mean(baseline_tvs))
        baseline_median = float(np.median(baseline_tvs))
        n_eligible_001 = sum(1 for t in baseline_tvs if t <= 0.01)
        n_eligible_005 = sum(1 for t in baseline_tvs if t <= 0.05)

        cont_omegas = {}
        for cont_name in CONTINUATIONS:
            post_tvs = []
            eligible_post = []
            for _, var, val in CELLS:
                ka_base = (var, val, pair_name, "a", "baseline")
                kb_base = (var, val, pair_name, "b", "baseline")
                if ka_base not in obs_idx or kb_base not in obs_idx:
                    continue
                d_pre = tv(obs_idx[ka_base], obs_idx[kb_base])

                ka = (var, val, pair_name, "a", cont_name)
                kb = (var, val, pair_name, "b", cont_name)
                if ka in obs_idx and kb in obs_idx:
                    d_post = tv(obs_idx[ka], obs_idx[kb])
                    post_tvs.append(d_post)
                    if d_pre <= 0.05:
                        eligible_post.append(d_post)

            omega_tau005 = float(np.max(eligible_post)) if eligible_post else None
            omega_mean = float(np.mean(eligible_post)) if eligible_post else None
            cont_omegas[cont_name] = {
                "omega_max_tau005": omega_tau005,
                "omega_mean_tau005": omega_mean,
                "n_eligible": len(eligible_post),
                "all_post_tvs": [round(t, 6) for t in post_tvs],
            }

        # Determine regime
        identity_omega = cont_omegas["identity"]["omega_max_tau005"]
        neutral_omega = cont_omegas["neutral"]["omega_max_tau005"]
        heavy_omega = cont_omegas["heavy"]["omega_max_tau005"]

        if identity_omega is not None and neutral_omega is not None:
            identity_ratio = identity_omega / max(neutral_omega, 1e-9)
        else:
            identity_ratio = None

        result = {
            "pair_name": pair_name,
            "category": category,
            "token_diff": token_diff,
            "len_diff": len_diff,
            "len_a": len_a,
            "len_b": len_b,
            "baseline_max": round(baseline_max, 6),
            "baseline_mean": round(baseline_mean, 6),
            "baseline_median": round(baseline_median, 6),
            "n_eligible_tau001": n_eligible_001,
            "n_eligible_tau005": n_eligible_005,
            "identity_omega": round(identity_omega, 6) if identity_omega is not None else None,
            "neutral_omega": round(neutral_omega, 6) if neutral_omega is not None else None,
            "heavy_omega": round(heavy_omega, 6) if heavy_omega is not None else None,
            "identity_to_neutral_ratio": round(identity_ratio, 3) if identity_ratio is not None else None,
            "continuations": cont_omegas,
        }
        pair_results.append(result)

        status = ""
        if identity_omega is not None and neutral_omega is not None:
            if identity_omega > neutral_omega:
                status = "AMPLIFY"
            else:
                status = "PRESERVE"

        print(f"\n{pair_name} [{category}]:")
        print(f"  Token diff: {token_diff}, Len diff: {len_diff}")
        print(f"  Baseline TV: max={baseline_max:.4f}, mean={baseline_mean:.4f}")
        print(f"  Eligible: tau=0.01: {n_eligible_001}, tau=0.05: {n_eligible_005}")
        if identity_omega is not None:
            print(f"  Omega(identity)={identity_omega:.4f}, Omega(neutral)={neutral_omega:.4f}, "
                  f"Omega(heavy)={heavy_omega:.4f}")
            print(f"  Identity/neutral ratio: {identity_ratio:.2f}x  -> {status}")

    # Sort by baseline TV for the ranking table
    pair_results.sort(key=lambda r: r["baseline_mean"])

    print("\n\n=== RANKED BY BASELINE TV (ascending) ===")
    print(f"{'Pair':<20s} {'Category':<18s} {'BasTV':>8s} {'Elig05':>6s} {'IdentW':>8s} "
          f"{'NeutW':>8s} {'HeavyW':>8s} {'I/N':>6s} {'TokDf':>5s}")
    print("-" * 100)
    for r in pair_results:
        io = f"{r['identity_omega']:.4f}" if r['identity_omega'] is not None else "N/A"
        no = f"{r['neutral_omega']:.4f}" if r['neutral_omega'] is not None else "N/A"
        ho = f"{r['heavy_omega']:.4f}" if r['heavy_omega'] is not None else "N/A"
        ratio = f"{r['identity_to_neutral_ratio']:.1f}x" if r['identity_to_neutral_ratio'] is not None else "N/A"
        print(f"{r['pair_name']:<20s} {r['category']:<18s} {r['baseline_mean']:8.4f} "
              f"{r['n_eligible_tau005']:>6d} {io:>8s} {no:>8s} {ho:>8s} {ratio:>6s} {r['token_diff']:>5d}")

    summary = {
        "n_readouts": sum(len(r["continuations"]["identity"]["all_post_tvs"]) * 8
                         for r in pair_results),
        "n_pairs": len(pair_results),
        "pairs_ranked_by_baseline_tv": pair_results,
    }
    return summary


if __name__ == "__main__":
    run_pair_property()
