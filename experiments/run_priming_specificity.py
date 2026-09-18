"""Priming Specificity Test (R6 Experiment #3).

Does neutral prime identity specifically, or does it prime ALL continuations?
If neutral_then_X is super-linear for all X, the mechanism is "neutral opens
an attentional gate." If only for identity, it's variable-reference-specific.

4 chains x 6 cells x 6 surfaces x 2 directions = ~288 readouts. Plus singles.
"""

import json, sys, time, gc
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix

MODEL_CFG = {"model_id": "Qwen/Qwen3-1.7B-Base"}

TEMPLATE = (
    "{var} = {outer_val}\n"
    "def f():\n"
    "    {var} = 99\n"
    "    def g():\n"
    "        {var} = 999\n"
    "        def h():\n"
    "            {var} = 9999\n"
    "            def k():\n"
    "                {var} = 99999\n"
    "                return {var}\n"
    "            k()\n"
    "            return {var}\n"
    "        h()\n"
    "        return {var}\n"
    "    g()\n"
    "    return {var}\n"
)

SURFACES = {
    "assert_var": "# {var} stays the same.\n",
    "assert_novar": "# State unchanged.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
    "minimal_ok": "# OK.\n",
    "minimal_good": "# Good.\n",
}

SINGLES = {
    "empty": "",
    "neutral": "_ = 0\n",
    "identity": "{var} = {var}\n",
    "heavy": "{var} = {var} + 0\n",
    "bare": "{var}\n",
    "nonvar_42": "_ = 42\n",
}

CHAINS = {
    "neutral_then_identity": ("_ = 0\n", "{var} = {var}\n"),
    "neutral_then_heavy": ("_ = 0\n", "{var} = {var} + 0\n"),
    "neutral_then_bare": ("_ = 0\n", "{var}\n"),
    "neutral_then_nonvar": ("_ = 0\n", "_ = 42\n"),
    "neutral_then_neutral": ("_ = 0\n", "_ = 0\n"),
    "nonvar_then_identity": ("_ = 42\n", "{var} = {var}\n"),
}

CELLS = [
    ("x", 1), ("x", 5),
    ("y", 2), ("y", 6),
    ("z", 3), ("z", 7),
]

QUERY = "f()\nprint({var})  # Output: "
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def tv(a, b):
    return 0.5 * np.abs(np.array(a) - np.array(b)).sum()


def run():
    print("=== Priming Specificity Test ===", flush=True)
    adapter = ModelAdapter(MODEL_CFG)
    observations = []
    t0 = time.time()
    n = 0

    for var, val in CELLS:
        prefix = build_prefix(TEMPLATE, var=var, outer_val=val)
        query = QUERY.format(var=var)
        ps = adapter.get_state_after_prefix(prefix)

        for sn, st in SURFACES.items():
            surf = st.replace("{var}", var)

            # Singles
            for cn, ct in SINGLES.items():
                cont = ct.replace("{var}", var)
                dist = adapter.get_dist_from_state(ps, surf + cont + query)
                n += 1
                ai = int(np.argmax(dist))
                observations.append({
                    "var": var, "val": val,
                    "surface": sn,
                    "chain_type": "single",
                    "continuation": cn,
                    "dist": dist.tolist(),
                    "argmax_label": DIGIT_LABELS[ai],
                    "argmax_prob": float(dist[ai]),
                    "correct_digit": str(val),
                    "is_correct": DIGIT_LABELS[ai] == str(val),
                })

            # Chains
            for chain_name, (c1, c2) in CHAINS.items():
                cont1 = c1.replace("{var}", var)
                cont2 = c2.replace("{var}", var)
                dist = adapter.get_dist_from_state(
                    ps, surf + cont1 + cont2 + query)
                n += 1
                ai = int(np.argmax(dist))
                observations.append({
                    "var": var, "val": val,
                    "surface": sn,
                    "chain_type": "chain",
                    "continuation": chain_name,
                    "dist": dist.tolist(),
                    "argmax_label": DIGIT_LABELS[ai],
                    "argmax_prob": float(dist[ai]),
                    "correct_digit": str(val),
                    "is_correct": DIGIT_LABELS[ai] == str(val),
                })

        if n % 72 == 0:
            print(f"  {n} readouts, {time.time()-t0:.0f}s", flush=True)

        del ps
        gc.collect()

    elapsed = time.time() - t0
    print(f"Done: {n} readouts in {elapsed:.1f}s")

    # Analysis: mean TV per continuation type across surface pairs
    pairs = [
        ("assert_var", "assert_novar"),
        ("boundary_1", "boundary_2"),
        ("minimal_ok", "minimal_good"),
    ]
    all_conts = list(SINGLES.keys()) + list(CHAINS.keys())
    tv_results = {}

    for cn in all_conts:
        tvs = []
        for s1, s2 in pairs:
            for var, val in CELLS:
                r1 = [o for o in observations
                      if o["surface"] == s1 and o["continuation"] == cn
                      and o["var"] == var and o["val"] == val]
                r2 = [o for o in observations
                      if o["surface"] == s2 and o["continuation"] == cn
                      and o["var"] == var and o["val"] == val]
                if r1 and r2:
                    tvs.append(tv(r1[0]["dist"], r2[0]["dist"]))
        if tvs:
            tv_results[cn] = {
                "mean_tv": round(float(np.mean(tvs)), 4),
                "max_tv": round(float(np.max(tvs)), 4),
                "n": len(tvs),
            }

    # Super-linearity for each chain
    super_linearity = {}
    for chain_name, (c1_template, c2_template) in CHAINS.items():
        # Find single names for components
        c1_single = None
        c2_single = None
        for sn, st in SINGLES.items():
            if st == c1_template:
                c1_single = sn
            if st == c2_template:
                c2_single = sn

        if c1_single in tv_results and c2_single in tv_results and chain_name in tv_results:
            sum_parts = tv_results[c1_single]["mean_tv"] + tv_results[c2_single]["mean_tv"]
            chain_tv = tv_results[chain_name]["mean_tv"]
            super_linearity[chain_name] = {
                "components": f"{c1_single} + {c2_single}",
                "sum_of_parts": round(sum_parts, 4),
                "chain": round(chain_tv, 4),
                "ratio": round(chain_tv / max(sum_parts, 0.0001), 2),
                "is_super_linear": chain_tv > 1.2 * sum_parts,
            }

    summary = {
        "n_readouts": n,
        "tv_per_continuation": tv_results,
        "super_linearity": super_linearity,
        "elapsed_s": round(elapsed, 1),
    }

    # Verdict
    neutral_id = super_linearity.get("neutral_then_identity", {})
    neutral_heavy = super_linearity.get("neutral_then_heavy", {})
    neutral_bare = super_linearity.get("neutral_then_bare", {})
    neutral_nonvar = super_linearity.get("neutral_then_nonvar", {})

    n_super = sum(1 for v in [neutral_id, neutral_heavy, neutral_bare, neutral_nonvar]
                  if v.get("is_super_linear", False))

    if n_super >= 3:
        summary["verdict"] = (
            f"GENERIC_GATE: neutral primes {n_super}/4 continuations super-linearly "
            "— mechanism is generic attentional gating")
    elif n_super == 1 and neutral_id.get("is_super_linear", False):
        summary["verdict"] = (
            "VARIABLE_SPECIFIC: only neutral→identity is super-linear "
            "— mechanism is variable-reference-specific")
    elif n_super > 0:
        summary["verdict"] = (
            f"PARTIAL: neutral primes {n_super}/4 super-linearly — mixed mechanism")
    else:
        summary["verdict"] = "NO_SUPER_LINEARITY: no chain exceeds 1.2x sum of parts"

    out_dir = Path("experiments/results/omega_priming_specificity")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    print(f"\n=== PRIMING SPECIFICITY RESULTS ===")
    print(f"\n  TV per continuation:")
    for cn, data in sorted(tv_results.items(), key=lambda x: -x[1]["mean_tv"]):
        print(f"    {cn:25s}: TV={data['mean_tv']:.4f}")

    print(f"\n  Super-linearity:")
    for cn, sl in super_linearity.items():
        marker = " ***" if sl["is_super_linear"] else ""
        print(f"    {cn:25s}: chain={sl['chain']:.4f} "
              f"sum={sl['sum_of_parts']:.4f} "
              f"ratio={sl['ratio']:.1f}x{marker}")

    print(f"\n  VERDICT: {summary['verdict']}")
    del adapter
    gc.collect()
    return summary


if __name__ == "__main__":
    run()
