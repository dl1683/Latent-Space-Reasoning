"""Deep Composition Chains (Experiment E5).

Tests whether the priming effect compounds: if neutral_then_identity is 2.12x
superadditive, does neutral_neutral_identity amplify even more?

Key question: is priming (1) a one-shot effect (N>1 neutral doesn't help more)
or (2) compounding (more context = stronger amplification)?

Chain designs:
- identity alone (baseline)
- neutral + identity (known 2.12x superadditive)
- neutral + neutral + identity
- neutral + neutral + neutral + identity
- verified + identity
- verified + verified + identity
- pass + identity
- pass + pass + identity
- neutral + heavy (known 0.63x of heavy alone — suppressive!)
- neutral + neutral + heavy

Also test: 3-element mixed chains
- neutral + verified + identity
- pass + neutral + identity
- verified + neutral + identity

~200 readouts. Fast.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter
from omega import tv


MODEL_CFG = {"model_id": "Qwen/Qwen3-1.7B-Base"}

TEMPLATE = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            def k():\n"
    "                {var} = 99999\n                return {var}\n"
    "            k()\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

SURFACES = {
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
    "minimal_ok": "# OK.\n",
    "minimal_good": "# Good.\n",
}

EQUIV_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

ATOMS = {
    "identity": "{var} = {var}\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "neutral": "# Processing.\n",
    "verified": "# Verified.\n",
    "pass": "pass\n",
}

CHAINS = {
    "empty": [],
    "identity": ["identity"],
    "neutral_identity": ["neutral", "identity"],
    "neutral_neutral_identity": ["neutral", "neutral", "identity"],
    "neutral_x3_identity": ["neutral", "neutral", "neutral", "identity"],
    "verified_identity": ["verified", "identity"],
    "verified_verified_identity": ["verified", "verified", "identity"],
    "pass_identity": ["pass", "identity"],
    "pass_pass_identity": ["pass", "pass", "identity"],
    "heavy": ["heavy"],
    "neutral_heavy": ["neutral", "heavy"],
    "neutral_neutral_heavy": ["neutral", "neutral", "heavy"],
    "neutral_verified_identity": ["neutral", "verified", "identity"],
    "pass_neutral_identity": ["pass", "neutral", "identity"],
    "verified_neutral_identity": ["verified", "neutral", "identity"],
    "identity_neutral": ["identity", "neutral"],
    "identity_neutral_neutral": ["identity", "neutral", "neutral"],
}

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
TAU = 0.05
QUERY = "f()\nprint({var})  # Output: "


def run():
    print("=== Deep Composition Chains ===")
    adapter = ModelAdapter({"model_id": MODEL_CFG["model_id"]})

    observations = []
    t0 = time.time()
    total = len(CELLS) * len(SURFACES) * len(CHAINS)
    done = 0

    for depth, var, val in CELLS:
        prefix_code = TEMPLATE.format(var=var, outer_val=val)
        query = QUERY.format(var=var)
        prefix_state = adapter.get_state_after_prefix(prefix_code)

        for surf_name, surf_text in SURFACES.items():
            surface = surf_text.format(var=var)

            for chain_name, chain_atoms in CHAINS.items():
                continuation = ""
                for atom_name in chain_atoms:
                    continuation += ATOMS[atom_name].format(var=var)

                suffix = surface + continuation + query
                dist = adapter.get_dist_from_state(prefix_state, suffix)

                argmax_idx = int(np.argmax(dist))
                observations.append({
                    "depth": depth, "var": var, "val": val,
                    "surface": surf_name, "chain": chain_name,
                    "chain_atoms": chain_atoms,
                    "chain_length": len(chain_atoms),
                    "dist": dist.tolist(),
                    "argmax_label": DIGIT_LABELS[argmax_idx],
                    "argmax_prob": float(dist[argmax_idx]),
                    "correct_digit": str(val),
                    "is_correct": DIGIT_LABELS[argmax_idx] == str(val),
                    "p_correct": float(dist[val]),
                    "p_9": float(dist[9]),
                })
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {done}/{total} ({done/elapsed:.1f}/s)")

    print(f"\nDone: {len(observations)} readouts in {time.time()-t0:.0f}s")

    out_dir = Path("experiments/results/omega_deep_composition")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f, indent=2)

    summary = analyze(observations)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_dir}")
    return summary


def analyze(observations):
    idx = {}
    for r in observations:
        key = (r['var'], r['val'], r['surface'], r['chain'])
        idx[key] = r

    # Compute omega (mean TV between equiv pair surfaces) per chain
    chain_omega = {}
    for chain_name in CHAINS:
        tvs = []
        for s1, s2, pair_name in EQUIV_PAIRS:
            for depth, var, val in CELLS:
                base_k1 = (var, val, s1, "empty")
                base_k2 = (var, val, s2, "empty")
                if base_k1 not in idx or base_k2 not in idx:
                    continue
                baseline = tv(idx[base_k1]['dist'], idx[base_k2]['dist'])
                if baseline > TAU:
                    continue

                k1 = (var, val, s1, chain_name)
                k2 = (var, val, s2, chain_name)
                if k1 not in idx or k2 not in idx:
                    continue
                cell_tv = float(tv(idx[k1]['dist'], idx[k2]['dist']))
                tvs.append(cell_tv)

        if tvs:
            chain_omega[chain_name] = {
                "mean_tv": round(float(np.mean(tvs)), 6),
                "max_tv": round(float(np.max(tvs)), 6),
                "n_eligible": len(tvs),
            }

    # Attractor analysis (non-assert surfaces)
    chain_attractor = {}
    for chain_name in CHAINS:
        sub = [r for r in observations
               if r['chain'] == chain_name
               and r['surface'] not in ['assert_novar_2', 'assert_var_2']]
        if sub:
            n_9 = sum(1 for r in sub if r['argmax_label'] == '9')
            mean_p_9 = float(np.mean([r['p_9'] for r in sub]))
            n_correct = sum(1 for r in sub if r['is_correct'])
            chain_attractor[chain_name] = {
                "n": len(sub),
                "attractor_9_rate": round(n_9 / len(sub), 4),
                "mean_p_9": round(mean_p_9, 4),
                "correct_rate": round(n_correct / len(sub), 4),
            }

    print(f"\n{'Chain':35s} {'omega':>8s} {'attr_9%':>8s} {'correct%':>9s} {'p(9)':>7s}")
    for chain_name in CHAINS:
        o = chain_omega.get(chain_name, {})
        a = chain_attractor.get(chain_name, {})
        omega = o.get('mean_tv', 0)
        attr_rate = a.get('attractor_9_rate', 0)
        correct = a.get('correct_rate', 0)
        p9 = a.get('mean_p_9', 0)
        print(f"  {chain_name:33s} {omega:8.4f} {attr_rate:8.1%} {correct:9.1%} {p9:7.4f}")

    # Priming compounding test
    n1 = chain_omega.get('neutral_identity', {}).get('mean_tv', 0)
    n2 = chain_omega.get('neutral_neutral_identity', {}).get('mean_tv', 0)
    n3 = chain_omega.get('neutral_x3_identity', {}).get('mean_tv', 0)
    id_only = chain_omega.get('identity', {}).get('mean_tv', 0)

    print(f"\n=== PRIMING COMPOUNDING ===")
    print(f"  identity:                  {id_only:.4f}")
    print(f"  neutral + identity:        {n1:.4f} ({n1/id_only:.2f}x)" if id_only else "")
    print(f"  neutral×2 + identity:      {n2:.4f} ({n2/id_only:.2f}x)" if id_only else "")
    print(f"  neutral×3 + identity:      {n3:.4f} ({n3/id_only:.2f}x)" if id_only else "")

    if n1 > 0 and n2 > 0:
        if n2 > n1 * 1.1:
            priming_verdict = "COMPOUNDS: more neutral context amplifies further"
        elif n2 < n1 * 0.9:
            priming_verdict = "SATURATES: second neutral reduces amplification"
        else:
            priming_verdict = "ONE_SHOT: additional neutral doesn't change much"
    else:
        priming_verdict = "insufficient data"

    summary = {
        "n_readouts": len(observations),
        "chain_omega": chain_omega,
        "chain_attractor": chain_attractor,
        "priming_compounding": {
            "identity_only": id_only,
            "neutral_1_identity": n1,
            "neutral_2_identity": n2,
            "neutral_3_identity": n3,
            "verdict": priming_verdict,
        },
    }

    print(f"\n=== PRIMING VERDICT: {priming_verdict} ===")
    return summary


if __name__ == "__main__":
    run()
