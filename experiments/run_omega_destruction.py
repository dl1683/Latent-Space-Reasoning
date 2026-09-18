"""Omega Variable Destruction Test (Experiment #64).

Tests what happens when continuations DESTROY the variable binding.
If identity amplification is about "refreshing" the variable,
then destruction should produce a qualitatively different signature.

Continuations:
- del {var}         (NameError context)
- {var} = None      (overwrite with None)
- {var} = "deleted" (overwrite with string)
- {var} = 0         (overwrite with different value)
- {var} = {var}     (identity, control)
- empty             (baseline control)

Key predictions:
- If omega measures variable binding strength: del and overwrite should
  produce MAXIMUM omega (destroy the binding)
- If omega measures distributional perturbation: del should spread
  probability mass across all digits equally
- If the model "knows" del causes NameError: argmax should shift to OTHER

~216 readouts. Very fast.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv


MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
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

CONTINUATIONS = {
    "empty": "",
    "identity": "{var} = {var}\n",
    "del_var": "del {var}\n",
    "overwrite_none": "{var} = None\n",
    "overwrite_str": '{var} = "deleted"\n',
    "overwrite_zero": "{var} = 0\n",
    "overwrite_other": "{var} = 42\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "nonvar_42": "_ = 42\n",
}

QUERY = "f()\nprint({var})  # Output: "

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
TAU = 0.05


def run():
    print("=== Variable Destruction Test (Exp #64) ===")
    adapter = ModelAdapter({"model_id": MODEL_CFG["model_id"]})
    template = MODEL_CFG["templates"]["depth4_single"]

    observations = []
    t0 = time.time()
    total = len(CELLS) * len(SURFACES) * len(CONTINUATIONS)
    done = 0

    for depth, var, val in CELLS:
        prefix_code = template.format(var=var, outer_val=val)
        query = QUERY.format(var=var)
        prefix_state = adapter.get_state_after_prefix(prefix_code)

        for surf_name, surf_text in SURFACES.items():
            surface = surf_text.format(var=var)

            for cont_name, cont_text in CONTINUATIONS.items():
                continuation = cont_text.format(var=var)
                suffix = surface + continuation + query
                dist = adapter.get_dist_from_state(prefix_state, suffix)
                done += 1

                argmax_idx = int(np.argmax(dist))
                observations.append({
                    "depth": depth, "var": var, "val": val,
                    "surface": surf_name, "continuation": cont_name,
                    "dist": dist.tolist(),
                    "argmax_label": DIGIT_LABELS[argmax_idx],
                    "argmax_prob": float(dist[argmax_idx]),
                    "correct_digit": str(val),
                    "is_correct": DIGIT_LABELS[argmax_idx] == str(val),
                    "p_other": float(dist[10]),
                    "entropy": float(-sum(p * np.log(p + 1e-12) for p in dist)),
                })

                if done % 50 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    print(f"  {done}/{total} ({rate:.1f}/s)")

    elapsed = time.time() - t0
    print(f"\nDone: {len(observations)} readouts in {elapsed:.1f}s")

    out_dir = Path("experiments/results/omega_destruction")
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
        key = (r['var'], r['val'], r['surface'], r['continuation'])
        idx[key] = r

    cont_profiles = {}
    for cont_name in CONTINUATIONS:
        correct_count = 0
        other_count = 0
        total = 0
        entropies = []
        p_others = []
        p_corrects = []

        for depth, var, val in CELLS:
            for surf_name in SURFACES:
                obs = idx[(var, val, surf_name, cont_name)]
                total += 1
                if obs['is_correct']:
                    correct_count += 1
                if obs['argmax_label'] == 'OTHER':
                    other_count += 1
                entropies.append(obs['entropy'])
                p_others.append(obs['p_other'])
                correct_idx = int(obs['correct_digit'])
                p_corrects.append(obs['dist'][correct_idx])

        cont_profiles[cont_name] = {
            "accuracy": round(correct_count / total, 4),
            "other_rate": round(other_count / total, 4),
            "mean_entropy": round(float(np.mean(entropies)), 4),
            "mean_p_other": round(float(np.mean(p_others)), 4),
            "mean_p_correct": round(float(np.mean(p_corrects)), 4),
            "total": total,
        }

    pair_omega = {}
    for s1, s2, pair_name in EQUIV_PAIRS:
        cont_omega = {}
        for cont_name in CONTINUATIONS:
            tvs = []
            flips = 0
            for depth, var, val in CELLS:
                base_k1 = (var, val, s1, "empty")
                base_k2 = (var, val, s2, "empty")
                if base_k1 not in idx or base_k2 not in idx:
                    continue
                baseline = tv(idx[base_k1]['dist'], idx[base_k2]['dist'])
                if baseline > TAU:
                    continue

                k1 = (var, val, s1, cont_name)
                k2 = (var, val, s2, cont_name)
                if k1 not in idx or k2 not in idx:
                    continue
                cell_tv = float(tv(idx[k1]['dist'], idx[k2]['dist']))
                tvs.append(cell_tv)
                if idx[k1]['argmax_label'] != idx[k2]['argmax_label']:
                    flips += 1

            if tvs:
                cont_omega[cont_name] = {
                    "mean_tv": round(float(np.mean(tvs)), 6),
                    "max_tv": round(float(np.max(tvs)), 6),
                    "n_eligible": len(tvs),
                    "n_flips": flips,
                }
        pair_omega[pair_name] = cont_omega

    pooled_omega = {}
    for cont_name in CONTINUATIONS:
        all_tvs = []
        all_flips = 0
        for pn, po in pair_omega.items():
            if cont_name in po:
                for pn2, po2 in pair_omega.items():
                    pass
        for pn, po in pair_omega.items():
            if cont_name in po:
                all_tvs.extend([po[cont_name]["mean_tv"]])
                all_flips += po[cont_name].get("n_flips", 0)
        if all_tvs:
            pooled_omega[cont_name] = {
                "mean_tv": round(float(np.mean(all_tvs)), 6),
                "total_flips": all_flips,
            }

    print(f"\n=== PROFILES ===")
    print(f"{'Continuation':20s} {'Acc':>6s} {'OTHER%':>7s} {'H':>6s} "
          f"{'p(correct)':>10s} {'p(OTHER)':>9s} {'omega_tv':>9s}")
    for cn in CONTINUATIONS:
        cp = cont_profiles[cn]
        po = pooled_omega.get(cn, {"mean_tv": 0})
        print(f"{cn:20s} {cp['accuracy']:6.1%} {cp['other_rate']:7.1%} "
              f"{cp['mean_entropy']:6.3f} {cp['mean_p_correct']:10.4f} "
              f"{cp['mean_p_other']:9.4f} {po['mean_tv']:9.4f}")

    del_acc = cont_profiles.get("del_var", {}).get("accuracy", 0)
    del_other = cont_profiles.get("del_var", {}).get("other_rate", 0)
    identity_acc = cont_profiles.get("identity", {}).get("accuracy", 0)
    empty_acc = cont_profiles.get("empty", {}).get("accuracy", 0)

    if del_other > 0.3:
        destruction_effect = "MODEL_KNOWS_DEL: high OTHER rate after del"
    elif del_acc < empty_acc - 0.2:
        destruction_effect = "DEL_DISRUPTS: accuracy drops significantly"
    else:
        destruction_effect = "DEL_IGNORED: model still predicts original value"

    ow_accs = [cont_profiles[cn]["accuracy"] for cn in
               ["overwrite_none", "overwrite_str", "overwrite_zero", "overwrite_other"]
               if cn in cont_profiles]
    mean_ow_acc = float(np.mean(ow_accs)) if ow_accs else 0

    if mean_ow_acc > 0.5:
        overwrite_effect = "OVERWRITES_IGNORED: model still predicts original"
    elif mean_ow_acc < 0.1:
        overwrite_effect = "OVERWRITES_EFFECTIVE: model tracks new value"
    else:
        overwrite_effect = "OVERWRITES_PARTIAL: mixed accuracy"

    print(f"\nDestruction: {destruction_effect}")
    print(f"Overwrites: {overwrite_effect}")

    return {
        "n_readouts": len(observations),
        "continuation_profiles": cont_profiles,
        "pair_omega": pair_omega,
        "pooled_omega": pooled_omega,
        "destruction_effect": destruction_effect,
        "overwrite_effect": overwrite_effect,
    }


if __name__ == "__main__":
    run()
