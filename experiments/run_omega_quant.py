"""Omega Quantization: Does quantization change the omega profile?

Compares Qwen3-1.7B at fp32 (baseline) vs int8 (dynamic quantization on Linear layers).
If the ranking is invariant, safety certificates transfer across precisions.
If the ranking changes, deployment at reduced precision breaks the guarantee.

6 cells x 2 pairs x 2 sides x 6 continuations = 144 readouts per precision = 288 total
"""

import copy
import json
import sys
import time
import numpy as np
from pathlib import Path

import torch
import torch.quantization

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import build_prefix


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


class QuantAdapter:
    """Model adapter that supports fp32 and int8 dynamic quantization."""

    def __init__(self, model_id, quantize=False):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32)
        self.mdl.eval()

        if quantize:
            self.mdl = torch.quantization.quantize_dynamic(
                self.mdl, {torch.nn.Linear}, dtype=torch.qint8)
            self.precision = "int8"
        else:
            self.precision = "fp32"

        self.digit_token_ids = {}
        for d in range(10):
            toks = self.tok.encode(str(d), add_special_tokens=False)
            assert len(toks) == 1, f"Digit {d} not single token: {toks}"
            self.digit_token_ids[d] = toks[0]
        self.call_count = 0

    def _extract_11bin(self, logits):
        probs = torch.softmax(logits, dim=0).numpy().astype(np.float64)
        bins = np.zeros(11, dtype=np.float64)
        for d in range(10):
            bins[d] = probs[self.digit_token_ids[d]]
        bins[10] = 1.0 - bins[:10].sum()
        return bins

    def get_state_after_prefix(self, text):
        ids = self.tok.encode(text, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            out = self.mdl(ids, use_cache=True)
        self.call_count += 1
        return copy.deepcopy(out.past_key_values)

    def get_dist_from_state(self, state, suffix_text):
        ids = self.tok.encode(suffix_text, add_special_tokens=False, return_tensors="pt")
        st = copy.deepcopy(state)
        with torch.no_grad():
            out = self.mdl(ids, past_key_values=st, use_cache=True)
        self.call_count += 1
        return self._extract_11bin(out.logits[0, -1, :])


def run_one_precision(model_id, quantize, label):
    print(f"\n--- {label} (quantize={quantize}) ---")
    adapter = QuantAdapter(model_id, quantize=quantize)

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
                        "precision": label,
                        "depth": depth, "var": var, "val": val,
                        "pair": pair_name, "side": side,
                        "continuation": cont_name,
                        "dist": dist.tolist(),
                    })

    elapsed = time.time() - t0
    print(f"  Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")
    del adapter
    import gc; gc.collect()
    return observations


def analyze(all_obs):
    print("\n=== FP32 vs INT8 Omega Comparison ===")

    precisions = sorted(set(o["precision"] for o in all_obs))

    for tau in [0.01, 0.02, 0.05]:
        print(f"\n--- tau={tau} ---")
        print(f"{'Precision':<12s} {'Pair':<20s} {'Cont':<12s} "
              f"{'mean':>8s} {'max':>8s} {'n':>3s} {'status':>8s}")

        for prec in precisions:
            obs = [o for o in all_obs if o["precision"] == prec]
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
                        print(f"  {prec:<10s} {pair_name:<20s} {cont_name:<12s} "
                              f"{mean_tv:8.5f} {max_tv:8.5f} {len(tvs):3d} "
                              f"{status:>8s}")

    # Ranking comparison
    print("\n=== Amplification Ranking Comparison (tau=0.01) ===")
    for prec in precisions:
        obs = [o for o in all_obs if o["precision"] == prec]
        obs_idx = {}
        for o in obs:
            key = (o["var"], o["val"], o["pair"], o["side"], o["continuation"])
            obs_idx[key] = np.array(o["dist"], dtype=np.float64)

        for pair_name in [p[2] for p in SURFACE_PAIRS]:
            cont_means = {}
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
                    if d_pre > 0.01:
                        continue
                    ka = (var, val, pair_name, "a", cont_name)
                    kb = (var, val, pair_name, "b", cont_name)
                    if ka in obs_idx and kb in obs_idx:
                        d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                        tvs.append(d)
                if tvs:
                    cont_means[cont_name] = float(np.mean(tvs))

            if cont_means:
                ranked = sorted(cont_means.items(), key=lambda x: -x[1])
                print(f"\n  {prec} / {pair_name}:")
                for i, (name, v) in enumerate(ranked, 1):
                    print(f"    {i}. {name:<12s} = {v:.5f}")

    # Cross-precision TV distance (how different are the raw distributions?)
    print("\n=== Cross-Precision Distribution Divergence ===")
    fp32_obs = {
        (o["var"], o["val"], o["pair"], o["side"], o["continuation"]):
        np.array(o["dist"], dtype=np.float64)
        for o in all_obs if o["precision"] == "fp32"
    }
    int8_obs = {
        (o["var"], o["val"], o["pair"], o["side"], o["continuation"]):
        np.array(o["dist"], dtype=np.float64)
        for o in all_obs if o["precision"] == "int8"
    }
    common = set(fp32_obs.keys()) & set(int8_obs.keys())
    if common:
        tvs = [float(0.5 * np.abs(fp32_obs[k] - int8_obs[k]).sum()) for k in common]
        print(f"  N={len(tvs)} common readouts")
        print(f"  Mean TV(fp32, int8) = {np.mean(tvs):.5f}")
        print(f"  Max  TV(fp32, int8) = {np.max(tvs):.5f}")
        print(f"  Min  TV(fp32, int8) = {np.min(tvs):.5f}")

        by_cont = {}
        for k in common:
            cn = k[4]
            by_cont.setdefault(cn, []).append(
                float(0.5 * np.abs(fp32_obs[k] - int8_obs[k]).sum()))
        print(f"\n  Per-continuation mean TV(fp32, int8):")
        for cn in sorted(by_cont.keys()):
            print(f"    {cn:<12s} = {np.mean(by_cont[cn]):.5f}")


def main():
    print("=== Omega Quantization Experiment ===")
    model_id = "Qwen/Qwen3-1.7B"
    result_dir = Path("experiments/results/omega_quant")
    result_dir.mkdir(parents=True, exist_ok=True)

    all_obs = []

    for quantize, label in [(False, "fp32"), (True, "int8")]:
        try:
            obs = run_one_precision(model_id, quantize, label)
            all_obs.extend(obs)
            with open(result_dir / f"observations_{label}.json", "w") as f:
                json.dump(obs, f)
        except Exception as e:
            print(f"  ERROR on {label}: {e}")
            import traceback
            traceback.print_exc()

    if all_obs:
        with open(result_dir / "observations_all.json", "w") as f:
            json.dump(all_obs, f)
        analyze(all_obs)

        summary = {
            "n_readouts": len(all_obs),
            "precisions": list(set(o["precision"] for o in all_obs)),
            "model": model_id,
        }
        with open(result_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nSaved {len(all_obs)} observations to {result_dir}/")
    return all_obs


if __name__ == "__main__":
    main()
