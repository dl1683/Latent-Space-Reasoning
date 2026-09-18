"""Omega Layer-wise Fingerprint: Does each layer have a distinct omega signature?

For each model layer L, extract hidden_states[L], apply the final LayerNorm + LM head
to get pseudo-logits, then compute the 11-bin distribution. Measure TV distances
between surface pairs at each layer to see how the omega profile develops through
the network depth.

Key question: Is omega flat (distributed), monotonically increasing (progressive),
or non-monotonic (compression bottleneck)?

Target: ~200-300 readouts across 7 layer checkpoints × 3 pairs × 5 continuations.
"""

import json
import sys
import time
import copy
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import build_prefix
from run_omega_sweep import MODEL_CFG, EQUIV_PAIRS

MODEL_ID = "Qwen/Qwen3-1.7B"

LAYER_CHECKPOINTS = [0, 4, 8, 12, 16, 20, 24, 27]

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

PAIRS = [
    ("assert_novar_1", "assert_var_1", "strong_assert"),
    ("assert_novar_2", "assert_var_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
]

SURFACES = {
    "none": "",
    "assert_novar_1": "# No changes.\n",
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_1": "# {var} is unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
}

CONTINUATIONS = {
    "none": "",
    "neutral": "# No changes.\n",
    "identity": "{var} = {var}\n",
    "heavy": "# Reassigning {var} now.\n",
    "type_check": "type({var})\n",
}


class LayerProbeAdapter:
    """Model adapter that can extract 11-bin distributions at any layer."""

    def __init__(self, model_id):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32)
        self.mdl.eval()

        self.digit_token_ids = {}
        for d in range(10):
            toks = self.tok.encode(str(d), add_special_tokens=False)
            assert len(toks) == 1, f"Digit {d} not single token: {toks}"
            self.digit_token_ids[d] = toks[0]

        self.final_norm = self.mdl.model.norm
        self.lm_head = self.mdl.lm_head
        self.n_layers = len(self.mdl.model.layers)
        self.call_count = 0
        print(f"Model loaded: {model_id}, {self.n_layers} layers", flush=True)

    def _extract_11bin(self, logits):
        probs = torch.softmax(logits, dim=0).detach().numpy().astype(np.float64)
        bins = np.zeros(11, dtype=np.float64)
        for d in range(10):
            bins[d] = probs[self.digit_token_ids[d]]
        bins[10] = 1.0 - bins[:10].sum()
        return bins

    def get_layer_dists(self, text, layers):
        """Get 11-bin distributions at specified layers for a given text.

        For each layer L: take hidden_states[L] at the last token position,
        apply the final LayerNorm, then the LM head, then extract 11-bin.
        Also returns the normal final-layer distribution for comparison.

        Returns dict mapping layer_index -> 11-bin numpy array.
        """
        ids = self.tok.encode(text, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            out = self.mdl(ids, output_hidden_states=True)
        self.call_count += 1

        result = {}
        with torch.no_grad():
            for L in layers:
                h = out.hidden_states[L][0, -1, :]
                normed = self.final_norm(h.unsqueeze(0)).squeeze(0)
                logits = self.lm_head(normed)
                result[L] = self._extract_11bin(logits)

            result["final"] = self._extract_11bin(out.logits[0, -1, :])
        return result


def tv(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()


def main():
    print("=== Omega Layer-wise Fingerprint ===", flush=True)

    adapter = LayerProbeAdapter(MODEL_ID)
    query_tmpl = MODEL_CFG["query_template"]
    observations = []
    t0 = time.time()
    n_readouts = 0

    for depth, var, val in CELLS:
        tmpl_key = f"depth{depth}_single"
        tmpl = MODEL_CFG["templates"][tmpl_key]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)

        for surf_name, surf_tmpl in SURFACES.items():
            surface_text = surf_tmpl.replace("{var}", var)
            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                full_text = prefix_text + surface_text + cont_text + query

                layer_dists = adapter.get_layer_dists(full_text, LAYER_CHECKPOINTS)
                n_readouts += 1

                obs = {
                    "depth": depth,
                    "var": var,
                    "val": val,
                    "surface": surf_name,
                    "continuation": cont_name,
                    "layer_dists": {str(k): v.tolist() for k, v in layer_dists.items()},
                }
                observations.append(obs)

                if n_readouts % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_readouts} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n_readouts} readouts in {elapsed:.1f}s, "
          f"{adapter.call_count} model calls", flush=True)

    result_dir = Path("experiments/results/omega_layers")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    print("\n=== Computing layer-wise omega ===", flush=True)
    layer_omega = compute_layer_omega(observations, PAIRS)

    with open(result_dir / "layer_omega.json", "w") as f:
        json.dump(layer_omega, f, indent=2)

    print_summary(layer_omega)

    summary = build_summary(layer_omega, n_readouts, elapsed)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    del adapter
    print(f"\nResults saved to {result_dir}/", flush=True)


def compute_layer_omega(observations, pairs, tau=0.01):
    """Compute omega at each layer for each pair and continuation type."""
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = obs["layer_dists"]

    all_layers = [str(L) for L in LAYER_CHECKPOINTS] + ["final"]
    results = {}

    for surf_a, surf_b, pair_name in pairs:
        pair_result = {}

        for layer_key in all_layers:
            baseline_eligible = set()
            for depth, var, val in CELLS:
                key_a = (depth, var, val, surf_a, "none")
                key_b = (depth, var, val, surf_b, "none")
                if key_a in obs_idx and key_b in obs_idx:
                    dist_a = np.array(obs_idx[key_a][layer_key])
                    dist_b = np.array(obs_idx[key_b][layer_key])
                    d = tv(dist_a, dist_b)
                    if d <= tau:
                        baseline_eligible.add((depth, var, val))

            cont_omega = {}
            for cont_name in CONTINUATIONS:
                post_tvs = []
                for cell in baseline_eligible:
                    depth, var, val = cell
                    key_a = (depth, var, val, surf_a, cont_name)
                    key_b = (depth, var, val, surf_b, cont_name)
                    if key_a in obs_idx and key_b in obs_idx:
                        dist_a = np.array(obs_idx[key_a][layer_key])
                        dist_b = np.array(obs_idx[key_b][layer_key])
                        d = tv(dist_a, dist_b)
                        post_tvs.append(d)

                if post_tvs:
                    cont_omega[cont_name] = {
                        "omega": float(max(post_tvs)),
                        "mean_tv": float(np.mean(post_tvs)),
                        "n_eligible": len(post_tvs),
                    }
                else:
                    cont_omega[cont_name] = {
                        "omega": None,
                        "mean_tv": None,
                        "n_eligible": 0,
                    }

            pair_result[layer_key] = {
                "n_eligible": len(baseline_eligible),
                "continuations": cont_omega,
            }

        results[pair_name] = pair_result

    return results


def print_summary(layer_omega):
    """Print a readable summary of layer-wise omega."""
    all_layers = [str(L) for L in LAYER_CHECKPOINTS] + ["final"]

    for pair_name, pair_data in layer_omega.items():
        print(f"\n=== {pair_name} ===")
        header = f"{'Layer':>8}"
        for cn in CONTINUATIONS:
            header += f"  {cn:>12}"
        header += f"  {'eligible':>8}"
        print(header)

        for lk in all_layers:
            ld = pair_data[lk]
            row = f"{lk:>8}"
            for cn in CONTINUATIONS:
                cd = ld["continuations"][cn]
                if cd["omega"] is not None:
                    row += f"  {cd['mean_tv']:>12.5f}"
                else:
                    row += f"  {'---':>12}"
            row += f"  {ld['n_eligible']:>8}"
            print(row)


def build_summary(layer_omega, n_readouts, elapsed_s):
    """Build a machine-readable summary with key findings."""
    all_layers = [str(L) for L in LAYER_CHECKPOINTS] + ["final"]

    profile_type = "unknown"
    identity_by_layer = {}

    for pair_name, pair_data in layer_omega.items():
        for lk in all_layers:
            ld = pair_data[lk]
            if "identity" in ld["continuations"]:
                cd = ld["continuations"]["identity"]
                if cd["mean_tv"] is not None:
                    if lk not in identity_by_layer:
                        identity_by_layer[lk] = []
                    identity_by_layer[lk].append(cd["mean_tv"])

    layer_means = {}
    for lk in all_layers:
        if lk in identity_by_layer and identity_by_layer[lk]:
            layer_means[lk] = float(np.mean(identity_by_layer[lk]))

    if len(layer_means) >= 3:
        vals = [layer_means.get(lk, 0) for lk in all_layers if lk in layer_means]
        if all(vals[i] <= vals[i+1] + 0.005 for i in range(len(vals)-1)):
            profile_type = "monotonically_increasing"
        elif all(vals[i] >= vals[i+1] - 0.005 for i in range(len(vals)-1)):
            profile_type = "monotonically_decreasing"
        else:
            mid = len(vals) // 2
            early_mean = np.mean(vals[:mid])
            late_mean = np.mean(vals[mid:])
            mid_val = vals[mid] if mid < len(vals) else 0
            if mid_val < early_mean and mid_val < late_mean:
                profile_type = "compression_bottleneck"
            else:
                profile_type = "non_monotonic"

    return {
        "n_readouts": n_readouts,
        "elapsed_s": elapsed_s,
        "model": MODEL_ID,
        "layers_probed": LAYER_CHECKPOINTS,
        "n_pairs": len(PAIRS),
        "n_continuations": len(CONTINUATIONS),
        "profile_type": profile_type,
        "identity_omega_by_layer": layer_means,
        "tau": 0.01,
    }


if __name__ == "__main__":
    main()
