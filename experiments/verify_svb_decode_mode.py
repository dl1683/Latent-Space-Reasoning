"""Verify SVB measurements: buggy multi-token vs correct single-token decode.

Runs a small subset of SVB-0 (depth 1-2, suffix s0 and s1) in both modes
on Falcon-H1 to directly compare the results. Also runs Mode F (full text)
as ground truth.

Expected: if the Mamba state bug matters, multi-token and decode modes
will differ significantly. Full-text mode is the reference.
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "tiiuae/Falcon-H1-1.5B-Instruct"
VARIABLES = ["x", "y", "z"]
OUTER_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
COMMENT = "# No changes.\n"
PASS_SUFFIX = "pass\n"
QUERY_TEMPLATE = "f()\nprint({var})  # Output: "

DEPTH1_TEMPLATE = """{var} = {outer_val}
def f():
    {var} = 0
{suffix}"""

DEPTH2_TEMPLATE = """{var} = {outer_val}
def f():
    {var} = 0
    def g():
        {var} = 0
{suffix}"""

SUFFIX_MAP = {
    "s0": "",
    "s1": COMMENT + PASS_SUFFIX,
}

print("Loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32)
mdl.eval()

digit_ids = {}
for d in range(10):
    toks = tok.encode(str(d), add_special_tokens=False)
    assert len(toks) == 1
    digit_ids[d] = toks[0]


def extract_11bin(logits):
    probs = torch.softmax(logits, dim=0).numpy().astype(np.float64)
    bins = np.zeros(11, dtype=np.float64)
    for d in range(10):
        bins[d] = probs[digit_ids[d]]
    bins[10] = 1.0 - bins[:10].sum()
    return bins


def get_cache(out):
    if hasattr(out, 'cache_params') and out.cache_params is not None:
        return out.cache_params
    return out.past_key_values


def forward_with_cache(ids, state):
    if hasattr(state, 'conv_states') or (hasattr(state, '__class__') and 'Mamba' in type(state).__name__):
        return mdl(ids, cache_params=state, use_cache=True)
    return mdl(ids, past_key_values=state, use_cache=True)


def mode_multitoken(prefix_text, suffix_text, query_text):
    """Buggy mode: prefix->cache->multi-token suffix+query continuation."""
    prefix_ids = tok.encode(prefix_text, add_special_tokens=False, return_tensors="pt")
    suffix_query = suffix_text + query_text
    sq_ids = tok.encode(suffix_query, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        out = mdl(prefix_ids, use_cache=True)
    state = copy.deepcopy(get_cache(out))
    with torch.no_grad():
        out2 = forward_with_cache(sq_ids, state)
    return extract_11bin(out2.logits[0, -1, :])


def mode_decode(prefix_text, suffix_text, query_text):
    """Correct mode: prefix->cache->single-token decode for suffix+query."""
    prefix_ids = tok.encode(prefix_text, add_special_tokens=False, return_tensors="pt")
    suffix_query = suffix_text + query_text
    sq_ids = tok.encode(suffix_query, add_special_tokens=False)
    with torch.no_grad():
        out = mdl(prefix_ids, use_cache=True)
    state = get_cache(out)
    for tok_id in sq_ids:
        with torch.no_grad():
            out = forward_with_cache(torch.tensor([[tok_id]]), state)
        state = get_cache(out)
    return extract_11bin(out.logits[0, -1, :])


def mode_fulltext(prefix_text, suffix_text, query_text):
    """Ground truth: full text in one pass, no cache."""
    full = prefix_text + suffix_text + query_text
    ids = tok.encode(full, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        out = mdl(ids, use_cache=False)
    return extract_11bin(out.logits[0, -1, :])


results = {}
call_count = 0
t0 = time.time()

for depth, template in [(1, DEPTH1_TEMPLATE), (2, DEPTH2_TEMPLATE)]:
    for sname, suffix in SUFFIX_MAP.items():
        key = f"d{depth}_{sname}"
        multi_sigmas = []
        decode_sigmas = []
        full_sigmas = []
        tvs_md = []
        tvs_mf = []
        tvs_df = []

        for var in VARIABLES:
            query_text = QUERY_TEMPLATE.replace("{var}", var)
            for val in OUTER_VALUES:
                prefix = template.replace("{var}", var).replace(
                    "{outer_val}", str(val)).replace("{suffix}", suffix)

                d_multi = mode_multitoken(prefix, "", query_text)
                d_decode = mode_decode(prefix, "", query_text)
                d_full = mode_fulltext(prefix, "", query_text)
                call_count += 3

                multi_sigmas.append(float(d_multi[val]))
                decode_sigmas.append(float(d_decode[val]))
                full_sigmas.append(float(d_full[val]))

                tv_md = 0.5 * np.abs(d_multi - d_decode).sum()
                tv_mf = 0.5 * np.abs(d_multi - d_full).sum()
                tv_df = 0.5 * np.abs(d_decode - d_full).sum()
                tvs_md.append(float(tv_md))
                tvs_mf.append(float(tv_mf))
                tvs_df.append(float(tv_df))

        results[key] = {
            "multi_sigma_mean": np.mean(multi_sigmas),
            "decode_sigma_mean": np.mean(decode_sigmas),
            "full_sigma_mean": np.mean(full_sigmas),
            "tv_multi_vs_decode": np.mean(tvs_md),
            "tv_multi_vs_full": np.mean(tvs_mf),
            "tv_decode_vs_full": np.mean(tvs_df),
        }
        elapsed = time.time() - t0
        print(f"{key}: multi={results[key]['multi_sigma_mean']:.4f} "
              f"decode={results[key]['decode_sigma_mean']:.4f} "
              f"full={results[key]['full_sigma_mean']:.4f} "
              f"| TV m/d={results[key]['tv_multi_vs_decode']:.4f} "
              f"m/f={results[key]['tv_multi_vs_full']:.4f} "
              f"d/f={results[key]['tv_decode_vs_full']:.4f} "
              f"({call_count} calls, {elapsed:.0f}s)", flush=True)

print(f"\nTotal: {call_count} calls, {time.time()-t0:.0f}s", flush=True)
print("\n=== SUMMARY ===", flush=True)
for k, v in results.items():
    diff_multi_full = v["multi_sigma_mean"] - v["full_sigma_mean"]
    diff_decode_full = v["decode_sigma_mean"] - v["full_sigma_mean"]
    print(f"{k}:", flush=True)
    print(f"  sigma: multi={v['multi_sigma_mean']:.4f} "
          f"decode={v['decode_sigma_mean']:.4f} "
          f"full={v['full_sigma_mean']:.4f}", flush=True)
    print(f"  delta from full: multi={diff_multi_full:+.4f} "
          f"decode={diff_decode_full:+.4f}", flush=True)
    print(f"  TV: multi-decode={v['tv_multi_vs_decode']:.4f} "
          f"multi-full={v['tv_multi_vs_full']:.4f} "
          f"decode-full={v['tv_decode_vs_full']:.4f}", flush=True)

out_path = Path("experiments/results/svb_decode_verification.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}", flush=True)
