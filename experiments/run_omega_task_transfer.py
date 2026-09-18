"""Omega Task Transfer (Experiment #50).

Do omega findings transfer to a different code task? All prior experiments
use the depth-4 variable scoping task. This tests omega on TWO new tasks:

1. Simple arithmetic: x = 3; y = x + 2; print(y)  # Output:
2. String formatting: name = "Alice"; greeting = f"Hello {name}"

If omega amplification replicates on these tasks, the findings are
task-general. If not, they're specific to variable scoping.

Continuation types: identity, heavy, neutral, safety_donot, depr_deprecated
(the key findings from prior experiments).
"""

import json
import sys
import time
import copy
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from omega import tv


def make_adapter():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B", trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", trust_remote_code=True, torch_dtype=torch.float32)
    mdl.eval()
    digit_ids = {}
    for d in range(10):
        toks = tok.encode(str(d), add_special_tokens=False)
        assert len(toks) == 1
        digit_ids[d] = toks[0]
    return tok, mdl, digit_ids


def extract_11bin(logits, digit_ids):
    probs = torch.softmax(logits, dim=0).numpy().astype(np.float64)
    bins = np.zeros(11, dtype=np.float64)
    for d in range(10):
        bins[d] = probs[digit_ids[d]]
    bins[10] = 1.0 - bins[:10].sum()
    return bins


def get_dist(tok, mdl, digit_ids, text):
    ids = tok.encode(text, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        out = mdl(ids, use_cache=True)
    return extract_11bin(out.logits[0, -1, :], digit_ids)


# Task 1: Arithmetic
ARITH_TEMPLATES = {
    "add": "x = {a}\ny = x + {b}\nprint(y)  # Output: ",
    "sub": "x = {a}\ny = x - {b}\nprint(y)  # Output: ",
    "mul": "x = {a}\ny = x * {b}\nprint(y)  # Output: ",
}

ARITH_SURFACES = {
    "standard": "",  # no surface
    "comment_ok": "# Correct.\n",
    "comment_good": "# Good.\n",
}

ARITH_PAIRS = [
    ("comment_ok", "comment_good", "arith_ok_good"),
]

# Task 2: String (measure using full vocab TV, not 11-bin)
# Skip this — full-vocab requires different infrastructure.
# Stick with arithmetic which gives numeric outputs.

ARITH_VALUES = [
    (3, 2),  # x=3, op=+2 -> 5
    (7, 4),  # x=7, op=+4 -> 11 (two digit!)
    (5, 1),  # x=5, op=+1 -> 6
    (9, 3),  # x=9, op=+3 -> 12
    (2, 8),  # x=2, op=+8 -> 10
    (4, 6),  # x=4, op=+6 -> 10
]

CONTINUATIONS = {
    "identity": "x = x\n",
    "heavy": "x = x + 1\n",
    "neutral": "pass\n",
    "safety_donot": "# SAFETY: do not modify\nx = x\n",
    "depr_deprecated": "# DEPRECATED: will be removed\nx = x\n",
}


def run_task_transfer():
    print("=== Omega Task Transfer (Exp #50) ===")
    print("Testing omega on arithmetic task\n")

    tok, mdl, digit_ids = make_adapter()
    observations = []
    t0 = time.time()
    n = 0

    for op_name, op_tmpl in ARITH_TEMPLATES.items():
        for a, b in ARITH_VALUES:
            for surf_name, surf_text in ARITH_SURFACES.items():
                for cont_name, cont_text in CONTINUATIONS.items():
                    code = op_tmpl.format(a=a, b=b)
                    # Insert surface + continuation before the query
                    parts = code.rsplit("\n", 1)
                    prefix = parts[0] + "\n"
                    query = parts[1] if len(parts) > 1 else ""
                    full_text = prefix + surf_text + cont_text + query

                    dist = get_dist(tok, mdl, digit_ids, full_text)
                    n += 1

                    observations.append({
                        "task": op_name,
                        "a": a,
                        "b": b,
                        "surface": surf_name,
                        "continuation": cont_name,
                        "dist": dist.tolist(),
                    })

                    if n % 50 == 0:
                        print(f"  {n} readouts, {time.time()-t0:.0f}s",
                              flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n} readouts in {elapsed:.1f}s")

    result_dir = Path("experiments/results/omega_task_transfer")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    # Analysis
    obs_idx = {}
    for obs in observations:
        key = (obs["task"], obs["a"], obs["b"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    summary = {"n_readouts": n, "elapsed_s": elapsed, "pairs": {}}

    print("\n=== Omega on Arithmetic Task ===")
    for surf_a, surf_b, pair_name in ARITH_PAIRS:
        for tau in [0.01, 0.05]:
            eligible = set()
            for op in ARITH_TEMPLATES:
                for a, b in ARITH_VALUES:
                    ka = (op, a, b, surf_a, "identity")
                    kb = (op, a, b, surf_b, "identity")
                    if ka in obs_idx and kb in obs_idx:
                        d = tv(obs_idx[ka], obs_idx[kb])
                        if d <= tau:
                            eligible.add((op, a, b))

            print(f"\n  {pair_name} tau={tau} ({len(eligible)} eligible):")
            pair_omega = {}
            for cont in CONTINUATIONS:
                tvs = []
                for cell in eligible:
                    op, a, b = cell
                    ka = (op, a, b, surf_a, cont)
                    kb = (op, a, b, surf_b, cont)
                    if ka in obs_idx and kb in obs_idx:
                        tvs.append(tv(obs_idx[ka], obs_idx[kb]))
                if tvs:
                    omega = float(np.max(tvs))
                    mean = float(np.mean(tvs))
                    pair_omega[cont] = {"omega": omega, "mean": mean}
                    print(f"    {cont:22s}: omega={omega:.5f} "
                          f"mean={mean:.5f}")

            summary["pairs"][f"{pair_name}_tau{tau}"] = pair_omega

    # Also report per-task baselines
    print("\n=== Baseline distances (surface pair TV) ===")
    for surf_a, surf_b, pair_name in ARITH_PAIRS:
        for op in ARITH_TEMPLATES:
            for a, b in ARITH_VALUES:
                ka = (op, a, b, surf_a, "identity")
                kb = (op, a, b, surf_b, "identity")
                if ka in obs_idx and kb in obs_idx:
                    d = tv(obs_idx[ka], obs_idx[kb])
                    print(f"  {op} a={a} b={b}: TV={d:.6f}")

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {result_dir}/")
    return observations, summary


if __name__ == "__main__":
    run_task_transfer()
