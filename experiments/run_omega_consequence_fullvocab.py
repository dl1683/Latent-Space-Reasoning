"""Full-Vocab Consequence Prediction (Experiment #55).

Tests whether full-vocabulary TV recovers predictive power on the base model.
The 11-bin consequence experiment showed strong prediction on instruct (r=0.677)
but weak on base (r=0.222). If 11-bin loses critical information, full-vocab
TV should improve the base-model correlation.

288 readouts: 6 cells x 6 surfaces x 8 continuations.
"""

import json
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from omega import tv


MODEL_ID = "Qwen/Qwen3-1.7B-Base"

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
    "nonvar_42": "_ = 42\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "neutral_pass": "pass\n",
    "type_check": "type({var})\n",
    "safety_comment": "# Do not modify {var}.\n",
    "deprecation": "# Deprecated: {var} is no longer used.\n",
}

PREFIX_TEMPLATE = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            def k():\n"
    "                {var} = 99999\n                return {var}\n"
    "            k()\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

QUERY_TEMPLATE = "f()\nprint({var})  # Output: "

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]


def get_digit_token_ids(tokenizer):
    ids = []
    for d in range(10):
        tok = tokenizer.encode(str(d), add_special_tokens=False)
        ids.append(tok[0] if tok else None)
    return ids


def run_fullvocab_consequence():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map="cpu")
    model.eval()
    digit_ids = get_digit_token_ids(tokenizer)

    observations = []
    full_dists = {}
    t0 = time.time()
    n_readouts = 0

    def get_dist(text):
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :]
        return torch.softmax(logits, dim=-1).numpy()

    for depth, var, val in CELLS:
        prefix = PREFIX_TEMPLATE.format(var=var, outer_val=val)
        query = QUERY_TEMPLATE.replace("{var}", var)

        for surf_name, surf_tmpl in SURFACES.items():
            surface = surf_tmpl.replace("{var}", var)

            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont = cont_tmpl.replace("{var}", var)
                text = prefix + surface + cont + query
                dist = get_dist(text)
                n_readouts += 1

                key = (depth, var, val, surf_name, cont_name)
                full_dists[key] = dist

                fullvocab_argmax = int(np.argmax(dist))
                fullvocab_argmax_token = tokenizer.decode([fullvocab_argmax])
                fullvocab_argmax_prob = float(dist[fullvocab_argmax])

                digit_probs = []
                for did in digit_ids:
                    digit_probs.append(float(dist[did]) if did is not None else 0.0)
                other_prob = 1.0 - sum(digit_probs)
                bin11 = digit_probs + [other_prob]
                bin11_argmax = int(np.argmax(bin11))
                bin11_label = str(bin11_argmax) if bin11_argmax < 10 else "OTHER"

                correct_digit = str(val)
                entropy = float(-np.sum(dist * np.log(dist + 1e-30)))

                observations.append({
                    "depth": depth, "var": var, "val": val,
                    "surface": surf_name, "continuation": cont_name,
                    "fullvocab_argmax_id": fullvocab_argmax,
                    "fullvocab_argmax_token": fullvocab_argmax_token.strip(),
                    "fullvocab_argmax_prob": fullvocab_argmax_prob,
                    "bin11": bin11,
                    "bin11_argmax": bin11_label,
                    "correct_digit": correct_digit,
                    "is_correct_fullvocab": fullvocab_argmax_token.strip() == correct_digit,
                    "is_correct_bin11": bin11_label == correct_digit,
                    "entropy": entropy,
                })

                if n_readouts % 48 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_readouts}/288 readouts, {elapsed:.0f}s",
                          flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done: {n_readouts} readouts in {elapsed:.1f}s")
    del model
    return observations, full_dists, elapsed


def analyze(observations, full_dists):
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = obs

    all_pairs = []
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for cont_name in CONTINUATIONS:
            for depth, var, val in CELLS:
                key_a = (depth, var, val, surf_a, cont_name)
                key_b = (depth, var, val, surf_b, cont_name)
                oa = obs_idx[key_a]
                ob = obs_idx[key_b]

                tv_bin11 = tv(oa["bin11"], ob["bin11"])
                tv_full = float(0.5 * np.abs(
                    full_dists[key_a] - full_dists[key_b]).sum())
                fullvocab_flip = oa["fullvocab_argmax_id"] != ob["fullvocab_argmax_id"]
                bin11_flip = oa["bin11_argmax"] != ob["bin11_argmax"]

                all_pairs.append({
                    "pair": pair_name,
                    "continuation": cont_name,
                    "cell": f"{var}_{val}",
                    "tv_bin11": float(tv_bin11),
                    "tv_fullvocab": tv_full,
                    "fullvocab_flip": fullvocab_flip,
                    "bin11_flip": bin11_flip,
                    "correct_a_fv": oa["is_correct_fullvocab"],
                    "correct_b_fv": ob["is_correct_fullvocab"],
                    "correct_a_bin": oa["is_correct_bin11"],
                    "correct_b_bin": ob["is_correct_bin11"],
                    "argmax_a": oa["fullvocab_argmax_token"],
                    "argmax_b": ob["fullvocab_argmax_token"],
                    "entropy_a": oa["entropy"],
                    "entropy_b": ob["entropy"],
                })

    tvs_bin11 = np.array([p["tv_bin11"] for p in all_pairs])
    tvs_fv = np.array([p["tv_fullvocab"] for p in all_pairs])
    fv_flips = np.array([1 if p["fullvocab_flip"] else 0 for p in all_pairs])
    b11_flips = np.array([1 if p["bin11_flip"] else 0 for p in all_pairs])

    def safe_corr(x, y):
        if x.std() > 0 and y.std() > 0:
            return float(np.corrcoef(x, y)[0, 1])
        return 0.0

    r_bin11_bin11flip = safe_corr(tvs_bin11, b11_flips)
    r_bin11_fvflip = safe_corr(tvs_bin11, fv_flips)
    r_fvtv_fvflip = safe_corr(tvs_fv, fv_flips)
    r_fvtv_bin11flip = safe_corr(tvs_fv, b11_flips)

    n_fv_flips = int(fv_flips.sum())
    n_b11_flips = int(b11_flips.sum())
    n_pairs = len(all_pairs)

    fv_correct_a = sum(1 for o in observations if o["is_correct_fullvocab"])
    b11_correct_a = sum(1 for o in observations if o["is_correct_bin11"])

    print("\n=== FULL-VOCAB CONSEQUENCE RESULTS ===\n")
    print(f"Pairs: {n_pairs}")
    print(f"Full-vocab flips: {n_fv_flips}/{n_pairs} ({n_fv_flips/n_pairs:.1%})")
    print(f"11-bin flips:     {n_b11_flips}/{n_pairs} ({n_b11_flips/n_pairs:.1%})")
    print(f"Full-vocab accuracy: {fv_correct_a}/{len(observations)} ({fv_correct_a/len(observations):.1%})")
    print(f"11-bin accuracy:     {b11_correct_a}/{len(observations)} ({b11_correct_a/len(observations):.1%})")
    print(f"\nMean full-vocab TV: {tvs_fv.mean():.4f} (std {tvs_fv.std():.4f})")
    print(f"Mean 11-bin TV:     {tvs_bin11.mean():.4f} (std {tvs_bin11.std():.4f})")
    print(f"Full-vocab/11-bin TV ratio: {tvs_fv.mean() / max(tvs_bin11.mean(), 1e-10):.1f}x")
    print(f"\nCorrelations with flips:")
    print(f"  bin11-TV   vs bin11-flip:    {r_bin11_bin11flip:.3f}")
    print(f"  bin11-TV   vs fullvocab-flip: {r_bin11_fvflip:.3f}")
    print(f"  fullvocab-TV vs fullvocab-flip: {r_fvtv_fvflip:.3f}  <-- KEY")
    print(f"  fullvocab-TV vs bin11-flip:    {r_fvtv_bin11flip:.3f}")

    print("\n--- Per-continuation ---")
    for cont_name in CONTINUATIONS:
        cells = [p for p in all_pairs if p["continuation"] == cont_name]
        fv_fr = sum(1 for c in cells if c["fullvocab_flip"]) / len(cells)
        b11_fr = sum(1 for c in cells if c["bin11_flip"]) / len(cells)
        mean_tv11 = np.mean([c["tv_bin11"] for c in cells])
        mean_tvfv = np.mean([c["tv_fullvocab"] for c in cells])
        print(f"  {cont_name:20s}: fv_flip={fv_fr:.1%} b11_flip={b11_fr:.1%} "
              f"tv11={mean_tv11:.4f} tvfv={mean_tvfv:.4f}")

    print("\n--- Sample argmax tokens ---")
    for p in all_pairs[:12]:
        flip_mark = "*" if p["fullvocab_flip"] else " "
        print(f"  {flip_mark} {p['pair']:20s} {p['continuation']:15s} "
              f"a='{p['argmax_a']}' b='{p['argmax_b']}' "
              f"tv11={p['tv_bin11']:.4f} tvfv={p['tv_fullvocab']:.4f}")

    summary = {
        "n_readouts": len(observations),
        "n_pairs": n_pairs,
        "n_fullvocab_flips": n_fv_flips,
        "n_bin11_flips": n_b11_flips,
        "fullvocab_flip_rate": n_fv_flips / n_pairs,
        "bin11_flip_rate": n_b11_flips / n_pairs,
        "corr_bin11tv_bin11flip": r_bin11_bin11flip,
        "corr_bin11tv_fvflip": r_bin11_fvflip,
        "corr_fvtv_fvflip": r_fvtv_fvflip,
        "corr_fvtv_bin11flip": r_fvtv_bin11flip,
        "mean_tv_fullvocab": float(tvs_fv.mean()),
        "mean_tv_bin11": float(tvs_bin11.mean()),
        "tv_ratio": float(tvs_fv.mean() / max(tvs_bin11.mean(), 1e-10)),
        "fullvocab_accuracy": fv_correct_a / len(observations),
        "bin11_accuracy": b11_correct_a / len(observations),
        "model": MODEL_ID,
        "all_pairs": all_pairs,
    }
    return summary


if __name__ == "__main__":
    out_dir = Path("experiments/results/omega_consequence_fullvocab")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Full-Vocab Consequence Prediction (Exp #55) ===")
    print("Testing: does full-vocab TV recover predictive power on base?\n")

    observations, full_dists, elapsed = run_fullvocab_consequence()

    obs_for_json = []
    for obs in observations:
        o = dict(obs)
        obs_for_json.append(o)
    with open(out_dir / "observations.json", "w") as f:
        json.dump(obs_for_json, f, indent=2)

    summary = analyze(observations, full_dists)
    summary["elapsed_s"] = elapsed

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")
