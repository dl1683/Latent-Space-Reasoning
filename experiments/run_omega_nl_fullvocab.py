"""Full-vocabulary NL omega measurement.

Exp #43. The 11-bin TV projection kills NL signal (exp #39 showed NL 11-bin
TV = 0.000001 vs full-vocab TV = 0.065). Measure NL omega with full-vocab TV
to determine if NL omega is truly comparable to code omega.
"""
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix

MODEL_ID = "Qwen/Qwen3-1.7B"

NL_PAIRS = [
    ("He said that he", "He mentioned that he", "said_mentioned"),
    ("He was tired after a long", "He was exhausted after a long", "tired_exhausted"),
    ("The house was built in", "The home was constructed in", "house_home"),
    ("The law was passed by", "The law was enacted by", "passive_voice"),
]

NL_CONTINUATIONS = {
    "none": "",
    "period": ". ",
    "comma": ", ",
    "and_conj": " and ",
    "exclamation": "! ",
    "question": "? ",
    "semicolon": "; ",
    "the": " the ",
    "day": " day ",
    "was": " was ",
}

CODE_PAIRS = [
    ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ("\n", "pass\n", "boundary_pair"),
]

CODE_CONTINUATIONS = {
    "none": "",
    "identity": "{var} = {var}\n",
    "neutral": "# No changes.\n",
    "heavy": "# Reassigning {var} now.\n",
    "type_check": "type({var})\n",
}

CODE_CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
]

CODE_TEMPLATE = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            def k():\n"
    "                {var} = 99999\n                return {var}\n"
    "            k()\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

QUERY_TEMPLATE = "f()\nprint({var})  # Output: "

DIGIT_TOKENS = None


def get_digit_token_ids(tokenizer):
    ids = []
    for d in range(10):
        token_id = tokenizer.encode(str(d), add_special_tokens=False)
        ids.append(token_id[0] if token_id else None)
    return ids


def tv(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()


class FullVocabAdapter:
    def __init__(self, model_id):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        self.model.eval()
        global DIGIT_TOKENS
        DIGIT_TOKENS = get_digit_token_ids(self.tokenizer)

    def _get_full_dist(self, text):
        tokens = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**tokens)
        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits, dim=0).numpy()
        return probs

    def _get_11bin_dist(self, text):
        probs = self._get_full_dist(text)
        bins = np.zeros(11)
        for i, tid in enumerate(DIGIT_TOKENS):
            if tid is not None:
                bins[i] = probs[tid]
        bins[10] = 1.0 - bins[:10].sum()
        return bins

    def get_state_after_prefix(self, prefix):
        tokens = self.tokenizer(prefix, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**tokens, use_cache=True)
        return {
            "past_key_values": out.past_key_values,
            "prefix_len": tokens["input_ids"].shape[1],
        }

    def get_dists_from_state(self, state, suffix):
        suffix_tokens = self.tokenizer(suffix, return_tensors="pt", add_special_tokens=False)
        past = state["past_key_values"]
        with torch.no_grad():
            out = self.model(
                input_ids=suffix_tokens["input_ids"],
                past_key_values=past,
                use_cache=False,
            )
        logits = out.logits[0, -1, :]
        full_probs = torch.softmax(logits, dim=0).numpy()
        bins = np.zeros(11)
        for i, tid in enumerate(DIGIT_TOKENS):
            if tid is not None:
                bins[i] = full_probs[tid]
        bins[10] = 1.0 - bins[:10].sum()
        return full_probs, bins


def run_nl_experiment(adapter):
    observations = []
    n = 0
    t0 = time.time()

    for text_a, text_b, pair_name in NL_PAIRS:
        state_a = adapter.get_state_after_prefix(text_a)
        state_b = adapter.get_state_after_prefix(text_b)

        for cont_name, cont_text in NL_CONTINUATIONS.items():
            if not cont_text:
                full_a = adapter._get_full_dist(text_a)
                full_b = adapter._get_full_dist(text_b)
                bin_a = adapter._get_11bin_dist(text_a)
                bin_b = adapter._get_11bin_dist(text_b)
            else:
                full_a, bin_a = adapter.get_dists_from_state(state_a, cont_text)
                full_b, bin_b = adapter.get_dists_from_state(state_b, cont_text)

            tv_full = tv(full_a, full_b)
            tv_11bin = tv(bin_a, bin_b)
            n += 2

            observations.append({
                "domain": "nl",
                "pair": pair_name,
                "text_a": text_a,
                "text_b": text_b,
                "continuation": cont_name,
                "tv_full": float(tv_full),
                "tv_11bin": float(tv_11bin),
            })

        if n % 20 == 0:
            print(f"  NL: {n} readouts, {time.time()-t0:.0f}s", flush=True)

    return observations, n


def run_code_comparison(adapter):
    observations = []
    n = 0
    t0 = time.time()

    for depth, var, val in CODE_CELLS:
        tmpl = CODE_TEMPLATE
        prefix = build_prefix(tmpl, var=var, outer_val=val)
        query = QUERY_TEMPLATE.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix)

        for surf_a_tmpl, surf_b_tmpl, pair_name in CODE_PAIRS:
            surf_a = surf_a_tmpl.replace("{var}", var)
            surf_b = surf_b_tmpl.replace("{var}", var)

            for cont_name, cont_tmpl in CODE_CONTINUATIONS.items():
                cont = cont_tmpl.replace("{var}", var)
                suffix_a = surf_a + cont + query
                suffix_b = surf_b + cont + query

                full_a, bin_a = adapter.get_dists_from_state(prefix_state, suffix_a)
                full_b, bin_b = adapter.get_dists_from_state(prefix_state, suffix_b)

                tv_full = tv(full_a, full_b)
                tv_11bin = tv(bin_a, bin_b)
                n += 2

                observations.append({
                    "domain": "code",
                    "pair": pair_name,
                    "var": var,
                    "val": val,
                    "continuation": cont_name,
                    "tv_full": float(tv_full),
                    "tv_11bin": float(tv_11bin),
                })

    return observations, n


def analyze(nl_obs, code_obs):
    print("\n=== NL FULL-VOCAB OMEGA ===\n")

    for pair_name in sorted(set(o["pair"] for o in nl_obs)):
        rows = [o for o in nl_obs if o["pair"] == pair_name]
        base = [o for o in rows if o["continuation"] == "none"]
        base_full = base[0]["tv_full"] if base else 0
        base_11bin = base[0]["tv_11bin"] if base else 0
        print(f"Pair: {pair_name}")
        print(f"  Baseline: full={base_full:.6f}  11bin={base_11bin:.6f}  ratio={base_full/max(base_11bin,1e-10):.0f}x")

        for o in sorted(rows, key=lambda x: -x["tv_full"]):
            if o["continuation"] == "none":
                continue
            ratio = o["tv_full"] / base_full if base_full > 0.001 else 0
            print(f"  {o['continuation']:15s}: full={o['tv_full']:.6f}  11bin={o['tv_11bin']:.6f}  ratio={ratio:.2f}x")
        print()

    print("=== CODE COMPARISON (full vs 11-bin) ===\n")
    for pair_name in sorted(set(o["pair"] for o in code_obs)):
        rows = [o for o in code_obs if o["pair"] == pair_name]
        print(f"Pair: {pair_name}")
        for cont in sorted(set(o["continuation"] for o in rows)):
            crows = [o for o in rows if o["continuation"] == cont]
            mean_full = np.mean([o["tv_full"] for o in crows])
            mean_11bin = np.mean([o["tv_11bin"] for o in crows])
            ratio = mean_full / max(mean_11bin, 1e-10)
            print(f"  {cont:15s}: full={mean_full:.6f}  11bin={mean_11bin:.6f}  ratio={ratio:.2f}x")
        print()

    nl_baselines_full = {o["pair"]: o["tv_full"] for o in nl_obs if o["continuation"] == "none"}
    nl_eligible_005 = {p: v for p, v in nl_baselines_full.items() if v <= 0.05}
    nl_eligible_010 = {p: v for p, v in nl_baselines_full.items() if v <= 0.10}

    return {
        "nl_baselines_full": nl_baselines_full,
        "nl_eligible_tau005": list(nl_eligible_005.keys()),
        "nl_eligible_tau010": list(nl_eligible_010.keys()),
        "n_nl_obs": len(nl_obs),
        "n_code_obs": len(code_obs),
    }


if __name__ == "__main__":
    out_dir = Path("experiments/results/omega_nl_fullvocab")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Full-Vocab NL Omega Experiment ===\n")
    adapter = FullVocabAdapter(MODEL_ID)

    nl_obs, nl_n = run_nl_experiment(adapter)
    code_obs, code_n = run_code_comparison(adapter)

    total_n = nl_n + code_n
    print(f"\nTotal readouts: {total_n}")

    all_obs = nl_obs + code_obs
    with open(out_dir / "observations.json", "w") as f:
        json.dump(all_obs, f, indent=2)

    summary = analyze(nl_obs, code_obs)
    summary["n_readouts"] = total_n

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    del adapter
    print(f"\nResults saved to {out_dir}")
