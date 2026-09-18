"""Omega Natural Language: Does the omega effect exist outside code?

Tests whether natural language surface pairs (semantically equivalent phrasings)
show amplification under continuations, analogous to the code omega effect.

Key difference from code experiments: uses FULL softmax TV distance instead of
11-bin digit readout, since NL next-tokens aren't digits.

Phase 1: Surface pair discovery — find NL pairs with baseline TV ≤ 0.05
Phase 2: Continuation test — do appended phrases amplify the difference?
"""

import copy
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path


class NLAdapter:
    """Adapter for NL omega — uses full softmax TV instead of 11-bin."""

    def __init__(self, model_id="Qwen/Qwen3-1.7B"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32)
        self.mdl.eval()
        self.vocab_size = self.mdl.config.vocab_size
        self.call_count = 0

    def get_full_dist(self, text):
        ids = self.tok.encode(text, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            out = self.mdl(ids)
        self.call_count += 1
        logits = out.logits[0, -1, :self.vocab_size]
        probs = torch.softmax(logits, dim=0).numpy().astype(np.float64)
        return probs

    def get_state_after_prefix(self, text):
        ids = self.tok.encode(text, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            out = self.mdl(ids, use_cache=True)
        self.call_count += 1
        return copy.deepcopy(out.past_key_values)

    def get_full_dist_from_state(self, state, suffix_text):
        ids = self.tok.encode(suffix_text, add_special_tokens=False, return_tensors="pt")
        st = copy.deepcopy(state)
        with torch.no_grad():
            out = self.mdl(ids, past_key_values=st, use_cache=True)
        self.call_count += 1
        logits = out.logits[0, -1, :self.vocab_size]
        probs = torch.softmax(logits, dim=0).numpy().astype(np.float64)
        return probs


def tv(a, b):
    return float(0.5 * np.abs(a - b).sum())


SURFACE_PAIRS = [
    # Original set (varying semantic distance)
    ("The capital of France is", "France's capital is"),
    ("The dog chased the", "The canine pursued the"),
    ("She walked to the", "She went to the"),
    ("The temperature was very", "The temp was extremely"),
    ("He said that he", "He mentioned that he"),
    ("The old man sat", "An elderly man sat"),
    ("It was raining", "Rain was falling"),
    ("The car drove down the", "The vehicle traveled down the"),
    ("She opened the door and", "She opened the door then"),
    ("The children were playing in the", "The kids were playing in the"),
    ("He picked up the phone and", "He grabbed the phone and"),
    ("The sun was setting over the", "The sun set over the"),
    ("They decided to go to the", "They chose to go to the"),
    ("The book was on the", "The book sat on the"),
    ("She smiled and said", "She grinned and said"),
    ("The river flowed through the", "The river ran through the"),
    ("He was tired after a long", "He was exhausted after a long"),
    ("The house was built in", "The home was constructed in"),
    ("She looked at the sky and", "She gazed at the sky and"),
    ("The food was very", "The meal was quite"),
    # Minimal-difference pairs (single word swap, same POS)
    ("The cat sat on the", "The cat lay on the"),
    ("He ran to the", "He walked to the"),
    ("She was happy about the", "She was glad about the"),
    ("They went to the store and", "They went to the shop and"),
    ("The big red", "The large red"),
    ("I think that the", "I believe that the"),
    ("He quickly ran to the", "He quickly walked to the"),
    ("The professor said that", "The teacher said that"),
    ("It was a beautiful", "It was a lovely"),
    ("She had a small", "She had a little"),
    # Near-identical pairs (punctuation / article / whitespace variants)
    ("The man said hello.", "The man said hello!"),
    ("It is a good day.", "It is a good day,"),
    ("He is a teacher.", "He is a teacher,"),
    ("The results were clear.", "The results were clear:"),
    ("I saw a bird in the", "I saw the bird in the"),
]

CONTINUATIONS = {
    "none": "",
    "neutral_period": ". ",
    "neutral_comma": ", and ",
    "topic_shift": " However, the weather ",
    "elaboration": " which was interesting because ",
    "negation": " but not ",
    "question": "? Well, ",
    "emphasis": "! Indeed, ",
    "contrast": " although some would say ",
    "temporal": " and then later ",
}


def run_phase1(adapter):
    """Phase 1: Discover NL surface pairs with small baseline TV."""
    print("\n=== Phase 1: Surface Pair Discovery ===")
    print(f"Testing {len(SURFACE_PAIRS)} candidate pairs...")

    results = []
    for i, (a, b) in enumerate(SURFACE_PAIRS):
        dist_a = adapter.get_full_dist(a)
        dist_b = adapter.get_full_dist(b)
        d = tv(dist_a, dist_b)

        top5_a = np.argsort(dist_a)[-5:][::-1]
        top5_b = np.argsort(dist_b)[-5:][::-1]
        top_a_tokens = [adapter.tok.decode([t]) for t in top5_a]
        top_b_tokens = [adapter.tok.decode([t]) for t in top5_b]

        status = "ELIGIBLE" if d <= 0.05 else ("MARGINAL" if d <= 0.15 else "DISTANT")
        print(f"  [{status:>8s}] TV={d:.4f}  {a[:40]:<40s} vs {b[:40]}")
        if d <= 0.15:
            print(f"           top-5 A: {top_a_tokens}  B: {top_b_tokens}")

        results.append({
            "pair_idx": i,
            "a": a, "b": b,
            "baseline_tv": d,
            "status": status,
            "top5_a": top_a_tokens,
            "top5_b": top_b_tokens,
        })

    eligible = [r for r in results if r["baseline_tv"] <= 0.05]
    marginal = [r for r in results if 0.05 < r["baseline_tv"] <= 0.15]
    print(f"\nEligible (TV <= 0.05): {len(eligible)}")
    print(f"Marginal (0.05 < TV <= 0.15): {len(marginal)}")
    print(f"Distant (TV > 0.15): {len(results) - len(eligible) - len(marginal)}")

    return results


def run_phase2(adapter, phase1_results, tau=0.05):
    """Phase 2: Test continuations on eligible pairs."""
    eligible = [r for r in phase1_results if r["baseline_tv"] <= tau]

    if not eligible:
        print(f"\n=== Phase 2: No eligible pairs at tau={tau} ===")
        print("Trying with marginal pairs (tau=0.15) instead...")
        tau = 0.15
        eligible = [r for r in phase1_results if r["baseline_tv"] <= tau]
        if not eligible:
            print("No pairs even at tau=0.15. NL omega cannot be tested with these pairs.")
            return []

    print(f"\n=== Phase 2: Continuation Test (tau={tau}, {len(eligible)} pairs) ===")

    observations = []
    for pair_info in eligible:
        a_text = pair_info["a"]
        b_text = pair_info["b"]
        pair_idx = pair_info["pair_idx"]
        baseline_tv = pair_info["baseline_tv"]

        print(f"\n  Pair {pair_idx}: \"{a_text[:35]}\" vs \"{b_text[:35]}\" (baseline={baseline_tv:.4f})")

        for cont_name, cont_text in CONTINUATIONS.items():
            full_a = a_text + cont_text
            full_b = b_text + cont_text

            dist_a = adapter.get_full_dist(full_a)
            dist_b = adapter.get_full_dist(full_b)
            d = tv(dist_a, dist_b)

            amplified = d > tau
            ratio = d / max(baseline_tv, 1e-8)
            marker = "AMPLIFY" if amplified else "preserve"

            if cont_name != "none":
                print(f"    {cont_name:<18s} TV={d:.4f} ({ratio:.1f}x baseline) [{marker}]")

            observations.append({
                "pair_idx": pair_idx,
                "a": a_text, "b": b_text,
                "continuation": cont_name,
                "cont_text": cont_text,
                "full_a": full_a, "full_b": full_b,
                "post_tv": d,
                "baseline_tv": baseline_tv,
                "ratio": ratio,
                "amplified": amplified,
                "tau": tau,
            })

    return observations


def analyze(phase1_results, phase2_observations, tau):
    """Summarize findings."""
    print("\n" + "=" * 60)
    print("=== Natural Language Omega: Summary ===")
    print("=" * 60)

    eligible = [r for r in phase1_results if r["baseline_tv"] <= tau]
    print(f"\nSurface pairs tested: {len(SURFACE_PAIRS)}")
    print(f"Eligible at tau={tau}: {len(eligible)}")

    if not phase2_observations:
        print("No observations to analyze.")
        return {}

    actual_tau = phase2_observations[0]["tau"]

    conts = sorted(set(o["continuation"] for o in phase2_observations if o["continuation"] != "none"))
    print(f"\nContinuations tested: {len(conts)}")
    print(f"\nPer-continuation omega (mean TV, max TV, amplification rate):")

    summary = {}
    for cont in conts:
        obs = [o for o in phase2_observations if o["continuation"] == cont]
        tvs = [o["post_tv"] for o in obs]
        ratios = [o["ratio"] for o in obs]
        n_amp = sum(1 for o in obs if o["amplified"])

        mean_tv = float(np.mean(tvs))
        max_tv = float(np.max(tvs))
        mean_ratio = float(np.mean(ratios))
        amp_rate = n_amp / len(obs)

        status = "AMPLIFY" if amp_rate > 0.5 else "preserve"
        print(f"  {cont:<18s} mean={mean_tv:.4f} max={max_tv:.4f} "
              f"ratio={mean_ratio:.1f}x amp_rate={amp_rate:.0%} [{status}]")

        summary[cont] = {
            "mean_tv": mean_tv,
            "max_tv": max_tv,
            "mean_ratio": mean_ratio,
            "amp_rate": amp_rate,
            "n": len(obs),
        }

    baselines = [o["baseline_tv"] for o in phase2_observations if o["continuation"] == "none"]
    if baselines:
        print(f"\nBaseline TV across tested pairs: mean={np.mean(baselines):.4f}, "
              f"max={np.max(baselines):.4f}")

    all_post = [o for o in phase2_observations if o["continuation"] != "none"]
    n_total_amp = sum(1 for o in all_post if o["amplified"])
    print(f"\nOverall amplification rate: {n_total_amp}/{len(all_post)} "
          f"({n_total_amp/max(len(all_post),1):.0%})")

    code_identity_mean = 0.050
    best_nl = max(summary.values(), key=lambda x: x["mean_tv"]) if summary else None
    if best_nl:
        print(f"\nComparison to code omega:")
        print(f"  Code identity mean (Qwen3-1.7B): {code_identity_mean:.4f}")
        print(f"  Best NL continuation mean:        {best_nl['mean_tv']:.4f}")
        print(f"  Ratio: {best_nl['mean_tv']/code_identity_mean:.1f}x")

    return summary


def run():
    print("=== Omega Natural Language Proof of Concept ===")
    print("Model: Qwen/Qwen3-1.7B | Readout: full softmax TV")

    result_dir = Path("experiments/results/omega_natural_lang")
    result_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    adapter = NLAdapter("Qwen/Qwen3-1.7B")

    phase1 = run_phase1(adapter)
    with open(result_dir / "phase1_pairs.json", "w") as f:
        json.dump(phase1, f, indent=2)

    tau_options = [0.05, 0.10, 0.15]
    best_tau = None
    for tau in tau_options:
        n_eligible = sum(1 for r in phase1 if r["baseline_tv"] <= tau)
        if n_eligible >= 3:
            best_tau = tau
            break

    if best_tau is None:
        best_tau = 0.15
        print(f"\nWarning: fewer than 3 pairs at any threshold. Using tau={best_tau}")

    phase2 = run_phase2(adapter, phase1, tau=best_tau)
    if phase2:
        with open(result_dir / "phase2_continuations.json", "w") as f:
            json.dump(phase2, f, indent=2)

    summary = analyze(phase1, phase2, best_tau)

    elapsed = time.time() - t0
    n_readouts = adapter.call_count
    print(f"\nTotal: {n_readouts} readouts in {elapsed:.1f}s ({n_readouts/elapsed:.1f}/s)")
    print(f"Saved to {result_dir}/")

    with open(result_dir / "summary.json", "w") as f:
        json.dump({
            "n_readouts": n_readouts,
            "elapsed_s": elapsed,
            "n_pairs_tested": len(SURFACE_PAIRS),
            "tau_used": best_tau,
            "n_eligible": sum(1 for r in phase1 if r["baseline_tv"] <= best_tau),
            "continuation_summary": summary,
        }, f, indent=2)

    return phase1, phase2, summary


if __name__ == "__main__":
    run()
