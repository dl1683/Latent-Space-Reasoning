"""Omega NL Pair Discovery v2: Systematic search for NL pairs that collapse at tau=0.05.

Insight from exp #32: collapsible NL pairs share HIGH-CONSTRAINT syntactic contexts
where the next token is strongly determined by position regardless of synonym swap.
The three that collapsed (tau=0.10):
  - "He said that he" vs "He mentioned that he" (TV=0.068)
  - "He was tired after a long" vs "He was exhausted after a long" (TV=0.065)
  - "The house was built in" vs "The home was constructed in" (TV=0.090)

Strategy: target high-constraint positions (after "that", temporal, quantifiers,
modals, embedded clauses, strong collocations).
"""

import json
import time
import numpy as np
import torch
import copy
from pathlib import Path


class NLAdapter:
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
        return torch.softmax(logits, dim=0).numpy().astype(np.float64)


def tv(a, b):
    return float(0.5 * np.abs(a - b).sum())


# 60 NEW candidate pairs organized by constraint type
PAIRS_V2 = [
    # --- THAT-clause embeddings (verb swap before "that the") ---
    ("He explained that the", "He stated that the"),
    ("She believed that the", "She thought that the"),
    ("They announced that the", "They declared that the"),
    ("He argued that the", "He claimed that the"),
    ("She realized that the", "She noticed that the"),
    ("He reported that the", "He announced that the"),
    ("She discovered that the", "She found that the"),
    ("They confirmed that the", "They verified that the"),

    # --- Preposition + determiner (arrival/location verbs) ---
    ("We arrived at the", "We got to the"),
    ("He returned to the", "He came back to the"),
    ("She entered the room and", "She walked into the room and"),
    ("They gathered at the", "They met at the"),
    ("He sat down at the", "He took a seat at the"),

    # --- Modal + infinitive (future-directed) ---
    ("She would like to", "She wants to"),
    ("He was going to", "He planned to"),
    ("They needed to", "They had to"),
    ("She was supposed to", "She was meant to"),
    ("He was about to", "He was going to"),

    # --- Temporal constraints ---
    ("Last Monday he", "On Monday he"),
    ("Yesterday morning she", "That morning she"),
    ("Every day they", "Each day they"),
    ("After dinner they", "Following dinner they"),
    ("Before sunrise the", "Prior to sunrise the"),

    # --- Quantifier + noun ---
    ("Most of the", "The majority of the"),
    ("All of the", "Every one of the"),
    ("Some of the", "A few of the"),
    ("None of the", "Not one of the"),
    ("Many of the", "A lot of the"),

    # --- Copula + adjective (high constraint on following NP) ---
    ("The meeting was very", "The meeting was extremely"),
    ("The project was quite", "The project was rather"),
    ("The situation was incredibly", "The situation was remarkably"),
    ("His performance was very", "His performance was extremely"),
    ("The road was very", "The road was extremely"),

    # --- Comparative structures ---
    ("It was much better than", "It was far superior to"),
    ("She was much older than", "She was far older than"),
    ("The problem was more complex than", "The problem was more difficult than"),

    # --- Passive constructions (agent suppressed = more constrained) ---
    ("The document was signed by", "The document was approved by"),
    ("The building was designed by", "The building was created by"),
    ("The law was passed by", "The law was enacted by"),
    ("The book was written by", "The book was authored by"),

    # --- Strong collocations ---
    ("Once upon a", "Long ago in a"),
    ("As a matter of", "As a point of"),
    ("For the sake of", "For the purpose of"),
    ("In the middle of the", "In the center of the"),
    ("At the end of the", "At the close of the"),

    # --- Embedded questions ---
    ("I wonder if the", "I wonder whether the"),
    ("She asked if the", "She asked whether the"),
    ("He wanted to know if the", "He wanted to know whether the"),

    # --- Reporting speech (very constrained after "said") ---
    ("He said he would", "He told them he would"),
    ("She said she had", "She mentioned she had"),
    ("They said they were", "They indicated they were"),

    # --- Verb + PP with strong selectional restriction ---
    ("He put the book on the", "He placed the book on the"),
    ("She threw the ball to the", "She tossed the ball to the"),
    ("He poured the water into the", "He emptied the water into the"),
    ("She hung the coat on the", "She draped the coat on the"),
    ("He drove the car to the", "He took the car to the"),

    # --- Existential constructions ---
    ("There was a large", "There stood a large"),
    ("There were many", "There existed many"),
    ("There was nothing but", "There was only"),
]

CONTINUATIONS = {
    "none": "",
    "neutral_period": ". ",
    "neutral_comma": ", and ",
    "negation": " but not ",
    "temporal": " and then later ",
    "elaboration": " which was interesting because ",
    "contrast": " although some would say ",
}


def run():
    print("=== Omega NL Pair Discovery v2 ===")
    print(f"Testing {len(PAIRS_V2)} high-constraint candidate pairs")
    print("Model: Qwen/Qwen3-1.7B | Readout: full softmax TV\n")

    result_dir = Path("experiments/results/omega_nl_v2")
    result_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    adapter = NLAdapter("Qwen/Qwen3-1.7B")

    # Phase 1: Screen all pairs
    print("=== Phase 1: Screening ===")
    phase1 = []
    for i, (a, b) in enumerate(PAIRS_V2):
        dist_a = adapter.get_full_dist(a)
        dist_b = adapter.get_full_dist(b)
        d = tv(dist_a, dist_b)

        top5_a = [adapter.tok.decode([t]) for t in np.argsort(dist_a)[-5:][::-1]]
        top5_b = [adapter.tok.decode([t]) for t in np.argsort(dist_b)[-5:][::-1]]

        if d <= 0.05:
            status = "ELIGIBLE_005"
        elif d <= 0.10:
            status = "ELIGIBLE_010"
        elif d <= 0.15:
            status = "MARGINAL"
        else:
            status = "DISTANT"

        print(f"  [{status:>12s}] TV={d:.4f}  \"{a[:35]}\" vs \"{b[:35]}\"")
        if d <= 0.15:
            print(f"               top5_a: {top5_a}  top5_b: {top5_b}")

        phase1.append({
            "pair_idx": i, "a": a, "b": b,
            "baseline_tv": d, "status": status,
            "top5_a": top5_a, "top5_b": top5_b,
        })

    n_005 = sum(1 for r in phase1 if r["baseline_tv"] <= 0.05)
    n_010 = sum(1 for r in phase1 if r["baseline_tv"] <= 0.10)
    n_015 = sum(1 for r in phase1 if r["baseline_tv"] <= 0.15)
    print(f"\n--- Screening results ---")
    print(f"  tau=0.05: {n_005} eligible")
    print(f"  tau=0.10: {n_010} eligible")
    print(f"  tau=0.15: {n_015} eligible (incl marginal)")
    print(f"  distant:  {len(phase1) - n_015}")

    # Phase 2: Test continuations on eligible pairs
    if n_005 >= 1:
        test_tau = 0.05
    elif n_010 >= 2:
        test_tau = 0.10
    else:
        test_tau = 0.15

    eligible = [r for r in phase1 if r["baseline_tv"] <= test_tau]
    print(f"\n=== Phase 2: Continuation test (tau={test_tau}, {len(eligible)} pairs) ===")

    phase2 = []
    for pair_info in eligible:
        a_text = pair_info["a"]
        b_text = pair_info["b"]
        baseline = pair_info["baseline_tv"]
        print(f"\n  Pair: \"{a_text[:35]}\" vs \"{b_text[:35]}\" (baseline={baseline:.4f})")

        for cont_name, cont_text in CONTINUATIONS.items():
            full_a = a_text + cont_text
            full_b = b_text + cont_text
            dist_a = adapter.get_full_dist(full_a)
            dist_b = adapter.get_full_dist(full_b)
            d = tv(dist_a, dist_b)
            ratio = d / max(baseline, 1e-8)
            amplified = d > test_tau

            if cont_name != "none":
                marker = "AMPLIFY" if amplified else "preserve"
                print(f"    {cont_name:<18s} TV={d:.4f} ({ratio:.1f}x) [{marker}]")

            phase2.append({
                "pair_idx": pair_info["pair_idx"],
                "a": a_text, "b": b_text,
                "continuation": cont_name, "cont_text": cont_text,
                "post_tv": d, "baseline_tv": baseline,
                "ratio": ratio, "amplified": amplified, "tau": test_tau,
            })

    elapsed = time.time() - t0
    n_readouts = adapter.call_count

    # Summary
    print(f"\n{'='*60}")
    print(f"=== Summary ===")
    print(f"Pairs tested: {len(PAIRS_V2)}")
    print(f"Eligible at tau=0.05: {n_005}")
    print(f"Eligible at tau=0.10: {n_010}")
    print(f"Phase 2 tau used: {test_tau}")
    print(f"Total readouts: {n_readouts} in {elapsed:.1f}s")

    if phase2:
        conts = sorted(set(o["continuation"] for o in phase2 if o["continuation"] != "none"))
        print(f"\nPer-continuation results:")
        cont_summary = {}
        for cont in conts:
            obs = [o for o in phase2 if o["continuation"] == cont]
            tvs = [o["post_tv"] for o in obs]
            ratios = [o["ratio"] for o in obs]
            n_amp = sum(1 for o in obs if o["amplified"])
            cont_summary[cont] = {
                "mean_tv": float(np.mean(tvs)),
                "max_tv": float(np.max(tvs)),
                "mean_ratio": float(np.mean(ratios)),
                "amp_rate": n_amp / len(obs) if obs else 0,
                "n": len(obs),
            }
            print(f"  {cont:<18s} mean={np.mean(tvs):.4f} ratio={np.mean(ratios):.1f}x "
                  f"amp={n_amp}/{len(obs)}")

    # Save
    with open(result_dir / "phase1_screening.json", "w") as f:
        json.dump(phase1, f, indent=2)
    if phase2:
        with open(result_dir / "phase2_continuations.json", "w") as f:
            json.dump(phase2, f, indent=2)
    with open(result_dir / "summary.json", "w") as f:
        json.dump({
            "n_readouts": n_readouts,
            "elapsed_s": elapsed,
            "n_pairs_v2": len(PAIRS_V2),
            "n_eligible_005": n_005,
            "n_eligible_010": n_010,
            "n_eligible_015": n_015,
            "phase2_tau": test_tau,
            "continuation_summary": cont_summary if phase2 else {},
            "best_pairs_005": [
                {"a": r["a"], "b": r["b"], "tv": r["baseline_tv"]}
                for r in sorted(phase1, key=lambda x: x["baseline_tv"])
                if r["baseline_tv"] <= 0.05
            ],
            "best_pairs_010": [
                {"a": r["a"], "b": r["b"], "tv": r["baseline_tv"]}
                for r in sorted(phase1, key=lambda x: x["baseline_tv"])
                if r["baseline_tv"] <= 0.10
            ],
        }, f, indent=2)

    print(f"\nSaved to {result_dir}/")
    return phase1, phase2


if __name__ == "__main__":
    run()
