"""Omega Referent/Name/Query Crossover (Experiment #51).

Fable R3 Rank 2: does amplification follow the referent being queried,
the surface token being repeated, or any extra context?

Design: Each history contains TWO variables. Continuations reference one
or the other. Queries ask about one or the other. The key contrast is:
does a continuation mentioning variable X amplify the X-query more than
the Y-query? If yes: referent-specific access. If equal: general context
sensitivity.

16 pair instances x 2 sides x 6 arms x 2 queries = 384 readouts.

Arms:
  1. empty (baseline)
  2. target_var_mention (mention the queried variable)
  3. other_var_mention (mention the OTHER variable)
  4. neutral_novar (# OK.\n - no variable mention)
  5. nonvar_expr (_ = 42\n)
  6. both_var_mention (mention both variables)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter
from omega import tv


MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
}

DUAL_VAR_TEMPLATE = (
    "{var1} = {val1}\n{var2} = {val2}\n"
    "def f():\n    {var1} = 99\n    {var2} = 88\n"
    "    def g():\n        {var1} = 999\n        {var2} = 888\n"
    "        return {var1} + {var2}\n"
    "    g()\n    return {var1}\n"
)

QUERY_TEMPLATES = {
    "query_var1": "f()\nprint({var1})  # Output: ",
    "query_var2": "f()\nprint({var2})  # Output: ",
}

SURFACE_PAIRS = [
    ("# {var1} is unchanged.\n", "# {var1} stays the same.\n", "assert_var1"),
    ("# State unchanged.\n", "# {var1} stays the same.\n", "assert_mixed"),
]

CELLS = [
    ("x", "y", 1, 2), ("x", "y", 5, 6),
    ("x", "z", 1, 3), ("x", "z", 5, 7),
    ("y", "z", 2, 3), ("y", "z", 6, 7),
    ("z", "x", 3, 1), ("z", "x", 7, 5),
]


def make_arms(var1, var2):
    return {
        "empty": "",
        "target_mention": f"# Check {var1}.\n",
        "other_mention": f"# Check {var2}.\n",
        "neutral_novar": "# OK.\n",
        "nonvar_expr": "_ = 42\n",
        "both_mention": f"# Check {var1} and {var2}.\n",
    }


def run_referent_crossover():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_id = MODEL_CFG["model_id"]
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map="cpu")
    model.eval()

    observations = []
    t0 = time.time()
    n_readouts = 0

    def get_dist(text):
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1).numpy()
        return probs

    for var1, var2, val1, val2 in CELLS:
        prefix = DUAL_VAR_TEMPLATE.format(
            var1=var1, var2=var2, val1=val1, val2=val2)

        for surf_a_tmpl, surf_b_tmpl, pair_name in SURFACE_PAIRS:
            surf_a = surf_a_tmpl.format(var1=var1, var2=var2)
            surf_b = surf_b_tmpl.format(var1=var1, var2=var2)

            arms = make_arms(var1, var2)

            for arm_name, arm_text in arms.items():
                for query_name, query_tmpl in QUERY_TEMPLATES.items():
                    query = query_tmpl.format(var1=var1, var2=var2)

                    text_a = prefix + surf_a + arm_text + query
                    text_b = prefix + surf_b + arm_text + query

                    dist_a = get_dist(text_a)
                    dist_b = get_dist(text_b)
                    d = tv(dist_a, dist_b)
                    n_readouts += 2

                    queried_var = var1 if "var1" in query_name else var2
                    arm_mentions_target = (
                        queried_var in arm_text and arm_name != "neutral_novar"
                    )

                    observations.append({
                        "var1": var1, "var2": var2,
                        "val1": val1, "val2": val2,
                        "pair": pair_name,
                        "arm": arm_name,
                        "query": query_name,
                        "queried_var": queried_var,
                        "arm_mentions_target": arm_mentions_target,
                        "tv": float(d),
                    })

                    if n_readouts % 48 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n_readouts}/384 readouts, {elapsed:.0f}s",
                              flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done: {n_readouts} readouts in {elapsed:.1f}s")
    del model
    return observations, elapsed


def analyze(observations):
    print("\n=== REFERENT CROSSOVER ANALYSIS ===\n")

    for pair_name in ["assert_var1", "assert_mixed"]:
        pair_obs = [o for o in observations if o["pair"] == pair_name]
        if not pair_obs:
            continue
        print(f"\n--- {pair_name} ---")

        bl_var1 = [o["tv"] for o in pair_obs
                    if o["arm"] == "empty" and o["query"] == "query_var1"]
        bl_var2 = [o["tv"] for o in pair_obs
                    if o["arm"] == "empty" and o["query"] == "query_var2"]
        if bl_var1:
            print(f"  Baseline query_var1: mean={np.mean(bl_var1):.6f}")
        if bl_var2:
            print(f"  Baseline query_var2: mean={np.mean(bl_var2):.6f}")

        for arm_name in ["target_mention", "other_mention", "neutral_novar",
                          "nonvar_expr", "both_mention"]:
            q1 = [o["tv"] for o in pair_obs
                  if o["arm"] == arm_name and o["query"] == "query_var1"]
            q2 = [o["tv"] for o in pair_obs
                  if o["arm"] == arm_name and o["query"] == "query_var2"]
            if q1 and q2:
                print(f"  {arm_name:20s}: q1={np.mean(q1):.6f} "
                      f"q2={np.mean(q2):.6f} "
                      f"ratio={np.mean(q1)/np.mean(q2):.2f}")

    print("\n=== KEY QUESTION: Is amplification target-specific? ===")
    target_on_target = [o["tv"] for o in observations
                        if o["arm"] == "target_mention"
                        and o["query"] == "query_var1"]
    target_on_other = [o["tv"] for o in observations
                       if o["arm"] == "target_mention"
                       and o["query"] == "query_var2"]
    other_on_target = [o["tv"] for o in observations
                       if o["arm"] == "other_mention"
                       and o["query"] == "query_var1"]
    other_on_other = [o["tv"] for o in observations
                      if o["arm"] == "other_mention"
                      and o["query"] == "query_var2"]

    if all([target_on_target, target_on_other, other_on_target, other_on_other]):
        tt = np.mean(target_on_target)
        to_ = np.mean(target_on_other)
        ot = np.mean(other_on_target)
        oo_ = np.mean(other_on_other)
        specificity = (tt - to_) - (ot - oo_)
        print(f"  Mention target, query target:  {tt:.6f}")
        print(f"  Mention target, query other:   {to_:.6f}")
        print(f"  Mention other, query target:   {ot:.6f}")
        print(f"  Mention other, query other:    {oo_:.6f}")
        print(f"  Target-specificity index:      {specificity:+.6f}")
        print(f"  ({'SPECIFIC' if specificity > 0.005 else 'GENERAL'})")

    neutral_tv = [o["tv"] for o in observations
                  if o["arm"] == "neutral_novar"]
    nonvar_tv = [o["tv"] for o in observations
                 if o["arm"] == "nonvar_expr"]
    if neutral_tv and nonvar_tv:
        print(f"\n  Neutral (no var): mean={np.mean(neutral_tv):.6f}")
        print(f"  Nonvar expr:      mean={np.mean(nonvar_tv):.6f}")
        print(f"  (Both should be similar if amplification is general)")

    return {
        "type": "referent_crossover",
        "n_observations": len(observations),
    }


if __name__ == "__main__":
    out_dir = Path("experiments/results/omega_referent_crossover")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Omega Referent/Name/Query Crossover (Exp #51) ===")
    print("Testing: referent-specific vs general context sensitivity\n")

    observations, elapsed = run_referent_crossover()

    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f, indent=2)

    summary = analyze(observations)
    summary["n_readouts"] = len(observations) * 2
    summary["elapsed_s"] = elapsed

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")
