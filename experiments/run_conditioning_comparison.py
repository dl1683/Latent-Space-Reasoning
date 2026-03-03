"""Head-to-head comparison: Pure Model vs Soft Prompt vs RNG Seed.

Tests 20 diverse questions across multiple categories and difficulties.
Saves full outputs for LLM-as-judge evaluation.

Categories:
- Simple arithmetic (easy baseline)
- Multi-step arithmetic (hard, chained operations)
- Word problems (requires comprehension + math)
- Logic / reasoning (non-numeric reasoning)
- General knowledge (factual questions)
"""

from __future__ import annotations

import json
import math
import sys
import time


def safe_print(text: str) -> None:
    """Print with ASCII fallback for Windows cp1252 compatibility."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from harness import (
    DecodeConfig,
    DecodeMode,
    auto_calibrate,
    decode_latent,
)
from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.decode.projection import make_row_orthonormal_W


@dataclass
class TestQuestion:
    qid: str
    category: str
    difficulty: str
    prompt: str
    expected: str  # Expected answer (for reference, LLM-as-judge does quality eval)


QUESTIONS: List[TestQuestion] = [
    # --- Simple Arithmetic (3) ---
    TestQuestion("arith_1", "arithmetic", "easy",
        "What is 47 + 86? Answer with just the number.", "133"),
    TestQuestion("arith_2", "arithmetic", "easy",
        "What is 15 * 12? Answer with just the number.", "180"),
    TestQuestion("arith_3", "arithmetic", "easy",
        "What is 1000 - 387? Answer with just the number.", "613"),

    # --- Multi-step Arithmetic (4) ---
    TestQuestion("multi_1", "multi_step", "medium",
        "Solve step by step:\n  a = 17 + 28\n  b = a * 3\n  c = b - 15\n  What is c? Answer with just the number.", "120"),
    TestQuestion("multi_2", "multi_step", "hard",
        "Solve step by step:\n  a = 13 * 7\n  b = a + 29\n  c = b mod 17\n  d = c * 5 + a\n  What is d? Answer with just the number.", "141"),
    TestQuestion("multi_3", "multi_step", "hard",
        "Solve step by step:\n  x = 256 / 8\n  y = x * x\n  z = y - x + 7\n  What is z? Answer with just the number.", "999"),
    TestQuestion("multi_4", "multi_step", "hard",
        "If I have 3 boxes, each containing 7 bags, and each bag has 4 marbles, "
        "how many marbles do I have total? Then subtract 15. What is the result?", "69"),

    # --- Word Problems (4) ---
    TestQuestion("word_1", "word_problem", "medium",
        "A train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours. "
        "What is the total distance traveled in miles?", "270"),
    TestQuestion("word_2", "word_problem", "medium",
        "Sarah has $120. She spends 1/3 on books and 1/4 of what remains on lunch. "
        "How much money does she have left?", "60"),
    TestQuestion("word_3", "word_problem", "hard",
        "A rectangular garden is 12 meters long and 8 meters wide. A path 1 meter wide "
        "runs around the outside. What is the area of the path in square meters?", "44"),
    TestQuestion("word_4", "word_problem", "hard",
        "Three friends split a restaurant bill. The meal costs $84, tax is 8%, and they "
        "want to leave a 20% tip on the pre-tax amount. How much does each person pay? "
        "Round to the nearest cent.", "35.84"),

    # --- Logic / Reasoning (5) ---
    TestQuestion("logic_1", "logic", "medium",
        "If all roses are flowers, and some flowers fade quickly, can we conclude that "
        "some roses fade quickly? Explain your reasoning in 2-3 sentences.", "No"),
    TestQuestion("logic_2", "logic", "medium",
        "I have a sequence: 2, 6, 18, 54, ... What are the next two numbers? "
        "Explain the pattern.", "162, 486"),
    TestQuestion("logic_3", "logic", "hard",
        "A farmer has chickens and cows. He counts 20 heads and 56 legs. "
        "How many chickens and how many cows does he have?", "12 chickens, 8 cows"),
    TestQuestion("logic_4", "logic", "hard",
        "You have 8 balls, one is slightly heavier. Using a balance scale, what is the "
        "minimum number of weighings needed to find the heavier ball? Explain your strategy.", "2"),
    TestQuestion("logic_5", "logic", "hard",
        "Three people check into a hotel room that costs $30. They each pay $10. "
        "The manager realizes the room should be $25 and gives the bellboy $5 to return. "
        "The bellboy keeps $2 and gives $1 back to each person. Now each paid $9, "
        "totaling $27. The bellboy has $2. That's $29. Where is the missing dollar? "
        "Explain clearly.", "There is no missing dollar"),

    # --- General Knowledge / Explanation (4) ---
    TestQuestion("know_1", "knowledge", "medium",
        "Explain what a prime number is and list all prime numbers between 20 and 40.", "23, 29, 31, 37"),
    TestQuestion("know_2", "knowledge", "medium",
        "What is the Fibonacci sequence? Write the first 10 numbers.", "0, 1, 1, 2, 3, 5, 8, 13, 21, 34"),
    TestQuestion("know_3", "knowledge", "hard",
        "Explain the difference between a stack and a queue data structure. "
        "Give a real-world analogy for each.", "LIFO vs FIFO"),
    TestQuestion("know_4", "knowledge", "hard",
        "What is the time complexity of binary search and why? Explain in simple terms.", "O(log n)"),
]


def run_pure_model(encoder, question: TestQuestion) -> str:
    """Run question with no conditioning at all."""
    inputs = encoder.tokenizer(question.prompt, return_tensors="pt")
    inputs = {k: v.to(encoder._device) for k, v in inputs.items()}
    with torch.no_grad():
        out = encoder.model.generate(
            **inputs, max_new_tokens=1024, temperature=0.3,
            do_sample=True, top_p=0.9, top_k=50,
            pad_token_id=encoder.tokenizer.pad_token_id,
            repetition_penalty=1.2,
        )
    resp = encoder.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()
    # Strip think tags
    if "<think>" in resp and "</think>" in resp:
        think_end = resp.find("</think>") + len("</think>")
        resp = resp[think_end:].strip()
    return resp


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Conditioning comparison")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--output", default=None, help="Override output path")
    args = parser.parse_args()

    model_name = args.model
    quantization = args.quantization
    model_short = model_name.split("/")[-1].lower().replace("-", "_")

    print("=" * 70)
    print("CONDITIONING COMPARISON: Pure Model vs Soft Prompt vs RNG Seed")
    print(f"Model: {model_name} ({quantization})")
    print(f"Questions: {len(QUESTIONS)} across {len(set(q.category for q in QUESTIONS))} categories")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    encoder = LLMEncoder(model_name=model_name, quantization=quantization)
    cal = auto_calibrate(encoder)
    d_latent = encoder.latent_dim
    embed_dim = cal["embed_dim"]
    target_rms = cal["embedding_rms"]

    # Check soft prompt compatibility early
    from harness import check_soft_prompt_compatibility
    soft_ok = check_soft_prompt_compatibility(encoder)
    print(f"Soft prompt compatible: {soft_ok}")

    # Setup soft prompt
    W = make_row_orthonormal_W(d_latent, 8 * embed_dim, seed=1234).to(encoder._device)
    torch.manual_seed(1000)
    seed_latent = encoder.encode("You calculate expressions and give numeric answers.")
    ball_radius = (1.0 / math.sqrt(0.5)) * 0.95
    target_init_norm = 0.5 * ball_radius
    lat = seed_latent.clone()
    lat_norm = lat.squeeze().norm().item()
    lat = lat * (target_init_norm / lat_norm)

    cfg_soft = DecodeConfig(
        geometry="euclidean", mode=DecodeMode.SOFT_PROMPT,
        W_soft=W, embed_dim=embed_dim, num_soft_tokens=8,
        target_rms=target_rms, curvature=0.5, max_new_tokens=1024, temperature=0.3,
    )
    cfg_rng = DecodeConfig(
        geometry="euclidean", mode=DecodeMode.RNG_SEED,
        curvature=0.5, max_new_tokens=1024, temperature=0.3,
    )

    results = []
    start = time.time()

    for i, q in enumerate(QUESTIONS):
        print(f"\n--- Q{i+1}/{len(QUESTIONS)}: [{q.category}/{q.difficulty}] {q.qid} ---")
        print(f"  Prompt: {q.prompt[:80]}...")

        entry = {
            "qid": q.qid,
            "category": q.category,
            "difficulty": q.difficulty,
            "prompt": q.prompt,
            "expected": q.expected,
        }

        # Pure model
        t0 = time.time()
        resp_pure = run_pure_model(encoder, q)
        entry["pure_model"] = resp_pure[:800]
        entry["pure_time"] = round(time.time() - t0, 1)
        safe_print(f"  PURE ({entry['pure_time']}s): {resp_pure[:120]}...")

        # Soft prompt (skip if model doesn't support inputs_embeds)
        if soft_ok:
            t0 = time.time()
            resp_soft = decode_latent(encoder, lat, q.prompt, cfg_soft)
            entry["soft_prompt"] = resp_soft[:800]
            entry["soft_time"] = round(time.time() - t0, 1)
            safe_print(f"  SOFT ({entry['soft_time']}s): {resp_soft[:120]}...")
        else:
            entry["soft_prompt"] = "[SKIPPED - inputs_embeds not supported]"
            entry["soft_time"] = 0
            print(f"  SOFT: SKIPPED")

        # RNG seed
        t0 = time.time()
        resp_rng = decode_latent(encoder, lat, q.prompt, cfg_rng)
        entry["rng_seed"] = resp_rng[:800]
        entry["rng_time"] = round(time.time() - t0, 1)
        safe_print(f"  RNG  ({entry['rng_time']}s): {resp_rng[:120]}...")

        results.append(entry)

    elapsed = time.time() - start

    # Save
    output = {
        "experiment": "conditioning_comparison",
        "model": model_name,
        "quantization": quantization,
        "soft_prompt_compatible": soft_ok,
        "calibration": cal,
        "n_questions": len(QUESTIONS),
        "elapsed_seconds": elapsed,
        "results": results,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(__file__).parent / f"conditioning_comparison_{model_short}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"DONE in {elapsed/60:.1f} min")
    print(f"Results saved to: {out_path}")
    print(f"{'=' * 70}")

    # Cleanup GPU
    del encoder
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
