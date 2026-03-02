"""
Evaluation script comparing Standard vs Grammar vs Hybrid evolution modes.

Tests 3 configurations:
1. Standard: Regular latent evolution (QD disabled)
2. Grammar: Fractal grammar evolution only
3. Hybrid: Both grammar + QD (takes best result)

Usage:
    python eval_grammar_modes.py --mode quick    # 5 questions
    python eval_grammar_modes.py --mode full     # All questions

================================================================================
CRITICAL: MANUAL EVALUATION IS THE MOST IMPORTANT METRIC
================================================================================
Statistical accuracy metrics (correct/total) are useful for quick validation,
but they are NOT sufficient for evaluating output quality.

ALWAYS perform manual review of outputs:
1. Have a human review the actual text outputs for coherence and usefulness
2. Use Codex or another AI to review outputs for quality and reasoning
3. Check that answers are not just "correct" but also well-reasoned
4. Verify outputs are sensible, coherent, and actually useful

Automated metrics can miss:
- Correct answers with flawed reasoning
- Nonsensical but pattern-matched responses
- Quality degradation that doesn't affect accuracy
- Outputs that are technically correct but unhelpful

Run manual review with: codex exec "Review these outputs for quality: [paste]"
================================================================================
"""

import sys
sys.path.insert(0, "C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning")

import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Test questions with known correct answers
# Use bare numbers/text to handle LaTeX formatting like $\boxed{0.05}$
QUESTIONS = [
    ("A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?", "0.05"),
    ("What is 15% of 80?", "12"),
    ("If Alice is taller than Bob, and Bob is taller than Carol, who is the shortest?", "Carol"),
    ("A lily pad doubles in size every day. If it takes 48 days to cover a pond, how long to cover half?", "47"),
    ("If you have 3 apples and take away 2, how many do you have?", "2"),
]


@dataclass
class ModeResult:
    answer: str
    score: float
    time: float
    mode: str


def run_mode(question: str, mode: str, generations: int = 6, chains: int = 6) -> ModeResult:
    """Run the engine with specified mode."""
    from latent_reasoning import Engine, Config

    config = Config()
    config.encoder.model = "Qwen/Qwen3-0.6B"
    config.encoder.quantization = "4bit"
    config.evolution.generations = generations
    config.evolution.chains = chains
    config.output.verbosity = 0

    # Configure mode
    if mode == "standard":
        config.qd.enabled = False
        config.grammar.enabled = False
    elif mode == "grammar":
        config.qd.enabled = False
        config.grammar.enabled = True
        config.grammar.population_size = 10
        config.grammar.num_rules = 6
        config.grammar.max_depth = 3
    elif mode == "hybrid":
        config.qd.enabled = True
        config.qd.novelty_weight = 0.3
        config.grammar.enabled = True
        config.grammar.population_size = 10
        config.grammar.num_rules = 6
        config.grammar.max_depth = 3
    else:
        raise ValueError(f"Unknown mode: {mode}")

    engine = Engine(config=config)
    start = time.time()
    result = engine.run(question)
    elapsed = time.time() - start

    return ModeResult(
        answer=result.plan or "",
        score=result.confidence,
        time=elapsed,
        mode=mode,
    )


def extract_final_answer(answer: str) -> str:
    """Extract the final answer from a response, looking for common patterns.

    Returns the extracted final answer text, or empty string if none found.
    """
    import re

    # Look for boxed answers (highest priority - this IS the answer)
    # Handle both \\boxed and \boxed (escaped and unescaped)
    boxed_patterns = [
        r'\\boxed\{\\text\{([^}]+)\}\}',  # \boxed{\text{Carol}}
        r'\\boxed\{([^}]+)\}',             # \boxed{Bob} or \boxed{12}
        r'boxed\{([^}]+)\}',               # boxed{X} without backslash
    ]

    for pattern in boxed_patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        if matches:
            return matches[-1].strip().lower()  # Last boxed is the final answer

    # Look for explicit answer statements
    answer_patterns = [
        r'(?:final )?answer[:\s]+\*{0,2}([A-Za-z0-9$., ]+?)[\s*]*[.!?\n]',
        r'(?:final )?answer[:\s]+\*{0,2}([A-Za-z0-9$., ]+?)$',
    ]

    for pattern in answer_patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        if matches:
            return matches[-1].strip().lower()

    # No clear final answer marker found
    return ""


def check_correct(answer: str, correct: str) -> bool:
    """Check if the final answer contains the correct response.

    IMPORTANT: This function checks the FINAL ANSWER only, not the entire text.
    An output that mentions the correct answer in reasoning but gives a wrong
    final answer will be marked INCORRECT.

    NOTE: This is still a heuristic. Manual review remains essential
    because automated checking cannot verify reasoning quality.
    """
    if not correct:
        return False

    # Extract what appears to be the final answer
    final = extract_final_answer(answer)

    # If we found a clear final answer, check it
    if final:
        for c in correct.split('/'):
            c_clean = c.lower().strip()
            if c_clean in final:
                return True
        # We found a final answer but it doesn't contain the correct answer
        # This means the model gave a WRONG answer
        return False

    # No clear final answer marker - fall back to checking full text
    # But be conservative: only mark correct if answer appears prominently
    import re
    answer_lower = answer.lower()
    for c in correct.split('/'):
        c_clean = c.lower().strip()
        # Must appear in a clear conclusion context
        patterns = [
            rf'(?:the )?(?:answer|result) is[:\s]*[^.]*\b{re.escape(c_clean)}\b',
            rf'(?:therefore|thus|hence|so)[,\s]+[^.]*\b{re.escape(c_clean)}\b[^.]*(?:is|are)[^.]*(?:shortest|tallest|answer)',
        ]
        for pattern in patterns:
            if re.search(pattern, answer_lower):
                return True

    return False


def run_evaluation(questions: list, output_file: str):
    """Run full evaluation comparing all modes."""
    modes = ["standard", "grammar", "hybrid"]
    results = []

    print("=" * 80)
    print("GRAMMAR MODE EVALUATION")
    print(f"Questions: {len(questions)} | Modes: {', '.join(modes)}")
    print("=" * 80)

    stats = {mode: {"correct": 0, "total": 0, "total_time": 0.0} for mode in modes}

    for i, (question, correct) in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {question[:60]}...")
        print("-" * 70)

        question_results = {"question": question, "correct": correct}

        for mode in modes:
            print(f"  {mode.upper():10s}: ", end="", flush=True)
            try:
                result = run_mode(question, mode)
                is_correct = check_correct(result.answer, correct)

                status = "CORRECT" if is_correct else "WRONG"
                print(f"Done ({result.time:.1f}s, score={result.score:.3f}) [{status}]")

                question_results[mode] = {
                    "answer": result.answer[:200],
                    "score": result.score,
                    "time": result.time,
                    "correct": is_correct,
                }

                stats[mode]["total"] += 1
                stats[mode]["total_time"] += result.time
                if is_correct:
                    stats[mode]["correct"] += 1

            except Exception as e:
                print(f"ERROR: {e}")
                question_results[mode] = {"error": str(e)}

        results.append(question_results)

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print("\nAccuracy by Mode:")
    for mode in modes:
        s = stats[mode]
        if s["total"] > 0:
            acc = 100 * s["correct"] / s["total"]
            avg_time = s["total_time"] / s["total"]
            print(f"  {mode.upper():10s}: {s['correct']}/{s['total']} ({acc:.1f}%) | Avg time: {avg_time:.1f}s")

    # Find best mode
    best_mode = max(modes, key=lambda m: stats[m]["correct"])
    print(f"\nBest Mode: {best_mode.upper()}")

    # Save results
    out_path = Path(__file__).parent / output_file
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'stats': stats,
            'results': results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate grammar evolution modes")
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--output', type=str, default='eval_grammar_modes_results.json')
    args = parser.parse_args()

    if args.mode == 'quick':
        questions = QUESTIONS[:3]
    else:
        questions = QUESTIONS

    run_evaluation(questions, args.output)


if __name__ == "__main__":
    main()
