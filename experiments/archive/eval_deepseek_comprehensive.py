"""
Comprehensive Evaluation with DeepSeek-R1-Distill Model

Tests Standard vs Grammar vs Hybrid modes on a diverse set of 25+ questions
including easy, medium, and hard difficulty levels.

Uses: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (strong reasoning model)

================================================================================
CRITICAL: MANUAL EVALUATION IS THE MOST IMPORTANT METRIC
================================================================================
After running, review outputs with Codex or manually to verify quality.
Automated metrics can miss reasoning errors that produce correct-looking answers.
================================================================================
"""

import sys
sys.path.insert(0, "C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning")

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import json
import argparse
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Comprehensive question bank with difficulty levels
QUESTIONS = {
    # ========== EASY (Basic math, simple logic) ==========
    "easy": [
        ("What is 15% of 80?", "12"),
        ("If Alice is taller than Bob, and Bob is taller than Carol, who is the shortest?", "Carol"),
        ("What is 7 x 8?", "56"),
        ("A train travels at 60 mph. How far does it go in 2.5 hours?", "150"),
        ("If all dogs are mammals, and Fido is a dog, is Fido a mammal?", "Yes"),
    ],

    # ========== MEDIUM (Multi-step, requires careful reading) ==========
    "medium": [
        ("A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?", "0.05"),
        ("A lily pad doubles in size every day. If it takes 48 days to cover a pond, how long to cover half?", "47"),
        ("If you have 3 apples and take away 2, how many do you have?", "2"),  # Tricky - you took them
        ("A farmer has 17 sheep. All but 9 die. How many sheep are left?", "9"),
        ("How many times can you subtract 5 from 25?", "1"),  # Only once, then it's 20
        ("Some months have 30 days, some have 31. How many months have 28 days?", "12"),  # All of them
        ("If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?", "5"),
        ("A doctor gives you 3 pills and tells you to take one every half hour. How long do they last?", "1"),  # 1 hour (0, 0.5, 1)
    ],

    # ========== HARD (Complex reasoning, multi-step logic) ==========
    "hard": [
        # Knights and Knaves
        ("On an island, knights always tell the truth and knaves always lie. Person A says 'We are both knaves.' What are A and B?", "A is knave, B is knight"),

        # Complex syllogism
        ("If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?", "No"),

        # Counterfactual
        ("If yesterday was tomorrow, today would be Friday. What day is today?", "Sunday"),

        # Math puzzle
        ("I am thinking of a number. If I double it and add 10, I get 26. What is the number?", "8"),

        # River crossing variant
        ("A man needs to cross a river with a wolf, a goat, and cabbage. The boat fits only the man and one item. Wolf eats goat if alone, goat eats cabbage if alone. What's the minimum number of crossings?", "7"),

        # Age puzzle
        ("Mary is twice as old as Ann was when Mary was as old as Ann is now. If Mary is 24, how old is Ann?", "18"),

        # Probability
        ("I flip a fair coin twice. Given that at least one flip was heads, what's the probability both were heads?", "1/3"),
    ],

    # ========== VERY HARD (Expert reasoning, tricky edge cases) ==========
    "very_hard": [
        # Three-way knights and knaves
        ("A says 'B is a knave'. B says 'A and C are the same type'. C says 'I am a knight'. If exactly one is a knight, who is it?", "C"),

        # Complex math
        ("What is the sum of all integers from 1 to 100?", "5050"),

        # Self-reference
        ("This sentence has how many letters?", "This requires counting - answer varies"),

        # Paradox awareness
        ("Can an omnipotent being create a stone so heavy they cannot lift it? Explain the logical issue.", "Paradox - omnipotence is self-contradictory in this formulation"),

        # Multi-constraint optimization
        ("You have 12 balls, one is heavier. Using a balance scale, what's the minimum weighings needed to find it?", "3"),

        # Sequence with trick
        ("What comes next: 1, 11, 21, 1211, 111221, ?", "312211"),  # Look-and-say sequence

        # Complex conditional
        ("If A implies B, and B implies C, and not C is true, what can we conclude about A?", "A is false"),
    ],

    # ========== CONCEPTUAL (No single correct answer, tests reasoning quality) ==========
    "conceptual": [
        ("Explain why the sky is blue in one sentence.", None),
        ("What is the relationship between entropy and information?", None),
        ("Design a rate limiting algorithm for an API.", None),
        ("What makes something 'alive' versus 'not alive'?", None),
    ],
}


@dataclass
class ModeResult:
    answer: str
    score: float
    time: float
    mode: str


def extract_final_answer(answer: str) -> str:
    """Extract the final answer from a response."""
    # Look for boxed answers
    boxed_patterns = [
        r'\\boxed\{\\text\{([^}]+)\}\}',
        r'\\boxed\{([^}]+)\}',
        r'boxed\{([^}]+)\}',
    ]

    for pattern in boxed_patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        if matches:
            return matches[-1].strip().lower()

    # Look for explicit answer statements
    answer_patterns = [
        r'(?:final )?answer[:\s]+\*{0,2}([A-Za-z0-9$.,/\- ]+?)[\s*]*[.!?\n]',
        r'(?:final )?answer[:\s]+\*{0,2}([A-Za-z0-9$.,/\- ]+?)$',
    ]

    for pattern in answer_patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        if matches:
            return matches[-1].strip().lower()

    return ""


def check_correct(answer: str, correct: str) -> bool:
    """Check if the final answer contains the correct response."""
    if not correct:
        return False  # Conceptual questions - can't auto-check

    final = extract_final_answer(answer)

    if final:
        for c in correct.split('/'):
            if c.lower().strip() in final:
                return True
        return False

    # Fallback: context-based checking
    answer_lower = answer.lower()
    for c in correct.split('/'):
        c_clean = c.lower().strip()
        patterns = [
            rf'(?:the )?(?:answer|result) is[:\s]*[^.]*\b{re.escape(c_clean)}\b',
            rf'(?:therefore|thus|hence|so)[,\s]+[^.]*\b{re.escape(c_clean)}\b',
            rf'\b{re.escape(c_clean)}\b[^.]*(?:is the answer|is correct)',
        ]
        for pattern in patterns:
            if re.search(pattern, answer_lower):
                return True

    return False


def run_mode(question: str, mode: str, model: str, generations: int = 8, chains: int = 8) -> ModeResult:
    """Run the engine with specified mode and model."""
    from latent_reasoning import Engine, Config

    config = Config()
    config.encoder.model = model
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
        config.grammar.population_size = 12
        config.grammar.num_rules = 8
        config.grammar.max_depth = 4
    elif mode == "hybrid":
        config.qd.enabled = True
        config.qd.novelty_weight = 0.3
        config.grammar.enabled = True
        config.grammar.population_size = 12
        config.grammar.num_rules = 8
        config.grammar.max_depth = 4
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


def run_evaluation(
    difficulties: List[str],
    model: str,
    modes: List[str],
    generations: int,
    chains: int,
    output_file: str,
):
    """Run full evaluation."""
    # Collect questions
    questions: List[Tuple[str, str, str, Optional[str]]] = []  # (difficulty, question, correct)
    for diff in difficulties:
        if diff in QUESTIONS:
            for q, correct in QUESTIONS[diff]:
                questions.append((diff, q, correct))

    print("=" * 80)
    print(f"COMPREHENSIVE DEEPSEEK EVALUATION")
    print(f"Model: {model}")
    print(f"Questions: {len(questions)} | Difficulties: {', '.join(difficulties)}")
    print(f"Modes: {', '.join(modes)}")
    print(f"Config: {generations} generations, {chains} chains")
    print("=" * 80)

    results = []
    stats = {mode: {"correct": 0, "total": 0, "total_time": 0.0, "by_diff": {d: {"correct": 0, "total": 0} for d in difficulties}} for mode in modes}

    for i, (diff, question, correct) in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] [{diff.upper()}] {question[:55]}...")
        print("-" * 70)

        question_results = {"question": question, "difficulty": diff, "correct_answer": correct}

        for mode in modes:
            print(f"  {mode.upper():10s}: ", end="", flush=True)
            try:
                result = run_mode(question, mode, model, generations, chains)
                is_correct = check_correct(result.answer, correct) if correct else None

                status = "CORRECT" if is_correct else ("WRONG" if is_correct is False else "N/A")
                print(f"Done ({result.time:.1f}s, score={result.score:.3f}) [{status}]")

                question_results[mode] = {
                    "answer": result.answer[:500] if result.answer else "",
                    "score": result.score,
                    "time": result.time,
                    "correct": is_correct,
                }

                stats[mode]["total"] += 1
                stats[mode]["total_time"] += result.time
                stats[mode]["by_diff"][diff]["total"] += 1
                if is_correct:
                    stats[mode]["correct"] += 1
                    stats[mode]["by_diff"][diff]["correct"] += 1

            except Exception as e:
                print(f"ERROR: {e}")
                question_results[mode] = {"error": str(e)}

        results.append(question_results)

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    # Filter to questions with correct answers for accuracy calc
    scorable = sum(1 for _, _, c in questions if c is not None)

    print(f"\nOverall Accuracy (on {scorable} scorable questions):")
    for mode in modes:
        s = stats[mode]
        scorable_total = sum(s["by_diff"][d]["total"] for d in difficulties if any(c for _, c in QUESTIONS.get(d, []) if c))
        if scorable_total > 0:
            acc = 100 * s["correct"] / scorable_total
            avg_time = s["total_time"] / s["total"] if s["total"] > 0 else 0
            print(f"  {mode.upper():10s}: {s['correct']}/{scorable_total} ({acc:.1f}%) | Avg time: {avg_time:.1f}s")

    print("\nBy Difficulty:")
    for diff in difficulties:
        print(f"\n  {diff.upper()}:")
        for mode in modes:
            d_stats = stats[mode]["by_diff"][diff]
            if d_stats["total"] > 0:
                # Only count scorable
                scorable_in_diff = sum(1 for q, c in QUESTIONS.get(diff, []) if c is not None)
                if scorable_in_diff > 0:
                    acc = 100 * d_stats["correct"] / scorable_in_diff
                    print(f"    {mode.upper():10s}: {d_stats['correct']}/{scorable_in_diff} ({acc:.1f}%)")

    # Find best mode
    best_mode = max(modes, key=lambda m: stats[m]["correct"])
    print(f"\nBest Mode: {best_mode.upper()}")

    # Save results
    out_path = Path(__file__).parent / output_file
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'config': {'generations': generations, 'chains': chains, 'difficulties': difficulties, 'modes': modes},
            'stats': stats,
            'results': results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {out_path}")
    print("\n" + "=" * 80)
    print("REMINDER: Run manual review on outputs for quality verification!")
    print("Use: codex exec \"Review these outputs for reasoning quality: [paste]\"")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive DeepSeek evaluation")
    parser.add_argument('--mode', choices=['quick', 'standard', 'full', 'hard_only'], default='standard',
                       help='Evaluation scope')
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                       help='Model to use')
    parser.add_argument('--generations', type=int, default=8)
    parser.add_argument('--chains', type=int, default=8)
    parser.add_argument('--output', type=str, default='eval_deepseek_comprehensive_results.json')
    parser.add_argument('--modes', type=str, default='standard,grammar,hybrid',
                       help='Comma-separated list of modes to test')
    args = parser.parse_args()

    # Determine difficulties based on mode
    if args.mode == 'quick':
        difficulties = ['easy', 'medium']
    elif args.mode == 'standard':
        difficulties = ['easy', 'medium', 'hard']
    elif args.mode == 'full':
        difficulties = ['easy', 'medium', 'hard', 'very_hard', 'conceptual']
    elif args.mode == 'hard_only':
        difficulties = ['hard', 'very_hard']

    modes = [m.strip() for m in args.modes.split(',')]

    run_evaluation(
        difficulties=difficulties,
        model=args.model,
        modes=modes,
        generations=args.generations,
        chains=args.chains,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
