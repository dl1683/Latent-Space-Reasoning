"""
Comprehensive Evaluation Script for Latent Space Reasoning

Consolidates all evaluation functionality:
- Baseline vs QD comparison
- Various question categories
- Output collection for external judging
- Statistical analysis

Usage:
    python eval_comprehensive.py --mode quick      # 5 questions, fast test
    python eval_comprehensive.py --mode standard   # 20 questions
    python eval_comprehensive.py --mode full       # All 60+ questions
    python eval_comprehensive.py --mode category --cat logic  # Specific category

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
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

# Question bank organized by category
QUESTIONS = {
    "math": [
        ("A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?", "$0.05"),
        ("What is 15% of 80?", "12"),
        ("If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "5 minutes"),
        ("A lily pad doubles in size every day. If it takes 48 days to cover a pond, how long to cover half?", "47 days"),
    ],
    "logic": [
        ("If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?", "No (invalid syllogism)"),
        ("All cats are animals. Some animals are pets. Can we conclude that some cats are pets?", "No (invalid syllogism)"),
        ("If no fish can fly and all salmon are fish, can salmon fly?", "No"),
        ("No reptiles have fur. All snakes are reptiles. Do snakes have fur?", "No"),
    ],
    "tricky": [
        ("If you have 3 apples and take away 2, how many do you have?", "2 (you took them)"),
        ("A farmer has 17 sheep. All but 9 die. How many sheep are left?", "9"),
        ("How many times can you subtract 5 from 25?", "1 (then it's 20)"),
        ("A doctor gives you 3 pills and tells you to take one every half hour. How long do they last?", "1 hour"),
        ("Some months have 30 days, some have 31. How many months have 28 days?", "12 (all of them)"),
    ],
    "transitive": [
        ("If Alice is taller than Bob, and Bob is taller than Carol, who is the shortest?", "Carol"),
        ("If A > B, B > C, and C > D, what is the relationship between A and D?", "A > D"),
        ("John is older than Mary. Mary is older than Tom. Tom is older than Sarah. Who is the oldest?", "John"),
    ],
    "puzzles": [
        ("A snail climbs 3 feet up a wall during the day but slides down 2 feet at night. How many days to climb 10 feet?", "8 days"),
        ("You have two jugs: one holds 3 liters, the other 5 liters. How do you measure exactly 4 liters?", "Fill 5L, pour to 3L, empty 3L, pour 2L to 3L, fill 5L, pour 1L to 3L = 4L left"),
        ("You have 8 balls. One is heavier. You have a balance scale. What's the minimum weighings to find the heavy ball?", "2"),
    ],
    "conceptual": [
        ("What is the relationship between entropy and information? Explain simply.", None),
        ("Why is the sky blue? Explain to a child.", None),
        ("What makes something 'alive' versus 'not alive'?", None),
        ("Explain recursion using a simple example.", None),
        ("What is the difference between correlation and causation?", None),
    ],
    "engineering": [
        ("Design a rate limiting system for an API.", None),
        ("Plan a database migration with zero downtime.", None),
        ("Troubleshoot a memory leak in a Python service.", None),
        ("Design a secure password reset flow.", None),
        ("Design a CI/CD pipeline for a monorepo.", None),
    ],
    "philosophical": [
        ("What would happen if humans could photosynthesize?", None),
        ("Why might a society that values freedom also need rules?", None),
        ("What's the difference between knowledge and wisdom?", None),
        ("Can a machine ever truly 'understand' something?", None),
    ],
}

# Config
DEFAULT_GENERATIONS = 6
DEFAULT_CHAINS = 6


@dataclass
class RunResult:
    answer: str
    score: float
    time: float
    archive_size: int = 0


@dataclass
class ComparisonResult:
    query: str
    category: str
    correct_answer: Optional[str]
    baseline: RunResult
    qd: RunResult
    winner: str  # "BASELINE", "QD", "TIE"
    reason: str


def run_engine(question: str, use_qd: bool, generations: int, chains: int) -> RunResult:
    """Run the engine with specified config."""
    from latent_reasoning import Engine, Config

    config = Config()
    config.encoder.model = "Qwen/Qwen3-0.6B"
    config.encoder.quantization = "4bit"
    config.evolution.generations = generations
    config.evolution.chains = chains
    config.qd.enabled = use_qd
    config.qd.novelty_weight = 0.3
    config.output.verbosity = 0

    engine = Engine(config=config)
    start = time.time()
    result = engine.run(question)
    elapsed = time.time() - start

    archive_size = 0
    if use_qd:
        orch = engine._get_orchestrator()
        if orch.qd_manager:
            archive_size = len(orch.qd_manager.archive)

    return RunResult(
        answer=result.plan or "",
        score=result.confidence,
        time=elapsed,
        archive_size=archive_size,
    )


def extract_final_answer(answer: str) -> str:
    """Extract the final answer from a response, looking for common patterns.

    Returns the extracted final answer text, or empty string if none found.
    """
    import re

    # Look for boxed answers (highest priority - this IS the answer)
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
        r'(?:final )?answer[:\s]+\*{0,2}([A-Za-z0-9$., ]+?)[\s*]*[.!?\n]',
        r'(?:final )?answer[:\s]+\*{0,2}([A-Za-z0-9$., ]+?)$',
    ]

    for pattern in answer_patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        if matches:
            return matches[-1].strip().lower()

    return ""


def check_correct(answer: str, correct: str) -> bool:
    """Check if the final answer contains the correct response.

    IMPORTANT: Checks FINAL ANSWER, not entire text. Manual review still essential.
    """
    if not correct:
        return False

    final = extract_final_answer(answer)

    if final:
        for c in correct.split('/'):
            if c.lower().strip() in final:
                return True
        return False

    # Fallback: context-based checking
    import re
    answer_lower = answer.lower()
    for c in correct.split('/'):
        c_clean = c.lower().strip()
        patterns = [
            rf'(?:the )?(?:answer|result) is[:\s]*[^.]*\b{re.escape(c_clean)}\b',
            rf'(?:therefore|thus|hence|so)[,\s]+[^.]*\b{re.escape(c_clean)}\b',
        ]
        for pattern in patterns:
            if re.search(pattern, answer_lower):
                return True

    return False


def compare_answers(baseline: RunResult, qd: RunResult, correct: Optional[str]) -> tuple:
    """Compare answers and determine winner."""
    # If we have a correct answer, check correctness
    if correct:
        baseline_correct = check_correct(baseline.answer, correct)
        qd_correct = check_correct(qd.answer, correct)

        if baseline_correct and not qd_correct:
            return "BASELINE", "Baseline got correct answer"
        elif qd_correct and not baseline_correct:
            return "QD", "QD got correct answer"
        elif baseline_correct and qd_correct:
            # Both correct, compare quality
            if qd.score > baseline.score + 0.05:
                return "QD", "Both correct, QD higher quality"
            elif baseline.score > qd.score + 0.05:
                return "BASELINE", "Both correct, Baseline higher quality"
            return "TIE", "Both correct, similar quality"
        else:
            return "TIE", "Both incorrect"

    # No correct answer, compare scores (with skepticism)
    diff = qd.score - baseline.score
    if abs(diff) < 0.01:
        return "TIE", "Scores similar"
    elif diff > 0:
        return "QD", f"QD score higher (+{diff:.3f})"
    else:
        return "BASELINE", f"Baseline score higher ({diff:.3f})"


def clean_text(s: str, max_len: int = 500) -> str:
    if not s:
        return ""
    s = s.strip().encode('ascii', 'replace').decode('ascii')
    return s[:max_len] + "..." if len(s) > max_len else s


def run_evaluation(
    categories: List[str],
    generations: int,
    chains: int,
    output_file: str,
) -> List[ComparisonResult]:
    """Run evaluation on specified categories."""
    results = []

    # Collect questions
    questions = []
    for cat in categories:
        if cat in QUESTIONS:
            for q, correct in QUESTIONS[cat]:
                questions.append((cat, q, correct))

    print("=" * 80)
    print(f"LATENT SPACE REASONING EVALUATION")
    print(f"Questions: {len(questions)} | Categories: {', '.join(categories)}")
    print(f"Config: {generations} generations, {chains} chains")
    print("=" * 80)

    stats = {"baseline_wins": 0, "qd_wins": 0, "ties": 0}

    for i, (cat, query, correct) in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] [{cat.upper()}] {query[:50]}...")
        print("-" * 70)

        # Run baseline
        print("  Baseline: ", end="", flush=True)
        try:
            baseline = run_engine(query, use_qd=False, generations=generations, chains=chains)
            print(f"Done ({baseline.time:.1f}s, score={baseline.score:.3f})")
        except Exception as e:
            print(f"ERROR: {e}")
            baseline = RunResult(f"ERROR: {e}", 0.0, 0.0)

        # Run QD
        print("  QD:       ", end="", flush=True)
        try:
            qd = run_engine(query, use_qd=True, generations=generations, chains=chains)
            print(f"Done ({qd.time:.1f}s, score={qd.score:.3f}, archive={qd.archive_size})")
        except Exception as e:
            print(f"ERROR: {e}")
            qd = RunResult(f"ERROR: {e}", 0.0, 0.0)

        # Compare
        winner, reason = compare_answers(baseline, qd, correct)
        print(f"  >>> {winner}: {reason}")

        if winner == "BASELINE":
            stats["baseline_wins"] += 1
        elif winner == "QD":
            stats["qd_wins"] += 1
        else:
            stats["ties"] += 1

        results.append(ComparisonResult(
            query=query,
            category=cat,
            correct_answer=correct,
            baseline=baseline,
            qd=qd,
            winner=winner,
            reason=reason,
        ))

        # Running tally
        total = stats["baseline_wins"] + stats["qd_wins"] + stats["ties"]
        print(f"  Running: QD={stats['qd_wins']} Baseline={stats['baseline_wins']} Ties={stats['ties']}")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    total = len(results)
    print(f"\nWinner Breakdown:")
    print(f"  QD:       {stats['qd_wins']:3d} ({100*stats['qd_wins']/total:.1f}%)")
    print(f"  Baseline: {stats['baseline_wins']:3d} ({100*stats['baseline_wins']/total:.1f}%)")
    print(f"  Ties:     {stats['ties']:3d} ({100*stats['ties']/total:.1f}%)")

    # Category breakdown
    print("\nBy Category:")
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        if cat_results:
            qd_w = sum(1 for r in cat_results if r.winner == "QD")
            b_w = sum(1 for r in cat_results if r.winner == "BASELINE")
            print(f"  {cat:12s}: QD={qd_w} Baseline={b_w} (n={len(cat_results)})")

    # Save results
    out_path = Path(__file__).parent / output_file
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {'generations': generations, 'chains': chains, 'categories': categories},
            'summary': stats,
            'results': [
                {
                    'query': r.query,
                    'category': r.category,
                    'correct': r.correct_answer,
                    'baseline_answer': clean_text(r.baseline.answer),
                    'baseline_score': r.baseline.score,
                    'qd_answer': clean_text(r.qd.answer),
                    'qd_score': r.qd.score,
                    'qd_archive': r.qd.archive_size,
                    'winner': r.winner,
                    'reason': r.reason,
                }
                for r in results
            ],
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation for Latent Space Reasoning")
    parser.add_argument('--mode', choices=['quick', 'standard', 'full', 'category'], default='quick',
                       help='Evaluation mode')
    parser.add_argument('--cat', type=str, help='Category for category mode (logic, math, tricky, etc.)')
    parser.add_argument('--generations', type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument('--chains', type=int, default=DEFAULT_CHAINS)
    parser.add_argument('--output', type=str, default='eval_comprehensive_results.json')
    args = parser.parse_args()

    # Determine categories based on mode
    if args.mode == 'quick':
        categories = ['math', 'logic']  # ~8 questions
    elif args.mode == 'standard':
        categories = ['math', 'logic', 'tricky', 'transitive']  # ~16 questions
    elif args.mode == 'full':
        categories = list(QUESTIONS.keys())  # All ~35 questions
    elif args.mode == 'category':
        if not args.cat or args.cat not in QUESTIONS:
            print(f"Available categories: {', '.join(QUESTIONS.keys())}")
            return
        categories = [args.cat]

    run_evaluation(
        categories=categories,
        generations=args.generations,
        chains=args.chains,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
