"""Quick test script for manual review of outputs."""
import sys
sys.path.insert(0, "C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning")

# Fix Windows console encoding issues
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from latent_reasoning import Engine, Config

# Test 5 diverse questions
questions = [
    "A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "What is 15% of 80?",
    "If Alice is taller than Bob, and Bob is taller than Carol, who is the shortest?",
    "Explain why the sky is blue in one sentence.",
    "What comes next in the sequence: 2, 4, 8, 16, ?"
]

config = Config()
config.evolution.generations = 6
config.evolution.chains = 8

engine = Engine(config=config)

print("=" * 80)
print("COMPREHENSIVE TEST - 5 QUESTIONS FOR MANUAL REVIEW")
print("=" * 80)

outputs = []
for i, q in enumerate(questions, 1):
    print(f"\n[Q{i}] {q}")
    print("-" * 60)
    result = engine.run(q)
    output = result.result if hasattr(result, "result") else str(result)
    outputs.append({"question": q, "answer": output})
    score = result.score if hasattr(result, "score") else 0
    gens = result.generations if hasattr(result, "generations") else 0
    print(f"Score: {score:.3f} | Gens: {gens}")

print("\n" + "=" * 80)
print("FULL OUTPUTS FOR MANUAL REVIEW")
print("=" * 80)
for i, o in enumerate(outputs, 1):
    print(f"\n{'='*20} Q{i} {'='*20}")
    print(f"QUESTION: {o['question']}")
    print(f"\nANSWER:\n{o['answer']}")
    print("-" * 50)
