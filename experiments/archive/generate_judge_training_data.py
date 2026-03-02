"""
Generate training data for latent scorer using Claude as strong judge.

This script:
1. Generates diverse prompts across categories
2. For each prompt, generates multiple responses with varying quality
3. Uses Claude as LLM-as-judge to score each response
4. Outputs JSON suitable for training the latent scorer
"""

import json
import random
import sys
import io
from datetime import datetime
from pathlib import Path
from anthropic import Anthropic

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Prompt categories for diverse training data
PROMPT_TEMPLATES = {
    "math": [
        "Prove that {concept} is {property}.",
        "Explain why {math_fact}.",
        "Solve: {math_problem}",
        "Derive the formula for {formula_topic}.",
    ],
    "coding": [
        "Write a function to {coding_task}.",
        "Implement {data_structure} with {operations}.",
        "Debug this code: {buggy_code_desc}",
        "Optimize {algorithm} for {constraint}.",
    ],
    "reasoning": [
        "If {premise1} and {premise2}, can we conclude {conclusion}?",
        "{logic_puzzle}",
        "What are the implications of {scenario}?",
        "Compare the trade-offs between {optionA} and {optionB}.",
    ],
    "planning": [
        "Design a system for {system_type}.",
        "Create a migration plan for {migration_task}.",
        "How would you architect {architecture_task}?",
        "Plan the implementation of {feature}.",
    ],
}

# Sample prompts for each category
SAMPLE_PROMPTS = {
    "math": [
        "Prove that the square root of 2 is irrational.",
        "Explain why 0.999... equals 1.",
        "Prove that there are infinitely many prime numbers.",
        "Derive the quadratic formula.",
        "Explain why the sum of angles in a triangle is 180 degrees.",
    ],
    "coding": [
        "Write a function to detect a cycle in a linked list.",
        "Implement a binary search tree with insert and search methods.",
        "Write a function to find the longest common subsequence.",
        "Implement a LRU cache.",
        "Write a function to serialize and deserialize a binary tree.",
    ],
    "reasoning": [
        "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "Three people check into a hotel room that costs $30. They each pay $10. Later, the manager realizes the room only costs $25, and gives the bellboy $5 to return. The bellboy pockets $2 and gives $1 back to each guest. Now each guest has paid $9 (total $27), plus the $2 the bellboy kept. Where did the other dollar go?",
        "What are the trade-offs between consistency and availability in distributed systems?",
        "You have 8 balls. 7 are identical, 1 is heavier. You have a balance scale. What's the minimum weighings to find the heavy ball?",
    ],
    "planning": [
        "Design a system for a ride-sharing app like Uber.",
        "Create a migration plan for moving a monolithic application to microservices.",
        "How would you architect a real-time collaborative document editor?",
        "Plan the implementation of a recommendation system for an e-commerce site.",
        "Design a distributed job scheduling system.",
    ],
}

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI-generated reasoning and responses.
Your task is to score responses on a scale of 0.0 to 1.0 based on:
- Correctness: Is the answer factually correct?
- Reasoning: Is the reasoning clear, logical, and well-structured?
- Completeness: Does it address all aspects of the question?
- Clarity: Is it easy to understand?

Score ranges:
- 0.9-1.0: Excellent - correct, clear, complete, well-reasoned
- 0.7-0.89: Good - mostly correct with minor issues
- 0.5-0.69: Adequate - some correctness but significant gaps
- 0.3-0.49: Poor - major errors or incomplete
- 0.0-0.29: Very poor - fundamentally wrong or irrelevant

Respond with ONLY a JSON object: {"score": <float>, "reasoning": "<brief explanation>"}"""


def score_response(client: Anthropic, query: str, response: str) -> dict:
    """Use Claude to score a response."""
    user_prompt = f"""Score this response to the given query.

QUERY: {query}

RESPONSE: {response}

Provide your score (0.0-1.0) and brief reasoning in JSON format."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = message.content[0].text.strip()

    # Parse JSON response
    try:
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        result = json.loads(text)
        return {
            "score": float(result.get("score", 0.5)),
            "judge_reasoning": result.get("reasoning", ""),
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  Warning: Failed to parse judge response: {e}")
        return {"score": 0.5, "judge_reasoning": f"Parse error: {text[:100]}"}


def generate_varied_responses(client: Anthropic, query: str, n: int = 3) -> list[str]:
    """Generate varied quality responses for a query."""
    responses = []

    # Generate responses with different quality levels
    quality_prompts = [
        # High quality - careful reasoning
        f"Think step by step carefully and provide a thorough, correct answer.\n\nQuestion: {query}",
        # Medium quality - quick response
        f"Answer this question:\n\n{query}",
        # Lower quality - rushed/partial
        f"Give a brief answer to: {query}",
    ]

    for i, prompt in enumerate(quality_prompts[:n]):
        try:
            temp = 0.3 + (i * 0.3)  # Vary temperature: 0.3, 0.6, 0.9
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=temp,
                messages=[{"role": "user", "content": prompt}],
            )
            responses.append(message.content[0].text)
        except Exception as e:
            print(f"  Warning: Failed to generate response {i}: {e}")

    return responses


def main():
    print("=" * 70)
    print("GENERATING TRAINING DATA FOR LATENT SCORER")
    print(f"Using Claude as strong judge")
    print("=" * 70)

    client = Anthropic()

    training_data = []

    # Process each category
    for category, prompts in SAMPLE_PROMPTS.items():
        print(f"\n{category.upper()}")

        for query in prompts[:3]:  # Limit to 3 per category for initial run
            print(f"\n  Query: {query[:50]}...")

            # Generate varied responses
            responses = generate_varied_responses(client, query, n=3)

            for i, response in enumerate(responses):
                print(f"    Response {i+1}...", end="", flush=True)

                # Score with Claude judge
                result = score_response(client, query, response)

                training_data.append({
                    "query": query,
                    "plan": response,  # "plan" is the field name expected by trainer
                    "score": result["score"],
                    "judge_reasoning": result["judge_reasoning"],
                    "category": category,
                    "response_type": ["thorough", "standard", "brief"][i],
                })

                print(f" score={result['score']:.2f}")

    # Save training data
    output_path = Path(f"experiments/claude_judge_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"Generated {len(training_data)} training samples")
    print(f"Saved to: {output_path}")

    # Print score distribution
    scores = [d["score"] for d in training_data]
    print(f"\nScore distribution:")
    print(f"  Min: {min(scores):.2f}")
    print(f"  Max: {max(scores):.2f}")
    print(f"  Mean: {sum(scores)/len(scores):.2f}")


if __name__ == "__main__":
    main()
