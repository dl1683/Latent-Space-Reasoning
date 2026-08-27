"""Gemini-based A/B judge for blind comparison of plans."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from enum import Enum

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class Winner(Enum):
    """Result of A/B comparison."""
    A = "A"
    B = "B"
    TIE = "TIE"


@dataclass
class ABResult:
    """Result of blind A/B comparison."""
    winner: Winner
    reasoning: str
    confidence: float  # 0-1
    a_was_baseline: bool  # For unblinding


AB_JUDGE_PROMPT = """You are an expert judge evaluating the quality of two plans for solving a problem.

QUERY: {query}

PLAN A:
{plan_a}

PLAN B:
{plan_b}

Evaluate both plans on these criteria:
1. **Clarity**: Are the steps clear and well-explained?
2. **Completeness**: Does it cover all important aspects?
3. **Actionability**: Are the steps concrete and actionable?
4. **Logic**: Is the ordering and reasoning sound?
5. **Depth**: Does it show deep understanding of the problem?

Based on your evaluation, which plan is better overall?

Respond in this EXACT format:
WINNER: [A/B/TIE]
CONFIDENCE: [0.0-1.0]
REASONING: [Your detailed reasoning in 2-3 sentences]"""


class GeminiJudge:
    """
    Gemini-based judge for blind A/B comparison of plans.

    Randomly shuffles which plan is A vs B to prevent bias.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.0,  # Deterministic judging
    ):
        """
        Initialize the Gemini judge.

        Args:
            model: Gemini model to use
            api_key: API key (defaults to GEMINI_API_KEY env var)
            temperature: Sampling temperature (0 = deterministic)
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model,
            generation_config={"temperature": temperature},
        )

    def compare(
        self,
        query: str,
        baseline_plan: str,
        lr_plan: str,
    ) -> ABResult:
        """
        Blind A/B comparison of baseline vs LR plan.

        Args:
            query: The original query
            baseline_plan: Plan from baseline (direct generation)
            lr_plan: Plan from latent reasoning

        Returns:
            ABResult with winner and reasoning
        """
        # Randomly assign to A/B to prevent position bias
        baseline_is_a = random.random() < 0.5

        if baseline_is_a:
            plan_a, plan_b = baseline_plan, lr_plan
        else:
            plan_a, plan_b = lr_plan, baseline_plan

        # Build prompt
        prompt = AB_JUDGE_PROMPT.format(
            query=query,
            plan_a=plan_a[:2000],  # Truncate for API limits
            plan_b=plan_b[:2000],
        )

        # Get judgment
        response = self.model.generate_content(prompt)
        result_text = response.text

        # Parse response
        winner, confidence, reasoning = self._parse_response(result_text)

        # Unblind: convert A/B winner to actual winner
        if winner == Winner.A:
            actual_winner = Winner.A if baseline_is_a else Winner.B
            # Translate: if baseline won, that's "baseline wins"
            # We want to know if LR won
            lr_won = not baseline_is_a
        elif winner == Winner.B:
            actual_winner = Winner.B if baseline_is_a else Winner.A
            lr_won = baseline_is_a
        else:
            actual_winner = Winner.TIE
            lr_won = None

        return ABResult(
            winner=Winner.B if lr_won else (Winner.A if lr_won is False else Winner.TIE),
            reasoning=reasoning,
            confidence=confidence,
            a_was_baseline=baseline_is_a,
        )

    def _parse_response(self, text: str) -> tuple[Winner, float, str]:
        """Parse the judge's response."""
        lines = text.strip().split("\n")

        winner = Winner.TIE
        confidence = 0.5
        reasoning = ""

        for line in lines:
            line = line.strip()
            if line.startswith("WINNER:"):
                w = line.replace("WINNER:", "").strip().upper()
                if "A" in w and "B" not in w:
                    winner = Winner.A
                elif "B" in w and "A" not in w:
                    winner = Winner.B
                else:
                    winner = Winner.TIE
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.replace("CONFIDENCE:", "").strip()
                    confidence = float(conf_str)
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        # If reasoning spans multiple lines, grab the rest
        if not reasoning and len(lines) > 2:
            reasoning = " ".join(lines[2:])

        return winner, confidence, reasoning

    def batch_compare(
        self,
        comparisons: list[tuple[str, str, str]],  # (query, baseline, lr)
    ) -> list[ABResult]:
        """
        Compare multiple query/plan pairs.

        Args:
            comparisons: List of (query, baseline_plan, lr_plan) tuples

        Returns:
            List of ABResult objects
        """
        results = []
        for query, baseline, lr in comparisons:
            try:
                result = self.compare(query, baseline, lr)
                results.append(result)
            except Exception as e:
                # On error, record as tie with error message
                results.append(ABResult(
                    winner=Winner.TIE,
                    reasoning=f"Error: {e}",
                    confidence=0.0,
                    a_was_baseline=True,
                ))
        return results


def judge_plans(
    query: str,
    baseline_plan: str,
    lr_plan: str,
) -> ABResult:
    """
    Convenience function to judge two plans.

    Args:
        query: Original query
        baseline_plan: Baseline generated plan
        lr_plan: Latent reasoning generated plan

    Returns:
        ABResult indicating which is better
    """
    judge = GeminiJudge()
    return judge.compare(query, baseline_plan, lr_plan)
