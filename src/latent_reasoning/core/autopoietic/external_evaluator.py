"""
External evaluator for grounding autopoietic judge.

The external evaluator uses a frontier model (e.g., Gemini) to provide
quality scores for decoded latent vectors. These scores serve as ground
truth for updating the internal scorer.

Key Features:
- Quality scoring on 0-1 scale
- Multiple evaluation criteria
- Error handling and retry logic
- Batch evaluation for efficiency
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class ExternalScore:
    """
    Result of external evaluation.

    Attributes:
        score: Overall quality score (0 to 1)
        criteria_scores: Individual criterion scores
        reasoning: Brief explanation of the score
        confidence: Evaluator's confidence in the score
        error: Error message if evaluation failed
    """
    score: float
    criteria_scores: dict[str, float] | None = None
    reasoning: str = ""
    confidence: float = 1.0
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        """Whether the evaluation succeeded."""
        return self.error is None


QUALITY_EVAL_PROMPT = """You are an expert evaluator assessing the quality of a response to a query.

QUERY: {query}

RESPONSE:
{response}

Evaluate the response on these criteria (each 0.0 to 1.0):
1. **Relevance**: How well does it address the query?
2. **Clarity**: Is it clear and well-structured?
3. **Depth**: Does it show deep understanding?
4. **Accuracy**: Does it seem factually correct?
5. **Completeness**: Does it cover the important aspects?

Provide your evaluation in this EXACT format:
RELEVANCE: [0.0-1.0]
CLARITY: [0.0-1.0]
DEPTH: [0.0-1.0]
ACCURACY: [0.0-1.0]
COMPLETENESS: [0.0-1.0]
OVERALL: [0.0-1.0]
REASONING: [Brief 1-2 sentence explanation]"""


LATENT_QUALITY_PROMPT = """You are evaluating whether a response demonstrates high-quality reasoning.

QUERY: {query}

RESPONSE:
{response}

Consider:
- Does the response show clear, logical thinking?
- Is the information relevant and accurate?
- Is the depth appropriate for the query?

Rate the overall quality from 0.0 (very poor) to 1.0 (excellent).

Respond in this EXACT format:
SCORE: [0.0-1.0]
REASONING: [Brief explanation]"""


class ExternalEvaluator:
    """
    External evaluator using Gemini for quality grounding.

    Provides ground truth quality scores by evaluating decoded text
    using a frontier model. These scores are used to update the
    internal scorer and prevent drift.

    Args:
        model: Gemini model to use
        api_key: API key (defaults to GEMINI_API_KEY env var)
        temperature: Sampling temperature (0 = deterministic)
        detailed: Whether to use detailed multi-criterion evaluation

    Usage:
        >>> evaluator = ExternalEvaluator()
        >>> score = evaluator.evaluate("What is AI?", "AI is...")
        >>> print(f"Quality: {score.score:.2f}")
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        detailed: bool = False,
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(
            model,
            generation_config={"temperature": temperature},
        )
        self.detailed = detailed

    def evaluate(
        self,
        query: str,
        response: str,
        max_retries: int = 2,
    ) -> ExternalScore:
        """
        Evaluate a single query-response pair.

        Args:
            query: The original query
            response: The generated response
            max_retries: Number of retries on failure

        Returns:
            ExternalScore with quality assessment
        """
        prompt = self._build_prompt(query, response)

        for attempt in range(max_retries + 1):
            try:
                result = self.model.generate_content(prompt)
                return self._parse_response(result.text)
            except Exception as e:
                if attempt == max_retries:
                    return ExternalScore(
                        score=0.5,  # Default to neutral on failure
                        error=str(e),
                        confidence=0.0,
                    )
                # Wait briefly before retry
                import time
                time.sleep(0.5 * (attempt + 1))

        return ExternalScore(score=0.5, error="Max retries exceeded", confidence=0.0)

    def evaluate_batch(
        self,
        pairs: list[tuple[str, str]],
        max_retries: int = 2,
    ) -> list[ExternalScore]:
        """
        Evaluate multiple query-response pairs.

        Args:
            pairs: List of (query, response) tuples
            max_retries: Number of retries per evaluation

        Returns:
            List of ExternalScore objects
        """
        return [
            self.evaluate(query, response, max_retries)
            for query, response in pairs
        ]

    def _build_prompt(self, query: str, response: str) -> str:
        """Build evaluation prompt."""
        # Truncate long responses
        response = response[:3000] if len(response) > 3000 else response

        if self.detailed:
            return QUALITY_EVAL_PROMPT.format(query=query, response=response)
        else:
            return LATENT_QUALITY_PROMPT.format(query=query, response=response)

    def _parse_response(self, text: str) -> ExternalScore:
        """Parse the evaluator's response."""
        lines = text.strip().split("\n")

        score = 0.5
        reasoning = ""
        criteria_scores = {} if self.detailed else None

        for line in lines:
            line = line.strip()

            if self.detailed:
                # Parse detailed criteria
                for criterion in ["RELEVANCE", "CLARITY", "DEPTH", "ACCURACY", "COMPLETENESS"]:
                    if line.upper().startswith(criterion + ":"):
                        try:
                            val = float(line.split(":")[-1].strip())
                            criteria_scores[criterion.lower()] = max(0.0, min(1.0, val))
                        except ValueError:
                            pass

                if line.upper().startswith("OVERALL:"):
                    try:
                        score = float(line.split(":")[-1].strip())
                        score = max(0.0, min(1.0, score))
                    except ValueError:
                        pass
            else:
                # Parse simple score
                if line.upper().startswith("SCORE:"):
                    try:
                        score = float(line.split(":")[-1].strip())
                        score = max(0.0, min(1.0, score))
                    except ValueError:
                        pass

            if line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[-1].strip()

        # If detailed, compute overall from criteria if not provided
        if self.detailed and criteria_scores and score == 0.5:
            score = sum(criteria_scores.values()) / len(criteria_scores)

        return ExternalScore(
            score=score,
            criteria_scores=criteria_scores,
            reasoning=reasoning,
            confidence=1.0,
        )

    def __repr__(self) -> str:
        return f"ExternalEvaluator(model={self.model_name}, detailed={self.detailed})"


class MockExternalEvaluator:
    """
    Mock evaluator for testing without API calls.

    Generates plausible scores based on simple heuristics.
    """

    def __init__(self, base_score: float = 0.6, noise: float = 0.1):
        self.base_score = base_score
        self.noise = noise

    def evaluate(
        self,
        query: str,
        response: str,
        max_retries: int = 2,
    ) -> ExternalScore:
        """Generate a mock score."""
        import random

        # Simple heuristics
        length_bonus = min(len(response) / 500, 0.2)  # Longer = slightly better
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap_bonus = len(query_words & response_words) / max(len(query_words), 1) * 0.2

        score = self.base_score + length_bonus + overlap_bonus
        score += random.gauss(0, self.noise)
        score = max(0.1, min(0.95, score))

        return ExternalScore(
            score=score,
            reasoning="Mock evaluation",
            confidence=0.8,
        )

    def evaluate_batch(
        self,
        pairs: list[tuple[str, str]],
        max_retries: int = 2,
    ) -> list[ExternalScore]:
        """Evaluate multiple pairs."""
        return [self.evaluate(q, r, max_retries) for q, r in pairs]

    def __repr__(self) -> str:
        return f"MockExternalEvaluator(base={self.base_score})"


def create_external_evaluator(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    mock: bool = False,
) -> ExternalEvaluator | MockExternalEvaluator:
    """
    Factory function to create an external evaluator.

    Args:
        model: Gemini model to use
        temperature: Sampling temperature
        mock: If True, return a mock evaluator for testing

    Returns:
        ExternalEvaluator or MockExternalEvaluator
    """
    if mock:
        return MockExternalEvaluator()

    try:
        return ExternalEvaluator(model=model, temperature=temperature)
    except ValueError:
        # Fall back to mock if no API key
        import warnings
        warnings.warn("GEMINI_API_KEY not found, using mock evaluator")
        return MockExternalEvaluator()
