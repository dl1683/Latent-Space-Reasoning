"""Compute budget management for the orchestrator."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class ComputeBudget:
    """Tracks and enforces compute budget limits."""

    max_generations: int
    max_evaluations: int
    max_time: float | None = None  # Seconds

    # Tracking
    generations_used: int = 0
    evaluations_used: int = 0
    start_time: float | None = None

    def start(self) -> None:
        """Start the budget timer."""
        self.start_time = time.time()

    def can_continue(self) -> bool:
        """Check if we can continue within budget."""
        # Check generations
        if self.generations_used >= self.max_generations:
            return False

        # Check evaluations
        if self.evaluations_used >= self.max_evaluations:
            return False

        # Check time
        if self.max_time is not None and self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed >= self.max_time:
                return False

        return True

    def record_generation(self, num_evaluations: int = 0) -> None:
        """Record completion of a generation."""
        self.generations_used += 1
        self.evaluations_used += num_evaluations

    def record_evaluation(self, count: int = 1) -> None:
        """Record evaluation(s)."""
        self.evaluations_used += count

    def get_remaining(self) -> dict:
        """Get remaining budget."""
        remaining = {
            "generations": self.max_generations - self.generations_used,
            "evaluations": self.max_evaluations - self.evaluations_used,
        }

        if self.max_time is not None and self.start_time is not None:
            elapsed = time.time() - self.start_time
            remaining["time"] = max(0, self.max_time - elapsed)

        return remaining

    def get_usage(self) -> dict:
        """Get current usage statistics."""
        usage = {
            "generations": self.generations_used,
            "evaluations": self.evaluations_used,
            "generations_pct": self.generations_used / self.max_generations * 100,
            "evaluations_pct": self.evaluations_used / self.max_evaluations * 100,
        }

        if self.start_time is not None:
            usage["elapsed_time"] = time.time() - self.start_time
            if self.max_time is not None:
                usage["time_pct"] = usage["elapsed_time"] / self.max_time * 100

        return usage

    def get_stop_reason(self) -> str | None:
        """Get the reason for stopping, if any."""
        if self.generations_used >= self.max_generations:
            return "max_generations"
        if self.evaluations_used >= self.max_evaluations:
            return "max_evaluations"
        if self.max_time is not None and self.start_time is not None:
            if time.time() - self.start_time >= self.max_time:
                return "max_time"
        return None

    def reset(self) -> None:
        """Reset the budget tracking."""
        self.generations_used = 0
        self.evaluations_used = 0
        self.start_time = None
