"""
Verifiable Tasks for Ground-Truth Evolution

This module provides tasks with PROGRAMMATIC VERIFICATION - no external APIs needed.
The key insight: evolve latents using pass/fail on real correctness, not style scoring.

Task types:
- Math: arithmetic, algebra with known answers
- Logic: propositional logic, truth tables
- Code: simple functions with unit tests

Each task has:
- prompt: the question to answer
- verifier: function that checks if answer is correct
- extract_answer: function to parse the model's answer
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class VerifiableTask:
    """A task with programmatic verification."""
    prompt: str
    correct_answer: Any
    verifier: Callable[[str, Any], bool]
    category: str
    difficulty: str = "easy"


class TaskGenerator(ABC):
    """Base class for task generators."""

    @abstractmethod
    def generate(self, n: int = 10) -> list[VerifiableTask]:
        """Generate n tasks."""
        pass


class ArithmeticTaskGenerator(TaskGenerator):
    """Generate arithmetic tasks with known answers."""

    def generate(self, n: int = 10) -> list[VerifiableTask]:
        tasks = []
        for _ in range(n):
            task_type = random.choice(["add", "subtract", "multiply", "divide", "mixed"])
            task = self._generate_task(task_type)
            tasks.append(task)
        return tasks

    def _generate_task(self, task_type: str) -> VerifiableTask:
        if task_type == "add":
            a, b = random.randint(1, 100), random.randint(1, 100)
            return VerifiableTask(
                prompt=f"What is {a} + {b}? Give only the numeric answer.",
                correct_answer=a + b,
                verifier=self._verify_number,
                category="arithmetic",
            )
        elif task_type == "subtract":
            a, b = random.randint(50, 150), random.randint(1, 50)
            return VerifiableTask(
                prompt=f"What is {a} - {b}? Give only the numeric answer.",
                correct_answer=a - b,
                verifier=self._verify_number,
                category="arithmetic",
            )
        elif task_type == "multiply":
            a, b = random.randint(2, 15), random.randint(2, 15)
            return VerifiableTask(
                prompt=f"What is {a} × {b}? Give only the numeric answer.",
                correct_answer=a * b,
                verifier=self._verify_number,
                category="arithmetic",
            )
        elif task_type == "divide":
            b = random.randint(2, 12)
            a = b * random.randint(2, 12)  # Ensure clean division
            return VerifiableTask(
                prompt=f"What is {a} ÷ {b}? Give only the numeric answer.",
                correct_answer=a // b,
                verifier=self._verify_number,
                category="arithmetic",
            )
        else:  # mixed
            a, b, c = random.randint(2, 20), random.randint(2, 10), random.randint(1, 10)
            answer = a * b + c
            return VerifiableTask(
                prompt=f"What is {a} × {b} + {c}? Give only the numeric answer.",
                correct_answer=answer,
                verifier=self._verify_number,
                category="arithmetic",
            )

    @staticmethod
    def _verify_number(response: str, correct: int) -> bool:
        """Extract and verify numeric answer."""
        # Find all numbers in response
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if not numbers:
            return False
        # Check if any number matches (allow for "The answer is 42" style)
        for num_str in numbers:
            try:
                # Handle both int and float
                num = float(num_str)
                if num == correct or int(num) == correct:
                    return True
            except ValueError:
                continue
        return False


class WordProblemGenerator(TaskGenerator):
    """Generate word problems with known answers."""

    TEMPLATES = [
        ("If you have {a} apples and buy {b} more, how many apples do you have in total?", lambda a, b: a + b),
        ("A store has {a} items. If {b} are sold, how many remain?", lambda a, b: a - b),
        ("There are {a} boxes with {b} items each. What is the total number of items?", lambda a, b: a * b),
        ("{a} cookies are shared equally among {b} children. How many does each child get?", lambda a, b: a // b),
        ("A train travels {a} miles in {b} hours. What is its speed in miles per hour?", lambda a, b: a // b),
    ]

    def generate(self, n: int = 10) -> list[VerifiableTask]:
        tasks = []
        for _ in range(n):
            template, solver = random.choice(self.TEMPLATES)
            # Generate appropriate numbers
            if "shared" in template or "speed" in template:
                b = random.randint(2, 10)
                a = b * random.randint(2, 15)
            else:
                a = random.randint(5, 50)
                b = random.randint(2, min(a-1, 20)) if "sold" in template else random.randint(2, 15)

            prompt = template.format(a=a, b=b) + " Give only the numeric answer."
            answer = solver(a, b)

            tasks.append(VerifiableTask(
                prompt=prompt,
                correct_answer=answer,
                verifier=ArithmeticTaskGenerator._verify_number,
                category="word_problem",
            ))
        return tasks


class LogicTaskGenerator(TaskGenerator):
    """Generate simple logic tasks."""

    def generate(self, n: int = 10) -> list[VerifiableTask]:
        tasks = []
        for _ in range(n):
            task_type = random.choice(["and", "or", "implies", "negation"])
            task = self._generate_task(task_type)
            tasks.append(task)
        return tasks

    def _generate_task(self, task_type: str) -> VerifiableTask:
        if task_type == "and":
            a, b = random.choice([True, False]), random.choice([True, False])
            a_str = "True" if a else "False"
            b_str = "True" if b else "False"
            return VerifiableTask(
                prompt=f"In logic, what is {a_str} AND {b_str}? Answer only True or False.",
                correct_answer=a and b,
                verifier=self._verify_bool,
                category="logic",
            )
        elif task_type == "or":
            a, b = random.choice([True, False]), random.choice([True, False])
            a_str = "True" if a else "False"
            b_str = "True" if b else "False"
            return VerifiableTask(
                prompt=f"In logic, what is {a_str} OR {b_str}? Answer only True or False.",
                correct_answer=a or b,
                verifier=self._verify_bool,
                category="logic",
            )
        elif task_type == "implies":
            a, b = random.choice([True, False]), random.choice([True, False])
            a_str = "True" if a else "False"
            b_str = "True" if b else "False"
            # P → Q is equivalent to (not P) or Q
            result = (not a) or b
            return VerifiableTask(
                prompt=f"In logic, if P={a_str} and Q={b_str}, what is P IMPLIES Q (P → Q)? Answer only True or False.",
                correct_answer=result,
                verifier=self._verify_bool,
                category="logic",
            )
        else:  # negation
            a = random.choice([True, False])
            a_str = "True" if a else "False"
            return VerifiableTask(
                prompt=f"In logic, what is NOT {a_str}? Answer only True or False.",
                correct_answer=not a,
                verifier=self._verify_bool,
                category="logic",
            )

    @staticmethod
    def _verify_bool(response: str, correct: bool) -> bool:
        """Verify boolean answer."""
        response_lower = response.lower()
        if correct:
            return "true" in response_lower and "false" not in response_lower.replace("true", "")
        else:
            return "false" in response_lower and "true" not in response_lower.replace("false", "")


class ComparisonTaskGenerator(TaskGenerator):
    """Generate comparison tasks."""

    def generate(self, n: int = 10) -> list[VerifiableTask]:
        tasks = []
        for _ in range(n):
            a, b = random.randint(1, 100), random.randint(1, 100)
            while a == b:
                b = random.randint(1, 100)

            task_type = random.choice(["greater", "less", "equal_check"])

            if task_type == "greater":
                tasks.append(VerifiableTask(
                    prompt=f"Is {a} greater than {b}? Answer only Yes or No.",
                    correct_answer=a > b,
                    verifier=self._verify_yesno,
                    category="comparison",
                ))
            elif task_type == "less":
                tasks.append(VerifiableTask(
                    prompt=f"Is {a} less than {b}? Answer only Yes or No.",
                    correct_answer=a < b,
                    verifier=self._verify_yesno,
                    category="comparison",
                ))
            else:
                # Sometimes make them equal
                if random.random() < 0.3:
                    b = a
                tasks.append(VerifiableTask(
                    prompt=f"Is {a} equal to {b}? Answer only Yes or No.",
                    correct_answer=a == b,
                    verifier=self._verify_yesno,
                    category="comparison",
                ))
        return tasks

    @staticmethod
    def _verify_yesno(response: str, correct: bool) -> bool:
        """Verify yes/no answer."""
        response_lower = response.lower()
        if correct:
            return "yes" in response_lower and "no" not in response_lower.split("yes")[0]
        else:
            return "no" in response_lower and "yes" not in response_lower.split("no")[0]


class VerifiableTaskSuite:
    """Suite of verifiable tasks for evolution."""

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

        self.generators = {
            "arithmetic": ArithmeticTaskGenerator(),
            "word_problem": WordProblemGenerator(),
            "logic": LogicTaskGenerator(),
            "comparison": ComparisonTaskGenerator(),
        }

    def generate_batch(self, n: int = 20, categories: list[str] | None = None) -> list[VerifiableTask]:
        """Generate a mixed batch of tasks."""
        if categories is None:
            categories = list(self.generators.keys())

        tasks = []
        per_category = max(1, n // len(categories))

        for cat in categories:
            if cat in self.generators:
                tasks.extend(self.generators[cat].generate(per_category))

        random.shuffle(tasks)
        return tasks[:n]

    def evaluate_response(self, task: VerifiableTask, response: str) -> bool:
        """Check if response is correct for task."""
        return task.verifier(response, task.correct_answer)

    def batch_evaluate(self, tasks: list[VerifiableTask], responses: list[str]) -> tuple[int, int]:
        """Evaluate a batch of responses. Returns (correct, total)."""
        correct = sum(
            self.evaluate_response(task, resp)
            for task, resp in zip(tasks, responses)
        )
        return correct, len(tasks)


def create_task_suite(seed: int | None = None) -> VerifiableTaskSuite:
    """Create a verifiable task suite."""
    return VerifiableTaskSuite(seed)


# Quick test
if __name__ == "__main__":
    suite = create_task_suite(seed=42)
    tasks = suite.generate_batch(n=10)

    print("Sample tasks:")
    for task in tasks[:5]:
        print(f"  [{task.category}] {task.prompt}")
        print(f"    Answer: {task.correct_answer}")
