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


class NestedArithmeticGenerator(TaskGenerator):
    """
    Generate nested/hierarchical arithmetic tasks.

    These have tree-like structure where hyperbolic should excel.
    Depth levels: 1 (simple), 2 (nested), 3 (deeply nested)
    """

    def generate(self, n: int = 10) -> list[VerifiableTask]:
        tasks = []
        for _ in range(n):
            depth = random.choice([1, 2, 2, 3])  # Bias toward medium depth
            task = self._generate_nested(depth)
            tasks.append(task)
        return tasks

    def _generate_nested(self, depth: int) -> VerifiableTask:
        """Generate nested expression with specified depth."""
        expr, answer = self._build_expr(depth)
        difficulty = ["easy", "medium", "hard"][min(depth - 1, 2)]

        return VerifiableTask(
            prompt=f"Calculate: {expr}. Give only the numeric answer.",
            correct_answer=answer,
            verifier=self._verify_number,
            category="nested_arithmetic",
            difficulty=difficulty,
        )

    def _build_expr(self, depth: int) -> tuple[str, int]:
        """Recursively build nested expression."""
        if depth <= 1:
            # Base case: simple operation
            a, b = random.randint(1, 20), random.randint(1, 10)
            op = random.choice(["+", "-", "*"])
            if op == "+":
                return f"({a} + {b})", a + b
            elif op == "-":
                return f"({a} - {b})", a - b
            else:
                return f"({a} * {b})", a * b
        else:
            # Recursive case: combine sub-expressions
            left_expr, left_val = self._build_expr(depth - 1)
            right_expr, right_val = self._build_expr(depth - 1)

            op = random.choice(["+", "-"])
            if op == "+":
                return f"({left_expr} + {right_expr})", left_val + right_val
            else:
                return f"({left_expr} - {right_expr})", left_val - right_val

    @staticmethod
    def _verify_number(response: str, correct: int) -> bool:
        """Extract and verify numeric answer."""
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if not numbers:
            return False
        for num_str in numbers:
            try:
                num = float(num_str)
                if num == correct or int(num) == correct:
                    return True
            except ValueError:
                continue
        return False


class MultiHopReasoningGenerator(TaskGenerator):
    """
    Generate multi-hop reasoning tasks.

    These require following chains of implications - hierarchical by nature.
    Hyperbolic should excel at maintaining solutions for different chain lengths.
    """

    def generate(self, n: int = 10) -> list[VerifiableTask]:
        tasks = []
        for _ in range(n):
            hops = random.choice([1, 2, 2, 3])  # Bias toward 2-hop
            task_type = random.choice(["chain", "transitive", "conditional"])
            task = self._generate_task(task_type, hops)
            tasks.append(task)
        return tasks

    def _generate_task(self, task_type: str, hops: int) -> VerifiableTask:
        difficulty = ["easy", "medium", "hard"][min(hops - 1, 2)]

        if task_type == "chain":
            return self._generate_chain(hops, difficulty)
        elif task_type == "transitive":
            return self._generate_transitive(hops, difficulty)
        else:
            return self._generate_conditional(hops, difficulty)

    def _generate_chain(self, hops: int, difficulty: str) -> VerifiableTask:
        """Generate numeric chain reasoning."""
        start = random.randint(1, 20)
        current = start
        ops = []

        for _ in range(hops):
            delta = random.randint(1, 10)
            if random.random() < 0.5:
                ops.append(f"add {delta}")
                current += delta
            else:
                ops.append(f"subtract {delta}")
                current -= delta

        prompt = f"Start with {start}. " + ". ".join([f"Then {op}" for op in ops]) + ". What is the result? Give only the numeric answer."

        return VerifiableTask(
            prompt=prompt,
            correct_answer=current,
            verifier=self._verify_number,
            category="multi_hop",
            difficulty=difficulty,
        )

    def _generate_transitive(self, hops: int, difficulty: str) -> VerifiableTask:
        """Generate transitive relation reasoning."""
        names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
        selected = random.sample(names, min(hops + 2, len(names)))

        # Build chain: A > B > C > ...
        relations = []
        for i in range(len(selected) - 1):
            relations.append(f"{selected[i]} is taller than {selected[i+1]}")

        question_pair = (selected[0], selected[-1])
        prompt = ". ".join(relations) + f". Is {question_pair[0]} taller than {question_pair[1]}? Answer only Yes or No."

        return VerifiableTask(
            prompt=prompt,
            correct_answer=True,  # Always yes due to chain construction
            verifier=self._verify_yesno,
            category="multi_hop",
            difficulty=difficulty,
        )

    def _generate_conditional(self, hops: int, difficulty: str) -> VerifiableTask:
        """Generate conditional chain reasoning."""
        conditions = ["it is raining", "the ground is wet", "people use umbrellas",
                      "puddles form", "grass grows", "flowers bloom"]
        selected = random.sample(conditions, min(hops + 1, len(conditions)))

        # Build if-then chain
        rules = []
        for i in range(len(selected) - 1):
            rules.append(f"If {selected[i]}, then {selected[i+1]}")

        premise = selected[0]
        conclusion = selected[-1]
        prompt = ". ".join(rules) + f". Given that {premise}, will {conclusion}? Answer only Yes or No."

        return VerifiableTask(
            prompt=prompt,
            correct_answer=True,
            verifier=self._verify_yesno,
            category="multi_hop",
            difficulty=difficulty,
        )

    @staticmethod
    def _verify_number(response: str, correct: int) -> bool:
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if not numbers:
            return False
        for num_str in numbers:
            try:
                num = float(num_str)
                if num == correct or int(num) == correct:
                    return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _verify_yesno(response: str, correct: bool) -> bool:
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
            "nested_arithmetic": NestedArithmeticGenerator(),
            "multi_hop": MultiHopReasoningGenerator(),
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


class FixedTaskPool:
    """
    Pre-generated fixed task pool with train/validation split.

    Key insight from Codex:
    - Use fixed pool to eliminate inter-generation distribution drift
    - Separate selection (can use stochastic tasks) from measurement (must be stable)
    - Train/validation split enables fair geometry comparison
    """

    def __init__(
        self,
        pool_size: int = 500,
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        """
        Create a fixed task pool.

        Args:
            pool_size: Total number of tasks to pre-generate
            val_ratio: Fraction of tasks for validation (default 0.2 = 100 tasks)
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

        # Create task suite and generate stratified pool
        suite = VerifiableTaskSuite(seed)

        # Generate equal tasks from each category
        categories = list(suite.generators.keys())
        per_category = pool_size // len(categories)

        all_tasks = []
        for cat in categories:
            tasks = suite.generators[cat].generate(per_category)
            all_tasks.extend(tasks)

        # Shuffle for randomization
        random.shuffle(all_tasks)

        # Split into train/val
        val_size = int(len(all_tasks) * val_ratio)
        self.val_tasks = all_tasks[:val_size]
        self.train_tasks = all_tasks[val_size:]

        # Track sampling state for reproducible sampling
        self._train_idx = 0
        self._sample_seed = seed

    def sample_train(self, n: int, seed: int | None = None) -> list[VerifiableTask]:
        """
        Sample n tasks from training pool.

        Uses seeded sampling so same seed gives same tasks.
        """
        if seed is not None:
            random.seed(seed)
        else:
            random.seed(self._sample_seed)
            self._sample_seed += 1

        return random.sample(self.train_tasks, min(n, len(self.train_tasks)))

    def get_validation(self) -> list[VerifiableTask]:
        """Get the full validation set (use for final evaluation)."""
        return self.val_tasks

    def sample_validation(self, n: int, seed: int | None = None) -> list[VerifiableTask]:
        """Sample n tasks from validation pool."""
        if seed is not None:
            random.seed(seed)
        return random.sample(self.val_tasks, min(n, len(self.val_tasks)))

    def evaluate(self, task: VerifiableTask, response: str) -> bool:
        """Check if response is correct for task."""
        return task.verifier(response, task.correct_answer)

    def batch_evaluate(self, tasks: list[VerifiableTask], responses: list[str]) -> tuple[int, int]:
        """Evaluate a batch. Returns (correct, total)."""
        correct = sum(self.evaluate(t, r) for t, r in zip(tasks, responses))
        return correct, len(tasks)

    @property
    def train_size(self) -> int:
        return len(self.train_tasks)

    @property
    def val_size(self) -> int:
        return len(self.val_tasks)

    def stats(self) -> dict:
        """Get pool statistics."""
        train_cats = {}
        for t in self.train_tasks:
            train_cats[t.category] = train_cats.get(t.category, 0) + 1

        val_cats = {}
        for t in self.val_tasks:
            val_cats[t.category] = val_cats.get(t.category, 0) + 1

        return {
            "train_size": len(self.train_tasks),
            "val_size": len(self.val_tasks),
            "train_categories": train_cats,
            "val_categories": val_cats,
        }


def create_fixed_pool(
    pool_size: int = 500,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> FixedTaskPool:
    """Create a fixed task pool with train/validation split."""
    return FixedTaskPool(pool_size=pool_size, val_ratio=val_ratio, seed=seed)


# Quick test
if __name__ == "__main__":
    suite = create_task_suite(seed=42)
    tasks = suite.generate_batch(n=10)

    print("Sample tasks:")
    for task in tasks[:5]:
        print(f"  [{task.category}] {task.prompt}")
        print(f"    Answer: {task.correct_answer}")

    print("\n--- Fixed Pool Test ---")
    pool = create_fixed_pool(pool_size=100, val_ratio=0.2, seed=42)
    print(f"Pool stats: {pool.stats()}")

    train_sample = pool.sample_train(10, seed=1)
    print(f"\nTrain sample (seed=1): {len(train_sample)} tasks")

    train_sample2 = pool.sample_train(10, seed=1)
    print(f"Train sample (seed=1 again): {len(train_sample2)} tasks")
    print(f"Same tasks? {[t.prompt for t in train_sample] == [t.prompt for t in train_sample2]}")
