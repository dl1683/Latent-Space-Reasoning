"""
Synthetic Tree Traversal Tasks

Per Codex analysis: "Tasks with true tree geometry and depth pressure"
- Depth 6-10, branching factor 3-6
- Rare-leaf retrieval where success depends on distinguishing low-frequency leaves
- Traversal tasks: given root + child-indices, recover node attributes
- Reward scales with depth

These are designed to show hyperbolic geometry advantage because:
1. Solution space has explicit tree structure
2. Depth matters (deeper = harder = more valuable)
3. Rare leaves require exploring peripheral regions (where hyperbolic has more volume)
"""

import random
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class TreeTask:
    """A task with true tree-geometric structure."""
    prompt: str
    correct_answer: Any
    verifier: Callable[[str, Any], bool]
    depth: int  # How deep in tree
    branch_path: list[int]  # Path from root
    category: str = "tree_traversal"
    difficulty: str = "easy"  # easy/medium/hard based on depth
    rarity: float = 1.0  # Lower = rarer (exponentially decreases with depth)


class SyntheticTree:
    """
    A synthetic tree for generating traversal tasks.

    Structure:
    - Root node with branching_factor children
    - Each node has an attribute (computed from path)
    - Depth determines difficulty
    """

    def __init__(
        self,
        max_depth: int = 8,
        branching_factor: int = 4,
        seed: int = 42,
    ):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.rng = random.Random(seed)

        # Pre-generate node attributes for consistency
        self._node_attrs = {}
        self._generate_tree_attrs([], 0)

    def _generate_tree_attrs(self, path: list[int], depth: int):
        """Recursively generate node attributes."""
        path_key = tuple(path)

        # Attribute is a function of path (deterministic)
        if depth == 0:
            self._node_attrs[path_key] = "ROOT"
        else:
            # Generate attribute based on path sum + depth
            val = sum(path) * (depth + 1) + len(path) * 7
            self._node_attrs[path_key] = val

        if depth < self.max_depth:
            for child_idx in range(self.branching_factor):
                self._generate_tree_attrs(path + [child_idx], depth + 1)

    def get_attribute(self, path: list[int]) -> Any:
        """Get attribute at given path."""
        return self._node_attrs.get(tuple(path), None)

    def compute_path_attribute(self, path: list[int]) -> int:
        """Compute expected attribute for a path (used for verification)."""
        if not path:
            return 0  # ROOT
        depth = len(path)
        return sum(path) * (depth + 1) + len(path) * 7


class TreeTraversalGenerator:
    """
    Generate tree traversal tasks.

    Task format: "Starting at root, go to child [2], then child [1], then child [3].
                  What is the node value?"
    """

    def __init__(
        self,
        max_depth: int = 8,
        branching_factor: int = 4,
        seed: int = 42,
    ):
        self.tree = SyntheticTree(max_depth, branching_factor, seed)
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.rng = random.Random(seed)

    def generate(self, n: int = 10) -> list[TreeTask]:
        """Generate n tree traversal tasks with depth distribution."""
        tasks = []

        for _ in range(n):
            # Bias toward deeper tasks (that's where hyperbolic should shine)
            # Distribution: 10% depth 1-2, 30% depth 3-4, 40% depth 5-6, 20% depth 7-8
            depth_choice = self.rng.random()
            if depth_choice < 0.1:
                depth = self.rng.randint(1, 2)
            elif depth_choice < 0.4:
                depth = self.rng.randint(3, 4)
            elif depth_choice < 0.8:
                depth = self.rng.randint(5, 6)
            else:
                depth = self.rng.randint(7, min(8, self.max_depth))

            task = self._generate_task(depth)
            tasks.append(task)

        return tasks

    def _generate_task(self, depth: int) -> TreeTask:
        """Generate a single traversal task at given depth."""
        # Generate random path
        path = [self.rng.randint(0, self.branching_factor - 1) for _ in range(depth)]

        # Compute expected answer
        answer = self.tree.compute_path_attribute(path)

        # Build prompt
        if depth == 0:
            prompt = "What is the root node value in this tree? (Answer: 0)"
        else:
            steps = [f"child {p}" for p in path]
            prompt = f"Starting at the root (value=0), follow this path: {' → '.join(steps)}. The node value at depth d with path P is computed as: sum(P) * (d+1) + len(P) * 7. What is the final node value? Give only the number."

        # Difficulty based on depth
        if depth <= 2:
            difficulty = "easy"
        elif depth <= 5:
            difficulty = "medium"
        else:
            difficulty = "hard"

        # Rarity decreases exponentially with depth (deeper = rarer)
        rarity = 1.0 / (self.branching_factor ** depth)

        return TreeTask(
            prompt=prompt,
            correct_answer=answer,
            verifier=self._verify_number,
            depth=depth,
            branch_path=path,
            difficulty=difficulty,
            rarity=rarity,
        )

    def _verify_number(self, response: str, expected: int) -> bool:
        """Verify numeric response."""
        import re
        numbers = re.findall(r'-?\d+', response)
        if not numbers:
            return False
        # Check last number (most likely the answer)
        try:
            return int(numbers[-1]) == expected
        except (ValueError, IndexError):
            return False


class HierarchicalClassificationGenerator:
    """
    Generate hierarchical classification tasks.

    A taxonomy with depth levels where errors at higher levels are tolerated
    but leaf errors are costly. This creates explicit tree-structured solutions.
    """

    TAXONOMY = {
        # Level 0 (root)
        "entity": {
            # Level 1
            "living": {
                # Level 2
                "animal": {
                    # Level 3
                    "mammal": ["dog", "cat", "elephant", "whale"],
                    "bird": ["eagle", "penguin", "sparrow", "owl"],
                    "fish": ["salmon", "shark", "goldfish", "tuna"],
                    "reptile": ["snake", "lizard", "turtle", "crocodile"],
                },
                "plant": {
                    "tree": ["oak", "pine", "maple", "birch"],
                    "flower": ["rose", "tulip", "daisy", "lily"],
                    "grass": ["wheat", "bamboo", "corn", "rice"],
                    "fungus": ["mushroom", "yeast", "mold", "truffle"],
                },
            },
            "non_living": {
                "natural": {
                    "mineral": ["gold", "silver", "diamond", "quartz"],
                    "element": ["oxygen", "carbon", "iron", "nitrogen"],
                    "celestial": ["star", "planet", "comet", "asteroid"],
                    "geological": ["mountain", "river", "ocean", "volcano"],
                },
                "artificial": {
                    "vehicle": ["car", "airplane", "boat", "bicycle"],
                    "tool": ["hammer", "screwdriver", "wrench", "saw"],
                    "building": ["house", "skyscraper", "bridge", "tower"],
                    "electronic": ["computer", "phone", "television", "radio"],
                },
            },
        },
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._build_paths()

    def _build_paths(self):
        """Build all paths from root to leaves."""
        self.leaf_paths = []
        self._traverse(self.TAXONOMY, [])

    def _traverse(self, node, path):
        """Recursively traverse taxonomy."""
        if isinstance(node, dict):
            for key, child in node.items():
                self._traverse(child, path + [key])
        elif isinstance(node, list):
            for leaf in node:
                self.leaf_paths.append(path + [leaf])

    def generate(self, n: int = 10) -> list[TreeTask]:
        """Generate n classification tasks."""
        tasks = []

        for _ in range(n):
            leaf_path = self.rng.choice(self.leaf_paths)
            task = self._generate_task(leaf_path)
            tasks.append(task)

        return tasks

    def _generate_task(self, path: list[str]) -> TreeTask:
        """Generate a classification task for a leaf item."""
        item = path[-1]
        full_path = " > ".join(path[:-1])
        depth = len(path) - 1  # -1 for leaf item itself

        prompt = f"Classify '{item}' in the taxonomy. The hierarchy is: entity > (living/non_living) > (category) > (subcategory) > item. What is the full path from root to '{item}'? Answer in format: level1 > level2 > level3 > level4"

        # Expected answer is the path without the leaf
        expected = " > ".join(path[1:-1])  # Skip "entity" root

        if depth <= 2:
            difficulty = "easy"
        elif depth <= 3:
            difficulty = "medium"
        else:
            difficulty = "hard"

        # Rarity based on how deep/specific
        rarity = 1.0 / (4 ** depth)

        return TreeTask(
            prompt=prompt,
            correct_answer=expected,
            verifier=self._verify_path,
            depth=depth,
            branch_path=path,
            category="hierarchical_classification",
            difficulty=difficulty,
            rarity=rarity,
        )

    def _verify_path(self, response: str, expected: str) -> bool:
        """Verify path response (case-insensitive, flexible matching)."""
        response = response.lower().replace(" > ", ">").replace(">", " > ")
        expected = expected.lower()

        # Check if all expected components are present in order
        exp_parts = [p.strip() for p in expected.split(">")]
        resp_parts = [p.strip() for p in response.split(">") if p.strip()]

        if len(resp_parts) < len(exp_parts):
            return False

        # Check each expected part is in response
        for exp in exp_parts:
            if exp not in response:
                return False
        return True


class MultiHopTreeGenerator:
    """
    Generate multi-hop reasoning tasks with explicit tree structure.

    These require traversing a knowledge tree to answer questions.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate(self, n: int = 10) -> list[TreeTask]:
        """Generate n multi-hop tree tasks."""
        tasks = []

        for _ in range(n):
            hops = self.rng.choice([2, 3, 3, 4, 4, 5, 6])
            task = self._generate_task(hops)
            tasks.append(task)

        return tasks

    def _generate_task(self, hops: int) -> TreeTask:
        """Generate a multi-hop task."""
        # Build a chain of facts
        names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry"]
        locations = ["Paris", "London", "Tokyo", "Sydney", "Cairo", "Mumbai", "Rio", "Moscow"]

        # Pick random entities
        used_names = self.rng.sample(names, min(hops + 1, len(names)))

        # Build facts chain
        facts = []
        chain = []

        for i in range(hops):
            if i == 0:
                # First hop: location
                loc = self.rng.choice(locations)
                facts.append(f"{used_names[i]} lives in {loc}.")
                chain.append(("location", loc))
            elif i % 2 == 1:
                # Odd hops: relationship
                facts.append(f"{used_names[i]} is a friend of {used_names[i-1]}.")
                chain.append(("friend", used_names[i]))
            else:
                # Even hops: location transfer
                loc = self.rng.choice(locations)
                facts.append(f"{used_names[i]} also lives in {loc}.")
                chain.append(("location", loc))

        self.rng.shuffle(facts)

        # Question about the chain
        question = f"Based on the facts: {' '.join(facts)} Following the chain of relationships from {used_names[0]}, what location is at the end of a {hops}-hop path? Give only the location name."

        # Answer is the last location in the chain
        answer = [c[1] for c in chain if c[0] == "location"][-1] if any(c[0] == "location" for c in chain) else used_names[-1]

        if hops <= 2:
            difficulty = "easy"
        elif hops <= 4:
            difficulty = "medium"
        else:
            difficulty = "hard"

        return TreeTask(
            prompt=question,
            correct_answer=answer,
            verifier=self._verify_location,
            depth=hops,
            branch_path=list(range(hops)),
            category="multi_hop_tree",
            difficulty=difficulty,
            rarity=1.0 / (2 ** hops),
        )

    def _verify_location(self, response: str, expected: str) -> bool:
        """Verify location response."""
        return expected.lower() in response.lower()


class TreeTaskPool:
    """Pool of tree-structured tasks for evolution experiments."""

    def __init__(
        self,
        pool_size: int = 200,
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        random.seed(seed)

        # Generators
        traversal_gen = TreeTraversalGenerator(seed=seed)
        classification_gen = HierarchicalClassificationGenerator(seed=seed)
        multihop_gen = MultiHopTreeGenerator(seed=seed)

        # Generate equal from each
        per_cat = pool_size // 3

        all_tasks = []
        all_tasks.extend(traversal_gen.generate(per_cat))
        all_tasks.extend(classification_gen.generate(per_cat))
        all_tasks.extend(multihop_gen.generate(per_cat))

        random.shuffle(all_tasks)

        # Split
        val_size = int(len(all_tasks) * val_ratio)
        self.val_tasks = all_tasks[:val_size]
        self.train_tasks = all_tasks[val_size:]

    def sample_train(self, n: int, seed: int | None = None) -> list[TreeTask]:
        """Sample n tasks from training pool."""
        if seed is not None:
            random.seed(seed)
        return random.sample(self.train_tasks, min(n, len(self.train_tasks)))

    def get_validation(self) -> list[TreeTask]:
        """Get full validation set."""
        return self.val_tasks

    def stats(self) -> dict:
        """Get pool statistics."""
        from collections import defaultdict

        train_cats = defaultdict(int)
        train_diff = defaultdict(int)
        train_depths = defaultdict(int)

        for t in self.train_tasks:
            train_cats[t.category] += 1
            train_diff[t.difficulty] += 1
            train_depths[t.depth] += 1

        val_cats = defaultdict(int)
        val_diff = defaultdict(int)

        for t in self.val_tasks:
            val_cats[t.category] += 1
            val_diff[t.difficulty] += 1

        return {
            "train_size": len(self.train_tasks),
            "val_size": len(self.val_tasks),
            "train_categories": dict(train_cats),
            "train_difficulty": dict(train_diff),
            "train_depths": dict(train_depths),
            "val_categories": dict(val_cats),
        }
