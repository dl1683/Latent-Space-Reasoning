"""
AND/OR tree structures for fractal latent grammars.

The grammar tree defines how rules are composed to generate latent vectors:
- LEAF nodes: Apply a single rule
- AND nodes: Combine children via weighted average
- OR nodes: Select best child via gating

The tree provides hierarchical composition of simple rules into complex
latent generation strategies.

Key Properties:
- Recursive: Trees can have arbitrary depth
- Compositional: Complex behaviors from simple rules
- Differentiable: All operations support gradients
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Iterator
import random

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from latent_reasoning.grammar.rules import RuleBank
    from latent_reasoning.config import GrammarConfig


class NodeType(Enum):
    """Types of nodes in the grammar tree."""
    LEAF = "leaf"  # Apply a rule
    AND = "and"    # Combine children (weighted average)
    OR = "or"      # Select best child (gating)


@dataclass
class GrammarNode:
    """
    A single node in the grammar tree.

    Node Types:
    - LEAF: Applies rule_idx to input, returns transformed latent
    - AND: Computes weighted average of children outputs
    - OR: Uses gating to select/blend children outputs

    Attributes:
        node_type: Type of node (LEAF, AND, OR)
        rule_idx: For LEAF nodes, which rule to apply
        children: For AND/OR nodes, list of child nodes
        alpha: For AND nodes, mixing weights for children
        gate: For OR nodes, gating weights
        depth: Depth of this node in the tree
    """
    node_type: NodeType
    rule_idx: int = 0  # For LEAF nodes
    children: list["GrammarNode"] = field(default_factory=list)  # For AND/OR
    alpha: Tensor | None = None  # Mixing weights for AND
    gate: Tensor | None = None  # Gating weights for OR
    depth: int = 0

    def expand(
        self,
        z: Tensor,
        rule_bank: "RuleBank",
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Recursively expand this node to produce a latent vector.

        Args:
            z: Input latent vector
            rule_bank: Bank of grammar rules
            temperature: Softmax temperature for OR gating

        Returns:
            Output latent vector
        """
        if self.node_type == NodeType.LEAF:
            # Apply the rule
            return rule_bank.apply(self.rule_idx, z)

        elif self.node_type == NodeType.AND:
            # Compute weighted average of children
            if not self.children:
                return z

            child_outputs = [
                child.expand(z, rule_bank, temperature)
                for child in self.children
            ]

            # Get or create mixing weights
            if self.alpha is None:
                # Equal weights by default
                alpha = torch.ones(len(self.children), device=z.device) / len(self.children)
            else:
                alpha = F.softmax(self.alpha.to(z.device), dim=0)

            # Weighted average
            result = torch.zeros_like(child_outputs[0])
            for w, output in zip(alpha, child_outputs):
                result = result + w * output
            return result

        elif self.node_type == NodeType.OR:
            # Gated selection of children
            if not self.children:
                return z

            child_outputs = [
                child.expand(z, rule_bank, temperature)
                for child in self.children
            ]

            # Get or create gating weights
            if self.gate is None:
                # Equal weights by default
                gate = torch.ones(len(self.children), device=z.device) / len(self.children)
            else:
                gate = F.softmax(self.gate.to(z.device) / temperature, dim=0)

            # Soft gating (weighted selection)
            result = torch.zeros_like(child_outputs[0])
            for w, output in zip(gate, child_outputs):
                result = result + w * output
            return result

        return z

    def count_nodes(self) -> int:
        """Count total nodes in subtree."""
        count = 1
        for child in self.children:
            count += child.count_nodes()
        return count

    def count_leaves(self) -> int:
        """Count leaf nodes in subtree."""
        if self.node_type == NodeType.LEAF:
            return 1
        return sum(child.count_leaves() for child in self.children)

    def get_max_depth(self) -> int:
        """Get maximum depth in subtree."""
        if not self.children:
            return self.depth
        return max(child.get_max_depth() for child in self.children)

    def get_rules_used(self) -> set[int]:
        """Get set of rule indices used in subtree."""
        if self.node_type == NodeType.LEAF:
            return {self.rule_idx}
        rules = set()
        for child in self.children:
            rules.update(child.get_rules_used())
        return rules

    def iter_nodes(self) -> Iterator["GrammarNode"]:
        """Iterate over all nodes in subtree (preorder)."""
        yield self
        for child in self.children:
            yield from child.iter_nodes()

    def clone(self, device: torch.device | None = None) -> "GrammarNode":
        """Create a deep copy of this node and subtree."""
        new_children = [child.clone(device) for child in self.children]

        new_alpha = None
        if self.alpha is not None:
            new_alpha = self.alpha.clone()
            if device is not None:
                new_alpha = new_alpha.to(device)

        new_gate = None
        if self.gate is not None:
            new_gate = self.gate.clone()
            if device is not None:
                new_gate = new_gate.to(device)

        return GrammarNode(
            node_type=self.node_type,
            rule_idx=self.rule_idx,
            children=new_children,
            alpha=new_alpha,
            gate=new_gate,
            depth=self.depth,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "node_type": self.node_type.value,
            "rule_idx": self.rule_idx,
            "children": [c.to_dict() for c in self.children],
            "alpha": self.alpha.tolist() if self.alpha is not None else None,
            "gate": self.gate.tolist() if self.gate is not None else None,
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GrammarNode":
        """Create from dictionary."""
        alpha = None
        if data.get("alpha") is not None:
            alpha = torch.tensor(data["alpha"])

        gate = None
        if data.get("gate") is not None:
            gate = torch.tensor(data["gate"])

        return cls(
            node_type=NodeType(data["node_type"]),
            rule_idx=data.get("rule_idx", 0),
            children=[cls.from_dict(c) for c in data.get("children", [])],
            alpha=alpha,
            gate=gate,
            depth=data.get("depth", 0),
        )

    def __repr__(self) -> str:
        if self.node_type == NodeType.LEAF:
            return f"LEAF(rule={self.rule_idx})"
        else:
            return f"{self.node_type.value.upper()}({len(self.children)} children)"


class GrammarTree:
    """
    Complete grammar tree with root node.

    Provides methods for tree manipulation, expansion, and analysis.

    Args:
        root: Root node of the tree
        device: Device for tensor operations

    Usage:
        >>> tree = GrammarTree.random(config, num_rules=8)
        >>> latent = tree.expand(seed, rule_bank)
        >>> mutated = tree.clone()
    """

    def __init__(
        self,
        root: GrammarNode,
        device: torch.device | str = "cpu",
    ):
        self.root = root
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def expand(
        self,
        z: Tensor,
        rule_bank: "RuleBank",
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Expand the tree to produce a latent vector.

        Args:
            z: Input/seed latent vector
            rule_bank: Bank of grammar rules
            temperature: Softmax temperature for OR nodes

        Returns:
            Output latent vector
        """
        return self.root.expand(z.to(self.device), rule_bank, temperature)

    @property
    def num_nodes(self) -> int:
        """Total number of nodes in tree."""
        return self.root.count_nodes()

    @property
    def num_leaves(self) -> int:
        """Number of leaf nodes."""
        return self.root.count_leaves()

    @property
    def max_depth(self) -> int:
        """Maximum depth of tree."""
        return self.root.get_max_depth()

    @property
    def rules_used(self) -> set[int]:
        """Set of rule indices used."""
        return self.root.get_rules_used()

    def get_node_at_path(self, path: list[int]) -> GrammarNode | None:
        """
        Get node at given path (list of child indices).

        Args:
            path: List of child indices from root

        Returns:
            Node at path, or None if invalid
        """
        node = self.root
        for idx in path:
            if idx >= len(node.children):
                return None
            node = node.children[idx]
        return node

    def iter_nodes(self) -> Iterator[GrammarNode]:
        """Iterate over all nodes."""
        return self.root.iter_nodes()

    def clone(self) -> "GrammarTree":
        """Create a deep copy of this tree."""
        return GrammarTree(
            root=self.root.clone(self.device),
            device=self.device,
        )

    def to(self, device: torch.device | str) -> "GrammarTree":
        """Move tree parameters to device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        # Move all alpha/gate tensors
        for node in self.iter_nodes():
            if node.alpha is not None:
                node.alpha = node.alpha.to(device)
            if node.gate is not None:
                node.gate = node.gate.to(device)
        return self

    @classmethod
    def random(
        cls,
        config: "GrammarConfig",
        num_rules: int,
        device: torch.device | str = "cpu",
    ) -> "GrammarTree":
        """
        Create a random grammar tree.

        Args:
            config: Grammar configuration
            num_rules: Number of available rules
            device: Device for tensors

        Returns:
            Random GrammarTree
        """
        if isinstance(device, str):
            device = torch.device(device)

        def build_node(depth: int) -> GrammarNode:
            # Decide node type based on depth and config
            if depth >= config.max_depth:
                # Force leaf at max depth
                return GrammarNode(
                    node_type=NodeType.LEAF,
                    rule_idx=random.randint(0, num_rules - 1),
                    depth=depth,
                )

            # Random type based on probabilities
            r = random.random()
            leaf_prob = 1.0 - config.and_prob - config.or_prob

            # Increase leaf probability with depth
            depth_factor = depth / config.max_depth
            adjusted_leaf_prob = leaf_prob + depth_factor * 0.3

            if r < adjusted_leaf_prob or depth >= config.max_depth - 1:
                return GrammarNode(
                    node_type=NodeType.LEAF,
                    rule_idx=random.randint(0, num_rules - 1),
                    depth=depth,
                )
            elif r < adjusted_leaf_prob + config.and_prob:
                # AND node
                num_children = random.randint(2, config.branching_factor)
                children = [build_node(depth + 1) for _ in range(num_children)]
                alpha = torch.randn(num_children, device=device) * 0.1
                return GrammarNode(
                    node_type=NodeType.AND,
                    children=children,
                    alpha=alpha,
                    depth=depth,
                )
            else:
                # OR node
                num_children = random.randint(2, config.branching_factor)
                children = [build_node(depth + 1) for _ in range(num_children)]
                gate = torch.randn(num_children, device=device) * 0.1
                return GrammarNode(
                    node_type=NodeType.OR,
                    children=children,
                    gate=gate,
                    depth=depth,
                )

        root = build_node(depth=0)
        return cls(root=root, device=device)

    @classmethod
    def balanced(
        cls,
        depth: int,
        branching: int,
        num_rules: int,
        device: torch.device | str = "cpu",
    ) -> "GrammarTree":
        """
        Create a balanced tree with specified depth.

        Args:
            depth: Tree depth
            branching: Branching factor
            num_rules: Number of rules
            device: Device for tensors

        Returns:
            Balanced GrammarTree
        """
        if isinstance(device, str):
            device = torch.device(device)

        def build_balanced(d: int) -> GrammarNode:
            if d >= depth:
                return GrammarNode(
                    node_type=NodeType.LEAF,
                    rule_idx=random.randint(0, num_rules - 1),
                    depth=d,
                )

            # Alternate AND/OR
            if d % 2 == 0:
                children = [build_balanced(d + 1) for _ in range(branching)]
                return GrammarNode(
                    node_type=NodeType.AND,
                    children=children,
                    alpha=torch.randn(branching, device=device) * 0.1,
                    depth=d,
                )
            else:
                children = [build_balanced(d + 1) for _ in range(branching)]
                return GrammarNode(
                    node_type=NodeType.OR,
                    children=children,
                    gate=torch.randn(branching, device=device) * 0.1,
                    depth=d,
                )

        return cls(root=build_balanced(0), device=device)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "root": self.root.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict, device: str = "cpu") -> "GrammarTree":
        """Create from dictionary."""
        return cls(
            root=GrammarNode.from_dict(data["root"]),
            device=device,
        )

    def __repr__(self) -> str:
        return (
            f"GrammarTree(nodes={self.num_nodes}, "
            f"leaves={self.num_leaves}, "
            f"depth={self.max_depth})"
        )
