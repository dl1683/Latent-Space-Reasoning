"""
FractalGrammar - Complete grammar combining rules and tree structure.

The FractalGrammar is the main abstraction for fractal latent grammars.
It combines:
- RuleBank: Collection of contractive transforms (the vocabulary)
- GrammarTree: AND/OR tree structure (the composition)

Together they define a generative process for latent vectors:
1. Start with a seed latent
2. Expand the tree recursively
3. Each LEAF applies a rule, AND/OR nodes compose results

Key Properties:
- Compression: Complex latents from simple rule composition
- Interpretability: Tree structure reveals generation process
- Evolvability: Can mutate both structure and parameters

Usage:
    >>> grammar = FractalGrammar.random(config, latent_dim=1024)
    >>> latent = grammar.expand(seed)  # Generate latent
    >>> mutated = grammar.mutate(temperature=0.1)  # Evolve
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import random

import torch
from torch import Tensor

from latent_reasoning.grammar.rules import RuleBank
from latent_reasoning.grammar.tree import GrammarTree, GrammarNode, NodeType

if TYPE_CHECKING:
    from latent_reasoning.config import GrammarConfig


@dataclass
class GrammarStats:
    """Statistics about a grammar's structure."""
    num_rules: int
    num_nodes: int
    num_leaves: int
    max_depth: int
    rules_used: int
    compression_ratio: float
    attractor_coverage: float


class FractalGrammar:
    """
    Complete fractal grammar combining rules and tree.

    The FractalGrammar generates latent vectors by:
    1. Expanding the tree structure
    2. Applying rules at LEAF nodes
    3. Composing via AND (average) and OR (gating)

    Attributes:
        rule_bank: Collection of grammar rules
        tree: AND/OR tree structure
        latent_dim: Dimension of latent vectors
        device: Device for computations

    Args:
        rule_bank: RuleBank with contractive rules
        tree: GrammarTree defining composition
        latent_dim: Dimension of latent space
        device: torch device

    Usage:
        >>> grammar = FractalGrammar.random(config, latent_dim=1024)
        >>> z = grammar.expand(seed)  # Expand tree to get latent
        >>> print(grammar.compression_ratio)  # How compressed is the grammar
    """

    def __init__(
        self,
        rule_bank: RuleBank,
        tree: GrammarTree,
        latent_dim: int,
        device: torch.device | str = "cpu",
    ):
        self.rule_bank = rule_bank
        self.tree = tree
        self.latent_dim = latent_dim
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Move to device
        self.rule_bank = self.rule_bank.to(device)
        self.tree = self.tree.to(device)

        # Cached attractors for seed generation
        self._cached_attractors: list[Tensor] | None = None

    def expand(
        self,
        seed: Tensor | None = None,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Expand the grammar tree to generate a latent vector.

        Args:
            seed: Input seed latent (uses attractor blend if None)
            temperature: Softmax temperature for OR nodes

        Returns:
            Generated latent vector
        """
        if seed is None:
            seed = self._get_seed_from_attractors()

        # Convert to device and dtype compatible with rule bank
        seed = seed.to(self.device)
        # Ensure seed dtype matches rule bank (grammar rules are float32)
        if seed.dtype != torch.float32:
            seed = seed.float()

        result = self.tree.expand(seed, self.rule_bank, temperature)

        # Return in original dtype if needed
        return result

    def expand_with_trace(
        self,
        seed: Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[Tensor, list[dict]]:
        """
        Expand grammar and return trace of node activations.

        Useful for interpretability and debugging.

        Args:
            seed: Input seed latent
            temperature: Softmax temperature

        Returns:
            (latent, trace) where trace is list of node info dicts
        """
        if seed is None:
            seed = self._get_seed_from_attractors()

        seed = seed.to(self.device)
        trace = []

        def expand_with_trace_inner(node: GrammarNode, z: Tensor, path: list[int]) -> Tensor:
            """Recursive expansion with tracing."""
            if node.node_type == NodeType.LEAF:
                result = self.rule_bank.apply(node.rule_idx, z)
                trace.append({
                    "path": path.copy(),
                    "type": "LEAF",
                    "rule_idx": node.rule_idx,
                    "input_norm": z.norm().item(),
                    "output_norm": result.norm().item(),
                })
                return result

            elif node.node_type == NodeType.AND:
                if not node.children:
                    return z

                child_outputs = []
                for i, child in enumerate(node.children):
                    out = expand_with_trace_inner(child, z, path + [i])
                    child_outputs.append(out)

                # Get mixing weights
                if node.alpha is None:
                    alpha = torch.ones(len(node.children), device=z.device) / len(node.children)
                else:
                    alpha = torch.softmax(node.alpha.to(z.device), dim=0)

                result = torch.zeros_like(child_outputs[0])
                for w, output in zip(alpha, child_outputs):
                    result = result + w * output

                trace.append({
                    "path": path.copy(),
                    "type": "AND",
                    "num_children": len(node.children),
                    "alpha": alpha.tolist(),
                    "output_norm": result.norm().item(),
                })
                return result

            elif node.node_type == NodeType.OR:
                if not node.children:
                    return z

                child_outputs = []
                for i, child in enumerate(node.children):
                    out = expand_with_trace_inner(child, z, path + [i])
                    child_outputs.append(out)

                # Get gating weights
                if node.gate is None:
                    gate = torch.ones(len(node.children), device=z.device) / len(node.children)
                else:
                    gate = torch.softmax(node.gate.to(z.device) / temperature, dim=0)

                result = torch.zeros_like(child_outputs[0])
                for w, output in zip(gate, child_outputs):
                    result = result + w * output

                trace.append({
                    "path": path.copy(),
                    "type": "OR",
                    "num_children": len(node.children),
                    "gate": gate.tolist(),
                    "output_norm": result.norm().item(),
                })
                return result

            return z

        result = expand_with_trace_inner(self.tree.root, seed, [])
        return result, trace

    def _get_seed_from_attractors(self) -> Tensor:
        """Generate seed as weighted blend of rule attractors."""
        if self._cached_attractors is None:
            self._cached_attractors = self.rule_bank.get_all_attractors()

        # Use rules referenced in tree
        rules_used = self.tree.rules_used
        if not rules_used:
            # Fallback to random
            return torch.randn(self.latent_dim, device=self.device)

        # Weighted average of attractors for used rules
        attractors = [self._cached_attractors[i] for i in rules_used if i < len(self._cached_attractors)]
        if not attractors:
            return torch.randn(self.latent_dim, device=self.device)

        seed = torch.stack(attractors).mean(dim=0)
        return seed.to(self.device)

    @property
    def compression_ratio(self) -> float:
        """
        Compute compression efficiency: tree_params / latent_dim.

        This measures how efficiently the tree structure compresses
        the grammar representation. Lower values indicate better
        compression (tree is simpler relative to latent space).

        Note: This excludes shared rule bank parameters since they
        are amortized across the population.
        """
        # Count tree-specific parameters only (rules are shared)
        num_tree_params = 0
        for node in self.tree.iter_nodes():
            if node.alpha is not None:
                num_tree_params += node.alpha.numel()
            if node.gate is not None:
                num_tree_params += node.gate.numel()
            if node.node_type == NodeType.LEAF:
                num_tree_params += 1  # rule index

        # Tree structure overhead (rough estimate)
        num_tree_params += self.tree.num_nodes * 2  # node type + depth

        if self.latent_dim == 0:
            return 0.0

        # Return tree overhead per latent dimension
        # Lower is better (simpler tree relative to output)
        return num_tree_params / self.latent_dim

    @property
    def stats(self) -> GrammarStats:
        """Get comprehensive statistics about this grammar."""
        rules_used = self.tree.rules_used

        # Compute attractor coverage (fraction of latent dim covered by attractors)
        if self._cached_attractors is None:
            self._cached_attractors = self.rule_bank.get_all_attractors()

        attractors = [self._cached_attractors[i] for i in rules_used if i < len(self._cached_attractors)]
        if attractors:
            # SVD to measure effective dimensionality
            stacked = torch.stack(attractors)
            if stacked.shape[0] > 1:
                _, s, _ = torch.svd(stacked)
                # Fraction of variance explained by top components
                total_var = (s ** 2).sum()
                if total_var > 0:
                    attractor_coverage = (s[0] ** 2 / total_var).item()
                else:
                    attractor_coverage = 1.0
            else:
                attractor_coverage = 1.0
        else:
            attractor_coverage = 0.0

        return GrammarStats(
            num_rules=self.rule_bank.num_rules,
            num_nodes=self.tree.num_nodes,
            num_leaves=self.tree.num_leaves,
            max_depth=self.tree.max_depth,
            rules_used=len(rules_used),
            compression_ratio=self.compression_ratio,
            attractor_coverage=attractor_coverage,
        )

    def clone(self) -> "FractalGrammar":
        """Create a deep copy of this grammar."""
        return FractalGrammar(
            rule_bank=self.rule_bank.clone(),
            tree=self.tree.clone(),
            latent_dim=self.latent_dim,
            device=self.device,
        )

    def to(self, device: torch.device | str) -> "FractalGrammar":
        """Move grammar to device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.rule_bank = self.rule_bank.to(device)
        self.tree = self.tree.to(device)
        self._cached_attractors = None  # Invalidate cache
        return self

    @classmethod
    def random(
        cls,
        config: "GrammarConfig",
        latent_dim: int,
        device: torch.device | str = "cpu",
    ) -> "FractalGrammar":
        """
        Create a random fractal grammar.

        Args:
            config: Grammar configuration
            latent_dim: Latent space dimension
            device: Device for tensors

        Returns:
            Random FractalGrammar
        """
        if isinstance(device, str):
            device = torch.device(device)

        # Create rule bank
        rule_bank = RuleBank(
            num_rules=config.num_rules,
            latent_dim=latent_dim,
            hidden_dim=config.rule_hidden_dim,
            contraction_factor=config.contraction_factor,
        )

        # Create random tree
        tree = GrammarTree.random(
            config=config,
            num_rules=config.num_rules,
            device=device,
        )

        return cls(
            rule_bank=rule_bank,
            tree=tree,
            latent_dim=latent_dim,
            device=device,
        )

    @classmethod
    def balanced(
        cls,
        config: "GrammarConfig",
        latent_dim: int,
        depth: int = 3,
        branching: int = 2,
        device: torch.device | str = "cpu",
    ) -> "FractalGrammar":
        """
        Create a balanced grammar with specified depth.

        Args:
            config: Grammar configuration
            latent_dim: Latent space dimension
            depth: Tree depth
            branching: Branching factor
            device: Device for tensors

        Returns:
            Balanced FractalGrammar
        """
        if isinstance(device, str):
            device = torch.device(device)

        rule_bank = RuleBank(
            num_rules=config.num_rules,
            latent_dim=latent_dim,
            hidden_dim=config.rule_hidden_dim,
            contraction_factor=config.contraction_factor,
        )

        tree = GrammarTree.balanced(
            depth=depth,
            branching=branching,
            num_rules=config.num_rules,
            device=device,
        )

        return cls(
            rule_bank=rule_bank,
            tree=tree,
            latent_dim=latent_dim,
            device=device,
        )

    @classmethod
    def from_single_rule(
        cls,
        config: "GrammarConfig",
        latent_dim: int,
        rule_idx: int = 0,
        device: torch.device | str = "cpu",
    ) -> "FractalGrammar":
        """
        Create a minimal grammar with a single LEAF node.

        Useful for testing and as a base for evolution.

        Args:
            config: Grammar configuration
            latent_dim: Latent space dimension
            rule_idx: Rule to use at the LEAF
            device: Device for tensors

        Returns:
            Single-rule FractalGrammar
        """
        if isinstance(device, str):
            device = torch.device(device)

        rule_bank = RuleBank(
            num_rules=config.num_rules,
            latent_dim=latent_dim,
            hidden_dim=config.rule_hidden_dim,
            contraction_factor=config.contraction_factor,
        )

        # Single LEAF tree
        root = GrammarNode(
            node_type=NodeType.LEAF,
            rule_idx=rule_idx,
            depth=0,
        )
        tree = GrammarTree(root=root, device=device)

        return cls(
            rule_bank=rule_bank,
            tree=tree,
            latent_dim=latent_dim,
            device=device,
        )

    def to_dict(self) -> dict:
        """Convert grammar to dictionary for serialization."""
        return {
            "latent_dim": self.latent_dim,
            "rule_bank": {
                "num_rules": self.rule_bank.num_rules,
                "latent_dim": self.rule_bank.latent_dim,
                "hidden_dim": self.rule_bank.hidden_dim,
                "contraction_factor": self.rule_bank.contraction_factor,
                "state_dict": {k: v.tolist() for k, v in self.rule_bank.state_dict().items()},
            },
            "tree": self.tree.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict, device: str = "cpu") -> "FractalGrammar":
        """Create grammar from dictionary."""
        # Recreate rule bank
        rb_data = data["rule_bank"]
        rule_bank = RuleBank(
            num_rules=rb_data["num_rules"],
            latent_dim=rb_data["latent_dim"],
            hidden_dim=rb_data.get("hidden_dim"),
            contraction_factor=rb_data.get("contraction_factor", 0.9),
        )

        # Load state dict
        state_dict = {k: torch.tensor(v) for k, v in rb_data["state_dict"].items()}
        rule_bank.load_state_dict(state_dict)

        # Recreate tree
        tree = GrammarTree.from_dict(data["tree"], device=device)

        return cls(
            rule_bank=rule_bank,
            tree=tree,
            latent_dim=data["latent_dim"],
            device=device,
        )

    def __repr__(self) -> str:
        stats = self.stats
        return (
            f"FractalGrammar(latent_dim={self.latent_dim}, "
            f"rules={stats.rules_used}/{stats.num_rules}, "
            f"nodes={stats.num_nodes}, "
            f"depth={stats.max_depth}, "
            f"compression={stats.compression_ratio:.2f}x)"
        )
