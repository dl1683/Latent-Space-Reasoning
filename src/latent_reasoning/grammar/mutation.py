"""
Grammar mutation and crossover strategies.

This module provides evolutionary operators for FractalGrammars:
- Structural mutations: Add/remove nodes, change types
- Parametric mutations: Perturb weights, change rules
- Crossover: Exchange subtrees between grammars

Key Design:
- Depth-adaptive: More structural changes early, more parametric later
- Respects constraints: Max depth, min leaves, rule bounds
- Differentiable weights: Parametric mutations preserve gradients
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from latent_reasoning.grammar.tree import GrammarNode, GrammarTree, NodeType

if TYPE_CHECKING:
    from latent_reasoning.grammar.grammar import FractalGrammar
    from latent_reasoning.config import GrammarConfig


@dataclass
class MutationStats:
    """Statistics about mutations applied."""
    structural_mutations: int = 0
    parametric_mutations: int = 0
    node_additions: int = 0
    node_removals: int = 0
    rule_changes: int = 0
    weight_perturbations: int = 0


class GrammarMutationStrategy:
    """
    Mutation strategy for FractalGrammars.

    Implements depth-adaptive mutation where:
    - Early evolution (low generation): More structural changes
    - Late evolution (high generation): More parametric refinement

    Mutation Types:
    1. **Structural** (changes tree topology):
       - Add node: Insert AND/OR node above a LEAF
       - Remove node: Collapse internal node to LEAF
       - Swap type: Change AND ↔ OR

    2. **Parametric** (changes values):
       - Perturb alpha: Modify AND mixing weights
       - Perturb gate: Modify OR gating weights
       - Change rule: Switch LEAF to different rule

    Args:
        config: Grammar configuration
        base_mutation_rate: Base probability of mutation
        structural_decay: How fast structural mutations decay with generation

    Usage:
        >>> strategy = GrammarMutationStrategy(config)
        >>> mutated = strategy.mutate(grammar, generation=10, temperature=0.1)
    """

    def __init__(
        self,
        config: "GrammarConfig",
        base_mutation_rate: float | None = None,
        structural_decay: float | None = None,
    ):
        self.config = config
        # Use config values if not explicitly provided
        self.base_mutation_rate = base_mutation_rate if base_mutation_rate is not None else config.mutation_rate
        self.structural_decay = structural_decay if structural_decay is not None else config.depth_decay
        # Wire additional config fields
        self.structure_mutation_rate = config.structure_mutation_rate
        self.param_mutation_scale = config.param_mutation_scale

    def mutate(
        self,
        grammar: "FractalGrammar",
        generation: int = 0,
        temperature: float = 1.0,
    ) -> "FractalGrammar":
        """
        Mutate a grammar with depth-adaptive strategy.

        Args:
            grammar: Grammar to mutate
            generation: Current generation (affects structural vs parametric ratio)
            temperature: Mutation strength (higher = more change)

        Returns:
            Mutated grammar (new instance)
        """
        from latent_reasoning.grammar.grammar import FractalGrammar

        # Clone grammar
        mutated = grammar.clone()

        # Compute structural vs parametric ratio
        # Early: more structural, Late: more parametric
        structural_ratio = self._compute_structural_ratio(generation)

        # Track mutations
        stats = MutationStats()

        # Decide mutation type
        if random.random() < self.base_mutation_rate * temperature:
            if random.random() < structural_ratio:
                # Structural mutation
                self._structural_mutation(mutated, stats)
            else:
                # Parametric mutation
                self._parametric_mutation(mutated, temperature, stats)

        # Always do some small parametric perturbation
        if random.random() < 0.5:
            self._perturb_weights(mutated, temperature * 0.5, stats)

        # Occasionally mutate rule bank weights for deeper exploration
        # This is key for grammar evolution - without it, rules stay static
        if random.random() < 0.2:  # 20% chance to mutate rules
            self.mutate_rule_bank(mutated, temperature * 0.3)
            # Invalidate grammar's cached attractors since rules changed
            mutated._cached_attractors = None

        return mutated

    def _compute_structural_ratio(self, generation: int) -> float:
        """Compute ratio of structural to parametric mutations."""
        # Use structure_mutation_rate from config as the base
        # Apply exponential decay based on generation
        base_ratio = self.structure_mutation_rate
        decay = (1.0 - self.structural_decay) ** generation
        ratio = base_ratio * (0.5 + 0.5 * decay)  # Range from base*0.5 to base*1.0
        return max(0.1, ratio)  # Keep minimum structural exploration

    def _structural_mutation(
        self,
        grammar: "FractalGrammar",
        stats: MutationStats,
    ) -> None:
        """Apply a structural mutation."""
        mutation_type = random.choice(["add_node", "remove_node", "swap_type"])

        if mutation_type == "add_node":
            self._add_node(grammar, stats)
        elif mutation_type == "remove_node":
            self._remove_node(grammar, stats)
        elif mutation_type == "swap_type":
            self._swap_node_type(grammar, stats)

        stats.structural_mutations += 1

    def _add_node(
        self,
        grammar: "FractalGrammar",
        stats: MutationStats,
    ) -> None:
        """Add an internal node above a random LEAF."""
        # Find all LEAF nodes that aren't at max depth
        candidates = [
            node for node in grammar.tree.iter_nodes()
            if node.node_type == NodeType.LEAF and node.depth < self.config.max_depth - 1
        ]

        if not candidates:
            return

        # Select random LEAF
        target = random.choice(candidates)

        # Replace it with AND or OR containing the original as child
        new_type = random.choice([NodeType.AND, NodeType.OR])
        num_children = random.randint(2, self.config.branching_factor)

        # Create new children (including original rule as one child)
        new_children = [
            GrammarNode(
                node_type=NodeType.LEAF,
                rule_idx=random.randint(0, self.config.num_rules - 1),
                depth=target.depth + 1,
            )
            for _ in range(num_children - 1)
        ]
        # Add original as first child
        original_leaf = GrammarNode(
            node_type=NodeType.LEAF,
            rule_idx=target.rule_idx,
            depth=target.depth + 1,
        )
        new_children.insert(0, original_leaf)

        # Mutate in place
        target.node_type = new_type
        target.children = new_children
        target.rule_idx = 0  # No longer used

        # Initialize weights
        if new_type == NodeType.AND:
            target.alpha = torch.randn(num_children, device=grammar.device) * 0.1
        else:
            target.gate = torch.randn(num_children, device=grammar.device) * 0.1

        stats.node_additions += 1

    def _remove_node(
        self,
        grammar: "FractalGrammar",
        stats: MutationStats,
    ) -> None:
        """Remove an internal node, replacing with a LEAF."""
        # Find AND/OR nodes (not root if it's the only node)
        candidates = [
            node for node in grammar.tree.iter_nodes()
            if node.node_type in (NodeType.AND, NodeType.OR)
        ]

        # Don't remove root if tree would become empty
        if len(candidates) == 1 and candidates[0] is grammar.tree.root:
            # Convert root to LEAF instead
            if grammar.tree.root.children:
                # Use rule from first child if it's a LEAF
                first_child = grammar.tree.root.children[0]
                if first_child.node_type == NodeType.LEAF:
                    rule_idx = first_child.rule_idx
                else:
                    rule_idx = random.randint(0, self.config.num_rules - 1)
            else:
                rule_idx = random.randint(0, self.config.num_rules - 1)

            grammar.tree.root.node_type = NodeType.LEAF
            grammar.tree.root.rule_idx = rule_idx
            grammar.tree.root.children = []
            grammar.tree.root.alpha = None
            grammar.tree.root.gate = None
            stats.node_removals += 1
            return

        if not candidates:
            return

        # Select random internal node
        target = random.choice(candidates)

        # Get a rule from descendants
        rules_used = target.get_rules_used()
        if rules_used:
            rule_idx = random.choice(list(rules_used))
        else:
            rule_idx = random.randint(0, self.config.num_rules - 1)

        # Convert to LEAF
        target.node_type = NodeType.LEAF
        target.rule_idx = rule_idx
        target.children = []
        target.alpha = None
        target.gate = None

        stats.node_removals += 1

    def _swap_node_type(
        self,
        grammar: "FractalGrammar",
        stats: MutationStats,
    ) -> None:
        """Swap AND ↔ OR for a random internal node."""
        candidates = [
            node for node in grammar.tree.iter_nodes()
            if node.node_type in (NodeType.AND, NodeType.OR)
        ]

        if not candidates:
            return

        target = random.choice(candidates)

        if target.node_type == NodeType.AND:
            target.node_type = NodeType.OR
            # Convert alpha to gate
            if target.alpha is not None:
                target.gate = target.alpha.clone()
                target.alpha = None
            else:
                target.gate = torch.randn(len(target.children), device=grammar.device) * 0.1
        else:
            target.node_type = NodeType.AND
            # Convert gate to alpha
            if target.gate is not None:
                target.alpha = target.gate.clone()
                target.gate = None
            else:
                target.alpha = torch.randn(len(target.children), device=grammar.device) * 0.1

        stats.structural_mutations += 1

    def _parametric_mutation(
        self,
        grammar: "FractalGrammar",
        temperature: float,
        stats: MutationStats,
    ) -> None:
        """Apply a parametric mutation."""
        mutation_type = random.choice(["change_rule", "perturb_weights"])

        if mutation_type == "change_rule":
            self._change_rule(grammar, stats)
        else:
            self._perturb_weights(grammar, temperature, stats)

        stats.parametric_mutations += 1

    def _change_rule(
        self,
        grammar: "FractalGrammar",
        stats: MutationStats,
    ) -> None:
        """Change rule index for a random LEAF."""
        leaves = [
            node for node in grammar.tree.iter_nodes()
            if node.node_type == NodeType.LEAF
        ]

        if not leaves:
            return

        target = random.choice(leaves)
        new_rule = random.randint(0, self.config.num_rules - 1)

        # Prefer nearby rules (smooth exploration)
        if random.random() < 0.5:
            delta = random.choice([-1, 1])
            new_rule = (target.rule_idx + delta) % self.config.num_rules

        target.rule_idx = new_rule
        stats.rule_changes += 1

    def _perturb_weights(
        self,
        grammar: "FractalGrammar",
        temperature: float,
        stats: MutationStats,
    ) -> None:
        """Perturb alpha/gate weights."""
        # Use param_mutation_scale from config instead of hardcoded 0.1
        scale = self.param_mutation_scale
        for node in grammar.tree.iter_nodes():
            if node.alpha is not None and random.random() < 0.5:
                noise = torch.randn_like(node.alpha) * temperature * scale
                node.alpha = node.alpha + noise
                stats.weight_perturbations += 1

            if node.gate is not None and random.random() < 0.5:
                noise = torch.randn_like(node.gate) * temperature * scale
                node.gate = node.gate + noise
                stats.weight_perturbations += 1

    def mutate_rule_bank(
        self,
        grammar: "FractalGrammar",
        temperature: float = 0.1,
    ) -> None:
        """
        Mutate the rule bank parameters.

        This is a more aggressive mutation that changes the underlying
        rules, not just which rules are used.

        Args:
            grammar: Grammar to mutate (in place)
            temperature: Mutation strength
        """
        for rule in grammar.rule_bank.rules:
            if random.random() < 0.3:
                # Perturb weights
                with torch.no_grad():
                    noise = torch.randn_like(rule.W) * temperature * 0.05
                    rule.W.add_(noise)

                    noise = torch.randn_like(rule.b) * temperature * 0.05
                    rule.b.add_(noise)

                # Invalidate cached attractor
                rule.invalidate_attractor()


class GrammarCrossoverStrategy:
    """
    Crossover strategy for FractalGrammars.

    Implements subtree exchange between two parent grammars.

    Types:
    1. **Subtree swap**: Exchange subtrees at random points
    2. **Rule blend**: Average rule bank parameters
    3. **Hybrid**: Combine tree from one, rules from other

    Args:
        config: Grammar configuration

    Usage:
        >>> strategy = GrammarCrossoverStrategy(config)
        >>> child1, child2 = strategy.crossover(parent1, parent2)
    """

    def __init__(self, config: "GrammarConfig"):
        self.config = config

    def crossover(
        self,
        parent1: "FractalGrammar",
        parent2: "FractalGrammar",
    ) -> tuple["FractalGrammar", "FractalGrammar"]:
        """
        Create two children via crossover.

        Args:
            parent1: First parent grammar
            parent2: Second parent grammar

        Returns:
            Tuple of two child grammars
        """
        crossover_type = random.choice(["subtree", "rule_blend", "hybrid"])

        if crossover_type == "subtree":
            return self._subtree_crossover(parent1, parent2)
        elif crossover_type == "rule_blend":
            return self._rule_blend_crossover(parent1, parent2)
        else:
            return self._hybrid_crossover(parent1, parent2)

    def _subtree_crossover(
        self,
        parent1: "FractalGrammar",
        parent2: "FractalGrammar",
    ) -> tuple["FractalGrammar", "FractalGrammar"]:
        """Exchange subtrees between parents."""
        from latent_reasoning.grammar.grammar import FractalGrammar

        # Clone parents
        child1 = parent1.clone()
        child2 = parent2.clone()

        # Get all nodes from each
        nodes1 = list(child1.tree.iter_nodes())
        nodes2 = list(child2.tree.iter_nodes())

        # Select crossover points (not root)
        if len(nodes1) > 1 and len(nodes2) > 1:
            point1 = random.choice(nodes1[1:])  # Skip root
            point2 = random.choice(nodes2[1:])

            # Find parents of crossover points and swap
            # This is complex with our structure, so we do a simpler swap:
            # Just swap the node contents
            self._swap_node_contents(point1, point2)

        return child1, child2

    def _swap_node_contents(
        self,
        node1: GrammarNode,
        node2: GrammarNode,
    ) -> None:
        """Swap contents between two nodes."""
        # Store node1 state
        type1 = node1.node_type
        rule1 = node1.rule_idx
        children1 = node1.children
        alpha1 = node1.alpha
        gate1 = node1.gate

        # Copy node2 to node1
        node1.node_type = node2.node_type
        node1.rule_idx = node2.rule_idx
        node1.children = node2.children
        node1.alpha = node2.alpha
        node1.gate = node2.gate

        # Copy stored node1 to node2
        node2.node_type = type1
        node2.rule_idx = rule1
        node2.children = children1
        node2.alpha = alpha1
        node2.gate = gate1

    def _rule_blend_crossover(
        self,
        parent1: "FractalGrammar",
        parent2: "FractalGrammar",
    ) -> tuple["FractalGrammar", "FractalGrammar"]:
        """Blend rule bank parameters between parents."""
        from latent_reasoning.grammar.grammar import FractalGrammar

        child1 = parent1.clone()
        child2 = parent2.clone()

        # Interpolation factor
        alpha = random.uniform(0.3, 0.7)

        # Blend rule parameters
        with torch.no_grad():
            for r1, r2 in zip(child1.rule_bank.rules, parent2.rule_bank.rules):
                # Child1 gets alpha blend toward parent2
                r1.W.copy_(alpha * r1.W + (1 - alpha) * r2.W.to(r1.W.device))
                r1.b.copy_(alpha * r1.b + (1 - alpha) * r2.b.to(r1.b.device))
                r1.invalidate_attractor()

            for r1, r2 in zip(child2.rule_bank.rules, parent1.rule_bank.rules):
                # Child2 gets inverse blend
                r2_orig = r1.W.clone()
                r1.W.copy_((1 - alpha) * r1.W + alpha * r2.W.to(r1.W.device))
                r1.b.copy_((1 - alpha) * r1.b + alpha * r2.b.to(r1.b.device))
                r1.invalidate_attractor()

        return child1, child2

    def _hybrid_crossover(
        self,
        parent1: "FractalGrammar",
        parent2: "FractalGrammar",
    ) -> tuple["FractalGrammar", "FractalGrammar"]:
        """Create children with tree from one parent, rules from other."""
        from latent_reasoning.grammar.grammar import FractalGrammar

        # Child1: parent1's tree + parent2's rules
        child1 = FractalGrammar(
            rule_bank=parent2.rule_bank.clone(),
            tree=parent1.tree.clone(),
            latent_dim=parent1.latent_dim,
            device=parent1.device,
        )

        # Child2: parent2's tree + parent1's rules
        child2 = FractalGrammar(
            rule_bank=parent1.rule_bank.clone(),
            tree=parent2.tree.clone(),
            latent_dim=parent2.latent_dim,
            device=parent2.device,
        )

        return child1, child2
