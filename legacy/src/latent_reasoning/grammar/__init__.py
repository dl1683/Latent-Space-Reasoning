"""
Fractal Latent Grammar module for compositional latent space reasoning.

This module provides a grammar-based approach to latent space exploration
where we evolve compositional structures (grammars) that GENERATE latent
vectors, rather than evolving the vectors directly.

Key Concepts:

**Grammar Rules**:
- Contractive transforms that map latents to latents
- T_r(z) = P_r * f(W_r @ z + b_r) with ||W_r||_2 < 1
- Contraction ensures convergence to fixed-point attractors

**AND/OR Trees**:
- LEAF: Apply a rule to generate/transform a latent
- AND: Combine children via weighted average (blend)
- OR: Select best child via learned gating (choose)

**Fractal Structure**:
- Self-similar composition enables complex patterns
- Rules are shared across tree for compression
- Natural hierarchy for reasoning decomposition

Key Components:

**GrammarRule (rules.py)**:
- Contractive linear transform with nonlinearity
- Spectral norm enforcement for contraction
- Fixed-point attractor computation

**GrammarNode, GrammarTree (tree.py)**:
- AND/OR/LEAF node types
- Recursive tree structure
- Expansion to latent vectors

**FractalGrammar (grammar.py)**:
- Combines rules + tree into complete grammar
- Latent generation via tree expansion
- Compression ratio metrics

**GrammarMutation (mutation.py)**:
- Depth-adaptive mutation rates
- Structural vs parametric mutations
- Crossover for grammar trees

**GrammarEvolutionLoop (grammar_loop.py)**:
- Evolution operating on grammars
- Integration with QD archive
- Grammar-specific selection

Quick Start:
    >>> from latent_reasoning.grammar import FractalGrammar
    >>> from latent_reasoning.config import GrammarConfig
    >>>
    >>> config = GrammarConfig(num_rules=8, max_depth=4)
    >>> grammar = FractalGrammar.random(config, latent_dim=1024)
    >>>
    >>> # Generate latent from grammar
    >>> latent = grammar.expand(seed_latent)
    >>>
    >>> # Mutate grammar
    >>> mutated = grammar.mutate(temperature=0.1)

References:
- Iterated Function Systems (IFS) - Barnsley (1988)
- Probabilistic Context-Free Grammars
- AND/OR search trees in AI planning
"""

from latent_reasoning.grammar.rules import (
    GrammarRule,
    RuleBank,
)
from latent_reasoning.grammar.tree import (
    NodeType,
    GrammarNode,
    GrammarTree,
)
from latent_reasoning.grammar.grammar import (
    FractalGrammar,
)
from latent_reasoning.grammar.mutation import (
    GrammarMutationStrategy,
    GrammarCrossoverStrategy,
)
from latent_reasoning.grammar.grammar_loop import (
    GrammarEvolutionLoop,
    GrammarEvolutionResult,
)

__all__ = [
    # Rules
    "GrammarRule",
    "RuleBank",
    # Tree
    "NodeType",
    "GrammarNode",
    "GrammarTree",
    # Grammar
    "FractalGrammar",
    # Mutation
    "GrammarMutationStrategy",
    "GrammarCrossoverStrategy",
    # Evolution
    "GrammarEvolutionLoop",
    "GrammarEvolutionResult",
]
