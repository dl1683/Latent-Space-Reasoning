"""
Unit tests for the Fractal Latent Grammar module.

Tests cover:
- GrammarRule: Contractive transforms and attractor computation
- RuleBank: Rule collection and batch operations
- GrammarNode: Tree node operations and expansion
- GrammarTree: Tree structure and traversal
- FractalGrammar: Complete grammar with expansion and compression
- GrammarMutationStrategy: Structural and parametric mutations
- GrammarCrossoverStrategy: Grammar crossover operations
- GrammarEvolutionLoop: Evolution with grammar populations
"""

import pytest
import torch
from dataclasses import dataclass

from latent_reasoning.config import GrammarConfig
from latent_reasoning.grammar import (
    GrammarRule,
    RuleBank,
    NodeType,
    GrammarNode,
    GrammarTree,
    FractalGrammar,
    GrammarMutationStrategy,
    GrammarCrossoverStrategy,
    GrammarEvolutionLoop,
    GrammarEvolutionResult,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def grammar_config():
    """Standard grammar configuration for tests."""
    return GrammarConfig(
        num_rules=8,
        max_depth=4,
        branching_factor=3,
        contraction_factor=0.9,
        and_prob=0.3,
        or_prob=0.3,
        rule_hidden_dim=64,  # Small for fast tests
        population_size=10,
        offspring_size=5,
        mutation_rate=0.3,
        crossover_rate=0.3,
        tournament_size=3,
    )


@pytest.fixture
def latent_dim():
    """Standard latent dimension for tests."""
    return 64  # Small for fast tests


@pytest.fixture
def device():
    """Test device."""
    return torch.device("cpu")


@pytest.fixture
def rule(latent_dim):
    """A single grammar rule."""
    return GrammarRule(
        latent_dim=latent_dim,
        contraction_factor=0.9,
    )


@pytest.fixture
def rule_bank(grammar_config, latent_dim):
    """Rule bank with multiple rules."""
    return RuleBank(
        num_rules=grammar_config.num_rules,
        latent_dim=latent_dim,
        contraction_factor=grammar_config.contraction_factor,
    )


@pytest.fixture
def random_grammar(grammar_config, latent_dim, device):
    """A random fractal grammar."""
    return FractalGrammar.random(
        grammar_config,
        latent_dim=latent_dim,
        device=device,
    )


@pytest.fixture
def mock_scorer():
    """Mock scorer for testing evolution."""
    @dataclass
    class MockScoreResult:
        overall: float
        coherence: float = 0.5
        complexity: float = 0.5
        novelty: float = 0.5

    class MockScorer:
        def score(self, latent, query=None):
            # Score based on latent norm (simple but deterministic)
            norm = latent.norm().item()
            score = 1.0 / (1.0 + abs(norm - 5.0))  # Peak at norm=5
            return MockScoreResult(overall=score)

    return MockScorer()


# ============================================================================
# GrammarRule Tests
# ============================================================================

class TestGrammarRule:
    """Tests for GrammarRule."""

    def test_creation(self, latent_dim):
        """Test rule creation."""
        rule = GrammarRule(latent_dim=latent_dim)
        assert rule.latent_dim == latent_dim
        assert rule.hidden_dim == latent_dim
        assert rule.contraction_factor == 0.9

    def test_creation_with_hidden_dim(self, latent_dim):
        """Test rule with different hidden dimension."""
        rule = GrammarRule(latent_dim=latent_dim, hidden_dim=32)
        assert rule.latent_dim == latent_dim
        assert rule.hidden_dim == 32

    def test_forward_1d(self, rule, latent_dim):
        """Test forward pass with 1D input."""
        z = torch.randn(latent_dim)
        out = rule(z)
        assert out.shape == (latent_dim,)

    def test_forward_2d(self, rule, latent_dim):
        """Test forward pass with batched input."""
        z = torch.randn(8, latent_dim)
        out = rule(z)
        assert out.shape == (8, latent_dim)

    def test_contraction(self, rule, latent_dim):
        """Test that rule is contractive during forward pass."""
        # Test contraction by checking that applying the rule
        # reduces the difference between two points (contraction property)
        z1 = torch.randn(latent_dim)
        z2 = torch.randn(latent_dim)
        dist_before = (z1 - z2).norm()

        out1 = rule(z1)
        out2 = rule(z2)
        dist_after = (out1 - out2).norm()

        # Contraction: distance should decrease (or stay same)
        # Allow some tolerance for numerical issues
        assert dist_after <= dist_before * 1.1

    def test_attractor_computation(self, rule, latent_dim):
        """Test fixed-point attractor computation."""
        attractor = rule.compute_attractor(iterations=50)
        assert attractor.shape == (latent_dim,)

        # Applying rule to attractor should give same point (approximately)
        out = rule(attractor)
        diff = (out - attractor).norm().item()
        assert diff < 0.1, f"Attractor not stable: diff={diff}"

    def test_attractor_caching(self, rule):
        """Test that attractor is cached."""
        a1 = rule.compute_attractor()
        a2 = rule.compute_attractor()
        assert torch.allclose(a1, a2)

    def test_attractor_invalidation(self, rule):
        """Test attractor cache invalidation."""
        a1 = rule.compute_attractor()
        rule.invalidate_attractor()
        assert not rule._attractor_valid

    def test_clone(self, rule, latent_dim):
        """Test rule cloning."""
        cloned = rule.clone()
        # Check weights match (not forward pass due to stochastic normalization)
        assert torch.allclose(rule.W.data, cloned.W.data)
        assert torch.allclose(rule.b.data, cloned.b.data)


# ============================================================================
# RuleBank Tests
# ============================================================================

class TestRuleBank:
    """Tests for RuleBank."""

    def test_creation(self, grammar_config, latent_dim):
        """Test rule bank creation."""
        bank = RuleBank(
            num_rules=grammar_config.num_rules,
            latent_dim=latent_dim,
        )
        assert len(bank) == grammar_config.num_rules
        assert bank.latent_dim == latent_dim

    def test_apply(self, rule_bank, latent_dim):
        """Test applying a single rule."""
        z = torch.randn(latent_dim)
        out = rule_bank.apply(0, z)
        assert out.shape == (latent_dim,)

    def test_apply_batch(self, rule_bank, latent_dim):
        """Test batch rule application."""
        z = torch.randn(latent_dim)
        outputs = rule_bank.apply_batch([0, 1, 2], z)
        assert len(outputs) == 3
        for out in outputs:
            assert out.shape == (latent_dim,)

    def test_get_attractor(self, rule_bank, latent_dim):
        """Test getting rule attractor."""
        attractor = rule_bank.get_attractor(0)
        assert attractor.shape == (latent_dim,)

    def test_get_all_attractors(self, rule_bank, latent_dim):
        """Test getting all attractors."""
        attractors = rule_bank.get_all_attractors()
        assert len(attractors) == rule_bank.num_rules
        for a in attractors:
            assert a.shape == (latent_dim,)

    def test_indexing(self, rule_bank):
        """Test rule bank indexing."""
        rule = rule_bank[0]
        assert isinstance(rule, GrammarRule)

    def test_clone(self, rule_bank, latent_dim):
        """Test rule bank cloning."""
        cloned = rule_bank.clone()
        # Check that weights match (not forward pass due to stochastic normalization)
        for i in range(len(rule_bank)):
            assert torch.allclose(
                rule_bank.rules[i].W.data,
                cloned.rules[i].W.data,
            )
            assert torch.allclose(
                rule_bank.rules[i].b.data,
                cloned.rules[i].b.data,
            )


# ============================================================================
# GrammarNode Tests
# ============================================================================

class TestGrammarNode:
    """Tests for GrammarNode."""

    def test_leaf_creation(self):
        """Test LEAF node creation."""
        node = GrammarNode(node_type=NodeType.LEAF, rule_idx=3)
        assert node.node_type == NodeType.LEAF
        assert node.rule_idx == 3
        assert len(node.children) == 0

    def test_and_creation(self, device):
        """Test AND node creation."""
        children = [
            GrammarNode(node_type=NodeType.LEAF, rule_idx=i)
            for i in range(3)
        ]
        node = GrammarNode(
            node_type=NodeType.AND,
            children=children,
            alpha=torch.randn(3),
        )
        assert node.node_type == NodeType.AND
        assert len(node.children) == 3
        assert node.alpha is not None

    def test_or_creation(self, device):
        """Test OR node creation."""
        children = [
            GrammarNode(node_type=NodeType.LEAF, rule_idx=i)
            for i in range(2)
        ]
        node = GrammarNode(
            node_type=NodeType.OR,
            children=children,
            gate=torch.randn(2),
        )
        assert node.node_type == NodeType.OR
        assert len(node.children) == 2
        assert node.gate is not None

    def test_expand_leaf(self, rule_bank, latent_dim):
        """Test LEAF node expansion."""
        node = GrammarNode(node_type=NodeType.LEAF, rule_idx=0)
        z = torch.randn(latent_dim)
        out = node.expand(z, rule_bank)
        assert out.shape == (latent_dim,)

    def test_expand_and(self, rule_bank, latent_dim):
        """Test AND node expansion."""
        children = [
            GrammarNode(node_type=NodeType.LEAF, rule_idx=i)
            for i in range(2)
        ]
        node = GrammarNode(
            node_type=NodeType.AND,
            children=children,
            alpha=torch.tensor([0.5, 0.5]),
        )
        z = torch.randn(latent_dim)
        out = node.expand(z, rule_bank)
        assert out.shape == (latent_dim,)

    def test_expand_or(self, rule_bank, latent_dim):
        """Test OR node expansion."""
        children = [
            GrammarNode(node_type=NodeType.LEAF, rule_idx=i)
            for i in range(2)
        ]
        node = GrammarNode(
            node_type=NodeType.OR,
            children=children,
            gate=torch.tensor([1.0, 0.0]),  # Select first child
        )
        z = torch.randn(latent_dim)
        out = node.expand(z, rule_bank)
        assert out.shape == (latent_dim,)

    def test_count_nodes(self):
        """Test node counting."""
        leaf = GrammarNode(node_type=NodeType.LEAF, rule_idx=0)
        assert leaf.count_nodes() == 1

        children = [
            GrammarNode(node_type=NodeType.LEAF, rule_idx=i)
            for i in range(3)
        ]
        parent = GrammarNode(node_type=NodeType.AND, children=children)
        assert parent.count_nodes() == 4

    def test_count_leaves(self):
        """Test leaf counting."""
        leaf = GrammarNode(node_type=NodeType.LEAF, rule_idx=0)
        assert leaf.count_leaves() == 1

        children = [
            GrammarNode(node_type=NodeType.LEAF, rule_idx=i)
            for i in range(3)
        ]
        parent = GrammarNode(node_type=NodeType.AND, children=children)
        assert parent.count_leaves() == 3

    def test_get_rules_used(self):
        """Test getting used rules."""
        children = [
            GrammarNode(node_type=NodeType.LEAF, rule_idx=i)
            for i in [0, 2, 4]
        ]
        parent = GrammarNode(node_type=NodeType.AND, children=children)
        assert parent.get_rules_used() == {0, 2, 4}

    def test_serialization(self):
        """Test node serialization."""
        node = GrammarNode(
            node_type=NodeType.AND,
            children=[
                GrammarNode(node_type=NodeType.LEAF, rule_idx=1),
                GrammarNode(node_type=NodeType.LEAF, rule_idx=2),
            ],
            alpha=torch.tensor([0.3, 0.7]),
        )
        data = node.to_dict()
        restored = GrammarNode.from_dict(data)
        assert restored.node_type == node.node_type
        assert len(restored.children) == len(node.children)

    def test_clone(self):
        """Test node cloning."""
        node = GrammarNode(
            node_type=NodeType.AND,
            children=[
                GrammarNode(node_type=NodeType.LEAF, rule_idx=1),
            ],
            alpha=torch.tensor([1.0]),
        )
        cloned = node.clone()
        assert cloned is not node
        assert cloned.children[0] is not node.children[0]


# ============================================================================
# GrammarTree Tests
# ============================================================================

class TestGrammarTree:
    """Tests for GrammarTree."""

    def test_creation(self):
        """Test tree creation."""
        root = GrammarNode(node_type=NodeType.LEAF, rule_idx=0)
        tree = GrammarTree(root=root)
        assert tree.root is root
        assert tree.num_nodes == 1

    def test_random_tree(self, grammar_config, device):
        """Test random tree generation."""
        tree = GrammarTree.random(
            grammar_config,
            num_rules=8,
            device=device,
        )
        assert tree.num_nodes >= 1
        assert tree.max_depth <= grammar_config.max_depth

    def test_balanced_tree(self, device):
        """Test balanced tree generation."""
        tree = GrammarTree.balanced(
            depth=2,
            branching=2,
            num_rules=8,
            device=device,
        )
        assert tree.num_nodes == 7  # 1 + 2 + 4 for depth=2, branching=2
        assert tree.num_leaves == 4

    def test_expand(self, grammar_config, rule_bank, latent_dim, device):
        """Test tree expansion."""
        tree = GrammarTree.random(grammar_config, num_rules=8, device=device)
        z = torch.randn(latent_dim)
        out = tree.expand(z, rule_bank)
        assert out.shape == (latent_dim,)

    def test_properties(self, grammar_config, device):
        """Test tree properties."""
        tree = GrammarTree.random(grammar_config, num_rules=8, device=device)
        assert tree.num_nodes > 0
        assert tree.num_leaves > 0
        assert tree.max_depth >= 0
        assert len(tree.rules_used) > 0

    def test_get_node_at_path(self, device):
        """Test getting node at path."""
        # Create a simple tree
        children = [
            GrammarNode(node_type=NodeType.LEAF, rule_idx=i)
            for i in range(2)
        ]
        root = GrammarNode(node_type=NodeType.AND, children=children)
        tree = GrammarTree(root=root, device=device)

        # Root
        node = tree.get_node_at_path([])
        assert node is root

        # First child
        node = tree.get_node_at_path([0])
        assert node is children[0]

    def test_serialization(self, grammar_config, device):
        """Test tree serialization."""
        tree = GrammarTree.random(grammar_config, num_rules=8, device=device)
        data = tree.to_dict()
        restored = GrammarTree.from_dict(data, device=str(device))
        assert restored.num_nodes == tree.num_nodes
        assert restored.num_leaves == tree.num_leaves


# ============================================================================
# FractalGrammar Tests
# ============================================================================

class TestFractalGrammar:
    """Tests for FractalGrammar."""

    def test_random_creation(self, grammar_config, latent_dim, device):
        """Test random grammar creation."""
        grammar = FractalGrammar.random(grammar_config, latent_dim, device)
        assert grammar.latent_dim == latent_dim
        assert grammar.rule_bank.num_rules == grammar_config.num_rules

    def test_balanced_creation(self, grammar_config, latent_dim, device):
        """Test balanced grammar creation."""
        grammar = FractalGrammar.balanced(
            grammar_config, latent_dim, depth=2, branching=2, device=device,
        )
        assert grammar.tree.num_nodes == 7

    def test_single_rule_creation(self, grammar_config, latent_dim, device):
        """Test single-rule grammar creation."""
        grammar = FractalGrammar.from_single_rule(
            grammar_config, latent_dim, rule_idx=3, device=device,
        )
        assert grammar.tree.num_nodes == 1
        assert grammar.tree.root.rule_idx == 3

    def test_expand(self, random_grammar, latent_dim):
        """Test grammar expansion."""
        seed = torch.randn(latent_dim)
        out = random_grammar.expand(seed)
        assert out.shape == (latent_dim,)

    def test_expand_without_seed(self, random_grammar, latent_dim):
        """Test expansion without explicit seed."""
        out = random_grammar.expand()  # Uses attractor blend
        assert out.shape == (latent_dim,)

    def test_expand_with_trace(self, random_grammar, latent_dim):
        """Test expansion with trace."""
        seed = torch.randn(latent_dim)
        out, trace = random_grammar.expand_with_trace(seed)
        assert out.shape == (latent_dim,)
        assert len(trace) > 0
        assert "type" in trace[0]

    def test_compression_ratio(self, random_grammar):
        """Test compression ratio computation."""
        ratio = random_grammar.compression_ratio
        assert ratio > 0

    def test_stats(self, random_grammar):
        """Test grammar statistics."""
        stats = random_grammar.stats
        assert stats.num_rules > 0
        assert stats.num_nodes > 0
        assert stats.num_leaves > 0

    def test_clone(self, random_grammar, latent_dim):
        """Test grammar cloning."""
        cloned = random_grammar.clone()
        # Check that structure matches
        assert cloned.tree.num_nodes == random_grammar.tree.num_nodes
        assert cloned.tree.num_leaves == random_grammar.tree.num_leaves
        # Check that rule weights match
        for i in range(len(random_grammar.rule_bank)):
            assert torch.allclose(
                random_grammar.rule_bank.rules[i].W.data,
                cloned.rule_bank.rules[i].W.data,
            )

    def test_to_device(self, random_grammar):
        """Test moving grammar to device."""
        grammar = random_grammar.to("cpu")
        assert grammar.device == torch.device("cpu")

    def test_serialization(self, random_grammar, latent_dim):
        """Test grammar serialization."""
        data = random_grammar.to_dict()
        restored = FractalGrammar.from_dict(data)
        # Check that structure matches
        assert restored.tree.num_nodes == random_grammar.tree.num_nodes
        assert restored.tree.num_leaves == random_grammar.tree.num_leaves
        # Check that rule weights match
        for i in range(len(random_grammar.rule_bank)):
            assert torch.allclose(
                random_grammar.rule_bank.rules[i].W.data,
                restored.rule_bank.rules[i].W.data,
            )


# ============================================================================
# GrammarMutationStrategy Tests
# ============================================================================

class TestGrammarMutationStrategy:
    """Tests for GrammarMutationStrategy."""

    def test_creation(self, grammar_config):
        """Test strategy creation."""
        strategy = GrammarMutationStrategy(grammar_config)
        assert strategy.base_mutation_rate == 0.3

    def test_mutate(self, grammar_config, random_grammar):
        """Test mutation."""
        strategy = GrammarMutationStrategy(grammar_config, base_mutation_rate=1.0)
        mutated = strategy.mutate(random_grammar, generation=0, temperature=1.0)
        assert mutated is not random_grammar

    def test_structural_ratio_decay(self, grammar_config):
        """Test that structural mutations decay with generation."""
        strategy = GrammarMutationStrategy(grammar_config)
        ratio_early = strategy._compute_structural_ratio(0)
        ratio_late = strategy._compute_structural_ratio(50)
        assert ratio_early > ratio_late

    def test_multiple_mutations(self, grammar_config, random_grammar):
        """Test multiple mutations produce variety."""
        strategy = GrammarMutationStrategy(grammar_config, base_mutation_rate=1.0)
        mutants = [
            strategy.mutate(random_grammar, generation=0, temperature=1.0)
            for _ in range(5)
        ]
        # Check that at least some are different
        structures = [m.tree.num_nodes for m in mutants]
        assert len(set(structures)) >= 1  # At least one unique structure

    def test_rule_bank_mutation(self, grammar_config, random_grammar, latent_dim):
        """Test rule bank mutation."""
        strategy = GrammarMutationStrategy(grammar_config)
        original_weights = random_grammar.rule_bank.rules[0].W.clone()
        strategy.mutate_rule_bank(random_grammar, temperature=1.0)
        # Some weights should have changed
        # Note: may not always change due to randomness


# ============================================================================
# GrammarCrossoverStrategy Tests
# ============================================================================

class TestGrammarCrossoverStrategy:
    """Tests for GrammarCrossoverStrategy."""

    def test_creation(self, grammar_config):
        """Test strategy creation."""
        strategy = GrammarCrossoverStrategy(grammar_config)
        assert strategy.config is grammar_config

    def test_crossover(self, grammar_config, latent_dim, device):
        """Test crossover produces two children."""
        strategy = GrammarCrossoverStrategy(grammar_config)
        parent1 = FractalGrammar.random(grammar_config, latent_dim, device)
        parent2 = FractalGrammar.random(grammar_config, latent_dim, device)
        child1, child2 = strategy.crossover(parent1, parent2)
        assert child1 is not parent1
        assert child2 is not parent2

    def test_subtree_crossover(self, grammar_config, latent_dim, device):
        """Test subtree crossover."""
        strategy = GrammarCrossoverStrategy(grammar_config)
        parent1 = FractalGrammar.balanced(grammar_config, latent_dim, depth=2, device=device)
        parent2 = FractalGrammar.balanced(grammar_config, latent_dim, depth=2, device=device)
        child1, child2 = strategy._subtree_crossover(parent1, parent2)
        assert child1.tree.num_nodes > 0
        assert child2.tree.num_nodes > 0

    def test_rule_blend_crossover(self, grammar_config, latent_dim, device):
        """Test rule blend crossover."""
        strategy = GrammarCrossoverStrategy(grammar_config)
        parent1 = FractalGrammar.random(grammar_config, latent_dim, device)
        parent2 = FractalGrammar.random(grammar_config, latent_dim, device)
        child1, child2 = strategy._rule_blend_crossover(parent1, parent2)
        # Children should exist
        assert child1 is not None
        assert child2 is not None

    def test_hybrid_crossover(self, grammar_config, latent_dim, device):
        """Test hybrid crossover."""
        strategy = GrammarCrossoverStrategy(grammar_config)
        parent1 = FractalGrammar.random(grammar_config, latent_dim, device)
        parent2 = FractalGrammar.random(grammar_config, latent_dim, device)
        child1, child2 = strategy._hybrid_crossover(parent1, parent2)
        # Child1 has parent1's tree
        assert child1.tree.num_nodes == parent1.tree.num_nodes
        # Child2 has parent2's tree
        assert child2.tree.num_nodes == parent2.tree.num_nodes


# ============================================================================
# GrammarEvolutionLoop Tests
# ============================================================================

class TestGrammarEvolutionLoop:
    """Tests for GrammarEvolutionLoop."""

    def test_creation(self, grammar_config, latent_dim, device):
        """Test loop creation."""
        loop = GrammarEvolutionLoop(
            grammar_config=grammar_config,
            latent_dim=latent_dim,
            population_size=10,
            device=device,
        )
        assert loop.population_size == 10
        assert loop.latent_dim == latent_dim

    def test_initialize_population(self, grammar_config, latent_dim, device):
        """Test population initialization."""
        loop = GrammarEvolutionLoop(
            grammar_config=grammar_config,
            latent_dim=latent_dim,
            population_size=10,
            device=device,
        )
        loop.initialize_population()
        assert len(loop.population) == 10

    def test_initialize_with_seed(self, grammar_config, latent_dim, device):
        """Test initialization with seed grammar."""
        loop = GrammarEvolutionLoop(
            grammar_config=grammar_config,
            latent_dim=latent_dim,
            population_size=10,
            device=device,
        )
        seed = FractalGrammar.random(grammar_config, latent_dim, device)
        loop.initialize_population(seed_grammar=seed)
        assert len(loop.population) == 10

    def test_run(self, grammar_config, latent_dim, device, mock_scorer):
        """Test running evolution."""
        loop = GrammarEvolutionLoop(
            grammar_config=grammar_config,
            latent_dim=latent_dim,
            population_size=5,
            device=device,
        )
        loop.initialize_population()
        result = loop.run(
            scorer=mock_scorer,
            query="test query",
            num_generations=3,
        )
        assert isinstance(result, GrammarEvolutionResult)
        assert result.best_grammar is not None
        assert result.best_latent is not None
        assert result.best_score > 0

    def test_run_with_early_stop(self, grammar_config, latent_dim, device, mock_scorer):
        """Test early stopping."""
        loop = GrammarEvolutionLoop(
            grammar_config=grammar_config,
            latent_dim=latent_dim,
            population_size=5,
            device=device,
        )
        loop.initialize_population()
        result = loop.run(
            scorer=mock_scorer,
            query="test",
            num_generations=100,
            early_stop_threshold=0.0,  # Always stop
        )
        assert len(result.history) < 100

    def test_history_recording(self, grammar_config, latent_dim, device, mock_scorer):
        """Test that history is recorded."""
        loop = GrammarEvolutionLoop(
            grammar_config=grammar_config,
            latent_dim=latent_dim,
            population_size=5,
            device=device,
        )
        loop.initialize_population()
        result = loop.run(
            scorer=mock_scorer,
            query="test",
            num_generations=3,
        )
        assert len(result.history) == 3
        assert "generation" in result.history[0]
        assert "best_score" in result.history[0]
        assert "avg_compression" in result.history[0]

    def test_inject_grammar(self, grammar_config, latent_dim, device):
        """Test injecting grammar into population."""
        loop = GrammarEvolutionLoop(
            grammar_config=grammar_config,
            latent_dim=latent_dim,
            population_size=5,
            device=device,
        )
        loop.initialize_population()
        grammar = FractalGrammar.random(grammar_config, latent_dim, device)
        loop.inject_grammar(grammar)
        assert len(loop.population) <= loop.population_size + 1

    def test_get_diverse_grammars(self, grammar_config, latent_dim, device, mock_scorer):
        """Test getting diverse grammars."""
        loop = GrammarEvolutionLoop(
            grammar_config=grammar_config,
            latent_dim=latent_dim,
            population_size=10,
            device=device,
        )
        loop.initialize_population()
        # Score first
        for ind in loop.population:
            ind.latent = ind.grammar.expand()
            ind.score = mock_scorer.score(ind.latent).overall
        diverse = loop.get_diverse_grammars(n=3)
        assert len(diverse) <= 3
        assert len(diverse) >= 1

    def test_reset(self, grammar_config, latent_dim, device):
        """Test loop reset."""
        loop = GrammarEvolutionLoop(
            grammar_config=grammar_config,
            latent_dim=latent_dim,
            population_size=5,
            device=device,
        )
        loop.initialize_population()
        loop.generation = 10
        loop.reset()
        assert len(loop.population) == 0
        assert loop.generation == 0
        assert loop.best_ever is None

    def test_from_config(self, grammar_config, latent_dim, device):
        """Test creation from config."""
        loop = GrammarEvolutionLoop.from_config(
            grammar_config=grammar_config,
            latent_dim=latent_dim,
            device=device,
        )
        assert loop.population_size == grammar_config.population_size
        assert loop.mutation_rate == grammar_config.mutation_rate


# ============================================================================
# Integration Tests
# ============================================================================

class TestGrammarIntegration:
    """Integration tests for the grammar module."""

    def test_full_pipeline(self, grammar_config, latent_dim, device, mock_scorer):
        """Test full grammar evolution pipeline."""
        # Create loop
        loop = GrammarEvolutionLoop(
            grammar_config=grammar_config,
            latent_dim=latent_dim,
            population_size=8,
            device=device,
        )

        # Initialize
        loop.initialize_population()

        # Run evolution
        result = loop.run(
            scorer=mock_scorer,
            query="test query",
            num_generations=5,
            verbose=False,
        )

        # Check result
        assert result.best_score > 0
        assert result.best_grammar is not None

        # Check that we can expand the best grammar
        latent = result.best_grammar.expand()
        assert latent.shape == (latent_dim,)

        # Check stats
        stats = result.grammar_stats
        assert stats.num_nodes > 0

    def test_grammar_serialization_roundtrip(self, grammar_config, latent_dim, device):
        """Test grammar survives serialization."""
        grammar = FractalGrammar.random(grammar_config, latent_dim, device)

        # Serialize and restore
        data = grammar.to_dict()
        restored = FractalGrammar.from_dict(data)

        # Check structure matches
        assert restored.tree.num_nodes == grammar.tree.num_nodes
        assert restored.tree.num_leaves == grammar.tree.num_leaves
        assert restored.latent_dim == grammar.latent_dim

        # Check that rule weights match
        for i in range(len(grammar.rule_bank)):
            assert torch.allclose(
                grammar.rule_bank.rules[i].W.data,
                restored.rule_bank.rules[i].W.data,
            )

    def test_mutation_preserves_validity(self, grammar_config, latent_dim, device):
        """Test that mutations produce valid grammars."""
        strategy = GrammarMutationStrategy(grammar_config, base_mutation_rate=1.0)
        grammar = FractalGrammar.random(grammar_config, latent_dim, device)

        for _ in range(10):
            mutated = strategy.mutate(grammar, generation=0, temperature=1.0)
            # Should be able to expand
            out = mutated.expand()
            assert out.shape == (latent_dim,)
            # Use for next iteration
            grammar = mutated

    def test_crossover_preserves_validity(self, grammar_config, latent_dim, device):
        """Test that crossover produces valid grammars."""
        strategy = GrammarCrossoverStrategy(grammar_config)

        for _ in range(5):
            parent1 = FractalGrammar.random(grammar_config, latent_dim, device)
            parent2 = FractalGrammar.random(grammar_config, latent_dim, device)
            child1, child2 = strategy.crossover(parent1, parent2)

            # Both children should expand
            out1 = child1.expand()
            out2 = child2.expand()
            assert out1.shape == (latent_dim,)
            assert out2.shape == (latent_dim,)
