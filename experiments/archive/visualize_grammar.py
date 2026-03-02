"""
Visualization suite for Fractal Latent Grammars.

This script creates visualizations to understand:
1. Grammar tree structures (AND/OR/LEAF composition)
2. Latent space trajectories through the tree
3. Attractor dynamics (how rules converge to fixed points)
4. Grammar evolution over generations
5. Rule effects in latent space

Run with: python experiments/visualize_grammar.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import torch
from dataclasses import dataclass

from latent_reasoning.config import GrammarConfig
from latent_reasoning.grammar import (
    FractalGrammar,
    GrammarTree,
    GrammarNode,
    NodeType,
    RuleBank,
    GrammarMutationStrategy,
    GrammarEvolutionLoop,
)


# Set style
plt.style.use('dark_background')
COLORS = {
    'leaf': '#4CAF50',      # Green
    'and': '#2196F3',       # Blue
    'or': '#FF9800',        # Orange
    'attractor': '#E91E63', # Pink
    'trajectory': '#00BCD4', # Cyan
    'highlight': '#FFEB3B', # Yellow
}


def create_config():
    """Create a grammar config for visualization."""
    return GrammarConfig(
        num_rules=8,
        max_depth=4,
        branching_factor=3,
        contraction_factor=0.9,
        and_prob=0.35,
        or_prob=0.35,
        rule_hidden_dim=64,
        population_size=20,
        mutation_rate=0.5,
    )


# =============================================================================
# 1. Grammar Tree Visualization
# =============================================================================

def visualize_tree(grammar: FractalGrammar, ax=None, title="Grammar Tree Structure"):
    """
    Visualize the AND/OR/LEAF tree structure.

    Shows:
    - Node types with different colors
    - Rule indices at LEAF nodes
    - Tree hierarchy
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Collect node positions using a simple layout algorithm
    positions = {}
    node_info = []

    def layout_tree(node: GrammarNode, x: float, y: float, width: float, idx: int = 0):
        """Recursively layout the tree."""
        positions[idx] = (x, y)

        # Store node info
        if node.node_type == NodeType.LEAF:
            color = COLORS['leaf']
            label = f"R{node.rule_idx}"
        elif node.node_type == NodeType.AND:
            color = COLORS['and']
            label = "AND"
        else:
            color = COLORS['or']
            label = "OR"

        node_info.append({
            'idx': idx,
            'pos': (x, y),
            'color': color,
            'label': label,
            'type': node.node_type,
            'children_idx': [],
        })

        # Layout children
        if node.children:
            child_width = width / len(node.children)
            start_x = x - width / 2 + child_width / 2
            for i, child in enumerate(node.children):
                child_idx = len(node_info)
                node_info[-1]['children_idx'].append(child_idx)
                layout_tree(child, start_x + i * child_width, y - 1, child_width * 0.8, child_idx)

    # Layout from root
    layout_tree(grammar.tree.root, 0, 0, 10)

    # Draw edges
    for info in node_info:
        for child_idx in info['children_idx']:
            child_info = node_info[child_idx]
            ax.plot(
                [info['pos'][0], child_info['pos'][0]],
                [info['pos'][1], child_info['pos'][1]],
                color='white', alpha=0.5, linewidth=1.5, zorder=1
            )

    # Draw nodes
    for info in node_info:
        circle = plt.Circle(info['pos'], 0.3, color=info['color'], zorder=2)
        ax.add_patch(circle)
        ax.text(info['pos'][0], info['pos'][1], info['label'],
                ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['leaf'], label='LEAF (Apply Rule)'),
        mpatches.Patch(color=COLORS['and'], label='AND (Blend)'),
        mpatches.Patch(color=COLORS['or'], label='OR (Select)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlim(-6, 6)
    ax.set_ylim(-grammar.tree.max_depth - 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')

    return ax


# =============================================================================
# 2. Latent Trajectory Visualization
# =============================================================================

def visualize_latent_trajectory(grammar: FractalGrammar, seed: torch.Tensor = None,
                                ax=None, title="Latent Trajectory Through Grammar"):
    """
    Visualize how a latent vector transforms as it flows through the grammar tree.

    Uses PCA to project to 2D and shows the path through transformations.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    latent_dim = grammar.latent_dim
    if seed is None:
        seed = torch.randn(latent_dim)

    # Collect latent vectors at each node
    trajectories = []

    def trace_expansion(node: GrammarNode, z: torch.Tensor, path: list):
        """Trace the expansion and collect intermediate latents."""
        trajectories.append({
            'latent': z.detach().clone(),
            'path': path.copy(),
            'type': node.node_type,
        })

        if node.node_type == NodeType.LEAF:
            result = grammar.rule_bank.apply(node.rule_idx, z)
            trajectories.append({
                'latent': result.detach().clone(),
                'path': path + ['out'],
                'type': 'output',
            })
            return result

        elif node.node_type == NodeType.AND:
            child_outputs = []
            for i, child in enumerate(node.children):
                out = trace_expansion(child, z, path + [f'and_{i}'])
                child_outputs.append(out)

            # Weighted average
            if node.alpha is None:
                alpha = torch.ones(len(node.children)) / len(node.children)
            else:
                alpha = torch.softmax(node.alpha, dim=0)

            result = sum(a * o for a, o in zip(alpha, child_outputs))
            trajectories.append({
                'latent': result.detach().clone(),
                'path': path + ['blend'],
                'type': 'blend',
            })
            return result

        elif node.node_type == NodeType.OR:
            child_outputs = []
            for i, child in enumerate(node.children):
                out = trace_expansion(child, z, path + [f'or_{i}'])
                child_outputs.append(out)

            # Gated selection
            if node.gate is None:
                gate = torch.ones(len(node.children)) / len(node.children)
            else:
                gate = torch.softmax(node.gate, dim=0)

            result = sum(g * o for g, o in zip(gate, child_outputs))
            trajectories.append({
                'latent': result.detach().clone(),
                'path': path + ['select'],
                'type': 'select',
            })
            return result

        return z

    # Trace the expansion
    with torch.no_grad():
        trace_expansion(grammar.tree.root, seed, ['root'])

    # Stack all latents and do PCA
    all_latents = torch.stack([t['latent'] for t in trajectories])

    # Simple PCA via SVD
    mean = all_latents.mean(dim=0)
    centered = all_latents - mean
    U, S, V = torch.svd(centered)
    projected = centered @ V[:, :2]

    # Plot trajectories
    points = projected.numpy()

    # Color by type
    type_colors = {
        NodeType.LEAF: COLORS['leaf'],
        NodeType.AND: COLORS['and'],
        NodeType.OR: COLORS['or'],
        'output': COLORS['attractor'],
        'blend': COLORS['and'],
        'select': COLORS['or'],
    }

    # Draw points
    for i, (traj, point) in enumerate(zip(trajectories, points)):
        color = type_colors.get(traj['type'], 'white')
        size = 150 if traj['type'] in ['output', 'blend', 'select'] else 80
        ax.scatter(point[0], point[1], c=color, s=size, zorder=3, edgecolors='white', linewidth=0.5)

    # Draw connections (simplified - just sequential)
    for i in range(len(points) - 1):
        ax.annotate('', xy=points[i+1], xytext=points[i],
                   arrowprops=dict(arrowstyle='->', color='white', alpha=0.3, lw=1))

    # Mark start and end
    ax.scatter(points[0, 0], points[0, 1], c=COLORS['highlight'], s=300, marker='*',
               zorder=4, label='Seed', edgecolors='white')
    ax.scatter(points[-1, 0], points[-1, 1], c=COLORS['attractor'], s=300, marker='s',
               zorder=4, label='Output', edgecolors='white')

    ax.set_xlabel('Principal Component 1', fontsize=10)
    ax.set_ylabel('Principal Component 2', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    return ax


# =============================================================================
# 3. Attractor Dynamics Visualization
# =============================================================================

def visualize_attractor_dynamics(rule_bank: RuleBank, rule_idx: int = 0,
                                  ax=None, title="Attractor Convergence"):
    """
    Visualize how repeated application of a rule converges to its attractor.

    Shows multiple trajectories from different starting points all converging.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    rule = rule_bank.rules[rule_idx]
    latent_dim = rule.latent_dim

    # Generate multiple starting points
    n_trajectories = 8
    n_steps = 15

    all_points = []

    with torch.no_grad():
        for traj_idx in range(n_trajectories):
            # Random starting point
            z = torch.randn(latent_dim) * 2
            trajectory = [z.clone()]

            # Apply rule repeatedly
            for step in range(n_steps):
                z = rule(z)
                trajectory.append(z.clone())

            all_points.append(torch.stack(trajectory))

    # Stack all and PCA
    all_stacked = torch.cat(all_points, dim=0)
    mean = all_stacked.mean(dim=0)
    centered = all_stacked - mean
    U, S, V = torch.svd(centered)

    # Project each trajectory
    cmap = plt.cm.viridis

    for traj_idx, trajectory in enumerate(all_points):
        centered_traj = trajectory - mean
        projected = (centered_traj @ V[:, :2]).numpy()

        # Create color gradient for this trajectory
        colors = cmap(np.linspace(0.2, 0.9, len(projected)))

        # Plot trajectory as line segments with color gradient
        for i in range(len(projected) - 1):
            ax.plot(projected[i:i+2, 0], projected[i:i+2, 1],
                   color=colors[i], linewidth=2, alpha=0.7)

        # Mark start
        ax.scatter(projected[0, 0], projected[0, 1], c='white', s=100,
                  marker='o', zorder=3, edgecolors='black')

    # Mark the attractor
    attractor = rule.compute_attractor(iterations=50)
    attractor_proj = ((attractor - mean) @ V[:, :2]).numpy()
    ax.scatter(attractor_proj[0], attractor_proj[1], c=COLORS['attractor'],
              s=400, marker='*', zorder=5, edgecolors='white', linewidth=2,
              label=f'Attractor (Rule {rule_idx})')

    ax.set_xlabel('Principal Component 1', fontsize=10)
    ax.set_ylabel('Principal Component 2', fontsize=10)
    ax.set_title(f"{title}\n(Rule {rule_idx}, contraction={rule.contraction_factor})",
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    # Add colorbar for iteration
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_steps))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Iteration')

    return ax


# =============================================================================
# 4. Grammar Evolution Visualization
# =============================================================================

def visualize_grammar_evolution(config: GrammarConfig, n_generations: int = 10,
                                 figsize=(16, 12)):
    """
    Visualize how grammars evolve over generations.

    Shows:
    - Tree structure changes
    - Score progression
    - Structural metrics (depth, nodes, rules used)
    """
    fig = plt.figure(figsize=figsize)

    # Create a simple mock scorer
    @dataclass
    class MockResult:
        overall: float

    class MockScorer:
        def score(self, latent, query=None):
            # Score based on latent characteristics
            norm = latent.norm().item()
            var = latent.var().item()
            score = 0.5 + 0.3 * np.tanh(var - 0.5) + 0.2 * np.tanh(5 - norm)
            return MockResult(overall=max(0, min(1, score)))

    # Run evolution
    loop = GrammarEvolutionLoop(
        grammar_config=config,
        latent_dim=64,
        population_size=15,
        device='cpu',
    )
    loop.initialize_population()

    scorer = MockScorer()

    # Track metrics over generations
    history = {
        'best_score': [],
        'avg_score': [],
        'avg_depth': [],
        'avg_nodes': [],
        'avg_rules_used': [],
    }

    best_grammars = []

    for gen in range(n_generations):
        # Expand and score
        for ind in loop.population:
            ind.latent = ind.grammar.expand()
            ind.score = scorer.score(ind.latent).overall

        # Record metrics
        scores = [ind.score for ind in loop.population]
        depths = [ind.grammar.tree.max_depth for ind in loop.population]
        nodes = [ind.grammar.tree.num_nodes for ind in loop.population]
        rules = [len(ind.grammar.tree.rules_used) for ind in loop.population]

        history['best_score'].append(max(scores))
        history['avg_score'].append(np.mean(scores))
        history['avg_depth'].append(np.mean(depths))
        history['avg_nodes'].append(np.mean(nodes))
        history['avg_rules_used'].append(np.mean(rules))

        # Save best grammar
        best = max(loop.population, key=lambda x: x.score)
        best_grammars.append(best.grammar.clone())

        # Selection and reproduction
        loop._selection_and_reproduction()
        loop.generation = gen + 1

    # Plot 1: Score progression
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(history['best_score'], 'o-', color=COLORS['highlight'], label='Best', linewidth=2)
    ax1.plot(history['avg_score'], 's--', color=COLORS['trajectory'], label='Average', linewidth=2, alpha=0.7)
    ax1.fill_between(range(n_generations), history['avg_score'], alpha=0.2, color=COLORS['trajectory'])
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Score')
    ax1.set_title('Score Evolution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Plot 2: Structural metrics
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(history['avg_depth'], 'o-', color=COLORS['and'], label='Avg Depth', linewidth=2)
    ax2.plot(history['avg_nodes'], 's-', color=COLORS['or'], label='Avg Nodes', linewidth=2)
    ax2.plot(history['avg_rules_used'], '^-', color=COLORS['leaf'], label='Avg Rules Used', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Count')
    ax2.set_title('Structural Evolution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    # Plot 3-6: Best grammar trees at different generations
    checkpoints = [0, n_generations // 3, 2 * n_generations // 3, n_generations - 1]
    for i, gen_idx in enumerate(checkpoints):
        ax = fig.add_subplot(2, 3, i + 3) if i < 2 else fig.add_subplot(2, 3, i + 3)
        visualize_tree(best_grammars[gen_idx], ax=ax,
                      title=f'Gen {gen_idx} (score={history["best_score"][gen_idx]:.3f})')

    plt.tight_layout()
    return fig


# =============================================================================
# 5. Rule Effects Visualization
# =============================================================================

def visualize_rule_effects(rule_bank: RuleBank, figsize=(14, 10)):
    """
    Visualize what each rule does to the latent space.

    Shows:
    - Input vs output distributions
    - Attractors for each rule
    - Contraction visualization
    """
    n_rules = min(rule_bank.num_rules, 8)
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()

    latent_dim = rule_bank.latent_dim
    n_samples = 100

    with torch.no_grad():
        # Generate random inputs
        inputs = torch.randn(n_samples, latent_dim)

        # Project all to 2D using shared PCA
        all_outputs = [inputs]
        attractors = []

        for i in range(n_rules):
            outputs = rule_bank.apply(i, inputs)
            all_outputs.append(outputs)
            attractors.append(rule_bank.get_attractor(i))

        # Compute shared PCA
        all_stacked = torch.cat(all_outputs + [torch.stack(attractors)], dim=0)
        mean = all_stacked.mean(dim=0)
        centered = all_stacked - mean
        U, S, V = torch.svd(centered)

        # Project inputs
        inputs_proj = ((inputs - mean) @ V[:, :2]).numpy()

        for i in range(n_rules):
            ax = axes[i]

            # Project outputs
            outputs = all_outputs[i + 1]
            outputs_proj = ((outputs - mean) @ V[:, :2]).numpy()

            # Project attractor
            attractor_proj = ((attractors[i] - mean) @ V[:, :2]).numpy()

            # Plot inputs (faded)
            ax.scatter(inputs_proj[:, 0], inputs_proj[:, 1],
                      c='gray', alpha=0.2, s=20, label='Input')

            # Plot outputs
            ax.scatter(outputs_proj[:, 0], outputs_proj[:, 1],
                      c=COLORS['trajectory'], alpha=0.6, s=30, label='Output')

            # Draw contraction arrows (sample)
            for j in range(0, n_samples, 10):
                ax.annotate('', xy=outputs_proj[j], xytext=inputs_proj[j],
                           arrowprops=dict(arrowstyle='->', color='white', alpha=0.3, lw=0.5))

            # Plot attractor
            ax.scatter(attractor_proj[0], attractor_proj[1],
                      c=COLORS['attractor'], s=200, marker='*',
                      edgecolors='white', linewidth=2, label='Attractor', zorder=5)

            ax.set_title(f'Rule {i}', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('Rule Effects: Input → Output Transformations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# 6. Mutation Effects Visualization
# =============================================================================

def visualize_mutation_effects(config: GrammarConfig, figsize=(16, 8)):
    """
    Visualize how mutations change grammar structure and behavior.
    """
    fig, axes = plt.subplots(2, 5, figsize=figsize)

    # Create original grammar
    original = FractalGrammar.random(config, latent_dim=64, device='cpu')

    # Apply mutations
    strategy = GrammarMutationStrategy(config, base_mutation_rate=1.0)

    # Show original
    visualize_tree(original, ax=axes[0, 0], title='Original')

    # Generate several mutants
    for i in range(4):
        mutant = strategy.mutate(original, generation=0, temperature=1.0)
        visualize_tree(mutant, ax=axes[0, i + 1], title=f'Mutant {i + 1}')

    # Show latent outputs
    seed = torch.randn(64)

    with torch.no_grad():
        original_out = original.expand(seed)

        # Collect all outputs for PCA
        all_outputs = [original_out]
        mutant_outputs = []

        for i in range(4):
            mutant = strategy.mutate(original, generation=0, temperature=1.0)
            out = mutant.expand(seed)
            all_outputs.append(out)
            mutant_outputs.append(out)

        # PCA
        all_stacked = torch.stack(all_outputs)
        mean = all_stacked.mean(dim=0)
        centered = all_stacked - mean

        # Handle case where all outputs are similar
        try:
            U, S, V = torch.svd(centered)
            projected = (centered @ V[:, :2]).numpy()
        except:
            projected = centered[:, :2].numpy()

        # Generate many mutants for scatter
        many_mutants = []
        for _ in range(50):
            mutant = strategy.mutate(original, generation=0, temperature=1.0)
            out = mutant.expand(seed)
            many_mutants.append(out)

        many_stacked = torch.stack(many_mutants)
        many_centered = many_stacked - mean
        try:
            many_proj = (many_centered @ V[:, :2]).numpy()
        except:
            many_proj = many_centered[:, :2].numpy()

    # Plot latent space
    ax = axes[1, 2]
    ax.scatter(many_proj[:, 0], many_proj[:, 1], c=COLORS['trajectory'], alpha=0.3, s=50, label='Mutants')
    ax.scatter(projected[0, 0], projected[0, 1], c=COLORS['highlight'], s=300, marker='*',
              edgecolors='white', linewidth=2, label='Original', zorder=5)
    ax.set_title('Mutation Effects in Latent Space', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Hide unused axes
    for ax in [axes[1, 0], axes[1, 1], axes[1, 3], axes[1, 4]]:
        ax.axis('off')

    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Fractal Latent Grammar Visualizations")
    print("=" * 60)

    # Create output directory
    output_dir = Path(__file__).parent / "plots" / "grammar_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config and grammar
    config = create_config()
    grammar = FractalGrammar.random(config, latent_dim=64, device='cpu')

    print(f"\nCreated grammar: {grammar}")
    print(f"  - Nodes: {grammar.tree.num_nodes}")
    print(f"  - Leaves: {grammar.tree.num_leaves}")
    print(f"  - Max depth: {grammar.tree.max_depth}")
    print(f"  - Rules used: {grammar.tree.rules_used}")

    # 1. Tree Structure
    print("\n[1/6] Visualizing tree structure...")
    fig, ax = plt.subplots(figsize=(12, 8))
    visualize_tree(grammar, ax=ax)
    fig.savefig(output_dir / "1_tree_structure.png", dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print(f"  Saved: {output_dir / '1_tree_structure.png'}")

    # 2. Latent Trajectory
    print("\n[2/6] Visualizing latent trajectory...")
    fig, ax = plt.subplots(figsize=(10, 10))
    visualize_latent_trajectory(grammar, ax=ax)
    fig.savefig(output_dir / "2_latent_trajectory.png", dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print(f"  Saved: {output_dir / '2_latent_trajectory.png'}")

    # 3. Attractor Dynamics
    print("\n[3/6] Visualizing attractor dynamics...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(min(8, config.num_rules)):
        ax = axes[i // 4, i % 4]
        visualize_attractor_dynamics(grammar.rule_bank, rule_idx=i, ax=ax,
                                     title=f"Rule {i}")
    plt.tight_layout()
    fig.savefig(output_dir / "3_attractor_dynamics.png", dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print(f"  Saved: {output_dir / '3_attractor_dynamics.png'}")

    # 4. Grammar Evolution
    print("\n[4/6] Visualizing grammar evolution (this may take a moment)...")
    fig = visualize_grammar_evolution(config, n_generations=15)
    fig.savefig(output_dir / "4_grammar_evolution.png", dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print(f"  Saved: {output_dir / '4_grammar_evolution.png'}")

    # 5. Rule Effects
    print("\n[5/6] Visualizing rule effects...")
    fig = visualize_rule_effects(grammar.rule_bank)
    fig.savefig(output_dir / "5_rule_effects.png", dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print(f"  Saved: {output_dir / '5_rule_effects.png'}")

    # 6. Mutation Effects
    print("\n[6/6] Visualizing mutation effects...")
    fig = visualize_mutation_effects(config)
    fig.savefig(output_dir / "6_mutation_effects.png", dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print(f"  Saved: {output_dir / '6_mutation_effects.png'}")

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)

    return output_dir


if __name__ == "__main__":
    output_dir = main()
