"""
Master Educational Animation for Fractal Latent Grammars

Combines all concepts into one comprehensive, well-paced explainer.
Target: ~3-4 minutes, understandable by anyone.
"""

import sys
sys.path.insert(0, "C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Polygon
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import torch
import subprocess

# Output directory
OUT_DIR = Path(__file__).parent / "plots" / "grammar_master"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use('dark_background')
COLORS = {
    'bg': '#0d1117',
    'text': '#ffffff',
    'highlight': '#ffd700',
    'accent1': '#58a6ff',  # Blue
    'accent2': '#f85149',  # Red
    'accent3': '#3fb950',  # Green
    'accent4': '#a371f7',  # Purple
    'accent5': '#f0883e',  # Orange
    'dim': '#8b949e',
    'card': '#161b22',
}

# Timing: 10 fps, so 10 frames = 1 second
# We want ~3 mins = 180 seconds = 1800 frames
FPS = 10

def seconds_to_frames(seconds):
    return int(seconds * FPS)

# Section timing (in seconds)
TIMING = {
    'title': 5,
    'challenge_intro': 4,
    'challenge_question': 5,
    'traditional_intro': 4,
    'traditional_demo': 6,
    'traditional_problem': 5,
    'latent_intro': 5,
    'latent_space_viz': 8,
    'latent_benefits': 6,
    'grammar_title': 4,
    'grammar_metaphor': 8,
    'grammar_tree': 8,
    'grammar_fractal': 5,
    'rules_title': 4,
    'rules_transform': 6,
    'rules_convergence': 12,
    'rules_insight': 5,
    'composition_title': 4,
    'and_explanation': 10,
    'or_explanation': 10,
    'composition_summary': 6,
    'pipeline_title': 4,
    'pipeline_step1': 5,
    'pipeline_step2': 5,
    'pipeline_step3': 8,
    'pipeline_step4': 6,
    'pipeline_power': 5,
    'evolution_title': 4,
    'evolution_cycle': 10,
    'evolution_result': 6,
    'summary_title': 4,
    'summary_points': 10,
    'end_card': 5,
}

def build_frame_map():
    """Build a map of section_name -> (start_frame, end_frame)"""
    frame_map = {}
    current_frame = 0
    for section, duration in TIMING.items():
        n_frames = seconds_to_frames(duration)
        frame_map[section] = (current_frame, current_frame + n_frames)
        current_frame += n_frames
    return frame_map, current_frame

FRAME_MAP, TOTAL_FRAMES = build_frame_map()

def get_section_progress(frame, section):
    """Get progress (0-1) within a section, or -1 if not in section"""
    start, end = FRAME_MAP[section]
    if frame < start or frame >= end:
        return -1
    return (frame - start) / (end - start)

def ease_in_out(t):
    """Smooth easing function"""
    return t * t * (3 - 2 * t)

def draw_title_card(ax, frame):
    """Opening title card"""
    p = get_section_progress(frame, 'title')
    if p < 0:
        return

    alpha = min(1, p * 3) if p < 0.8 else max(0, 1 - (p - 0.8) * 5)

    ax.text(7, 5.5, "FRACTAL LATENT GRAMMARS", fontsize=32, ha='center',
           color=COLORS['highlight'], fontweight='bold', alpha=alpha)
    ax.text(7, 4.2, "A New Approach to AI Reasoning", fontsize=18, ha='center',
           color=COLORS['text'], alpha=alpha)
    ax.text(7, 2.5, "How AI can 'think' before it speaks", fontsize=14, ha='center',
           color=COLORS['dim'], alpha=alpha, style='italic')


def draw_challenge(ax, frame):
    """The Challenge section"""
    # Intro
    p = get_section_progress(frame, 'challenge_intro')
    if p >= 0:
        alpha = min(1, p * 3)
        ax.text(7, 6.5, "THE CHALLENGE", fontsize=28, ha='center',
               color=COLORS['accent2'], fontweight='bold', alpha=alpha)
        if p > 0.3:
            ax.text(7, 5, "How can AI reason through complex problems?",
                   fontsize=18, ha='center', color=COLORS['text'], alpha=min(1, (p-0.3)*3))
        return

    # Question example
    p = get_section_progress(frame, 'challenge_question')
    if p >= 0:
        ax.text(7, 6.5, "THE CHALLENGE", fontsize=28, ha='center',
               color=COLORS['accent2'], fontweight='bold')
        ax.text(7, 5, "How can AI reason through complex problems?",
               fontsize=18, ha='center', color=COLORS['text'])

        alpha = min(1, p * 2)
        box = FancyBboxPatch((1.5, 2.5), 11, 1.8, boxstyle="round,pad=0.15",
                            facecolor=COLORS['card'], edgecolor=COLORS['accent1'],
                            alpha=alpha, linewidth=2)
        ax.add_patch(box)
        ax.text(7, 3.4, '"What are the implications of quantum entanglement',
               fontsize=13, ha='center', va='center', color=COLORS['text'],
               alpha=alpha, style='italic')
        ax.text(7, 2.9, 'for modern cryptography?"',
               fontsize=13, ha='center', va='center', color=COLORS['text'],
               alpha=alpha, style='italic')

        if p > 0.5:
            ax.text(7, 1.3, "This requires deep, multi-step reasoning...",
                   fontsize=14, ha='center', color=COLORS['dim'], alpha=min(1, (p-0.5)*3))


def draw_traditional(ax, frame):
    """Traditional approach section"""
    p = get_section_progress(frame, 'traditional_intro')
    if p >= 0:
        alpha = min(1, p * 3)
        ax.text(7, 6.5, "TRADITIONAL APPROACH", fontsize=24, ha='center',
               color=COLORS['accent5'], fontweight='bold', alpha=alpha)
        if p > 0.3:
            ax.text(7, 5, "Large Language Models generate text word-by-word",
                   fontsize=16, ha='center', color=COLORS['text'], alpha=min(1, (p-0.3)*3))
        return

    p = get_section_progress(frame, 'traditional_demo')
    if p >= 0:
        ax.text(7, 6.5, "TRADITIONAL APPROACH", fontsize=24, ha='center',
               color=COLORS['accent5'], fontweight='bold')
        ax.text(7, 5.2, "Generate answer one word at a time:", fontsize=16,
               ha='center', color=COLORS['text'])

        words = ["The", "implications", "of", "quantum", "entanglement", "are", "..."]
        n_words = min(len(words), int(p * len(words) * 1.5) + 1)
        text = " ".join(words[:n_words])

        box = FancyBboxPatch((1.5, 3), 11, 1.5, boxstyle="round,pad=0.1",
                            facecolor=COLORS['card'], edgecolor=COLORS['accent1'],
                            linewidth=2)
        ax.add_patch(box)
        ax.text(7, 3.75, text, fontsize=16, ha='center', va='center',
               color=COLORS['accent1'])

        # Cursor blink
        if int(p * 10) % 2 == 0:
            ax.text(7 + len(text)*0.08, 3.75, "|", fontsize=16, ha='left',
                   color=COLORS['highlight'])
        return

    p = get_section_progress(frame, 'traditional_problem')
    if p >= 0:
        ax.text(7, 6.5, "THE PROBLEM", fontsize=24, ha='center',
               color=COLORS['accent2'], fontweight='bold')

        problems = [
            ("No time to 'think'", "Each word is generated immediately"),
            ("Commits early", "First words lock in the direction"),
            ("No exploration", "Can't consider alternative approaches"),
        ]

        for idx, (title, desc) in enumerate(problems):
            show_at = idx * 0.25
            if p > show_at:
                alpha = min(1, (p - show_at) * 4)
                y = 4.8 - idx * 1.3
                ax.text(3, y, "X", fontsize=20, ha='center', color=COLORS['accent2'],
                       alpha=alpha, fontweight='bold')
                ax.text(4, y, title, fontsize=16, ha='left', color=COLORS['text'],
                       alpha=alpha, fontweight='bold')
                ax.text(4, y - 0.5, desc, fontsize=12, ha='left', color=COLORS['dim'],
                       alpha=alpha)


def draw_latent_approach(ax, frame):
    """Our latent space approach"""
    p = get_section_progress(frame, 'latent_intro')
    if p >= 0:
        alpha = min(1, p * 3)
        ax.text(7, 6.5, "OUR APPROACH", fontsize=24, ha='center',
               color=COLORS['accent3'], fontweight='bold', alpha=alpha)
        if p > 0.3:
            ax.text(7, 5, "Before generating words, explore an 'idea space'",
                   fontsize=16, ha='center', color=COLORS['text'], alpha=min(1, (p-0.3)*3))
        if p > 0.6:
            ax.text(7, 4.2, "where similar ideas are close together",
                   fontsize=14, ha='center', color=COLORS['dim'], alpha=min(1, (p-0.6)*3))
        return

    p = get_section_progress(frame, 'latent_space_viz')
    if p >= 0:
        ax.text(7, 7.2, "THE IDEA SPACE", fontsize=20, ha='center',
               color=COLORS['accent3'], fontweight='bold')

        # Draw concept clusters
        concepts = [
            ("Quantum\nPhysics", 3, 4.5, COLORS['accent1']),
            ("Entanglement", 4.5, 5, COLORS['accent1']),
            ("Security", 10, 5, COLORS['accent2']),
            ("Encryption", 9, 4, COLORS['accent2']),
            ("Cryptography", 7, 4.5, COLORS['accent4']),
            ("Key Exchange", 6, 3.5, COLORS['accent4']),
        ]

        # Fade in concepts
        for idx, (label, x, y, color) in enumerate(concepts):
            show_at = idx * 0.1
            if p > show_at:
                alpha = min(0.8, (p - show_at) * 2)
                circle = Circle((x, y), 0.7, facecolor=color, alpha=alpha)
                ax.add_patch(circle)
                ax.text(x, y, label, fontsize=9, ha='center', va='center',
                       color='white', alpha=min(1, alpha*1.5))

        # Draw connections
        if p > 0.5:
            conn_alpha = min(0.4, (p - 0.5) * 0.8)
            connections = [
                (3, 4.5, 4.5, 5),
                (9, 4, 10, 5),
                (7, 4.5, 9, 4),
                (7, 4.5, 6, 3.5),
            ]
            for x1, y1, x2, y2 in connections:
                ax.plot([x1, x2], [y1, y2], 'w-', alpha=conn_alpha, linewidth=2)

        if p > 0.7:
            ax.text(7, 1.5, "Related ideas cluster together",
                   fontsize=14, ha='center', color=COLORS['highlight'],
                   alpha=min(1, (p-0.7)*4))
        return

    p = get_section_progress(frame, 'latent_benefits')
    if p >= 0:
        ax.text(7, 6.5, "WHY THIS MATTERS", fontsize=22, ha='center',
               color=COLORS['highlight'], fontweight='bold')

        benefits = [
            "Think before speaking",
            "Explore multiple reasoning paths",
            "Find unexpected connections",
            "Produce more thoughtful answers"
        ]

        for idx, benefit in enumerate(benefits):
            show_at = idx * 0.2
            if p > show_at:
                alpha = min(1, (p - show_at) * 3)
                ax.text(7, 5 - idx * 0.9, f"+ {benefit}", fontsize=15,
                       ha='center', color=COLORS['accent3'], alpha=alpha)


def draw_grammar_concept(ax, frame):
    """What is a grammar?"""
    p = get_section_progress(frame, 'grammar_title')
    if p >= 0:
        alpha = min(1, p * 3)
        ax.text(7, 5.5, "FRACTAL LATENT GRAMMARS", fontsize=26, ha='center',
               color=COLORS['highlight'], fontweight='bold', alpha=alpha)
        if p > 0.4:
            ax.text(7, 4, "A structured way to explore the idea space",
                   fontsize=16, ha='center', color=COLORS['text'], alpha=min(1, (p-0.4)*3))
        return

    p = get_section_progress(frame, 'grammar_metaphor')
    if p >= 0:
        ax.text(7, 7, "GRAMMAR = RECIPE FOR IDEAS", fontsize=20, ha='center',
               color=COLORS['highlight'], fontweight='bold')

        # Two columns: cooking vs AI
        if p > 0.1:
            alpha = min(1, (p - 0.1) * 3)
            ax.text(3.5, 5.8, "Cooking", fontsize=16, ha='center',
                   color=COLORS['accent5'], fontweight='bold', alpha=alpha)
            ax.text(10.5, 5.8, "AI Reasoning", fontsize=16, ha='center',
                   color=COLORS['accent1'], fontweight='bold', alpha=alpha)
            ax.plot([7, 7], [2, 5.5], '--', color=COLORS['dim'], alpha=alpha*0.5)

        if p > 0.25:
            alpha = min(1, (p - 0.25) * 3)
            ax.text(3.5, 4.8, "Ingredients", fontsize=13, ha='center', color=COLORS['text'], alpha=alpha)
            ax.text(3.5, 4.2, "Flour, eggs, sugar", fontsize=11, ha='center', color=COLORS['dim'], alpha=alpha)
            ax.text(10.5, 4.8, "Concepts", fontsize=13, ha='center', color=COLORS['text'], alpha=alpha)
            ax.text(10.5, 4.2, "Physics, Security, Math", fontsize=11, ha='center', color=COLORS['dim'], alpha=alpha)

        if p > 0.5:
            alpha = min(1, (p - 0.5) * 3)
            ax.text(3.5, 3.3, "Recipe steps", fontsize=13, ha='center', color=COLORS['text'], alpha=alpha)
            ax.text(3.5, 2.7, "Mix - Fold - Bake", fontsize=11, ha='center', color=COLORS['accent5'], alpha=alpha)
            ax.text(10.5, 3.3, "Grammar rules", fontsize=13, ha='center', color=COLORS['text'], alpha=alpha)
            ax.text(10.5, 2.7, "Transform - Combine - Refine", fontsize=11, ha='center', color=COLORS['accent1'], alpha=alpha)

        if p > 0.75:
            alpha = min(1, (p - 0.75) * 4)
            ax.text(3.5, 1.6, "Output: Cake", fontsize=14, ha='center', color=COLORS['highlight'], alpha=alpha)
            ax.text(10.5, 1.6, "Output: Refined Idea", fontsize=14, ha='center', color=COLORS['highlight'], alpha=alpha)
        return

    p = get_section_progress(frame, 'grammar_tree')
    if p >= 0:
        ax.text(7, 7.2, "GRAMMAR STRUCTURE", fontsize=20, ha='center',
               color=COLORS['highlight'], fontweight='bold')

        # Root
        if p > 0.05:
            alpha = min(1, (p - 0.05) * 5)
            root = Circle((7, 5.2), 0.55, facecolor=COLORS['accent1'], alpha=alpha)
            ax.add_patch(root)
            ax.text(7, 5.2, "BLEND", fontsize=11, ha='center', va='center',
                   color='white', fontweight='bold', alpha=alpha)

        # Children - OR nodes
        if p > 0.25:
            alpha = min(1, (p - 0.25) * 4)
            for x in [4.5, 9.5]:
                c = Circle((x, 3.5), 0.5, facecolor=COLORS['accent5'], alpha=alpha)
                ax.add_patch(c)
                ax.text(x, 3.5, "PICK", fontsize=10, ha='center', va='center',
                       color='white', fontweight='bold', alpha=alpha)
            ax.plot([7, 4.5], [4.65, 4], 'w-', alpha=alpha*0.7, linewidth=2)
            ax.plot([7, 9.5], [4.65, 4], 'w-', alpha=alpha*0.7, linewidth=2)

        # Leaves - Rules
        if p > 0.5:
            alpha = min(1, (p - 0.5) * 4)
            leaves = [(3, 1.8), (6, 1.8), (8, 1.8), (11, 1.8)]
            for i, (x, y) in enumerate(leaves):
                c = Circle((x, y), 0.45, facecolor=COLORS['accent3'], alpha=alpha)
                ax.add_patch(c)
                ax.text(x, y, f"R{i+1}", fontsize=10, ha='center', va='center',
                       color='white', fontweight='bold', alpha=alpha)

            ax.plot([4.5, 3], [3, 2.25], 'w-', alpha=alpha*0.7, linewidth=2)
            ax.plot([4.5, 6], [3, 2.25], 'w-', alpha=alpha*0.7, linewidth=2)
            ax.plot([9.5, 8], [3, 2.25], 'w-', alpha=alpha*0.7, linewidth=2)
            ax.plot([9.5, 11], [3, 2.25], 'w-', alpha=alpha*0.7, linewidth=2)

        # Legend
        if p > 0.7:
            alpha = min(1, (p - 0.7) * 4)
            ax.text(1, 6, "BLEND = combine all", fontsize=10, color=COLORS['accent1'], alpha=alpha)
            ax.text(1, 5.4, "PICK = choose best", fontsize=10, color=COLORS['accent5'], alpha=alpha)
            ax.text(1, 4.8, "R = transform rule", fontsize=10, color=COLORS['accent3'], alpha=alpha)
        return

    p = get_section_progress(frame, 'grammar_fractal')
    if p >= 0:
        ax.text(7, 6.5, 'WHY "FRACTAL"?', fontsize=22, ha='center',
               color=COLORS['highlight'], fontweight='bold')

        if p > 0.2:
            ax.text(7, 5, "The same pattern repeats at every level",
                   fontsize=16, ha='center', color=COLORS['text'], alpha=min(1, (p-0.2)*3))
        if p > 0.5:
            ax.text(7, 4, "Like a tree where each branch looks like the whole tree",
                   fontsize=14, ha='center', color=COLORS['dim'], alpha=min(1, (p-0.5)*3))
        if p > 0.7:
            ax.text(7, 2.5, "Simple rules  -->  Complex thoughts",
                   fontsize=18, ha='center', color=COLORS['accent3'], alpha=min(1, (p-0.7)*4),
                   fontweight='bold')


def draw_rules(ax, frame):
    """How rules transform ideas"""
    p = get_section_progress(frame, 'rules_title')
    if p >= 0:
        alpha = min(1, p * 3)
        ax.text(7, 5.5, "HOW RULES WORK", fontsize=26, ha='center',
               color=COLORS['accent4'], fontweight='bold', alpha=alpha)
        if p > 0.4:
            ax.text(7, 4, "Each rule transforms one idea into another",
                   fontsize=16, ha='center', color=COLORS['text'], alpha=min(1, (p-0.4)*3))
        return

    p = get_section_progress(frame, 'rules_transform')
    if p >= 0:
        ax.text(7, 7, "RULE = TRANSFORMATION", fontsize=20, ha='center',
               color=COLORS['accent4'], fontweight='bold')

        if p > 0.1:
            alpha = min(1, (p - 0.1) * 3)
            # Input box
            box1 = FancyBboxPatch((1.5, 3.5), 3, 1.8, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['card'], edgecolor=COLORS['accent1'],
                                 alpha=alpha, linewidth=2)
            ax.add_patch(box1)
            ax.text(3, 4.4, "Input Idea", fontsize=14, ha='center', color=COLORS['accent1'], alpha=alpha)
            ax.text(3, 3.9, "[0.2, -0.5, 0.8, ...]", fontsize=10, ha='center', color=COLORS['dim'], alpha=alpha)

        if p > 0.4:
            alpha = min(1, (p - 0.4) * 3)
            # Arrow with "RULE"
            ax.annotate("", xy=(8.5, 4.4), xytext=(5, 4.4),
                       arrowprops=dict(arrowstyle="->", color=COLORS['highlight'], lw=3),
                       alpha=alpha)
            ax.text(6.75, 5.2, "RULE", fontsize=16, ha='center', color=COLORS['highlight'],
                   alpha=alpha, fontweight='bold')
            ax.text(6.75, 3.5, "rotate + shrink", fontsize=11, ha='center', color=COLORS['dim'], alpha=alpha)

        if p > 0.7:
            alpha = min(1, (p - 0.7) * 4)
            # Output box
            box2 = FancyBboxPatch((9, 3.5), 3, 1.8, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['card'], edgecolor=COLORS['accent3'],
                                 alpha=alpha, linewidth=2)
            ax.add_patch(box2)
            ax.text(10.5, 4.4, "Output Idea", fontsize=14, ha='center', color=COLORS['accent3'], alpha=alpha)
            ax.text(10.5, 3.9, "[0.1, -0.3, 0.5, ...]", fontsize=10, ha='center', color=COLORS['dim'], alpha=alpha)
        return

    p = get_section_progress(frame, 'rules_convergence')
    if p >= 0:
        ax.text(7, 7.3, "THE MAGIC: CONVERGENCE", fontsize=18, ha='center',
               color=COLORS['highlight'], fontweight='bold')

        # Generate trajectories
        np.random.seed(42)
        n_traj = 5
        colors = [COLORS['accent1'], COLORS['accent2'], COLORS['accent3'],
                  COLORS['accent4'], COLORS['accent5']]

        # Simulated trajectories converging to center
        trajectories = []
        for t in range(n_traj):
            start = np.random.randn(2) * 2
            traj = [start.copy()]
            for step in range(15):
                # Move toward origin with some rotation
                theta = 0.3
                scale = 0.85
                rot = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
                traj.append(scale * rot @ traj[-1])
            trajectories.append(np.array(traj))

        # Scale and offset for display
        for traj in trajectories:
            traj[:, 0] = traj[:, 0] * 1.5 + 7
            traj[:, 1] = traj[:, 1] * 1.2 + 4

        step = min(15, int(p * 18))

        for idx, traj in enumerate(trajectories):
            n_show = min(step + 1, len(traj))
            alpha = 0.8

            # Path
            ax.plot(traj[:n_show, 0], traj[:n_show, 1],
                   color=colors[idx], alpha=alpha*0.7, linewidth=2)

            # Current point
            ax.scatter(traj[n_show-1, 0], traj[n_show-1, 1],
                      c=colors[idx], s=120, zorder=5, edgecolor='white', linewidth=2)

            # Start marker
            ax.scatter(traj[0, 0], traj[0, 1],
                      c=colors[idx], s=60, marker='s', alpha=0.5)

        # Attractor marker
        if step > 8:
            alpha = min(1, (step - 8) / 5)
            ax.scatter(7, 4, c=COLORS['highlight'], s=200, marker='*', zorder=10, alpha=alpha)
            ax.text(7, 2.5, "ATTRACTOR", fontsize=14, ha='center',
                   color=COLORS['highlight'], alpha=alpha, fontweight='bold')

        # Explanation text
        ax.set_xlim(2, 12)
        ax.set_ylim(1.5, 7)

        if step < 5:
            ax.text(7, 1.8, "Different starting ideas (colored squares)",
                   fontsize=12, ha='center', color=COLORS['dim'])
        elif step < 10:
            ax.text(7, 1.8, "Each step applies the rule - ideas get closer",
                   fontsize=12, ha='center', color=COLORS['text'])
        else:
            ax.text(7, 1.8, "All paths converge to the same stable point!",
                   fontsize=12, ha='center', color=COLORS['highlight'])
        return

    p = get_section_progress(frame, 'rules_insight')
    if p >= 0:
        ax.text(7, 6.5, "KEY INSIGHT", fontsize=24, ha='center',
               color=COLORS['highlight'], fontweight='bold')

        insights = [
            ("Rules are CONTRACTIVE", COLORS['accent3']),
            ("They always shrink distances", COLORS['text']),
            ("This guarantees convergence", COLORS['text']),
            ("= Stable, predictable reasoning", COLORS['highlight']),
        ]

        for idx, (text, color) in enumerate(insights):
            show_at = idx * 0.2
            if p > show_at:
                alpha = min(1, (p - show_at) * 3)
                ax.text(7, 5 - idx * 1, text, fontsize=16, ha='center',
                       color=color, alpha=alpha)


def draw_composition(ax, frame):
    """AND vs OR composition"""
    p = get_section_progress(frame, 'composition_title')
    if p >= 0:
        alpha = min(1, p * 3)
        ax.text(7, 5.5, "COMBINING IDEAS", fontsize=26, ha='center',
               color=COLORS['accent1'], fontweight='bold', alpha=alpha)
        if p > 0.4:
            ax.text(7, 4, "Two fundamental operations: AND and OR",
                   fontsize=16, ha='center', color=COLORS['text'], alpha=min(1, (p-0.4)*3))
        return

    p = get_section_progress(frame, 'and_explanation')
    if p >= 0:
        ax.text(7, 7.2, "AND = BLEND ALL IDEAS", fontsize=20, ha='center',
               color=COLORS['accent1'], fontweight='bold')

        ideas = [
            (3, 4.5, "Quantum", COLORS['accent1']),
            (7, 5.2, "Security", COLORS['accent2']),
            (11, 4.5, "Networks", COLORS['accent3']),
        ]

        # Show input ideas
        if p > 0.1:
            alpha = min(1, (p - 0.1) * 3)
            for x, y, label, color in ideas:
                circle = Circle((x, y), 0.6, facecolor=color, alpha=alpha*0.8)
                ax.add_patch(circle)
                ax.text(x, y, label, fontsize=10, ha='center', va='center',
                       color='white', alpha=alpha)

        # Show blending animation
        if p > 0.35:
            blend_p = min(1, (p - 0.35) / 0.4)

            # Arrows toward center
            for x, y, _, _ in ideas:
                end_x = x + (7 - x) * blend_p * 0.5
                end_y = y + (3 - y) * blend_p * 0.5
                ax.annotate("", xy=(end_x, end_y), xytext=(x, y - 0.6),
                           arrowprops=dict(arrowstyle="->", color='white', alpha=0.4))

            if blend_p > 0.6:
                result_alpha = (blend_p - 0.6) * 2.5
                result = Circle((7, 2.5), 0.8, facecolor=COLORS['highlight'],
                               alpha=result_alpha*0.9, edgecolor='white', linewidth=2)
                ax.add_patch(result)
                ax.text(7, 2.5, "Combined", fontsize=10, ha='center', va='center',
                       color='black', alpha=result_alpha)

        if p > 0.8:
            ax.text(7, 1, "AND combines ALL perspectives with learned weights",
                   fontsize=13, ha='center', color=COLORS['text'], alpha=min(1, (p-0.8)*5))
        return

    p = get_section_progress(frame, 'or_explanation')
    if p >= 0:
        ax.text(7, 7.2, "OR = SELECT BEST IDEA", fontsize=20, ha='center',
               color=COLORS['accent5'], fontweight='bold')

        ideas = [
            (3, 4.5, "Option A", COLORS['accent1']),
            (7, 5.2, "Option B", COLORS['accent3']),
            (11, 4.5, "Option C", COLORS['accent2']),
        ]

        # Show input ideas
        if p > 0.1:
            alpha = min(1, (p - 0.1) * 3)
            for x, y, label, color in ideas:
                circle = Circle((x, y), 0.6, facecolor=color, alpha=alpha*0.8)
                ax.add_patch(circle)
                ax.text(x, y, label, fontsize=10, ha='center', va='center',
                       color='white', alpha=alpha)

        # Show selection
        if p > 0.35:
            select_p = min(1, (p - 0.35) / 0.4)
            winner_idx = 1  # Option B wins

            for idx, (x, y, label, color) in enumerate(ideas):
                if idx == winner_idx:
                    if select_p > 0.3:
                        ring_alpha = min(1, (select_p - 0.3) * 2)
                        ring = Circle((x, y), 0.85, facecolor='none',
                                     edgecolor=COLORS['highlight'], linewidth=4,
                                     alpha=ring_alpha)
                        ax.add_patch(ring)
                else:
                    if select_p > 0.5:
                        fade = max(0.3, 1 - (select_p - 0.5) * 1.4)
                        circle = Circle((x, y), 0.6, facecolor=color, alpha=fade*0.8)
                        ax.add_patch(circle)

            if select_p > 0.7:
                ax.text(7, 2.5, "Selected: Option B", fontsize=16, ha='center',
                       color=COLORS['highlight'], alpha=min(1, (select_p-0.7)*4),
                       fontweight='bold')

        if p > 0.8:
            ax.text(7, 1, "OR picks the BEST option using learned gating",
                   fontsize=13, ha='center', color=COLORS['text'], alpha=min(1, (p-0.8)*5))
        return

    p = get_section_progress(frame, 'composition_summary')
    if p >= 0:
        ax.text(7, 6.8, "AND vs OR: SUMMARY", fontsize=22, ha='center',
               color=COLORS['highlight'], fontweight='bold')

        if p > 0.15:
            alpha = min(1, (p - 0.15) * 3)
            # AND column
            ax.text(4, 5.2, "AND", fontsize=20, ha='center',
                   color=COLORS['accent1'], fontweight='bold', alpha=alpha)
            ax.text(4, 4.3, '"Use all perspectives"', fontsize=11, ha='center',
                   color=COLORS['text'], alpha=alpha, style='italic')
            ax.text(4, 3.5, "Best for:\ncomprehensive\nanalysis", fontsize=10, ha='center',
                   color=COLORS['dim'], alpha=alpha)

            # OR column
            ax.text(10, 5.2, "OR", fontsize=20, ha='center',
                   color=COLORS['accent5'], fontweight='bold', alpha=alpha)
            ax.text(10, 4.3, '"Pick best strategy"', fontsize=11, ha='center',
                   color=COLORS['text'], alpha=alpha, style='italic')
            ax.text(10, 3.5, "Best for:\ndecision\nmaking", fontsize=10, ha='center',
                   color=COLORS['dim'], alpha=alpha)

            # Divider
            ax.plot([7, 7], [3, 5.5], '--', color=COLORS['dim'], alpha=alpha*0.5)

        if p > 0.6:
            ax.text(7, 1.5, "Together, they create flexible reasoning structures",
                   fontsize=14, ha='center', color=COLORS['highlight'], alpha=min(1, (p-0.6)*3))


def draw_pipeline(ax, frame):
    """Complete pipeline"""
    p = get_section_progress(frame, 'pipeline_title')
    if p >= 0:
        alpha = min(1, p * 3)
        ax.text(7, 5.5, "THE COMPLETE PIPELINE", fontsize=26, ha='center',
               color=COLORS['highlight'], fontweight='bold', alpha=alpha)
        if p > 0.4:
            ax.text(7, 4, "From question to thoughtful answer",
                   fontsize=16, ha='center', color=COLORS['text'], alpha=min(1, (p-0.4)*3))
        return

    # Helper to draw persistent elements
    def draw_step_indicator(current_step):
        for i in range(4):
            x = 2 + i * 3.5
            color = COLORS['highlight'] if i < current_step else COLORS['dim']
            ax.text(x, 7.2, f"Step {i+1}", fontsize=10, ha='center', color=color)
            if i < 3:
                ax.plot([x + 0.8, x + 2.7], [7.2, 7.2], '-', color=COLORS['dim'], alpha=0.5)

    p = get_section_progress(frame, 'pipeline_step1')
    if p >= 0:
        draw_step_indicator(1)
        ax.text(7, 6.2, "STEP 1: User Question", fontsize=18, ha='center',
               color=COLORS['accent1'], fontweight='bold')

        if p > 0.2:
            alpha = min(1, (p - 0.2) * 3)
            box = FancyBboxPatch((2, 3), 10, 2, boxstyle="round,pad=0.15",
                                facecolor=COLORS['card'], edgecolor=COLORS['accent1'],
                                alpha=alpha, linewidth=2)
            ax.add_patch(box)
            ax.text(7, 4, '"Explain the relationship between', fontsize=13,
                   ha='center', color=COLORS['text'], alpha=alpha, style='italic')
            ax.text(7, 3.4, 'quantum computing and cryptography"', fontsize=13,
                   ha='center', color=COLORS['text'], alpha=alpha, style='italic')
        return

    p = get_section_progress(frame, 'pipeline_step2')
    if p >= 0:
        draw_step_indicator(2)
        ax.text(7, 6.2, "STEP 2: Encode to Idea Space", fontsize=18, ha='center',
               color=COLORS['accent3'], fontweight='bold')

        # Show question box (smaller)
        box1 = FancyBboxPatch((1, 3.5), 4, 1.5, boxstyle="round,pad=0.1",
                             facecolor=COLORS['card'], edgecolor=COLORS['accent1'],
                             linewidth=1)
        ax.add_patch(box1)
        ax.text(3, 4.25, "Question", fontsize=11, ha='center', color=COLORS['accent1'])

        if p > 0.3:
            alpha = min(1, (p - 0.3) * 3)
            ax.annotate("", xy=(6.5, 4.25), xytext=(5.2, 4.25),
                       arrowprops=dict(arrowstyle="->", color=COLORS['highlight'], lw=2),
                       alpha=alpha)
            ax.text(5.85, 4.9, "encode", fontsize=10, ha='center',
                   color=COLORS['highlight'], alpha=alpha)

        if p > 0.6:
            alpha = min(1, (p - 0.6) * 3)
            box2 = FancyBboxPatch((7, 3.5), 4, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['card'], edgecolor=COLORS['accent3'],
                                 alpha=alpha, linewidth=2)
            ax.add_patch(box2)
            ax.text(9, 4.5, "Latent Vector", fontsize=11, ha='center', color=COLORS['accent3'], alpha=alpha)
            ax.text(9, 3.9, "[0.2, -0.5, 0.8, ...]", fontsize=9, ha='center', color=COLORS['dim'], alpha=alpha)
        return

    p = get_section_progress(frame, 'pipeline_step3')
    if p >= 0:
        draw_step_indicator(3)
        ax.text(7, 6.2, "STEP 3: Grammar Processing", fontsize=18, ha='center',
               color=COLORS['accent4'], fontweight='bold')

        # Grammar box
        if p > 0.1:
            alpha = min(1, (p - 0.1) * 3)
            box = FancyBboxPatch((2, 1.5), 10, 4, boxstyle="round,pad=0.15",
                                facecolor=COLORS['card'], edgecolor=COLORS['accent4'],
                                alpha=alpha, linewidth=2)
            ax.add_patch(box)

        # Mini tree
        if p > 0.25:
            alpha = min(1, (p - 0.25) * 3)
            # Root
            root = Circle((7, 4.3), 0.35, facecolor=COLORS['accent1'], alpha=alpha)
            ax.add_patch(root)
            ax.text(7, 4.3, "AND", fontsize=8, ha='center', va='center', color='white', alpha=alpha)

            # Children
            c1 = Circle((5, 3.2), 0.3, facecolor=COLORS['accent5'], alpha=alpha)
            c2 = Circle((9, 3.2), 0.3, facecolor=COLORS['accent5'], alpha=alpha)
            ax.add_patch(c1)
            ax.add_patch(c2)
            ax.text(5, 3.2, "OR", fontsize=8, ha='center', va='center', color='white', alpha=alpha)
            ax.text(9, 3.2, "OR", fontsize=8, ha='center', va='center', color='white', alpha=alpha)

            ax.plot([7, 5], [3.95, 3.5], 'w-', alpha=alpha*0.5, linewidth=1.5)
            ax.plot([7, 9], [3.95, 3.5], 'w-', alpha=alpha*0.5, linewidth=1.5)

            # Leaves
            for lx in [4, 6, 8, 10]:
                leaf = Circle((lx, 2.2), 0.25, facecolor=COLORS['accent3'], alpha=alpha)
                ax.add_patch(leaf)
            ax.plot([5, 4], [2.9, 2.45], 'w-', alpha=alpha*0.5, linewidth=1)
            ax.plot([5, 6], [2.9, 2.45], 'w-', alpha=alpha*0.5, linewidth=1)
            ax.plot([9, 8], [2.9, 2.45], 'w-', alpha=alpha*0.5, linewidth=1)
            ax.plot([9, 10], [2.9, 2.45], 'w-', alpha=alpha*0.5, linewidth=1)

        if p > 0.6:
            alpha = min(1, (p - 0.6) * 3)
            ax.text(3, 5, "Transform", fontsize=10, color=COLORS['dim'], alpha=alpha)
            ax.text(7, 5, "Combine", fontsize=10, ha='center', color=COLORS['dim'], alpha=alpha)
            ax.text(11, 5, "Refine", fontsize=10, ha='right', color=COLORS['dim'], alpha=alpha)
        return

    p = get_section_progress(frame, 'pipeline_step4')
    if p >= 0:
        draw_step_indicator(4)
        ax.text(7, 6.2, "STEP 4: Decode to Answer", fontsize=18, ha='center',
               color=COLORS['accent2'], fontweight='bold')

        # Refined latent
        box1 = FancyBboxPatch((1, 3.5), 4, 1.5, boxstyle="round,pad=0.1",
                             facecolor=COLORS['card'], edgecolor=COLORS['highlight'],
                             linewidth=2)
        ax.add_patch(box1)
        ax.text(3, 4.25, "Refined Idea", fontsize=11, ha='center', color=COLORS['highlight'])

        if p > 0.3:
            alpha = min(1, (p - 0.3) * 3)
            ax.annotate("", xy=(6.5, 4.25), xytext=(5.2, 4.25),
                       arrowprops=dict(arrowstyle="->", color=COLORS['accent3'], lw=2),
                       alpha=alpha)
            ax.text(5.85, 4.9, "decode", fontsize=10, ha='center',
                   color=COLORS['accent3'], alpha=alpha)

        if p > 0.5:
            alpha = min(1, (p - 0.5) * 2.5)
            box2 = FancyBboxPatch((7, 2.8), 5.5, 2.5, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['card'], edgecolor=COLORS['accent1'],
                                 alpha=alpha, linewidth=2)
            ax.add_patch(box2)
            ax.text(9.75, 4.5, "Thoughtful Answer", fontsize=12, ha='center',
                   color=COLORS['accent1'], alpha=alpha, fontweight='bold')
            ax.text(9.75, 3.7, '"Quantum computing threatens', fontsize=10,
                   ha='center', color=COLORS['text'], alpha=alpha, style='italic')
            ax.text(9.75, 3.2, 'current encryption, but also enables', fontsize=10,
                   ha='center', color=COLORS['text'], alpha=alpha, style='italic')
            ax.text(9.75, 2.7, 'quantum-safe cryptography..."', fontsize=10,
                   ha='center', color=COLORS['text'], alpha=alpha, style='italic')
        return

    p = get_section_progress(frame, 'pipeline_power')
    if p >= 0:
        ax.text(7, 6.2, "THE POWER OF THIS APPROACH", fontsize=20, ha='center',
               color=COLORS['highlight'], fontweight='bold')

        powers = [
            ("Explore", "before committing", COLORS['accent1']),
            ("Combine", "multiple perspectives", COLORS['accent3']),
            ("Refine", "through iteration", COLORS['accent4']),
            ("Produce", "thoughtful answers", COLORS['highlight']),
        ]

        for idx, (verb, rest, color) in enumerate(powers):
            show_at = idx * 0.18
            if p > show_at:
                alpha = min(1, (p - show_at) * 4)
                ax.text(7, 4.8 - idx * 0.9, f"{verb} {rest}", fontsize=16,
                       ha='center', color=color, alpha=alpha)


def draw_evolution(ax, frame):
    """How grammars evolve"""
    p = get_section_progress(frame, 'evolution_title')
    if p >= 0:
        alpha = min(1, p * 3)
        ax.text(7, 5.5, "HOW GRAMMARS IMPROVE", fontsize=26, ha='center',
               color=COLORS['accent3'], fontweight='bold', alpha=alpha)
        if p > 0.4:
            ax.text(7, 4, "Grammars evolve through generations",
                   fontsize=16, ha='center', color=COLORS['text'], alpha=min(1, (p-0.4)*3))
        return

    p = get_section_progress(frame, 'evolution_cycle')
    if p >= 0:
        ax.text(7, 7.2, "THE EVOLUTION CYCLE", fontsize=20, ha='center',
               color=COLORS['accent3'], fontweight='bold')

        # Cycle nodes
        cycle = [
            (3, 4.8, "Population", COLORS['accent1']),
            (7, 6, "Evaluate", COLORS['accent5']),
            (11, 4.8, "Select Best", COLORS['accent3']),
            (9, 2.5, "Mutate", COLORS['accent4']),
            (5, 2.5, "Next Gen", COLORS['highlight']),
        ]

        for idx, (x, y, label, color) in enumerate(cycle):
            show_at = idx * 0.12
            if p > show_at:
                alpha = min(1, (p - show_at) * 4)
                circle = Circle((x, y), 0.8, facecolor=color, alpha=alpha*0.85)
                ax.add_patch(circle)
                ax.text(x, y, label, fontsize=9, ha='center', va='center',
                       color='white', alpha=alpha, fontweight='bold')

                # Arrow to next
                if idx < len(cycle) - 1 and p > show_at + 0.1:
                    next_x, next_y, _, _ = cycle[idx + 1]
                    ax.annotate("", xy=(next_x - 0.7, next_y),
                               xytext=(x + 0.7, y),
                               arrowprops=dict(arrowstyle="->", color='white',
                                             alpha=0.4, connectionstyle="arc3,rad=0.2"))

        # Loop back
        if p > 0.7:
            alpha = min(1, (p - 0.7) * 4)
            ax.annotate("", xy=(3.5, 4), xytext=(4.5, 2.9),
                       arrowprops=dict(arrowstyle="->", color='white',
                                     alpha=alpha*0.4, connectionstyle="arc3,rad=0.3"))

        if p > 0.8:
            ax.text(7, 1, "Each cycle, grammars get better",
                   fontsize=13, ha='center', color=COLORS['text'], alpha=min(1, (p-0.8)*5))
        return

    p = get_section_progress(frame, 'evolution_result')
    if p >= 0:
        ax.text(7, 6.5, "THE RESULT", fontsize=22, ha='center',
               color=COLORS['highlight'], fontweight='bold')

        if p > 0.15:
            alpha = min(1, (p - 0.15) * 3)
            ax.text(4, 5, "Generation 1", fontsize=12, ha='center', color=COLORS['dim'], alpha=alpha)
            ax.text(10, 5, "Generation 50", fontsize=12, ha='center', color=COLORS['dim'], alpha=alpha)

            # Quality bars
            bar1 = FancyBboxPatch((3, 3.5), 2, 0.8, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['accent2'], alpha=alpha*0.8)
            ax.add_patch(bar1)
            ax.text(4, 3.9, "Low", fontsize=10, ha='center', va='center', color='white', alpha=alpha)

            bar2 = FancyBboxPatch((9, 3.5), 2, 2.2, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['accent3'], alpha=alpha*0.8)
            ax.add_patch(bar2)
            ax.text(10, 4.6, "High", fontsize=10, ha='center', va='center', color='white', alpha=alpha)

            ax.annotate("", xy=(8.8, 4.6), xytext=(5.2, 3.9),
                       arrowprops=dict(arrowstyle="->", color=COLORS['highlight'], lw=2), alpha=alpha)

        if p > 0.5:
            ax.text(7, 1.8, "Grammars automatically discover", fontsize=14,
                   ha='center', color=COLORS['text'], alpha=min(1, (p-0.5)*3))
            ax.text(7, 1.1, "effective reasoning patterns", fontsize=14,
                   ha='center', color=COLORS['highlight'], alpha=min(1, (p-0.5)*3), fontweight='bold')


def draw_summary(ax, frame):
    """Final summary"""
    p = get_section_progress(frame, 'summary_title')
    if p >= 0:
        alpha = min(1, p * 3)
        ax.text(7, 5.5, "PUTTING IT ALL TOGETHER", fontsize=26, ha='center',
               color=COLORS['highlight'], fontweight='bold', alpha=alpha)
        return

    p = get_section_progress(frame, 'summary_points')
    if p >= 0:
        ax.text(7, 7, "FRACTAL LATENT GRAMMARS", fontsize=22, ha='center',
               color=COLORS['highlight'], fontweight='bold')

        points = [
            ("1.", "Encode questions into 'idea space'", COLORS['accent1']),
            ("2.", "Use grammar rules to transform ideas", COLORS['accent4']),
            ("3.", "Combine with AND (blend) and OR (select)", COLORS['accent3']),
            ("4.", "Evolve grammars to find best strategies", COLORS['accent5']),
            ("5.", "Decode refined ideas into thoughtful answers", COLORS['highlight']),
        ]

        for idx, (num, text, color) in enumerate(points):
            show_at = idx * 0.15
            if p > show_at:
                alpha = min(1, (p - show_at) * 4)
                ax.text(2, 5.5 - idx * 0.95, num, fontsize=14, color=color, alpha=alpha, fontweight='bold')
                ax.text(2.8, 5.5 - idx * 0.95, text, fontsize=14, color=COLORS['text'], alpha=alpha)

        if p > 0.85:
            ax.text(7, 0.8, "AI that thinks before it speaks",
                   fontsize=16, ha='center', color=COLORS['highlight'],
                   alpha=min(1, (p-0.85)*7), fontweight='bold', style='italic')
        return

    p = get_section_progress(frame, 'end_card')
    if p >= 0:
        alpha = min(1, p * 2)
        ax.text(7, 5, "FRACTAL LATENT GRAMMARS", fontsize=28, ha='center',
               color=COLORS['highlight'], fontweight='bold', alpha=alpha)
        ax.text(7, 3.5, "Latent Space Reasoning Project", fontsize=16, ha='center',
               color=COLORS['text'], alpha=alpha)


def create_master_animation():
    """Create the complete master animation"""
    print(f"Creating master animation ({TOTAL_FRAMES} frames, ~{TOTAL_FRAMES/FPS:.0f} seconds)...")

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=COLORS['bg'])

    def animate(frame):
        ax.clear()
        ax.set_facecolor(COLORS['bg'])
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.axis('off')

        # Call all section drawers - only active one will draw
        draw_title_card(ax, frame)
        draw_challenge(ax, frame)
        draw_traditional(ax, frame)
        draw_latent_approach(ax, frame)
        draw_grammar_concept(ax, frame)
        draw_rules(ax, frame)
        draw_composition(ax, frame)
        draw_pipeline(ax, frame)
        draw_evolution(ax, frame)
        draw_summary(ax, frame)

        # Progress bar at bottom
        progress = frame / TOTAL_FRAMES
        ax.plot([0.5, 0.5 + 13 * progress], [0.15, 0.15], '-',
               color=COLORS['highlight'], linewidth=3, alpha=0.7)
        ax.plot([0.5, 13.5], [0.15, 0.15], '-',
               color=COLORS['dim'], linewidth=1, alpha=0.3)

        return []

    print("  Generating frames...")
    anim = FuncAnimation(fig, animate, frames=TOTAL_FRAMES, interval=1000/FPS, blit=True)

    gif_path = OUT_DIR / "fractal_grammar_explainer.gif"
    print(f"  Saving GIF to {gif_path}...")
    anim.save(gif_path, writer=PillowWriter(fps=FPS),
              savefig_kwargs={'facecolor': COLORS['bg']})
    plt.close()
    print(f"  GIF saved: {gif_path}")

    return gif_path


def convert_to_video(gif_path):
    """Convert GIF to MP4 video using ffmpeg"""
    print("\nConverting to video...")

    video_path = gif_path.with_suffix('.mp4')

    # Try ffmpeg
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', str(gif_path),
            '-movflags', 'faststart',
            '-pix_fmt', 'yuv420p',
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v', 'libx264',
            '-crf', '18',
            '-preset', 'slow',
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  Video saved: {video_path}")
            return video_path
        else:
            print(f"  ffmpeg error: {result.stderr}")
    except FileNotFoundError:
        print("  ffmpeg not found, trying alternative...")

    # Try with moviepy as fallback
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(str(gif_path))
        clip.write_videofile(str(video_path), fps=FPS, codec='libx264')
        clip.close()
        print(f"  Video saved: {video_path}")
        return video_path
    except ImportError:
        print("  moviepy not installed")
    except Exception as e:
        print(f"  moviepy error: {e}")

    # Manual frame-by-frame approach using PIL and cv2
    try:
        from PIL import Image
        import cv2

        # Read GIF frames
        gif = Image.open(gif_path)
        frames = []
        try:
            while True:
                frame = gif.copy().convert('RGB')
                frames.append(np.array(frame))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        if frames:
            height, width = frames[0].shape[:2]
            # Ensure dimensions are even for video encoding
            width = width if width % 2 == 0 else width - 1
            height = height if height % 2 == 0 else height - 1

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, FPS, (width, height))

            for frame in frames:
                frame = frame[:height, :width]
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f"  Video saved: {video_path}")
            return video_path
    except ImportError as e:
        print(f"  Missing library: {e}")
    except Exception as e:
        print(f"  cv2 error: {e}")

    print("  Could not convert to video. Please install ffmpeg or moviepy.")
    return None


def main():
    print("=" * 60)
    print("Creating Master Educational Animation")
    print("=" * 60)

    # Print timing breakdown
    print("\nSection timing:")
    total_time = 0
    for section, duration in TIMING.items():
        total_time += duration
        print(f"  {section}: {duration}s")
    print(f"\nTotal duration: {total_time} seconds ({total_time/60:.1f} minutes)")
    print()

    gif_path = create_master_animation()
    video_path = convert_to_video(gif_path)

    print()
    print("=" * 60)
    print("COMPLETE!")
    print(f"  GIF: {gif_path}")
    if video_path:
        print(f"  Video: {video_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
