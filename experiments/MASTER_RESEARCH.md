# Latent Space Reasoning - Research Master Document

## Mission
Explore advanced theoretical frameworks and discover how they can revolutionize our latent space reasoning system.

## Theoretical Domains Under Investigation
1. **Category Theory & Monads** - Compositional structures, functors, natural transformations
2. **Evolutionary Algorithms** - Beyond basic GA, coevolution, novelty search, MAP-Elites
3. **Self-Organizing Systems** - Autopoiesis, emergence, swarm intelligence
4. **Fractals & Information Geometry** - Self-similarity, compression, manifold structure
5. **Neural Cellular Automata** - Local neural rules → global self-organization (NEW: Jan 2026)

## Current System Limitations (Baseline)
- Weak latent-to-decode coupling (only RNG seeding)
- Scorer doesn't evaluate correctness, only style
- Diversity collapses without explicit bonus
- Evolution searches mostly over random seeds, not semantic space
- No learned manifold projection

---

## Unit Progress Tracker

| Unit | Focus | Key Finding | Status |
|------|-------|-------------|--------|
| 1 | Theoretical Landscape Discovery | QD most applicable, Fractals most novel, Autopoiesis for scorer | Complete |
| 2 | Quality Diversity Deep Dive | DNS+RFF is minimal viable QD path | Complete |
| 3 | Fractal Latent Grammars | AND/OR recursive rules with contractive transforms | Complete |
| 4 | Autopoietic Judge Co-evolution | External grounding + homeostasis for adaptive scoring | Complete |
| 5 | Category Theory Formalization | Monads for evolution, functors for encode/decode, algebras for grammars | Complete |
| 6 | Framework Synthesis | Unified architecture: QD on grammar space with autopoietic judge | Complete |
| 7-10 | Advanced Theory | Grammar mutation, BD design, latent geometry, scaling | Complete |
| 11-25 | Experiment Design | Detailed experimental protocols for all hypotheses | Complete |
| 26-50 | Cross-Domain Applications | Math, code, creative, language, scientific reasoning | Complete |
| 51-100 | Frontier Exploration | Neural grammars, meta-learning, theoretical frontiers | Complete |

---

## Accumulated Insights

### Category Theory & Monads
- **Evolution monad T**: Stochastic variation + selection; `T(X) = Dist(Pop(X))`
- **Encode/decode functors**: `E: Txt → Lat`, `D: Lat → Txt`; approximate adjunction
- **QD archive as store comonad**: Context-dependent selection
- **Fractal grammar as F-algebra**: Recursive expansion via catamorphisms
- **Autopoietic judge as state monad**: Co-evolving scorer with external grounding
- Key papers: Categorical Deep Learning (ICML 2024), Category-Theoretical ML Survey (March 2025)

### Evolutionary Algorithms (Quality Diversity)
- **Key insight**: Don't optimize single "best latent" - build diverse repertoire
- **DNS (Dominated Novelty Search)**: Feb 2025, gridless QD, outperforms MAP-Elites in high-dim
- **VQ-Elites**: April 2025, learns BDs unsupervised via VQ-VAE
- **AutoQD**: June 2025, random Fourier features for BD embedding
- **Implementation path**: RFF for 8-16 dim BD, DNS for archive management

### Self-Organizing Systems (Autopoiesis)
- **Free-energy principle**: Systems minimize prediction error
- **Neural autopoiesis**: Stimulus avoidance maintains self-boundaries (MIT Press 2020)
- **Judge co-evolution**: Two-time-scale updates prevent collusion
- **Homeostasis controller**: `T_{t+1} = T_t * exp(k * (D* - D_t))`
- **External grounding**: Periodic expensive evaluation calibrates internal judge

### Fractals & Information Geometry
- **Neural Collages (NeurIPS 2022)**: Differentiable fractal representations via PIFS
- **Rule definition**: `T_r(z) = P_r * f(W_r @ z + b_r)` with ||W_r||_2 < 1 (contractive)
- **AND/OR composition**: Weighted sum vs gated selection
- **Attractor projection**: `z ← E(D(z))` iterated for stability
- Key insight: Grammars enable compositional, interpretable latent generation

### Neural Cellular Automata (NEW: Jan 2026)
- **Key paper**: "Neural cellular automata: applications to biology and beyond classical AI" (Hartl, Levin, Pio-Lopez, BioSystems 2025)
- **Core idea**: Embed ANNs as local decision-making centers; global behavior emerges from local rules
- **Connection to our architecture**: Grammar rules ≈ NCA update rules; both use trainable neural components
- **Latent space relevance**: NCA Manifold framework represents developmental trajectories in structured latent spaces
- **Reasoning applications**: Successfully applied to ARC-AGI reasoning benchmark
- **Iterative refinement**: NCA state updates parallel our attractor projection `z ← E(D(z))`
- **Stochastic extensions**: Mixture of NCA (MNCA) adds probabilistic rule selection (like our OR nodes)
- **Key insight**: Validates local-to-global self-organization for AI reasoning tasks

### Cross-Domain Synthesis (Unit 6 Unified Architecture)
- **QD on grammar space**: Archive stores (Grammar, BD, Quality), not raw latents
- **Grammar → Latent → Score pipeline**: Recursive expansion, RFF BD, autopoietic judge
- **Categorical composition**: Store comonad (archive) + evolution monad (T) + state monad (judge)
- **Two-level diversity**: Grammar diversity (archive) + latent diversity (per-grammar)

---

## UNIFIED ARCHITECTURE (Unit 6 Output)

```
            ┌────────────────────────────────────────────────────┐
            │                    Control Loop                    │
            │  (two-time-scale, homeostasis, novelty/quality)    │
            └───────────────┬────────────────────────────────────┘
                            │
                      ┌─────▼─────┐
                      │ QD Archive │  Store comonad S
                      │  (DNS)     │  (diverse high-quality grammars)
                      └─────┬─────┘
                            │ select/variation (Kleisli T)
                      ┌─────▼─────┐
                      │ Grammar G │  F-algebra (AND/OR tree + rules)
                      └─────┬─────┘
                            │ expand (recursive)
                      ┌─────▼─────┐
                      │ Latent z  │
                      └─────┬─────┘
                 ┌──────────┴───────────┐
                 │                      │
          ┌──────▼──────┐        ┌──────▼──────┐
          │ BD via RFF  │        │ Autopoietic │  State monad J
          │ (8–16 dim)  │        │   Judge     │  + external grounding
          └──────┬──────┘        └──────┬──────┘
                 │                     │
                 └──────────┬──────────┘
                            │ score + novelty
                      ┌─────▼─────┐
                      │   Update  │  Archive + Judge
                      └───────────┘
```

---

## Experiment Designs (Not Yet Run)

### Experiment 1: QD-Grammar System (Priority: HIGH)
- **Hypothesis**: QD on grammar space produces more diverse, higher-quality outputs than flat latent evolution
- **Variables**:
  - Grammar depth (3-5), rules (4-8), BD dim (8-16)
  - Archive size (100-500), novelty k (5-15)
- **Metrics**: Archive coverage, QD-score, output diversity, task accuracy
- **Baseline**: Current flat latent EA with diversity bonus

### Experiment 2: Autopoietic Judge (Priority: HIGH)
- **Hypothesis**: Co-evolving judge maintains higher external correlation than static scorer
- **Variables**:
  - Judge update frequency (3-10 gens)
  - External eval budget (1-5 per gen)
  - Homeostasis target (0.3-0.6)
- **Metrics**: Judge-external correlation, diversity over time, stability
- **Baseline**: Current static trained latent scorer

### Experiment 3: Fractal Compression (Priority: MEDIUM)
- **Hypothesis**: Grammar parameters << flat latent dimension with comparable quality
- **Variables**:
  - Total grammar params vs 1024-dim latent
  - Recursion depth
- **Metrics**: Compression ratio, reconstruction quality, generalization
- **Baseline**: Direct 1024-dim latent optimization

### Experiment 4: Category Theory Tests (Priority: LOW)
- **Hypothesis**: Monad laws and naturality detect bugs before deployment
- **Tests**:
  - Associativity: `(mut ; sel ; mut) == (mut ; (sel ; mut))`
  - Unit laws: Starting from seed consistency
  - Naturality: Encoder swap compatibility
- **Metrics**: Law violation rate, bug detection rate

---

## Implementation Candidates (Ranked by Potential)

1. **QD-Grammar System** (Units 2+3+6)
   - Immediately novel: QD on grammar space unprecedented
   - Addresses diversity collapse completely
   - Enables compositional reasoning

2. **Autopoietic Judge** (Unit 4)
   - Addresses known scorer weakness (0.07 correlation)
   - Self-improving over time
   - External grounding prevents exploitation

3. **Category Theory Framework** (Unit 5)
   - Provides formal foundation
   - Enables principled module swapping
   - Debugging via law checking

4. **Fractal Grammars Alone** (Unit 3)
   - High compression potential
   - Interpretable structure
   - Novel contribution

---

## Key Research Questions (Open)

### Theoretical
1. What is the correct symmetry group G for latent space?
2. Can we prove encode-decode adjunction formally?
3. What is minimal autopoietic closure for stability?

### Practical
1. How to learn behavioral descriptors without supervision?
2. What's the right RFF gamma for our latent space?
3. How to initialize grammar rules to avoid trivial collapse?

### Novel Directions
1. Can grammar structure serve as behavioral descriptors? (QD-Fractal synergy)
2. Can rules co-evolve with judge? (Fractal-Autopoiesis synergy)
3. Topos-theoretic view of transformer conditioning?

---

## Unit Artifacts Index

| Unit | Artifact | Path |
|------|----------|------|
| 1 | Discovery Summary | `unit_artifacts/unit_001_discovery.md` |
| 1 | Experiment Designs | `unit_artifacts/unit_001_experiment_designs.md` |
| 2 | QD Deep Dive | `unit_artifacts/unit_002_qd_deepdive.md` |
| 3 | Fractal Grammars | `unit_artifacts/unit_003_fractal_grammars.md` |
| 4 | Autopoietic Judge | `unit_artifacts/unit_004_autopoietic_judge.md` |
| 5 | Category Theory | `unit_artifacts/unit_005_category_theory.md` |
| 6 | Synthesis | `unit_artifacts/unit_006_synthesis.md` |

---

## Unit Artifacts Index (Complete)

| Units | Artifact | Path |
|-------|----------|------|
| 1 | Discovery Summary | `unit_artifacts/unit_001_discovery.md` |
| 1 | Experiment Designs | `unit_artifacts/unit_001_experiment_designs.md` |
| 2 | QD Deep Dive | `unit_artifacts/unit_002_qd_deepdive.md` |
| 3 | Fractal Grammars | `unit_artifacts/unit_003_fractal_grammars.md` |
| 4 | Autopoietic Judge | `unit_artifacts/unit_004_autopoietic_judge.md` |
| 5 | Category Theory | `unit_artifacts/unit_005_category_theory.md` |
| 6 | Synthesis | `unit_artifacts/unit_006_synthesis.md` |
| 7-10 | Advanced Theory | `unit_artifacts/unit_007_010_advanced_theory.md` |
| 11-25 | Experiments | `unit_artifacts/unit_011_025_experiments.md` |
| 26-50 | Applications | `unit_artifacts/unit_026_050_applications.md` |
| 51-100 | Frontier | `unit_artifacts/unit_051_100_frontier.md` |

---

## Changelog
- Unit 1 Complete: Discovered 4 frameworks, designed 3 experiments
- Unit 2 Complete: DNS+RFF identified as minimal viable QD path
- Unit 3 Complete: Fractal grammar architecture with AND/OR trees
- Unit 4 Complete: Autopoietic judge with external grounding + homeostasis
- Unit 5 Complete: Category theory formalization (monads, functors, algebras)
- Unit 6 Complete: Unified architecture synthesizing all frameworks
- Units 7-10 Complete: Grammar mutation, BD design, latent geometry, scaling
- Units 11-25 Complete: Detailed experiment protocols
- Units 26-50 Complete: Cross-domain applications (math, code, creative, etc.)
- Units 51-100 Complete: Frontier exploration and research agenda

**ALL 100 UNITS COMPLETE**

---

## New Research Connections (Post-100 Units)

### Neural Cellular Automata Integration (Jan 2026)

**Discovery**: The BioSystems 2025 review paper on NCA by Hartl, Levin, and Pio-Lopez provides strong theoretical and empirical support for our unified architecture approach.

**Key Parallels**:
| Our Architecture | NCA Framework |
|------------------|---------------|
| Grammar rules `T_r(z)` | Local update rules with embedded NNs |
| AND/OR tree composition | Local-to-global emergence |
| Attractor projection `z ← E(D(z))` | Iterative state refinement |
| OR nodes (probabilistic selection) | MNCA stochastic rule assignments |
| Grammar structure as BD | NCA Manifold latent representations |

**Novel Synthesis Opportunity**:
- **NCA-Grammar Hybrid**: Replace fixed grammar rules with learnable NCA-style update rules
- **Differentiable grammar expansion**: End-to-end trainable via NCA gradients
- **Multi-scale reasoning**: NCA's hierarchical structure matches reasoning decomposition

**New Experiment Design: NCA-Grammar Fusion**
```yaml
experiment:
  name: nca_grammar_hybrid
  hypothesis: NCA-style learnable rules outperform fixed grammar rules

  conditions:
    A_fixed_grammar:
      rules: handcrafted contractive transforms

    B_nca_grammar:
      rules: learned via NCA training (gradient or evolution)
      state_channels: 16
      update_steps: 4-8

  metrics:
    - rule_generalization: Performance on unseen queries
    - training_efficiency: Samples to convergence
    - interpretability: Can we visualize learned rules?
```

**Related Papers to Explore**:
- arXiv:2509.11131 - Neural cellular automata: applications to biology and beyond classical AI
- arXiv:2506.20486 - Mixtures of Neural Cellular Automata (MNCA)
- arXiv:2506.15746 - Neural Cellular Automata for ARC-AGI

---

## EVALUATION PHILOSOPHY

### Manual Review is the Most Important Metric

**CRITICAL**: Statistical accuracy metrics (correct/total, QD-score, coverage) are useful for quick validation but are **NOT SUFFICIENT** for evaluating output quality.

### Required Evaluation Protocol

1. **Human Review**: Always have a human review actual text outputs for coherence and usefulness
2. **AI Review**: Use Codex or another AI to review outputs for reasoning quality
3. **Beyond Accuracy**: Check that answers are not just "correct" but well-reasoned
4. **Usefulness Check**: Verify outputs are sensible, coherent, and actually helpful

### What Automated Metrics Miss

- **Correct but flawed**: Right answer with broken reasoning
- **Pattern matching**: Nonsensical but accidentally correct responses
- **Quality drift**: Subtle degradation that doesn't affect accuracy
- **Unhelpful correctness**: Technically correct but useless outputs

### Case Study: False Positive Detection (Jan 2026)

**Scenario**: Standard mode output for "Who is the shortest?" question was marked CORRECT by automated tests but was actually WRONG.

**What happened**:
- Question: "If Alice is taller than Bob, and Bob is taller than Carol, who is the shortest?"
- Model output stated "Bob is shorter than Carol" (WRONG - inverted the premise)
- Model concluded "Bob is the shortest" (WRONG answer)
- But "Carol" appeared in the text during reasoning
- Simple substring matching marked it CORRECT

**Detection method**: Codex review identified the logical error:
```
- High: Misstates premise as "Bob < Carol" instead of "Bob > Carol"
- High: Concludes Bob is shortest, contradicting Alice > Bob > Carol
- Medium: Internal inconsistency in reasoning
```

**Fix**: Improved `check_correct()` to extract actual final answer from `\boxed{}` or explicit "Answer:" statements, not just search for correct answer anywhere in text.

**Lesson**: Even simple transitive reasoning questions can expose model errors that automated tests miss. Manual review or AI-assisted review (Codex) is essential.

### How to Run Manual Review

```bash
# Use Codex to review outputs
codex exec "Review these model outputs for quality, coherence, and reasoning.
Grade each on: (1) Reasoning quality, (2) Coherence, (3) Usefulness.
Be critical - don't just say 'good', identify specific issues:

[paste outputs here]"
```

### Evaluation Hierarchy (Most to Least Important)

1. **Manual human review** - Is this actually useful and well-reasoned?
2. **AI-assisted review** - Does Codex find issues we missed?
3. **Qualitative analysis** - Do outputs make sense in context?
4. **Statistical accuracy** - Does it get the right answer?
5. **Internal metrics** - Scores, diversity, coverage (least reliable alone)
