# Unit 1: Theoretical Landscape Discovery

## Unit Goal
Establish ground truth about theoretical frameworks that could revolutionize latent space reasoning.

## Subagents Run
1. **Web Research Scout** - Category theory, QD algorithms, self-organizing systems, fractals
2. **Codex Analyst** - First-principles synthesis of each framework's applicability

---

## Key Findings by Framework

### 1. Category Theory & Monads

**Core Insight:** Treat encode→evolve→decode as a compositional program with explicit algebraic structure, making latent transformations lawful and interoperable.

**Concrete Mechanism:**
- Define latent space as an object in a category
- Define evolution as a monad (stochasticity + selection as effects)
- Implement monad algebras to enforce equivariance
- Replace RNG-only decoding with a **functor from latent morphisms into decoder conditioning space**

**Key Papers:**
- [Categorical Deep Learning is an Algebraic Theory of All Architectures](https://arxiv.org/abs/2402.15332) (2024)
- [Category Theory Framework for DNNs](https://dl.acm.org/doi/10.1145/3759355.3759375) (ACM 2025)
- [Category-Theoretical and Topos-Theoretical Frameworks in ML Survey](https://www.mdpi.com/2075-1680/14/3/204) (March 2025)

**Potential Experiment:**
- Implement monadic EA where mutation/crossover are Kleisli arrows
- Compare semantic consistency under latent perturbations
- Test if commutative diagrams predict decoder outcomes

---

### 2. Quality Diversity (MAP-Elites, Novelty Search)

**Core Insight:** Don't optimize a single "best latent" - build a structured repertoire of diverse high-quality latents.

**Concrete Mechanism:**
- Run QD directly in latent space with learned behavioral descriptors
- Use Dominated Novelty Search (2025) to avoid rigid grids
- Maintain archive keyed by learned descriptors
- Score by both judge AND task-specific correctness proxy

**Key Papers:**
- [Dominated Novelty Search](https://arxiv.org/html/2502.00593v1) (February 2025) - removes need for grid structure
- [MAP-Elites in latent spaces of VAEs](https://arxiv.org/pdf/2102.12463) - already proven for game level generation
- [Quality-Diversity Algorithms](https://quality-diversity.github.io/papers.html) - comprehensive list

**Potential Experiment:**
- Compare archive coverage and decoding quality to baseline EA
- Measure "latent coverage" via distance-to-archive
- Test if novel latents yield better decoder conditioning

---

### 3. Self-Organizing Systems & Autopoiesis

**Core Insight:** Create a closed, self-maintaining latent ecology where evaluation, mutation, and selection emerge from internal constraints.

**Concrete Mechanism:**
- Make the judge a dynamical subsystem that co-evolves with latents
- Introduce free-energy minimization: latents predict their own decoded outputs
- Prediction error drives evolution
- Add boundary-maintenance: penalize diversity collapse

**Key Papers:**
- [From Intelligence to Autopoiesis](https://www.frontiersin.org/journals/communication/articles/10.3389/fcomm.2025.1585321/full) (May 2025)
- [Self-orthogonalizing attractor networks from free energy principle](https://arxiv.org/html/2505.22749v1) (May 2025)
- [Neural Autopoiesis](https://direct.mit.edu/artl/article/26/1/130/93271/Neural-Autopoiesis-Organizing-Self-Boundaries-by)

**Potential Experiment:**
- Build recurrent latent ecosystem with online-trained judge
- Compare stability (diversity retention), coherence, emergence of attractor basins

---

### 4. Fractals & Self-Similarity

**Core Insight:** Encode reasoning as self-similar transformations so the system can compress and reuse structure across scales.

**Concrete Mechanism:**
- Define a fractal latent generator: recursive transformation rules that expand into full latents
- Use Neural Collage-style operators for self-referential transforms
- Decoding conditions on rule sets rather than single vector
- Enables scalable, interpretable compositional reasoning

**Key Papers:**
- [Neural Collages: Differentiable Fractal Representations](https://arxiv.org/abs/2204.07673)
- [Self-Similarity Analysis in Deep Neural Networks](https://arxiv.org/html/2507.17785) (July 2025)
- [FractalNet: Ultra-Deep Networks without Residuals](https://arxiv.org/abs/1605.07648)

**Potential Experiment:**
- Replace direct latents with "latent grammars" (recursive rule sets)
- Compare compression ratio, generalization, controllability
- Evaluate if scaling recursion depth yields coherent multi-step reasoning

---

## Cross-Framework Synergies

| Framework A | Framework B | Synergy |
|-------------|-------------|---------|
| Category Theory | Quality Diversity | QD as exploring morphism space |
| Category Theory | Autopoiesis | Monads formalize closed loops |
| Category Theory | Fractals | Endofunctors with fixed points model recursion |
| Quality Diversity | Autopoiesis | Archive as self-sustaining ecology |
| Quality Diversity | Fractals | Multi-scale novelty descriptors |
| Autopoiesis | Fractals | Self-similarity as internal regularizer |

---

## Revolutionary Potential Ranking

1. **Quality Diversity + Latent Space** - Immediately applicable, proven in VAE latents
2. **Fractal Latent Grammars** - Novel, high compression, interpretable
3. **Autopoietic Judge Co-evolution** - Self-improving system, addresses scorer weakness
4. **Categorical Pipeline Formalization** - Theoretical foundation, enables principled design

---

## Open Questions

1. How to learn behavioral descriptors for QD without supervision?
2. Can fractal generators be differentiably learned?
3. What's the minimal autopoietic closure that yields stability?
4. How to define the "correct" category for latent space reasoning?

---

## Cycle Handoff Pack

**Unit Goal:** Theoretical Landscape Discovery
**Cycle Number:** 1/4 (Discovery)
**Subagents Run:** Web Research (4 parallel), Codex Analyst
**Key Decisions Made:**
- All four frameworks have revolutionary potential
- QD most immediately applicable
- Fractal grammars most novel
**Artifacts Produced:** This document, research links, Codex synthesis
**Open Questions:** See above
**Next Cycle Must Start By:** Synthesizing frameworks into concrete design proposals
**Acceptance Criteria for Next Cycle:** Executable experiment designs with clear success metrics
