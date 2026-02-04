# Unit 4: Autopoietic Judge Co-evolution

## Unit Goal
Design a system where judge and latent population co-evolve, addressing the known scorer weakness.

## Research Sources

### Autopoiesis in AI (2025)
- [Rethinking AI Through Systems Theory](https://www.frontiersin.org/journals/communication/articles/10.3389/fcomm.2025.1585321/full) (May 2025)
- Self-producing, self-maintaining systems
- Boundary between self/environment dynamically regulated
- Structural coupling enables co-evolution

### Free Energy Principle
- [Wikipedia Overview](https://en.wikipedia.org/wiki/Free_energy_principle)
- Systems minimize variational free energy (prediction error)
- Action and perception unified as inference

### Neural Autopoiesis (MIT Press 2020)
- [Paper](https://direct.mit.edu/artl/article/26/1/130/93271/Neural-Autopoiesis-Organizing-Self-Boundaries-by)
- Stimulus avoidance maintains self-boundaries
- STDP enables autonomous self-organization

### Surrogate-Assisted Evolution (2025)
- [Survey](https://link.springer.com/article/10.1007/s41965-024-00165-w)
- Generation-based strategy works best
- Accuracy > 0.6 consistently outperforms no-surrogate

---

## Mathematical Formulation

### State Variables
```
Population at gen t: P_t = {z_i^t} for i=1..N
Decoder: x_i^t = D(z_i^t)
Judge score: s_i^t = J_θ_t(z_i^t, x_i^t, ctx_i^t)
```

### Evolution Dynamics
```
Selection distribution: p_t(z) ∝ exp(β * s(z))
Mutation: z_i^(t+1) = M(z_i^t, mod_i^t, T_t)
```

### External Grounding
```
For subset G_t ⊂ P_t, external signal y_i = E(x_i^t)
```

### Judge Update (Free-Energy Style)
```
L_θ = E_{(z,y) ∈ D_ext}[(J_θ(z) - y)²]           # External grounding
    + λ_rank * RankLoss(J_θ; pairs)              # Ranking consistency
    + λ_homeo * KL(p_θ(s|A) || target)           # Homeostasis
    + λ_drift * ||θ - θ_ema||²                   # Stability

θ_(t+1) = θ_t - η * ∇_θ L_θ
```

---

## Key Design Decisions

### 1. Judge Update Strategy

**Decision**: Generation-based updates every K generations

| Approach | Pros | Cons |
|----------|------|------|
| Online (every gen) | Most responsive | Unstable, expensive |
| Batch (every K gens) | Stable, efficient | Slower adaptation |
| **Hybrid** | Balance | Moderate complexity |

**Implementation**:
- Every generation: micro-update with tiny η
- Every K generations: full batch update
- Keep EMA of θ for stability

### 2. External Grounding Signal

**Options** (in order of cost/quality):
1. **Task-based validation**: factuality, retrieval accuracy
2. **LLM judge**: High-cost frontier model evaluation
3. **Human evaluation**: Gold standard but expensive
4. **Anchor set**: Fixed tasks with known scores

**Recommended**: Combination of anchor set (cheap, stable) + periodic LLM judge (expensive, accurate)

### 3. Preventing Judge-Latent Collusion

**Multi-layer defense**:

| Mechanism | How It Helps |
|-----------|--------------|
| Two-time-scale | Judge slow, evolution fast |
| Anchor holdout | Fixed validation set detects drift |
| Adversarial negatives | High-judge/low-external examples |
| Trust gating | Weight judge by correlation |
| Boundary penalty | OOD latents penalized |

### 4. Homeostatic Diversity Maintenance

**Control loop**:
```
D_t = population_diversity(P_t)
D* = target diversity (e.g., 0.4)

T_(t+1) = clamp(T_t * exp(k_T * (D* - D_t)), T_min, T_max)
w_div_(t+1) = clamp(w_div_t + k_w * (D* - D_t), 0, w_max)
```

**Additional mechanisms**:
- Novelty archive for exploration bonus
- Stability monitor: freeze judge if oscillation detected
- Boundary loss: high prediction error → avoidance (neural autopoiesis)

---

## Architecture

```
         ┌─────────────────────────────────────────┐
         │           External Evaluator E          │
         │  (LLM judge, task validation, anchors)  │
         └─────────────────┬───────────────────────┘
                           │ y_i (expensive)
                           ▼
┌──────────────────────────────────────────────────────────┐
│                    AutopoieticJudgePanel                 │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │ Internal Judge│  │ Experience    │  │ Homeostasis  │ │
│  │ J_θ(z,x,ctx)  │  │ Buffer        │  │ Controller   │ │
│  └───────┬───────┘  └───────┬───────┘  └──────┬───────┘ │
│          │                  │                  │         │
│          └──────────────────┴──────────────────┘         │
└────────────────────────┬─────────────────────────────────┘
                         │ s_i, calibrated
                         ▼
              ┌─────────────────────┐
              │   EvolutionLoop     │
              │  selection/mutation │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Latent Population  │
              │     P_t = {z_i}     │
              └─────────────────────┘
```

---

## Integration with EvolutionLoop

### Hook Points in `loop.py`

```python
class EvolutionLoop:
    def run(self, seed_latent, query):
        # ...existing code...

        for gen in range(self.config.generations):
            # 1. Evaluate with judge
            scores = self.judge_panel.evaluate(chains, context)

            # NEW: Store to experience buffer
            self.experience_buffer.add(chains, scores, context)

            # NEW: Sample for external evaluation
            if gen % self.config.external_eval_freq == 0:
                subset = self._sample_for_external(chains)
                external_scores = self.external_evaluator.evaluate(subset)
                self.experience_buffer.add_external(subset, external_scores)

            # NEW: Update judge every K generations
            if gen % self.config.judge_update_freq == 0:
                self.judge_panel.update(self.experience_buffer)
                self.judge_panel.calibrate()

            # NEW: Homeostasis control
            diversity = population_diversity(chains)
            self._update_temperature(diversity)

            # ...existing selection/mutation code...
```

### New Components

```python
class ExternalEvaluator:
    """Interface for expensive external validation."""
    def evaluate(self, chains: List[ChainState]) -> List[float]:
        # Call LLM judge, task validator, etc.
        pass

class ExperienceBuffer:
    """Stores (z, x, ctx, s_internal, s_external) tuples."""
    def add(self, chains, scores, context): ...
    def add_external(self, chains, external_scores): ...
    def sample_batch(self, n): ...

class AutopoieticJudgePanel(JudgePanel):
    """Judge that updates online with external grounding."""
    def update(self, buffer: ExperienceBuffer):
        batch = buffer.sample_batch(self.batch_size)
        loss = self._compute_loss(batch)
        self.optimizer.step(loss)
        self.theta_ema = 0.99 * self.theta_ema + 0.01 * self.theta

    def calibrate(self):
        # Adjust trust weight based on anchor performance
        anchor_corr = self._eval_anchors()
        self.trust_weight = max(0.3, anchor_corr)
```

---

## Hyperparameters

| Parameter | Recommended | Range | Notes |
|-----------|-------------|-------|-------|
| judge_update_freq (K) | 5 | 3-10 | Generations between judge updates |
| external_eval_freq | 1 | 1-5 | How often to sample for external |
| external_sample_size | 3 | 1-5 | Latents evaluated externally per gen |
| judge_lr (η) | 1e-4 | 1e-5 to 1e-3 | Judge learning rate |
| ema_decay | 0.99 | 0.9-0.999 | EMA for theta stability |
| target_diversity (D*) | 0.4 | 0.3-0.6 | Homeostasis target |
| k_T | 0.1 | 0.05-0.2 | Temperature control gain |

---

## Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Reward hacking | High | High | Anchors + correlation gating |
| Collusion collapse | Medium | High | Adversarial negatives + external checks |
| Over-regularization | Medium | Medium | Novelty bonus + diversity control |
| Budget blow-up | Medium | Medium | Stratified sampling + K-gen batching |
| Instability/oscillation | Medium | Medium | Two-time-scale + EMA |
| Stale grounding | Low | Medium | Periodic anchor refresh |

---

## Revolutionary Potential

**Why This Matters**:
- **Addresses core weakness**: Current scorer has 0.07 correlation with external quality
- **Self-improving**: Judge gets better as it sees more data
- **Adaptive**: Responds to distribution shift in latent population
- **Robust**: Multiple layers of anti-exploitation defense

**Potential Impact**:
- 2-5x improvement in external quality correlation
- More stable evolution (less exploitation of scorer quirks)
- Better generalization to new query types

---

## Unit 4 → Unit 5 Handoff

**Key Finding**: Co-evolutionary dynamics with external grounding + homeostasis provides a principled framework for adaptive scoring

**Synergy with Previous Units**:
- Unit 2 (QD): Autopoietic judge can score archive entries
- Unit 3 (Fractals): Grammar rules can co-evolve with judge

**Next Unit Recommendation**: Category Theory formalization
- Provides mathematical foundation for all previous frameworks
- Could unify QD, Fractals, and Autopoiesis

**Open Questions**:
1. What's the right balance of internal vs external evaluation?
2. How to handle non-stationary external evaluators?
3. Can we learn the homeostasis target D* adaptively?
