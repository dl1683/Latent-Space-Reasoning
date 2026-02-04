# Unit 6: Framework Synthesis

## Unit Goal
Unify all frameworks (QD, Fractals, Autopoiesis, Category Theory) into a coherent architecture.

---

## Unified Architecture

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

## Framework Integration Matrix

| Framework | Role in Unified System | Categorical Structure |
|-----------|----------------------|----------------------|
| QD (DNS) | Explore grammar space | Store comonad S |
| Fractals | Generate latents compositionally | F-algebra |
| Autopoiesis | Adaptive scoring with grounding | State monad J |
| Category Theory | Composition semantics | Kleisli arrows, distributive laws |

---

## Key Integration Decisions

### 1. QD Operates on Grammar Space (Not Raw Latents)
**Why**: Grammars preserve compositional structure; evolving grammars is more meaningful than evolving flat vectors

**How**:
- Archive stores `(Grammar, BD, Quality)` tuples
- Grammar → sample k latents → aggregate BD → evaluate
- DNS maintains diversity in BD space

### 2. Grammar-to-Latent is Stochastic
**Why**: Same grammar can produce different latents (AND node weights, OR node choices)

**How**:
- `Grammar.sample_latents(k, rng)` returns k diverse latents
- Aggregate scoring: mean or top-k of judge scores
- BD computed from latent statistics

### 3. Judge Scores Individual Latents, Not Grammars
**Why**: Latents are what get decoded; quality is at latent level

**How**:
- `Judge.score(latent) → float`
- Grammar quality = aggregate of latent scores
- External evaluator validates decoded outputs

### 4. Two-Time-Scale Updates
**Why**: Prevent judge-grammar collusion

**How**:
- Many grammar generations per judge update
- Homeostasis adjusts novelty/quality weights
- Judge frozen during grammar evolution phases

---

## Data Flow Specification

```
INPUTS:
  - Initial grammar population P₀
  - Judge parameters θ₀
  - External evaluator E (expensive)
  - Archive state A₀ (empty or seeded)
  - Homeostasis target D*

MAIN LOOP (gen = 1..G):
  1. SELECT grammar G from archive A via DNS policy
  2. VARY G → G' via mutation/crossover (Kleisli T)
  3. EXPAND G' → {z₁, ..., zₖ} via recursive AND/OR tree
  4. COMPUTE BD from {zᵢ} using RFF projection
  5. SCORE each zᵢ with judge J → {s₁, ..., sₖ}
  6. AGGREGATE Q(G') = mean/top-k of {sᵢ}
  7. COMPUTE novelty N(G') = distance to archive neighbors
  8. UPDATE archive A with (G', BD, Q + αN)

  IF gen % K == 0:  # Slow time-scale
    9a. SAMPLE subset for external evaluation
    9b. UPDATE judge J with external grounding
    9c. CALIBRATE via homeostasis controller

OUTPUTS:
  - Archive A_final with diverse high-quality grammars
  - Best grammar G* and its latent samples
  - Evolved judge J_final
```

---

## Component Interfaces

### Grammar (F-algebra)
```python
class Grammar:
    rules: List[Rule]           # Contractive affine transforms
    tree: TreeNode              # AND/OR structure
    params: GrammarParams       # Weights, gates, seeds

    def sample_latents(self, k: int, rng: RNG) -> List[Tensor]:
        """Expand grammar recursively, sample k latents."""
        pass

    def mutate(self, rng: RNG, temperature: float) -> Grammar:
        """Mutate rules and/or tree structure."""
        pass

    def crossover(self, other: Grammar, rng: RNG) -> Grammar:
        """Combine two grammars."""
        pass

    def signature(self) -> GrammarSig:
        """Structural summary for BD computation."""
        pass
```

### Descriptor (RFF)
```python
class RFFDescriptor:
    projector: RFFProjector     # Fixed random Fourier features

    def compute(self, latents: List[Tensor]) -> np.ndarray:
        """Compute 8-16 dim BD from latent statistics."""
        stats = self._compute_stats(latents)  # mean, var, etc.
        return self.projector.project(stats)

    def distance(self, bd1: np.ndarray, bd2: np.ndarray) -> float:
        """L2 distance in BD space."""
        return np.linalg.norm(bd1 - bd2)
```

### Judge (State Monad)
```python
class AutopoieticJudge:
    model: LatentScorer         # Neural network
    ema_params: Tensor          # EMA for stability
    trust_weight: float         # Confidence in internal scoring

    def score(self, latent: Tensor, context: Context) -> float:
        """Score individual latent."""
        pass

    def update(self, batch: List[Tuple[Tensor, float]]) -> None:
        """Train on (latent, external_score) pairs."""
        pass

    def calibrate(self, homeostasis: HomeostasisState) -> None:
        """Adjust trust weight based on correlation."""
        pass
```

### Archive (Store Comonad)
```python
class DNSArchive:
    entries: List[ArchiveEntry]  # (grammar, bd, quality)
    novelty_k: int               # Neighbors for novelty

    def insert(self, grammar: Grammar, bd: np.ndarray, quality: float) -> None:
        """DNS insertion: keep if dominates or novel."""
        pass

    def select(self, rng: RNG, policy: SelectionPolicy) -> Grammar:
        """Sample grammar weighted by quality + novelty."""
        pass

    def novelty(self, bd: np.ndarray) -> float:
        """Average distance to k nearest neighbors."""
        pass

    def stats(self) -> ArchiveStats:
        """Coverage, quality distribution, etc."""
        pass
```

### Controller (Homeostasis)
```python
class HomeostasisController:
    target_diversity: float
    k_T: float                  # Temperature control gain
    k_w: float                  # Weight control gain

    def adjust(self, archive_stats: ArchiveStats,
               judge_stats: JudgeStats) -> ControlParams:
        """Compute new temperature, novelty weight, etc."""
        diversity = archive_stats.coverage
        delta = self.target_diversity - diversity

        new_temp = clamp(current_temp * exp(self.k_T * delta))
        new_novelty_weight = clamp(current_weight + self.k_w * delta)

        return ControlParams(temperature=new_temp,
                           novelty_weight=new_novelty_weight)
```

---

## Categorical Composition

### How Components Fit Together

```
S (Archive) ────extract────> Grammar ────expand────> Latent
     ↑                          │                      │
     │                          │ Kleisli T            │ Functor E
     │                          ↓                      ↓
     └────insert────< (Grammar, BD, Q) <────score────< Judge J
```

### Distributive Law: T and J
```
The evolution monad T and judge state monad J interact via:

T(J(X)) → J(T(X))

This ensures scoring distributes over evolution:
- Evolve grammars, then score
- OR score, then evolve
- Both give same result (up to isomorphism)
```

---

## Phased Implementation Roadmap

### Phase 1: Grammar Engine (Week 1-2)
- Implement Rule class with contractive transforms
- Build AND/OR tree structure
- Create Grammar.sample_latents() with recursive expansion
- Test: grammar → latent → decode produces coherent output

### Phase 2: QD Archive (Week 2-3)
- Implement RFFDescriptor with fixed projector
- Build DNSArchive with novelty computation
- Create selection policy (quality + novelty weighted)
- Test: archive maintains diversity across generations

### Phase 3: Autopoietic Judge (Week 3-4)
- Wrap existing LatentScorer in AutopoieticJudge
- Add update() method with external grounding
- Implement two-time-scale update schedule
- Test: judge correlation improves over time

### Phase 4: Homeostasis (Week 4-5)
- Implement HomeostasisController
- Connect to archive stats and judge stats
- Add adaptive novelty/quality weighting
- Test: system maintains target diversity

### Phase 5: Integration (Week 5-6)
- Wire all components into unified evolution loop
- Add categorical interfaces for swappability
- Comprehensive testing with real queries
- Benchmark against baseline system

---

## Key Tradeoffs

| Decision | Pros | Cons |
|----------|------|------|
| QD on grammars (not latents) | Compositional, interpretable | Stochastic evaluation |
| RFF for BD | Fast, differentiable | Less interpretable |
| Two-time-scale judge | Stable, prevents collusion | Slower adaptation |
| DNS over MAP-Elites | No grid needed, high-dim friendly | More complex implementation |
| Aggregate scoring (mean) | Robust to outliers | Misses best-case quality |

---

## Revolutionary Potential

**This unified architecture enables**:
1. **Compositional reasoning** via grammar structure
2. **Diverse solutions** via QD archive
3. **Adaptive evaluation** via autopoietic judge
4. **Principled composition** via category theory

**Key innovation**: QD over grammar space with autopoietic scoring is entirely novel - no prior work combines these.

---

## Unit 6 → Unit 7+ Handoff

**Key Achievement**: Complete unified architecture with all frameworks integrated

**Next Units**:
- Unit 7-10: Deep dives into specific components (grammar mutations, BD design, etc.)
- Unit 11-25: Experiment design and ablation studies
- Unit 26-50: Application to specific reasoning tasks
- Unit 51-100: Optimization, scaling, and frontier exploration
