# Unit 2: Quality Diversity Deep Dive

## Unit Goal
Design concrete QD integration for latent space reasoning with actionable architecture decisions.

## Research Sources (2025)

### VQ-Elites (April 2025)
- [Paper](https://arxiv.org/abs/2504.08057)
- Uses VQ-VAE to learn behavioral descriptors unsupervised
- Organizes behavior space into structured grid via clustering latent representations
- No manual BD definition needed - learns from data

### AutoQD (June 2025)
- [Paper](https://arxiv.org/abs/2506.05634)
- Uses random Fourier features to embed policy occupancy measures
- Theoretically grounded BD generation
- Distances reflect meaningful behavioral differences

### Dominated Novelty Search (Feb 2025)
- [Paper](https://arxiv.org/abs/2502.00593)
- [GitHub](https://github.com/adaptive-intelligent-robotics/Dominated-Novelty-Search)
- Gridless QD using dynamic fitness transformations
- No predefined bounds or parameters needed
- Outperforms MAP-Elites in high-dimensional spaces

### AURORA (Foundational)
- Trains autoencoder on generated trajectories
- Latent space of autoencoder = behavior space
- Online training as QD progresses

---

## Key Architecture Decisions

### 1. Behavioral Descriptors for Text Reasoning

**Decision: Hybrid Features**

| Component | What It Captures | Dimension |
|-----------|------------------|-----------|
| Output embedding cluster | Semantic strategy | 4-8 dims |
| Structural stats | Step count, token length | 2-4 dims |
| Latent trajectory | Delta norms, directions | 4-8 dims |

**Total BD dimension: 8-16** (compact enough for efficient search)

**Rationale**: Text reasoning doesn't have obvious "behavior" like robotics. We must define it via:
- **Output semantics**: Different responses to same query
- **Reasoning structure**: How the latent evolved
- **Latent geometry**: Where in the space the solution lives

### 2. Grid vs Gridless

**Decision: Gridless DNS as primary, with optional VQ-Elites archive**

**Rationale**:
- Our 1024-dim latent space has unknown bounds
- DNS requires no predefined grid structure
- DNS outperforms MAP-Elites in high-dimensional spaces (proven in 2025 paper)
- Can add VQ-Elites archive for structured analysis later

### 3. Handling 1024-dim Latent Space

**Decision: Random Fourier Features (RFF) for dimensionality reduction**

```python
class RFFProjector:
    def __init__(self, input_dim=1024, output_dim=16, gamma=0.1):
        self.W = torch.randn(input_dim, output_dim // 2) * gamma
        self.b = torch.rand(output_dim // 2) * 2 * np.pi

    def project(self, x):
        z = x @ self.W + self.b
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)
```

**Why RFF over autoencoder**:
- No training required (instant use)
- Preserves relative distances (kernel approximation)
- AutoQD (June 2025) validates this approach

### 4. Ensuring Semantic Diversity

**Decision: Multi-objective fitness with semantic distance check**

```python
def compute_qd_fitness(latent, decoded_text, archive):
    # Primary fitness from scorer
    fitness = scorer.score(latent)

    # Behavioral descriptor
    bd = compute_bd(latent, decoded_text)

    # Novelty (distance to nearest neighbors in BD space)
    novelty = archive.novelty(bd, k=10)

    # Combined QD fitness (DNS style)
    qd_fitness = fitness + alpha * novelty

    # Semantic diversity check (decoded output must differ)
    decoded_embedding = embed_text(decoded_text)
    if archive.has_similar_output(decoded_embedding, threshold=0.95):
        qd_fitness *= 0.5  # Penalize semantic duplicates

    return qd_fitness, bd
```

---

## Integration Architecture

### New Module Structure
```
src/latent_reasoning/qd/
├── __init__.py
├── behavior.py      # BD computation
├── archive.py       # DNS/VQ-Elites archives
├── novelty.py       # Novelty metrics
└── manager.py       # QDManager orchestration
```

### Integration Points

**1. config.py - New QDConfig**
```python
class QDConfig(BaseModel):
    enabled: bool = False
    bd_dim: int = 16
    novelty_k: int = 10
    novelty_weight: float = 0.3
    archive_type: Literal["dns", "vq_elites", "hybrid"] = "dns"
    semantic_threshold: float = 0.95
```

**2. loop.py - QD in evolution cycle**
```python
# After scoring, before selection:
if self.qd_manager:
    bds = self.qd_manager.compute_bds(chains)
    novelty_scores = self.qd_manager.compute_novelty(bds)
    scores = self.qd_manager.combine_fitness(scores, novelty_scores)
    self.qd_manager.update_archive(chains, bds, scores)
```

**3. selection.py - QD-aware selection**
```python
class QDSelection(SelectionStrategy):
    """Selection that samples from QD archive for parent selection."""

    def select(self, chains, scores, bds):
        # Sample parents from diverse regions of archive
        parents = self.archive.sample_diverse(n=self.config.survivors)
        return parents
```

---

## Hyperparameter Recommendations

| Parameter | Recommended | Range | Notes |
|-----------|-------------|-------|-------|
| bd_dim | 16 | 8-32 | Lower = faster, higher = more expressive |
| novelty_k | 10 | 5-20 | sqrt(population) is common heuristic |
| novelty_weight | 0.3 | 0.1-0.5 | Balance fitness vs exploration |
| rff_gamma | 0.1 | 0.01-1.0 | Controls kernel bandwidth |
| semantic_threshold | 0.95 | 0.9-0.99 | Higher = stricter duplicate detection |

---

## Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| BD doesn't capture meaningful behavior | Medium | High | Ablate BD components, try AURORA online learning |
| Novelty dominates fitness | Low | Medium | Tune novelty_weight, use DNS dominance |
| Archive grows too large | Low | Low | Prune by fitness, limit size |
| Decode semantic duplicates | Medium | Medium | Embed decoded text, penalize similarity |
| Computational overhead | Medium | Medium | Lazy BD computation, batch novelty |

---

## Minimal Viable Implementation (MVP)

**Phase 1: Simple DNS (2-3 hours code)**
1. Add RFF projector for BD computation
2. Implement simple unstructured archive (list of (latent, bd, fitness))
3. Add novelty bonus to existing diversity_weight mechanism
4. No config changes - hardcode for testing

**Phase 2: Full Integration (1-2 days code)**
1. QDConfig in config.py
2. QDManager class
3. Proper archive with insertion/pruning
4. QD-aware selection strategy

**Phase 3: Advanced (future units)**
1. VQ-Elites for structured archive
2. Online autoencoder (AURORA-style)
3. Semantic diversity via decoded embedding

---

## Unit 2 → Unit 3 Handoff

**Key Finding**: DNS + RFF is the minimal viable path to QD integration

**Next Unit Focus**: Should investigate:
1. **Fractal Latent Grammars** - Most novel approach from Unit 1
2. OR **Autopoietic Judge** - Addresses scorer weakness directly

**Recommendation**: Unit 3 → Fractal Latent Grammars (higher novelty potential)

**Open Questions Carried Forward**:
1. What's the right RFF gamma for our latent space?
2. How much does semantic duplicate detection help?
3. Can we learn better BDs online?
