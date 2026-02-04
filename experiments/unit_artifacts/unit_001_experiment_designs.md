# Unit 1: Experiment Designs (Cycle 2 Output)

## Experiment 1: Quality Diversity (MAP-Elites) in Latent Space

### Complexity: Medium
### Priority: HIGH (immediately applicable, proven in VAE latents)

### Core Hypothesis
QD algorithms maintain diverse archives of high-quality solutions, preventing premature convergence that plagues our current system.

### Implementation Blueprint

**Phase 1: Behavioral Descriptor Learning**
```
Input: latent vectors from encoder
Output: 2-3D behavioral descriptors (BDs)

Option A: PCA on latent space → use top 2-3 components as BDs
Option B: Train small autoencoder, use bottleneck as BDs
Option C: Use semantic similarity to reference "anchor" prompts as axes
```

**Phase 2: Archive Structure**
```
- Grid-based (classic MAP-Elites): discretize BD space into cells
- CVT-based: use Voronoi tessellation for flexible boundaries
- Dominated Novelty (DNS): remove grid entirely, use Pareto dominance
```

**Phase 3: Selection & Mutation**
```
1. Sample parent from archive (uniform or fitness-proportional)
2. Apply mutation in latent space
3. Evaluate: (fitness, behavioral_descriptors)
4. Insert into archive if:
   - Cell is empty, OR
   - New solution has higher fitness than occupant
```

### Key Variables to Tune
| Variable | Range | Purpose |
|----------|-------|---------|
| grid_resolution | 10-50 per dimension | Archive granularity |
| bd_dimensions | 2-4 | Descriptor complexity |
| selection_strategy | uniform/fitness/curiosity | Exploration balance |
| mutation_sigma | 0.1-2.0 | Search radius |

### Success Metrics
1. **Archive coverage**: % of cells filled after N evaluations
2. **QD-score**: sum of fitness across all filled cells
3. **Decode diversity**: semantic distance between decoded outputs
4. **Final answer quality**: judge score on best decoded response

### Failure Modes & Mitigations
| Failure | Detection | Mitigation |
|---------|-----------|------------|
| BD collapse | Low archive coverage | Increase BD dimensions, use DNS |
| Fitness stagnation | QD-score plateau | Increase mutation, add niching |
| Decoder ignores latent | Random outputs | Train latent-conditioned decoder |

---

## Experiment 2: Fractal Latent Grammars

### Complexity: High
### Priority: MEDIUM (most novel, high potential)

### Core Hypothesis
Instead of evolving flat latent vectors, evolve recursive transformation rules that generate latents. This enables compositional reasoning and better compression.

### Implementation Blueprint

**Phase 1: Grammar Definition**
```
Grammar = {
    rules: List[TransformationRule],
    axiom: Tensor,  # Starting point
    depth: int      # Recursion depth
}

TransformationRule = {
    condition: Tensor → bool,  # When to apply
    transform: Tensor → Tensor  # What to do
}
```

**Phase 2: Latent Generation**
```
def generate_latent(grammar, depth):
    current = grammar.axiom
    for _ in range(depth):
        for rule in grammar.rules:
            if rule.condition(current):
                current = rule.transform(current)
    return current
```

**Phase 3: Evolution on Grammars**
```
- Mutate rule parameters (weights, biases)
- Add/remove rules
- Adjust recursion depth
- Modify axiom
```

### Key Variables
| Variable | Range | Purpose |
|----------|-------|---------|
| max_rules | 3-10 | Grammar complexity |
| rule_hidden_dim | 64-256 | Transform capacity |
| max_depth | 2-8 | Recursion limit |
| axiom_dim | Same as latent | Starting vector |

### Success Metrics
1. **Compression ratio**: grammar_params / latent_dim
2. **Generalization**: performance on unseen query types
3. **Interpretability**: can we understand what rules do?
4. **Scaling**: does deeper recursion yield better reasoning?

### Failure Modes
| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Grammar collapse | All rules identical | Diversity pressure on rules |
| Recursion explosion | NaN/Inf values | Gradient clipping, layer norm |
| No semantic structure | Random outputs | Pre-train on meaningful data |

---

## Experiment 3: Autopoietic Judge Co-evolution

### Complexity: High
### Priority: MEDIUM-HIGH (addresses scorer weakness)

### Core Hypothesis
The judge and latent population should co-evolve, creating a self-sustaining system where evaluation criteria emerge from interaction.

### Implementation Blueprint

**Phase 1: Online Judge Training**
```
JudgeUpdate:
1. Sample latent batch from population
2. Decode to text
3. Get external signal (task completion, coherence metric)
4. Train judge to predict external signal from latent
```

**Phase 2: Co-evolutionary Loop**
```
for generation in range(max_generations):
    # Evolve latents using current judge
    latents = evolve(population, judge)

    # Update judge using external validation
    external_scores = validate(decode(latents))
    judge.train(latents, external_scores)

    # Check for homeostasis
    if diversity(population) < threshold:
        inject_random_latents()
```

**Phase 3: Free Energy Minimization**
```
- Latents predict their own decoded outputs
- Prediction error drives evolution
- Judge measures prediction accuracy
- System minimizes surprise while maintaining diversity
```

### Key Variables
| Variable | Range | Purpose |
|----------|-------|---------|
| judge_update_freq | 1-10 generations | Co-evolution speed |
| external_signal | task_score/coherence/diversity | What judge learns |
| homeostasis_threshold | 0.3-0.7 | Diversity floor |
| prediction_weight | 0.0-1.0 | Self-prediction importance |

### Success Metrics
1. **Judge-external correlation**: does internal scoring match external validation?
2. **Stability**: does system maintain diversity over time?
3. **Emergence**: do new evaluation criteria appear?
4. **Task performance**: final answer quality

### Failure Modes
| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Mode collapse | Diversity crash | Homeostasis injection |
| Judge-latent collusion | High internal, low external scores | External validation frequency |
| Oscillation | Scores fluctuate wildly | Slower judge updates, momentum |

---

## Cross-Experiment Synergies

| Experiment A | Experiment B | Synergy |
|--------------|--------------|---------|
| QD | Fractals | Use grammar structure as behavioral descriptors |
| QD | Autopoiesis | Archive as ecological niche structure |
| Fractals | Autopoiesis | Rules co-evolve with judge |

---

## Recommended Execution Order

1. **QD (MAP-Elites)** - Lowest risk, highest immediate impact
2. **Autopoietic Judge** - Addresses known scorer weakness
3. **Fractal Grammars** - Most novel, requires more theory work first

---

## Unit 1 → Unit 2 Handoff

**Key Decision**: Unit 2 should deep-dive into Quality Diversity implementation details, specifically:
1. How to learn behavioral descriptors without supervision
2. Whether to use grid-based or gridless (DNS) archives
3. Integration with existing evolution loop

**Open Research Questions**:
1. Can we use the encoder's internal structure to define natural behavioral descriptors?
2. What's the right balance between fitness and novelty?
3. How does archive size scale with latent dimension?
