# Units 7-10: Advanced Theory Deep Dives

## Unit 7: Grammar Mutation Strategies

### Key Question
How should we mutate fractal grammars to balance exploration and exploitation?

### Mutation Types

#### 1. Rule Parameter Mutation
```python
def mutate_rule_params(rule, temperature):
    """Mutate W, b, P while preserving contractivity."""
    # Perturb weight matrix
    delta_W = torch.randn_like(rule.W) * temperature
    new_W = rule.W + delta_W

    # Enforce contractivity via spectral normalization
    sigma = torch.linalg.matrix_norm(new_W, ord=2)
    if sigma >= 1.0:
        new_W = new_W * 0.9 / sigma

    # Perturb bias
    delta_b = torch.randn_like(rule.b) * temperature
    new_b = rule.b + delta_b

    return Rule(W=new_W, b=new_b, P=rule.P)
```

#### 2. Tree Structure Mutation
```python
def mutate_tree(node, temperature):
    """Mutate AND/OR tree structure."""
    p = random.random()

    if p < 0.3:  # Swap AND ↔ OR
        node.type = 'OR' if node.type == 'AND' else 'AND'
    elif p < 0.5:  # Add child
        node.children.append(create_leaf_node())
    elif p < 0.7 and len(node.children) > 1:  # Remove child
        node.children.pop(random.randint(0, len(node.children)-1))
    elif p < 0.9:  # Change rule assignment
        node.rule_id = random.randint(0, num_rules-1)
    # else: no structural change

    return node
```

#### 3. Depth-Adaptive Mutation
```python
def adaptive_mutation(grammar, generation, max_gen):
    """Start with high exploration, end with exploitation."""
    progress = generation / max_gen

    # Early: structure mutations
    # Late: parameter refinement
    if random.random() > progress:
        return mutate_tree(grammar.root, temperature=1.0)
    else:
        return mutate_rule_params(grammar.rules, temperature=0.1)
```

### Crossover Strategies

#### 1. Subtree Exchange
```python
def subtree_crossover(g1, g2):
    """Exchange subtrees between two grammars."""
    node1 = select_random_node(g1.root)
    node2 = select_random_node(g2.root)

    # Swap subtrees
    swap_subtrees(node1, node2)
    return g1, g2
```

#### 2. Rule Interpolation
```python
def rule_crossover(g1, g2, alpha=0.5):
    """Blend rule parameters from two grammars."""
    new_rules = []
    for r1, r2 in zip(g1.rules, g2.rules):
        new_W = alpha * r1.W + (1 - alpha) * r2.W
        new_b = alpha * r1.b + (1 - alpha) * r2.b
        new_rules.append(Rule(W=new_W, b=new_b, P=r1.P))
    return Grammar(rules=new_rules, tree=g1.tree)
```

---

## Unit 8: Behavioral Descriptor Design

### Key Question
What behavioral descriptors best capture semantic diversity for text reasoning?

### Option 1: Latent Statistics (RFF-based)
```python
class LatentStatsBD:
    """BD from latent vector statistics."""

    def __init__(self, bd_dim=16, gamma=0.1):
        self.rff = RFFProjector(input_dim=1024*4, output_dim=bd_dim, gamma=gamma)

    def compute(self, latents: List[Tensor]) -> np.ndarray:
        # Compute statistics across latent samples
        stack = torch.stack(latents)
        stats = torch.cat([
            stack.mean(dim=0),      # Mean
            stack.std(dim=0),       # Std
            stack.max(dim=0)[0],    # Max
            stack.min(dim=0)[0],    # Min
        ])
        return self.rff.project(stats)
```

### Option 2: Grammar Structure BD
```python
class GrammarStructureBD:
    """BD from grammar tree structure."""

    def compute(self, grammar: Grammar) -> np.ndarray:
        features = [
            grammar.depth,                  # Tree depth
            grammar.num_nodes,              # Total nodes
            grammar.num_and_nodes / grammar.num_nodes,  # AND ratio
            grammar.num_or_nodes / grammar.num_nodes,   # OR ratio
            len(set(grammar.rule_ids)),     # Unique rules used
            grammar.avg_branching_factor,   # Avg children per node
        ]
        return np.array(features)
```

### Option 3: Decoded Output BD
```python
class DecodedOutputBD:
    """BD from decoded text semantics."""

    def __init__(self, embed_model="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embed_model)
        self.pca = PCA(n_components=16)

    def compute(self, decoded_texts: List[str]) -> np.ndarray:
        embeddings = self.embedder.encode(decoded_texts)
        # Use mean embedding reduced to BD dim
        mean_embed = embeddings.mean(axis=0)
        return self.pca.transform(mean_embed.reshape(1, -1))[0]
```

### Recommended: Hybrid BD
```python
class HybridBD:
    """Combine multiple BD sources."""

    def compute(self, grammar, latents, decoded) -> np.ndarray:
        # 4 dims from structure
        struct_bd = self.structure_bd.compute(grammar)[:4]
        # 6 dims from latent stats
        latent_bd = self.latent_bd.compute(latents)[:6]
        # 6 dims from decoded semantics
        decoded_bd = self.decoded_bd.compute(decoded)[:6]

        return np.concatenate([struct_bd, latent_bd, decoded_bd])  # 16 total
```

---

## Unit 9: Latent Space Geometry

### Key Question
What geometric structure does our latent space have, and how can we exploit it?

### Manifold Hypothesis
- Latent vectors lie on low-dimensional manifold in R^1024
- Evolution should follow manifold, not arbitrary directions

### Geometric Analysis Tools

#### 1. Intrinsic Dimensionality Estimation
```python
def estimate_intrinsic_dim(latents: List[Tensor], method='mle'):
    """Estimate manifold dimension via maximum likelihood."""
    from sklearn.neighbors import NearestNeighbors

    X = torch.stack(latents).numpy()
    nn = NearestNeighbors(n_neighbors=20)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    # MLE estimator
    k = 10
    mu = distances[:, 1:k+1].mean(axis=1)
    dim = 1 / (np.log(mu[:, -1] / mu[:, 0]) / np.log(k))
    return np.median(dim)
```

#### 2. Curvature Estimation
```python
def estimate_curvature(latents: List[Tensor]):
    """Estimate local curvature via second derivatives."""
    # Fit local quadratic approximation
    # High curvature → need smaller mutation steps
    pass
```

### Geometry-Aware Mutations

#### 1. Tangent Space Mutation
```python
def tangent_mutation(latent, neighbors, temperature):
    """Mutate along estimated tangent space."""
    # Compute local PCA to estimate tangent space
    X = torch.stack(neighbors)
    mean = X.mean(dim=0)
    centered = X - mean
    U, S, V = torch.linalg.svd(centered)

    # Keep top-k principal components as tangent space
    k = min(10, len(neighbors))
    tangent_basis = V[:k]

    # Project random noise onto tangent space
    noise = torch.randn(k) * temperature
    delta = tangent_basis.T @ noise

    return latent + delta
```

#### 2. Geodesic Crossover
```python
def geodesic_crossover(z1, z2, alpha=0.5):
    """Interpolate along geodesic (approximated as straight line)."""
    # For Euclidean approximation
    return alpha * z1 + (1 - alpha) * z2

    # For more accurate geodesic, would need:
    # - Riemannian metric tensor
    # - Exponential/log maps
```

### Symmetry Groups

#### Potential Symmetries in Latent Space
1. **Permutation symmetry**: Reordering tokens shouldn't change meaning
2. **Rotation symmetry**: Encoding orientation arbitrary
3. **Scale symmetry**: Magnitude vs direction

#### Equivariant Evolution
```python
def equivariant_mutation(latent, group_element):
    """Mutation that respects symmetry group."""
    # Apply group action before mutation
    transformed = group_element @ latent
    mutated = mutate(transformed)
    # Apply inverse to return to canonical frame
    return group_element.T @ mutated
```

---

## Unit 10: Scaling and Efficiency

### Key Question
How to scale the unified architecture to practical use cases?

### Computational Bottlenecks

| Component | Cost | Optimization Strategy |
|-----------|------|----------------------|
| Grammar expansion | O(depth × branching) | Cache partial expansions |
| BD computation | O(k × latent_dim) | Batch RFF projection |
| Judge scoring | O(k × model_size) | Batch inference |
| Archive novelty | O(archive_size) | KD-tree for NN search |
| External eval | O(LLM_cost) | Amortize via sampling |

### Optimization Strategies

#### 1. Lazy Grammar Evaluation
```python
class LazyGrammar:
    """Only expand grammar when needed."""

    def __init__(self, grammar):
        self.grammar = grammar
        self._latent_cache = None
        self._bd_cache = None

    @property
    def latent(self):
        if self._latent_cache is None:
            self._latent_cache = self.grammar.expand()
        return self._latent_cache

    def invalidate(self):
        """Call after mutation."""
        self._latent_cache = None
        self._bd_cache = None
```

#### 2. Batched Judge Scoring
```python
def batch_score(judge, latents: List[Tensor], batch_size=32):
    """Score latents in batches for GPU efficiency."""
    scores = []
    for i in range(0, len(latents), batch_size):
        batch = torch.stack(latents[i:i+batch_size])
        batch_scores = judge.forward(batch)
        scores.extend(batch_scores.tolist())
    return scores
```

#### 3. Approximate Novelty with KD-Tree
```python
class FastNoveltyArchive:
    """Use KD-tree for O(log n) novelty computation."""

    def __init__(self, k=10):
        self.k = k
        self.tree = None
        self.bds = []

    def add(self, bd):
        self.bds.append(bd)
        self.tree = KDTree(np.array(self.bds))

    def novelty(self, bd):
        if len(self.bds) < self.k:
            return 1.0  # Maximum novelty for sparse archive
        distances, _ = self.tree.query(bd.reshape(1, -1), k=self.k)
        return distances.mean()
```

#### 4. Adaptive External Evaluation
```python
class AdaptiveExternalEval:
    """Evaluate externally only when uncertain."""

    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def should_evaluate(self, judge_score, judge_confidence):
        """Evaluate if judge is uncertain or score is borderline."""
        if judge_confidence < self.threshold:
            return True
        if 0.4 < judge_score < 0.6:  # Borderline
            return True
        return False
```

### Scaling Targets

| Scale | Archive Size | Gens | Compute | Use Case |
|-------|-------------|------|---------|----------|
| Tiny | 50 | 20 | 1 GPU-min | Quick experiments |
| Small | 200 | 50 | 10 GPU-min | Development |
| Medium | 500 | 100 | 1 GPU-hour | Validation |
| Large | 2000 | 500 | 10 GPU-hours | Research |
| XL | 10000 | 2000 | 100 GPU-hours | Publication |

---

## Units 7-10 Summary

| Unit | Topic | Key Insight |
|------|-------|-------------|
| 7 | Grammar Mutation | Depth-adaptive: structure early, params late |
| 8 | BD Design | Hybrid (structure + latent + decoded) best |
| 9 | Latent Geometry | Tangent space mutations respect manifold |
| 10 | Scaling | KD-tree + batching + lazy eval = 10x speedup |

## Next Units Direction
- Units 11-25: Detailed experiment protocols
- Units 26-50: Application to specific reasoning tasks
- Units 51-100: Frontier exploration and optimization
