# Unit 3: Fractal Latent Grammars

## Unit Goal
Design a novel architecture where evolution operates on recursive RULES that generate latents, rather than on flat 1024-dim vectors.

## Research Sources

### Neural Collages (NeurIPS 2022)
- [Paper](https://arxiv.org/abs/2204.07673)
- [GitHub](https://github.com/ermongroup/self-similarity-prior)
- Represent data as parameters of self-referential transformations
- Based on Partitioned Iterated Function Systems (PIFS)
- Hypernetwork generates fractal code in single forward pass
- Resolution-agnostic via affine transformations
- Trained VDVAEs with collage parameters as latent variables

### Navigating Latent Space Dynamics (May 2025)
- [Paper](https://arxiv.org/abs/2505.22785)
- Iteratively applying encoding-decoding creates vector field
- Attractor points emerge from standard training
- No additional training needed - just iterate

### AOGNets (CVPR 2019)
- [Paper](https://arxiv.org/abs/1711.05847)
- AND-OR Grammar as network generator
- Splits features into N groups, parses as 'sentence'
- Compositional + reconfigurable

### IFS/PIFS Foundations
- Self-similarity encoded as contractive affine maps
- Fixed point of iterated application = fractal
- Very high compression ratios (up to 10,000:1)

---

## Core Architecture: Fractal Latent Grammar (FLG)

### What is a Rule?

A rule is a **reusable, contractive transform** with a composition type:

```
Rule r = (W_r, b_r, P_r, type_r, gate_r)

T_r(z) = P_r * f(W_r @ z + b_r)

Where:
- W_r ∈ R^{1024×1024}, constrained: ||W_r||_2 ≤ c < 1 (contractive)
- b_r ∈ R^{1024} (bias)
- P_r is a projection or block mask (for interpretability)
- type_r ∈ {AND, OR}
- gate_r controls OR selections
- f is nonlinearity (tanh or relu)
```

### How Rules Compose

The grammar is a tree/DAG with nodes labeled by rules:

```
For node n with rule r and children C(n):

If n is leaf:
    z_n = P_n @ v_n  (trainable seed vector)

If n is AND:
    z_n = Σ_{c ∈ children(n)} α_{n,c} * T_{r_n}(z_c)

If n is OR:
    c* = argmax(gate_{n,c})
    z_n = T_{r_n}(z_{c*})

Final latent: z = z_root
```

**Optional attractor projection**:
```
z ← E(D(z))  repeated k times
```

### Architecture Diagram

```
        Evolution
   (grammar + rule params)
             |
             v
   +---------------------+
   | Fractal Grammar G   |
   |  - 4-8 rules        |
   |  - AND/OR tree      |
   |  - depth 3-5        |
   +---------------------+
             |
      recursive expand
             v
      z ∈ R^1024
             |
     optional: z ← E(D(z))^k
             |
             v
       Decoder D
             |
         Output
             |
          Fitness
```

---

## What Evolution Optimizes

### Structure
- Tree shape (depth, branching)
- Rule IDs at nodes
- AND/OR choices

### Parameters
- W_r, b_r per rule
- P_r (projection masks)
- α weights for AND aggregation
- gate_r for OR selection
- v_n leaf seed vectors

### Regularizers
- Contraction strength: ||W_r||_2 < 0.9
- Sparsity of P_r
- Diversity of rule usage

---

## Keeping Latents Meaningful

1. **Distribution Matching**: Enforce mean/var or MMD between z and E(x) latents
2. **Cycle Consistency**: Penalize ||z - E(D(z))|| to stay on decoder manifold
3. **Attractor Projection**: Run k iterations of E(D(.)) to land on stable latents
4. **Latent Validity**: Discourage large norms or out-of-range values

---

## Minimal Implementation

### Configuration
- Rules: 4 rules total
- Each rule uses block-diagonal W_r (interpretable)
- Depth: 3-4, binary tree
- Composition: AND = weighted sum, OR = hard choice

### Pseudocode

```python
class Rule:
    def __init__(self, latent_dim=1024, block_size=256):
        self.W = nn.Parameter(torch.eye(latent_dim) * 0.9)  # Contractive
        self.b = nn.Parameter(torch.zeros(latent_dim))
        self.P = nn.Parameter(torch.eye(latent_dim))  # Projection

    def apply(self, z):
        return self.P @ torch.tanh(self.W @ z + self.b)

class GrammarNode:
    def __init__(self, rule_id, node_type, children=None):
        self.rule_id = rule_id
        self.type = node_type  # 'AND', 'OR', or 'LEAF'
        self.children = children or []
        self.alpha = None  # For AND
        self.gate = None   # For OR
        self.seed = None   # For LEAF

def expand(node, rules, depth):
    if depth == 0 or node.type == 'LEAF':
        return node.seed

    child_latents = [expand(c, rules, depth-1) for c in node.children]
    rule = rules[node.rule_id]

    if node.type == 'AND':
        weighted = sum(a * rule.apply(z) for a, z in zip(node.alpha, child_latents))
        return weighted
    else:  # OR
        c_star = node.gate.argmax()
        return rule.apply(child_latents[c_star])

# Evolution fitness
def evaluate(grammar, rules, encoder, decoder, target_latent):
    z = expand(grammar.root, rules, depth=4)

    # Optional attractor projection
    for _ in range(3):
        decoded = decoder(z)
        z = encoder(decoded)

    # Fitness
    sim = cosine_similarity(z, target_latent)
    cycle_loss = (z - encoder(decoder(z))).norm()

    return sim - 0.1 * cycle_loss
```

---

## Integration with Existing System

### What Changes
- Replace direct latent evolution with grammar evolution
- Add `FractalGrammar` class to `src/latent_reasoning/`
- Modify `EvolutionLoop` to evolve grammars instead of vectors

### What Stays
- Encoder E unchanged
- Decoder D unchanged
- Scorer/Judge unchanged (scores the generated latent z)

---

## Advantages Over Flat Latent Evolution

| Aspect | Flat (Current) | Fractal Grammars |
|--------|----------------|------------------|
| Parameters | 1024 per individual | ~100-500 per grammar |
| Compositionality | None | Explicit tree structure |
| Interpretability | Opaque | Rules map to subspaces |
| Stability | Can diverge | Guaranteed (contraction) |
| Compression | 1:1 | 10-100x potential |
| Reuse | None | Rules shared across nodes |

---

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Collapse to trivial fixed points | High | Lower contraction (0.7 instead of 0.9) |
| Decoder exploitation | High | Cycle consistency loss, attractor projection |
| Search space explosion | Medium | Limit depth (4), rules (8), branching (2) |
| Dense W_r loses interpretability | Medium | Use block-diagonal or low-rank |
| Training instability | Medium | Spectral normalization on W_r |

---

## Revolutionary Potential Assessment

**Novelty**: HIGH - No one has applied fractal grammars to latent space reasoning
**Difficulty**: HIGH - Complex implementation, many design choices
**Potential Payoff**: VERY HIGH
- Could enable compositional multi-step reasoning
- Natural hierarchy maps to reasoning depth
- Rule reuse could capture reasoning patterns

---

## Unit 3 → Unit 4 Handoff

**Key Finding**: FLG provides a mathematically grounded framework for compositional latent generation via recursive rules

**Recommended Next Unit**: Autopoietic Judge Co-evolution
- Addresses the known scorer weakness
- Could combine with FLG: co-evolve rules AND judge

**Open Questions**:
1. How to initialize rules to avoid trivial collapse?
2. What's the right balance of AND vs OR nodes?
3. Can we learn the grammar structure or must we fix it?
4. How does depth affect reasoning quality?
