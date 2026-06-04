# Attractor Landscape Mapping (ALM) — Deep System Design

## Status: DESIGN PHASE — Codex Review Required

## Why ALM Is Priority #2

Codex Wave 2 Review: "ALM is the best Wave 2 contribution because it directly supports decorrelation measurement and duplicate-basin suppression."

ALM is not just an architecture — it's the MEASUREMENT INFRASTRUCTURE for CDE. Without knowing the attractor landscape, CDE operates blind: it can measure error correlation post-hoc but can't predict or prevent trajectory duplication during generation.

ALM answers the most fundamental question: **How many distinct reasoning paths does this model have for this task, and what determines which path is taken?**

---

## 1. Theoretical Foundation

### The Attractor Basin Model

For a frozen autoregressive model M, generation can be modeled as a discrete dynamical system:

```
s_{t+1} = f(s_t, x)
```

where:
- s_t = model state at step t (KV cache + current position)
- x = input (question + any prefix)
- f = one step of generation (forward pass + argmax/sample)

With greedy decoding, this system is DETERMINISTIC given the input. Different inputs (different prefixes) start the system at different initial conditions, which may converge to different ATTRACTORS.

**An attractor basin** is the set of initial conditions (prefixes) that all converge to the same output trajectory. If the model has K attractor basins for a given question, there are at most K distinct outputs achievable by prefix perturbation.

### Why Basins Exist

Transformer generation has strong convergence properties:
1. **Autoregressive feedback**: Each generated token becomes context for the next. Small initial differences may amplify or attenuate.
2. **Softmax attention**: Attention is approximately winner-take-all for peaked distributions. Minor input changes may not shift the winning attention target.
3. **Layer normalization**: LayerNorm constrains activations to a sphere, creating natural basin boundaries where states on different sides of a norm boundary evolve differently.

The key insight: **the number of basins K is a property of the MODEL × TASK interaction, not just the model or just the task.** A model may have many basins for task A (diverse reasoning paths) and few basins for task B (strong default attractor).

### Relationship to CDE

In CDE terms:
- **rho (error correlation)** measures WHETHER candidates are in different basins (low rho = different basins)
- **K (trajectory classes)** measures HOW MANY basins exist
- **ALM** directly measures BOTH and additionally maps the GEOMETRY of the basin structure

CDE without ALM: knows that operators decorrelate (rho < 0.3) but not WHY or how to IMPROVE decorrelation.
CDE with ALM: knows the basin structure and can actively STEER candidates into different basins.

---

## 2. Complete System Design

### Component 1: Probe Generator

Generate a dense sample of trajectories to map the landscape.

```python
class ProbeGenerator:
    """
    Generate N trajectory probes by sampling the input space systematically.
    """
    def __init__(self, model, embed_dim, embed_rms):
        self.model = model
        self.embed_dim = embed_dim
        self.embed_rms = embed_rms
    
    def generate_probes(self, question_embeds, N=100, probe_length=32):
        """
        Generate N short probes (first 32 tokens only) with random prefixes.
        Returns: list of ProbeResult(seed, prefix_embeds, first_32_tokens, 
                                      first_32_embedding, logprobs, kv_cache)
        """
        probes = []
        for seed in range(N):
            torch.manual_seed(seed)
            prefix = torch.randn(1, 2, self.embed_dim)
            prefix = prefix * (self.embed_rms / prefix.norm(dim=-1, keepdim=True))
            
            full_input = torch.cat([prefix, question_embeds], dim=1)
            output = self.model.generate(
                inputs_embeds=full_input,
                max_new_tokens=probe_length,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
            
            probes.append(ProbeResult(
                seed=seed,
                prefix_embeds=prefix,
                token_ids=output.sequences[0, full_input.shape[1]:],
                logprobs=compute_logprobs(output.scores),
                # Optionally save kv_cache for later continuation
            ))
        
        return probes
```

**Design decisions:**
- **N=100 probes**: Enough to sample the landscape with reasonable density. 100 × 32 tokens = 3200 tokens of generation (cheap).
- **Probe length = 32 tokens**: Enough to distinguish trajectory classes in most cases. First 32 tokens typically establish the reasoning approach.
- **Greedy decoding**: Probes should be deterministic. We want to map the effect of PREFIX variation, not sampling variation.
- **RMS-matched prefixes**: Same scale as existing experiments. The probes sample the same input distribution our method uses.

### Component 2: Trajectory Embedder

Convert each probe's first-32-tokens into a fixed-size embedding for clustering.

```python
class TrajectoryEmbedder:
    """
    Multiple embedding strategies for trajectory classification.
    """
    def embed_token_sequence(self, token_ids):
        """Strategy 1: Average token embeddings (model's own embedding layer)."""
        embeds = self.model.embed_tokens(token_ids)  # (32, embed_dim)
        return embeds.mean(dim=0)  # (embed_dim,)
    
    def embed_token_hash(self, token_ids):
        """Strategy 2: Binary vector of token presence (bag-of-tokens)."""
        bag = torch.zeros(self.vocab_size)
        for tid in token_ids:
            bag[tid] = 1.0
        return bag
    
    def embed_logprob_trajectory(self, logprobs):
        """Strategy 3: Use the logprob sequence as a trajectory fingerprint."""
        # Logprobs capture the model's confidence trajectory
        # Pad/truncate to fixed length
        return torch.tensor(logprobs[:32])
    
    def embed_combined(self, token_ids, logprobs):
        """Strategy 4: Concatenate token and logprob embeddings."""
        token_embed = self.embed_token_sequence(token_ids)
        logprob_embed = self.embed_logprob_trajectory(logprobs)
        return torch.cat([token_embed, logprob_embed])
```

**Which embedding to use?**
This is an empirical question. Run all four on the 100 probes and measure:
- Which embedding produces the CLEAREST cluster separation (highest silhouette score)?
- Which embedding's clusters best predict final-answer correctness?
Choose the winner.

### Component 3: Basin Clusterer

Cluster probe embeddings into attractor basins.

```python
class BasinClusterer:
    """
    Identify attractor basins from trajectory embeddings.
    """
    def cluster(self, embeddings, method='hdbscan'):
        """
        Cluster embeddings into basins.
        
        HDBSCAN preferred over DBSCAN:
        - Doesn't require eps parameter
        - Handles varying cluster densities
        - Labels noise points explicitly
        
        Returns: BasinMap(cluster_labels, centroids, basin_sizes, noise_fraction)
        """
        if method == 'hdbscan':
            clusterer = HDBSCAN(min_cluster_size=5, min_samples=2)
            labels = clusterer.fit_predict(embeddings)
        elif method == 'spectral':
            # For comparison: spectral clustering with automatic K selection
            # Use silhouette score to choose K from {2, 3, 4, ..., 20}
            best_k, best_score = 2, -1
            for k in range(2, min(21, len(embeddings)//3)):
                labels_k = SpectralClustering(n_clusters=k).fit_predict(embeddings)
                score = silhouette_score(embeddings, labels_k)
                if score > best_score:
                    best_k, best_score = k, score
            labels = SpectralClustering(n_clusters=best_k).fit_predict(embeddings)
        
        # Compute basin properties
        basins = {}
        for basin_id in set(labels):
            if basin_id == -1:  # noise
                continue
            mask = labels == basin_id
            basins[basin_id] = BasinProperties(
                centroid=embeddings[mask].mean(dim=0),
                size=mask.sum(),
                radius=embeddings[mask].std(dim=0).norm(),
                members=[i for i, l in enumerate(labels) if l == basin_id],
            )
        
        return BasinMap(
            labels=labels,
            basins=basins,
            noise_fraction=(labels == -1).sum() / len(labels),
            n_basins=len(basins),
        )
```

**Critical parameter**: `min_cluster_size`. Too small (2-3) finds many spurious basins. Too large (20+) misses small basins. Default = 5 (5% of 100 probes = reasonable minimum basin size).

### Component 4: Basin Quality Assessor

For each basin: what's the probability that a candidate in this basin is correct?

```python
class BasinQualityAssessor:
    """
    Assess quality of each basin by generating full outputs from representative probes.
    """
    def assess(self, basin_map, probes, question, ground_truth, samples_per_basin=3):
        """
        For each basin, generate full outputs from 3 representative members
        and evaluate correctness.
        """
        basin_quality = {}
        for basin_id, basin in basin_map.basins.items():
            # Select representative probes (closest to centroid)
            reps = self.select_representatives(basin, probes, n=samples_per_basin)
            
            # Generate full outputs
            correct_count = 0
            for probe in reps:
                full_output = self.model.generate(
                    prefix=probe.prefix_embeds,
                    question=question,
                    max_new_tokens=1024,
                )
                if evaluate(full_output, ground_truth):
                    correct_count += 1
            
            basin_quality[basin_id] = BasinQuality(
                correct_rate=correct_count / samples_per_basin,
                avg_logprob=mean([p.logprobs.mean() for p in reps]),
                avg_length=mean([len(p.token_ids) for p in reps]),
                health_score=self.compute_health(reps),
            )
        
        return basin_quality
```

**Cost**: K basins × 3 samples × 1024 tokens = K × 3072 tokens of full generation. With K=5 basins: 15,360 tokens. Manageable but not free.

**Optimization**: Only assess quality for the TOP-K basins by size (larger basins are more likely to be real). Skip tiny basins (size < 3 probes) as they may be noise.

### Component 5: Basin Navigator

Use the basin map to steer candidate generation into specific basins.

```python
class BasinNavigator:
    """
    Generate candidates that target specific basins.
    """
    def target_basin(self, basin_map, target_basin_id, question_embeds, n_attempts=5):
        """
        Generate a candidate that lands in the target basin.
        
        Strategy: use the prefix closest to the target basin's centroid.
        """
        target = basin_map.basins[target_basin_id]
        
        # Find the probe whose embedding is closest to the target centroid
        # among probes NOT already in this basin (for diversity)
        closest_probe = min(
            [p for p in probes if basin_map.labels[p.seed] == target_basin_id],
            key=lambda p: distance(embedder.embed(p), target.centroid)
        )
        
        return closest_probe.prefix_embeds
    
    def diverse_basin_sampling(self, basin_map, basin_quality, N=10):
        """
        Allocate N candidates across basins proportional to basin quality × basin size.
        
        High-quality large basins get more candidates.
        Every basin with quality > 0 gets at least 1 candidate.
        """
        # Score each basin
        scores = {
            bid: bq.correct_rate * basin_map.basins[bid].size
            for bid, bq in basin_quality.items()
            if bq.correct_rate > 0  # skip basins with 0% correctness
        }
        
        if not scores:
            # No quality basins found — fall back to random
            return [random_prefix(seed=i) for i in range(N)]
        
        # Allocate proportionally
        total = sum(scores.values())
        allocation = {bid: max(1, round(N * score / total)) for bid, score in scores.items()}
        
        # Generate candidates
        candidates = []
        for basin_id, n_cands in allocation.items():
            basin_probes = [p for p in probes if basin_map.labels[p.seed] == basin_id]
            for i in range(n_cands):
                # Use different probes within the basin for slight within-basin diversity
                prefix = basin_probes[i % len(basin_probes)].prefix_embeds
                candidates.append(prefix)
        
        return candidates[:N]
```

### Component 6: Landscape Visualizer

For interpretability and paper figures:

```python
class LandscapeVisualizer:
    """
    2D/3D visualization of the attractor landscape.
    """
    def plot_landscape(self, embeddings, labels, quality_scores):
        """
        UMAP reduction to 2D, colored by basin, sized by quality.
        """
        coords_2d = UMAP(n_components=2).fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        for basin_id in set(labels):
            mask = labels == basin_id
            color = 'green' if quality_scores.get(basin_id, {}).get('correct_rate', 0) > 0.5 else 'red'
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                      c=color, alpha=0.6, label=f'Basin {basin_id}')
        
        ax.set_title('Attractor Landscape: Green=Quality, Red=Trap')
        ax.legend()
        return fig
```

---

## 3. Integration with CDE

### ALM as CDE Operator O14

ALM-guided prefix selection IS an operator in the CDE framework:
- **Input**: question, basin_map (pre-computed)
- **Output**: N candidates, each targeting a different quality basin
- **Properties**: high K (by design), low rho (by design), potentially high A (targeted at quality basins)
- **Cost**: N × 1.0x generation + one-time 100 × 32 tokens probe cost

### ALM as CDE Measurement Tool

Beyond being an operator, ALM provides MEASUREMENT for all other operators:
- Run probes using EACH operator → get operator-specific basin maps
- Compare: do different operators produce different basin structures?
- If O2 (prefix) produces 6 basins and O5 (temp 0.6) produces 4 basins with 60% overlap → they're partially complementary
- If they produce the SAME 6 basins → they're redundant despite superficially different mechanisms

### ALM as CDE Diversity Controller

Use ALM's basin map for Architecture 21 (DRCG) and Architecture 18 (DO-BoN):
- After generating each candidate, classify its basin using the pre-computed map
- If the candidate lands in a basin that already has K representatives → reject (duplicate suppression)
- This replaces the crude "first-32-token fingerprint" with a PRINCIPLED basin classification

---

## 4. Experimental Protocol

### Phase 1: Landscape Survey (Per-Task)

For each of 25 arithmetic tasks:
1. Generate 100 probes (32 tokens each) with random prefixes
2. Embed using all 4 embedding strategies
3. Cluster with HDBSCAN
4. Record: K basins, basin sizes, noise fraction
5. Visualize with UMAP

**Output**: A table showing K basins per task. This directly answers: how many distinct reasoning paths does the model have?

### Phase 2: Basin Quality Assessment

For each task, for each basin:
1. Generate 3 full outputs from representative probes
2. Evaluate correctness
3. Record: basin quality (correct_rate)
4. Label basins: "correct basin" (rate > 0.5), "trap basin" (rate = 0), "uncertain" (0 < rate < 0.5)

**Output**: Quality-annotated basin map. Shows which parts of the landscape contain correct reasoning.

### Phase 3: Basin Navigation Test

For each task:
1. ALM-guided: generate 10 candidates targeting quality basins (diverse_basin_sampling)
2. Random: generate 10 candidates with random prefixes (current method)
3. Compare: oracle accuracy, individual accuracy, trajectory diversity

**Pre-registered success criterion**: ALM-guided oracle ≥ random oracle + 2 tasks AND ALM-guided mean accuracy ≥ random mean accuracy.

### Phase 4: Cross-Operator Basin Comparison

For each task:
1. Generate 100 probes with random prefix (operator O2)
2. Generate 100 probes with temperature 0.6 (operator O5)
3. Generate 100 probes with prompt rephrase (operator O8)
4. Cluster each operator's probes independently
5. Pool all 300 probes and cluster together

**Output**: Do different operators access different basins? This is the STRONGEST possible evidence for CDE's thesis.

---

## 5. Failure Modes and Pre-Mortems

### Failure 1: No Basin Structure
**What**: HDBSCAN finds 1 giant cluster (or 100 singletons). No discrete basin structure.
**Probability**: Medium. If the model's generation is smooth in prefix space (small prefix changes → small output changes), there are no discrete basins.
**Impact**: ALM fails. CDE loses its measurement infrastructure for trajectory diversity.
**Mitigation**: Try different embedding strategies. If token-level embedding shows no structure, try hidden-state embedding (layer 16 activations at token 32). Basin structure may exist in internal representation even if not in output tokens.
**If still no structure**: This is a FINDING. Report: "The model's generation landscape for arithmetic is smooth, not basin-structured. Prefix perturbation provides continuous diversity, not discrete trajectory switching."

### Failure 2: Basins Are Task-Specific
**What**: Task A has 5 basins, Task B has 2 basins, and the basin structures share nothing in common.
**Probability**: High. Different arithmetic operations likely have different attractor structures.
**Impact**: ALM must be run per-task. The one-time offline cost becomes per-query.
**Mitigation**: Look for task-TYPE patterns. Do all multiplication tasks share similar basin structure? If yes, compute one basin map per operation type, not per specific problem.

### Failure 3: Quality Basins Are Small
**What**: The "correct-answer basin" exists but contains only 2-3% of probes.
**Probability**: Consistent with existing data (32% baseline accuracy → roughly 32% of seeds land in a correct basin).
**Impact**: Random probing has a 32% chance of hitting the correct basin per probe. ALM can GUARANTEE hitting it with basin navigation. This is exactly why ALM is valuable.
**Not a failure**: This is the EXPECTED case. ALM's value comes from navigating TO the small quality basins that random search often misses.

### Failure 4: Basin Navigation Doesn't Work
**What**: Using a probe's prefix that maps to the "correct basin" doesn't produce a correct full output.
**Probability**: Medium. The probe's first 32 tokens may be in the correct basin, but the full 1024-token generation may diverge.
**Impact**: ALM is diagnostic (shows the landscape) but not actionable (can't steer to quality basins).
**Mitigation**: Use the probe's prefix directly (not an approximation). If the probe at seed=47 was in the correct basin, use seed=47's exact prefix for full generation.

### Failure 5: 100 Probes Is Insufficient
**What**: The landscape has many small basins that 100 probes miss.
**Probability**: Depends on basin count. If K=20, 100 probes averages 5 per basin. Some basins may have 0 probes.
**Impact**: Missing basins = missing potentially correct trajectories.
**Mitigation**: Increase to N=200 or N=500 probes. Each probe costs only 32 tokens of generation. 500 × 32 = 16,000 tokens ≈ 16 full generations. Still cheap.

---

## 6. Cost Analysis

### Per-Task ALM Cost

| Phase | Generations | Tokens | Time (est.) |
|---|---|---|---|
| Probe | 100 × 32 tokens | 3,200 | ~30 sec |
| Embedding | Compute from probes | 0 | ~2 sec |
| Clustering | HDBSCAN on 100 points | 0 | <1 sec |
| Quality assessment | K basins × 3 × 1024 tokens | ~15,000 | ~90 sec |
| Navigation (10 candidates) | 10 × 1024 tokens | 10,240 | ~60 sec |
| **Total per task** | | ~28,000 | **~3 min** |

### Full Calibration Set (25 Tasks)

| Phase | Total Time |
|---|---|
| Probe all tasks | ~12 min |
| Cluster all tasks | <1 min |
| Quality assess all tasks | ~37 min |
| Navigation test all tasks | ~25 min |
| **Total** | **~75 min** |

### Comparison with Random Best-of-N (N=10)

Random best-of-N: 25 tasks × 10 × 1024 tokens = 256,000 tokens ≈ 25 min
ALM-guided best-of-N: ~75 min (3x more expensive)

**BUT**: ALM provides diagnostic information (basin count, basin quality, landscape structure) that random best-of-N doesn't. The 3x overhead pays for UNDERSTANDING, not just PERFORMANCE.

**After one-time mapping**: ALM-guided generation per task costs the same as random (10 × 1024 tokens). The overhead is entirely in the initial mapping.

---

## 7. Paper Contribution

If ALM works, it provides THREE paper contributions:

### Contribution 1: The Attractor Landscape Map
First-ever visualization of a small LLM's reasoning landscape for specific tasks. Shows how many distinct reasoning paths exist, which contain correct answers, and how they relate to each other. This is interpretability + intervention in one result.

### Contribution 2: Quantifying Prefix Perturbation's Mechanism
The basin map directly answers: "What does prefix perturbation do?" Answer: "It moves the system to a different attractor basin. The model has K basins for this task; random prefixes sample uniformly; our method's success rate is proportional to the fraction of basins containing correct trajectories."

### Contribution 3: Principled Diversity Optimization
ALM-guided generation replaces random exploration with targeted basin sampling. The improvement (if any) demonstrates that trajectory diversity is not random — it has structure that can be exploited.

---

## 8. Relationship to Other Top-5 Architectures

### ALM + CDE = Natural Integration
ALM is CDE's diversity measurement engine. Without ALM, CDE measures error correlation (post-hoc). With ALM, CDE predicts and controls diversity (a priori).

### ALM + DDC (#9)
Decompose-Dispatch-Compose creates easier sub-problems. ALM maps the landscape of each sub-problem. If sub-problems have simpler landscapes (fewer basins, larger quality basins), DDC + ALM is the optimal combination.

### ALM + Neuro-Symbolic (#14)
For arithmetic: neuro-symbolic provides exact verification. ALM provides diverse formalization candidates. Combined: map the landscape of Python code generation, navigate to basins that produce correct formalizations.

### ALM + CIR (#11)
CIR identifies WHICH features cause basin transitions. ALM maps WHERE basins are. Together: CIR explains the landscape; ALM exploits it.
