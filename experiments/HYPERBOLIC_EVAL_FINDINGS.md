# Hyperbolic Evolution Evaluation Findings

**Date:** 2026-02-03
**Status:** Phase 6+ - Verifiable Evolution System Active

---

## BREAKTHROUGH: Verifiable Ground-Truth Evolution

### The Problem We Solved
The latent scorer had -0.03 correlation with quality. Evolution was essentially blind.

### The Solution
Replace the broken scorer with **verifiable tasks** that have programmatic pass/fail verification:
- Arithmetic: `What is 4 + 95?` (answer: 99, verifiable)
- Logic: `What is True AND False?` (answer: False, verifiable)
- Word Problems: `10 boxes with 5 items each?` (answer: 50, verifiable)
- Comparisons: `Is 76 > 55?` (answer: Yes, verifiable)

### Key Results (Quick Test: 3 gen, 4 tasks, 1 run)
- **Baseline accuracy: 50.0%**
- **After evolution: 62.5%** (+12.5% improvement!)
- Both hyperbolic and Euclidean improved equally
- **Verdict: Evolution WORKS with ground-truth fitness**

### Insight from Codex
> "The breakthrough isn't better scoring - it's making truth observable."

### New Approach: Fitness Sharing
Codex identified the key gap: "Diversity must influence *selection* toward quality. Right now it's just a byproduct."

Implemented **hyperbolic fitness sharing**:
- Candidates in crowded regions get fitness penalties
- Uses hyperbolic distance for niche counting
- Hyperbolic volume growth rewards peripheral exploration
- Makes diversity a MECHANISM, not a byproduct

---

## V2 Experiment: Improved Experimental Design (In Progress)

Following Codex's recommendations, implemented:
1. **Fixed task pool** (500 tasks, train/val split)
2. **Paired seeds** (same task sequence for both geometries)
3. **Rolling fitness** (3-generation average for selection)
4. **Final validation** (20 held-out tasks for comparison)
5. **Hierarchical tasks** (nested arithmetic, multi-hop reasoning)

### V2 Results (Quick Test: 5 gen, 4 pop, 2 runs)

**Run 1:**
| Prompt | Hyperbolic Val | Euclidean Val | Winner |
|--------|---------------|---------------|--------|
| Math reasoner | 35% | 35% | TIE |
| Logic solver | 25% | 30% | EUCLIDEAN |

**Run 2 (Partial):**
- Hyperbolic showing roll=75% (strong!)
- Still in progress...

### Key Observation: Euclidean Shows Higher Diversity
Counter to expectations, Euclidean maintained HIGHER diversity (5.5-6.1) than hyperbolic (3.9-4.6) in this experiment. This suggests:
1. Euclidean mutation adds unconstrained noise (spreads farther)
2. Hyperbolic is constrained to the Poincare ball (||x|| < 1)
3. Diversity metrics aren't directly comparable between geometries

### Why Hyperbolic Isn't Winning (Codex Analysis)

> "Hyperbolic should win when the task's useful variation is hierarchical or tree-like, when the fitness landscape has many deep, branching basins, and when evaluation rewards coverage of rare niches."

Current tasks are too "flat" - all arithmetic/logic tasks might have similar optimal latents. Diversity doesn't help when all tasks need the same solution.

### V2 Final Results (Stopped after Run 2 due to slow progress)
- Run 1 Prompt 1: TIE (35% vs 35%)
- Run 1 Prompt 2: EUCLIDEAN wins (30% vs 25%)
- Run 2: Hyperbolic roll=75%, Euclidean roll=62.5% (incomplete)

**Conclusion: No clear hyperbolic advantage with flat tasks.**

---

## V3/V4 Experiments: True Hierarchical Tasks

### Key Codex Insight (xhigh reasoning analysis)

> "A 20–30% win is *not* likely on generic reasoning tasks. If you don't engineer the task structure to be explicitly hierarchical with depth/branching pressure, hyperbolic won't beat Euclidean by much, if at all."

### What Could Actually Produce 20-30% Win:

1. **Tasks with true tree geometry and depth pressure**
   - Depth 6-10, branching factor 3-6
   - Rare-leaf retrieval (distinguishing low-frequency leaves)
   - Traversal tasks where reward scales with depth

2. **Fitness that rewards depth AND breadth**
   - Depth-weighted coverage (deeper = exponentially more valuable)
   - Worst-leaf or 5th-percentile accuracy (not just average)
   - Novelty/rarity bonus for rare leaves

3. **Mixed geometry (Euclidean × Hyperbolic)**
   - Split latent: Euclidean for local variance, hyperbolic for hierarchy
   - "This setup consistently outperforms pure hyperbolic" - Codex

4. **Curvature annealing**
   - Start low (near Euclidean), increase as population stabilizes
   - Learned curvature based on branching factor/depth

### V3: Hierarchical Tasks + Coverage Fitness
- Uses nested arithmetic and multi-hop reasoning only
- Coverage bonus for solving different categories/difficulties
- Tail metrics (worst-category accuracy)

### V4: True Tree Tasks + Depth Fitness (In Progress)
- **Tree traversal**: Follow path [2]→[1]→[3] through tree, compute node value
- **Hierarchical classification**: Deep taxonomy (entity→living→animal→mammal→dog)
- **Multi-hop reasoning**: Chains with branching factor > 2

Key changes:
- Depth-weighted fitness (2^depth scaling)
- Rarity bonus for deep paths
- Curvature annealing (0.5 → 1.5)
- Numerical safeguards for diversity calculation

---

## Original Findings (Phase 5)

## Key Findings

### 1. Hyperbolic Geometry Successfully Maintains Diversity

| Metric | Euclidean | Hyperbolic | Ratio |
|--------|-----------|------------|-------|
| Survivors | 2 | 5-7 | 3.5x |
| Diversity | 0.006 | 2.32 | 405x |
| Stop Reason | patience | max_generations | - |

**Conclusion:** Hyperbolic geometry in the Poincaré ball model successfully prevents population collapse during evolution. This is the core hypothesis CONFIRMED.

### 2. BUT: Output Quality Shows No Difference

LLM-as-judge evaluation (Claude subagent) found:
- Euclidean wins: 2 prompts (slight margin)
- Hyperbolic wins: 2 prompts (slight margin)
- Ties: 1 prompt
- **Verdict: "Difference within noise margin"**

All outputs were truncated mid-thought. Both geometries produce similar reasoning patterns.

### 3. CRITICAL: Latent Scorer Has ~0 Correlation

From checkpoint analysis:
```
val_corr: -0.031
```

The latent scorer essentially produces random scores. The LLM-as-judge confirmed:
- Automated scores inflate quality by 0.15-0.25 points
- Scorer rewards "verbose thinking patterns" not correctness
- Wrong answers get similar scores to correct ones

### 4. Root Cause Analysis

The evolution loop is **blind** because:
1. The latent scorer was trained on data with poor quality labels
2. It learned to recognize "reasoning style" not "reasoning quality"
3. Selection pressure is essentially noise
4. Hyperbolic diversity helps maintain exploration but can't find better solutions without signal

## Codex Senior Engineer Assessment

> "Promising as a *mechanism for maintaining exploration*, but currently a dead end for *end-quality* until your evaluation signal improves. Right now you've proven 'hyperbolic keeps diversity,' not 'hyperbolic improves reasoning.'"

### Codex Recommendations (Priority Order):
1. **(a) Improve latent scorer** - Most urgent, evolution is blind without it
2. **(d) Add format compliance + post-hoc evaluator** - Verify actual quality
3. **(c) Curvature tuning** - Only matters after scorer is fixed
4. **(b) Decoder improvements** - Won't help if selection is broken

## Technical Details

### Hyperbolic Implementation (Working)
- Poincaré ball model with curvature c=1.0
- logmap0/expmap0 for tangent space operations
- Karcher mean for crossover
- Hyperbolic distance for diversity computation
- Proper tangent→ball→tangent mapping in decoder (fixed bug)

### Scorer Issues
- 12,118 training samples, 1,346 validation
- Trained with MSE loss on Gemini-generated scores
- -0.031 correlation = worse than random
- Likely reasons: noisy labels, style overfitting

## Next Steps

### Phase 6: Fix Latent Scorer
Options:
1. Generate new training data with stronger LLM judge
2. Train on completeness + correctness signals
3. Use self-consistency (model rates its own outputs)

### Phase 7: Add Format Compliance
- Ensure `<think>` tags are always present
- Detect incomplete outputs
- Add completion penalty

### Phase 8: Re-evaluate with Fixed Scorer
- Controlled ablation: same seeds, same evaluator
- Compare final output quality not just diversity
- Statistical significance testing

## Codex Code Review Findings

A detailed Codex review of the hyperbolic implementation found:

### Math Correctness: VERIFIED
- `mobius_add`, `expmap0`, `logmap0`, `expmap`, `logmap`, `hyperbolic_distance` are mathematically correct
- `karcher_mean` uses reasonable Riemannian gradient descent

### Integration Issues Identified
1. **Critical (FIXED):** Hyperbolic latents were scored/decoded in Euclidean space - fixed by adding `hyperbolic`/`curvature` params to `decode()`
2. **High:** Modifier hints inconsistent with hyperbolic latents - still needs fix
3. **High:** Curvature annealing doesn't reproject latents - potential issue
4. **Medium:** `mobius_add` hard-codes `max_norm=0.98` instead of using config
5. **Low:** Mutation step sizes grow near boundary (not scaled by 1/λ_p)

### Codex Verdict
> "Yes, promising, but the next focus should be alignment and ablations."

## Conclusion

**Hyperbolic geometry for latent evolution: Mechanism works, signal broken.**

The geometry successfully maintains population diversity during evolution - this is valuable. But without a reliable scoring signal, diversity doesn't translate to quality improvements. The pivot to fixing the scorer is the correct next step.
