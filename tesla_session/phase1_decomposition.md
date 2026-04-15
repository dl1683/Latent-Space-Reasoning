# Tesla Mode Phase 1: Decomposition — Next Generation of Latent Space Reasoning

## The Problem

We pioneered embedding-space perturbation for inference-time reasoning improvement in small LLMs. Others are now adopting this paradigm. We need to define and build the next generation. The question: **What mechanisms remain unexploited? What architectural evolution unlocks capabilities the current system cannot reach? How do we stay ahead?**

## Current System Summary

- Prepending 2 random embedding-scale tokens to an LLM shifts greedy decoding trajectory
- +19.6pp arithmetic (Qwen3-4B), 92% legal oracle wins, attention sink rescue in planning
- Direction-agnostic: random noise = optimized projections (p=1.0)
- Cross-domain validated: arithmetic, planning, legal reasoning
- Judge-heavy: system ceiling determined by scorer quality
- Evolution barely works (mean wins: Base 6, Pert 5, Evo 1 in legal) — scorer is barely trained

## Component Decomposition

### Component 1: The Perturbation Mechanism
- **What**: 2 random embedding-scale tokens prepended to shift greedy trajectory
- **Depends on**: Attention sink positions, model embedding geometry, greedy decoding, latent capabilities model can't access by default
- **Downstream**: Everything — entire improvement gated by this
- **Interactions**: Quantization level, model capacity ceiling, token budget, task difficulty
- **Failure mode**: Direction-agnostic = no targeting. Mean improvement can be zero (DeepSeek) even when oracle is 100%. The mechanism is a *lottery* — diversifies trajectories but can't steer without a judge

### Component 2: The Judge/Scorer
- **What**: Scores candidate latents to guide evolution or oracle selection
- **Depends on**: Training data quality, architecture, what "quality" means in latent space
- **Downstream**: Evolution effectiveness. THE bottleneck
- **Interactions**: Scorer operates in latent space but quality defined in output space — fundamental proxy gap
- **Failure mode**: Scores style not substance. -inf bugs. Non-deterministic projections. Doesn't understand task semantics

### Component 3: The Evolution Loop
- **What**: Population-based search (selection + mutation + crossover) in latent space
- **Depends on**: Judge quality (fundamentally), mutation operators, population size
- **Downstream**: Whether we can systematically find good perturbations vs. random lottery
- **Interactions**: Latent space geometry — smooth? convex? local optima?
- **Failure mode**: Bad scorer → evolution wanders randomly. Current state: barely better than random perturbation

### Component 4: The Projection (Latent → Soft Prompt)
- **What**: Fixed row-orthonormal matrix projecting 1024d latent → 8×2560 soft prompt embeddings
- **Depends on**: Latent/embedding dimensionality, RMS scaling
- **Downstream**: Whether latent space structure maps meaningfully to embedding space
- **Interactions**: RMS scaling must match real token scale (~0.022)
- **Failure mode**: Untrained — preserves structure but doesn't learn useful mappings. Random W = structured W (p=1.0)

### Component 5: The Generation Model
- **What**: The LLM being perturbed (Qwen3-4B, 8B, etc.)
- **Depends on**: Weights, quantization, vocabulary, architecture
- **Downstream**: Everything — model's latent capability space IS the ceiling
- **Interactions**: Quantization modulates whether noise can exploit trajectory landscape
- **Failure mode**: Model doesn't have latent capability → perturbation can't help

### Component 6: The Oracle Selection
- **What**: Best-of-N selection using external judge (LLM-as-judge, human)
- **Depends on**: N diverse candidates, reliable external judge
- **Downstream**: "Real" performance ceiling
- **Interactions**: N candidates × generation cost = total cost
- **Failure mode**: Low candidate diversity → oracle saturates. Biased judge → biased oracle

## Key Tensions Identified

1. **Lottery vs. Targeting**: The mechanism is powerful but undirected. We diversify but can't aim.
2. **Latent Space vs. Output Space**: Scorer operates where optimization is cheap (latent) but quality lives where evaluation is expensive (output). Fundamental proxy gap.
3. **Direction Independence Paradox**: Random noise works as well as optimized directions. This is robust but means the projection/evolution machinery adds no value over simple random sampling.
4. **Judge Bottleneck**: Everything depends on judge quality, but the current judge is barely trained and operates on the wrong signal.
5. **Model Ceiling**: The mechanism can only unlock what the model already knows. It cannot add knowledge.

## Open Questions for Next-Gen Design

1. Can we make perturbation *targeted* without losing the robustness of direction-agnostic approach?
2. What would a judge look like that operates in output space but at latent-space cost?
3. Is there a way to learn the perturbation landscape so evolution actually converges?
4. Can we perturb at layers beyond input embeddings (intermediate layers, attention patterns)?
5. Can we combine this with other inference-time methods (sampling, self-consistency, verifiers)?
6. What happens when we scale to larger models? Does the effect vanish, transform, or amplify?
7. Can we make the system self-improving — where each generation's outputs train the next generation's scorer?
