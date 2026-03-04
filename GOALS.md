# Goals

## Mission
- Liberate high-quality intelligence for everyone through low-cost, low-resource, transparent, and auditable systems.
- Prefer large efficiency gains (10x-100x lower cost) even with modest quality tradeoffs versus maximum-benchmark systems.

## Active Goals

### 1. Validate Soft Prompt Conditioning as a Real Improvement
- [x] Build conditioning comparison framework (20 questions, 3 conditions, LLM-as-judge)
- [x] Run on multiple model sizes (0.6B, 4B, 8B, 14B)
- [x] Confirm conditioning eliminates phantom hallucination at all sizes
- [ ] Expand to non-Qwen model families for generalizability
- **Current status:** Both conditioning methods beat pure model. Non-monotonic scaling discovered.

### 2. Find a Search Algorithm That Exploits the Latent Landscape
- [x] Replace dense_score with actual task accuracy as fitness signal
- [x] V15b: Accuracy-based fitness geometry isolation (DONE -- evolution still hurts)
- [x] Prove landscape IS exploitable (32% range, p=0.006, sensitivity analysis)
- [ ] Random search baseline: sample 20 latents, pick best (RUNNING)
- [ ] Large-noise evolution (noise_scale=1.0+ to explore globally)
- [ ] CMA-ES: learned covariance for adaptive jumps
- [ ] QD with large noise: diversity pressure + global exploration
- **Current status:** Local evolution can't exploit the 32% global range. The problem is search radius (noise=0.1 in 2560d), not geometry or fitness. Need global search.

### 3. Determine If Hyperbolic Geometry Matters
- [x] V5-V8: Early mixed signals (fragile, high variance)
- [x] V15a: No geometry effect under identical soft prompt (both 60%) -- fitness was broken
- [x] V15b: No geometry effect with accuracy fitness (both 68%) -- geometry doesn't matter
- **CONCLUDED:** Hyperbolic geometry provides no benefit over Euclidean for soft prompt mutation. Both produce identical results. The conditioning CHANNEL matters (finding 1), not the mutation geometry.

### 4. Novel Algorithmic Breakthroughs
- [x] CMA-ES in Poincare ball (implemented, tested)
- [x] Mixture-of-curvature evolution (implemented, tested)
- [x] Active Inference surrogate (implemented, V17 runner ready)
- [x] Quality-Diversity archive with DNS (implemented, V18 runner ready)
- [ ] Test with global search (large noise or CMA-ES) before running V17/V18
- [ ] V19: Physarum-bioelectric hybrid search (designed, deferred)
- **Current status:** V17/V18 use local mutations and will likely fail for the same reason as V15b. Must fix search radius first.

## Completed Goals
- [x] AIM-v1 accessibility milestone (efficiency + quality tradeoffs validated)
- [x] Autonomy scaffolding (AGENTS.md, MEMORY.md, GOALS.md, TASKS.md, WORKLOG.md)
- [x] Unified experiment harness (harness.py, decode subpackage)
- [x] Cross-model conditioning comparison (4 models, 20 questions, LLM-as-judge)
- [x] Experiment documentation (EXPERIMENTS.md, ledger.jsonl)
- [x] V15b: geometry isolation with accuracy fitness (concluded: geometry doesn't matter)

## Key Research Findings (Validated)
1. **Conditioning bandwidth matters** — soft prompt (20,480 continuous values) vs RNG seed (31-bit integer) vs nothing
2. **Both conditioning methods eliminate phantom hallucination** — robust across all model sizes
3. **Non-monotonic scaling** — soft prompt isn't universally better; RNG seed wins at 0.6B and 8B
4. **Landscape IS exploitable GLOBALLY** — 32% accuracy range across random latents (p=0.006)
5. **Local evolution FAILS** — small mutations (noise=0.1) can't find improvements; good latents are far apart
6. **Hyperbolic geometry doesn't matter** — identical to Euclidean under same conditioning (V15a and V15b)
7. **Novel research combination** — QD + Poincare + Active Inference + soft prompt (validated by 2025-2026 lit review)
8. **All components independently validated** — MAP-Elites for prompts (CEC 2025), HypLoRA (NeurIPS 2025), Coconut (COLM 2025), LLM-SAEA (2025)
9. **Chain-of-thought contributes ~16% accuracy** — no-think baseline 72% vs thinking 88% on easy_nested
