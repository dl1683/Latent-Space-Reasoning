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

### 2. Fix Evolutionary Fitness Function
- [x] Replace dense_score with actual task accuracy as fitness signal
- [ ] Re-run V15 geometry isolation with accuracy-based fitness (V15b RUNNING)
- [ ] Determine if evolution can actually improve output quality (not just Goodhart)
- **Current status:** Accuracy fitness implemented. V15b running. Sensitivity analysis proves landscape IS exploitable (32% range, p=0.006).

### 3. Determine If Hyperbolic Geometry Matters
- [x] V5-V8: Early mixed signals (fragile, high variance)
- [x] V15a: No geometry effect under identical soft prompt (both 60%) -- fitness was broken
- [ ] V15b: Re-test geometry with accuracy-based fitness (RUNNING NOW)
- **Current status:** Blocked on V15b results. Fitness fix should unblock the geometry comparison.

### 4. Novel Algorithmic Breakthroughs
- [x] CMA-ES in Poincare ball (implemented, tested)
- [x] Mixture-of-curvature evolution (implemented, tested)
- [x] Active Inference surrogate (implemented, V17 runner ready)
- [x] Quality-Diversity archive with DNS (implemented, V18 runner ready)
- [ ] Run V17 diagnostic: Active Inference vs standard evolution
- [ ] Run V18 diagnostic: QD archive vs elitist selection
- [ ] V19: Physarum-bioelectric hybrid search (designed, deferred)
- **Current status:** All infrastructure built. V17/V18 runners ready. Awaiting V15b baseline.

## Completed Goals
- [x] AIM-v1 accessibility milestone (efficiency + quality tradeoffs validated)
- [x] Autonomy scaffolding (AGENTS.md, MEMORY.md, GOALS.md, TASKS.md, WORKLOG.md)
- [x] Unified experiment harness (harness.py, decode subpackage)
- [x] Cross-model conditioning comparison (4 models, 20 questions, LLM-as-judge)
- [x] Experiment documentation (EXPERIMENTS.md, ledger.jsonl)

## Key Research Findings (Validated)
1. **Conditioning bandwidth matters** — soft prompt (20,480 continuous values) vs RNG seed (31-bit integer) vs nothing
2. **Both conditioning methods eliminate phantom hallucination** — robust across all model sizes
3. **Non-monotonic scaling** — soft prompt isn't universally better; RNG seed wins at 0.6B and 8B
4. **Evolution with dense_score hurts** — Goodhart's Law; need accuracy-based fitness
5. **Landscape IS exploitable** — 32% accuracy range across random latents (Cochran's Q=23.2, p=0.006)
6. **Novel research combination** — QD + Poincare + Active Inference + soft prompt (validated by 2025-2026 lit review, no existing work combines all five)
7. **All components independently validated** — MAP-Elites for prompts (CEC 2025), HypLoRA (NeurIPS 2025), Coconut (COLM 2025), LLM-SAEA (2025)
