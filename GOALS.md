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
- [ ] Replace dense_score with actual task accuracy as fitness signal
- [ ] Re-run V15 geometry isolation with accuracy-based fitness
- [ ] Determine if evolution can actually improve output quality (not just Goodhart)
- **Current status:** Blocked. V15 showed evolution hurts with current fitness (90% -> 60%).

### 3. Determine If Hyperbolic Geometry Matters
- [x] V5-V8: Early mixed signals (fragile, high variance)
- [x] V15: No geometry effect under identical soft prompt conditioning (both 60%)
- [ ] Re-test geometry after fixing fitness function
- **Current status:** Inconclusive. Codex says "conditioning bandwidth matters, not geometry."

### 4. Long-Horizon: Novel Algorithmic Breakthroughs
- [ ] CMA-ES in Poincare ball (implemented, untested at scale)
- [ ] Mixture-of-curvature evolution (implemented, untested)
- [ ] Activation injection / steering vectors (designed, not implemented)
- **Current status:** Infrastructure built, waiting on fitness fix before meaningful experiments.

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
5. **Geometry effect is entangled with conditioning** — can't isolate geometry until fitness works
