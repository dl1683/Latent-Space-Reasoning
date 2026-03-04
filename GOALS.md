# Goals

## Mission
- Liberate high-quality intelligence for everyone through low-cost, low-resource, transparent, and auditable systems.
- Prefer large efficiency gains (10x-100x lower cost) even with modest quality tradeoffs versus maximum-benchmark systems.

## Active Goals

### 1. CRITICAL: Resolve Warm-Start vs Direction Confound
- [ ] Run random-noise control: 8 tokens of torch.randn at target_rms (NOT via W projection) on same 25 sweet-spot tasks
- [ ] Compare 3 conditions: baseline (no tokens), random noise, latent-projected
- [ ] If noise matches latent-projected: direction doesn't matter (warm-start only) -> pivot
- [ ] If noise matches baseline: direction carries signal -> scale to 50 latents
- **Current status:** All 10 latents beat baseline on sweet-spot (+12.4% mean), but Cochran's Q not significant (p=0.504). This is consistent with warm-start hypothesis. Random-noise control is the single most important experiment.

### 2. Validate Soft Prompt Conditioning as a Real Improvement
- [x] Build conditioning comparison framework (20 questions, 3 conditions, LLM-as-judge)
- [x] Run on multiple model sizes (0.6B, 4B, 8B, 14B)
- [x] Confirm conditioning eliminates phantom hallucination at all sizes
- [ ] Expand to non-Qwen model families for generalizability
- **Current status:** Both conditioning methods beat pure model. Non-monotonic scaling discovered.

### 3. Find a Search Algorithm That Exploits the Latent Landscape (BLOCKED on Goal 1)
- [x] Replace dense_score with actual task accuracy as fitness signal
- [x] V15b: Accuracy-based fitness geometry isolation (DONE -- evolution still hurts)
- [x] Prove latents affect accuracy differently (Cochran's Q=23.2, p=0.006 on easy_nested)
- [ ] Random-noise control to prove direction matters (Goal 1)
- [ ] If direction matters: 50-latent random search, CMA-ES, QD
- [ ] If warm-start only: pivot to studying warm-start mechanism
- **Current status:** BLOCKED until warm-start confound is resolved.

### 4. Determine If Hyperbolic Geometry Matters — CONCLUDED
- [x] V5-V8: Early mixed signals (fragile, high variance)
- [x] V15a: No geometry effect under identical soft prompt (both 60%) -- fitness was broken
- [x] V15b: No geometry effect with accuracy fitness (both 68%) -- geometry doesn't matter
- **CONCLUDED:** Hyperbolic geometry provides no benefit over Euclidean for soft prompt mutation.

## Completed Goals
- [x] AIM-v1 accessibility milestone (efficiency + quality tradeoffs validated)
- [x] Autonomy scaffolding (AGENTS.md, MEMORY.md, GOALS.md, TASKS.md, WORKLOG.md)
- [x] Unified experiment harness (harness.py, decode subpackage)
- [x] Cross-model conditioning comparison (4 models, 20 questions, LLM-as-judge)
- [x] Experiment documentation (EXPERIMENTS.md, ledger.jsonl)
- [x] V15b: geometry isolation with accuracy fitness (concluded: geometry doesn't matter)

## Key Research Findings (Codex-Validated)
1. **Conditioning bandwidth matters** — soft prompt (20,480 continuous values) vs RNG seed (31-bit integer) vs nothing
2. **Both conditioning methods eliminate phantom hallucination** — robust across all model sizes
3. **Non-monotonic scaling** — soft prompt isn't universally better; RNG seed wins at 0.6B and 8B
4. **Latents DO differ from each other** — Cochran's Q=23.2, p=0.006 on easy_nested tasks
5. **BUT conditioning mostly hurts on easy tasks** — mean 85.6% vs 92% baseline; only 1/10 beats baseline
6. **Sweet-spot: all latents beat baseline** — +12.4% mean improvement, 3/10 individually significant
7. **UNRESOLVED: warm-start vs direction** — sweet-spot Q not significant (p=0.504), all latents improve similarly
8. **Local evolution FAILS** — small mutations (noise=0.1) can't find improvements
9. **Hyperbolic geometry doesn't matter** — identical to Euclidean under same conditioning
10. **Chain-of-thought IS the steering mechanism** — no-think landscape flat (4% range vs 32% with thinking)
11. **Cochran's Q bug found and fixed** — original code had swapped matrix axes, producing null Q values
