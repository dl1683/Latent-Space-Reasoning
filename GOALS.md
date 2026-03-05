# Goals

## Mission
- Liberate high-quality intelligence for everyone through low-cost, low-resource, transparent, and auditable systems.
- Prefer large efficiency gains (10x-100x lower cost) even with modest quality tradeoffs versus maximum-benchmark systems.

## Active Goals

### 1. Characterize the Warm-Start Mechanism (IN PROGRESS)
**Core finding:** Prepending random embedding-scale tokens improves Qwen3-4B arithmetic by up to +28pp (32% -> 60% at 2 tokens). Direction doesn't matter. The dose-response is non-monotonic. This is a TRAJECTORY PERTURBATION effect.

**Completed:**
- [x] Mean-embedding control: 36% = zero → token diversity doesn't help for identical tokens
- [x] Zero-embedding control: 36% (+4pp) → embedding values matter, not just sequence extension
- [x] 1-token dose-response: 42.7% (+10.7pp)
- [x] 2-token dose-response: 60.0% (+28pp) → NON-MONOTONIC PEAK, zero variance across 3 latents
- [x] Nested-easy noise control: noise = latent-projected (p=1.0), confirmed cross-difficulty
- [x] Error taxonomy: redistribution, not clean improvement (3 fixed, 6 regressed)
- [x] Qualitative output analysis: policy shift from formal to exploratory reasoning

**In progress / next:**
- [ ] Repeated-noise (1 vector x 8): within-prefix diversity test
- [ ] Attention masking (--mask-prefix): attention sink vs trajectory perturbation
- [ ] Suffix position: does position matter?
- [ ] max_new_tokens sweep: token budget mediator test
- [ ] Scale to n=100+ for statistical power

### 2. Test Warm-Start Generality (DEFERRED until mechanism characterized)
- [ ] Non-Qwen models: Llama-3.2-3B, Phi-3-mini, Gemma-2-2B
- [ ] Non-arithmetic tasks: GSM8K subset, logic puzzles
- [ ] Larger models: does the effect vanish at scale?

### 3. Paper: "Prefix Perturbation as Policy Switch in Small Language Models"
- [ ] Complete mechanism characterization (Goal 1)
- [ ] Frame as redistribution/policy change, not pure improvement
- [ ] Complete generality tests (Goal 2)
- [ ] Address: n=25 fragility, token budget mediation, mechanism evidence

## Concluded / Falsified Goals

### Latent Space Search — FALSIFIED (2026-03-04)
- Random noise = latent-projected (p=1.0). Direction carries no signal.
- V17, V18, V19: DEAD CODE

### Hyperbolic Geometry — CONCLUDED (2026-03-03)
- Euclidean = Hyperbolic under same conditioning. Geometry adds no value.

## Completed Goals
- [x] Update article: ARTICLE_UPDATE.md
- [x] Research brief: RESEARCH_BRIEF.md with 7 figures
- [x] README.md updated to reflect warm-start findings
- [x] AIM-v1 accessibility milestone
- [x] Autonomy scaffolding
- [x] Unified experiment harness
- [x] Cross-model conditioning comparison
- [x] Experiment documentation
- [x] V15b geometry isolation
- [x] Cochran's Q bug fix and reprocessing
- [x] Warm-start control experiment

## Key Research Findings (Codex-Validated)
1. **TRAJECTORY PERTURBATION** — random prefix tokens shift generation from formal→exploratory mode
2. **Redistribution effect** — some tasks improve, others regress. Net +12pp mean
3. **Non-monotonic optimum** — 2 tokens = 60% (best), 8 tokens = 44% (overshoot)
4. **Token diversity matters** — diverse tokens (+12pp) >> identical tokens (+4pp)
5. **Direction irrelevant** — random noise = W-projected latents (p=1.0)
6. **CoT mediates** — no-think mode eliminates the effect entirely
7. **Token budget correlation** — wrong answers hit max_new_tokens (~80s = 1024 tokens)
