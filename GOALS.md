# Goals

> **ARCHIVED**: This file is a historical planning snapshot.  
> For the current evidence and execution surface, start with [START_HERE.md](START_HERE.md).

## Mission
- Liberate high-quality intelligence for everyone through low-cost, low-resource, transparent, and auditable systems.
- Prefer large efficiency gains (10x-100x lower cost) even with modest quality tradeoffs versus maximum-benchmark systems.

## Active Goals

### 1. NeurIPS Paper: PGRMS — Perturbation-Gated Reasoning Mode Selection
**Core finding:** Prepending random embedding-scale tokens improves Qwen3-4B arithmetic by up to +28pp (32% -> 60% at 2 tokens). The dose-response is non-monotonic. The mechanism decomposes into think-mode gating (+8pp) and perturbation-specific optimization (+20pp).

**Completed experiments:**
- [x] Dose-response: 0, 1, 2, 3, 8 tokens (non-monotonic peak at 2)
- [x] Controls: zero-embedding, mean-embedding, force-think
- [x] 3-tok n=10 replication (44.0%, SD=1.33, equalization DEAD)
- [x] Oracle coverage analysis (2-tok k=3 = 88%, 3-tok k=10 = 80%)
- [x] Force-think decomposition (think=+8pp, noise=+20pp)
- [x] Deterministic chaos / invariance length analysis

**Running:**
- [ ] 2-tok n=10 rerun (EXISTENTIAL — tests equalization at scale)

**Queued (post 2-tok):**
- [ ] Think-gate probe (~5 min) — <think> logit analysis under perturbation
- [ ] Shi discrete token control t=2 (~30 min) — continuous vs discrete comparison
- [ ] Word problem cross-task replication (~90 min) — external validity

### 2. Test Generality (DEFERRED until paper MVP)
- [ ] Non-Qwen models: Llama-3.2-3B, Phi-3-mini, Gemma-2-2B
- [ ] Non-arithmetic tasks: GSM8K subset, logic puzzles
- [ ] Larger models: does the effect vanish at scale?

## Concluded Explorations

### Directional Latent Search — CONCLUDED (2026-03-04)
- Random noise = latent-projected (p=1.0). The mechanism is direction-agnostic.
- The improvement comes from token presence/diversity, not specific latent directions.

### Hyperbolic Geometry — CONCLUDED (2026-03-03)
- Euclidean = Hyperbolic under same conditioning. Geometry doesn't differentiate outcomes.

### 3-tok Equalization — CONCLUDED (2026-03-06)
- N1-N10: [11,11,11,10,13,12,9,9,12,12], SD=1.33, p=0.335
- Equalization is 2-TOKEN-SPECIFIC. Does not persist at 3 tokens.

## Key Research Findings (Codex-Validated)
1. **Non-monotonic optimum** — 2 tokens = 60% (best), 1 = 42.7%, 3 = 44%, 8 = 44%
2. **Two-component decomposition** — think-mode gating (+8pp) + noise perturbation (+20pp)
3. **Oracle efficiency** — 3 runs at 2-tok = 88% coverage; 10 runs at 3-tok = only 80%
4. **Solve-count equalization** — 2-tok specific (p=0.031); dead at 3-tok (p=0.335)
5. **Direction steers task selection** — same accuracy, different task subsets (Fleiss kappa=0.278)
6. **Direction-agnostic** — random noise = W-projected latents (p=1.0)
7. **CoT mediates** — no-think mode eliminates the effect entirely
8. **Deterministic chaos** — greedy decoding, byte-identical invariance lengths decrease with energy
