# Goals

## Mission
- Liberate high-quality intelligence for everyone through low-cost, low-resource, transparent, and auditable systems.
- Prefer large efficiency gains (10x-100x lower cost) even with modest quality tradeoffs versus maximum-benchmark systems.

## Active Goals

### 1. Characterize the Warm-Start Mechanism
The core finding: prepending 8 random embedding-scale tokens improves Qwen3-4B arithmetic by +12pp (32% -> 44%). Direction doesn't matter. Characterize WHY this works.
- [ ] Mean-embedding control (does token diversity matter, or do 8 identical tokens also work?)
- [ ] Zero-embedding control (does the attention mask extension suffice?)
- [ ] Token count dose-response (1, 2, 4, 8, 16, 32 tokens)
- [ ] RMS scale sweep (0.5x, 1x, 2x, 5x target_rms)
- [ ] Nested-easy noise control (does noise show Cochran's Q significance on easy tasks?)

### 2. Test Warm-Start Generality
Is warm-start Qwen3-specific or a general phenomenon?
- [ ] Non-Qwen models: Llama-3.2-3B, Phi-3-mini, Gemma-2-2B
- [ ] Non-arithmetic tasks: GSM8K subset, ARC-Easy, logic puzzles
- [ ] Larger models: does the effect vanish at scale?
- [ ] Multiple task counts (50-100 tasks for statistical power)

### 3. Paper: "Free Lunch: Random Embedding Tokens Improve Small-Model Reasoning"
- [ ] Complete mechanism characterization (Goal 1)
- [ ] Complete generality tests (Goal 2)
- [ ] Position: zero-cost inference improvement, no training needed
- [ ] Address reviewer concerns: efficiency argument, scale dependency

## Concluded / Falsified Goals

### Latent Space Search — FALSIFIED (2026-03-04)
- Random noise matches latent-projected performance (44% vs 44.4%, Mann-Whitney p=1.0)
- Latent direction carries no signal. Improvement is warm-start, not direction.
- Evolution, CMA-ES, QD, Active Inference — all futile because nothing to search for.
- V17, V18, V19 runners: DEAD CODE (do not run)

### Hyperbolic Geometry — CONCLUDED (2026-03-03)
- Euclidean = Hyperbolic under same conditioning (V15a, V15b)
- Geometry adds no value.

## Completed Goals
- [x] AIM-v1 accessibility milestone
- [x] Autonomy scaffolding
- [x] Unified experiment harness
- [x] Cross-model conditioning comparison
- [x] Experiment documentation
- [x] V15b geometry isolation
- [x] Cochran's Q bug fix and reprocessing
- [x] Warm-start control experiment

## Key Research Findings (Codex-Validated)
1. **WARM-START IS THE MECHANISM** — prepending random embedding tokens improves accuracy by +12pp
2. **Latent direction does not matter** — random noise = W-projected latents (p=1.0)
3. **Chain-of-thought IS the mediator** — no-think mode eliminates the effect entirely
4. **Latents DO differ on easy tasks** — Cochran's Q=23.2 (p=0.006) but mostly in harmful directions
5. **Local evolution fails** — small mutations can't exploit landscape, and landscape isn't exploitable anyway
6. **Hyperbolic = Euclidean** — geometry irrelevant under same conditioning
7. **Both conditioning methods eliminate hallucination** — robust across model sizes
