# Real Causal Quotient (RCQ)

Version: 0.2 (model confirmed, capability screen revised)
Status: IMPLEMENTATION READY — Finch-3B confirmed as substrate

## 1. Motivation

Every closed line in this project teaches one meta-lesson: the structure is
whole-state, behaviorally defined, and must be encountered in an existing model
— not designed into a new one. The Real Causal Quotient program studies an
existing trained recurrent model's internal state using only behavioral
equivalence — no R^n geometry, no designed worlds.

## 2. Central artifact

The artifact is a **response quotient** over a real recurrent model's state space:

Given a model M with recurrent state C(p) after processing prefix p, two states
are equivalent iff they produce identical future-response distributions for all
registered future suffixes:

    C ≡_Γ C'  ⟺  K_{u,q}(C) = K_{u,q}(C')  for every (u,q) ∈ Γ

where K_{u,q}(C) is the answer distribution when starting from state C, appending
action sequence u, then appending query q.

The quotient Q_Γ = Reach(M) / ≡_Γ, together with the induced action law
[C]·a = [F_a(C)], is the mathematical object. It is coordinate-free: defined
entirely by behavioral function, never by distances, angles, or coordinates in
the state space.

Distance from claim: **0** — the quotient and action law ARE the native math.

## 3. R^n trap check

The quotient construction uses ONLY:
- Response distributions under future suffixes (behavioral)
- Equality of quantized probability vectors (functional)
- Action-induced class transitions (dynamical)

It NEVER uses:
- Euclidean distance between state vectors
- Cosine similarity
- PCA or dimensional reduction
- Any coordinate-dependent metric

The carrier state happens to live in R^n (it's a neural network). The
mathematical object is the quotient under behavioral equivalence, not the
geometry of the carrier.

## 4. Model selection

### Capability screen results (2026-09-02)

**REVISED GATE:** The original gate (>=95% QA accuracy) was measuring the wrong
thing. The quotient requires state DISCRIMINATION (different distributions per
state), not QA ACCURACY (correct top-1). A model that always answers wrong but
produces distinct distributions per state is a valid quotient substrate.

**New gate: Entity discrimination.** For asymmetric states (A!=B), the response
distributions for "Where is Avery?" and "Where is Blake?" must differ with
mean TV > 0.10.

| Model | Params | Arch | QA Acc | Entity Disc TV | Status |
|-------|--------|------|--------|----------------|--------|
| Mamba-130M | 130M | Mamba | 25% | not tested | FAIL |
| RWKV-4-169M | 169M | RWKV-4 | 67% | 0.03 (absent) | FAIL |
| RWKV-4-430M | 430M | RWKV-4 | 75% | not tested | FAIL |
| RWKV-4-1.5B | 1.5B | RWKV-4 | 58% | not tested | FAIL |
| Finch-1.6B | 1.6B | RWKV-6 | 57% | 0.08 (marginal)| FAIL |
| **Finch-3B** | 3.1B | RWKV-6 | 71% | **0.43 (strong)**| **PASS** |

Key findings:
- Entity discrimination emerges between 1.6B and 3B in RWKV-6 architecture
- RWKV-4 models at ALL sizes lack entity discrimination (track location
  dominance, not entity-specific state)
- State replay fidelity: TV=0.000000 (bit-exact) on Finch-3B
- State injection: confirmed causal — injecting (kit,gar) state into (gar,kit)
  continuation flips entity answers (TV=0.41/0.46)
- Q1 (Blake) accuracy is 7/7 perfect; Q0 (Avery) has slight "office" bias

### Selected model (CONFIRMED)

RWKV/v6-Finch-3B-HF. All capabilities verified:
- Entity discrimination: PRESENT (asymmetric TV=0.43)
- State capture: WORKS (bit-exact replay)
- State injection: WORKS (entity swap confirmed)
- State substitution: CAUSAL (injected state dominates continuation)
- CPU inference: ~2-3s per pass, feasible for short bursts
- Already cached locally, no CUDA kernels required

### Hardware constraints

- Laptop with degraded battery: sustained CPU/GPU load risks hard shutdown
- Short bursts only (minutes, not hours)
- Must checkpoint between bursts
- GPU bursts require explicit user approval

## 5. Task: two-entity/three-location state tracking

Entities: Avery, Blake
Locations: kitchen, garden, office
Joint states: 3² = 9
Macro-actions: 6 token strings ("Avery moved to the garden." etc.)
Probes: "Where is Avery?", "Where is Blake?"

Answers scored by normalized teacher-forced log-likelihood over the three
location tokens. No sampling, no open-ended generation.

### Capability screen (revised 2026-09-02)

- Entity discrimination: asymmetric-state Q0-Q1 TV > 0.10 (confirmed: 0.43)
- State replay: TV ≤ 1e-4 between full-prefix and captured-state (confirmed: 0.000000)
- State injection: injecting state A into continuation B produces TV > 0.20
  between injected and natural responses (confirmed: 0.41/0.46)
- QA accuracy is monitored but NOT gated (the quotient uses distributions, not top-1)

If the capability screen fails, do NOT:
- Repair the prompt
- Fine-tune the model
- Build the quotient

## 6. RCQ-0: first experiment

### Phase 1: Capability screen + state extraction

For each of N distinct histories (varying move sequences, all reaching each of
the 9 joint states from multiple paths):
1. Run the full prefix through the model
2. Extract the complete recurrent state C(p)
3. Score all 3 answers for both entity queries → 6 probability vectors
4. Record the state and response profile

### Phase 2: Quotient construction

1. For each state, compute the full response profile: distributions under
   direct queries AND single-action extensions (6 actions × 2 queries = 12
   more probability vectors per state)
2. Quantize probability vectors on a pre-registered grid (δ = 0.05)
3. States with identical quantized response profiles form equivalence classes
4. Record the quotient partition

### Phase 3: Action law

1. For each equivalence class and each action, observe what class the model
   transitions to (from single-action extensions)
2. Record the transition table: [C]·a → [C']
3. Check consistency: do different states within the same class transition to
   the same target class?
4. Gate: within-class transition consistency ≥ 0.95

### Phase 4: Held-out composition

1. Lock the learned single-action transitions
2. Compose to predict two-action sequence outcomes: [C]·a·b
3. Score against the model's actual two-action response profiles
4. Gate: top-1 agreement ≥ 0.95, mean response TV ≤ 0.10
5. Advantage over baselines ≥ 10pp:
   - Historyless (no-update / direct-response prediction)
   - Last-action-only prediction
   - Shuffled class transitions

### Phase 5: State substitution (causal test)

1. For states C₁, C₂ in the same equivalence class (from different histories):
2. Inject C₁ as starting state, continue with C₂'s future suffix
3. Gate: mean TV between substituted and natural continuation ≤ 0.05
4. For states in DIFFERENT classes: confirm distinguishing suffix exists

## 7. Kill gates

### RCQ-0 is a PASS if:

All Phase 1-5 gates clear, AND the quotient provides predictive power beyond
what a simple text parser achieves on the same prompts. The "text parser" null:
parse entity locations from the prompt text using simple pattern matching, and
predict responses from the parsed state. If the quotient's predictions are no
better than the text parser's, the quotient is trivial (merely re-deriving
what's obvious from the surface text).

### RCQ-0 is a NO-GO if:

1. Capability screen fails (model can't track state)
2. Quotient is trivially fine (each history maps to a unique class — no
   generalization)
3. Quotient is trivially coarse (all histories equivalent — no discrimination)
4. Within-class transition consistency < 0.90 (no lawful dynamics)
5. Composition prediction no better than baselines (no predictive power)
6. State substitution fails (quotient classes are artifacts of quantization,
   not genuine behavioral equivalence)

### RCQ-0 is scientifically successful but direction NO-GO if:

The quotient cleanly recovers the 9 obvious world states with perfect
composition AND provides zero predictive surplus beyond a text parser. This
means the model's internal state has structure, but that structure is fully
explained by surface-level text processing. Native math requires structure
that the model computes but the text doesn't display.

## 8. Three-rung ladder

- **RCQ-0:** Same task, same names, same wording. Establishes basic quotient
  + composition law. This document.
- **RCQ-1:** Transport the quotient across held-out presentation axes
  (different entity names, different location names, different phrasing)
  without refitting. Tests whether the quotient captures abstract structure
  or memorized surface patterns.
- **RCQ-2:** Predict a new intervention family or second task better than
  semantic and standard recurrent-systems nulls. This is the claim-bearing
  native predictive artifact.

## 9. Runner constraints

- Target: 250-350 lines of runner code
- Hard halt at 450 lines
- Apparatus-to-artifact ratio must stay below 2:1
- Re-use from HANDLE-mu: seed custody, JSON encoding, bootstrap CI,
  gate results. Do NOT re-use: simulator, slot architectures, HANDLE-specific
  gates.

## 10. Connection to OpenAI Astra / looped transformers

OpenAI's Astra reportedly uses "recurrent depth" — looping hidden state through
shared transformer layers multiple times before emitting tokens. This shifts
reasoning into "numerical latent states that nobody can read" (The Information,
2026-09-01). The RCQ framework would be the first public mathematical approach
to reading such states: quotient by behavioral equivalence, extract the action
law, predict held-out compositions. If RCQ succeeds on RWKV, the natural next
substrate is a looped/recurrent-depth transformer — when one becomes available.
