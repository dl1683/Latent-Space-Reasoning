# PMO-0R: Path-Memory Observability (Revised)

Version: 1.0 (Codex R1+R2 locked, 2026-09-02)
Status: LOCKED — launch authorized
Distance from claim: 1 (bounded continuation-distinguishability witness)

## 1. Claim wall

PMO-0R may establish only:

> H_PMO: On the pinned Finch revision and locked entity-location grammar,
> same-endpoint, last-two-matched histories retain a materially observable
> entity-conditioned path trace under the registered suffix/query family.

Never infer intrinsic fibers, forgetting, memory capacity, topology, a groupoid,
or a general latent-space law.

A competence failure closes this response interface as unusable. A scientific
non-pass closes H_PMO and further attempts to derive a path calculus from this
exact family. It does NOT close: Finch entity discrimination, causal state
injection, bit-exact replay, or independently motivated experiments with a
different mathematical claim and estimand.

## 2. Model and interface

- Model: `RWKV/v6-Finch-3B-HF`
- Revision: `c17eed9625bba0bc71c4b67db39f6d34ac9846fc`
- CPU, float32, batch one.

Query template:

    \nQuestion: Where is {entity}? Answer exactly one of: kitchen, garden, office.\nAnswer: The

Response: full next-token law pushed through four fixed bins:

    {kitchen_token (52085), garden_token (46078), office_token (46701), OTHER}

P(OTHER) = 1 - P(kitchen) - P(garden) - P(office).

Conditional three-logit renormalization is prohibited.

## 3. Population

### 3.1 Roots

All nine initial configurations in {kitchen, garden, office}^2 for (Avery, Blake).
Root phrasing: "Avery is in the {loc}. Blake is in the {loc}."

### 3.2 Asymmetric panels

| Panel | A1 | A2 | B1 | B2 | Final (Avery, Blake) |
|-------|-----|-----|-----|-----|---------------------|
| KG | A→garden | A→kitchen | B→kitchen | B→garden | (kitchen, garden) |
| GO | A→office | A→garden | B→garden | B→office | (garden, office) |
| OK | A→kitchen | A→office | B→office | B→kitchen | (office, kitchen) |

All panels produce asymmetric endpoints (Avery ≠ Blake location).
Every location appears twice as a correct answer across panels (balanced).

### 3.3 Linear extensions

Six orderings preserving A1<A2 and B1<B2:

1. A1 A2 B1 B2
2. A1 B1 A2 B2
3. A1 B1 B2 A2
4. B1 A1 A2 B2
5. B1 A1 B2 A2
6. B1 B2 A1 A2

### 3.4 Primary matched pairs

Last-two-matched pairs for commutation defect:

- Pair 1: `A1 B1 A2 B2` vs `B1 A1 A2 B2` (last 2: A2 B2)
- Pair 2: `A1 B1 B2 A2` vs `B1 A1 B2 A2` (last 2: B2 A2)

These isolate the early A1/B1 swap while holding the final two action strings
exactly fixed.

### 3.5 Common continuation

Carrier macro: ` Nothing else changes.`
Pinned tokenizer IDs: `[50487, 30746, 51165, 47]` (4 tokens, scales linearly).

Evaluate at macro repetitions: 0, 1, 2, 4.
Advance sequentially from saved recurrent states.
Assert canonical whole-string tokenization equals repetition of pinned sequence.

### 3.6 Population census

- 162 four-action history states (9 roots × 3 panels × 6 extensions)
- 324 response rows per suffix length (162 × 2 queries)
- 1,296 scientific response rows total (324 × 4 suffix lengths)
- Maximum ~1,500 logical forward invocations (including competence, replay, etc.)
- CPU forecast: ~21 minutes (well under 90-minute abort threshold)

## 4. Competence staircase

Use the same query/interface throughout. Every rung requires:

- Overall accuracy ≥ 0.95
- Accuracy ≥ 0.90 in every (depth, entity, target-location) arm
- Correct location must beat the OTHER bin
- No prompt repair after the one locked template

Staircase:

1. Direct facts: 36 rows (9 roots × 2 statement orders × 2 entities)
2. Two actions: 108 rows (9 roots × 3 panels × 2 A1B1/B1A1 orders × 2 entities)
3. Four actions at suffix 0: 324 rows
4. Suffix lengths 1, 2, 4: one rung at a time

A failed rung stops advancement.

## 5. Observables

### 5.1 Contextual commutation defect (κ)

For matched paths π, π', suffix h:

    κ = (TV(P_π^A, P_π'^A) + TV(P_π^B, P_π'^B)) / 2

where TV is total variation on the 4-bin response.

### 5.2 Entity interaction (ι)

For matched paths and suffix h, let δ_e = P_π^e - P_π'^e:

    ι = (1/4) × ||δ_A - δ_B||_1

Entity-specific path memory requires BOTH κ and ι to pass.
If only κ passes: global path trace (not entity-specific).

### 5.3 Geometric-mean null

Parameter-free null for the equal-information parent profiles:

    q_geo(j) ∝ sqrt(q_L(j) × q_R(j))

where q_L, q_R are the two matched-path responses.

## 6. Null ladder

Cross-fitted on 3 root folds (6 train / 3 test roots per fold).
All methods use the same 4-bin response law and denominator.

1. Identity: zero contrast (predict target = source)
2. Panel centroid: panel-conditioned contrast ignoring path identity
3. Last-1 action features
4. Last-2 action features
5. Discounted action history (lambda selected inside training roots)
6. Panel-additive action-position main effects
7. Exact path-ID lookup (saturated descriptive comparator)
8. Shuffled-path lookup (fixed seed 42)
9. Geometric-mean null (parameter-free)

## 7. Gates

### 7.1 Path-witness gates (per suffix rung)

Two bands:

**Registered witness (0.020 band):**
- Mean TV (κ) ≥ 0.020
- One-sided 95% root-cluster lower bound > 0.005
- Replicated across ≥ 2 target-location assignments
- Replay maximum TV ≤ 0.0001
- One-shot vs saved-state discrepancy ≤ 0.0002

**Strong/material witness (0.050 band):**
- Mean TV (κ) ≥ 0.050
- Root-cluster lower bound > 0.020
- Same replication requirement

**Entity specificity:**
- ι ≥ 0.020 with root-cluster LB > 0.005

### 7.2 Null-ladder gates

For stable path specificity, path-ID must beat every substantive null by:
- Mean TV advantage ≥ 0.01
- Root-cluster lower bound > 0

### 7.3 Decisive statistic

Aggregated over the complete locked suffix family {0, 1, 2, 4}.
Per-suffix results are mandatory profiles but do not provide four
independent opportunities to pass.

No monotonicity gate is permitted. Report M_0, M_1, M_2, M_4
and ratios M_h/M_0 descriptively.

## 8. Adjudication

| First failed condition | Verdict and action |
|---|---|
| Hash/tokenization/replay/population/cache/schema/call-census | `INVALID_IMPLEMENTATION` — no scientific interpretation |
| Any competence rung fails | `TASK_POPULATION_VOID` — close this response interface |
| Suffix-0 witness gate fails (κ < 0.020) | `NO_REGISTERED_PATH_WITNESS` — close H_PMO; do not say no memory exists |
| Witness passes but ι < 0.020 | `GLOBAL_TRACE_ONLY` — path trace exists but not entity-specific |
| Witness passes but path-ID does not beat null ladder | `LOW_ORDER_PRESENTATION_SUFFICIENT` — retain bounded distinction |
| A later suffix rung fails competence | stop staircase; report highest-passing suffix |
| All witness and null gates pass through length 4 | `BOUNDED_TRACE_MEMORY_WITNESS` — return to math dialogue only |

Any scientific non-pass closes H_PMO. Post-PMO pivot: nested scope/variable
binding on a code-capable recurrent model, regardless of PMO sign.

## 9. Pre-run kills

- CPU forecast above 90 minutes
- Measurement-to-artifact ratio above 5:1
- Untyped gates, incomplete lifetime-call accounting, cache ambiguity
- Failed replay/injection checks
- Any change to locked population after viewing scientific effects

## 10. Measurement-to-artifact ratio

Declared: **1:1** — the observation data and its statistical analysis IS the
deliverable. There is no separate learned artifact; the mathematical claim is
the behavioral witness itself. Runner code is the measurement apparatus.

## 11. Design provenance

- Codex PMO-0 design gate R1: identified two PFC-0 defects (symmetric endpoints,
  hand-picked logit renormalization), proposed PMO-0R replacement spec
- Codex PMO-0R design gate R2: addressed four Claude objections, locked two-band
  threshold, added entity-interaction statistic ι, scoped closure, sequenced
  variable-binding pivot
- Post-hoc analyses: log-space composition failure (universal, not simplex-specific),
  geometric mean as best equal-info predictor, kitchen calibration bias (47.6%)
