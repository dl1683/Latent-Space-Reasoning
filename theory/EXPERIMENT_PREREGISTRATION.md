# Semantic-Holonomy Crossover: Pre-registration

Status: DRAFT pre-registration. Awaiting Codex review.
Date: 2026-09-04.
Tests: Intensional Descent Criterion (theory/INTENSIONAL_DESCENT.md).

## Hypothesis

Pretrained transformers compress execution histories into equality-witness
classes that are strictly finer than denotational state. The model's
response action factors through intensional role (persistence, rewrite,
restoration, observation) but NOT through extensional denotation alone.

## Design: Randomized Semantic-Holonomy Crossover

### Factor structure (2×N×M×K)

Orthogonalize four factors:

1. **Denotation** (held exactly fixed by symbolic execution): all trace
   variants produce the same final program state. Verified automatically,
   not by the experimenter.

2. **Intensional role** (the treatment variable): which KIND of "doing
   nothing" a trace performs.
   - ASSERT: declarative assertion that state is unchanged
   - REWRITE: identity assignment (x = x)
   - RESTORE: assign different value then reassign original (x=5; x=3)
   - OBSERVE: read-only check (print(x) or assert x == 3)
   - BOUNDARY: structural delimiter only (newline, pass)

3. **Surface realization** (nuisance variable — must NOT predict behavior):
   many paraphrases per intensional role.
   - ASSERT realizations: "# No changes", "# State unchanged",
     "# Nothing modified", "# Variables preserved", etc.
   - REWRITE realizations: "x = x", "x = +x", "x = int(x)",
     "x = (lambda v: v)(x)", etc.
   - RESTORE realizations: "x = 5\nx = 3", "x = 0\nx = 3",
     "tmp = x\nx = 99\nx = tmp", etc.
   - OBSERVE realizations: "_ = x", "assert x == x",
     "type(x)", etc.
   - BOUNDARY realizations: "\n", "pass\n", "\n\n", "...\n", etc.

4. **Lexical assignment** (the nonce-legend control): randomized mapping
   from physical markers to intensional roles.
   - In condition A: marker α means ASSERT, marker β means REWRITE
   - In condition B: marker α means REWRITE, marker β means ASSERT
   - Same physical tokens, swapped meanings
   - If behavior follows the marker (not the role), this is lexical
     cueing, not intensional structure

### Hold-out structure

Train/test split by:
- Entire paraphrase families (not individual paraphrases)
- Nonce legend assignments
- Variable names (x, y, z, a, b, etc.)
- Outer values (0-9)
- Operation compositions (atomic → composed)
- Domains (Python scoping → other task families if Gate 1 passes)

### Gate 1: Semantic Descent

**Question:** Does the model's behavior follow the randomly assigned
intensional role, not the surface form?

**Method:**
1. For each intensional role R and each surface realization s of R,
   measure the response operator K_s on the (C, L, R) quotient.
2. Compute within-role variance (different surfaces, same role) and
   between-role variance (different roles).
3. Fit a classifier: surface features → behavior vs. intensional role →
   behavior.

**Pass criterion:** Between-role variance >> within-role variance, AND
the role-based classifier generalizes to held-out surface realizations
with accuracy > 0.8, AND held-out nonce-legend swaps produce behavior
following the assigned role, not the physical marker.

**Kill criterion:** Effect follows surface features (token identity,
position, length) rather than role. Within-role variance ≥ between-role
variance. Nonce swap reverses behavior.

### Gate 2: Composition (only if Gate 1 passes)

**Question:** Do witness transport operators compose predictably?

**Method:**
1. Estimate atomic witness transport operators K_R for each role R from
   Gate 1 data.
2. Generate UNSEEN compositions: ASSERT then REWRITE, OBSERVE then
   RESTORE, etc.
3. Predict composed operators from atomic ones (using the context-indexed
   composition model from D14).
4. Compare predictions to actual measurements.

**Pass criterion:** Predicted composed operators match actual within
measurement noise (TV distance < 0.05), for held-out composition orders
and operand combinations.

**Kill criterion:** Each composition must be fitted separately — atomic
operators cannot predict unseen compositions. Prediction error > 0.15 TV.

### Gate 3: Causal Transfer (only if Gate 2 passes)

**Question:** Can the equality-witness state be transferred causally?

**Method:**
1. Take two prefixes x₁, x₂ with the same denotational state but
   different intensional histories (e.g., x₁ ends with ASSERT, x₂
   ends with REWRITE).
2. Extract internal state (KV cache) from x₁ at the witness-bearing
   positions.
3. Transplant into x₂'s KV cache at the corresponding positions.
4. Test whether downstream behavior follows the DONOR's intensional
   role, not the recipient's surface history.
5. Control: transplant from a surface-matched, confidence-matched
   position that has a DIFFERENT intensional role.

**Pass criterion:** Donor-transplant shifts behavior toward donor's
witness class. Control-transplant does not. Effect size > 2× control
for at least 3 of 5 role types.

**Kill criterion:** Transplant has no effect, or surface-matched control
transplant produces the same shift. Internal state cannot be separated
from surface history.

## Analysis plan

### Primary analysis
- Mixed-effects model: response_operator ~ intensional_role +
  (1|surface_realization) + (1|variable) + (1|outer_value)
- Random effect for surface realization MUST be small relative to
  fixed effect of intensional role

### Secondary analyses
- Kernel operator estimation per role (mean and CI)
- Composition prediction accuracy (held-out)
- Cross-model replication (if all gates pass, run on ≥2 additional
  models from different families)

### Multiple comparisons
- 5 roles × 3 gates = 15 primary tests
- Bonferroni correction: α = 0.05/15 = 0.0033 per test
- Pre-registered: no fishing

## Sample size

### Gate 1 (behavioral)
- 5 intensional roles × ≥8 surface realizations × 3 variables ×
  9 outer values × 5 depths = 5×8×3×9×5 = 5,400 observations minimum
- Plus hold-out: 5 roles × 4 held-out surfaces × same = 2,700

### Gate 2 (composition)
- 5×4 = 20 ordered pairs of roles × 3 variables × 9 values × 3 depths
  = 1,620 compositions
- Hold-out: 10 unseen compositions × same = 810

### Gate 3 (causal transfer)
- 5 roles × 4 transplant directions × 3 variables × 9 values × 3 depths
  = 1,620 interventions
- Plus matched controls: same count

## Model and hardware

- Primary: Qwen3-1.7B-Base (continuing from SVB series)
- CPU-only (no GPU approval)
- Estimated time: Gate 1 ~30 min, Gate 2 ~15 min, Gate 3 ~45 min
  (if KV cache surgery is feasible on CPU)

## Decision tree

```
Gate 1 PASS → Gate 2
Gate 1 FAIL → CLOSE "intensional native mathematics" as contextual cueing
Gate 2 PASS → Gate 3
Gate 2 FAIL → Weaken claim to "semantic sensitivity without composition"
Gate 3 PASS → Cross-model replication → PUBLISH if replicates
Gate 3 FAIL → Weaken claim to "behavioral, not internal, witness structure"
```
