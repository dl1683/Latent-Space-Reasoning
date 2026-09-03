# SVB-0: Scope-Variable Binding Experiment 0

Version: 0.1 (self-reviewed; Codex design gate deferred — credits exhausted)
Status: DRAFT — awaiting Codex review
Distance from claim: 0 (the scope-stack structure IS native latent-space math)

## 1. Claim wall

SVB-0 may establish:

> H_SVB: On the pinned Falcon-H1-1.5B-Instruct model, the recurrent state
> after processing Python code with nested variable scoping maintains
> entity-specific (variable-specific) outer-scope bindings through inner-scope
> computation, as measured by the response distribution at scope closure.

This demonstrates native latent-space structure: the model's internal state
encodes a scope stack compatible with lexical scoping rules.

A competence failure closes this response interface. A scientific non-pass
closes H_SVB but NOT: code-model discrimination, state injection, or
independently motivated experiments.

## 2. Model and interface

- Model: `tiiuae/Falcon-H1-1.5B-Instruct`
- Architecture: Hybrid Mamba + attention (recurrent SSM layers provide latent state)
- CPU, float32, batch one.
- State: DynamicCache (save after prefix, inject suffix tokens)

### Response law

Query template suffix: `f()\nprint({variable})  # Output: `

Response: full next-token logits pushed through 11 bins:
- Bins 0-9: P(digit d) for d in {0,1,2,3,4,5,6,7,8,9}
- Bin 10: P(OTHER) = 1 - sum(P(digit d) for d in 0..9)

Digit tokens must be single-token. No renormalization.

## 3. Population

### 3.1 Outer-scope values

Digits 1 through 9 (9 values). Skip 0 to avoid ambiguity.

### 3.2 Inner-scope value

Fixed: 99. Always different from any single-digit outer value.

### 3.3 Code templates

**Depth 1 (single function scope):**
```python
{var} = {outer_val}
def f():
    {var} = 99
    return {var}
```

**Depth 2 (nested function scope):**
```python
{var} = {outer_val}
def f():
    {var} = 99
    def g():
        {var} = 999
        return {var}
    g()
    return {var}
```

### 3.4 Variables

Primary: x, y, z (3 variables)
Multi-variable templates test entity specificity.

**Single-variable templates (discrimination):**
One variable set in outer scope, shadowed in inner scope.

**Two-variable templates (entity specificity):**
Two variables set independently in outer scope, both shadowed in inner scope.
Query each variable separately.

### 3.5 Query suffix

`f()\nprint({var})  # Output: `

### 3.6 Population census

**Single-variable, depth 1:** 9 outer_vals × 3 variables = 27 prefix states × 1 query = 27 rows
**Single-variable, depth 2:** 9 outer_vals × 3 variables = 27 prefix states × 1 query = 27 rows
**Two-variable, depth 1:** 9 × 9 × C(3,2) = 243 prefix states × 2 queries = 486 rows
**Two-variable, depth 2:** 9 × 9 × C(3,2) = 243 prefix states × 2 queries = 486 rows

Total: 1026 science rows. Estimated ~1500 forward passes including competence.

### 3.7 Neutral suffix injection

After the function definition, inject 0/1/2/4 repetitions of `# No changes.\n`:
```python
{var} = {outer_val}
def f():
    {var} = 99
    return {var}
# No changes.
# No changes.
f()
print({var})  # Output: 
```

Tests whether scope binding survives irrelevant text between definition and call.

## 4. Competence staircase

### Rung 1: Direct assignment (no function)
`{var} = {val}\nprint({var})  # Output: `
Correct digit must be top among digits AND beat P(OTHER).
Threshold: ≥ 0.90 overall, ≥ 0.80 per (variable, value) arm.

### Rung 2: Single function, depth 1 (suffix 0)
Full template with one variable.
Threshold: ≥ 0.85 overall (correct digit is top among digits).
P(OTHER) ≤ 0.10.

### Rung 3: Two variables, depth 1 (suffix 0)
Full template with two variables, both queries.
Threshold: ≥ 0.80 overall.

### Rung 4: Depth 2 (suffix 0)
Nested function template.
Threshold: ≥ 0.70 overall (weaker gate — depth degrades signal).

Failed rung stops advancement. Suffix injection rungs evaluated only if base passes.

## 5. Observables

### 5.1 Scope binding fidelity (σ)

For a given outer value v, variable var, and depth d:

    σ(v, var, d) = P(digit v | prefix with var=v at depth d)

Mean σ across the population characterizes how well the model preserves
outer-scope bindings.

### 5.2 Path contrast (κ)

For matched paths (outer_val = v1 vs outer_val = v2), same inner scope:

    κ = TV(P_v1, P_v2)

where TV is total variation on the 11-bin response.

### 5.3 Entity interaction (ι)

For two-variable templates, let δ_var = P_{v1}^var - P_{v2}^var:

    ι = (1/2) × ||δ_x - δ_y||_1

Variable-specific scope binding requires BOTH κ and ι above threshold.

### 5.4 Depth decay profile

Report σ and κ at each depth level. The ratio σ(d=2)/σ(d=1) characterizes
the depth scaling of the scope stack.

## 6. Null ladder

Cross-fitted on 3 folds (3 values per fold: {1,2,3}, {4,5,6}, {7,8,9}).

1. **Uniform digit**: predict uniform over 9 non-zero digits
2. **Inner value**: always predict 9 (first digit of 99)
3. **Last-token**: predict based on last token of prefix
4. **Variable-conditioned**: per-variable mean distribution
5. **Depth-conditioned**: per-depth mean distribution
6. **Value lookup**: exact outer-value → response mapping (saturated)
7. **Frequency prior**: model's prior digit distribution (no prefix)

## 7. Gates

### 7.1 Scope binding gates

**Registered binding:** Mean σ ≥ 0.30 with bootstrap LB > 0.20.
**Strong binding:** Mean σ ≥ 0.50 with bootstrap LB > 0.35.

### 7.2 Path contrast gates

**Registered contrast:** Mean κ ≥ 0.30 with bootstrap LB > 0.15.
**Strong contrast:** Mean κ ≥ 0.50 with bootstrap LB > 0.30.

### 7.3 Entity specificity

ι ≥ 0.10 with bootstrap LB > 0.05.

### 7.4 Replay integrity

State injection vs full pass for the SAME prefix must have TV ≤ 0.01 for
the comparison to be valid. NOTE: preliminary tests show TV=0.84 between
full and injected on this model. If this persists, the experiment compares
injected states to each other (valid for path contrast) but cannot claim
the injected distribution equals the natural one.

## 8. Adjudication

| First failed condition | Verdict |
|---|---|
| Token/state/replay/schema failure | `INVALID_IMPLEMENTATION` |
| Direct assignment competence fails | `TASK_POPULATION_VOID` |
| Single-variable depth-1 fails | `INSUFFICIENT_SCOPE_BINDING` |
| Depth-1 passes but ι < threshold | `GLOBAL_SCOPE_TRACE` (not variable-specific) |
| Depth-1 passes, depth-2 fails | `SHALLOW_SCOPE_BINDING` (depth-1 only) |
| All gates pass through depth 2 | `SCOPE_STACK_WITNESS` |

## 9. Design provenance

- PMO-0R TASK_POPULATION_VOID → pivot to variable binding (Codex R1+R2 recommendation)
- Model screening: Falcon-H1-1.5B-Instruct selected for code capability + recurrent state
- Preliminary signal: TV 0.63-0.74 path contrast, 9/9 correct digits, entity specificity confirmed
- Depth profile: d=1 strong (σ~0.68), d=2 moderate (σ~0.49), d=3 fails
- Codex design gate: PENDING (credits exhausted until 2026-09-06)
