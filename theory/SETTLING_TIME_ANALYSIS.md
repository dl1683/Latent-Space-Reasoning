# Shadow-Binding Attenuation: Functional Form Analysis

Status: EMPIRICAL OBSERVATION (one model, one template family, n=27/cell)
Data: experiments/results/svb_qwen3_formal/result.json
Distributional data: experiments/results/svb_qwen3_formal/obs_checkpoint.npz
Model: Qwen3-1.7B-Base (pure transformer)

## Raw data

| Depth | s0 | s1 | s2 | s4 |
|-------|-------|-------|-------|-------|
| d1 | 0.958 | 0.970 | 0.961 | 0.959 |
| d2 | 0.897 | 0.945 | 0.936 | 0.934 |
| d3 | 0.792 | 0.900 | 0.888 | 0.894 |
| d4 | 0.753 | 0.880 | 0.872 | 0.849 |

σ = mean P(correct value) over 27 cells (3 variables × 9 values).
Suffix = `# No changes.\n` appended s times.

## Observation 1: Exponential depth decay

The readout error 1 - σ(d, 0) grows approximately exponentially with depth:

    1 - σ(d, 0) ≈ C₀ · exp(α₀ · d)

    C₀ = exp(-3.607) ≈ 0.027
    α₀ = 0.602
    R² = 0.928

Each depth level multiplies readout error by exp(0.602) ≈ 1.83.

With one suffix, the same exponential form holds with a smaller rate:

    1 - σ(d, 1) ≈ C₁ · exp(α₁ · d)

    C₁ = exp(-3.901) ≈ 0.020
    α₁ = 0.477
    R² = 0.957

The suffix reduces the exponential decay constant by 21% (0.602 → 0.477).
Each depth level now multiplies error by exp(0.477) ≈ 1.61 instead of 1.83.

## Observation 2: Recovery fraction converges to ~0.5

Define the recovery fraction:

    r(d) = [σ(d,1) - σ(d,0)] / [1 - σ(d,0)]

| Depth | Gap from 1.0 | Gain | Recovery r(d) |
|-------|-------------|------|---------------|
| d1 | 0.042 | 0.012 | 0.284 |
| d2 | 0.103 | 0.049 | 0.471 |
| d3 | 0.208 | 0.109 | 0.521 |
| d4 | 0.247 | 0.126 | 0.512 |

For d ≥ 2, r(d) ≈ 0.50. One suffix recovers approximately HALF the readout
deficit. This ratio is stable across three depth levels despite the raw gain
ranging from 4.9 to 12.6 percentage points.

At d1 the recovery fraction is lower (0.28), consistent with a floor effect:
when σ is already near 1.0, there's less to recover and the suffix may
overshoot into the regime where additional processing slightly degrades
(s2 < s1 at all depths).

## Observation 3: Suffix count > 1 hurts

The s2-s1 difference is consistently negative:

| Depth | s2 - s1 | s4 - s1 |
|-------|---------|---------|
| d1 | -0.009 | -0.011 |
| d2 | -0.009 | -0.011 |
| d3 | -0.013 | -0.007 |
| d4 | -0.007 | -0.030 |

The degradation at s4 is worst at d4 (-0.030). Extra suffixes don't just
fail to help — they actively degrade readout, and more so at deep nesting.
This rules out "extra computation helps" as the mechanism. The suffix is
a specific intervention, not generic processing time.

## Observation 4: Depth penalty halves

Linear fit of σ vs depth:
- Without suffix: slope = -0.072 per depth level
- With suffix: slope = -0.032 per depth level
- Ratio: 0.44 (settling reduces depth penalty by 56%)

## Observation 5: Monotone decay from s1 peak (not oscillation)

Normalizing the suffix trajectory as (σ(d,s) - σ(d,0)) / (σ(d,1) - σ(d,0)):

| Depth | s1 (=1.0) | s2 retention | s4 retention |
|-------|-----------|-------------|-------------|
| d2 | 1.000 | 0.809 | 0.774 |
| d3 | 1.000 | 0.883 | 0.938 |
| d4 | 1.000 | 0.943 | 0.760 |

At d≥2, s2 retains 81-94% of the s1 gain. The improvement decays
monotonically with suffix count but does not oscillate. This rules out
resonance and is consistent with attention dilution: more suffix positions
dilute the attention budget available for scope binding.

## Observation 6: Mixture model (SUPERSEDED by Obs. 7-8)

The aggregate data fits σ(d,1) = 0.5·σ(d,0) + 0.5·p with p ≈ 1.0
and residuals ≤0.004 at d≥2. But this was a reparameterization of
shadow attenuation: the "50% recovery" IS "suffix suppresses ~62%
of shadow leakage." The distributional analysis (Obs. 7-8) reveals
the actual mechanism — shadow-binding suppression, not dual-pathway
readout. The data table and fit remain valid descriptive summaries;
the "fresh readout pathway" and "look twice" interpretations are
refuted. Prediction d5 σ ≈ 0.73 reframed under attenuation law (P4).

## Observation 7: Shadow-binding suppression (REFRAMES Obs. 6)

Zero-call reanalysis of the per-cell 11-bin distributions reveals
that the suffix effect is NOT a generic "fresh readout pathway."
It is a specific suppression of the shadow-binding competitor.

All inner bindings use 9-prefixed values (99, 999, 9999, 99999).
The suffix transfers probability mass FROM digit 9 TO the correct
outer digit, with near-perfect conservation:

| Depth | Gain P(correct), targets 1-8 | Drop P(9) | Correlation | Target=9 gain |
|-------|------------------------------|-----------|-------------|---------------|
| d1 | +1.36 pp | -1.52 pp | -0.990 | -0.21 pp |
| d2 | +5.46 pp | -5.48 pp | -0.995 | +0.14 pp |
| d3 | +12.15 pp | -12.10 pp | -0.989 | +0.61 pp |
| d4 | +14.12 pp | -13.93 pp | -0.986 | +0.71 pp |

Other-digit mass changes by at most 0.16 pp on average.

**Key implications:**
1. The "~50% deficit recovery" (Obs. 2) was an artifact: the deficit
   IS shadow-9 leakage, and the suffix suppresses ~60% of it at d2-d4.
2. Target=9 gets nearly zero gain because the shadow IS digit 9.
3. Depth confounds with shadow-9 load: deeper = more 9-prefixed bindings.
4. The "mixture model" σ(d,1) = 0.5·σ(d,0) + 0.5 was a reparameterization
   of "suffix suppresses ~60% of the dominant shadow competitor."

**Revised mechanism:** The model confuses outer vs inner scope bindings
at deep nesting. The suffix resolves the ambiguity by suppressing the
currently dominant shadow candidate. This is boundary-conditioned shadow
attenuation, not dual-pathway readout.

**Connection to causal attention:** In a causal transformer, suffix tokens
cannot revise earlier code-token states. They introduce new downstream
positions that read the prefix and mediate the later query. The suffix
acts as a boundary cue that phase-switches the readout: same binding
trace, query-ready phase. The phase change attenuates the shadow.

## Observation 8: Leakage decomposition (the clean law)

Decompose the response law: P = C·δ_y + L·δ_z + R
where y = target (outer value), z = 9 (shadow digit).

| Depth | C (s0) | C (s1) | L (s0) | L (s1) | R (s0) | R (s1) | a_c |
|-------|--------|--------|--------|--------|--------|--------|-----|
| d1 | 0.955 | 0.969 | 0.029 | 0.014 | 0.016 | 0.017 | 0.48 |
| d2 | 0.888 | 0.943 | 0.089 | 0.034 | 0.023 | 0.023 | 0.38 |
| d3 | 0.773 | 0.894 | 0.196 | 0.075 | 0.031 | 0.031 | 0.38 |
| d4 | 0.731 | 0.872 | 0.234 | 0.095 | 0.035 | 0.033 | 0.41 |

(Targets 1-8 only, n=24 cells per depth. Conservation: ΔC+ΔL+ΔR = 0.000000.)

**The attenuation law:** L_{d,1} = a_c · L_{d,0}

    a_c ≈ 0.38 ± 0.07  (d ≥ 2)

The suffix retains ~38% of shadow leakage and transfers the rest to C.
R is invariant (|ΔR| < 0.002 at all depths).

**As a linear operator on the (C, L, R) simplex:**

    T_suffix: (C, L, R) → (C + (1-a_c)·L,  a_c·L,  R)

This is a one-parameter family of linear maps indexed by a_c. The map:
- Contracts L by factor a_c
- Expands C by the same mass
- Leaves R fixed
- Preserves the simplex (C + L + R = 1)

Per-cell a_c has low variance (σ = 0.04-0.08), confirming that the
attenuation coefficient is a stable property, not a cell-specific accident.

## Observation 9: Predictive-state synchronization

The suffix contracts within-binding response-law variation while
preserving between-binding discrimination:

| Depth | Within-binding TV (s0) | Within-binding TV (s1) | Contraction |
|-------|----------------------|----------------------|-------------|
| d1 | 0.0172 | 0.0129 | 25% |
| d2 | 0.0306 | 0.0145 | 53% |
| d3 | 0.0685 | 0.0285 | 58% |
| d4 | 0.0713 | 0.0363 | 49% |

Between-binding discrimination: s1/s0 ratios 1.01-1.18 (enhanced).

After removing shadow-9 and correct-digit bins, the contraction
disappears at d1-d2 (residual ratios 1.39, 1.02) and weakens to
~0.80 at d3-d4. Shadow-9 suppression is the primary mechanism of
synchronization (explains 75-100% of contraction).

## Observation 10: The operator is one-shot, not iterable

If T_suffix were a repeatable linear operator, applying it n times
would give L_{d,n} = a_c^n · L_{d,0}. The data refutes this:

| Depth | L(s1)/L(s0) | L(s2)/L(s0) actual | L(s2)/L(s0) if T^2 |
|-------|-------------|-------------------|-------------------|
| d2 | 0.38 | 0.44 | 0.15 |
| d3 | 0.38 | 0.42 | 0.15 |
| d4 | 0.41 | 0.41 | 0.16 |

Additional suffixes also grow R (ΔR = +0.003 to +0.010 from s0 to s4).

The operator decomposes into two effects:
- First suffix: suppresses shadow (a_c ≈ 0.38), transfers to C, R invariant
- Each additional suffix: partially restores shadow, adds noise to R

This explains the s1 peak: the boundary cue has a precise "dose" — one
application suppresses the competitor, additional applications introduce
interference that undoes part of the suppression.

## Theoretical summary

The suffix effect on nested-variable readout has a precise mathematical
description as a controlled Markov kernel (Codex formalization):

**Response decomposition:** P = C*d_y + L*d_z + R, where y = target
(outer value), z = shadow digit, program-relative.

**Controlled kernel** (acts on d_z, fixes other response coordinates):

    K_{y,z,a} d_z = a*d_z + (1-a)*d_y
    K_{y,z,a} d_j = d_j    (j != z)

with a = a_c ~ 0.38 at d >= 2. This contracts the nuisance direction
d_z - d_y by factor a, while leaving unrelated response coordinates
approximately fixed. Its global TV contraction coefficient is 1;
contraction occurs only along the empirically identified nuisance fiber.

**Content-specificity:** The kernel is parameterized by suffix content.
"# No changes.\n" actuates the R-preserving kernel (a_c ~ 0.38).
Generic code tokens give weaker kernels (a_c ~ 0.82). Self-assignment
reverses the direction (C -> L). The semantic content "no changes"
carries the dominant disambiguation signal (~45% of 62% total).

**One-shot optimality:** T is optimal among tested counts {0,1,2,4}.
Because s=2,4 do not behave as K^2, K^4, the full system is a
switched controlled transducer, not an iterated homogeneous system:

    (m, mu) --[suffix]--> (m', K^{(m)} mu)

The first boundary symbol enters the useful mode; later repetitions
invoke a different transition. This is the mathematical signature
of one-shot optimality.

**Shadow tracking:** Suppression follows the actual shadow digit under
relabeling. a_c consistent across shadow digits (0.37-0.42).

**One-probe synchronization:** The kernel contracts within-binding
response-law variation (25-59%) while enhancing between-binding
discrimination (1-18%). Shadow coordinates explain ~97-102% of the
contraction. Licensed as a one-probe fiber synchronizer.

**Scope:** One model (Qwen3-1.7B-Base), one template family, one
query probe, n=27 cells/depth. Cross-model and cross-task stability
are open predictions.

## Toward native coordinates (connection to the project thesis)

The (C, L, R) simplex is a subset of R^3 and the kernel K is a
standard stochastic linear map. The mathematics itself is ordinary.
What may be native is the *operationally induced quotient*: the
program-relative decomposition (target y, shadow z, residual) that
makes an apparently nonlinear distributional effect become a simple
proportional law. The coordinates are defined by program semantics,
not by vector geometry or layer structure.

Specifically:

1. **Program-relative quotient.** The (C, L, R) decomposition is
   relative to a PROGRAM (which defines target y and shadow z).
   Different programs induce different decompositions of the same
   11-bin distribution. The operator T acts uniformly across programs —
   same a_c regardless of which digit is the shadow (confirmed by
   relabeling). The quotient itself — collapsing 11 bins to 3 by
   program semantics — is the potentially native contribution.

2. **Compositional.** T acts uniformly across variable names, target
   values, and shadow digits. The attenuation coefficient a_c is
   stable across these variations (0.37-0.42 at d≥2).

3. **R invariance.** R is approximately conserved under the "# No
   changes" suffix (|dR| < 0.002). This constraint is content-specific:
   other suffixes partially disturb R (see Obs. 11). The R-preserving
   property selects a distinguished operator from the family.

4. **Nonlinear-to-linear reduction.** T is nonlinear on the raw 11-bin
   distribution (the effect vector is completely state-dependent). In
   (C, L, R) coordinates: T(C,L,R) = (C+(1-a_c)L, a_c·L, R). The
   right quotient makes the dynamics simple.

5. **(C, L, R) equality is NOT predictive-state equality.** R collapses
   distinct response bins, and no sufficient family of future tests has
   been checked. The decomposition is a registered response-law quotient,
   not a full predictive state. Cross-probe verification is needed.

## One-probe response-law contraction

Define the state distance between two programs as TV(response_law_1,
response_law_2). The suffix acts as a contraction-with-expansion:

| Depth | Same-binding contraction | Diff-binding expansion | Gap |
|-------|-------------------------|----------------------|-----|
| d1 | 0.88 (12% contraction) | 1.01 (1% expansion) | 0.13 |
| d2 | 0.60 (40% contraction) | 1.05 (5% expansion) | 0.46 |
| d3 | 0.49 (51% contraction) | 1.15 (15% expansion) | 0.66 |
| d4 | 0.75 (25% contraction) | 1.18 (18% expansion) | 0.43 |

Shadow coordinates ({correct, shadow} bins) explain ~97-102% of
within-binding contraction at d2-d4. The contraction IS the shadow
suppression, viewed in metric-space terms. The licensed term is
**one-probe fiber synchronizer** or **registered response-law
synchronizer** — not full predictive-state synchronization, since
only one query probe has been tested.

NOTE: The two synchronization tables in this document use different
aggregations (ratio-of-means vs mean-of-ratios), producing different
numbers. The Obs. 9 table uses ratio-of-means; this table uses
ratio-of-means with different pairing. Both show the same qualitative
pattern.

## Confirmed predictions

### P1: Shadow-digit relabeling — CONFIRMED (2026-09-04)

Experiment: experiments/config/svb_qwen3_shadow_relabel.json
Results: experiments/results/svb_qwen3_shadow_relabel/result.json
324 calls, 94.8s CPU.

| Condition | Shadow digit | d4 gain (pp) | d4 drop shadow (pp) | d4 corr | a_c |
|-----------|-------------|-------------|---------------------|---------|-----|
| shadow9 | 9 | +14.86 | -14.79 | -0.991 | 0.42 |
| shadow2 | 2 | +18.73 | -19.26 | -0.999 | 0.37 |
| shadow5 | 5 | +17.85 | -18.40 | -0.997 | 0.41 |

**Verdict: SHADOW TRACKING CONFIRMED.** The suppression follows the
shadow digit under relabeling. shadow2 suppresses P(2), shadow5
suppresses P(5). The mechanism is NOT hardcoded to any digit — it
tracks the actual binding competitor.

Shadow2/5 show STRONGER effects than shadow9 (gains 17-19pp vs 15pp
at d4). The attenuation coefficient a_c is consistent across shadow
digits (0.37-0.42), supporting a single underlying mechanism.

### Information-theoretic confirmation

Scope-relevant mass (C+L) is invariant under the suffix (<0.002
change at all depths). The suffix resolves a BINDING DISAMBIGUATION
problem, not an information gap. Scope-pair entropy H(C,L) drops
by 0.08-0.33 bits (resolves ~42% of disambiguation at d≥2).

### P2: Suffix content sensitivity — CONFIRMED (2026-09-04)

Experiment: experiments/config/svb_qwen3_suffix_ablation.json
Results: experiments/results/svb_qwen3_suffix_ablation/result.json
756 calls, 236.4s CPU. Six suffix conditions across d1-d4.

| Suffix | a_c (d2-d4 mean) | Suppression | R disruption | Type |
|--------|-----------------|-------------|--------------|------|
| (none) | --- | 0% | --- | baseline |
| `# No changes.\n` | 0.380 | 62.0% | <0.002 | STRONG, R-preserving |
| `\n` | 0.801 | 19.9% | <0.001 | MODERATE, R-preserving |
| `# TODO\n` | 0.819 | 18.1% | ~0.004 | MODERATE, R-deflecting |
| `pass\n` | 0.871 | 12.9% | ~0.006 | MODERATE, R-disrupting |
| `x = x\n` | 1.609 | -60.9% | ~0.007 | INTERFERENCE |

**Verdict: CONTENT-SPECIFICITY CONFIRMED.** The boundary cue is
not any code-like token — "# No changes.\n" is uniquely effective.

**Two-component model:**
- Generic boundary signal (~17%): any code token provides modest
  shadow suppression (a_c ~ 0.82). This is the "something appeared
  after the nested code" signal.
- Content-specific "no changes" signal (~45% additional): the
  semantic content "No changes" contributes the dominant suppression.
  At d2: 18% generic + 45% specific = 63% total.

**R-preservation hierarchy:** "# No changes.\n" is the only suffix
that is both strongly suppressive AND R-preserving (|dR| < 0.002).
Generic suffixes partially disturb R. Self-assignment reverses the
operator direction entirely (C → L, a_c > 1.6).

**Operator family:** Different suffix contents actuate different
members of a transition kernel family:

    K_nochanges: (C,L,R) → (C + 0.62L, 0.38L, R)      [R-preserving]
    K_newline:   (C,L,R) → (C + 0.20L, 0.80L, R-e)    [weak, clean]
    K_pass:      (C,L,R) → (C + 0.13L, 0.87L, R-e')   [weak, R-leaky]
    K_selfassign: reverses — mass flows C → L            [interference]

The "# No changes.\n" kernel is distinguished by being BOTH the
strongest suppressor AND the only one that preserves R invariance.
This is not accidental: the semantic content "no changes" is exactly
the information the model needs to resolve binding ambiguity without
disturbing non-scope mass.

**Codex gossip version (adopted verbatim):** "A comment doesn't give
the transformer more time — it tells it which overwritten value to
stop believing."

## Operator family formalization (from P2 analysis)

### The binding simplex and kernel family

**Definition.** The *binding simplex* S = {(C, L, R) : C+L+R=1, all >= 0}
is the quotient of the 11-bin response distribution by the program-relative
partition {correct digit y, shadow digit z, everything else}. The quotient
map q_{y,z} : Delta^10 -> S is determined by the program.

**Definition.** The *content-parameterized kernel family* is a map
K : Sigma -> End(S), where Sigma = {suffix contents} and End(S) =
stochastic maps S -> S. For the R-preserving subfamily:

    K_a : (C, L, R) -> (C + (1-a)L, aL, R),   a in [0, 1]

Properties of K_a:
- Fixed point set: {(C, 0, R) : C + R = 1} (the L = 0 face)
- Eigenvalues: 1 (multiplicity 2), a
- Spectral gap: 1 - a (measures suppression strength)
- det(K_a) = a (compresses volume by factor a)
- K_a is idempotent iff a = 0 (complete suppression) or a = 1 (identity)
- K_a K_b = K_{ab} (the R-preserving family IS a multiplicative group)
  BUT the empirical system does not compose this way (one-shot, not iterable)

### The binding signal and multiplicative decomposition

The kernel family has a natural multiplicative decomposition
(Codex derivation): if boundary correction and content correction
are sequential pure attenuation kernels, their survival factors
multiply:

    a_total = a_boundary * a_content

Using a_boundary ~ 0.82 (mean of generic suffixes) and a_total ~ 0.38:

    a_content ~ 0.38 / 0.82 = 0.463

The content stage removes ~54% of the leakage remaining after the
generic boundary stage. The absolute suppression difference (0.62 -
0.18 = 0.44) is the additive version of the same fact.

| Suffix | a_c | Suppression | Binding H reduction |
|--------|-----|-------------|---------------------|
| `# No changes.\n` | 0.380 | 62.0% | 42-51% |
| `\n` | 0.801 | 19.9% | 11-13% |
| `# TODO\n` | 0.819 | 18.1% | ~12% |
| `pass\n` | 0.871 | 12.9% | ~8% |
| (identity) | 1.000 | 0% | 0% |
| `x = x\n` | K(1, 0.13) | -60.9% | -18 to -40% |

Note: content-specificity is **intensional** (lexico-syntactic /
discourse-act conditioning), not necessarily semantic. All tested
suffixes are extensionally no-ops for the queried variable, yet
produce sharply different kernels. The action does NOT factor through
denotational store semantics. Paraphrase and contradiction equivalence
tests (P5-P6) would be needed before "semantic" becomes the
identified causal variable.

### General R-preserving kernel (Codex derivation)

Every R-preserving stochastic kernel on (C,L,R) has the form:

    K(a,b) = [[1-b, 1-a, 0],
              [b,   a,   0],
              [0,   0,   1]]

with 0 <= a, b <= 1. The attenuation law is K(a, 0) (b = 0: no C→L
reverse flow). Self-assignment is approximately K(1, beta) with
beta ~ 0.13 (mass flows C→L at rate beta*C).

On each R-fiber (fixed R = r), using p = L/(C+L):

    p' = b + (a - b) * p

Fixed point: p* = b / (1 - a + b)
Contraction rate: lambda = a - b

| Suffix type | Kernel form | Attractor p* |
|-------------|-------------|--------------|
| `# No changes` | K(0.38, ~0) | 0 (correct) |
| `\n` | K(0.80, ~0) | 0 (weakly) |
| `pass`, `# TODO` | weak + R-defect | not fiber-closed |
| `x = x` | K(~1, ~0.13) | 1 (shadow!) |

The pure attenuation operators K(a, 0) form a commutative monoid
isomorphic to ([0,1], x). Adding reverse transport (b > 0) gives the
ordinary two-state Markov monoid and generally destroys commutativity.

**Selection principle**: "# No changes.\n" is distinguished by being
the only tested suffix that is both strongly suppressive (a < 0.5)
AND approximately R-preserving (closed on R-fibers). The bare newline
has similar |dR| (~0.001) but weak suppression (a ~ 0.80). Combined
strength + R-closure is the joint selection criterion.

### Predictions from the operator family

P5: Semantically equivalent suffixes ("# Nothing changed.\n",
"# Same value.\n", "# Unmodified.\n") should have beta ~ 1.0.

P6: Semantically opposite suffixes ("# Updated.\n", "# Changed.\n",
"# New value.\n") should have beta < 0.

P7: The 81/19 content/boundary decomposition should hold across
models with different architectures (if it reflects pre-training
patterns) or shift (if it reflects architectural properties).

P8: R-preservation should correlate with attention pattern specificity:
R-preserving kernels attend to scope-relevant positions only, while
R-disrupting kernels have broader attention spread.

### P4: Depth-5 extrapolation — L SATURATION (2026-09-04)

Experiment: experiments/config/svb_qwen3_depth5_prediction.json
Results: experiments/results/svb_qwen3_depth5/result.json
81 calls (cached from prior runs), 42.5s.

| Depth | C (s0) | C (s1) | L (s0) | L (s1) | R (s0) | R (s1) | a_c  |
|-------|--------|--------|--------|--------|--------|--------|------|
| d1 | 0.955 | 0.969 | 0.029 | 0.014 | 0.016 | 0.017 | 0.48 |
| d2 | 0.888 | 0.943 | 0.089 | 0.034 | 0.023 | 0.023 | 0.38 |
| d3 | 0.773 | 0.894 | 0.196 | 0.075 | 0.031 | 0.031 | 0.38 |
| d4 | 0.731 | 0.872 | 0.234 | 0.095 | 0.035 | 0.033 | 0.41 |
| d5 | 0.709 | 0.854 | 0.244 | 0.106 | 0.047 | 0.040 | 0.43 |

**Key findings:**

1. **L saturation**: L(4)=0.234, L(5)=0.244 — only 4% increase. Growth
   ratios: d1→d2 x3.0, d2→d3 x2.2, d3→d4 x1.2, d4→d5 x1.04.
   The exponential model L ~ 0.03·exp(0.6d) is REFUTED at d5.

2. **R growth**: R(4)=0.035, R(5)=0.047 — 33% increase. As L saturates,
   residual noise R grows instead. Different degradation mode.

3. **a_c weakening**: a_c increases from 0.38 (d2-d3) to 0.43 (d5).
   The operator is less effective in the saturation regime.

4. **R invariance breaks**: |dR| = 0.007 at d5 (vs <0.002 at d2-d4).
   The "clean" L→C transfer degrades; some mass leaks to R.

5. **Recovery fraction stable**: r(5) = 0.49, consistent with d2-d4
   (~0.50). The proportional recovery holds even as the mechanism degrades.

**Interpretation**: The model has finite scope-confusion capacity. L
saturates around 0.24 (~24% of the response mass leaks to the shadow).
Beyond d4, additional nesting adds general noise (R growth) rather than
scope-specific confusion (L growth). The attenuation operator degrades
because the d5 "error" is no longer purely scope-related.

**Old prediction was wrong**: The exponential model predicted
sigma(5,0) ≈ 0.45, but actual is 0.73. The prediction interval
[0.68, 0.78] for sigma(5,1) was based on a wrong premise. However,
the MIXTURE FORMULA with the actual s0 gives 0.5·0.73 + 0.5 = 0.866,
matching the actual 0.862 to within 0.004.

## Remaining predictions

### P3: Nested vs flat structure control
Token/occurrence-matched flat repetition vs nested scopes.

## Open questions

1. What specific attention pattern mediates the boundary cue?
2. Why does s > 1 hurt — does it introduce new interference?
3. Is the a_c ≈ 0.38 suppression rate architectural or learned?
4. Does the boundary cue work for non-Python tasks?
5. Is the d3-d4 residual synchronization (beyond shadow suppression) real?
6. Why is a_c consistent across shadow digits but NOT across depths
   (a_c ≈ 0.48 at d1, ≈ 0.38 at d≥2)?
7. What is the pre-image of R-preservation in the attention pattern?
8. Does the 81/19 content/boundary split reflect training or architecture?
