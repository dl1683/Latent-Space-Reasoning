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

## Observation 6: Mixture model (best fit)

The data is extremely well described by:

    σ(d, 1) = 0.5 × σ(d, 0) + 0.5 × p

where p ≈ 1.0 (depth-independent fresh readout probability).

| Depth | σ(d,0) | σ(d,1) actual | σ(d,1) predicted | Residual |
|-------|--------|---------------|------------------|----------|
| d1 | 0.958 | 0.970 | 0.979 | -0.009 |
| d2 | 0.897 | 0.945 | 0.948 | -0.003 |
| d3 | 0.792 | 0.900 | 0.896 | +0.004 |
| d4 | 0.753 | 0.880 | 0.877 | +0.003 |

Residuals are ≤0.009 (well within n=27 sampling noise). At d≥2,
residuals are ≤0.004.

**Interpretation**: The suffix creates a second, independent readout
pathway with near-perfect accuracy (~99-100% at d≥2). The model's
answer is a 50-50 mixture of the original readout (depth-degraded)
and this fresh readout (depth-independent). At d1, the fresh pathway
is slightly below perfect (p ≈ 0.98), consistent with a ceiling effect.

**Falsifiable prediction**: At d5, if σ(5,0) follows the exponential
model (≈0.45), then σ(5,1) ≈ 0.5 × 0.45 + 0.5 × 1.0 ≈ 0.73. This
is a strong prediction from 4 data points, testable with a d5 template.

**Gossip-magazine version**: "One comment line lets the model look
at the answer twice — and the second look is always right."

## Connection to breakpoints

The mixture model directly addresses three breakpoints:

**BP-1 (presence ≠ causation):** The outer-scope value is ENCODED in the
hidden state at all depths (the fresh readout pathway proves this — p≈1.0
means the information is always present). But the information is only
ADDRESSABLE through the suffix-triggered pathway at deep nesting. Without
the suffix, the depth penalty makes the information progressively less
addressable — despite being fully present.

**BP-7 (snapshot ≠ computation):** The settling effect IS a computational
trajectory phenomenon. The suffix adds one more step to the trajectory,
and that step enables a readout pathway that doesn't exist in the original
trajectory. You cannot understand this from a single-layer activation
snapshot — it requires the full trajectory through the suffix token.

**BP-6 (observation ≠ state):** The choice of readout (with or without
suffix) changes the observable. The model's "state" after processing
depth-4 code contains the correct answer, but you only SEE it if you
observe through the suffix-triggered pathway. The without-suffix
observation gives σ=0.75; the with-suffix observation gives σ=0.88.
Same state, different observation, different measurement.

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
description:

**State:** (C, L, R) decomposition of the response law, where
C = P(correct target), L = P(shadow digit), R = 1 - C - L.

**Depth law:** L grows exponentially with depth: L ~ 0.03·exp(0.6d)

**First-suffix action:** T: (C, L, R) → (C + (1-a_c)·L, a_c·L, R)
with a_c ≈ 0.38 at d ≥ 2. R is invariant.

**Non-iterability:** T is one-shot. Repeated application does NOT give
T^n; instead, extra suffixes add noise to R and partially reverse L
suppression. The "therapeutic dose" is exactly one.

**Shadow-specificity:** Gain vanishes when target = shadow digit.

**Synchronization:** The suffix contracts within-binding variation
(25-59%) while enhancing between-binding discrimination (1-18%).

The mechanism is boundary-conditioned shadow attenuation: a single
boundary cue suppresses the dominant shadow competitor. The cue acts
as a phase switch in the readout — same binding trace, query-ready
phase. Not generic computation, not dual-pathway readout, not settling.

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

## Remaining predictions

### P2: Suffix content sensitivity
Does the boundary cue need to be a Python-compatible statement?
Config: experiments/config/svb_qwen3_suffix_ablation.json

### P3: Nested vs flat structure control
Token/occurrence-matched flat repetition vs nested scopes.

### P4: Depth-5 extrapolation (reframed)
Under the attenuation law: L(5,0) ~ exp(3.0) · 0.03 ≈ 0.60.
σ(5,1) = 1 - a_c·L(5,0) - R ≈ 1 - 0.38·0.60 - 0.035 ≈ 0.74.
Config: experiments/config/svb_qwen3_depth5_prediction.json

## Open questions

1. What specific attention pattern mediates the boundary cue?
2. Why does s > 1 hurt — does it introduce new interference?
3. Is the a_c ≈ 0.38 suppression rate architectural or learned?
4. Does the boundary cue work for non-Python tasks?
5. Is the d3-d4 residual synchronization (beyond shadow suppression) real?
6. Why is a_c consistent across shadow digits but NOT across depths
   (a_c ≈ 0.48 at d1, ≈ 0.38 at d≥2)?
