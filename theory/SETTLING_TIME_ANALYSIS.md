# Settling Time: Functional Form Analysis

Status: EMPIRICAL OBSERVATION (one model, one template family, n=27/cell)
Data: experiments/results/svb_qwen3_formal/result.json
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

## Open questions

1. Is the ~50% recovery fraction a coincidence of this template/model,
   or does it hold across template families and models?
2. Does the exponential depth decay extend to d5+? (Would require
   deeper nesting templates.)
3. What is the mechanism by which one suffix halves the readout error?
   Candidates: attention redistribution, hidden-state realignment,
   position-dependent readout recalibration.
4. Why does s > 1 hurt? Is the extra suffix pushing the hidden state
   past the optimal readout configuration?
5. Is there a suffix OTHER than `# No changes.` that recovers more or
   less than 50%? If different suffixes recover different fractions,
   the fraction is suffix-dependent, not a depth property.
