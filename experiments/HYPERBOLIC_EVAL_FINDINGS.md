# Hyperbolic Evolution Evaluation Findings

**Date:** 2026-02-03
**Status:** Phase 5 Complete - Pivot Required

## Key Findings

### 1. Hyperbolic Geometry Successfully Maintains Diversity

| Metric | Euclidean | Hyperbolic | Ratio |
|--------|-----------|------------|-------|
| Survivors | 2 | 5-7 | 3.5x |
| Diversity | 0.006 | 2.32 | 405x |
| Stop Reason | patience | max_generations | - |

**Conclusion:** Hyperbolic geometry in the Poincaré ball model successfully prevents population collapse during evolution. This is the core hypothesis CONFIRMED.

### 2. BUT: Output Quality Shows No Difference

LLM-as-judge evaluation (Claude subagent) found:
- Euclidean wins: 2 prompts (slight margin)
- Hyperbolic wins: 2 prompts (slight margin)
- Ties: 1 prompt
- **Verdict: "Difference within noise margin"**

All outputs were truncated mid-thought. Both geometries produce similar reasoning patterns.

### 3. CRITICAL: Latent Scorer Has ~0 Correlation

From checkpoint analysis:
```
val_corr: -0.031
```

The latent scorer essentially produces random scores. The LLM-as-judge confirmed:
- Automated scores inflate quality by 0.15-0.25 points
- Scorer rewards "verbose thinking patterns" not correctness
- Wrong answers get similar scores to correct ones

### 4. Root Cause Analysis

The evolution loop is **blind** because:
1. The latent scorer was trained on data with poor quality labels
2. It learned to recognize "reasoning style" not "reasoning quality"
3. Selection pressure is essentially noise
4. Hyperbolic diversity helps maintain exploration but can't find better solutions without signal

## Codex Senior Engineer Assessment

> "Promising as a *mechanism for maintaining exploration*, but currently a dead end for *end-quality* until your evaluation signal improves. Right now you've proven 'hyperbolic keeps diversity,' not 'hyperbolic improves reasoning.'"

### Codex Recommendations (Priority Order):
1. **(a) Improve latent scorer** - Most urgent, evolution is blind without it
2. **(d) Add format compliance + post-hoc evaluator** - Verify actual quality
3. **(c) Curvature tuning** - Only matters after scorer is fixed
4. **(b) Decoder improvements** - Won't help if selection is broken

## Technical Details

### Hyperbolic Implementation (Working)
- Poincaré ball model with curvature c=1.0
- logmap0/expmap0 for tangent space operations
- Karcher mean for crossover
- Hyperbolic distance for diversity computation
- Proper tangent→ball→tangent mapping in decoder (fixed bug)

### Scorer Issues
- 12,118 training samples, 1,346 validation
- Trained with MSE loss on Gemini-generated scores
- -0.031 correlation = worse than random
- Likely reasons: noisy labels, style overfitting

## Next Steps

### Phase 6: Fix Latent Scorer
Options:
1. Generate new training data with stronger LLM judge
2. Train on completeness + correctness signals
3. Use self-consistency (model rates its own outputs)

### Phase 7: Add Format Compliance
- Ensure `<think>` tags are always present
- Detect incomplete outputs
- Add completion penalty

### Phase 8: Re-evaluate with Fixed Scorer
- Controlled ablation: same seeds, same evaluator
- Compare final output quality not just diversity
- Statistical significance testing

## Conclusion

**Hyperbolic geometry for latent evolution: Mechanism works, signal broken.**

The geometry successfully maintains population diversity during evolution - this is valuable. But without a reliable scoring signal, diversity doesn't translate to quality improvements. The pivot to fixing the scorer is the correct next step.
