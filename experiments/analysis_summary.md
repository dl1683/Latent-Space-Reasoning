# Analysis Summary: Perturbation-Gated Reasoning Mode Selection
Date: 2026-03-05 (Tesla Workflow Deep Analysis)

## 1. Dose-Response Curve (As of 2026-03-05)

| Tokens | Mode | Mean Acc | Std | n_lat | Delta |
|--------|------|----------|-----|-------|-------|
| 0 | baseline | 32.0% | - | - | - |
| 1 | random_noise | 42.7% | 1.9% | 3 | +10.7pp |
| **2** | **random_noise** | **60.0%** | **0.0%** | **3** | **+28.0pp** |
| 3 | random_noise | 44.0% | 0.0% | 3+ (RUNNING 10) | +12.0pp |
| 8 | random_noise | 44.0% | 2.8% | 4 | +12.0pp |
| 8 | latent_projected | 44.4% | 7.0% | 10 | +12.4pp |
| 8 | zero_embedding | 36.0% | 0.0% | 3 | +4.0pp |
| 8 | mean_embedding | 36.0% | 0.0% | 1 | +4.0pp |

## 2. CRITICAL CORRECTION: "Zero Variance" Is Misleading (2026-03-05)

The 2-token "zero variance" (std=0.0%) means all 3 latents get **the same accuracy** (60%), but they solve **DIFFERENT TASKS**:

| Metric | Value |
|--------|-------|
| Unanimous correct (all 3 latents) | 9/25 tasks |
| Disagreement (1 or 2 latents correct) | 13/25 tasks |
| Stuck (all wrong) | 3/25 tasks |
| Oracle (any 1 of 3 correct) | 22/25 = **88%** |
| Majority vote (2/3) | 14/25 = 56% |
| Fleiss kappa | 0.278 (fair agreement) |
| Permutation test p-value | 0.024 (more correlated than chance) |
| Cochran Q | 0.0, p=1.0 (marginals identical, MISLEADING) |

**Correct interpretation**: Direction does NOT affect how many tasks are solved, but it DOES affect which tasks are solved. Different random perturbations open different "reasoning channels."

### Answer magnitude predicts task category:
- Unanimous correct: mean |answer| = 599
- Disagreement: mean |answer| = 2619
- Stuck: mean |answer| = 7092

## 3. Task-Level Recovery (2-token vs baseline)

- **14/17** baseline failures at least partially recovered (by at least 1 of 3 latents)
- **2** full regressions at per-latent level (nest_007, nest_014 baseline-correct but not all latents correct)
- **3** stuck tasks: nest_005 (8360), nest_008 (7278), nest_021 (5639)
- All stuck tasks involve multiple large two-digit multiplications

## 4. Complexity Predictors

| Feature | r with recovery | p-value direction |
|---------|-----------------|-------------------|
| max_operand | **-0.805** | Strong negative |
| |answer| | -0.620 | Moderate negative |
| n_digits | -0.551 | Moderate negative |

## 5. Lyapunov-Like Invariance Length Analysis

Under greedy decoding (T=0, do_sample=False), different noise vectors produce
byte-identical text for a "invariance length" that depends on perturbation energy:

| Tok | n_lat | Mean Prefix | Identical% | Unanimous% | Oracle | MajVot | Acc |
|-----|-------|-------------|-----------|------------|--------|--------|-----|
| 1 | 3 | 349 chars | 44% | 56% | 68% | 36% | 42.7% |
| 2 | 3 | 323 chars | 35% | 48% | 88% | 56% | 60.0% |
| 8 | 10 | 93 chars | 6% | 20% | 92% | 32% | 44.4% |

**Key discovery**: This is DETERMINISTIC CHAOS, not sampling noise (T=0).
- 5 disagreement tasks have 500/500 chars byte-identical across all 3 latent pairs
- Yet different correctness outcomes (divergence after stored text)
- More tokens = more perturbation energy = shorter invariance = earlier bifurcation
- 2-token sweet spot: late divergence preserves reasoning quality + high oracle coverage

## 6. Force-Think Decomposition (CONFIRMED 2026-03-05)

**Force-think baseline: 40.0% (10/25)** -- prediction confirmed!

| Condition | Think Rate | Accuracy | Component |
|-----------|-----------|---------|-----------|
| Baseline | 16% (4/25) | 32.0% | -- |
| Force-think | 100% (forced) | 40.0% | Think mode = +8pp |
| Zero embedding | 100% (25/25) | 36.0% | Think + zero positions = +4pp |
| 1-tok noise | 100% (25/25) | 42.7% | Think + 1 random = +10.7pp |
| 2-tok noise | 100% (25/25) | 60.0% | Think + optimal noise = +28pp |
| 8-tok noise | ~100% | 44.0% | Think + excess noise = +12pp |

**Clean decomposition at 2 tokens**:
- Think mode alone: +8pp (32% -> 40%)
- Noise perturbation beyond think: +20pp (40% -> 60%)
- **Noise contributes 2.5x more than think mode activation**

**Force-think task-level changes vs normal baseline**:
- Gained: nest_002 (6193), nest_015 (7693), nest_016 (10), nest_018 (4)
- Lost: nest_007 (1044), nest_014 (20) -- overthinking regression

**Zero embedding determinism**: 25/25 byte-identical responses across 3 latents (no chaos).
Zero vs force-think NOT statistically significant (10/25 vs 9/25, p >> 0.3 at n=25).

**Codex caution**: Three-component model (think gate + positional perturbation + stochastic
diversity), not simple two-component. Need n=100 to distinguish force-think from zero.

## 7a. Infrastructure Confound (RESOLVED)

- `decode_with_raw_soft_prompt` preserves `<think>` tags
- `run_zero_shot` and `decode_latent` STRIP `<think>` tags
- Fixed: now stores `response_raw` alongside stripped `response` (2000 chars)
- Baseline: 16% think mode (4/25), 2-token: 100% think mode (25/25)

## 7b. Timing Confound (UNDER INVESTIGATION, Codex Analysis 2026-03-05)

**The 2-tok condition is 1.6x slower than all other conditions:**

| Condition | Mean time/task | Total (3 latents) |
|-----------|-----------------|-------------------|
| Baseline | 74.0s | 1851s |
| 1-tok noise | 73.1s | 5481s |
| 2-tok noise | **118.4s** | **8883s** |
| Zero embedding | 72.1s | 5408s |
| 8-tok latent | 75.1s | 18769s (10 latents) |

- 73% of excess 2-tok wall-clock concentrated in 5 tasks (nest_013 to nest_017)
- nest_017 latent 2: 1151s (19 min!) — not explainable by 1024-token cap (~80s)
- Slowdown is specific to 2-tok noise, NOT think mode or positions generally
- Slow != correct: nest_015 appears at 73s, 378s, 82s with mixed correctness
- **Codex interpretation**: 2-tok noise induces longer, less stable rollouts;
  part of the accuracy advantage may be mediated by extra compute, not just
  better reasoning channel selection

**Critical instrumentation gap** (fixed in harness):
- Added `generated_tokens`, `prompt_tokens`, `terminated_by_eos`, `tokens_per_sec`
  to distinguish "more tokens" from "slower tokens"

**Required experiment**: Downward max_new_tokens sweep (128, 256, 512, 1024)
with force-think baseline, 2-tok zero, 2-tok noise conditions.

**3-tok timing (preliminary, latent 1)**: Normal range (34-93s), no outliers.
If 3-tok accuracy ~45-50% with normal timing, materially weakens compute-time story.

### TIMING CONFOUND RESOLUTION (2026-03-05)

The timing anomaly is a **single-latent artifact**, NOT systematic:

| Latent | Accuracy | Mean time | Max time | Outliers (>200s) |
|--------|----------|-----------|----------|-------------------|
| 0 | 60% | 73.6s | 94.2s | 0 |
| 1 | 60% | 189.7s | 1151.3s | 5 |
| 2 | 60% | 92.0s | 129.0s | 0 |

- Point-biserial r(time, correct) = -0.206 (slow runs tend to be WRONG)
- Latent 0 achieves 60% at 73.6s mean = baseline-identical timing
- All three latents achieve the SAME accuracy despite wildly different timing
- **The 2-tok advantage is NOT from extra computation time**
- Budget sweep downgraded from "critical" to "reviewer-facing control" (Codex)

**Paper-ready claim (Codex-approved, narrow)**:
"At least one 2-token perturbation direction achieves the full 60% accuracy at
baseline-matched latency (73.6s vs 74.0s baseline), while slower runs are not
more accurate (point-biserial r = -0.206)."

## 7c. Theoretical Model: PGRMS (Updated)

**Perturbation-Gated Reasoning Mode Selection**
- Random prefix tokens ACTIVATE think mode via energy perturbation
- Direction steers WHICH reasoning channels are activated (not how many)
- At 2 tokens: optimal energy, all directions yield same accuracy but different task sets
- At 8 tokens: excessive energy, direction-dependent accuracy AND tasks
- Oracle ceiling suggests 88% of tasks are solvable with diverse perturbations

## 8. Operation Type Stratification

| Category | n | Base | 1-tok | 2-tok | 8-tok |
|----------|---|------|-------|-------|-------|
| Mod (<50) | 7 | 14.3% | 23.8% | **66.7%** | 34.3% |
| Small (<=1k) | 14 | 35.7% | 54.8% | **76.2%** | 57.1% |
| Medium (1k-5k) | 3 | **100%** | 55.6% | 77.8% | 50.0% |
| Large (>5k) | 8 | 0.0% | 16.7% | 25.0% | 20.0% |

Key: Modular arithmetic benefits most (+52pp at 2-tok). Medium tasks REGRESS (overthinking).

## 8b. Sensitive Task Analysis (STRICT categorization, Codex 2026-03-05)

**Strict definition**: Always-solved = correct by baseline AND every latent in every condition.
Never-solved = wrong by baseline AND every latent in every condition.

- Always solved (strict): 2/25 (nest_001, nest_024)
- Never solved (strict): 1/25 (nest_008, answer=7278)
- Sensitive: 22/25 tasks
- Full cross-condition oracle: 24/25 = **96%** (only nest_008 unsolvable)

| Condition | k lat | Counts/lat | Std | Oracle | Unsolved |
|-----------|-------|-----------|-----|--------|----------|
| Baseline | 1 | [6] | - | 6/22=27.3% | 16 tasks |
| 1-tok | 3 | [8,9,9] | 0.47 | 15/22=68.2% | 7 tasks |
| **2-tok** | **3** | **[13,13,13]** | **0.00** | **21/22=95.5%** | **1 task** |
| 8-tok | 10 | [7..12] | 1.76 | 21/22=95.5% | 1 task |
| Zero | 3 | [7,7,7] | 0.00 | 7/22=31.8% | 15 tasks |

**Key findings:**
- 2-tok and 8-tok reach the SAME oracle ceiling (95.5%), same missed task (nest_005)
- 2-tok does it with 3 directions; 8-tok needs 10 (3x more efficient)
- 2-tok equalizes per-direction count (std=0.0); 8-tok varies widely (std=1.76)
- Zero embedding also shows equalization (std=0.0) but at much lower level (7/22)
- **Task-specific resonance windows**: nest_005 (8360) ONLY solved by 1-tok; nest_021 (5639) ONLY solved by 8-tok
  - This shows non-monotonicity at the task level in both directions
- McNemar per-latent (2-tok vs baseline):
  - Latent 0: 7 gains, 0 losses (chi2=5.14, p≈0.023) — purely additive
  - Latent 1: 9 gains, 2 losses (chi2=3.27)
  - Latent 2: 8 gains, 1 losses (chi2=4.00)
- Effect size: Cohen's h = 0.570 (medium-large)

### Coarsely-stable categorization (supplement-only)
Previous looser definition (solved by baseline + all 2-tok latents): 5 always-solved, 2 never-solved, 18 sensitive.
Use "coarsely stable" label, not "always/never", per Codex guidance.

## 8b. Data Integrity Issue: 500-Char Response Truncation (2026-03-05)

**All pre-3-tok experiments** store responses truncated at 500 chars (`resp[:500]`).
The current code stores `resp[:2000]` + `response_raw[:2000]`.

**Impact on scoring**: verify_answer() was called on FULL responses at runtime, so the stored
`correct` flags are based on full generation outputs. The stored responses are for inspection
only and cannot independently verify the scores.

| Experiment | Mismatches | Total | % Unverifiable |
|-----------|-----------|-------|----------------|
| mean_embedding | 12 (3 base + 9 noise) | 36% | includes nest_008 |
| zero_embedding | 30 (3 base + 27 noise) | 36% | |
| latent_8tok | 113 (3 base + 110 noise) | 44% | |
| noise_2tok | 49 (3 base + 46 noise) | 61% | |
| baseline | 3/25 | 12% | shared across files |

**Mismatch direction** (verify_stored_data.py audit, 2026-03-05):
- 199/204 mismatches: stored=True, replayed=False (answer truncated past 500 chars)
- 5/204 mismatches: stored=False, replayed=True (all nest_018, expected=4, tiny answer
  accidentally appears in truncated text; runtime found different final number in full response)

**Codex assessment (2026-03-05)**: Not a hard submission blocker if claims stay conservative.
Re-run priority: mean_embedding (decides 24/25 vs 25/25) -> noise_2tok (main result) ->
latent_8tok (oracle comparison arm). Full reruns before camera-ready.

**Hardening (implemented)**: Current code now stores `extracted_answer` field alongside
`correct` flag, capturing the exact number verify_answer() matched against.

**Assessment**: This is a REPRODUCIBILITY issue, not a correctness issue. The accuracy numbers
(32%, 42.7%, 60%, 44%, 36%) are based on runtime full-response scoring and are likely correct.
However, reviewers cannot independently verify from stored data. The 3-tok experiment (running now)
uses the fixed code. All experiments should eventually be re-run with current code for publication.

## 9. Paper Contributions (Revised Ranking, Codex 2026-03-05, updated 2026-03-05c)

### Codex Equalization Review (2026-03-05)
- **Paper framing**: Solve-count equalization = main SCIENTIFIC INSIGHT; oracle efficiency = main EMPIRICAL PAYOFF
- **Terminology**: Per-perturbation categories → `unanimous`/`frozen`/`sensitive` (not `always`/`never` to avoid collision with cross-condition strict definition)
- **Naming**: Paper term = "solve-count equalization", interpretive = "fixed-capacity regime"
- **8-tok mechanism**: Regime boundary. Moderate perturbation selects among equal channels; over-perturbation makes channel quality direction-dependent
- **Paper sentence (Codex-approved)**: "At fixed perturbation magnitude, random directions enter an equalized solve-count regime: each direction solves the same number of tasks, but not the same tasks. Oracle gains arise by covering the perturbation-sensitive subset."

1. **MAIN SCIENTIFIC INSIGHT**: Solve-count equalization (Codex: "fixed-capacity regime")
   - Confirmed at 2-tok [15,15,15] and 3-tok [11,11,11], both std=0.00
   - P=0.004 under independence for 3-tok
   - Structural decomposition: unanimous/frozen/sensitive explains mechanism
2. **MAIN EMPIRICAL PAYOFF**: Oracle efficiency / coverage-vs-budget curve (Codex: "strongest claim")
   - 2-tok: 88% oracle with 3 runs; 8-tok: 92% oracle with 10 runs (3x more efficient)
   - 25/25 combined oracle as capstone endpoint (CI: [0.86, 1.0])
3. **STRONG**: Direction changes WHICH tasks, not HOW MANY (equalization at 2-tok, std=0.0)
   - "Constant count, different support" — task redistribution without marginal change
3. **STRONG**: Noise contributes 2.5x more than think mode (force-think decomposition)
4. **STRONG**: Deterministic chaos under greedy decoding (T=0, different outputs from noise)
5. **STRONG**: Condition-specific rescue windows (Shapley: 3 families each uniquely rescue 1 task)
   - noise_1tok: nest_005; latent_8tok: nest_021; mean_8tok: nest_008 (revalidate)
6. **STRONG**: Timing confound resolved — latent 0 achieves 60% at baseline timing (73.6s)
7. **STRONG**: Non-monotonic dose-response CONFIRMED: 3-tok N1=N2=44%, sharp drop from 2-tok 60%
   - 3-tok equalization: both latents solve EXACTLY 11/25 (std=0.00, P=2.5% under independence)
   - 3-tok solve set entirely subset of 2-tok oracle (adds 0 new tasks)
8. **STRONG**: Pooled sign test: 24 gains / 3 losses across 3 latents, p=0.000049
   - Per-latent McNemar exact: L0 p=0.016, L2 p=0.039 (significant)
   - Fisher exact pooled: OR=3.19, p=0.021
   - Cohen's h=0.570 (medium effect)
9. **MODERATE**: Large-answer regression (100% -> 78%) = policy switch, not generic boost
10. **WEAK**: Answer-magnitude Spearman on delta accuracy: r=-0.295, p=0.15 (NS)
    - Raw accuracy correlation is confounded by baseline difficulty

## 10. Cross-Experiment Oracle Analysis (2026-03-05)

### Combined Oracle = 24/25 (96%) VERIFIED, 25/25 UNVERIFIABLE

Every task is solvable by at least one perturbation condition — but nest_008 (mean_8tok) has
a scoring bug (response truncated at 500 chars, stored `correct: True` is unverifiable).
Conservative claim: 24/25 = 96%. Need to re-run mean_embedding with current code.

**Total perturbation budget**: 20 runs across 6 condition types (3×1tok + 3×2tok + 10×8tok-latent + 3×zero + 1×mean).

### Oracle Scaling Efficiency

| Group | k latents | Oracle | Coverage |
|-------|-----------|--------|----------|
| noise_2tok | 1 | 15/25 | 60% |
| noise_2tok | 2 | 20/25 | 80% |
| noise_2tok | 3 | 22/25 | 88% |
| latent_8tok | 1 | 9/25 | 36% |
| latent_8tok | 3 | 15/25 | 60% |
| latent_8tok | 5 | 19/25 | 76% |
| latent_8tok | 10 | 23/25 | 92% |

**2-tok is ~3x more oracle-efficient than 8-tok** (3 directions → 88% vs 10 directions → 92%).

### Unique Solvers (tasks solvable by ONLY one condition group)

| Task | Answer | Unique solver | Solve rate across all 20 runs |
|------|--------|--------------|------------------------------|
| nest_005 | 8360 | noise_1tok (1/3 latents) | 5% |
| nest_008 | 7278 | mean_8tok (1/1) | 5% |
| nest_021 | 5639 | latent_8tok (2/10 latents) | 10% |

All unique-solver tasks have |answer| > 5000. Each condition type covers a slightly different region of task space.

### Zero is Strict Subset of Noise

Zero embedding solves 9 tasks. All 9 are also solved by latent_8tok. Zero adds nothing unique.

### Task Categories (across ALL conditions)

| Category | Count | Description |
|----------|-------|-------------|
| always-solved | 5 | Correct in baseline + all perturbation conditions |
| RECOVERED by 2-tok | 7 | Baseline wrong, 2-tok majority correct |
| sensitive | 6 | Mixed across conditions |
| never-solved | 6 | All |answer| > 5000 except nest_014 (ans=20, anomaly) |
| REGRESSED by 2-tok | 1 | nest_007 (ans=1044), baseline correct but 2-tok mostly wrong |

### Answer Magnitude Gradient

| Bin | n | Baseline | 1-tok | 2-tok | 8-tok-L | Delta (2-tok) |
|-----|---|----------|-------|-------|---------|---------------|
| tiny (≤10) | 4 | 0% | 17% | 75% | 40% | +75pp |
| small (11-100) | 3 | 33% | 33% | 56% | 27% | +22pp |
| medium (101-1000) | 7 | 57% | 86% | 86% | 80% | +29pp |
| large (1001-5000) | 3 | 100% | 56% | 78% | 50% | -22pp |
| huge (>5000) | 8 | 0% | 17% | 25% | 20% | +25pp |

**Spearman(|answer|, 2tok_acc) = -0.458, p = 0.021** — significant negative correlation.

### Oracle vs Independence: Positive Correlation Structure

Permutation test (5000 permutations, preserving per-latent marginals):

| Group | k latents | Observed Oracle | Independence Null | z-score | p |
|-------|-----------|----------------|-------------------|---------|---|
| 2-tok | 1 | 60% | 60% | 0.00 | 1.000 |
| 2-tok | 2 | 80% | 84% | -0.84 | 0.902 |
| 2-tok | 3 | 88% | 94% | -1.35 | 0.963 |
| 8-tok-L | 1 | 36% | 36% | 0.00 | 1.000 |
| 8-tok-L | 3 | 60% | 77% | -3.04 | 1.000 |
| 8-tok-L | 5 | 76% | 94% | -4.25 | 1.000 |
| 8-tok-L | 10 | 92% | 100% | -8.25 | 1.000 |

**Key insight**: Oracle is BELOW independence, not above. Latents are positively correlated —
they fail on the same hard tasks (large answers). 2-tok has least redundancy (closest to
independence null), explaining its higher oracle efficiency per latent.

The combined oracle = 100% is trivially expected under independence with 21 conditions at ~44%
average accuracy. The INTERESTING finding is that 2-tok achieves near-independence oracle rates
with only 3 directions, while 8-tok shows massive redundancy (z=-8.25 at k=10).

### Leave-One-Condition-Out Oracle (Shapley-style, Codex-recommended)

| Dropped | Oracle | Lost Task | Interpretation |
|---------|--------|-----------|----------------|
| noise_1tok | 24/25 | nest_005 (8360) | Unique contribution |
| noise_2tok | 25/25 | - | Dominated by others |
| latent_8tok | 24/25 | nest_021 (5639) | Unique contribution |
| zero_8tok | 25/25 | - | Dominated, adds nothing |
| mean_8tok | 24/25 | nest_008 (7278) | **UNVERIFIABLE** (see below) |

**SCORING BUG (nest_008, mean_8tok)**: The mean_embedding experiment used an older code version
that stored only 500 chars of response. verify_answer() was called on the full response
(which may have been longer), but the stored 500-char response does NOT contain "7278".
Reproducing verify_answer on the stored response gives False, but the stored `correct` flag is True.
**This result is UNVERIFIABLE** from the stored data. Need to re-run mean_embedding with
current code (which stores 2000 chars + response_raw).

**Impact**: Combined oracle is 24/25 (verified) or 25/25 (if nest_008 mean_8tok is genuinely correct).
Conservative claim: 24/25 = 96%, consistent with original analysis.

Three condition families each rescue exactly one unique task. noise_2tok and zero_8tok are fully dominated.

### Spearman on Delta Accuracy (Codex-corrected test)

**Raw accuracy**: Spearman(|answer|, 2tok_acc) = -0.458, p = 0.021 (significant)
**Delta accuracy**: Spearman(|answer|, delta_acc) = -0.295, p = 0.152 (NOT significant)

Codex noted: the raw-accuracy correlation conflates baseline difficulty with treatment effect.
When testing on GAIN over baseline (the proper SR variable), the correlation vanishes.
This WEAKENS the stochastic resonance framing — the answer-magnitude gradient is driven by
baseline difficulty, not by differential noise benefit.

### Clopper-Pearson CI for 25/25 Oracle

- 95% CI (two-sided): [0.863, 1.000]
- 95% lower bound (one-sided): 0.887
- Reject H0: p <= 0.85 (p = 0.017)
- Fail to reject H0: p <= 0.90 (p = 0.072)
- **Claim**: oracle coverage > 85% is defensible; > 90% is not

### Budget-Matched Oracle Comparison (Codex-recommended test)

Bootstrap subsampling: random k-of-10 from 8-tok vs 2-tok at same k

| k | 2-tok Oracle | 8-tok Mean [95% CI] | P(8tok < 2tok) |
|---|-------------|---------------------|----------------|
| 1 | 60% | 44.2% [36, 56] | 1.000 |
| 2 | 80% | 59.2% [48, 72] | 1.000 |
| 3 | 88% | 68.5% [52, 80] | 1.000 |

**Oracle AUC** (k=1..3): 2-tok = 0.760, 8-tok random-3 = 0.572 [0.467, 0.680], P = 1.000

2-tok is universally more oracle-efficient at every budget level.
This is NOT explained by higher per-latent accuracy alone (60% vs 44%) —
2-tok latents also explore more diverse task-space regions.

### Codex Oracle Framing (2026-03-05)

- **100% combined oracle**: descriptive benchmark endpoint, NOT generalizable standalone claim
- **Oracle efficiency** (coverage-vs-budget curve): MAIN RESULT
  - Make the main figure a coverage-vs-budget curve
  - 2-tok reaches 88% in 3 runs; 8-tok needs 10 runs for 92%
  - Treat 25/25 as the rightmost capstone point
- **Unique solvers**: frame as "condition-specific rescue windows"
  - Use leave-one-condition-out table (above)
  - Caution: mean_8tok singleton needs manual revalidation
- **Answer magnitude**: supports bounded SR but only weakly
  - Test on delta accuracy, not raw (Spearman becomes NS)
  - Use logistic regression with quadratic term for inverted-U
- **Must do**: held-out task set (n=100) with pre-registered 20-run budget

### Statistical Notes
- Pairwise agreement between 2-tok latents: 60-76% (substantial task redistribution)
- Fleiss kappa: 0.278 (fair agreement, not random but not homogeneous)
- Combined oracle = 100% is EXPECTED under independence (trivial, not publishable as standalone)
- 2-tok oracle EFFICIENCY (88% with just 3 directions) IS the publishable result
- Need: held-out test set (n=100 tasks) to validate oracle scaling curve

## 9b. Universal Fraction and Two-Stage Model (Codex 2026-03-05)

### Per-Perturbation Structural Decomposition
| Condition | Unan | Froz | Sens | Per-lat counts | Std | Sens frac (q) |
|-----------|------|------|------|----------------|-----|---------------|
| zero (3) | 9 | 16 | 0 | [9,9,9] | 0.00 | n/a |
| 1-tok (3) | 6 | 8 | 11 | [10,11,11] | 0.47 | 0.424 |
| **2-tok (3)** | **9** | **3** | **13** | **[15,15,15]** | **0.00** | **0.462** |
| 3-tok (4) | 7 | 10 | 8 | [11,11,11,10] | 0.43 | 0.44 |
| 8-tok (10) | 3 | 2 | 20 | [9..14] | 1.76 | 0.405 |

### Codex Two-Stage Model (2026-03-05)
1. **Magnitude sets the task-level threshold**: which tasks are frozen, unanimous, or sensitive
2. **Direction selects** which sensitive tasks get solved, at roughly fixed capacity (~42%)

### Equalization vs Independence (Codex)
- Expected SD under iid Bernoulli(q=0.42): 1-tok=1.64, 2-tok=1.78, 3-tok=1.31, 8-tok=2.21
- Observed SD: 1-tok=0.47, **2-tok=0.00**, 3-tok=0.43, 8-tok=1.76
- 2-tok and 3-tok are MUCH more equalized than iid → **fixed-quota model**, not independent Bernoulli
- 8-tok is close to iid (1.76 vs 2.21 expected)

### Endogeneity Warning (Codex)
- 42% is a CONDITIONAL occupancy rate (sensitive tasks exclude all-correct and all-wrong)
- Under iid Bernoulli at k=3: q_cond = (1+p)/3, so q_cond ≈ 0.42 implies p ≈ 0.26
- Categories are endogenous to the directions that defined them

### Held-Out Direction Test (N4 as natural held-out)
- N4 solves 3/7 = 42.9% of N1-N3's sensitive tasks → fraction CONFIRMED on held-out
- BUT N4 breaches nest_003 (unanimous in N1-N3) → categories are NOT perfectly stable

### N4 Result (2026-03-05)
- N4 = 10/25 (40%): equalization breaks from [11,11,11] to [11,11,11,10]
- 3-tok std now 0.43 (was 0.00 with N1-N3 only)
- Key: equalization is APPROXIMATE at 3-tok, EXACT at 2-tok (so far)
- N5-N10 will refine the 3-tok distribution

## 11. Codex Shi et al. Positioning (2026-03-05)

**Codex assessment** of Shi et al. (arXiv:2510.01032, Oct 2025):

- **Positioning**: Same umbrella phenomenon, different regime. Shi = closest related work.
  They establish discrete-token perturbations help; we study continuous embedding-space
  perturbations with stronger mechanistic structure.
- **Mechanism**: No conflict. Multi-level reconciliation:
  `prefix perturbation → MLP activation redistribution → gate crossing / mode selection → altered reasoning trajectory → task-level redistribution`
  Shi = proximate MLP substrate. PGRMS = systems-level behavioral model.
  Present think-mode gating as **Qwen3-4B instantiation**, not universal explanation.
- **Venue**: Pushes toward NeurIPS-style pitch. Need: (1) Shi-style token control,
  (2) internal mechanism probe, (3) broader replication. Without these, EMNLP safer but weaker.
- **Effect size**: Don't sell +28pp vs +1-5% as simple superiority. Ours = peak resonance
  in sensitive regime; theirs = broad cross-model average. Report headroom + matched controls.
- **Immediate paper changes**: Remove "first to show" claims. Reframe as continuous
  perturbation control + dynamical structure. Add Shi comparison table.

**Positioning sentence (Codex-approved)**:
"Shi et al. (2025) show that repeated punctuation tokens can produce modest, non-monotonic
reasoning gains via activation redistribution. We study a distinct continuous-perturbation
regime, where random embedding-scale prefixes induce much larger, direction-independent but
task-selective effects, revealing mode gating, oracle efficiency, and task-specific resonance."

## 11. Pending Experiments (Priority Order, per Codex 2026-03-05, updated post-Shi)

### Behavioral Experiments
1. **Shi-style discrete token control** -- HIGHEST PRIORITY (Codex).
   Run repeated `/` and `?` on Qwen3-4B at 1,2,3,8 tokens. Implemented in harness.
2. **3-tok dose-response** -- RUNNING (Noise 2/10 in progress).
   - Noise 1: 44% (11/25), Noise 2: 44% (11/25) — EQUALIZATION AT 3-TOK TOO!
   - N1 and N2 solve EXACTLY 11/25 but 4 different tasks (2 swaps each way)
   - 3-tok solve set is ENTIRELY a subset of 2-tok oracle (adds 0 new tasks)
   - 3-tok gains from baseline: nest_006, _010, _015, _017, _019 (5 gains, 2 losses)
   - Combined oracle unchanged at 24/25 (nest_008 still unsolved)
   - Dose-response: 0=32%, 1=42.7%, 2=60%, 3=44% (N1+N2), 8=44%
   - Equalization hierarchy: 2-tok [15,15,15] std=0, 3-tok [11,11] std=0 (n=2 only!)
   - N3 CONFIRMED: [11, 11, 11] across 3 latents (P=0.004 under independence)
   - STRUCTURAL FINDING: equalization = fixed capacity per perturbation level
     - 2-tok: 9 always + 3 never + 13 sensitive, per-latent = 9 + 6/13 = 15
     - 3-tok: 8 always + 10 never + 7 sensitive, per-latent = 8 + 3/7 = 11
     - Both achieve 100% oracle coverage of sensitive tasks from 3 latents
     - More perturbation energy freezes more tasks (10 vs 3 never-solved)
     - 3-tok oracle (3 lat) = 15/25 = 60% = exactly 2-tok per-latent accuracy
3. **Scale 2-tok to n=100 tasks, n=10 latents** -- replicate oracle + zero-variance
4. Dense dose-response (4,5,6,7 tokens)
5. Cross-model (Qwen3-8B)

### Mechanism Probes (Codex-ordered for NeurIPS, 2026-03-05)
1. **D. Think-gate probe** (HIGHEST ROI) -- Script ready: run_think_gate_probe.py
   - Single forward pass, measure <think> log-prob under all perturbation conditions
   - Tests PGRMS gating claim directly
2. **C. MLP redistribution probe** -- Hook MLP blocks on last prompt + first generated tokens
   - Under teacher forcing to control for different text
   - Measure layer-wise MLP output norm, top-k neuron overlap, activation entropy
   - Bridges directly to Shi et al.'s mechanism
3. **E. Reduced causal tracing** -- Activation patching at candidate layers
   - Patch baseline residual/MLP into perturbed run; check if <think> logit gain collapses
   - First-token effects only (full rollout too expensive)
4. **A. Hidden-state scouting** -- Identify candidate layers for C and E
   - Residual stream comparison: L2, cosine sim, norm changes per layer

### Lower Priority
- Reduced max_new_tokens sweep (256, 512) -- reviewer-facing control
- Force-think + 2 noise tokens -- stochastic decomposition
- Antipodal pair experiment
