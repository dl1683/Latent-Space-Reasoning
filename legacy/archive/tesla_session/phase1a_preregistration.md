# Phase 1A Pre-Registration: Prefix vs Temperature Comparison

## Status: REVISED v2.3 — Codex R5+R7+R8+R10 corrections applied (2026-04-15)

### Codex R5 Key Corrections
1. **Primary metric changed**: DM13 → margin (p - max(q_i)). DM13 is secondary.
2. **4th operator added**: temp=1.0 for dose-response.
3. **Tie-breaking revised**: Fractional expected correctness for confirmatory; sensitivity analyses for all policies.
4. **H3 tautology fixed**: Split-half out-of-sample prediction (estimate on seeds 0-8, predict on 9-16).
5. **Trivial-attractor analysis added**: Cross-task wrong-answer mass for values 0, 1, 2, -1.
6. **Response-level analysis added**: Edit distance, first-divergence, output length.
7. **Extraction failure hard rule**: >5% failure suspends confirmatory claims.
8. **N=17 over N=16**: Odd number eliminates exact ties for correct-vs-top-wrong.

## Date: 2026-04-15
## Pre-registered by: Claude (Tesla session) + Codex R3/R5 approval

---

## 1. Research Question

**Does random soft-prefix perturbation produce higher plurality margin (p - max(q_i)) than temperature sampling on frozen small LLMs, and does this translate into better plurality-vote accuracy?**

Secondary: Does the margin hierarchy hold across temperature levels (dose-response)?

---

## 2. Hypotheses (Pre-Registered)

### H1 (SECONDARY — demoted from primary per Codex R10): Prefix produces higher plurality margin than temperature
- **Measure**: margin = p - max(q_i) per task
- **Prediction**: margin_prefix > margin_temperature (at temp=0.6)
- **Operationalized**: Paired Wilcoxon signed-rank test on per-task margin values
- **Required decomposition**: Report p, max(q_i), oracle coverage, and tie rate separately

### H1b: max(q_i) is lower for prefix than temperature
- **Measure**: max(q_i) per task
- **Prediction**: max(q_i)_prefix < max(q_i)_temperature
- **This is the mechanism claim**: prefix spreads errors more evenly

### H2: Plurality voting accuracy is higher for prefix than temperature
- **Measure**: Number of tasks where plurality vote selects correct answer
- **Prediction**: plurality_correct_prefix > plurality_correct_temperature
- **Operationalized**: McNemar test on paired task outcomes (recognized as underpowered at N=25)
- **Supplementary**: Effect size + bootstrap CI reported regardless of significance

### H3: The p > max(q_i) condition predicts plurality success OUT-OF-SAMPLE
- **Measure**: Estimate p and max(q_i) from seeds 0-8 (training half); predict plurality outcome on seeds 9-16 (held-out half)
- **Prediction**: >80% prediction accuracy on held-out half
- **Tie scoring**: Held-out plurality on seeds 9-16 uses same fractional expected correctness rule as primary analysis
- **This avoids the tautology** of computing condition from same data as outcome

### H4: Temperature dose-response on margin
- **Measure**: margin at temp=0.6 vs temp=1.0
- **Prediction**: margin_temp0.6 > margin_temp1.0 (higher temperature collapses p faster than it spreads max(q_i))
- **If H4 fails**: temperature diversity IS sufficient and prefix adds no unique value

### H5: Perturbation increases EOS (completion) rate
- **Measure**: Per-task fraction of seeds that reach end-of-sequence
- **Prediction**: EOS_rate_prefix > EOS_rate_baseline (perturbation helps model complete reasoning)
- **Critical context**: Adversarial audit revealed P(correct|EOS) = 1.000 across all 500 perturbation responses. ALL wrong answers come from truncated responses. EOS rate may be the fundamental mechanism, not answer diversity.
- **Required analysis**: EOS rate per operator, P(correct|EOS) per operator, mean token length EOS vs truncated

### H6 (Codex R7+R8 — REQUIRED before full Phase 1A): Token budget sweep
- **MUST RUN FIRST**: TWO operators at max_new_tokens = {512, 768, 1024, 1536, 2048} on all 25 tasks
  - **Greedy**: 1 generation per task per budget (125 total)
  - **Prefix**: N=10 per task per budget (1,250 total)
- **Codex R8 MANDATORY**: Greedy sweep is REQUIRED — prefix-only does not answer "why not just raise max_new_tokens?"
- **Prediction**: EOS rate increases monotonically with budget; plurality accuracy tracks EOS rate
- **Critical test**: If max_new_tokens=2048 greedy achieves ≥72% accuracy, prefix is dominated
- **Purpose**: Determine whether the mechanism is path-switching (perturbation) or budget-insufficient (verbosity)
- **Store**: Full raw output (NO truncation), stop reason, token count, EOS/truncation status
- **Score with**: last-integer, answer-anywhere, strict final-answer parser
- **Report**: Accuracy-vs-total-tokens Pareto curves (Codex R8 mandatory)
- **GPU cost**: ~2-3 hours (1,375 generations × 5 budgets)
- **Changelog**: v2.1 had prefix-only; v2.2 adds greedy per Codex R8 review

### H7 (Codex R10 — NEW PRIMARY MECHANISM): Prefix produces higher EOS rate than temperature
- **Measure**: Per-task EOS rate (fraction of N=17 seeds reaching end-of-sequence)
- **Prediction**: EOS_rate_prefix > EOS_rate_temp0.6
- **Operationalized**: Wilcoxon signed-rank test on per-task EOS rates
- **This replaces H1 as the primary mechanism test** (Codex R10 correction: EOS rate is more fundamental than margin)
- **If H7 holds**: Prefix finds more completing trajectories → more correct plurality candidates
- **If H7 fails**: EOS rate is similar → prefix and temperature are equivalent for completion

### H8 (Codex R10 — NEW): P(correct|EOS) is maintained across operators
- **Measure**: P(correct|EOS) per operator
- **Prediction**: P(correct|EOS) ≈ 1.0 for greedy and prefix; P(correct|EOS) < 1.0 for temperature
- **Critical test**: If temperature produces completed-but-wrong responses, prefix has a qualitative advantage (deterministic convergence)
- **Report with**: task-clustered bootstrap CIs, not just response-level counts

### H_null: Temperature sampling achieves similar EOS rate and margin to prefix
- If H_null holds: Prefix perturbation is not special; any diversity-generating method works. Paper story shifts from "prefix perturbation" to "any completion-increasing method" as mechanism.

---

## 3. Experimental Design

### 3.1 Models
| Model | Quantization | Role |
|-------|-------------|------|
| Qwen3-4B | 4-bit NF4 | Primary (known to benefit from prefix) |
| DeepSeek-1.5B | 4-bit NF4 | Generalization test (known to be hurt by prefix) |

### 3.2 Tasks
- **Same 25 arithmetic tasks** as existing N=10 data
- task_type: nested, difficulty: sweet_spot
- These are pre-existing tasks, not selected post hoc
- **Acknowledged limitation**: These are the discovery set, not held-out. Results are exploratory on known tasks. Confirmatory claims require held-out replication.

### 3.3 Operators (4 total — revised from 3)
| Operator | Config | N per task | Stochastic? |
|----------|--------|-----------|-------------|
| O1: Greedy baseline | temp=0, no prefix | 1 | No |
| O2: Random soft prefix | 2-token, RMS-matched, temp=0 | 17 | Yes (random prefix) |
| O3: Temperature 0.6 | temp=0.6, no prefix | 17 | Yes (sampling) |
| O4: Temperature 1.0 | temp=1.0, no prefix | 17 | Yes (sampling) |

**N=17 rationale**: Odd number means correct answer vs top wrong answer cannot have exact equal counts when they together account for all candidates (eliminating one class of tie). Marginally more seeds than N=16 at negligible cost.

### 3.4 Generation Config
- max_new_tokens: 1024
- Greedy decoding for O1, O2 (temp=0, do_sample=False)
- Stochastic for O3 (temp=0.6, do_sample=True, top_p=1.0)
- Stochastic for O4 (temp=1.0, do_sample=True, top_p=1.0)
- O2 prefix: 2 tokens, random Gaussian, RMS-matched to embedding scale
- O2 seeds: 0-16 (17 independent random prefixes)
- O3/O4 seeds: 0-16 (17 independent sampling seeds via torch.manual_seed)

### 3.5 Total Generations
| Component | Generations | Estimated Time |
|-----------|------------|----------------|
| O1 (greedy) × 25 tasks × 2 models | 50 | ~10 min |
| O2 (prefix) × 25 tasks × 17 seeds × 2 models | 850 | ~105 min |
| O3 (temp0.6) × 25 tasks × 17 seeds × 2 models | 850 | ~105 min |
| O4 (temp1.0) × 25 tasks × 17 seeds × 2 models | 850 | ~105 min |
| **Total** | **2,600** | **~5.5 hours** |

---

## 4. Pre-Registered Selectors

All selectors applied identically to all operators. No selector is developed or tuned post hoc.

### S1: Random Pick
- Select one candidate uniformly at random
- Expected accuracy = mean individual accuracy
- This is the BASELINE selector. Everything must beat this.

### S2: Plurality Vote (PRIMARY)
- Extract answer from each candidate using `extract_answer()` (last integer extraction)
- Count votes for each unique extracted answer
- Select the answer with the most votes
- **Tie-breaking policy** (pre-registered):
  - **Confirmatory**: Report expected accuracy under uniform random tie-breaking (fractional credit: if correct answer ties with k-1 others, credit = 1/k)
  - **Sensitivity analysis**: Also report earliest-seed, latest-seed, worst-case, best-case tie resolution
  - Rationale: eliminates any hidden bias from deterministic tie-breaking

### S3: Formal Verifier (DS2)
- For arithmetic tasks: execute the expression using Python eval()
- If any candidate's extracted answer matches eval() result: select that candidate
- If no match: fall back to S2 (plurality vote)
- This is the UPPER BOUND for arithmetic (100% selector reliability)

### S4: Oracle (Diagnostic Only)
- Select the correct answer if any candidate produced it
- NOT a deployable selector — evaluation-only ceiling

---

## 5. Answer Extraction Protocol (Pre-Registered, Locked)

### Function: extract_answer()
- Extract the LAST integer appearing in the decoded text output
- Regex: `r'-?\d+'` applied globally, take the last match
- No normalization beyond integer extraction
- Edge cases:
  - No integers found → answer = None (counts as wrong)
  - Multiple integers → last one wins (existing behavior)
  - Negative integers → preserved with sign

### Extraction validation
- Before running: manually verify extraction on 5 sample outputs per operator
- After running: report extraction failure rate per operator
- **Hard rule**: If extraction failure rate > 5% for any primary operator (O2, O3, O4): confirmatory claims are SUSPENDED. The run becomes exploratory. A revised extractor must be pre-registered and the experiment rerun for confirmatory status.

---

## 6. Metrics (Pre-Registered)

### Primary Metrics (Locked Hierarchy — Codex R5→R10 REVISED)

**Codex R10 correction**: EOS/completion metrics promoted to primary mechanism; margin/DM13 demoted to secondary. This reflects the EOS-completion discovery.

| Priority | Metric | Description | Level | Role |
|----------|--------|-------------|-------|------|
| **1** | **EOS rate** | Fraction of seeds reaching end-of-sequence per task | Per-task-per-operator | **PRIMARY mechanism** |
| **2** | **P(correct\|EOS)** | Correctness of normally-completing responses | Per-operator | **PRIMARY mechanism** |
| **3** | **Plurality accuracy** | Fraction correct under S2 (fractional tie-breaking) | Per-operator | **PRIMARY outcome** |
| **4** | **Oracle coverage** | Fraction of tasks where any candidate correct | Per-operator | Diagnostic |
| **5** | **p** | Individual correctness probability | Per-task-per-operator | Diagnostic |

### Secondary Metrics (Previously Primary — Demoted per Codex R10)
| Priority | Metric | Description | Level |
|----------|--------|-------------|-------|
| **6** | **Margin** | p - max(q_i) per task | Per-task-per-operator |
| **7** | **max(q_i)** | Top wrong answer mass per task | Per-task-per-operator |

### EOS-Detail Metrics
| Metric | Description | Level |
|--------|-------------|-------|
| **Mean EOS tokens** | Average generated tokens for EOS responses (CORRECTED for inputs_embeds bug) | Per-task-per-operator |
| **Mean trunc tokens** | Average generated tokens for truncated responses (should be max_new_tokens) | Per-task-per-operator |
| **5-bucket classification** | EOS+correct, EOS+wrong, trunc+correct-stated, trunc+correct-somewhere, trunc+no-correct | Per-response |

### Secondary Diversity Metrics
| Metric | Description | Level |
|--------|-------------|-------|
| DM13: Answer-level diversity | K/N_wrong (unique wrong answers / total wrong) | Per-task-per-operator |
| Tie rate | Fraction of tasks where plurality has ties | Per-operator |
| Entropy | Shannon entropy of answer distribution per task | Per-task-per-operator |

### Diagnostic Metrics
| Metric | Description |
|--------|-------------|
| Trivial-attractor mass | Cross-task wrong-answer mass for values {0, 1, 2, -1} (excluding tasks where value is correct) |
| Extraction failure rate | % of candidates where extract_answer() returns None |
| Output length | Token count per candidate |
| Pairwise edit distance | Levenshtein distance between within-operator response pairs |
| First-divergence position | Token index where within-operator responses first differ |

### Selection Quality Metrics
| Metric | Description |
|--------|-------------|
| SQ1: Margin prediction accuracy | % of tasks where sign(margin) matches plurality outcome (on held-out half) |
| SQ2: Selector gap | (Oracle - Plurality) / (Oracle - Random) |
| SQ3: Tiebreak sensitivity | Range of plurality accuracy across 4 tiebreak policies |

---

## 7. Statistical Tests (Pre-Registered)

### Test 1: Prefix vs Temperature margin (H1) — PRIMARY
- Wilcoxon signed-rank test on per-task margin values (all 25 tasks)
- α = 0.025 (Bonferroni-corrected for 2 primary tests)
- Supplementary: paired bootstrap CI on mean margin difference

### Test 2: Prefix vs Temperature plurality accuracy (H2)
- McNemar exact test on paired task outcomes
- α = 0.025 (Bonferroni-corrected)
- **Power caveat**: With 25 tasks, this test is underpowered. Need ≥7 discordant pairs with 0 opposite discordance for significance. Report effect size + bootstrap CI regardless.

### Test 3: max(q_i) comparison (H1b)
- Wilcoxon signed-rank on per-task max(q_i) values
- Exploratory (not Bonferroni-corrected)

### Test 4: Temperature dose-response (H4)
- Wilcoxon signed-rank on per-task margin: temp=0.6 vs temp=1.0
- Exploratory

### Test 5: Out-of-sample prediction (H3)
- Compute p and max(q_i) on seeds 0-8; predict plurality outcome on seeds 9-16
- Report prediction accuracy (fraction of tasks correctly predicted)
- No significance test — descriptive evaluation

### Test 6: Cross-model comparison
- Report all metrics for both Qwen and DeepSeek
- If DeepSeek prefix fails (as expected): report as NEGATIVE result
- Exploratory

### Test 7 (NEW PRIMARY): Prefix vs Temperature EOS rate (H7)
- Wilcoxon signed-rank test on per-task EOS rates (all 25 tasks)
- α = 0.0167 (Bonferroni-corrected for 3 primary tests: T7, T2, T8)
- Supplementary: task-clustered bootstrap CI on mean EOS rate difference

### Test 8 (NEW): P(correct|EOS) across operators (H8)
- Task-clustered bootstrap CI for P(correct|EOS) per operator
- α = 0.0167 (Bonferroni-corrected)
- If any operator shows P(correct|EOS) < 0.95: report as significant finding

### Multiple testing (REVISED per Codex R10)
- 3 primary tests (T7, T2, T8) → Bonferroni correction: α = 0.0167 each
- T1 (margin) reclassified as SECONDARY/exploratory
- All other analyses are exploratory and reported as such
- Permutation tests used where possible for robustness
- Task-clustered bootstrap used alongside response-level stats (Blind Spot 17)

---

## 8. Pre-Registered Analyses (Non-Test)

### A1: Full answer histograms
- For every task × operator: report the complete answer frequency table
- Visualization: bar chart of answer counts per task

### A2: Trivial-attractor audit
- Compute cross-task frequency of wrong answers = {0, 1, 2, -1}
- Report per-operator: what fraction of total wrong votes go to trivial values?
- If trivial attractors dominate: this is a systematic extraction/model bias, not genuine diversity

### A3: Response-level structural analysis
- Pairwise edit distance between all within-operator response pairs per task
- Mean first-divergence position (how early do responses diverge?)
- Output length distribution per operator
- Prediction: prefix responses diverge earlier than temperature responses

### A4: Reuse of existing data
- Existing N=10 prefix data for Qwen3-4B (seeds 0-9) can be partially reused
- New seeds 10-16 must be generated for Phase 1A
- Temperature data must be generated entirely fresh
- DeepSeek must be generated fresh for all operators

---

## 9. Success Criteria (Pre-Registered)

### STRONG SUCCESS
All of:
- H1 confirmed: margin_prefix > margin_temperature (p < 0.025)
- H1b confirmed: max(q_i)_prefix < max(q_i)_temperature
- H2 directionally consistent (plurality_prefix > plurality_temperature, even if McNemar underpowered)
→ Paper claim: "Prefix perturbation produces lower top-wrong concentration, enabling more effective verifier-free plurality selection."

### MODERATE SUCCESS
Either:
- H1 confirmed but H2 not (margin is higher but plurality accuracy is similar due to ties)
- H2 confirmed but H1 not (plurality is better for reasons other than margin)
→ Paper claim: "Prefix perturbation offers advantages but the mechanism is [margin/something else]."

### NULL RESULT
Neither H1 nor H2 confirmed:
→ Paper claim: "Plurality voting works for any diverse candidate generator; prefix is not special."
→ This is still publishable — the plurality mechanism is valuable regardless of operator.

### DOSE-RESPONSE FINDING (H4)
If margin_temp0.6 > margin_temp1.0:
→ Supports "there is an optimal diversity-fidelity tradeoff"
If margin_temp1.0 > margin_temp0.6:
→ Higher temperature is better; prefix may still beat both or not

### NEGATIVE RESULT
Plurality voting fails for temperature AND fails for prefix at N=17:
→ Investigate extraction failures, N-scaling, or task-set confounds

---

## 10. Timeline

| Step | Duration | GPU? |
|------|----------|------|
| Implement harness modifications (add temp operators) | 2-4 hours | No |
| Validate on 1 task × 4 operators | 20 min | Yes |
| Full Phase 1A run | ~5.5 hours | Yes |
| Analysis + figure generation | 2-3 hours | No |
| Codex review of results | 1 hour | No |
| **Total** | **~12 hours elapsed** | **~6 hours GPU** |

---

## 11. Changelog

| Version | Date | Changes |
|---------|------|---------|
| v1 | 2026-04-15 | Initial draft (3 operators, N=16, DM13 primary) |
| v2 | 2026-04-15 | Codex R5 revision: margin primary, 4 operators, N=17, fractional ties, H3 split-half, trivial attractors, response-level analysis, extraction hard rule |
| v2.1 | 2026-04-15 | Codex R7 (EOS review): Added H5 (EOS rate), H6 (token budget sweep REQUIRED first), EOS-mechanism metrics. Added 5-bucket output classification. |
| v2.2 | 2026-04-15 | Codex R8 (compute cost review): H6 expanded to include greedy budget sweep (not just prefix). Must report accuracy-vs-total-tokens Pareto curves. Greedy sweep is mandatory to answer "why not just raise max_new_tokens?" |
| v2.3 | 2026-04-15 | Codex R10 (convergence check): EOS rate + P(correct\|EOS) promoted to PRIMARY mechanism metrics. Margin/DM13/max(q_i) demoted to SECONDARY. Added 5-bucket classification to EOS-detail metrics. Token counting must be corrected (inputs_embeds bug). Task-clustered CIs required. |
