# Tesla Mode Phase 5: Hypotheses, Predictions, and Preregistration

## Purpose
Formally state hypotheses H1-H6 before data collection. Preregister estimator, features, windows, and acceptance criteria. These predictions create accountability — when Phase A data is collected, we check them.

---

## Preregistration (Locked Before Data Collection)

### Estimator Choice
**MI estimator**: k-NN mutual information estimator (Kraskov-Stögbauer-Grassberger, k=5)
**Why**: Non-parametric, no distributional assumptions, works on continuous features.
**Alternative considered**: Discretized MI (equal-frequency binning, 10 bins) — NOT chosen as primary because bin count is a hyperparameter that could be tuned to inflate results.
**Null distribution**: 1000 permutations (shuffle correctness labels within task groups, preserving task-group structure).
**CI method**: Bootstrap by task group (not candidate row), 1000 iterations, 95% CI.

### Feature Set (Frozen)
Features used in MI analysis (cannot be added after labels are inspected):
1. `mean_logprob` at window w
2. `logprob_slope` at window w (None if n_observed < 2 → excluded from MI at w=1)
3. `token_entropy_mean` at window w
4. `token_entropy_slope` at window w (None if n_observed < 2)
5. `cumulative_logprob` at window w
6. `attention_sink_mass` at window w (None if attentions unavailable → excluded)
7. `repetition_rate` at window w (excluded for w ≤ 8 — defined as 0 for all)
8. `think_token_fraction` at window w
9. `eos_appeared` at window w (binary)
10. `truncated_at_window` at window w (binary)

**Composite feature** (single number for oracle recall routing):
`routing_score = mean_logprob - 0.5 × attention_sink_mass + 0.1 × (1 - token_entropy_mean)`
(weights are pre-registered, not optimized on data)

### Windows (Frozen)
`w ∈ {1, 4, 8, 16, 32, 64, 128}` generated tokens.

### Train/Test Split
70% of task groups → MI training (feature selection if needed)
30% of task groups → held-out oracle recall evaluation
Split is pre-registered by task_id hash before data collection.

### Labels
- **Arithmetic**: `correct = 1` if `extract_answer(output) == ground_truth`, `0` otherwise. `answer_anywhere_correct` searched in output EXCLUDING `<think>` blocks.
- **Legal/Planning**: `correct = 1` if this candidate is the oracle winner in its task group AND oracle margin ≥ 0.5 points above median. Task groups with no valid oracle winner are excluded.

---

## H1: Early Features Carry Meaningful MI at 64 Tokens (Arithmetic)

**Statement**: Early trajectory features at w=64 contain statistically significant mutual information with final correctness for arithmetic tasks (Qwen3-4B Q4).

**Prediction**: `I(features_64; correct) > 0.1 bits` with p < 0.05 vs permutation null, measured on held-out arithmetic task groups.

**Counter-prediction**: MI stays below 0.05 bits at all windows ≤ 64. All candidates produce near-identical early token sequences regardless of final correctness. The trajectory diversity that matters for correctness only emerges after w=64.

**Why this matters**: If H1 is false, the entire Observer-Router concept (Phase B) is killed. The system cannot route cheaper than full generation.

**Operationalization**:
- Dataset: 100 arithmetic task groups × 11 candidates (Qwen3-4B Q4)
- Metric: `I(features_64; correct)` via k-NN estimator, bootstrap CI by task group
- Acceptance: > 0.1 bits AND p < 0.05 vs permutation null AND lower CI bound > 0.05 bits
- Rejection: ≤ 0.05 bits OR not significant vs permutation null

---

## H2: Early Features Predict Convergence More Than Computation (Qwen3-4B Q4)

**Statement**: For Qwen3-4B Q4 arithmetic, early features at w=64 carry more MI with `converged` (answer in final position) than with `answer_anywhere_correct` (answer computed anywhere in output).

**Prediction**: `I(features; converged) > I(features; answer_anywhere_correct)` at w=64 for Qwen3-4B Q4.

**Rationale**: The Phase 1 decomposition showed that Qwen3-4B at 4-bit is "convergence-limited" — answer-anywhere is already high (80%) at baseline. Perturbation mainly helps answers reach the final position. Early features that predict convergence (e.g., attention patterns, output length forecast) should dominate over features that predict whether the model computed the answer at all.

**Counter-prediction**: No significant difference. Features predict both equally (or neither significantly).

**Operationalization**:
- Compare MI values: `I(features_64; converged)` vs `I(features_64; answer_anywhere_correct)`
- Bootstrap paired comparison by task group
- Report decomposition: `P(correct) = P(anywhere) × P(converged|anywhere)` with feature-specific MI per component

---

## H3: Qwen3-8B Q8 Shows More Computation Signal Than Qwen3-4B Q4

**Statement**: For Qwen3-8B Q8, early features at w=64 carry MORE MI with `answer_anywhere_correct` (computation component) than for Qwen3-4B Q4.

**Rationale**: Qwen3-8B Q8 is "computation-limited" — baseline answer-anywhere is only 32%. Perturbation improves the model's ability to actually compute the answer (+18pp answer-anywhere). This should manifest as higher MI with answer-anywhere-correct at early windows for 8B Q8 vs 4B Q4.

**Prediction**: `I(features_64; answer_anywhere_correct)[8B-Q8] > I(features_64; answer_anywhere_correct)[4B-Q4]` with bootstrap CI not overlapping.

**Counter-prediction**: No significant difference. The mechanism difference between models is not visible in early features.

**Operationalization**:
- Requires 100 arithmetic task groups × 11 candidates for BOTH models
- Compare MI distributions via bootstrap CIs
- Report: within-model and cross-model MI decomposition

---

## H4: attention_sink_mass Predicts Collapse/Truncation Better Than Correctness

**Statement**: `attention_sink_mass` at w=32 predicts whether the output will be truncated (hit max_new_tokens limit) with AUROC > 0.65, but predicts final correctness with AUROC < 0.65.

**Rationale**: Attention sink patterns lock the model into "presentation mode" (high formatting, low computation) which causes token budget exhaustion → truncation. This is a formatting/convergence effect, not a computation effect. Perturbation breaks the sink pattern, preventing truncation, but whether the computation is correct is a separate question.

**Prediction**:
- `AUROC(attention_sink_mass_32 → truncated) > 0.65` (attention sink predicts truncation)
- `AUROC(attention_sink_mass_32 → correct) < 0.65` (attention sink does NOT predict correctness beyond truncation)

**Counter-prediction**: attention_sink_mass predicts both equally (or neither). It may be a length proxy, not a semantic quality predictor.

**Note**: This hypothesis requires attention output to be available. If `output_attentions=True` fails or is unavailable, H4 is marked inconclusive (not rejected).

**Operationalization**:
- Compute attention sink mass as mean fraction of attention in prefix positions {0,1} across layers {1,2,3,4}, heads all
- Compute AUROC for `attention_sink_mass_32 → truncated` and `attention_sink_mass_32 → correct`
- Bootstrap CIs by task group

---

## H5: A Held-Out Promotion Rule Retains ≥90% Oracle Winners at ≤50% Promotion

**Statement**: The pre-registered composite routing score (`mean_logprob - 0.5 × attention_sink_mass + 0.1 × (1 - token_entropy_mean)`) at w=64, applied to held-out arithmetic task groups, retains ≥90% of oracle winners while promoting ≤50% of candidates.

**Prediction**: On held-out task groups, promote top-5 of 10 candidates by routing score. Measure: P(oracle winner ∈ promoted set) ≥ 0.90.

**Counter-prediction**: Oracle winner recall < 80% at 50% promotion. The routing score cannot distinguish oracle winners from non-winners at this budget level.

**Why the counter-prediction matters**: If H5 is false, Phase B is still killed even if H1 passes. High MI does not imply an operationally useful router.

**Operationalization**:
- Promote top-5 by routing score (composite, pre-registered weights, not optimized)
- Measure oracle winner recall on held-out 30% of task groups
- Bootstrap CI by task group
- Report recall@K curve for K = 1, 2, 3, 4, 5 out of 10

---

## H6: Legal/Planning MI Is Weaker Than Arithmetic MI

**Statement**: Early features at w=64 carry LESS MI with correctness for legal and planning tasks than for arithmetic tasks, even if length/collapse MI is present.

**Rationale**: Arithmetic has a deterministic ground truth (exact answer), making the MI target crisp. Legal/planning quality is multidimensional, judge-dependent, and can manifest late in the output (a critical distinction may appear at token 800, not token 64). The oracle-relative label for qualitative tasks also introduces judge noise.

**Prediction**: `I(features_64; correct)[arithmetic] > I(features_64; correct)[legal]` with non-overlapping bootstrap CIs.

**Sub-prediction**: MI with `truncated` (collapse signal) will be similar or higher for legal/planning vs arithmetic (collapse is universal; quality discrimination is not).

**Counter-prediction**: Legal/planning MI is comparable to arithmetic MI. The oracle-relative label captures enough signal for early routing.

**Operationalization**:
- Requires separate 100-group pilots for legal and planning (separate from arithmetic)
- Report domain-specific MI with side-by-side comparison
- Include collapse signal (MI with `truncated`) as a sanity check

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `inputs_embeds` sequence misalignment | HIGH | CRITICAL | Smoke test before data collection |
| VRAM OOM from `output_attentions` | HIGH | HIGH | Collect attentions only for first 128 tokens |
| Oracle-relative label degeneracy | MEDIUM | HIGH | Require ≥0.5 point margin; exclude degenerate task groups |
| Near-zero feature variance at w≤32 | MEDIUM | MEDIUM | Report feature variance stats; don't interpret zero MI as absence of signal |
| Judge drift across legal/planning runs | MEDIUM | MEDIUM | Pin judge model version; hash rubric |
| Sample size insufficient for reliable MI | MEDIUM | MEDIUM | Use bootstrap CIs; 100 task groups minimum for pilot |
| RMS calibration differs across quantizations | LOW | HIGH | Calibrate per-model per-quantization; assert actual_prefix_rms |
| CUDA non-determinism | LOW | MEDIUM | Smoke test determinism; log flag per record |

---

## Validation Plan

**For each hypothesis**: When Phase A data collection is complete, run the analysis in this order:
1. H4 first (attention sink → truncation AUROC): sanity check that attentions are working correctly
2. H2 (convergence vs computation decomposition): establish the P decomposition
3. H3 (8B Q8 vs 4B Q4 comparison): cross-model check
4. H1 (MI > 0.1 bits at w=64): the primary Phase B gate
5. H5 (oracle recall ≥ 90% at ≤ 50% promotion): the operational gate
6. H6 (legal/planning < arithmetic MI): qualitative domain check

If H1 AND H5 both pass: design Phase B Observer-Router.
If either fails: Phase B is killed. Publish Phase A as the system.

---

## What Phase A Alone Delivers (If H1/H5 Fail)

Phase A is not just infrastructure for Phase B. Even if Phase B is killed:

1. **A clean, reproducible full-generation oracle system** — replaces the broken `orchestrator.decode()` path
2. **Validated soft-prefix mechanism in a package** — currently the research mechanism and package are different systems
3. **A candidate-level atlas** — the first structured dataset of (perturbation, telemetry, outcome) triples for this mechanism
4. **The MI analysis itself** — understanding WHY early features do or don't predict correctness is publishable science (negative result with solid methodology is still a contribution)
5. **The P decomposition** — `P(correct) = P(anywhere) × P(converged|anywhere)` per model/quantization is a clean empirical result

**Phase A is independently valuable.** Phase B is the conditional upside.
