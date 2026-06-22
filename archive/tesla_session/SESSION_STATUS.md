# Tesla Session Status — 2026-04-15 (Updated: Post-R9 Full Synthesis)

## OVERALL STATE: Deep adversarial audit complete. Ready for instrumented experiments.

All claims have been stress-tested through Codex R7-R9 and 16 blind spot audits. The mechanism story has shifted from "error diversity" to "completion rate," with significant caveats documented. The experiment plan is tightened with mandatory controls.

---

## 1. EOS-COMPLETION MECHANISM (Codex R7 VALIDATED, R8+R9 REFINED)

### Core finding (substantively real):
- P(correct|EOS) = 1.000 across 100 EOS responses (94 perturbation + 6 baseline)
- ALL wrong answers come from truncated responses
- Perturbation EOS rate: 37.6% (94/250) vs baseline: 24% (6/25)
- Plurality decomposition: ALL-EOS tasks 100% correct, PARTIAL-EOS 100%, NO-EOS 37%

### Corrected token statistics (Blind Spot 14 — token counting bug):
- Perturbation EOS: mean **718 tokens** (stored: 661), range **396-1009** (stored: 345-952)
- Perturbation truncated: **1024 tokens** (stored: ~969) — hitting budget exactly
- Max EOS response: **1009 tokens** — only 15 tokens from budget!
- Bug: `inputs_embeds` path undercounts by `n_input` (~55-59 tokens)

### Caveats (Codex R8+R9):
- P(correct|EOS)=1.0 is an **extractor-conditioned selection effect**, not a universal mechanism
- EOS responses have clean answer sections; truncated responses yield random intermediates
- 95% lower bound on EOS correctness: ~0.964 (rule of three on 100 responses)
- May not generalize to harder tasks where model's self-verification is insufficient
- Think-mode template is a confound (Blind Spot 15): model's `</think>` gate = confidence proxy

### Honest claim (Codex R8 approved):
> "Perturbation sometimes finds shorter completing trajectories under a 1024-token cap."

NOT yet defensible: "Baseline is on the wrong path and more tokens will not help."

---

## 2. COMPUTE COST ANALYSIS (Codex R8 REVIEWED)

### Key numbers:
| Method | Total tokens | Tasks correct | Per correct task |
|--------|-------------|---------------|-----------------|
| Greedy 1024 | 23,358 | 8/25 (32%) | 2,920 |
| Prefix N=10 × 1024 | ~256,000 (corrected) | 18/25 (72%) | ~14,222 |
| Greedy 2048 (est.) | ~46,716 | ???/25 | ??? |

### Path vs Budget vs Verbosity (three-way distinction):
1. **Budget problem**: Model needs more tokens for a valid strategy
2. **Verbosity problem**: Model has correct answer but over-verifies before finalizing
3. **Path problem**: Model is stuck in non-terminating loop

Evidence for (2): 70% of truncated responses contain the correct answer inside `<think>` block. The model computes correctly but can't stop re-checking.

Loop detection evidence is **suggestive but not conclusive** (Codex R8) — hedging markers confounded by response length.

### Kill zone: If greedy at 2048 achieves ≥72% accuracy, prefix is dominated on cost AND accuracy.

---

## 3. TAUTOLOGY AUDIT (Codex R9 REVIEWED)

### Verdict: Substantially real, with extractor bias
- NOT a measurement artifact: the model genuinely never produces wrong final answers on EOS
- BUT: the extractor changes semantic role across stop modes (final-answer parser for EOS, tail-of-working parser for truncated)
- P(correct|EOS) is real but the 1.000 rate is partially inflated by extraction alignment

### Extractor-agnostic claim (Codex R9 approved):
> "Prefix perturbation increases the rate of responses that both terminate normally and contain a parseable final answer equal to ground truth."

---

## 4. ADVERSARIAL BLIND SPOTS (16 total, 6 new this session)

### New blind spots from this session:
| # | Severity | Finding | Status |
|---|----------|---------|--------|
| 11 | CRITICAL | Discovery-set overfitting: Wilson 95% CI for 72% is [52%, 86%] | Documented |
| 12 | HIGH | Repetition penalty confound: rp=1.2 applied to all ops; interaction with inputs_embeds path unknown | Documented |
| 13 | HIGH | P(correct|EOS) tautology: extractor-conditioned selection effect | Codex R9 reviewed |
| 14 | CRITICAL | Token counting bug: inputs_embeds path undercounts by n_input | CONFIRMED empirically |
| 15 | HIGH | Think-mode template: `</think>` gate = confidence proxy, not reasoning quality | Documented |
| 16 | MEDIUM | EOS token ID: standard HuggingFace detection, likely correct | Low risk |

### Pre-existing blind spots (from original audit):
| # | Severity | Finding |
|---|----------|---------|
| 1 | CRITICAL | Majority vote catastrophe: worse than random at all N |
| 1b | CRITICAL | Universal across both models |
| 2 | CRITICAL | Statistical power: 25 tasks underpowered for modest effects |
| 3 | HIGH | Framework over-engineering |
| 4 | HIGH | DeepSeek model-dependence bomb |
| 5 | MEDIUM | Answer normalization bottleneck |
| 6 | MEDIUM | CLAUDE.md contradiction |
| 7 | MEDIUM | Novelty risk vs Soft Reasoning |
| 8 | MEDIUM | Existing data not fully analyzed |
| 9 | HIGH | Haven't tested simplest version |
| 10 | LOW | No early stopping in long experiments |

---

## 5. PHASE 1A PRE-REGISTRATION: v2.2 (Codex R5+R7+R8 Approved)

### Changes from v2.1 → v2.2:
- H6 expanded: greedy budget sweep added (not just prefix) per Codex R8
- Must report accuracy-vs-total-tokens Pareto curves
- Token counting fix required before running

### Experiment priority (ALL Codex reviews converge):

#### Step 0 (MANDATORY — Codex R7+R8+R9): Instrumented Infrastructure
0a. **Fix token counting bug** in `inputs_embeds` path
0b. **Build instrumented generation wrapper** ensuring path equivalence:
    - Same repetition-penalty semantics for `inputs_embeds` vs `input_ids`
    - True generated-token count from both paths
    - Full raw output storage (no 2000-char truncation)
    - Verified EOS/length stop reason
    - Identical extraction inputs across all operators
0c. **Validate on 2-3 tasks**: confirm fix produces consistent counts

#### Step 1 (REQUIRED — Codex R7+R8): Token Budget Sweep (~3 hours GPU)
- **Greedy**: max_new_tokens = {512, 768, 1024, 1536, 2048} × 25 tasks × 1 generation
- **Prefix N=10**: max_new_tokens = {512, 768, 1024, 1536, 2048} × 25 tasks × 10 seeds
- Total: ~1,375 generations
- Score with: last-integer, strict final-answer, answer-anywhere
- Report: EOS rate, P(correct|EOS), accuracy, Pareto curves
- **This sweep determines whether the full Phase 1A is worth running**

#### Step 2 (Conditional on Step 1): Full Phase 1A (~6 hours GPU)
- 4 operators × 25 tasks × N=17 seeds × 2 models = ~2,600 generations
- Only proceed if Step 1 shows prefix has advantage over greedy at 2048
- Add sensitivity controls: rep_penalty=1.0 on 5-10 tasks, --no-think on 5-10 tasks

#### Step 3 (Required for publishability): Held-out tasks
- Generate 30-50 new tasks with different seed
- Run same protocol — this is the CONFIRMATORY analysis
- If held-out plurality <55%: discovery-set finding was partially anomalous

---

## 6. MECHANISM PREDICTIONS (EOS-REVISED)

### Primary predictions for Phase 1A:
| ID | Prediction | Confidence |
|----|-----------|------------|
| RP1 | EOS_rate: prefix > temp0.6 > temp1.0 > greedy | LOW — directional only |
| RP2 | P(correct\|EOS): prefix ≈ 1.0, temp0.6 < 1.0 | MEDIUM |
| RP3 | Plurality ≈ f(EOS_rate × P(correct\|EOS)) | MEDIUM |
| RP4 | Corrected EOS tokens: mean ~718, range 396-1009 | OBSERVED |
| RP5 | Answer diversity is downstream of truncation, not primary | MEDIUM |

### Critical scenarios:
| Scenario | Outcome | Paper impact |
|----------|---------|-------------|
| A: Prefix wins EOS + P(correct\|EOS) | Strong paper | Clear mechanism story |
| B: Prefix wins EOS, equal P(correct\|EOS) | Moderate paper | Completion-rate story |
| C: Temperature matches prefix | General finding | "Any diversity works" |
| D: Greedy 2048 beats everything | Devastating | Perturbation story collapses |
| E: Temperature wins via extraction | Unexpected | Revision needed |
| F: Temperature higher EOS but more wrong | Novel finding | Quality-diversity tradeoff |

---

## 7. SOFT REASONING DIFFERENTIATION (Updated)

### New framing (post-EOS):
> "Both methods increase reasoning-path completion probability. Ours: random noise + verifier-free plurality. Theirs: directed search + verifier. The question is whether random prefix perturbation at equal compute budget matches or exceeds BO-guided perturbation."

### Required comparison (future):
- 5th operator: first-output-token perturbation with random noise (Soft Reasoning proxy)
- This isolates whether perturbation POSITION matters for EOS rate

---

## 8. Documents Updated This Session

| File | Changes | Codex Status |
|------|---------|-------------|
| critical_finding_eos_mechanism.md | +Section 12 (compute cost, R8 corrected), +Section 13 (tautology, R9 reviewed), corrected token stats | R7+R8+R9 |
| phase1a_preregistration.md | v2.1→v2.2: greedy budget sweep, Pareto curves | R8 |
| prefix_vs_temperature_mechanism.md | Sections 5-10 rewritten with EOS-revised predictions | R9 reviewed |
| adversarial_self_audit.md | +6 new blind spots (#11-16) | Partially reviewed |
| codex_compute_cost_review.txt | NEW: R8 compute cost review | Complete |
| codex_tautology_review.txt | NEW: R9 tautology + path equivalence review | Complete |

---

## 9. WHAT CODEX SAYS MUST HAPPEN NEXT (Convergence of R7+R8+R9)

1. **Fix the measurement infrastructure** (token counting, path equivalence, full output storage)
2. **Run the budget sweep** (greedy + prefix, 5 budgets, 25 tasks)
3. **Only then decide** whether full Phase 1A is justified
4. **Held-out tasks** are the critical path to publishability

### What we can claim NOW (Codex R10 calibrated):
- In the Qwen 25-task discovery set, prefix perturbation produced a higher observed normal-completion rate under a 1024-token cap than greedy baseline
- In the observed 4B+8B EOS set, completed responses were correct under last-integer extraction: 134/134 (descriptive; task-clustered CIs needed)
- Plurality over extracted answers achieved 18/25 = 72% on the discovery set (Wilson 95% CI: 52-86%; no held-out confirmation)
- Current evidence points to completion/final-answer formatting as the primary mechanism; error diversity appears downstream of truncation

### What we CANNOT claim without more data:
- That prefix beats increasing the token budget (H6 required)
- That the effect generalizes beyond these 25 tasks (held-out required)
- That it works without think-mode or repetition penalty (controls required)
- That P(correct|EOS) ≈ 1.0 on harder tasks or with temperature (Phase 1A required)
- That it generalizes to "small quantized models" broadly (only tested on Qwen)
- That P(correct|EOS) = 1.0 is a stable law (it's an observation with statistical uncertainty)
