# Adversarial Self-Audit: CDE Design Blind Spots

## Status: CODEX-VALIDATED (Codex R1 confirms all findings)

## Context
After an intensive Tesla session producing 32 architectures, a converged CDE framework, and extensive design documents, this audit asks: **what are we missing?** The analysis is grounded in actual computation on existing N=10 arithmetic data, not speculation.

---

## BLIND SPOT 1 (CRITICAL): The Majority Vote Catastrophe

### Discovery
Running CDE metrics on existing N=10 random-prefix Qwen3-4B data reveals:

| Selector | N=3 | N=5 | N=7 | N=10 |
|----------|-----|-----|-----|------|
| Majority vote | 56% | 52% | 52% | 40% |
| Random pick (mean) | 60% | 58% | 55% | 52% |
| Oracle | 88% | 100% | 100% | 100% |

**Majority vote is WORSE than random selection.** It degrades from 56% to 40% as N increases. This is because:

- 13 of 19 sensitive tasks have per-seed accuracy <50%
- On these tasks, more votes = more confident wrong answer
- Only 6 sensitive tasks have >50% accuracy (where majority vote helps)
- 6 tasks are always correct (no signal either way)

### Why This Matters for CDE
CDE's primary selector (DS3: Operator-Stratified Consensus) is a variant of majority voting. If the underlying problem is that individual accuracy is too low for any voting-based method, then:

1. **DS3 doesn't fix this.** DS3 weights by operator diversity, but if ALL operators produce <50% accuracy on hard tasks, cross-operator voting still picks the wrong answer.
2. **The selector ceiling analysis** shows that even a 70%-reliable binary selector only achieves 78%. You need >80% selector reliability to beat baseline (85%).
3. **This contradicts the paper's central claim**: "a deployable selector converts decorrelation into usable gains." Decorrelation is necessary but not sufficient — you also need individual accuracy above the voting threshold.

### What CDE v2 Gets Right
CDE v2 correctly identifies selected accuracy as the primary endpoint and oracle as diagnostic. But the design ASSUMES that a good selector exists for the low-accuracy regime. It doesn't.

### The Fix That's Missing
CDE needs a selector that works in the **minority-correct regime** (individual accuracy <50%). Options:
1. **Formal verification (DS2)**: For arithmetic, eval() is a perfect verifier. This gives 100% selector reliability on verifiable tasks. But this is domain-specific and won't work for legal/planning.
2. **Self-consistency with answer normalization**: Don't vote on raw outputs — extract and normalize answers first. If Task 8 has 10% accuracy but the 1 correct seed produces "56" and the 9 wrong seeds produce 9 different wrong answers, the correct answer might still win a plurality vote.
3. **Confidence calibration**: Use logprob entropy or perplexity as a correctness signal. Low confidence → likely wrong.
4. **Domain-specific verifiers**: For legal reasoning, an LLM judge. For planning, constraint satisfaction checks.

### Impact on Paper
If DS3 can't beat random selection, the paper has no deployable result. It collapses to "oracle improvement only" — the weaker claim CDE was specifically designed to avoid.

---

## BLIND SPOT 1B (CRITICAL): Majority Vote Catastrophe Is UNIVERSAL

### 8B 8-bit N=10 (Second Model)
| Metric | 4B Q4 | 8B 8-bit |
|--------|-------|----------|
| Baseline | 32% | 16% |
| Mean prefix | 51.6% | 28.8% |
| Oracle N=10 | 100% | 80% |
| **Majority vote N=10** | **40%** | **12%** |
| Baseline-only tasks | 0 | 0 |
| Prefix-only tasks | 17 | 16 |
| Error rho (sensitive) | 0.140 | 0.148 |

The 8B model shows the SAME pattern, only worse:
- Majority vote at N=10 (12%) is WORSE than baseline (16%)
- Same zero baseline-only tasks — prefix is again a strict oracle superset
- Similar low error correlation on sensitive tasks (~0.15)

### Legal Reasoning v2 (Different Domain)
Legal data shows the identical oracle-selector gap:
- Oracle (best-of-5 perturbation) beats baseline in **11/12 tasks (92%)**
- Mean perturbation wins only **5/12 tasks**
- This is the same pattern: huge oracle potential, deployable selection unresolved

### Conclusion
**The majority vote catastrophe is universal** across models (4B, 8B), domains (arithmetic, legal), and evaluation methods (binary correct, 10-point quality scale). It's a fundamental property of the low-accuracy regime, not an artifact. Any CDE paper MUST address this.

---

## BLIND SPOT 2 (CRITICAL): Statistical Power

### The Problem
CDE Phase 1 uses 25 arithmetic tasks. We need to detect whether selected accuracy under CDE ensemble > best single operator.

### The Math
- Baseline (best single operator): ~60% (15/25 correct)
- Meaningful improvement: +2 tasks → 68% (17/25 correct)
- This is a 2/25 difference = 8 percentage points
- McNemar test on paired binary data with N=25 has terrible power for a 2-task difference
- At α=0.05, detecting a 2-task difference with 80% power requires ~100 tasks (not 25)

### What This Means
Even if CDE works, we may not be able to prove it statistically with 25 tasks. We'd need:
- A larger task set (100+ tasks) for publishable power
- OR a much larger effect size (~5+ tasks, which is 20pp)
- OR a different statistical approach (bootstrap over task sets, effect size estimation rather than hypothesis testing)

### The Fix
CDE v2 already acknowledges "25 tasks = PILOT." But the implication is clearer now: Phase 1 can only detect LARGE effects. If the effect is small (1-2 tasks), we'll get a null result even if CDE works. The paper must frame this honestly.

---

## BLIND SPOT 3 (HIGH): Framework Over-Engineering

### The Concern
A reviewer will ask: "Why do you need 32 architectures, a measurement framework, 8 operators, 4 selector tiers, and a 3,625-generation experiment to answer the question: does prefix perturbation help?"

### The Honest Answer
We DON'T need all of that. The minimum viable experiment is:
1. Run greedy baseline on 25 tasks → accuracy A
2. Run random prefix N=10 → accuracy B (with oracle C)
3. Run temperature sampling N=10 → accuracy D (with oracle E)
4. Compare A, B, C, D, E

This takes ~2-3 hours GPU, not 8-10. And it answers the core question: is prefix perturbation a distinct operator from temperature sampling?

### Why CDE Still Has Value
CDE's value is as a GENERALIZABLE framework for comparing ANY inference-time operators. But the paper should lead with the simple experiment and introduce CDE as the lens through which to analyze it — not the other way around.

### The Fix
Restructure Phase 1:
- Phase 1A (MUST RUN): Greedy vs prefix vs temperature. 3 operators, 25 tasks, ~2 hours.
- Phase 1B (NICE TO HAVE): Full 8-operator CDE comparison. 25 tasks, ~8 hours.
- If Phase 1A shows nothing, Phase 1B is wasted. If Phase 1A shows something, Phase 1B characterizes it.

---

## BLIND SPOT 4 (HIGH): The DeepSeek Problem Is A Model-Dependence Bomb

### Known Facts
- Qwen3-4B: prefix perturbation helps (+19.6pp on last-integer)
- Qwen3-8B: prefix perturbation helps (+12.8pp)
- DeepSeek-1.5B: prefix perturbation HURTS (-1.6pp mean, -2pp by metric)
- phi-2: prefix perturbation helps slightly (+6.7pp)

### The Blind Spot
The CDE framework doesn't address model-dependence AT ALL. If the method only works on Qwen models, then:
1. The paper is about Qwen-specific artifacts, not general inference-time reasoning
2. All 32 architectures are designed around an assumption that may only hold for one model family
3. The "attractor landscape" story may be a Qwen-specific phenomenon

### The Fix
CDE Phase 1 should include at least 2 model families. Ideally:
- Qwen3-4B (primary, known to work)
- DeepSeek-1.5B (known to fail — explains WHEN method doesn't work)
- One other model (phi-2, Llama, Mistral)

If the paper can explain WHY it works on Qwen but not DeepSeek, that's actually MORE interesting than showing it works everywhere.

---

## BLIND SPOT 5 (MEDIUM): Answer Normalization Is The Actual Bottleneck

### The Problem
DS3 (operator-stratified consensus) votes on normalized answers. For arithmetic: "56", "The answer is 56", "= 56" all map to 56. This is solvable.

But for legal reasoning, normalization is MUCH harder:
- Two 700-word essays that reach the same conclusion but phrase it differently
- How do you detect they're "the same answer"?
- Sentence embedding similarity? LLM judge? Exact string match on conclusion?

If answer normalization fails, DS3 fails, and selected accuracy collapses to random. This is the actual engineering bottleneck, not the operator portfolio.

### How This Interacts With Blind Spot 1
Even with perfect answer normalization, if individual accuracy is <50%, plurality voting STILL might fail. But with imperfect normalization, it's even worse — correct answers that look different get split across clusters, making it harder for the correct answer to win.

---

## BLIND SPOT 6 (MEDIUM): The CLAUDE.md Contradiction

### The Tension
CLAUDE.md says: "Automated Scorer Scores Are IRRELEVANT... The ONLY Valid Evaluation: LLM-as-Judge with Manual Review"

But CDE's entire selector stack (DS1-DS4) is automated scoring. Specifically:
- DS2 (formal verifier): automated eval()
- DS3 (operator-stratified consensus): automated voting
- DS4 (self-certainty): automated logprob analysis

### Resolution
This is actually NOT a contradiction, but it needs to be stated clearly:
- The CLAUDE.md warning is about the PROJECT'S latent scorer (the trained judge that operates in latent space)
- CDE's selectors are DIFFERENT: they operate on decoded text using principled methods (formal verification, consensus, logprobs)
- DS2 (formal verification) is 100% reliable for arithmetic — this is categorically different from a learned scorer
- DS3 (consensus) is a well-understood voting method with known failure modes (see Blind Spot 1)

The fix: Explicitly distinguish "CDE deployable selectors" from "the project's latent scorer" in all documentation.

---

## BLIND SPOT 7 (MEDIUM): Novelty Risk

### What's Actually Novel
1. **CDE framework**: A measurement contract for comparing inference-time operators. Closest work: arXiv:2502.11027 does prompt perturbation + best-of-N but without a systematic operator comparison framework.
2. **Random continuous prefix perturbation**: Existing work uses discrete prompt perturbation (word-level rephrasings). Continuous embedding perturbation is distinct.
3. **Operator-stratified consensus**: Weighting votes by operator diversity rather than raw count. This is a novel selector design.

### What's NOT Novel
1. Best-of-N resampling — very well known
2. Prefix perturbation at the token level — arXiv:2502.11027
3. Ensemble diversity matters — textbook machine learning
4. Verifier ceiling bounds — well-established

### The Risk
A reviewer could argue: "This is just best-of-N with a different perturbation method. The 'framework' adds no empirical value." The defense requires showing that CDE's metrics (decorrelation, Jaccard, trajectory classes) provide actionable insights that raw accuracy comparisons don't.

---

## BLIND SPOT 8 (MEDIUM): Existing Data We Haven't Analyzed

### Available Data Not Yet Used for CDE Analysis
1. **Per-latent per-task binary results** (N=10, 25 tasks) — NOW ANALYZED (see above)
2. **Legal reasoning v2**: 12 tasks × 5 seeds × 3 conditions — Has per-task binary results. Can compute cross-condition Jaccard and error correlation.
3. **Planning task results**: 5 tasks × 5 seeds — Small but could show if decorrelation patterns differ by domain.
4. **8B 8-bit results**: N=10, 25 tasks — Cross-model comparison of correlation structure.
5. **Full generation outputs**: Stored in JSON files. Could compute token-level trajectory similarity (a CDE metric) from existing outputs WITHOUT new GPU work.

### What Analysis Would Tell Us RIGHT NOW (No GPU Needed)
- Do legal reasoning seeds show the same correlation structure as arithmetic?
- Is the "majority vote is worse than random" pattern domain-specific or universal?
- Does the 8B model show different correlation patterns than 4B?
- Can we extract answer diversity from existing outputs to prototype DS3?

---

## BLIND SPOT 9 (HIGH): We Haven't Tested The Simplest Version

### The Untested Hypothesis
Before CDE, before 32 architectures, before any framework — the simplest test is:

> For each of the 25 tasks, does random prefix perturbation ever find the correct answer when greedy baseline doesn't?

We know the oracle is 100% (every task solved by at least one prefix seed) and the baseline is 32%. But we haven't checked: **does the baseline get the SAME 8 tasks right, or different ones?**

If the baseline gets tasks {0,1,10,11,15,16,19,24} correct (always-correct tasks + a few others), and prefix perturbation adds tasks from the Medium/Hard tier, that's the core scientific finding. No framework needed.

### COMPUTED — Results
| Category | Count | Tasks |
|----------|-------|-------|
| Both correct (baseline + prefix) | 8 | 0,1,3,7,11,14,20,24 |
| Baseline-only (baseline right, prefix always wrong) | **0** | (none) |
| Prefix-only (prefix right, baseline wrong) | **17** | 2,4,5,6,8,9,10,12,13,15,16,17,18,19,21,22,23 |
| Neither correct | 0 | (none) |
| **Combined oracle** | **25/25** | 100% |

**Key finding: Prefix perturbation is a STRICT SUPERSET of baseline in the oracle sense.** Every task baseline solves, prefix also solves. But prefix unlocks 17 additional tasks. The baseline provides zero unique coverage.

Notable: Tasks 10, 15, 16, 19 have 70-100% prefix accuracy but baseline=0%. These are tasks where the model CAN solve them but greedy decoding systematically fails. Prefix perturbation exposes latent capability.

---

## BLIND SPOT 10 (LOW): 10-Hour GPU Experiment With No Early Stopping

### The Problem
CDE Phase 1 runs all 8 operators on all 25 tasks (~3,625 generations, ~8-10 hours). There is no early-stopping or priority ordering.

### The Fix
Run operators in priority order:
1. Greedy baseline (10 min) — establishes reference
2. Random soft prefix N=16 (2 hours) — the primary operator
3. Temperature 0.6 N=16 (2 hours) — the main competitor
4. STOP: Analyze. Is prefix ≠ temperature? Are they decorrelated?
5. If yes: continue with remaining operators
6. If no: the entire CDE thesis is in trouble, don't waste 4 more hours

---

## TOP 5 PEER REVIEW KILLERS

1. **"Your method is worse than random selection."** Majority vote on your own data gives 40% vs 52% random. If you can't show a deployable selector that beats random, the paper has no practical contribution.

2. **"Single model, narrow task domain."** Qwen3-4B on 25 arithmetic tasks. The method hurts DeepSeek. No generalization evidence.

3. **"Underpowered experiment."** 25 binary tasks can't detect the effects you're claiming. The confidence intervals are too wide to be meaningful.

4. **"The framework adds nothing over a simple head-to-head comparison."** CDE's value needs to be demonstrated empirically, not just described theoretically. If a simple "prefix vs temperature vs greedy" comparison answers the question, CDE is overhead.

5. **"arXiv:2502.11027 already does this."** Prompt perturbation + best-of-N is prior art. The continuous embedding angle and operator-stratified consensus are new, but the reviewer needs to be convinced these matter.

---

## RECOMMENDED IMMEDIATE ACTIONS (Before GPU Work)

### Action 1: Analyze Legal + 8B Data Through CDE Lens
Run the same error correlation / Jaccard / majority-vote analysis on:
- Legal reasoning v2 (12 tasks, 5 seeds, 3 conditions)
- 8B 8-bit results (25 tasks, 10 seeds)
This tells us if the majority-vote catastrophe is universal or arithmetic-specific.

### Action 2: Compute Baseline-Prefix Complementarity
From existing data: what tasks does baseline solve that prefix misses, and vice versa? This is the single most important CDE number and it's computable NOW.

### Action 3: Prototype Answer Normalization on Existing Legal Outputs
Take the 60 legal reasoning outputs (12 tasks × 5 seeds) and manually check: do correct answers cluster? Can a simple normalization distinguish them? This tests whether DS3 can work for non-arithmetic domains.

### Action 4: Restructure Phase 1 Into 1A/1B
Phase 1A: greedy + prefix + temperature only (3 operators, ~2 hours GPU).
Phase 1B: full 8-operator run (only if 1A is positive).

### Action 5: Add DeepSeek to Phase 1A
Include DeepSeek-1.5B to test model dependence. This is CRITICAL for any generalization claim.

---

## WHAT 10 HOURS OF ANALYSIS WOULD HAVE YIELDED (Instead of 32 Architectures)

If we had spent the Tesla session analyzing existing data rather than designing architectures, we would know:

1. Error correlation structure across seeds (DONE NOW: rho=0.14 on sensitive tasks)
2. That majority vote fails catastrophically (DONE NOW: 40% < 52% random)
3. Whether legal reasoning shows the same pattern
4. Whether 8B shows the same pattern
5. Baseline-prefix complementarity (which tasks each solves uniquely)
6. N-scaling curves for both oracle and selected accuracy (DONE NOW)
7. Whether trajectory classes predict task difficulty
8. Whether token-level similarity differs between correct and incorrect outputs
9. Whether logprob entropy correlates with correctness (confidence calibration)
10. A complete pre-registration of Phase 1 hypotheses grounded in empirical priors

The design work wasn't wasted — CDE is a good framework. But the data analysis should have come FIRST. Codex was right: "Stop designing. Start measuring."

---

## CDE Data Summary (From Existing N=10 Qwen3-4B Random Prefix)

| Metric | Value |
|--------|-------|
| N seeds | 10 |
| N tasks | 25 |
| Mean accuracy (per seed) | 51.6% |
| Baseline accuracy | 32% |
| Oracle (N=10) | 100% |
| Majority vote (N=10) | 40% |
| Random pick (mean) | 51.6% |
| Mean error correlation (all tasks) | 0.394 |
| Mean error correlation (sensitive tasks only) | 0.140 |
| Mean Jaccard similarity | 0.545 |
| Max Jaccard similarity | 0.733 |
| Unique trajectory classes | 10/10 |
| Frozen tasks | 6 |
| Sensitive tasks | 19 |
| Tasks with <50% accuracy | 13 |
| Tasks with >50% accuracy | 6 |
| Complementary tasks per pair (mean) | 7.7/25 |

### Key Insight
The data shows high ORACLE potential (100%) but terrible SELECTOR potential with naive voting (40%). This is exactly the gap CDE claims to bridge — but it needs a selector that works in the minority-correct regime, not just a better voting scheme.

---

## CODEX R1 ADVERSARIAL AUDIT — Key Findings (gpt-5.4, 2026-04-15)

Codex independently confirmed all findings above and added:

### On Existing Data Gap
> "That is the paper in miniature: huge oracle, weak selector. It should have been the first CDE figure."
> "Legal already shows 'oracle best-of-5 looks promising, deployable selection is unresolved.' The design loop kept adding operators and abstractions instead of doing the uncomfortable selector audit on data that already exists."

### On Statistical Power (DEVASTATING)
> "In an exact paired McNemar/sign test, 2 gains, 0 losses gives two-sided p=0.50. Even 5 gains, 0 losses is still p=0.0625. You need at least 6 gains, 0 losses just to cross p<0.05, meaning the smallest clean significant effect is 24pp."
> "Power for a true 2-task/8pp effect is roughly 1-2% two-sided."
> "A 10-hour, 25-task CDE Phase 1 is underpowered for anything but large effects."

### On Framework Trap
> "Why do I need Controlled Decorrelation Ensemble, ALM, 32 architectures, and a selector protocol before you have shown prefix beats temperature, nucleus, rephrasing, and discrete prompt perturbation under equal compute?"
> "Until [the minimum publishable comparison] exists, CDE is an internal measurement harness, not the contribution."

### On Novelty Risk
> "arXiv:2502.11027 is a serious threat. It already covers diversified prompt perturbation, best-of-N scaling, diversity-fidelity tradeoffs, verifier/LLM-judge selection, and explicitly warns that majority voting may not benefit from diversity."
> "What remains potentially novel is narrower: frozen-model, inference-only, random continuous soft-prefix perturbation; deterministic greedy trajectory bifurcation from embedding perturbations."

### On DeepSeek
> "DeepSeek is not a footnote. If perturbation hurts DeepSeek, then perturbation is not a general reasoning improvement. It is a model-regime-dependent trajectory intervention."

### Codex's Top 5 Rejection Objections
1. "This is just diversified best-of-N prompt perturbation with weaker baselines." (arXiv:2502.11027 prior art)
2. "The experiment is underpowered and overfit." (25 tasks, sweet spots)
3. "The method is oracle-dependent." (Deployable selection is weak/unproven)
4. "It does not generalize." (DeepSeek degrades, model-dependence)
5. "Evaluation validity is unstable." (Last-integer scoring, truncated outputs, broken scorer history)

### Codex's Recommended Action (Instead of CDE Phase 1)
> "Run a CDE-0 offline audit before any new GPU work."
> "That would have told you the key thing: perturbation diversity is real, but CDE lives or dies on selection. The current design session mostly built machinery around the part already shown to work and under-tested the part most likely to fail."

---

## BLIND SPOT 11 (CRITICAL — Post-EOS Discovery): Discovery-Set Overfitting Risk

### The Issue
ALL results (72% plurality, 37.6% EOS rate, P(correct|EOS)=1.0) were observed on the SAME 25 tasks. Phase 1A uses the SAME 25 tasks with more seeds. No result has been validated on held-out data.

### Statistical Reality

**Plurality accuracy confidence interval**:
- Observed: 18/25 = 72%
- Wilson 95% CI: **[52.4%, 85.7%]**
- The true plurality rate could be as low as 52% — barely above baseline's 32%

**How likely is 72% if the true rate is lower?**

| True rate | P(observing ≥18/25) | Interpretation |
|-----------|--------------------|-|
| 50% | 2.2% | Unlikely but possible — we'd reject null |
| 55% | 6.4% | Marginal |
| 60% | 15.4% | Plausible — 1 in 6.5 chance |
| 65% | 30.6% | Quite possible — we may be overestimating by 7pp |
| 70% | 51.2% | Median expectation — could easily be 70% not 72% |

**EOS rate confidence intervals**:
- Perturbation: 37.6% (95% CI: [31.6%, 43.6%]) — reasonably tight at N=250
- Baseline: 24% (95% CI: [7.3%, 40.7%]) — ENORMOUS uncertainty at N=25

**Statistical power for Phase 1A (same 25 tasks, different seeds)**:
- Tasks needed for 72% vs 50% at 80% power (α=0.025): **38 tasks**
- Tasks needed for 72% vs 50% at 90% power: **50 tasks**
- Tasks needed for modest 10pp effect: **194 tasks**
- Current: 25 tasks — **UNDERPOWERED** for anything but large effects (>20pp)

### What this means
1. Phase 1A on the same 25 tasks CANNOT confirm the effect — it can only characterize the mechanism
2. The 72% figure may shrink on held-out tasks (regression toward the mean)
3. The EOS rate (37.6%) may shrink if the 25 tasks are biased toward "perturbation-friendly" expressions
4. Held-out validation (30-50 new tasks) is the CRITICAL path to a publishable result

### Task difficulty distribution concern
The 25 tasks were generated with seed=42 from 6 pattern templates (sweet_spot difficulty). The difficulty distribution depends on:
- Which pattern was randomly selected for each task
- The random operand values (e.g., 20×99 vs 20×21 have very different difficulty)
- The resulting answer magnitude and number of computation steps

A different seed could produce a task set with very different baseline accuracy and perturbation sensitivity.

### Recommendation
1. Phase 1A proceeds on these 25 tasks (mechanism characterization)
2. Held-out task generation (seed=43 or similar) MUST produce 30-50 new tasks
3. Run SAME protocol on held-out tasks — this is the CONFIRMATORY analysis
4. If held-out plurality accuracy falls below 55%: the discovery-set finding was partially anomalous
5. All paper claims must clearly distinguish discovery vs confirmatory evidence

---

## BLIND SPOT 12 (HIGH — Post-EOS Discovery): Repetition Penalty Confound

### The Issue
All generation in the experiment uses `repetition_penalty=1.2` (HuggingFace). This is applied equally to greedy baseline, prefix perturbation, and (planned) temperature sampling.

### How repetition_penalty=1.2 works
- For each previously-generated token: if its logit is positive, divide by 1.2; if negative, multiply by 1.2
- This makes previously-used tokens ~20% less likely to be re-selected
- It is designed to prevent degenerate repetition (e.g., repeating "the the the the")

### Why this could be a confound

**Interaction with the EOS mechanism**: The EOS discovery shows that truncated responses contain "verification loops" — the model repeatedly re-checks its work. But repetition_penalty could be the CAUSE of this behavior:

1. **Natural reasoning involves repetition**: Arithmetic verification naturally repeats sub-expressions ("57 × 20 = 1140, let me verify: 57 × 20..."). With `repetition_penalty=1.2`, the model is PENALIZED for repeating the exact tokens it just generated.
2. **Forced paraphrasing**: Instead of efficiently re-checking ("57 × 20 = 1140"), the model must use different words ("let me recalculate: the product of fifty-seven and twenty gives us one thousand one hundred and forty"). This uses MORE tokens for the same verification step.
3. **Longer paths → more truncation**: If repetition penalty forces the model into verbose paraphrasing, paths that would naturally complete in <1024 tokens might not.

**Interaction with perturbation**:
4. **Perturbation + repetition penalty**: Different perturbation seeds generate different initial tokens → different tokens are "used" → different penalty profiles downstream. This could AMPLIFY the diversity effect of perturbation (making seeds more independent than they would be naturally).
5. **Without repetition penalty**: Perturbation seeds might converge to more similar trajectories (the natural greedy path exerts stronger pull without penalty-induced deviation).

### Assessment of severity

| Concern | Severity | Why |
|---------|----------|-----|
| Rep penalty causes loops | MEDIUM | All operators share the same penalty, so the RELATIVE comparison is fair. But absolute EOS rates might change without it. |
| Rep penalty amplifies perturbation diversity | LOW-MEDIUM | Plausible but hard to test without data. The effect should be similar for temperature diversity. |
| Effect disappears without rep penalty | LOW | Perturbation shifts initial attention patterns, which is upstream of repetition penalty. The attention-sink rescue (planning tasks) probably doesn't depend on rep penalty. |
| Rep penalty × prefix interaction is unique | LOW | No known mechanism for why prefix would interact with rep penalty differently than temperature. |

### Recommendation for Phase 1A
- **Include `repetition_penalty=1.0` as a sensitivity analysis** (not a primary operator)
- Run on a SUBSET of tasks (5-10) to check if EOS rates change dramatically
- If EOS rates change by >10pp → report as a significant confound
- If EOS rates change by <5pp → rep penalty is not a primary driver
- This adds ~30 min GPU time for 5 tasks × 4 budgets × 10 seeds

### What this means for the paper
- **Must report**: repetition_penalty=1.2 was used in all experiments
- **Should test**: sensitivity to rep_penalty removal on a subset
- **Risk**: If the EOS effect reverses without rep penalty, the entire mechanism needs reframing
- **Mitigation**: The comparison BETWEEN operators is still fair (same penalty for all)

---

## BLIND SPOT 13 (HIGH — Post-EOS Discovery): P(correct|EOS) Tautology Risk

### The Issue
P(correct|EOS)=1.000 across 100 EOS responses may be partly an extraction artifact. See Section 13 of `critical_finding_eos_mechanism.md` for full analysis.

### Summary
- EOS responses have clean answer sections after `</think>` → last-integer extractor captures the final answer perfectly
- Truncated responses have incomplete think blocks → last-integer extractor captures intermediate values
- The extractor is an "answer-if-complete, random-intermediate-if-not" function
- P(correct|EOS)=1.0 is REAL (the model never produces wrong final answers on these tasks) but INFLATED relative to harder tasks where the model might complete but state a wrong answer

### Impact
- Phase 1A must use MULTIPLE extractors per Codex R7
- The paper should use the extractor-agnostic reformulation from Section 13.5
- Temperature sampling may produce EOS+wrong at higher rates (sampling noise corrupts intermediate steps)

---

## BLIND SPOT 14 (CRITICAL — Codex R9 CONFIRMED): Token Count Bug in inputs_embeds Path

### The Bug
The soft-prompt generation path uses `inputs_embeds` instead of `input_ids`. HuggingFace `generate()` returns output_ids containing ONLY generated token IDs (no input placeholders) when `inputs_embeds` is used. But the code computes:
```python
n_generated = output_ids.shape[1] - n_input  # n_input = soft_tokens + text_tokens
```
This UNDERCOUNTS generated tokens by `n_input` (~55-59 tokens).

### Verification
- ALL 156 truncated perturbation responses satisfy: `stored_gen + prompt_tokens == 1024` (the true budget)
- Stored gen ~969, true gen = 1024
- EOS stored mean=661, true mean=**718**, range **396-1009** (not 345-952)
- Baseline uses `input_ids` path → correct counting (gen=1024 for truncated)

### Impact on claims
| Claim | Impact |
|-------|--------|
| EOS vs truncated classification | NOT AFFECTED (based on last token, not count) |
| "EOS responses 300+ tokens shorter" | REDUCED — true gap is ~306 tokens (1024-718), not ~307 (968-661). Similar. |
| "EOS range 345-952" | WRONG — true range is 396-1009. Max is only 15 tokens from budget! |
| Compute cost analysis | UNDERSTATED — perturbation uses 1024 tokens per truncated response, not 969 |
| "Perturbation finds paths completing in 345-952" | WRONG — paths complete in 396-1009 tokens. Some barely fit within budget. |

### Critical concern: max EOS at 1009
Some EOS responses complete with just 15 tokens to spare. On slightly harder task variants, these would truncate. This means:
- The boundary between EOS and truncated is NOT sharp — it's task-difficulty-dependent
- A small increase in task difficulty could push many EOS responses into truncation
- The "EOS=correct, truncation=wrong" dichotomy is partially an artifact of the 1024 budget choice

### Required fix for Phase 1A
1. Fix token counting: for `inputs_embeds` path, use `output_ids.shape[1]` directly (no subtraction)
2. Store corrected token counts
3. Report both corrected and uncorrected for transparency

---

## BLIND SPOT 15 (HIGH — Post-EOS Discovery): Think-Mode Template as EOS Confound

### The Issue
Qwen3's thinking mode produces: `<think>...reasoning...</think>\n\nFinal Answer: X\n\nEOS`

The EOS mechanism may be entirely about the `<think>` template structure, not about reasoning capability:
1. Complete responses: `<think>reasoning</think>` → answer section → EOS
2. Truncated responses: `<think>reasoning...` (no `</think>`, no answer section)

### Empirical findings
- **Zero** of 156 truncated perturbation responses contain `</think>` in stored text
- **All** EOS responses have complete `<think>...</think>` structure
- The model only emits EOS after the answer section that follows `</think>`
- When truncated, the model is INSIDE the think block, doing verification

### What this means
P(correct|EOS) = 1.0 may be a three-step tautology:
1. EOS requires the model to emit `</think>` + answer section + EOS token
2. The model only emits `</think>` when it's satisfied with its reasoning
3. "Satisfied with reasoning" ≈ "reasoning is correct" for these tasks

This is a SELECTION EFFECT: the model's internal confidence gate (`</think>`) correlates with correctness. On tasks where the model CAN verify its own answer (arithmetic), this gate is reliable. On tasks where self-verification is harder, the gate may leak wrong answers through.

### Critical example
nest_002 L0 (truncated, wrong): The model CORRECTLY computes 6193, states "Therefore, the final answer is 6193" INSIDE the think block, then starts a re-verification pass ("Let me just cross-check once more...") and gets truncated. The model had the right answer but its confidence gate (re-verification loop) prevented it from closing `</think>`.

### Implication
- The mechanism is NOT "perturbation helps reasoning" — the model already reasons correctly
- The mechanism IS "perturbation helps the model reach its confidence threshold faster"
- On tasks where the model CAN'T verify (harder math, non-arithmetic), the confidence gate may behave differently
- Without think mode, the model structure would be entirely different — the EOS finding may not apply

### Recommendation for Phase 1A
- Add `--no-think` control on a subset (5-10 tasks) to test think-mode dependency
- If EOS rate changes dramatically without think mode → the finding is think-mode-specific
- Document think-mode as a required condition for the mechanism

---

## BLIND SPOT 16 (MEDIUM — Post-EOS Discovery): EOS Token ID Misidentification

### The Issue
The `terminated_by_eos` flag checks: `output_ids[0, -1] == eos_token_id`. But:
- What if there are MULTIPLE EOS-like tokens (e.g., `<|endoftext|>`, `<|im_end|>`, `</s>`)?
- What if the model stops generating for OTHER reasons (e.g., `max_new_tokens` reached but the last token happens to be EOS)?
- What if `eos_token_id` is set to one specific token but the model can also stop via `stopping_criteria`?

### Assessment
- Qwen3's tokenizer likely has a single canonical EOS token ID
- HuggingFace's `generate()` stops when it sees the EOS token OR hits max_new_tokens
- If the last token is EOS, generation stopped because of EOS (not budget)
- If the last token is NOT EOS, generation stopped because of budget (not EOS)
- **Risk**: LOW — the detection logic is standard and correct for HuggingFace

### Recommended verification
- Confirm Qwen3-4B's eos_token_id value
- Check if there are additional stop tokens in the tokenizer config
- Log the exact stop reason from HuggingFace generate (if available)

---

## BLIND SPOT 17 (HIGH — Codex R10): Pseudoreplication / Clustered Inference

### The Issue
Response-level counts like `94/250` and `100/100` (P(correct|EOS)) are useful descriptively but not sufficient inferentially. Responses are nested inside 25 tasks with task identity dominating difficulty. A task where 10/10 seeds reach EOS contributes 10 "successes" that are NOT independent — they're all the same easy task.

### Impact
- P(correct|EOS) = 100/100 with task-clustered bootstrap might have wider CIs than rule-of-three suggests
- EOS rate of 37.6% (94/250) is dominated by a few high-EOS tasks (nest_000, nest_001, nest_010, nest_011, nest_020, nest_024 = 60/94 EOS from 6 tasks)
- Task-paired tests (Wilcoxon signed-rank on per-task EOS rates) are the correct inferential tool

### Recommendation
- Report response-level stats as DESCRIPTIVE
- Use task-level paired tests for INFERENCE
- Add cluster-bootstrap CIs for all key claims

---

## BLIND SPOT 18 (HIGH — Codex R10): Compute-Equivalent Budget Baseline

### The Issue
The budget sweep (H6) stops greedy at 2048 tokens. But prefix N=10 at 1024 costs ~10,240 generated tokens per task (10 × 1024). A fair compute-equivalent comparison would be greedy at ~10,240 tokens — or greedy at 4096/8192 for tasks still truncated at 2048.

### Impact
- If a reviewer calculates that prefix N=10 costs 10x and greedy at 2048 only costs 2x, they'll ask "what about greedy at 4096?"
- For the 4 genuinely hard tasks, even 2048 may not suffice, but 4096 or 8192 might
- The paper needs to show the FULL compute-accuracy Pareto frontier, not just a few budget points

### Recommendation
- Extend H6 greedy sweep to at least 4096 for tasks still truncated at 2048
- Consider adaptive continuation until EOS or 8192 for the hardest tasks
- Report: accuracy vs total generated tokens, not just accuracy vs max_new_tokens

---

## BLIND SPOT 19 (MEDIUM — Codex R10): Preregistration Drift After EOS Discovery

### The Issue
The preregistration's primary hypothesis (H1) still frames margin as the key comparison. But the EOS discovery changes the primary question to: "Does prefix increase EOS rate relative to temperature?" The metric hierarchy was corrected in v2.3, but the hypotheses H1-H4 are still margin-centric.

### Impact
- A strict confirmatory analysis would evaluate H1 (margin) as the primary test, even though we now know EOS rate is the better metric
- This creates a risk: H1 might fail (margin is a weak signal) while EOS rate shows a clear effect
- The paper would then need to report "primary hypothesis not confirmed" even if the more meaningful metric shows a result

### Recommendation
- v2.3 should add H7 (primary): "Prefix produces higher EOS rate than temperature"
- H1 (margin) should be reclassified as secondary/exploratory
- This is a legitimate preregistration update BEFORE seeing Phase 1A data, so it's not p-hacking

---

## REVISED PRIORITY AFTER AUDIT (Updated Post-R10)

Based on data analysis + Codex R1-R10, the priority order:

### Original Priority (Pre-Audit)
1. CDE Phase 1 smoke test → 2. Full run → 3. Analysis

### Post-R10 Priority (CURRENT)
0. **Fix measurement infrastructure** (NO GPU): token counting bug, path equivalence, full output storage, strict final-answer parser, task-clustered analysis hooks
1. **Token budget sweep** (~3 hours GPU): greedy + prefix × {512,768,1024,1536,2048,4096*} × 25 tasks
   - *4096 for tasks still truncated at 2048 only
   - Report accuracy-vs-total-tokens Pareto curves
2. **Full Phase 1A** (~6 hours GPU): 4 operators × 25 tasks × N=17 × 2 models (CONDITIONAL on Step 1)
   - Add: rep_penalty=1.0 control, --no-think control on subset
3. **Held-out tasks** (CRITICAL for publishability): 30-50 new tasks, same protocol
4. **Soft Reasoning proxy**: 5th operator (first-output-token perturbation)
5. **DeepSeek replication** and non-Qwen model
