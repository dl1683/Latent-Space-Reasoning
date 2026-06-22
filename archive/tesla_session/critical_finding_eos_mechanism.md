# CRITICAL FINDING: EOS-Completion Is the Real Mechanism

## Status: ADVERSARIAL DISCOVERY — Changes the entire paper story. Needs Codex review.

---

## 1. The Discovery

### P(correct | EOS) = 1.000

Across ALL 250 perturbation responses (4B) and ALL 250 perturbation responses (8B):
- When the model reaches EOS (completes its response): **it is ALWAYS correct**
- When the model truncates (hits max_new_tokens): it is mostly wrong

| | 4B (n=250) | 8B (n=250) |
|---|---|---|
| EOS + correct | 94 (37.6%) | 32 (12.8%) |
| EOS + wrong | **0 (0.0%)** | **0 (0.0%)** |
| Truncated + correct | 35 (14.0%) | 40 (16.0%) |
| Truncated + wrong | 121 (48.4%) | 178 (71.2%) |
| **P(correct \| EOS)** | **1.000** | **1.000** |
| P(correct \| truncated) | 0.224 | 0.183 |

The same holds for baseline (greedy without perturbation):
- 4B baseline: 6 EOS (all correct), 19 truncated (2 lucky correct)
- 8B baseline: 2 EOS (all correct), 23 truncated (2 lucky correct)

### Token length distribution

**NOTE (Codex R9 + Blind Spot 14)**: The `inputs_embeds` generation path has a token-counting bug that UNDERCOUNTS by `n_input` (~55-59 tokens). Corrected values shown; stored values in parentheses.

| | EOS (corrected) | Truncated (corrected) |
|---|---|---|
| 4B mean tokens | **718** (stored: 661), range **396-1009** (stored: 345-952) | **1024** (stored: ~969) |
| 8B mean tokens | ~801 (stored: 744), range ~508-1017 (stored: 451-960) | **1024** (stored: ~968) |

EOS responses are ~306 tokens SHORTER than truncated ones on average. The max EOS response is 1009 tokens — only 15 tokens from the 1024 budget. This means some efficient paths barely complete within the budget.

---

## 2. What This Means for the Mechanism Story

### OLD STORY (before this finding):
"Prefix perturbation produces diverse reasoning strategies, leading to diverse wrong answers. Correct answers cluster while wrong answers are dispersed. Plurality voting exploits this dispersion."

### NEW STORY (after this finding):
"Prefix perturbation helps the model find reasoning paths that complete within the token budget. ALL completing paths reach the correct answer (the correct answer is an attractor in the model's computation space). Truncated paths produce random working-values that are naturally diverse. Plurality voting simply selects the answer from completing paths."

### Why this is simpler and more interesting:
1. **No error decorrelation theory needed**: Wrong answer diversity is just truncation noise
2. **DM13 ≈ 0.84 is expected**: Random intermediate values from different truncation points are naturally unique
3. **max(q_i) ≈ 0.20 is expected**: Random working numbers rarely cluster (except trivial values like 1, 2)
4. **Plurality voting works because**: All completing responses agree (they all reached the same correct answer via greedy decoding from different but convergent initial states)

---

## 3. Revised Mechanism: Perturbation as Computation Shortcut

### The computation path model:
1. Each perturbation seed creates a different initial state for the reasoning chain
2. Some initial states lead to DIRECT reasoning paths → model finishes in ~600-700 tokens → EOS → correct
3. Some initial states lead to LOOPING reasoning paths → model hits 1024 token limit → truncated → wrong
4. The perturbation doesn't change WHAT the model knows — it changes HOW EFFICIENTLY it accesses that knowledge

### Evidence:
- EOS responses are 300+ tokens shorter than truncated ones
- All EOS responses reach the same correct answer (deterministic convergence)
- Perturbation INCREASES EOS rate: 94/250 (37.6%) vs baseline 6/25 (24%)
- This matches the attention-sink rescue finding for planning tasks: perturbation breaks the model out of stuck loops

### Connection to existing findings:
- **Attention sink rescue** (planning task 4): Baseline generates 14 words (catastrophic loop), all perturbation seeds generate 650-710 words. Same mechanism — perturbation breaks loops.
- **DeepSeek hurting**: Perturbation might LENGTHEN DeepSeek's reasoning (making it less efficient), causing more truncation
- **Dose-response** (2 tokens optimal): Too much perturbation might disrupt the reasoning process itself, creating less efficient paths

---

## 4. Impact on Answer Diversity Metrics

### DM13 (answer-level diversity):
Previously interpreted as "diverse error strategies." 
Now: random truncation noise. Each truncated response has a different last-integer because:
- Different seed → different computation order → different intermediate values at truncation point
- No mechanism needed beyond "different truncation context"

### max(q_i) (top wrong answer mass):
Previously: "the key metric for plurality success."
Now: a consequence of random truncation noise. max(q_i) ≈ 0.20 means the most common wrong answer appears 2/10 times — exactly what you'd expect from random sampling from different working contexts.

### The REAL metric that matters: EOS rate
| Task | 4B EOS rate | 4B p | 4B plurality |
|------|------------|------|-------------|
| nest_000 | 10/10 | 1.00 | OK |
| nest_003 | 6/10 | 0.60 | OK |
| nest_006 | 5/10 | 0.50 | OK |
| nest_023 | 0/10 | 0.40 | OK (lucky extraction) |
| nest_004 | 0/10 | 0.20 | FAIL |
| nest_008 | 0/10 | 0.10 | FAIL |

**EOS rate = p for tasks with no lucky extraction.** Where p > 0, it's driven by EOS (except for lucky extractions from truncated text).

---

## 5. Impact on Phase 1A

### The key comparison changes:
- OLD: "Does prefix produce higher DM13 / lower max(q_i) than temperature?"
- NEW: "Does prefix produce higher EOS rate than temperature?"

### Temperature predictions under new mechanism:
- Temperature sampling adds noise to token selection → may cause the model to explore LONGER reasoning chains → LOWER EOS rate → WORSE plurality
- OR: temperature occasionally shortcuts past loop-causing tokens → HIGHER EOS rate → BETTER plurality
- The answer is an empirical question, but the mechanism prediction is sharper

### What Phase 1A must now report:
1. **EOS rate per operator** (prefix vs temp=0.6 vs temp=1.0 vs greedy)
2. **P(correct | EOS) per operator** — is it still 1.000 for temperature?
3. **Mean token length for EOS vs truncated responses** per operator
4. **Plurality accuracy** — should now be strongly predicted by EOS rate

---

## 6. Impact on Trivial Attractor Analysis

### 100% of trivial-value extractions (answers = 0, 1, 2, -1) come from truncated responses.

These are not "systematic attractor wrong answers." They are random numbers from reasoning text:
- "18 × 1 = 18" → last integer before truncation = 18 or 1
- "Step 1: compute..." → last integer = 1
- "carry over 1" → last integer = 1

**The trivial attractor concern is resolved**: they're truncation artifacts, not model errors. The model never "chose" 1 as its answer — it was just working through a calculation when it ran out of tokens.

---

## 7. Honest Re-Assessment of Claims

### STRENGTHENED claims:
1. "Perturbation helps the model complete computation" — directly evidenced by EOS rates
2. "The model has latent knowledge it can't access via default path" — P(correct|EOS)=1.0 means the knowledge is ALWAYS there
3. "Plurality voting is a completion-aware selector" — it picks the answer from completed responses

### WEAKENED claims:
1. "Error decorrelation is the key mechanism" — no, truncation noise IS the decorrelation
2. "Answer-level diversity (DM13) is the right metric" — no, EOS rate is the right metric
3. "The p > max(q_i) theory is a predictive model" — it's a restatement of "EOS rate > truncation noise concentration"

### UNCHANGED claims:
1. Oracle coverage demonstrates latent capacity
2. Legal domain shows oracle gap
3. Planning domain shows attention-sink rescue
4. Non-monotonic dose-response exists

---

## 8. The Revised Paper Story

### Before:
"Prefix perturbation produces diverse reasoning strategies. Wrong answers are decorrelated at the answer level. Plurality voting exploits error dispersion to select correct answers even when individual accuracy is below 50%."

### After:
"Prefix perturbation helps small quantized language models find efficient reasoning paths that complete within the token budget. When a reasoning path completes, the model reliably reaches the correct answer (P(correct|EOS) = 1.000). Incomplete paths produce incidentally diverse 'answers' — random intermediate values from reasoning-in-progress. Plurality voting works because all completed paths agree on the correct answer, while incomplete paths produce scattered noise. The perturbation doesn't teach the model anything new — it helps it navigate its existing knowledge more efficiently."

### This connects to:
- **Attention sink theory**: Default greedy paths get stuck in attention sinks (repetitive loops). Perturbation shifts initial attention distribution, avoiding sinks.
- **Soft Reasoning differentiation**: Both methods help the model find completing paths. Our contribution: (1) random noise is sufficient (no optimization), (2) completion + plurality voting is sufficient (no verifier).

---

## 9. Implications for max_new_tokens

### Currently: max_new_tokens = 1024

If the mechanism is "EOS = correct, truncation = wrong," then **increasing max_new_tokens should directly increase accuracy** (more truncated responses get to finish).

### Testable prediction:
- At max_new_tokens = 2048: EOS rate increases → accuracy increases → plurality improves
- At max_new_tokens = 512: EOS rate decreases → accuracy decreases → plurality fails
- The TRANSITION should be sharp: around the typical EOS token count (~600-700 tokens)

### This is a CHEAP experiment for Phase 1A:
Run one operator at max_new_tokens = {512, 768, 1024, 1536, 2048} and measure EOS rate + accuracy. ~15 minutes extra GPU.

---

## 10. Plurality Decomposition by EOS Category

### The clean pattern:
| Category | 4B | 8B |
|----------|----|----|
| ALL-EOS (all 10 seeds complete) | 6/6 (100%) plurality correct | 0/0 |
| PARTIAL-EOS (1-9 seeds complete) | 8/8 (100%) plurality correct | 7/7 (100%) plurality correct |
| NO-EOS (0 seeds complete) | 4/11 (36%) plurality correct | 7/18 (39%) plurality correct |
| **Total** | **18/25 (72%)** | **14/25 (56%)** |

### Interpretation:
- When **any seed reaches EOS**, plurality ALWAYS works (21/21 across both models)
- When **no seed reaches EOS**, plurality is essentially random (11/29 = 38%, driven by lucky extraction)
- The 4B's 72% advantage over 8B's 56% is entirely because 4B has fewer NO-EOS tasks (11 vs 18)

### The genuine mechanism accounts for:
- 4B: 14/25 tasks via EOS mechanism + 4/25 via lucky extraction = 18/25
- 8B: 7/25 tasks via EOS mechanism + 7/25 via lucky extraction = 14/25

### Lucky extraction is NOT a mechanism we can claim:
The 4+7 = 11 tasks where plurality succeeds without any EOS are pure noise — the correct answer happened to be the last integer in enough truncated responses. This is unreliable and wouldn't hold on new tasks.

**The honest paper claim is**: "On tasks where at least one perturbation seed produces a complete response, plurality voting reliably selects the correct answer (21/21 tasks, both models)."

---

## 11. The Extraction Gap: The Model Is Better Than We Measure

### Discovery: 70% of truncated responses contain the correct answer

Of 156 truncated responses (4B), 110 (70.5%) contain the correct answer somewhere in the reasoning text. But the last-integer extractor only captures 35 of these (22.4% of truncated, or 27 of the 110 that have the answer).

### If we had an oracle extractor: plurality goes to 100%

| Extractor | 4B Plurality | Notes |
|-----------|-------------|-------|
| Last-integer (current) | 18/25 = 72% | Misses correct answer in truncated responses |
| Oracle any-match | 25/25 = 100% | Correct answer found ANYWHERE in text |

With the oracle extractor, every task has p_any ≥ 0.3, and plurality succeeds on ALL 25 tasks.

### What this means:
1. **The model computes the correct answer almost always** — even in truncated responses
2. **The bottleneck is extraction, not reasoning**
3. **Perturbation doesn't improve reasoning** in the sense of "producing more correct computations" — the model already computes correctly ~90% of the time
4. **Perturbation improves OUTPUT FORMATTING**: it helps the model produce concise, completable responses where the correct answer is the LAST thing written

### The real mechanism (refined):
1. The model has arithmetic capability sufficient to compute the correct answer
2. On default greedy path: model produces verbose reasoning → runs out of tokens → answer buried in text
3. With perturbation: some seeds find more direct paths → model finishes → correct answer at end
4. Plurality voting + last-integer extraction = "pick the answer from seeds that finished cleanly"

### Non-oracle extractor design (for Phase 1A):
Instead of last-integer, try:
- **Repeated-value extractor**: Extract the integer that appears most frequently in the last 200 tokens
- **Answer-pattern extractor**: Find the integer following "=", "answer is", "equals" closest to end
- **Sentence-final extractor**: Find the integer in the last complete sentence

These don't require ground truth and could capture more correct answers from truncated text.

### Impact on paper claims:
- The 72% UNDERESTIMATES the model's latent capacity
- With better extraction, the gain from perturbation could be much larger
- This opens a new research direction: extraction-aware inference

---

## 12. The "Just Raise max_new_tokens" Counter-Argument (Adversarial Audit)

### The critique (Codex R7):
> "If P(correct|EOS)=1.0 and wrong answers come from truncation, why not just increase the token budget? That's simpler and cheaper than running N perturbation seeds."

### 12.1 Compute Cost Reality

| Method | Total tokens (corrected) | Tasks correct | Tokens per correct task |
|--------|--------------------------|---------------|------------------------|
| Greedy baseline (1024) | 23,358 | 8/25 (32%) | 2,920 |
| Prefix N=10 (1024) | ~256,000 (corrected*) | 18/25 (72%) ** | ~14,222 |
| Hypothetical greedy (2048) | ~46,716 | ???/25 | ??? |

\* Token counting bug (Blind Spot 14): stored values undercounted; true perturbation tokens are 1024 per truncated response, not ~969.
\** 18/25 = plurality voting accuracy (not mean individual accuracy of 51.6%)

Prefix perturbation uses **~11x more tokens** than greedy baseline (corrected), and **~4.9x more tokens per correct task**. If simply doubling max_new_tokens to 2048 could solve many more tasks at 2x the cost, that would be far more efficient than N=10 perturbation.

### 12.2 Task Classification: Where Does Each Method Help?

The 25 tasks decompose cleanly into four categories:

| Category | Count | Description | Perturbation helps? |
|----------|-------|-------------|-------------------|
| Easy | 6/25 | Baseline EOS+correct | No — greedy already works |
| Perturbation rescues | 8/25 | Baseline truncated, perturbation finds EOS | **YES — the core value** |
| Verbosity/extraction | 7/25 | No EOS, but answer present in 7-9/10 responses | Maybe — higher budget may suffice |
| Genuinely hard | 4/25 | No EOS, answer rarely present (2-4/10) | Uncertain — different approach may be needed |

**Key insight for Category 2**: In ALL 8 perturbation-rescue tasks, the correct answer appears in 8-10/10 truncated response texts. The model COMPUTES correctly — perturbation helps it FINISH sooner, producing a clean EOS response where the answer is extractable.

**Key insight for Category 3**: 7 tasks have the answer present in most responses but NO perturbation seed reaches EOS. These are the primary targets for the token budget sweep. If greedy at 2048 rescues these via longer output, the perturbation advantage narrows from 8 tasks to fewer.

**Key insight for Category 4**: 4 tasks rarely have the answer even in the reasoning text. These may be genuinely beyond the model's reliable arithmetic capability at this quantization level.

### 12.2b Where Would Higher Budget Help? (Detailed)

The 19 tasks where baseline truncates at 1024 fall into three categories (not two):

**Category A: Perturbation finds EOS paths (8 tasks)**

| Task | Pert EOS seeds | Mean EOS tokens | Max EOS tokens |
|------|---------------|-----------------|----------------|
| nest_006 | 5/10 | 735 | ~950 |
| nest_010 | 10/10 | 714 | ~950 |
| nest_014 | 3/10 | 898 | ~950 |
| nest_015 | 7/10 | 831 | ~950 |
| nest_016 | 3/10 | 871 | ~950 |
| nest_017 | 2/10 | 902 | ~950 |
| nest_018 | 3/10 | 792 | ~950 |
| nest_019 | 5/10 | 890 | ~950 |

ALL perturbation EOS responses complete within 952 tokens — well within the 1024 budget. This means **the efficient reasoning path exists within 1024 tokens**. The baseline doesn't find it because it takes the WRONG path, not because the budget is too small.

For these 8 tasks, raising max_new_tokens helps the baseline ONLY IF the longer budget allows the model to recover from its inefficient path and still produce a correct final answer. But the loop evidence (below) suggests it won't.

**Category B: NO perturbation seed reaches EOS (11 tasks)**

| Task | Pert correct/10 | Pattern: all at ~968 tokens |
|------|-----------------|---------------------------|
| nest_002 | 3/10 | Yes — all seeds truncate |
| nest_004 | 2/10 | Yes |
| nest_005 | 2/10 | Yes |
| nest_007 | 3/10 | Yes (baseline correct despite truncation) |
| nest_008 | 1/10 | Yes |
| nest_009 | 1/10 | Yes |
| nest_012 | 2/10 | Yes |
| nest_013 | 1/10 | Yes |
| nest_021 | 3/10 | Yes |
| nest_022 | 2/10 | Yes |
| nest_023 | 4/10 | Yes |

These tasks are genuinely harder. Neither the default path NOR any of 10 random perturbation seeds can find a completing path within 1024 tokens. The "correct" extractions here are lucky last-integer matches from truncated reasoning text.

For these tasks, raising max_new_tokens MIGHT help — but the loop evidence suggests otherwise.

### 12.3 Loop Detection: Suggestive but Not Conclusive (Codex R8 Corrected)

**Codex R8 correction**: Raw marker counts are confounded by length. Truncated responses are longer by construction, so they contain more hedging markers by default. The analysis below is SUGGESTIVE, not conclusive. Proper loop detection requires: per-token marker rates, matched-length controls against long EOS traces, repeated-subexpression detection, and sliding-window progress metrics.

Truncated responses show elevated markers of re-verification:

| Loop marker | Mean occurrences per truncated response |
|-------------|----------------------------------------|
| "wait" | 4.6 |
| "let me" | 6.1 |
| "actually" | 0.5 |
| "hmm" | 1.3 |
| "check" | 1.1 |
| "verify" | 0.6 |
| **Total** | **~14.2** |

Compare: EOS responses average ~3.1 hedging markers (mostly in early reasoning), then proceed to a clean conclusion.

**Per-task loop classification of 11 no-EOS tasks:**

| Task | Classification | Evidence |
|------|---------------|----------|
| nest_002 | LIKELY LOOP | High recheck count (~12-18 markers), model re-derives same intermediate values |
| nest_004 | LIKELY LOOP | Model recalculates sub-expressions repeatedly |
| nest_005 | LIKELY LOOP | Similar pattern: compute, doubt, recompute |
| nest_007 | LIKELY LOOP | Despite correct extraction, reasoning is circular |
| nest_008 | LIKELY LOOP | Very high marker count, model oscillates between steps |
| nest_009 | LIKELY LOOP | Multiple "wait, let me recalculate" sequences |
| nest_012 | PROGRESSIVE | Model appears to make forward progress but runs out |
| nest_013 | LIKELY LOOP | Re-verification dominates the token budget |
| nest_021 | LIKELY LOOP | Repetitive step-checking patterns |
| nest_022 | PROGRESSIVE | Moderate loop markers, some forward progress |
| nest_023 | POSSIBLE LOOP | Mixed signals — some forward motion, some re-checking |

**8/11 are LIKELY LOOP, 2/11 progressive, 1/11 ambiguous.**

**Codex R8 WARNING**: These classifications are unvalidated. High marker counts could reflect legitimate thorough verification rather than pathological looping. The critical test is whether increasing max_new_tokens allows these tasks to complete (favoring "verbose but valid") or whether the model produces more of the same (favoring "loop"). **This can only be resolved empirically by the H6 token budget sweep, which must include GREEDY budget sweep, not just prefix.**

### 12.4 The Critical Distinction: Path vs Budget vs Verbosity (Codex R8 Corrected)

**Codex R8 correction**: The binary path/budget distinction is premature. There are three plausible explanations, not two:

**Budget problem**: The model has the right strategy but needs more tokens to execute it.
→ Fix: increase max_new_tokens. Cheap and simple.

**Verbosity problem** (Codex R8 "middle ground"): The model has a valid strategy that produces the correct answer, but the strategy includes extensive verification that runs out the budget before the model finalizes its output. More tokens would allow the model to finish verifying and produce EOS.
→ Fix: increase max_new_tokens OR reduce verification verbosity.

**Path problem**: The model is stuck in a non-terminating loop regardless of budget.
→ Fix: change the model's initial conditions (perturbation) to find a different strategy.

**Current evidence is consistent with all three** for different tasks:
1. Perturbation EOS paths complete in 345-952 tokens — short efficient paths EXIST
2. Baseline takes all 1024 tokens and doesn't complete — could be wrong path OR verbose valid path
3. 70% of truncated responses contain the correct answer — strongly favors verbosity over wrong-path for many tasks
4. Loop markers are elevated in truncated responses — but this is confounded by length (Codex R8)

**The honest current claim** (Codex R8 approved):
> "Perturbation sometimes finds shorter completing trajectories under a 1024-token cap."

**NOT yet defensible**:
> "Baseline is on the wrong path and more tokens will not help."

The distinction between these three explanations can ONLY be resolved by the H6 token budget sweep, which must include greedy at multiple budgets.

### 12.5 Honest Assessment: What the Token Budget Sweep Will Show (Codex R8 Flagged)

**Codex R8 correction**: These predictions are "hope-shaped" — they assume loops persist. Present as EXPLICIT RISKY PREDICTIONS, not as a defense. The sweep will FALSIFY or CONFIRM them.

**RISKY Prediction for H6 — Prefix operator (max_new_tokens sweep):**

| Budget | EOS rate prediction | Confidence | Why |
|--------|-------------------|------------|-----|
| 512 | ~20% (drop from 37.6%) | HIGH | Cuts off tasks that need 512-952 tokens |
| 768 | ~30% | MEDIUM | Recovers some moderate-length tasks |
| 1024 | 37.6% (current) | OBSERVED | Baseline |
| 1536 | ~42-48% | LOW | Assumes loops persist — unproven |
| 2048 | ~45-52% | LOW | Assumes loops persist — unproven |

**CRITICAL ADDITION (Codex R8 mandatory): Greedy operator budget sweep:**

| Budget | Accuracy prediction | Confidence | Why |
|--------|-------------------|------------|-----|
| 512 | ~16-20% | HIGH | Many current EOS tasks need 500+ tokens |
| 768 | ~24-28% | MEDIUM | Recovers some shorter tasks |
| 1024 | 32% (observed) | OBSERVED | Baseline |
| 1536 | ~40-56% | LOW | Key discriminator: loops vs verbose |
| 2048 | ~44-64% | LOW | If >60%: prefix story narrows dramatically |

**The kill zone**: If greedy at 2048 achieves ≥18/25 correct (72%), prefix N=10 at 1024 is dominated on both accuracy AND cost. This is the scenario that collapses the entire perturbation narrative for arithmetic.

**Codex R8 required output**: Accuracy-vs-total-tokens Pareto curves, NOT just raw accuracy. The paper must show where each method sits on the compute-accuracy frontier.

### 12.6 The Paper Defense (Codex R8 Corrected — Downgraded)

**Codex R8 rejected the original defense as overclaiming.** Revised version:

> Reviewer: "Why not just raise max_new_tokens?"

> Response: "We include a compute-normalized comparison in Table X. At {512, 768, 1024, 1536, 2048} tokens, we report greedy accuracy alongside prefix plurality accuracy, with accuracy-vs-total-tokens Pareto curves. On our task set, perturbation finds shorter completing trajectories: EOS responses average 661 tokens (range 345-952), well within the 1024 budget. Whether greedy at higher budgets can match this depends on whether the baseline strategy is verbose-but-valid or non-terminating — our sweep addresses this directly. We note that the perturbation advantage, if it exists, is in providing N diverse path samples rather than a single deeper search, analogous to parallel restarts in optimization."

**Outstanding weaknesses** (Codex R8):
- Monte Carlo independence claim needs pairwise correlation analysis
- "N independent path samples" is too strong without effective sample size measurement
- The defense is ENTIRELY contingent on H6 results — cannot be pre-armed

---

## 13. Tautology Audit: Is P(correct|EOS)=1.0 an Artifact?

### 13.1 Code Path Analysis

The determination of `terminated_by_eos` and `correct` follows this path:

```
1. Generation produces output_ids tensor
2. terminated_by_eos = (output_ids[0, -1] == eos_token_id)
   → Checks if LAST generated token is the EOS special token
3. raw = tokenizer.decode(output_ids[n_prompt:], skip_special_tokens=True)
4. resp = raw with <think>...</think> stripped
5. correct = verify_answer(resp, expected)
   → Extracts LAST integer from resp, compares to expected
6. extracted_answer = extract_answer(resp) → last integer from resp
```

### 13.2 Potential Tautological Mechanisms

**Mechanism T1: EOS implies complete think block → clean answer section**
- EOS response: `<think>...reasoning...</think>\n\nFinal Answer: $1140$`
- `resp` after stripping: `Final Answer: $1140$`
- Last integer: 1140 (the answer the model explicitly produced)
- Truncated response: `<think>...reasoning...57 × 20 = 1140, wait let me check... 34 + 23 = 57` (cut off)
- `</think>` NOT found → resp includes think block content
- Last integer: 57 (intermediate value, NOT the final answer)

**Is this a tautology?** PARTIALLY. The extraction method (last-integer) is biased toward correctness for EOS responses because:
- EOS responses have a clean answer section AFTER the think block
- The last integer in "Final Answer: 1140" IS the answer
- Truncated responses expose intermediate arithmetic values as the "last integer"

However, this is NOT a complete tautology because:
- The model COULD complete and produce a wrong final answer → EOS + wrong
- The model COULD make an arithmetic error in its final statement
- P(correct|EOS)=1.0 means the model NEVER makes errors in its final answer for these 25 tasks
- This is a SUBSTANTIVE finding about model capability, not an extraction artifact

**Mechanism T2: EOS as confidence proxy**
- The model produces EOS when it's "done thinking"
- "Done thinking" ≈ "satisfied with answer" ≈ "answer is verified"
- So EOS ≈ "model verified its answer" → of course it's correct

This is a SELECTION EFFECT, not a measurement artifact. But it weakens the claim:
- On easy arithmetic, the model CAN verify its own answer (mental math)
- On harder tasks (multi-step with carry errors, modular arithmetic, etc.), self-verification may fail
- P(correct|EOS) < 1.0 is expected on harder tasks where the model's self-verification is unreliable

**Mechanism T3: Last-integer extractor is biased FOR EOS responses**
- EOS response format: `<think>...long reasoning...</think>\n\n**Final Answer:** $X$`
- After think-stripping: `**Final Answer:** $X$` — last integer IS the final answer
- Truncated response format: `<think>...long reasoning...intermediate calculations...`
- After failed think-stripping: entire text including all intermediate numbers
- Last integer is whatever number happens to be at the end of the reasoning chain

This IS a measurement bias. The extractor is effectively an "answer-if-complete, random-intermediate-if-not" function. This doesn't invalidate the EOS finding, but it means:
- EOS correctness is partly about the model knowing the answer + the extractor capturing it cleanly
- Truncated incorrectness is partly about the extractor failing, not just the model failing

### 13.2b Think-Close Pattern Statistics

Systematic check across all stored responses:

| Model | EOS + `</think>` | EOS total | Trunc + `</think>` | Trunc total |
|-------|-------------------|-----------|---------------------|-------------|
| 4B perturbation | 83 | 94 | **1** | 156 |
| 4B baseline | — | 6 | 0 | 19 |
| 8B perturbation | 20 | 32 | 0 | 218 |
| 8B baseline | — | 2 | 0 | 23 |

The EOS responses missing `</think>` in stored text (11 for 4B, 12 for 8B) have the tag beyond the 2000-char storage truncation.

**The one exception**: nest_006 L8 (4B perturbation, truncated=true, correct=false). This response:
1. Closed `</think>` at char 1701 after extensive reasoning
2. Started producing a clean step-by-step answer section
3. Got truncated mid-answer ("$1626 \% 27$. Break...") — the answer section's formatting was too verbose
4. Last integer in stored text: 11 (wrong, from partial "Step 3" calculation)
5. Correct answer: 6

This is the VERBOSITY PROBLEM in pure form: the model found the right reasoning path, closed the think gate, but its output formatting consumed too many tokens. With just ~50 more tokens, this response would have been EOS+correct.

### 13.3 What Would Falsify P(correct|EOS)=1.0?

1. **Strict final-answer parser on EOS responses**: If a parser looking for "Final Answer: X" or "= X" in the answer section disagrees with last-integer on ANY EOS response → there's extraction noise
2. **Temperature EOS responses with wrong answers**: If temp=0.6 produces completed responses where the model confidently states a wrong answer → P(correct|EOS) is task/method-dependent, not universal
3. **Harder arithmetic tasks**: Tasks where the model's self-verification is insufficient (e.g., 5-step nested operations) might produce EOS+wrong
4. **Non-arithmetic tasks**: Legal/planning EOS responses cannot be extracted by last-integer, so the entire framework doesn't apply

### 13.4 Verdict: Substantially Real, Partially Tautological

**What's real (strong claim):**
- The model reliably computes correct answers for these 25 arithmetic tasks when it completes its reasoning
- Completion rate (EOS rate) is the primary bottleneck, not reasoning capability
- Perturbation increases completion rate compared to greedy baseline (37.6% vs 24%)

**What's partially tautological (weaker claim):**
- P(correct|EOS)=1.000 is inflated by the extractor bias: EOS responses have a clean answer section that the last-integer extractor captures perfectly, while truncated responses have messy intermediate values
- The TRUE accuracy of completed reasoning might be 0.95-0.99, not 1.000 — but we can't distinguish without a stricter parser
- The 22.4% accuracy on truncated responses is DEFLATED by the same bias: the model may have the correct answer in its reasoning but the extractor picks up an intermediate value

**What this means for the paper:**
- Report P(correct|EOS) with a caveat about extraction method alignment
- Phase 1A must use MULTIPLE extractors (last-integer, strict final-answer, answer-anywhere) per Codex R7
- The 5-bucket classification (EOS+correct, EOS+wrong, trunc+correct-stated, trunc+correct-somewhere, trunc+no-correct) is essential for disambiguating real mechanism from extraction artifact
- **The strongest version of the claim avoids P(correct|EOS) entirely**: "Perturbation increases the rate at which the model produces responses that terminate with a correctly-formatted final answer."

### 13.5 The Extractor-Agnostic Reformulation

Instead of "P(correct|EOS)=1.0" (which is extraction-dependent), the paper should use:

**Claim**: "Under prefix perturbation, N=10 random seeds produce at least one response that (a) reaches end-of-sequence and (b) contains the correct answer as the final stated result, for 14/25 tasks (4B) vs 6/25 baseline tasks. Plurality voting over all N responses selects the correct answer in 18/25 tasks."

This formulation:
- Doesn't depend on P(correct|EOS) being exactly 1.0
- Acknowledges that EOS rate is the mechanism driver
- Keeps plurality voting as the selection mechanism
- Is robust to extractor choice

---

## 14. Open Questions for Codex Review

1. Does the tautology analysis in Section 13 correctly identify the extraction bias? Is there a deeper confound?
2. Is the "extractor-agnostic reformulation" (Section 13.5) sufficient, or does it lose too much punch?
3. How does temperature affect the EOS + extraction alignment? (Temperature may produce EOS responses with non-standard formatting → last-integer extractor fails)
4. For the 2 "progressive" no-EOS tasks (nest_012, nest_022): would 1536 or 2048 tokens allow completion?
5. Is the Monte Carlo analogy apt, or does it overstate the independence of perturbation paths?
6. How does the compute cost argument change at N=17 (Phase 1A)? At N=5 (lower cost)?
7. What is the minimum N needed for reliable plurality? Is there a phase transition?
8. Should we run a small manual audit of 10-20 EOS responses with a strict final-answer parser before Phase 1A?
