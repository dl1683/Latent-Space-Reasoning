**Bottom Line**

CDE is ahead of the evidence. The existing data already says the central problem is not “can perturbations create diverse successful trajectories?” It is “can any deployable selector reliably pick them?” Right now the answer is mostly unproven, and in some existing data it looks bad.

Also: `memory/MEMORY.md` is not present in this workspace. I read the requested session docs that exist, plus the relevant experiment summaries/raw files.

**1. Existing Data Gap**

The N=10 Qwen3-4B Q4 arithmetic data can answer several CDE questions now:

- Prefix mean accuracy: `51.6%` vs greedy baseline `32%`.
- Best single prefix seed: `60%`.
- Prefix oracle: `25/25 = 100%`.
- Strict majority over correctness: only `10/25 = 40%`; tie-random is about `11/25 = 44%`.
- Estimated oracle scaling from task sensitivities:
  `N=1: 51.6%`, `N=2: ~67%`, `N=3: ~76.8%`, `N=5: ~88.6%`, `N=10: 100%`.

That is the paper in miniature: huge oracle, weak selector. It should have been the first CDE figure.

The 12 legal tasks say the same thing:

- Baseline average: `4.12/10`.
- Perturbation average: `4.32/10`, only a small mean lift.
- Best perturbation average: `5.73/10`.
- Best perturbation beats baseline on `11/12` tasks.
- But random perturbation average wins only `5/12` tasks, and the evolution scorer was broken on `9/12`.

So legal already shows “oracle best-of-5 looks promising, deployable selection is unresolved.” Why hasn’t this been done? Because the design loop kept adding operators and abstractions instead of doing the uncomfortable selector audit on data that already exists.

**2. Statistical Power**

No, CDE Phase 1 cannot detect a 2-task difference on 25 binary tasks with significance.

A 2-task lift is `8pp`. In an exact paired McNemar/sign test, `2 gains, 0 losses` gives two-sided `p=0.50`. Even `5 gains, 0 losses` is still `p=0.0625`. You need at least `6 gains, 0 losses` just to cross `p<0.05`, meaning the smallest clean significant effect is `24pp`.

Power for a true 2-task/8pp effect is terrible. Under an optimistic no-regression model, two-sided power is roughly `1-2%`; one-sided is roughly `4-5%`. A looser independent-proportion approximation is still only around `8-10%`.

So yes: a 10-hour, 25-task CDE Phase 1 is underpowered for anything but large effects. It can screen for “obvious win” or “obvious failure.” It cannot adjudicate subtle operator differences.

**3. Framework Trap**

Yes, CDE is over-engineered relative to the current evidence.

A reviewer will ask: “Why do I need Controlled Decorrelation Ensemble, ALM, 32 architectures, and a selector protocol before you have shown prefix beats temperature, nucleus, rephrasing, and discrete prompt perturbation under equal compute?”

The minimum publishable comparison is:

- greedy baseline
- soft prefix N-matched
- temperature/nucleus N-matched
- prompt rephrase N-matched
- random token prefix N-matched
- same deployable selector across all

Until that exists, CDE is an internal measurement harness, not the contribution.

**4. Novelty Risk**

arXiv:2502.11027 is a serious threat. It already covers diversified prompt perturbation, best-of-N scaling, diversity-fidelity tradeoffs, verifier/LLM-judge selection, and explicitly warns that majority voting may not benefit from diversity. That overlaps with much of CDE’s story.

COCONUT also weakens any broad “continuous latent reasoning” novelty claim: it already frames reasoning beyond language tokens using continuous hidden states.

What remains potentially novel is narrower:

- frozen-model, inference-only, random continuous soft-prefix perturbation
- deterministic greedy trajectory bifurcation from embedding perturbations
- task-level solve-set redistribution/oracle coverage on small quantized models
- quantization/model-specific trajectory effects
- empirical oracle-selector gap for continuous perturbations

That is not “new reasoning architecture.” It is “continuous prefix perturbations expose latent trajectory diversity, but deployable selection is the bottleneck.”

**5. DeepSeek Problem**

DeepSeek is not a footnote. If perturbation hurts DeepSeek answer-anywhere `84% -> 78%`, then perturbation is not a general reasoning improvement. It is a model-regime-dependent trajectory intervention.

That invalidates every design that assumes soft prefix is a privileged primary operator. At best, prefix becomes one candidate in a routed operator portfolio. At worst, it is a cherry-picked Qwen3-4B Q4 phenomenon.

For a reviewer, DeepSeek means: show model-family robustness, or explicitly scope the paper to “model-specific trajectory diversification.”

**6. CLAUDE.md Tension**

There is a real contradiction unless you separate selector from evaluator.

`CLAUDE.md` says automated scorer scores are irrelevant for assessing quality. CDE says deployable selectors choose outputs. That is acceptable only if:

- selector scores are treated as part of the method, not evidence of quality
- final quality is evaluated by ground truth or blind manual/LLM-judge review
- legal/open-ended tasks do not use the broken latent scorer as a truth proxy

The legal experiment already proves the danger: the automated evolution scorer failed on most tasks. If CDE claims legal improvement via automated selection, this paper is dead.

**7. Top 5 Rejection Objections**

1. “This is just diversified best-of-N prompt perturbation with weaker baselines.” arXiv:2502.11027 already covers much of the framing.

2. “The experiment is underpowered and overfit.” 25 tasks cannot support architecture conclusions, sweet spots, or small lift claims.

3. “The method is oracle-dependent.” Your best results are oracle/best-of-N; deployable selection is weak or unproven.

4. “It does not generalize.” DeepSeek degrades, 8B/4B quantization behaves differently, and the Qwen3-4B result may be a lucky regime.

5. “Evaluation validity is unstable.” Last-integer scoring, answer-anywhere gaps, truncated outputs, legal hallucinations, and broken scorer history all weaken “reasoning improvement” claims.

**8. What Should Have Been Done Instead**

Run a CDE-0 offline audit before any new GPU work:

1. Compute prefix-only oracle/selected curves from existing N=10 Qwen data.
2. Compute pairwise correct-set Jaccard, unique solves, seed stability, and majority/plurality selector accuracy.
3. Simulate deployable selectors from existing extracted answers, EOS status, token count, time, and answer clusters.
4. Reanalyze legal: can any non-oracle rule pick the best perturbation among B1-B5?
5. Do the power analysis and resize Phase 1 before spending 8-10 GPU hours.

That would have told you the key thing: perturbation diversity is real, but CDE lives or dies on selection. The current design session mostly built machinery around the part already shown to work and under-tested the part most likely to fail.

Sources: local `CLAUDE.md`, `tesla_session/SESSION_STATUS.md`, `tesla_session/cde_measurement_protocol_v2.md`, `experiments/analysis_summary.md`, `experiments/CRITICAL_ANALYSIS.md`, `experiments/cross_model_task_analysis.md`, `experiments/legal_v2_results_summary.md`; arXiv [2502.11027](https://arxiv.org/html/2502.11027), [COCONUT / 2412.06769](https://arxiv.org/abs/2412.06769), and [Verification Limits Code LLM Training](https://arxiv.org/abs/2509.20837).tokens used
198,774
**Bottom Line**

CDE is ahead of the evidence. The existing data already says the central problem is not “can perturbations create diverse successful trajectories?” It is “can any deployable selector reliably pick them?” Right now the answer is mostly unproven, and in some existing data it looks bad.

Also: `memory/MEMORY.md` is not present in this workspace. I read the requested session docs that exist, plus the relevant experiment summaries/raw files.

**1. Existing Data Gap**

The N=10 Qwen3-4B Q4 arithmetic data can answer several CDE questions now:

- Prefix mean accuracy: `51.6%` vs greedy baseline `32%`.
- Best single prefix seed: `60%`.
- Prefix oracle: `25/25 = 100%`.
- Strict majority over correctness: only `10/25 = 40%`; tie-random is about `11/25 = 44%`.
- Estimated oracle scaling from task sensitivities:
  `N=1: 51.6%`, `N=2: ~67%`, `N=3: ~76.8%`, `N=5: ~88.6%`, `N=10: 100%`.

That is the paper in miniature: huge oracle, weak selector. It should have been the first CDE figure.

The 12 legal tasks say the same thing:

- Baseline average: `4.12/10`.
- Perturbation average: `4.32/10`, only a small mean lift.
- Best perturbation average: `5.73/10`.
- Best perturbation beats baseline on `11/12` tasks.
- But random perturbation average wins only `5/12` tasks, and the evolution scorer was broken on `9/12`.

So legal already shows “oracle best-of-5 looks promising, deployable selection is unresolved.” Why hasn’t this been done? Because the design loop kept adding operators and abstractions instead of doing the uncomfortable selector audit on data that already exists.

**2. Statistical Power**

No, CDE Phase 1 cannot detect a 2-task difference on 25 binary tasks with significance.

A 2-task lift is `8pp`. In an exact paired McNemar/sign test, `2 gains, 0 losses` gives two-sided `p=0.50`. Even `5 gains, 0 losses` is still `p=0.0625`. You need at least `6 gains, 0 losses` just to cross `p<0.05`, meaning the smallest clean significant effect is `24pp`.

Power for a true 2-task/8pp effect is terrible. Under an optimistic no-regression model, two-sided power is roughly `1-2%`; one-sided is roughly `4-5%`. A looser independent-proportion approximation is still only around `8-10%`.

So yes: a 10-hour, 25-task CDE Phase 1 is underpowered for anything but large effects. It can screen for “obvious win” or “obvious failure.” It cannot adjudicate subtle operator differences.

**3. Framework Trap**

Yes, CDE is over-engineered relative to the current evidence.

A reviewer will ask: “Why do I need Controlled Decorrelation Ensemble, ALM, 32 architectures, and a selector protocol before you have shown prefix beats temperature, nucleus, rephrasing, and discrete prompt perturbation under equal compute?”

The minimum publishable comparison is:

- greedy baseline
- soft prefix N-matched
- temperature/nucleus N-matched
- prompt rephrase N-matched
- random token prefix N-matched
- same deployable selector across all

Until that exists, CDE is an internal measurement harness, not the contribution.

**4. Novelty Risk**

arXiv:2502.11027 is a serious threat. It already covers diversified prompt perturbation, best-of-N scaling, diversity-fidelity tradeoffs, verifier/LLM-judge selection, and explicitly warns that majority voting may not benefit from diversity. That overlaps with much of CDE’s story.

COCONUT also weakens any broad “continuous latent reasoning” novelty claim: it already frames reasoning beyond language tokens using continuous hidden states.

What remains potentially novel is narrower:

- frozen-model, inference-only, random continuous soft-prefix perturbation
- deterministic greedy trajectory bifurcation from embedding perturbations
- task-level solve-set redistribution/oracle coverage on small quantized models
- quantization/model-specific trajectory effects
- empirical oracle-selector gap for continuous perturbations

That is not “new reasoning architecture.” It is “continuous prefix perturbations expose latent trajectory diversity, but deployable selection is the bottleneck.”

**5. DeepSeek Problem**

DeepSeek is not a footnote. If perturbation hurts DeepSeek answer-anywhere `84% -> 78%`, then perturbation is not a general reasoning improvement. It is a model-regime-dependent trajectory intervention.

That invalidates every design that assumes soft prefix is a privileged primary operator. At best, prefix becomes one candidate in a routed operator portfolio. At worst, it is a cherry-picked Qwen3-4B Q4 phenomenon.

For a reviewer, DeepSeek means: show model-family robustness, or explicitly scope the paper to “model-specific trajectory diversification.”

**6. CLAUDE.md Tension**

There is a real contradiction unless you separate selector from evaluator.

`CLAUDE.md` says automated scorer scores are irrelevant for assessing quality. CDE says deployable selectors choose outputs. That is acceptable only if:

- selector scores are treated as part of the method, not evidence of quality
- final quality is evaluated by ground truth or blind manual/LLM-judge review
- legal/open-ended tasks do not use the broken latent scorer as a truth proxy

The legal experiment already proves the danger: the automated evolution scorer failed on most tasks. If CDE claims legal improvement via automated selection, this paper is dead.

**7. Top 5 Rejection Objections**

1. “This is just diversified best-of-N prompt perturbation with weaker baselines.” arXiv:2502.11027 already covers much of the framing.

2. “The experiment is underpowered and overfit.” 25 tasks cannot support architecture conclusions, sweet spots, or small lift claims.

3. “The method is oracle-dependent.” Your best results are oracle/best-of-N; deployable selection is weak or unproven.

4. “It does not generalize.” DeepSeek degrades, 8B/4B quantization behaves differently, and the Qwen3-4B result may be a lucky regime.

5. “Evaluation validity is unstable.” Last-integer scoring, answer-anywhere gaps, truncated outputs, legal hallucinations, and broken scorer history all weaken “reasoning improvement” claims.

**8. What Should Have Been Done Instead**

Run a CDE-0 offline audit before any new GPU work:

1. Compute prefix-only oracle/selected curves from existing N=10 Qwen data.
2. Compute pairwise correct-set Jaccard, unique solves, seed stability, and majority/plurality selector accuracy.
3. Simulate deployable selectors from existing extracted answers, EOS status, token count, time, and answer clusters.
4. Reanalyze legal: can any non-oracle rule pick the best perturbation among B1-B5?
5. Do the power analysis and resize Phase 1 before spending 8-10 GPU hours.

That would have told you the key thing: perturbation diversity is real, but CDE lives or dies on selection. The current design session mostly built machinery around the part already shown to work and under-tested the part most likely to fail.

Sources: local `CLAUDE.md`, `tesla_session/SESSION_STATUS.md`, `tesla_session/cde_measurement_protocol_v2.md`, `experiments/analysis_summary.md`, `experiments/CRITICAL_ANALYSIS.md`, `experiments/cross_model_task_analysis.md`, `experiments/legal_v2_results_summary.md`; arXiv [2502.11027](https://arxiv.org/html/2502.11027), [COCONUT / 2412.06769](https://arxiv.org/abs/2412.06769), and [Verification Limits Code LLM Training](https://arxiv.org/abs/2509.20837).
