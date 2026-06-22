**Verdict**

Phase A is the right architecture direction, but the revised design is **not ready as written** for Phase 3/5. It fixes the conceptual mistakes from Round 2, but it still has one critical implementation error and one critical experiment-design error:

1. The proposed soft-prefix RMS normalization is wrong.
2. The MI gate cannot be run from the existing logs as specified.

Gate decision: **do not proceed to Phase 3/5 until a short Phase 2.1 correction is made.** This is no longer an architectural block; it is a data-validity block.

**1. Required Revisions**

Phase A addresses the six Round 2 requirements mostly, but not completely.

- Hypotheses marked: yes. `attention_sink_mass`, 32-token classification, and 6-feature routing are now explicitly hypotheses in [phase2_revised.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase2_revised.md:27).
- Multi-surface removed: yes. Mainline is now `input_prefix` only, with multi-surface excluded in [phase2_revised.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase2_revised.md:345).
- Observer contract: partial. Phase A defines a generation telemetry contract, but not a real future Observer/KV-cache continuation contract. That is acceptable only because Phase B is explicitly not built yet.
- Fallback fixed: mostly. The design rejects `encoder.decode()` in [phase2_revised.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase2_revised.md:350), but the “all candidates abstained -> best by mean_logprob” fallback in [phase2_revised.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase2_revised.md:375) must be marked operational-only, not validation evidence.
- Unbiased atlas logging: yes in intent. It logs all candidates and uses `selected_as_best` as a flag, not a filter, in [phase2_revised.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase2_revised.md:274).
- MI/AUROC/oracle-recall gates: yes in intent, but missing confidence intervals, task-held-out splits, and sample-size requirements.

**2. MI Experiment**

The acceptance thresholds are directionally appropriate but too weak unless statistical reliability is added.

- `MI > 0.1 bits` is a reasonable kill/proceed lower bound, not proof of routability.
- `>=90% oracle recall at <=50% promotion` is the real operational gate. MI alone must not pass Phase B.
- AUROC per feature is useful descriptively. A single-feature `AUROC > 0.65` should not justify a router by itself.
- The router gate must be evaluated **within task groups**. Candidate-level `I(features; correct)` can be inflated by task difficulty. The router needs to retain the winning candidate among candidates for the same query.

The methodology needs correction:

- Use a pre-registered MI estimator, not “discretization or k-NN” interchangeably after seeing results.
- Use permutation nulls and bootstrap confidence intervals.
- Split by `task_id`, not candidate row, or the same prompt leaks across train/test.
- Evaluate the promotion policy on held-out tasks.
- For qualitative legal/planning tasks, define the target label before collection: binary pass/fail, top-rubric score, above-baseline improvement, or oracle-winner rank. Right now `correct` is `None` for qualitative outputs, so the MI target is undefined.

Existing logs are insufficient for the official MI gate. The validated sensitivity script stores text snippets and coarse metadata only: `response[:2000]`, `response_raw[:2000]`, correctness, generated token count, and EOS status in [run_latent_sensitivity.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_latent_sensitivity.py:1384). It does not store exact token IDs, logprobs, entropy, attentions, hidden states, or full raw traces needed for the proposed `EarlyFeatures`.

Existing logs can support a crude text-only pilot. They cannot support the hard gate.

Minimum sample size: current arithmetic `25 tasks x 10 seeds = 250 candidates` is enough to debug the pipeline, not enough for a hard MI gate. I would require at least:

- **Pilot:** >=100 task groups x 10 candidates per model/task class.
- **Hard gate:** preferably 200-500 task groups x 10 candidates per model/task class.
- Each label class should have at least 100 positive and 100 negative candidate records, and oracle recall should be bootstrapped by task, not by candidate.

Legal/planning with 12 and 5 tasks respectively is far below reliable MI scale.

**3. Component Contracts**

Component 2 has a critical bug. The revised pseudo-code uses:

```python
noise = noise * (rms_scale / noise.norm(dim=-1, keepdim=True))
```

That sets each token’s L2 norm to `0.022`, not its RMS. For `embed_dim=2560`, the per-dimension RMS becomes about `0.022 / sqrt(2560)`, roughly 50x too small.

The validated script scales by tensor RMS:

```python
current_rms = sp.square().mean().sqrt().clamp_min(1e-8)
sp = sp * (effective_rms / current_rms)
```

See [run_latent_sensitivity.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_latent_sensitivity.py:1157). This must be fixed before data collection.

The `inputs_embeds` path exists in the encoder, but only as latent-projected soft prompt text generation, not Phase A telemetry. `decode_with_soft_prompt()` projects a latent through `soft_prompt_projector` in [encoder.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/src/latent_reasoning/core/encoder.py:893), then calls `generate(inputs_embeds=...)` in [encoder.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/src/latent_reasoning/core/encoder.py:943). It returns only text. Phase A needs a new raw-prefix applicator.

Component 2 also needs `output_scores=True` to compute logprobs and entropy. `output_attentions=True` and `output_hidden_states=True` are not enough.

Component 3 is sufficient for a first cheap-feature MI pass, but incomplete. Add:

- explicit definitions for `attention_sink_mass` aggregation by layer/head/token;
- `n_observed` and `feature_available` flags per window;
- top-k logits or score summaries;
- prompt length and baseline/no-prefix candidate features;
- group-relative features within the same task;
- exact handling for slope at `w=1`.

Component 4 is sound in principle. Abstain-over-heuristic is the correct rule. But if all candidates abstain, returning `mean_logprob` must not create a validation label or enter MI as correctness.

Component 5 needs more contamination guards. Add model revision/HF commit, quantization config, generation config, code commit, schema version, actual soft-prefix hash, judge prompt/rubric hash, task hash, and train/test split ID. `model_id + tokenizer_hash + judge_model_version` is not enough.

**4. Phase A Value**

Phase A is valuable by itself. It fixes the gap between the validated research mechanism and the package path. It also creates a reproducible best-of-N validation system for arithmetic and a proper atlas for legal/planning.

But Phase A is not yet a cheap deployed router. It is a full-generation validated candidate system. Its value is scientific rigor, oracle-ceiling measurement, and validated selection when a real output validator exists.

**5. Risks That Could Corrupt Phase B**

The main corruption risks are:

- wrong RMS scaling;
- running the MI gate on old logs without logprobs/attentions/token IDs;
- treating candidate rows as independent instead of task clusters;
- using full-output or truncation-derived information as “early” features;
- choosing windows/features after inspecting labels;
- judge drift or unblinded qualitative scoring;
- mixing model/tokenizer/quantization/generation configs;
- undefined qualitative target labels;
- imputation of unavailable attentions;
- failing to log the no-prefix baseline candidate.

**6. Gate Decision**

No, not as written.

Required changes before Phase 3/5:

1. Fix soft-prefix RMS scaling to match the validated mechanism exactly.
2. State that existing logs are pilot-only; the hard MI gate requires new Phase A telemetry collection.
3. Add `output_scores=True`, exact logprob/entropy extraction, and exact sequence-slicing tests for `inputs_embeds`.
4. Define task-held-out MI/router evaluation with bootstrap CIs.
5. Complete atlas contamination fields, including model revision, quantization config, generation config, code commit, soft-prefix hash, task hash, and judge/rubric hash.
6. Add baseline/no-prefix as a logged candidate.
7. Define qualitative labels before using legal/planning in MI.

After that, proceed to Phase 3 as a **Phase A stress test**, not as Phase B implementation.

The hypotheses to formalize after the corrections are:

- H1: early features at 64 tokens contain `>0.1 bits` of task-held-out information about final correctness for arithmetic.
- H2: early features predict convergence/truncation more strongly than answer-anywhere computation for Qwen3-4B Q4.
- H3: Qwen3-8B Q8 shows more answer-anywhere/computation signal than Qwen3-4B Q4.
- H4: `attention_sink_mass` predicts collapse/truncation better than semantic correctness.
- H5: a held-out promotion rule can retain `>=90%` of oracle winners while promoting `<=50%` of candidates.
- H6: qualitative legal/planning will have weaker early correctness MI than arithmetic, even if it has length/collapse MI.

**7. Priority Directive**

The single most important Phase A implementation decision is this:

Build the raw-prefix applicator and candidate record as the source of truth, with exact validated RMS scaling, exact generation config, exact per-token telemetry, and immutable atlas logging.

If that mechanism is even slightly different from the validated `run_latent_sensitivity.py` path, the Phase B gate will measure a new system while pretending to measure the old one.