**Gate Verdict**

Phase 2.1 addresses the 7 Round 3 requirements at the architecture-document level. I would now unblock Phase 3 and Phase 5, but only as Phase A stress-test and hypothesis work. Do not treat this as approval to run the hard MI gate on existing logs or to begin Phase B router work.

The remaining gaps are implementation-spec gaps, not conceptual blockers: exact RMS field naming, sequence alignment for `inputs_embeds`, blinded qualitative judging, and a few missing provenance fields.

**1. Seven Corrections**

The 7 corrections map cleanly to the Round 3 required changes:

1. RMS scaling: addressed, but see subtle issue below.
2. Existing logs pilot-only: addressed.
3. `output_scores=True` and logprob extraction: addressed in intent.
4. Task-held-out MI with bootstrap CIs: addressed.
5. Atlas contamination fields: mostly addressed.
6. No-prefix baseline: addressed.
7. Qualitative labels: addressed with Option A.

Still incomplete:

- The correction document uses `rms_scale: 0.022` in the atlas, but in [run_latent_sensitivity.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_latent_sensitivity.py:1143), `rms_scale` is a multiplier, not the target RMS. The atlas should store `embedding_rms`, `rms_multiplier`, and `effective_rms`.
- The `inputs_embeds` sequence slicing must be empirically tested. The codebase has inconsistent assumptions: the sensitivity script decodes `output_ids[0, :]`, while [encoder.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/src/latent_reasoning/core/encoder.py:969) skips `prompt_len`.
- Qualitative judging needs an explicit blinding rule: judge sees output text and task/rubric only, not seed, prefix metadata, early features, candidate order, or baseline identity.

**2. RMS Scaling**

Assuming the missing expression in your prompt is `sp.square().mean().sqrt()`: yes, that is tensor-level RMS:

```python
sqrt(mean(sp ** 2))
```

over all elements in the soft-prefix tensor, i.e. batch × prefix tokens × embedding dimensions.

The validated script does this:

```python
target_rms = cal["embedding_rms"]
effective_rms = target_rms * args.rms_scale
sp = torch.randn(1, num_soft_tokens, embed_dim, generator=noise_gen)
current_rms = sp.square().mean().sqrt().clamp_min(1e-8)
sp = sp * (effective_rms / current_rms)
```

So `0.022` is a rounded shorthand for Qwen3 embedding RMS, not the exact mechanism. The actual Qwen3-4B value in existing result files is about `0.021953146904706955`; Qwen3-8B is about `0.022050151601433754`.

Therefore: the correction is mathematically right, but still subtly wrong if implemented with literal `0.022` or if the atlas calls that value `rms_scale`. Use calibrated `effective_rms`.

**3. Oracle-Relative Label**

Option A is the right qualitative MI label for the first legal/planning pilot because it asks the operational question: can early features identify the candidate the judge would select as best within the same task group?

Its weakness is that it is relative, not absolute. It can degenerate when all candidates are bad, all candidates tie, the judge is noisy, or the judge mostly rewards length/style. In that case MI may measure “predicts judge preference among weak outputs,” not “predicts quality.” Add tie handling, score margins, blinded candidate order, and report how often the oracle winner is only marginally better than the runner-up.

**4. EarlyFeatures Contract**

Complete enough to start implementation, not complete enough to trust without added fields/tests.

Add:

- `prompt_token_count`, `combined_input_length`, `generation_start_index`
- `prefix_token_count`, `prefix_positions`
- actual generated token IDs from `output.sequences`, not inferred by argmax
- exact semantics for `ended_before_window`, `eos_in_window`, and `max_new_tokens_reached`
- `feature_extractor_version`
- either define hidden-state features or remove `hidden_states_available` from the MI contract
- slope behavior for `n_observed < 2`

The `inputs_embeds` path exists in [encoder.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/src/latent_reasoning/core/encoder.py:943), but it is a latent-projected soft prompt path, not the raw-prefix Phase A telemetry path. Phase A should build a dedicated raw-prefix applicator.

**5. Atlas Guard**

Mostly sufficient for the 10 Round 3 corruption risks. The remaining guardrails I would add:

- `torch`, `transformers`, `bitsandbytes`, CUDA, driver versions
- prompt template hash and full task input hash
- feature extractor version
- judge blinding flag and candidate shuffle/order hash
- `effective_rms` separately from RMS multiplier
- explicit “missing attention/logprob was not imputed” policy in analysis metadata

Also freeze the feature list and windows before inspecting labels.

**6. Minimum New Data**

Existing arithmetic/legal/planning generations count as zero for pilot MI because they lack token IDs, logprobs, entropy, attentions, hidden states, and full traces.

Minimum for an arithmetic pilot MI:

- 100 task groups
- 10 random-prefix candidates per group
- 1 no-prefix baseline per group
- total: 1,100 new candidate generations with Phase A telemetry

If reusing old arithmetic prompts, you may rerun the existing 25 task groups with telemetry and add 75 fresh groups, but all 100 groups must be newly generated through Phase A. Do not pool arithmetic, legal, and planning to reach 100; they are different task classes and label regimes.

For legal and planning pilots separately:

- legal: rerun 12 existing prompts with telemetry plus 88 more groups, or collect 100 fresh groups
- planning: rerun 5 existing prompts with telemetry plus 95 more groups, or collect 100 fresh groups

Practical RTX 5090 plan:

1. Smoke test: 5 arithmetic groups × baseline+3 prefixes, `max_new_tokens=256`, validate schema and slicing.
2. Pilot: 100 arithmetic groups × baseline+10 prefixes, Qwen3-4B Q4, greedy, full output text/token IDs, early telemetry through 128 tokens.
3. Use batch size 1 and checkpoint one JSONL record per candidate.
4. Aggregate attentions/hidden states online or only for early windows; do not retain full 1024-token attention tensors.
5. Only after arithmetic works, repeat for legal/planning with blinded judge scoring.

**7. Gate Decision**

Yes: proceed to Phase 3 and Phase 5.

But the approved scope is:

- Phase 3: Phase A telemetry stress test
- Phase 5: formalize H1-H6 and preregister estimator/features/windows

Not approved yet:

- hard MI gate
- Phase B Observer/Router
- any claim based on old logs beyond crude text-only pilot debugging

**8. Priority Directive**

The single most important Phase A implementation requirement is sequence-aligned telemetry from the raw `inputs_embeds` prefix path.

Every candidate record must prove that the same run produced the soft prefix, prompt boundary, generated token IDs, selected-token logprobs, entropy, attention aggregates, full output, and validation label. If that alignment is off by one token, or if RMS is hardcoded instead of calibrated, the MI result is not interpretable.