**Gate Verdict**

Proceed to Phase 6, but only to write the Blueprint. Do not start full Phase A implementation or data collection until two amendments are made:

1. Add a missing Phase 3 stress test for generation-path confounding: position IDs/RoPE offset, KV-cache continuation, and telemetry observer effects.
2. Tighten Phase 5 preregistration around the MI estimator, feature normalization, H5 candidate universe, and sample-size/inconclusive rules.

The core direction is sound. The current documents are good enough to blueprint the system, not yet good enough to execute the preregistered experiment.

**Code Validation**

RMS is correctly implemented in the sensitivity script as a multiplier over calibrated embedding RMS, not as an absolute RMS value: [run_latent_sensitivity.py:1143](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_latent_sensitivity.py:1143), with tensor RMS normalization at [run_latent_sensitivity.py:1159](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_latent_sensitivity.py:1159). Phase A should log `embedding_rms`, `rms_multiplier`, `effective_rms`, and `actual_prefix_rms`.

The `inputs_embeds` sequence assumption is not validated. Worse, the repo contains incompatible assumptions. The sensitivity path computes generated length as if `output_ids` includes the prompt, then decodes the whole sequence: [run_latent_sensitivity.py:152](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_latent_sensitivity.py:152), [run_latent_sensitivity.py:163](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_latent_sensitivity.py:163). `encoder.py` explicitly skips `combined_embeds.size(1)`: [encoder.py:969](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/src/latent_reasoning/core/encoder.py:969), [encoder.py:970](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/src/latent_reasoning/core/encoder.py:970). So yes: sequence misalignment is actually the highest risk.

**1. Stress Test Validity**

The 9 stress tests are the right set and mostly correctly ranked. Sequence alignment deserves the top slot because an off-by-prompt bug silently corrupts every windowed feature, every selected-token logprob, and every MI result.

The critical missing failure mode is generation-path confounding. Prepending raw soft tokens changes more than content: it can shift RoPE/position IDs for all prompt tokens, change attention-mask-derived positions, alter KV-cache layout, and produce different kernels when telemetry is enabled. Phase 3 needs a new stress test proving:

- no-prefix Phase A baseline equals direct `input_ids` generation byte-for-byte;
- zero-prefix and masked-prefix controls separate prefix content from position shift;
- telemetry-enabled generation produces the same token sequence as lean generation;
- two-phase “observe first 128, continue lean” generation is exact.

The VRAM calculation is directionally right but not exact. With cached generation, attention storage is closer to `layers * heads * sum(prompt_len + t)` across generated steps, not simply one full final square matrix, and not full square per step. Still, full 1024-token `output_attentions=True` is GB-scale, slow, and may disable optimized attention. The first-128 mitigation is sufficient only if implemented as a custom loop or exact KV-cache continuation, then verified against lean generation.

Oracle-relative degeneracy is real for legal/planning, less relevant for arithmetic. A winner-vs-median margin helps, but it is not enough. Add an absolute quality floor, winner-vs-runner-up margin, tie policy, and judge-stability check. Otherwise H6 can become “predicts judge style” rather than “predicts quality.”

**2. Hypothesis Validity**

H1-H6 are the right hypothesis families: primary MI, mechanism decomposition, cross-model mechanism, attention-collapse sanity check, operational routing, and qualitative-domain transfer.

But H1’s `>0.1 bits` threshold is only a pilot-level effect-size threshold. With 100 task groups, 10 features, clustered candidates, binary labels, and k-NN estimation, it is not calibrated enough for a hard kill/proceed gate unless the observed effect is far above threshold and the CI is clean. Keep `>0.1 bits`, but require a positive-control/null calibration before treating it as decisive.

H5’s `>=90% oracle recall at <=50% promotion` is operationally correct: a router that drops too many oracle winners is useless. But the spec is inconsistent: data has baseline + 10 prefixes, while H5 says top-5 of 10. Decide whether baseline is excluded. Also, 30 held-out groups gives a wide recall CI; 27/30 looks like 90% but is not a strong estimate.

H2 is testable only after tightening the labels. Raw MI comparison between `converged` and `answer_anywhere_correct` is confounded by different base rates and nested structure. Use normalized MI and a conditional target: `converged_given_anywhere`.

The composite routing weights are arbitrary. Pre-registration prevents tuning leakage, but it does not make the weights principled. Treat this as a frozen heuristic baseline. You must also freeze feature normalization, probably train-split z-scoring or within-task ranks. Without normalization, `mean_logprob`, entropy, and attention mass live on incompatible scales.

**3. Preregistration Completeness**

KSG k-NN with `k=5` is acceptable as a pilot estimator for continuous features, but standard KSG is not the right estimator for a binary target. Use a mixed discrete-continuous MI estimator, or explicitly name an implementation such as a classifier-target k-NN MI estimator and convert nats to bits. Permutation testing helps significance, but does not fix biased absolute MI.

The preregistration must specify feature scaling, missing-value handling, tie/jitter policy, binary feature handling, exact estimator library/version, random seeds, one-sided vs two-sided tests, multiple-window correction, and whether MI is multivariate over all 10 features or reported per feature.

The 70/30 split is too thin for both MI estimation and H5. Since the feature set is frozen, H1 can use all 100 groups with task-bootstrap CIs, while H5 should use a larger held-out set or be declared pilot-only. If you keep 70/30, do not overclaim recall precision.

For independent replication, add exact model revision, quantization config, tokenizer/chat-template config, prompt generator seed, full task manifests, soft-prefix seed/hash protocol, position-ID policy, attention implementation, package versions, judge model/version/rubric, candidate shuffle hash, exclusion rules, schema version, and analysis commit hash.

**4. Data Collection Plan**

The sequence smoke test → 100 arithmetic groups → legal/planning order is sound, but add a preflight suite before the 100 groups:

- baseline equivalence;
- `inputs_embeds` generation boundary;
- selected-token score alignment;
- RMS assertion;
- telemetry-vs-lean equivalence;
- JSONL crash/resume validation.

100 arithmetic groups × 11 candidates is enough for a feasibility pilot and for detecting a large H1 signal. It is not enough for a stable near-threshold hard gate, especially with 10D k-NN MI. If the measured MI is around 0.08-0.14 bits, the result should be considered ambiguous. For a publishable hard gate, expect 200-300 groups.

For H3, use paired task groups across 4B-Q4 and 8B-Q8. 100 paired groups is a pilot. I would require about 200 paired groups minimum, and closer to 300 if the expected cross-model MI difference is small.

Cheaper H1 signal path: run 25-30 arithmetic groups × baseline + 5 prefixes, collect only token IDs, scores/logprobs, entropy through w=64, skip attentions/hidden states, and cap generation lower if arithmetic answers complete reliably. Treat it as a non-confirmatory feasibility pilot.

**5. Phase A Standalone**

Phase A alone is publishable if it becomes a reproducible candidate-generation and telemetry atlas system with clean empirical findings. It is deployable only as an offline oracle/candidate evaluation pipeline, not as an online router.

To make Phase A stronger standalone work, add position-shift controls, zero/masked-prefix controls, RMS dose-response, token-count dose-response, a hardened atlas schema, public analysis scripts, and the decomposition `P(correct) = P(anywhere) * P(converged | anywhere)` across model/quantization settings.

**6. Blueprint Requirements**

Phase 6 Blueprint should contain:

- exact `RawSoftPrefixApplicator` tensor contract;
- position-ID and attention-mask policy;
- `generation_start_index` discovery and validation;
- score/logprob alignment contract;
- telemetry collection design for first 128 tokens;
- atlas schema and atomic JSONL/resume behavior;
- validator contracts for arithmetic and qualitative tasks;
- MI estimator implementation details;
- split, normalization, CI, permutation, and exclusion rules;
- preflight test matrix;
- explicit “no Phase B unless H1 and H5 pass” gate.

Phase 3 should add the generation-path confounding stress test. Phase 5 should be amended before any confirmatory data collection.

**7. Priority Directive**

Resolve this first in Phase A code: a minimal raw `inputs_embeds` generation probe that proves sequence boundary and score alignment.

It must output and assert `prompt_token_count`, `prefix_token_count`, `combined_input_length`, `generation_start_index`, generated token IDs, `len(scores)`, selected-token logprobs, decoded generated text, actual prefix RMS, and baseline equivalence. Until that probe passes, every other Phase A component is downstream of an untrusted token boundary.