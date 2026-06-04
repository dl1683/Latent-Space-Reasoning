# Diffusion Reader Guide

This guide is the navigation layer for the diffusion reasoning work. The repo
has a lot of generated evidence, so use this file to decide what to read first.

## Fast Path

If you only have ten minutes:

1. [README.md](../README.md): current public score/cost headline and entry paths.
2. [DIFFUSION_PUBLIC_BENCHMARK.md](../DIFFUSION_PUBLIC_BENCHMARK.md): the
   public three-arm benchmark: greedy/fixed, random perturbation, latent repair.
3. [docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md](DIFFUSION_REASONING_GEOMETRY_THEORY.md):
   the mathematical theory: denoise trajectories, information loss, proof
   obligations, and error functions.
4. [docs/DIFFUSION_THEORY_CLAIM_LEDGER.md](DIFFUSION_THEORY_CLAIM_LEDGER.md):
   theory assertions mapped to evidence, assumptions, falsifiers, and next proof
   obligations.
5. [CLAIM_EVIDENCE_MAP.md](../CLAIM_EVIDENCE_MAP.md): evidence ledger for every
   public claim.
6. [DIFFUSION_GROUND_TRUTH_INDEX.md](../DIFFUSION_GROUND_TRUTH_INDEX.md):
   canonical score/report/raw artifact pointers and content hashes.

## Current Public Claims

| Claim | Read |
| --- | --- |
| Top-score latent repair beats greedy/fixed and random on the lean GPU mixed benchmark: `0.531116` at `2.625000x`. | [DIFFUSION_PUBLIC_BENCHMARK.md](../DIFFUSION_PUBLIC_BENCHMARK.md), [CLAIM_EVIDENCE_MAP.md](../CLAIM_EVIDENCE_MAP.md) |
| Budget-favored value-proxy repair reaches `0.508705` at `2.375000x`, matching cheap-tier cost while scoring higher. | [DIFFUSION_BUDGET_VALUE_PROXY_AUDIT.md](../DIFFUSION_BUDGET_VALUE_PROXY_AUDIT.md), [DIFFUSION_REPAIR_VALUE_GEOMETRY.md](../DIFFUSION_REPAIR_VALUE_GEOMETRY.md) |
| The mechanism is denoise-trajectory repair, not broad sampling. | [DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md](../DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md), [DIFFUSION_DENOISE_PHASE_GEOMETRY.md](../DIFFUSION_DENOISE_PHASE_GEOMETRY.md) |
| Public evidence is claim-gated and hash-indexed. | [CLAIM_EVIDENCE_MAP.md](../CLAIM_EVIDENCE_MAP.md), [DIFFUSION_GROUND_TRUTH_INDEX.md](../DIFFUSION_GROUND_TRUTH_INDEX.md) |

## Theory Layer

Read these when evaluating the deeper research direction:

| Document | What It Gives You |
| --- | --- |
| [DIFFUSION_REASONING_GEOMETRY_THEORY.md](DIFFUSION_REASONING_GEOMETRY_THEORY.md) | Formal objects, information-loss definition, propositions, proof sketches, and candidate error functions. |
| [DIFFUSION_THEORY_CLAIM_LEDGER.md](DIFFUSION_THEORY_CLAIM_LEDGER.md) | Conservative ledger of theory claims, evidence, assumptions, falsifiers, and next proof obligations. |
| [DIFFUSION_REASONING_FIELD_IMPLICATIONS.md](DIFFUSION_REASONING_FIELD_IMPLICATIONS.md) | Public narrative for why diffusion changes the latent-reasoning substrate. |
| [DIFFUSION_RESEARCH_TRANSLATION_NOTES.md](DIFFUSION_RESEARCH_TRANSLATION_NOTES.md) | Translation from world-model, prediction-error, and local research signals into runnable operators. |
| [DIFFUSION_NATIVE_REASONING_ARCHITECTURE.md](DIFFUSION_NATIVE_REASONING_ARCHITECTURE.md) | Running architecture log for schedules, repair operators, source policies, gates, and selectors. |

Core theory claim:

`reasoning improvement = controlled reduction of task-relevant information loss over an editable denoise trajectory`

This means the important research objects are not only final answers. They are:

- denoise-state feature geometry
- verifier-visible information loss
- marginal repair value
- source trust and retention
- anchor realization
- energy-aware repair spending
- promotion value after a repair is generated

Current proof direction: train and test a decomposed error function with
separate availability, promotion-value, source-trust, retention, and cost heads.
This is the repo's path from observed GPU benchmark gains to deeper claims
about the geometry of diffusion reasoning space.
The newest larger transfer slice adds one more term to the availability head:
repair availability is relative to the selected trajectory state. Read
[DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V3.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V3.md)
for the 16-planning-row evidence and the pointer to the live
`trajectory_relative_decomposed_spend` CUDA run.
[DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md](../DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md)
then turns that boundary into a learned trigger,
`learned_availability_predictor_v1`, with a matching CUDA confirmation.
The next fresh slice,
[DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V4.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V4.md),
is the guardrail: the learned v3 cutoff does not transfer. It blocks all three
historically labeled v4 repairs, and under corrected repair-only labels still
fails as a stable availability rule. The availability head is now a boundary
condition, and promotion needs its own post-repair target.
The calibrated trigger then gets its own fresh slice in
[DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V5.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V5.md).
It improves score over fixed and random, but the corrected repair-only labels
show three calibrated availability errors and remaining oracle headroom. The
next proof step is now explicit:
[DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md](../DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md)
separates post-repair promotion from pre-repair spend gating.
The fresh v6 continuation keeps that split:
[DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V6.md](../DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V6.md)
again gives `candidate_aware_promotion_v1` zero promotion errors, while
[DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V6.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V6.md)
shows calibrated availability misses spend decisions. The generated
[DIFFUSION_SPEND_POLICY_DECISION.md](../DIFFUSION_SPEND_POLICY_DECISION.md)
then makes the current cost decision explicit: keep `candidate_aware_promotion_v1`
fixed, use denoise-phase repairability as the incumbent spend trigger, and test
any learned spend gate offline against accumulated transfer targets before
buying another full GPU slice. The fresh v7 run did that no-retuning check:
[DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V7.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V7.md)
shows the incumbent still beats fixed and random but spends on six no-lift
repairable rows, while
[DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V7.md](../DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V7.md)
keeps the post-repair promotion head at zero errors.
[DIFFUSION_SPEND_GATE_V7_FIT.md](../DIFFUSION_SPEND_GATE_V7_FIT.md)
then searches simple offline gates over v5/v6/v7 and finds no deployable
replacement yet: the best rule still has five errors. The v8 and v9
counterexample probes make that failure sharper rather than resolving it:
[DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V9.md](../DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V9.md)
keeps `candidate_aware_promotion_v1` at zero errors on generated candidates,
but [DIFFUSION_SPEND_GATE_V9_FIT.md](../DIFFUSION_SPEND_GATE_V9_FIT.md) still
has 12 simple-gate errors across 40 accumulated spend rows. Read
[DIFFUSION_SPEND_COUNTEREXAMPLE_WORKBENCH.md](../DIFFUSION_SPEND_COUNTEREXAMPLE_WORKBENCH.md)
next: it names the low-gap profitable false negatives and high-gap no-lift
false positives that any future spend controller must explain. The first richer
offline challenger,
[DIFFUSION_SPEND_VALUE_MODEL_V1.md](../DIFFUSION_SPEND_VALUE_MODEL_V1.md),
is a negative control: nearest-prototype pre-repair geometry protects most
positives only by spending too broadly and still drops a held-out positive.
[DIFFUSION_SOURCE_VALUE_SIGNALS_V1.md](../DIFFUSION_SOURCE_VALUE_SIGNALS_V1.md)
then audits source-text structure directly. It also is not deployable, but it
shows the next useful channel is source realization/degeneracy rather than more
prompt-gap tuning.
[DIFFUSION_SOURCE_DEGENERATION_AUDIT_V1.md](../DIFFUSION_SOURCE_DEGENERATION_AUDIT_V1.md)
splits that channel into repeated-token, punctuation-run, and meta-leakage
defects. It keeps the gate closed: degeneration is explanatory evidence, not a
standalone spend trigger, because defective sources include both repairable
positives and no-lift traps.
[DIFFUSION_PRE_REPAIR_EDGE_PROXY_V1.md](../DIFFUSION_PRE_REPAIR_EDGE_PROXY_V1.md)
then measures the named pre-repair edge-proxy question directly: can frozen
source and span diagnostics predict candidate promotion edge before live repair
spend?
[DIFFUSION_COUNTERFACTUAL_CONTROLLER_ARCHITECTURE_V1.md](../DIFFUSION_COUNTERFACTUAL_CONTROLLER_ARCHITECTURE_V1.md)
is the next architecture contract: frozen features can only triage whether to
buy a cheap counterfactual observation; they cannot promote full repair spend.
[DIFFUSION_COUNTERFACTUAL_PROBE_TARGETS_V1.md](../DIFFUSION_COUNTERFACTUAL_PROBE_TARGETS_V1.md)
turns that contract into the first replayable probe-target sheet for the named
counterexamples.
[DIFFUSION_COUNTERFACTUAL_PROBE_POLICY_FIT_V1.md](../DIFFUSION_COUNTERFACTUAL_PROBE_POLICY_FIT_V1.md)
then fits the deterministic scaffold offline. It finds a one-error rule, but
keeps the result diagnostic until measured micro-probe deltas replace the
generated predictions.
[DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_RUNNER_HOOK_V1.md](../DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_RUNNER_HOOK_V1.md)
adds the runner-facing `counterfactual_micro_probe_v1` trigger. It records
measured probe diagnostics in gate rows while forcing `should_run=false`, so
probe triage cannot accidentally become full repair spend.
[DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_SMOKE_V1.md](../DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_SMOKE_V1.md)
is the first GPU smoke for that hook: one `plan_070` probe generation, measured
deltas in the gate row, and no repair score credit.
[DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_COUNTEREXAMPLES_V1.md](../DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_COUNTEREXAMPLES_V1.md)
then runs the measured hook across all 12 named v5-v9 counterexamples. It buys
seven cheap probes, skips five no-lift rows, records zero triage errors, and
still keeps `should_run=false`.
[DIFFUSION_COUNTERFACTUAL_MEASURED_PROBE_VALUE_POLICY_V1.md](../DIFFUSION_COUNTERFACTUAL_MEASURED_PROBE_VALUE_POLICY_V1.md)
then reruns the same counterexample set in all-shadow probe mode so negatives
also get measured probe deltas. The measured-only Stage 1 rule still has two
errors, while the zero-error all-feature rule is just the old prompt-gap /
`would_probe` Stage 0 boundary, so full repair spend remains blocked.
[DIFFUSION_COUNTERFACTUAL_PROBE_TEXT_FIDELITY_V1.md](../DIFFUSION_COUNTERFACTUAL_PROBE_TEXT_FIDELITY_V1.md)
then treats the measured probe text as tomography. It finds four malformed
`FULL_REPAIR_AUTHORIZED=false` strings, five punctuation/spelling-defect rows,
and a best post-probe text rule with two errors. The next probe has to be a
stable diagnostic instrument before it can be a value-of-information controller.
[DIFFUSION_COUNTERFACTUAL_TOMOGRAPHY_PROBE_TEXT_FIDELITY_V1.md](../DIFFUSION_COUNTERFACTUAL_TOMOGRAPHY_PROBE_TEXT_FIDELITY_V1.md)
adds the strict `strict_tomography_probe_v1` prompt policy and reruns the 12-row
all-shadow GPU measurement. The sentinel improves to 12/12 exact and zero
malformed authorization rows, but only 8/12 rows pass the stricter Stage 1
validity gate and the best post-probe text rule still has one false positive.
[DIFFUSION_COUNTERFACTUAL_VALIDATED_PROBE_STAGE1_GATE_V1.md](../DIFFUSION_COUNTERFACTUAL_VALIDATED_PROBE_STAGE1_GATE_V1.md)
then applies the actual controller discipline: invalid probe rows cannot be
used as value evidence. Under that validity-required rule, three profitable
rows become invalid-positive misses and the best validated Stage 1 rule has
five errors. Full repair spend remains blocked.
[DIFFUSION_COUNTERFACTUAL_KEY_VALUE_PROBE_TEXT_FIDELITY_V2.md](../DIFFUSION_COUNTERFACTUAL_KEY_VALUE_PROBE_TEXT_FIDELITY_V2.md)
and
[DIFFUSION_COUNTERFACTUAL_KEY_VALUE_VALIDATED_PROBE_STAGE1_GATE_V2.md](../DIFFUSION_COUNTERFACTUAL_KEY_VALUE_VALIDATED_PROBE_STAGE1_GATE_V2.md)
test a `key_value_tomography_probe_v2` prompt that removes placeholder
exemplars. It lowers the validated gate fit to two false negatives, but raw
diagnostic reliability regresses to 6/12 valid rows and two malformed
authorization strings, so it is a negative control rather than a promoted probe.
[DIFFUSION_COUNTERFACTUAL_COMPACT_PROBE_TEXT_FIDELITY_V3.md](../DIFFUSION_COUNTERFACTUAL_COMPACT_PROBE_TEXT_FIDELITY_V3.md)
and
[DIFFUSION_COUNTERFACTUAL_COMPACT_VALIDATED_PROBE_STAGE1_GATE_V3.md](../DIFFUSION_COUNTERFACTUAL_COMPACT_VALIDATED_PROBE_STAGE1_GATE_V3.md)
then test `compact_tomography_probe_v3`: short `A=`, `B=`, `C=`, `Z=false`
slots restore 12/12 runtime-format-valid rows and zero malformed
authorizations at a 48-token / 24-step budget, but the stricter semantic audit
finds only 4/12 rows valid: seven template echoes, five malformed compact-key
rows, and one duplicate authorization row. Under semantic-required Stage 1
validity, five profitable rows become invalid-positive misses and the best
validated rule has six errors, so v3 is not a live spend gate.
[DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_TEXT_FIDELITY_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_TEXT_FIDELITY_V4.md)
and
[DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_STAGE1_GATE_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_STAGE1_GATE_V4.md)
then test `span_tomography_probe_v4`: copied-span `X0=`, `X1=`, `X2=`, `N=0`
slots remove the v3 template-echo failure mode. The same 12-row all-shadow CUDA
surface now has 10/12 semantic-valid rows, zero malformed authorizations, zero
template echoes, one malformed span key, and zero invalid-positive misses. The
first semantic-valid Stage 1 fit has one false-positive no-lift row; adding an
X0/X2 slot-overlap penalty to retention-risk visibility then yields a zero-error
local validated rule. That is a fresh-slice challenger, not a promoted spend
gate.
[DIFFUSION_COUNTERFACTUAL_PROBE_TRANSFER_TARGETS_V1.md](../DIFFUSION_COUNTERFACTUAL_PROBE_TRANSFER_TARGETS_V1.md),
[DIFFUSION_COUNTERFACTUAL_SPAN_DISTINCT_RETENTION_RULE_V4_FRESH_PLANNING.md](../DIFFUSION_COUNTERFACTUAL_SPAN_DISTINCT_RETENTION_RULE_V4_FRESH_PLANNING.md),
and
[DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_TRANSFER_MATRIX_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_TRANSFER_MATRIX_V4.md)
then close that obligation on an independent `plan_001`-`plan_008` planning
slice. The fresh GPU run has 8/8 semantic-valid probe rows, but the frozen local
rule `valid_measured_distinct_retention_risk_visibility_ge_0p927195` has three
transfer errors: false positives on `plan_002` and `plan_003`, and a false
negative on `plan_004`. The gate remains `diagnostic_only`; the next controller
must learn a cross-slice value-of-information feature rather than reusing this
retention threshold.
[DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_CONJUNCTION_TRANSFER_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_CONJUNCTION_TRANSFER_V4.md)
then exhaustively searches two-condition validated probe rules as a diagnostic
next-hypothesis generator. The train-ranked winner is still the failing
distinct-retention threshold, but transfer screening finds a challenger:
`measured_expected_gap_visibility_gain >= 0.666667` and
`measured_expected_span_evidence_gain <= 0.600000`. It has one named-surface
error and zero errors on the fresh planning slice, so it is a frozen next-slice
hypothesis, not a promoted gate.
[DIFFUSION_COUNTERFACTUAL_SPAN_GAP_SPAN_RULE_V4_TRANSFER_V3_PLANNING.md](../DIFFUSION_COUNTERFACTUAL_SPAN_GAP_SPAN_RULE_V4_TRANSFER_V3_PLANNING.md)
then freezes that challenger on the next non-overlapping `plan_017`-`plan_024`
slice. The probe remains mechanically clean with 8/8 semantic-valid rows, but
the rule has three strict transfer errors: false positives on `plan_020` and
`plan_023`, plus a false negative on positive `plan_018`. The conjunction is
retired as a controller candidate; future work needs a richer probe signature or
learned value-of-information model rather than another hand threshold.
[DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_MODEL_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_MODEL_V4.md)
then audits that next step across all three span-v4 slices with a nearest-prototype
signature model over measured probe value, gap, realization, span, retention,
slot overlap, source quality, and probe-text validity features. It preserves all
15 positive rows in sample and under leave-one-slice-out, but it still selects
11 no-lift negatives. That makes it useful diagnostic evidence, not a live spend
gate; the next model must learn no-lift specificity or an explicit cost penalty
before any promotion.
[DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_UTILITY_FRONTIER_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_UTILITY_FRONTIER_V4.md)
adds that explicit cost penalty as a threshold frontier over the same
leave-one-slice-out signature scores. The result is still a negative gate:
selection penalties through `0.020000` keep the broad 26-row policy with 11 false
positives, while `0.050000` drops to three selected rows but misses 13 positives.
The failure is now sharper: the current signature score is recall-biased and
does not expose a cheap no-lift cutoff.
[DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_NO_LIFT_VETO_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_NO_LIFT_VETO_V4.md)
then tests the obvious two-stage controller: keep the high-recall signature
selection, but train a separate one- or two-condition veto over selected rows to
drop likely no-lift spends. Leave-slice-out transfer rejects it. The veto lowers
false positives from 11 to 8, but introduces five false negatives, reducing
practical-penalty utility from `0.625500` to `0.289429`. The next model needs
signed realized-lift prediction, not threshold-fragment vetoing.

## Benchmark And Cost Layer

Read these when checking whether the claim is cheap, reproducible, and not
hidden behind a large search stack:

| Document | What It Gives You |
| --- | --- |
| [docs/LEAN_GPU_DIFFUSION_BENCHMARK_PROTOCOL.md](LEAN_GPU_DIFFUSION_BENCHMARK_PROTOCOL.md) | The allowed GPU protocol and reproduction commands. |
| [DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md](../DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md) | The validated denoise phase-window cost ladder. |
| [DIFFUSION_BUDGET_POLICY_LOSS.md](../DIFFUSION_BUDGET_POLICY_LOSS.md) | Cost-aware selector objective: `utility = aggregate_score_lift - lambda * marginal_relative_cost`. |
| [DIFFUSION_BUDGET_VALUE_PROXY_AUDIT.md](../DIFFUSION_BUDGET_VALUE_PROXY_AUDIT.md) | Label-free value proxy and fresh CUDA confirmation. |
| [DIFFUSION_REPAIR_VALUE_GEOMETRY.md](../DIFFUSION_REPAIR_VALUE_GEOMETRY.md) | Feature geometry behind the value proxy. |
| [DIFFUSION_ERROR_FUNCTION_GEOMETRY.md](../DIFFUSION_ERROR_FUNCTION_GEOMETRY.md) | Data-derived error-function assertions linking repair value, source trust, retention, and realization losses. |
| [DIFFUSION_DECOMPOSED_SELECTOR_AUDIT.md](../DIFFUSION_DECOMPOSED_SELECTOR_AUDIT.md) | Direct four-term selector comparison against single repairability labels. |
| [DIFFUSION_COMPOSITE_SELECTOR_TARGETS.md](../DIFFUSION_COMPOSITE_SELECTOR_TARGETS.md) | Supervised target rows for training the four-term selector. |
| [DIFFUSION_COMPOSITE_SELECTOR_FIT.md](../DIFFUSION_COMPOSITE_SELECTOR_FIT.md) | Tiny interpretable four-head selector fit over the target rows. |
| [DIFFUSION_SELECTOR_HOLDOUT_EVAL.md](../DIFFUSION_SELECTOR_HOLDOUT_EVAL.md) | Leave-one-task-out check of decomposed heads against the single repairability baseline. |
| [DIFFUSION_COMPOSITE_SELECTOR_RUNNER_POLICY.md](../DIFFUSION_COMPOSITE_SELECTOR_RUNNER_POLICY.md) | CLI trigger and diagnostics for running the four-head selector in the benchmark runner. |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER.md) | Independent planning-slice spend-head transfer boundary result using repair-oracle lift. |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V2.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V2.md) | Expanded eight-planning-row transfer result. |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V3.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V3.md) | Larger 16-planning-row proof-object transfer result under corrected repair-only labels: trajectory-relative availability is useful but still has one error. |
| [DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md](../DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md) | Corrected availability predictor fit over v3 rows; the best pre-repair rule still has one error. |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V4.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V4.md) | Fresh-slice falsifier for fixed pre-repair availability rules: two repair-promotion positives and one calibrated availability error. |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V5.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V5.md) | Fresh-slice test for `calibrated_availability_predictor_v1`: three errors, positive score lift, and remaining oracle headroom. |
| [DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md](../DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md) | Post-repair promotion labels showing `candidate_aware_promotion_v1` has zero local errors on v5 repair candidates. |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V6.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V6.md) | Fresh-slice test showing calibrated spend gating has four errors on v6. |
| [DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V6.md](../DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V6.md) | Post-repair promotion labels showing `candidate_aware_promotion_v1` has zero local errors again on v6 repair candidates. |
| [DIFFUSION_SPEND_POLICY_DECISION.md](../DIFFUSION_SPEND_POLICY_DECISION.md) | Current v5-v9 spend-policy decision and live v6 relative-cost comparison. |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V7.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V7.md) | Fresh no-retuning v7 spend check: repairable-denoise spending finds two positives and six no-lift rows. |
| [DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V7.md](../DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V7.md) | Fresh v7 promotion labels showing `candidate_aware_promotion_v1` remains zero-error on generated candidates. |
| [DIFFUSION_SPEND_GATE_V7_FIT.md](../DIFFUSION_SPEND_GATE_V7_FIT.md) | Offline simple-gate fit over v5/v6/v7 showing no pre-repair spend gate is ready for GPU promotion. |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V8.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V8.md) | Fresh v8 spend check: four profitable repairs, four no-lift rows, and continued spend-head errors. |
| [DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V8.md](../DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V8.md) | Fresh v8 promotion labels showing `candidate_aware_promotion_v1` remains zero-error on generated candidates. |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V9.md](../DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V9.md) | Counterexample-probe spend check: five profitable repairs, three no-lift rows, and low-gap/high-gap boundary stress. |
| [DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V9.md](../DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V9.md) | Counterexample-probe promotion labels showing zero candidate-aware promotion errors with five positives. |
| [DIFFUSION_SPEND_GATE_V9_FIT.md](../DIFFUSION_SPEND_GATE_V9_FIT.md) | Offline simple-gate fit over v5-v9 showing the best threshold still misses seven positives and admits five no-lift repairs. |
| [DIFFUSION_SPEND_COUNTEREXAMPLE_WORKBENCH.md](../DIFFUSION_SPEND_COUNTEREXAMPLE_WORKBENCH.md) | Active controller workbench naming the false-negative and false-positive clusters the next spend model must solve. |
| [DIFFUSION_SPEND_VALUE_MODEL_V1.md](../DIFFUSION_SPEND_VALUE_MODEL_V1.md) | Offline nonlinear prototype challenger over pre-repair diagnostics; useful negative showing local geometry alone is still not deployable. |
| [DIFFUSION_SOURCE_VALUE_SIGNALS_V1.md](../DIFFUSION_SOURCE_VALUE_SIGNALS_V1.md) | Label-free source-text signal audit over v5-v9 rows; identifies source realization and degeneracy features as the next controller channel to test. |
| [DIFFUSION_SOURCE_DEGENERATION_AUDIT_V1.md](../DIFFUSION_SOURCE_DEGENERATION_AUDIT_V1.md) | Source realization defect audit over v5-v9 rows; separates repeated-token, punctuation-run, and meta-leakage clusters without promoting them to a spend gate. |
| [DIFFUSION_PRE_REPAIR_EDGE_PROXY_V1.md](../DIFFUSION_PRE_REPAIR_EDGE_PROXY_V1.md) | Offline pre-repair promotion-edge proxy over v5-v9 rows; joins frozen source, degeneration, and span diagnostics before any live spend promotion. |
| [DIFFUSION_COUNTERFACTUAL_CONTROLLER_ARCHITECTURE_V1.md](../DIFFUSION_COUNTERFACTUAL_CONTROLLER_ARCHITECTURE_V1.md) | Next-controller contract: use frozen features for probe triage, then learn value-of-information from cheap counterfactual probe rows before another live spend gate. |
| [DIFFUSION_COUNTERFACTUAL_PROBE_TARGETS_V1.md](../DIFFUSION_COUNTERFACTUAL_PROBE_TARGETS_V1.md) | First deterministic counterfactual-probe target sheet over named spend counterexamples; diagnostic scaffold for the future measured micro-probe run. |
| [DIFFUSION_COUNTERFACTUAL_PROBE_POLICY_FIT_V1.md](../DIFFUSION_COUNTERFACTUAL_PROBE_POLICY_FIT_V1.md) | Offline value-of-information rule fit over the deterministic probe scaffold; one-error diagnostic result, not a promoted spend gate. |
| [DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_RUNNER_HOOK_V1.md](../DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_RUNNER_HOOK_V1.md) | Runner hook for `--repair-spend-trigger counterfactual_micro_probe_v1`; emits bounded probe records while blocking full repair spend. |
| [DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_SMOKE_V1.md](../DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_SMOKE_V1.md) | First GPU smoke of the measured micro-probe hook on `plan_070`; confirms measured gate deltas with `should_run=false`. |
| [DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_COUNTEREXAMPLES_V1.md](../DIFFUSION_COUNTERFACTUAL_MICRO_PROBE_COUNTEREXAMPLES_V1.md) | Measured micro-probe run over all named v5-v9 counterexamples; zero triage errors but still diagnostic-only. |
| [DIFFUSION_COUNTERFACTUAL_MEASURED_PROBE_VALUE_POLICY_V1.md](../DIFFUSION_COUNTERFACTUAL_MEASURED_PROBE_VALUE_POLICY_V1.md) | All-shadow measured probe fit over the 12 named counterexamples; measured-only Stage 1 still has two errors, so full repair spend remains blocked. |
| [DIFFUSION_COUNTERFACTUAL_PROBE_TEXT_FIDELITY_V1.md](../DIFFUSION_COUNTERFACTUAL_PROBE_TEXT_FIDELITY_V1.md) | Post-probe text fidelity audit over the all-shadow probe rows; malformed authorization and weak tomography keep Stage 1 diagnostic-only. |
| [DIFFUSION_COUNTERFACTUAL_TOMOGRAPHY_PROBE_TEXT_FIDELITY_V1.md](../DIFFUSION_COUNTERFACTUAL_TOMOGRAPHY_PROBE_TEXT_FIDELITY_V1.md) | Strict tomography prompt GPU run; fixes the authorization sentinel but still leaves one post-probe false positive and four invalid diagnostic rows. |
| [DIFFUSION_COUNTERFACTUAL_VALIDATED_PROBE_STAGE1_GATE_V1.md](../DIFFUSION_COUNTERFACTUAL_VALIDATED_PROBE_STAGE1_GATE_V1.md) | Validity-required Stage 1 gate fit; invalid diagnostics include three profitable rows, so the best validated rule has five errors. |
| [DIFFUSION_COUNTERFACTUAL_KEY_VALUE_PROBE_TEXT_FIDELITY_V2.md](../DIFFUSION_COUNTERFACTUAL_KEY_VALUE_PROBE_TEXT_FIDELITY_V2.md) | Key-value tomography v2 text-fidelity result; removes placeholders but regresses exact authorization and valid-row count. |
| [DIFFUSION_COUNTERFACTUAL_KEY_VALUE_VALIDATED_PROBE_STAGE1_GATE_V2.md](../DIFFUSION_COUNTERFACTUAL_KEY_VALUE_VALIDATED_PROBE_STAGE1_GATE_V2.md) | Key-value tomography v2 validity-required gate fit; two false negatives but still diagnostic-only because probe reliability regressed. |
| [DIFFUSION_COUNTERFACTUAL_COMPACT_PROBE_TEXT_FIDELITY_V3.md](../DIFFUSION_COUNTERFACTUAL_COMPACT_PROBE_TEXT_FIDELITY_V3.md) | Compact tomography v3 text-fidelity result; runtime format is stable, but semantic validity falls to 4/12 because of template echoes, malformed compact keys, and duplicate authorization. |
| [DIFFUSION_COUNTERFACTUAL_COMPACT_VALIDATED_PROBE_STAGE1_GATE_V3.md](../DIFFUSION_COUNTERFACTUAL_COMPACT_VALIDATED_PROBE_STAGE1_GATE_V3.md) | Compact tomography v3 semantic-validity-required gate fit; five profitable rows become invalid-positive misses and the best measured Stage 1 rule has six errors. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_TEXT_FIDELITY_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_TEXT_FIDELITY_V4.md) | Span tomography v4 text-fidelity result; 10/12 semantic-valid rows, zero template echoes, and one malformed span key. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_STAGE1_GATE_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_STAGE1_GATE_V4.md) | Span tomography v4 semantic-validity-required gate fit; X0/X2-overlap-penalized retention risk clears the local 12-row fit but remains diagnostic. |
| [DIFFUSION_COUNTERFACTUAL_PROBE_TRANSFER_TARGETS_V1.md](../DIFFUSION_COUNTERFACTUAL_PROBE_TRANSFER_TARGETS_V1.md) | Independent `plan_001`-`plan_008` transfer target sheet for measured counterfactual probe gates: 6 positives and 2 negatives. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_DISTINCT_RETENTION_RULE_V4_FRESH_PLANNING.md](../DIFFUSION_COUNTERFACTUAL_SPAN_DISTINCT_RETENTION_RULE_V4_FRESH_PLANNING.md) | Frozen distinct-retention rule transfer check: 3 errors on the fresh planning slice, so the local zero-error rule is rejected for promotion. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_TRANSFER_MATRIX_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_TRANSFER_MATRIX_V4.md) | Cross-slice matrix separating discovery-fit rules from fresh-only diagnostic upper bounds; best local v4 rule has 3 transfer errors. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_CONJUNCTION_TRANSFER_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_CONJUNCTION_TRANSFER_V4.md) | Two-condition measured-probe rule search; transfer-screened gap/span conjunction is the next frozen hypothesis, not a promoted controller. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_GAP_SPAN_RULE_V4_TRANSFER_V3_PLANNING.md](../DIFFUSION_COUNTERFACTUAL_SPAN_GAP_SPAN_RULE_V4_TRANSFER_V3_PLANNING.md) | Frozen next-slice check for the gap/span conjunction: 3 errors on `plan_017`-`plan_024`, retiring the challenger as diagnostic-only. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_MODEL_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_MODEL_V4.md) | Leave-slice-out signature-model audit across 28 span-v4 rows: all positives are preserved, but 11 no-lift false positives block promotion. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_UTILITY_FRONTIER_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_UTILITY_FRONTIER_V4.md) | Cost-penalized frontier over the signature score: low penalties keep 11 false positives; higher penalties miss most positives. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_NO_LIFT_VETO_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_NO_LIFT_VETO_V4.md) | Leave-slice-out no-lift veto search over 73,154 threshold-fragment rules; specificity improves weakly but five positives are newly missed. |
| [DIFFUSION_MOONSHOT_REASONING_ARCHITECTURE_V1.md](DIFFUSION_MOONSHOT_REASONING_ARCHITECTURE_V1.md) | Cross-repo synthesis and build target for the next controller: signed realized-value tomography with external anchors and consolidation. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_V4.md) | First signed realized-value head; improves strict signed utility but remains below the M1 promotion bar. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_CONTROLS_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_CONTROLS_V4.md) | Negative-control audit for the signed-value head; withholding text, source, gap/span, or retention families all degrades utility. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_WEAK_SLICE_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_WEAK_SLICE_V4.md) | Weak-slice diagnostic for `plan_017`-`plan_024`: all eight rows are selected, but only two are positive, blocking a fresh GPU promotion run. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COHORT_RISK_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COHORT_RISK_V4.md) | Risk-adjusted neighbor cohort sweep; clears the old M1 utility bar but still fails weak-slice cohort calibration. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_TRAJECTORY_RELATIVE_GATE_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_TRAJECTORY_RELATIVE_GATE_V4.md) | Composite audit that adds trajectory-relative source evidence; clears the weak-slice blocker offline but changes the information-channel contract. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_TRAJECTORY_RELATIVE_CONTROLS_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_TRAJECTORY_RELATIVE_CONTROLS_V4.md) | Negative-control audit for the trajectory-relative channel; all four controls degrade against the true composite. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_FREEZE_V4.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_FREEZE_V4.md) | Frozen v10 protocol for the composite gate: new task preset, measured span-probe pass, all-repairable label pass, and replay promotion gates. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_V10_REPLAY.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_V10_REPLAY.md) | Fresh v10 replay result: useful repair labels, but the frozen composite remains diagnostic-only after selecting all rows and missing the utility bar. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_V10_VETO_STRESS_FREEZE.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_V10_VETO_STRESS_FREEZE.md) | Fixed-source v10 follow-up protocol designed to create source-vs-trajectory deltas and test the trajectory-relative veto. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_V10_FIXED_SOURCE_REPLAY.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_V10_FIXED_SOURCE_REPLAY.md) | Fixed-source replay result: the trajectory-relative veto removes one no-lift row, but the composite remains diagnostic-only. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_V10_NO_LIFT_SPECIFICITY.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_V10_NO_LIFT_SPECIFICITY.md) | Post-label v10 diagnostic frontier: a measured probe-value floor removes the remaining no-lift rows on fixed-source replay, but must be frozen before any fresh transfer check. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_FREEZE.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_FREEZE.md) | Frozen v11 source-divergent proof obligation for the measured probe-value floor, with fixed source policy and measurement-cost gates declared before GPU labels. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_MEASUREMENT.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_MEASUREMENT.md) | Fresh v11 measurement boundary: the GPU pass completed, but all planning rows had zero source-vs-trajectory delta, so the declared source-divergent gate failed before labels. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_RANDOM_SOURCE_FREEZE.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_RANDOM_SOURCE_FREEZE.md) | Frozen random-source addendum for v11: uses the stable random arm as repair/probe source to force a source-divergence stress test before labels. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_RANDOM_SOURCE_MEASUREMENT.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_RANDOM_SOURCE_MEASUREMENT.md) | Fresh random-source v11 measurement: source divergence appears on five planning rows, so the predeclared label pass is now meaningful. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_RANDOM_SOURCE_REPLAY.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_RANDOM_SOURCE_REPLAY.md) | Fresh random-source v11 replay falsifier: the frozen value floor selects no-lift rows, misses the only positive, and goes negative after probe cost. |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_COUNTEREXAMPLES.md](../DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_COUNTEREXAMPLES.md) | Source-aware v11 counterexample map: high probe value on negative-delta random sources is no-lift, the selected-repair positive is source-aligned and below the old floor, and `plan_084` separates generation value from promotion realization. |
| [DIFFUSION_SOURCE_AWARE_LIFT_DIRECTION_V12_FREEZE.md](../DIFFUSION_SOURCE_AWARE_LIFT_DIRECTION_V12_FREEZE.md) | Frozen v12 source-aware lift-direction proof obligation over `plan_089`-`plan_096`, with random-source measurement and label commands plus selected-repair/oracle realization gates before any live spend trigger. |
| [DIFFUSION_SOURCE_AWARE_LIFT_DIRECTION_V12_MEASUREMENT.md](../DIFFUSION_SOURCE_AWARE_LIFT_DIRECTION_V12_MEASUREMENT.md) | Fresh v12 measurement boundary: the random-source pass creates three negative source-delta rows, blocks high-probe disagreement rows, and selects only `plan_091` before labels. |
| [DIFFUSION_SOURCE_AWARE_LIFT_DIRECTION_V12_REPLAY.md](../DIFFUSION_SOURCE_AWARE_LIFT_DIRECTION_V12_REPLAY.md) | Fresh v12 replay falsifier: the static source-aware surface misses `plan_093`, selects no-lift `plan_091`, and points next at denoise-phase realization rather than harder gap/coverage thresholds. |
| [DIFFUSION_DENOISE_PHASE_REALIZATION_V13_FREEZE.md](../DIFFUSION_DENOISE_PHASE_REALIZATION_V13_FREEZE.md) | Frozen v13 denoise-phase realization proof obligation over `plan_097`-`plan_104`, with source-aligned skeleton/phase evidence and selected-vs-oracle replay controls before GPU labels. |
| [DIFFUSION_DENOISE_PHASE_REALIZATION_V13_MEASUREMENT.md](../DIFFUSION_DENOISE_PHASE_REALIZATION_V13_MEASUREMENT.md) | Fresh v13 measurement boundary: source disagreement appears on `plan_097` and `plan_098`, skeleton-only selects all planning rows, and the frozen realization surface selects `plan_099` plus `plan_102` before labels. |
| [DIFFUSION_DENOISE_PHASE_REALIZATION_V13_REPLAY.md](../DIFFUSION_DENOISE_PHASE_REALIZATION_V13_REPLAY.md) | Fresh v13 replay falsifier: the frozen realization surface catches `plan_099` but admits no-lift `plan_102`, misses positive `plan_104`, and points next at a realization-value head. |
| [DIFFUSION_SPEND_TRANSFER_RULE_FIT.md](../DIFFUSION_SPEND_TRANSFER_RULE_FIT.md) | Transfer-rule fit showing current decomposed spend is the best repair-availability rule. |
| [DIFFUSION_SPEND_TRANSFER_RULE_FIT_V2.md](../DIFFUSION_SPEND_TRANSFER_RULE_FIT_V2.md) | Expanded transfer-rule fit over the eight-row independent slice. |
| [DIFFUSION_TRANSFER_PROMOTION_VALUE.md](../DIFFUSION_TRANSFER_PROMOTION_VALUE.md) | Transfer promotion-value result showing named `--repair-selector transfer_promotion_value` realizes the low-margin repair. |
| [DIFFUSION_TRANSFER_HEAD_FIT.md](../DIFFUSION_TRANSFER_HEAD_FIT.md) | First separate availability and promotion-value head fit over original plus transfer rows. |
| [DIFFUSION_REASONING_PROOF_OBJECT.md](../DIFFUSION_REASONING_PROOF_OBJECT.md) | Canonical proof-object ledger for the decomposed diffusion reasoning heads, falsifiers, and next GPU validations. |

## Mechanism Layer

Read these when checking what actually causes the lift:

| Document | What It Gives You |
| --- | --- |
| [DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md](../DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md) | Productive spend versus no-lift skip diagnostics. |
| [DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md](../DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md) | 53,460-point geometry/phase sweep and score-cost frontier. |
| [DIFFUSION_DENOISE_PHASE_GEOMETRY.md](../DIFFUSION_DENOISE_PHASE_GEOMETRY.md) | When repairable skeletons appear in denoise histories. |
| [DIFFUSION_PHASE_HYBRID_MECHANISM_AUDIT.md](../DIFFUSION_PHASE_HYBRID_MECHANISM_AUDIT.md) | Why phase history is selector evidence before it is a replacement source. |
| [DIFFUSION_PHASE_SOURCE_POLICY_AUDIT.md](../DIFFUSION_PHASE_SOURCE_POLICY_AUDIT.md) | Source-choice policy audit for final versus history source. |
| [DIFFUSION_PHASE_SOURCE_THRESHOLD_SWEEP.md](../DIFFUSION_PHASE_SOURCE_THRESHOLD_SWEEP.md) | GPU threshold sweep showing weak phase-history replacement regresses. |

## Anchor And Retention Layer

Read these when evaluating compact control terms and information preservation:

| Document | What It Gives You |
| --- | --- |
| [DIFFUSION_ANCHOR_RETENTION_LOSS.md](../DIFFUSION_ANCHOR_RETENTION_LOSS.md) | Constraint-retention loss for history/final anchor choice. |
| [DIFFUSION_HISTORY_ANCHOR_REPAIR_AUDIT.md](../DIFFUSION_HISTORY_ANCHOR_REPAIR_AUDIT.md) | History-anchor diagnostics and why dual spending is too expensive. |
| [DIFFUSION_REALIZATION_QUALITY.md](../DIFFUSION_REALIZATION_QUALITY.md) | Compact seed realization quality and anchor leakage checks. |
| [DIFFUSION_PHASE_ANCHOR_BOUNDARY.md](../DIFFUSION_PHASE_ANCHOR_BOUNDARY.md) | Boundary for pre-generation phase anchors. |

## Development Map

| Surface | File |
| --- | --- |
| Main GPU runner | [experiments/run_diffusion_three_arm_benchmark.py](../experiments/run_diffusion_three_arm_benchmark.py) |
| Public claim builder | [experiments/build_diffusion_claim_evidence.py](../experiments/build_diffusion_claim_evidence.py) |
| Claim validator | [experiments/validate_diffusion_claim_evidence.py](../experiments/validate_diffusion_claim_evidence.py) |
| Theory claim validator | [experiments/validate_diffusion_theory_claim_ledger.py](../experiments/validate_diffusion_theory_claim_ledger.py) |
| Stale public-doc scanner | [experiments/scan_stale_diffusion_docs.py](../experiments/scan_stale_diffusion_docs.py) |
| Repair-value geometry audit | [experiments/analyze_diffusion_repair_value_geometry.py](../experiments/analyze_diffusion_repair_value_geometry.py) |
| Error-function geometry audit | [experiments/analyze_diffusion_error_function_geometry.py](../experiments/analyze_diffusion_error_function_geometry.py) |
| Decomposed selector audit | [experiments/analyze_diffusion_decomposed_selector.py](../experiments/analyze_diffusion_decomposed_selector.py) |
| Composite selector target builder | [experiments/build_diffusion_composite_selector_targets.py](../experiments/build_diffusion_composite_selector_targets.py) |
| Composite selector fitter | [experiments/fit_diffusion_composite_selector.py](../experiments/fit_diffusion_composite_selector.py) |
| Selector holdout evaluator | [experiments/evaluate_diffusion_selector_holdout.py](../experiments/evaluate_diffusion_selector_holdout.py) |
| Independent spend-transfer evaluator | [experiments/evaluate_diffusion_independent_spend_transfer.py](../experiments/evaluate_diffusion_independent_spend_transfer.py) |
| Spend-transfer rule fitter | [experiments/fit_diffusion_spend_transfer_rule.py](../experiments/fit_diffusion_spend_transfer_rule.py) |
| V7 spend-gate fitter | [experiments/fit_diffusion_spend_gate_v7.py](../experiments/fit_diffusion_spend_gate_v7.py) |
| Transfer promotion-value evaluator | [experiments/evaluate_diffusion_transfer_promotion_value.py](../experiments/evaluate_diffusion_transfer_promotion_value.py) |
| Transfer-head fitter | [experiments/fit_diffusion_transfer_heads.py](../experiments/fit_diffusion_transfer_heads.py) |
| Source degeneration audit | [experiments/analyze_diffusion_source_degeneracy.py](../experiments/analyze_diffusion_source_degeneracy.py) |
| Pre-repair edge proxy audit | [experiments/analyze_diffusion_pre_repair_edge_proxy.py](../experiments/analyze_diffusion_pre_repair_edge_proxy.py) |
| Counterfactual probe target builder | [experiments/build_diffusion_counterfactual_probe_targets.py](../experiments/build_diffusion_counterfactual_probe_targets.py) |
| Counterfactual probe policy fitter | [experiments/fit_diffusion_counterfactual_probe_policy.py](../experiments/fit_diffusion_counterfactual_probe_policy.py) |
| Counterfactual micro-probe run analyzer | [experiments/analyze_diffusion_counterfactual_micro_probe_run.py](../experiments/analyze_diffusion_counterfactual_micro_probe_run.py) |
| Counterfactual micro-probe runner trigger | [experiments/run_diffusion_three_arm_benchmark.py](../experiments/run_diffusion_three_arm_benchmark.py) with `--repair-spend-trigger counterfactual_micro_probe_v1` |
| Proof-object builder | [experiments/build_diffusion_proof_object.py](../experiments/build_diffusion_proof_object.py) |
| Four-head runner trigger | [experiments/run_diffusion_three_arm_benchmark.py](../experiments/run_diffusion_three_arm_benchmark.py) with `--repair-spend-trigger decomposed_four_head_selector` |
| Transfer-rule runner trigger | [experiments/run_diffusion_three_arm_benchmark.py](../experiments/run_diffusion_three_arm_benchmark.py) with `--repair-spend-trigger decomposed_spend_transfer_rule` |
| Trajectory-relative transfer trigger | [experiments/run_diffusion_three_arm_benchmark.py](../experiments/run_diffusion_three_arm_benchmark.py) with `--repair-spend-trigger trajectory_relative_decomposed_spend` |
| Learned availability trigger | [experiments/run_diffusion_three_arm_benchmark.py](../experiments/run_diffusion_three_arm_benchmark.py) with `--repair-spend-trigger learned_availability_predictor_v1` |
| Transfer promotion-value selector | [experiments/run_diffusion_three_arm_benchmark.py](../experiments/run_diffusion_three_arm_benchmark.py) with `--repair-selector transfer_promotion_value --repair-promotion-margin 0.0` |
| Diffusion backend and repair code | [src/latent_reasoning/diffusion](../src/latent_reasoning/diffusion) |

## How To Read Claims

Do not trust a sentence just because it is in a Markdown file. A promoted
diffusion benchmark claim should have:

- a score JSON
- a rendered report
- raw generations
- a run ID
- a content hash in the ground-truth index
- validator coverage in the claim evidence gate

If a claim is not in [CLAIM_EVIDENCE_MAP.md](../CLAIM_EVIDENCE_MAP.md), treat it
as exploratory until it is promoted.

A theory claim should also appear in
[DIFFUSION_THEORY_CLAIM_LEDGER.md](DIFFUSION_THEORY_CLAIM_LEDGER.md), with a
status, falsifier, and next proof obligation. Treat claims missing from both
ledgers as working notes.
