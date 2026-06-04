# Latent Space Reasoning

This repo studies inference-time latent reasoning control: changing a frozen
model's reasoning trajectory without fine-tuning it. It started with
autoregressive soft-prefix perturbations, but the current frontier is
language-diffusion repair because diffusion denoise trajectories expose editable
intermediate states.

**Current public result:** on the lean GPU mixed benchmark, LLaDA-MoE
diffusion-native latent repair beats both greedy/fixed denoise and random
perturbation:

| Public arm | Score | Relative GPU cost |
|------------|------:|------------------:|
| Greedy/fixed denoise | 0.412277 | 1.000000x |
| Random perturbation | 0.372125 | 1.000000x |
| **Latent repair** | **0.531116** | **2.625000x** |

**Public entry path:** README -> [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md)
-> [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md) ->
[DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md). The public
benchmark now shows both the top-score latent repair point (`0.531116` at
`2.625000x`) and the budget-favored decomposed-selector point (`0.508705` at
`2.375000x`). Use the first for the headline reasoning-lift claim and the
second for the cost-aware controller claim: it exposes the fitted
spend/source/retention/realization selector heads in runner diagnostics while
matching the lower-cost value-proxy score/cost point.

**Theory entry path:** start with
[docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md](docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md).
It formalizes the current view of diffusion reasoning as controlled reduction
of task-relevant information loss over an editable denoise trajectory, then
states the proof obligations and candidate error functions that the benchmark
artifacts are now testing.
The newest theory work is deliberately decomposed: repair availability,
promotion value, source trust, retention, and cost are separate error-function
terms so the repo can test where information is lost, gained, preserved, or
made too expensive.
The current proof-object report is
[DIFFUSION_REASONING_PROOF_OBJECT.md](DIFFUSION_REASONING_PROOF_OBJECT.md):
six falsifiable heads, 60 target rows, explicit information channels, and GPU
validation obligations for the next slice. One head is now deliberately marked
`boundary`: v4 falsified the v3 learned availability cutoff, v5 showed that
pre-repair availability geometry still misses promotion value, and v6 confirms
that the post-repair promotion head generalizes while calibrated spend gating
does not.
The first larger proof-object GPU slice is now complete: v3 adds eight fresh
planning prompts, finds three positive repair-availability rows, and shows why
availability must be trajectory-relative. Single repairability makes `4` errors,
the older decomposed spend head makes `1`, and the new
`trajectory_relative_decomposed_spend` head makes `0` on the 16 planning rows;
the executable CUDA run `diffusion-106f05c6dd5532ee` closes oracle repair
headroom at `0.000000` while beating fixed by `0.021920` and random by
`0.083580` on repair-covered planning tasks.
That boundary is now fitted into `learned_availability_predictor_v1`:
[DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md](DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md)
learns `prompt_gap_count <= 8`, `source_quality <= 0.256429`, and
`source_task_delta_vs_trajectory >= 0`; CUDA run `diffusion-d0c2962992c50178`
reproduces the same three selected repairs and zero oracle headroom.
The follow-up v4 slice is the key negative result: without retuning thresholds,
[DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V4.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V4.md)
finds repair-promotion positives on `plan_026` and `plan_031`, and shows the
learned v3 source-quality cutoff does not transfer. Executable CUDA run
`diffusion-865e5acb0ee73e8a` is retained as historical evidence for the stale
cutoff; the corrected label path now pushes the work toward separate spend and
post-repair promotion targets.
That calibrated trigger is now executable as
`calibrated_availability_predictor_v1`. It is a useful pre-repair spend trigger,
but the corrected repair-only labels show that availability is not promotion.
The fresh v5 slice is the next boundary:
[DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V5.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V5.md)
has four repair-promotion positives and calibrated availability makes three
errors. CUDA run `diffusion-c4f0d7bc21768f21` beats fixed by `0.044866` and
random by `0.074438` on repair-covered planning tasks while leaving `0.011920`
oracle headroom. The new
[DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md](DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md)
artifact shows `candidate_aware_promotion_v1` has zero promotion errors over
the seven generated v5 repair candidates, including recovered `plan_037` and
blocked `plan_033`, `plan_038`, and `plan_040`. The fresh v6 slice repeats the
same split: [DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V6.md](DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V6.md)
has zero promotion errors over eight generated repair candidates, while
[DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V6.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V6.md)
shows calibrated availability makes four spend errors. The best current lean
policy is therefore repairable-denoise spending plus candidate-aware promotion:
run `diffusion-ae7a4edd5c22ca20`, `+0.086696` vs fixed, `+0.125929` vs random,
`+0.068009` vs trajectory, and `0.000000` oracle headroom at `1.000000` extra
generation per task. The generated decision artifact
[DIFFUSION_SPEND_POLICY_DECISION.md](DIFFUSION_SPEND_POLICY_DECISION.md)
summarizes the v5/v6/v7 target rows and the latest calibrated live cost
comparison: calibrated spend saves `0.375000` relative extra generations per
task on v6, but repairable-denoise spending buys `0.039661` more score and
`0.105762` incremental lift per added generation.
The fresh v7 slice is now run as the no-retuning check:
[DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V7.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V7.md)
shows only `2` profitable repair rows and `6` no-lift repairable spends, while
[DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V7.md](DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V7.md)
keeps `candidate_aware_promotion_v1` at `0` promotion errors. CUDA run
`diffusion-711ea5fcfd8c07e5` still beats fixed by `+0.023036` and random by
`+0.082063` on repair-covered tasks, but leaves `0.016875` oracle headroom,
so the next work is a spend gate over v5/v6/v7 labels without changing the
promotion head. The first offline search is in
[DIFFUSION_SPEND_GATE_V7_FIT.md](DIFFUSION_SPEND_GATE_V7_FIT.md): the best
simple pre-repair rule still has `5` errors, so it is not promoted to GPU.
For navigation across the whole diffusion stack, use
[docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md). For theory
claim discipline, use
[docs/DIFFUSION_THEORY_CLAIM_LEDGER.md](docs/DIFFUSION_THEORY_CLAIM_LEDGER.md).

**What is worth sharing publicly:** a frozen diffusion language model improves
short planning reasoning by repairing denoise trajectories at inference time.
The measured lift is `+0.118839` over greedy/fixed denoise and `+0.158991`
over random perturbation at `2.625000x` relative GPU cost. The newest strict
phase-hybrid run keeps the same score/cost point while making the mechanism
cleaner: denoise history helps when it passes repairability, retention-safety,
and source-advantage checks; raw phase-source replacement is not enough. The
latest audit now emits five phase-source loss targets, so the next step is not
"try more benchmarks" but learning the source-choice policy that decides when
to trust denoise history and when to preserve final-state repair. The first
policy audit selects the strict calibrated rule
`phase_safe_repairable_count > 0`, `target_similarity >= 0.96`, and
`text_similarity >= 0.96`, with zero weighted error on the current source-choice
targets; looser "trust any safe phase" replacement would produce three false
history-source switches. The benchmark runner now exposes these as explicit
`PHASE_SOURCE_*` thresholds in the phase-hybrid source switch, with CLI knobs
`--phase-source-target-similarity-min`,
`--phase-source-text-similarity-min`, and
`--phase-source-history-char-ratio-min` for GPU policy sweeps. The first loose
`0.90/0.90/0.90` CUDA sweep confirms the important boundary: loose promotion
scores `0.524554` at the same `2.625000x` cost, below strict `0.531116`,
because it switches `plan_003` from final to history and loses `0.052500` on
that task. A too-strict `0.97/0.97/0.95` run also scores `0.531116` while
keeping all repairs final-sourced, so the frontier lesson is not "always use
history"; it is "do not trust weak phase history as a source." That plateau is
now a named benchmark operator: `constraint_span_phase_final_preserve_seeded_gated`.
It keeps phase-denoise evidence for repair spend gating, but forces the repair
source to the final denoise state so the public mechanism is easy to reproduce
without hiding behind threshold tuning. Fresh CUDA validation
`diffusion-175cbd422107ee5e` confirms the named operator ties the frontier:
`0.531116` at `2.625000x`, with all five selected repairs final-sourced.

Promoted run: `diffusion-3b42951db77c5aa6`; canonical promoted report:
`eval_results/diffusion_language/llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_preservation_seed_fresh_v1_report.md`.
The fresh cap-31/cap-32 frontier confirmations and lower-cost cap ladder are summarized in
[DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md](DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md)
under "Fresh Phase-Window Confirmations", and the derived minimal cap map is in
[DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md](DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md).
That map predicts all eleven fresh CUDA cap confirmations with `0` score/cost
mismatches: cap `9` is the no-repair floor, cap `10-19` is the three-repair
plateau, cap `20-30` is the four-repair plateau, and cap `31+` is the full
five-repair frontier. The benchmark runner now exposes those tiers directly as
`--repair-phase-budget floor`, `cheap`, `mid`, and `frontier`, so future GPU
runs can request the public budget tier instead of memorizing raw denoise-step
caps. Fresh CUDA runs of `--repair-phase-budget floor`, `cheap`, `mid`, and
`frontier` land on their expected content IDs: floor mode is
`diffusion-fae5a3498468b66f` at `0.414598` / `2.000000x`, cheap mode is
`diffusion-f8f6ae3e209d502b` at `0.472500` / `2.375000x`, mid mode is
`diffusion-65f906724fed3cbc` at `0.496607` / `2.500000x`, and frontier mode is
`diffusion-175cbd422107ee5e` at `0.531116` / `2.625000x`.
The next theoretical layer is now generated in
[DIFFUSION_BUDGET_POLICY_LOSS.md](DIFFUSION_BUDGET_POLICY_LOSS.md): it turns
the phase-window ladder into a cost-aware selector loss
`utility = aggregate_score_lift - lambda * marginal_relative_cost`. This shows
why the cap ladder is the right public control surface but not the final learned
policy: an oracle task-gated selector at lambda `0.18` would keep
`plan_004`/`plan_006`/`plan_007`, score `0.508705` at `2.375000x`, and gain
`+0.022589` objective over the best cap policy at that cost penalty.
That oracle is now a runner-ready spend trigger. The generated
[DIFFUSION_BUDGET_VALUE_PROXY_AUDIT.md](DIFFUSION_BUDGET_VALUE_PROXY_AUDIT.md)
calibrates `denoise_phase_value_proxy` to
`source_quality <= 0.301429` inside the public prompt-gap band, and a fresh
CUDA run with the stable CLI threshold `--repair-value-proxy-source-quality-max
0.31` confirms the lower-cost point: `diffusion-a343e942cbfb0a93` scores
`0.508705` at `2.375000x`, spending only on `plan_004`, `plan_006`, and
`plan_007`. That result is now rendered in
[DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md) as the
budget-favored latent repair point and in
[CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md) as claim
`moe_mixed_phase_final_preserve_seeded_value_proxy_budget`.
The same score/cost point is now confirmed through the executable decomposed
four-head selector trigger: `diffusion-62476b492c9e592c` uses
`--repair-spend-trigger decomposed_four_head_selector`, repairs
`plan_004`/`plan_006`/`plan_007`, scores `0.508705` at `2.375000x`, and records
all four selector head IDs on every repair-spend gate row. The generated public
benchmark now treats that run as the budget-favored latent repair claim
`moe_mixed_decomposed_four_head_selector_budget`.
The winning policy uses automatic preservation-seeded compact control plus
denoise-trajectory repair. It keeps the public benchmark intentionally lean:
greedy baseline, random perturbation, and selected latent repair only, with
short open-ended planning as the primary slice plus small math/symbolic/science
guards.

**Read this first if you are evaluating or sharing the claim:**

Public path: start here, then verify the three-arm result in
[DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md), trace the
claim in [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md), and use
[DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md) to reach the
canonical score/report/raw artifacts.

| Start here | Why it matters |
|------------|----------------|
| [docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md) | Reader-facing map of the public claims, theory layer, benchmark/cost layer, mechanism audits, anchor-retention work, and development surfaces |
| [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md) | The current public result: greedy/fixed denoise, random perturbation, top-score latent repair, and budget-favored latent repair with relative GPU cost |
| [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md) | The evidence ledger tying every public claim to run IDs, reports, raw generations, validation checks, and the comparable MoE mixed score/cost ledger |
| [DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md) | Canonical score/report/raw artifact pointers and content hashes for promoted claims |
| [docs/DIFFUSION_THEORY_CLAIM_LEDGER.md](docs/DIFFUSION_THEORY_CLAIM_LEDGER.md) | Theory claim ledger mapping assertions to evidence, assumptions, falsifiers, and next proof obligations |
| [DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md](DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md) | Repair-spend gate audit showing productive spends, skipped no-lift cases, and forced-spend controls |
| [DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md](DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md) | 53,460-point source-geometry plus denoise-phase gate sweep, including fresh cap-9, cap-10/cap-16, cap-20/cap-30, and cap-31/cap-32 confirmation rows |
| [DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md](DIFFUSION_PHASE_WINDOW_BUDGET_MAP.md) | Minimal denoise phase-window budget map: cap `9`, `10-19`, `20-30`, and `31+`, with all fresh CUDA confirmations matching the derived score/cost model |
| [DIFFUSION_BUDGET_POLICY_LOSS.md](DIFFUSION_BUDGET_POLICY_LOSS.md) | Cost-aware repair selector loss and task-level marginal value targets for learning which denoise repairs are worth the extra GPU generation |
| [DIFFUSION_BUDGET_VALUE_PROXY_AUDIT.md](DIFFUSION_BUDGET_VALUE_PROXY_AUDIT.md) | Runner-ready source-quality value proxy plus fresh CUDA confirmation of the `0.508705` / `2.375000x` lower-cost point |
| [DIFFUSION_REPAIR_VALUE_GEOMETRY.md](DIFFUSION_REPAIR_VALUE_GEOMETRY.md) | Feature-geometry audit showing why the value proxy is not just early repairability: source quality, prompt-gap band, and late denoise phase separate high-value repair spends from low-value spends |
| [DIFFUSION_ERROR_FUNCTION_GEOMETRY.md](DIFFUSION_ERROR_FUNCTION_GEOMETRY.md) | Data-derived assertions for the next error functions: cost-aware repair value, retention-gated source trust, and composite denoise reasoning loss |
| [DIFFUSION_DECOMPOSED_SELECTOR_AUDIT.md](DIFFUSION_DECOMPOSED_SELECTOR_AUDIT.md) | Direct comparison showing the decomposed value/source/retention/realization selector locally dominates a single repairability-label controller |
| [DIFFUSION_COMPOSITE_SELECTOR_TARGETS.md](DIFFUSION_COMPOSITE_SELECTOR_TARGETS.md) | First supervised target surface for the four-term diffusion controller: task rows plus realization-policy rows |
| [DIFFUSION_COMPOSITE_SELECTOR_FIT.md](DIFFUSION_COMPOSITE_SELECTOR_FIT.md) | Tiny CPU-safe interpretable selector fit over the four-term target surface |
| [DIFFUSION_SELECTOR_HOLDOUT_EVAL.md](DIFFUSION_SELECTOR_HOLDOUT_EVAL.md) | Leave-one-task-out selector check: decomposed heads make `4` errors over `21` labels versus `12` for a single repairability controller |
| [DIFFUSION_COMPOSITE_SELECTOR_RUNNER_POLICY.md](DIFFUSION_COMPOSITE_SELECTOR_RUNNER_POLICY.md) | Runner-facing implementation of the four-head selector as `--repair-spend-trigger decomposed_four_head_selector` |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER.md) | Independent spend-head transfer check using repair-oracle lift: `plan_012` is a positive low-margin repair case, and the decomposed spend head has zero errors on the four-row slice |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V2.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V2.md) | Expanded eight-planning-row transfer check: the same `plan_012` low-margin repair remains the only positive repair-availability label |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V3.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V3.md) | Larger 16-planning-row proof-object transfer check under repair-only labels: single repairability has `5` errors, old decomposed spend has `2`, and trajectory-relative decomposed spend has `1` |
| [DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md](DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md) | Corrected v3 availability predictor fit: the best pre-repair rule still has one error, making availability a boundary rather than a solved head |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V4.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V4.md) | Fresh-slice falsifier: two repair-promotion positives and calibrated availability with one error, showing fixed pre-repair rules remain brittle |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V5.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V5.md) | Fresh calibrated-trigger boundary: four repair-promotion positives, three calibrated availability errors, positive CUDA score lift, and remaining oracle headroom |
| [DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md](DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md) | Post-repair promotion target artifact: `candidate_aware_promotion_v1` has zero errors over seven generated v5 repair candidates |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V6.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V6.md) | Fresh v6 spend-gate falsifier: calibrated availability has four errors and misses high-value `plan_048` |
| [DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V6.md](DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V6.md) | Fresh v6 promotion target artifact: `candidate_aware_promotion_v1` again has zero errors, selecting five positive generated repairs |
| [DIFFUSION_SPEND_POLICY_DECISION.md](DIFFUSION_SPEND_POLICY_DECISION.md) | Current v5/v6/v7 cost decision: repairable-denoise spending preserves positives, but the next gate must remove no-lift spend |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V7.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V7.md) | Fresh v7 no-retuning spend check: only two profitable repair rows, six no-lift repairable spends, and five calibrated availability errors |
| [DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V7.md](DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V7.md) | Fresh v7 promotion target artifact: `candidate_aware_promotion_v1` has zero errors and selects `plan_054` and `plan_056` |
| [DIFFUSION_SPEND_GATE_V7_FIT.md](DIFFUSION_SPEND_GATE_V7_FIT.md) | Offline v5/v6/v7 spend-gate fit showing the best simple gate still has five errors and should not be promoted |
| [DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_TRANSFER_MATRIX_V4.md](DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_TRANSFER_MATRIX_V4.md) | Counterfactual span-probe transfer matrix: the zero-error local distinct-retention rule has 3 errors on fresh planning rows, keeping Stage 1 diagnostic-only |
| [DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_CONJUNCTION_TRANSFER_V4.md](DIFFUSION_COUNTERFACTUAL_SPAN_VALIDATED_PROBE_CONJUNCTION_TRANSFER_V4.md) | Exhaustive two-condition probe-rule search: gap/span challenger has 1 local error and 0 fresh-slice errors, but remains diagnostic because it was transfer-screened |
| [DIFFUSION_COUNTERFACTUAL_SPAN_GAP_SPAN_RULE_V4_TRANSFER_V3_PLANNING.md](DIFFUSION_COUNTERFACTUAL_SPAN_GAP_SPAN_RULE_V4_TRANSFER_V3_PLANNING.md) | Frozen next-slice falsifier: the gap/span challenger has 3 errors on `plan_017`-`plan_024`, so Stage 1 still needs richer probe signatures |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_MODEL_V4.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_MODEL_V4.md) | Leave-slice-out signature-model audit: all 15 positives are preserved across 28 span-v4 rows, but 11 false positives keep the gate diagnostic-only |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_UTILITY_FRONTIER_V4.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNATURE_UTILITY_FRONTIER_V4.md) | Cost-penalized signature frontier: `0.020000` keeps 11 false positives, while `0.050000` misses 13 positives |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_NO_LIFT_VETO_V4.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_NO_LIFT_VETO_V4.md) | No-lift veto falsifier: 73,154 threshold-fragment rules reduce false positives from 11 to 8 but introduce five false negatives |
| [docs/DIFFUSION_MOONSHOT_REASONING_ARCHITECTURE_V1.md](docs/DIFFUSION_MOONSHOT_REASONING_ARCHITECTURE_V1.md) | Moonshot-aligned next architecture: replace scalar/veto gates with signed realized-value tomography, external anchors, and consolidation |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_V4.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_V4.md) | M1 signed-value head: improves signed utility to `0.582500` with 9 false positives and 0 false negatives, but does not clear the `0.625500` promotion bar |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_CONTROLS_V4.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_CONTROLS_V4.md) | M2 signed-value controls: all matched-k and best-withheld feature-family controls degrade, supporting distributed signature use but not promotion |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_WEAK_SLICE_V4.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_SIGNED_VALUE_WEAK_SLICE_V4.md) | Weak-slice diagnosis: `plan_017`-`plan_024` selects 8 rows for only 2 positives and `0.001429` signed utility, so the next feature must calibrate cohort value density |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COHORT_RISK_V4.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COHORT_RISK_V4.md) | Neighbor-risk calibration: improves global signed utility to `0.685500`, but still selects the weak cohort wholesale, so GPU promotion remains blocked |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_TRAJECTORY_RELATIVE_GATE_V4.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_TRAJECTORY_RELATIVE_GATE_V4.md) | Trajectory-relative composite gate: improves signed utility to `0.805500` with 0 false negatives and 0 weak-slice false positives, but needs channel controls before GPU |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_TRAJECTORY_RELATIVE_CONTROLS_V4.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_TRAJECTORY_RELATIVE_CONTROLS_V4.md) | Trajectory-channel controls: withholding, delta-only, inverted, and rotated controls all degrade relative to the true composite |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_FREEZE_V4.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_FREEZE_V4.md) | Frozen v10 proof-obligation manifest: new `plan_073`-`plan_080` task slice, measured span-probe pass, all-repairable label pass, and replay gates before any GPU spend |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_V10_REPLAY.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_V10_REPLAY.md) | Fresh v10 GPU replay: preserves all selected-repair positives but selects all 8 planning rows, admits 3 no-lift repairs, and stays below the frozen promotion utility bar |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_V10_VETO_STRESS_FREEZE.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_V10_VETO_STRESS_FREEZE.md) | Frozen fixed-source follow-up to stress the trajectory-relative veto missing from the first v10 replay |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_V10_FIXED_SOURCE_REPLAY.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_COMPOSITE_V10_FIXED_SOURCE_REPLAY.md) | Fixed-source veto-stress replay: trajectory-relative veto blocks no-lift `plan_074`, but two no-lift rows remain and utility stays below promotion |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_V10_NO_LIFT_SPECIFICITY.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_V10_NO_LIFT_SPECIFICITY.md) | Post-label v10 diagnostic frontier: a measured probe-value floor removes the remaining no-lift rows on fixed-source replay, but must be frozen and tested on a fresh source-divergent slice |
| [DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_FREEZE.md](DIFFUSION_COUNTERFACTUAL_SPAN_PROBE_VALUE_FLOOR_V11_FREEZE.md) | Frozen v11 proof-obligation manifest for the measured probe-value floor: new `plan_081`-`plan_088` source-divergent slice, fixed source policy, replay gates, and measurement-cost accounting before GPU labels |
| [DIFFUSION_SPEND_TRANSFER_RULE_FIT.md](DIFFUSION_SPEND_TRANSFER_RULE_FIT.md) | Transfer-rule fit showing `current_decomposed_spend` is the best repair-availability predictor; the older `0.3075` source-task floor is too conservative |
| [DIFFUSION_TRANSFER_PROMOTION_VALUE.md](DIFFUSION_TRANSFER_PROMOTION_VALUE.md) | Historical transfer-promotion report; after repair-only relabeling, current promotion evidence moves to the v5 candidate target artifact |
| [DIFFUSION_TRANSFER_HEAD_FIT.md](DIFFUSION_TRANSFER_HEAD_FIT.md) | Historical first separate transfer-head fit, superseded for current promotion claims by the corrected v5 candidate-promotion labels |
| [DIFFUSION_REASONING_PROOF_OBJECT.md](DIFFUSION_REASONING_PROOF_OBJECT.md) | Canonical proof-object ledger: six heads, 60 target rows, an availability boundary, and a zero-error candidate-aware promotion target |
| [docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md) | Navigation map for the diffusion work so readers can move from public claims to theory, mechanisms, cost controls, and source files |
| [DIFFUSION_PHASE_ANCHOR_BOUNDARY.md](DIFFUSION_PHASE_ANCHOR_BOUNDARY.md) | Full mixed-suite diagnostic showing when pre-generation phase anchors help and why they should remain conditional |
| [DIFFUSION_PHASE_HYBRID_MECHANISM_AUDIT.md](DIFFUSION_PHASE_HYBRID_MECHANISM_AUDIT.md) | Error-correction audit and phase-source loss-target extraction for the strict phase-hybrid run: detect repairable phase, diagnose retention safety, select final/history source, repair, and verify source lift |
| [DIFFUSION_PHASE_SOURCE_POLICY_AUDIT.md](DIFFUSION_PHASE_SOURCE_POLICY_AUDIT.md) | Source-choice policy audit showing why denoise phase is selector evidence first: strict calibrated similarity has zero weighted error, while naive safe-phase history replacement creates three false positives |
| [DIFFUSION_PHASE_SOURCE_THRESHOLD_SWEEP.md](DIFFUSION_PHASE_SOURCE_THRESHOLD_SWEEP.md) | Fresh CUDA threshold sweep showing loose `0.90/0.90/0.90` phase-source promotion regresses to `0.524554`, while strict `0.96` and too-strict `0.97` policies both keep `0.531116`; this motivates the named final-preserving phase operator |
| [docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md](docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md) | Mathematical theory layer: denoise trajectories, verifier geometry, information loss, repair-value proofs, and candidate error functions |
| [docs/DIFFUSION_REASONING_FIELD_IMPLICATIONS.md](docs/DIFFUSION_REASONING_FIELD_IMPLICATIONS.md) | The public narrative for why diffusion-native denoise trajectories change the latent reasoning substrate |
| [docs/DIFFUSION_NATIVE_REASONING_ARCHITECTURE.md](docs/DIFFUSION_NATIVE_REASONING_ARCHITECTURE.md) | The mechanism log for denoise-state selection, repair spend gates, phase windows, and current operator boundaries |
| [docs/LEAN_GPU_DIFFUSION_BENCHMARK_PROTOCOL.md](docs/LEAN_GPU_DIFFUSION_BENCHMARK_PROTOCOL.md) | The lean benchmark protocol: fixed/greedy, random perturbation, and latent repair only |

**Current engineering frontier:** the promoted public result above is still the
claim to share. The newest full GPU tests show why raw phase-source replacement
is too blunt, then fix it with a phase-conditioned hybrid. The strict hybrid
keeps the promoted preservation-seeded repair controls, records denoise phase
timing, and switches to history only when the phase state also passes strict
retention/source-advantage checks. It recovers the same `0.531116` at
`2.625000x` public score/cost point while making phase evidence explicit. The
boundary and hybrid result are recorded in
[DIFFUSION_PHASE_ANCHOR_BOUNDARY.md](DIFFUSION_PHASE_ANCHOR_BOUNDARY.md) and
as claim `moe_mixed_phase_hybrid_preserve_seeded_equivalent_frontier` in the
generated evidence ledger. The mechanism-level audit in
[DIFFUSION_PHASE_HYBRID_MECHANISM_AUDIT.md](DIFFUSION_PHASE_HYBRID_MECHANISM_AUDIT.md)
shows the strict hybrid as an error-correction loop: five selected repairs,
one history-source switch, four final-source keeps, and five positive
repair-vs-source deltas. It also writes
`eval_results/diffusion_language/diffusion_phase_hybrid_loss_targets.jsonl`
with one `trust_history_source` target and four `preserve_final_source` targets,
weighted by observed repair lift. That is the current training objective for
making the source switch less hand-coded. The follow-up
[DIFFUSION_PHASE_SOURCE_POLICY_AUDIT.md](DIFFUSION_PHASE_SOURCE_POLICY_AUDIT.md)
calibrates that objective into the current source-choice rule and records the
failure mode of looser phase replacement. The fresh CUDA threshold sweep in
[DIFFUSION_PHASE_SOURCE_THRESHOLD_SWEEP.md](DIFFUSION_PHASE_SOURCE_THRESHOLD_SWEEP.md)
confirms the boundary on GPU: the loose `0.90/0.90/0.90` rule adds one extra
history switch, lowers `plan_003` by `0.052500`, and drops the planning repair
score by `0.006563` at unchanged cost. The `0.97/0.97/0.95` run removes the
remaining `plan_001` history switch with no score loss, showing the current
public frontier is a strict/final-preserving plateau rather than proof that
history sourcing is required on every repair. The executable policy name for
that plateau is now `constraint_span_phase_final_preserve_seeded_gated`: phase
history still decides when repair is worth spending, but final-state repair
stays the source of truth. Fresh run `diffusion-175cbd422107ee5e` validates the
named operator directly: `0.531116` at `2.625000x`, matching strict `0.96` and
strict `0.97` while using `0` history sources and `5` final sources.

**Public sharing capsule:** we now have a cheap, reproducible GPU benchmark where
a frozen diffusion language model improves reasoning by repairing denoise
trajectories at inference time. The promoted run shows selected latent repair at
`0.531116` versus `0.412277` for greedy/fixed denoise and `0.372125` for random
perturbation, at `2.625000x` relative GPU cost. The repair-spend gate is not
hand-waved: the current audit spends compute on all five productive repairable
cases, skips all three forced no-lift controls, and records `5 / 0 / 3 / 0`
true-positive / false-positive / true-negative / false-negative gate behavior.
The current sweep tests 53,460 source-geometry plus denoise-phase gate settings
and keeps the promoted gate on the score/cost frontier. It also exposes a real
budget knob: cap the first repairable denoise skeleton at step `9` and the
named phase/final operator spends no repair compute for `0.414598` at
`2.000000x`; cap at step `10` or `16` and it spends three repairs for
`0.472500` at `2.375000x`; both caps share run/content ID
`diffusion-f8f6ae3e209d502b` because they select the same repaired tasks and
produce identical generations;
cap at step `20` or `30` and the fresh CUDA runs
`diffusion-419fbf63c9d8e30b` and `diffusion-65f906724fed3cbc` spend four
repairs for `0.496607` at `2.500000x`; the named operator's cap-20 and
cap-30 rows share run/content ID `diffusion-65f906724fed3cbc` because both
select the same four repaired tasks and still skip late `plan_007`; cap at
step `31` or `32` and the fresh runs
`diffusion-3b42951db77c5aa6` and `diffusion-175cbd422107ee5e` use 27
generations, spend five repairs, accept `plan_007` when its first repairable
skeleton appears at step `31`, and recover
the promoted `0.531116` at `2.625000x`. That is the strongest public claim
right now:
diffusion-native latent reasoning is measurable, cheaper than broad sampling
sweeps, and grounded in denoise-state repairability rather than vibes.

To reproduce the fresh cap-32 promoted public run on CUDA:

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --task-preset lean_gpu_mixed --candidates llada-moe-7b-a1b-instruct-hf --limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 --repair-pack constraint_span_anchor_instability_claim_auto_compat_preserve_seeded_gated --repair-source-policy fixed --repair-spend-trigger denoise_phase_repairability --repair-source-min-chars 240 --repair-source-prompt-gap-min 2 --repair-source-prompt-gap-max 9 --repair-source-prompt-coverage-min 0.4 --repair-source-prompt-coverage-max 1.0 --repair-denoise-skeleton-max-step 32 --repair-selector planning_quality_seed_realization_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --evolved-promotion-margin 0.015 --device cuda --dtype bfloat16 --raw-output eval_results\diffusion_language\llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_phase32_fresh_v1_raw.jsonl --scores-output eval_results\diffusion_language\llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_phase32_fresh_v1_scores.json --report-output eval_results\diffusion_language\llada_moe_mixed_compact_span_anchor_instability_claim_auto_compat_preserve_seeded_gated_phase32_fresh_v1_report.md
```

Validated cleaner named phase/final frontier operator:

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --task-preset lean_gpu_mixed --candidates llada-moe-7b-a1b-instruct-hf --limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 --repair-pack constraint_span_phase_final_preserve_seeded_gated --repair-source-policy fixed --repair-spend-trigger denoise_phase_repairability --repair-source-min-chars 240 --repair-source-prompt-gap-min 2 --repair-source-prompt-gap-max 9 --repair-source-prompt-coverage-min 0.4 --repair-source-prompt-coverage-max 1.0 --repair-phase-budget frontier --repair-selector planning_quality_seed_realization_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --evolved-promotion-margin 0.015 --device cuda --dtype bfloat16
```

Then refresh and validate the public claim ledger:

```powershell
python experiments\build_diffusion_claim_evidence.py
python experiments\validate_diffusion_claim_evidence.py
python experiments\validate_diffusion_theory_claim_ledger.py
python experiments\scan_stale_diffusion_docs.py
python experiments\analyze_diffusion_error_function_geometry.py
python experiments\analyze_diffusion_decomposed_selector.py
python experiments\build_diffusion_composite_selector_targets.py
python experiments\fit_diffusion_composite_selector.py
python experiments\evaluate_diffusion_selector_holdout.py
```

Runner-facing four-head selector trigger:

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --task-preset lean_gpu_mixed --candidates llada-moe-7b-a1b-instruct-hf --limit-schedules 2 --limit-evolved-schedules 0 --limit-repair-candidates 1 --repair-pack constraint_span_phase_final_preserve_seeded_gated --repair-source-policy fixed --repair-spend-trigger decomposed_four_head_selector --repair-source-min-chars 240 --repair-source-prompt-gap-min 2 --repair-source-prompt-gap-max 9 --repair-source-prompt-coverage-min 0.4 --repair-source-prompt-coverage-max 1.0 --repair-phase-budget frontier --repair-value-proxy-source-quality-max 0.301429 --repair-selector planning_quality_seed_realization_guarded --repair-promotion-margin 0.02 --trajectory-selector planning_state --evolved-promotion-margin 0.015 --device cuda --dtype bfloat16
```

Historical context: prepending random embedding-scale prefix tokens improves
Qwen3-4B arithmetic by +19.6pp mean over baseline (32% -> 51.6%, n=10
directions), and cross-domain validation on complex planning tasks showed that
perturbation can break attention-sink-induced degenerate generation while
evolved latent vectors surface qualitatively different reasoning. Those results
motivated the current diffusion-native work. See `paper/main.tex` for the
NeurIPS paper draft.

**Original article:** [How to Teach LLMs to Reason for $0.50](https://www.artificialintelligencemadesimple.com/p/how-to-teach-llms-to-reason-for-50)
**Update article:** [ARTICLE_UPDATE.md](ARTICLE_UPDATE.md) — latest findings including planning task cross-domain validation

## Headline Findings

### 1. Diffusion-Native Latent Repair Beats Greedy And Random

The newest result tests the thesis that diffusion models are a better substrate
for latent reasoning because their denoising trajectory remains editable. The
public benchmark deliberately stays narrow and cheap: greedy/fixed denoise,
random perturbation, and one selected latent repair arm over short planning plus
math/symbolic/science guards.

| Arm | Score | Relative cost |
|-----------|:--------:|:------:|
| Greedy/fixed denoise | 0.412277 | 1.000000x |
| Random perturbation | 0.372125 | 1.000000x |
| **Latent repair** | **0.531116** | **2.625000x** |

This is not just best-of-N sampling. The current repair policy scores compact
semantic anchors for compatibility, uses denoise-history geometry to decide
where repair compute is worth spending, and then repairs masked spans directly.
The current automatic preservation-seeded run ties the hand-built seed frontier
while making the seed choice automatic and removing explicit seed/anchor meta
wording from the frontier task.

The latest promoted run keeps the same aggregate score and cost as the prior
automatic compatibility-scored run, but it cleans up the frontier task: `plan_004`
recovers `0.621786` without explicit seed/anchor meta text by fixing
`oracle selected results; preserve claim if disappears` into the denoise tail.

### 2. Random Prefix Tokens Improve Arithmetic Reasoning

Prepending **2 random embedding-scale tokens** to the input of Qwen3-4B (Q4) improves arithmetic accuracy from **32% to 51.6% mean** (+19.6pp, n=10 directions). No training, no fine-tuning, no optimization — just noise at the right scale.

| Condition | Accuracy | Change | n |
|-----------|:--------:|:------:|:-:|
| Baseline (no prefix) | 32.0% | — | 1 |
| Zero embedding (8 tokens) | 36.0% | +4pp | 3 |
| Mean embedding (8 identical) | 36.0% | +4pp | 1 |
| Random noise (1 token) | 42.7% | +10.7pp | 3 |
| **Random noise (2 tokens)** | **51.6%** | **+19.6pp** | **10** |
| Random noise (3 tokens) | 44.0% | +12pp | 10 |
| Random noise (8 tokens) | 44.4% | +12.4pp | 10 |

**Direction doesn't matter for total count** — solve counts vary normally (p=0.66 vs iid). But directions solve **different task subsets**: 10 two-token directions achieve 100% oracle coverage (25/25). The dose-response is **non-monotonic**: 2 tokens is optimal, more tokens degrades back to ~44%.

### 3. Perturbation Breaks Attention-Sink Degenerate Generation

On complex planning tasks (system design, incident response, cache debugging), greedy baseline can fail catastrophically — the **cache debugging task produces only 14 words** before the model gets trapped in an attention-sink loop. All 5 random perturbation seeds rescue this into **650-710 word complete diagnostic plans**. This demonstrates that soft prompt perturbation breaks degenerate greedy generation paths induced by attention sink patterns in the first few positions.

### 4. Evolution Surfaces Qualitatively Different Reasoning

Evolved latent vectors (via trained scorer + evolutionary search) don't just produce more words — they produce **genuinely different reasoning**. On the incident response task, evolution surfaces honeypot deployment, MITRE ATT&CK framework analysis, tiered credential rotation with HSM integration, immutable container rebuilds, and DMZ isolation strategies. The baseline never produces these concepts. This is not style variation — it's accessing different knowledge and reasoning paths in the model's parameter space.

### A New Axis of Improvement

This effect is **orthogonal to all known LLM improvement methods**:
- **Scaling**: Adds parameters. We change zero parameters.
- **Fine-tuning**: Updates weights. We leave weights frozen.
- **Prompt engineering**: Optimizes discrete tokens. We inject continuous embeddings.
- **RAG**: Adds external knowledge. We unlock internal knowledge.
- **Sampling (best-of-N)**: Generates N outputs, picks best. We run N cheap scorer evals on latent vectors + 1 generation pass.

The efficiency advantage over best-of-N is significant: evolution needs N tiny MLP forward passes (the latent scorer) plus a single generation pass, versus N full autoregressive generation passes for best-of-N sampling.

## What's Actually Happening

### In Arithmetic: Trajectory Perturbation

The prefix shifts the model from "formal presentation mode" (structured LaTeX, truncates before computing) into "exploratory computation mode" (informal, but actually does math). This is **trajectory perturbation** — a policy change, not a capability gain.

- **Chain-of-thought mediates**: disabling thinking eliminates the effect entirely
- **Trajectory modulation**: first-token logit probe shows <think> is saturated (>99.99%) under all conditions — perturbation modulates the reasoning chain, not mode entry
- **Task-selective**: different directions solve different tasks, enabling oracle coverage
- **Token budget**: wrong answers hit max_new_tokens ceiling, correct answers finish early

### In Planning: Attention Sink Avoidance + Knowledge Unlocking

On complex planning tasks, two distinct mechanisms emerge:

1. **Attention sink avoidance**: Greedy decoding can get trapped when early tokens (attention sinks) lock the model into degenerate generation paths. Soft prompt perturbation in the first 2 positions disrupts this, breaking the degeneracy. The most dramatic example: the cache debugging task baseline produces only 14 words before collapsing, while every perturbation seed produces a complete 650+ word diagnostic plan.

2. **Latent knowledge access via evolution**: Evolved soft prompts don't just break attention sinks — they steer the model into different regions of its knowledge space. The evolved incident response plan includes honeypot deployment, MITRE ATT&CK framework references, and HSM-backed credential rotation — none of which appear in baseline or random perturbation outputs. The model *knows* these concepts but doesn't access them under default greedy decoding.

### The Underlying Mechanism

The soft prompt system consistently improves over the bare baseline. The mechanism operates at two levels:

- **Random perturbation** (direction-agnostic): breaks degenerate attention patterns and shifts output policy. Random noise matches W-projected latents (p = 1.0). Robust and requires zero optimization.
- **Evolved perturbation** (direction-sensitive): the trained latent scorer guides evolution toward soft prompts that access specific knowledge and reasoning modes. Currently limited by a barely-trained scorer, but already surfaces qualitatively different outputs.

See [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md) for the full technical summary. Details on the warm-start mechanism in [ARTICLE_UPDATE.md](ARTICLE_UPDATE.md).

## Installation

```bash
git clone https://github.com/devansh/latent-space-reasoning.git
cd latent-space-reasoning
pip install -e .
```

Optional dependencies:

```bash
pip install -e ".[dev]"    # tests/lint/type-check
pip install -e ".[quant]"  # bitsandbytes 4-bit quantization support
```

### Requirements

- **Python**: 3.10+ (tested with 3.13)
- **PyTorch**: 2.0+ with CUDA support recommended
- **Memory**:
  - Minimum: ~2GB VRAM (Qwen3-0.6B)
  - Recommended: ~8GB VRAM (Qwen3-4B)
  - CPU-only: Supported but slower

## Quick Start

### Compare Methods (Recommended)

The best way to see the difference is to run both baseline and latent reasoning on the same query:

```bash
# Basic comparison - see the difference immediately
latent-reason compare "How do I implement user authentication?"

# Accessibility-first profile (CPU, low-resource defaults)
latent-reason compare "How do I implement user authentication?" --config configs/aim_v1_low_resource.yaml

# With a larger model
latent-reason compare "Design a REST API" --encoder Qwen/Qwen3-4B

# Save results for analysis
latent-reason compare "Optimize database queries" --output results.json
```

### Simple Usage

```bash
latent-reason run "How do I implement caching?"
latent-reason run "Design a microservices architecture" --encoder Qwen/Qwen3-1.7B
latent-reason run "Optimize database performance" --chains 8 --generations 15
```

### Python API

```python
from latent_reasoning import reason, compare, Engine

result = reason("How do I implement caching?")
print(result.plan)

cmp = compare("How do I implement rate limiting?")
print(cmp["baseline"])
print(cmp["latent_reasoning"])

engine = Engine()
advanced = engine.run("Design an API")
print(advanced.generations, advanced.evaluations)
```

### Check Your Setup

```bash
latent-reason check-gpu
latent-reason models
```

## Models

| Model | Size | VRAM | Best For |
|-------|------|------|----------|
| `Qwen/Qwen3-4B` | 4B | ~8 GB | Best quality output |
| `Qwen/Qwen3-1.7B` | 1.7B | ~4 GB | Balance of speed/quality |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | ~3 GB | Strong reasoning, efficient |
| `Qwen/Qwen3-0.6B` | 0.6B | ~2 GB | Fast iteration, CPU-friendly |
| `microsoft/phi-2` | 2.7B | ~6 GB | Alternative option |
| `ibm-granite/granite-4.0-h-1b` | 1B | ~2 GB | Compact alternative |

Qwen3 models generally produce the highest quality output. DeepSeek-R1-Distill is particularly strong for reasoning tasks.

## Configuration

Use `config.example.yaml` as the full schema reference. For accessibility-focused runs, start with `configs/aim_v1_low_resource.yaml`.

```bash
latent-reason run "query" --config config.yaml
latent-reason compare "query" --config config.yaml
```

## Repository Map

```
src/latent_reasoning/
  engine.py              # Main Engine class - primary interface
  reason.py              # Simple reason() function
  config.py              # Configuration schema and defaults
  cli/main.py            # CLI commands
  core/
    encoder.py           # LLMEncoder: encode/decode with transformer models
    judge.py             # Scoring: TrainedLatentJudge, ScorerJudge, etc.
    panel.py             # JudgePanel: aggregates multiple scorers
    chain.py             # ChainState: tracks evolution history
  decode/
    projection.py        # Orthogonal W projection for soft prompts
    steering.py          # Intermediate layer steering
  diffusion/
    backends.py          # Local diffusion model loading and generation backends
    trajectory.py        # Denoise trajectory/state scoring
    repair.py            # Masked-span repair policies and selectors
    control.py           # Compact control terms and denoise-tail steering
    candidates.py        # Repair candidate packs used by benchmark runners
  evolution/
    loop.py              # EvolutionLoop: main evolution algorithm
    selection.py         # Selection strategies
    mutation.py          # Mutation strategies
    crossover.py         # Crossover strategies
  orchestrator/
    orchestrator.py      # Coordinates full pipeline
  utils/
    hyperbolic.py        # Poincare ball / hyperbolic geometry utilities
    logging.py           # Structured logging and progress display
experiments/
  run_diffusion_three_arm_benchmark.py  # Public fixed/random/latent-repair GPU ledger
  build_diffusion_claim_evidence.py     # Regenerates public benchmark and evidence map
  validate_diffusion_claim_evidence.py  # Validates promoted claim artifacts
  validate_diffusion_theory_claim_ledger.py  # Validates theory statuses, falsifiers, and proof obligations
  analyze_diffusion_error_function_geometry.py  # Derives next-loss assertions from repair/source targets
  analyze_diffusion_decomposed_selector.py    # Compares four-term decomposed selectors against single repairability labels
  build_diffusion_composite_selector_targets.py  # Builds trainable rows for the four-term controller
  fit_diffusion_composite_selector.py        # Fits a tiny interpretable four-head selector baseline
  evaluate_diffusion_selector_holdout.py     # Leave-one-task-out check against a single-label repairability controller
  evaluate_diffusion_independent_spend_transfer.py  # Independent spend-head transfer check from all-repairable GPU labels
  scan_stale_diffusion_docs.py          # Guards public docs against stale diffusion claims
  run_latent_sensitivity.py   # Main experiment runner (all controls)
  analyze_error_taxonomy.py   # Per-task error analysis
  create_figures.py           # Publication-quality figure generation
  harness.py                  # Unified experiment harness
  EXPERIMENTS.md              # Full experiment log (reverse chronological)
  ledger.jsonl                # Machine-readable experiment ledger
  figures/                    # Generated figures (7 publication plots)
tests/                        # Unit and integration tests (342 tests)
```

## Key Documentation

| Document | Purpose |
|----------|---------|
| [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md) | Current public three-arm diffusion result: greedy, random perturbation, and selected latent repair with relative GPU cost |
| [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md) | Promoted claim ledger tying public diffusion claims to score, report, raw-generation, run ID, and validation artifacts |
| [DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md) | Canonical score/report/raw pointers and content hashes for promoted diffusion claims |
| [docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md) | Reader-facing map of the current diffusion work and evidence hierarchy |
| [docs/DIFFUSION_THEORY_CLAIM_LEDGER.md](docs/DIFFUSION_THEORY_CLAIM_LEDGER.md) | Conservative theory claim ledger with falsifiers and next proof obligations |
| [DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md](DIFFUSION_REPAIRABILITY_GEOMETRY_AUDIT.md) | Denoise-phase repair-spend gate audit with true-positive/true-negative skip controls |
| [DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md](DIFFUSION_REPAIRABILITY_GEOMETRY_SWEEP.md) | Source-geometry plus denoise-phase gate sweep for the current score/cost frontier |
| [DIFFUSION_BUDGET_POLICY_LOSS.md](DIFFUSION_BUDGET_POLICY_LOSS.md) | Cost-aware marginal repair-value objective for learned repair spending |
| [DIFFUSION_BUDGET_VALUE_PROXY_AUDIT.md](DIFFUSION_BUDGET_VALUE_PROXY_AUDIT.md) | Runner-ready low-cost value proxy with fresh CUDA confirmation |
| [DIFFUSION_REPAIR_VALUE_GEOMETRY.md](DIFFUSION_REPAIR_VALUE_GEOMETRY.md) | Feature-geometry map for the learned repair-value controller |
| [DIFFUSION_ERROR_FUNCTION_GEOMETRY.md](DIFFUSION_ERROR_FUNCTION_GEOMETRY.md) | Generated bridge from collected repair/source rows to next-generation loss functions |
| [DIFFUSION_DECOMPOSED_SELECTOR_AUDIT.md](DIFFUSION_DECOMPOSED_SELECTOR_AUDIT.md) | Generated selector comparison proving the current four-term decomposed controller beats single-label repairability on the local target rows |
| [DIFFUSION_COMPOSITE_SELECTOR_TARGETS.md](DIFFUSION_COMPOSITE_SELECTOR_TARGETS.md) | Trainable target rows for learning the four-term diffusion controller |
| [DIFFUSION_COMPOSITE_SELECTOR_FIT.md](DIFFUSION_COMPOSITE_SELECTOR_FIT.md) | Interpretable four-head selector fit that later learned controllers must beat on held-out target surfaces |
| [DIFFUSION_SELECTOR_HOLDOUT_EVAL.md](DIFFUSION_SELECTOR_HOLDOUT_EVAL.md) | Leave-one-task-out test showing the decomposed selector keeps a `66.666667%` relative error reduction over a single repairability baseline |
| [DIFFUSION_COMPOSITE_SELECTOR_RUNNER_POLICY.md](DIFFUSION_COMPOSITE_SELECTOR_RUNNER_POLICY.md) | CLI and diagnostics for the executable `decomposed_four_head_selector` spend trigger |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER.md) | Historical four-prompt transfer result from the pre-v5 repair-oracle labeling path |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V2.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V2.md) | Expanded eight-planning-row transfer result with the same positive `plan_012` repair-availability label |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V3.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V3.md) | Larger proof-object transfer result over 16 planning rows under repair-only labels; trajectory-relative spend adds useful selected-trajectory evidence but still has one error |
| [DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md](DIFFUSION_AVAILABILITY_PREDICTOR_FIT.md) | Interpretable corrected availability predictor fit showing pre-repair geometry is useful but not solved |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V4.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V4.md) | Fresh v4 transfer boundary for availability: fixed source-quality and gap rules do not transfer cleanly |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V5.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V5.md) | Fresh v5 transfer boundary: calibrated pre-repair availability is useful but insufficient without candidate-aware promotion value |
| [DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md](DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V5.md) | Candidate-aware promotion labels from v5 repair diagnostics; the named post-repair selector has zero local promotion errors |
| [DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V6.md](DIFFUSION_INDEPENDENT_SPEND_TRANSFER_V6.md) | Fresh v6 transfer boundary showing calibrated spend gating fails while generated repairs remain valuable |
| [DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V6.md](DIFFUSION_CANDIDATE_PROMOTION_TARGETS_V6.md) | Candidate-aware promotion labels from v6 repair diagnostics; zero promotion errors over eight generated candidates |
| [DIFFUSION_SPEND_TRANSFER_RULE_FIT.md](DIFFUSION_SPEND_TRANSFER_RULE_FIT.md) | Transfer rule fit: current decomposed spend has zero repair-availability errors; source-task floors above `0.295357` create a false negative |
| [DIFFUSION_SPEND_TRANSFER_RULE_FIT_V2.md](DIFFUSION_SPEND_TRANSFER_RULE_FIT_V2.md) | Expanded transfer fit confirming the same rule boundary over eight independent planning rows |
| [DIFFUSION_TRANSFER_PROMOTION_VALUE.md](DIFFUSION_TRANSFER_PROMOTION_VALUE.md) | Historical promotion-value boundary; current repair-only promotion evidence is the v5 candidate-promotion target file |
| [DIFFUSION_TRANSFER_HEAD_FIT.md](DIFFUSION_TRANSFER_HEAD_FIT.md) | Separate availability and promotion-value transfer heads over current generated rows |
| [DIFFUSION_REASONING_PROOF_OBJECT.md](DIFFUSION_REASONING_PROOF_OBJECT.md) | Falsifiable decomposed reasoning proof object tying theory heads to target rows, information channels, evidence, and next GPU tests |
| [docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md](docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md) | Mathematical theory layer for information loss, denoise geometry, proof obligations, and next error functions |
| [docs/DIFFUSION_REASONING_FIELD_IMPLICATIONS.md](docs/DIFFUSION_REASONING_FIELD_IMPLICATIONS.md) | Public narrative for why denoise-trajectory control is the next latent reasoning substrate |
| [docs/LEAN_GPU_DIFFUSION_BENCHMARK_PROTOCOL.md](docs/LEAN_GPU_DIFFUSION_BENCHMARK_PROTOCOL.md) | Cheap local GPU protocol for fixed, random, and latent repair comparisons |
| [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md) | Technical summary with data tables and figures |
| [ARTICLE_UPDATE.md](ARTICLE_UPDATE.md) | Accessible article covering all findings |
| [GOALS.md](GOALS.md) | Active research goals and completed milestones |
| [TASKS.md](TASKS.md) | Current task board and experiment queue |
| [experiments/EXPERIMENTS.md](experiments/EXPERIMENTS.md) | Full experiment log with methodology |

## Development

```bash
make install-dev
make test      # 342 tests
make lint
make check
```

## Current Research Status

**Phase: Cross-domain validation and mechanism characterization** (see [TASKS.md](TASKS.md))

Completed:
- Non-monotonic dose-response: 2 tokens optimal (+19.6pp mean, n=10)
- Oracle coverage: 100% from 10 two-token directions (vs 80% 3-tok, 92% 8-tok)
- Think-gate probe: mode gating falsified, mechanism is trajectory modulation
- Controls: zero embedding, mean embedding, no-think, explicit think-prefix
- Equalization negative result: n=3 pattern did not replicate at n=10
- **Cross-domain validation**: 3-way comparison on 5 complex planning tasks (baseline vs perturbation vs evolution, all at 2048 tokens)
- **Attention sink avoidance**: perturbation rescues catastrophic baseline failures
- **Evolution quality**: evolved latents surface qualitatively different reasoning
- Multi-model validation: Qwen3-4B, Qwen3-8B (8-bit), DeepSeek-1.5B, phi-2
- **Model-dependent mechanism**: 4B aids convergence only (80% answer-anywhere); 8B 8-bit aids both computation (+18pp answer-anywhere) and convergence

### Cross-Model Arithmetic Results

| Model | Quant | n | Baseline | +Noise | Delta | Oracle | McNemar p |
|-------|-------|---|----------|--------|-------|--------|-----------|
| Qwen3-4B | 4-bit | 10 | 32% | 51.6% | +19.6pp | 100% | 0.000015 |
| Qwen3-8B | 8-bit | 10 | 16% | 28.8% | +12.8pp | 80% | 0.000177 |
| DeepSeek-1.5B | 4-bit | 10 | 76% | 74.4% | -1.6pp | 100% | 0.031 |
| phi-2 | none | 3 | 12% | 18.7% | +6.7pp | 28% | 0.125 |

**Convergence vs computation**: For Qwen3-4B (high computational ceiling, 80% answer-anywhere), perturbation primarily aids convergence — the model already computes correctly but fails to put the answer last. For Qwen3-8B 8-bit (low ceiling, 32% answer-anywhere), perturbation improves actual computation (+18pp answer-anywhere) as well as convergence.

### Planning Task 3-Way Comparison

| Task | Baseline | Perturbation | Evolution | Winner |
|------|----------|-------------|-----------|--------|
| Fraud Detection | 5.6/10 | 4.8/10 | **5.8/10** | Evolution |
| Incident Response | 5.6/10 | **7.4/10** | 5.8/10 | Perturbation |
| Data Platform | 5.8/10 | **7.2/10** | 5.8/10 | Perturbation |
| Cache Debugging | 29/50 | **39/50** | 25/50 | Perturbation |
| DB Migration | 14/50 | 24/50 | **35/50** | Evolution |

**Judge tally**: Perturbation 3/5, Evolution 2/5, Baseline 0/5.

### Legal Reasoning 3-Way Comparison (NEW)

12 complex legal scenarios across 5 categories, blind-reviewed by Codex CLI. All 12 tasks reviewed.

| Metric | Result |
|--------|--------|
| Oracle perturbation beats baseline | **11/12 tasks (92%)** |
| Average oracle lift | **+1.6 points** (10-point scale) |
| Peak improvement | **+3.4 points** (negotiation, contractor misclass) |
| Mean wins | Base 4, Perturbation 4, Evolution 1 |

**Key finding: The model has latent legal knowledge it cannot access via greedy decoding.** Perturbation consistently unlocks better analysis — multi-jurisdictional reasoning, regulatory frameworks, and strategic analysis that standard generation misses entirely.

| Task | Category | Baseline | Best Perturbation | Lift |
|------|----------|:---:|:---:|:---:|
| FTC Unfairness | Regulatory | 5.2 | 7.2 (evolution) | +2.0 |
| Negotiation Leverage | Strategic | 2.0 | 5.4 | +3.4 |
| Contractor Misclass | Scenario | 2.2 | 5.6 | +3.4 |
| IP Risk Portfolio | IP | 3.6 | 6.4 | +2.8 |
| SaaS Contract | Transactional | 4.0 | 5.6 | +1.6 |
| Disparate Impact | Employment | 6.0 | 6.8 | +0.8 |

> **This system is explicitly judge-heavy.** The perturbation mechanism reliably accesses latent knowledge (proven by the 92% oracle win rate). The degree to which that knowledge is captured depends on judge/scorer quality. The minimally-trained scorer used here captures some of the ceiling (evolution wins on task 01), but better judges — like those from [Irys](https://irys.ai) or [Iqidis](https://iqidis.ai) — would capture substantially more.

Full showcase data: [`experiments/legal_showcase.json`](experiments/legal_showcase.json)

Next experiments:
- Clean re-run with fixed scorer (deterministic projection)
- Better latent scorers for more consistent evolution gains
- Larger planning/legal task sets for statistical power
- Attention probing to confirm the attention sink mechanism directly

## Limitations

- **Single model for planning/legal**: All cross-domain comparisons on Qwen3-4B. Arithmetic tested on 4 models.
- **Modest n**: 25 arithmetic tasks, 5 planning tasks, 12 legal tasks.
- **Effect is redistribution** in arithmetic: some tasks improve, others regress.
- **Judge-heavy by design**: This is a feature, not a bug. The system's quality ceiling is determined by judge quality. The current trained latent judge is barely trained — evolution results are promising but inconsistent. Better judges and evolution strategies (e.g., [Iqidis](https://iqidis.ai) / [Irys](https://irys.ai) approaches) should yield more reliable gains. The oracle analysis proves the latent knowledge exists; the judge determines how much is captured.

## Contributing

Contributions welcome! Areas of interest:
- New evolution strategies (selection, mutation, crossover)
- Alternative scoring methods (semantic, heuristic, learned)
- Evaluation benchmarks and metrics
- Model architecture experiments
- Performance optimizations

The point of open sourcing is to push the boundaries and explore crazy ideas, so don't be scared to explore a lot.

### Monthly Bounty Program ($2,000/month)

[Iqidis](https://iqidis.ai) sponsors a monthly bounty pool for the top 10 contributors:

| Rank | Bounty |
|------|--------|
| 1st | $500 |
| 2nd | $350 |
| 3rd | $275 |
| 4th | $200 |
| 5th | $175 |
| 6th | $150 |
| 7th | $125 |
| 8th | $100 |
| 9th | $75 |
| 10th | $50 |

**Additional perks:**
- All Top 10 contributors listed in README
- Active contributors offered interviews at [Iqidis](https://iqidis.ai) and access to our network of **1.5M+ members** including engineers, managers, and builders from Google, Nvidia, OpenAI, Anthropic, Meta AI, and other top AI organizations

Bounties given out monthly on the 15th.

## Exclusive Access for AI Made Simple Founding Members

**Founding members of [AI Made Simple](https://www.artificialintelligencemadesimple.com/subscribe)** get exclusive access to:

- **391-query comprehensive test set** - Extensive evaluation across different model families, configurations, and setups
- **Detailed analysis** - Full breakdown of performance across various scenarios
- **Research updates** - Early access to findings from ongoing V10-V14+ experiments

### Production Considerations

This open-source release provides the core engine and research artifacts. For production systems, you would likely need:

1. **Better Judge Models**: The shared checkpoint is a basic trained scorer. Production systems benefit from judges trained on domain-specific data with more sophisticated architectures.

2. **Smarter Aggregation**: This implementation uses simple mean pooling to combine evolved latents. Production systems can use more sophisticated approaches. For example, [Iqidis](https://iqidis.ai) (the team behind this repo) uses a **reverse Mixture of Experts** architecture - a learned MLP that analyzes all evolved latents natively, scores them, and determines the optimal way to combine them into the final output.

3. **Continuous Training**: Judge models improve with ongoing training on new data and feedback loops.

**Bottom line**: The results shown here use the simple shared checkpoint and open research code. Better judges, conditioning methods, and aggregation strategies can yield significantly better results, but those components require substantial investment to develop and are often proprietary.

## License

MIT - Use or modify the code for whatever you want. All commercial applications are welcome and encouraged.

## Citation

```bibtex
@software{latent_space_reasoning,
  title={Latent Space Reasoning Engine},
  author={Devansh},
  year={2025},
  url={https://github.com/devansh/latent-space-reasoning}
}
```
