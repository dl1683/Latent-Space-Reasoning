# Diffusion Report Archive

This folder contains generated and historical diffusion reports.

These files are kept for audit and reproduction, but they are not the recommended
entry path for new readers. Start with the root [README](../../../README.md),
[DIFFUSION_PUBLIC_BENCHMARK.md](../../../DIFFUSION_PUBLIC_BENCHMARK.md),
[CLAIM_EVIDENCE_MAP.md](../../../CLAIM_EVIDENCE_MAP.md), and
[DIFFUSION_GROUND_TRUTH_INDEX.md](../../../DIFFUSION_GROUND_TRUTH_INDEX.md).

Current high-signal generated reports:

- [LATENT_TRAJECTORY_AGGREGATION_SCOUT.md](LATENT_TRAJECTORY_AGGREGATION_SCOUT.md):
  deterministic component-aggregation scaffold showing one local compositional
  promotion and two blocked safety cases; protocol evidence only, not a
  promoted model result.
- [LATENT_AGGREGATION_RUBRIC_REPLAY.md](LATENT_AGGREGATION_RUBRIC_REPLAY.md):
  post-hoc replay over existing planning rubric-hit labels; estimates oracle
  component-union headroom, not inference-time aggregation.
- [LATENT_AGGREGATION_INFERENCE_V1_FREEZE.md](LATENT_AGGREGATION_INFERENCE_V1_FREEZE.md):
  pre-label contract for the first inference-time aggregation validation,
  including frozen task IDs, extractor limits, realization requirements, and
  statistical gates.
- [LATENT_AGGREGATION_INFERENCE_SMOKE_REPLAY.md](LATENT_AGGREGATION_INFERENCE_SMOKE_REPLAY.md):
  deterministic smoke replay for the inference-time extractor/fuser/realizer
  harness; useful for pipeline validation, but explicitly not frozen GPU
  evidence.
- [LATENT_AGGREGATION_INFERENCE_V1_REPLAY.md](LATENT_AGGREGATION_INFERENCE_V1_REPLAY.md):
  real frozen GPU replay over 16 planning tasks; negative aggregation result
  with `0/16` online promotions and failed predeclared statistical gates.
- [LATENT_AGGREGATION_EXTRACTOR_FAILURE_V1.md](LATENT_AGGREGATION_EXTRACTOR_FAILURE_V1.md):
  post-hoc extractor failure diagnostic showing the current literal threshold
  is too conservative on the frozen replay labels; useful for the next freeze,
  not a promoted online result.
- [LATENT_AGGREGATION_INFERENCE_V1_THRESHOLD01_REPLAY.md](LATENT_AGGREGATION_INFERENCE_V1_THRESHOLD01_REPLAY.md):
  post-hoc threshold replay using support threshold `0.1`; component extraction
  recovers the frozen labels, mean realized score rises, but predeclared
  promotion gates still fail with only `1/16` online promotions.
- [LATENT_AGGREGATION_GAIN_FAILURE_THRESHOLD01.md](LATENT_AGGREGATION_GAIN_FAILURE_THRESHOLD01.md):
  diagnostic over the threshold `0.1` replay showing the remaining bottleneck:
  `12/16` tasks lift final score without positive component gain, split between
  best-single reformatting and multi-source selection with no new components.
- [LATENT_AGGREGATION_SCORE_DIMENSION_GAP_THRESHOLD01.md](LATENT_AGGREGATION_SCORE_DIMENSION_GAP_THRESHOLD01.md):
  diagnostic showing why rubric-only component gain is too narrow; `12/16`
  score-lift/no-gain tasks improve through non-rubric scoring dimensions, and
  `4/16` do so after the best single already saturates rubric coverage.
- [LATENT_AGGREGATION_MULTI_ASPECT_V2_FREEZE.md](LATENT_AGGREGATION_MULTI_ASPECT_V2_FREEZE.md):
  frozen held-out v2 contract over `plan_025..plan_048` that promotes
  aggregation from rubric-item stitching to multi-aspect latent fusion across
  rubric, causal, specificity, constraint, and risk aspects.
- [LATENT_AGGREGATION_MULTI_ASPECT_V2_LABEL_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V2_LABEL_REPORT.md):
  GPU label-generation report for the frozen v2 held-out slice; repair coverage
  is `24/24` eligible tasks with repair task score `0.378586` versus fixed
  `0.336354` on repair-covered tasks.
- [LATENT_AGGREGATION_MULTI_ASPECT_V2_HEADROOM.md](LATENT_AGGREGATION_MULTI_ASPECT_V2_HEADROOM.md):
  held-out diagnostic over the v2 GPU rows showing modest complement material:
  `9/24` tasks have any complement aspect and `5/24` have dimension complements.
- [LATENT_AGGREGATION_MULTI_ASPECT_V2_REPLAY.md](LATENT_AGGREGATION_MULTI_ASPECT_V2_REPLAY.md):
  deterministic held-out v2 replay; `9/24` tasks promote locally and win-count
  gates pass, but the overall frozen gate fails because mean non-rubric lift is
  `0.027083` below the predeclared `0.030000` threshold.
- [LATENT_AGGREGATION_MULTI_ASPECT_V2_FAILURE.md](LATENT_AGGREGATION_MULTI_ASPECT_V2_FAILURE.md):
  post-replay failure analysis showing the miss is coverage-driven: complement
  tasks average `0.072222` non-rubric lift, but only `9/24` tasks have complement
  material, diluting the all-task mean below threshold.
- [LATENT_AGGREGATION_MULTI_ASPECT_V2_COVERAGE_GAP.md](LATENT_AGGREGATION_MULTI_ASPECT_V2_COVERAGE_GAP.md):
  coverage blocker diagnostic showing `14/15` no-complement tasks are anchor
  dominance on the frozen aspect ontology and only `1/15` is a below-threshold
  near miss.
- [LATENT_AGGREGATION_MULTI_ASPECT_V3_FREEZE.md](LATENT_AGGREGATION_MULTI_ASPECT_V3_FREEZE.md):
  pre-label v3 freeze over new `plan_201`-`plan_224` tasks, with targeted
  aspect-deficit probes, separate coverage and conditional-quality gates, probe
  cost accounting, and equal-budget best-of controls.
- [LATENT_AGGREGATION_MULTI_ASPECT_V3_PROBE_DRY_RUN_FAILURE.md](LATENT_AGGREGATION_MULTI_ASPECT_V3_PROBE_DRY_RUN_FAILURE.md):
  failed first GPU probe dry run; the command used `--limit-repair-candidates 0`,
  so the runner skipped the probe gate and generated `0` counterfactual probes.
- [LATENT_AGGREGATION_MULTI_ASPECT_V3_PROBE_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V3_PROBE_REPORT.md):
  corrected GPU probe run with `24` counterfactual probe generations over the
  v3 task slice.
- [LATENT_AGGREGATION_MULTI_ASPECT_V3_PROBE_ANALYSIS.md](LATENT_AGGREGATION_MULTI_ASPECT_V3_PROBE_ANALYSIS.md):
  v3 probe analysis showing `23/24` stage-1-valid probe texts, zero full-repair
  authorizations, and negative mean probe-task utility versus the source.
- [LATENT_AGGREGATION_MULTI_ASPECT_V3_LABEL_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V3_LABEL_REPORT.md):
  frozen v3 GPU label run showing `24/24` eligible repair coverage, repair task
  score `0.350000`, and `+0.036042` task-score lift over the selected trajectory.
- [LATENT_AGGREGATION_MULTI_ASPECT_V3_REPLAY.md](LATENT_AGGREGATION_MULTI_ASPECT_V3_REPLAY.md):
  deterministic frozen v3 aggregation replay; selected complements promote on
  `6/6` covered tasks, but the full v3 gate fails because complement coverage is
  only `6/24`.
- [LATENT_AGGREGATION_MULTI_ASPECT_V3_FAILURE.md](LATENT_AGGREGATION_MULTI_ASPECT_V3_FAILURE.md):
  post-replay v3 failure analysis showing the binding failure is complement
  coverage, not conditional complement quality.
- [LATENT_AGGREGATION_MULTI_ASPECT_V3_PROBE_AUGMENTED_REPLAY.md](LATENT_AGGREGATION_MULTI_ASPECT_V3_PROBE_AUGMENTED_REPLAY.md):
  replay that adds the corrected probe raw rows as an extra latent source;
  coverage improves only from `6/24` to `7/24`, so the frozen gate still fails.
- [LATENT_AGGREGATION_MULTI_ASPECT_V3_PROBE_AUGMENTED_FAILURE.md](LATENT_AGGREGATION_MULTI_ASPECT_V3_PROBE_AUGMENTED_FAILURE.md):
  failure analysis for the probe-augmented replay showing that even with probes,
  the binding gap remains the `12/24` complement-coverage gate.
- [LATENT_AGGREGATION_MULTI_ASPECT_V3_COVERAGE_GAP.md](LATENT_AGGREGATION_MULTI_ASPECT_V3_COVERAGE_GAP.md):
  baseline v3 coverage-gap diagnostic showing all `18` no-complement tasks are
  anchor-dominance cases, with no below-threshold near misses.
- [LATENT_AGGREGATION_MULTI_ASPECT_V3_PROBE_AUGMENTED_COVERAGE_GAP.md](LATENT_AGGREGATION_MULTI_ASPECT_V3_PROBE_AUGMENTED_COVERAGE_GAP.md):
  probe-augmented coverage-gap diagnostic showing probes add one covered task
  but leave `17` anchor-dominance no-complement tasks.
- [LATENT_AGGREGATION_MULTI_ASPECT_V3_DIVERSITY_EXTENSION_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V3_DIVERSITY_EXTENSION_REPORT.md):
  GPU diversity-extension run adding LLaDA evolved/revision schedule candidates
  as post-failure source-generation evidence.
- [LATENT_AGGREGATION_MULTI_ASPECT_V3_DIVERSITY_AUGMENTED_REPLAY.md](LATENT_AGGREGATION_MULTI_ASPECT_V3_DIVERSITY_AUGMENTED_REPLAY.md):
  post-failure multi-source replay over label, probe, and diversity-extension
  rows; numeric v3 gates pass with `13/24` complement coverage and `13` local
  promotions, but this is diagnostic design evidence rather than the original
  predeclared v3 promotion.
- [LATENT_AGGREGATION_MULTI_ASPECT_V3_DIVERSITY_AUGMENTED_COVERAGE_GAP.md](LATENT_AGGREGATION_MULTI_ASPECT_V3_DIVERSITY_AUGMENTED_COVERAGE_GAP.md):
  coverage-gap diagnostic for the diversity-augmented replay showing remaining
  no-complement tasks drop to `11`, all still anchor-dominance cases.
- [LATENT_AGGREGATION_MULTI_ASPECT_V4_FREEZE.md](LATENT_AGGREGATION_MULTI_ASPECT_V4_FREEZE.md):
  fresh `plan_225`-`plan_248` replication contract that predeclares the
  diversity-extension source mix before labels, removing the v3 post-failure
  caveat.
- [LATENT_AGGREGATION_MULTI_ASPECT_V4_LABEL_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V4_LABEL_REPORT.md):
  fresh v4 label/source run over Dream and LLaDA rows; LLaDA selected repair
  covers `24/24` eligible tasks with mean task score `0.331313`.
- [LATENT_AGGREGATION_MULTI_ASPECT_V4_PROBE_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V4_PROBE_REPORT.md):
  fresh v4 counterfactual probe source run with `24` measured probe generations.
- [LATENT_AGGREGATION_MULTI_ASPECT_V4_DIVERSITY_EXTENSION_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V4_DIVERSITY_EXTENSION_REPORT.md):
  fresh predeclared v4 LLaDA evolved/revision diversity-extension source run,
  with evolved mean task score `0.301789`.
- [LATENT_AGGREGATION_MULTI_ASPECT_V4_REPLAY.md](LATENT_AGGREGATION_MULTI_ASPECT_V4_REPLAY.md):
  passing fresh predeclared v4 replay over label, probe, and diversity sources:
  `14/24` complement coverage, `14` local promotions, all frozen gates passed.
- [LATENT_AGGREGATION_MULTI_ASPECT_V4_COVERAGE_GAP.md](LATENT_AGGREGATION_MULTI_ASPECT_V4_COVERAGE_GAP.md):
  v4 coverage diagnostic showing the remaining `10` no-complement tasks are
  anchor-dominance cases under the current aspect ontology.
- [LATENT_AGGREGATION_MULTI_ASPECT_V5_FREEZE.md](LATENT_AGGREGATION_MULTI_ASPECT_V5_FREEZE.md):
  frozen 48-task v5 replication contract over `plan_249`-`plan_296`; keeps the
  v4 source mix fixed and adds robustness gates for medians, leave-one-out lift,
  high-leverage tasks, source-family ablations, theme buckets, and
  cost-normalized lift before any v5 labels exist.
- [LATENT_AGGREGATION_MULTI_ASPECT_V5_LABEL_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V5_LABEL_REPORT.md):
  fresh v5 label/source run over the frozen 48-task slice; selected repair
  reaches `0.323549` task score on repair-covered tasks versus fixed
  `0.284007`.
- [LATENT_AGGREGATION_MULTI_ASPECT_V5_PROBE_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V5_PROBE_REPORT.md):
  fresh v5 probe source run with `48` measured counterfactual probes; useful as
  replay source evidence, not as an independently promoted repair result.
- [LATENT_AGGREGATION_MULTI_ASPECT_V5_DIVERSITY_EXTENSION_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V5_DIVERSITY_EXTENSION_REPORT.md):
  fresh v5 LLaDA evolved/revision diversity-extension source run; evolved task
  score is `0.309671` versus fixed `0.284007` and random `0.257126`.
- [LATENT_AGGREGATION_MULTI_ASPECT_V5_REPLAY.md](LATENT_AGGREGATION_MULTI_ASPECT_V5_REPLAY.md):
  passing fresh 48-task v5 replay over label, probe, and diversity sources:
  `34/48` complement coverage, `34/14/0` wins/ties/losses, mean realized score
  `0.402750` versus anchor `0.340964`, and all `24` frozen gates passed.
- [LATENT_AGGREGATION_MULTI_ASPECT_V5_COVERAGE_GAP.md](LATENT_AGGREGATION_MULTI_ASPECT_V5_COVERAGE_GAP.md):
  v5 coverage diagnostic showing the remaining `14` no-complement tasks:
  `13` anchor-dominance cases and `1` positive-below-threshold near miss.
- [LATENT_AGGREGATION_MULTI_ASPECT_V6_FREEZE.md](LATENT_AGGREGATION_MULTI_ASPECT_V6_FREEZE.md):
  active fresh `plan_297`-`plan_344` coverage-targeting contract; keeps the v5
  replay mechanism fixed and adds anchor-deficit constraint-gap rescue rows as
  a new source family, with stricter coverage and explicit incremental-cost
  reporting gates.
- [LATENT_AGGREGATION_MULTI_ASPECT_V6_LABEL_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V6_LABEL_REPORT.md):
  fresh v6 label/source run over the coverage-targeting 48-task slice; selected
  repair covers `48/48` eligible tasks with mean task score `0.306533` versus
  fixed `0.263757` and random `0.218954` on repair-covered tasks.
- [LATENT_AGGREGATION_MULTI_ASPECT_V6_PROBE_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V6_PROBE_REPORT.md):
  fresh v6 counterfactual probe source run with `288` raw source rows over the
  coverage-targeting slice; useful as replay source evidence, not as an
  independently promoted repair result.
- [LATENT_AGGREGATION_MULTI_ASPECT_V6_DIVERSITY_EXTENSION_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V6_DIVERSITY_EXTENSION_REPORT.md):
  fresh v6 LLaDA evolved/revision diversity-extension source run; evolved task
  score is `0.274015` versus fixed `0.263757` and random `0.218954`.
- [LATENT_AGGREGATION_MULTI_ASPECT_V6_ANCHOR_DEFICIT_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V6_ANCHOR_DEFICIT_REPORT.md):
  fresh v6 targeted anchor-deficit source run; selected repair covers `48/48`
  tasks and scores `0.278735` versus fixed `0.263757` and random `0.218954`.
- [LATENT_AGGREGATION_MULTI_ASPECT_V6_REPLAY.md](LATENT_AGGREGATION_MULTI_ASPECT_V6_REPLAY.md):
  failed fresh v6 replay over label, probe, diversity, and anchor-deficit
  sources; coverage is `27/48` against the `36/48` gate, while all-task
  non-rubric lift remains positive at `0.043118` with `0` unsupported additions
  and `0` hard contradictions.
- [LATENT_AGGREGATION_MULTI_ASPECT_V6_COVERAGE_GAP.md](LATENT_AGGREGATION_MULTI_ASPECT_V6_COVERAGE_GAP.md):
  v6 coverage diagnostic showing the remaining `21` no-complement tasks:
  `19` anchor-dominance cases and `2` positive-but-below-threshold cases.
- [LATENT_AGGREGATION_MULTI_ASPECT_V6_THRESHOLD_SENSITIVITY.md](LATENT_AGGREGATION_MULTI_ASPECT_V6_THRESHOLD_SENSITIVITY.md):
  no-generation diagnostic showing the v6 failure is not primarily caused by
  the frozen dimension threshold; positive-floor coverage is only `29/48`
  against the `36/48` gate.
- [LATENT_AGGREGATION_MULTI_ASPECT_V7_PREFREEZE.md](LATENT_AGGREGATION_MULTI_ASPECT_V7_PREFREEZE.md):
  pre-freeze v7 design derived from the v6 failure; requires fresh
  `plan_345`-`plan_392` tasks, expanded aspects, and new source families before
  a real promotion freeze can be built.
- [LATENT_AGGREGATION_MULTI_ASPECT_V7_FREEZE.md](LATENT_AGGREGATION_MULTI_ASPECT_V7_FREEZE.md):
  v7 task and ontology freeze over fresh `plan_345`-`plan_392`. Expanded-aspect
  replay support, v7 source-family mapping, and label-leakage gates are now
  implemented in the shared replay runner. The baseline label, ontology-probe,
  and cross-latent source families are populated; the frozen multi-source v7
  replay is complete and failed its promotion gates.
- [LATENT_AGGREGATION_MULTI_ASPECT_V7_LABEL_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V7_LABEL_REPORT.md):
  fresh v7 baseline label/source run over the frozen 48-task slice; selected
  latent repair covers `48/48` eligible tasks with mean task score `0.327170`
  versus fixed `0.270320` and random `0.246198` on repair-covered tasks.
- [LATENT_AGGREGATION_MULTI_ASPECT_V7_ONTOLOGY_PROBE_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V7_ONTOLOGY_PROBE_REPORT.md):
  fresh v7 ontology-probe source run over the frozen 48-task slice; generated
  `48` counterfactual probe rows with `span_tomography_probe_v4`, plus the
  shared baseline rows needed for source-family replay.
- [LATENT_AGGREGATION_MULTI_ASPECT_V7_CROSS_LATENT_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V7_CROSS_LATENT_REPORT.md):
  fresh v7 cross-latent source run over the frozen 48-task slice; generated
  `336` full model generations and `192` arm selections. Evolved cross-latent
  selection scores `0.288632` versus `0.270320` fixed and `0.246198` random,
  with oracle task score `0.306` and remaining selector regret of `0.017`.
- [LATENT_AGGREGATION_MULTI_ASPECT_V7_REPLAY.md](LATENT_AGGREGATION_MULTI_ASPECT_V7_REPLAY.md):
  failed fresh v7 expanded-ontology replay over label, ontology-probe, and
  cross-latent sources. Coverage is `24/48` against the `36/48` gate, online
  promotions are `23`, wins/ties/losses are `24/24/0`, and Wilson lower bound
  is `0.344713` against `0.600000`; safety remains clean with `0` unsupported
  additions, `0` hard contradictions, and a passing label-leakage check.
- [LATENT_AGGREGATION_MULTI_ASPECT_V7_FAILURE.md](LATENT_AGGREGATION_MULTI_ASPECT_V7_FAILURE.md):
  no-generation diagnostic over the failed v7 replay. It converts the failure
  into an executable next-source target: at current `n=48`, a new source family
  needs at least `13` newly promoted covered tasks to clear the Wilson gate, so
  the next run should target the `24` uncovered tasks directly.
- [LATENT_AGGREGATION_MULTI_ASPECT_V8_TARGETED_SOURCE_CONTRACT.md](LATENT_AGGREGATION_MULTI_ASPECT_V8_TARGETED_SOURCE_CONTRACT.md):
  predeclared source-generation contract for the next aggregation experiment.
  It freezes the `24` v7 no-complement tasks and emits a
  `targeted_history_contrast` GPU command; this is not a promotion artifact
  until the generated rows are replayed against the frozen v7 evidence.
- [LATENT_AGGREGATION_MULTI_ASPECT_V8_TARGETED_HISTORY_CONTRAST_REPORT.md](LATENT_AGGREGATION_MULTI_ASPECT_V8_TARGETED_HISTORY_CONTRAST_REPORT.md):
  targeted GPU source run over the `24` v7 no-complement tasks. The repair arm
  covers `24/24` tasks and scores `0.292167` versus `0.275045` fixed and
  `0.281318` evolved; useful local repair evidence, but not a promotion result.
- [LATENT_AGGREGATION_MULTI_ASPECT_V8_TARGETED_HISTORY_CONTRAST_REPLAY.md](LATENT_AGGREGATION_MULTI_ASPECT_V8_TARGETED_HISTORY_CONTRAST_REPLAY.md):
  negative post-failure replay adding the targeted history-contrast rows to the
  frozen v7 source mix. Coverage remains `24/48`, online promotions remain
  `23`, and source-family ablation shows `targeted_history_contrast` contributes
  no selected complements under the expanded extractor.
- [LATENT_AGGREGATION_MULTI_ASPECT_V8_TARGETED_SOURCE_GAP.md](LATENT_AGGREGATION_MULTI_ASPECT_V8_TARGETED_SOURCE_GAP.md):
  no-generation diagnostic explaining the targeted-source miss. Across the `24`
  targeted repairs, mean delta versus the original v7 anchor is `-0.049479`,
  only `1/24` beats the original anchor, only `1/24` adds an expanded complement
  against that anchor, and `23/24` are not-stronger/no-new-expanded-aspect cases.
- [LATENT_AGGREGATION_MULTI_ASPECT_V6_EXPANDED_ONTOLOGY_BACKTEST.md](LATENT_AGGREGATION_MULTI_ASPECT_V6_EXPANDED_ONTOLOGY_BACKTEST.md):
  post-hoc no-generation diagnostic over existing v6 raw rows; the expanded
  ontology recovers `12` of `21` v6 no-complement tasks, motivating v7
  implementation while requiring false-positive audits on fresh tasks.
- [LATENT_AGGREGATION_INFERENCE_V1_LABEL_REPORT.md](LATENT_AGGREGATION_INFERENCE_V1_LABEL_REPORT.md):
  generated label report for the frozen Dream/LLaDA trajectory run that feeds
  the inference aggregation replay.
- [DIFFUSION_REPAIR_VALUE_TOMOGRAPHY.md](DIFFUSION_REPAIR_VALUE_TOMOGRAPHY.md):
  behavior-tomography audit for the cost-aware repair-spend controller.
- [DIFFUSION_LAMBDA_REPAIR_CONTROLLER_TRANSFER.md](DIFFUSION_LAMBDA_REPAIR_CONTROLLER_TRANSFER.md):
  transfer audit showing where the lambda-aware repair controller does and does not generalize.
- [DIFFUSION_LAMBDA_REPAIR_ACTIVE_TARGETS.json](DIFFUSION_LAMBDA_REPAIR_ACTIVE_TARGETS.json):
  runner-ready task-id manifest for the next focused active-data collection.

Generated report builders may still emit root-level markdown unless their
defaults have been migrated. Treat new root-level generated reports as temporary
audit output and move durable reports back under this archive or a specific
reader-facing `docs/` page.
