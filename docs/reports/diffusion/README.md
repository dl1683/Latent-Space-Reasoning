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
