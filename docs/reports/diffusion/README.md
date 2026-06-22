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
