# Latent Aggregation Multi-Aspect V3 Probe Dry-Run Failure

## Verdict

The first v3 probe measurement run is a failed dry run. It completed on GPU, wrote base trajectory outputs, and produced a runner report, but it did not generate any counterfactual probe records.

## Failed Command

```powershell
python experiments\run_diffusion_three_arm_benchmark.py --task-ids plan_201,plan_202,plan_203,plan_204,plan_205,plan_206,plan_207,plan_208,plan_209,plan_210,plan_211,plan_212,plan_213,plan_214,plan_215,plan_216,plan_217,plan_218,plan_219,plan_220,plan_221,plan_222,plan_223,plan_224 --candidates dream-7b-instruct-hf,llada-8b-instruct-hf --limit-schedules 3 --limit-evolved-schedules 0 --limit-repair-candidates 0 --repair-spend-trigger counterfactual_micro_probe_v1 --counterfactual-probe-mode all --counterfactual-probe-policy span_tomography_probe_v4 --trajectory-selector planning_state --device cuda --dtype bfloat16 --raw-output eval_results\diffusion_language\latent_aggregation_multi_aspect_v3_probe_raw.jsonl --scores-output eval_results\diffusion_language\latent_aggregation_multi_aspect_v3_probe_scores.json --report-output docs\reports\diffusion\LATENT_AGGREGATION_MULTI_ASPECT_V3_PROBE_REPORT.md
```

## Observed Failure

- Runner output: `Counterfactual probe generations: 0`
- Raw output: `eval_results\diffusion_language\latent_aggregation_multi_aspect_v3_probe_dry_run_failure_raw.jsonl`
- Scores output: `eval_results\diffusion_language\latent_aggregation_multi_aspect_v3_probe_dry_run_failure_scores.json`
- Runner report: `docs\reports\diffusion\LATENT_AGGREGATION_MULTI_ASPECT_V3_PROBE_DRY_RUN_FAILURE_RUNNER_REPORT.md`

## Cause

The command set `--limit-repair-candidates 0`. The runner only enters the repair/probe gate when `_should_run_repairs(...)` is true, so a zero repair-candidate limit skipped the counterfactual probe section entirely. The counterfactual trigger would have kept `should_run=false`, but the probe source path still needed a nonzero repair-candidate limit to be reached.

## Improvement

The v3 freeze command now uses `--limit-repair-candidates 1` with the same diagnostic-only probe trigger:

```powershell
--limit-repair-candidates 1 --repair-spend-trigger counterfactual_micro_probe_v1 --counterfactual-probe-mode all --counterfactual-probe-policy span_tomography_probe_v4
```

The next run must be treated as the first valid v3 probe measurement only if `counterfactual_probe_generation_count > 0`.
