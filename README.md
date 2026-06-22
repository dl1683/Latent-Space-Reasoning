# Latent Space Reasoning

This repo studies inference-time latent reasoning control: changing a frozen
model's reasoning trajectory without fine-tuning it.

The current promoted result is diffusion-native latent repair. The active
research question is whether editable latent or denoise trajectories can provide
reliable reasoning gains while preserving a clean evidence trail.

## Current Result

On the lean GPU mixed benchmark, LLaDA-MoE latent repair beats both greedy/fixed
denoise and random perturbation:

| Public arm | Score | Relative GPU cost |
|------------|------:|------------------:|
| Greedy/fixed denoise | 0.412277 | 1.000000x |
| Random perturbation | 0.372125 | 1.000000x |
| Latent repair | 0.531116 | 2.625000x |

There is also a lower-cost decomposed-selector point at `0.508705` and
`2.375000x`. Use the top-score point for the headline reasoning-lift claim and
the lower-cost point for the controller/cost claim.

## Read First

| Path | Purpose |
|------|---------|
| [DIFFUSION_PUBLIC_BENCHMARK.md](DIFFUSION_PUBLIC_BENCHMARK.md) | Public benchmark table and promoted score/cost claims |
| [CLAIM_EVIDENCE_MAP.md](CLAIM_EVIDENCE_MAP.md) | Claim-to-artifact ledger for promoted evidence |
| [DIFFUSION_GROUND_TRUTH_INDEX.md](DIFFUSION_GROUND_TRUTH_INDEX.md) | Canonical raw/report/score artifact pointers and hashes |
| [docs/DIFFUSION_READER_GUIDE.md](docs/DIFFUSION_READER_GUIDE.md) | Reader map for diffusion claims, theory, and validation surfaces |
| [docs/DIFFUSION_THEORY_CLAIM_LEDGER.md](docs/DIFFUSION_THEORY_CLAIM_LEDGER.md) | Conservative theory ledger with falsifiers and next proof obligations |
| [meditations/README.md](meditations/README.md) | Private question-first notes used to maintain the paradigm layer of the project |
| [docs/README.md](docs/README.md) | Single-page documentation index for onboarding and navigation |
| [docs/GATED_ATTENTION_PROBE.md](docs/GATED_ATTENTION_PROBE.md) | Current Qwen3-Next gated-attention probe status and blockers |
| [docs/reports/diffusion/README.md](docs/reports/diffusion/README.md) | Historical/generated diffusion report archive |
| [docs/reports/diffusion/DIFFUSION_REPAIR_VALUE_TOMOGRAPHY.md](docs/reports/diffusion/DIFFUSION_REPAIR_VALUE_TOMOGRAPHY.md) | Behavior-tomography audit for the next repair-spend controller |
| [experiments/EXPERIMENTS.md](experiments/EXPERIMENTS.md) | Chronological experiment log |

Generated one-off reports and raw artifacts are retained for auditability, but
they are not the front-page navigation path. Start from the files above unless
you are reproducing a specific historical run.

## Repository Layout

```text
experiments/     Experiment runners, report builders, validators, and analysis scripts
eval_results/    Generated run outputs, ledgers, raw generations, and score reports (typically git-ignored)
docs/            Reader guides, theory docs, runbooks, and consolidated status pages
tests/           Unit and regression tests for runners, builders, validators, and controls
src/             Shared package code used by the experiment stack
```

## Reproduce The Public Diffusion Evidence

```bash
python experiments/build_diffusion_claim_evidence.py
python experiments/validate_diffusion_claim_evidence.py
python experiments/validate_diffusion_theory_claim_ledger.py
```

The promoted public result is intentionally narrow: greedy/fixed denoise,
random perturbation, and selected latent repair, with relative GPU cost reported
beside score.

## Current Gated-Attention Status

Current blocker/runner status for Qwen3-Next is maintained in
[docs/GATED_ATTENTION_PROBE.md](docs/GATED_ATTENTION_PROBE.md). Use that file for
the latest environment and dependency notes.

## Development

```bash
python -m pytest tests -q
python -m compileall experiments src tests
```

For focused gated-attention validation:

```bash
python -m pytest tests/test_build_gated_attention_null_probe_freeze.py tests/test_build_gated_attention_probe_execution_plan.py tests/test_build_gated_attention_artifact_decision.py tests/test_build_gated_attention_wsl_bootstrap.py tests/test_run_latent_sensitivity_position_shift.py -q
```
