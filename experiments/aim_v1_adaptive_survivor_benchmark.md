# AIM-v1 Adaptive Survivor Benchmark

- Generated: `2026-02-16T04:44:41.520865+00:00`
- Model: `hf-internal-testing/tiny-random-gpt2`
- Queries: `6`
- Repeats: `3`
- Warmup per trial: `True`

> Interpretation note: Latent scorer values below are internal evolutionary-guidance telemetry only; they are not direct output-quality metrics.

## Fixed vs Adaptive

| Metric | Fixed | Adaptive |
|---|---:|---:|
| Avg latent scorer output (guidance only) | 0.7357447942097982 | 0.7357933123906454 |
| Median trial avg scorer output (guidance only) | 0.7490471402804056 | 0.7497121493021647 |
| Avg evaluations | 14.61111111111111 | 12.333333333333334 |
| Median trial avg evaluations | 14.0 | 11.833333333333334 |
| Avg latent duration (s) | 0.23746366110410438 | 0.2326372888928745 |
| Median trial avg latent duration (s) | 0.23759838332383273 | 0.23383681666261205 |
| Avg evolution duration (s) | 0.018243683317753796 | 0.014932055559863025 |
| Median trial avg evolution duration (s) | 0.016547299999122817 | 0.014244200002091626 |
| Avg evolution time/eval (s) | 0.0012486171091998797 | 0.0012107072075564615 |
| Avg evaluations/quality | 16.98099225479753 | 14.331284444626121 |

## Deltas

- Quality delta (adaptive - fixed): `0.0006650090217590332`
- Evaluation reduction ratio: `0.1547619047619047`
- Latency reduction ratio: `0.015831617238295292`
- Evolution latency reduction ratio: `0.13918282723787448`
- Evolution time/eval reduction ratio: `0.04218934298298723`
