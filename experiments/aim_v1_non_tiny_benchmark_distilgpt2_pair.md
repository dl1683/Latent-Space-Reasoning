# AIM-v1 Adaptive Survivor Benchmark

- Generated: `2026-02-16T05:38:00.238225+00:00`
- Model: `distilgpt2`
- Queries: `2`
- Repeats: `1`
- Warmup per trial: `True`

> Interpretation note: Latent scorer values below are internal evolutionary-guidance telemetry only; they are not direct output-quality metrics.

## Fixed vs Adaptive

| Metric | Fixed | Adaptive |
|---|---:|---:|
| Avg latent scorer output (guidance only) | 0.7614727020263672 | 0.7614727020263672 |
| Median trial avg scorer output (guidance only) | 0.7614727020263672 | 0.7614727020263672 |
| Avg evaluations | 7.0 | 5.5 |
| Median trial avg evaluations | 7.0 | 5.5 |
| Avg latent duration (s) | 0.5103366000112146 | 0.5169574999890756 |
| Median trial avg latent duration (s) | 0.5103366000112146 | 0.5169574999890756 |
| Avg evolution duration (s) | 0.00700209999922663 | 0.0059905999805778265 |
| Median trial avg evolution duration (s) | 0.00700209999922663 | 0.0059905999805778265 |
| Avg evolution time/eval (s) | 0.0010002999998895185 | 0.0010891999964686956 |
| Avg evaluations/quality | 7.923915641761859 | 6.234291080751501 |

## Deltas

- Quality delta (adaptive - fixed): `0.0`
- Evaluation reduction ratio: `0.21428571428571427`
- Latency reduction ratio: `-0.012973594246847039`
- Evolution latency reduction ratio: `0.1444566656803704`
- Evolution time/eval reduction ratio: `-0.08887333458861946`
