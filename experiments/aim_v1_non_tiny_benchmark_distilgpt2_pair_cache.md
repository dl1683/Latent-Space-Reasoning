# AIM-v1 Adaptive Survivor Benchmark

- Generated: `2026-02-16T05:39:18.200539+00:00`
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
| Avg latent duration (s) | 0.49353305003023706 | 0.5084191999631003 |
| Median trial avg latent duration (s) | 0.49353305003023706 | 0.5084191999631003 |
| Avg evolution duration (s) | 0.007992850005393848 | 0.006248699995921925 |
| Median trial avg evolution duration (s) | 0.007992850005393848 | 0.006248699995921925 |
| Avg evolution time/eval (s) | 0.0011418357150562639 | 0.0011361272719858046 |
| Avg evaluations/quality | 7.923915641761859 | 6.234291080751501 |

## Deltas

- Quality delta (adaptive - fixed): `0.0`
- Evaluation reduction ratio: `0.21428571428571427`
- Latency reduction ratio: `-0.030162417556334208`
- Evolution latency reduction ratio: `0.21821377960238353`
- Evolution time/eval reduction ratio: `0.004999355857578904`
