# AIM-v1 Adaptive Survivor Benchmark

- Generated: `2026-02-16T04:46:57.045002+00:00`
- Model: `hf-internal-testing/tiny-random-gpt2`
- Queries: `6`
- Repeats: `3`
- Warmup per trial: `True`

> Interpretation note: Latent scorer values below are internal evolutionary-guidance telemetry only; they are not direct output-quality metrics.

## Fixed vs Adaptive

| Metric | Fixed | Adaptive |
|---|---:|---:|
| Avg latent scorer output (guidance only) | 0.7357447942097982 | 0.7355683909522163 |
| Median trial avg scorer output (guidance only) | 0.7490471402804056 | 0.7495338122049967 |
| Avg evaluations | 14.555555555555555 | 13.11111111111111 |
| Median trial avg evaluations | 14.0 | 12.0 |
| Avg latent duration (s) | 0.27618651667338173 | 0.3638578277702133 |
| Median trial avg latent duration (s) | 0.2720195333434579 | 0.27073711667132255 |
| Avg evolution duration (s) | 0.021352077782568004 | 0.022110955563322123 |
| Median trial avg evolution duration (s) | 0.020351083347729098 | 0.018242500023916364 |
| Avg evolution time/eval (s) | 0.0014669366415504736 | 0.0016864288141516875 |
| Avg evaluations/quality | 16.90940877542753 | 15.239113852952274 |

## Deltas

- Quality delta (adaptive - fixed): `0.00048667192459106445`
- Evaluation reduction ratio: `0.14285714285714285`
- Latency reduction ratio: `0.004714428616110255`
- Evolution latency reduction ratio: `0.10361037237106216`
- Evolution time/eval reduction ratio: `-0.04578789890042756`
