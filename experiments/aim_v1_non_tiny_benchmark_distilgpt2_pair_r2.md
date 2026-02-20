# AIM-v1 Adaptive Survivor Benchmark

- Generated: `2026-02-16T05:40:13.114800+00:00`
- Model: `distilgpt2`
- Queries: `2`
- Repeats: `2`
- Warmup per trial: `True`

> Interpretation note: Latent scorer values below are internal evolutionary-guidance telemetry only; they are not direct output-quality metrics.

## Fixed vs Adaptive

| Metric | Fixed | Adaptive |
|---|---:|---:|
| Avg latent scorer output (guidance only) | 0.7406120002269745 | 0.7406120002269745 |
| Median trial avg scorer output (guidance only) | 0.7406120002269745 | 0.7406120002269745 |
| Avg evaluations | 6.5 | 5.5 |
| Median trial avg evaluations | 6.5 | 5.5 |
| Avg latent duration (s) | 0.505054499997641 | 0.46792335000645835 |
| Median trial avg latent duration (s) | 0.505054499997641 | 0.46792335000645835 |
| Avg evolution duration (s) | 0.006841575013822876 | 0.006245949989533983 |
| Median trial avg evolution duration (s) | 0.006841575013822876 | 0.006245949989533983 |
| Avg evolution time/eval (s) | 0.0010525500021265964 | 0.0011356272708243605 |
| Avg evaluations/quality | 7.462550442418369 | 6.3428985291100695 |

## Deltas

- Quality delta (adaptive - fixed): `0.0`
- Evaluation reduction ratio: `0.15384615384615385`
- Latency reduction ratio: `0.07351909544683999`
- Evolution latency reduction ratio: `0.08705963511113717`
- Evolution time/eval reduction ratio: `-0.08255746614068012`
