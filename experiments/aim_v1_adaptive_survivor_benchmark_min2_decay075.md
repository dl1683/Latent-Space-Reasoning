# AIM-v1 Adaptive Survivor Benchmark

- Generated: `2026-02-16T04:43:32.664475+00:00`
- Model: `hf-internal-testing/tiny-random-gpt2`
- Queries: `6`
- Repeats: `3`
- Warmup per trial: `True`

> Interpretation note: Latent scorer values below are internal evolutionary-guidance telemetry only; they are not direct output-quality metrics.

## Fixed vs Adaptive

| Metric | Fixed | Adaptive |
|---|---:|---:|
| Avg latent scorer output (guidance only) | 0.7357447942097982 | 0.7355542249149747 |
| Median trial avg scorer output (guidance only) | 0.7490471402804056 | 0.7494913140932719 |
| Avg evaluations | 14.444444444444445 | 13.166666666666666 |
| Median trial avg evaluations | 13.5 | 12.166666666666666 |
| Avg latent duration (s) | 0.2244365666701924 | 0.23124630001257174 |
| Median trial avg latent duration (s) | 0.2198328500186714 | 0.23061311667940268 |
| Avg evolution duration (s) | 0.01681055556077303 | 0.015126872232132074 |
| Median trial avg evolution duration (s) | 0.016793433353692915 | 0.014087100008813044 |
| Avg evolution time/eval (s) | 0.0011638076926689023 | 0.0011488763720606638 |
| Avg evaluations/quality | 16.795172946296507 | 15.302861372500518 |

## Deltas

- Quality delta (adaptive - fixed): `0.00044417381286621094`
- Evaluation reduction ratio: `0.09876543209876548`
- Latency reduction ratio: `-0.049038470182302916`
- Evolution latency reduction ratio: `0.16115426118535447`
- Evolution time/eval reduction ratio: `-0.03240190737202579`
