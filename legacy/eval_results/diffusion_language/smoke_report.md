# Diffusion Language Smoke Report

Records: `6`

| Run | Candidate | Schedule | Score | Steps | Temp | History | First visible | First final | Mask-free | Final chars | Text |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dream-7b-instruct-hf | - | - | 64 | 0.2 | 64 | - | - | - |  | Run the baseline job first and collect its measurements. Then, run the risky intervention job... |
| 2 | dream-7b-instruct-hf | - | - | 64 | 0.2 | 64 | 14 | 39 | 64 | 198 | Run the baseline job first and collect its measurements. Then, run the risky intervention job... |
| 3 | dream-7b-instruct-hf | - | - | 64 | 0.2 | 64 | 1 | 64 | 64 | 108 | Collect measurements from the baseline job as the first result and run the second job with th... |
| 4 | llada-8b-instruct-hf | - | - | 32 | 0.0 | None | - | - | - |  | Collect the baseline measurement. Even if the intervention fails, the baseline measurement wi... |
| 5 | llada-8b-instruct-hf | - | - | 32 | 0.0 | 32 | 1 | - | 1 | 184 | Collect the baseline measurement. Even if the intervention fails, the baseline measurement wi... |
| 6 | llada-8b-instruct-hf | - | - | 32 | 0.0 | 32 | 7 | - | 32 | 184 | Collect the baseline measurement. Even if the intervention fails, the baseline measurement wi... |

## Latest Trajectory Samples

- step `1`: masks `31`, eos `1`, visible chars `0`
- step `7`: masks `25`, eos `1`, visible chars `43`
- step `13`: masks `19`, eos `1`, visible chars `70`
- step `20`: masks `12`, eos `1`, visible chars `120`
- step `26`: masks `6`, eos `1`, visible chars `153`
- step `32`: masks `0`, eos `1`, visible chars `194`
