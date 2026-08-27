# Diffusion Language Smoke Report

Records: `2`

| Run | Candidate | Schedule | Score | Steps | Temp | History | First visible | First final | Mask-free | Final chars | Text |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | llada-8b-instruct-hf | low_confidence_32 | 0.688 | 32 | 0.0 | 32 | 7 | 32 | 32 | 184 | Collect the baseline measurement. Even if the intervention fails, the baseline measurement wi... |
| 2 | llada-8b-instruct-hf | random_32 | 0.685 | 32 | 0.0 | 32 | 1 | 32 | 32 | 172 | Collect the measurements from the baseline job before running the risky reasoning interventio... |

## Latest Trajectory Samples

- step `1`: masks `31`, eos `0`, visible chars `3`
- step `7`: masks `25`, eos `0`, visible chars `40`
- step `13`: masks `19`, eos `0`, visible chars `71`
- step `20`: masks `12`, eos `0`, visible chars `104`
- step `26`: masks `6`, eos `0`, visible chars `151`
- step `32`: masks `0`, eos `1`, visible chars `172`
