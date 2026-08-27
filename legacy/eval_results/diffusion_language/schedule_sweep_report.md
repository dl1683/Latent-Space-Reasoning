# Diffusion Language Smoke Report

Records: `3`

| Run | Candidate | Schedule | Score | Steps | Temp | History | First visible | First final | Mask-free | Final chars | Text |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dream-7b-instruct-hf | entropy_32 | 0.561 | 32 | 0.2 | 32 | 1 | 13 | 32 | 117 | Run the baseline job first and collect its measurements. Then, run the intervention job and c... |
| 2 | dream-7b-instruct-hf | entropy_64 | 0.727 | 64 | 0.2 | 64 | 14 | 39 | 64 | 198 | Run the baseline job first and collect its measurements. Then, run the risky intervention job... |
| 3 | dream-7b-instruct-hf | origin_64 | 0.121 | 64 | 0.2 | 64 | 14 | 64 | 64 | 26 | I recommend the following, |

## Latest Trajectory Samples

- step `1`: masks `64`, eos `0`, visible chars `0`
- step `14`: masks `51`, eos `11`, visible chars `2`
- step `26`: masks `46`, eos `16`, visible chars `2`
- step `39`: masks `34`, eos `28`, visible chars `2`
- step `51`: masks `15`, eos `46`, visible chars `6`
- step `64`: masks `0`, eos `59`, visible chars `26`
