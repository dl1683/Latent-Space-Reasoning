# Diffusion Language Smoke Report

Records: `2`

| Run | Candidate | Schedule | Score | Steps | Temp | History | First visible | First final | Mask-free | Final chars | Text |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | llada-8b-instruct-hf | low_confidence_32 | 0.680 | 32 | 0.0 | 32 | 7 | - | 32 | 184 | Collect the baseline measurement. Even if the intervention fails, the baseline measurement wi... |
| 2 | llada-8b-instruct-hf | random_32 | 0.407 | 32 | 0.0 | 32 | 1 | - | 32 | 113 | You should collect the baseline first. If the intervention fails, you can still publish the b... |

## Latest Trajectory Samples

- step `1`: masks `31`, eos `0`, visible chars `3`
- step `7`: masks `25`, eos `2`, visible chars `27`
- step `13`: masks `19`, eos `5`, visible chars `36`
- step `20`: masks `12`, eos `8`, visible chars `58`
- step `26`: masks `6`, eos `9`, visible chars `93`
- step `32`: masks `0`, eos `11`, visible chars `123`
