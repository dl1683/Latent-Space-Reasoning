# Keyword-Stuffing Audit

Verdict: **RED**

## Summary

- Gaming ratio: `4.6974`
- Keyword mean lift: `0.471380`
- Realized mean lift: `0.100350`
- Keyword promotions: `96/96`
- Realized promotions: `87/96`
- Keyword promotion ratio: `1.1034`
- Keyword beats realized: `96/96`

## Thresholds

| Level | Gaming Ratio | Keyword Promotion Ratio |
| --- | --- | --- |
| Green | <= 0.25 | <= 10% of realized |
| Yellow | <= 0.50 | <= 25% of realized |
| Red | above | rubric gameable |

## Interpretation

**WARNING**: The planning rubric is gameable by keyword stuffing. Keyword bags reproduce a significant fraction of the aggregation lift. Automatic planning scores cannot be the primary endpoint. Switch to blinded pairwise review of decoded outputs.

## Per-Task Results

| Task | Anchor | Realized | Keyword | KW > Anchor | KW >= Realized | Decision |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `plan_441` | 0.4914 | 0.5539 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_442` | 0.4129 | 0.6132 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_443` | 0.3129 | 0.4968 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_444` | 0.3814 | 0.4454 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_445` | 0.4289 | 0.4929 | 0.9000 | yes | yes | `online_promoted_local` |
| `plan_446` | 0.4089 | 0.5454 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_447` | 0.4989 | 0.5629 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_448` | 0.4129 | 0.5932 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_449` | 0.3629 | 0.4882 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_450` | 0.4507 | 0.5146 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_451` | 0.4343 | 0.4968 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_452` | 0.5539 | 0.6179 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_453` | 0.3014 | 0.3654 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_454` | 0.4025 | 0.5664 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_455` | 0.4329 | 0.4329 | 0.8500 | yes | yes | `blocked_no_complement_material` |
| `plan_456` | 0.5314 | 0.5954 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_457` | 0.3364 | 0.3989 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_458` | 0.5739 | 0.6364 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_459` | 0.3829 | 0.4668 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_460` | 0.3429 | 0.4454 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_461` | 0.3114 | 0.4314 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_462` | 0.3343 | 0.4582 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_463` | 0.4032 | 0.5257 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_464` | 0.3871 | 0.4296 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_465` | 0.4289 | 0.4914 | 0.9500 | yes | yes | `online_promoted_local` |
| `plan_466` | 0.4629 | 0.5468 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_467` | 0.3943 | 0.5818 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_468` | 0.3039 | 0.4254 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_469` | 0.3818 | 0.4243 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_470` | 0.2800 | 0.3439 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_471` | 0.3014 | 0.3654 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_472` | 0.3964 | 0.4589 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_473` | 0.3979 | 0.5043 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_474` | 0.3775 | 0.5239 | 0.9500 | yes | yes | `online_promoted_local` |
| `plan_475` | 0.3550 | 0.4189 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_476` | 0.3757 | 0.5346 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_477` | 0.3739 | 0.5554 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_478` | 0.4829 | 0.6682 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_479` | 0.4189 | 0.4829 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_480` | 0.3604 | 0.4029 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_481` | 0.4089 | 0.6718 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_482` | 0.3743 | 0.4582 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_483` | 0.3764 | 0.4189 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_484` | 0.4314 | 0.5704 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_485` | 0.3529 | 0.5368 | 1.0000 | yes | yes | `online_promoted_local` |
| `plan_486` | 0.3764 | 0.4404 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_487` | 0.3700 | 0.5039 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_488` | 0.3729 | 0.6432 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_489` | 0.3789 | 0.5218 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_490` | 0.3100 | 0.4575 | 0.9500 | yes | yes | `online_promoted_local` |
| `plan_491` | 0.3354 | 0.3354 | 0.8500 | yes | yes | `blocked_no_complement_material` |
| `plan_492` | 0.3139 | 0.3779 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_493` | 0.3729 | 0.3729 | 0.8500 | yes | yes | `blocked_no_complement_material` |
| `plan_494` | 0.4343 | 0.5168 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_495` | 0.4939 | 0.6329 | 0.9000 | yes | yes | `online_promoted_local` |
| `plan_496` | 0.4239 | 0.5089 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_497` | 0.3514 | 0.4529 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_498` | 0.3889 | 0.3889 | 0.8500 | yes | yes | `blocked_no_complement_material` |
| `plan_499` | 0.2900 | 0.4475 | 0.9500 | yes | yes | `online_promoted_local` |
| `plan_500` | 0.6043 | 0.6043 | 0.8500 | yes | yes | `blocked_no_complement_material` |
| `plan_501` | 0.3618 | 0.4243 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_502` | 0.2625 | 0.4400 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_503` | 0.4189 | 0.4829 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_504` | 0.3389 | 0.4029 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_505` | 0.4189 | 0.5204 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_506` | 0.3329 | 0.4368 | 0.9000 | yes | yes | `online_promoted_local` |
| `plan_507` | 0.3829 | 0.5793 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_508` | 0.3943 | 0.6343 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_509` | 0.3929 | 0.5168 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_510` | 0.4075 | 0.5464 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_511` | 0.3764 | 0.4189 | 0.9000 | yes | yes | `online_promoted_local` |
| `plan_512` | 0.3114 | 0.3939 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_513` | 0.4529 | 0.5918 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_514` | 0.3600 | 0.4654 | 0.9500 | yes | yes | `online_promoted_local` |
| `plan_515` | 0.4329 | 0.5918 | 0.9500 | yes | yes | `online_promoted_local` |
| `plan_516` | 0.4343 | 0.6021 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_517` | 0.3443 | 0.3868 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_518` | 0.3764 | 0.4189 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_519` | 0.3175 | 0.3814 | 0.9500 | yes | yes | `online_promoted_local` |
| `plan_520` | 0.4529 | 0.4529 | 0.8500 | yes | yes | `blocked_no_complement_material` |
| `plan_521` | 0.3779 | 0.4618 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_522` | 0.3629 | 0.4454 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_523` | 0.3043 | 0.3868 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_524` | 0.5239 | 0.6079 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_525` | 0.3729 | 0.3729 | 0.8500 | yes | yes | `blocked_no_complement_material` |
| `plan_526` | 0.3939 | 0.3939 | 0.8500 | yes | yes | `blocked_no_complement_material` |
| `plan_527` | 0.2714 | 0.4589 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_528` | 0.3504 | 0.5093 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_529` | 0.4568 | 0.5207 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_530` | 0.4743 | 0.4743 | 0.8500 | yes | yes | `blocked_no_complement_material` |
| `plan_531` | 0.3989 | 0.4629 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_532` | 0.3514 | 0.4529 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_533` | 0.3314 | 0.4579 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_534` | 0.3764 | 0.4404 | 0.8500 | yes | yes | `online_promoted_local` |
| `plan_535` | 0.4329 | 0.7257 | 0.9000 | yes | yes | `online_promoted_local` |
| `plan_536` | 0.3543 | 0.4582 | 0.8500 | yes | yes | `online_promoted_local` |
