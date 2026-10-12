# Correctness hunt: Q1-prime Amendment A implementation (READ-ONLY; do not edit any file)

Read in detail, citing file:line: experiments/cloud/lsr_jobs/q1pp.py (all), the helpers it imports from
experiments/cloud/lsr_jobs/native.py (block, changed, collection, value_positions, states_at_pos, Hook, run_prompts),
experiments/cloud/lsr_jobs/common.py load(), experiments/cloud/configs/q1pp_8b_smoke.json, and the registration
"Q1-prime Amendment A" at the end of theory/LATENT_BRANCH_CAPACITY.md.
Hunt for: bugs that would make the smoke or sweep wrong (position alignment at the post/closing site, pair labels,
pair-mean leakage between fit and test, norm matching, bootstrap/Bonferroni/r90 logic, gate logic, claim rule
mismatch with the registration, resume behaviour, multi-GPU device placement for 72B, collateral definition),
and any mismatch between code and registration text. A smoke is already running on this code; say for each issue
whether it invalidates the smoke results or only needs fixing before the sweep. Under 400 words, file:line.
