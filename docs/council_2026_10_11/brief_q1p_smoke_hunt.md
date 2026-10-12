# Fresh error hunt: Q1-prime 8B positive-control smoke (READ-ONLY; do not edit files)

Read in detail, citing file:line: experiments/cloud/lsr_jobs/native.py (FAM, COLORS, collection, changed, block,
value_positions, q1p_pairs, states_at_pos, Hook, run_prompts, q1p_cell, fit_basis_value, q1prime),
experiments/cloud/configs/q1p_8b_smoke.json, theory/LATENT_BRANCH_CAPACITY.md (Registration Q1-prime, Standing rule),
and experiments/results/cloud/q1p_8b_smoke_summary.jsonl.

Observation: all 6 cells (W1/W2 x k=1,4,16; n=64, dev names, Qwen3-8B layer 9) pass the gate: full donor-follow 1.0,
sham 0.0, collateral 0-0.008, receiver accuracy 1.0. proj retention: r=1 0.14-0.27, r=4 1.0 everywhere; frozen random
retention ~0 at all ranks; r90 = 4 in every cell; bf16 absorption ~0.17%.

Questions:
1. Hunt bugs and overclaims in the instrument and the summary.
2. Construction confound: there are 4 colour options. At the value token the donor-minus-receiver difference may be
   dominated by the token-identity difference of two colour words, whose centered span is at most 3 dims, regardless
   of k or context. If so r90<=4 is near-forced by construction and the registered k-growth and cross-model
   predictions cannot be tested by this intervention. Is this correct? Check whether fit_basis_value centers the
   differences (does centering remove dims?), and whether a rank-3 vs rank-4 distinction matters.
3. Should the full 512-collection 8B sweep and the 14B/32B/72B scaling still run as registered, or is spending on it
   wasted? If wasted, give the minimal pre-registered amendment (before any further data) that would make Q1-prime
   test context-dependent dimension rather than token identity — e.g. more options (16-64 single-token values),
   patching at a later position (e.g. the name token after the fact, or a downstream token) with the value-token
   difference controlled, or a token-embedding-difference baseline (patch only the embedding-layer difference
   propagated) — and predict what it would show.
4. Give the licensed one-sentence statement of the smoke result.
Answer in under 450 words.
