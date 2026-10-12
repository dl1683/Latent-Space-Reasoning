# Fresh error hunt: Q1-prime Amendment A 8B smoke (READ-ONLY; do not edit files)

Read in detail (file:line): experiments/cloud/lsr_jobs/q1pp.py, experiments/cloud/configs/q1pp_8b_smoke.json and
q1pp_8b_sweep.json, the "Q1-prime Amendment A" registration at the end of theory/LATENT_BRANCH_CAPACITY.md,
docs/council_2026_10_11/codex_q1pp_code.md (your previous code review; note the smoke ran on commit 2f4e8fd, before
fixes in 0b130d6), and experiments/results/cloud/q1pp_8b_smoke_summary.jsonl.

Observation: post (closing-token) site, dev admission k=4 n=64: full donor-follow 0.0 and sham 0.0 at layers 9/18/27
in both families -> inadmissible. Value site layer 9: all 6 cells pass the gate (full 0.98-1.0, sham 0, collateral
<=0.4%, receiver acc 1.0); pm alone 1.0; resid alone 0.0; pm+top-r = full; r90 = 0; no contextual claims; pair-mean
energy fraction 0.96-0.99 (k=1) and 0.72-0.77 (k=4/16). Model revision was null in the smoke (snapshot lookup
failed; since fixed to the hub API and pinned in the sweep). The value-site test-name sweep (512, fit 2048) is running.

1. Hunt errors/overclaims. 2. Give the licensed sentence for the smoke. 3. Is the inadmissible post site evidence
of anything beyond "single-position additive patching at the closing token does not move answers in this model"?
4. Given pm-alone already reproduces the value-site effect, does the registered scaling rule (scale only if the 8B
post site is admitted or a value-site contextual claim survives) now say: do not scale? 5. In the program's
interest (do not close the direction), what is the highest-leverage next experiment that tests whether a frozen
model holds context-dependent, high-dimensional fact information at a single position — e.g. multi-position or
"set" patching at the closing token, a fact-free receiver, or reading rather than writing (linear decodability
of all k facts from the closing state with held-out names)? Under 450 words.
