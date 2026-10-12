# Fresh error hunt: Registration R 8B full run + 14B/32B/72B smokes (STRICTLY READ-ONLY; do not edit any file)

Read in detail (file:line): experiments/cloud/lsr_jobs/readout.py (seen-split amendment), configs readout_8b_full.json,
readout_{14b,32b,72b}_smoke.json, Registration R + Amendment 1 at the end of theory/LATENT_BRANCH_CAPACITY.md, prior
hunts docs/council_2026_10_11/codex_readout_*.md, and experiments/results/cloud/readout_{8b_full,14b_smoke,32b_smoke,
72b_smoke}_summary.jsonl (fields splits.unseen / splits.seen, seen_minus_unseen, rank_curve, *_by_layer).
Summary of what I see: instrument gates pass in every cell for all four models (pc 1.0). 8B full (n_train 8192,
2048 unseen + 2048 seen tests): unseen k=4 E holds (0.80/0.83 vs hist 0.53), k=16 E fails (0.43/0.45 vs 0.38);
seen k=4 ~1.0, seen k=16 0.86/0.81 (E_seen holds), gap k=16 +0.43/+0.37; r_req unseen k4 6/12, seen k4 8, seen k16 48.
Larger models (smokes, n_train 2048) replicate: unseen k4 0.73-0.86, unseen k16 fails, seen k16 0.69-0.82, gap
+0.26..+0.41. Dev selects the lowest candidate layer (0.25 depth) in every cell of every model. 32B k=1 shuffle 0.127
vs majority 0.27 (below chance).
Questions: 1. Bugs/overclaims, incl. comparability across models (n_train differs), layer-selection pattern, the
below-chance shuffle, the k=1 overlap. 2. Score each registered and Amendment-1 prediction PASS/FAIL. 3. Licensed
sentences (8B verdict; cross-model observation). 4. What could produce a large seen-name advantage other than
name-bound colour information in the closing state (e.g., name-token identity features + slot-position features
allowing a seen-name lookup, reader capacity / overfitting on name identities, tokenization length)? Propose the
decisive control. 5. Highest-leverage next step for the program (do not close the direction). Under 500 words.
