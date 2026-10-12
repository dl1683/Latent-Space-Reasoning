# Fresh error hunt: Registration R 8B readout smoke (STRICTLY READ-ONLY; do not edit any file)

Read in detail (file:line): experiments/cloud/lsr_jobs/readout.py, configs/readout_8b_smoke.json, "Registration R"
(+ pre-smoke amendments) at the end of theory/LATENT_BRANCH_CAPACITY.md, docs/council_2026_10_11/codex_readout_code.md,
and experiments/results/cloud/readout_8b_smoke_summary.jsonl (fields full_rank_dev_acc_by_layer / _test_acc_by_layer).
Observations (Qwen3-8B, rev b968826, n_test 512 TEST-name collections): instrument gates pass in all 6 cells (pc 1.0;
shuffle within 0.03 of majority). k=1 E1 holds (1.0, r_req 2). k=4 E holds: mean slot acc 0.80/0.83 vs hist-mode
0.54/0.52; changed-slot margin +0.38/+0.42; r_req 8 (W1) / 16 (W2). k=16 E fails (margin 0.01/0.03 < 0.10) though
changed-slot margin +0.055/+0.085 with CI>0; per-slot acc shows first/last slots readable, middle near chance.
KEY: dev accuracy (new collections, same 20 DEV names as train) is ~1.0 at k=4 and 0.72/0.68 at k=16, vs test
(held-out names) 0.80-0.87 and 0.40-0.48. Dev selected layer 9 while test is best at 18 for W1k4, W1k16, W2k16.
Questions: 1. Bugs/overclaims; licensed sentence. 2. Is the dev-vs-test gap evidence that the closing state carries
name-specific binding codes readable for seen names (k=16 ~0.7 vs 0.38 hist-mode), or can it be explained by dev
selection bias / leakage (check seeds, name overlap, collection duplication probability)? 3. Propose the minimal
pre-registered amendment before the 8B full run: e.g. add a fresh seen-name test split (new collections, DEV names,
never used for selection) with its own E_seen / r_req_seen claims, and whether layer selection should stay on dev.
4. Is the 8B instrument admissible for scaling to 14B/32B/72B under the registration? Under 450 words, file:line.
