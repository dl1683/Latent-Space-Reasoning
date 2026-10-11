# Re-council after cross-project extraction (2026-10-11, round A1)

The user directed a deep read of two sibling projects: Matrix Native Math (matrices as the atom: precision/bit laws, structure, dynamic range) and the LLM Genome / Neural Dissection Lab (how stored items form, die, consolidate under optimizer steps). The distillation is theory/CROSS_PROJECT_INSIGHTS.md. You may read the sibling repos themselves read-only if a claim matters (paths are in the file); do not edit them. The frozen pre-registration is theory/LATENT_BRANCH_CAPACITY.md; council history in docs/council_2026_10_11/. User reminder: the program is the mathematics of high-dimensional spaces; LLM latent spaces are one instance.

Status: the runner is validated (smoke passed; auto-off self-test passes on each VM). Probe C pilot exposed truncation (direct mode 0 percent termination at 16 tokens, CoT 37-42 percent at 384); fixed by an answer prefill with newline stop and a 1024-token CoT budget, re-piloting now. A non-claim-bearing throughput pilot of the 0.6B gate is running. No claim-bearing data exists yet, so amendments are still clean.

Answer concisely (under 900 words):
1. For each proposed change in section E of the insights file: ADOPT, REJECT or MODIFY, one line each with reason. Especially: the PR estimator fix (n at least 8r or bias-corrected, isotropic null reported); quantized-z eval arm; C at 80/90/95 percent; per-cell training logs; whether the concurrent-formation risk justifies changing the training recipe now (staged curriculum) or only logging; the Jacobian block-structure diagnostic; the random-init backbone null.
2. Do any insights change the portfolio itself? In particular, is the proposed new bet (capacity as a function of width r and precision b, interference regime vs information regime, crossover where sqrt(PR)-limited capacity meets the r times b over 16 ceiling) worth adding, replacing, or folding into the flagship? Is there a better high-dimensional bet that the three projects together suggest?
3. Any change to the numerical predictions (the 1.5-2.7x bands; the 64-to-256 comparison if r=64 is precision- or floor-limited)?
4. Write the exact text of a dated amendment (Amendment 1, 2026-10-11) you would sign.
Write markdown to your output. No experiments, no spend, no file edits.
