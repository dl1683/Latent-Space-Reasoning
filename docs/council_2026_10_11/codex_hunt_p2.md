**A. P2 FAIL stands, with narrower wording and several corrections.**

Licensed sentence: “At the tested exposure, consistency training reduces variance-normalised encoder-target defect but misses P2’s registered answerability and selective-edit thresholds in all three seeds. This construction establishes neither update closure nor functional equivalence from reduced geometric defect.”

The two-update gains are +0.68/+0.39/+0.59 pp at k=1 and +2.69/+10.62/−1.59 pp at k=4. No arm reaches 90% at every selector. These numbers support FAIL independently of significance labels. [Results:103](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/cloud/p2_summary.txt:103)

The error hunt found:

- **Denominators check out.** All 54 accuracy cells contain 1,024 collections; k=4 exhausts four selectors, giving 4,096 answers. Raw collection counts reproduce stored accuracies. Defect uses a separate 512-collection sample; causal tests use 512 witnesses, including **286 informative futures**. Informative edit-and-undo success is conditional on those 286, not all 512. [Evaluator:239](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/branch.py:239), [informative subset:360](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/branch.py:360)
- **Defects are comparable within seed.** Raw normalisers are identical across arms: 0.581658, 0.654691, 0.749736 for seeds 11/23/47. However, seed-11 k=1 u=2 improves only 2.464→1.415, **1.74×**, missing halving. Avoid blanket “2–4×.” The normaliser comes from k=4 dev states, so do not treat cross-k defect magnitudes as calibrated functional distances. [Normalisation:192](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/p2.py:192)
- **Sham equality does not prove ignored updates.** Seed-11 consistency’s main/sham accuracies coincide; answer-only differs slightly. Aggregate equality cannot establish identical predictions. Zero/shuffled-state controls near 50% show reliance on retained state, but cannot distinguish retained initial answers from correct updating. Sham strings also differ from toggle strings; exact token/FLOP matching is not enforced. [Controls:262](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/branch.py:262), [instruction strings:112](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/branch.py:112)
- **Undo is a signed ratio**, `(undo + follow − 1)/(base + follow − 1)`, with tiny denominators when targeted effects are small. Negative values mean undo falls below the post-edit original-answer accuracy; they are not negative percentages of recovered edits. Separately, this ratio measures immediate undo, not recovery after subsequent updates. [Formula:373](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/branch.py:373)
- **Correct the shuffled-arm claim:** 18.9/24.6/2.7 pp targeted and 17.5/32.7/2.0 pp collateral. The draft omits seed 47. Seed-11 k=4 already lacks encoder/reader competence: minimum-selector dev accuracy 0.652. Seeds 23/47 pass that proximal check, so this caveat does not explain every failure. [Results:1](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/cloud/p2_summary.txt:1)
- **Exposure is configured identically at 192,000**, and every arm computes consistency targets; measured FLOPs within 5% remain unverified. Competence is dev-only and does not gate training. Test data enter evaluation, not the loss. Reachability-mode graph patterns are **not disjoint**; only update pairs are split. [Training:100](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/p2.py:100), [sampler:80](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/branch.py:80)
- Collection pairing is appropriate, but the printed bootstrap-tail “p” is not a calibrated null test; Holm output also lacks cumulative monotonic adjustment. Do not report exact-zero significance or multiplicity-adjusted CIs. [Aggregation:204](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/aggregate.py:204)

**Never say:** “consistency never helps,” “updates are ignored,” “geometric closeness cannot transfer,” “all shuffled edits are large,” “intrinsic capacity/scaling refuted,” or “native mathematics established.”

**B. The P1′ stop rule fires; do not launch remaining seeds.**

The quoted 0.898/0.962 are **mean** dev accuracies. Promotion requires minimum-selector accuracy: across recorded promotion checks, r64 never exceeds 0.640625 and r128 never exceeds 0.859375. Synchronisation implements the conjunctive two-check rule correctly. [Mastery:497](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/branch.py:497)

Both leaders’ test minimum-selector accuracies fail even at k=1: 0.691/0.826. Report **no positive C90 established; floor-censored below k=1**, not capacity inferred from curriculum failure. Scaling remains NOT TESTED. The nominal 300k allowance implements 299,968 presentations through integer division; this small shortfall does not explain these margins. [Raw r64:70](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/cloud/p1b-s11/shards/005/out/results.jsonl:70), [stop rule:217](C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/LATENT_BRANCH_CAPACITY.md:217)

**C. Continue the mission; next test future-distinguishability of the frozen state.**

So what: *does the hidden state remember enough to answer a question whose answer changes tomorrow?*

Define equivalence by answers under allowed future operations. Construct paired k=1 graphs with identical current reachability but opposite answers after the same two toggles. Freeze existing encoders; train fresh matched readers given state, operations and query. Arms: encoded state, shuffled state, current-answer-only state, full graph text. Split graph patterns and operation pairs explicitly; cluster statistics by witness pair.

Proposed prediction: state ≥90%, ≥20 pp above shuffled/current-answer controls; full text ≥95%. Cap at 30,000 training collections per arm, three existing encoder seeds, 1,024 held-out pairs per seed, six cloud GPU-hours. Kill this recipe if state misses prediction while full text passes; full-text failure invalidates the instrument.

Success locates the bottleneck downstream of retained information; failure establishes only a decoding limitation under this recipe. This replaces geometric distance with future-observable distinctions, while acknowledging that behavioral equivalence is established mathematics and novelty remains open.

No files changed or experiments launched. Blackboard state/synthesis calls were unavailable because the tools required approval under the never-approval policy.

