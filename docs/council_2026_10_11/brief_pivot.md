# Council: gate outcome and what would work (2026-10-11 ~22:20Z)

Read fresh: theory/LATENT_BRANCH_CAPACITY.md (pre-registration, Amendments 1-2, Deviation 1), theory/CROSS_PROJECT_INSIGHTS.md, experiments/EXPERIMENTS.md (top), docs/council_2026_10_11/droid_review_2.md, the job code experiments/cloud/lsr_jobs/branch.py. Mission reminder: mathematics of high-dimensional spaces; LLM latent spaces are one instance. You are goal-aligned adversaries; never close the mission.

Evidence (DEV, curriculum arm, Qwen3-0.6B, 3 seeds, step 4250/4687; locked test eval follows within ~2h):
- Accuracy is IDENTICAL across nested widths 64/256/1024 in every cell and seed (to 0.01). Under nested-mask training the model writes everything into the first 64 shared coordinates; width never binds. The pre-registered width-scaling test therefore degenerates (branch PR will not be width-dependent -> "not tested" by the frozen rules).
- Storage (u=0): k=1 0.96-1.00; k=4 0.88-0.98; k=16 0.59-0.67; k=64 0.53-0.56. Capacity at the 90% bar is about 4 graphs (16 bits each) in at most 64 coordinates.
- Updates (fresh-context passes): k=1 one update 0.77-0.78, two updates 0.60-0.61 (chance 0.5). k=4 with updates is HIGHER (0.80-0.88) because half the ops are branch-addressed and mostly miss the queried branch. Carrying a state through a fresh-context update is the failure point.
- Concurrent arm: constant output through step ~2000 (3/3 seeds), finishing ~23:55Z.
- Gate (frozen): single-branch >= 90% at every update count will almost surely FAIL (u1 ~0.77, u2 ~0.61). Pre-registered consequence: no flagship escalation.
Other probes: B (latent search) ES == matched random perturbation in 14/14 models (no direction search); random final-position perturbation beats tuned temperature sampling only in Qwen3-8B/1.7B (confirmation on 512 new problems per family running). C (frozen recurrence) trending clean negative on admissible families; graph direct-mode "gains" were a constant-answer artifact.
Window: until 2026-10-16 (~4.5 days). Spend so far ~$450 of $5,000; GPUs are plentiful (L4/A100/H100 Spot).

Questions (answer in under 900 words, concrete):
1. Do you agree the gate fails if the locked test confirms these dev numbers, and that the flagship (1.7B) must not escalate under the current registration?
2. "This is not working - what would work?" Propose the next registered experiment(s) for the remaining 4 days, keeping the mission and the high-dimensional framing central. Candidates from the operator (attack them, or replace them):
   (a) STORAGE CAPACITY LAW, done right: separately trained models per width r in {2,4,8,16,32,64,128} (no nested sharing), u=0 only, many seeds, measuring C(r) and branch-difference PR (n=8192, isotropic null) -> test C ~ sqrt(PR) vs C ~ PR (linear, information-limited) vs a precision ceiling, plus the quantized-z arm. Cheap and wide (0.6B learns u=0 fast).
   (b) UPDATE CLOSURE as a trained law: add a commutation (consistency) objective that makes the fresh-context update operator commute with encoding, U(E(G), op) ~ E(apply(G, op)), using the model's own encoder as the target (no external labels); then measure the commutation defect as a function of k, width and number of updates. Is this admissible under the council's "no auxiliary labels" rule? Does it make the flagship claim trivial or meaningful?
   (c) Any better idea.
3. Give the exact pre-registration text (predictions with numbers, kill criteria, controls, budget) for what you choose. Sign it.
Write markdown. No experiments, no spend, no file edits.
