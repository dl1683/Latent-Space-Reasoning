# Amendment 3 request (2026-10-11 ~23:35Z): contingent on the P1 stop rule

> **SUPERSEDED.** This operator draft proposed independent mastery promotion. The signed Amendment 3 (theory/LATENT_BRANCH_CAPACITY.md, Amendment 3 section) replaces it with synchronized promotion. Retained for the council record.

Read theory/LATENT_BRANCH_CAPACITY.md (all sections incl. Registrations P1/P2), experiments/EXPERIMENTS.md (top: LOCKED gate verdict).

Facts:
- LOCKED gate verdict (curriculum arm, test, 3 seeds): FAIL. Storage k=1 0.97-0.99; C90 = 2-4 reachability answers; k=1 one update 0.84-0.86, two updates ~0.76; branch-addressed edits ~0 pp targeted effect; edit-then-advance tracks the unedited graph (0.41); one seed ignores global updates (main == sham). No flagship escalation.
- P1 seed 11 (all 7 widths r=2..128, edge-query task, registered schedule 10% k=1, 10% k<=2, then full k mix): dev k=1 accuracy is ~0.60 at EVERY width (r=2 identical to r=128) at steps 250 and 500 of 1,562, loss ~0.687 (chance 0.693); the full-text positive control learned (loss 0.02 by step 225). This matches the concurrent-mixture stall seen in the original gate arm (constant output under the full mix) and the Genome lab's concurrent-formation law. The P1 stop rule (r=64 and r=128 must reach 95% at k=1 on dev within the exposure allowance) will formally evaluate at step 1,562 (~00:20Z); it is very likely to fire.

Proposed Amendment 3 (operator draft), effective only if the P1 stop rule fires:
(a) P1 STOPS as registered (remaining seeds not launched); its seed-11 runs are reported as a stop-rule outcome, no capacity claim.
(b) P1' (new registration, same task, widths, seeds, measures, predictions, controls and budget as P1) with ONE change: a mastery-gated curriculum: k_max starts at 1 and doubles when the dev min-selector accuracy at the current k_max reaches 0.95 (checked every 100 steps, dev n=64), training on k uniform over powers of two <= k_max; exposure cap raised to 300,000 collections; checkpoint selection after k_max reaches 64 or at the cap, whichever first, by mean dev accuracy over the full k grid; a run that never reaches k_max=64 reports C90 as censored at its final k_max. Stop rule unchanged but evaluated at the cap.
(c) P2 base changed: P2 starts from the LOCKED gate curriculum checkpoints (reachability task, r=64, seeds 11/23/47), which store well (k=1 0.97-0.99) and fail updates; frozen encoder and reader; separate updater exactly as registered; the query is reachability (as in the gate), k in {1,4}, updates 0-2; all P2 predictions and controls unchanged. Rationale: P2 tests whether a consistency target fixes precisely the measured update failure, and no longer waits on P1.

Answer in under 250 words: SIGN as written, or give exact replacement sentences. If you think the mastery-gated curriculum invalidates comparability across widths (different exposure per width), say how to fix it.
