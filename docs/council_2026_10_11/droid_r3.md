## Round 3 verdict

**Yes, with two amendments to `codex_r2.md`:**

- Remove frozen Probe A from the required schedule and mark it skipped before any run. Injecting a last-layer residual as a fresh input embedding tests a weak interface; a null would not inform the trained gate. Do not treat the unrun frozen prediction as evidence.
- Freeze the storage/delayed-computation control’s architecture, training allowance, state budget, and comparison rule **before flagship training**. It may be trained only after admission, but it must be evaluated on the same locked problems before any computation claim.

**Operator notes:** (a) Agree, use one configurable trainer for the 0.6B gate and 1.7B flagship. (b) Agree, subject to the second amendment; passing admission does not itself establish computation. (c) Agree, subject to explicitly amending the frozen schedule rather than silently dropping a preregistered test.

**The result that would most change my view:** On held-out compositions, a single state supports branch-addressed edit, undo, and subsequent updates at $k\geq4$, while beating both the matched storage/delayed-computation control and the strongest equal-FLOP alternative. That would make update-closed, selective answerability worth developing as a mathematical primitive. A participation-ratio scaling fit alone would not.

No experiments, cloud actions, spend, or file edits occurred.
