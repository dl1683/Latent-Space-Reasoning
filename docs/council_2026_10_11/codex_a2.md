**Do not sign as written.** Retain the proposed stage schedule; replace the surrounding sentences with:

> “The concurrent arm and its frozen repair remain unchanged. Register the curriculum before launching it, using fresh, seed-matched initialization rather than diagnostic checkpoints. Match optimizer, learning rate, LoRA, widths, batch size, optimizer steps and total presentations exactly to the unrepaired concurrent arm, within existing spending caps; the curriculum receives no additional repair.
>
> Stage boundaries are floor(fraction × total steps). Within each stage, sample k uniformly from its listed powers of two, update count uniformly from its listed set, and width uniformly as frozen; preserve the applicable 50/50 toggle mixture. Record presentations and token/FLOP exposure by cell: equal presentations do not imply equal compute.
>
> Select checkpoints only after the final boundary using the frozen dev score. Report both arms’ gate verdicts under unchanged thresholds. Choose the flagship recipe using dev results only, preferring concurrent if both pass; freeze that choice before locked evaluation. Apply the chosen recipe consistently to the flagship and mandatory storage control. Preserve all remaining definitions and verdict rules.”

**Bug check first:** The reported proximal success supports interface learnability, but cannot exclude mixed-path bugs. Verify mixed-batch graph/update labels, selector indexing, masking, binary-head scoring, and gradients through E/D and update passages; compare sampled labels with an independent graph oracle. Constant predictions warrant these checks, but do not establish a bug. Step 250/4687 is early; retain the concurrent run.

Blackboard state access and synthesis were rejected by the automatic approval policy.

