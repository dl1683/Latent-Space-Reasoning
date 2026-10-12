**Replace, do not sign yet.** The proposal follows direction 3, but it leaves the training dose undefined and does not isolate the optimizer mechanism ([codex_stepback.md:41–53](docs/council_2026_10_11/codex_stepback.md#L41-L53); [branch.py:469–538](experiments/cloud/lsr_jobs/branch.py#L469-L538)).

Replace “identical examples, identical total optimizer steps” with: “Pre-register a numeric step count and identical per-family multisets and exposure counts across arms; reserve the same final 10% of steps for balanced consolidation.”

Replace “a separate AdamW state per family” with: “Use family-specific first- and second-moment buffers, but synchronize bias-correction age and the global learning-rate schedule; otherwise optimizer age confounds moment separation.”

Add: “Report each family’s accuracy, option-choice frequencies, and answer-format failures at every checkpoint. Held-out wording tests paraphrase transfer for trained facts, not generalization to unseen facts.” The sibling formation result is an analogy, not LSR evidence ([CROSS_PROJECT_INSIGHTS.md:49–56](theory/CROSS_PROJECT_INSIGHTS.md#L49-L56)).
