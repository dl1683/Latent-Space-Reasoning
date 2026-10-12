**Verdict: Q1 mediation is unestablished; Q2 fails the tested transplant interface. Reject the draft’s inference that “facts stay at their own token positions.”**

I read `native.py` fully and verified that the captured run bundle matches it.

- **Zeros need instrument validation.** Right padding, batch slicing, add/set operations and last-token scoring are internally consistent ([native.py:54](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/native.py:54)). But closing positions come from standalone blocks, without asserting that their token IDs remain the full prompt’s prefix; Q2’s placeholder assertion similarly checks standalone text ([native.py:104](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/native.py:104), [native.py:200](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/native.py:200)). The ledger’s 10×-norm diagnostic produced 0/64 flips on the **voided, shorter-suffix run**; it is a failed positive control, not proof of routing. The CPU check on another model does not validate this execution ([ledger.jsonl:1118](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/ledger.jsonl:1118)). Diagnostic comments promise additional swaps that are never executed ([native.py:308](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/native.py:308)).

- **Denominators and licensed results.** Raw Q1 records contain 512 donor-follow indicators per variant per cell, all zero; these are donor-follow outcomes, not complete targeted-flip records. Collateral is absent at k=1, zero over 1,536 queried unchanged facts at k=4 and 2,048 at k=16. Retention is undefined. Q2 uses 512 collections: 512 answers at k=1, 2,048 clustered answers at k=4. Patched accuracy is **0.2227–0.2715**, empirical constant baselines **0.2544–0.2891**, full-context accuracy **0.9990–1.0000**. Outputs vary but show strong option preferences. Termination is N/A for option-logit scoring.

- **Protocol deviations.** Q2 seeds include m, producing different collections across ladder rungs; raw gold arrays confirm this, contradicting the registration. Rolled donors lack explicit length/position matching ([native.py:209](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/native.py:209), [native.py:281](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/native.py:281)). Captured model/tokenizer revisions are null; identical-input noise is unreported.

- **PR and layer choice.** All fits use 2,048 pairs, so unequal fit counts do not explain the PR trend. Increasing k changes sampled fact identities, positions and context lengths; centered PCA therefore remains descriptive scaffolding ([native.py:177](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/native.py:177)). Layer 18 wins an all-zero dev tie, establishing no privileged mediator.

**Continue the program; scaling this interface is not highest leverage.** Next validate full-prompt token alignment, actual post-hook residuals and logit changes, then establish a localized causal positive control.

**Never say:** “facts stay at their tokens”; “suffix states contain no facts”; “pretrained models do not superpose facts”; “PR measures causal capacity”; “the hook was validated.”

Blackboard inspection and synthesis were denied because those calls require approval and this session’s policy is `never`.

