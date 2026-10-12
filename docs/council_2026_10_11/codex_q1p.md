**(A)** The draft overclaims. Residual/logit measurements use only the first prompt per cell; logits cover the whole vocabulary ([native.py:371–398](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/native.py:371)). “All positions” actually means every third position through the batch’s minimum closing index ([native.py:380–382](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/native.py:380)).

Licensed sentence:

“On the reported Qwen3-8B validation cells (W1/W2, k=4, n=128 each), closing-position additions flipped 0/128 four-option answers; first-prompt diagnostics measured residual changes of 9.98–10.03× and maximum whole-vocabulary logit changes of 0.1875–0.875. Sparse multi-position additions flipped 6.25–28.125%. Value-token donor replacement produced 98.44–100% donor-follow at layers 9/18 and 43.75–53.125% at layer 27, versus 0% unpatched. This supports functioning interventions and localized mediation under retained receiver context; it does not establish that the closing position is absent from the answer path.” ([summary](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/cloud/q8b_validate_summary.jsonl:1))

**(B) DO NOT SIGN as written. Exact replacement/addition sentences:**

“Measure conditional single-fact edit dimension, not joint branch capacity; four answer choices alone imply no rank bound.”

“Fix the zero-based decoder-layer output index at round(0.25×depth); this depth is validated only for 8B. Pin model/tokenizer revisions, assert full-prompt token alignment and single-token distinct options, and measure bf16 intervention absorption.”

“Before each model’s sweep, pass a separate n=64 smoke in every (k,family) cell: full-minus-sham ≥0.80 and collateral ≤0.05; collateral is N/A at k=1. Repeat these gates on confirmation data.”

“Fit centered dev differences separately per (model,k,family); match random-projection norms to top-r projections and repeat five frozen random-subspace seeds.”

“Retain paired predictions, collection identifiers, collateral outcomes, option frequencies and constant-answer baselines; report simultaneous collection-bootstrap intervals for gates, retention, r90 and comparisons.” Current serialization drops collateral pairs ([native.py:524–531](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/cloud/lsr_jobs/native.py:524)).

“Define r90 on the tested grid; threshold uncertainty is inconclusive, ranks beyond 64 are censored, and random matching requires a predeclared equivalence margin.”

“The program should continue; the amended 8B probe is the highest-leverage immediate step. PCA remains Rⁿ scaffolding; selective-edit composition, restoration and update closure are the native-mathematics destination.”

Blackboard read/write/synthesis were denied because approval is unavailable; no board record was made.

