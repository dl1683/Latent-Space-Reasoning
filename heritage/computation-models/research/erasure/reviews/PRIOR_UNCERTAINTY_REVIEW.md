| Gate | Decision |
|---|---|
| **LOWER** | **PASS — PROVED AT SCOPE**, universal over all finite circuits in the stated basis |
| **UPPER** | **PASS — PROVED AT SCOPE**, including the uniform comparison and K=1 cliff |
| **CODE** | **PASS at the revised hashes below**, under the explicit snapshot-launcher and observed-exit admission contract |

External edits superseded the initial code snapshot during review; that snapshot is **VOID for current-source admission**. I reread the revisions and verified the final snapshot remained stable. CPU only; no file changes, additional agents, or FULL execution.

**LOWER.** A RESET’s complete-state fibres have size at most two, so its collisions are disjoint pairs. Four pairs provide eight distinct predecessor classes. Selecting one original input from each gives a valid eight-atom prior—even after earlier coalescence—with entropy loss \(3-2=1\) bit.

The quantifier order is legitimate: admission requires one circuit to satisfy **every prior at every gate**. A counterexample prior may depend on the proposed circuit and attacked gate; it remains one fixed original-input distribution.

Initially there are \(2^n\) complete states; exact cleanup leaves precisely two. Reversible gates preserve cardinality. Consequently,

\[
G\ge R\ge\left\lceil\frac{2^n-2}{3}\right\rceil.
\]

This covers arbitrary finite initialized workspace, reversible recoding, and depth. Required registers give \(W\ge n+1\); targeting every input and the distinct output also gives \(G\ge n+1\).

**UPPER.** The Gray construction removes every scratch distinction involving the flipped bit before RESET. Each actual merge absorbs an original singleton of mass \(b\le1/8\), so

\[
L(a,b)\le h_2(b)\le h_2(1/8)<3/4.
\]

The exact comparison is \(2^{18}=262144<7^7=823543\). Zero-mass singletons cause no problem; the final AND-zero move into the all-ones encoding loses zero entropy. Full input, output, and workspace states remain correctly accounted for.

Gray amortization supplies \(O(2^n+n)\) gates and width \(2n+4\), establishing robust work \(\Theta(2^n)\). Under uniform input, the rational constructor uses a constant three-bit residual stage and linear cleanup work, giving \(\Theta(n)\). Uniform feasibility follows from
\(K_*\le h_2(2^{-n})\le h_2(1/8)<3/4\).

At K=1, every one-bit RESET loses at most one bit under any prior, permitting linear clean computation. The generalized collision lemma also survives **with its explicit \(2t\le M\) condition**. It supplies a capacity bound, not a general tight frontier; \(t=1\) forbids class collapse.

**CODE — finite evidence, EMPIRICAL.**

- Independently reconstructed all four smoke circuits and all 33 smoke priors, matching every saved trace and witness: **8,336 full-support transitions, 38 RESETs, 40 collision pairs**.
- Reproduced the n=4 uniform witness: eight distinct complete states become four, with exact ratio **256**, exceeding the cap ratio **64**. A separate synthetic probe verified witness selection after genuine coalescence.
- Ten additional rational priors passed robust cap and cleanup checks. Small K=1 circuits passed; Gray counts and amortization passed through 12 dimensions.
- Nine injected gate, output, width, and count defects were rejected. Actual `-O`/`-OO` subprocesses and optimized code objects were rejected.
- Case-two failures recorded KILL/VOID, the first failing unit and traceback, returned 1, and stopped further cases. Source mismatches produced VOID.
- The revised launcher binds compilation to captured source bytes. Strict descendant-file confinement and exclusive temporary publication passed modeled boundary, link, replacement, and failure probes.

Publication can still precede a later stdout failure, leaving JSON CONFIRM. **The separately observed zero exit, matching attempt, source hashes, artifact hash, inventory, and independent raw audit are therefore mandatory.** Filesystem publication probes were mocked; live Windows atomicity and hostile directory swaps were not tested.

FULL remains gated here: **16 circuits across n=3..10 and all 12,871 eight-subset priors were not executed or admitted**. CONFIRM/KILL/VOID criteria remain unchanged; independent raw admission is required after FULL.

**Priority and mission uncertainty.** Strongest dismissal: the separation follows from an imposed logical peak cap, an adversarial-prior quantifier, elementary collision counting, and an existing singleton Gray construction. Distribution-sensitive reversibility and irreversibility accounting already have substantial foundations in [Frank](https://arxiv.org/abs/1806.10183) and [Demaine et al.](https://erikdemaine.org/papers/Energy_ITCS2016/paper.pdf); this limited comparison establishes no priority.

The returned-fair-bit objection is particularly serious: in that changed product-prior contract, conditioning each reset on one fair-bit value bounds its loss by \(1/2\), enabling linear cleanup. Six finite cases supported it. This leaves the fixed no-seed theorem intact while limiting its practical interpretation. Useful-world significance remains unresolved; this review grants no whole-mission conclusion.

Final SHA-256 fingerprints:

```text
research/PRIOR_UNCERTAINTY_CANDIDATE.md
209c7184e948a96dd025d243bcfab01a5899e276c9ea5f861466e811decfffe0
research/SUCCINCT_ERASURE_CANDIDATE.md
c84b31b4292d176015ac6c17a701fcebcbd4effe74953b9c25a0a1089008d760
experiments/prior_uncertainty.py
881ca25349ab1ab7222cb212118ec0a1ce6f9e21cfb962613a590fab5fa90afc
experiments/succinct_erasure.py
a8c1d33a98822de83792f344842c9ac90b709134eb6cb10d012d71b6f3f0e71c
experiments/evidence_io.py
2bf4310800821fabf69f9a95bb71561ddf1a595d9263ec6f8a3fa90d1675386f
experiments/run_snapshot.py
2681e422d7cb5ba7b50f61a7bb973b6e308cd7a4f9b8da7386437c48038406a1
results/prior_uncertainty_smoke.json
d741dfa7deaae8d11b07a567ffb0aceb05fbebfe2f35b7e49aa541eae0ba3b0c
results/snapshot_smoke_exit_receipts.json
dfda00e13ea9d2ac0960b6de67f52ae628b30a519e137dc71cb0de83aeba1937
research/SUCCINCT_ERASURE_REVIEW_1.md
aa2754fe31b9b3e8bb17c8f60b284aa194bec6bfef91c6cdd7ec9bca27b26895
research/SUCCINCT_ERASURE_REVIEW_2.md
0ed24ca738d22c88614f0f1cf960729492769d1c448ad87b773756021fb5f525
EXPERIMENTS.md
241430b0586af39b4c22c73fc09c293f83cde02347a9dfcba59a8d9d697361b7
experiments/ledger.jsonl
d83508374abbb24c4d7be4a2cbd785085d3a71240cf4f0048e6bce072a98ab46
AGENTS.md
e9b7400be1f4bf550fc46ee4cc0fe495b20d32d7dda5c30d370e1922ee75b6b7
dual-loop/SKILL.md
334a7d272c572b4dbc8c36a5fde15385fa7962df5c234ba8772c199628af0151
```