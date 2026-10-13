**WEIGHTED EVIDENCE: PASS. ROBUST EVIDENCE: PASS. COROLLARY PROOF: PASS for the corrected file.**

Read-only CPU review completed. I read AGENTS.md, the required dual-loop skill, approved reviews, frozen registrations, sources, and environment records. I independently rebuilt the calculations without importing either runner or calling `main()`.

| Experiment | Independent verification |
|---|---|
| Weighted | Exactly **15,225 unique cases and states**; **108,159** merge transitions; **76,693** descending-prefix comparisons. Every recorded optimum and prediction agreed exactly. Reconstructed optimal paths attained every optimum. |
| Robust | Exactly **428 unique cases**, split **40/69/130/189** by support size. Rebuilt **96 prior grids**, **3,349 case-state entries**, **4,602 case-edge maxima**, and **7,186 transitions**. All **702 saved path steps and maximizing witnesses** passed, including optimum and count checks. |

Both pre-run records match the reviewed sources, CPU environment, and optimization level zero. Sources, raw data, environment records, and approved reviews retained identical before/after hashes. Frozen ranges and objectives remained unchanged. Concurrent document updates changed status text, corrected the corollary quantifier, and appended the robust FULL receipt; existing ledger entries were preserved.

Concrete probes:

- Weighted `(1,1,3,3)` has exact optimum **64**, attained by merging `(3,3)` first.
- Robust `(M,W,u)=(4,8,3)`, constant output: fixed-schedule optimum **16777216/84375**; distribution-adaptive optimum **65536/729**. The required quantifier order is preserved.
- Outside frozen controls, `(M,W,u)=(2,3,2)` has priors `(1,2),(2,1)` and grid optimum **27/4**, versus continuous prediction **8**. The balanced prior is absent; continuous agreement is **not** extended to odd-denominator grids.

The corrected [corollary](/<LOCAL_PROJECTS>/new-computation-model/research/WEIGHTED_ENTROPY_COROLLARY.md:21) passes. Both distributions have nine atoms, total weight 24, and
\[
\prod_i w_i^{w_i}=2^{40},
\qquad H=\log_2(24)-5/3.
\]

Every descending prefix was compared using exact rational cross-products:

| Prefix size | A merge pair | Against A peak | B merge pair | Against B peak |
|---:|---|:---:|---|:---:|
| 2 | `(4,4)` | `<` | `(8,2)` | `<` |
| 3 | `(8,4)` | `<` | `(10,2)` | `<` |
| 4 | `(12,4)` | `<` | `(12,2)` | `<` |
| 5 | `(16,4)` | `=` | `(14,2)` | `<` |
| 6 | `(20,1)` | `<` | `(16,2)` | `<` |
| 7 | `(21,1)` | `<` | `(18,2)` | `<` |
| 8 | `(22,1)` | `<` | `(20,2)` | `<` |
| 9 | `(23,1)` | `<` | `(22,2)` | `=` |

In particular, A’s largest tail loss is below its peak:
\[
R(23,1)<R(16,4)
\iff 2^{104}3^{24}<5^{20}23^{23}.
\]

At \(K=1/2\), the exact comparisons are:
\[
5^{20}=95367431640625>17592186044416=2^{44},
\]
\[
2^{36}3^{24}=19408409961765342806016
<81402749386839761113321=11^{22}.
\]

Thus **A has no finite-depth realization; B has a finite realization** under the reviewed weighted contract. The original “B CAN at every finite depth” wording was corrected during review: \(24^6>2^{25}\) proves \(H>5/2\), excluding depth five or less. Four-bit encoding with seven zero-probability states fits the weighted support contract.

SHA-256 fingerprints:

| Artifact | Hash |
|---|---|
| `weighted_erasure_full.json` | `10bf1d8196ab84d7da559d1bee0e51e8e00bcb6e43dee65915e9769f66d51c61` |
| `robust_erasure_full.json` | `2ffcef27f5ad5298950fe1d4245673b469a1ea8883cbb790da0e797b88ca6cfd` |
| `weighted_erasure.py` | `9592d370d94f3b5df5649b92834e45bee9b8aa5e0b3a53176bb0a87c1dccdbf3` |
| `robust_erasure.py` | `c04022347ae4cfb6be1318df3582b41518a1e2c80d6fa74f779afe3ca229e104` |
| `weighted_full_run_environment.json` | `7c5648e6b61926b4f662bf426d6b22e3264fb6d10a809a7d0266a04af94d25e6` |
| `robust_full_run_environment.json` | `69d080b122c357a9c382e7118445a19d3e957d100e83526e053fa827fe34e711` |
| Corrected `WEIGHTED_ENTROPY_COROLLARY.md` | `cf7a4642c972f18cbff46c49fc1756916b5512e6a48d4e82ba402de91db18a5b` |

The experiment admissions are **EMPIRICAL exact finite abstraction evidence**. General theorem status rests on proof review. Weighted gate-compiler validation remains separate.