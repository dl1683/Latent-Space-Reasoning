**GATE A — REPAIR; no whole-repository correctness/integrity admission.** The examined October theorem core survived my attacks at its stated scopes. I found a repairable clean-output contract error in one review report, but no fatal counterexample to the central frontiers. Coverage limitations also prevent an entire-repository PASS.

**GATE B — NO.** The repository has substantial scoped capability and impossibility results. It has not yet earned useful-world significance, strict primary-literature standing, or completion of the new-computation-model mission. I am not independently won over by the entire repository.

**Coverage and changing state**

I inventoried every tracked artifact. The final checked committed inventory, at `0d20365a`, contained **331 files, 330,176,376 bytes**: 167 documents/registries, 14 Python files, 50 PDFs, 49 extracted texts, 49 result JSON files and two visual artifacts.

I read the constitution, full dual-loop skill, required startup surfaces, all 14 Python sources, the current October definitions/proofs and their scoped review reports. I parsed every result JSON, inspected receipts and source bindings, and viewed the plotted results. Historical documents received structural and selected substantive inspection; **I did not semantically read every line of the large QUESTIONS/WORKLOG archives or every page of the 1,465-page PDF corpus**. The scanned Reif–Tyagi technical report lacks usable extracted text. Those are explicit whole-repository coverage limits.

My bounded attacks included **1,390,456 independently interpreted transitions across 22 small masked/generic circuits**, plus **204,032 transitions** through a padded Gray counterconstruction. I also checked all 116 masked words’ recorded counts and all 18,718 RESET certificate records without replaying their transitions. No edits, agents, GPU, commits or registered FULL runs occurred.

The separate raw reviewer finished during this review. [MASKED_BLOCK_FULL_EVIDENCE_REVIEW.md](<LOCAL_PROJECTS>/new-computation-model/research/MASKED_BLOCK_FULL_EVIDENCE_REVIEW.md) now records **EMPIRICAL admission of all 116 circuits**. Its artifact hash and bindings agree with the saved results. I did not duplicate its 429,592,644,496-transition reconstruction or treat that admission as a universal theorem.

**Defects and decisive attacks**

1. **Nonfatal integrity repair: the min-atom review’s shortcut violates the output contract.**  
   [MIN_ATOM_BRANCH_REVIEW.md:14](<LOCAL_PROJECTS>/new-computation-model/research/MIN_ATOM_BRANCH_REVIEW.md:14) proposes leaving R unchanged, with `3n−1` gates and `2n+r` registers. The common contract requires R in distinct output registers and its original registers cleared. For `r=1, R=1`, that shortcut leaves a dirty input and an unset context output.

   Add two CNOTs per context bit. The corrected construction uses **`3n+2r−1` gates and `2n+2r` registers**. The branch-cap upper remains valid; the shortcut and its resource figures require correction.

2. **Finite counterexample: uniform marginals do not support the product-mask guarantee.**  
   For the [masked compiler](<LOCAL_PROJECTS>/new-computation-model/experiments/masked_block.py:28), take `n=r=4`, uniform R, and
   ```
   X = R xor 6   when R < 8
   X = R         when R >= 8.
   ```
   X is also uniform, hence satisfies the `1/8` atom cap. At RESET index **38**, the flag probability is **1/2**. Its local cost is **1**, exceeding `h₂(1/16)`.

   This is outside the reviewed product contract. It confirms that independence is essential and that a returned uniform marginal cannot substitute for it. Pointwise return likewise does not guarantee independence from later workloads formed using R.

3. **Finite sensitivity witness: small mask bias breaks the exact uniform cap.**  
   Independently choose X uniformly on `0,…,7`, and assign R probability `17/256` on `0,…,7`, `15/256` otherwise. The same RESET has flag probability **`17/256`**, giving
   \[
   h_2(17/256)>h_2(1/16).
   \]
   This refutes any tolerance-free extension of the exact-uniform cap. The new [MASK_SOURCE_REVIEW.md](<LOCAL_PROJECTS>/new-computation-model/research/MASK_SOURCE_REVIEW.md) correctly permits bias by changing the promise and feasibility floor; it does not preserve the old cap.

4. **The broad padding and source-unification objections were wrong—and are now withdrawn.**  
   I verified a clean `n=r=4` construction that returns R, ignores it during incoming-singleton Gray cleanup, and inserts charged reversible padding. Its **797 gates** satisfy window-2 local cap `h₂(1/16)`: each window contains at most one RESET, whose cost is at most `h₂(1/8)<2h₂(1/16)`.

   Thus excluding a unit-loss RESET does not exclude every unmasked circuit. Conversely, padding cannot hide that unit loss when `wK<1`. The latest [advisor round 3](<LOCAL_PROJECTS>/new-computation-model/research/SUPERVISOR_OCTOBER_2_ROUND_3.md) withdraws the overbroad exclusion, single-τ unification and unsupported sampling exponent. These are corrected steering errors, not current theorem failures.

No fatal infinity leak emerged in the examined core. Programs have finite words and registers; predicates expand into charged gates; probability laws are quantified promises rather than machine-readable real oracles. Preparation and authentication remain uninstantiated costs, which limits operational claims rather than invalidating the stated logical theorems.

**Surviving capability and impossibility**

These statements survive my examination within **finite deterministic, fixed-schedule NOT/CNOT/Toffoli/RESET circuits**, with their particular source promises and cleanup contracts:

| Scope | Surviving result |
|---|---|
| Uniform finite inputs | Exact all-depth joint-loss threshold `φ(q)/M`; selective singleton absorption supplies a charged finite compiler, and first fibre formation supplies the universal lower. |
| Positive rational weighted support | Exact descending heaviest-prefix frontier. The implemented lexicographic compiler is not weighted-optimal evidence. |
| Unknown bounded-atom priors | The piecewise robust feasibility frontier survives. Continuous-prior statements cannot be transferred unchanged to restricted probability grids. |
| Known uniform AND | At every feasible rational joint-loss cap, optimal elementary work `Θ(n+r+n/K)` and linear width; `n=1` remains a reversible exception. |
| Unknown capped X, no helper | At joint cap `3/4`, optimal `Θ(2^n)` work; cap one permits linear work. The uniform comparison changes the source promise explicitly. |
| Returned fair marginal | Linear cleanup with joint loss at most `1/2`, including arbitrary X/R correlation. Fair coordinates alone do not bound complete-context masses. |
| Local product-mask AND | Optimal critical work `Θ(n+r+2^r max(1,n/r))`, uniformly in growing r. |
| Generic cleanup | Charged `C_f` plus mask cleanup; exact RESET marginal `2^-r` for every unknown X law under independent uniform R. Exact noninjective universal local floor `h₂(2^-r)`. |
| Correlated robustness | For `n≥3,r≥4`, critical work `Θ(n2^r+r)`. The total-cost lower `Ω_C(nr)` requires the fixed allowance `G≤Cn2^r`; it does not cover arbitrarily slow circuits. |
| Branch meter | Uniform exact floor `log₂3`, with optimal linear work. Without a minimum positive atom, no finite universal cap for `n≥4`. The min-atom result gives an enclosure within three bits, not an exact frontier or a fixed probability-grid theorem. |

The [common model](<LOCAL_PROJECTS>/new-computation-model/research/STATISTICAL_CONTEXT_MODEL.md:60) mostly preserves these distinctions correctly. The correlated total-cost restriction is explicit in [TOTAL_LOCAL_CANDIDATE.md:54](<LOCAL_PROJECTS>/new-computation-model/research/TOTAL_LOCAL_CANDIDATE.md:54). The newer window/fixed-design extension remains candidate material at my read boundary.

July remains parked. [HANDOFF section 17](<LOCAL_PROJECTS>/new-computation-model/research/HANDOFF.md:264) leaves bridge semantics, observation transport/equivariance and the semantic CANNOT open. Defined transformations and failed recorded routes do not establish the old completed pair. No terminal or READY claim is resurrected.

**Major mission obstacles**

The strongest obstacle is an **operational resource contract**. J, L and V measure different things. A fixed rational design supplies finite data, but does not supply a finite-time implementation attaining its scalar logarithmic cost. Source preparation, conditional reuse, controller cleanup, averaging response and reversible residual costs still need connected accounting. A purely CS application remains a valid route; it needs a task where these resources have an independently meaningful use.

Priority is also unresolved. Checked primary passages already contain the uniform singleton-split ingredient, local/global entropy accounting, circuit design priors, and peak/accumulated erasure resources. [Meila](https://sites.stat.washington.edu/mmp/Papers/compare-jmva-revised.pdf), [Thermodynamics of Modularity](https://arxiv.org/pdf/1708.03030), [Minimal thermodynamic cost of computing with circuits](https://arxiv.org/html/2504.04031v1), [Energy Complexity of Regular Languages](https://arxiv.org/pdf/2204.06025). These antecedents do not establish equivalence of the complete new frontiers. They do prevent novelty from resting merely on entropy accounting or a theorem pair.

The historical [DFT flagship](<LOCAL_PROJECTS>/new-computation-model/research/FLAGSHIP_RESULT.md:375) remains a narrow routed-subclass CANNOT result with no matched optical CAN or binding device adjudication. The corrected Extropic audit is useful evidence accounting, but neither artifact establishes whole-model achievement.

**Strongest decision-changing next question**

> Can one specified finite operational contract preserve the masked advantage against all admitted competitors after charging duration, design precision, source preparation, reuse and reversible overhead?

A sharp conditional calculation is available: for `n≥r`, masked work is `Θ(n2^r/r)`. If every executed gate contributes residual cost ρ, preserving total `O(n)` requires
\[
\rho=O(r2^{-r}).
\]
The next unit should determine whether that shrinking overhead is attainable under fixed, charged constraints—or whether an admitted comparator removes the advantage. Treating ρ as an independently adjustable knob would leave the decisive question unanswered.

The scoped mathematics merits continuation. **Whole-repository admission and mission completion remain unearned.**