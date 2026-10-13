**Corrected UPPER — PASS. Unchanged lower/enclosure — PASS at scope.** No remaining repair or fatal counterexample found.

Accepted scope: `n≥4`, finite integer `r≥0`, independent uniform R, rational `0<δ≤1/16`, and every positive X mass in `[δ,1/8]`. Correctness and cleanup hold pointwise, including zero-probability inputs.

1. **Direct construction and complete cleanup pass.** The [erratum’s transfer](<LOCAL_PROJECTS>/new-computation-model/research/MIN_ATOM_BRANCH_ERRATUM.md:13) maps each `(Rin_i,Rout_i)=(b,0)` to `(0,b)` using two CNOTs. The AND tree uses `n−1` Toffolis; copying its root and reversing the tree clears all scratch. Resetting X then leaves exactly `(AND(X),original R)` in distinct outputs, satisfying the [common contract](<LOCAL_PROJECTS>/new-computation-model/research/STATISTICAL_CONTEXT_MODEL.md:19).

   Exact elementary work and allocated width are
   \[
   G=2r+2(n-1)+1+n=3n+2r-1,\qquad
   W=(n+r)+(1+r)+(n-1)=2n+2r.
   \]
   These formulas cover `r=0` and every finite r. Encoding, program generation and source preparation remain separately charged.

2. **Universal V upper passes, including zero atoms.** Complete states partition original labels. Different original R values cannot reach the same state: deterministic continuation would then contradict pointwise R return. Writing `β=2^{-r}`, every positive predecessor class contains an entire positive original atom, hence has mass at least `βδ`; every successor class has mass at most `β`. Therefore
   \[
   V=\log_2(m_{\rm post}/m_{\rm pre})\le\log_2(1/\delta).
   \]
   Earlier mergers cannot split atoms. Merging zero-mass labels adds no mass, and zero-mass predecessors require no ratio. Reversible gates preserve class masses exactly.

3. **The finite-prior lower passes unchanged.** For the [nine-atom witness](<LOCAL_PROJECTS>/new-computation-model/research/MIN_ATOM_BRANCH_CANDIDATE.md:33), `p₀=δ` and `p₁…₈=(1−δ)/8` satisfy normalization and both atom bounds. Labels `0…8` all have AND zero when `n≥4`. Within each original R value, clean output forces their eventual coalescence. At the δ-atom’s first positive merger, its predecessor mass remains `βδ`; the other predecessor contains at least one whole large atom. Thus
   \[
   V\ge\log_2\frac{1+7\delta}{8\delta}.
   \]
   Extra depth, scratch, zero-mass mergers and prior-aware circuit selection cannot avoid this event.

Consequently,
\[
\boxed{\log_2\frac{1+7\delta}{8\delta}
\le K^*_\delta(n,r)\le\log_2(1/\delta).}
\]
For `δ=2^{-b}`, `b≥4`, this gives `b−3<K^*≤b`, with enclosure width strictly below three bits. It establishes neither an exact frontier nor a b-fractional-bit probability-grid theorem.

**EMPIRICAL support:** independent exact CPU probes passed 16 circuits (`n=4…7`, `r=0…3`), all 3,600 pointwise terminals, 244 rational-prior cases, 254,802 positive branch checks and 1,530 subset checks. The universal verdict rests on the proofs above.

Read-only review completed without private memory, edits, agents, GPU or FULL runs. This is scoped logical proof admission; whole-repository and mission-completion admission remain unearned.