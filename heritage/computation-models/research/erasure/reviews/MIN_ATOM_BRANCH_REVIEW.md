**LOWER PASS. UPPER PASS** for the finite min-atom promise in [MIN_ATOM_BRANCH_CANDIDATE.md](/<LOCAL_PROJECTS>/new-computation-model/research/MIN_ATOM_BRANCH_CANDIDATE.md). Both survive independent mathematical attack.

The accepted scope is \(n\ge4\), finite \(r\), independent uniform context \(R\), rational \(0<\delta\le1/16\), and positive \(X\)-atom masses in \([\delta,1/8]\). Circuits are finite, deterministic, fixed-schedule, and return \((\operatorname{AND}(X),R)\) pointwise with every other register zero, including on zero-probability inputs.

**UPPER — PASS.** At each gate boundary, complete states partition the original input labels. If two labels with different original \(R\) reached the same complete state, their deterministic futures would coincide, contradicting pointwise return. This applies even if the physical context registers have been overwritten or recoded.

Writing \(\beta=2^{-r}\), every positive predecessor class therefore has mass at least \(\beta\delta\), and every successor class has mass at most \(\beta\). Consequently,
\[
V_g=\log_2\frac{m_{\mathrm{post}}}{m_{\mathrm{pre}}}
\le \log_2(1/\delta).
\]
Reversible gates preserve class masses. Previous RESETs cannot invalidate either inequality.

Existence also holds independently of the previously reviewed compiler: compute AND reversibly into \(n-1\) scratch bits, copy it into the output, reverse the computation, then RESET all \(n\) data bits. Leave \(R\) unchanged. This uses \(3n-1\) elementary gates and \(2n+r\) registers, with finite charged program construction. Thus the cap is achievable by one circuit independent of the unknown prior.

**LOWER — PASS.** Choose labels \(0,\ldots,8\), all AND-zero when \(n\ge4\), with
\[
p_0=\delta,\qquad p_1=\cdots=p_8=a=\frac{1-\delta}{8}.
\]
Normalization is exact. Also \(a\le1/8\) and \(a\ge\delta\), since \(\delta\le1/16<1/9\).

Fix an original context label and follow \(X_0\)’s class. Clean output forces all nine positive labels in that context into one final state, so a first merger with another positive label must occur. Before that merger, the tracked class has mass exactly \(\beta\delta\): earlier mergers with zero-mass labels contribute nothing. Deterministic evolution cannot split classes or fractions of original atoms.

The other predecessor contains at least one entire large atom and therefore has mass at least \(\beta a\). The merger must occur at RESET, giving
\[
\frac{m_{\mathrm{post}}}{m_{\mathrm{pre}}}
\ge\frac{\delta+a}{\delta}
=\frac{1+7\delta}{8\delta}.
\]
Extra depth, finite scratch, zero-state mergers, or prior-aware circuit selection cannot avoid this argument. Crucially, the **same explicit prior defeats every circuit** below the bound.

If \(K^*_{\delta}(n,r)\) denotes the infimum universal peak-\(V\) cap over correct finite circuits, the proved enclosure is
\[
\log_2\frac{1+7\delta}{8\delta}
\le K^*_{\delta}(n,r)
\le\log_2(1/\delta).
\]
For \(\delta=2^{-b}\), integer \(b\ge4\),
\[
b-3<\log_2\frac{2^b+7}{8}
\le K^*_{\delta}(n,r)\le b.
\]
The gap is strictly below three bits; this does **not** establish an exact frontier.

One precision distinction matters: \(b\) specifies the **minimum-atom scale**. The nine-atom witness uses probabilities with denominator \(2^{b+3}\); the accepted promise does not restrict every probability to a \(b\)-fractional-bit grid. All descriptions remain finite and charged.

**EMPIRICAL supporting probes:** 12 independently constructed cleanup circuits (\(n=4\ldots7,\ r=0\ldots2\)) passed exact rational checks across four floors, including all zero-mass input labels. They covered 29,904 state transitions and 12,376 positive branch checks; 15,555 subset checks corroborated the first-merge inequality. The universal conclusions rest on the proofs above.

The [fixed-design scalar bridge](/<LOCAL_PROJECTS>/new-computation-model/research/METER_SOURCE_TRANSLATION.md:63) remains candidate-only. These passes establish a logical branch-surprise law, without identifying actual-workload \(V\) with calibrated device heat or granting mission completion. Read-only CPU review completed; no agents, edits, GPU, or FULL execution.