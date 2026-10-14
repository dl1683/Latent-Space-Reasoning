**Verdict: PASS-WITH-LISTED-FIXES.** The quadratic mathematics passes. Its application to E1/E2 needs an explicit bridge contract. No files were created, edited or deleted.

References below use `T = theory/CORRECTION_READY_RESPONSE.md` and `P = experiments/analysis/correction_capsule_exact.py`.

- **PROVED — definitions, retained F/G, resolvent and corrected response** (`T:15–24,51–73`). Energy minimization gives the stated Schur complement. Woodbury with update factor \(CD^{1/2}\) gives the positive correction \(FR_G(D)F^T\). Push-through establishes all three resolvent forms, including singular D/G and noncommuting products. The two-world witness is correct.

- **PROVED-WITH-FIX — legal domain and signed updates** (`T:26–28`). \(0\preceq D\preceq I\) preserves SPD of the corrected world, but finite storage also requires grid membership. **Required edit:** define legality explicitly as \(D_{\rm new}\in\mathcal D_{\rm grid}\cap[0,I]\). For example, \(D_{\rm new}=I/3\) satisfies the matrix inequalities but violates a declared dyadic grid.

- **PROVED — supplied retraction; historical limitation** (`T:29–31`). Subtracting a supplied term restores the corresponding cumulative correction. Cumulative D cannot authenticate named events: histories adding T once versus adding T/2 twice have identical D.

- **PROVED — normalized scalar contract, support restriction, safe Q and certificate** (`T:43–47,77–110`). PSD zero diagonals force zero rows. Principal-block spectral bounds survive in Q. The second-resolvent identity gives
  \(y-p=-x^TEx+(Ex)^TR_G(Ex)\).
  The bounds \(R_G\preceq D\) and \(R_G\preceq(4/7)I\) justify both remainder bounds, including entrywise \(|D|\), and the sharper known-t interval.

- **PROVED — diagonal bound, headroom, mixing witness, realization** (`T:114–146`). The error bound is \((4/7)(1/4)(1/1024)=1/7168\). Both separating responses and the SPD congruence construction are correct.

- **PROVED — acquisition counterexample, admission boundary and interval repair** (`T:150–182`). The displayed fractions check. Transcript-consistent completions separated by \(1/240>2/4096\) preclude a common acceptable answer. Same-query interval intersection preserves containment; midpoint admission is sufficient, without guaranteeing progress.

- **PROVED — outward numerical construction** (`T:184–197`). Q-similarity gives the stated inverse-norm bound. Residual inflation, eta and widened endpoints cover solve/readout error. This proves an enclosure recipe, not a floating-point implementation.

- **PROVED, conditional — rational g, costs, rank-one formula and optimal ambiguity radius** (`T:33–41,201–225`). Derived-entry length g is an assumed source contract, not implied by input precision. Counts are valid for the normalized setting; intermediate bit growth remains additional. Compactness applies to the closed bounded susceptibility family, and bounded rational encodings make possibilities finite.

**Checker:** I ran `python -B experiments/analysis/correction_capsule_exact.py` without `--output`: **12 PASS**, exactly matching `experiments/results/analysis/correction_capsule_exact.json`. It verifies finite witnesses only. In-memory probes produced PASS for zero cases, altered `expected_outputs`, and \(G=-I\) in the D=0 fixture. Expected outputs and general domain premises are not validated (`P:249–280,729–768`). It can therefore pass while those premises—or an untested universal claim—are false.

**UNSUPPORTED — transfer to E1/E2** (`T:227–230`; `experiments/cloud/RESEARCH_QUEUE.md:35–47`). **Required edit:** specify the destination state/actions/readout and either prove an operation-preserving quadratic-to-dependency map, or explicitly make E1/E2 independent graph constructions with their own correctness contract. A learned quadratic constructor is the direct alternative.

I formed these conclusions before reading `internal/analysis_2026_10_14/codex_proof_retry_audit.md`. I agree with its algebra; my additional concerns are grid legality, checker coverage and bridge specification. The cited [Kron-update](https://motion.me.ucsb.edu/pdf/2011d-db.pdf) and [PSD-sketch](https://www.cs.cmu.edu/afs/cs/user/dwoodruf/www/ackqwz16.pdf) attributions match their stated scopes.