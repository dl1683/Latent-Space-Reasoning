# Matrix-Native Arithmetic: A Research Monograph

*The complete research content of the matrix-native-math program: every result, every proof with its reasoning, every attempted route that failed and why, every measurement, and every open problem with its full context. Written to be evaluated.*

---

## Errata (2026-09-28; E11–E14 added 2026-09-29) — read before evaluating

An independent outside audit (`research/external_review_2026_09/CHATGPT_SESSION_2026_09_19_to_28.md`) plus an in-repo check corrected the following. Items marked VERIFIED were re-derived or recomputed in this repository; PENDING items are unverified claims that should not be cited as results.

- **E1 (VERIFIED, logic).** r_pf = 49 does not imply rank(MM₄) = 48; only rank(MM₄) ≤ 48 is known (no lower bound near 48 exists). The "locked together" claim and the ladder's "gap exactly 1 at rung 2" are unsupported. r_pf is a question about Kronecker-refined rank, i.e. the six-party tensor square ⟨2,2,2⟩⊠⟨2,2,2⟩ — the Draisma–Saptharishi / Christandl–Jensen–Zuiddam tensor-product rank-multiplicativity question (arXiv:1705.09379), which this record had not cited.
- **E2 (VERIFIED).** The registry theorem's "exact descriptors are lists" is false as a storage statement (affine counterexample in §5.1). The word is an O(k log m)-bit exact descriptor.
- **E3 (VERIFIED).** Polynomial growth ⇔ virtually nilpotent, not virtually solvable (the gated source `research/COUNTING_THEOREM_V1.md` had this right).
- **E4 (VERIFIED).** DPS48 was 65% slower than Strassen at n = 4096 (not 39%); the printed crossover evaluates to ≈ 4.0×10³ (not 2.6×10³).
- **E5 (DERIVED, unreviewed).** For compact K ≤ U(n) with B = Φ_K(Q) (Hilbert–Schmidt projection onto the commutant): E_K S = 2‖Q − B‖²_F and E_K S ≤ sup_K S ≤ 2 E_K S. Hence the cliff constant improves from 12C to 6C (Σ(r_b³ − r_b) ≤ 6C), and the isotypic floor improves to E S ≥ (1/6)Σ_b m_b(d_b³ − d_b) (84, not 7, for d = 8). The cocycle route of §3.3 is unnecessary in finite dimension. Not yet through the independent proof gate.
- **E6 (VERIFIED by construction; not yet gated).** (i) Aperiodic single chain (§3.4, 9.6(iii)): the increment sequence T_{a1},T_{a1},T_{a2},T_{a2},… (adjacent swaps, non-periodic index sequence) has S ≤ 2 for all partial products while generating S_n — the unrestricted "forces near-MASA" statement is false. (ii) Non-unitary cliff (9.6(i)): A_n = ½I + J_n/(4n), B_n = ½I + J_nᵀ/(4n) have every positive-word spread ≤ 1 yet generate all of M_n — the unnormalized contraction analogue fails. (iii) The snake example (§3.4, E.4) cannot be an all-positive-words statement: for a finite permutation σ⁻¹ = σ^{ord−1}, so positive words generate the group. Read it as a chain-level example.
- **E7 (VERIFIED by code reading and derivation; no new timing run; see also `research/external_review_2026_09/VERIFICATION_2026_09_28.md` for the independently verified 17/3 staged construction, which is an instance of Beniamini–Schwartz sparse decomposition).** The 960-vs-172 per-node "obstruction" (§7.1) is an implementation artifact, not a cost of the scheme. `code/dps48_race_final.py:43-55` applies the *composed* maps U·P, V·Q, DEC·W (448/288/336 nonzeros → 400+240+320 = 960 fma) inside every recursive node. An alternative-basis algorithm applies P^{⊗d}, Q^{⊗d}, DEC^{⊗d} once on the whole matrix at O(n² log n) total, with core recursion B(d) = 48·B(d−1) + 172·16^{d−1}, because (U⊗U)(P⊗P) = (UP)⊗(UP); nothing requires P to be Kronecker-decomposable across the internal 2×2 split, so the bilocated ranks 13/15/14 are irrelevant at 4×4 recursion granularity. The repo's own `research/DPS48_RETAINED_RECONCILIATION_V1_COST_SCHEDULE.md:7-28` already states this accounting. An op-count model (not wall time) puts DPS48-with-transform-once ahead of Winograd-15 beyond n ≈ 10⁴ (0.96× at n = 2¹⁴) and roughly at parity with a Karstadt–Schwartz Strassen model (1.01×) at 2¹⁴, versus 1.4× *behind* under the per-node model. `DPS48_RACE_V1_DISPOSITION.md`'s PRACTICAL_NEGATIVE therefore describes a suboptimal implementation only. Memory traffic (48 live buffers), stability under a once-applied transform, and non-power-of-4 sizes are unmeasured. The product-form question (r_pf) is not needed for 4×4-granularity recursion; it matters only for radix-2 composition.
- **E8 (PENDING).** (a) §8.3 at ε = 0 the family {A₀+εE₁, A₀+εE₂} is one generator, so span ≤ k+1 — the reported curve needs the setup (two distinct clean generators?) checked against the code. (b) §8.4: exp(iM/‖M‖_F) is unitary only for Hermitian M; west0167 is nonsymmetric — preprocessing and ‖U*U − I‖ must be recorded. (c) Random-integer exactness is not an identity certificate; check all 16×16 basis pairs (4,096 coefficient equations).
- **E9 (VERIFIED, weak).** A generalized Koszul flattening (3-dim subspace {I,Z,X} of one leg; 12×12 skew block matrix, rank 12, det −16; pure terms map to rank ≤ 2; Kronecker square rank 144, per-term ≤ 4) gives r_pf ≥ 36, beating the AS bound 31. It is the square of a bound of 6 that lies below border rank 7, so it cannot approach 49 and says nothing about 48 vs 49. Reconstructed independently in `research/external_review_2026_09/verification_code/t4_proofs/t4_flat.py`.
- **E10 (VERIFIED).** §3.8/E.3: the delocalized-torus spread is the torus-Haar *average*, not a contribution present at every element (S(identity) = 0). Exact criterion for a multiplicity-free torus: E S = 2Σ_j(⟨v_j,Q²v_j⟩ − ⟨v_j,Qv_j⟩²).
- **E11 (EMPIRICAL re-measurement, 2026-09-29; independent evidence gate pending).** The "workload pipeline v2: 2.8–14× over 24-thread BLAS" (unit `unconstructed_workload_v2`, `code/compiled_pipeline.py`; ledger rows 1162–1166) is a win over a chain of dense GEMMs only. The timed kernel `diag_extract_axis_fft` is a hand-written direct DFT summation (four nested loops, O(N²) per member with sin/cos in the inner loop) that returns only the N eigenvalues; it is timed against `P = P @ members[k]` (T−1 dense O(N³) GEMMs producing the full N×N product), while the same script's correctness reference `np.fft.fft2` (O(N log N) per member) is the right specialist and is faster. Single thread (`code/record_corrections_2_checks.py pipeline`, `results/record_corrections_2/pipeline.json`; n_side = 16/32/32, T = 16/16/64): dense chain 19 ms / 1.12 s / 4.73 s; compiled kernel 18 ms / 0.315 s / 1.30 s (only 1.05–3.6× the dense chain single-threaded; the historical figure compared a multi-threaded kernel with 24-thread BLAS); `np.fft.fft2` pointwise product 0.33 ms / 0.60 ms / 2.1 ms, i.e. 56× / 526× / 616× faster than the compiled kernel and 59× / 1870× / 2240× faster than the dense chain; materialising the full N×N product from the FFT eigenvalues (ifft2 + circulant gather) is still 18× / 150× / 451× faster than the dense chain (all agree to ≤ 2.4×10⁻¹³). Even a perfect 24× parallel speed-up of the compiled kernel would leave the single-thread FFT ≥ 2.3× (n_side = 16) and ≥ 22× (n_side = 32) faster. **Correct label:** the win is over dense GEMM chains only; the right specialist is an FFT and is faster; the structure (translation invariance of a periodic Kronecker-sum Laplacian, closed-form eigenvalues) is domain-obvious, as the unit's own gate already flagged. The unit's cold-detection results (centralizer dimensions 454/196/340; 64/64 translations) are unaffected.
- **E12 (EMPIRICAL re-measurement, 2026-09-29; independent evidence gate pending).** The Wedderburn instrument's "fair advantage 1.9–5× from m ≥ 24" and "measured speedup over the strongest baseline" (`research/WEDDERBURN_INSTRUMENT_V1_RESULT.md`; ledger rows 1146–1148) are relative to a dense real-GEMM chain. The specialist for a block-circulant family is an FFT along the block axis of each member's first block row plus one batched (m, nb, nb) BLAS matmul per composition. Single thread, N = 1536, T = 16 (`code/record_corrections_2_checks.py wedderburn M NB T`), m×nb = 24×64 / 32×48 / 48×32: dense chain 3.79 / 3.34 / 3.37 s; numba-fused emitter 0.252 / 0.118 / 0.0805 s (complex128 cast kept off the clock as in the frozen protocol; the cast itself costs 0.12–0.16 s); specialist 0.0765 / 0.0418 / 0.0259 s, i.e. **3.3× / 2.8× / 3.1× faster than the emitter** (5.4–9.3× if the cast is charged) and 50–130× faster than the dense chain; agreement with the dense product 10⁻¹⁴–2×10⁻¹³. This corroborates the fresh-eyes lens rerun (3–3.9×). **Correct label:** detection of hidden classical block-circulant structure plus an emitter that beats a dense GEMM chain; the strongest baseline was not raced and beats the emitter by about 3×. Reproducibility: line 19 of the hash-bound `code/fused_emitter.py` (`members[k].dtype != np.complex128` inside `@njit`) does not type under numba 0.62.1; the check used a copy with that line replaced by `M = members[k]`; the bound file is untouched. Detection results are unaffected.
- **E13 (EMPIRICAL re-measurement + code reading, 2026-09-29; independent evidence gate pending).** The labeling-recovery withdrawal's stated reason — "joint diagonalization beats/dominates recovery+FFT on materialized products at n ≤ 384" (§7.2, G.3, cemetery item 5; ledger row 1217) — compared unlike work. `code/labeling_cost_table.py` materialises each joint-diagonalization product with a dense complex GEMM `Vc @ (d[:,None]*Vc.conj().T)` (O(n³); lines 272, 278 of the pre-annotation file) but each FFT-pipeline product with `decode()` (O(n²): n `np.roll` calls and one fancy-index scatter), and it pins no BLAS threads (the thread count of the recorded run is not logged, but the recorded 3.63 ms joint-diagonalization and 12.63 ms BLAS-chain times at n = 384 correspond to about 125 and 143 GFLOP/s of complex GEMM, i.e. 3.4–3.9× the 37 GFLOP/s one thread reaches here and above the ≈ 88 GFLOP/s AVX2 single-core ceiling, so the recorded GEMMs were multi-threaded — an inference — while the decode is single-threaded NumPy). Single thread, decode / GEMM in ms: 0.49/0.16 (n = 96), 1.34/1.72 (192), 5.0/12.3 (384), 19.7/94 (768), 77.7/819 (1536); GEMM÷decode = 0.32, 1.28, 2.4, 4.8, 10.5, with local exponents ≈ 1.4–2.0 against 2.8–3.4 (`code/record_corrections_2_checks.py labeling`). So the result is a small-n constant-factor observation (single-thread crossover n ≈ 150); asymptotically the cyclic/FFT route wins on materialized products and is classical (circulant recognition after Muzychuk / Evdokimov–Ponomarenko, then FFT arithmetic). In the diagonal domain both routes cost O(n) per step, the FFT route storing O(n) against O(n²) for the eigenbasis; both setups are O(n³) as implemented (`eigh`). **Consequence:** the Middle-tier withdrawal stands, but its reason is that the pipeline is classical recognition + FFT (no new representation class; the strongest structure-aware baseline is the script's own "oracle", which it can at best tie), not that joint diagonalization dominates; "joint diagonalization is the right tool for one-shot materialized products" (`research/MARKET_QUESTION_STATE_V1.md`) holds only below n ≈ 150 single-thread. Not measured: the crossover against a multi-thread GEMM.
- **E14 (FIXED 2026-09-29, commit dc7434c8).** AGENTS.md §7.1 stated "Semisimple subalgebras inherit ω — no route to faster matmul through structure". The theorem (`research/SUBALGEBRA_LANDSCAPE_V1.md`) is exponent-only: it excludes a better *exponent* inside semisimple structure and says nothing against constant-factor, log-factor or asymptotic speed-ups for structured families (circulants/FFT, block-diagonal algebras, the specialist algorithms of E11–E12), which this record itself measures. AGENTS.md §7.1 now says so; the Scope notes below already carried the correct statement. Frozen checkpoints (`PROGRAM_SELF_REFLECTION_2026_09_19_V9`–`V12`, `2026_09_20_V10`), ledger rows 1138–1148, 1162–1166, 1216–1217 and `WORKLOG.md` still quote the uncorrected numbers and are left as frozen receipts; read them with E11–E13.
- **Scope notes.** Spread S is unnormalized (⊕ of n/2 adjacent swaps has bandwidth 1 but S = n) and is a position statistic, not an arithmetic-hardness measure (the reversal permutation has cubic S and linear-time arithmetic). The five-field "unification" is a research program, not a theorem. The semisimple-inheritance obstruction excludes only *exponent* improvement inside semisimple structure; diagonal algebras and bounded blocks are semisimple and cheap.

---

## Chapter 0. What this is and how to evaluate it

The program's goal: find a representation of linear algebra in which the matrix — not the scalar — is the atomic unit, and in which matrix computation is **fundamentally** cheaper than in the scalar-derived formalism. Success was graded on three tiers from the start: **(Floor)** a unifying theory with theorems about which matrix classes admit composition-friendly compressed encodings and which provably do not; **(Middle)** a new representation with a *provable* speedup for a useful family, in running code; **(Ceiling)** progress on the exponent of matrix multiplication reached from the representation side.

At the time of writing: the Floor tier is substantially delivered but contested (see the Errata below) — a gated theory with laws on three independent axes and exact constants. The Middle tier was pursued through two full constructive programs (a certified sub-classical multiplication circuit, and a hidden-structure recovery pipeline); both are recorded here with their complete cost accounting, and neither currently supports a Middle-tier claim. The Ceiling is untouched directly. The program's sharpest open problem (r_pf = 49?) is a question about *Kronecker-refined* rank; it does NOT determine the ordinary rank of 4×4 multiplication (Erratum E1).

**What a reviewer should evaluate:** Chapters 3–5 contain the theory (spread-energy composition law; telescoping-rank classification; word-growth counting). Chapter 6 is the atlas of obstructions — the walls, with their mechanisms. Chapter 7 is the constructive record with honest costs. Chapter 8 is everything measured on real computational data. Chapter 9 states every open problem with everything the program knows about it. Chapter 10 is the research cemetery: every route that was tried, how far it got, and what broke — this chapter exists because a negative result with a mechanism is a map; one without is a grave.

**Rigor conventions.** Every theorem statement carries its scope. Where a proof is short it is given in full. Where it is long, the proof mechanism is given step-by-step with the load-bearing lemmas stated precisely. Numerical claims state their protocols. No claim anywhere is labeled proved that did not survive independent adversarial review; where such review changed or killed a claim, the original claim and the reason it died are recorded.

---

## Chapter 1. The problem landscape

### 1.1 The motivating observation

Arithmetic as standardly formalized is scalar-atomic: the number line is generated by 1; operations are defined by their action on units; matrices are arrays of scalars. Yet two adjacent layers of the computational stack have already abandoned this: in quantum mechanics observables are operators (matrices) and scalars emerge only at measurement; in modern accelerators the dense-compute primitive is a matrix product (a TPU v4 core performs 128×128 matrix multiplies as single systolic instructions). The mismatch is real at the hardware level: a modern GPU/TPU achieves a large fraction of peak only on large dense matrix products, while the surrounding software stack spends most of its effort marshaling scalars into that form.

The bet is not that scalar arithmetic is wrong, but that it is a *choice* — one coordinate system among several — and that other choices may expose structure that the scalar view hides. The concrete question: **is there a representation of matrices (a choice of "digits" and "operations") in which the operations that dominate real computation — products, powers, repeated composition — are asymptotically cheaper, with the conversion costs honestly counted?**

### 1.2 Why this is not immediately hopeless, and not immediately hopeful

Known structured representations already prove that representation change can buy composition speedups: the Fourier basis turns convolution-powers into pointwise operations (O(n log n) vs O(n²)); displacement structure turns Toeplitz products into low-rank updates; hierarchical formats turn elliptic-operator arithmetic into O(n log n) or O(n) flows. These are the program's "civilizations" — each a full arithmetic built around a non-scalar atomic structure.

But each known civilization covers a family that is visibly special (commutative, or algebra-generated, or geometrically localized). The program's central negative results (Chapters 3, 5, 6) say, in effect: the known cheap families are cheap *for law-like reasons*, and the laws have boundaries that are now theorems. The open question is whether any *useful* family outside the known civilizations admits a cheaper arithmetic — and the program has mapped exactly where such a family can and cannot live.

### 1.3 The three questions

Everything the program did organizes under three questions, which became its three axes:

1. **(Registry)** If I compose elements of a family F repeatedly, how does the number of *distinct* products grow, and what is the minimum data structure representing them all? — Exact composition bookkeeping.
2. **(Walk)** If I multiply matrices together, what happens to their *position structure* (bandedness, locality relative to a fixed basis)? — Approximate composition geometry.
3. **(Self-similarity)** When does a family's natural basis factor across scales (Kronecker-style), so that arithmetic composes level-by-level like the FFT? — Telescoping structure.

The axes are independent (a family can be cheap on one and expensive on another), and every classical fast algorithm is a cheap point on at least one axis with the other two honestly priced. This trichotomy is itself a program result: it was discovered by autopsy — by classifying why each failed attempt failed (Chapter 10).

---

## Chapter 2. The framework in full

### 2.1 The registry axis

**Objects.** For a family F of n×n matrices, the *registry* is the algebra ⟨F⟩ it generates. For k ≥ 0 let W_k be the set of words of length exactly k, and D(k) the minimum size of a data structure from which every element of ⟨F⟩ of word-length ≤ k can be recovered exactly. D(k) is the *descriptor cost*.

The central registry question: when is D(k) subexponential? The counting theorem (Chapter 5.1) answers: descriptor *information* is the log of the word count; whether a cheap descriptor with cheap composition exists is not settled by counting (Erratum E2). This is the axis on which "compressed exact arithmetic" lives or dies.

### 2.2 The walk axis

**Objects.** Fix the standard basis e₀,…,e_{n−1} of ℂⁿ and the *position operator* Q = diag(0,1,…,n−1). For a matrix u define the **spread energy**

  S(u) = ‖[Q,u]‖²_F = Σ_{i,j} (i−j)² |u_{ij}|².

S is zero exactly on diagonal matrices; a unitary with bandwidth b has S ≤ n·b²·(mass); we will see that a Haar-typical unitary has S concentrated at exactly n(n²−1)/6. Associated to each unitary g is the **displacement cocycle** b(g) = g*Qg − Q, which satisfies the cocycle identity b(gh) = h*b(g)h + b(h) on the conjugation representation.

**Why this functional.** S measures the failure of a matrix to commute with position — i.e., how much a matrix "moves" mass across the index line under conjugation. It is the finite, basis-relative shadow of the commutator-with-localization that controls: Lieb–Robinson velocities in quantum lattices (group velocities of operators); the elliptic/hyperbolic split in hierarchical-matrix arithmetic (operators with bounded walk can be compressed hierarchically, ballistic ones cannot); the forward/reverse-mode split in automatic differentiation of long chains; and positional carry propagation in digit arithmetic. The program's unification result: all four are the same phenomenon at different scales — a spread-energy law (Chapter 3).

The walk axis's central question: **can a family's products keep S uniformly bounded (or geometrically tailed) in n, without the family being trivially decomposable?** The dichotomy theorem (Chapter 3.7) answers no, with an exact mixing constant.

### 2.3 The self-similarity axis

**Objects.** A basis {B_N} of ℂ^N "telescopes" at a split N = MN if B_{MN} = B_M ⊗ B_N (possibly permuted). FFT (B_N = Fourier, split-any), wavelet cascades, and Strassen-type bilinear recursions all live on this axis: their O(N log N) or O(N^{log₂7}) costs are realizations of telescoping. The formal object the program isolated: for the 4×4 matrix-multiplication tensor with each 16-dimensional leg split 4⊗4 (the block split), a bilinear scheme is **product-form** if every linear form in it is a pure tensor across the split; **r_pf** is the minimum number of scalar multiplications over product-form schemes. This is the refined tensor rank — the CP rank of the tensor viewed as 6-way with 4-dimensional legs.

The axis's central question: how much does telescoping structure save, and where does it exist? The separable-schemes theorem and the classification (Chapter 4) answer: at aligned splits, nothing (naive forced); on the four-member L-family of transversal splits, exactly one recursion's worth (49 vs 64), and conjecturally not one multiplication more.

### 2.4 The landmark map

| Family | Registry | Walk | Self-similarity | Verdict |
|---|---|---|---|---|
| Circulants / diagonalized-by-Fourier | cheap (commutative, dim n) | cheap (S≡0 in Fourier basis) | yes (FFT telescopes) | the model civilization |
| Toeplitz | **full** (generates everything) | structured via displacement | partial | superfast solvers exist *because* the other two axes carry it |
| Banded families | cheap (finite-band exact) | bounded if bandwidth fixed | no | banded arithmetic |
| H-matrix, elliptic regime | full | bounded | hierarchical | O(n log n) arithmetic exists |
| H-matrix, transport regime | full | ballistic | — | no compression (law, not accident: Ch. 3) |
| Generic pair | full | mixing at the universal rate | no | BLAS is optimal-honest |
| Heisenberg/Zak | cheap (nilpotent growth) | expensive | — | registry-only civilization |
| Strassen 2×2, lifted | — | — | the L-family split | the only known transversal telescoping |
| DPS48 rank-48 scheme | excellent (48 products) | — | **broken** (basis doesn't telescope) | Ch. 7.1 |

---

## Chapter 3. The Walk Law

This chapter is the program's core: a complete composition law for position structure under matrix multiplication. It exists in four pieces — the cliff (§3.3), the product-to-group transfer (§3.4), the tail-to-spread transfer (§3.5), and the forced-mixing converse (§3.6) — assembled into a dichotomy in §3.7. Each section states the result, gives the proof (in full where short), and records the routes that failed on the way to it.

### 3.1 The empirical law that started it

Before any theorem, five numerically observed regimes suggested one law:

1. **Quantum lattices:** the spread of local Hamiltonian evolutions grows ballistically (Lieb–Robinson); of dissipative/gapped evolution, diffusively.
2. **Iterative vs direct solvers:** operator powers (resolvents) of banded matrices fill in at a forced linear width rate (Chapter 5.3, Theorem B) — iterative composition cannot stay banded.
3. **Automatic differentiation:** forward-mode Jacobian-vector chains preserve structure; reverse-mode chains (adjoints) accumulate displacement — the walk distinguishes them.
4. **Hierarchical matrices:** elliptic-problem operators admit bounded-walk arithmetic; transport-problem operators force ballistic walks — the admissibility condition of H-matrices is precisely a walk bound.
5. **Positional carries:** carry propagation in base-q digit arithmetic is a displacement phenomenon with the same tail/forcing structure.

The unification conjecture: *position structure under repeated composition either persists because the family is (up to bounded noise) decomposable, or is destroyed at a forced rate; nothing in between persists.* The rest of this chapter is the theorem.

### 3.2 Basic properties of spread energy

Directly from the definition:

- S(u) = 0 ⟺ u diagonal.
- Unitaries: S(g) = ‖b(g)‖²_F exactly (the cocycle norm).
- Subadditivity structure: S(gh) ≤ 2S(g) + 2S(h)·‖g‖² (from the cocycle identity and the triangle inequality) — so bounded families can have unbounded products only through coherent accumulation.
- Expected value for Haar U(n): E S = (1/n)Σ_{i,j}(i−j)² = n(n²−1)/6 (a single Schur-orthogonality computation; its universality is Theorem M1 below).
- Extremes: the reversal permutation has S = (n³−n)/3 ≈ 2× the Haar mean; the full shift has S = n(n²−1)/... computed: for the cyclic shift σ(i) = i+1 mod n, S = Σ_i (i − (i+1))² over i < n−1 plus the wrap term = (n−1)·1 + (n−1)² = n(n−1); i.e., S(shift) = n(n−1) — linear, not cubic, because mass moves one step. The *banded* unitaries are exactly the low-S ones.

Two artifacts discovered while measuring these (recorded to save successors time): sampling a chain only at even indices can show S ≡ 0 spuriously (a period-2 chain whose odd elements are the identity); and clustering eigenvalues at a fixed tolerance conflates exact eigenvalue-1 with near-unit contractive values, producing phantom pairings — use exact least-squares Fredholm residual tests instead.

### 3.3 The group cliff

**Theorem 3.3 (composition cliff; group form).** Let G be a group of unitaries on ℓ²({0,…,n−1}) with sup_{g∈G} S(g) ≤ C. Then:

(i) there is a self-adjoint Hilbert–Schmidt A with ‖A‖_F ≤ √C such that [Q − A, g] = 0 for all g ∈ G;
(ii) writing r₁, r₂, … for the dimensions of the eigenspaces of Q − A, the nontrivial blocks satisfy Σ_{r_b ≥ 2} (r_b³ − r_b) ≤ 12C and Σ_{r_b ≥ 2} (r_b − 1) ≤ 2C;
(iii) the algebra generated by G has dimension ≤ n + f(C) with f(C) = O(C), **independent of n**: a MASA plus a non-abelian core of bounded total excess.

The constant 12 is tight within a factor of 4.

**Proof.**

*Step 1 (cocycle to coboundary).* Consider the real Hilbert space HS₀ of self-adjoint Hilbert–Schmidt operators with the Frobenius inner product. The conjugation representation g·X = gXg* acts unitarily, and b(g) = g*Qg − Q is a 1-cocycle: b(gh) = h*b(g)h + b(h). By hypothesis ‖b(g)‖_F = S(g)^{1/2} ≤ √C for all g. The Johnson–Bounded-cocycle theorem (a bounded cocycle of a unitary representation on a Hilbert space is a coboundary — via the center of the minimal ball of the cocycle's orbit) supplies a self-adjoint A ∈ HS₀ with b(g) = g*Ag − A for all g and ‖A‖_F ≤ √C (the center bound: b(e) = 0 lies in the orbit's convex hull). Rearranged: [Q − A, g] = 0 for every g ∈ G.

*Step 2 (degeneracy budget).* Q − A is self-adjoint with ‖A‖_F ≤ √C. Hoffman–Wielandt: pairing the sorted spectra, Σᵢ (λᵢ(Q − A) − λᵢ(Q))² ≤ ‖A‖²_F ≤ C. Q's spectrum is {0, 1, …, n−1}, simple with unit gaps. If an eigenspace of Q − A has dimension r, its r equal eigenvalues must pair (in sorted order) with r *consecutive* integers (simplicity forces order-preserving pairing), and r equal values against consecutive integers minimize total squared deviation when placed at their mean: the minimum is Σ_{j=0}^{r−1} (j − (r−1)/2)² = r(r²−1)/12. Summing over nontrivial blocks (deviations in different blocks add, being disjoint index sets): Σ_{r_b≥2} r_b(r_b²−1)/12 ≤ C, i.e. Σ(r_b³ − r_b) ≤ 12C. The excess bound follows since r³ − r ≥ 6(r−1) for r ≥ 2 with equality at r = 2.

*Step 3 (structure).* Every g commutes with Q − A, so G lies in the direct product of the unitary groups of the eigenspaces, acting trivially across. On one-dimensional eigenspaces the action is by characters — commuting, MASA-like. All non-abelian behavior lives on ⊕_{r_b ≥ 2} E_b, whose total dimension is Σ r_b ≤ 2C + (number of blocks) ≤ 4C by Step 2. The generated algebra has dimension at most n + Σ_{r_b≥2} r_b². ∎

**Sharpness.** Example families show 12C cannot fall below 3C: the block-diagonal family ⊕ U(2) on n/2 adjacent pairs has S ≤ 2·(mass per block)·1² per block, total S = Θ(n), while Σ(r³ − r) = n/2 · 6 = 3n — the ratio Σ(r³−r)/S is 3 asymptotically; the remaining factor-4 slack in the theorem is the gap between the Hoffman–Wielandt step and the true optimal pairing in extremal cases.

**What fails without unitarity.** Step 1 uses the conjugation representation's unitarity. For general contractions the cocycle can be bounded without being a coboundary (the orbit need not lie in a ball around a fixed center in the right way), and indeed nonunitary bounded-spread semigroups exist whose closure is not of cliff form — the non-unitary version of the cliff is open (Chapter 9.6).

### 3.4 From products to groups, and the chain forms

The cliff is a statement about groups. The operational question concerns *products* (positive words). The gap is real: there are families whose positive words all have bounded S while the generated group contains elements of enormous S. Recorded counterexample (from an early review of the draft): a snake-loop permutation family with sup over positive words S ≈ 1,594 whose group contains an element with S ≈ 21,333,200. So a product-to-group transfer needs hypotheses.

**Proposition 3.4a (all-words transfer).** Let F be a finite unitary family with sup over *all* positive words w of S(w) ≤ C. Then S(g) ≤ C for every g in the group ⟨F⟩, with the same constant, and the cliff applies.

**Proof.** In HS₀: the orbit O = {δ_w = w*Qw − Q : w ∈ F⁺ ∪ {e}} lies in the ball of radius √C. For y ∈ F⁺ define T_y(X) = y*Xy + δ_y. The cocycle identity gives exactly T_y(δ_w) = δ_{wy}, so T_y maps O into O; each T_y is an isometry of HS₀ (X ↦ y*Xy is unitary on HS₀; translation by δ_y). Hence T_y maps the closed convex hull K of O into itself. At finite n, HS₀ is finite-dimensional, so K is compact. An isometric self-embedding of a compact metric space is surjective (total boundedness: an isometry preserves the separation δ between K and T_y(K); iterating gives infinitely many disjoint δ-balls in a totally bounded space unless the gap is zero). Hence T_y is onto, so T_y⁻¹ preserves K; T_y⁻¹ = T_{y⁻¹} in the affine sense, and iterating, every group word's cocycle δ_g = T_g(δ_e) ∈ K ⊆ ball(√C). ∎

**Failed route (recorded).** The first attempted proof used a "lag-1 factorization": write a group word h with an inverse at the end as h = y⁻¹(yh) and induct. This is invalid: yh need not be a positive word (for h itself containing inverses, e.g. commutators UVU⁻¹V⁻¹, the intermediate factors leave the semigroup). The convex-hull isometry argument above is the repair; it appeared during adversarial review of the draft.

**Partial result for permutation unitaries.** For families of permutation matrices, a chain-versus-group gap of at most polynomial type exists: if products have S ≤ C then group elements have S ≤ C³, via integer-entry displacement counting (each group word is a product of at most C/2 "span units"; the mass-transport interpretation of S for permutations). No analogue is known for general unitaries; the gap is the reason the chain-form theorem below needs its own hypothesis.

**Theorem 3.4b (periodic chain form).** Let (u₁, u₂, …) be periodic with period p, and suppose the partial products P_j = u_j⋯u₁ satisfy sup_j S(P_j) ≤ C. Then the period product W = u_p⋯u₁ has S(W) ≤ C, and the family {W-generator structure} is near-MASA: the displacement of W satisfies a phase-resonance budget — the Cesàro averages of the intermediate displacements satisfy Σ_{j=1}^{p} ‖Σ_{i≤j} δ_{u_i}‖²_F/(p·C) ≤ 12 — forcing W into the cliff with the period's phases aligned. (Gated; the exact constant bookkeeping in the research file.)

**Open (aperiodic single chain).** Whether sup_j S(P_j) ≤ C along one aperiodic sequence forces near-MASA increments is open. A genuinely aperiodic vacuous walk exists (verified): concatenated returning blocks of varying lengths with increments confined to {I, U, U²} for U of order 3 never become globally periodic while all partial products stay in a bounded set — so the aperiodic case is not vacuous, but the structural conclusion is not proven either. The typical-chain version (i.i.d. increments; boundedness in probability of S(P_j)) is a labeled conjecture; deterministic-rotation examples (geometric sums ΣRᵏξ) show transported-increment chains can stay bounded with nonvanishing increments, so the random-recursion toolbox must be used with care.

### 3.5 From geometric tails to bounded spread: the transfer chain

The operational hypothesis in applications is not bounded S but geometric decay of off-band mass. The bridge:

**Theorem 3.5 (transfer chain).** Suppose every product u of the family satisfies

  tail(L) := ‖u·1(|i−j|>L)‖²_F ≤ C₀·a^L for all L ≥ 0,

with a < 1 and C₀, a independent of the word, its length, and n. Then S(u) ≤ C₀(1+a)/(1−a)² =: C₁ for every product, **independent of n**, and the cliff applies with core ≤ 12·C₁.

**Proof (the tail-sum identity).** Let m_d = Σ_{|i−j|=d} |u_{ij}|² be the shell mass at distance d, so tail(L) = Σ_{d>L} m_d. The key identity: for each entry with |i−j| = d,

  (i−j)² = d² = Σ_{k=1}^{d} (2k−1)  (sum of the first d odd numbers).

Summing over entries and exchanging the order of summation: shell d contributes m_d to tail(k−1) for every k ≤ d, hence

  S(u) = Σ_{i,j} (i−j)²|u_{ij}|² = Σ_{k≥1} (2k−1) Σ_{d≥k} m_d = Σ_{k≥1} (2k−1)·tail(k−1).

Now apply the hypothesis at L = k−1 (every instance L ≥ 0 is used; see below):

  S(u) ≤ C₀ Σ_{k≥1} (2k−1) a^{k−1}.

The sum evaluates in closed form: Σ_{k≥1} a^{k−1} = 1/(1−a) and Σ_{k≥1} k·a^{k−1} = 1/(1−a)², so Σ (2k−1)a^{k−1} = 2/(1−a)² − 1/(1−a) = (1+a)/(1−a)². ∎

**Tightness.** Take shell masses m_d = (1−a)a^d (d ≥ 1): then tail(L) = a^{L+1}, so C₀ = a works, and S = Σ d²(1−a)a^d = a(1+a)/(1−a)² = C₁ *exactly*. The constant cannot improve for this hypothesis form.

**Why the L = 0 instance is load-bearing.** The disjoint product of adjacent transpositions D_n = T₀T₂T₄⋯ has tail(L) = 0 for all L ≥ 1 (all entries at distance ≤ 1) yet S(D_n) = n: its C₀ at L = 0 is ‖D_n·1(|i−j|>0)‖² = n, not uniform. Without uniformity at L = 0 the theorem is empty for exactly the families that "hide" linear spread at distance 1. Conversely the theorem's shell-accounting shows distance-0 mass never enters S at all: diagonal mass, however large, is invisible to spread.

**Numerical verification.** The identity S = Σ(2k−1)tail(k−1) verified to relative error 2.4×10⁻¹⁶ on random Givens-word products (n = 256); the bound verified across n = 32…4096 with S ∈ [0.63, 0.76] against C₁ = 2.37 at the test parameters; the tightness example attains equality to machine precision.

**History of the proof (what was tried first).** The first draft bounded S by splitting at a fixed bandwidth L*: S ≤ L*²·‖u‖²_F + Σ_{d>L*} d²·(tail mass). This carries a spurious factor n (from ‖u‖²_F = n for unitaries) unless L* is chosen adapting to n, and the bookkeeping collapses to O(n) at best. The repair — the tail-sum identity — eliminates the split entirely by using *every* hypothesis instance L = 0, 1, 2, … as the shell cap. The repair was found in adversarial review; the draft's error is recorded here because it is the natural first attempt and it silently loses exactly the n-independence that makes the theorem valuable.

### 3.6 The converse: forced mixing

The cliff says bounded spread ⟹ decomposable structure. The converse direction: if the family is *not* decomposable, spread must grow. The measure of non-decomposition is isotypic.

**Setup.** For a compact group K ≤ U(n) (any closed subgroup), the representation of K on ℂⁿ decomposes as ⊕_b (V_b ⊗ M_b) with pairwise inequivalent irreducibles π_b of dimension d_b = dim V_b and multiplicity m_b = dim M_b. Define the **non-abelian isotypic support** Ω(K) = Σ_{d_b ≥ 2} m_b(d_b − 1) = n − Σ_b m_b. Ω = 0 ⟺ K acts through characters on all of ℂⁿ (the "abelian part"), with the caveat recorded in §3.8.

**Theorem 3.6a (M1: universal Haar mean).** If K acts irreducibly on ℂⁿ (single isotypic block, d = n, m = 1), then for Haar u ∈ K:

  E S(u) = n(n²−1)/6, exactly.

**Proof.** Schur orthogonality: for an irreducible unitary representation of a compact group, E[u_{ij} \overline{u_{i'j'}}] = δ_{ii'}δ_{jj'}/n — in any orthonormal basis. Hence E|u_{ij}|² = 1/n for all i, j, and E S = (1/n)Σ_{i,j}(i−j)² = (1/n)·n²(n²−1)/6. ∎

This single computation explains why n(n²−1)/6 kept appearing as the empirical mixing constant: it is not a property of U(n); it is the Schur mean of *every* irreducible compact group. (Verified numerically on SU(2) irreducibles — spin-j representations with n = 2j+1 built from angular-momentum matrices and Haar quaternions: means within 0.2–1.3 standard errors of exact at n = 32, 64, 128.)

**Theorem 3.6b (M2: isotypic floor; general K).** For Haar u ∈ K:

  E S(u) ≥ Ω(K).

Moreover, in a basis in which the isotypic blocks are coordinate-aligned, the exact identity E S = Σ_b (1/d_b) Σ_{v≠v'} Σ_m Δ(pos(v,m), pos(v',m))² holds, where Δ is the standard-basis distance between the slots carrying the (v,m) and (v',m) coordinates; Ω is the worst-case (distance-1 interleave) value of the right side.

**Proof mechanism (the basis-free part).** Let M_{ij} = E|u_{ij}|². Then: (i) Σ_{i,j} M_{ij} = E‖u‖²_F = n and (i−j)² ≥ 1 off-diagonal, so E S ≥ Σ_{i≠j} M_{ij} = n − tr M. (ii) tr M = Σᵢ E|u_{ii}|²; decompose eᵢ into isotypic components; the diagonal coefficients from inequivalent blocks are L²-orthogonal (Schur), so M_{ii} = Σ_b E|f_{i,b}|² with f_{i,b} the diagonal coefficient of block b. (iii) Writing X for the d_b×m_b coordinate matrix of e_i's block component, f_{i,b}(u) = tr(X X*ᵖπ_b(u))-style, and Schur's two-index orthogonality gives E|f_{i,b}|² = (1/d_b)‖XX*‖²_F; since XX* is positive semidefinite, ‖XX*‖²_F = Σσ_k⁴ ≤ (Σσ_k²)² = ‖X‖⁴_F =: t_{i,b}². (iv) t_{i,b} ∈ [0,1] (orthogonal components of unit vectors) so t² ≤ t, and Σᵢ t_{i,b} = rank of the block = m_b d_b. Hence tr M ≤ Σ_b (1/d_b)·m_b d_b = Σ_b m_b, and E S ≥ n − Σ_b m_b = Ω. ∎

**History (the falsified identity).** The first draft claimed the coordinate-aligned identity as a general theorem. It is false: for an isotypic block not aligned with the standard basis, E|u_{ij}|² involves interference between multiplicity copies with equal coefficient functions, plus deterministic contributions from character (d_b = 1) blocks. A specific delocalized 2-plane block in n = 4 gives E S = 6.01 where the aligned identity is not even evaluable. The inequality survives with the basis-free proof above; Monte Carlo attack on ~10⁴ randomly conjugated blocks (all (d,m) ∈ {(2,2),(2,3),(3,2),(3,3),(2,5),(4,2)}) never produced E S below Ω, with equality attained exactly at aligned configurations.

**Sharpness.** K = U(2)^{⊕n/2} on adjacent pairs: each block contributes (1/2)(1² + 1²) = 1 = m(d−1); measured E S = 32.007 vs Ω = 32 at n = 64. Interleaved-at-distance-1 arrangements attain the floor; contiguous arrangements exceed it (the identity, not just the floor, tracks the arrangement: 84 vs 84.2 and 1344 vs 1342.4 at two tested interleavings).

**Theorem 3.6c (M3: positive words are dense in the group).** For any unitary family F, the closure of the positive-word semigroup F⁺ equals the compact group closure K of ⟨F⟩. Consequently, for any continuous f: sup_{w ∈ F⁺} f(w) = max_{k ∈ K} f(k).

**Proof.** Let S = cl(F⁺) (a compact subsemigroup: products of limits are limits of products). Claim: S is a group. Fix x ∈ S. Consider G₀ = cl{xᵗ : t ∈ ℤ}, a compact abelian group; let μ be its Haar measure (positive on nonempty opens). If the translates U, x⁻¹U, x⁻²U, … of a nonempty open U ⊆ G₀ were pairwise disjoint up to index T, their total measure would exceed 1 — impossible; so there are s < r with x^{−s}U ∩ x^{−r}U ≠ ∅, i.e. x^{r−s} ∈ U·U⁻¹. Taking U from a neighborhood basis of e (chosen to exclude x itself, and excluding x² as well by shrinking: choose the basis inside an open W with x ∉ W — possible unless x = e, which is trivial), this yields t_j ≥ 2 with x^{t_j} → e; then x^{t_j − 1} = x⁻¹·x^{t_j} → x⁻¹, and each x^{t_j−1} ∈ {xᵗ : t ≥ 1} ⊆ S, so x⁻¹ ∈ S by closedness. A closed subsemigroup of a group containing all inverses is a subgroup. S ⊆ K trivially (positive words are group words); S ⊇ F and S a group ⟹ S ⊇ ⟨F⟩⁻ = K. ∎

**Theorem 3.6d (M4: no cap survives) and Junction J.** If K_n are the group closures of a tower of families and Ω(K_n) → ∞, then for every C there exist n and a *single positive word* w with S(w) > C. (E over Haar ≥ Ω by M2 ⟹ max ≥ Ω; M3 lifts the max to a positive word.) Contrapositive with Theorem 3.5: no n-uniform geometric tail hypothesis survives a tower with Ω → ∞. Junction J: conversely, under the cliff (Σ_{r_b ≥ 2}(r_b − 1) ≤ 2C, Σ(r_b³ −r_b) ≤ 12C), every d ≥ 2 isotypic block of K lies wholly inside the sum of the non-abelian eigenspaces (a block intersecting the character part would make π_b a constituent of a character sum, forcing d_b = 1), so Ω(K) ≤ Σ r_b ≤ 4C. ∎

### 3.7 The dichotomy assembled

> **Theorem 3.7 (walk dichotomy; the program's central result).** For towers of unitary factor families: either
> **(A)** the products satisfy an n-uniform geometric tail bound — and then by the transfer chain S ≤ C₀(1+a)/(1−a)² uniformly, by the all-words transfer the group obeys the same bound, and by the cliff the family is structurally a MASA plus a non-abelian core of dimension ≤ 12C₁ independent of n, with Ω ≤ 4C₁; or
> **(B)** no such bound holds — and when Ω(K_n) → ∞ the failure is *forced*: for every cap C, a single positive word exceeds it, and Haar-typical products mix with E S → n(n²−1)/6 (exactly, in the irreducible case; universally bounded below by Ω).
> **There is no third regime.**

**Scope.** Unitary families; quantification over all words; constants uniform in n. The non-unitary (non-expansive) analogue of the cliff is open (§9.6). The all-words hypothesis in (A) cannot be weakened to chains (§3.4's counterexample). Boundary phenomena next.

### 3.8 The boundary: abelian families that still mix

The dichotomy's two vertices are not exhaustive in the sense "Ω = 0 ⟹ budget-keeping possible": a tower can have Ω = 0 (fully abelian group closures) and still force mixing. Counterexample (recorded, mechanism understood): u_n = V_n diag(e^{iθ₁}, …, e^{iθ_n}) V_n* with V_n generic delocalized (e.g., Haar) and θ generic irrationally independent. K_n is a torus — Ω = 0 — but sup over positive words S = the torus-Haar mean Σ_{i,j} ρ_{ij}(i−j)² with ρ_{ij} = Σ_v |V_{iv}|²|V_{jv}|² ≈ 1/n for delocalized V, giving Θ(n³). Mechanism: character blocks with delocalized eigenvectors carry fixed off-diagonal projection patterns that contribute to every product. What Ω = 0 *does* guarantee (by M2's proof mechanism): no *Schur-driven* growth — the contribution is deterministic, the same for every group element. The classification of which abelian towers keep budgets is exactly the registry axis's question in another dress, and is open (§9.5).

### 3.9 Numerical evidence base for the walk law

1. **Pre-registered n = 150 prediction.** Before the M1 theorem existed, the mixing law E S(P_j) → n(n²−1)/6 was registered with prediction 150·(150²−1)/6 = 562,475 for an equidistributing family; measured 559,895 at 30,000 steps (0.5%). The exact constant was then proven universal (M1).
2. **SU(2) spin-j check of M1** (n = 32, 64, 128; 400 Haar samples each): means within 0.23/0.60/1.25 standard errors of exact.
3. **M2 floor and identity** (U(2)^{⊕32} at n = 64: 32.007 vs 32; single U(8) block, contiguous vs interleaved slots: 84.24/1342.26 vs identities 84/1344; general-position conjugates: 5.4–108.2 vs Ω = 2–6, never below).
4. **M3 bridge diagnostic** (two generic U(16) generators, 200 words × 1500 steps): positive-word max S 810 vs signed-word max 792 — no lag; entry statistics at Haar values.
5. **Transfer identity** at machine precision (above).
6. **Real-data confirmations** (Chapter 8.4): unitary-exponential families of real matrices lock onto n(n²−1)/6 at n = 24 (99.9% by 25 products) and n = 167 (99.8% by 200).

## Chapter 4. The Self-Similarity Axis: Telescoping Rank

This chapter classifies where telescoping (Kronecker-factorizing) arithmetic structure can save work in matrix multiplication — the exact question left open by the program's most complete constructive failure (the DPS48 race, Chapter 7.1) and connected at its sharp end to a classical open rank problem.

### 4.1 The question in its operational form

A bilinear multiplication scheme for m×m matrices is a decomposition of the structure tensor into r rank-1 terms — r scalar multiplications plus linear encodings of the inputs and decodings of the outputs. A scheme's *certified* cost model amortizes the linear maps only if they **telescope**: if the encoding of a 4×4 scheme applied recursively to 16×16, 64×64, … operands reuses the same small transforms level after level, which happens exactly when the scheme's linear forms are Kronecker-decomposable across the block split. This motivates:

**Definition (product-form rank).** Split each 16-dimensional leg of the 4×4 multiplication tensor as a 4⊗4 Kronecker product — the block split, leg A = (i,k) with i = (i₁,i₂), k = (k₁,k₂), and level-1 coordinates (i₁,k₁), level-2 coordinates (i₂,k₂); similarly B = (k,l), C = (i,l). A scheme is **product-form** if every input form uⱼ and output form wⱼ is a pure tensor across its leg's split: uⱼ = xⱼ ⊗ yⱼ, etc. **r_pf** := minimum number of terms over product-form schemes. Equivalently: the CP rank of the tensor viewed as 6-way with 4-dimensional legs (the *refined* rank).

The operational meaning: r_pf = 64 means a telescoping scheme of that block structure saves nothing over naive; r_pf = 49 means Strassen's 7/8 saving per level is exactly achievable with fully telescoping transforms; r_pf = 48 would revive the DPS48 program (Chapter 7.1) with certified cost realizable.

### 4.2 Exact baselines

Both endpoints are certified in exact rational arithmetic:

**Naive (r = 64).** The digitwise scheme: T₄ = T₂ ⊗ T₂ as a pure tensor across the level cut (block multiplication is carry-free at both levels). Each of the 64 terms is a standard basis vector on each leg — pure across any split. Identity verified exactly.

**Two-level Strassen (r = 49).** Compose the classical 7-multiplication 2×2 scheme with itself: each composite factor is a Kronecker product of the 2×2 scheme's factors (u_{st} = u_s ⊗ u_t, etc. over the 49 pairs (s,t)). The composite identity C = W_c((U_c a)⊙(V_c b)) is exact over Fractions (verified on random integer inputs; factor purity audited by exact outer-product reconstruction: every factor satisfies M = x·yᵀ exactly). So **r_pf ≤ 49**.

**DPS48's certified basis is not product-form.** The rank-48 DPS scheme (Chapter 7.1) exists — tensor rank ≤ 48 — but its certified basis transforms P, Q, decode have bilocated (matrix-matrix realization) ranks 13/15/14 of a possible 16: sums of many Kronecker products, not one. So tensor rank and refined rank genuinely differ at 4×4 **if** r_pf = 49 — and the entire question is whether that difference is exactly one.

### 4.3 The separable-schemes theorem (aligned splits are naive, at every size)

**Theorem 4.3.** Fix m ≥ 2 and any field. In any bilinear scheme for m×m multiplication where every input form factors across the row|column split of its operand (uⱼ = xⱼ(i) ⊗ yⱼ(k)) and every output form factors across its row|column split (wⱼ = sⱼ(i) ⊗ tⱼ(l)), the number of scalar multiplications is at least m³. The naive scheme attains m³.

**Proof.** Regard the scheme as a CP decomposition of the 6-leg tensor

  T = Σ_{i,k,l ∈ [m]} eᵢ ⊗ e_k ⊗ e_k ⊗ e_l ⊗ eᵢ ⊗ e_l  on legs (i_A, k_A, k_B, l_B, i_C, l_C),

where the sum's index identifications enforce the three pairings (i_A = i_C), (k_A = k_B), (l_B = l_C). **Cut one leg from each pair:** rows = (i_A, k_B, l_C), columns = (i_C, k_A, l_B). The flattening of T across this *pairing cut* is the identity matrix I_{m³}: entry [(i,k,l), (i′,k′,l′)] = 1 exactly when i = i′, k = k′, l = l′ — because T's support is the set of consistent triples and the cut groups each triple's three "row" copies against its three "column" copies.

Now flatten each product-form term. The term (x⊗y)⊗(p⊗q)⊗(s⊗t) has flattening (x ⊗ p ⊗ t)(s ⊗ y ⊗ q)ᵀ: rank ≤ 1 — and here is the crux: the cut respects the row|column splits leg by leg. Leg A contributes its row index i to rows and its column index k to columns — opposite sides; likewise B (k vs l) and C (i vs l). So each separable term's cut-matrix is an outer product. A sum of r matrices of rank ≤ 1 equals the identity of size m³ only if r ≥ m³ = rank(I). ∎

**Strengthening (exact approximation floor).** The pairing-cut flattening is an entry permutation of the tensor, hence Frobenius-isometric; the flattened model has rank ≤ r; by the Eckart–Young theorem the best r-term separable approximation satisfies ‖T − scheme‖²_F ≥ m³ − r, attained by any r terms of the naive scheme (each naive term is a unit vector). Consequence: the program's numerical searches for separable (m³−1)-term schemes stall at residual exactly 1 — which they did, to 15 digits at m = 2, before the theorem was known.

**Generalization (any aligned split).** The proof only uses that the split's sub-vector spans refine leg-by-leg into the pairing cut. Any split with this property ("aligned") is naive-forced. At 4×4 this covers 23 of the 27 splits (§4.4).

**Meaning.** Row|column separability is the arithmetic of "each multiplication touches one row and one column independently" — the natural first design for telescoping. The theorem says this design buys exactly nothing, at every size, in every field, with a sharp approximation floor besides. All savings must come from splits transversal to the pairings.

### 4.4 The alignment classification

**Setup.** The 4×4 tensor's twelve bit-legs: A carries (i₁, k₁, i₂, k₂), B carries (k₁, l₁, k₂, l₂), C carries (i₁, l₁, i₂, l₂). Six Φ-pairings of legs carrying equal values: (i₁: A↔C), (i₂: A↔C), (k₁: A↔B), (k₂: A↔B), (l₁: B↔C), (l₂: B↔C). A product-form split assigns each leg's four bits to two 2-bit spans. Per-leg patterns: **L** = (i₁k₁ | i₂k₂) [level], **R** = (i₁i₂ | k₁k₂) [row|column], **M** = (i₁k₂ | i₂k₁) [mixed]; 3³ = 27 splits.

**The constraint graph.** Vertices = the twelve bit-legs. SAME-edges (6): the two bits of each sub-vector span — a cut realizing the split must put each span's bits on one side for the term-flattening argument to apply. OPPOSITE-edges (6): the two legs of each Φ-pairing — a pairing-respecting cut must split each pair for the flattening to be a permutation (the identity). A valid aligned cut exists iff this signed system is satisfiable.

**Lemma 4.4 (bipartiteness criterion).** The system is satisfiable iff no cycle of the constraint graph contains an odd number of OPPOSITE-edges. (Standard GF(2) solvability: SAME = parity-0 equation, OPPOSITE = parity-1 equation; inconsistency = a cycle with odd parity sum.)

**Theorem 4.4 (classification).** Of the 27 splits, exactly 23 are aligned (satisfiable) and 4 are transversal (unsatisfiable). The transversal set is the **L-family** {(L,L,L), (M,M,L), (L,M,M), (M,L,M)} — a single orbit under the tensor's bit symmetries. On all 23 aligned splits, r_pf = 64 by Theorem 4.3's generalization. On the L-family, r_pf ≤ 49 (Strassen; the (M,M,L) etc. realizations are σₖ-conjugates, machine-verified as exact rank-49 product-form schemes in integer arithmetic).

Witness cut for R (rows = {i₁A, i₂A, k₁B, k₂B, l₁C, l₂C}: every SAME-edge within a side, every Φ-pair split). Odd cycles for the L-family, e.g. at (L,L,L):

  i₁A —x→ k₁A —Φ(k₁)→ k₁B —p→ l₁B —Φ(l₁)→ l₁C —s→ i₁C —Φ(i₁)→ i₁A

— three OPPOSITE edges (odd), so no valid cut exists. Verified by exhaustive enumeration of all 64 pair-parity assignments per split (and at review, by exact integer arithmetic throughout).

**An instructive correction.** The program first believed there were three split types up to symmetry, with the "mixed" M-pattern empirically naive-forced. Both parts of that belief were wrong, and the corrections matter:

- The split the numerics actually tested was (M,M,M) — mixed on all three legs — which **admits** the witness cut rows = {i₁A, k₂A, k₁B, l₂B, l₁C, i₂C} (a 12-cycle with six OPPOSITE edges — even). So (M,M,M) is *aligned*, its r_pf = 64 is *provable* by Theorem 4.3's generalization, and the measured drop-exact plateaus (residual 1.00000000000003 at 63 terms; 2.0000009 at 62 — exactly the Eckart–Young floor 64 − r) were a 14-digit numerical confirmation of the theorem, not an open case.
- The originally printed (M,M,L) spans are the σₖ-conjugate of (L,L,L) (swapping the k-bits is an exact symmetry of the tensor): they carry the rank-49 conjugated Strassen and are decidedly not naive-forced.

**Corollary 4.4 (the cut landscape; why new technology is mandatory).** For every L-family split, every pairing-respecting cut straddles at least two sub-vectors (the two disjoint odd cycles each force one), so each product-form term's cut-flattening has rank up to 2⁶, and brute force over all 2¹² cuts gives a best bound of 2⁶⁻² = 16. Cuts that fail to split a pair cap at flattening rank 2⁵ = 32 < 48. Since separating r_pf (which wants ≥ 49) from tensor rank needs a cut bound of 64, **no coordinate-cut flattening argument can prove the L-family case** (generalized/Koszul flattenings are not excluded; an outside audit reports r_pf ≥ 36 from one — unverified, Erratum E9). This corollary closed an entire proof route by exhaustion — recorded so no successor re-walks it.

### 4.5 The rank-48 question

Everything now reduces to: **is r_pf(L-family) = 49?** — i.e., is rank-48 impossible with telescoping transforms? The evidence:

**Numerical (calibrated optimization).** Protocol: parameterize 48 product-form factors; optimize the CP residual. Calibration first — perturbed known solutions must polish back to ~0 (passes: 3.5×10⁻⁸ at n-legs; random-start recovery fails, the classical CP-swap swamp, so all conclusions rest on structured starts: disclosed). Then: 60 random restarts (all ≥ 1.0); all 1,176 pair-merges of the Strassen-49 solution screened (best 1.199) with the top 8 long-polished (12,000 iterations, tolerances 10⁻¹⁸): every one converges to the same floor 1.0000005–1.0000014, none below 1, gradients still shrinking (3×10⁻⁴…1.2×10⁻³). Residual mass concentrates in one symmetric cluster around a single multiplication-table entry (peak 0.139 with four level-symmetric satellites ≈ 0.0424). Reading: 48 product-form terms capture 63 of the tensor's 64 energy units — strictly better than any 48-of-49 subset of Strassen (which misses 16) — but never 0. The exactness of the floor 1.0 is the conjecture.

**Baby instance (m = 2, row|column split).** The analogous question — separable rank of the 2×2 tensor, with 8 = naive and 7 = Strassen — is settled by Theorem 4.3: r = 8 > 7. All-pairs merge searches from the naive scheme stall at exactly the drop-one energy; multistarts plateau at exactly 1.0000000000000. The baby instance is the base case of the ladder below.

**The reduction (what rank-48 is equivalent to).** The refined 6-way tensor is the structure tensor of the algebra M₂ ⊗ M₂ ≅ M₄ *in the Kronecker basis*: the level-split bilinear map ((A₁⊗A₂),(B₁⊗B₂)) ↦ (A₁B₁)⊗(A₂B₂) is literally multiplication in M₄ on decomposable tensors, which span. Since tensor rank is invariant under invertible leg changes, r_pf equals the rank of MM₄ *when every rank-1 term must be Kronecker-decomposable*. Consequences: (a) r_pf ≥ rank₃-way(MM₄); (b) DPS48 gives rank(MM₄) ≤ 48; (c) r_pf = 49 does **NOT** imply rank(MM₄) = 48: the known inequalities rank(MM₄) ≤ 48 and rank(MM₄) ≤ r_pf are compatible with rank(MM₄) < 48 and r_pf = 49 (ERRATUM E1: the earlier 'locked together' claim assumed an unknown lower bound of 48); (d) the best unconditional lower bound on r_pf is the Alder–Strassen algebra bound 2·dim(M₄) − 1 = **31**.

**Failed proof routes (all recorded, none re-walkable by accident):**

1. *Cut/flattening bounds* — capped at 16 by Corollary 4.4's brute force.
2. *Alder–Strassen directly* — basis-independent, blind to the Kronecker constraint; caps at 31 for every refinement.
3. *Multiplicativity of tensor rank* (rank(A⊗B) = rank(A)rank(B)) — false in general; the DPS48 scheme is precisely a counterexample instance at the unrefined level (rank(MM₄) ≤ 48 < 49 = 7²). The refined version is the open question itself, so no circular help.
4. *Slicing* — contracting one leg of the refined tensor gives a matrix-tensor product whose rank factorizes with the sliced side only (rank(M ⊗ T) = rank(M)·rank(T) when M is a matrix — this IS provable, standard), yielding 4·7 = 28. Each additional slice loses a factor; cannot reach 49.
5. *The coupled-flattening hope* (interactions of two cuts detecting decomposability) — the natural first idea; the 2¹² brute force shows even the best single cut gives 16, and coupled versions were not found to beat the counting.

**The staged attack (for a successor).** The ladder formulation:

| Rung | Statement | Status |
|---|---|---|
| 0 | rank(T₂) = 7 | classical (Alder–Strassen: rank ≥ 2·dim A − 1 for algebra multiplication) |
| 1 | refined (row\|col) rank of T₂ = 8 — gap exactly 1 over rung 0 | **proved** (Theorem 4.3 at m = 2) |
| 2 | refined (L-family) rank of MM₄ = 49 — gap exactly 1 over rung ≤ 48 | **open**; numerical floor; H0 armed |

The proof must go *through* the refinement: develop a substitution-induction (the Alder–Strassen engine: peel rank-1 terms against the algebra's structure) in which every peeled term is forced to be Kronecker-decomposable. The entry condition to attempt: in the AS induction at the refined level, show the substitution sequence can be chosen decomposable. The base-case precedent is exactly Theorem 4.3's cut mechanism at rung 1 — an argument the AS bound is blind to. The key lemma to try: in a 6-way decomposition, a term nonzero on two different pair-values cannot be shared across levels the way DPS48 shares terms — make precise via support combinatorics on the 64-point support (0/1 entries, no cancellations on supports; each term's support box must cover table points, and boxes spanning two levels pay both).

**The constructive flip side (H0, armed).** If instead some search ever certifies an exact 48-term product-form scheme (rational reconstruction + Fractions verification mandatory), then r_pf = 48 (giving only rank(MM₄) ≤ 48, already known), and — if the revival premise survives Erratum E7 — the DPS48 program revives with certified telescoping cost — the race's obstruction (Chapter 7.1) would be circumvented by construction rather than repaired.

---

## Chapter 5. The Registry Axis: Word Growth and Closure

### 5.1 The counting theorem

**Theorem 5.1 (descriptor floors = word counts; two independent gates).** Let F be a family of matrices over a field with enough elements, generating a free-enough semigroup (precisely: the evaluation map from words to matrices is injective up to the algebra's defining relations — the generic case, and all exponential-growth cases). Then for every k the minimum descriptor size D(k) satisfies D(k) = |W_{≤k}| · (1 + o(1)) — the number of words of length ≤ k, up to first order — and the numeral-separation lemma gives the matching lower bound constructively:

**Numeral separation (mechanism).** For generic generator entries, the (0,1)-entry of a product word w = g_{j₁}⋯g_{jk} is a polynomial in the generators' entries that encodes a base-q numeral: expanding along the word, the coefficient picks up one factor per letter position, and two distinct words produce polynomials agreeing on a set of codimension ≥ 1 — distinct words are separable by a single scalar functional. Hence any descriptor recovering all products must distinguish |W| objects with log₂-information ≥ log₂|W| bits each — and the list (storing one pointer per word) achieves it: Θ(k log m) total for m generators with exponential growth.

**The dichotomy corollary (via Tits).** A finitely generated family either is virtually nilpotent (word growth polynomial; Gromov/Bass–Guivarc'h) — solvable non-nilpotent groups already have exponential growth — or has exponential word growth (registry-full). There is no middle regime for matrix families.

*Meaning and use.* The word itself is an exact O(k log m)-bit descriptor, so counting alone gives no storage barrier; whether composition, evaluation and *required queries* stay cheap is a separate question this theorem does not address. (ERRATUM E2: an earlier version said 'exact descriptors are lists' and 'possible only in the essentially-commutative world'. Counterexample: G0=diag(2,1), G1=[[2,1],[0,1]] — exponential word growth, non-commuting, yet each product is (k,b) with (k,b)*(l,c)=(k+l, b+2^k c), O(k) bits, with growing-integer bit cost to be charged.) This theorem is the backbone of: the premium meter's registry axis; the Gottesman–Knill-type reading (stabilizer workloads are abelian-chain-cheap; one generic element exits the abelian sector into exponential word growth — the magic-state boundary as a registry cliff); and several obstructions in Chapter 6.

### 5.2 The subalgebra landscape

**Theorem 5.2 (semisimple inheritance).** Semisimple subalgebras inherit the matrix-multiplication exponent: composition inside semisimple structure offers no route below the world exponent. For direct sums, the frontier relation β = (ω−1)α + (2−ω) is tight (gated). Corollary: any representation-based saving must involve non-semisimple structure — radicals, approximations, or nonlinear closure.

### 5.3 Composition closure: the two dichotomies

**Theorem 5.3a (exact closure exists).** There are families whose fixed-accuracy approximant class is exactly closed under the native product — e.g., positive rank-one-dominant families with an additive certified budget — so composition never pays a truncation penalty and the certification composes interface-exactly.

**Theorem 5.3b (additive budgets up to a threshold).** For (K, a)-local uniform families, typed error budgets are additive up to k = Θ(a⁻ᴸ) for width-L descriptors.

**Theorem 5.3c (forced width).** For resolvent-type families — powers (I − bA)⁻¹-type operators of banded A — any band-L descriptor incurs width growing as Ω(ka) in the power k: fixed-width tracking is impossible exactly where iterative methods live. (This is the walk law's forcing phenomenon appearing on the registry axis; the two proofs are independent.)

**History note.** Theorem 5.3 family passed one adversarial and three delta gates, with the honest scope disclosure that the positive side (a) is conditional on a hypothesis class whose nontrivial membership is not exhibited — vacuity was priced and recorded rather than hidden.

### 5.4 An instructive falsification (recorded)

An early claimed "threshold theorem" asserted rank multiplicativity under Kronecker products of representations, which would have implied a known-false bound on ω. It died by internal arithmetic before any gate. The surviving fragment: Kronecker bases are tensor products of bases (true, kept). The corpse's lesson is now a standing rule: multiply any claimed inequality against the known walls before believing it.

## Chapter 6. The Obstruction Atlas

Every wall the program proved, with its statement and what it forecloses. These are the negative-space specifications a new representation must satisfy (the program's constitution reads barriers as maps, not tombstones).

| # | Obstruction | Statement (all gated) | Forecloses |
|---|---|---|---|
| 1 | **GIRS hypercube dimension** | The literal all-graphs DV representation conjecture fails: required ambient dimension grows as the hypercube count, exceeding any polynomial encoding; an explicit analytic family carries an exact-width obstruction, with low-rank approximation of the same family quantified | Duality-based universal graph representations |
| 2 | **Tree-composition growth** | In known tree-quotient systems, direct descriptor products achieve exact minimal-state composition; growth under commuting factors is sharp — no further compression exists (implementation certified: 26 fixtures, 260 exact comparisons) | Tree-indexed representation hopes beyond bookkeeping |
| 3 | **Separator support** | Real scalar-interpolation representations of the target class require a large shared global separator cover; pruning to local separators is impossible | Sparse-interpolation arithmetic for dense-product families |
| 4 | **Multivariate minor obstruction** | Fixed-grid minor-polynomial representations fail at n = 3 and n ≥ 5 (n = 4 inconclusive — recorded honestly) | One whole family of minor-closed representation formats |
| 5 | **Unitary separator** | Complete unitary families require a large shared block; no small-support decomposition exists | Unitary-analog sparsification |
| 6 | **Small-block templates** | No exact ℚ/ℂ restriction into M₂³ ⊕ scalars exists (exact template closed); coupled scalar-zero pivots reduce to two rational covering normal forms; the overlap search for ≤ 22 products terminates at 23 with frozen controls | The literal sub-Strassen small-block program |
| 7 | **Nonlinear closure scope** | A classical nonlinear family (product-closed, full linear span) falsifies the unrestricted linear-closure dichotomy; the corrected scope is recorded | Over-broad closure claims — this obstruction falsified the program's own landscape and fixed it |
| 8 | **Semisimple inheritance** (Thm 5.2) | — | Subalgebra-route speedups |
| 9 | **The counting theorem** (Thm 5.1) | — | Exact compressed arithmetic off the abelian world |
| 10 | **The walk dichotomy** (Ch. 3) | — | Position-structure persistence off the cliff |
| 11 | **The alignment classification** (Ch. 4) | — | Telescoping savings off the L-family |

The atlas's cumulative message: the known cheap families are cheap for law-like reasons spanning all three axes; a new cheap family must dodge every row simultaneously.

---

## Chapter 7. The Constructive Record

### 7.1 The DPS48 arc: certification, race, obstruction, revival question

**The object.** DPS48 is a rank-48 bilinear scheme for 4×4 matrix multiplication (entries in {0, ±1}; input additions 52 and 44, output additions 76), delivered with a certified alternative-basis realization (invertible 16×16 transforms P, Q, decode) and certified recurring cost: 172 additions per recursive node, giving amortized per-entry cost c = (L + 32)/32 against 192 for the best published scheme — a 10% certified operation-count improvement.

**Certification method (transferable).** Every identity was verified in exact rational arithmetic (Python Fractions) at every boundary: the composite identity C = DEC·W((UPa) ⊙ (VQb)) against direct products on random integer inputs, multiple seeds; the recursive driver against Python-integer references; the classical Strassen baseline (7M/18A) verified exact in the same harness before any timing. Forward errors at race sizes: 5.0×10⁻¹⁵ / 6.5×10⁻¹⁵ / 2.0×10⁻¹⁴ at n = 1024/2048/4096 — comparable to Strassen (no precision penalty beyond 1.7×).

**The race (pre-registered, run once).** Harness: NumPy/BLAS float64, identical for all contestants, BLAS leaf at 128, min of 2–3 reps after warmup. Kill rule registered before running: "> 15% slower than classic Strassen at 4096."

| n | BLAS | Strassen 18A | DPS48-172 |
|---|---|---|---|
| 1024 | 0.01 s | 0.11 s | 0.22 s |
| 2048 | 0.06 s | 0.95 s | 0.83 s (anomalous rep; not reproduced) |
| 4096 | 0.49 s | 6.89 s | **11.36 s** |

DPS48 is 65% slower than the *weaker* baseline at n = 4096 (11.36/6.89 = 1.649; the earlier '39%' was the reciprocal); kill rule fires. Working set: 48 concurrent operand buffers per node vs Strassen's 7 (6.9× cache-pressure disadvantage).

**The obstruction (measured, mechanism identified).** The certified cost model charges the basis transforms amortized ~32 ops/node — realizable only if the transforms **telescope** across recursion levels (Kronecker self-similarity: basis at size 16n = basis at 16 ⊗ identity-at-n). They do not: P, Q, decode are general sparse dyadic matrices whose bilocated (matrix-matrix realization) ranks are **13 / 15 / 14** of 16 — sums of many Kronecker products, not one. Every recursive node therefore applies the full basis: composed per-node cost 400 + 240 + 320 = 960 fused multiply-adds naive (best greedy synthesis 1541; tensordot evaluation 960) versus the certified 172 — a **5.6× per-node inflation intrinsic to the basis**, independent of implementation language. Charging the real 960 moves the certified crossover versus alternative-basis Strassen by ≈ (960/172)^{1/log₄(64/48)} ≈ 4.0×10³ (arithmetic corrected from 2.6×10³; the exponent gap log₄(64/48) compares against naive-64, not the actual competitor, so the crossover must be recomputed — and the per-node 960 premise itself is under re-examination, Erratum E7).

**The revival question.** Whether *any* rank-48 scheme admits telescoping transforms — precisely r_pf = 48 vs 49 — is the rank-48 question of Chapter 4.5, with its full staged attack and armed constructive flip side.

### 7.2 Labeling recovery: hidden structure, found cold, costed honestly

**The setting.** A matrix family is presented in an arbitrary vertex relabeling (a uniformly random permutation π applied to rows and columns). No metadata. The pipeline: detect → recover → exploit. The retraction of an earlier claim in this area had set the bar: the relabeling mechanism must actually be exercised.

**Detection (trivial and invariant).** The centralizer dimension Σ (eigenvalue multiplicities)² is similarity-invariant: identical canonical vs relabeled (measured 190/190, 222/222 across instances; C(14,7) = 3432 for the 7-cube). The earlier unit's "24/24 shift inclusion" test, by contrast, collapses to 1/24 under relabeling — recorded as the retraction's core finding.

**The recovery premise (a reduction).** The adapted arithmetic does not need the true π — it needs **any single-cycle order-n permutation matrix S in the centralizer of the family**: any cyclic generator's eigenbasis is a Fourier basis (which automorphism labels the characters is irrelevant to the arithmetic). Recovery thereby reduces to: find a permutation matrix in a linear subspace.

**Route 1 — alternation (works, moderate rate).** Iterate: project a random permutation onto the centralizer subspace (least squares in the n²-dimensional vec space) → round to the nearest permutation (Hungarian assignment) → repeat; accept iff the result is a single n-cycle commuting with the family to 10⁻⁹. Results: 3/3 relabeled Cayley instances at n = 96 recover cold (4/20, 1/20, 1/20 restarts; ~5 ms per restart); scales to n = 384 (2/12).

**Route 2 — phase sorting (deterministic, found in review of Route 1).** For a Cayley family on Z_n with a coprime generator, the eigenspace of the second-largest-character eigenvalue is 2-dimensional (characters ±1), and for ANY orthonormal basis (v₁, v₂) of it, z = v₁ + i·v₂ has phase 2πx/n + const at every vertex x (a rotation of (cos, sin) basis only rotates z's global phase). Sorting vertices by arg(z) yields a cyclic order whose successor permutation commutes with the whole family — membership exactly 0.0, no restarts, 3.2 ms at n = 96, 43 ms at n = 384. Scope: requires a coprime first generator (non-coprime generators collapse all eigenspaces to dimension ≥ 2·gcd — a replayed failure mode with cluster histogram {6:2, 12:31}, recorded).

**Exploitation (exact).** Through the recovered shift S: FFT-ordered diagonal arithmetic (eigenvalues via first-column FFT in the cycle order) reproduces dense word products to 4–6×10⁻¹⁶; native-diagonal per-step cost ≈ 3–7 µs, equal to the metadata oracle's; the BLAS crossover collapses to ~5 products (recovery now nearly free).

**The honest termination (the strongest baseline).** For commuting families — exactly the recovered case — **joint diagonalization** needs no recovery and no cyclic structure: diagonalize member 1; inside each degenerate eigenspace, diagonalize member 2's projection; the common eigenbasis gives the same elementwise word arithmetic at ~n² storage. Measured: joint diagonalization *beats* the recovery+FFT pipeline on materialized products at every size tested (0.143 vs 0.607 ms at n = 96; 3.63 vs 4.69 ms at n = 384, both exact to ~10⁻¹³) **[Erratum E13: a small-n constant-factor observation — the joint-diagonalization column materializes with an O(n³) GEMM (thread count unlogged), the FFT column with an O(n²) decode; single-thread crossover n ≈ 150, decode 10.5× cheaper at n = 1536; the withdrawal stands because the route is classical recognition + FFT]**. The pipeline's surviving advantages are **representational only**: O(n) storage vs O(n²); near-linear registration of each new family member vs quadratic projection; native-diagonal per-step at oracle parity. The measured implementation wall: ~2× if products are materialized each step; ~1500–3000× if the workload lives in the diagonal representation.

**The class boundary (kill rule fired, mechanism probed).** On Z₂ᵏ Cayley families (no n-cycles exist; the target is a fixed-point-free involution — one level of the recursive recovery), alternation fails completely: 0/100 restarts across two rounding operators (Hungarian assignment; greedy vertex-pairing on the symmetrized projection), and the diagnostic is decisive — the iterates never even reach the subspace (membership > 0 at every iteration's end) despite the subspace being *denser* (3280–3432 of 16384 vs 190–766 of 9216 for the cyclic class). Interpretation (labeled): small-multiplicity centralizers (mostly 2×2 blocks) have block-monomial permutation elements forming an attractive basin; high-multiplicity centralizers (blocks to 35×35) bury the target translations (which exist — they are the hidden x ↦ x⊕g maps) among non-permutation near-misses. Registered successor target: translations are Krawtchouk-encoded unitaries inside the multiplicity blocks — recovering them is an open problem; the classical recognition route (polynomial-time circulant/Cayley recognition by Muzychuk and Evdokimov–Ponomarenko) is the baseline any algebraic method must beat.

### 7.3 The commutant engine (foundational infrastructure)

The MKKK/Maehara–Murota engine (commutant as commutator nullspace; generic-element block-diagonalization) and its physics workload results: exact sector counts for pair-hopping chains (Fibonacci law), the 2×L dipole law ((F_{L+3}−1)²), the multiplicity-table measurements. Findings that shaped the program: enrichment repairs sampling, not commutation; multi-seed consensus recovers only the abelian (basis-diagonal) part of the symmetry — single generic seed optimal; the MPO reformulation is the only route past L = 10. A wrong centralizer-dimension belief was caught and corrected here (generic centralizer is the n-dimensional polynomial algebra, not 1 — Cayley–Hamilton; derogatory matrices larger).

### 7.4 The Wedderburn instrument

Centralizer-based hidden-block detection via a two-cluster log-spectrum estimator. Detection proven on synthetic towers. Fair advantage (all transform costs counted; dtype-matched baselines) from block dimension 24 upward. An 8× claim was retracted in review (excluded transform costs; complex128 baseline on a real family) — the corrected fair numbers stand **[Erratum E12: "fair" = against a dense GEMM chain only; an FFT + batched-matmul specialist is ≈ 3× faster than the emitter]**.

### 7.5 The premium meter

The two-axis instrument (registry word-span premium × walk constants), validated on seven retrodictions: circulants premium-free; simultaneously diagonalizable premium-free; **Toeplitz full word-span premium yet superfast via the displacement axis** (the two-axis distinction itself); generic pairs full premium; low-rank chains cheap by rank closure; stabilizer workloads abelian-chain premium-free (Gottesman–Knill as a registry statement); the PDE walk split (advection ballistic — band-8 error = 1 beyond k = 8; heat diffusive — √k growth 0.05/0.07/0.17 at k = 4/16/64; resolvent saturating). Prospective call 1 confirmed (connectivity threshold in sparse-orthogonal compressibility). Prospective call 2 falsified (Chapter 8.3) — producing the noise law.

---

## Chapter 8. Measurements on Real Data

### 8.1 Prevalence of hidden structure

Fourteen real matrices from the SuiteSparse collection (structural bcsstk01/03/04; NASA nos1; fluid pores_1, fs_541_1; circuit can_24; chemical west0167, steam1, steam2; reservoir saylr1/3/4; n = 24…3564). Profile: centralizer dimension Σ(multiplicities²) vs matched random baselines, then the **tolerance ladder** (re-measure at 10⁻⁶/10⁻⁸/10⁻¹⁰/10⁻¹²) applied to every candidate:

| Matrix | n | Verdict |
|---|---|---|
| bcsstk03 | 112 | **Robust bearer**: 142–168 > n at every tolerance; 2-dimensional eigenspaces throughout (plausibly geometric symmetry) |
| steam1 | 240 | **Robust bearer**: 280–420 > 240 at every tolerance |
| fs_541_1 | 541 | **Artifact**: 741 = 1.37n at τ = 10⁻⁸; exactly generic at τ ≤ 10⁻¹⁰; 58,599 at 10⁻⁶ (near-degeneracy ladder) |
| steam2 | 600 | **Artifact**: generic at τ ≤ 10⁻¹⁰ |
| saylr4 | 3564 | **Artifact**: generic at τ = 10⁻¹² (signal 1.14n at 10⁻⁸ is near-degeneracy) |
| six others | 24–237 | **Generic** at all tolerances (centralizer = n exactly, matching random) |

**Tally: 2 of 14 robust structure-bearers**, both plausibly physical symmetry — territory where classical symmetry-aware arithmetic already applies. Three would-be false positives were caught by the ladder (a matrix can look structured at one tolerance and be exactly generic at another; the ladder is now the mandatory instrument for any structure claim on numerical data).

### 8.2 The family-data reality

The pipeline pays when a *family* of same-size operators shares hidden structure (recover once; compose many). The standard collections contain no true operator families — the "chemical plant family" is three different sizes (240/600/80), the "reservoir family" three more (238/1000/3564). The family-level market question is blocked on data availability, not on instruments; the registered pivot is same-mesh parametric sources.

### 8.3 The noise law (measured; from a falsified pre-registration)

Setting: circulant A₀ plus sparse perturbations E₁, E₂ (8 nonzeros) at amplitude ε; family {A₀+εE₁, A₀+εE₂}; measure the span dimension of products up to length k (n = 12; tolerance 10⁻¹⁰; both degenerate and distinct-generator controls). Pre-registered predictions (hash-committed): both wrong; measured law:

- **ε = 0:** span grows as the monomial count, min(C(k+2,2), n) — reaching the hidden algebra (12) at k = 4. (The prediction had said saturation at k = 2 — it confused the algebra with the span of short words.)
- **Visible ε ≥ 10⁻⁸:** non-circulant corrections enter from the **second** product (the ε-order terms A₀E + EA₀ are two-sided circulant sandwiches — non-circulant for generic E); span exceeds the algebra's dimension by k = 3 (15 > 12); growth then tracks the *word count* toward full genericity (144) at k ≈ 8 = log₂(n²). (The prediction had said an instant cliff to n² — the corrections enter per-word, not all at once.)
- **Sub-tolerance ε = 10⁻¹⁴:** numerically identical to ε = 0 — structure survival is exactly tolerance-relative.

**Statement.** Hidden registry structure under model noise dies compositionally at word-count rate; survival is a joint property of (noise amplitude, tolerance, composition depth). Consequence for the market: recovered structure supports exact arithmetic only inside the tolerance window of the workload's own noise.

### 8.4 The Schurian mean on real matrices

Unitary-exponential families exp(iM/‖M‖_F) built from real matrices (design note: no two real matrices share a size, so each is paired with a random GOE partner — the generic-side prediction is thereby still tested; the bearer-contrast needs same-size structured siblings, unavailable):

- **can_24 (circuit simulation, n = 24):** E S reaches **99.9% of n(n²−1)/6 by k = 25** and stays locked (0.92–1.00) through k = 800.
- **west0167 (chemical process, n = 167):** 96.5% by k = 50; **99.8% by k = 200**; locked 0.997–1.000 through k = 2000.
- All five tested families grow toward the mean with **no plateau** — the mixing signature; a structure-bounded family would flatten below its isotypic floor (Ω).
- Protocol disclosure: the pre-registered decision rule ("within 10% by k = 10") was mis-calibrated about mixing times (the cutoff grows with n; earlier synthetic runs needed k to 3×10⁴); the rule fired as written and the re-analysis is the record.

The walk law's exact constant is what arithmetic built from real matrices actually does.

## Chapter 9. Open Problems (each with everything known)

### 9.1 The rank-48 question (the program's sharpest open problem)

**Is r_pf(L-family) = 49?** Equivalently: no rank-48 bilinear scheme for 4×4 multiplication has Kronecker-decomposable forms; equivalently (via the Chapter 4.5 reduction): every rank-48 scheme requires bit-coupled forms. Known: ≤ 49 (Strassen composes); the floor-1.0 numerical phenomenon (48 terms capture 63 of 64 units, never 0); no cut argument can work (cap 16); AS caps at 31; the baby instance holds with gap exactly 1 (8 vs 7). Proving it would NOT settle rank(MM₄) (Erratum E1); it is a statement about Kronecker-refined rank only. Refuting it constructively (an exact certified 48-term product-form scheme) revives the DPS48 program with realizable certified cost. The staged attack (decomposability-sensitive substitution induction) is in Chapter 4.5.

### 9.2 The refinement-gap ladder, rungs ≥ 3

Rung 2 is the 4×4 instance. Does the pattern "refined rank = unrefined rank + 1 exactly" continue up (8×8: 345 vs ≤ 7·49 = 343… the arithmetic of the ladder at higher rungs is itself worth computing)? Nothing is known beyond rung 1.

### 9.3 Representation-native workloads (the last Middle-tier door)

For workloads that can live in the recovered representation — long composition chains, streaming registration of new family members, O(n) storage regimes — the diagonal-domain arithmetic is measured at 1500–3000× per step against materializing baselines, but no *strongest classical specialist* for that exact regime has been costed against it (the natural candidate: maintaining products in a joint eigenbasis without materializing it). The market question (do real workloads have this shape?) decides the Middle tier; it requires same-mesh parametric operator families, which public collections do not currently provide.

### 9.4 High-multiplicity recovery

Recover the hidden translation structure of Z₂ᵏ-Cayley families from the relabeled adjacency alone. Known: alternation fails decisively (0/100, membership never achieved); the targets exist (translations x ↦ x⊕g are permutation matrices in the centralizer); their algebraic signature is Krawtchouk-encoded unitary blocks inside the M_{C(k,j)} multiplicities; the classical recognition route is polynomial-time and uncosted in practice. Any algebraic method must beat it.

### 9.5 The abelian mixing boundary

Classify which abelian towers (Ω = 0) keep walk budgets. Known: delocalized-eigenbasis tori mix to Θ(n³) sup-word spread despite Ω = 0 (deterministic contribution — same for every group element; no Schur growth). The question is exactly the registry axis's "which abelian structure is visible in position space," and is open.

### 9.6 Walk-law residues

(i) The non-unitary cliff: bounded S semigroups of contractions — Step 1's cocycle argument uses unitarity; no counterexample and no theorem. (ii) The half-line variant of Proposition 3.4a (Q unbounded on ℓ²(ℕ)): isometric self-embeddings of non-compact convex sets need not be surjective — the unilateral shift embeds cl-conv{eₖ} into itself; norm-isometries are norm-distal, blocking the naive fixed-point route; weak compactness + Ryll-Nardzewski may work modulo unbounded-Q bookkeeping. (iii) The aperiodic single-chain form (§3.4). (iv) Chain-level equidistribution: typicality along random-product walks (existence of exceeding words is proven; the *rate* along typical walks is the open part — the real-data measurements of Chapter 8.4 are the current empirical base).

### 9.7 Descriptors under approximation

The counting theorem is exact. The truncation dichotomies (5.3) cover the two ends. The middle — approximate descriptors with certificates, between additive budgets and forced width — is mapped only at its endpoints.

---

## Chapter 10. The Research Cemetery

Every route tried, how far it got, what broke. (A negative result with a mechanism is a map.)

1. **The threshold theorem** (asserted rank multiplicativity under ⊠). Died by arithmetic against a known wall. Fragment kept (Kronecker bases = tensor products).
2. **Enrichment/multi-seed consensus for symmetry detection.** Dies structurally for non-abelian symmetries: consensus sees only the basis-diagonal part. Single generic seed is optimal (generic elements are ones in MKKK).
3. **DPS48 as a practical scheme.** Died at the race (65% behind weaker Strassen at n = 4096) through the non-telescoping basis (960 vs 172 per node). Mechanism fully measured; question refined into 9.1.
4. **"Non-obvious" workload detection claims.** Died in review: the relabeling mechanism was never exercised; circulant recognition is classical. Refined into the recovery premise (any cyclic generator suffices) which then *worked*.
5. **The recovery pipeline as a speedup claim.** Died against joint diagonalization on materialized products **[Erratum E13: reason corrected — a small-n constant-factor artifact; the route is classical recognition + FFT]**. Survivors: the phase-sort recovery; the representational advantages; the exact cost table.
6. **Z₂ᵏ recovery by alternation.** Died at 0/100 with a decisive diagnostic (subspace never reached). Mechanism probed; successor target named (Krawtchouk blocks).
7. **Prospective noise prediction v1.** Both quantitative claims wrong (monomial growth; per-word destruction). Corpse became the noise law of 8.3.
8. **Mixed-split classification v1.** The referent error (tested (M,M,M) = aligned; printed (M,M,L) = LEVEL-conjugate). Correction upgraded an empirical case into a 14-digit theorem confirmation.
9. **The draft M2 identity.** False off aligned bases; Monte Carlo kill; basis-free proof built in repair.
10. **The draft transfer-chain bound.** Lost an n via the fixed-split accounting; repaired by the tail-sum identity (the reviewer supplied it; the draft error is recorded as the natural first attempt).
11. **Cut-based proofs of rank-48.** Exhaustively capped at 16 by the 2¹² brute force after the classification — an entire route closed by computation.
12. **The early small-block program** (exact restrictions into M₂³ ⊕ scalars; overlap searches to 22 products). Exact templates closed by proof; searches terminated at 23 with frozen controls. The literal route is closed; the obstruction rows remain in the atlas.
13. **Dense-Kronecker implementation of adapted bases.** An N³-cost transform implementation bug masked the structure (fixed with axis-wise FFTs; then a one-sided-conjugation convention bug; then a normalization mismatch) — three implementation layers, each of which silently destroyed the asymptotics, caught only by exactness asserts. Recorded as the implementation wall's concrete anatomy.

---

## Chapter 11. Notation and Index

- **S(u), Q, b(g):** spread energy ‖[Q,u]‖²_F; position operator diag(0..n−1); displacement cocycle g*Qg − Q.
- **Ω(K):** non-abelian isotypic support Σ_{d_b≥2} m_b(d_b − 1).
- **MASA:** maximal abelian self-adjoint algebra.
- **r_pf, product-form:** Chapter 4.1; **L-family:** the four transversal splits; **aligned:** split refining a pairing cut.
- **tail(L):** ‖u·1(|i−j|>L)‖²_F; the **tail-sum identity** S = Σ(2k−1)tail(k−1).
- **D(k), W_k:** descriptor cost; words of length k.
- **DPS48:** the rank-48 4×4 scheme family; **bilocated rank:** rank of the matrix-matrix realization of a transform under the block split.
- **The tolerance ladder:** measure at 10⁻⁶/10⁻⁸/10⁻¹⁰/10⁻¹²; robust structure survives all.
- **The implementation wall:** structure not exploited by the runtime is structure not had.
- **CP swamp:** random-start ALS/gradient recovery fails on tensor landscapes; structured starts only.

## Chapter 12. Bibliography (the base the theorems stand on)

- K. R. Moudgalya, R. Nandkishore (2023), *Commutant algebras and fragmentation* — the commutator-nullspace framework.
- T. Maehara, K. Murota (2010–11); Murota et al. (MKKK) — generic-element simultaneous block-diagonalization.
- B. Johnson (1967) via the BdHV analysis text, Prop. 2.2.9 — bounded cocycles are coboundaries for unitary representations.
- Hoffman–Wielandt (1953); Weyl — eigenvalue perturbation.
- Schur orthogonality; Peter–Weyl theorem — compact-group harmonic analysis.
- V. Strassen (1969, 1986/87); A. Alder, V. Strassen (1981) — bilinear rank; the 2·dim A − 1 bound; degeneration.
- Eckart–Young (1936) — best low-rank approximation.
- A. Smirnov (1983); DPS preprints (2025–26) — rank-48 4×4 schemes.
- C. Van Loan, N. Pitsianis (1993) — approximation with Kronecker products.
- AlphaTensor (Fawzi et al., Nature 2022) — the 𝔽₂ 47 scheme (field distinction noted: char-0 lower bounds are far weaker).
- R. Muzychuk (2004); I. Evdokimov, A. Ponomarenko — circulant/Cayley recognition is polynomial.
- Lootens–Delcamp–Verstraete (2021–23); Sala et al. (2020); Khudorozhkov et al. (2022) — fragmentation physics substrate.
- GNVW index theory arXiv:0910.3675; Freedman–Haah–Hastings arXiv:1910.07998; Ranard–Walter–Witteveen arXiv:2012.00741 — cited within the QCA imports of the walk program.

---

*End of monograph. Composed 2026-09-21 from the full program record (final repository commit 6af5a139; 1,232 ledger rows; all primary documents in `research/`, all raw captures in `results/`). Every theorem here survived at least one independent adversarial review (which, as the Errata show, is not the same as being correct); every failure here carries its mechanism. The open problems are stated with everything the program knows about them.*

# Appendices and Enrichment

## Appendix E. Worked examples and computations (Chapter 3 companion)

### E.1 Spread energy of the standard families

Computed exactly from S(u) = Σ_{i,j}(i−j)²|u_{ij}|²:

- **Diagonal unitaries:** S = 0. The MASA vertex.
- **Adjacent transposition** Tᵢ (swaps eᵢ, eᵢ₊₁): S = 2 (two entries at distance 1). The minimal non-abelian step.
- **Disjoint product D_n = T₀T₂T₄…:** S = n/2 · 2 = n. All entries at distance 1 — tail(L) = 0 for every L ≥ 1 — yet S = Θ(n). This example is why the transfer chain's L = 0 instance is load-bearing: the family {D_n} has tail(L) ≡ 0 for L ≥ 1 but its L = 0 value ‖D_n·1(|i−j|>0)‖² = n is not uniform in n, and the theorem correctly refuses to apply.
- **Cyclic shift** σ: entries at distance 1 except the wrap (distance n−1): S = (n−1)·1 + (n−1)² = n(n−1). Linear growth — the shift is the canonical "bounded-bandwidth ballistic" family.
- **Reversal** ρ(i) = n−1−i: S = Σᵢ(2i−(n−1))² = (n³−n)/3 ≈ 2× the Haar mean. The maximum over permutation unitaries (mass-transport: S(π) = Σᵢ(i−π(i))², maximized by the anti-diagonal assignment).
- **Haar U(n):** E S = n(n²−1)/6 exactly (Theorem 3.6a); concentration within O(std) ≈ n^{2.5}-scale for n large (measured std/mean ≈ 2–3% at n = 128–150).
- **Givens words** (products of ~40 random adjacent/near-adjacent rotations, n = 256): the family used for machine-precision verification of the tail-sum identity — S computed two ways (direct sum; tail-sum) agree to 2.4×10⁻¹⁶ relative.

### E.2 The cliff's constants, step by step

For a block of dimension r collapsed onto r consecutive integers, the Hoffman–Wielandt forced deviation is Σ_{j=0}^{r−1}(j − (r−1)/2)² = r(r²−1)/12. The chain: Σ(r³−r) ≤ 12C; then r³ − r ≥ 6(r−1) ⟹ Σ(r−1) ≤ 2C; then #blocks ≤ 2C (each contributes ≥ 6) ⟹ Σr_b ≤ 4C; the algebra dimension n + Σr_b² with Σr_b² ≤ (Σr_b)² ≤ 16C² for the square-sum, or ≤ max(r)·Σr_b ≤ ((12C)^{1/3}+1)·4C linear-in-C bookkeeping — the recorded O(C) with explicit constants.

Extremal check: ⊕U(2) on n/2 adjacent pairs has S = n/2 blocks × (1² + 1²) = n... precisely each U(2) block contributes (1/2)(1²+1²) = 1 expected, total E S = n/2, while Σ(r³−r) = (n/2)·6 = 3n = 6·E S: the constant 12 is slack by exactly 2 at this family (C = 2·E S by the Frobenius bound, giving Σ = 3·C/... the recorded factor-4 total slack decomposes as 2 here and 2 in the pairing).

### E.3 The delocalized torus (boundary counterexample, full construction)

n arbitrary; V = Haar unitary; θ₁,…,θ_n generic rationally independent; u = V diag(e^{iθⱼ}) V*. The family {u} is a single unitary: K = closure{uᵗ} = V·𝕋ⁿ·V*, abelian, Ω(K) = 0. The positive words are u, u², u³, …, dense in V·𝕋ⁿ·V*. For Haar-typical torus elements, E|w_{ij}|² = ρ_{ij} := Σ_v |V_{iv}|²|V_{jv}|² (cross terms vanish by character orthogonality); Σρ_{ij} = n; for delocalized V, ρ_{ij} ≈ 1/n uniformly, so sup-word S ≥ E-torus S = Σ_{i,j} ρ_{ij}(i−j)² ≈ (1/n)·n²(n²−1)/6 = Θ(n³). Mechanism, stated sharply: character blocks with delocalized eigenbases contribute a *fixed* off-diagonal projection pattern |V_{iv}V_{jv}|² to every product — deterministic mass, present at every group element alike, invisible to Ω (which counts only Schur-driven growth). Consequence: budget-keeping among abelian towers is a property of the *eigenbasis delocalization*, not of the group — see Problem 9.5.

### E.4 The snake-loop counterexample (products vs group, full mechanism)

A family of permutation unitaries built from a path-snake traversal: increments whose products are confined (sup over positive words S ≈ 1,594) while the generated group contains a long-range rearrangement with S ≈ 21,333,200 ≈ 1.3×10⁴× larger. Mechanism: positive words from the family return (the snake folds back), but group words can unfold (commutator-type combinations stretch across the index line). This is why Proposition 3.4a's all-words hypothesis is genuinely needed and why the permutation-unitary polynomial-loss lemma (C → C³) is the best known bridge for single chains. Recorded numbers: chain budget 1594; group element 21,333,200; ratio ≈ 1.3×10⁴ < C³ consistent with the lemma.

### E.5 The five-field unification, field by field

1. **Quantum lattices (Lieb–Robinson).** For local H, the Heisenberg evolution of a local operator A(t) = e^{iHt}Ae^{−iHt} has support growing ballistically: in finite-dimensional chains with the index line as position, S(A(t)) grows quadratically in t (velocity squared). Gapped dissipative (Liouvillian) dynamics: √t. The walk law's reading: the local Hamiltonians have bounded per-step tails; their products (evolutions) mix unless the family is cliff-decomposed — and physically it never is; hence ballistic/diffusive transport as forced mixing.
2. **Iterative vs direct solvers.** Resolvent powers ((I − bA)⁻¹)ᵏ of banded A: Theorem 5.3c's forced width. Direct factorizations (LU) of banded matrices stay banded — the family of elimination steps is walk-bounded by construction. The dichotomy is a design constraint: iterative composition cannot stay local, so any compressed-iterative scheme must pay registry instead.
3. **Automatic differentiation.** Forward-mode chains compose Jacobians applied to a fixed vector: position structure of the seed propagates with bounded displacement (the walk stays bounded along the chain). Reverse-mode accumulates adjoint products whose displacement adds: the walk mixes. The walk law predicts which mode preserves sparsity — matching the known practical rule (exploit forward-mode structure when it exists).
4. **Hierarchical matrices.** H-matrix admissibility (well-separated interactions) is a walk condition: elliptic-problem operators have bounded-walk structure (information decays with distance), transport operators are ballistic (information moves). The admissibility boundary is the walk dichotomy in engineering dress: measured — advection band-8 error = 1 beyond k = 8 (ballistic); heat √k growth 0.05/0.07/0.17 at k = 4/16/64; resolvent saturating.
5. **Positional carries.** Adding in base q: the carry operator is a displacement phenomenon with tail structure identical to Theorem 5.3c's resolvent forcing — carry width grows linearly in the number of digits added without cascading tricks; the digit-torus cascade absorption analysis (from the self-similarity program) is the constructive escape on that axis.

## Appendix F. The complete 27-split enumeration (Chapter 4 companion)

Patterns per leg: L = (i₁k₁|i₂k₂), R = (i₁i₂|k₁k₂), M = (i₁k₂|i₂k₁). Status after the bipartiteness enumeration (A = aligned, r_pf = 64 proved; T = transversal, the L-family):

| # | A | B | C | Status | Note |
|---|---|---|---|---|---|
| 1 | L | L | L | **T** | the level split; Strassen achieves 49 |
| 2 | R | R | R | A | the row|column split; Theorem 4.3's home cut |
| 3 | M | M | M | A | the tested "mixed" split; witness cut {i₁A,k₂A,k₁B,l₂B,l₁C,i₂C} |
| 4 | M | M | L | **T** | σₖ-conjugate of (L,L,L); conjugated rank-49 verified |
| 5 | L | M | M | **T** | conjugate |
| 6 | M | L | M | **T** | conjugate |
| 7–27 | the remaining 21 combinations | | | A | each admits a pairing-respecting cut; 64 forced |

Symmetry classes under the tensor's bit-swap × leg-permutation group: 5 orbits of sizes 12+6+4+4+1; the L-family is a single orbit. Brute-force cut bounds on every transversal member: best 2⁴ = 16; non-pairing rank cap 2⁵ = 32.

## Appendix G. Complete numerical tables

### G.1 Walk law

| Experiment | Parameters | Result |
|---|---|---|
| Pre-registered mixing prediction | n = 150, equidistributing family, k → 3×10⁴ | predicted 562,475; measured 559,895 (0.5%) |
| M1 on SU(2) spin-j | n = 32/64/128; 400 Haar samples | means 5491.0/42915.7/362131.3 vs exact 5456/43680/349504 (0.23/0.60/1.25 se) |
| M2 floor U(2)^{⊕32} | n = 64; 1000 samples | 32.007 vs Ω = 32 (0.07 se) |
| M2 identity, arrangement | U(8) block, n = 32; contiguous/interleaved | 84.24 / 1342.26 vs identities 84 / 1344 |
| M2 general position | (d,m) ∈ {(2,2),(2,3),(3,2),(3,3)}; conjugated | 5.43/28.03/32.25/108.24, all ≥ Ω = 2/3/4/6 |
| M3 bridge | 2 generators U(16); 200 words × 1500 | positive max S 810.0 vs signed 792.3; E|w₀₀|² = 0.0667/0.0636 vs 1/16 = 0.0625 |
| Tail-sum identity | Givens words, n = 256 | worst relative deviation 2.4×10⁻¹⁶ |
| Transfer bound vs truth | n = 32…4096 | S ∈ [0.63, 0.76] vs C₁ = 2.37 |
| Tightness example | m_d = (1−a)a^d | S = C₁ to machine precision (442.0 vs 442.96 at a = 0.25) |

### G.2 Self-similarity searches

| Experiment | Result |
|---|---|
| Strassen-49 exact residual | 0.0 |
| Perturbed-recovery calibration | 3.5×10⁻⁸ (49), 3.6×10⁻¹¹ (naive-8 baby) |
| Random-start recovery | fails (96.4 vs 0) — CP swamp; disclosed, structured starts only |
| Baby (2×2 separable): all 28 pair-merges at r = 7 | min = 1.000000000000032 = drop-one energy exactly |
| Parent: 1,176 pair-merges screened | min 1.199, median 1.66, max 4.88 |
| Parent: top-8 long polish | 1.0000005–1.0000014; none below 1; gradients 3×10⁻⁴…1.2×10⁻³ |
| Residual structure | peak entry 0.139 + 4 satellites 0.0424 (level-symmetric cluster) |
| (M,M,M) split: 2016 merges at 63 | min exactly 1.00000000000003 (Eckart–Young 64−63) |
| (M,M,M) cascade to 62 | exactly 2.0000009 (64−62) |
| DPS48 basis bilocated ranks | P = 13, Q = 15, decode = 14 of 16 |

### G.3 Constructive programs

| Program | Numbers |
|---|---|
| DPS48 certification | 172 vs 192 ops; additions 52/44/76; exactness 8 seeds; errors 5.0e-15…2.0e-14 (n = 1024…4096) |
| DPS48 race | 0.22/0.83*/11.36 s vs Strassen 0.11/0.95/6.89 vs BLAS 0.01/0.06/0.49 (*anomalous rep) |
| Per-node cost | 960 fused madds vs certified 172 (5.6×); greedy synthesis 1541 |
| Crossover displacement | (960/172)^{1/log₄(64/48)} ≈ 4.0×10³ (corrected) → N ~ 10⁷ claimed, under re-examination |
| Phase-sort recovery | 3.2 ms (n = 96) / 43 ms (n = 384); membership exactly 0.0 |
| Alternation recovery | 4/20, 1/20, 1/20 (n = 96); 2/12 (n = 384); ~5 ms/restart |
| Z₂ᵏ alternation | 0/100 (two operators); in-subspace count 0 in all runs |
| Exploitation exactness | 4–6×10⁻¹⁶ vs dense; joint-diag 5.7×10⁻¹⁴…2.5×10⁻¹³ |
| Cost table n = 96 | joint-diag 0.143 ms; pipeline 0.607; BLAS 0.546; native 3 µs |
| Cost table n = 384 | joint-diag 3.63; pipeline 4.69; BLAS 12.63; native 7 µs; recovery 43 ms one-time |
| Cost-table caveat (E13) | joint-diag column = O(n³) GEMM materialization, pipeline column = O(n²) decode; single-thread decode / GEMM (ms): 0.49/0.16 (n = 96), 5.0/12.3 (384), 19.7/94 (768), 77.7/819 (1536) |

### G.4 Market

| Probe | Result |
|---|---|
| Prevalence | 2/14 robust (bcsstk03: 142–168 > 112 all tolerances; steam1: 280–420 > 240); 3 artifacts; 9 generic |
| fs_541_1 ladder | 741 @10⁻⁸ → 541 @10⁻¹⁰ → 58,599 @10⁻⁶ |
| Noise law | clean: min(C(k+2,2), n); noisy: exceeds algebra at k = 3 (15 > 12), generic at k ≈ 8; sub-τ ≡ clean |
| Schur lock-on | can_24: 99.9% @k=25, locked to k = 800; west0167: 96.5% @50, 99.8% @200, locked to k = 2000 |

## Appendix H. Computational methodology (how every number above was produced)

1. **Exact arithmetic as the default.** All identity verifications, rank computations on integer/dyadic data, and certificate checks run in Python `Fractions` (or exact GF(2)/integer enumeration). Floating-point is used only for optimization and measurement, and every floating result that supports a claim is either (a) verified exactly after rational reconstruction, or (b) labeled EMPIRICAL with its protocol.
2. **Calibrated optimization.** Before any plateau is believed, the optimizer must recover a *known* solution from perturbation (e.g., perturbed Strassen-49 → 3.5×10⁻⁸). Random-start recovery failures are disclosed and never silently relied upon (the CP swamp is real: random-start ALS stalls at 96.4 on a problem with a known 0).
3. **In-script invariant asserts.** Similarity-invariant quantities are computed twice (canonical vs relabeled) and asserted equal — this caught the directed-graph construction bug when the centralizer dimension disagreed with itself (3958 vs 108). Naive constructions are asserted to give residual exactly 0 before any search runs — this caught the mixed-coordinate factor bug (residual 96).
4. **Pre-registration.** Predictions, null models, and kill rules are hash-committed to the ledger before measurement. In this record: the n = 150 mixing prediction (confirmed), the noise-law prediction (falsified in both claims — the measured law is the result), the Schur-rate rule (mis-calibrated about cutoffs — the re-analysis is recorded).
5. **The tolerance ladder.** Any structure claim on numerical data must survive measurement at 10⁻⁶/10⁻⁸/10⁻¹⁰/10⁻¹². Near-degeneracy ladders impersonate multiplicity (3 of 14 real matrices).
6. **Independent adversarial review.** Every PROVED label here survived at least one zero-context reviewer attacking exactly its statement; repairs were applied and re-reviewed (delta). The reviews materially changed: the transfer chain's proof (supplied the tail-sum identity), M2's proof (falsified the identity form, built the basis-free version), the classification (found the referent error, upgraded it to a proof), the recovery claim (found the dominating baseline and the better algorithm).

## Appendix I. Extended cemetery (earlier lines, complete)

Beyond Chapter 10's entries, the early program's closed lines:

- **CRC approximation by finite vertical range** (proved as construction): for the registered row-snake family, relative spectral error ≤ 972·2⁻ᴸ with width (L+1)(2L+2) per flow and O(N·L²) arithmetic. Narrative value: exact state growth does not force growing width at fixed accuracy — representation-side positive result, no speedup claimed (the retained word already applied in O(N)).
- **GIRS explicit approximation**: exact analytic obstruction + low-rank approximation of the same family; known construction, no novelty claim.
- **Tree-native reduction** (proved, scoped): native exact minimal-state derivation and sharp growth under commuting factors; implementation authorized but no speedup.
- **Displacement generalization**: classical implementation; measurements section left incomplete — recorded as control, closed honestly.
- **The deep-anatomy program** (six mechanism-laws and the five-field unification) — DERIVED, unreviewed as theorems, but every one of its six laws is now either proved (Burnside envelope → counting theorem; closure trichotomy → 5.3; noncommutativity premium → walk law) or embedded in open problems (rigidity, carry-resolvent, fixed-point). The anatomy's role in this record: it was the map that the theorem arc walked.

## Appendix J. What a reviewer should probe (self-assessment of weak points)

Intellectual honesty about where this monograph's claims are thinnest:

1. **The walk dichotomy's unitarity.** All of Chapter 3 lives under unitarity. Real workload matrices are not unitary (the real-data measurements used unitary exponentials — a modeling choice, disclosed). A reviewer should decide how much this limits the law's reach; the program's view: unitary families are the correct first case (the cocycle is honest there), and the non-unitary extension is a well-posed open problem (9.6), not a hidden flaw — but it is the single most consequential scope limit in the theory.
2. **The one-unit floor is numerics, not proof.** The rank-48 conjecture rests on calibrated optimization. The calibration is strong (perturbed recovery, structured starts, exhaustive merges), but a constructivist reviewer should weigh H0 seriously: the search space is continuous and the floor phenomenon, while perfectly clean to 10⁻⁶, is not a certificate.
3. **Market prevalence at n ≤ 3564.** All real-data claims are at sizes where O(n²) constants dominate. The representational-advantage door (9.3) is untested at scales where it would matter.
4. **The anatomy→theorem arc.** Two of the six anatomy laws remain unproven as stated (rigidity, fixed-point). They motivated proofs but should not be cited as results.
5. **Gates are adversarial, not infallible.** The review protocol caught real errors and improved real proofs, but every gate is one fresh reviewer (sometimes two, with delta). A mathematical reviewer of this monograph is the next gate — the statements are written to be attackable.
