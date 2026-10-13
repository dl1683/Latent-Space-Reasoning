# Landscape of composition-closed subalgebras with fast native multiply (V1)

**2026-09-07 qualification.** The semisimple calculations below are not a
classification of all associative algebras. G3's complex exact-bilinear-rank
specialization is refuted by the explicit
square-zero counterexample and universal semisimple comparison in
[REPRESENTATION_FRONTIER_V2.md](REPRESENTATION_FRONTIER_V2.md), after independent
proof and delta reviews. This is not a total-arithmetic or runtime separation.
The measured triangular/dual examples do not exhaust radical
algebras. Existing measurements remain unchanged.

**Closure scope correction.** The unrestricted exact-closure dichotomy formerly
stated in section 2 is withdrawn. A multiplication-closed nonlinear family need
not be a linear subalgebra. The reviewed classical
[symmetric-power counterexample](NONLINEAR_COMPOSITION_SCOPE_V1.md) has compact
native products, scalar commutant, and full linear span. This leaves the
linear-subalgebra classification and historical measurements intact.

Status: research unit, 2026-09-06. Labels: PROVED / EMPIRICAL / CONJECTURE / SPECULATION.
Code: `code/experiments/subalgebra_landscape.py` (one runner; modes `bench`,
`closure`, `precision`, `selftest`). Tests: `code/tests/test_subalgebra_landscape.py`.
Data: `results/subalgebra_landscape_v1/` (`bench.json`, `bench_table.md`,
`pinned1/bench.json`, `closure.json`, `precision.json`).

Question. The "matrix as primitive" program needs LARGE subalgebras A of M_n(R)
that are closed under multiplication and natively cheap, so a matrix can stay
compressed across many products. Block-circulant (B389) is one. What else exists,
how do they compare, and where are the gaps?

Coordinates. dim A = n^alpha, native multiply cost = n^beta. Dense is (2, 3);
scalars are (0, 0). The task asked whether anything sits at 1 < alpha < 2 with
beta < 3 alpha / 2.

## 1. The frame (PROVED, Wedderburn-Artin plus counting)

Every semisimple subalgebra of M_n(C) is conjugate to a block-diagonal algebra
(+)_i M_{s_i} in which the i-th block appears with multiplicity m_i,
sum_i s_i m_i <= n. Then

    dim A = sum_i s_i^2,        native cost = sum_i s_i^omega  (in Wedderburn coordinates).

For k equal blocks with multiplicity 1 and k s = n:  dim = n s,  cost = k s^3 = dim^2 / n.
In exponents the FRONTIER of exact semisimple families is

    beta = 2 alpha - 1                     (schoolbook inside blocks)
    beta = (omega - 1) alpha + (2 - omega) (fast matmul inside blocks),

a straight line from (1, 1) to (2, omega). It lies strictly below the 3 alpha / 2
line for every alpha < 2, so the task's question 3 is answered YES by any direct
sum; e.g. k = s = sqrt(n) gives dim n^1.5 at cost n^2 (vs n^2.25 for the line).

Lower bound on the same frontier: each M_{s_i} block needs at least s_i^2
multiplications, so cost >= dim for every semisimple A. The whole intermediate
regime is therefore pinned between beta = alpha and beta = 2 alpha - 1, and the
only unknown in it is omega. Structured subalgebras INHERIT omega on their blocks;
they never improve it. (Cohn-Umans runs the same inequality backwards: embed
matmul into C[G] so that n^omega <= sum d_rho^omega.)

What distinguishes subalgebras of the same Wedderburn type is not the multiply
but three side quantities:

1. Intertwiner cost: FFT-like (n^2 log n) vs generic conjugation (2 GEMMs = n^3).
2. Intertwiner conditioning: unitary (*-closed algebras) vs kappa(S) >> 1.
3. The radical (non-semisimple part).

## 2. The map

| family | closed? | dim (alpha at n=2048) | native cost model | intertwiner | position |
|---|---|---|---|---|---|
| dense M_n | yes | n^2 (2.00) | n^3 | none | (2, 3) |
| block-diagonal, k x M_s | yes | k s^2 | k s^3 | identity | ON frontier |
| block-circulant M_s(C[Z_k]) (B384/B389) | yes | k s^2 | ~2 k s^3 real + FFTs | FFT (x) I, unitary, n^2 log k | ON frontier (same type as block-diagonal) |
| M_s(C[G]), G abelian | yes | |G| s^2 | |G| s^3 | Kronecker FFT, unitary | ON frontier |
| M_s(C[G]), G non-abelian (dihedral implemented) | yes | |G| s^2 | (sum d_rho^3) s^3 >= |G| s^3 | fast FT exists for some G | ABOVE frontier by sum d_rho^3 / |G| (D_m: ~2x) |
| commutant of X with eigen-multiplicities | yes | sum m_i^2 | sum m_i^3 | generic S: n^3, kappa(S) | ON frontier, but conversion-dominated |
| commutative semisimple (diagonalizable) | yes | n (1.00) | n | eigenbasis | endpoint (1, 1); FFT only for group-like cases |
| Kronecker { X (x) I_b } (multiplicity b) | yes | n^2 / b^2 | (n/b)^3 = dim^1.5 | identity | EXACTLY on 3 alpha / 2 line (multiplicity wastes n) |
| upper triangular | yes | n(n+1)/2 (1.91) | n^3 / 6 | identity | alpha = 2, beta = 3: constant factor only (radical n(n-1)/2) |
| matrix dual numbers M_s[eps] (B388) | yes | 2 s^2 | 3 s^3 | 2-channel Hadamard | ABOVE frontier: M_s (+) M_s costs 2 s^3 at equal (n, dim) |
| band width b | NO | n(2b+1) | n b^2 | - | product has width 2b: filtration |
| H-matrices, rank k off-diagonal | NO (approx.) | n k log n | n k^2 log^2 n | - | product has rank 2k; truncation = precision cost |
| displacement rank r (Toeplitz-like) | NO | ~2 r n | r^2 n log n | - | product has rank r1 + r2 |
| butterfly / Monarch | NO | n log n | n log n (apply) | - | product of two butterflies is not a butterfly |
| Lie algebra envelopes | yes | Burnside: irreducible rep => all of M_n | - | - | nothing new beyond Wedderburn |

Within a linear subspace, multiplication closure gives an associative algebra;
the preceding algebra classification applies within its hypotheses. It does not
classify nonlinear multiplication-closed families. The former unrestricted
"if and only if" dichotomy and "only on the frontier" conclusion are withdrawn.
Specific growth claims for the listed band, hierarchical, displacement, and
butterfly representations retain their stated scopes. A full Lie-algebra
envelope does not rule out compact descriptors for individual group elements.

## 3. Measurements (EMPIRICAL, float64, CPU, all relative errors <= 7e-16)

Full tables: `results/subalgebra_landscape_v1/bench_table.md` (24-thread
OpenBLAS) and `results/subalgebra_landscape_v1/pinned1/bench.json` (1 thread,
the B389 convention). "1-product" charges decode of two promised dense inputs
plus the dense n^2 output; break-even is the number of chained products after
which the native path wins including all conversions.

n = 2048, dim 65536 (alpha = 1.45), 1 thread / 24 threads:

| representation | flops ratio | native speedup | 1-product speedup | break-even |
|---|---|---|---|---|
| block_diagonal k=64,s=32 | 4096 | 1787x / 235x | 104x / 16.6x | 1 / 1 |
| block_circulant k=64,s=32 | 1181 | 237x / 40.5x | 36.5x / 4.7x | 1 / 1 |
| dihedral_block m=32,s=32 | 756 | 171x / 15.2x | 33.2x / 3.5x | 1 / 1 |

n = 2048, dim 262144 (alpha = 1.64):

| representation | flops ratio | native (1 thr / 24 thr) | 1-product (1 thr / 24 thr) |
|---|---|---|---|
| block_diagonal k=16,s=128 | 256 | 175x / 7.0x | 55x / 5.0x |
| block_circulant k=16,s=128 | 103 | 56x / 4.7x | 24x / 2.0x |
| dihedral_block m=8,s=128 | 49.5 | 20x / 2.3x | 12.4x / 1.2x |
| kronecker X(x)I_4, a=512 | 64 | 74x / 16.5x | 15x / 3.7x |

Other rows at n = 2048 (1 thread / 24 threads): commutative_conjugate (dim 2048,
generic orthogonal S) native 282,000x / 107,000x but 1-product 0.32x / 0.26x and
NEVER breaks even (each decode is 2 GEMMs = 0.207 s vs 0.177 s for the dense
product at 24 threads); upper_triangular native 1.53x / 0.84x (the 6x is a flop
count; dtrmm on a triangular right operand does ~n^3/2, and multithreaded dgemm
wins outright); matrix_dual_numbers native 2.36x / 1.35x, 1.67x / 0.45x with IO.
The radical families gain a constant factor at best, and lose it to a better GEMM.

Readings:
- B389 reproduced: 36.5x and 24x single-thread dense-IO speedups at the B389
  shapes (B389: 29.5x and 16.4x). Against a 24-thread GEMM the same paths give
  4.7x and 2.0x because the n^2 single-threaded decode/assembly becomes the floor.
  The B389 headline is a 1-thread number; stated as such it holds.
- Dense-vs-dense timing noise is +/-15% (best-of-5); differences under that are
  not readable.
- Prediction (2), non-abelian penalty ~2x: 1.2x-2.7x measured, ~2x where block
  products dominate the FFTs.
- Prediction (3): Kronecker multiplicity b costs exactly b times the equal-dim
  direct sum in flops (64 vs 256 at dim 262144).
- Wall-clock ratios track flop ratios only loosely: batched small GEMMs and
  numpy FFTs are less efficient than one large GEMM, which is why the flop model
  is reported alongside every timing.

## 4. Precision axis (EMPIRICAL)

Same algebra type (16 blocks of 16 in M_256), native-path relative error vs a
longdouble reference, as a function of the intertwiner S:

| S | kappa(S) | native-path error | bits lost | dense GEMM error |
|---|---|---|---|---|
| FFT (x) I (block-circulant) | 1 | 4.5e-16 | 1 | - |
| random orthogonal | 1 | 1.2e-15 | 2.4 | 1.6e-15 |
| dense, prescribed | 1e2 | 2.0e-14 | 6.5 | 1.1e-14 |
| dense, prescribed | 1e4 | 4.9e-11 | 17.7 | 5.1e-13 |
| dense, prescribed | 1e6 | 2.3e-7 | 30 | 3.2e-11 |
| dense, prescribed | 1e8 | 1.4e-3 | 42.6 | 2.4e-9 |

Roughly 1.5 log2(kappa) bits per product vs 0.5 log2(kappa) for the dense
product on the same inputs. This is the two-axis trade of
`research/MATRIX_NATIVE_INFORMATION_THEORY.md` section 2 made concrete: the
multiply count of a subalgebra is fixed by its Wedderburn type, and a
non-unitary intertwiner moves the saved multiplications onto the precision axis.
*-closed subalgebras (unitarily block-diagonalizable) are the ones that are free
on both axes; block-circulant is *-closed, a generic commutant is not.

## 5. Generated algebras (EMPIRICAL, exact over GF(1000003))

`closure` mode takes integer generators, closes under multiplication exactly in
GF(p), and reports dim, radical (trace-form radical, Dickson), centre (commutant
of the generators inside A) and, when semisimple, the block partition over C from
a generic central element. n = 8..24, seed 0:

- 2 or 3 random sparse generators with >= 2 nonzeros per row: all of M_n in every
  case (dim n^2, one block). Burnside: irreducible action => everything.
- 1 nonzero per row: dim 0.44-0.94 n^2 with radical 25-103 (random n-nonzero
  matrices are singular, so the closure is radical-heavy, not fast).
- one generator: dim <= n, commutative, as it must be.
- two random permutations: M_{n-1} (+) M_1 (permutation rep = trivial + standard).
- two band-1 (tridiagonal) matrices: all of M_n. Bands are not a subalgebra.
- dihedral regular representation D_m: 2-dim irreps with multiplicity 2,
  schoolbook cost 36 vs 20 for Z_20 at equal dimension.
- random elements of a commutant of X with eigen-multiplicities 2: recovers the
  partition (2,1)^(n/2) exactly.

Verdict: sparsity alone never produces a small closure. Small, fast closures come
only from a shared invariant-subspace / commutant structure fixed in advance,
which is the Wedderburn partition again. "Pick random sparse matrices and close"
is not a construction route.

Method note: floating-point closure is unreliable and was replaced. Word bases
are exponentially ill-conditioned (Krylov), so a dependent word can have a
computed residual of 5.7e-8 (observed at n = 20) that exceeds any fixed
tolerance, is normalized to a unit vector, and spawns a fake full algebra. The
exact GF(p) path is the one to trust; the float path is used only for the block
partition and only when its dimension agrees with the exact one.

## 6. Where the gaps are

G1 (the real one). Everything on the exact frontier reduces to omega inside
blocks. No subalgebra route lowers the exponent; the program's ceiling target is
untouched by this map.

G2. Fast unitary intertwiners are known for group-like Wedderburn types (abelian
FFT, some non-abelian FFTs) and their Kronecker products. Which commutant-type
subalgebras admit an O(n^2 log n) *-preserving intertwiner is open (blackboard
signal s2). For a generic commutant the conversion is 2 GEMMs and never pays.

G3. CONJECTURE: at equal (n, dim) the radical never beats the semisimple direct
sum. Holds for dual numbers (3 s^3 vs 2 s^3) and triangular (constant factor);
no general proof here.

G4. Not addressed: which USEFUL matrices lie near a frontier algebra (the
fingerprint-plus-error program, section 6 of the information-theory note). The
map says where exact compressed composition is possible; it does not say that
any application's matrices live there.

## 7. Re-run

    python code/experiments/subalgebra_landscape.py selftest
    python code/experiments/subalgebra_landscape.py bench --sizes 512 1024 2048 --ks 4 16 64 --repeats 5
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/subalgebra_landscape.py bench --sizes 2048 --ks 4 16 64 --repeats 3 --out results/subalgebra_landscape_v1/pinned1
    python code/experiments/subalgebra_landscape.py closure --closure-sizes 8 12 16 20 24
    python code/experiments/subalgebra_landscape.py precision --precision-n 256 --precision-k 16
    python -m pytest code/tests/test_subalgebra_landscape.py -q
