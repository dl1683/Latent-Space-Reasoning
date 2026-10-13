# OZAKI_FAIR_ENGINE_V1 - a fair own CRT (Ozaki-II) INT8 emulation engine with inside scaling vs the vendor's FP64 emulation, accuracy-matched

Status: IN PROGRESS (started 2026-09-29). Distance from claim: 1 for the engine (it IS an alternative arithmetic
representation of the matrix product: a matrix held as residues modulo pairwise-coprime 8-bit moduli under power-of-two
row/column/inner scalings, multiplied by exact integer GEMMs); the race against the vendor is a distance-2 measurement
inside this unit. Mission tie-in: AGENTS.md sections 1 and 3 (a representation in which the matrix product is cheaper,
with conversion, precision and memory all charged), section 11 (constructive follow-through of `beyond_bilinear_v1`,
in which only the vendor's own emulation won); CURRENT_FOCUS "Fresh-eyes review" items 1 and 7.
Code: `code/ozaki_engine.py` (single module; reuses the harness pieces of `code/beyond_bilinear.py`).
Data: `results/ozaki_fair_engine_v1/`. Scratch (not evidence): the session scratchpad `uOE/`.

## 0. Prior art and gap (three closest; retrieved 2026-09-29)

1. Ozaki, Uchino, Imamura, "Ozaki Scheme II: a GEMM-oriented emulation of floating-point matrix multiplication using an
   integer modular technique" (arXiv 2504.08009; GEMMul8): CRT with pairwise-coprime moduli, ~14-16 INT8 GEMMs for FP64.
   Did NOT: use inside (contraction-index) scaling, give an a-priori componentwise error contract that picks the modulus
   count, or compare against cuBLAS 13.1's built-in emulation on a Blackwell laptop part (sm_120).
2. Abdelfattah, Dongarra, Fasi, Mikaitis, Tisseur, "Analysis of Floating-Point Matrix Multiplication Computed via
   Integer Arithmetic" (arXiv 2506.11277): rounding-error analysis of the Ozaki-I slicing; shows badly scaled inputs
   (A = A~D, B = D^-1 B~ with per-row circular rotation) defeat 18 slices. Did NOT treat the CRT variant, inside scaling,
   or a cost-aware choice of the number of slices.
3. Demmel, Henry, Kozachenko, Langou, Li, Riedy, Vanover, "How to grade the accuracy of the BLAS" (arXiv 2609.12307,
   section 5.2.1): both Ozaki I and II pass the graded-exponent test only with inner scaling, and the rotated/permuted
   test defeats inner scaling; cuBLAS detects the wide range and defaults to native FP64. Did NOT measure speed, nor say
   how many moduli a given contract needs.
Also: NVIDIA cuBLAS 13.x fixed-point emulation (docs: dynamic or fixed mantissa control, `cublasSetWorkspace` "avoids
both the allocation cost and the fallback path"); Ootomo-Ozaki-Yokota (arXiv 2306.11975); Ballard et al. 2016 (inside
scaling for Strassen-like accuracy). Gap this unit fills: a cost-honest, accuracy-matched head-to-head of a FAIR own CRT
engine against the strongest vendor configuration on this machine, plus an a-priori, checkable precision contract.

## 1. PRE-REGISTRATION (frozen before any measurement; hash by `python code/prereg.py ozaki_fair_engine_v1 research/OZAKI_FAIR_ENGINE_V1.md`)

### 1.1 Why the previous unit's own engine (M1) was handicapped (read from the code and records, no new measurement)

(a) Ozaki-I slicing needs 36-91 INT8 GEMMs where CRT needs ~14-16. (b) `beyond_bilinear.py` lines 130 and 244-246 pass a
row-major right operand to `torch._int_mm` (PyTorch issue 165230: the rhs is transposed internally). (c) The measured INT8
rate (64-91 TOPS, `micro.json`) equals the FP16 rate, although on GeForce Ada/Blackwell INT8 tensor peak is 4x FP16 with
FP32 accumulate (datasheet: RTX 5090 Laptop INT8 dense about 254 TOPS at 1.515 GHz = 82 SM x 2048 op/clk/SM). (d) The S
family needed 13 slices; inside scaling removes that. (e) The vendor baseline never called `cublasSetWorkspace`, and the
vendor result was about 13x more accurate than native FP64 (E1 1.2e-16 vs 1.5e-15 at n = 2048, G), so the vendor spends
about 4 bits more than the matched-accuracy criterion needs. (f) 7 of 13 cells of the previous unit violated its own 5%
drift rule.

### 1.2 Machine and baselines

RTX 5090 Laptop (sm_120, 82 SMs, 24 GB), torch 2.12 dev cu128 (cuBLAS 12.8, no emulation), system CUDA 13.1 toolkit
(cuBLAS 13.1 and cuBLASLt via ctypes, NVRTC 13.x for own kernels). Native FP64 about 0.69 TFLOPS peak. The GPU throttles
(83-87 C, SM clock 1087-1900 MHz under load); the owner's CPU jobs may saturate the CPU.
Baselines (float64 in, float64 out, n x n):
 - V-native: `torch.matmul` float64 (cuBLAS DGEMM). All speedups are relative to it, bracketed in every round.
 - V-emu variants (cuBLAS 13.1 DGEMM with math mode 8 = FP64_EMULATED_FIXEDPOINT):
   Vd = out of the box (strategy PERFORMANT, dynamic mantissa, default cap, no user workspace; as in the previous unit);
   screening grid per cell (see 1.6): strategy {PERFORMANT, EAGER} x workspace {none, 2 GiB via `cublasSetWorkspace_v2`}
   x max-mantissa cap {default, 128} (dynamic mantissa control), plus FIXED-mantissa variants b in {40, 48, 56, 64}
   (strategy EAGER, workspace 2 GiB), which are ORACLE-calibrated (the user must know b).
   V*dyn = the fastest accuracy-matching DYNAMIC variant of the grid in that cell (the vendor's own a-priori rule);
   V*fix = the fastest accuracy-matching FIXED variant (oracle-level). If no variant matches, V* = V-native.
   Cross-check only (screening): V-native13 = cuBLAS 13.1 DGEMM with math mode 0.
 - Overlap of two IQR intervals [q1, q3] (over valid rounds) means max(q1_a, q1_b) <= min(q3_a, q3_b).

### 1.3 Own engine E-CRT (specification frozen; details and code in section 2)

Moduli (pairwise coprime, verified in code): 256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199, 197,
193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127, 113, 109, 107 (first N are used, N <= 32). A modulus
p contributes signed residues in [-(p-1)/2, (p-1)/2] (p = 256: [-128, 127]), so every INT8 GEMM of inner dimension k <= 131071 is
exact in INT32. Pipeline (everything inside the timed region, inputs and preallocated buffers resident):
 1. Statistics: inner exponents a_l = max_i e(A_il), b_l = max_j e(B_lj) (e = frexp exponent), inside scaling
    gamma_l = floor((b_l - a_l)/2) (0 if a column/row is zero); row exponents r_i = max_l (e(A_il) + gamma_l), column exponents
    c_j = max_l (e(B_lj) - gamma_l); s^A_i = sum_l |A_il| 2^gamma_l, s^B_j = sum_l |B_lj| 2^-gamma_l; the norms nu^A =
    max_i (sum_l (A_il 2^(gamma_l - r_i))^2)^(1/2), nu^B likewise. Non-finite input => fall back to native FP64.
 2. Scaled integers for a chosen p: A'_il = rn(A_il 2^(p - r_i + gamma_l)), B'_lj = rn(B_lj 2^(p - c_j - gamma_l)) (rn = round to
    nearest, ties away from zero, so |A'|, |B'| <= 2^p and the quantization error is <= 1/2).
 3. Residues of A' and of B' modulo each of the N moduli by exact integer arithmetic on the GPU (mantissa/exponent
    extracted from the FP64 bits; for a left shift the residue is (mant mod p)(2^sh mod p) mod p from a table), stored as
    int8 with the contraction index contiguous for both operands (B is written transposed: the "TN" layout INT8 tensor
    cores need).
 4. N INT8 GEMMs with INT32 accumulation (cuBLASLt TN, or `torch._int_mm` with a column-major B: chosen by the section 1.8
    diagnosis and frozen before the race), each followed by X_t = C32 mod p_t stored as uint8.
 5. Mixed-radix (Garner) reconstruction of X = A'B' exactly in multi-limb integers, symmetric-range correction, one
    correctly rounded conversion to FP64, multiplication by 2^-(alpha_i + beta_j).
 Lemma L1 (elementary; PROVED-pending-gate): if the CRT stage is exact (|X_ij| < M/2) the result satisfies
 |C_hat_ij - C_ij| <= E_ij(p) + 2^-53 |C_hat_ij| with
   E_ij(p) = (1/2) 2^(r_i - p) s^B_j + (1/2) 2^(c_j - p) s^A_i + (1/4) k 2^(r_i + c_j - 2p).
 Range: |X_ij| <= ||A'_i|| ||B'_j|| <= (2^p nu^A + sqrt(k)/2)(2^p nu^B + sqrt(k)/2) =: Xmax(p).
 A-PRIORI RULE (the precision contract), frozen: p_max(N) = the largest integer p with 2 Xmax(p) + 1 <= M_N (M_N = product
 of the first N moduli). N* = the smallest N in [6, 32] such that E_ij(p_max(N)) <= tau D^_ij / 2 on the probe set P, with
 tau = 2^-52 and D_ij = sum_l |A_il||B_lj| evaluated in FP64 on P. P = 4096 uniformly random pairs (seeded) united with the
 pairs formed by the 32 rows of largest 2^(r_i) k / s^A_i and the 32 columns of largest 2^(c_j) k / s^B_j. If N* would
 exceed 32 the call falls back to native FP64. The oracle N is the smallest N (same p_max(N)) whose measured E1 and E2
 are both <= native's on the shared sample. N*, p, and the scaling exponents use only the inputs (no reference, no native
 error). Setup excluded from timing: NVRTC compilation, cuBLASLt algorithm autotune per shape, buffer allocation (all
 preallocated once per size and reused; the same courtesy as the vendor's user workspace). A fixed-N graph capture of the
 heavy part may be used to remove host-launch jitter; the statistics pass, the host synchronisation and the rule are
 always inside the timed region.

### 1.4 Input families (float64, seeded, n x n both operands, n in {2048, 4096, 8192})

G, X, S, R exactly as in `beyond_bilinear.make_inputs` (G Gaussian; X graded, (U-0.5)exp(2N); S = inner-index similarity
A = G D, B = D^-1 G' with D = 2^u, u uniform integers in [-16, 16]; R row/column scalings 2^s_i, 2^t_j, s, t in [-20, 20]).
W (new, "ill-scaled/rotated", after Abdelfattah et al. section 5.2): A~, B~ ~ N(0,1); d_c = 2^(20 c/(k-1)); A = A~ diag(d),
B = diag(d)^-1 B~; then A[i, l] <- A[i, (l + i) mod k] and B[l, j] <- B[(l + j) mod k, j]. Inside scaling cannot balance W.

### 1.5 Accuracy protocol (matched accuracy; unchanged from `beyond_bilinear_v1`)

On the shared sample (48 random rows x 48 random columns plus the argmax row/column of the native result) against a
correctly rounded reference (Dekker TwoProduct + `math.fsum`): E1 = max |C_hat - C_ref| / max |C_ref|;
E2 = max |C_hat - C_ref| / (|A||B|)_ij. A method is accuracy-matched in a cell iff E1(method) <= E1(V-native) and
E2(method) <= E2(V-native) on the same inputs and sample. A win is only claimable for the A-PRIORI configuration; a
configuration that matches only at the oracle N is labelled ORACLE-N and cannot support H1a. Deviation rule: if the
a-priori configuration fails to match in more than 20% of valid cells, tau is divided by 4, all cells are re-run and the
change is disclosed as a deviation.

### 1.6 Timing protocol and drift rule (enforced by the harness, not by reading)

Screening (per cell, untimed for decisions): every vendor variant of the grid is run 3 times in interleaved random order
and checked for accuracy; the two fastest matching dynamic variants and the fastest matching fixed variant, plus Vd,
enter the main rounds together with native, own-cert (the a-priori configuration) and own-oracle (when its N differs).
In the decisions below V*dyn is the carried dynamic variant (Vd included) with the lowest main-round median seconds and
V*fix is the carried fixed variant. Main rounds (per cell): 2 warm-up rounds discarded; then rounds until 7 VALID rounds; at most 30
attempts. A round is [native pre] [all other methods in a fresh random order] [native post] with CUDA events and a
synchronise per call; per-round ratio = t_method / mean(native_pre, native_post). A round is VALID iff
|native_post - native_pre| / min(native_pre, native_post) <= 0.05; invalid rounds are discarded whole and counted. If fewer than 7 valid
rounds after 30 attempts, the block is re-run once after a 60 s cool-down; if still fewer the cell is INVALID(drift), reported and
excluded from the H0/H1 decisions. No outlier removal. SM clock, temperature, power and throttle reasons are logged (NVML) after
every call; the benchmark process runs at high priority; CPU load is logged; the run is not paused for owner load.
Statistics per method: median and IQR (25/75 percentile) of the per-round ratio and of the seconds; paired per-round
rho = t_own / t_V*dyn (median, IQR).

### 1.7 Hypotheses, predictions, decision rules, kill rule

H0: for every valid cell and family, the a-priori own engine (matched accuracy) is NOT more than 10% faster than V*dyn:
median rho >= 0.90 OR the IQR intervals of the native-normalised ratios of the two overlap; and it does not beat native by
more than 1.3x on S or W in a cell where Vd (out of the box) falls back to native (speedup over native <= 1.05).
H1a: in at least one valid (n, family) cell the a-priori own engine at matched accuracy has median rho <= 0.90 and IQR
intervals (native-normalised ratios) that do not overlap with V*dyn's, over >= 7 valid rounds.
H1b: on S or W for some n, Vd's speedup over native is <= 1.05 (fallback) while the a-priori own engine at matched accuracy is
> 1.3x faster than native (median ratio < 0.77 and the q3 of its per-round ratio < 0.77).
Secondary (labelled): own-oracle vs V*fix (oracle level); own vs native everywhere.
Quantitative predictions, written before running: the number of moduli chosen by the a-priori rule N*: G, R 15-17, X 16-19,
S 15-17 with inside scaling (>= 21 without), W 23-28; oracle N about 2 below N*. Speedup over native at n = 4096-8192: G, R 6-14x,
X 5-11x, S 6-14x, W 3-7x. Ratio to V*dyn: 0.5-1.0 on G and R (a win if INT8 GEMM is faster than the vendor's), 0.4-1.2 elsewhere.
P(H1a) = 0.5, P(H1b) = 0.85. E-CRT reproduces A'B' bit-exactly (integer-exactness test, n <= 256, Python integers).
Precision contract: measured E_ij <= E_ij(p) at every sampled entry (never violated), and E_ij(p)/measured has median
about 2^5-2^8 on G (worst-case bound versus typical random-sign errors); the a-priori configuration matches native accuracy
in >= 80% of valid cells.
Kill rule: if H0 survives (no cell satisfies H1a or H1b) the unit records the mechanism from the kernel-level time breakdown
(INT8 GEMM TOPS actually reached vs the section 1.8 ceiling, conversion, reduction, reconstruction) and the next method is chosen
from: own INT8 GEMM with a fused modular-reduction epilogue; FP8/MX block-scaled tensor-core route; fused conversion kernels;
stochastic rounding with a Hoeffding certificate (l2 instead of l1 norms). The unit does not stop on H0.

### 1.8 INT8 throughput diagnosis (Task 2), hypotheses and predictions

D0: the previous INT8 = FP16 result reflects a hardware or power ceiling: no INT8 path exceeds 1.3x the FP16 rate.
D1: it reflects a slow path: some path (`torch._int_mm` with column-major B, cuBLASLt TN INT8, cublasGemmEx INT8) reaches
>= 1.5x the FP16 rate (>= 100 TOPS at n >= 4096). P(D1) = 0.6.
Predictions: `torch._int_mm` row-major B vs `B.t().contiguous().t()` differ by <= 20% at n >= 4096 (the transpose is O(n^2)); the
best cuBLASLt algorithm reaches 120-200 TOPS at n = 4096-8192; a bare mma.sync s8 loop kernel (NVRTC) reaches 82 x 2048 x f_SM
within 15% at the logged SM clock f_SM; FP8 e4m3 (`torch._scaled_mm`) is 1.3-2.5x the FP16 rate. Rates measured with the
GPU in its loaded, throttled state (interleaved with FP16 as a clock witness). The section 1.3 GEMM path is frozen after this
diagnosis as the fastest path that is bit-exact against `torch._int_mm` on random int8 operands.

### 1.9 Cost accounting reported for every cell

Seconds of every method (median, IQR), speedup over native, N and p (own) or mantissa variant (vendor), workspace bytes
(own buffers vs vendor 2 GiB workspace), kernel-level breakdown of the own engine (statistics, residues, GEMMs, reduction,
reconstruction), achieved INT8 TOPS in the GEMM stage, clocks/temperature, E1/E2 of every method, and whether the
a-priori configuration matched. Conversion, reconstruction and the statistics pass are inside the timed region; nothing
is amortised across calls (a persistent-operand variant is described, not claimed).

### 1.10 Gates and rigor labels

Before the race: a pre-run code review of `code/ozaki_engine.py` and the harness by a fresh subagent with no shared context.
After the race: an evidence gate against raw outputs by a fresh subagent. Timings EMPIRICAL (this machine). Lemma L1 and
the range bound are PROVED only after the independent check; the a-priori rule's coverage of the true maximum over all
(i, j) is HEURISTIC (probe set), checked empirically. Distance from claim named in each unit opening: 1 (engine) or 2
(race).

## 2. Method specification and implementation notes

Distance from claim in this section: 1 (engine) and 2 (measurements of it). Rigor labels: EMPIRICAL unless stated.

### 2.0 Dated amendments (2026-09-30; the hashed text of section 1 is unchanged; each amendment follows the pre-run review or a coordinator rule)

A1 (vendor baseline strengthened; independent pre-run code review, major finding M2). Besides the registered grid the screening now varies the
   user workspace {library-managed default, 2 GiB, 8 GiB}, `SetEmulationSpecialValuesSupport(NONE)` (inputs are finite; the own engine falls back
   on non-finite input), and, as oracle-class variants, fixed mantissa b in {40, 48, 56, 64} and a dynamic mantissa offset {-8, -16, -24}. 27
   variants per cell. Class "a-priori" = the library's own dynamic rule with static settings; class "oracle" = needs a calibration against the
   native error. V*dyn = fastest carried a-priori variant that meets the contract (Vd, the out-of-the-box call, is always carried; variants that
   fall back to native meet the contract trivially at native speed and are included, so V*dyn is never slower than native); V*fix = fastest
   carried oracle-class variant. The carried set is Vd + the 3 fastest other a-priori variants + the 2 fastest oracle variants. The retained-mantissa-bits
   pointer (`cublasSetFixedPointEmulationMantissaBitCountPointer`) is logged: it reports the bits the library RECOMMENDS (62 for G, 91 for S, 80 for W
   at n = 2048), not whether it fell back (a default-cap fallback shows only in the time; probe `bits_probe`, exploratory).
A2 (thermal gating; drift rule unchanged). The laptop GPU throttles from 2392 MHz to about 1800 MHz within ~10 s of sustained load at 87-89 C, and
   a debug race at n = 8192 without gating had 25 of 30 rounds invalid and the same kernel varying 6x within a round. Every round (warm-up rounds
   included) and every screening pass therefore starts only after NVML reports <= 74 C (at most 150 s wait); a cool-start probe showed native
   n = 8192 constant at 1.60-1.66 s for the first three calls and the own engine at 0.088-0.10 s. This replaces the 15 s soak of D4. Numbers are
   therefore COOL-START, BOOST-REGION numbers: a round still heats from <= 74 C to 78-85 C, rounds that heat through the 5% band are discarded by the registered
   rule, and the behaviour under sustained throttled load (clocks 700-1800 MHz) is not what the race measures; its effect on rho is unknown, so one UNGATED
   sensitivity cell is run and labelled after the race. NVML clock/temperature samples are coarse (identical for all calls of a round, refreshed on a
   ~0.1-1 s scale) and are context only; the regime witness is the native pre/post time against the nominal 0.688 TFLOPS time (`native_med_s / native_nominal_s`
   per cell), and rounds whose gate temperature was not reached are counted (`n_gate_missed`). A tiny primer kernel follows the native pre-call: the first
   submission after a long kernel was delayed by 30-90 ms in about 60% of debug rounds (WDDM scheduling; position-0 median ratio to the method median 3.73 without
   the primer, 0.99 with it). The race began with code sha256 prefix 107607450c87 (`code_sha256` is in every record). INVALID cells are re-run and appended, so an
   evidence gate must use the latest record of a cell.
A3 (wording of the contract; coordinator rule 2026-09-30). The a-priori rule is a SAMPLED (statistical) contract: Lemma L1 bounds the error at
   EVERY entry, but the comparison with tau |A||B| is evaluated on a probe set of about 9k of the n^2 pairs, so nothing in the timed path is a guarantee
   of E_ij <= tau D_ij at every entry. Two rigorous companions exist. (i) The data-independent scaled contract of the configuration table (section 2.5):
   |C_hat - C|_ij <= sigma(p) k 2^(r_i + c_j) + 2^-53 |C_hat_ij| for all finite inputs. (ii) An optional FULL-matrix certificate (`full_certificate`, not
   in the timed path): one FP16 tensor-core GEMM of the scaled magnitudes rounded down gives a lower bound on every D_ij (assumption: FP32 accumulation error of
   non-negative terms below 8 k 2^-24 relative); at n = 2048 it confirms the sampled N in 5 of 5 families (N_full = N_sampled; the sampled required bits were 1 bit
   short for R and W, absorbed by p_max(N)); unoptimised cost 7-12 ms at n = 2048 (the engine takes about 3 ms; an optimised fused version is bounded below by one FP16 GEMM,
   about 10-12% of the engine time).
A4 (baseline / contract / lifecycle statement required before a speedup claim; AGENTS.md section 12, 2026-09-30).
   Baseline: cuBLAS 13.1 DGEMM emulation at its fastest configuration of the 27-variant grid that meets the SAME accuracy contract, with and without its fallback
   behaviour (Vd and V*dyn/V*fix, all carried and timed in the same rounds as the own engine); native FP64 (`torch.matmul`) is the bracket.
   Output contract: C = A B in float64 for square, 32-aligned, finite float64 inputs, alpha = 1, beta = 0. Accuracy: (a) "matched" = E1 and E2 no larger than native's on the
   shared 49 x 49 sample against the correctly rounded reference (sampled, EMPIRICAL); (b) for the own engine additionally Lemma L1 at every entry and the full-matrix
   cross-check against the N = 32 result at n = 2048 (`check`); both sides typically OVER-deliver in the a-priori mode (own about 26x, vendor about 13x more accurate
   than native), so a-priori rows compare contracts, the oracle rows compare calibrated minimal configurations.
   Lifecycle: every timed call includes statistics, residue conversion, N INT8 GEMMs, reductions, reconstruction and the host read of (N, p); nothing is amortised across
   calls (persistent-operand reuse would remove the A/B conversion, about 20-25% of the engine time, and is described, not claimed); buffers are preallocated (own
   3 N n^2 + 4 n^2 bytes with capacity 20 moduli, vendor 2 or 8 GiB workspace); analytic DRAM traffic of the own engine is about (72 + 12 N) n^2 bytes per call
   (statistics 48 n^2, conversion 16 n^2 + 2 N n^2, INT32 products and reductions 9 N n^2, reconstruction (N + 8) n^2; GEMM operand re-reads excluded), i.e. 264 n^2 bytes
   for N = 16; the vendor's traffic is not observable. Results are reported as time, bytes and error, not GEMM count alone.
A5 (configuration as input; coordinator point 4). `run(config={'N':..,'p':..})` and `pick_config(table, k, sigma_target)` read a JSON table
   (`results/ozaki_fair_engine_v1/config_table.json`, schema `ozaki_crt_int8_config_v1`, entries (k, N, p, sigma) for the moduli list above) so that theory (the
   `digit_gemm_complexity_v1` table of certified configurations) can select (N, p). A table p above what the data range allows is refused (native fallback); test
   `python code/ozaki_engine.py config`: 6 of 6 configurations for sigma 2^-40 and 2^-50 on G, X, W meet their guaranteed contract with 0 violations.

A7 (fixed operand; coordinator 2026-10-01). See 3.6: `PersistA`, `persist`; the encode is charged; the kill test (q = 16) was not evaluable under the 5% drift rule; this run used OZAKI_GATE_C=80 because 74 C was unreachable that day.
A8 (host-share trace; decision rule registered 2026-10-01 before the run): host read/dispatch >= 20% of raced wall time -> device-resident dispatch next; < 10% -> reject the host-bottleneck premise; between -> repeat. Result 1.2-1.5%: premise rejected (3.3).
A9 (audit gate; coordinator 2026-10-01): see 3.5; promotion only if lcb95 > 1.25 against the best matched vendor mode; an independent reviewer is to audit `cmd_audit`, `cmd_persist`, `cmd_trace` and the raw records.

A10 (certified mode; coordinator 2026-10-01; dated 2026-10-01, registered BEFORE the price run, after the correctness run of `certify`). Mode `run(graph=True, cert=True)`: the plan (N, p) is chosen on the device with NO sampled index. Assumptions, stated: finite inputs (else native fallback); exact CRT stage
   (2 Xmax(p) + 1 <= prod of moduli, INT32 products exact for k <= 131071); quantisation by round-half-away (|delta| <= 1/2); final conversion correctly rounded; contract |C_hat - C|_ij <= tau D_ij with tau = 2^-52 and D = |A||B| in exact arithmetic, split as E_ij(p) <= tau D_ij / 2 (Lemma L1, all entries) plus 2^-53 |C_hat| <= tau D / 2.
   D_ij is bounded BELOW on every entry by two exact INT8 GEMMs of the scaled magnitudes rounded DOWN to two 7-bit digits (floor(127(1-2^-50)x) twice; the A2 B2 term dropped; int32 sums exact; derated by 2^-21 for the FP32 conversion), O(n^3 / 8) extra INT8 work plus O(n^2) elementwise work; p_req from the exact quadratic in 2^-p over all n^2 entries; entries with lower bound 0 cannot be certified -> native fallback.
   Pre-run prediction (price): N_cert = N_sampled on G (16); extra end-to-end time of certified over sampled mode 30-45% at n = 4096 (two extra INT8 GEMMs of K = k and 2k = 3/16 of N = 16 GEMM work plus about 8 n^2-sized elementwise passes); W: lower bound too weak, N_cert > N_sampled or fallback. H0: certification costs <= 25% extra time and no entry violates; H1: it costs more. KILL: extra time > 25% vs the sampled rule, or any entry violating its bound. P(kill by cost) = 0.6.

### 2.1 Implementation (what runs; `code/ozaki_engine.py`, one module, boilerplate ctypes wrappers written by the executor and reviewed)

NVRTC-compiled kernels (sm_120), cuBLASLt TN INT8 GEMM (13.1) and torch buffers. Timed call (`run_fast`, CUDA graphs): graph 1 = statistics
(6 kernels) + the a-priori rule on the device (probe D^ by a small FP64 GEMM, closed-form solution of E <= tau D^/2 for the required bits
p_req, closed-form p_max(N), N* = smallest N with p_max(N) >= p_req) -> one host read of (N, p) -> graph 2 = residue kernels for A and
(transposed) B, N x [cuBLASLt INT8 GEMM + reduction mod p_t to uint8], Garner mixed-radix reconstruction in 32-bit limbs with one
correctly rounded conversion. Buffers are preallocated per size (capacity 20 moduli, grown on demand). Fallback to native FP64 on
non-finite input or when N* > 32.

### 2.2 Disclosed differences from the frozen text (none changes a decision rule)

D1. Probe set: the union grid I x J with I = 64 random + 32 hardest rows, J likewise (about 96 x 96 pairs) instead of 4096 random pairs +
    a 32 x 32 hard grid (gather efficiency; a superset in coverage). The 32 hardest rows/columns are by 2^r k / s (spikiest first).
D2. The device rule uses p_req = ceil(max(-log2 x_max) + 1e-9) from the exact quadratic solution instead of an integer search
    (conservative by at most one bit; equal to the host integer search in every audited cell: n = 1024, 2048, five families).
D3. Lemma L1 audit adds the rounding of the correctly rounded reference: |C_hat - C_ref| <= E + 2^-53(|C_hat| + |C_ref|) (a first audit
    without that term flagged 1-2 spurious violations in X and W).
D4. Rounds in the first 15 s of a block (soak) are discarded in addition to the 2 warm-up rounds; the oracle scan starts at N* - 6.
D5. Vendor handles: `SetWorkspace` is not called at all for the "ws0" variants (a call with (NULL, 0) forces the native fallback; found in a
    debug run, before any evidence run). Vendor "ws1" variants share one 2 GiB user workspace.
D6. INT8 GEMM path (protocol 1.8): torch col-major, cuBLASLt and GemmEx have overlapping IQRs at every n; cuBLASLt is used (fewer layers,
    graph-capturable). cuBLASLt returned exactly one heuristic algorithm, so no autotune is possible.

### 2.3 Task 2 result: INT8 throughput diagnosis (`results/ozaki_fair_engine_v1/int8_diag.json`, 11 interleaved random-order rounds, loaded state)

| n | FP16 (fp32 acc) | FP8 e4m3 | `_int_mm` row-major B | `_int_mm` column-major B | cuBLASLt TN | GemmEx INT8 | NVML SM clock |
|---|---|---|---|---|---|---|---|
| 2048 | 27.5 | 42.9 | 46.6 | 116.0 | 96.0 | 102.7 | 1402 MHz |
| 4096 | 42.4 | 79.8 | 43.4 | 140.0 | 140.6 | 140.0 | 1087 MHz |
| 8192 | 29.4 | 51.7 | 29.1 | 110.2 | 110.0 | 110.9 | 697 MHz |

(T-op/s; datasheet 254 TOPS at 1.515 GHz.) D1 holds: the row-major-B path of the previous unit was 2.4-3.8x slower than the same kernel with a
column-major B; INT8 is 3.3x FP16 and 1.7-2.4x FP8 at n >= 4096. The bare mma.sync.m16n8k32 s8 loop reaches 401 TOPS at NVML SM clock 2392 MHz
= 2046 op/clk/SM (nominal 2048); FP16 (fp32 acc) 100.5 TFLOPS; e4m3 201 TFLOPS (ratios 4 : 1 : 2). In sustained GEMMs the limiter is the SM clock
(697-1402 MHz at 85-88 C, software thermal slowdown), and cuBLASLt sits at 77% (n = 4096) and 94% (n = 8192) of the clock-scaled peak
82 x 2048 x f. INT8 peak per clock is 512x the FP64 peak per clock (168k vs 328 op/clk).

### 2.4 Task 3 result: correctness (`selftest.json`, `check_1024.json`, `check_2048.json`)

Integer exactness: 48 cases (N = 8, 14, 20, 30; p = p_max and p_max - 9; shapes 64^3 and 32 x 96 x 64; Gaussian, graded and "wide" data
with zeros, subnormals and 2^+-90 exponent ranges) reproduce the Python big-integer product bit-for-bit; device statistics equal a numpy
reference; graph replay equals eager. Accuracy (n = 2048; native E1/E2 -> cert configuration): the a-priori configuration matched native in all
five families and both sizes.

| family | cert N (p) | oracle N | cert N without inside scaling | Lemma L1 audit (violations; median bound/err) |
|---|---|---|---|---|
| G | 16 (59) | 14 | not run | 0; 38 |
| X | 18 (68) | 16 | 19 | 0; 4.5 |
| S | 16 (59) | 14 | 23 | 0; 38 |
| R | 16 (58) | 15 | not run | 0; 53 |
| W | 20 (74) | 18 | 20 | 0; 15 |

The S family after inside scaling is bit-identical to G in N, p and errors (inside scaling exactly undoes the similarity). The price of the
certificate is 1-2 moduli (about 7-14% of the GEMM work) over the oracle N.

### 2.5 Configuration table (theory-facing interface; `results/ozaki_fair_engine_v1/config_table.json`, `config_test.json`)

Schema `ozaki_crt_int8_config_v1`: for each inner dimension k in {1024, ..., 16384} and each N in 6..32 the largest data-independent p (nu_A = nu_B = sqrt(k),
the worst case of the range bound 2 Xmax(p) + 1 <= prod(moduli[:N])) and the guaranteed scaled contract sigma(p) = 2^-p (1 + 2^(-p-2)):
|C_hat - C|_ij <= sigma k 2^(r_i + c_j) + 2^-53 |C_hat_ij| for ALL finite inputs (Lemma L1 with s^A_i < k 2^r_i and s^B_j < k 2^c_j; PROVED-pending-gate).
Example (k = 8192): N = 15 -> p = 53, N = 16 -> p = 58 (sigma 2^-58), N = 12 -> p = 41. This contract is relative to the row/column SCALES, not to the componentwise
|A||B|; it is what a theory unit can certify without seeing the data. The componentwise contract needs the data (sampled rule) or the full certificate.

## 3. Results (race, protocol section 1.6; filled after the run)

(Dated 2026-10-01; all timings EMPIRICAL, this machine, cool-start boost region (amendment A2); raw: `results/ozaki_fair_engine_v1/race_results.jsonl` (latest record per cell), `race_run*.log`, `audit_results.jsonl`, `trace_results.jsonl`, `persist_results.jsonl`; one-command replay `results/ozaki_fair_engine_v1/replay.sh`; table generator `python code/ozaki_engine.py report`.)

### 3.1 Verdict on the registered hypotheses


**EVIDENCE GATE (independent audit, Droid on the owner's default model, 2026-10-01; Stateful Codex's run did not start because its binary was missing; transcript outside the repo).** Verdict: the 4096-cell race results are SUPPORTED as cool-start, sampled-accuracy measurements; no invalid rounds entered the reported 4096 decisions; headline numbers recomputed from the JSONL match (e.g. 4096 S own/native 13.47x, 4096 W paired rho 0.4546 [0.4307, 0.4588], 8192 R own/native 16.67x, audit G nominal lcb 1.0697). Findings applied or recorded: (1) MAJOR - the measured code is not reproducible exactly: records carry code hashes 107607450c87 (2048/4096) and aa51d99f33e6 (8192), neither is a committed version, and the committed file hashes to 16c1b0ed900b; the exact measured sources were NOT preserved; the 8192 split-bracket cells (amendment A6) are an AMENDED, not registered, protocol; the five 4096 H1a decisions do not depend on it; RULE from now on: commit the code before any run so hash = commit. (2) MAJOR - 'best of 27' narrowed to 'fastest carried variant after accuracy screening' (only about six variants were timed in the main rounds). (3) MAJOR - the headline now separates the calibrated-fixed range (about 1.2-1.7x) from the dynamic-rule range (about 1.7-3x); at 4096 G and R the calibrated margin is 1.23x and 1.21x. No audit promotion is supported (every audit block failed its drift check; three-block bounds are indicative only). (4) MINOR - q = 2 and q = 8 for non-G families were NOT RUN, not INVALID; the q = 16 kill test is unevaluable; the fixed-operand product computes a fallback flag for later operands but does not act on it, and first/last sampled checks do not establish accuracy of every intermediate product. (5) MINOR - the report command prints every appended row; deduplicate by (n, family) before quoting. 'H0 rejected' should read 'the registered seven-round median/IQR rule is met'. UNSUPPORTED by these records: an unchanged-protocol 8192 win, a hot sustained-load win, a passed 1.25x promotion, guaranteed full-matrix accuracy in the timed path. The full-matrix reference is the engine's own N = 32 result, not an independent correctly rounded oracle.

Registered race, 15 cells (n = 2048, 4096, 8192; G, X, S, R, W), drift rule as registered: 12 cells VALID, 3 INVALID(drift) and excluded (8192 S and 8192 W: bracket 1 never reached 7 valid rounds in two tries; 8192 G under the registered single bracket: 60 of 60 rounds failed, median native drift +9.1% from self-heating, so n = 8192 used split brackets, amendment A6, and its G cell is VALID there). n = 16384 not run (time).
**H0 is rejected for n >= 4096 and survives at n = 2048.** H1a (a-priori own engine more than 10% faster than V*dyn, non-overlapping IQRs, >= 7 valid rounds) holds in all 5 cells at n = 4096 and in the 3 valid cells at n = 8192, in none at n = 2048 (rho 0.92 G, 0.70 X, 0.51 S, 1.01 R, 0.64 W, but IQRs overlap: host launch/sync jitter of a 3 ms call under ~90% ambient CPU). H1b (S/W: the out-of-box call falls back to native while own > 1.3x native) holds at n = 4096 (S, W); the raised-cap/workspace vendor variants do NOT fall back, so H1b is a statement about the out-of-box call only.

| n | fam | N cert (p) / oracle N | Vd (out of box) | V*dyn | V*fix (oracle class) | own_cert | own_oracle | rho cert/V*dyn med [q1,q3] | H1a |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | G | 16 (59) / 14 | 3.17 | 6.46 | 8.15 | 7.02 | 9.63 | 0.92 [0.91,1.21] | no (IQR overlap) |
| 2048 | X | 18 (68) / 16 | 2.55 | 4.63 | 6.49 | 6.54 | 8.77 | 0.70 [0.70,0.99] | no |
| 2048 | S | 16 (59) / 14 | 0.82 | 3.43 | 3.95 | 6.78 | 9.37 | 0.51 [0.49,0.96] | no |
| 2048 | R | 16 (58) / 15 | 2.16 | 6.51 | 8.28 | 6.43 | 8.90 | 1.01 [0.97,1.35] | no |
| 2048 | W | 20 (74) / 18 | 0.86 | 3.99 | 5.64 | 6.21 | 7.91 | 0.64 [0.64,0.97] | no |
| 4096 | G | 16 (58) / 14 | 6.07 | 8.51 | 10.76 | 13.21 | 16.30 | 0.66 [0.60,0.66] | YES |
| 4096 | X | 18 (67) / 15 | 0.95 | 5.05 | 8.67 | 12.58 | 15.80 | 0.40 [0.39,0.41] | YES |
| 4096 | S | 16 (58) / 14 | 0.95 | 4.36 | 5.01 | 13.47 | 16.70 | 0.32 [0.24,0.33] | YES |
| 4096 | R | 16 (58) / 15 | 5.62 | 8.61 | 11.16 | 13.52 | 15.56 | 0.64 [0.60,0.69] | YES |
| 4096 | W | 20 (74) / 18 | 0.95 | 5.10 | 7.17 | 11.24 | 13.53 | 0.45 [0.43,0.46] | YES |
| 8192 | G | 16 (58) / 14 | 6.23 | 8.77 | 10.45 | 17.04 | 20.96 | 0.51 [0.39,0.55] | YES (split brackets) |
| 8192 | X | 18 / 15 | - | 4.70 | - | 13.92 | 18.89 | 0.33 [0.30,0.36] | YES (split brackets) |
| 8192 | R | 16 / 15 | - | 8.60 | - | 16.67 | 18.64 | 0.52 [0.49,0.53] (unpaired) | YES (split brackets) |
| 8192 | S, W | | | | | | | INVALID(drift) | excluded |

Speedups are medians of per-round ratios to native `torch.matmul` (native/nominal 1.01-1.05 everywhere, i.e. the cool 0.688 TFLOPS regime). The own timed call = statistics + rule + residues + N INT8 GEMMs + reductions + reconstruction + one host read of (N, p), inputs and buffers resident; V*dyn and V*fix are the fastest carried variants of the 27-variant grid (A1), chosen by screening (cost in 3.4).

### 3.2 Accuracy (precision-contract wording; rigor labels)

- Matched accuracy (E1, E2 no larger than native's on the shared 49 x 49 sample against the correctly rounded reference) is met by the a-priori configuration in all valid cells (EMPIRICAL, SAMPLED). In the a-priori mode both sides over-deliver: own is 20-1000x more accurate than native (E2 0.00-0.04 x native), the vendor dynamic rule about 13x; the oracle columns (own_oracle vs V*fix) compare minimal calibrated configurations and are the "matched accuracy" comparison proper.
- The a-priori rule is a SAMPLED (statistical) contract, never a guarantee (A3). RIGOROUS parts: Lemma L1 holds at every entry (PROVED-pending-independent-gate; the code reviewer re-derived it, found it attained with equality, 0 violations in exact-rational simulations, 0 violations in our audits); the data-independent scaled contract of the configuration table (6/6 test configurations, 0 violations). CERTIFIED for ALL entries only when the optional full-matrix certificate is run (`full_certificate`; n = 2048: N_full = N_sampled in 5/5 families, 0 uncertified entries; assumption on tensor-core FP32 accumulation stated; unoptimised cost 7-12 ms, not in the timed path).
- Full-matrix checks: at n = 2048 the a-priori result is no worse than native over all entries (5 families). In the audit at n = 4096 the full-matrix componentwise E2 of own_cert is 2.2e-17 (G), 3.0e-17 (W), 1.7e-16 (X) against native 6.8e-16, 1.1e-14, 1.2e-14; but the SAMPLE-matched oracle configurations are NOT always matched over the whole matrix: X own_oracle E2 1.7e-14 and vendor off-24 E2 7.0e-14 exceed native's 1.2e-14. The reference of full-matrix numbers is our own N = 32 result (quantisation error about 2^-100, CRT stage bit-exact by the selftest), not an independent library.
- The vendor's retained-mantissa recommendation is 62 (G, R), 79-81 (X, W), 91 (S) bits, i.e. Ozaki-I with about 8-12 slices (36-78 INT8 GEMMs) against 16-20 moduli: the advantage is structural (GEMM count), not a better kernel.

### 3.3 Cost accounting and breakdown (time, bytes, error)

Warm eager profile of the a-priori configuration (cell records `own.cert.prof_s`; the 2048 G profile was taken on a cold GPU and is unreliable). n = 4096, N = 16: statistics 2.25 ms (15%), residues 1.62 ms (11%), INT8 GEMMs + reductions 10.2 ms (68%; 216 TOPS in the stage, about 54% of the 401 TOPS bare mma loop), reconstruction 0.88 ms (6%); total 14.9 ms. n = 8192: 13.1 / 10.1 / 108.8 / 5.8 ms (GEMM stage 79%). By the registered rule GEMMs are 68-79%: product count matters; the O(n^2) stages (21-32%) run at about 420 of 896 GB/s. Analytic DRAM traffic (72 + 12 N) n^2 bytes (4.4 GB at n = 4096, N = 16). Buffers 3 N n^2 + 4 n^2 bytes (1.0 GiB at 4096, 4.0 GiB at 8192 with capacity 20) against the vendor's 2 or 8 GiB user workspace. Nothing is amortised across calls (3.6).
Host-bottleneck premise RETIRED (A8): trace of the TIMED path at n = 4096: stats/rule graph 3.1 ms, core graph 10.9 ms (G) / 13.7 ms (W), host read + sync + dispatch share of wall time 1.5% (G) and 1.2% (W) (rule: < 10% rejects the premise); end to end 14.3 / 17.6 ms against V*dyn 22.1 / 40.0 ms. The 17.8 ms "stats+rule_wall" of the eager profile path is not a saving of the timed path.

### 3.4 Selection charge

V*dyn and V*fix come from screening 27 variants (about 100 vendor calls: 2.8 s of GPU at G, 11 s at W and X, plus the calibration against native error); the own oracle N costs 7 scans of 12-30 ms (0.09-0.19 s). The a-priori own configuration needs no calibration.

### 3.5 Audit gate (A9; `python code/ozaki_engine.py audit`, n = 4096; G, W, X; separate evidence)

Contenders: native, best vendor DYNAMIC and best vendor FIXED/offset variant (from the race record), own a-priori, own oracle; random sweep order, cooled blocks of 15 s, per-call timing, full-matrix errors. RESULT: the audit stopped early under its own rule (every block "unreliable": native drifts 14-56% inside a block because a 15 s block heats the GPU; contenders are interleaved, but the pairing criterion failed), so only 3 blocks per family exist and the confidence bounds (t, df = 2) are INDICATIVE. Speedup of own_cert vs best vendor modes (mean / lcb95): G 1.67 / 1.37 vs dynamic, 1.19 / 1.07 vs fixed b48; W 2.60 / 2.21 and 1.65 / 1.39; X 2.81 / 1.72 and 1.53 / 1.39. own_oracle: G 2.14 / 1.66 and 1.52 / 1.39; W 3.17 / 2.80 and 2.01 / 1.85; X 3.51 / 2.81 and 1.91 / 1.39. The promotion rule (lcb95 > 1.25 against the best matched vendor mode) is met by own_oracle in all three and by own_cert in W and X, NOT by own_cert in G (1.07). Because the audit's own pairing check failed, NO promotion is claimed.

### 3.6 Fixed operand (A7; `persist`, n = 4096)

A is encoded once at the first product's (N, p) with inside scaling off; the encode and every decode are charged. q = 1: persistent / non-persistent own 0.88 (G), 0.94 (X), 0.87 (W), 0.89 (R), 1.23 (S: inside scaling off costs 7 moduli); persistent / vendor 0.37-0.59 (part of the q = 1 gain is the removed inside-scaling pass, not reuse). q = 2: 0.84; q = 4: 0.77 (G is the only valid q = 4 cell). q = 8, 16 and the other families' q = 2, 4 were INVALID under the 5% drift rule or NOT RUN (evidence gate: q = 2 and q = 8 for families other than G were not run) (long streams heat the GPU), so the registered kill test (q = 16 at least 10% faster, non-overlapping IQRs) could NOT be evaluated. The reuse gain is bounded by the A-side encode share (about 20%); measured 12-23% at q <= 4. First and last products matched native accuracy (sampled).

### 3.7 Honest headline and so-what

At n >= 4096, cool-start, finite square FP64 inputs, an own CRT/INT8 engine delivers FP64-class output about 1.2-1.7x faster than the fastest carried CALIBRATED fixed-mantissa vendor mode (4096: 1.21x on R and 1.23x on G, higher on W and X; audit G 1.19x with nominal lower bound 1.07) and about 1.7-3x faster than the vendor's own dynamic rule, where the 27-variant grid was accuracy-screened and only the fastest carried variants were timed in the main rounds and 11-17x faster than native FP64 on this GPU, with sampled (not guaranteed) componentwise accuracy 20-1000x better than native. It is an engineering win from using about 2.5x fewer INT8 GEMMs (CRT instead of Ozaki-I) with a data-driven number of moduli; it does not change the exponent, needs no new arithmetic, relies on a laptop GPU whose FP64 is 1/64 of FP32, and the margin against a calibrated vendor mode is small (about 1.2x). Importance verdict (AGENTS 4.2): WEAK as a standalone headline. Next: own INT8 GEMM with a fused modular-reduction epilogue (removes about 9 N n^2 bytes, 2-3 ms at n = 4096), fused conversion kernels (non-GEMM stages at about 47% of DRAM peak), a certificate inside the timed path, a hot-state sensitivity cell (not run), n = 16384 (not run).

### 3.8 Certified mode (A10; code = commit 3d8b553c; `certify`, `price`; raw `results/ozaki_fair_engine_v1/certify_results.json`, `price_results.jsonl`)

CORRECTNESS (EMPIRICAL on the registered checks; derivation of E_ij(p) and the range bound = Lemma L1, PROVED-pending-independent-gate; the lower bound on D is elementary: floor-rounded digits, exact int32 sums, FP32 derating): full-matrix check of ALL entries against the contract |C_hat - C| <= tau D (tau = 2^-52): G, X, Z (cancellation-heavy: A = [Ah, Ah + 2^-40 dl], B = [Bh; -Bh]) at n = 256 (reference = correctly rounded exact result of every entry),
1024 and 2048 (reference = N = 32 engine result, quantisation ~2^-100): 0 violations of tau D and 0 violations of Lemma L1 at any entry in all 9 (family, n) cases; max err / (tau D) 0.10 (G), 0.64-0.79 (X), 0.001-0.002 (Z). Negative control: N_cert - 5 moduli (N = 11-13) violates tau D at 65 thousand to 4.2 million entries, caught in 9/9. For contrast native FP64 violates the same contract at 247 (G, n = 256) up to 1.9e6 (X, n = 2048) entries (the contract is stricter than native, by design).
LIMIT: wide-range W (2^20 within rows) is NOT certifiable by the two-digit lower bound (entries with lower bound 0 remain, so the mode falls back to native FP64), at n = 256, 1024, 2048 and 4096. The certified N equals the sampled N in every certified case (G 16, X 17-18, Z 16), i.e. no extra moduli.
PRICE (n = 4096, 3 cooled blocks of 8 s, paired, interleaved with native): G certified / sampled end-to-end time 1.90, 1.97, 1.98 (+95%, 27.4-29.6 ms vs 14.4-15.0 ms); W falls back to native (222-238 ms vs 17.5 ms, x13). KILL RULE TRIGGERED (> 25% extra time): the implemented certificate is too expensive; the cost is the n^2 elementwise torch passes (digit extraction, FP32 quadratic, log2: about 13 ms), not the two INT8 GEMMs (about 1.2 ms). The prediction "30-45%" was too low.
Next (HEURISTIC, not measured): one fused kernel doing digit extraction, the lower-bound GEMM epilogue and the quadratic would leave about two INT8 GEMMs plus one n^2 pass (about 10% of the engine at n = 4096); a third digit for W. Until then the timed path remains the SAMPLED contract; the certified mode is a verification/offline tool, not a speed result.


## A11. Fused all-entry certificate pilot (registered 2026-10-01 BEFORE any run; meditation: Stateful Codex, Droid default model, Antigravity agree on the design; the 'CORRECT' wording of A10 is EMPIRICAL on 9 full-matrix cases, no proof gate yet; 'no extra planes' holds only for the tested G/X/Z cells (W fell back); A10's 'all finite inputs' is wrong at the subnormal boundary)

Design (Codex/Droid): kernel 1 per operand reads FP64 once and emits the packed INT8 digit planes directly (no xa/xb/digit temporaries); the two exact INT8 lower-bound GEMMs stay; kernel 2 reads the INT32 outputs and tests E(p) <= tau*D_lower/2 against the finite precomputed p(N) table by binary search (D_lower rounded DOWN, E rounded UP; no log2/sqrt), block-reduces the maximum required N and an uncertifiable counter; tile fast-accept (tile-minimum D_lower vs tile-maximum error coefficient; failed tiles still get entrywise tests); for wide-range W an adaptive extra-digit/exponent-binned lower bound only on failed tiles with a work cap, then native fallback. DOMAIN: entries whose tau*D_lower falls below the subnormal spacing (2^-1075 allowance) are counted as UNCERTIFIABLE (never silently certified); the guarantee is stated for the normal-output domain.
Pre-run predictions (n = 4096, G and W, interleaved cooled blocks, budget about 2-4 GPU-minutes, roughly 80-140 W): H0: fused certified time <= 1.25 x the sampled mode (overhead <= 25%) on G; H1: <= 1.12 x. Kill rule: if fused overhead > 25% on G, or any entry violates tau*D (checked by the existing `certify` suite at n = 256/1024/2048 including the N-5 negative control, which must still be caught), or any rounding direction is wrong, the fused design is abandoned and tile-level bounds are tried next. 'Under 10%' is a HEURISTIC hope, not a prediction.


## A11 RESULT (2026-10-02; implemented by the implementer, lead-reviewed; EMPIRICAL, NOT yet independently gated)

Fused all-entry certificate (two single-pass extraction kernels writing packed INT8 digit planes, the two exact INT8 lower-bound GEMMs, one check kernel with downward-rounded D_lower, upward-rounded E, binary search over the p(N) table, subnormal domain counted as uncertifiable): the `certify` suite run on the FUSED path gives 0 violations of tau*D and of Lemma L1 at every entry for G, X (wide range) and Z (cancellation-heavy) at n = 256/1024/2048, the N-5 negative control is caught in all 9 cases, and wide-range W still falls back to native (uncertifiable; the recorded 'preq' diagnostic for W changed from 78-80 to 0 with the new path). Price, n = 4096, family G, 3 interleaved cooled blocks (native 204-210 ms, so thermally steady): sampled 14.3-15.4 ms, old certified 27.4-28.3 ms (1.89x of sampled), FUSED 14.3-16.0 ms = 1.017x of sampled (about +2%). Code hash e036866b... (record fused_cert_results.jsonl). Pre-registered H1 (<= 1.12x) MET, kill rule (> 1.25x) NOT triggered. Caveats: one family timed, 3 blocks (indicative), no tile fast-accept/W adaptive digits yet, reference for n >= 1024 is the engine's own N = 32 result, certificate proof still 'independent re-check pending'. Consequence for the headline: an all-entry certified mode at about sampled-mode cost would turn 'fast with sampled accuracy' into 'fast with a per-entry guarantee on G/X/Z-type inputs' (native FP64 itself violates the same contract at 247 to 1.9M entries). NEXT: independent evidence gate on the timing + kernels; time X and Z at 4096; W adaptive digits.


## A11 CORRECTION (2026-10-02, independent evidence gate: Droid on the owner's default model and Stateful Codex, both FAILED the gate; supersedes the A11 RESULT price claim)

The '+2%' price of the fused certificate (fused 1.017x sampled) is RETRACTED: run_fast's CUDA-graph cache key omitted `fused`, and cmd_price created the sampled graph first, so every timed `fused=True` call replayed the SAMPLED graph (no extraction, no certificate GEMMs, no check). The earlier unfused certified price (+89 to +95%) stands. Further findings, all accepted: (1) the error upper bound E is not rigorously upward-rounded because sA/sB are accumulated with round-to-nearest additions/atomicAdd (needs directed accumulation or a proved inflation factor); (2) the N-5 'negative control' uses force_N, which bypasses the fused selector, so it tests the arithmetic, not the detector (needs a control feeding the bad candidate to cert_check_k); (3) G2 can reach 2k*126^2, overflowing INT32 for k > 67,633 (needs a guard/fallback); (4) the subnormal cutoff test must reject exponent <= -1075; (5) the kernel scans 27 candidates linearly, not a binary search. SURVIVED: the lower-bound derivation for D_lower (valid inside the exact-accumulator domain) and the Acat/Bcat layout. 'Zero observed violations on G/X/Z' remains an EMPIRICAL observation (sampled-path products), NOT evidence for the fused detector. A rigorous per-entry guarantee is UNSUPPORTED until these are fixed and independently proved. Lesson recorded: a timing win that is much better than the mechanism predicts must be checked for skipped work before it is written down (verify captured kernels per mode).


## A12 RESULT (2026-10-02; fixes for all A11-correction findings applied by the implementer, lead-reviewed; EMPIRICAL, NOT yet independently re-gated)

Fixed: graph cache key now includes fused/cert (distinct graph ids recorded per mode); sA/sB use a rounded-up inflation allowance; detector-level control added to `certify` (positive fires on a truncated p(N) table, negative does not fire at the correct table: caught in all rows); INT32 guard for large k; subnormal boundary `<=`. `certify`: 0 violations of tau*D and of Lemma L1 at every entry for G, X, Z at n = 256/1024/2048, W falls back. HONEST PRICE (n = 4096, G, 3 cooled blocks, native 217-231 ms, distinct captured graphs): sampled 15.2-15.6 ms, old unfused certified 28.2-30.4 ms (1.89x sampled), FUSED 22.2-24.1 ms = 1.50x sampled (+50%). Pre-registered H1 (<= 1.12x) NOT met; kill rule (> 1.25x) TRIGGERED for the fused design as specified: the fusion cut the certificate overhead from about +89% to about +50% but not below the line. The earlier '+2%' was the retracted graph-cache artifact. Method change (registered next, not a stop): (a) per-kernel breakdown of the remaining ~7 ms (two extra INT8 GEMMs over 2k, extraction passes, the 27-candidate check) to see what is irreducible; (b) tile fast-accept using tile-level bounds so most entries skip the exact check; (c) reduce to one extra GEMM with a coarser (single-digit) lower bound where the margin allows; (d) decide whether the certified mode is offered as an OPTIONAL mode ('+50% for a per-entry guarantee') rather than the default.


## A13 (2026-10-02): kernel-optimisation attempt REJECTED and reverted; diagnosis and next design

The implementer's attempt to speed up the fused certificate (block-reduced atomics, coalesced B extraction) was reviewed and REVERTED: its price record was impossible (certified 13.4 ms FASTER than sampled 14.4 ms, so the certified path was not doing the full work or was timed wrongly), the check kernel got slower in the new breakdown (6.4 ms vs 4.9 ms), no bitwise-equality evidence was recorded in the result files, and an unrelated result file (int8_diag.json) was silently rewritten. The extraction-B improvement (1.5 -> 0.7 ms) is plausible but unverified and is not kept. DIAGNOSIS (lead, arithmetic): cert_check_k runs ~150 double-precision operations per entry (directed divisions, multiplies, ldexp, up to five loop iterations) over 16.7M entries = ~2.5 GFLOP on a GPU whose FP64 rate is 0.69 TFLOPS (1/64 of FP32): the check is FP64-THROUGHPUT-bound (about 3.6 ms), not atomic- or bandwidth-bound; this also explains the 27 GB/s. NEXT DESIGN: (1) tile-level fast accept: one integer reduction pass (int64 min of 127*C32+G2 per tile, bandwidth-bound, no FP64) plus per-row-tile/column-tile bounds of F1 (O(n) work), a single FP64 evaluation per tile, entrywise exact check only for failing tiles; (2) alternatively move the per-entry arithmetic to integer/FP32 with a proved margin. Pre-registered prediction: certificate overhead <= 20% of the sampled mode at n = 4096 G; kill: > 25%.


## A13 TILE RESULT AND GATE (2026-10-02; independent evidence gate: Droid on the owner's default model and Stateful Codex; both: soundness algebra holds, evidence incomplete)

Tile-level all-entry certificate (integer tile-min pass, row/column tile maxima rounded up, one directed-rounding test per tile, entrywise fallback only for failing tiles): n = 4096 G, 3 cooled blocks, distinct graph ids, nesting sampled <= tile <= fused <= unfused respected: sampled 14.61-15.18 ms, TILE 17.28-18.30 ms (geometric mean 1.179x of sampled, +18%), old fused 22.0-23.2 ms (+50%), unfused 27.4-28.8 ms (+89%). Pre-registered prediction (<= 20%) met, kill line (25%) not triggered. GATE FINDINGS (accepted): (1) SOUNDNESS: no tile-pass/entry-fail counterexample found by either reviewer; the tile bound is conservative under the existing assumptions (exact INT8 accumulation, k <= 67,633, directed rounding, minimum-exponent cutoff). (2) The `certify` zero-violation records and the detector controls validated the FUSED output, not the TILE output (the tile output was discarded): tile-specific correctness is UNSUPPORTED until tile output is checked at every entry. (3) Tile-vs-entry required-N was compared only at n = 256/1024 and not recorded; N_tile >= N_entry is not proved by the construction (separately rounded bounds). (4) The number of failing tiles (fail_cnt) was not recorded, so it is unknown whether the timed G case ran any entrywise fallback; the +18% is a measured ratio, 'fallback-free' is not established. (5) native drift gate used `> 10.0` (1000%) instead of 0.10. (6) fused_breakdown.json was overwritten by an unrelated rerun and is stale. STATUS: +18% is EMPIRICAL as a three-block G timing; the per-entry guarantee for the tile path is UNPROVEN pending the fixes above and a proof pass.


## A14 RESULT: certified tile path vs the vendor, clean blocks (2026-10-02; EMPIRICAL; implementer-run, lead-recomputed; independent re-gate PENDING)

Fixes since the gate: tile output checked at every entry (`certify`: 0 violations of tau*D and Lemma L1 for BOTH the fused and tile paths on G, X, Z at n = 256/1024/2048, W falls back; the previous 4.19M 'violations' for Z at 2048 were a HARNESS RACE: E.run returns before its core graph finishes and the next mode overwrote the shared workspace; fixed by synchronize between modes); detector control passes through the tile path; N_tile >= N_entry in every certified row (X needs more planes: 17 -> 18, 18 -> 19, 18 -> 20 at n = 256/1024/2048); fail_cnt = 0 and fallback_entries = 0 recorded in the timed cases. Price design: 6 short blocks (about 3-4 s), GPU cooled below 55 C between blocks, strict 10% native-drift gate (measured drift 0.6-2.0%, 0 blocks dropped), vendor variants timed in the SAME blocks, distinct graph ids, nesting sampled <= tile <= fused <= unfused respected in all 12 blocks. n = 4096 (geometric mean of per-block ratios; [min, max]): FAMILY G: certified tile / sampled = 1.179 [1.158, 1.205]; fastest calibrated vendor fixed mode (fix_b48, NO per-entry guarantee) / certified tile = 1.033 [1.023, 1.051]; vendor dynamic (dyn_s1_ws8_cap0) / tile = 1.346 [1.317, 1.386]; native FP64 / tile = 11.1x. FAMILY X (wide range): tile / sampled = 1.247 [1.222, 1.271] (tile needs 20 planes vs 18 sampled); vendor off-24 / tile = 1.136 [1.102, 1.173]; vendor dynamic / tile = 1.987 [1.946, 2.024]; native / tile = 9.6x. READING (Pareto frontier, Codex/Droid framing): at n = 4096 the engine WITH a deterministic per-entry certificate is about 1.03x (G) to 1.14x (X) FASTER than the vendor's fastest calibrated mode, which has no such certificate, and 1.35x to 2.0x faster than the vendor's dynamic rule; G is close to a tie, so the honest claim is 'at least matches', not a blanket win. Caveats: one size (4096), two families, cool-start short blocks (not sustained), reference at larger n is the engine's own N = 32 result, W (wide-range rows) still falls back, the soundness argument has not passed a proof gate, the vendor's calibrated mode is calibrated on the sampled accuracy of these inputs.


## A14 RE-GATE (2026-10-02; Droid on the owner's default model and Stateful Codex; no FATAL findings; ratios SUPPORTED; wording narrowed)

Both reviewers recomputed the ratios from the raw records: G tile/sampled 1.1786 [1.158, 1.205], calibrated vendor / tile 1.0333 [1.023, 1.051], dynamic / tile 1.3458; X 1.2470, 1.1362 [1.102, 1.173], 1.9871; native/tile 11.14 and 9.62. NARROWED WORDING (accepted): (1) 'fastest calibrated vendor mode' means the fastest PREVIOUSLY CARRIED accuracy-screened vendor variant (sampled-native-error oracle on the same inputs; calibration cost excluded, which favours the vendor), not an exhaustive vendor optimum; the engine replays CUDA graphs while the vendor contenders are direct cuBLAS calls (whether vendor graph capture narrows the 3.3% G margin is untested); (2) 'zero tile violations' is supported only for n <= 2048 (full-output checks); at n = 4096 the price run checks only fail_cnt = 0 (no tile needed entrywise fallback), NOT numerical error; (3) the 'cooled below 55 C' step is real but its success and the starting temperatures are not recorded in the artifacts; (4) the detector-control calls lack the inter-mode synchronizations (minor). Claims that stand: ratios and same-block ordering; zero recorded failing tiles in the timed cases; faster than the tested carried vendor modes in cool-start blocks. Claims NOT yet earned: a guaranteed per-entry certificate at 4096; 'fastest vendor'. Follow-up delegated: validate the 4096 tile output at every entry, record cooling success and temperatures, add synchronizations to the controls, and time the vendor with graph capture where possible.


## A14 FOLLOW-UP (2026-10-02; implementer run, lead-reviewed; Droid strategy review flagged the invalid columns)

KEPT (EMPIRICAL): the n = 4096 TILE output checked at every entry against the high-precision reference: 0 violations of tau*D and of Lemma L1 for families G and X, plus 0 violations in an independent 256x256 sub-block reference (exact summation) for both. This closes the 'zero violations only to 2048' gap for G and X (reference at 4096 is still the engine's own N = 32 result except for the exact sub-block). INVALID and marked as such in fused_cert_results.jsonl (rows 6-7): the added `vend:*_graph` columns (0.04-0.07 ms per call vs 19-43 ms eager: impossible, the captured vendor call did not run the GEMM) and the price blocks themselves, because wait_cool(55) was not reached (start temperatures 58-68 C, 30 s wait). The A14 numbers (rows 4-5, clean cooled blocks) are unaffected; the question 'does vendor graph capture narrow the 3.3% G margin' REMAINS OPEN. Rule added: a timing column whose value is orders of magnitude below the mechanism's lower bound is invalid until the captured kernels are shown to execute.
