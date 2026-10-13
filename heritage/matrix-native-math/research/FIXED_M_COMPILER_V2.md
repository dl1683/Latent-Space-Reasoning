# FIXED_M_COMPILER_V2: fix the control and chase the real win

Date: 2026-08-02

Status: **CONTROLS PASS; INDEPENDENT EVIDENCE REVIEW PASS; DEPLOYABLE WIN**


## 0. Owned surfaces and precommitted verdicts

Unit id: `FIXED_M_COMPILER_V2`.

Owned surfaces are this file, `research/run_fixed_m_compiler_v2.py`,
`research/verify_fixed_m_compiler_v2.py`, `results/fixed_m_compiler_v2/`,
and append-only V2 rows in `WORKLOG.md` and `experiments/ledger.jsonl`.
The pruning-ideal round, T4 step-two census, their processes, locks, code, and
result trees are excluded and untouchable.

The following tokens are frozen before the runner exists:

```text
PASS_FIXED_M_COMPILER_V2_CONTROLS
KILL_FIXED_M_COMPILER_V2_FALSE_DISPATCH
FIXED_M_COMPILER_V2_DEPLOYABLE_WIN
FIXED_M_COMPILER_V2_NO_DEPLOYABLE_WIN
VOID_FIXED_M_COMPILER_V2_RESOURCE_OR_CORPUS
```

- `PASS_FIXED_M_COMPILER_V2_CONTROLS` requires all V1 identity/apply smoke
  obligations used by V2, exact byte-for-byte recovery of the new planted
  `BCCB(16,32)` matrix, an exact zero representation certificate, executable
  apply validation, automatic selection of `BCCB(16,32), rank 0`, and a strict
  fully charged lifecycle win at the control's frozen `q=10000`. It also
  requires `NO_WIN` on generic dense and both spectrum-matched negative
  controls.
- `KILL_FIXED_M_COMPILER_V2_FALSE_DISPATCH` fires if any reportable negative
  control dispatches. It dominates every positive token.
- `FIXED_M_COMPILER_V2_DEPLOYABLE_WIN` requires controls PASS and at least one
  member of the frozen positive population -- Poisson Green, irregular RBF,
  barbell heat, OPT-125M q-projection, or OPT-125M output projection -- to dispatch with exact representation certificate,
  executable error within the frozen bound, online win, finite strict
  break-even, and strict fully charged lifecycle win at its precommitted
  deployment `q`, followed by fresh independent evidence review.
- `FIXED_M_COMPILER_V2_NO_DEPLOYABLE_WIN` is legal only after controls PASS and
  fresh evidence review if no non-planted real operator satisfies that
  conjunction.
- `VOID_FIXED_M_COMPILER_V2_RESOURCE_OR_CORPUS` is reserved for a missing
  required corpus or the registered CPU/RSS boundary. It is not a scientific
  negative.

## 1. V1 failure diagnosis

**EXACT COMPUTATION from committed V1 receipts.** The planted matrix was
certified as `BCCB(16,16), rank 0` and its FFT apply agreed with dense to
`5.4609e-16` relative Frobenius error. The failure was not planting,
projection-family recognition, or error certification. It was the control's
online and lifecycle scale: at `n=256,b=16`, BCCB apply took `90.0 us` versus
`55.7 us` dense, so no finite break-even existed. V1 also charged `6.609865 s`
offline. The control asked a correct compiler to reject its planted candidate.

**NUMERICAL DIAGNOSIS.** The V1 class-average projector can perturb an exactly
stored family member by last-bit summation rounding even though the ideal real
projector is the identity. That did not cause the dispatch failure, but it made
exact recovery needlessly expensive: the general dyadic residual certificate
had to scan the reconstructed matrix. V2 adds a logically prior exact-member
path that uses a representative descriptor and verifies byte equality against
the fully materialized family member. It never treats a float screen as a
certificate.


## 2. Frozen V2 compiler path

The V1 dictionary and apply implementations remain unchanged. V2's winner path
is deliberately narrow and complete for exact rank-0 members:

1. enumerate every applicable V1 family in canonical order;
2. extract a representative descriptor from stored matrix entries;
3. materialize the represented matrix and accept exact membership iff
   true bitwise equality of the contiguous float64 arrays (including signed
   zero);
4. prepare and time only exact members, against the same resident dense input,
   full-output allocation policy, three warmups, nine repeats, pinned logical
   CPU, one BLAS/FFT thread, float64, and batch 16;
5. use the exact certificate `eta=0`; independently validate executable apply;
6. dispatch the least fully charged lifecycle candidate with canonical family
   tie-break, or return `NO_WIN`.

This is not an approximate low-rank compiler and does not claim completeness
outside bitwise materializer-image rank-0 membership. That narrowing is the compile-cost
intervention: it removes all SVDs, alternating least-squares rounds, float
projection screens, and non-candidate exact dyadic scans from the successful
path. The V1 exhaustive rank-0/rank-4 result remains the comparison, not hidden
work in V2.

### Theorem 1 -- exact representative recovery

**PROVED, pending fresh proof review.** For each V1 family and stored binary64 matrix `M` in the bitwise image of
that family's executable materializer, the representative descriptor
materializes byte-for-byte to `M`. This does not call a signed-zero variant
with different bits a bitwise member, even when it is the same real matrix.

**Proof.** Diagonal uses the stored diagonal and verifies all off-diagonal
zeros. Circulant uses column zero, whose entry at row `d` is `c_d`. Toeplitz
uses column zero and row zero. Block-circulant with block size `s` uses block
column zero, so block row `d` is `C_d`. `BCCB(p,q)` uses scalar column zero,
reshaped as the kernel, so row `(d,e)` is `c_(d,e)`. The frozen materializers
reconstruct every entry by the defining difference index. True bitwise equality, not NumPy numeric equality, is then checked over the
complete matrix, so acceptance implies the stored
operator is exactly recovered. `square`

### Theorem 2 -- zero certificate

**PROVED, pending fresh proof review.** If the complete byte-equality check
passes, the exact-real dyadic stored matrix and encoded structured operator are
entrywise identical. Hence `max_ij |M_ij-Mhat_ij|=0`,
`eta=n max_ij |M_ij-Mhat_ij|=0`, and the V1 operator-norm/apply certificate
holds for every input.

### Theorem 3 -- lifecycle and dispatch

**PROVED, inherited from V1 pending delta review.** For finite nonnegative
offline cost `A`, resident dense time `D`, encoded apply time `C`, and zero
decode, strict break-even is `q>A/(D-C)` when `C<D`, with minimum integer
`floor(A/(D-C))+1`; otherwise it is infinite. V2 dispatches only certified,
apply-valid candidates whose charged `A+qC` is strictly below `qD`.

## 3. Control rebuild and fixed populations

- Positive control: a deterministic random exact `BCCB(16,32)` at `n=512`,
  batch 16, `q=10000`. The larger scale tests the same executable BCCB path in
  the regime where V1's non-planted Poisson operator already showed an online
  win. The plant is generated from the frozen PRNG and dictionary materializer;
  no family label is supplied to the compiler.
- Non-planted target: the V1 periodic screened-Poisson Green operator on the
  `16 x 32` torus, batch 16, `q=10000`.
- Negative controls: V1 generic dense, a spectrum-matched planted control, and
  a spectrum-matched Poisson control. All use `q=10000`.
- Continuity diagnostics: V1 irregular RBF and barbell-heat matrices, both
  `q=10000`. They can return `NO_WIN` without defining the V2 positive
  verdict.


## 4. Real inference-weight census

V2 reads the locally cached, trained `facebook/opt-125m` checkpoint at Hugging
Face revision `27dcfa74d334bc871f3234de431e71c6eeba5dd6`, requires the frozen checkpoint SHA-256
`2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9`
and config SHA-256
`54162c9cd0377110df1859dfabebb106fc0988c8aadb88207ce1ee0d34985df6`,
and copies two complete square float16 checkpoint tensors to sealed float64 V2
inputs by exact value conversion: layer-0 self-attention `q_proj.weight` and `out_proj.weight`, each
`768 x 768`. These are diagnostic real inference weights, not positive controls.

For each tensor V2 measures the full singular spectrum, stable rank,
participation rank, rank-4 Frobenius tail, numerical rank, and minimum exact
rank-0 family projection defect. The deployment count is precommitted as
`q=1,000,000,000` to test the billion-application premise. A weight can cross
break-even only if the compiler actually finds a certified structured candidate;
high reuse never converts an unsupported representation into a win. Checkpoint
loading and spectrum diagnostics are reported separately and are not hidden
inside compiler cost or claimed as deployment timing.

## 5. Cost accounting delta

V2 charges from compiler entry through completed dispatch: exact-member search,
all retained plan preparation, dense and candidate calibration, executable
validation, complete candidate decision-state construction before the dispatch
decision, and measured dispatcher time. Offline cost is the exact sum of the
three individual binary64 dyadic values for case wall time, complete dispatch
calibration wall time, and the charged dispatch bound; V2 does not round a
float sum before rationalization. Instead of V1's arbitrary one-second reserve, V2 premeasures
1000 executions of the actual pure dispatch reducer over a worst-case synthetic
candidate table and charges both the complete 1000-call calibration wall time and the maximum
observed per-call time plus `10 us` to every compiled case. The actual
dispatch must not exceed this bound. Input generation, checkpoint
loading, and post-outcome spectrum diagnostics are excluded because they are
not used by the automatic decision and are reported separately.

The full run is CPU-only with 20-minute and 4-GiB absolute-RSS checks at every
declared phase boundary. These are fail-closed sampled boundaries, not a
continuous peak monitor: an overrun inside a long phase is detected and voided
at the next boundary. A smoke run and fresh code/proof review are mandatory
before the reportable full run.

## 6. Frozen evidence boundary

No V2 positive claim is legal until the independent verifier regenerates all
synthetic matrices, rehashes the external checkpoint and sealed tensors,
replays exact family membership and materialization, recomputes all certificates,
apply errors, timing medians, exact rational break-even/lifecycle arithmetic,
dispatches, controls, spectra, resource bounds, and artifact hashes from the raw
receipts. Timing samples are empirical; exact membership, certificate and
adjudication arithmetic are exact computations over stored bytes and values.


## 7. First implementation smoke

**EXACT COMPUTATION.** The first V2 smoke returned
`PASS_FIXED_M_COMPILER_V2_SMOKE` across all 15 registered direct
family/layout checks. Every representative descriptor rematerialized
byte-for-byte to its planted input, and every executable apply agreed with
resident dense within the frozen limit.

**EXACT COMPUTATION.** For the dedicated `BCCB(16,32)` control smoke, the V2
representative path recovered the stored matrix exactly with `eta=0`.
The V1 class-average numerical projector on the same exactly planted stored
matrix had maximum entry delta `3.3306690738754696e-16`. This confirms that
V1's ideal projector theorem was sound while its binary64 averaging path
needlessly destroyed byte identity.

The repaired pure-dispatch accounting persists all 1000 finite positive
calibration samples and charges the complete calibration wall plus the maximum
sample and frozen `10 us` margin to every case. Exact measured values live in
the regenerated smoke receipt so this report can be frozen before the final
smoke source hash is bound. This is smoke evidence only. No reportable V2
lifecycle run has executed. Fresh proof/code review remains mandatory.


## 8. First fresh proof/code review -- `NO_GO_FIXED_M_COMPILER_V2_FULL_BENCHMARK`

**EXACT COMPUTATION.** The first fresh zero-context reviewer blocked the full
run on eight issues: fail-open verifier surfaces; incomplete artifact
transactionality; rounded rather than termwise-exact offline rationalization
and stale receipt wording; signed-zero unsoundness in the byte-equality claim;
unstated float16 checkpoint provenance and unfrozen source hashes; overstated
continuous resource semantics; a positive-population mismatch; and stale smoke
bytes.

**EXACT COMPUTATION.** The repair uses true uint8 bitwise equality; freezes and
checks the OPT checkpoint/config hashes, config identity, tensor keys, shapes,
and float16 source dtype; exact-sums each charged binary64 time as a dyadic
rational; freezes one five-case positive population shared by runner and
verifier; narrows the resource claim to sampled phase boundaries; and expands
the verifier to replay the 15-layout smoke, require the 1000 finite positive
dispatch samples and every nine-sample timing block, enforce frozen constants
and native types, bind canonical run/artifact paths and exact schemas, reject
extra NPZ members, replay all candidates and adjudication fields, and bind
resource checks to literal limits. The reportable full run remains unopened.
A new smoke and second fresh review of the final bytes are required.


## 9. Second fresh code review -- `NO_GO_FIXED_M_COMPILER_V2_FULL_BENCHMARK`

**EXACT COMPUTATION.** The second fresh reviewer found four remaining verifier
holes: resource receipts lacked an independently expected phase sequence;
representation NPZ names were derived from untrusted mappings and duplicate
candidates could collapse; checkpoint/config paths were not checked literally;
and residual native-type coercions remained.

**EXACT COMPUTATION.** The repair independently constructs every resource phase
from regenerated cases and dictionary membership, requires raw samples to equal
the receipt minus the terminal `raw_results_written` boundary, and enforces
nonnegative monotone elapsed/RSS values. It independently constructs every
descriptor/plan NPZ name, rejects duplicate candidates, binds candidate family
objects literally, requires exact checkpoint/config paths, and adds strict
fraction, CPU, fixed-contract, smoke, timing, checkpoint, manifest, dense,
offline-component, spectrum, and adjudication schemas/native types. Canonical
JSON comparison is used where Python equality could equate Boolean and integer
values. The full run remains unopened pending a third fresh review.


## 10. Third fresh code review -- `NO_GO_FIXED_M_COMPILER_V2_FULL_BENCHMARK`

**EXACT COMPUTATION.** The third fresh reviewer found a fatal literal mismatch:
the runner's persisted fixed search contract did not equal the verifier's frozen
bitwise-materializer wording, so an honest full run could not pass. It also
found that aggregate compile-cost components were not algebraically tied back
to membership, preparation, dense-calibration, candidate-calibration and
dispatch-calibration timing rows, and that the inference-spectrum map and best
rank-0 projection receipt were not exact-key/exact-value checked.

**EXACT COMPUTATION.** The repair makes the runner and verifier search literals
identical; checks all exposed profiling identities, the enclosing search lower
bound, and the residual wall-time identity; requires exactly the two frozen OPT
spectrum keys; and canonically compares the complete rank-0 defect list and best
projection receipt against independent recomputation. The full run remains
unopened pending a fourth fresh review of the regenerated smoke and final
three-file bytes.


## 11. First reportable full run -- superseded

**HISTORICAL EVIDENCE.** Sections 11--16 record the first audited run. Section
17 explains why its closeout was rejected and its source freeze superseded.
Section 18 is the sole current reportable run and replaces all timing, resource,
hash, and final-review statements in Sections 11--16.

**EXACT COMPUTATION.** After the three blocking reviews above were repaired, a
fourth fresh reviewer returned `GO_FIXED_M_COMPILER_V2_FULL_BENCHMARK` on the
frozen runner, verifier, report and regenerated smoke bytes. It independently
replayed all 15 family/layout smoke cases, exercised signed-zero and receipt
tampering falsifiers, checked the termwise-dyadic cost and lifecycle equations,
and rehashed the frozen OPT checkpoint/config. The reportable full run is
`190741ded93fc7d1fb785307`.

**EXACT COMPUTATION.** Independent audit returned all three earned tokens:

```text
PASS_FIXED_M_COMPILER_V2_CONTROLS
PASS_FIXED_M_COMPILER_V2_EVIDENCE_REVIEW
FIXED_M_COMPILER_V2_DEPLOYABLE_WIN
```

The three registered negative controls all returned `NO_WIN`; the false-dispatch
set is empty. The planted `BCCB(16,32)` control was recovered bit-for-bit with
`eta=0`, executable relative error `5.0173e-16`, and automatic selection of
`BCCB(16,32), rank 0`. At `q=10000`, its online median was `122.4 us` versus
`196.5 us` dense (`1.6054x`), strict break-even was `q=3683`, and fully
charged lifecycle was `1.496859 s` versus `1.965000 s` dense (`1.3127x`).
The V1 control failure was therefore a bad scale choice, not failed planting or
recognition; the new control earns the intended positive-control obligation.


## 12. Deployable win and full lifecycle table

**EMPIRICAL timing / EXACT COMPUTATION dispatch and break-even.** Times are
medians of nine repeats after three warmups on one pinned logical CPU, float64,
batch 16, resident input/plan, full output materialized, zero decode. Offline
charges compiler-entry-to-dispatch wall, the complete 1000-call dispatch
calibration wall, and the calibrated maximum-plus-`10 us` dispatch bound.

| Case | q | dense us | selected apply us | online x | offline s | strict break-even q | dispatch |
|---|---:|---:|---:|---:|---:|---:|---|
| generic dense n32 control | 10000 | 1.800 | -- | -- | 0.199721 | -- | NO_WIN |
| planted BCCB(16,32) control | 10000 | 196.500 | 122.400 | 1.6054 | 0.272859 | 3683 | REPRESENTATION |
| spectrum-matched planted control | 10000 | 201.200 | -- | -- | 0.240657 | -- | NO_WIN |
| screened-Poisson Green | 10000 | 201.700 | 120.800 | 1.6697 | 0.272830 | 3373 | REPRESENTATION |
| spectrum-matched Poisson control | 10000 | 201.600 | -- | -- | 0.240420 | -- | NO_WIN |
| irregular RBF | 10000 | 49.200 | -- | -- | 0.202770 | -- | NO_WIN |
| barbell heat | 10000 | 49.800 | -- | -- | 0.202585 | -- | NO_WIN |
| OPT-125M layer-0 q projection | 1000000000 | 486.500 | -- | -- | 0.268862 | -- | NO_WIN |
| OPT-125M layer-0 output projection | 1000000000 | 487.100 | -- | -- | 0.268377 | -- | NO_WIN |


**DEPLOYABLE WIN.** The non-planted periodic screened-Poisson Green operator is
the real crossing class. Its stored binary64 matrix is exactly
`BCCB(16,32)`, with bitwise representation certificate `eta=0` and executable
relative error `4.8805e-16 <= 5e-12`. The descriptor is 4096 bytes, the prepared
transform is 4352 bytes, the declared traffic proxy is 278784 bytes, and its
batch-16 apply uses 4352 complex pointwise products versus the dense baseline's
4194304 real products. At frozen `q=10000`, candidate lifecycle is `1.480831 s`
versus `2.017000 s` dense: a fully charged `1.3621x` end-to-end win. This earns this unit's precommitted deployable-win token and supplies a useful
running result for a periodic operator family. It does not earn the project's
middle success tier, because BCCB/FFT is an existing representation class rather
than a new one.

## 13. Where compile cost went and what V2 removed

**EMPIRICAL.** Poisson's fully charged offline cost fell from V1's
`25.420513 s` to `0.2728305 s`, a `93.17x` reduction. Its strict break-even
fell from `285946` to `3373`, an `84.77x` reduction. V2 does not use truncated
or randomized SVD on this path: it obtains the stronger result by proving exact
rank-0 membership first, so every SVD, alternating least-squares iteration, and
general residual-certificate scan is bypassed without weakening the error
claim.

Poisson's V2 case wall before charged dispatch accounting was `0.0742436 s`:
`0.0402746 s` exact member search, `0.0284700 s` candidate calibration,
`0.0025027 s` dense calibration, `0.0003841 s` reference apply, and
`0.0026122 s` residual overhead. The intentionally conservative global
1000-call dispatch calibration contributes `0.1982523 s`; the charged dispatch
bound contributes `0.0003346 s`. Thus the current break-even is already below
the selected deployment count even though 72.7 percent of charged offline cost
is conservative calibration overhead. No conversion cost is hidden: the input
arrives dense, exact descriptor extraction/materialization is charged, the plan
is prepared and resident, output is fully materialized, and decode is zero.


## 14. Honest inference-weight result

**EXACT COMPUTATION / EMPIRICAL spectra.** The two tested real weights are the
stored float16 OPT-125M layer-0 attention q-projection and output-projection,
exactly value-converted to float64 for analysis. Both are full numerical rank
768 and neither is an exact member of the frozen dictionary, so both return
`NO_WIN` even at precommitted `q=1000000000`. Repetition cannot amortize a
representation that does not certify.

| Weight | best rank-0 frozen family | relative Frobenius defect | stable rank | participation rank | rank-4 tail |
|---|---|---:|---:|---:|---:|
| layer-0 q projection | block-circulant s=128 | 0.913113 | 31.0604 | 123.8684 | 0.942166 |
| layer-0 output projection | block-circulant s=128 | 0.912517 | 13.8601 | 61.7941 | 0.881006 |

These measurements are evidence against the frozen exact structured dictionary
for these two matrices, not against all inference weights or approximate model
compression. No approximate candidate was searched, and no claim is made that
billion-use learned weights generally support this win.


## 15. Evidence identity and resource accounting

**EXACT COMPUTATION.** The full run used only CPU, one-thread numerical pools,
and sampled phase-boundary resource guards. Its terminal recorded transaction boundary
was `3.462069 s` and `721305600` bytes RSS; the maximum sampled RSS was
`729227264` bytes at an earlier membership boundary. Both are below the frozen
1200-second and 4-GiB limits. This is not a continuous peak-RSS claim. Checkpoint provenance is
OPT-125M revision `27dcfa74d334bc871f3234de431e71c6eeba5dd6`, checkpoint SHA-256
`2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9`,
and config SHA-256
`54162c9cd0377110df1859dfabebb106fc0988c8aadb88207ce1ee0d34985df6`.

```text
run id: 190741ded93fc7d1fb785307
repository HEAD bound inside raw evidence: 9c3e880055b47e9ef2a4549c941fa3f7683f3ec4
raw_results.json SHA-256: cc1024f056dc082a3477fbce21c675ab1789a6a2d1d1def8ac08fcba696a8cee
independent_audit.json SHA-256: 91ef57c6efcf9500a90c15885f9be5d4df12ce9cb3c25accf30f6c4952e83c36
artifact_manifest.json SHA-256: 5447856b4aaff5b308511a3b2735a93f4b486152385e42e55ebc77d1cab93844
resource_stop_receipt.json SHA-256: a9e927a3a8e94f16325b6a9e4fab00854d9b9eeb7456ce55675bb04d9f29070f
latest_status.json SHA-256: 53fcfb90a4f873c5913dbdfb8cc1ee0446896aab7c58084be23d13a63a271033
reviewed_hashes.json SHA-256: 1e4bd6de29873b706950de4a0ee977a349d303c5686dfa26a4e8d4a47c4f2f23
pre_run_review.json SHA-256: fa1b7a09905a32a0cdbe968586233cc5617b5b49265b3fe3492ca40767afd086
smoke.json SHA-256: 199ec7b924e61c29abac18e9c301b2661dc15ba604249c2c91c887055cbb1197
reviewed runner SHA-256: 8a45c6808dc58989a5b7b8be98b8c5dfd54243dff8a6da195e898ce0c9cababd
reviewed verifier SHA-256: 57b96a006f6b1c61655e33ce3fb5d09e104c163339f52f868daab8f476024ed8
report SHA-256 at pre-run review: 7c4fde31fd64c8e75ef2e165e96615fab0c5cc31abb7580a1ebe8f8a1f0d70c0
```

## 16. Honest so-what

**EMPIRICAL -- one-sentence honest so-what.** V2 fixes the failed control and
turns V1's real `1.67x` Poisson online signal into a rigorously certified,
fully charged `1.36x` end-to-end deployment win at `q=10000` by replacing the
expensive general approximate fitter with exact structured-membership
compilation; this is a deployable win for repeated periodic screened-Poisson
operators, not a general dense-matrix or neural-weight breakthrough.

**CONJECTURE / next information-gain direction only.** The exact-member compiler
should next be tested on other naturally repeated translation-invariant and
block-periodic operators, where exact structure and high deployment counts are
plausible. Learned dense weights should not receive more exact-dictionary spend
without a precommitted approximate certificate that survives the observed
0.91 projection defects and charges model-quality impact. No such next unit was
started here.


## 17. First final zero-context review -- `FAIL_FIXED_M_COMPILER_V2_FINAL_EVIDENCE_REVIEW`

**EXACT COMPUTATION.** The first final zero-context reviewer reproduced the
controls and Poisson lifecycle win, then rejected two closeout claims and one
replay ergonomics gap. The ledger confused terminal RSS with maximum sampled
RSS; Section 12 incorrectly equated this known BCCB/FFT deployment result with
the project's new-representation middle tier; and the verifier relied on the
caller's one-thread shell environment during deterministic corpus regeneration.

**EXACT COMPUTATION.** The ledger now records the actual maximum sampled RSS
`729227264` bytes while this report separately records the terminal boundary.
Section 12 explicitly denies the project middle-tier claim. The verifier now
enters `threadpool_limits(limits=1)` internally before any audit regeneration,
so plain invocation enforces the numerical-pool contract. Because verifier and
report bytes changed, the earlier run remains historical evidence only: a new
source freeze, fresh pre-run review, full run, independent audit, and fresh
zero-context final review are required below.


## 18. Definitive self-enforcing replacement run

**EXACT COMPUTATION.** Fresh rerun gate
`GO_FIXED_M_COMPILER_V2_FULL_BENCHMARK` froze runner
`8a45c6808dc58989a5b7b8be98b8c5dfd54243dff8a6da195e898ce0c9cababd`,
self-enforcing verifier
`3b86fa76059f5b97d7e4e3cf2e50e7227b7f5c86684bdd32b746d84368d4bfe7`,
and report-at-review
`3f10c6916179c416c460258cd1645c08511a4c206a9a28fa617838f5db901449`.
The replacement full run is `154f27c839d474ba8f05fe76`. Plain invocation
`python -B research/verify_fixed_m_compiler_v2.py --audit`, with no shell thread
overrides, passed because the verifier now holds numerical pools at one thread.
It returned:

```text
PASS_FIXED_M_COMPILER_V2_CONTROLS
PASS_FIXED_M_COMPILER_V2_EVIDENCE_REVIEW
FIXED_M_COMPILER_V2_DEPLOYABLE_WIN
```

**EXACT COMPUTATION / EMPIRICAL timing.** All three negative controls again
returned `NO_WIN`. The exact planted `BCCB(16,32)` control again recovered
bit-for-bit with `eta=0`, selected itself automatically, and won: `123.4 us`
apply versus `194.3 us` dense (`1.5746x` online), `0.2771109 s` fully charged
offline, strict break-even `q=3909`, and `1.511111 s` versus `1.943000 s`
lifecycle at `q=10000` (`1.2858x`). Thus
`PASS_FIXED_M_COMPILER_V2_CONTROLS` is earned with zero false dispatches.

| Final-run case | q | dense us | selected apply us | online x | offline s | break-even q | dispatch |
|---|---:|---:|---:|---:|---:|---:|---|
| generic dense n32 control | 10000 | 2.100 | -- | -- | 0.202979 | -- | NO_WIN |
| planted BCCB(16,32) control | 10000 | 194.300 | 123.400 | 1.5746 | 0.277111 | 3909 | REPRESENTATION |
| spectrum-matched planted control | 10000 | 210.300 | -- | -- | 0.246582 | -- | NO_WIN |
| screened-Poisson Green | 10000 | 195.600 | 122.300 | 1.5993 | 0.277820 | 3791 | REPRESENTATION |
| spectrum-matched Poisson control | 10000 | 196.400 | -- | -- | 0.245774 | -- | NO_WIN |
| irregular RBF | 10000 | 89.300 | -- | -- | 0.208973 | -- | NO_WIN |
| barbell heat | 10000 | 65.600 | -- | -- | 0.206808 | -- | NO_WIN |
| OPT-125M layer-0 q projection | 1000000000 | 465.600 | -- | -- | 0.273758 | -- | NO_WIN |
| OPT-125M layer-0 output projection | 1000000000 | 475.000 | -- | -- | 0.271578 | -- | NO_WIN |

**DEPLOYABLE WIN.** The non-planted screened-Poisson Green operator again
crosses end-to-end. The exact `BCCB(16,32)` certificate remains `eta=0`; the
executable relative error remains `4.8805e-16`; representation size/work
accounting is unchanged from Section 12. Its `122.3 us` apply versus `195.6 us`
dense is `1.5993x` online. At `q=10000`, its fully charged lifecycle is
`1.500820 s` versus `1.956000 s` dense, a `1.3033x` end-to-end win, with strict
break-even `q=3791`. Relative to V1, fully charged offline cost fell from
`25.420513 s` to `0.2778205 s` (`91.50x`) and break-even fell from `285946`
to `3791` (`75.43x`). This earns the unit's deployable-win token, not the
project middle tier: BCCB/FFT is known, not a new representation class.

**EMPIRICAL cost profile.** Poisson's case wall was `0.0759714 s`:
`0.0399705 s` exact member search, `0.0306234 s` candidate calibration,
`0.0024095 s` dense calibration, `0.0002667 s` reference apply, and
`0.0027013 s` residual overhead. The fully charged global dispatch calibration
was `0.2013734 s`; the maximum-plus-margin reserve was `0.0004757 s`. Exact
membership again eliminates V1's SVD/ALS/general-certificate cost without using
randomization or weakening the certificate. The two OPT weights remain
`NO_WIN` at `q=10^9`; their independently replayed spectrum measurements in
Section 14 are input-deterministic and unchanged.

**EXACT COMPUTATION.** Terminal sampled transaction boundary was `3.412063 s`
and `731803648` bytes RSS. Maximum sampled RSS was `739950592` bytes at an
earlier membership boundary. Both are below the frozen limits under the honest
phase-boundary, non-continuous semantics.

```text
run id: 154f27c839d474ba8f05fe76
repository HEAD bound inside final raw evidence: a9e12d4d6739225155840c033ae1950146330a52
raw_results.json SHA-256: a1b2a3de274d8a5f6fef629d0a912e8700c1ce3c3ada1896743cd662b5ebbdc0
independent_audit.json SHA-256: 0aa2ab555b249002d391160e32f0ae7da73e5a1238aebb7e400eb817b4615d8a
artifact_manifest.json SHA-256: aa5facb194203d4e37a3e1a2efb07724e5d1ab05e0964375f3638288d0f81b11
resource_stop_receipt.json SHA-256: 641a91bf9ebc7fa66fb06fd7d4979c8f89e66233a5180409d4b4dd5a405559b4
latest_status.json SHA-256: 3e6cde45a0656a3f247b76b8359fb23eb7717d69c665751d771d15327af8c778
reviewed_hashes.json SHA-256: 14b16d131dcc25ade7f6c84ea71bcc8e0d90e5e9225585653ba2a49b0b061dfa
pre_run_review.json SHA-256: d02f48a72a75a9bbc1d8de4f54eeae0fee798eb78084cd30a5e53807a60ad6d0
smoke.json SHA-256: cd53f8051c8eec1ab40ec402c7ac224399b6aa32d6f19d50ecd46a71bb257a64
```

**EMPIRICAL -- final one-sentence honest so-what.** A complete exact-member
compiler over the known frozen structured dictionary turns the real periodic
screened-Poisson signal into an audited, fully charged `1.30x` deployment win
at `q=10000`, while honestly rejecting the tested neural weights and making no
claim to a new representation class or general matrix-multiplication advance.
