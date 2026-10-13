# Independent Proof Review 6 - CIC Noisy X/Z Fusion and FrCC Freshness Pair

**Review status:** COMPLETE (2026-07-14). This file is written as a standalone verification pass over the remaining `PROVED-PENDING-REVIEW` claims in `research/CIC_GATE0.md` and `research/FRCC_GATE0.md`.

**Reviewer posture:** fresh, zero-context, adversarial mathematical review of explicit frozen statements. No private project memory is evidence. The frozen source scope is finite and CPU-only; no repository file except this review is edited.

**Rigor convention:** each mathematical finding is marked `PROVED / SOUND`, `PROVED / DEFECT`, or `PROVED / COUNTEREXAMPLE`; interpretive policy choices use `HEURISTIC`. Final claim verdicts use only `UPHELD`, `UPHELD-WITH-REPAIR`, or `REFUTED`.

**Citation convention:** `CIC:Section` and `FRCC:Section` refer to `research/CIC_GATE0.md` and `research/FRCC_GATE0.md` sections/claims.

## 1. Exact claim statements and definitions under review

### 1.1 Frozen contracts used by both files

**CIC frozen contract.** A one- or two-photon stock is `rho^(\otimes k)` for `k\in\{1,2\}` with `rho` a finite `2x2` density matrix, protocols are finite instrument compositions with one move per specimen call, and cost is specimen count. The move table in section 2.2 defines the only detection primitives; only these moves consume specimens and all operations are finite algebraic.

**FrCC frozen contract.** A finite source state is `(t,D,v_obs,e,transcript,published)` with fixed finite lease contract and source values `(v_obs[i],e_i)`, moves are `read`, `eval`, `compile-cert`, `publish`, and `wait`. A protocol may pay source reads, Boolean operations, output cost, and time; all are charged.

**Heuristic scope normalization.** `Cert(f(v),v)` denotes the coordinate sets `S` that force `f(v)` under fixed valuation on `S`, and
`H_f(v,e)=max_{S\in Cert} min_{i\in S} e_i` with the convention `min(\emptyset)=+\infty` for constant functions. In FrCC-EPOCH, epoch length `L\ge1` is fixed and every renewal-read returns an expiration through epoch end.

### 1.2 Claim 1 (CIC) - explicit noisy X/Z fusion CAN

**Exact pre-review statement.** Under section 2's contract, `fuse_half` with effects
`G_{ab}=(I+(a/2)X+(b/2)Z)/4` has marginals
`sum_b G_{ab}=M^{(X,1/2)}_a` and `sum_a G_{ab}=M^{(Z,1/2)}_b`. Therefore
`nu({M^(X,1/2),M^(Z,1/2)})=1`, while independent separate one-specimen moves consume two.

### 1.3 Claim 2 (CIC) - explicit sharp X/Z non-fusion CANNOT

**Exact pre-review statement.** No one-specimen finite parent on `\mathbb C^2` has marginals exactly `(M^{(X,1)}_a)` and `(M^{(Z,1)}_b)` simultaneously for every `rho`. The two-specimen family `H_{ab}=M^{(X,1)}_a\otimes M^{(Z,1)}_b` is feasible, so
`nu({M^(X,1),M^(Z,1)})=2`.

### 1.4 Claim 3 (FrCC) - static exact horizon CAN/CANNOT

**Exact pre-review statement.** For any finite Boolean `f`, valuation `v`, and finite lease vector `e`, with
`H_f(v,e)` as above:

1. For every `t\le H_f(v,e)`, `f(v)` is certified without reads.
2. For every `h>H_f(v,e)`, every read-free protocol is unsound on some lease-respecting adversarial completion.

### 1.5 Claim 4 (FrCC) - epoch theorem `FRCC-EPOCH`

**Exact pre-review statement.** In synchronized lease epochs, the minimum worst-case number of reads needed to output correct value and live certificate in one epoch is exactly the deterministic decision-tree complexity `D(f)`.

### 1.6 Claim 5 (FrCC) - strict `2<3` mux saving

**Exact pre-review statement.** For function `mux(s,x0,x1)` on three leased bits, the minimum worst-case epoch refresh reads is exactly `2` (same as `D(mux)`), so adaptive querying beats refresh-all (`3`).

## 2. Attack log - CIC claims

### Increment 1: CAN construction correctness

1. **PROVED / SOUND - `G_ab` is a valid POVM.** Since eigenvalues of `aX/2+bZ/2` are `\pm1/\sqrt2`, each
`G_{ab}` has eigenvalues `(1\pm1/\sqrt2)/4`, hence is positive semidefinite. Summing over all four outcomes yields `I`, so the family is a finite POVM.

2. **PROVED / SOUND - marginals are exactly the noisy Pauli effects.** Algebra gives
`\sum_b G_{ab}=(I+aX/2)/2=M^{(X,1/2)}_a` and `\sum_a G_{ab}=(I+bZ/2)/2=M^{(Z,1/2)}_b`, matching the requested sharpness and contrast definitions by inspection on every `rho`.

3. **PROVED / SOUND - lower bound `nu\ge1` is immediate from contract typing.** There is no zero-specimen detector in the move table; any protocol with `k=0` would produce output independent of `rho`, while both target marginals depend on `rho`. Hence at least one specimen is required.

4. **PROVED / SOUND - exactness of the claimed saving.** Naive separate noisy moves consume two specimens and two detector calls, while `fuse_half` consumes one specimen and one call, so the invariant is exactly one and the saved quantity is exactly one in both coordinates.

5. **PROVED / SOUND - boundary checks.** For any physically admissible density matrix with finite two-state support, the above operator identities hold identically. The argument does not invoke canonical relabelling, unbounded advice, or oracle access.

### Increment 2: CANNOT non-fusion proof

6. **PROVED / SOUND - parent reduction to four-outcome POVM is valid.** Any finite one-specimen protocol with classical post-processing can be aggregated to a four-outcome parent POVM `{J_ab}`.

7. **PROVED / SOUND - support intersections force zero effects.** From exact marginals, `sum_b J_ab=P^X_a`, hence `0\le J_ab\le P^X_a`; similarly `J_ab\le P^Z_b`. A positive operator bounded by each rank-one projector is supported in each projector's range, but the X- and Z-eigenspaces are distinct one-dimensional lines, with zero intersection. So each `J_ab=0`.

8. **PROVED / SOUND - contradiction and optimality.** Summing zero effects contradicts `\sum_{a,b}J_ab=I`, so no one-specimen parent exists. The two-specimen `H_ab` protocol is valid and has exactly the two-shot marginals, hence `nu(...)=2` by construction+obstruction.

9. **PROVED / SOUND - edge-case check.** For `k=0` output is impossible by same contract; for `k\ge3` only the two-assay target is evaluated, so exact two-copy lower/upper witnesses remain stable at values `1` and `2`.

## 3. Attack log - FrCC claims

### Increment 3: static `FRCC-STATIC` CAN

10. **PROVED / SOUND - canonical correctness horizon is attainable.** Let `S*` attain the finite maximum in `H_f(v,e)` (or `+\infty` for constants). For `t\le \min_{i\in S*}e_i`, every coordinate in `S*` is unchanged by lease contract; therefore all completions consistent with `S*` share value `f(v)`. So the published pair `(f(v),H_f,S*)` is sound through `H_f`.

11. **PROVED / SOUND - formula compiler is explicit and typed.** On monotone formula trees, the bottom-up rule set at AND/OR gates described in section 3 preserves correctness and computes exact certificate witnesses with `O(|D|)` operations and explicit finite output size.

12. **PROVED / SOUND - boundary scope.** If `f` is constant, `Cert` contains `\emptyset`, so by convention `H_f=+\infty`; the CAN clause remains true at all finite times and the CANNOT quantifier is vacuous for `h>+\infty`.

13. **PROVED / SOUND - no-free-choice edge case.** Finite exact leases and finite formula guarantee `S*` exists unless `f` is constant, which is handled above; there is no hidden non-finite object.

### Increment 4: static `FRCC-STATIC` CANNOT

14. **PROVED / SOUND - the `L_h` adversarial complement is exact.** If `h>H_f`, then `L_h={i:e_i\ge h}` is not a certificate (by definition of maximum), so there is completion `w` agreeing on `L_h` with `f(w)\ne f(v)`.

15. **PROVED / SOUND - indistinguishability without reads transfers the contradiction.** A read-free protocol has identical transcript in the run that keeps coordinates fixed and the run that switches expired coordinates to `w` after expiration, yet outputs diverge at time `h`; hence no read-free protocol can certify beyond `H_f`.

16. **PROVED / SOUND - boundary consistency.** If `h` is integer/non-integer or if some `e_i` equal `h`, the definition uses `\ge h` in `L_h`, which keeps exactly coordinates guaranteed through time `h`; all larger `h` values remain covered by the same complement argument.

### Increment 5: `FRCC-EPOCH` equivalence

17. **PROVED / SOUND - `D(f)` upper bound.** An optimal deterministic decision tree queried in one epoch induces a freshness policy that reads one source per tree query, returns a path certificate at the leaf, and is correct for every completion consistent with that path; leaf-to-leaf depth bound gives reads `\le D(f)`.

18. **PROVED / SOUND - `D(f)` lower bound.** Any exact policy in one epoch induces a deterministic decision tree by replacing adaptive reads with queries and terminal publications with leaf labels. Lease contract fixes all unqueried coordinates for the path, so correctness implies the tree computes `f`; worst-case reads are at least tree depth, so `\ge D(f)`.

19. **PROVED / SOUND - equivalence is contract-preserving.** The synchronous-epoch semantics identify reads with adaptive query steps and certificates with root-to-leaf partial assignments, with no extra structure added.

### Increment 6: strict `2<3` saving claim

20. **PROVED / SOUND - direct upper bound and witness.** For `mux`, two reads (`s`, then `x_s`) certify output through the epoch and publish the path certificate; reads are exactly `2` on all valuations.

21. **PROVED / SOUND - tightness via 2-vs-3 split.** Any one-read policy fails by symmetry: first reading `s` leaves data bit unknown; first reading `x0` or `x1` leaves the branch selected by `s` unknown, giving two completions with same read transcript and opposite outputs. Therefore `D(mux)=2` and `D(fresh\!\!\!?)` strict saving is exact.

22. **PROVED / SOUND - edge-case on non-mux functions.** The theorem is stated for arbitrary finite `f`; the strict member is one explicit witness within the claimed family and does not over-generalize to all functions.

## 4. Final verdicts and certified scopes

### 4.1 CIC-TOY

**Final verdict: `UPHELD`.**

**Certified core.** Under the frozen finite contract of section 2, one specimen suffices for noisy `X/Z` and two specimens are necessary for sharp `X/Z`. The one-specimen construction is the explicit `G_ab` parent and the two-specimen obstruction is the support intersection of rank-one sharp projections.

**Boundary scope.** No repair is needed for the theorem statements as written; the definitions of specimen consumption, finite instruments, and admissible operations are already explicitly fixed.

### 4.2 FRCC-STATIC

**Final verdict: `UPHELD-WITH-REPAIR`.**

**Certified core.** The exact horizon law and tight no-refresh lower bound hold on the stated finite lease contract.

**Minimal repair.** Add an explicit convention that `D(f)` and the FrCC protocols are deterministic and exact (zero-error), and state separately that constant functions use the `H_f=+\infty` convention so the CANNOT clause is vacuous there.

**Boundary scope.** These are bookkeeping repairs only; the mathematical statements and proofs for nonconstant finite `f` are sound.

### 4.3 FRCC-EPOCH and mux saving

**Final verdict: `UPHELD`.**

**Certified core.** In synchronized epochs, worst-case read complexity is exactly `D(f)` via direct bijection between policies and decision trees, and the three-source `mux` instance achieves and matches cost `2<3`.

**Boundary scope.** The result is valid for the frozen epoch contract where read operations are the only paid adaptive resource and all queries are exact over one epoch.

## 5. Inventory and pending-claim completion status

**COMPLETE PASS INVENTORY.** The reviewed `PROVED-PENDING-REVIEW` assertions in this pass are: CIC noisy-CAN, CIC sharp-CANNOT, FrCC-STATIC CAN/CANNOT, FRCC-EPOCH exact equivalence, and the strict `2<3` mux witness. No additional `PROVED-PENDING-REVIEW` claims surfaced in the target files outside these items.

## 6. One-paragraph summary

**PROVED / SUMMARY.** The CIC miniature is mathematically sound: a finite explicit parent POVM gives one-specimen noisy `X/Z`, and sharp `X/Z` cannot be done with one specimen because any finite one-specimen parent must be zero by sharp-projector support intersection. Both values `nu=1` and `nu=2` are therefore correct under the same finite stock contract. FrCC-STATIC and FrCC-EPOCH are also sound at the finite level: certificates plus expirations certify exactly the forced-horizon region, and in synchronized epochs the optimal refresh policy is exactly deterministic decision-tree complexity. The only repairs are formalization hygiene around deterministic/exactness and the constant-function horizon convention; the core mathematical content is upheld.