# PFPWO — Post-Fabrication Programmable Wave Operator Artillery Record

**Work-loop batch:** 2, iterations 6–8
**Date:** 2026-07-14
**Frozen registration:** research/DESIGN_GATE_2.md sections 3 and 5
**Final Stage-A token — HEURISTIC:** **ABSORBED**
**Occupation token — EMPIRICAL SOURCES + HEURISTIC GATE:** **OCCUPIED-KILL**
**Approximate-DFT transfer token — HEURISTIC:** **TRANSFER-FAIL**

**EMPIRICAL — EXECUTION RECORD.** This unit used the repository source dossier and
targeted primary-source retrieval. It ran no computation, touched no GPU, and made no
Git write. Sources that were not available as full text are identified as unresolved
obligations; none is treated as novelty clearance.

## 0. Rigor and frozen object

- **PROVED** is reserved for a complete proof that survived fresh independent Codex
  review. This document introduces no PROVED claim.
- **EMPIRICAL** marks what a named source or execution record establishes in its
  stated scope.
- **CONJECTURE** marks an unproved mathematical target.
- **HEURISTIC** marks the algebra-to-stakeholder interpretation, resource comparison,
  or gate inference.
- **SPECULATION** marks a possible engineering payoff without a cleared theorem
  route.

The frozen family is

\[
\mathcal A^{\rm circ}_{N,b}
=\left\{A_\theta=F_N^*D_\theta F_N:
D_\theta=\operatorname{diag}(e^{2\pi i\theta_j/2^b}),
\ \theta\in\{0,\ldots,2^b-1\}^N\right\}.
\]

The program contains exactly \(Nb\) bits. The temporal order, \(Q\)-use
amortization, \(b_x\)-bit input codebook, all-mode coherent output, \(\ell_2\)
error, failure probability, phase reference, and downstream channel
\(C_\theta=A_\theta^*\) remain exactly as registered. No family, stakeholder,
precision rule, endpoint, or competitor was changed after the attacks.

---

## 1. Iteration 6 — task, stakeholder, and transfer screen

### 1.1 Stakeholder source check

**EMPIRICAL — PRIMARY-SOURCE SCOPE.** Armstrong's optical-OFDM tutorial identifies
OFDM as a modulation method for dispersive optical channels and treats phase noise,
frequency offset, coherent reception, and the IFFT/FFT signal chain as central
constraints ([JLT 27, 189–204, 2009](https://doi.org/10.1109/JLT.2008.2010061)).
Rha et al. explicitly implement coherent-optical-OFDM chromatic-dispersion
precompensation with full-band inverse DFTs and subcarrier phase rotation
([IEEE PTL 27, 30–33, 2015](https://doi.org/10.1109/LPT.2014.2360931)).
These sources support a real coherent-transmitter stakeholder and, more
importantly, show that subcarrier-domain phase programming is already part of the
ordinary transmitter design space.

**HEURISTIC — STAKEHOLDER SCOPE.** The stakeholder can require an emitted coherent
field at the link boundary. It does not require that field to have been produced by a
standalone post-IFFT wave operator. Its operational objective is received-symbol
loss after the channel and receiver, not preservation of a particular internal
factorization.

### 1.2 Direct-task substitution is algebraically exact

Let \(z=F_Nx\). Then

\[
A_\theta x=F_N^*D_\theta z.
\]

**HEURISTIC — EXACT FINITE ALGEBRA, UNREVIEWED AS A FORMAL THEOREM.** Under the
strict registered input contract, a baseline given only \(x\) must compute
\(z=F_Nx\), multiply by \(D_\theta\), and inverse-transform before optical
emission. This paragraph does not erase that FFT cost.

The named OFDM stakeholder, however, naturally begins one stage earlier with
frequency-bin data symbols \(z\); its ordinary unprecompensated time waveform is
\(x=F_N^*z\). At that actual end-to-end boundary, a matched electronic or hybrid
transmitter multiplies each already-present subcarrier \(z_j\) by the \(b\)-bit
coefficient in \(D_\theta\), runs its ordinary IFFT, and drives its coherent
IQ/optical modulation chain. It emits
\(F_N^*D_\theta z=A_\theta x\) and does not return an unmaterialized array. The
precompensated channel gives

\[
C_\theta A_\theta x=A_\theta^*A_\theta x=x.
\]

Thus the stakeholder loss is achieved by fusing the program into the ordinary
per-subcarrier map. A wave path that insists on receiving the intermediate
time-domain \(x\) must first synthesize that \(x\) from the same data symbols; the
field contract has inserted the IFFT output as a privileged internal task boundary.
If the card instead insists that \(x\) is the externally supplied user input, the
source check verifies a generic coherent filtering task but not the claimed ordinary
OFDM-transmitter boundary. Either reading defeats the stakeholder-forced mismatch.
The comparison that remains is between two implementations of a known spectral
phase filter, not between a wave-only task and a wire-only task.

**HEURISTIC — ALTERNATIVE INTERPRETATION.** A high-rate optical pulse shaper could
still occupy a favorable constant-factor or Pareto point by avoiding some repeated
electronic arithmetic. That is a legitimate accelerator question. It is not evidence
that the finite task requires a new computation-model primitive.

### 1.3 Output, phase, location, and materialization controls

| Control | Rigor-labelled finding | Disposition |
|---|---|---|
| Full coherent field | **HEURISTIC:** a coherent transmitter ultimately emits such a field, but both the hybrid baseline and the wave card do so at the same optical boundary. | Does not force a wave-only computation. |
| Downstream quotient | **HEURISTIC:** the stakeholder consumes received symbols after \(C_\theta\); at the natural data-symbol input, the fused subcarrier implementation attains the same loss. | The registered \(x\)-field boundary is an artificial internal strengthening. |
| Global phase | **EMPIRICAL + HEURISTIC:** coherent receivers carry phase-recovery machinery; the registered absolute phase reference is stronger than many symbol-loss contracts. | Both full-field error and the operational phase quotient must be reported; the stronger row cannot create the separation. |
| Input location | **HEURISTIC:** digital \(b_x\)-bit symbols originate before the IFFT on both compared transmitters. | No free resident optical input. |
| Output location | **HEURISTIC:** both sides terminate at the same modulator/link boundary and pay optical emission/coupling. | No electronic-side-only materialization tax. |
| Program arrival | **HEURISTIC:** both sides receive the same \(Nb\) bits after fabrication and may retain them for exactly \(Q\) uses. | Fixed-instance fabrication is blocked symmetrically. |

### 1.4 Approximate-DFT transferability screen

The required transferable lemma had to cover the finite codebook, global
\(\ell_2\) error, failure probability, \(Nb\) program bits, named optical
terminals, and the downstream quotient for the whole family rather than for one exact
DFT.

**EMPIRICAL — SOURCE BOUNDARY.** Thompson's lower bound concerns an exact DFT over a
finite ring with word-valued terminals. Reif–Tyagi's DFT-VLSIO admits a unit-time
optical DFT primitive. Ailon's adjacent \(\Omega(N\log N)\) result is for
well-conditioned layered linear circuits computing the Fourier transform, not for a
post-fabrication family with an optical-emission endpoint. Miller's mode theory and
Li–Monticone's spatial-complexity line constrain finite wave channels; they do not
supply an electronic lower bound for this end-to-end transmitter task. These scopes
are documented in research/sources/OFFT_SOURCES.md.

**HEURISTIC — FAMILY OBSTRUCTION.**

1. The family contains the identity (\(\theta=0\)) and global phase maps, so no
   nontrivial lower bound can apply uniformly to every member.
2. A worst-case-\(\theta\) theorem would still have to defeat the fused
   frequency-domain diagonal implementation. Exact-word DFT separation does not do
   that.
3. A cut/rank bound on materializing all modes must charge the same optical emission
   boundary on both sides; it cannot count only electronic internal traffic.
4. At the natural OFDM data-symbol boundary, the stakeholder quotient allows the
   diagonal phase map to be compiled into the transmitter symbols before the IFFT.
   If only \(x\) is supplied, that is a different generic filtering task and the
   named stakeholder no longer forces the registration.

**Iteration-6 transfer token — HEURISTIC:** **TRANSFER-FAIL / DFT-SPECIFIC.** No
repository or retrieved source supplies a robust same-task lower-bound route that
transfers to \(\mathcal A^{\rm circ}_{N,b}\). No DFT theorem attempt was made.

### 1.5 Iteration-6 token

**TASK-CONTRACT-FROZEN — HEURISTIC.** The finite family, stakeholder, temporal order,
endpoint, quotient, and loss are truth-apt and remain unchanged. The contract is
therefore frozen rather than VOID. The direct-task construction is a blocking
absorption attack, carried to the full closure and physical ledgers in iterations 7
and 8. The missing transfer route already prevents end-of-batch clearance.

---

## 2. Iteration 7 — occupation and closure-class gate

### 2.1 Primary-source occupation matrix

| Line | Source fact — EMPIRICAL IN SOURCE SCOPE | PFPWO consequence — HEURISTIC |
|---|---|---|
| Universal unitary meshes | Reck et al. give an algorithmic decomposition of any finite-dimensional unitary into two-mode beam-splitter transformations ([PRL 73, 58, 1994](https://doi.org/10.1103/PhysRevLett.73.58)). | Post-fabrication programmable modal unitary maps are established theory, not a new permission. |
| Rectangular MZI meshes | Clements et al. give a programmable universal multiport with \(N(N-1)/2\) beam splitters, optical depth \(N\), an explicit programming decomposition, and improved loss tolerance ([Optica 3, 1460, 2016](https://doi.org/10.1364/OPTICA.3.001460)). | Arbitrary-unitary photonic closure contains the registered circulant subfamily, albeit with a denser \(O(N^2)\) control fabric. |
| Self-configuring linear optics | Miller constructs universal linear optical components with progressive local training and drift correction ([Photonics Research 1, 1, 2013](https://doi.org/10.1364/PRJ.1.000001)). | Calibration, detectors, feedback, and training are part of the established model story, not afterthoughts unique to this audit. |
| Field-programmable photonic arrays | Capmany and Pérez explicitly propose FPGA-like post-fabrication photonic arrays and discuss building blocks, programming, and physical implementation ([Optics Express 26, 27265, 2018](https://doi.org/10.1364/OE.26.027265)). | The headline permission “fixed fabric, program arrives later, linear optical function changes” is occupied. |
| Multipurpose processor | Pérez et al. demonstrate a common hexagonal silicon-photonic mesh reconfigured for different linear signal-processing functions ([Nature Communications 8, 636, 2017](https://doi.org/10.1038/s41467-017-00714-1)). | Reconfigurable photonic and hybrid baselines are mandatory and already concrete. |
| Self-characterization/control | Milanizadeh et al. require electronic control, architecture data, full bias calibration, auto-routing, and presets for a field-programmable photonic mesh ([Nature Communications 11, 6359, 2020](https://doi.org/10.1038/s41467-020-19608-w)). | The program-to-control compiler and calibration table are known load-bearing resources. |
| Self-calibration | Xu et al. use an optical reference path, phase recovery, and iterative calibration to obtain programmable complex impulse responses under thermal cross-talk ([Nature Photonics 16, 595–602, 2022](https://doi.org/10.1038/s41566-022-01020-z)). | Calibration can be engineered, but it is neither free nor a new formal resource by itself. |
| Programmable spectral phase | Ferdous et al. characterize and programmably line-by-line shape the phase of an on-chip-generated microresonator comb using a Fourier-transform pulse shaper ([Nature Photonics 5, 770–776, 2011](https://doi.org/10.1038/nphoton.2011.255)). | The \(F^*D_\theta F\) operation is known pulse-shaping practice, not merely contained abstractly in a universal mesh; this source does not establish an integrated size-indexed upper for the registered card. |
| Electronic analog closure | Floating-gate FPAAs provide post-fabrication configurable analog blocks and interconnect ([JSSC 46, 42–46, 2011](https://doi.org/10.1109/JSSC.2010.2056832)). | “Analog and programmed after manufacture” is not a wave-exclusive permission; exact high-bandwidth matching remains device-specific. |
| Optical capacity/VLSIO | Barakat–Reif/Reif–Tyagi already model finite optical communication capacity, volume/time, energy, and a DFT primitive; Miller and Li–Monticone supply modern finite-mode/spatial-limit tools. | A new audited bound may refine the ledger, but the model/cost storyline is occupied. |

**EMPIRICAL — SOURCE COMPLETENESS LIMIT.** Full or author-posted text was available
for Reck, Clements, Miller, Capmany–Pérez, and the named programmable-photonics
demonstrations through publisher/author repositories or arXiv. The original
Barakat–Reif and Tyagi–Reif papers remain paywalled and are represented by their
official abstracts and full-text Reif–Tyagi transcriptions already on disk. This
limitation does not create novelty: later full-text sources explicitly instantiate
the post-fabrication programmable mesh and its calibration/control stack.

### 2.2 Closure-class comparison

| Comparator | Same-task route — HEURISTIC | Closure effect |
|---|---|---|
| FFT ASIC/FPGA | at the natural stakeholder input, phase-rotate \(N\) data subcarriers, then use the existing transmitter IFFT and IQ modulator; if only \(x\) is supplied, also charge the forward FFT | Exact end-to-end direct-task route; the strict intermediate-field route remains an architecture comparison. |
| Near-memory / analog electronic | store \(N\) phase coefficients near a streaming FFT/filter fabric | Device point unresolved; cannot be excluded from CANNOT by definition. |
| Universal MZI mesh | program the desired unitary directly | Contains the family with \(O(N^2)\) tuners and known calibration/loss burdens. |
| Programmable pulse shaper | disperse bins, set \(N\) phase pixels/tuners, recombine | Known specialized wave constructor closest to the candidate upper. |
| Hybrid photonic transmitter | do the diagonal phase map digitally or in a small optical substage and share the same laser/modulator/output | Removes the artificial electronics-versus-optics interface split. |
| Direct downstream algorithm | compile \(D_\theta\) into the subcarrier constellation/equalizer | Attains received-symbol loss without preserving a standalone PFPWO boundary. |

### 2.3 Strict-delta test

**HEURISTIC — STRONGEST SURVIVING SENTENCE.** A previously absent theorem might still
give a fully audited Pareto comparison between one specialized \(N\)-phase optical
pulse shaper and a declared electronic transmitter class on the same finite error
contract. That would be an architecture theorem.

**HEURISTIC — STRICT-DELTA FAILURE.** No model-level sentence survives beyond:
programmable linear optics, pulse shaping, FPGA/FPAA-style post-fabrication control,
VLSIO/capacity accounting, and an ordinary architecture-specific lower bound. The
registered permission and CAN story are occupied; the only potentially unoccupied
residue is the modern end-to-end bill plus a lower bound that this batch could not
even route.

**Iteration-7 token — EMPIRICAL SOURCES + HEURISTIC GATE:** **OCCUPIED-KILL.**
Source absence does not supply clearance, and the sources found are affirmative
occupation evidence. Iteration 8 therefore performs the supervisor-required Stage-A
and overfit verdict as a bounded death certificate, not as physical/theorem
clearance.

---

## 3. Iteration 8 — physical artillery and Stage-A verdict

### 3.1 Finite candidate-upper ledger

Let \(B_{\rm ctrl}\) be verified controller-to-device program bandwidth,
\(\tau_{\rm set}\) and \(E_{\rm set}\) the measured setting latency and
energy per controlled phase element, \(K_{\rm cal}\) the finite calibration
probes, \(f_{\rm ref}\) the refresh rate, \(L\) the optical path length,
\(M\) the number of supported orthogonal modes, and \(P_{\rm ph}\) the photons
used per \(Q\)-vector batch. None is assigned unit cost.

| Resource | Candidate optical bill — HEURISTIC SYMBOLIC LEDGER | Matched baseline bill / unresolved evidence |
|---|---|---|
| Program | read, decode, verify, and store \(Nb\) bits; \(T_{\rm load}\ge Nb/B_{\rm ctrl}\); finite setting/reset bill for \(N\) phase elements | same \(Nb\) bits are stored as per-subcarrier coefficients; no asymptotic loading advantage is established |
| Calibration | acquire \(K_{\rm cal}\) probes, identify pixel/tuner response and cross-talk, store the table, compile \(\theta\) to controls, refresh at \(f_{\rm ref}\), recover after failure | digital coefficients also require quantization/verification, but photonic sources show extra reference, thermal, and bias-calibration paths |
| Reuse | divide one finite load/calibration bill by exactly \(Q\), while retaining footprint, static power, and refresh | same \(Q\) on FFT/FPGA, analog, mesh, and hybrid baselines |
| Input | ingest \(QNb_x\) finite input bits, generate complex electrical waveforms, DAC/modulate, couple to the \(N\) modes | the hybrid baseline uses the same symbol source, DAC/IQ modulation, laser, and endpoint; these costs cannot be charged only to one side |
| Transform | finite aperture/volume, \(M\ge N\) resolvable modes, loss, \(L/c\) propagation, dispersion, phase-mask action, recombination | FFT/diagonal/IFFT can be fused to diagonal-plus-existing-IFFT for this stakeholder; universal meshes and pulse shapers are additional baselines |
| Precision | phase setting resolution, dynamic range, cross-talk, drift, photons/SNR, \(\ell_2\) error, joint failure \(1-\delta\), repetitions | the same codebook/norm/failure is required, but no source gives the composed photon–phase–mode upper or same-task electronic lower |
| Output | phase reference, coherent emission/coupling at the registered link boundary, and any monitoring used for feedback | the matched transmitter must and can emit the same optical field; interface cost is shared |
| Space/energy | aperture, tuner count, controller/DAC area, thermal/static tuning, laser wall-plug energy, peak power, packaging | electronic/analog/hybrid area, memory, arithmetic, DAC, laser, and package must be measured on the same throughput point |

**HEURISTIC — FINITE-UPPER VERDICT.** The ledger is finite and exposes the necessary
measurements, but the repository does not contain a size-indexed physical
construction with values or proved scaling for these coordinates. Therefore no
nonempty optical Pareto region is established. Inferring one from propagation time
alone would reintroduce exactly the unit-cost wave, mode, SLM-write, photon, and
calibration leaks forbidden by the design gate.

### 3.2 Best-baseline attack

**HEURISTIC.** The strongest baseline is not a planar CPU. It is the existing
coherent optical transmitter with digital per-subcarrier phase rotation, a streaming
IFFT, matched DAC/IQ modulation and laser, optionally augmented by a programmable
photonic filter. For the registered circulant family it:

1. receives the same \(Nb\) program bits after fabrication;
2. stores exactly \(N\) phase coefficients;
3. at the natural data-symbol stakeholder input, applies \(D_\theta\) before the
   IFFT already required to form the OFDM waveform; if the artificial contract gives
   only \(x\), it pays the forward FFT explicitly;
4. emits the same all-mode coherent field at the same link boundary; and
5. attains the same received-symbol loss after \(C_\theta\).

The candidate optical pulse shaper may reduce a hardware constant or move an
energy/latency point. No electronic CANNOT applies to this fused route, and the
universal photonic and hybrid baselines already contain the claimed permission.

### 3.3 Lower-bound and capacity attack

**EMPIRICAL — SOURCE SCOPE.** Thompson does not cover the approximate coherent task.
Reif–Tyagi assume the optical DFT primitive and already compare it with electronics.
Miller and Li–Monticone constrain finite optical modes/geometry. Photon-limited ONN
papers constrain task accuracy in their architectures. Meech et al. empirically show
that conversion/data movement can dominate an optical Fourier accelerator. No
retrieved source composes these into the required same-task electronic lower bound.

**HEURISTIC — RESULT.** The transfer screen failed, so there is no robust cut/rank
weapon to apply. A mode-capacity theorem constrains the optical upper as well as
communication; it does not by itself exclude the fused hybrid transmitter. The
claimed residual primitive—“finite-capacity coherent interference and transport
through a post-programmed wave volume”—is therefore not isolated from ordinary
programmable filtering plus matched optical emission.

### 3.4 Supervisor-required audit-checklist overfit test

| Overfit question | Finding |
|---|---|
| Did post-fabrication arrival expose a new mismatch? | **HEURISTIC — NO.** It removes instance-specific fabrication, but both the digital transmitter and established photonic meshes accept post-fabrication coefficients. |
| Did coherent all-mode output force the new primitive? | **HEURISTIC — NO.** It forces both sides to pay a coherent optical endpoint; the hybrid transmitter emits the field after fused per-tone programming. |
| Did finite \(b,\epsilon,\delta,Q\) prevent hidden infinity? | **HEURISTIC — PARTLY.** The syntax is finite, but no finite photon/calibration/mode scaling proves the upper, and finite syntax does not create an advantage. |
| Is the mismatch load-bearing for stakeholder loss? | **HEURISTIC — NO.** \(A_\theta\) is diagonal in the OFDM subcarrier basis. At the natural data-symbol boundary direct programming achieves the same loss; giving only \(x\) manufactures an internal transform boundary the stakeholder does not force. |
| Did the card merely pass controls? | **HEURISTIC — YES.** The card was tailored to avoid advice, output, and precision objections, yet the tailored operator aligns exactly with a standard transmitter factorization and an occupied programmable-photonics line. |

**HEURISTIC — OVERFIT VERDICT.** The card overfit the constraint map. Its controls
make the comparison honest, but the honest comparison removes the alleged
computational mismatch. This is the supervisor's load-bearing distinction: formal
audit survival is not positive evidence.

### 3.5 Infinity-leak audit

| Leak | Status |
|---|---|
| arbitrary real phase | **SEALED BY DEFINITION:** \(b\)-bit phase alphabet |
| per-instance fabrication | **SEALED BY TEMPORAL ORDER** |
| infinite reuse | **SEALED BY FIXED \(Q\)** |
| unbounded mode count | **NOT SEALED BY A PHYSICAL UPPER:** \(M\), aperture, and wavelength remain measurements/obligations |
| unit-time propagation/DFT | **REJECTED:** \(L/c\), dispersion, and device path must be charged |
| calibration oracle | **REJECTED BUT UNQUANTIFIED:** compiler/table/probes/refresh are explicit, not yet bounded |
| free photons/phase reference | **REJECTED BUT UNQUANTIFIED** |
| unmaterialized output | **SEALED SYMMETRICALLY:** both sides emit at the optical boundary |
| nonuniform size-specific fabric/table | **CHARGED, NOT ELIMINATED:** uniform construction and table acquisition remain unproved |

The unsealed physical-upper obligations independently block clearance. They are not
used to claim that the optical implementation is impossible.

### 3.6 Stage-A verdict

**ABSORBED — HEURISTIC, FULL ARGUMENT.** The finite task is truth-apt, so the verdict
is not VOID. It does not return **TASK-MISMATCH-CLEAR** because:

1. the adaptive coherent-OFDM stakeholder, at its natural data-symbol boundary,
   admits an exact direct-task implementation by per-subcarrier phase rotation
   before its ordinary IFFT; insisting on \(x\) instead makes the registered
   intermediate field artificial rather than stakeholder-forced;
2. that matched hybrid implementation emits the same coherent field at the same
   physical boundary and receives the same \(Nb\)-bit program after fabrication;
3. post-fabrication programmable optical linear maps, universal meshes,
   pulse-shaping operations, control compilers, and calibration are established
   lines, so there is no strict model-level delta;
4. no same-codebook, same-norm, same-failure lower-bound route transfers from the
   exact DFT literature; and
5. the symbolic optical ledger establishes no nonempty region after loading,
   calibration, precision, photons, modes, input, and output are charged.

The strongest remaining claim is a possible architecture benchmark between a
specialized pulse shaper and a fused coherent transmitter. That does not identify a
new first-class computational resource.

**DEPENDENCY DISPOSITION — HEURISTIC.** PFPWO founding work ends here. No CAN or
CANNOT theorem is registered. Per research/DESIGN_GATE_2.md section 6, failure of the
transfer route, occupation of the model story, and absence of a nonempty physical
region each independently fire the end-of-batch escalation row.

---

## 4. Explicit unresolved source and measurement obligations

These obligations are preserved for any future architecture-only use. They do not
hold the founding card open.

1. **SOURCE-INCOMPLETE — EMPIRICAL:** obtain the full Barakat–Reif 1987 and
   Tyagi–Reif 1990 papers rather than relying on official abstracts and Reif–Tyagi's
   full-text restatements.
2. **SOURCE-INCOMPLETE — EMPIRICAL:** complete a theorem-level search for lower
   bounds on programmable circulant filters under approximate coherent output.
   Exact query: “programmable circulant unitary” lower bound coherent output
   electronic area-time.
3. **MEASUREMENT-OPEN — EMPIRICAL:** choose one actual pulse shaper and obtain
   size-indexed program bandwidth, calibration probes/time, drift refresh, insertion
   loss, phase error, aperture/mode count, photons, laser/controller energy, and
   packaging data at fixed \(N,b,Q,\epsilon,\delta\).
4. **BASELINE-OPEN — EMPIRICAL:** obtain the matched streaming-FFT/OFDM-transmitter
   energy, latency, area, DAC, modulator, and laser point under the same throughput
   and error contract.
5. **THEOREM-OPEN — CONJECTURE:** a substrate-neutral composed
   mode–geometry–time–photon–error capacity theorem remains absent from the retrieved
   record. It would audit both sides; it does not by itself rescue PFPWO.

---

## 5. Iterations 6–8 falsification ledger

| Claim under test | Rigor-labelled failure | Surviving kernel |
|---|---|---|
| A post-fabrication circulant phase family forces a wave-specific useful task | **HEURISTIC — FALSIFIED FOR THE REGISTERED STAKEHOLDER:** at the natural data-symbol input the phase law fuses into the OFDM subcarrier map; giving only \(x\) manufactures an internal boundary | architecture benchmark for optical pulse shaping |
| Exact/approximate DFT lower-bound technology transfers | **HEURISTIC — TRANSFER-FAIL** | approximate coherent-output lower bounds remain open as a general architecture question |
| Programmable-wave permission is a strict model delta | **EMPIRICAL SOURCES + HEURISTIC — OCCUPIED-KILL** | modern full-ledger theorem could refine known programmable photonics |
| Full physical accounting leaves a nonempty wave-only region | **HEURISTIC — NOT ESTABLISHED** | a future measured point may show a constant/Pareto hardware benefit |
| Checklist survival predicts a real mismatch | **HEURISTIC — FALSIFIED BY OVERFIT TEST** | the controls remain useful as a gate, not as positive evidence |

## NARRATIVE

**Honest one-sentence story — HEURISTIC.** *The programmable light machine passed the
paperwork but failed the job test: for its chosen optical-OFDM task, an ordinary
transmitter can load the same phase program into the subcarriers it already creates
and emit the same light.*

**“Isn't that obvious?” test — HEURISTIC.** It is obvious once the operator is written
as \(F^*D_\theta F\) next to the OFDM IFFT. The non-obvious work was forcing the
same endpoint, program timing, precision, and stakeholder loss, then checking that
programmable photonics had already occupied the purported permission.

**“So what?” test — HEURISTIC.** This prevents a constraint-map-engineered card from
becoming a founding model merely because it survived a checklist. It does not produce
a positive capability or theorem pair.

**Mission test — HEURISTIC.** A random person would say “that is sensible,” not “that
changes everything for me.” PFPWO is a killed search card and possible accelerator
benchmark, not the moonshot. The binding next action is program-level
**ESCALATE-REFRAME**, not a rotation to another material.
