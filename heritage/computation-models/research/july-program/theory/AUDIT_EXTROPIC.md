# Extropic TSU-versus-GPU Audit Dry Run

**Date refreshed:** 2026-07-14  
**Claim under audit:** approximately 10,000× less inference energy for a thermodynamic/probabilistic computer running a Denoising Thermodynamic Model (DTM) than GPU-based generative models on a simple image benchmark  
**Verdict:** **UNVERIFIABLE END TO END**  
**Verdict scope:** this is not a finding that the estimate is false. It is a finding that the available primary record does not price a matched preparation-to-delivered-output system.

## 1. Fresh primary basis

The original Extropic TSU-101 URL was fetched on 2026-07-14, but its current page did not expose the article body to the available crawler. The audit therefore does not reconstruct its wording from third-party summaries.

The refreshed primary basis is stronger:

1. [Jelinčič et al., “An efficient probabilistic hardware architecture for diffusion-like models,” npj Unconventional Computing (published 2026-07-02)](https://www.nature.com/articles/s44335-026-00075-3), authored principally at Extropic and funded by Extropic.
2. [Extropic's current THRML documentation](https://docs.thrml.ai/en/latest/), which defines the software stack as block Gibbs sampling for hypergraphical/energy-based models and identifies the same hardware structure.

**EMPIRICAL — source statement.** The paper says a future all-transistor probabilistic device could match GPU generative-model performance on binarized Fashion-MNIST at about 10,000× less energy per generated sample. It explicitly describes this as a combination of measured RNG-circuit data, circuit models, and simulations—not an end-to-end full-chip measurement.

**EMPIRICAL — source caveat.** The test chip measured individual probabilistic circuits and did not implement an Ising machine or computational architecture. The paper says the hardware experiments used simulations because the probabilistic hardware is currently unavailable, calls the system a first step rather than a finished product, and states that the model omits practical effects best resolved by a real-chip demonstration.

This claim is therefore classified as:

> **VENDOR-AFFILIATED PRIMARY MODEL CLAIM, PARTLY GROUNDED IN COMPONENT MEASUREMENTS; NOT A MEASURED FULL-SYSTEM BENCHMARK.**

## 2. Matched output contract

The narrowest contract supported by the primary source is:

- **task:** inference from a trained generative model on binarized Fashion-MNIST;
- **delivered unit:** one generated image sample;
- **quality metric:** Fréchet Inception Distance (FID);
- **comparison:** DTM chains of 2–8 EBMs against GPU VAE, GAN, and DDPM points at varying inference costs;
- **energy boundary:** inference energy per generated sample;
- **excluded unless separately stated:** training, model compilation, weight flashing, host-to-device control, sample transfer to a host, and downstream use.

**HEURISTIC — contract status: PARTIALLY MATCHED.** Comparing different algorithms at a common FID is a legitimate task-level comparison in principle. The headline does not by itself give a fixed target FID, estimator sample count/confidence interval, named GPU/software/precision point, independence criterion, or amortization count. The paper also states that its probabilistic models are far from state-of-the-art raw modeling quality and that the datasets are much smaller than frontier generative tasks. The 10,000× statement must not be generalized to “AI” or “GPUs” without this narrow contract.

## 3. Translation into (V, L, B, D, I, S)

The vector is kept unscalarized. “Reported” means present in the refreshed primary paper; “missing” means unavailable for a matched end-to-end verdict, not necessarily absent from every supplement or internal record.

| Coordinate | Primary-source translation | Audit status |
|---|---|---|
| **V — causal/parallel volume** | Most reported EBMs use a 70×70 grid (4,900 binary sampling cells) with typically 12 neighbors per variable. DTM depth is 2–8 sequential EBMs; the largest simulated DTM has about 50,000 nodes total. Local wire communication is in the cell-energy model and is reported as about half its budget for the chosen connectivity. | **PARTIAL.** Topology and simulated active volume are stated. A fabricated DTCA's total area, routing, controllers, memory, chip-to-chip links, yield, and idle infrastructure are not measured. |
| **L — trajectory/mixing length** | The energy model uses 250 sampling iterations per EBM for DTM inference. With 2–8 EBM stages, the charged modeled work is proportional to 2–8 × 250 × 4,900 conditional cell updates. | **PARTIAL.** A fixed iteration budget is explicit. “Satisfactorily mix” is not translated into a certified total-variation/output independence tolerance for the delivered sample. |
| **B — resolution/conditioning** | Binary stochastic cells implement a sigmoidal conditional update; the paper reports preliminary nonideality effects relative to 5–8-bit weight quantization. It discusses drift, process variation, bias nonlinearity, and deviations from the ideal update. | **UNVERIFIABLE.** No complete tolerance chain maps fabrication/calibration/weight error to FID and energy at a declared confidence. Precision equivalence with the GPU computation is not fully priced. |
| **D — description/preparation** | The model consists of trained DTM weights, grid connectivity, a sampling schedule, and per-stage programming. The paper permits distinct circuits, multiple chips, or reprogramming the same hardware between EBMs. | **UNVERIFIABLE.** Training/compilation energy and time, weight-description bits, programming hardware, calibration, and the cost/frequency of flashing weights are outside the headline energy model. |
| **I — input/output** | Generative inference begins from the model's prescribed noisy state and delivers an image sample. The paper explicitly identifies external communication—weight flashing and transferring samples to a host—as the most important omitted energy contribution. | **UNVERIFIABLE.** No measured host/device input, control, output-transfer, reset, or final materialization energy/latency is supplied. |
| **S — stochastic sampling/confidence** | Gibbs updates use physical random-bit cells; Kmix = 250 is used per EBM. FID is the quality metric and autocorrelation is used during training to pressure layers toward tractable sampling. | **UNVERIFIABLE.** The headline lacks a certified independence/effective-sample-size target, FID estimator uncertainty, repeat count, failure probability, and energy cost of any rejection/repetition. |

## 4. Side energy account

The paper's modeled probabilistic-device inference energy is

    E_model = T × K_mix × L_grid² × E_cell,

with reported values K_mix = 250, L_grid = 70, and E_cell approximately 2 fJ per conditional cell update. T is the number of DTM EBM stages.

**EMPIRICAL INPUTS + ARITHMETIC (not measurement).** These values imply approximately 2.45 nJ per EBM stage and 4.9–19.6 nJ per generated sample for 2–8 stages, before the explicitly omitted external communication and other unmodeled system effects. This calculation only restates the vendor-affiliated physical model; it is not an independent energy measurement.

The cell model includes:

- random-bit generation;
- biasing circuitry;
- local neighbor communication/wire capacitance; and
- clock distribution.

It excludes or incompletely treats:

- flashing trained weights;
- moving generated samples to external hardware;
- host/controller/memory energy;
- calibration, drift correction, reset, and yield;
- nonideal full-network sampling behavior;
- training and compilation;
- cooling/power-delivery/package losses; and
- the amortization threshold required to make setup and I/O negligible.

The GPU side counts model FLOPs and divides by a manufacturer FLOP-per-joule specification. The authors state this underestimates GPU energy, which is conservative in favor of the GPU. It is nevertheless a modeled peak/specification calculation rather than a measured, named, software-matched end-to-end GPU run.

## 5. Missing-data decision table

| Required comparison fact | Available? | Consequence |
|---|---:|---|
| exact finite task and dataset | yes | binarized Fashion-MNIST generation |
| one delivered output definition | partial | image sample; independence/downstream boundary unspecified |
| common quality target | partial | FID used, but headline target/uncertainty not fixed |
| measured full TSU/DTCA system | no | projected model, not system benchmark |
| named GPU, software, precision, utilization | not in refreshed headline/main-text basis | baseline cannot be reconstructed end to end |
| TSU parameter precision/calibration tolerance | partial | 5–8-bit discussion, no full error-to-output certificate |
| preparation/programming cost | no | D coordinate open |
| I/O and host energy | explicitly omitted | I coordinate open |
| mixing/independence/confidence target | no complete contract | S coordinate open |
| amortization count | no threshold | setup/I/O cannot be normalized |
| inference-only boundary on both sides | broadly yes | training excluded, but programming/load boundary remains |
| reproducibility assets | yes | code/library are linked; this supports scrutiny, not hardware validation |

## 6. Verdict

> **UNVERIFIABLE END TO END.**

**HEURISTIC.** The refreshed paper is materially better evidence than the older marketing summary: it specifies the task, FID metric, architecture, mixing iteration count, cell-energy terms, reproducibility code, and explicit omissions. It still does not support the registered end-to-end verdict because the numerator is a future full-system estimate built partly from component measurements, the most important external communication costs are omitted, the amortization threshold is unspecified, and sampling/quality confidence is not fully matched to a named measured GPU execution.

UNVERIFIABLE is not “disproved.” A several-orders component or modeled inference advantage may survive. The current evidence cannot determine whether the approximately 10,000× ratio survives preparation, calibration, host control, I/O, sample independence, full-chip nonidealities, and matched baseline implementation.

## 7. Minimal verification packet Extropic would need to publish

1. A fabricated DTCA/TSU system running the stated DTM, with rail- or wall-energy measurements.
2. A named GPU, software versions, numerical precision, batch size, utilization, and measured energy.
3. A predeclared binarized Fashion-MNIST model-quality target and identical FID estimator, sample count, and confidence interval.
4. A definition of an independent delivered sample and measured autocorrelation/effective sample size.
5. Energy/latency for weight load, calibration, reset, host control, and output transfer.
6. The number of inferences over which preparation is amortized, with break-even curves rather than a single ratio.
7. Sensitivity to 5–8-bit weights, drift, process variation, bias nonlinearity, yield, and temperature.
8. Separate training/compilation and inference accounts for both systems.
9. Results across scale, including where connectivity/routing and output materialization stop being negligible.
10. Independent reproduction or access sufficient to rerun the full measurement.

## 8. Relation to the killed EFRS direction

**HEURISTIC.** This audit demonstrates that an end-to-end checklist catches real evidentiary gaps, but it does not reverse the novelty KILL. The translation required no new architecture-neutral simulation theorem; it is the floor byproduct the registration explicitly barred from becoming the success path.

## 9. Honest audit sentence

**HEURISTIC.** *Extropic now provides a serious, reproducible model-backed reason to investigate its 10,000× claim, but the claimed ratio still ends before a real programmed machine delivers a confidence-qualified sample against a measured matched GPU.*

---

## 10. Post-review amendment (2026-07-14)

This amendment preserves the original audit above as the error record and supersedes
it where the two conflict. It implements research/REVIEW_1_FINDINGS.md, finding 4,
repair 6, nonfatal repair 9, and final-verdict item 4. The supplement used below is
the paper's [Supplementary
Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs44335-026-00075-3/MediaObjects/44335_2026_75_MOESM1_ESM.pdf).

### 10.1 Original claim, review finding, and corrected headline

**ORIGINAL CLAIM — HEURISTIC.** Sections 2–6 called the comparison **UNVERIFIABLE END
TO END**, said the named GPU/software/precision point was unavailable, treated the
GPU comparison as a manufacturer-spec calculation rather than a named measured
execution, and marked four coordinates **UNVERIFIABLE**. The source pass had not read
the paper's supplement.

**REVIEWER FINDING — EMPIRICAL, PRIMARY-SOURCE VERIFIED.** The full DTCA is still a
projection rather than a measured working architecture, but the supplement names the
NVIDIA A100 and Float32 specification route, reports empirical GPU energy measured
with Zeus, supplies FID/energy comparison points, models initialization and readout,
states its weight-programming amortization assumption, and reports autocorrelation
estimation over time and multiple independent Gibbs chains. Those facts directly
contradict the original “unavailable” wording.

> **CORRECTED HEADLINE — HEURISTIC, WEAKENED:** **MODEL-BACKED, PARTLY
> COMPONENT-MEASURED, REPRODUCIBLE PROJECTION; NOT AN END-TO-END MEASURED HARDWARE
> BENCHMARK.**

This is not a finding that the projected ratio is false or content-free. It is also
not evidence that a fabricated DTM-hardware system has delivered the projected
end-to-end result.

### 10.2 Supplement evidence omitted by the original audit

| Fact | Corrected status | Primary provenance | Residual scope |
|---|---|---|---|
| GPU identity and specification route | **REPORTED** | Supplement F, p. 17: NVIDIA A100; theoretical route uses JAX/PyTorch FLOP estimates and NVIDIA's 19.5 TFLOP/s Float32, 400 W specifications. | A spec route is still a model, but it is named and reconstructible. |
| Empirical GPU energy | **COMPONENT/SYSTEM EXECUTION MEASURED ON GPU** | Supplement F, p. 17: batches were sampled and GPU energy and time were measured with Zeus. | This measures the GPU side, not the unbuilt DTCA side. |
| FID/energy points | **REPORTED** | Supplement Table S3 gives VAE FID 30.5, 27.4, and 17.9 with empirical energy per sample $6.1\times10^{-5}$ J, $1.5\times10^{-4}$ J, and $2.5\times10^{-3}$ J; theoretical values are $2.3\times10^{-5}$ J, $0.4\times10^{-4}$ J, and $1.7\times10^{-3}$ J. | These are task/model points, not a universal “GPU” curve or frontier result. |
| Initialization/readout energy | **MODELLED** | Supplement E.3–E.4, pp. 14–15: $E=T(E_{\rm samp}+E_{\rm init}+E_{\rm read})$, with initialization of all nodes and readout of data nodes included. | The equations price on-chip transport to/from the chip boundary, not a measured host-delivery path. |
| Fuller DTCA estimate | **MODELLED** | Supplement E.4, p. 15: about $1.6T$ nJ, with $E_{\rm init}+E_{\rm read}\approx0.01T$ nJ; for $T=2$–8 this is about 3.2–12.8 nJ total and 0.02–0.08 nJ for initialization plus readout. | This is a high-level future-device estimate, not a rail/wall measurement. |
| Weight programming and amortization | **REPORTED ASSUMPTION / OPEN MEASUREMENT** | Supplement E.5, p. 16: parameter memory writes cost more than maintenance; the estimate assumes infrequent writes—program once, then run many sampling programs—so maintenance dominates. | No measured programming bill or numeric break-even/amortization count is supplied. |
| Sampling and correlation diagnostics | **REPORTED** | Supplement E.4 uses $K=250$ conservatively after layer autocorrelation and FID-versus-chain-length checks; Supplement G, pp. 17–18, estimates autocorrelation by averaging over time and multiple independent Gibbs chains. | This is an empirical diagnostic, not a proof-grade TV or universal independence certificate. |

**EMPIRICAL INPUTS + ARITHMETIC, RETAINED.** The original side calculation
$250\times70^2\times2$ fJ $=2.45$ nJ per EBM is arithmetically correct for its
simplified inputs. **MODELLED, SUPERSEDING SYSTEM ESTIMATE.** The supplement's fuller
model gives about $1.6T$ nJ and explicitly includes initialization/readout. Both
numbers are projections; neither is a measurement of a complete DTCA.

### 10.3 Reissued (V, L, B, D, I, S) table with provenance

The statuses below replace **UNVERIFIABLE** with **REPORTED**, **MODELLED**, and
**OPEN**. Multiple statuses are intentional: a coordinate may contain reported inputs
and modelled costs while retaining an open end-to-end boundary.

| Coordinate | Corrected status | Reported/modelled content | Provenance | Open boundary |
|---|---|---|---|---|
| **V — causal/parallel volume** | **REPORTED + MODELLED + OPEN** | The 70×70 grids, typically 12 neighbors, 2–8 stages, and modeled local communication/clock fabric are stated. | Main text, DTCA description; Supplement E.1–E.4. | Fabricated DTCA area, routing, controllers, memory, chip-to-chip links, yield, idle/supporting load, and scaling are not measured end to end. |
| **L — trajectory/mixing length** | **REPORTED + MODELLED + OPEN** | $K=250$ is used for the energy estimate; layer autocorrelation decays in tens of iterations and FID shows little benefit beyond about 250 in the reported model. | Supplement E.4, Figs. S6–S7; Supplement G. | The diagnostic does not establish a proof-grade mixing guarantee or an architecture-wide independent-sample contract. |
| **B — resolution/conditioning** | **REPORTED + MODELLED + OPEN** | Sigmoidal RNG behavior, PDK/process-variation simulations, and 5–8-bit weight/nonideality discussion are reported; circuit behavior is propagated through a physical model. | Main text circuit measurements/nonideality discussion; Supplement E. | No measured full-network tolerance chain maps fabrication, calibration, drift, and coefficient error to FID and energy with system error bars. |
| **D — description/preparation** | **REPORTED ASSUMPTION + MODELLED + OPEN** | Weights/biases require on-chip memory; write energy is acknowledged and treated as amortized by infrequent programming over many runs. Training is outside the explicitly inference-only headline. | Supplement E.5(a), p. 16. | Programming hardware, measured write/load energy, calibration, model-parameter storage, and a numeric break-even count remain open. |
| **I — input/output** | **MODELLED + OPEN** | Initialization and readout are explicit terms, about $0.01T$ nJ together in the fuller model; within-chip communication to the edge is modelled. | Supplement E.3–E.5, pp. 14–16. | Host/device transfer, control, package/PCB/system integration, reset, and delivered-output energy/latency are estimated incompletely or omitted from the headline. |
| **S — stochastic sampling/confidence** | **REPORTED + MODELLED + OPEN** | FID is the empirical task metric; $K=250$, autocorrelation curves, FID-versus-chain-length, time averaging, and multiple independent Gibbs chains are reported. | Main text FID comparisons; Supplement E.4, F, and G. | FID estimator uncertainty, exact repeats/sample counts in the headline comparison, full-chip correlations/nonidealities, and independent reproduction remain unresolved. |

### 10.4 Corrected comparison interpretation

**EMPIRICAL — GPU SIDE.** The baseline is not unnamed: the A100 execution has empirical
Zeus energy measurements, and a separate Float32 peak-spec route is compared against
those measurements. The original audit's contrary statement is withdrawn.

**HEURISTIC — METRIC SCOPE.** The paper's stated claim is inference energy at matched
observed FID on binarized Fashion-MNIST. A proof-grade terminal-law TV certificate is
not a prerequisite for that empirical comparison. FID uncertainty, sampling
protocol, autocorrelation, and correlation across delivered samples remain legitimate
scrutiny. The comparison must not be generalized to all AI, all GPUs, or
state-of-the-art generative quality.

**HEURISTIC — BOUNDARY SCOPE.** Excluding training is not itself a mismatch when both
sides' headline is explicitly inference-only. Training/compilation may be reported
separately. Within inference, weight loading/programming, host I/O, control, reset,
and amortization remain part of the matched boundary and are not erased by excluding
training.

### 10.5 Surviving cautions

1. **EMPIRICAL — OPEN.** The test chip measures individual probabilistic circuits; it
   does not implement a working Ising machine, DTM, or complete DTCA.
2. **MODELLED / OPEN.** The DTCA numerator is a future-system projection based on
   component measurement, circuit models, and simulation. The supplement itself says
   the high-level model likely underestimates a complete device.
3. **MODELLED / OPEN.** Initialization/readout are included, and a more detailed PCB
   communication analysis is reported as not changing the system conclusion, but
   off-chip integration and the complete host/control/power-delivery boundary remain
   estimated or omitted rather than measured.
4. **HEURISTIC — OPEN.** The infrequent-write assumption is an amortization model, not
   a measured break-even curve.
5. **EMPIRICAL — OPEN.** Code is published, but no independent reproduction of the
   complete model-to-comparison pipeline or hardware result is recorded in this
   audit.

### 10.6 Reissued verdict and residual verification target

> **HEURISTIC — REISSUED VERDICT:** **MODEL-BACKED, PARTLY COMPONENT-MEASURED,
> REPRODUCIBLE PROJECTION; NOT AN END-TO-END MEASURED HARDWARE BENCHMARK.**

The corrected residual verification target is bounded: fabricate and measure the
complete programmed DTCA; report full-chip and host/off-chip/control energy and
nonideality error bars; publish programming/load amortization curves; report the FID
sampling protocol and uncertainty at the compared points; and obtain independent
reproduction. The named A100, Float32 route, Zeus measurements, initialization/readout
model, and autocorrelation analysis are no longer listed as missing.

### 10.7 NARRATIVE

**Honest one-sentence story — HEURISTIC.** *Extropic has a reproducible, component-
grounded projection and a measured A100 comparison—not a fabricated end-to-end
thermodynamic computer delivering the projected result.*

**“Isn't that obvious?” / “so what?” status — HEURISTIC.** The distinction is
sensible audit hygiene, not the project's moonshot. Its value is that the record now
credits real evidence without converting a future-system projection into a measured
hardware fact.

---

## 11. ACDFT theorem activation after Proof Review 2

**ACTIVATION STATUS — PROVED THEOREM INPUT + HEURISTIC AUDIT DISPOSITION.** The
reviewed ACDFT chain now gives the audit pillar a theorem-backed test, but only for
the exact high-accuracy full coherent-DFT claim class in
`research/ACDFT_LEMMA.md`. It does not make every physical-computing audit a
theorem application.

### 11.1 Extropic typecheck

| ACDFT premise | Extropic claim | Disposition |
|---|---|---|
| every input lies in the frozen finite complex grid and the task is normalized DFT | trained DTM generative inference from a noisy state | **TYPE MISMATCH** |
| all $N$ coherent complex outputs are resident at named terminals | one image sample judged by FID | **OUTPUT MISMATCH** |
| global coherent error $d_{\rm coh}\le\epsilon$ in the theorem's narrow positive regime | empirical FID/autocorrelation evidence; no coherent global-$\ell_2$ contract | **METRIC MISMATCH** |
| finite interactive capacity certificate for every balanced named-output cut | component/model energy accounting, not an ACDFT cut-capacity certificate | **PREMISE ABSENT** |
| conditional area–time use additionally supplies $C_{\rm cut}\le\kappa\sqrt A$ | no matching layer/pitch/channel-aware ACDFT geometry law | **PREMISE ABSENT** |

> **HEURISTIC — EXTROPIC VERDICT UNCHANGED.** (RCUT) and the repaired conditional
> area–time theorem provide no new lower bound on the DTM energy projection. The
> claim fails the task and output type before any capacity or geometry comparison.
> The corrected section-10 verdict remains the operative audit result.

### 11.2 New teeth and exact limit

**PROVED-AT-SCOPE CONSEQUENCE.** For any real implementation claim that *does*
promise the frozen full coherent DFT on the finite grid, with the reviewed failure
and finite operational cut-capacity premises, (RCUT) rules out sublinear
cut-information-time in its fixed small-error regime. Interaction, feedback,
randomness, channel noise, and local coding do not evade that statement. If the
claim also proves the separate geometry law (G), the repaired conditional
area–time inequality applies.

**HEURISTIC — CORPUS STANDING.** This is new audit leverage at the level of a real
claim class: an admitted high-accuracy coherent-DFT accelerator cannot treat
cross-cut information as free. It is not yet a concrete device adjudication. No
source currently audited in this repository supplies the identical input grid,
full coherent terminal output, theorem-level error/failure promise, operational
capacity certificate, and geometry constant. Intensity-only optical inference,
generic matrix-vector multiplication, partial Fourier output, remote-field output,
and decision tasks do not inherit the theorem.

**MILESTONE-C CONSEQUENCE — HEURISTIC.** The audit pillar is activated but Milestone
C is not complete: this typecheck neither instantiates a second materially different
physical claim nor supplies the required false-positive/false-negative attack. The
next admissible corpus use is a source-backed coherent-DFT accelerator whose claim
can be normalized line by line; absent that, the theorem must not be presented as
having adjudicated today's photonic-accelerator market.

