# Cross-project insights for LSR (2026-10-11)

> Historical extraction. The projects are now unified here; use
> [UNIFIED_EVIDENCE](../docs/UNIFIED_EVIDENCE.md) and [heritage](../heritage/README.md)
> for current corrected reading and relocation. Original absolute source paths
> below describe the capture environment, not surviving research homes.

These are insights taken from two sibling projects, plus the trades between them. Both repos stay separate and read-only.

- **Matrix Native Math (MNM):** `C:/Users/devan/OneDrive/Desktop/Projects/matrix-native-math`, the public mirror `C:/Users/devan/matrix-native-public`, and `C:/Users/devan/matrix-native-public_claims_check.md`.
- **LLM Genome / Neural Dissection Lab:** `C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome` (written GEN below). This covers HEAD `c6f2ed4`, the history below `106aeff`, and its `INBOX_from_matrix-native-math.md` and `REPLY_to_matrix-native-math.md`.

**Method.** Three independent read-only sub-agents did the deep reads, and the operator checked their results. Each insight records:
- **Status:** MEASURED (with its scope) or CONJECTURED. A transfer to LSR is always conjectured unless an LSR run tests it.
- **Effect on LSR:** which hypothesis, prediction or experiment the insight changes.

**Framing.** The question is how mathematics changes at d >> 1. LLM latent spaces are one instance. Most GEN numbers come from a single 2.74M-parameter char-level transformer (width 192) and are n=1 per cell. Its recipes fade at 10M parameters, and its facts live in weights, while LSR's branches live in an activation vector. Read GEN results as analogies, not measurements of LSR objects.

## A. Insights that bear on the flagship (latent branch capacity)

**A1. Spread is not function. Isotropic spread is the null.**
- **Source:** GEN `NOTES.md` x6/x10, `THINKING.md` T250, and `THE_LAWS_V3.md` Laws 3 and 5.
- **MEASURED:** GEN's "73.5% fill law" (energy at 50/90/99% = 0.12/0.44/0.735 of the room) is identical whether the stored fact is alive or dead. Two sub-agents independently computed that the isotropic-Gaussian null is 0.124/0.444/0.734, so the "law" is that null.
- **LSR effect:** participation ratio is a spread measure of the same kind. It can fall inside the admissible band for a code that stores nothing, so it stays scaffolding. Report PR beside its isotropic null (PR/r) and with the top principal component removed.

**A2. Finite-sample bias in PR can void the 256->1024 scaling test.**
- **Source:** operator-side calculation by a GEN reader sub-agent.
- **CONJECTURED, but checkable analytically:** for a perfectly isotropic state, the expected sample PR at n = 1,024 samples is about 60/205/511 for r = 64/256/1024. That makes the 256->1024 PR ratio about 2.5x, below the 3.2x admissibility floor. At n = 4,096 the PR values are about 63/241/818.
- **LSR effect:** the pre-registered PR uses 512 differences, which falls into exactly this trap.
- **Amendment (proposed):** use n >= 8r (8,192 at r = 1024) or a bias-corrected PR. Report the isotropic-null PR at the same n next to every measured PR.

**A3. Capacity is a pair (width, precision), and an information ceiling can bind at r = 64.**
- **Source:** MNM `research/MATRIX_NATIVE_INFORMATION_THEORY.md` §2, `DIGIT_GEMM_COMPLEXITY_V1.md` Thms 2–3, and ledger `TWO_AXIS_CROSSOVER_DERIVED` (03f72180).
- **Status:** the counting and bit laws are proved-pending-gate in MNM. The transfer to LSR is conjectured.
- **LSR effect.**
  - Each branch carries up to 16 bits. The information ceiling is C <= r·b_eff/16, so C(64) <= 32 with an effective 8 bits per coordinate.
  - At r = 64 this ceiling may bind before the sqrt(PR) interference law does. That would push the fitted exponent toward 1 and contaminate the 64->256 ratio.
- **New eval-only arm (no retraining):** quantize z to b in {2, 3, 4, 8} bits per coordinate before readout.
  - An interference-limited (superposed) code should keep C(r) flat down to about 4 bits.
  - A cliff at 6–8 bits means an amplitude or precision code, which is a different mechanism.

**A4. Near a capacity edge, expect cliffs, floors and large seed scatter.**
- **Source:** GEN e272, e280 and the Day 12 report.
- **MEASURED** (weight-space rank ladder, n=1):
  - Formation is dead at 1k dimensions at any dose. Reads ramp from 0.027 at 2k to 0.27 at 10k.
  - The floor sits at (1k, 2k] under AdamW and at (2k, 5k] under SGD-M. Above the floor the slopes agree within about 20%.
  - Redrawing the room shifts results by about 21%, and up to 2.9x.
- **LSR effect.**
  - C(r) at a 90% threshold edge can scatter by up to 2x across seeds. Report full per-seed accuracy-vs-k curves, and C at 80/90/95% as a sensitivity analysis.
  - Refine the k-grid densely at the low end for r = 64.
  - If r = 64 sits near a floor, use a floor-offset diagnostic C = a(sqrt(PR) − c), which is not the prediction itself.
  - Orderings replicated in GEN much more reliably than magnitudes (confident seat claims went 0/9; registered leans scored 29/55). So give the ordering C(64) < C(256) < C(1024) its own verdict, separate from the exponent.

**A5. Concurrent formation collides; serial formation survives.**
- **Source:** GEN e268–e273 and e284, Law 2a.
- **MEASURED:** a write formed alone survives (0.2646). The same write formed while other training traffic flows dies (0.00004, a 6,609x gap). Persistent small shared-gradient components dominate a shared momentum buffer.
- **LSR effect (CONJECTURED):**
  - The frozen recipe trains every k (1..64), every update count (0/1/2), and both global and addressed toggles concurrently through one LoRA plus E/D. That is a concurrent-formation regime.
  - Rare or hard cells (high k, two-update chains, addressed edits) may never form.
  - **Mitigation (diagnostic only, no recipe change yet):** log per-cell dev accuracy (k × updates × toggle type) during training.
  - The pre-registered repair (double the exposure, halve the lr) is a dose change, and GEN found dose does not rescue a sub-floor room (A6). If cells stall, a staged curriculum is the GEN-informed repair, and it would need a dated amendment.

**A6. Exposure does not create capacity; only width does. Adam's early steps are sign steps.**
- **Source:** GEN e272, opt1/opt2, e188 and T255.
- **MEASURED:** a 1k room at 3.73x the lr moved more than a working 10k room and still stayed dead. Adam's step size grows as lr·sqrt(N), regardless of gradient size.
- **GEN overclaims flagged:** the "80% in-room" figure is the identity sqrt(2/pi). "Adam kills, SGD teaches" compares runs whose movement differs by 1,683x.
- **LSR effect:** if r = 64 fails, read that as a width verdict, not a training deficit. The repair cannot be expected to rescue it.

**A7. A stored item's readout depends on its context, so collateral damage is coupling, not overwrite.**
- **Source:** GEN e290 and e283, Law 2b, T263–T269.
- **MEASURED** (one class, n=1):
  - A readout dies from orthogonal drift of only 0.07–0.32% of the write norm, with survival roughly proportional to 1/drift.
  - Meanwhile 84–93% of the write itself is still in place (cosine 0.993).
  - Removing the out-of-room drift brings the readout back 4,328x.
- **LSR effect:** two additional diagnostics.
  - (i) Measure collateral as a dose-response over the edit magnitude ||Δz||, using scaled edits.
  - (ii) Run the projection-out test: remove the edit's component outside the edited branch's subspace and see whether collateral recovers. This distinguishes overwrite from readout coupling.
  - Also run repeated sham round-trips through E/D to fit retention per pass. That gives a decay clock for longer update chains.

**A8. Non-structure-preserving updates mix at a forced rate.**
- **Source:** MNM `RESEARCH_MONOGRAPH.md` Ch. 3 (Thms 3.3, 3.5, 3.6a–d, 3.7; commits a365bf45, dc12f900) and §8.3 (ef4997b5).
- **Status:** proved for unitary families. Products whose spread stays bounded commute with a fixed operator; otherwise they drift to the Haar mean. MEASURED: a pre-registered n=150 prediction was off by 0.5%. Under perturbation, approximately separated structure dies at the word-count rate (n=12).
- **LSR effect (CONJECTURED transfer, since LSR updates are nonlinear).** Maintaining k branches through updates requires the update map's Jacobian, restricted to branch-variation directions, to be near block-diagonal. For global toggles the natural closed form is I_k ⊗ U, which is Kronecker-closed with no descriptor growth.
  - **New diagnostic:** take the finite-difference Jacobian of one update on the branch-variation subspace. Report its off-block energy and its distance from Kronecker rank 1.
  - **Prediction:** without that structure, collateral grows with the number of updates rather than staying flat. This gives a computation-vs-storage signature that does not depend on accuracy parity.

**A9. Tolerance belongs in every capacity law.**
- **Source:** MNM `research/GIRS_EXPLICIT_APPROXIMATION_V1.md` (GIRS) and the CRC disposition. Proved and reviewed.
- **Finding:** exact width can be Ω(sqrt(n/log n)) while width-2 approximants reach error ≤ 2/n.
- **LSR effect:** C(r) is defined at 90%, and the exponent can depend on that threshold. This is a second reason, beyond A4, to report C at 80/90/95%.

**A10. Recovery by construction is a live hazard.**
- **Source:** GEN `code/mc001_qwen3_controlled_v10_head_localization.py:130-165` at commit 106aeff.
- **MEASURED:** the Genome era's only bounded card claimed an exact 1.0000 mediation recovery. A reader verified that this is an identity by construction.
- **LSR effect:** audit undo and self-swap so that they cannot reinstate a cached state.
  - In branch.py, undo re-applies the toggle through the network (z_u = step(step(z0))) and never copies z0, so it passes this check.
  - Record this explicitly and keep it that way.
  - Custody comparisons need a tolerance, never bit-equality (see C1).

**A11. HANDLE is prior art for the flagship and came with a NO-GO rationale.**
- **Source:** GEN `research/04_MATH_AGENDA.md` §11 (fdb0825) and `research/59*/60*`.
- **CONJECTURED:** a finite-state key-value store that writes an answer-sufficient code explains every planned I/O observation. HANDLE-0 was never trained.
- **LSR effect:** this strengthens the need for the storage control. It also suggests two additions:
  - a matched explicit-cache or GRU competitor;
  - a cheapest-mechanism null: the same E/D and LoRA on a random-init backbone. In GEN g125, interface-only training closed 42.7% of a gap.

**A12. Interference geometry is randomized structure.**
- **Source:** GEN e283.
- **MEASURED:** a random direction overlaps a k-of-N subspace at sqrt(k/N) (0.0604 = sqrt(10k/2.74M)). This is a clean real-network check of the concentration that HP-E assumes.
- **Related MNM result:** compositions of structured operators saturate to dense at about sqrt(d) factors (Monarch; `results/monarch_growth_v1/REPORT.md`, 22/22 cells for q = 2–5; novelty unestablished). This is an independent appearance of sqrt(d) as a mixing scale.
- **LSR effect:** sqrt(d) appears as a mixing scale in several unrelated mechanisms. That is an analogy (corrected after Droid review 1); it is not evidence for the flagship's capacity law, which remains a prediction. Real weights have PR of about 0.1d (MNM `FIXED_M_COMPILER_V2.md` §14, two OPT matrices), so use d_eff = PR, never d.

## B. Insights that bear on Probe B (latent search) and Probe C (recurrence)

**B1. Residual steering in frozen models never beat random or matched controls.**
- **Source:** GEN MC001, MC005 and MC006 cards at 106aeff, and DAY_ONE L2.
- **MEASURED:**
  - In Qwen3-0.6B (n=288), a random vector at h14 gave a margin of +0.776, more than the learned direction (+0.487).
  - Damage follows perturbation energy, not content.
  - Renormalizing the stream at eval time costs 3.56 nats.
- **LSR effect:**
  - B's latent_rand arm is essential and must keep its norm schedule matched.
  - **Prediction:** random-direction search recovers most of any latent_es gain.
  - **C:** add a norm-matched (renormalized) loop arm. Choose the looped block per checkpoint on dev data, since GEN found the causal gate layer moves with seed (r 0.50–0.80).
  - **C:** a contiguous block of about 3 layers is a natural unit. MC005 found a superadditive three-layer block at L24–26.

**B2. Off-position and late writes are transient; mid-depth, decision-position writes act.**
- **Source:** GEN E055, E056b, E056c.
- **MEASURED (2.74M):** a teacher-forced state written at depth >= 4/6 rescues knowledge (0.374–0.494 vs a shuffled 0.000). The mean donor beats every individual donor. Off-position writes decay within 2 tokens.
- **LSR effect on B:** add an arm that perturbs only the final prompt position at mid-depth (Codex r2 already specified the final prompt position). Add an averaged-candidate direction to the ES. The current all-positions arm is kept as a contrast.

**B3. Per-row dynamic range absorbs small perturbations in bf16.**
- **Source:** GEN `runs/x1/range_census.csv`, and MNM `research/OZAKI_FAIR_ENGINE_V1.md` §2.4 and §3.8.
- **MEASURED:** per-token spread is 6.8–8.1 bits, and up to 13.5 bits on the worst token.
- **LSR effect:** a shared-scale isotropic delta loses its components in massive-activation channels and attention-sink tokens to bf16 rounding. Log the absorbed fraction ||(h+δ)_bf16 − h − δ|| / ||δ|| per run, and scale per token.

**B4. Worst-case norm bounds overstate instability by 10²–10⁶.**
- **Source:** MNM `results/certificate_law_v1/REPORT.md`.
- **MEASURED** on one synthetic family.
- **LSR effect on C:** judge "instability" by measured trajectory divergence per loop (iterate deltas and finite-perturbation growth), never by spectral-norm bounds.

## C. Instrument insights (all probes)

**C1. Batch shape is a rounding boundary.**
- **Source:** GEN `REPLY_to_matrix-native-math.md` x3, and MNM `research/CURRENT_FOCUS.md` lines 149–152.
- **MEASURED:** one 256-row GEMM versus two 128-row GEMMs re-rounds 85% of output bits. Same-shape reruns are bit-exact.
- **LSR effect:** each arm has its own batch composition (B: B copies vs B deltas vs sequential repair). Measure the noise floor per arm and per batch shape, and report the answer-flip rate under batch-shape perturbation.

**C2. Matching outputs does not mean matching states, and cross-model bases are near-orthogonal without a learned map.**
- **Source:** GEN E041, E108, T053, and the era-1 g061/g124/g182 results.
- **MEASURED:** cosines between independently trained nets are 0.016–0.06. Leave-one-architecture-out chart fits give R² of −11 to −19.
- **LSR effect on deferred bet 4:** only a learned map judged by intervention is admissible.
- **The narrow surviving variant:** same-tokenizer transfer, anchored on shared token rows.
  - GEN g191 and g194 (MEASURED, Qwen3 family only): exact-match token rows held by an anchor give +0.465 nats, and the direction carries the effect.
  - The dissection lab found token rows to be an inert intervention seat (x22, x26).
  - So token rows are a coordinate system for training anchors, not a read or write basis. Do not use them for the bottleneck Q or for the B basis.

## D. Overclaims in the sibling projects (do not import)

- **GEN:**
  - The fill law is the Gaussian null.
  - Law 1 (every death is a step-death) is true by definition.
  - The 80% in-room figure is sqrt(2/pi).
  - The rank-ladder results are intrinsic-dimension measurements (Li et al. 2018), uncited.
  - The MC005 recovery is an identity.
  - Its laws rest on n=1 cells at one scale.
- **MNM:**
  - The public engine headline (11–17x) compares against a 1/64-rate FP64 laptop GPU. The honest margin is about 1.03–1.14x at n=4096.
  - Several timings were retracted (the cuda-graph artifact).
  - The novelty of the Monarch law is unestablished.
  - The "|λ2/λ1| = 1.000 curiosity" (GEN runs/x2) is a complex-conjugate eigenvalue pair.

## E. Proposed changes (for the re-council; nothing is adopted until a dated amendment)

1. **PR estimator:** n >= 8r samples or a bias-corrected estimator, with the isotropic null reported next to it (A1, A2). This must be fixed before any flagship data.
2. **New eval-only arms and diagnostics for the flagship:**
   - quantized-z capacity (A3);
   - C at 80/90/95% and per-seed curves (A4, A9);
   - per-cell training logs (A5);
   - collateral dose-response, projection-out and sham round-trip decay (A7);
   - the Jacobian block-structure diagnostic, with collateral tracked against update count (A8);
   - a random-init backbone null (A11).
3. **Probe B:** add a final-position mid-depth arm and an averaged-candidate direction (B2), log the bf16 absorbed fraction (B3), and measure the per-arm batch-shape noise floor (C1).
4. **Probe C:** add a norm-matched loop arm, a contiguous 3-layer block in the dev sweep (B1), and trajectory-divergence instability metrics (B4).
5. **Candidate new high-dimensional bet:** the precision-width tradeoff of superposed states. Measure capacity as a function of (r, b) and test whether it follows a single law in r·b_eff or separate interference and information regimes, with a crossover where sqrt(PR)-limited capacity meets the r·b/16 ceiling. MNM's two-axis crossover is the template.
