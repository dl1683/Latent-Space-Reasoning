# Experiments Log

## Unified-program authority — 2026-10-13

All future experiments belong to this repository's [shared research program](../theory/RESEARCH_PROGRAM.md).
Current execution and adoption decisions are in [STATE.md](../STATE.md).
The dated records below preserve their original scopes and controls; their historical
"next" instructions do not queue a run. Consolidation itself launches no experiment
and contributes no scientific outcome to this log's denominators.

## Gemma 4 31B instrument controls v1/v2 — passed (run IDs dated 2026-10-15)

- `gemma4-controls-2026-10-15`: identity difference 0; live-hook difference 2.8125; donor-follow 1.0 (64 cases), unpatched 0, shuffled 0.28125, majority 0.359375. [V1 results](results/cloud/gemma4_controls_summary.jsonl); [independent review](../docs/council_2026_10_15_gemma_controls_codex.md).
- `gemma4-controls-v2-2026-10-15`: identity difference 0; live-hook difference 3.96875; donor-follow 1.0 (64 cases). Full token-difference checks: 0/64 failures. Batch-shape option-logit difference 0.375, argmax agreement 1.0 (16 cases), donor residual difference 2.0. Receiver-self-SET difference 0 (64 cases). [V2 results](results/cloud/gemma4_controls_v2_summary.jsonl).
- Scope: Amendment G4 layer-15 W1 four-option instrument smoke only; no portable latent memory, general mediation, learned or economic result. V1 review requested the controls recorded in v2; no separate v2 fresh-review verdict is claimed. No run was executed during this documentation sync.
- Run names retain 2026-10-15; spend custody records completion on 2026-10-14 UTC. Historical execution commit was not recorded in the summaries; ledger references identify source-history snapshots, not a verified execution receipt.

## Fixed-layer re-score: 300 names beat 20 by 0.06-0.10 at both layers; 10-point k=4 prediction not replicated (2026-10-12 20:39, Codex hunt passed)

- These fitted per-slot linear readers show higher held-out-name accuracy after training on 300 names than 20 at both fixed layers and both tested k values; the k=4 ten-point prediction is not replicated.

## Training-name ablation: fewer names lowers held-out-name readout ~0.08-0.11; registered k=4 criterion met for W2 only (2026-10-12 20:06, Codex hunt passed)

- Under the registered selection procedure, restricting reader training to 20 names reduces held-out-name linear readout accuracy in both families; only W2 meets the registered k=4 magnitude criterion, and differing selected layers limit causal attribution.
- What we learned: name count explains part of R's held-out gap; layer choice is a live confound.

## N-control: slot queries read the closing state almost perfectly at k=4, even for novel names (2026-10-12 19:29, Codex hunt passed)

- In this smoke, closing states support slot-indexed colour recovery above histogram mode across both templates and rename/reorder conditions; the tested name-query recipes remain inconclusive.
- k=4: 0.99 in all rename/reorder quads (baseline 0.53); k=16: 0.62-0.73 (baseline 0.38). Name queries fail on the same states.
- What we learned: the information is slot-ordered and name-general with 300 training names; R's held-out gap likely (not proven) came from 20 names.

## Registration N binding smoke: gates pass, reader inconclusive; no full run (2026-10-12 18:58, Codex hunt passed)

- All four N smoke cells pass the instrument gates but remain reader-inconclusive: the selected entity-query readers do not exceed histogram mode; these results establish neither name-addressable retrieval nor its absence.
- What we learned: with 300 training names the entity-query reader cannot even reach the colour-mix baseline, so the
  R readout's success may be slot-ordered or reader-specific. Next: slot-query control on the same states.

## Registration R closing-state readout: binding information readable at k=4; at k=16 only for seen names (2026-10-12 18:09, Codex hunt passed)

- 8B's registered readers recover slot-ordered colour information beyond histogram information at k4; selected-layer unseen k16 fails E, while seen k16 passes. Larger-model smokes show the same qualitative split pattern.
- 8B full (n_test 2048 per split): unseen-name k4 0.80/0.83 vs colour-histogram 0.53; k16 0.43/0.45 vs 0.38 (E fails);
  seen-name k16 0.86/0.81 (gap +0.43/+0.37); rank needed: unseen k4 6/12, seen k4 8, seen k16 48.
- Not scaling evidence (different n_train/grids/seeds across models). Seen-name advantage is not yet name-addressable binding.
- What we learned: the closing state is not empty - a frozen position that single additive patches cannot move still carries
  linearly readable slot-colour information, far above the colour-mix baseline, and much more of it for familiar names.
  Next decisive control: rename x reorder with an entity-query reader.

## Q1-prime Amendment A 8B value-site sweep: colour-pair means carry the whole edit; no contextual dimension; no scaling (2026-10-12 17:01, Codex hunt passed)

- Across six layer-9 value-site cells with 512 TEST-name collections each, DEV-fitted colour-pair means achieved 99.61-100% donor-follow and retained >=90% of the sham-subtracted full effect; all gates passed, r90=0 with bootstrap CI [0,0], and no registered contextual-dimension comparison passed.
- What we learned: on held-out names the value-token edit is fully lexical at this layer; the open question moves to whether
  binding information is *readable* at the closing position (Registration R, smoke running).

## Q1-prime Amendment A 8B smoke: value-site effect is the colour-pair mean; closing site inadmissible (2026-10-12 16:18, Codex hunt passed)

- On six 64-collection DEV smoke cells, layer-9 value-token interventions passed the registered gate; fitted colour-pair means alone achieved 100% donor-follow, while residual-only edits achieved 0%, licensing no contextual-dimension claim; closing-position admission failed in both families at layers 9/18/27.
- What we learned: the value-token edit is carried by the 12 lexical pair means (residual adds nothing); a single additive patch at the
  closing token moves no answers at layers 9/18/27. Scaling is not warranted under the registered rule. Next lead (Codex): read the
  closing state (held-out linear readers per fact slot, rank-bottleneck curve) before more writing.

## Q1-prime 8B smoke: gates pass 6/6, full sweep DEFERRED (2026-10-12 15:50, Codex hunt passed)

- Full and rank-4 value-token patches: 100% donor-follow, sham 0%, collateral <=0.8%, frozen random ~0 (n=64 x 6 cells, dev).
- Licensed: local mediation under retained receiver context; NOT context-dependent dimension or branch capacity.
- What we learned: with 4 colour values the value-token difference is near-lexical (<=3-dim span), so r90=4 is near-forced;
  the registered k-growth and cross-model predictions are not testable by this intervention. Amendment needed before more spend.

## Registration Q1-prime registered; 8B instrument validated (non-claim) (2026-10-12 ~15:38Z)

- q8b-validate (Codex+Droid hunted): the 8B hook works and value-token donor replacement drives answers (98-100%
  donor-follow at layers 9/18, 44-53% at layer 27, 0% unpatched). This does not establish that the closing position is
  absent from the answer path. Registration Q1-prime (theory/LATENT_BRANCH_CAPACITY.md) moves the patch to the
  changed fact's value token, with frozen random controls, per-cell gates, and a separate n=64 8B smoke before any sweep.
- What we learned: the earlier Q null came from where we patched, not from a dead hook.

## Registration Q (frozen Qwen3-8B, native selectable information): NO VERDICT — instrument unvalidated (2026-10-12 15:25Z)

Licensed (Codex hunt): "Q1 mediation is unestablished; Q2 fails the tested transplant interface." Q1 full patches at the
closing position: 0/512 donor-following answers in every cell; Q2 transplants of the last 1-8 fact-independent suffix
positions: chance (0.22-0.27) at k=1 and 4 while full context is 1.00. The hook positive control failed and token
alignment was not asserted, so the zeros are not yet interpretable. (A first run, q8b, was voided: the registered
>= 8-token suffix had not been applied and its "perfect m=8 transfer" was the last fact's own tokens.)

## Registration F (formation collisions, Qwen3-8B LoRA): INSTRUMENT INVALID (2026-10-12 14:36Z, Codex hunt passed)

Staged positive control fails (no seed has both staged orders >= 0.90 in every family); staged - interleaved after
consolidation +2.1 to +2.9 pp (bar 15). Licensed: "Sequential training shows substantial forgetting; interleaving
retains more before consolidation" (pre-consolidation staged means 0.44-0.47 with earlier families at 0.17-0.37;
interleaved 0.88-0.90). "The moments recipe reaches means 0.990-0.996 before consolidation, with unresolved optimizer
confounds" (global-step bias correction inflates its effective step 1.5-1.9x). The Genome-style concurrent-formation
penalty was not observed in this pretrained LoRA setting; the dominant effect was sequential forgetting.

## Registration P3 (future distinguishability): INSTRUMENT INVALID (2026-10-12 14:20Z, Codex error hunt passed)

"P3 is instrument-invalid under its registered FULL-control kill rule; descriptively, STATE scored 0.731-0.751,
exceeding SHUFFLED by 18.6-20.2 pp and CURRENT by 23.1-25.1 pp, with unresolved SHUFFLED sampling-role confounding."
(FULL positive control 0.905-0.916 < 0.95.)

## Registration P2 (update closure): registered prediction FAILS (2026-10-12 13:45Z, Codex error hunt passed)

"At the tested exposure, consistency training reduces variance-normalised encoder-target defect but misses P2's
registered answerability and selective-edit thresholds in all three seeds. This construction establishes neither
update closure nor functional equivalence from reduced geometric defect." Paired two-update gains (consistency vs
answer-only): k=1 +0.7/+0.4/+0.6 pp, k=4 +2.7/+10.6/-1.6 pp (bar +15 pp); defect k=1 u=2 reduced 1.74x/3.9x/4.6x;
no arm reaches 0.90 at every selector; shuffled-target edits change the target only with large collateral.
What we learned: moving the updated state closer (in squared distance) to the encoder's state for the updated
graphs did not make the reader answer the updated question — a dissociation between geometric and functional
closeness in this construction.

## Registration P1-prime pilot: stop rule FIRES (2026-10-12 13:45Z)

Synchronized mastery curriculum, seed 11: mean dev k=1 bit accuracy rose to 0.90 (r=64) / 0.96 (r=128) (vs ~0.60 under
P1's schedule) but promotion needs every selector >= 0.95 at both leaders; max min-selector 0.64/0.86; k_max stayed 1.
No positive C90 established; width scaling not tested; remaining seeds not launched.

## Registration P1: stop rule FIRED (2026-10-12 04:35Z)

Random-access edge storage, Qwen3-0.6B, seed 11, independent widths r=2..128, registered schedule (20% easy then the
full k mix): pooled dev k=1 bit accuracy flat at ~0.60 at every width from step 250 to the cap; the restarted r=64/r=128
pilots finished step 1,562 (99,968 presentations) at 0.603/0.624 vs the 0.95 bar (48 dev collections x 16 selectors;
an oracle constant answer per selector reaches only 0.566, so this is not a label artifact; Codex error hunt passed). Per the registration: remaining seeds not launched, no capacity claim. Amendment 3 activates (P1-prime
synchronized mastery curriculum; P2 update closure on the locked gate checkpoints). Observation (descriptive): the
same flat-at-chance-plus-epsilon signature as the concurrent gate arm.

## LOCKED: concurrent gate arm FAILS (2026-10-12 01:22Z)

Frozen concurrent recipe, 3 seeds, test: seeds 11 and 47 stayed at constant output for the full run (all cells at
base rate); seed 23 learned k=1 storage (0.902) and nothing else. The registered repair (double exposure within
caps) is inapplicable: the full cap was already used. Neither gate arm passes (arms not pooled): no escalation.
Cross-project note (descriptive): concurrent mixture stalls, staged mixture learns — the same contrast the Genome
lab reports for weight-space formation (CROSS_PROJECT_INSIGHTS A5); here n=3 seeds per arm, one task.

## Probe B (latent search): registered prediction FAILS (2026-10-12 00:27Z)

14-checkpoint dev screen + 4-checkpoint confirmation (512 new problems/family, budgets 8 and 32, text arms at
dev-tuned temperature with sequential feedback repair). Verifier-guided residual ES never beats a matched-norm
random perturbation in any confirmation configuration (4/4). Best latent-vs-text gap: Qwen3-8B +2.6 pp (B8) / +2.5 pp (B32),
below the 5 pp bar and in one family only; at budget 32 text repair beats latent by 9-18 pp in the other three
checkpoints. Licensed sentence (evidence gate 1): "The tested confirmation configurations miss the registered
improvement threshold and show no replicated ES advantage over random perturbations; matched-compute superiority
remains unestablished." (Correction: one dev cell had ES > random, Qwen3-0.6B l14: 0.070 vs 0.047; verifier calls
and FLOPs were not exactly matched; the countdown verifier accepted '**', now fixed.)

## Probe C (frozen recurrent depth): registered prediction FAILS (2026-10-12 00:10Z)

14-checkpoint screen + 4-checkpoint confirmation (512 new problems/family, dev-frozen configs). CoT: every
dev-admitted loop lowered admissible-family accuracy (-3 to -9 pp); 8/14 models admitted no loop on dev.
Direct mode: Qwen2.5-7B countdown +2.5 pp [0.4, 4.9] (< 3 pp bar); Qwen3-1.7B/4B null/negative; Qwen3-8B
"+6.6 pp" is an artifact (the unlooped model emits empty answers for 507/512 problems; looping unblocks output).
Licensed sentence (evidence gate 1): "No admissible tested configuration establishes the registered
replicated recurrence benefit; confounded cells remain instrument failures." (Controls matched nominal layer/token
allowances, not measured FLOPs; the Qwen2.5-7B +2.5 pp interval extends above 3 pp.) Instrument lessons: constant-answer baselines and empty-answer rates are mandatory
beside termination.

## LOCKED: admission gate FAILS (2026-10-11 23:15Z)

Curriculum arm (Amendment 2), Qwen3-0.6B, 3 seeds, test split, corrected evaluator. k=1: no updates 0.97-0.99,
one update 0.84-0.86, two updates ~0.76 (bar 0.90 at every update count). Branch-addressed edits: targeted
effect ~0 pp (bar 20 pp); edit-then-advance accuracy 0.41 (below chance). Coarse-grid storage (every selector) passes through k=2-4. Width 64/256/1024 nearly identical -> width scaling not tested. Consequence (pre-registered): no
1.7B flagship escalation. Licensed sentence (evidence gate 1): "The documented curriculum construction fails admission in all three
seeds; these results license no flagship escalation and leave width scaling untested." (Storage C90 values are
coarse-grid; main~sham at one seed does not prove updates are ignored; the 0.41 edit-then-advance accuracy is below
chance but was not paired-scored against the unedited graph; the concurrent arm is recorded FAIL in all three seeds (ledger `gate_verdict_concurrent`); arms not pooled.) Next: Registrations P1 (random-access storage capacity vs width) and
P2 (update closure with a consistency target).

## GCP latent-reasoning program (2026-10-11, running)

Pre-registration: theory/LATENT_BRANCH_CAPACITY.md — Amendment 1 (PR at 8,192 pairs + isotropic null, quantized-
state arms, C80/C95, per-cell logs; motivated by theory/CROSS_PROJECT_INSIGHTS.md, the user-directed extraction from
Matrix Native Math and the LLM Genome lab), Amendment 2 (curriculum gate arm), Deviation 1, Registrations P1/P2,
Amendment 3 (contingent). Council and audits: docs/council_2026_10_11/. Runner: experiments/cloud/lsr_cloud.py.
Spend: experiments/cloud/SPEND.md.

- **Runner smoke (L4):** auto-off self-test PASS; KV-safe layer looper exact at loops=1 (logit max diff 0.0);
  identical-input reruns identical.
- **Probe C pilot v1 (instrument fail; remedy configured, not yet validated):** truncation confound caught by evidence gate 1 — direct mode
  0% termination at 16 tokens; CoT 37-42% at 384. Remedy (answer prefill + newline stop; CoT budget 1024)
  before any interpretation. What we learned: Qwen3 non-thinking mode still reasons in "answer only" mode.
  Pilot v3 after the remedy (Qwen3-1.7B, n=24 dev): direct 64-token termination 0.875-0.958, CoT 2048-token
  termination 0.83-0.92 -> STILL below the 95% gate; capability interpretation stays blocked per family/arm.
- **Gate throughput pilot (A100, 60 steps):** pipeline end to end; 10.6 s/step at batch 64; all cells at chance.
- **Probe C instrument lesson (constant-answer artifact):** the graph shortest-path family has a modal
  answer ("3") on 62-66% of items (generator keeps distances >= 3). In direct mode, dev-selected loops "gained"
  by shifting the constant answer from "2" to "3" (Qwen2.5-0.5B test: base 0.091 -> loop 0.198 overall; graph
  0.258 -> 0.586 with 3 distinct predictions, still below the 0.617 constant-answer baseline). Every C/B
  comparison now reports the per-family constant-answer baseline and prediction diversity; graph direct-mode
  cells are uninformative. Early CoT test reads on the admissible arithmetic family (termination 1.00): loops
  0.727 -> 0.695 (Qwen2.5-0.5B), 0.055 -> 0.023 (Qwen3-0.6B) — no gain so far (screening, n=128/family).
- **Admission gate (Qwen3-0.6B, 3 seeds, H100):** running. DEV observation (not a verdict): the frozen
  concurrent recipe emits a constant answer in every dev cell through step 750 in 3/3 seeds (loss 0.695-0.697).
- **Diagnostics diag-k1 (k=1, u=0 only):** text positive control dev 0.984 (step 400); bottleneck state
  variant 0.926 at step 250 (lr 2e-4); state at lr 1e-3 collapses to chance. Interface is learnable.
- **Curriculum arm (Amendment 2, 3 seeds, H100):** DEV at step 1250 (seeds 23/47; still in stage k<=8):
  k=1,u=0 0.94-0.97; k=4,u=0 0.74-0.84; k=1,u=1 0.73-0.78; u=2 0.60-0.73; k>=16 chance. Same sampler and
  code as the stalled concurrent arm; only the k/update schedule differs (cf. CROSS_PROJECT_INSIGHTS A5).
  What we learned so far: carrying a state forward through update passages is the hard part.

## HP-D Orlicz arc: optimality stack at the settled compression optimum (2026-09-20)

### s=1 gap diagnosis (12 seeds x 6000 steps, frozen decision rule)

At (B=8, rho=0.3, s=1): d=2 best 0.0857 (all seeds within 0.0005), 6.2x the
Fisher prediction 0.0139; d=4 best 0.0253, 4.7x. VERDICT: genuine
large-deviation gap — the optimizer is not the limitation; the quadratic
expansion is. Runner: internal/s1_gap_lean.py (lean rerun after the original
40-run job stalled under contention); data s1_gap_diagnosis.json.

### CGF decomposition at the settled optimum

Excess 0.0859 = quadratic component AT the optimum 0.0966 (7x the
Fisher-optimal 0.0139, ABOVE the total) with kappa_3 = -0.005, kappa_4 =
+0.014 partially canceling. The CGF optimum accepts ~7x the variance to buy
negative-skew cumulant structure: the argmin changes qualitatively.

### Envelope numerics (withdrawn conjecture)

The two-point variance-KL envelope's optimizer runs to the boundary; inf KL
= 0 at any fixed variance (rare huge negative jumps). The v/2 - c v^{3/2}
discount law is withdrawn; variance does not govern KL compression — the
one-sided exponential moment does.

### Stationarity and curvature certificates

- Stationarity: off-space gradient energy = 1.0000 at the settled optimum
  (gradient entirely in the complement block — E-Y-type orthogonality;
  condition statement corrected en route, both directions recorded).
- Lanczos Hessian (analytic gradients, 45 iters, 1040-dim factor space):
  lambda_min = 2.4e-05 > 0, lambda_max = 0.252 — CERTIFIED STRICT LOCAL
  MINIMUM; near-flat valley explains the 0.0005 seed spread.
- Failed schemes, logged: alternating metric-reweighting (0.0117 invalid)
  and unconstrained DV dual (unbounded inner problem; 0.390 spurious) —
  both establish that KL compression lacks E-Y's norm geometry; the dual
  route requires regularization or a simplex-native formulation.

**Ledger:** hpd_s1_gap_verdict_d2, hpd_s1_gap_complete, hpd_cgf_decomposition,
hpd_variance_kl_envelope (withdrawn), hpd_orlicz_correction, hpd_dv_core,
hpd_tilted_svd_probe_invalid, hpd_first_order_characterization,
hpd_first_order_verified, hpd_dual_attempt_failed, hpd_second_order_evidence,
hpd_strict_local_certified
**Theory:** theory/HPD_ALLOCATION_THEOREM.md sections 12-12j

---

## HP-E: aggregate interference theorem — proved, corrected, validated (2026-09-19)

### Isotropic quadratic theorem (COMPLETE)

For unit-norm isotropic features and the energy readout M = I:
E[A_2] = k(k-1)/d EXACTLY; Var(A_2) = k(k-1)(2d-2)/(d^2(d+2)) EXACTLY (all
shared-index covariances vanish identically via conditional independence +
exact unit norms); two-phase capacity frontier k* = Theta(sqrt(d)) with
Chebyshev bounds (mean-field below, aggregate-interference failure above).

### Validation (pre-registered, frozen header in the runner)

P1 anchor: 8/9 cells within 0.5-4.6% (one k=4 finite-size cell 13%);
P2 concentration: pass; P4 crossing at sqrt(d): pass (A_2 = 0.87-0.97 at
k = sqrt(d), 14.8-15.7 at 4 sqrt(d)). P3 REFUTED the registered general-t
power: measured E[term^2] = ||T||^2/d^t to the digit — lemma corrected; the
Theta(1) diagonal channel is special to quadratic/projection structure, and
the degree>=3 extension is open with that explicit obstruction.

### E-struct theorem (anisotropic; expectation level + validated)

Unit-energy features with covariance Sigma: per-pair interference =
tr(Sigma^2)/(tr Sigma)^2 = 1/PR(Sigma) EXACTLY; capacity frontier
k* = Theta(sqrt(PR)); power-law corollary k* = Theta(d^{1-alpha}) for
alpha > 1/2 — a power-law spectrum taxes quadratic mean-field computation by
d^{alpha-1/2} vs isotropy. Validated across alpha in {0,.5,.75,.9},
d in {512,2048}: per-pair exact to 0.4-2.4% at alpha<=0.5; crossings
bracketed 8/8; two formal registered fails carry diagnosed benign causes
(exact-vs-expectation normalization at alpha=0.9; grid-gap). Variance
identity: isotropic cancellation breaks under anisotropy, replaced by a
PR(Sigma^2)-bounded covariance with alpha-dependent burn-in
k >> d^{2alpha-1}; intermediate band open (martingale route suggested).

**Data:** experiments/results/hpe_aggregate/{anchor,estruct}_{check,verdicts}.json
**Runners:** experiments/run_hpe_{anchor,estruct}_check.py
**Theory:** theory/HPE_AGGREGATE_INTERFERENCE.md sections 3b-3i
**Ledger:** hpe_e5_anchor_validation, hpe_estruct_validation,
hpe_isotropic_theorem_complete, hpe_anisotropic_variance
**What we learned:** spectrum -> capacity is lawfully bridgeable via
participation ratios (no fitted slopes); the naive degree-p ladder does not
exist (per-vector 1/d pairing); trained power-law geometry REDUCES quadratic
mean-field capacity relative to isotropy — the opposite of the packing
intuition.

---

## HP-A A1: QSAE optimum proven sparse-exact; mean field refuted (2026-09-18)

### Theorem (sparse-exact optimum of the quadratic superposition autoencoder)

For the QSAE loss L(W) = Σ_i p_i ‖W^T W e_i − e_i‖² with usage weights p,
code dimension d < n: writing G = W^T W (PSD, rank ≤ d) and P = diag(p),
L = tr((G−I) P (G−I)). On ker G (dim n−d), G−I acts as −I, so
L ≥ tr(P|_{ker G}), minimized by ker G = span of tail coordinates; the
minimum is the tail mass Σ_{i>d} p_i, attained by any orthonormal unit-norm
storage of the top-d usage tokens. The optimal frame operator is W W^T = I_d:
**completely flat spectrum.**

### Verification (L-BFGS, multi-init, 6 configs)

s ∈ {0.5, 1.0} × (n,d) ∈ {(64,16),(256,64),(512,128)}, 5 independent inits
each. Results (medians): loss − tail-mass bound = **0.000000 at all six
configs**; nonzero-norm count k0 = d exactly; fitted power-law exponent at
the optimum α ≈ 0.00; the P1′ mean-field bulge profile fits at R² = 0.00–0.08
(threshold for refutation: < 0.8). **Mean-field ansatz REFUTED at the
optimum; the optimum anti-mixes (organizes orthogonality) rather than mixes.**

### Consequence for the program

Power-law spectra cannot come from the quadratic objective's optimum — they
must come from optimization dynamics (transients) or from softmax retrieval
geometry. The rate-distortion s/2 benchmark and mean-field bulge are retired
as QSAE comparators. Prediction P-E added (before any HP-B run): converged
QSAE GD endpoints are flat (α ≤ 0.35), ≈ d nonzero norms, loss → tail mass.
The XE (softmax) arm is now the main event of HP-B.

**Data:** experiments/results/hpa_random_frame/a1_meanfield.json
**Runner:** experiments/run_hpa_random_frame.py (shared: hp_spectral_lib.py)
**Ledger:** hpa_a1_result
**NOTEBOOK entry:** 119

### A2: the spectral estimator itself is biased — measurement-validity result

Random-frame Monte Carlo (norm profiles × random directions, d ∈ {32..1024},
γ ∈ {4,16}, 20 draws; corrected instrument: ambient dimension d, not the
positive-norm count — bug found and fixed before any result was used; A1
unaffected, a2/a3 killed and rerun):

- **A2-i REFUTED:** top-90% log-log OLS does not recover the norm-profile
  exponent at d ≥ 256: max deviation 0.19 (γ=4, β=0.25) and 0.19 (γ=16,
  β=1.0). Bias is a function of (γ, β) only — asymptotic in d by ~128.
- **A2-iii REFUTED at γ=4:** uniform (flat) input reads α̂ = 0.40–0.45
  (threshold 0.35). Passes at γ=16 (0.20–0.23 — matching the record's
  "flat init ≈ −0.25" readings at real-model γ).
- **A2-ii (specificity) CONFIRMED:** bulge inputs manufacture steep slopes
  (α̂ up to 8.9–12.0 at s=1) but the curved-alternative diagnostic correctly
  rejects them (ΔR² = 0.13–0.28 in favor of the quadratic-in-log fit; power-law
  R² only 0.34–0.50).
- Even TRUE power-law inputs at shallow β show curved fits winning (ΔR² up to
  0.17): the random-frame map bends the log-spectrum (head spikes + MP floor).

**Consequence for the record:** spectral exponents measured this way are
comparable only within fixed (γ, window). Entry 101's cross-γ exponent sweep
embeds this bias (magnitude partially artifact; the direction
"higher overcompleteness → flatter" may survive but needs bias-corrected
rereading).

### A3: transient rate + the undershoot signature

Independent pilot (exact gradient flow, seeds 90–92, (64,16,s=1)): α rises
0.53 → 1.03+ over t=250..1250 — rich-get-richer confirmed at the exact
dynamics level. Calibrated rate ρ = 2.755 (single free constant, first
checkpoint only). Prediction map at (1024,64,s=1) uses the A1-theorem
sparse-exact endpoint. Mechanism discovered inside the map: a saturated
head-only random frame at γ_eff=1 has a Marchenko–Pastur spectrum that the
estimator reads as α ≈ 1.15 — so a late-time measured α BELOW the norm-only
prediction is the dynamic signature of learned orthogonality (directional
organization beyond norms), cleanly separable by the N1-t control.

**Data:** experiments/results/hpa_random_frame/{a2_spectral_map,a3_transient}.json
**Ledger:** hpa_a2_result, hpa_a3_result

---

## Controlled Eigenspace Vulnerability — Entry 92 Overturned (2026-09-18)

### Controlled eigenspace perturbation (entry 118, SmolLM2-135M)

Injected fixed-magnitude perturbations into individual gate_proj eigenspaces
at the MLP input (post-LN) across layers 0, 15, 29. All 576 eigenspaces
tested at each layer. Result: POSITIVE beta at all layers (sensitivity
INCREASES with eigenvalue, not decreases). Top eigenspaces are 3-14x more
sensitive than tail. Overturns entry 92's "tail vulnerability" interpretation.
The SNR prediction (beta=-0.5) rejected; observed beta = +0.25 to +0.41.

Entry 92's correlation was confounded: natural perturbations that project into
top eigenspaces at late layers are SMALL (absorbed by earlier layers), while
tail-projecting perturbations are LARGE (unabsorbed). The tail vulnerability
is a multi-layer processing artifact, not an eigenspace property.

**Data:** experiments/results/frame_analysis/eigenspace_vulnerability.json
**Script:** scratchpad/eigenspace_vulnerability.py
**NOTEBOOK entry:** 118

---

## General HD Mathematics — Analytical Results (2026-09-18)

**Direction pivot:** From model-specific experiments to general mathematical
exploration of high-dimensional spaces. Scope: pure math, statistical physics,
information theory, random matrix theory, kernel methods.

### Interference hierarchy (entry 117, analytical)

[WITHDRAWN 2026-09-19, see ledger hpe_e5_anchor_validation] Derived: in R^d, t-body interference terms in degree-p polynomial maps scale
as d^{-(t-1)/2} (RMS). Proven for p=2 (quadratic). Connects to cluster
expansion (stat phys), 1/N expansion (QFT), Hoeffding decomposition (statistics).
The "mean field" of HD spaces is independent feature processing.

Surviving scoped result: for isotropic independent unit features and the specified normalized quadratic readout, E[A2] = k(k-1)/d with exact variance and a crossing at Theta(sqrt(d)); no general degree-p hierarchy or operational capacity law follows (ledger `hpe_isotropic_theorem_complete`).

### Representation-computation gap (entry 117, analytical)

[WITHDRAWN 2026-09-19, see ledger droid_review_2hr_1] Derived: store O(exp(d)) features (packing), linear computation on O(√d)
simultaneously (coherence-limited), bilinear computation on O(1) simultaneously
(dimension-independent). The gap is exponential. Training closes it by structuring
weights. Intermediate dimension m provides computation bandwidth, not d.

Correction: the gap is not a universal capacity law; orthonormal features permit exact bilinear computation up to k <= d (ledger `droid_review_2hr_1`; NOTEBOOK entry 121).

### Rate-distortion optimal allocation (entry 117, analytical)

[WITHDRAWN 2026-09-19, see ledger droid_review_2hr_1] Proved: for Zipf(s) inputs, optimal eigenvalue exponent α* = s/2. Lagrange
optimization. Observed model exponents (0.35-0.90) are steeper than optimal.
The 1-body (mean-field) allocation becomes exact as d → ∞.

Correction: the allocation calculation has no established eigenspectrum or d-to-infinity interpretation (ledger `droid_review_2hr_1`; NOTEBOOK entry 121).

**Theory notes:** scratchpad/interference_hierarchy.md [artifact missing as of 2026-10-12], scratchpad/hd_computation_capacity.md [artifact missing as of 2026-10-12]
**NOTEBOOK entry:** 117
[WITHDRAWN 2026-09-19, see ledger hpe_e5_anchor_validation and droid_review_2hr_1] **What we learned:** HD spaces have a natural hierarchy where independent feature
processing is the dominant term, with corrections suppressed by powers of d.
This connects neural network architecture (depth, width, gating) to fundamental
HD geometry. The representation-computation gap explains why transformers need
both width (for packing) and depth × intermediate width (for computation).

---

## Row-Permutation Series: Composition is a Tensor Phenomenon — 2026-09-18

**Central question:** What invariant captures MLP composition? Can static
weight analysis characterize gated MLP function, or is composition
fundamentally data-dependent?

### Row-permutation challenge (entry 103)

Prescribed by Droid adversarial review (entry 102). Permuted up_proj rows
randomly (seed 42) on SmolLM2-135M, all 30 layers, 10 prompts.

**Result:** Permutation preserves all D_k (diff < 5e-08) and all Gram
operators, but destroys MLP output (mean relative error 139%, range
1.04-1.64). Function-preserving control (consistent permutation across
gate/up/down) has zero error (2.5e-07). Separation: 5.5 million to 1.

### Row-wise cosine analysis (entry 104)

Computed cos(gate_i, up_i) for all 1536 neurons per layer.

**Result:** Mean cosine = -0.001, fraction positive = 0.499. Indistinguishable
from random. In d=576, all pairwise cosines concentrate at zero by HD
geometry. Row pairing matters but carries no per-pair directional signature.

### Cross-matrix SVD (entry 105)

Computed SVD of G = W_gate^T @ W_up (d x d) for original and permuted pairing.

**Result:** Nuclear norm ratio 1.004, effective rank 410 vs 413, power-law
exponent 0.372 vs 0.370. Nearly identical. The cross-matrix spectrum does
NOT distinguish correct from random pairing.

### Bilinear vs gated sensitivity (entry 106)

Three modes under same permutation: SwiGLU, linear-gated (no silu), additive.

**Result:** SwiGLU error = 1.39, linear-gated = 1.28 (ratio 1.09),
additive = 0.88. The silu nonlinearity adds only 9% extra sensitivity.
The elementwise multiply (gating) adds 46% over additive. The multiplicative
tensor structure, not the activation function, drives composition sensitivity.

**Converging conclusion:** No tested weight-only invariant captures gated MLP
composition. D_k, row cosines, and cross-matrix SVD are all blind or washed
out. The compositional structure is a 3-way tensor phenomenon created by
multiplicative gating, not reducible to pairwise matrix statistics.

### Predictive rewiring challenge (entries 108-109)

Droid-prescribed decision experiment. 60 partial permutations per layer,
5 test layers, train/test prompt split.

**Entry 108:** Input-conditioned tensor predictor (gate+input, no silu) has
R2 = 0.90 on held-out prompts vs DU (gate-blind) R2 = -2.4. But DU's poor
R2 is calibration transfer failure, not ranking failure (correlation = 0.97).
Note: activation-weighted predictor was tautologically perfect (same formula).

**Entry 109 (decisive):** Weight-only tensor ||ΔT||_F (includes gate via
Hadamard product tr(GG^T ⊙ D^TD ⊙ ΔUΔU^T)) vs weight-only DU ||DΔU||_F.
R2: tensor 0.94, DU 0.95. Nearly identical. Gate structure in weights adds
NOTHING to damage prediction. Input-conditioned tensor only marginally better
(R2 = 0.96).

**LINE CLOSED.** Per Droid's decision rule: DU suffices, gate-value tensor
coupling is not a discoverable invariant. The sensitivity to row permutation
is about how D amplifies changes in U, not about gate-up coupling.

**NOTEBOOK entries:** 103-109
**Data:** experiments/results/row_permutation/
**Scripts:** row_permutation_challenge.py, row_aware_analysis.py,
cross_matrix_svd.py, bilinear_vs_gated.py, predictive_rewiring.py,
weight_only_predictor.py
**What we learned:** D_k (Gram-operator invariants) are insufficient for
composition — valid negative (entry 103). But there is no deeper weight-only
compositional invariant hiding in the gate-up coupling. Permutation damage
is predicted by the DU product. The gate's role is input-specific modulation,
not weight-level coupling. The "tensor composition" framing was overclaimed;
the correct framing is hidden-unit gauge symmetry.

---

## Mixed Moment Critical Exponent — 2026-09-18

**Central question:** Can pairwise spectral statistics (eigenspectrum,
coherence) predict frame operator composition, or do mixed moments carry
essential information about eigenvector alignment?

### Synthetic proof (entry 97)

5 ensembles x {d=64-1024} x {gamma=2,4} x 30 trials. Power-law frame
operators with alpha=0.8 (matching real models). Shared top-25% eigenvectors
vs random. 6/6 pre-registered predictions confirmed.

**Key result:** D1 separates shared from random at z > 49 across all d.
The separation GROWS as d^{0.575} (predicted d^{0.6}). Critical exponent
alpha = 0.5: below it pairwise geometry suffices; above, mixed moments are
essential. Real models (alpha=0.6-0.9) are in the "fails" regime.

### Real model verification (entry 98)

SmolLM2-135M (d=576) and Qwen3-0.6B (d=1024). Normalized D1_norm between
gate_proj and up_proj frame operators at every layer.

**Key result:** D1_norm positive at ALL layers (0.05-1.32). Random baseline
= 0. Sharing increases through depth, peaking at last layers (Qwen3 L27:
1.32, 10x early baseline). Invisible to per-component spectral analysis.

### D-scaling across SmolLM2 ladder (entry 100)

SmolLM2-135M (d=576), 360M (d=960), 1.7B (d=2048). Same architecture
family, 3.5x range of dimensions.

**Key result:** D1_norm CONSTANT at ~0.21 across all dimensions. Training
co-adapts alpha (spectral exponent decreases with d) to maintain constant
cross-component coupling. Top-4 eigenvector overlap increases (0.35 -> 0.42)
but per-vector contribution decreases (flatter spectra). Larger models
distribute coupling more evenly across depth.

**Formal theorem:** theory/mixed_moment_critical_exponent.md
**Data:** experiments/results/frame_composition/
**NOTEBOOK entries:** 97-100
**What we learned:** Pairwise geometry is provably insufficient for trained
networks. Mixed moments are the native cross-component information. A phase
transition at alpha=0.5 separates two regimes. Real models maintain constant
cross-component coupling (~0.21) by co-adapting spectral shape with dimension.

---

## Frame-Theoretic Analysis of Trained Transformers — 2026-09-18

**Central question:** Do trained transformer weight matrices approximate the
tight frame / equiangular tight frame structure assumed by formal superposition
theory? What mathematical structure do they have instead?

### Frame analysis (3 models)

Analyzed embedding and MLP weight matrices of SmolLM2-135M (d=576),
Qwen3-0.6B (d=1024), and Phi-1.5 (d=2048) as overcomplete frames.

**Key finding:** Real models are 100x-10,000x from tight frame ideal.
Condition numbers: 640-12,000 for embeddings, 180-25,888 for MLPs.
Coherence 15-30x the Welch bound. The formal theory's assumptions are
massively violated. Yet models achieve superposition.

### Eigenspectrum shape

Spectra follow power laws (exponent -0.35 to -0.90), NOT Marchenko-Pastur
(random matrix) or uniform (tight frame). Tail exponents steepen through
depth (-1.3 to -1.8). SmolLM2 embedding is effectively 4-dimensional
(participation ratio 3.9/576). Gate_proj is steeper than up_proj universally.

### Eigenprojection test (H1b: tail vulnerability)

Perturbation vectors projected into gate_proj eigenbasis at 5 layers per model.
High-sensitivity perturbations project LESS onto top eigenspaces at the input
layer (SmolLM2 L0: rho=-0.43; Qwen3 L0: rho=-0.83, p=0.04). Interpretation:
top eigenspaces are robust (high energy); tail eigenspaces are fragile. The
power-law spectrum IS the vulnerability gradient. Late-layer patterns diverge
across architectures (compensation strategies differ).

### Rate-distortion prediction and first-principles derivation (entries 95, 99, 101)

Monte Carlo test: random embedding directions with information-weighted norms
produce FLAT spectra indistinguishable from uniform random. Power law requires
directional structure from training. Phi-1.5 calibration falsifies monotone
alpha(d); controlling variable is n/d (overcompleteness ratio), not d alone.

**First-principles confirmation:** Single-layer embedding trained with SGD on
Zipfian tokens + cross-entropy produces power-law eigenspectra. Uniform tokens
→ flat spectrum. Power law exponent controlled by (Zipf s, n/d ratio).
n/d=2 → exponent -2.39; n/d=128 → exponent -0.97. Zipf s=0 → flat (-0.25);
s=1 → steep (-1.75).

Per-token eigenspace rank: SmolLM2 median 32/576 (5.6%), Qwen3 326/1024
(31.8%). Tokens are hierarchically clustered in eigenspace.

### Effective Zipf measurement and quantitative prediction (entry 107)

All three tokenizers show effective Zipf s ≈ 0.5 (not s = 1.0). BPE subword
tokenization flattens the frequency distribution. 5×5 grid of toy models
(s ∈ {0.3-0.7}, n/d ∈ {8-128}) quantifies the joint dependence. Prediction
for Phi-1.5 (s=0.53, n/d=25): within 21%. High-n/d predictions fail because
toy training (5000 steps) is insufficient for convergent organization.

Three-variable model: exponent = f(Zipf s, n/d, training convergence).

### Spectral-composition bridge (entry 110) — NEGATIVE

Tested whether training on Zipfian data jointly organizes spectral structure
AND compositional gate-up pairing. Minimal model (n=1024, d=64, h=128, untied
unembed), 5 seeds, 4 conditions (gated/frozen-embed/additive/uniform).
Fable-reviewed pair-sensitive signed-product statistic with Bonferroni correction.

**Result:** Bridge hypothesis REJECTED (Bonferroni p=1.0). Spectral structure
does not predict composition. SURPRISE: additive MLPs develop z=-7.79 spectral
anti-pairing; gating disrupts this, explaining why gated MLPs are pairing-sensitive.

### Spectral sensitivity test (entry 115) — NEGATIVE

Zero-ablation of up_proj at each layer of SmolLM2-135M. Measure output divergence.

**Result:** Spectral exponent does NOT predict sensitivity (rho=+0.28, p=0.14).
Sensitivity is U-shaped (L0 and L29 critical, middle redundant). Position
dominates over spectral structure. The depth gradient is descriptive, not
predictive of functional importance.

### Scale test and LN confound (entries 113-114)

SmolLM2-360M confirms up_proj steepening (rho=-0.862). Post-LN spectra still
flatten with depth (rho=+0.45, p=0.01), resolving the layer norm confound.
Steepening rate approximately scale-invariant (-0.162 vs -0.142 normalized).

### Hidden state spectra — complementary specialization (entry 113)

Forward pass of SmolLM2-135M on 677 tokens of real text. Hidden state
covariance eigenspectrum at each layer.

**Result:** Hidden state spectra FLATTEN with depth (rho=+0.871, p<0.001).
Weight spectra steepen (rho=-0.891). Anti-correlated. Rules out residual
accumulation as the cause of weight steepening. Interpretation: complementary
specialization — richer representations require more selective weights.

### Spectral depth gradient (entry 112)

Comprehensive depth profile across MLP and attention components. Both models.
MLP gate/up Gram exponents, attention QK/OV effective ranks, gate-up pairing z-scores.

**Result:** MLP up_proj exponent steepens monotonically with depth: SmolLM2
rho=-0.891, Qwen3 rho=-0.649 (both p<0.001). Gate follows. Attention QK/OV
effective ranks show NO depth trend. Creates a generalization-to-specialization
gradient: early layers use many spectral directions (flat spectra), late layers
concentrate on few (steep spectra). The pairing dipole (entry 111) sits at
the extreme end. QK effective rank = 80% of d_head; OV = 93%.

### Real-model pairing verification (entry 111)

Computed signed-product pairing statistic across all 30 layers of SmolLM2-135M.
200 shuffles per layer, embedding eigenbasis. Pure weight analysis.

**Result:** Aggregate top-5 z = -0.11 (p=0.375, not significant). 28/30 layers
show z ≈ 0 — confirms gating destroys anti-pairing in real models. ANOMALY:
layer 28 (penultimate) shows POSITIVE pairing: top-5 z = +2.48, 137/576
positive-significant eigenvectors, max z = 11.5, Bonferroni p ≈ 1e-6. Suggests
near-output layers develop spectral co-alignment from concentrated loss gradients.

**Data:** experiments/results/frame_analysis/real_model_pairing.json
**NOTEBOOK entries:** 88-101, 107, 110-115
**What we learned:** The theory-practice gap is RESOLVED for embeddings.
Power law emerges from Zipfian data + gradient descent. The mechanism is
directional structure (organized token-eigenspace alignment). The degree of
organization depends on overcompleteness ratio n/d, not d alone. Effective
Zipf s ≈ 0.5 for BPE tokenizers. Spectral structure and compositional
pairing are INDEPENDENT — the additive anti-pairing discovery (entry 110)
shows that gating decouples pairing from spectral constraints.

---

## HD Geometry Probes — 2026-09-18

High-dimensional geometry of transformer hidden states across layers.
Model: SmolLM2-1.7B (24 layers, d=2048). All probes: last-token extraction, fp32 CPU.

**Central question:** Do transformer representations exhibit native high-dimensional mathematical structure beyond standard R^n geometry?

### Probe 1: Arithmetic expressions (ARTIFACT)

- 100 arithmetic expressions (`x = a op b`), all sharing suffix `\nprint(x)\n# Output: `
- **ARTIFACT:** Shared suffix made all last tokens identical → cos=1.0 at L0
- Apparent "spread-then-concentrate" hunchback was context propagation artifact
- **Status:** Invalidated. Confirmed by Probes 2-3 which showed no hunchback.

### Probe 2: Diverse inputs (clean)

- 150 inputs: 50 arithmetic, 50 natural language, 50 code (bare text, no shared suffix)
- **NO hunchback** for any class or combined. All trajectories monotonically concentrate.
- Cross-class convergence zone at L12-L20 (~0.52-0.65), then L24 separates domains
- Arith-code stay close (0.70); NL splits from both (0.35-0.37)
- Per-class final cosine: arith 0.956, NL 0.806, code 0.824

### Probe 3: Controlled similarity threshold (clean)

- 120 inputs: 4 sets × 30 (identical_prefix, paraphrases, same_domain, cross_domain)
- **NO hunchback** for ANY input set. All monotonically concentrate.
- Final cosine preserves input similarity ordering (identical_prefix 0.871 > cross_domain 0.764)
- Differentiation threshold hypothesis not supported

### What survived Fable + Droid dual review

1. Cross-domain mid-layer convergence → final-layer separation (real but not novel)
2. Context-propagation mechanism (identical last token → hunchback) is concrete finding
3. Per-class final cosine reflects next-token distribution similarity
4. **All patterns are known phenomenology using standard R^n tools** — not breakthrough level

### What was learned

- ID hunchback, anisotropy, trajectory geometry are well-studied (5+ papers cited by reviewers)
- Input cosine ≠ feature superposition (category error corrected)
- eff_dim bounded by sample size n (rank ceiling), not d
- cos_mean = (n||ȳ||² - 1)/(n-1) — the "dissociation" is elementary

## MP-Deviation & Interference Composition — 2026-09-18

Zero-cost reanalysis of 7-model balanced kill data (69 cells × 24 perturbations).

### MP-Deviation Analysis (KILLED)

Tested whether eigenvalue spikes above Tracy-Widom threshold in behavioral Gram
matrices predict transfer hierarchy. Recommended by Fable strategic review R2.

- All 7 models: 1-3 spikes above TW (MP deviations exist)
- Spike overlap does NOT predict held-out transfer (rho near 0, all splits)
- Spike count ~ gap correlation: split-dependent artifact (p=0.034 interleaved → p=0.562 full)
- MP framework adds no predictive power over standard PCA
- **Kill criterion #2 triggered:** deviations exist but don't predict

### Interference Composition / Transitivity (KILLED)

Tested whether perturbation correlations follow product rule r(i,k) ~ r(i,j)*r(j,k).
Deviations would indicate a composition law.

- Raw: positive residual +0.13 to +0.20 (all 7 models) — apparent superadditivity
- 3-factor PCA captures 79-97% variance
- After factoring: residual flips to -0.02 to -0.12 — superadditivity ENTIRELY from shared factors
- Factors are architecture-family-specific (Qwen3-StableLM cluster; SmolLM2 cluster)
- **Kill criterion triggered:** composition explained by ~3 trivial factors

### Behavioral Rank Bootstrap (POSITIVE)

Bootstrap test: is behavioral rank model-intrinsic or perturbation-set-dependent?
Zero-cost reanalysis of balanced kill data.

- Behavioral rank IS model-intrinsic (matches synthetic rank-k, NOT random)
- Rank varies 1 (SmolLM2-360M) to 4+ (SmolLM2-1.7B-Instruct)
- PR predicts k80 perfectly: rho=0.954, p=0.001
- Instruction tuning doubles PR (1.32→2.62)

### Weight Spectral Bridge (KILLED)

Tests whether weight matrix spectral properties predict behavioral PR. CPU only.

- No weight spectral quantity predicts behavioral PR (all p>0.25)
- **Decisive:** SmolLM2 base/instruct have identical weight spectra (PR delta 0.1%)
  but 98% different behavioral PR
- Weight spectra encode capacity, not behavioral routing

### Jacobian Derivation (KILLED — Fable R4)

Theory-first: derive sensitivity hierarchy from attention Jacobian. Value/routing
channel decomposition, causal masking argument.

- Value/routing dichotomy stated backwards (both zero at t<p, both nonzero at T)
- Perturbations not in Jacobian domain (variable length changes)
- d_add_zero falsification: appends "+0" (zero original positions touched), should
  behave A-like under theory. FAILS in 6/7 models (rho vs A = 0.06-0.86 vs AA 0.64-0.94)
- Not novel: QK/OV circuit split (Elhage 2021) + adversarial Jacobian (Goodfellow 2015)
- R3 kill criteria: 3/4 met. See scratchpad/fable_jacobian_r4.md

### Attention Routing Diversity (KILLED)

Tests whether base vs instruct routing diversity explains behavioral rank difference.
CPU, ~50 forward passes per model on SmolLM2-1.7B base/instruct.

- Routing nearly identical (delta <2% in heads changed, <1% in Jaccard)
- 50-97% of heads change for every perturbation (not sparse)
- Routing change is positional (token insertion), not content-based
- Behavioral rank 2× difference does NOT come from routing diversity

### Cumulative assessment

Behavioral data (69×24) is intrinsically rank-3. All structure reduces to 3 architecture-
family-specific vulnerability factors. RMT, interference composition, transitivity,
weight spectra, Jacobian theory, and routing diversity add nothing beyond PCA.
Dataset is exhausted for high-d analysis.

**HD Math direction KILLED** (Fable R3 criterion #1 triggered):
3 theory-first cycles (concentration/routing, weight spectral bridge, Jacobian),
0 surviving testable predictions. One salvage finding: d_add_zero dissociates
positional from computational accounts of the hierarchy.

## TRHO Transfer Rho Spectrum — 2026-09-17

Behavioral transfer rho across language model architectures, training strategies, and perturbation types.
2 model families, 9 configurations, 40-50 cells per experiment.

**Central question:** Is low behavioral transfer (rho) a universal property of language models,
and does it vary with perturbation type?

### Key Findings

1. **Transfer hierarchy established:**
   - Within-surface rho: 0.63-0.72 (trivial perturbations are self-consistent)
   - Cross-type rho: 0.08-0.36 (surface cannot predict semantic; Qwen p=0.13 NOT significant)
   - Within-semantic rho: 0.14-0.29 (structural changes unpredictable)

2. **Effective dimensionality (model-independent):**
   - Surface perturbations: ~1.9 effective dimensions
   - Semantic perturbations: ~4.1 effective dimensions
   - Ratio: 2.2x in BOTH Qwen3 and SmolLM2
   - Semantic PC1 captured by surface (74-93%), but PC3/PC4 nearly orthogonal (2-4%)

3. **1/sqrt(d_eff) bound:**
   - Matches surface rho within 2% (predicted 0.731, observed 0.718)
   - Overshoots semantic by 3.7x (predicted 0.497, observed 0.135)
   - Excess non-transfer: semantic perturbations have adversarial structure beyond dimensionality

4. **Perturbation correlation confirms hierarchy:**
   - Surface correlations: 0.48-0.65 (perturbations probe same subspace)
   - Semantic correlations: 0.18-0.25 (each perturbation probes different pathway)
   - Commutative reorderings (a+b vs b+a) produce non-zero TV: refutes complexity confound

5. **Bootstrap CIs (500 iterations):**
   - Qwen cross-type: [-0.031, 0.341] — includes zero, cannot reject null
   - SmolLM2 cross-type: [0.185, 0.709] — excludes zero
   - All within-surface CIs exclude zero

6. **Readout dependence:**
   - 11-bin (task-specific): rho 0.13-0.30
   - Full-logit cosine (full behavioral state): rho 0.25-0.68
   - Phi-1.5 full-logit ERASES the semantic gap entirely (surface 0.70, semantic 0.66)
   - Task-specific information transfers worst

7. **Biology >> LLM:** Protein/yeast rho 0.88-0.98 vs LLM semantic 0.14-0.29.

8. **Noise calibration clean:** Deterministic TV=0.0, permutation p<0.001, 4.9σ above null.

9. **Perturbation transfer matrix (full-logit):**
   - Qwen: semantic block mean rho=0.361, worst pair varname_y<->assign_twice at 0.044
   - SmolLM2: semantic block mean rho=0.613 (intermediate)
   - Phi-1.5: semantic block mean rho=0.754 (explains gap erasure: high within-semantic correlation)
   - No negative correlations in full-logit; 11-bin negatives were readout artifacts
   - func_call universally lowest-correlated semantic perturbation across all models

10. **NL math transfer rho (Qwen3-1.7B-Base):**
    - ALL rho values collapse vs code. 11-bin surface: 0.518 (vs 0.685 code), semantic: 0.094 (vs 0.309)
    - NL "surface" perturbations are high-dimensional (3.81 vs 1.62): every word is semantic
    - Dimensionality ratio shrinks (1.44x NL vs 1.99x code) because NL surface is also high-dim
    - Implication: code alignment is partially certifiable (trivial perturbation class exists);
      NL alignment is uncertifiable (no trivial perturbation class in NL)

11. **Common-factor null (fragility control — CORRECTED):**
    - Per-cell fragility accounts for most of raw rho (within-cell null means 0.48-0.87)
    - ORIGINAL "semantic collapses" claim RETRACTED — was covariate contamination artifact
    - Clean partial (other-type covariate): Qwen FL surface 0.76, semantic 0.43 (gap persists)
    - Semantic retains condition-specific structure in ALL models (clean partial 0.42-0.77)
    - SmolLM2 FL surface is ENTIRELY fragility (within-cell null p=0.26)

12. **NL multi-model comparison (SmolLM2, Phi-1.5, full-logit):**
    - Phi-1.5 gap erasure is CODE-SPECIFIC: ratio 1.05x (code) -> 1.82x (NL)
    - SmolLM2 is DOMAIN-STABLE: ratio 1.49x (code) -> 1.50x (NL)
    - Qwen3 COLLAPSES in NL: surface 0.661->0.175, semantic 0.227->0.048
    - Model ordering reverses in NL: SmolLM2 > Phi-1.5 > Qwen3
    - CAUTION: Phi-1.5 P(digit)=0.001 in NL; 11-bin unreliable, full-logit valid

13. **Improved theoretical bound (mac/sqrt(d_eff)):**
    - mac = mean absolute pairwise correlation between perturbation effects
    - mac/sqrt(d_eff) has mean error 6.7% vs 15.6% for 1/sqrt(d_eff) alone
    - For Qwen3 semantic: 11% error (mac/sqrt) vs 146% error (1/sqrt)
    - All predictors rank-order models correctly (n=6 too small to discriminate)

14. **Paired-gap bootstrap (direct CI on surface−semantic):**
    - Qwen FL: gap +0.241 [+0.084, +0.430] — EXCLUDES ZERO (P(gap>0)=0.998)
    - Qwen 11-bin: gap +0.147 [-0.030, +0.330] — includes zero
    - SmolLM2 FL: gap +0.046 [-0.144, +0.212] — includes zero
    - SmolLM2 11-bin: gap +0.173 [+0.028, +0.373] — EXCLUDES ZERO
    - Phi FL: gap +0.002 [-0.095, +0.107] — no gap (50/50)
    - Headline: Qwen full-logit gap is statistically established

15. **Disattenuated cross-type correlation:**
    - Qwen FL: cross=0.367 vs predicted=0.387 (one factor fits, disatt_r=0.948)
    - All others: one factor rejected (disatt_r=0.35-0.76)
    - 4/5 model×readout combos reject the single-factor model
    - Qwen FL is the exception: surface and semantic fragility rankings coincide

16. **Held-out perturbation prediction:**
    - LOO within original 12: surface rho=0.76, semantic rho=0.51 (fragility predicts within-family)
    - Cross-type: surface fragility → semantic held-out: rho 0.30-0.53
    - Novel perturbations (dead_code, redundant_parens, int_str_roundtrip): ALL rho ≈ 0 (p>0.4)
    - Fragility is perturbation-family-specific, NOT a universal cell property

17. **Cross-domain fragility decomposition (NL vs code):**
    - Fragility loading is MODEL x DOMAIN specific — same model allocates it differently
    - Qwen3 CODE: fragility dominates both surface (R2=88%) and semantic (R2=83%)
    - Qwen3 NL: surface is fragility-INDEPENDENT (R2=0.6%!), semantic is 92% fragility
    - SmolLM2 CODE: surface is 94% fragility, semantic only 30% fragility
    - SmolLM2 NL: both are 80-90% fragility, both collapse to ~0 after removal
    - KEY: Qwen3 partial gap is CONSERVED: +0.255 (code) vs +0.256 (NL)
    - The gap is an architectural invariant across domains despite completely different fragility allocation
    - SmolLM2 code retains genuine semantic structure (partial rho=0.377) but SmolLM2 NL does not

18. **StableLM-2-1.6B (4th architecture family, PARTIAL):**
    - Code full-logit: surf=0.918, cross=0.763, sem=0.827 (ratio 0.97x, NO GAP)
    - Code 11-bin: surf=0.735, cross=0.419, sem=0.406 (ratio 1.36x)
    - NL full-logit: surf=0.640, cross=0.443, sem=0.206 (ratio 1.91x, HUGE gap)
    - NL 11-bin: surf=0.612, cross=0.091, sem=0.025 (ratio 1.55x)
    - Same pattern as Phi-1.5: code gap erasure + NL gap massive
    - Falcon-H1-1.5B (hybrid SSM-attention) still running

19. **Dimensionality scaling with perturbation vocabulary:**
    - Qwen: d_eff = 0.253*n + 0.28, R2=0.976 — LINEAR growth, no saturation
    - d_eff/n constant (0.26-0.29): each perturbation adds ~0.25 new dimensions
    - SmolLM2: d_eff/n decreases from 0.23 to 0.18 — SATURATING
    - Qwen rho DECREASES as vocabulary grows (0.87 -> 0.63); SmolLM2 INCREASES (0.87 -> 0.92)
    - Model-dependent sensitivity landscape compactness predicts cross-type transfer

20. **Cross-model fragility transfer (zero-cost):**
    - Code: Qwen3 <-> SmolLM2 fragility rho=0.243 (p=0.13, NOT SIGNIFICANT)
    - NL: Qwen3 <-> SmolLM2 fragility rho=-0.055 (p=0.74, ZERO)
    - Cross-domain: Qwen3 code<->NL rho=0.163 (NS); SmolLM2 rho=0.416 (p=0.008)
    - Fragility is (model x domain x perturbation-family)-specific, maximally non-transferable
    - Different models find different cells sensitive on the SAME 40 arithmetic problems

21. **Fable R13 review — claim triage:**
    - WITHDRAWN: "Qwen3 collapses in NL" (fp16 floor), "SmolLM2 domain-stable" (fragility-dependent),
      "Phi gap erasure code-specific" (precision mismatch), mac/sqrt(d_eff) (in-sample, fails holdout),
      "no negative correlations in full-logit" (false in NL), biology >> LLM (no backing data),
      d_eff as independent finding (restatement of correlation)
    - SURVIVES: "Per-input behavioral sensitivity is a perturbation x input interaction,
      not an input property." Surface class transfers because its 6 members are ONE perturbation
      in 6 spellings (d_eff ~ 1.9). Everything else doesn't transfer.
    - POSITION CONFOUND identified: surface appends AFTER assignment; semantic rewrites the assignment.
      Class is confounded with "touches the line the answer depends on."
    - KILL EXPERIMENT designed: 4 groups x 6 perts, fp32, CPU. Resolves fp16, position, held-out.

22. **R13 kill experiment (COMPLETED — DECISIVE):**
    - Qwen3-1.7B, fp32 CPU, 40 cells x 24 perturbations x 2 surfaces, 352s
    - (a) 6 original inert appends after assignment
    - (b) 6 NEW inert appends after assignment (held-out test)
    - (c) 6 inert insertions BEFORE assignment (position control)
    - (d) 6 meaning-preserving REWRITES of the assignment (matched position)
    - **Results (full-logit Spearman rho):**
      - a<->a = 0.785: within original inerts (baseline)
      - a<->b = 0.811: held-out inerts TRANSFER (R13's "near-duplicates" prediction REFUTED)
      - a<->c = 0.611: position control WORKS — before/after gap is ~0.2, not the whole story
      - a<->d = 0.333: GAP IS REAL — rewrites don't transfer to inerts even at same position
      - d<->d = 0.224: rewrites barely transfer to EACH OTHER
    - **Results (11-bin):** same gradient: a<->b=0.746, a<->c=0.565, a<->d=0.328, d<->d=0.182
    - **Gradient:** same-type inerts (0.81) > cross-position inerts (0.61) > cross-type (0.33) > within-rewrite (0.22)
    - **Three confounds resolved:** (1) held-out test PASSES — surface class is real, not 6 near-duplicates;
      (2) position contributes ~0.2 but doesn't explain the full gap; (3) fp32 consistent with 11-bin (no floor issue)
    - Key: the factor is whether the perturbation TOUCHES THE COMPUTATION PATH, not position
    - R14 REVIEW (PARTIAL CONFIRM): gradient real but "computation path" unfalsifiable; position ~35%
      of gap; a<->b>a<->a is group-homogeneity noise; fp32 confirms 11-bin not fixes full-logit;
      needs SmolLM2 replication + syntactic novelty + inert-at-assignment controls.

23. **Kill experiment — SmolLM2-1.7B replication (COMPLETED — GRADIENT REPLICATES):**
    - Same 4 groups x 6 perts, fp32 CPU, 40 cells, 357s, 95% accuracy
    - **Results (full-logit):** a<->a=0.864, a<->b=0.861, a<->c=0.672, a<->d=0.583, d<->d=0.443
    - Same gradient: same-type > cross-position > cross-type > within-rewrite
    - Gap COMPRESSED vs Qwen3 (0.28 vs 0.45) but structure preserved
    - ZERO distances below 1e-3 — no fp16/fp32 floor concern
    - SmolLM2 more uniform in perturbation sensitivity (consistent with "domain-stable")

24. **R14 control experiments (COMPLETED — SURFACES DISAGREE):**
    - Groups: A (reference inert, 6), E (unusual inert appends, 6), F (inert assignment-line mods, 6), D (rewrites, 6)
    - Qwen3-1.7B-Base, fp32 CPU, 40 cells × 24 perts × 2 surfaces, 366s, 100% accuracy
    - **Full-logit ordering:** A(0.785) > E(0.645) > F(0.471) > D(0.224)
    - **11-bin ordering:** A(0.686) > F(0.583) > E(0.442) > D(0.182)
    - **E and F SWAP between surfaces** — two readouts disagree about which control group is "like A"
    - Full-logit says: unusual appends transfer well (E≈A), assignment-line mods don't (F≈D)
      → assignment-line TOKEN CONTENT matters for representation-level sensitivity
    - 11-bin says: assignment-line mods transfer well (F≈A), unusual appends don't (E≈D)
      → DISTRIBUTIONAL NOVELTY matters for decision-level sensitivity
    - Both surfaces preserve the A > D gradient — the gap between inerts and rewrites is robust
    - Cross-group: full-logit a<->e=0.609, a<->f=0.565; 11-bin a<->e=0.430, a<->f=0.572 (same swap)
    - Distance magnitudes: E median slightly above A, F slightly above E, D highest (consistent gradient)
    - **CALIBRATION (per-cell bootstrap):** E-F swap NOT significant on either surface
      - fulllogit: E<->E minus F<->F = +0.171 CI [-0.013, 0.355] (includes 0)
      - 11-bin: E<->E minus F<->F = -0.138 CI [-0.341, 0.048] (includes 0)
      - Mann-Whitney A<->E vs A<->F: fulllogit p=0.79, 11-bin p=0.067
    - Within-group heterogeneity dominates: f_semicolon (0.872 fl, 0.810 11b) > any E member;
      e_list_pop (-0.021 11b) < any F member. Groups E/F not internally coherent.
    - **Honest conclusion:** gradient from inerts to rewrites is robust across readout surfaces,
      but the intermediate region cannot be cleanly attributed to either "computation path" or
      "syntactic novelty". Individual perturbation identity dominates group membership.
    - **Property analysis (18 non-A perturbations):**
      - Position (after=0, at=1, replace=2) predicts FL rho: r=-0.538, p=0.021 uncorrected
        R15: does NOT survive Bonferroni for 10 tests (5 predictors × 2 surfaces). SUGGESTIVE only.
      - Mean distance predicts 11-bin rho: r=-0.703, p=0.001 (SURVIVES Bonferroni correction)
        R15: but tautological — perturbations with larger distances have noisier rankings (rank-SNR)
      - tokens_added, unusual, modifies_assign: all p > 0.1
      - R15 CORRECTION: "two independent axes" downgraded to one confirmed but tautological predictor
        (mean distance) plus one suggestive uncorrected predictor (position ordinal)

25. **Kill experiment — Phi-1.5 (COMPLETED — NO GRADIENT):**
    - microsoft/phi-1_5, fp32 CPU, 40 cells × 24 perts × 2 surfaces, 306s, 92.5% accuracy
    - **Full-logit:** a<->a=0.807, a<->b=0.739, a<->c=0.709, a<->d=0.752, d<->d=0.763
    - **NO GRADIENT:** a<->d (0.752) ≈ a<->a (0.807), d<->d (0.763) ≈ a<->a. MWU p=0.52 (NS)
    - Rewrites transfer to inerts AND to each other — Phi-1.5 is perturbation-robust
    - 11-bin: compressed range 0.514-0.601, 84% of distances < 1e-3 (resolution floor)
    - Confirms gap erasure from earlier spectrum study under controlled kill experiment design
    - **MODEL-DEPENDENT gradient:** Qwen3 (strong), SmolLM2 (compressed), Phi-1.5 (absent)
    - R15 CONDITIONAL: result depends on P(digit) verification. If P(digit)≈0.001, 11-bin is noise
      and full-logit measures non-arithmetic behavior. 84% of 11-bin distances < 1e-3 is consistent
      with near-zero digit probability. MUST verify before claiming certifiability.
    - R15: "code-specialized training" attribution confounded with arch/tokenizer/size (n=1)

26. **Perturbation dimensionality analysis (zero model runs):**
    - PCA on perturbation correlation matrices across all kill experiments
    - d_eff (participation ratio): Qwen3 kill 2.72/2.87, controls 3.55/3.59, SmolLM2 1.97/2.78
    - PC1 = fragility (common factor), captures 49-70% of variance
    - Groups barely separable: between/within distance ratio 1.07-1.50
    - 5-8 PCs for 90% variance — perturbation space is ~5-8 dimensional in practice
    - SmolLM2 most one-dimensional (d_eff=1.97 FL) = most fragility-dominated
    - R15: d_eff is algebraic identity (participation ratio = n/(1+(n-1)·mean_r²)); confirms #19
      dimensionality results in kill experiment context, NOT an independent finding (R13 withdrew this)

27. **Kill experiment — StableLM-2-1.6B (4th architecture, COMPLETED — GRADIENT PRESENT):**
    - StableLM-2-1.6B (general, 1.6B), fp32 CPU, 40 expression cells x 24 perts x 2 surfaces
    - Full-logit: a<->a=0.927, a<->b=0.937, a<->c=0.587, a<->d=0.628, d<->d=0.566
    - Gap = 0.299 (MWU p=4.5e-13) — moderate gradient, between SmolLM2 and Qwen3
    - Highest a<->a (0.927) and a<->b (0.937) of all 4 models — strongest inert transfer
    - 82.5% accuracy (lowest of the 4; 7 cells wrong)
    - Confirms: general-purpose models show gradient; magnitude varies by architecture

28. **Rewrite Interchangeability Index (RII, zero model runs; R15 REVIEWED):**
    - RII = mean within-group-D Spearman correlation across cells
    - Measures whether a model treats meaning-preserving rewrites as INTERCHANGEABLE
    - Phi-1.5: RII=0.763, StableLM-2: RII=0.566, SmolLM2: RII=0.443, Qwen3: RII=0.224
    - RII vs gradient: Pearson r=-0.952 (p=0.048) among 4 comparable models (1.3-1.7B, >80% acc)
    - Cross-model perturbation rank: Qwen3 is outlier (r~0 with SmolLM2); others cluster at r~0.5
    - R15 CORRECTION: "code-specialized → high RII" was OVERREACH (n=1, confounded)
    - RII is a descriptive index correlating with gradient; causal driver unknown

29. **Kill experiment — CodeGen-350M-mono (5th architecture, COMPLETED — GRADIENT PRESENT):**
    - CodeGen-350M-mono (code-specialized, 350M), fp32 CPU, 40 cells x 24 perts x 2 surfaces
    - Full-logit: a<->a=0.762, a<->b=0.731, a<->c=0.354, a<->d=0.444, d<->d=0.368
    - Gap = 0.318 (MWU p=2.3e-7) — REFUTES "code-specialized → no gradient"
    - RII = 0.368 — REFUTES "code-specialized → high RII"
    - Accuracy: 32.5% (barely above chance) — MAJOR CONFOUND with task incompetence
    - 11-bin DEAD: 100% distances < 1e-3 (model puts near-zero probability on digits)
    - Not directly comparable to 1.3-1.7B models due to scale × accuracy confound
    - Interpretations: (a) code-specialization needs scale to produce certifiability,
      (b) accuracy/task-competence is the real predictor, or (c) Phi's no-gradient is
      architecture-specific (parallel attention), not training-domain-specific

30. **RII decomposition and P(digit) verification (zero model runs, 2026-09-18):**
    - P(digit) verified from 11-bin distance magnitudes (no model loading needed):
      Qwen3 (median 0.015, 0% < 1e-3), SmolLM2 (0.094, 0%), StableLM (0.089, 0%),
      Phi-1.5 (0.0003, 83%), CodeGen (0.00001, 100%). Clean binary split: 3 digit-predicting,
      2 code-continuation. 11-bin INVALIDATED for Phi/CodeGen; full-logit valid for all
    - **RETRACTED (R16 R2): z-RII and FDI are null artifacts.** Row-centering k=6 D columns
      gives expected pairwise Spearman = -1/(k-1) = -0.20; pure-noise matrices give -0.192±0.002;
      the five models give -0.174 to -0.191. "Universal z-RII ≈ -0.18" IS the null, not a finding.
      FDI = RII - z_RII = RII + 0.18 (constant shift), so FDI vs gap r=-0.952 is the existing
      RII vs gap r=-0.944 relabeled. All z-RII/FDI claims withdrawn
    - **REPLACEMENT (R16 E-corrected, NOW WITH CIs):** Fragility-partialled group rhos (rank
      regression on per-cell mean distance): residual RII collapses near zero for all models
      (CIs all include 0). RII variation IS largely fragility share. But the GAP survives
      partialling with bootstrap 95% CIs (1000 resamples, all exclude 0 except Phi):
      Qwen3 0.294 [0.099, 0.454], CodeGen 0.251 [0.106, 0.412], StableLM 0.633 [0.303, 0.803],
      SmolLM2 0.274 [0.060, 0.516]. Phi 0.044 [-0.057, 0.150] includes 0 (NS).
      The gradient is NOT a fragility artifact; it is real perturbation-type-specific structure
    - **PERMUTATION NULL (independent test):** Shuffle perturbation labels within each cell (500
      perms). Qwen3 Z=8.8, SmolLM2 Z=6.0, StableLM Z=6.2, CodeGen Z=5.5 (all p<0.001).
      Phi Z=1.0 (p=0.168, NS). Perturbation identity matters beyond fragility for 4/5 models.
      Permutation 95th percentile ≈ 0.08-0.10 for all models
    - **CELL-EXCLUSION (Phi):** Leave-1-out gap [0.048, 0.064]; leave-3-out (500 samples) max
      0.076 (none > 0.10). Phi's no-gradient is robust to cell exclusion

### Fable R15 adversarial review (2026-09-18):
- 7 weaknesses identified; key corrections applied to STATE.md
- CONFIRMED: multi-model gradient replication (now 4/5 models), Phi no-gradient
- OVERREACH: "code-specialized → certifiable" (n=1 confounded; CodeGen REFUTES simplest version)
- OVERREACH: "two independent axes" (position ordinal p=0.021 fails Bonferroni at 10 tests)
- RESTATEMENT: d_eff finding confirms #19, not independent
- Review: experiments/results/fable_strategy_r15.md

31. **RETRACTED: Copy-vs-compute control (R17 — cell-schema mismatch, 2026-09-18):**
    - `copy_vs_compute.py` used the 20-cell `make_cells()` schema from `run_transfer_spectrum.py`,
      but kill experiment JSONs use 40-cell arithmetic-expression cells (zero literal-copy cells).
      The "simple/complex" split was an arbitrary answer-digit partition, not computation type.
      Cell descriptions in STATE.md were factually false. Additionally, the 12-cell gap sits within
      the random-12 null distribution (48th-88th percentile), and the MWU tested amplitude difference
      not the rho gap. All claims WITHDRAWN. Process violation: entered STATE.md without review

32. **Per-perturbation distance profiles — DOWNGRADED to descriptive (R17, 2026-09-18):**
    - "Qwen3 outlier" contrast NS at n=24 (Fisher z=1.66, p=0.10). Cluster is group-level effect
    - Two most "model-dependent" perturbations (c_newline_before, a_whitespace) are tokenizer-confounded:
      Qwen3's tokenizer merges them differently from other models
    - d_temp_var is length-confounded (only D pert that adds a line)
    - After group-demeaning, Qwen3-StableLM rises to r=0.62 (p=0.001); "cluster" collapses to NS
    - Gradient survives exclusion of tokenizer-sensitive perturbations (all five models unchanged)
    - Amplitude ordering model-dependent (Qwen3 D>A p=2e-9; others D<A all p<0.01) but gap is
      rank-based (amplitude-invariant by construction); not an anomaly, a clarification

33. **Cell exclusion robustness, symmetric (R17 correction, 2026-09-18):**
    - Phi exhaustive L3O: max gap 0.077 (cells 6, 26, 30) across all 9880 combinations
    - Gradient models L3O (2000 random triples, R17): min gap Qwen3 0.387, SmolLM2 0.232,
      StableLM 0.238, CodeGen 0.255. Gradient never collapses under any cell exclusion
    - L3O 2.5th percentiles: Qwen3 0.41, SmolLM2 0.25, StableLM 0.26, CodeGen 0.28

34. **Phi readout saturation — CONFIRMED as readout artifact (R16 Step 1 + R18 restructure):**
    - Removing ONE trailing space from query (`# Output: ` → `# Output:`) flips Phi from code
      continuation (P(digit)=0.001) to digit prediction. Tokenizer: Q0→space token→newline, Q1→colon→digit
    - **PRIMARY (40-cell kill-matched, same cells as other 4 models):** gap=0.217 [0.123, 0.355],
      p=5.26e-11. P(digit)=0.980, accuracy=95%. Pre-registered gates PASS. Q0 control: gap=0.055.
    - **SUPPLEMENTARY (20-cell, different cells/surfaces — NOT comparable to kill):** gap=0.335
      [0.141, 0.494], p=8.76e-9. P(digit)=0.843, accuracy=85%. Larger gap partly confounded by
      combined cell/surface/query changes (R18). RII under Q1: 0.235 [0.051, 0.412]
    - Raw gap: Qwen3 (0.452) > CodeGen (0.318) > StableLM (0.299) > SmolLM2 (0.281) > Phi-Q1 (0.217)
    - **STATUS: 5/5 CONFIRMED (R18 definitive test).** Phi Q1 passes ALL gates:
      Z=4.6 (p<0.001), residual gap 0.386 [0.171, 0.564] excl 0, PC1 share 84.6%.
      Residual gap is SECOND HIGHEST (above Qwen3/CodeGen). All 5 models confirmed.
    - Qwen3 Q1 control: Q1 INADMISSIBLE on Qwen3 (P(digit)=0.019). Readout effect model-specific

35. **Bootstrap CIs + permutation null on fragility-partialled quantities (R16 Step 0):**
    - 1000 bootstrap resamples, rank regression partialling on Q0 kill data
    - **Raw gap CIs:** Qwen3 [+0.315, +0.593], CodeGen [+0.202, +0.436], StableLM [+0.191, +0.438],
      SmolLM2 [+0.197, +0.385] — all excl 0. Phi Q0 [+0.005, +0.106] barely excl 0.
    - **Residual gap CIs:** Qwen3 +0.230 [+0.077, +0.430], CodeGen +0.277 [+0.100, +0.421],
      StableLM +0.560 [+0.256, +0.750], SmolLM2 +0.310 [+0.091, +0.533] — all excl 0.
      Phi Q0 +0.005 [-0.076, +0.110] includes 0 (NS).
    - **Residual RII CIs:** Qwen3 +0.062 [-0.037, +0.142], SmolLM2 +0.051 [-0.046, +0.164],
      StableLM +0.061 [-0.027, +0.218], CodeGen +0.078 [-0.025, +0.197] — all include 0.
      Phi +0.125 [+0.029, +0.263] — ONLY model with resid RII excl 0.
    - **Permutation null for resid RII (500 shuffles):** Qwen3 bias-corrected +0.099 (significant),
      SmolLM2 +0.071 (NS), StableLM +0.078 (NS), Phi +0.142 (significant), CodeGen +0.118 (significant)
    - Confirms: gradient is real perturbation-type structure, not common-factor artifact

36. **TRANSFER RHO — RETRACTED "2/5 transfer" claim (R19 review, 2026-09-18):**
    - Raw A→D correlation 0.56-0.92: real, driven by common fragility (all methods agree)
    - Same-locus inert exchangeability: A→B residual 0.83-0.97, all 5 models (SOLID)
    - **RETRACTED:** "2/5 models (Qwen3, StableLM) show type-specific transfer"
      R19 found: StableLM flips sign under rank partialling (-0.42); outlier cell (10%2) drives it.
      Qwen3 drops to 0.00 under rank+PC1+PC2; Simpson mixture across operator strata.
      The 2/5 vs 3/5 split was COVARIATE COMPOSITION (C-PC1 vs B-PC1 controls the outcome).
      a_empty in Group A means "inert sensitivity" IS baseline fragility (rho 0.87-0.99 with baseline).
    - **All-PC1 anti-transfer also retracted** (suppression artifact, caught earlier)
    - **NEW: Fragility has a second edit-locus dimension** (PC2, 8-17% variance).
      After removing PC1+PC2: zero of five models shows residual A→D transfer.
    - No model ordering is stable across methods (forking-paths hazard)
    - Power 0.43 at rho=0.3, n=40 — cannot claim null or positive
    - **Defensible:** fragility dominates; inert appends are exchangeable; no type-specific
      transfer currently established for any model at n=40.
    - Next: cell×perturbation interaction test (direct test), or ≥113-cell pre-registered run

37. **Tokenization-covariate regression (R17 Priority 2, zero cost, 2026-09-18):**
    - Does tokenizer explain per-perturbation distance profile?
    - Token-count delta between minimal/verbose surfaces is CONSTANT (4 tokens) across all 24
      perturbations for 4/5 models (the only surface difference is `# Python code\n` prefix)
    - Qwen3 has 1 exception: c_newline_before (3 tokens, tokenizer merge artifact)
    - R^2: Qwen3 0.347 (below 0.5 threshold), all others 0.000 (degenerate — constant input)
    - **VERDICT: Per-perturbation distance variation is NOT a tokenizer artifact.**
    - The variation comes from how models process different perturbations, not from tokenization

38. **Operator-class test (R17 Priority 6, pre-registered, zero cost, 2026-09-18):**
    - Pre-registered H1: D-sensitivity varies by operator class AND A-sensitivity does not
    - Operators tested: +, -, *, %, // (** excluded, only 1 cell)
    - **H1 FAILED for all 5 models.** Both A and D sensitivity vary by operator class:
      Qwen3 (A p=0.003, D p=0.012), SmolLM2 (A p=0.000, D p=0.001),
      StableLM (A p=0.002, D p=0.024), Phi (A p=0.000, D p=0.000),
      CodeGen (A p=0.022, D p=0.319 — exception: D uniform)
    - Operator class is a COMMON FRAGILITY axis, not D-specific
    - "Pattern structure" is empty as a term — the sensitivity structure is just fragility
    - Admissible under R17 rules (robustness check, not additive claim)

39. **RETRACTED: Row-permutation interaction test (2026-09-19, retracted same day by R20):**
    - Row-permutation of 40x24 distance matrix is invariant for interaction SS (row means travel
      with rows). All 20 p-values were floating-point artifacts (null_std ~ 1e-20). "5/5 additive",
      "type-blind", "monolithic fragility" all withdrawn. See fable_strategy_r20.md.
    - **Correctly specified test (R20, nested F + group-label perm, log scale):**
      4/5 models show cell x group interaction (group-perm p: Qwen3=0.0003, SmolLM2=0.0007,
      StableLM=0.0003, CodeGen=0.009, Phi=0.31 null). Cell x group share of residual: 0.26-0.40
      vs 0.13 expected under H0.
    - Within-group D coherence: rho 0.22-0.76. D is six unrelated probes, not a type.
    - Process lesson: null_std ~ 1e-20 and p=1.000 across all conditions is a stop signal.

40. **Scale ladder — SmolLM2 (135M, 360M) + Qwen3-0.6B (2026-09-18):**
    - Identical 40-cell, 24-pert kill experiment at smaller scales. fp32 CPU.
    - SmolLM2-135M: gap=0.121 [0.065, 0.205], acc=52.5%, P(digit)=0.994, Perm Z=2.7 (p=0.004)
    - SmolLM2-360M: gap=0.318 [0.207, 0.431], acc=90.0%, P(digit)=0.993, Perm Z=6.0 (p<0.001)
    - Qwen3-0.6B:   gap=0.489 [0.347, 0.628], acc=100%, P(digit)=1.000, Perm Z=8.8 (p<0.001)
    - **NON-MONOTONIC SCALING:** SmolLM2 gap peaks at 360M (0.318), drops at 1.7B (0.281).
      Qwen3-0.6B gap (0.489) EXCEEDS Qwen3-1.7B (0.452). Accuracy scales monotonically.
    - **Gradient exists at 135M parameters** with only 52.5% accuracy — significant at p=0.004.
    - RII: SmolLM2-135M=0.712 (D perturbations barely distinguishable), 360M=0.184 (differentiated),
      1.7B=0.443. Qwen3-0.6B=0.169, 1.7B=0.224.
    - **Qwen3-0.6B residual analysis:** residual gap 0.309 [0.115, 0.487], PC1=53.1% (LOWEST
      fragility dominance of all models). ALL GATES PASS. Despite HIGHEST raw gap, residual gap
      is middle-of-pack. Residual RII 0.015 [-0.061, 0.111] (near zero after fragility removal).
    - **Confound:** Qwen3-0.6B uses `Qwen/Qwen3-0.6B` (instruct), 1.7B uses `Qwen3-1.7B-Base`.
      Scale comparison confounded by instruction tuning. SmolLM2 ladder is clean (all same family).
    - **PC1 ordering:** Qwen3-1.7B (51%) < Qwen3-0.6B (53%) << SmolLM2-135M (70%) < SmolLM2-1.7B
      (78%) < SmolLM2-360M (91%). Lower PC1 = more complex sensitivity structure.

41. **Tukey non-additivity + multiplicative structure (2026-09-19):**
    - Tukey 1-df: F=60-490 (p < 1e-13) all 5 models — data IS multiplicative on raw scale.
    - Log transform reduces residual from 16-44% to 12-19%.
    - Log-space permutation p > 0.39 — BUT this used the same broken row-permutation (R20).
      The correctly specified group-label permutation finds 4/5 interaction (see finding 39).
    - Variance decomposition (log): Cell 13-46%, Pert 37-75%, Residual 12-19%.
    - The 12-19% log-space residual is REAL deterministic structure (fp32 CPU, no noise term).
    - In 4/5 models it is organized by perturbation GROUP (cell x group share 0.26-0.40 vs 0.13 H0).

42. **Kill experiment — Qwen3-0.6B-Base (INSTRUCT CONFOUND RESOLVED, 2026-09-19):**
    - Qwen/Qwen3-0.6B-Base, fp32 CPU, 40 cells x 24 perts x 2 surfaces, 314s, 100% accuracy
    - P(digit) median = 0.996. Admissible readout (Q0).
    - **Full-logit:** a<->a=0.910, a<->d=0.586, d<->d=0.394. Gap=0.087 [0.052, 0.129], Z=5.0
    - **5.6x LOWER than instruct variant** (0.087 vs 0.489). LARGEST EFFECT IN THE PROGRAM.
    - Same architecture, same tokenizer, same 40 cells — instruction tuning is the variable.
    - Within-D coherence: 0.394 (Base) vs 0.169 (Instruct) — tuning destroys D-group structure
    - parens<->int_cast: 0.729 (Base) vs 0.183 (Instruct) — NOT tokenization (same tokenizer)
    - Cross-variant fragility rho = 0.333 (p=0.036) — weak, tuning reorganizes fragility
    - Mean distance: 0.025 (Base) vs 0.019 (Instruct) — Base has LARGER distances but LOWER gap
    - Residual gap: 0.241 [0.169, 0.307], PC1=87.8%. ALL GATES PASS. Residual > raw (2.8x)
      same suppressor pattern as SmolLM2-360M (R21 caution at high PC1).
    - **Qwen3 Base scale is MONOTONIC:** 0.6B (0.087) → 1.7B (0.452) = 5.2x increase.
      Prior "non-monotonic" claim was artifact of comparing instruct (0.6B) vs Base (1.7B).
    - **Implications for alignment:** Instruction tuning/RLHF creates massive surface sensitivity.
      Base model treats perturbation types as nearly interchangeable (gap 0.087); post-training
      creates a 5.6x gap. Fine-tuning is the problem, not the architecture.

### Status: Active — R21 CORRECTIONS APPLIED + INSTRUCT CONFOUND RESOLVED (2026-09-19).
  GRADIENT: 8 models confirmed (gap 0.087-0.49, all Perm Z>=2.7). REAL. Exists at 135M params.
  GRADIENT PERM Z: RESOLVED — within-row column shuffle, all Z>2 STAND.
  Phi Q1 DEFINITIVE: gap=0.217, Z=4.6, residual gap 0.386 [0.171, 0.564] — ALL GATES PASS.
  STRUCTURE: MULTIPLICATIVE dominant (first-order Taylor in small-angle regime, R21 derivation).
    R2=1.000 RETRACTED. Correct log-space R2=0.82-0.88. CONTENT is in the 12-25% log residual.
  INTERACTION: 7/8 models show cell x group interaction (nested F + group-label perm).
    R21 CRITICAL: interaction EXISTS AMONG INERTS ALONE (AB vs C, p≤0.004 in 7/8).
    Comment-vs-statement within A+B significant in 7/8. A vs B null in 8/8 (negative control).
    Declared ABCD groups are NOT the axes models use to organize residuals.
  ATTENUATION: A-D cross rho / sqrt(within-A * within-D) = 0.79-1.01. Single common cell
    factor reproduces A-D gap. Inerts predict rewrites as well as other rewrites do (8/8).
    "Not predictable from inert" WITHDRAWN; "less reliable" is supported.
  WITHIN-D COHERENCE: rho 0.17-0.76. D perturbations don't agree on cell ordering.
  D-COHERENCE OVERCLAIMS CORRECTED (R21): All rhos carry CIs (n=40). "Anti-correlated" →
    uncorrelated. "Arithmetic identities" → cluster includes parens. "cond_true = control flow"
    → SmolLM2-1.7B only, driven by identity cells. Tokenization confound: parens/int_cast
    share space-token deletion, are top-3 D pair in 5/8.
  COPY CONTROL (Qwen3): Gap=0.403 [0.288, 0.516], Z=7.9. Expression gap=0.452.
    H_surface PASS (diff=0.049): gradient persists on LITERAL assignments (x=5, no computation).
    H_compute FAIL (ratio=0.89): gradient NOT driven by computation sensitivity.
  LOCUS CONFOUND: 5/8 negative type effect. Interaction primarily POSITIONAL.
  POSITION vs CONTENT (zero-cost, 7 models): Same-position/diff-content rho=0.705 vs
    same-content/diff-position rho=0.530. Position wins 6/7 models (+0.175 mean).
    Perturbations at the same position are MORE similar than same-content at different positions.
  SAME-LINE LOCUS (zero-cost, 7 models): Same-line inerts +0.070 more D-correlated than
    different-line inerts. 7/7 positive, 2/7 significant. Small but consistent.
  DIRECTION STRUCTURE: 3/4 models perm-validated (Z=2.9-5.8) but residual is 12-25% of total.
    NOT D-SPECIFIC: Groups A (Z=8-14) and B (Z=5-7) show STRONGER direction divergence than D.
    Direction divergence is a general property of context modulation, not semantic-type-specific.
  NOISE FLOOR: Qwen3 SNR=316x, SmolLM2-135M SNR=10,408x. Both CLEAN. 0% below noise.
  D-GROUP CROSS-MODEL (8 models): Within-D rho 0.169-0.712. Cross-model consistency 0/28
    significant. No D-perturbation consistently most anomalous. D-coherence does NOT predict
    residual gap (rho=+0.05, p=0.91). Raw gap does NOT predict residual gap (rho=-0.11, p=0.80).
  SCALE LADDER RESIDUALS: SmolLM2-360M residual (0.384) > raw (0.318) but R21 flags suppressor
    artifact at PC1=90.5%. 135M fails (CI incl 0). Non-monotonic not significant (CIs overlap, n=3).
  FABLE R21 ADVERSARIAL REVIEW (this session): 2/6 tautological, 2/6 underpowered, 1/6 inflated
    framing, 1/6 real and damaging (locus confound). Corrections applied to STATE.md.
  COPY CONTROL (4 models complete, 3 admissible):
    | Model | Copy Gap | Expr Gap | Ratio | H_surface | Admissible |
    | Qwen3-1.7B | 0.403 | 0.452 | 89% | PASS (0.049) | YES |
    | SmolLM2-1.7B | 0.386 | 0.281 | 137% | PASS (0.105) | YES |
    | StableLM-2-1.6B | 0.193 | 0.299 | 65% | PASS (0.106) | YES |
    | CodeGen-350M | 0.241 | 0.318 | 76% | PASS (0.077) | NO (70% acc) |
    | Phi-1.5 | 0.197 | 0.217 | 91% | PASS (0.020) | NO (P(digit)=0.001) |
    ALL H_surface PASS, ALL H_compute FAIL. Ratio 65-137%. 3/3 admissible from 3 families.
    2 inadmissible (CodeGen: low acc; Phi: low P(digit)) show same pattern.
    Gradient is surface-driven, not computation-driven. SERIES COMPLETE.
  INSTRUCT CONFOUND: RESOLVED. Qwen3-0.6B-Base gap=0.087 vs instruct 0.489 (5.6x).
    Instruction tuning is the largest single effect. Qwen3 Base scale is monotonic.
    Base within-D rho 0.394 vs instruct 0.169. parens<->int_cast 0.729 vs 0.183.
  INSTRUCT POSITION EFFECT (Qwen3-0.6B base vs instruct, zero-cost):
    Instruction tuning SELECTIVELY reshapes correlations:
    A<->C (inert cross-position): 0.410 -> 0.503 (UP — learns inerts are "same kind")
    A<->D (cross-type): 0.586 -> 0.301 (DOWN — becomes more type-discriminative)
    D<->D: 0.394 -> 0.169 (DOWN — destroys rewrite coherence)
    Gap: 0.324 -> 0.489 (UP — gradient AMPLIFIED)
    Qwen3 IT amplifies gap 1.5x (like-for-like). BUT SmolLM2-1.7B-Instruct REPLICATION
    FAILS: gap 0.281→0.251 (DECREASES). SmolLM2 IT uniformly amplifies all groups 2-2.8x
    (no selective D reduction). Effect is ARCHITECTURE-DEPENDENT, not general.
  MEANING-CHANGE TRANSFER (Group E, 2 models):
    Qwen3-1.7B: D→E rho=0.536 (sig); Inert→E rho=0.261 (NS). D predicts, inert does not.
    SmolLM2-1.7B REPLICATION: D→E rho=0.727 (sig); Inert→E rho=0.514 (sig). BOTH predict.
    UNIVERSAL: D→E > Inert→E (Δ=0.213-0.275). D is ALWAYS the better predictor.
    NOT UNIVERSAL: Whether inert alone predicts depends on model's A-D correlation baseline.
    Per-pert: additive shifts (+1, -1) and reassignment transfer well; multiplicative/negation do not.
  SPACE-TOKEN CONFOUND — DISMISSED (zero-cost, 7 models): R21 flagged parens/int_cast
    share space-token deletion. Test: Jaccard of token-level edits vs D-pair behavioral rho.
    Grand r=0.094, p=0.340. NULL. Control: add_zero/zero_plus Jaccard=1.0 but rho=0.023-0.777.
    parens/int_cast coincidence, not causal. One exception: Qwen3-1.7B r=0.523 (p=0.045).
  POSITION SHIFT MECHANISM (zero-cost, 7 models): Token-count similarity does NOT predict
    C-group pair correlation. Grand r=0.010, p=0.918, n=105 pooled pairs. 5/7 models negative.
    Prediction was: C pairs with same token-count shift should correlate more. FALSIFIED.
    SmolLM2-1.7B highest C pair (rho=0.875) has LARGEST shift diff (4 tokens) — opposite.
    Position organization is QUALITATIVE (before/after in causal attention window), not
    QUANTITATIVE (how many tokens shift the position). Mechanism: before-perturbations enter
    the assignment's attention context, after-perturbations don't.
  OPEN: (1) DONE — Copy control complete (5/5). (2) Noise floor on remaining models.
  (3) DONE — Space-token confound DISMISSED (grand r=0.094, NS).
  (4) Locus-crossed design. (5) Instance replicates (k>=3).
  (6) Nested F + group-label perm on Qwen3-0.6B-Base. (7) Base model for other families.
  (8) DONE — IT replication on SmolLM2: FAILS (architecture-dependent).
  (9) Design causal transfer test. (10) DONE — SmolLM2 meaning-change: D→E 0.727, Inert→E 0.514.
      R24 CORRECTION: on non-identity cells (n=24), Inert→E drops to 0.143 (NS); D→E to 0.393
      (borderline). Transfer driven by 2 of 6 D members (add_zero, cond_true). Token-prefix null.
  (11) DONE — Header relocation (R22 E1) on Qwen3: gap halved (0.453→0.234). Header in middle
      disrupts all correlations. ~50% header effect, ~50% genuine structural. C_C decreased (not
      increased as adjacency predicted). D_D unchanged.
  (12) Readout comparison: gap survives on 11-bin in all 4 models (0.19-0.37). But Qwen3
      cell-profile cross-readout rho ≈ 0 — full-logit and 11-bin measure different cells.
  (13) Balanced cell v3 on Qwen3: Gap=0.500, CI [0.411, 0.588].
  (14) DONE — Balanced cell v3 on SmolLM2-1.7B: Gap=0.275, CI [0.202, 0.362], z=4.33.
      DD < AD (0.531 < 0.669). Residual gap 0.389 (HIGHER than raw — operator strata absorb
      some gap). Replicates Qwen3 finding at lower magnitude.
  (15) DONE — Prefix-form crossover on Qwen3 (2x2: add_zero/zero_plus x plus_one/one_plus).
      Diagonal mean rho 0.349, off-diagonal 0.063 (non-identity n=24). Partial prefix support:
      add_zero->plus_one=0.634 (strong, same prefix), add_zero->one_plus=-0.301 (anti, diff prefix).
      BUT zero_plus->plus_one=0.428 > zero_plus->one_plus=0.063 (breaks prefix prediction).
      E-E rho (plus_one vs one_plus) = -0.143: two E members are independent.
      Prefix form is partial mediator; add_zero is the dominant D member.
  (16) DONE — Prefix-form crossover REPLICATION on SmolLM2-1.7B: signs replicate Qwen3.
      add_zero->plus_one=0.676, add_zero->one_plus=-0.405, zero_plus->plus_one=0.390,
      zero_plus->one_plus=-0.159. E-E rho: **-0.470 (p=0.021, SIG)**.
  (17) DONE — Variance analysis (zero-cost): add_zero has 3.5-4.3x more variance than
      zero_plus. d_temp_var REVERSES column effect. Mean D -> E is NS in both models.
      Neither prefix nor gradient hypothesis survives D-level aggregation. Transfer is
      perturbation-member-specific, not group-specific. E-E anti-correlation is robust.
  (18) DONE — 11-bin crossover reanalysis (Qwen3, 160 fwd passes, old 40 cells, 2026-09-19):
      E-E flips from -0.143 (NS) to +0.758 (p<0.001). ALL four entries positive on 11-bin.
      Off-diagonal > diagonal. Token-prefix null REFUTED on answer distribution.
      Gap reframed as D-diversity, not D-disconnection.
  (19) DONE — 11-bin crossover replication (SmolLM2, 160 fwd passes, old 40 cells, 2026-09-19):
      E-E flips from -0.470 (p=0.021) to +0.124 (NS, p=0.563). Anti-correlation artifact
      CONFIRMED (sign flips in both models). BUT broad positive transfer is QWEN3-SPECIFIC:
      SmolLM2 has 1/4 significant vs 4/4. Only add_zero→plus_one survives in both.
  (20) DONE — Fable R25 adversarial review (2026-09-19): DEVASTATING.
      2×2 is a COLUMN main effect, not diagonal. `plus_one` transfers, `one_plus` doesn't.
      `one_plus` is a garden-path column (precedence confound on `* // %` cells).
      Operator strata REVERSE Qwen3 diagonal (within-stratum ad→plu = -0.23).
      SmolLM2 ad→plu survives within-operator (+0.52) but interaction collapses to 0.04.
      RETRACTIONS: "prefix form partial mediator" → downgrade; "E-E independence" → retract;
      "meaning change determines cells" → withdraw; "token-prefix sharing" → narrow to
      late-edit locality. Experiment was on old 40-cell set (violation of balanced-v3 mandate).
      PROPOSED EXPERIMENTS: (1) meaning-preserving late twins on balanced v3 + both readouts
      (the single most informative cheap test); (2) edit-position gradient; (3) `1+(e)` control;
      (4) rerun 2×2 on balanced v3 + 11-bin; (5) report four cells with CIs, not diag/offdiag.
  (21) DONE — Within-operator stratum analysis (balanced v3, zero-cost, 2026-09-19):
      Gap SURVIVES and GROWS within operator strata. Qwen3: residual gap 0.515 > raw 0.500.
      SmolLM2: residual 0.365 > raw 0.275. DD < AD in 5/5 (Qwen3), 3/5 (SmolLM2) strata.
      d_add_zero and d_zero_plus have LOWEST AD correlations (0.068/-0.171 Qwen3) — the
      crossover's D members are the least representative. R25's operator-stratum attack applies
      to the crossover, not to the gap itself.
  (22) DONE — Late-twin disambiguation (balanced v3, 6 perts, both readouts, 2026-09-19):
      D (e+0, e-0) vs E (e+1, e-1, e+2, e+3). All append-after-e. 828 fwd passes per model.
      **MEANING IS DEAD ON 11-BIN.** Replicated across Qwen3 and SmolLM2.
      | | Qwen3 FL | Qwen3 11b | SmolLM2 FL | SmolLM2 11b |
      |---|---|---|---|---|
      | D-D - D-E | +0.234 [+0.082,+0.382] | -0.053 | +0.236 [+0.106,+0.357] | +0.015 [-0.171,+0.200] |
      FL: meaning matters (CI excludes zero). 11-bin: meaning IRRELEVANT (CI spans zero).
      E-E coherence structured by answer-shift direction (same-sign=0.836 vs opposite=0.725).
  (23) DONE — Cross-position transfer (zero-cost reanalysis, balanced v3, 2026-09-19):
      Reanalyzed existing kill data for A (after) vs C (before) position effects.
      A-A > A-C in both models (CI excludes zero). A-D ~ C-D in both (rewrites equally low).
      Position matters for after-perts, less so for before-perts (C-C < A-C in Qwen3).
  (24) DONE — StableLM-2-1.6B late-twin disambiguation (3rd arch, 828 fwd passes, 2026-09-19):
      EDIT LOCALITY ON BOTH READOUTS. FL: D-D-DE=-0.017 CI[-0.170,+0.129]. 11b: -0.044 CI[-0.201,+0.122].
      Unlike Qwen3/SmolLM2, StableLM does NOT distinguish +/- in formatting geometry.
      FL meaning distinction is MODEL-SPECIFIC. 11-bin edit locality now 3/3 models.
  (25) DONE — Trivial pert inflation test (zero-cost, both models, 2026-09-19):
      Dropping trivial A perts reduces gap by 10-12%. Gap robust to pert selection.
  (26) DONE — Adversarial self-review (2026-09-19):
      Strongest attack: D has 1 pair. "Fail to detect" not "meaning dead." Expand D planned.
  (27) DONE — Expanded-D experiment (3 models, 6 D-D pairs each, 2026-09-19):
      11-bin: 0/3 positive meaning. Qwen3 REVERSES (D-D-DE=-0.111, CI excl zero negative).
      SmolLM2 +0.085 (CI spans zero), StableLM +0.062 (CI spans zero).
      FL: SmolLM2 only clear (+0.180, CI excl zero). Qwen3 VANISHES. StableLM marginal.
      Operator-family decomposition: within same family, meaning=+0.002/+0.008/-0.090 on 11-bin.
  (28) DONE — Operator family decomposition (zero-cost, 2026-09-19):
      All 3 models: ZERO within-family meaning effect on 11-bin. Family > meaning.
  (29) DONE — SmolLM2-360M expanded-D (4th model, 2026-09-20):
      FL: D-D-DE=-0.031 CI[-0.076,+0.009]. 11b: -0.035 CI[-0.097,+0.024]. BOTH REVERSED.
      Supports operator-family confound critique: D-D dragged down by cross-family pairs.
  (30) DONE — Adversarial review of expanded-D (Fable, 2026-09-20):
      1 SOLID (FL=family-clustering), 2 QUALIFIED (meaning-dead, family>meaning), 1 FRAGILE
      (expanded-D introduced family confound worse than 1-pair variance).
      CRITICAL: D={+0,-0,*1,**1} spans 3 families. D-D is cross-family, D-E includes
      same-family. Comparison structurally confounded. Fix: same-family D expansion.

  (31) DONE → RETRACTED — Isomorphic same-family D (4 models, 2026-09-20):
      D={+0,-0,+(0),-(0)}, E={+1,-1,+(1),-(1)}. Isomorphic sign×form. 552 fwd/model.
      RETRACTED: DD-DE comparison confounded by answer identity (D→x, E→x±1).
      Correct comparison (DD-EE): nothing significant. Answer identity test:
      D-D <= E-E-same-sign in ALL 8 tests. Meaning has NO independent signal.
      In arithmetic, meaning ≡ answer identity → cannot be separated.
  (32) DONE — Answer identity confound analysis (zero-cost reanalysis, 2026-09-20):
      Repartitioned all 28 pairwise correlations by answer identity (same/diff answer).
      D-D ~ E-E-same-sign in SmolLM2-1.7B FL (+0.012 diff). REVERSE in 6/8 tests.
      Dominant factor: sign matching (~0.18 gap), not meaning (~0.01 after control).
      Meaning question CLOSED for arithmetic (meaning ≡ answer identity).

---

## OMT-0 Omega Modulus Transfer Test — 2026-09-15/17

Systematic empirical test of the omega modulus as a behavioral preservation predictor.
~30,000+ readouts across 65+ experiments on Qwen3 (0.6B/1.7B/4B), Pythia, StableLM, Gemma, GPT-2.

**Central question:** Can omega (modulus of continuity under behavioral distance)
trained on one set of conditions predict preservation under different conditions?

### Experiments (reverse chronological):

| # | Experiment | Readouts | Key Finding |
|---|---|---|---|
| 81 | **Pre-registered identity prediction** | 72 | PRE-REGISTERED PREDICTIONS: 0/4 CORRECT. Two-factor model (meaningfulness x baseline) COMPLETELY FAILS on held-out pairs. var_echo: predicted amplify, actual 0.1x (baseline 0.136, much higher than expected). english_synonym: predicted suppress, actual 4.0x (meaningful words CAN amplify). code_return: predicted amplify, actual 1.5x. pragma: predicted amplify, actual 0.2x. CONCLUSION: identity amplification is NOT predictable from simple surface token features. The two-factor model overfit the training pairs. Omega's pair-dependence is an irreducible empirical quantity — must be MEASURED, not derived |
| 80 | **Surface type dissection** | 108 | THE MISSING FACTOR: Identity amplification depends on whether surface tokens are MEANINGFUL or ARBITRARY, not structural vs semantic. comment_short ("# a" vs "# b", arbitrary labels) has id=5.3x; comment_long ("# note here" vs "# info here", real words) has id=0.3x. SAME baselines (0.013 vs 0.011). Model's language predictions about real words compete with code-execution chain. Heavy shows opposite pattern: stmt_vs_stmt (8.3x) and cross_type (7.7x) are strongest — heavy wins when surfaces contain actual code. blank_vs_blank is INERT (0.4x for both). Clean rule: arbitrary-token surfaces let identity dominate; meaningful-token surfaces suppress it |
| 79 | **Regime classifier (Fable E4)** | 144 | Three regimes are NOT a clean continuum. Spearman baseline vs id_amp: rho=-0.571, p=0.139 (not significant). CRITICAL: strong_explicit (most semantically assertive, "x is definitely 1") has identity SUPPRESSION (0.7x) and high baseline (0.025) — assertion strength ≠ regime. boundary (0.013 baseline) has 6.0x identity but comment_generic (0.019 baseline) has 0.3x — similar baselines, opposite identity effects. code_noise has heavy=5.4x (high). Heavy is qualitatively universal (always positive delta) but quantitatively varies 5.4x in delta_TV across pairs. Per-cell baselines vary up to 20x within a pair |
| 78 | **Throwaway variable boundary** | 204 | Cancellation on strong_assert tracks pure-underscore names: _ (0.48x), __ (0.53x), ___ (0.68x) all cancel; _0 (0.96x), _x (0.89x), tmp (0.90x), counter (1.04x) do NOT cancel. Boundary: pure-underscore = throwaway → cancellation; any alphanumeric = real → no cancellation. `result` outlier: 30.0x single amplification (highest ever for collision!) and 0.60x cancellation (moderate, possibly ceiling effect). On boundary_pair: ALL names accumulate (ratio > 1), no cancellation for any name. SEMANTIC: amplification depends on variable name's semantic relationship to output: result (30x) >> tmp/__ (14-15x) >> counter (8.5x) |
| 77 | **Verbatim vs renamed repeat (Fable E3)** | 120 | Renaming `_` to `__` does NOT break double-collision dip (4.8x vs 5.4x verbatim), but `_x` DOES (13.5x). Cancellation is throwaway-variable redundancy recognition, not token-level copy suppression. double_mixed beats single (13.5x vs 11.2x). identity_plus_read (var=var + _=var) = 19.6x on strong_assert (highest ever). On boundary_pair: double_verbatim INCREASES (3.7x vs 2.8x single) — no cancellation, opposite of strong_assert |
| 76 | **Stacking grid on new pairs (Fable E1)** | 192 | Composition algebra CONFIRMED pair-conditioned across 4 pairs. identity ranges 13.5x (strong_assert) → 6.0x (boundary) → 1.3x (assert_weak) → 0.3x SUPPRESSION (comment_generic). ONLY 2 UNIVERSAL CONDITIONS: heavy (+) and identity+heavy (+). 13/15 conditions are MIXED sign across pairs. Pair correlation matrix: two weak clusters (sa+bp r=0.55, aw+cg r=0.38), near-zero cross-cluster. assert_weak has HIGH empty baseline (0.056) — most conditions suppress below it |
| 75 | **Recency gate (Fable E2)** | 144 | Identity decays with distance on BOTH pairs. strong_assert: 13.5x → 10.1x → 4.5x → 2.5x (monotonic decay). boundary_pair: 6.0x → 3.6x → 3.1x → 4.2x (dips then RECOVERS). Heavy: OPPOSITE pattern — decays on strong_assert but GROWS on boundary_pair (2.7x → 8.1x). Per-cell: x_1 resistant to decay, y cells decay faster. Fable half-right: both pairs show decay, but different mechanisms (positional vs structural recovery). Heavy's value-mutation is amplified by intervening pass lines on boundary_pair |
| 74 | **Collision repetition curve** | 120 | EVEN-ODD OSCILLATION: match collision 1=11.2x, 2=5.4x (DIP!), 3=12.3x (recovery!), 4=14.3x. Mismatch monotonic (no dip). At rep=2 mismatch overtakes match (6.0x vs 5.4x). Explains #73 double_collision self-cancel: even-phase destructive interference. Two matching copies pair-resolve ambiguity; three create new odd-man-out. Envelope grows: 4 reps = 14.3x |
| 73 | **Amplification stacking** | 192 | Stacking is REAL, non-commutative, heterogeneous. collision+isinstance=2.1x superadditive (strongest). collision→identity=2.0x but identity→collision=1.7x (order matters). double_collision=0.5x SELF-CANCELLING (destructive interference!). double_identity=1.2x. Most combos near-additive (0.8-1.1x). Collision stacks with everything except itself |
| 72 | **Collision form generalization** | 96 | Value-mentioning forms (type(val), print(val), assert var==val) ALL amplify 8-11x. eval_var (_ = var) only 1.9x — variable read barely amplifies. assign_match strongest (11.2x). Category: value_mention (9.6x) >> var_reference (5.4x) >> control (2.9x). CRITICAL: identity (var=var, 13x) >> eval_var (_ = var, 1.9x) proves identity paradox is SELF-ASSIGNMENT (circular binding), not variable evaluation |
| 71 | **Literal-value collision** | 168 | COLLISION CONFIRMED: value-matching `_ = val` amplifies 8.1x on strong_assert (vs 4.1x mismatch). Match+identity=28.1x (doubles identity's 13.2x). Match+heavy=27.2x. Two causal paths to same value create CONSTRUCTIVE interference via representational ambiguity. Per-cell range: 3.3x-11.8x (consistent). New amplification primitive alongside identity |
| 70 | **Depth gradient** | 144 | Pair-dependent depth scaling. On strong_assert: identity peaks at depth 3 (13.5x), non-monotonic. Heavy roughly monotonic (→11.6x). On boundary_pair: baseline TV grows with depth (0.005→0.046), drowning perturbation — identity=1.0x at depth 4. Scope-chain hypothesis PARTIALLY CONFIRMED: real effect visible only on well-controlled pairs. Measurement requires strongly-anchored surfaces |
| 69 | **Pair vulnerability spectrum** | 120 | Anchoring hypothesis REFUTED. Vulnerability is COMPOSITION-SPECIFIC: assert_weak most absolutely vulnerable, comment_generic PROTECTS against identity+safety (27x lower TV than boundary) but vulnerable to heavy (6x). No pair is universally weak/strong. Margin gradient: assert_strong (0.55) > assert_weak (0.40) > boundary/comment (0.14-0.19) |
| 68 | **Non-commutativity structure** | 432 | 6x6 composition matrix across 2 pairs. Empty is identity element (perfectly commutative). Mostly subadditive. Biggest asymmetry: nonvar→safety 3.54x. Only 1 superadditive pair. **KEY**: algebra is SURFACE-PAIR-DEPENDENT — identity→safety = 0.0118 (strong_assert) vs 0.1129 (boundary), a 9.6x difference. Same compositions, different pairs, dramatically different effects |
| 67 | **Error correction test** | 72 traj | **PREDICTIONS WRONG at step-40 (52.8%)** but **PARTIALLY RIGHT with robust scoring (66.7%)**. Step-40 TV contaminated by 2-step oscillation artifact (digit vs separator tokens). Robust metric: heavy/reassign correctly predicted to diverge (4/4 each). nonvar_42/safety_comment correctly converge (4/4). empty incorrectly diverges (baseline instability). nonvar_expr incorrectly diverges (any token insertion destabilizes). Single-readout vs trajectory INVERSION confirmed but oscillation complicates interpretation | Heavy ALWAYS converges (0/4 predicted correctly) — mutation acts as attractor reset. Reassign ALWAYS converges (0/4). Empty mostly DIVERGES (3/4) — surface differences compound chaotically. **ACTUAL MECHANISM**: trajectory fate governed by DETERMINISTIC STATE ANCHORING, not mutation detection. Single-readout TV and trajectory fate are INVERSE: mutations converge trajectories (same attractor), no-ops diverge them (chaotic amplification) |
| 67z | **Variable access gradient (zero-compute)** | 0 | Exp #55 data reanalysis: clean monotonic gradient by variable interaction type. none/comment: 0% flips, margin=0.81-0.86. code-read (type_check): 11.1% flips, margin=0.59. noop-write (identity): 11.1%, margin=0.63. mutate (heavy): 22.2%, margin=0.39. Critical jump at comment→code-read boundary. type_check can CORRECT errors (9→correct_digit) |
| 67y | **Composition algebra detail (zero-compute)** | 0 | Chain data: non-commutativity scales with strength difference (IN/NI=6.33x). ALL superadditive chains end with identity (TI=1.71x, OI=1.81x). Identity is universal amplifier, neutral is universal damper. INI sandwich = 2.7x superadditive. Identity self-composition peaks at length 2 (III < II) |
| 66a | **Bifurcation mechanism (zero-compute)** | 0 | TWO MECHANISMS found in #56 data. Type A: boundary_pair attractor capture at step 1 (surface_b locks onto 9-attractor). Type B: strong_assert error correction — step-4 spike resolves based on mutation semantics: heavy 4/4 diverge, nonvar_42 1/4 diverge, identity 3/4 diverge. Model performs AUTOREGRESSIVE ERROR CORRECTION |
| 66b | **Composition algebra (zero-compute)** | 0 | #65 data reveals NON-COMMUTATIVE mostly SUBADDITIVE algebra. neutral→identity=0.0998 (1.77x superadditive, 4x stronger than reverse). Most compositions are subadditive (0.3-0.6x). Only "weak-then-strong" orderings create superadditivity. Weak perturbations PRIME for amplification by subsequent strong perturbations |
| 60b | **Bifurcation fate predictor** | 42 pairs (partial) | Initial omega (step-0 TV) is a WEAK predictor of autoregressive grow-vs-collapse fate: TV AUC=0.632 (<0.7 threshold), margin AUC=0.600, TV-fate corr=0.339. **Continuation type dominates fate**: nonvar_42 always collapses (0%), safety_comment mostly collapses (14%), identity/heavy/type_check ~57% grow. All 42/108 pairs eligible. Omega does not beat simple confidence margin. Supports Fable R5: bimodality is continuation-type-driven, not omega-predictable |
| 58r | **R6-corrected cross-Q reanalysis** | 0 (analysis) | Fair margin, baseline-flip filter, direct Spearman on #58 data. KEY: assert↔print ρ=0.651 (p=0.003), pooled ρ=0.222. Nothing transfers to addition (neither TV nor margin). Family-dependent, not universal negative |
| R6 | **Fable strategic review R6** | — | Five problems with #58 design: comparison Q readout-mismatched, margin baseline unfair, low power, no baseline-flip filter, Spearman is primary quantity. Reframes: TV<margin/2 is a theorem. Omega's value is as certificate, not flip predictor |
| E1 | **Token-persisted bifurcation rerun** | ~144 trajectories | RUNNING (5/72 conditions). R5 Priority #1. Early data: 3/5 conditions show UNSTABLE labels (change with horizon), supporting greedy-artifact hypothesis |
| E2 | **Null control + distributional TV** | ~1500 fwd | QUEUED. R5 Priority #2. ARM1: arbitrary edits (not equiv pairs) give same grow rate? ARM2: T=0.7 sampling preserves bimodality? |
| 65 | **Composition ordering** | 473 | **NON-COMMUTATIVE**: neutral_then_identity amplifies 2.1x above identity alone (0.100 vs 0.047). 7.3x asymmetry on verified/bare pair. 4x on neutral/identity vs reverse. Order of continuations matters dramatically. Static omega data (not trajectory) |
| 64 | **Perturbation zoo** | 360 | 15 novel perturbation types mapped. Flip DIRECTION matters: if_pass pushes toward attractor 9, assign_zero pulls away. bool_var highest TV amplification (12.9x) but zero flips. TV magnitude ≠ flip risk — directional relationship to decision boundary determines outcome. 5 flips across 168 eligible combinations |
| 63b | **Full-vocab incremental** | 0 (analysis) | Confirms #55b: full-vocab ≡ 11-bin (corr=1.0, AUC=0.832 all 4 combos). Flip cells 2.95x higher TV. Per Fable R5: same-readout robustness check, not direction-changing |
| 63 | **Full-vocab consequence** | 288 | Full-vocab TV adds nothing over 11-bin for code pairs. TV ratio fullvocab/11-bin = 1.0004. Same 6 flips, same correlations (0.347), same accuracy (78.1%). 11-bin readout is sufficient for this domain |
| 62 | **Omega incremental value** | 0 (analysis) | Margin alone AUC=0.977, adding omega DECREASES to 0.963. ⚠️ **Fable R5 caveat**: same-readout null on 7 flips, no eligibility filter. Margin trivially wins when both computed from same 11-bin readout. UNTESTED, not dead — cross-question #58 is the admissible test |
| 61 | **Bifurcation predictor** | 0 (analysis) | Initial TV predicts autoregressive growth with AUC=0.676, r=+0.327. BUT continuation type dominates: identity 88% grow, heavy 75%, empty/nonvar 38%. ⚠️ **Fable R5**: autoregressive bimodality (#56) is greedy-decode artifact — need temperature experiment |
| 60 | **Baseline feature comparison** | 0 (analysis) | Margin (r=-0.406, F1=0.632) beats omega (r=+0.272, F1=0.326). ⚠️ **Fable R5 caveat**: same-readout null on 7 flips. Margin beating omega is trivially expected when both share readout. Cross-question test (#58) needed |
| 59 | **Digit-prior sweep** | 576 | **RIVAL_FAILS on stable cells**: Leading-digit match rate 9.7% (chance). Argmax perfectly tracks cell value {1,2,3,5,6,7} regardless of `_ = k` literal — zero flips across all 576 readouts. Attractor prediction (digit=9 lower TV) holds weakly (0.027 vs 0.033). ⚠️ **Caveat**: cells used (vals 1-7) are high-confidence; original R4 digit-copying claim was on uncertain cells. Test does NOT cover the uncertain-cell regime. **New finding**: literal-cell-value INTERFERENCE — when literal matches cell val, TV is HIGHER (e.g. nonvar_3 on val=3: 0.075 vs 0.027 on val≠3), suggesting collision not reinforcement |
| 58 | **Cross-question consequence** | 500 (80 pairs after filtering) | **FAMILY-DEPENDENT TRANSFER (R6-corrected).** Mean transfer TV AUC=0.498 (chance). BUT assert↔print Spearman ρ=0.651 (p=0.003) — genuine intra-family transfer. Addition shows no stability (ρ=-0.135, 0.057). Fair margin ALSO fails at transfer (AUC 0.386). Same-Q certificate: 0/49 violations; cross-Q: 7-20% violations. Verdict: omega is question-FAMILY-dependent. Within coherent families (questions probing same state), omega transfers. Across family boundaries, nothing transfers |
| 57 | **Length scaling** | 1008 | **TYPE-DEPENDENT**: heavy shows INVERTED-U (peak n=1-2, omega 0.106-0.126, then declines to 0.077 at n=12). identity is STEP FUNCTION (2.25x at n=1, flat after). nonvar shows persistent LOG growth (0.029 to 0.121 at n=12). comment SUPPRESSIVE (0.91x, below baseline). Steepest initial step: heavy 3.69x. Most sustained: nonvar. Pair-dependent: strong_assert_2/nonvar monotonic to 0.241; boundary_pair/heavy peaks n=1-2 then drops to baseline. More heavy lines = LESS sensitivity |
| 56 | **Autoregressive compounding** | 1984 | **BIMODAL**: Omega over 30 generated tokens grows 12.1x overall (TV 0.048->0.582). BUT outcome is binary: 19/32 pairs grow to TV~0.95 (near-total divergence), 13/32 collapse to TV~0.0001 (convergence). Zero stable. identity most reliable amplifier (7/8 grow, final TV=0.855); heavy 6/8; empty and nonvar_42 coin flips (3/8). Critical transition at steps 4-8: abrupt phase jump within 1-2 tokens. Pair type secondary to continuation type |
| 55 | **Cross-model consequence** | 432 | **NEGATIVE**: tau=0.3 from 1.7B does NOT transfer. 0.6B: 2.3% acc (near-chance), r=0.307, best F1=0.143 at tau=0.05, 1 flip/108 pairs. 4B: 91.2% acc, r=0.284, best F1=0.333 at tau=0.15, 3 flips/108 pairs. Ref 1.7B: r=0.677, F1=0.75. Sweet spot at 1.7B; 0.6B too weak, 4B too robust. Correlation NOT monotonic with scale |
| 54 | **Identity paradox (BASE model)** | 288 | Identity HAS 11.1% flip rate on base model (TV=0.042). 0% flip rate in #49 was instruct-model bug. nonvar_42: 0% flips despite TV=0.034 (rank_corr=0.990, top3=1.00) — true rotation-like change. heavy: 22.2% flips, TV=0.074. Different continuations produce structurally different distributional changes |
| 53 | **Prospective consequence (BASE)** | 576 | **NEGATIVE on base**: Train F1=0.220, Test F1=0.129±0.075, generalization 58.6%, test r=0.222. Fork agent's instruct results (F1=0.696) do NOT replicate on base model. Omega is a weak argmax-flip predictor without RLHF |
| 52 | **State reconstruction** | 240 | Canonical records tautological TV=0 (both surfaces → identical text). Appended overwrite AMPLIFIES 6x on target_query, reduces ortho_query. Repair shifts sensitivity |
| 51 | **Referent/name/query crossover** | 384 | Dual-variable design: target-specificity index = -0.008 (negative). Amplification general, not referent-specific. Baseline gap explains apparent 5x |
| 50c | **Referent test** | 252 | **REVERSAL**: tracked-var continuations SUPPRESS omega (0.085), not amplify. tracked_value (var=val) near-zero (0.035). other_var_identity (w=w) HIGHEST (0.339), exceeding baseline (0.296). Gradient tracked>other>none is FALSE. Tracked var acts as re-anchoring mechanism |
| 50b | Task transfer (arithmetic) | 270 | heavy is TASK-GENERAL (omega=0.186, 19x identity). safety_donot is TASK-SPECIFIC (0.003 on arithmetic vs 0.09 on scoping). Neutral near-zero on arithmetic |
| 49b | **Consequence prediction (INSTRUCT)** | 288 | On Qwen3-1.7B (instruct, model-id bug): r_pb=0.677, F1=0.750 at tau=0.3. Does NOT replicate on base model (#53). RLHF-specific |
| 48b | Safety comments on instruct | 432 | RLHF POLARIZES omega: heavy amplified 1.7-1.9x, neutral 3.8x on tight pairs. Safety/certainty SUPPRESSED (0.07-0.19x). Pair-dependent: safety_donot 7.4x tight, 1.0x wide |
| — | **Revival** | 1584 | **MAJOR**: contracted distinctions CAN be revived. reassign_42 contracts 0.32x; none_assign+isinstance_check revives 2.6x. del_var amplifies 3.27x. "Close now" ≠ "interchangeable later" |
| — | State reconstruction vs repair | 240 | Canonical records tautological TV=0 (design flaw). Appended overwrite AMPLIFIES 6x. Shifts sensitivity, doesn't repair |
| — | Referent/name/query crossover | 384 | Amplification is **GENERAL**, not referent-specific. Target-specificity index = -0.008. Mentioning target amplifies OTHER query 5x more |
| — | Boundary-preserving syntax | 192 | Confound real but effect survives. bare_var_nl (25x) strongest. nonvar_expr (10x) = identity. Computation context, not variable reference |
| — | **Fable R3 corrections** | 0 (review) | **#38 boundary confound**; **#40 NN has no var**; **#45 circular score**; **#39 heterogeneous**. Pivot: predict failures, not catalog amplifiers |
| 50 | NL full-vocab omega | 160 | **MAJOR**: NL omega (5-17x) EXCEEDS code omega (2-5x) with full-vocab TV. Domain gap was entirely 11-bin artifact |
| 39 | Matched measurement reanalysis | 0 (reanalysis) | CORRECTS #33: on matched cells instruct heavy RISES #3→#1 (4.6x); NL 11-bin TV=0 (artifact) |
| 49 | Meta-analysis | 0 (analysis) | heavy_bare ONLY universal amplifier (100%); identity_bare universal preserver (0%); neutral 67%; all comments preserve; pass > identity |
| 48 | Safety comments on instruct | — | Running: does safety_donot amplification hold on instruct model? |
| 47 | Chain amplification | 300 | Neutral GROWS monotonically (4.4x over 3 steps); identity saturates; HN=0.564 worst combo; neutral accumulation more dangerous than mutations |
| 46 | Baseline predictor | 0 (analysis) | Baseline TV predicts omega (rho=0.74); BUT inverted on tight pairs (rho=-0.31); safest-looking pairs most vulnerable |
| 45 | Minimal diagnostic set | 0 (analysis) | 3-probe set {comment_var, type_check, var_only} achieves rho=0.61; binary screening works, point prediction limited |
| 44 | Comment taxonomy | 504 | Safety comments AMPLIFY 7.45x; deprecation SUPPRESSES 0.15x; "do not modify" increases sensitivity opposite of intent |
| 43 | Authority framing | 288 | Doc frame SUPPRESSES neutral 97% on tight pairs but AMPLIFIES 3.5x on wide pairs; omega is frame-sensitive |
| 48 | Fixed-cell comparison | 360 | On fixed cells: nonvar_42 beats identity 2/3 pairs (max), 3/3 (mean). Value matters: 42 vs 99 = 2.5x. Double nonvar < single. Token-specific modulation |
| 46 | Boundary-preserving syntax crossover | 192 | Boundary artifact CONFIRMED (4.19x). Valid amplification survives (25x). `_ = 42` beats identity at tau=0.01 — amplification is computation context, not variable reference |
| 44 | Pair property analysis | 576 | REFUTES strong/tight hypothesis: identity amplifies only 2/8 pairs; heavy universal; no simple predictor (category/TV/tokens) |
| 41 | Layer scaling (0.6B vs 1.7B) | 210 | 0.6B onset at L8 (vs L20 in 1.7B); more distributed profile; larger models concentrate omega in late layers |
| 40 | Repetition disentanglement | 540 | Pair-dependent opposing effects: strong pairs additive/last-wins; tight pairs suppressive/first-wins; I_paraphrase ≠ I |
| 38 | Partial identity tokens | 720 | No single trigger; progressive buildup x(0.057)→x=(0.304)→x=x(0.687); newline REDUCES 4x; pair-dependent |
| 37 | Systematic NL pair discovery v2 | 132 | FIRST NL pair at tau=0.05: passive voice "passed by"/"enacted by" (TV=0.028); 12x amplification |
| 36 | Cross-arch instruct (StableLM) | 288 | RLHF reshaping REPLICATES but INVERTED: StableLM heavy #5→#1, neutral #1→#4 (opposite of Qwen3) |
| 35 | Layer-wise fingerprint | 210 | Monotonically increasing; omega=0 in L0-16; onset at L20; 100x jump L16→L20; concentrated in final 30% |
| 34 | Quantization (int8) | 288 | INT8 DESTROYS identity signal (#1→#5); safety certs NOT quant-invariant; TV(fp32,int8)=0.378 |
| 33 | Base vs instruct | 288 | RLHF reshapes omega: heavy #1→#4, neutral #5→#2 (amplifies); omega detects alignment fingerprint |
| 32 | Natural language omega | 130 | Omega EXISTS in NL; tau=0.10 needed (not 0.05); ". " strongest amplifier (3.1x); domain-general framework |
| 31 | Phase transition | 204 | Step function at rep=1 (0.003→0.047); NO gradual ramp; rep_2 < rep_1 (interference) |
| 30 | Bootstrap CIs | 0 (analysis) | Cohen's d=3.49, p=0.0004 for identity vs neutral |
| 29 | Held-out prediction | 312 | 83.3% at tau=0.05, +33pp over naive baseline |
| 28 | Multi-variable scope | 288 | type_check 2.3× stronger with 2+ vars in scope |
| 27 | Variable name sensitivity | 144 | Name-length independent pattern; magnitude varies |
| 26 | Gemma-3-1b replication | 144 | CONFIRMS omega; pass>neutral>identity (inverted) |
| 25 | Architecture replication (Pythia, StableLM) | 288 | Pythia=zero; StableLM=intermediate |
| 24 | Structure transfer (4 code structures) | 576 | class_method=zero amplification; nested_func strongest |
| 23 | Rank-correlation transfer | 0 (analysis) | Spearman rho=0.52; rank predictor 71-86% at tau=0.01 |
| 22 | Max-over-pairs analysis | 0 (analysis) | Does NOT fix multi-pair mixing at tight tau |
| 21 | Perturbation diversity (14 types) | 360 | ALL 14 novel perturbation types amplify at tau=0.01 |
| 20 | Scaling (0.6B/1.7B/4B) | 432 | Identity amplification scales monotonically: 0.003/0.014/0.051 |
| 19 | Pair x depth interaction | 720 | All 5 pairs work at all depths; identity increases with depth |
| 18 | Multi-pair randomized splits | 0 (analysis) | Omega is pair-dependent; loses to code-comment at tau=0.01 |
| 17 | Cross-model (Qwen3-0.6B, GPT-2) | 288 | GPT-2 zero; 0.6B intermediate; scales with capacity |
| 16 | Cross-query transfer | 288 | 100% accuracy at tau=0.02 both directions |
| 15 | Pair transfer (leave-one-out) | 480 | +27.5pp over baseline at tau=0.01 |
| 14 | Repair/undo chains | 432 | Neutral AMPLIFIES identity; amplification irreversible |
| 13 | Measurement stability | 96 | Perfectly deterministic (zero TV between runs) |
| 12 | Token ablation on identity | 480 | Variable mention drives amplification; just_var > identity |
| 11 | Surface pair discovery | 360 | 12+ eligible pairs from 30 candidates |
| 10 | Templates (5 structures) | 720 | All 5 structures show same omega pattern |
| 9 | Transfer (cross-variable) | 432 | Train on x, predict y,z: 83-92% |
| 8 | Gradient (semantic) | 384 | Smooth gradient from neutral to heavy |
| 7 | Queries (format sensitivity) | 576 | Robust across query formats |
| 6 | Values (1-9 sensitivity) | 1,080 | Value-independent |
| 5 | Depth (2/3/4 nesting) | 1,152 | Deeper nesting = stronger amplification |
| 4 | Composition (chain effects) | 1,032 | Non-linear; neutral catalyzes identity |
| 3 | Controls (baselines) | 984 | Baselines validated |
| 2 | Prospective prediction test | 0 (analysis) | Omega predicts preservation at 79% |
| 1 | Fast sweep (core effect) | 3,528 | Core omega effect established |

### Verdict (updated post-architecture replication):
**Structured positive, cross-architecture.** Omega is a real phenomenon confirmed in
3 architecture families (Qwen3, StableLM, Gemma). Code continuations amplify behavioral
distance universally, but WHICH continuations amplify most is model-dependent:
- Qwen3: identity > type_check > heavy > neutral
- StableLM: neutral > type_check > heavy > identity
- Gemma: pass > neutral > type_check > identity
- Pythia: zero (insufficient code training at 1.4B)

Scales monotonically with model capacity (Qwen3 family). Generalizes across variables,
code structures, query formats, perturbation types (14/14 amplify). Partially transfers
across surface-pair families via ranking (rho=0.52). Operational threshold: tau=0.05.

~32,000+ readouts across 62+ experiments, 5 model architectures. Results in
experiments/results/omega_*/. Fable reviews in experiments/results/fable_omega_review_{1,2,3,4}.md.

### Fable R4 Corrections (2026-09-17):
**Devastating reanalysis.** R4 identified that the program quietly stopped measuring
omega (eligibility filter dropped since #46). Key corrections:
- nonvar_42 is **digit copying** (argmax→"4" in 20/36 readouts), not computation context
- Revival revised: 9/30 chains on eligible cells (stronger than R4's 1/25 on unfiltered)
- Consequence AUC=0.918 on eligible cells survives correction
- Gradient reversal, state confirmation anchor, TSI, value-matters 2.5x: **WITHDRAWN**
- Cross-question transfer: **UNTESTED** — that's the real omega claim (exp #58 running)
- Margin (confidence gap) **DOMINATES** omega for same-question flip prediction (AUC 0.977 vs 0.838)
- Omega has **NEGATIVE** incremental value beyond margin (-0.018 delta AUC)
- Length scaling reveals 3 dynamical signatures: heavy decays, identity saturates, nonvar accumulates

### Revised Verdict (post-R4):
**Omega is a real measurable phenomenon but NOT a useful predictor beyond simple confidence margin
for same-question tasks.** The remaining theoretical claim — cross-question certification —
is under test (exp #58). If omega predicts behavior under Q2 from measurements under Q1
better than running Q2 directly, it has genuine value as a cheap certification tool.
If not, the program reduces to "TV between surface pairs is a noisy proxy for confidence."


## S65 supplied-revision diagnostic and user pause — 2026-09-14

Research is paused; no run is active or queued. Retry completed all 18 responses,
all terminated (3/3 per arm). Original citizenship 2/3, supplied citizenship 3/3,
capital prerequisites 0/3, supplied composed capital 0/3. Zero prerequisite-eligible
cases; do not infer missing knowledge or failed competent composition. All outputs
were manually reviewed by root and two independent reviewers. One wrong-subject
record is misattributed at country level. Full evidence: `results/relation_revision_v1/`.

First attempt failed serialization after one generation: 81.698 seconds, zero saved
responses, unrecorded 1–16 forwards. Retry: 51.476 seconds,111 recorded forwards;
combined133.174 seconds,112–127 forwards. Session: nine probe blocks,666 recorded
plus1–16 unrecorded calls (667–682 total),1710 apparatus/0 artifact lines. No GPU.
Natural capital questions are an unrun restart candidate, not a queued experiment.


## S64 recovery attribution and direction check — 2026-09-14

Temporal-state migration is not selected before a useful upgrade. The bounded
source contrast separates added-feature segmentation quality from recovery-specific
evidence and identifies memory/search without new weights as a direct comparator.
Final-trajectory selection differs from immediate decisions. Canonical S64 and
ledger preserve sources and reviewer provenance. No new probe, forward or apparatus.

## S63 temporal-operation source check — 2026-09-14

One primary source and independent review identify contributed features inside
next-frame object association and granularity control. The encoder is frozen but
the decoder and new temporal components receive training. Mixed ablation effects,
oracle-mask diagnostic scope and inconsistent default table figures are preserved;
no general information-absence or repeated-acquisition claim. No new empirical
block, model forward or apparatus. Canonical S63 and ledger retain provenance.

## S62 later-attribute interface comparison — 2026-09-14 (completed)

On fourteen disjoint qualifying test pairs / 31 color questions, mean within-pair
annotation-agreement scores are .630952 for specialist features, .607143 for all
raw logits and .690476 for existing recipient features. Features beat logits on
two pairs, lose on two and tie on ten. Independent audit exactly reproduces all
predictions, vote aggregation, spectra and scores. Thirty-six readers attain 15
slope degrees of freedom; nine are deliberate constants. These are same-family
supervised readers, not zero-shot later questions.

Selected visual review finds body-region/visibility ambiguity in three pairs.
No exclusions or rescoring: retain dataset-agreement scope, not verified color
perception. This panel does not justify requiring specialist features; no universal
logit sufficiency or intrinsic representation-hole result. Close without tuning.

New cost: 180 CPU forward calls, 19.627 supervised extraction seconds; 45 reader
preparations, 5.464 supervised fitting seconds; 4,686,432 test-annotation bytes.
All reproduced logits match the previous cache exactly. Runner 222 apparatus lines;
session eight blocks, 555 forward calls (fifteen earlier batched), 1573/0 code lines.
No GPU or new weights. Raw votes, design, attempts, extraction and result are in
`results/attribute_comparison_v1/`; S62 holds full qualifications and timing scopes.

## S61 source resolution and S62 annotation feasibility — 2026-09-14

S61 distinguishes useful resolution repair from matched-effort causal attribution
and prior-service retention. Its source inquiry is closed. S62 independently
verifies 26 discordant training pairs across six species for fixed breast-color
questions: 54 pair–color comparisons, not independent observations. All 1800 raw
training votes reproduce the aggregates (1440 known, 360 unknown); there is one
vote per image–attribute. Initial extraction: 3,949,152 bytes / 21.62 seconds;
additional raw-vote custody fetch: 3,900,000 bytes / 1.586 seconds. Test attributes
remain unexamined. Frozen design selects a bounded matched-reader probe, not yet
run. No new model block, forward, fitted reader or runner at this checkpoint.
Receipts: `results/attribute_comparison_v1/`; canonical S61/S62 and ledger.

## S59–S60 strategy after the teaching result — 2026-09-14

Two-round review declines a hierarchical-versus-flat classifier run: the direct
union head can retain the old scoped head and reuse the same features. No actual
resource constraint motivates the added hierarchy. S60 independently checks a
primary multi-teacher representation precedent, distinguishing frozen-backbone
downstream readers from unchanged teacher consumers. No local run or new apparatus;
seven blocks, 375 backbone evaluations and 1351/0 code lines remain unchanged.
Canonical S59/S60 and the ledger retain the analytical and source boundaries.

## S58 bounded specialist-teaching precursor — 2026-09-14 (completed)

Fresh review corrects an unrun selection criterion; the selected limited-label
teaching question now has a result, independently reproduced from cached tensors.

| Test correct / 60 | Label draw 0 | Label draw 1 | Label draw 2 |
|---|---:|---:|---:|
| 18 labels only | 52 | 49 | 52 |
| Hard teacher targets | 59 | 59 | 58 |
| Soft teacher targets | 57 | 52 | 55 |
| 120-label reference | 59 | 59 | 59 |

All draws reuse the same sixty test images. Hard teaching corrects 7/10/6 mistakes
with zero regressions against label-only. Teacher hard targets equal all 120
training truths: ordinary additional correct supervision explains the improvement.
Teacher test accuracy is 56/60 full-vocabulary, 59/60 restricted to the six classes.
No soft-information advantage, new backbone knowledge or cumulative claim.

CPU preparation: 285.14 seconds, 18.85 MB images and 52.64 MB model archives.
Compute supervisors: 34.95 seconds total; 360 image-level backbone evaluations,
2400 head optimization steps, twelve candidate heads. All attempts completed.
Runner: 286 apparatus lines, zero reusable artifact-bearing code lines. Session:
seven probe blocks, 375 backbone evaluations, 1351/0 lines. No GPU or panel tuning.
Manifest, preparation, attempts, cache blocks and predictions are retained in
`results/species_teaching_v1/`; canonical S58 explains scope and provenance.

## S57 prepared intervention source contrast — 2026-09-14

One source and independent review identify an offline-prepared recipient write
vector, distinct from S56 live paired reconstruction. Method-level dependency
and selected behavioral illustrations do not establish implementation timings,
new expertise or cumulative composition. No model run or new apparatus.

## S56 feature-to-operation source check — 2026-09-14

One primary report and pinned implementation distinguish behavioral modulation
from executable grafting. Independent review checks the offered-name scorer,
live paired-model input and the masked recipient write path. A static wrong-tool
counterexample is not a model failure rate. No model run, scorer execution, new
probe block or runner; six blocks/fifteen forwards and1065/0 lines unchanged.
The canonical S56 section and ledger hold provenance and qualifications.

## S55 prediction-resource resolution — 2026-09-14

A bounded check identifies a small published species predictor and accessible
image/caption row metadata. No model or image was downloaded. Dataset authors
warn of ImageNet overlap; label order and mirror split provenance are unbound.
Independent review declines the isolated species-head run because its outcomes
do not resolve the active operational-composition decision. Availability is no
longer the deferral reason. No new empirical block, forward or runner line.

## S55 species-teaching comparator clarification — 2026-09-14

Two-round strategy dialogue selects one bounded resource-feasibility allocation
for a possible explicit species service. A generalist without species outputs is
not a valid species-teaching comparator. Label-only, soft specialist teaching and
hard pseudo-label teaching must declare supervision and cost. This is ordinary
assimilation feasibility, not shared-suffix reuse or joint reasoning. No new run,
empirical block, model forward or apparatus line. Details remain in S55.

## S53 reaction-consequence source contract — 2026-09-14

One specialist source and two-round direction review select conditional structural
questions about a predicted product as a consumer-access candidate. A Table 1
illustration was manually traced for four input/output membership cases. This is
not a parser/model run, reaction benchmark or new local empirical block. RDKit
availability was checked read-only; no dependency was installed. The reviewed cost
comparison credits decoding/retaining a graph once. No source generation score is
promoted to local quality evidence; model/reader selection and fitting remain unrun.

## S52 predicted consequences and later-use objective — 2026-09-14

Analytical application of the existing consumer loss separates marginal answers
from shared-analysis use. Independent two-round review verifies the KL decomposition
and keeps correctness distinct from teacher agreement. Separate donor-trained ports
need not modify previous ports. A direct learner may predict and retain a sufficient
record once; repeated-inference baselines would be unfair. No new theorem framework,
runner, local empirical block or model forward. Six blocks/fifteen scored forwards
and1065 apparatus/0 artifact-bearing lines remain unchanged.

## Fresh periodic approach review — 2026-09-14, around 15:30 UTC

After62f85fe and separate entropy checking, independent fresh subagent review
distinguishes unchanged connected growth from shared updates and distinguishes
native-input from induced-input component replay. Joint-service replay is not
necessarily required. A worked hypothetical depth-boundary failure prioritizes
available output/image evidence and direct learned repair as comparators; it is
not an empirical block. Next selection needs a testable useful benefit of sharing
or assimilation, not prior proof of superiority. No real CLI audit or new runner.

## S51 depth-selection interface and endpoint boundary — 2026-09-14

A raw relative-depth interface makes category/depth pixel selection expressible
without learned alignment. This is source/code analysis, not demonstrated quality.
A limited check did not bind the specific trained semantic/spatial endpoint;
local execution of that exact episode closes without extending the search or
substituting a new model under old claims. Independent two-round dialogue favored
the complete-episode dependency check over a standalone depth demonstration.
No new empirical block, model forward or runner; accounting remains unchanged.

## S50 concrete acquisition reference — 2026-09-14

Source study of a semantic/spatial merger identifies original-operation retention
measurements and a useful joint behavior, with teacher, prompt and resolution
confounds limiting attribution. The original spatial API accepts dense masks as
well as points; matching that guidance is expressible but untested in this study.
Canonical S50 retains final-paper custody, separately identified author appendices,
the pinned predictor interface and independent review provenance. This is a source
result, not a local empirical block. Six blocks, fifteen scored model forwards,
1065 apparatus /0 artifact-bearing lines remain unchanged. No reproduction selected.
The worked replay-source proposal finds that unlabeled images can support generated
joint-service targets only after selecting vocabulary/request support and a sampling
policy. The student must run its own path. Equal-budget ingredient replay is the
comparison; no training or new empirical block follows from this derivation.

## Mission-coverage consolidation — 2026-09-14

Current map and routing consolidated through S49; no new empirical block. Earlier
four negatives, two failed S37 attempts within its block, and S47's direct-method
success retain their scopes. Six blocks, fifteen scored model forwards,1065
apparatus /0 artifact-bearing code lines. Current notebook's stale958/five count
corrected. Useful contribution need not add novel information. The missing
acquisition episode must identify supported prior A, new B and retained joint use;
repeated queries over a saved export alone do not satisfy that requirement.

## S49 saved exports and later requests — 2026-09-14

Pinned source inspection finds that supplying a saved reconstruction export
bypasses fresh query encoding. A separate prepared context-compression interface
actually combines memory with a later prompt, supported by source training and
evaluation. Its multi-span result required extra training on the concatenation
pattern; it is not independent-donor composition. No local model run or acquired
artifact. Canonical S49 and two `s49_*_20260914` ledger entries retain code/PDF
hashes and independent reviewer provenance. Accounting remains unchanged.
Further S49 review retains consumer-response distillation as optional and adds a
complete trained-reasoning comparator with rationale-preserving no-alignment
control. Traced hidden targets depend on changing shared parameters. Official
small-checkpoint availability is confirmed, then closed without download or
execution after independent direction review; two additional S49 ledger records
preserve the receipt and disposition. No new theorem section or runner.

## S48 transported operation contract — 2026-09-14

Primary source and independent algebra separate a transported direction from the
complete read/write intervention. Affine transport, descent through a noninjective
export and clamp-order effects are diagnostic; the inspected method recalibrates
the recipient and does not promise exact conjugacy. A later operation must justify
its remaining recipient-specific preparation against direct learning. No new model
run or apparatus; local totals remain six probe blocks,15 scored model forwards
and1065 apparatus /0 artifact-bearing physical code lines. Canonical S48 and ledger
`s48_transported_operation_contract_20260914` carry provenance and limits.
Continuation: accumulating a full positive centroid during recipient clamp
calibration needs no extra model forwards. A declared reusable reference enables
a direct direction comparator on those positives; reference quality is unproven.
Readout reuse and state-update transfer have different preparation obligations.
Ledger `s48_shared_calibration_comparator_20260914`; no new empirical block.

## S47 real evidence selection and direct remedy — 2026-09-14

Fixed five report pairs, ten prompts; four pairs semantically usable with an exact
source-bound annotation per question. One ratio/denominator mismatch excluded
without replacement; one malformed offset retained alongside an exact alternative.
Fifteen of sixteen abstract annotation rows match exclusive-end indexing, including
duplicates. On eight eligible structured outcome queries, ordinary lexical matching
selects adequate evidence for7/8 across full abstracts and8/8 with supplied RESULTS
filtering; paired adequacy3/4 versus4/4. One mixed-quality top tie remains in the
RESULTS arm. Root and independent reviewer read every selected sentence and tie.
CPU-only subprocess0.3406s from cached sources, excluding preparation; no model.
Canonical storage/I-O replay preserves all16 decisions and input custody. Panel
closed without learned model, expansion or acquisition claim. One runner107 lines,
one result JSON. Totals: six local probe blocks,15 scored model forwards,1065
apparatus /0 artifact-bearing physical code lines; S43 external inspection separate.
See canonical S47 and ledger `s47_evidence_selection_null_result_20260914`.

## S46 direct operation teaching comparator — 2026-09-14

Two-round independent dialogue prioritizes direct conditional preference teaching
over full-answer representations for the specified retrieval endpoint. A contrast
dictionary otherwise hides an unproven semantic interpreter. Donor-written bounded
executable predicates remain a conditional alternative; no parser is built. Richer
exports may earn their cost through later-operation reuse, which retrieval alone
cannot show. Theory only; empirical totals unchanged. Ledger:
`s46_direct_preference_competitor_20260914`.

## S44-S45 conditional evidence selection — 2026-09-14

One source-reconfirmed worked retrieval request fixes the offered operation. A
prospective reversal contrast asks whether the same evidence is preferred
appropriately for different questions on unfamiliar records. Fresh donor answers
at inference are additional information, not an equal-information teaching arm.
Independent strategy dialogue rejected an uninformative four-record model check.
No model loaded, runner added or acquisition demonstrated; numerical thresholds
and an empirical panel remain unselected. See canonical S44-S45 and ledger
`s44_s45_conditional_evidence_selection_20260914`. Empirical totals unchanged.

## S43 bounded teaching-target inspection — 2026-09-14

Read 32 published responses for a fixed first16 identical queries; row-service
revision headers match dataset commit, ids/text match, no transport truncation.
Generation finish metadata is absent and many stored answers visibly stop early.
One complete counting correction is independently verified:136 versus76; shared
errors and partial answers make whole donor responses unreliable as gold. No
comparative accuracy, termination, acquisition or latent-transfer claim. Fixed slice
closed without expansion. One retained result JSON; no new runner/model inference.
Local totals remain five probe blocks and15 scored CPU forwards; this external-
output inspection is counted separately. Ledger: s43_teaching_target_slice_20260914.

## S42 response-derived teaching and computation reuse — 2026-09-14

Primary-source and independent strategy reviews distinguish answer generator,
embedding teacher and predictive backbone. A pooled retrieval vector differs from
a decodable soft-prompt sequence; latent chaining remains future work. Smaller
cache adapters need not make total fitting cheap. A causal-prefix reuse argument
is independently checked, with isolated-branch, frozen-policy and generation-
vocabulary qualifications. This supports an ordinary teaching comparator and
conditional shared-predictor idea, not a build or local acquisition claim. See
canonical S42 and ledger. Source/code reads only; empirical totals unchanged.

## S40–S41 separate preparation and joint use — 2026-09-14

Primary source and pinned implementation review find a constructive precedent for
joint use of separately trained donor interfaces. Parallel residual summation and
sequential fusion are distinct operators; neither freezes an entire prior receiver
trajectory. Released candidate interfaces exist, but exact joint-run configuration
and item-level retention remain unbound in the checked material. A frozen-query
attention identity is independently verified and retained as a diagnostic, not a
new instrumentation program. See canonical S40–S41 and matching ledger entries.
No local model run, training, weights download or new runner; totals unchanged:
15 scored CPU forwards, five probe blocks, 958 apparatus / 0 artifact-bearing lines.

## S39 live latent communication source check — 2026-09-14

Pinned source and independent strategy review separate same-model cache reuse
from heterogeneous contributor admission. The official core uses one model across
roles. Pinned community extensions implement actual heterogeneous transport.
A function-body versus complete-function prompt difference identifies the first
no-message receiver control; no causal explanation of author results is claimed.
Probability-mediated transport preserves a conditional message, not arbitrary
latent state. See canonical S39 and ledger. No new execution; totals unchanged.

## S38 strategy resumed — 2026-09-14

Latest user instruction keeps mission fixed and pivots around unavailable tooling.
No S37 rerun or CLI repair loop. Conditional transcript simulation sharpened by
independent review; teacher target construction and student training pathway are
separate acquisition choices. Primary-source checks support an ordinary text-
counterfactual comparator and qualify random-network nonlinear alignment through
a disjoint-name test. Symbolic-teacher intervention training supplies a student-
pathway precedent; ordinary augmentation already solves one behavioral endpoint.
A conditional factorization argument identifies when both training paths coincide.
See canonical S38 and ledger `s38_*` entries.
No new model forwards or apparatus: totals remain 15 scored CPU forwards, five
probe blocks, 958 apparatus / 0 artifact-bearing lines, no GPU or training.


## S37 execution outcome — 2026-09-14

Two bounded attempts, **zero scored forwards**, no scientific result. Attempt 1
hit the 180 s compute deadline (200.42 s total, peak RSS 592,384,000 bytes).
Attempt 2 reached alignment in 84.57 s (peak RSS 734,183,424 bytes), then failed
because the downloader saved HTTP-compressed metadata without decoding it.
Seven gzip files and one Brotli configuration were normalized offline; decoded
tokenizers are byte-identical. This is a harness error, not incompatible models.
Downloader corrected; no third run in this allocation. Exact raw attempts, hashes,
diagnosis and cache correction are in ledger entries
`agreement_teaching_preflight_attempt1_20260914` and
`agreement_teaching_preflight_attempt2_20260914`. Two real CLI audit attempts remain incomplete because file reads failed; separate
collaboration custody checks verified counts, reconstructed sources and normalized
metadata, not a scientific result or independently replayed decompression.

Accounting: five probe blocks (two execution attempts in S37), 15 total scored
CPU forwards, 958 apparatus / 0 artifact-bearing lines, zero building rounds,
no GPU or training. Scientific competence/headroom/intervention claims untested.

## S37 preflight locked — 2026-09-14

User-requested renewed reviews demote S36 query compatibility as first priority.
One fixed CPU-only agreement intervention preflight is justified after actual-file
design review and resource/alignment/checkpoint corrections. No training or
generation. Lock, receipts and hashes: `agreement_teaching_preflight_lock_20260914`.
Execution failures are inconclusive. Result pending; historical selections below
are superseded by this reassessment.

## Application selected — 2026-09-14, 03:42 UTC

Two fresh dialogues select S36's scientific evidence-finding use and initial
query-side compatibility hypothesis, resolving the map audit's selection gap.
Mechanism-testing and contrary evidence are the intended new behavior; established
topic searches must remain adequate. Expansion, rebuilding and no acquisition are
real outcomes. No local collection/model pair/run is claimed ready. Strategy selection
is complete; empirical adequacy and value remain open. Receipts and hashes:
`application_dialogue_r1_20260914` / `application_dialogue_r2_20260914`.
No additional local forwards, GPU or training; four probes and 15 CPU forwards total.

## Strategy map review — 2026-09-14, 03:34 UTC

Fresh unprimed CLI inspected consolidated 7d0a1bf after independent entropy review.
Continue; the general map is sufficient, but consequential use and preferred route
remain unselected. More definitions are not the next step. S36 works one application
choice. Corrected mandatory prior-preparation, cost-superiority and already-successful
direct-route wording across current surfaces. No new numerical correction in inspected
receipts, not a full numerical reproduction. Audit e1095–e1096; root synthesis complete.
Ledger `strategy_map_adversarial_review_20260914`. Next review around 05:34 UTC;
accounting remains four probes, 15 CPU forwards, 628:0 code lines, no GPU/training.

## Inheritance framing complete — 2026-09-14

Two fresh direction dialogues close S35's paper analysis. A retained fitting choice
is reconstructible from declared historical evidence; its usefulness after a new
provider still requires quality/cost evidence. This is a legitimate cheap baseline,
not a new admission-learning algorithm or selected experiment. The three empirical
uncertainties are useful contribution, retained interpretation/repair and total
effort. Stronger learning-to-admit is optional. No model calls or new runner.
Receipts: `inheritance_dialogue_r1_20260914` and `inheritance_dialogue_r2_20260914`.

## Candidate proposal decision — 2026-09-14, deferred

S34 now specifies 32 possible fixed candidates, accessible generator/object/scene
interfaces, supported truncation control and resource ceilings; no weights acquired
or models loaded. Design round 1 added scene-swap and uniform-selection controls,
paired object spoils, practical primary thresholds and a 45-minute total-effort cap.
Fresh round 2 and root defer execution: uncertain endpoint/compatibility effort
outweighs a narrow interface result with no concrete next use. An authenticated source-image reader
opened the image but mislocalized colors: file access works, endpoint reliability
is unproved. Root and an independent reader cross-checked the figure. This separate
source-image call adds no local probe forwards. Receipts and hashes are in
`candidate_proposal_dialogue_r1_20260914`, `candidate_proposal_dialogue_r2_20260914`
and `source_image_judge_readiness_20260914`.

## Fourth periodic approach review — 2026-09-14, 02:40 UTC

Fresh real CLI reviewed consolidated e72f83d after a separate entropy sweep:
continue, revise evidence allocation. Prepare one concrete bounded candidate-data
proposal with accessible interfaces, independent coverage/selection judgments,
direct controls and resource ceilings; no execution. Low coverage does not choose
guidance over other remedies; provider replacement differs from capability growth.
Corrections propagated to canonical/current surfaces. Audit e1086-e1087, root
synthesis complete; `strategy_periodic_approach_review_4_20260914` preserves receipt.
No numerical correction or new local behavioral result. Accounting remains four
probes, 15 CPU forwards, 628:0 code lines. Next review around 04:40 UTC.

## Third periodic approach review — 2026-09-14, 00:54 UTC

Fresh real CLI inspected consolidated 89b34ae and recommends continuing while
ending further interface cautions as the default. S27 develops one <=45-minute
constructive refresh-versus-maintenance episode, with full refresh as a serious
reference and all consumption/decision learning charged. Root challenges whether
generic caching would repeat the closed S24 contract. Two direction responses
now close S27 through sufficient typed connected refresh and adopt S28's reusable
consumer target. Targeted source inspection and S28.1's incremental-reader algebra
are complete; neither is a new numerical experiment.
Audit e1061-e1062; exact receipt/hash in ledger
`strategy_periodic_approach_review_3_20260914`. No new experiment or model call.
Accounting remains 628 apparatus / 0 artifact-bearing code lines, four probes /
zero building rounds, 15 model forwards. Next periodic review around 02:54 UTC.

## Reusable operations and grafting — 2026-09-14, source/strategy only

Canonical S29 works a source/operation reuse contract. S30 checks two primary
heterogeneous fusion mechanisms and a pinned code release with independent readers.
Positive interface reuse and folded weight transfer remain distinct from sequential
acquisition. Learning, retained tails, conditional normalization and folding scopes
are recorded in the canonical document and `heterogeneous_grafting_source_20260914`.
No experiment, model loading or new runner.

## Retained interpretation and migration — 2026-09-14, source/strategy only

Three fresh direction dialogues refine S31's future-operation teaching contract
and S32's provider-replacement case. Independently checked published migration
tables show why the currently deployed query/gallery pair is the retention
reference. No model or economic retirement decision was executed. Source versions,
reported points and independent scope checks live in canonical S32 and ledger
`migration_retirement_contract_20260914`. Source study adds no empirical probe.

## Intervention teaching contract — 2026-09-14, source/strategy only

Canonical S33 records a primary causal-distillation recipe independently checked
against its final paper, plus two fresh direction dialogues. Each model intervenes
on its own states; the reusable teaching episode still needs donor-specific
correspondence. S33.1 separates auxiliary teaching from promised operations.
Positive published task scores do not establish superiority over targeted ordinary
teaching at matched total expenditure. Ledger `causal_distillation_source_20260914`
and `teaching_retention_contract_20260914` preserve scope. No model or new probe.

## Shared-candidate inference lifecycle — 2026-09-14, source/strategy only

Canonical S34 records an independently checked compositional inference account
and three fresh direction rounds. The candidate-reader lifecycle
dependency/cost account favors completed-candidate coverage and clean evaluation
before trajectory guidance. A pinned source-repository inspection identifies no
eligible candidate bank in its limited scope.
Published prepared composition does not establish the proposed new contribution.
Ledger `compositional_inference_source_20260914` and dialogue receipts preserve
scope. Accounting remains four probes / 15 CPU forwards / 628:0 code lines.
No model, numerical experiment or implementation.

## Continuation and growth contracts — 2026-09-14, mathematics/source only

Canonical S8.1 gives a conditional mean-state closure proof and separates
fixed-action prediction from observation-then-choice. DeepMDP and successor-feature
primary sources supply existing task-relative representation and transfer contracts.
S8.2 gives an exact counterexample to monotonicity of recomputed one-pass GPI
under policy-library inclusion; its standard constituent-policy theorem remains
intact. Retaining and evaluating the previous composed policy restores conditional
retention bounds. Fresh real CLI audit e1056-e1057 checked the witness and scope.
No numerical benchmark, learned-model failure, new latent law or compute run.
Receipts: ledger `mean_state_closure_20260914`, `deepmdp_contract_source_20260914`
and `gpi_growth_math_audit_20260914`. Accounting remains four probes, 15 model
forwards, 628 apparatus / 0 artifact-bearing code lines.

## Autonomous conditional continuation — audited local closure, 2026-09-13

Canonical S26 specifies one bounded CPU strategy diagnostic after two fresh real
CLI direction rounds. All retained variant rows consistently reconstruct four
assay reference constructs (lengths 47/69/64/50); no fitness scores select pairs.
Sixteen fixed pairs test independent, autonomous, reverse, repeat, two oracle and
cyclic-content conditions. Confidence-first scheduling uses no extra forwards.
Installed mask-dropout normalization requires the equal-mask control. Seven
forwards / 112 rows maximum, 180 seconds including startup; timeout is incomplete.
Recovery thresholds are local allocation criteria, not significance or general
reasoning evidence. The single run completed in 10.1455 seconds, seven CPU
forwards / 112 rows. Primary both-site recovery 0/16 matches independent 0/16;
second-site recovery 2/16 matches independent 2/16 and trails cyclic control
3/16. All 16 fixed-position cases complete per arm. Initial first guesses are
correct only 1/16, bounding achievable pair recovery below the locked threshold.
Close this procedure locally; no strong negative about competent composition.
Fresh real CLI audit e1044-e1045 independently reproduced every count and checked
source/custody/code paths; root and auditor read all 16 decoded predictions.
Reverse/confidence-first each correct one and spoil one. Timing is from main,
excluding interpreter startup; raw audit runtime typo is corrected in the ledger. Exact arrays
and provenance are in `results/encoder_continuation_v1/result.json`. Ledger `autonomous_continuation_design_20260913`.


## Second periodic approach review — 2026-09-13, 23:03 UTC

Fresh real CLI review of `f83ce4b` recommends continuing strategy but ending
unspecified revision prediction and generic vocabulary expansion as the default.
Its decision question is what acquisition or use improves because a revision
relation is learned. Next block: at most 45 minutes, one concrete operation,
donor evidence, receiver interface and retained use; conclude typed import is
sufficient, specify a decision-bearing empirical contrast, or close on an exact
missing dependency. No competence proof is required before a probe measuring it.
This is an approach review, not a new experiment. Audit entries e1039-e1040;
receipt and hash: ledger `strategy_periodic_approach_review_2_20260913`.
The subsequent bounded contract closed with outcome 1: S24's authoritative
scoped external-record import and stipulated calculator supply the useful
operation. No learned predictor is needed there; no real purchasing donor or
receiver is asserted. Independent strategy check agreed. Mission e1041 and ledger
`scoped_revision_contract_closeout_20260913` retain the scoped result.
Accounting at that checkpoint was 421 apparatus / 0 artifact-bearing lines, three probes /
zero construction rounds, eight model forwards, no GPU or training.


## Revision consequence identification (2026-09-13; no compute)

Two fresh real Codex CLI design-dialogue rounds produced S24's hypothetical
purchasing example and corrected conflation of dependency with observed answer
change. The example identifies a missing scope distinction and keeps acquisition
evidence separate from later consequence tests. It instantiates prior information
arguments and does not establish a learned law. Typed repair is sufficient under
the stipulated two-hypothesis contract. A future learned-structure claim must
predict actual model behavior without caller-supplied propagation supplying it.
No new model call, runner, benchmark, artifact or selected run. Dialogue receipts
are in the ledger; this does not replace the scheduled periodic approach review.
The existing Geiger2021 causal-abstraction framework now supplies the explicit
method bridge in S24, with its task/phrase supervision and sampled scope qualified.

## Acquisition reuse and revision precedents (2026-09-13; strategy only)

Canonical S2.2 specifies a reusable acquisition procedure whose learned outputs may
change, with declared evidence/budgets and fresh-input use. S12 adds conditional
tied-head parameter bookkeeping; no proprietary model or measured runtime claim.
S25 records full-method source readings of MEMOIR v4, RippleEdits TACL2024 and
StableEdit v1. These establish relevant published approaches and task contracts,
not project replications. Independent reviews distinguish sequential target learning
from donor-computation transfer and qualify the semantics of dependency rules.
No selected experiment, new runner, model load or training; accounting unchanged.

## Source interpretation and interface repair (2026-09-13; no run)

S3 now incorporates the full official stitching paper: its clustered noise has
shared class centres across train/test, and stitch supervision can learn output
correspondence. Functional acquisition and preserved donor computation are distinct
claims. S24 traces crop-local mutation and missing inverse-coordinate metadata in
the pinned table pipeline. These create caller obligations for a persistent page
contract, not evidence of a defective latent space or a selected wrapper project.

README's old "content-driven, not surface-driven" wording is narrowed to the
measured assertion-versus-misleading and variable-mention contrasts. The 6.3×
number is unchanged; this does not isolate semantics from all surface differences.
The historical Gate 1b interpretation below is qualified accordingly. No new
model run, training or apparatus; session accounting is unchanged.

## Worked growth episode: source inspection only (2026-09-13)

Canonical S24 uses a released table illustration to assign successive donor
contributions and the source of joint-use relationships. OCR supplies text;
structure predictions and fixed binding rules supply cells; an existing executor
supplies later queries. Two-round design dialogue and independent code reading
corrected the attribution of binding and strengthened the structured comparator.
No OCR/table pipeline, training or benchmark was run. Metadata confirms exposed
component resources only, not task competence. Source image hash and code/model
revisions are in the ledger. Session model/probe accounting remains unchanged.

## Specialist task contract: documented case is not discriminating (2026-09-13)

Source inspection, not a new model experiment. At pinned public data revision
9f0ef941a362f0e308e11861ed82c36338e23586, the documented PFN1 test example
(row 125) has the exact question and answer of four training rows. Independent
verification checked sequence edits, question equality and source hashes. A
question dictionary answers this example without DNA; no full-benchmark score
or trained-model attribution follows. The curation formatter omits nucleotide
changes from questions, and the evaluation loader combines 146 test and 144
validation rows. The latter explains the paper's 290-example table without
establishing historical model-selection details.

Close this candidate for the proposed joint-contribution demonstration: missing
independent effect contrasts, receiver role-competence evidence and a verified
usable original receiver. No new harness, model load or reproduction. The
[canonical S8 contract](../theory/CUMULATIVE_CAPABILITY_ACQUISITION.md#s8-what-reasoning-over-an-encoder-could-mean)
and ledger retain the exact scope, row locators and hashes. Probe accounting is
unchanged; bounded source reads are not additional model experiments.

## Periodic strategy review: task-first selection (2026-09-13; no run)

Fresh real Codex CLI review of HEAD 0a93fb7, audit e1027, finds diminishing
returns from availability-led diagnostics. Continue strategy with one documented
specialist-encoder joint-task eligibility block, at most 90 minutes; compare
structured and richer ports without building either. Require competence in each
assigned role, not standalone superiority for all donors. Receiver adaptation
must be declared rather than universally banned. S22 stays closed; its failed
average does not bar a separately justified richer interface. See the
[canonical next-work boundary](../theory/CUMULATIVE_CAPABILITY_ACQUISITION.md#next-work-boundary).
No new empirical result. Session accounting remains 421/0 apparatus/artifact
lines, three probes/zero building rounds, eight CPU model forwards.

## Cached donor output screen: fixed rule loses (2026-09-13; AUDITED)

S22 prespecified four assays, two columns and an equal within-assay midrank
average before outcome calculation. [Runner](run_cached_donor_screen.py),
[result and subset provenance](results/cached_donor_screen_v1/result.json).
All 8,610 variants share finite identity-matched support; no exclusions. Initial
screen 23.86 seconds including bounded archive retrieval, zero model calls or
training. Corrected combination correlations: -0.205586, 0.313539, 0.238012,
-0.301905; each is below the retrospectively better individual score. Individual
signed correlations match the official per-assay summaries to reported precision.

Fresh real Codex CLI audit e1022 upheld the bounded negative and caught split
mathematical ties from floating normalization. Integer doubled-midrank sums
correct the largest correlation by 0.00001343, without changing the verdict.
Offline correction matches independent SciPy recomputation. Original source and
result remain in commit 0f61942, hashes retained in the corrected result/ledger.
Source agreement is the recorded runner check; audit independently checked the
retained subsets, not discarded full archive members. No tuning or new run.
This is failure of one fixed rule on four descriptive units, not proof of no
complementarity, internal transfer limits, generalization or biological utility.
Session apparatus/artifact: 421/0 lines, three probes, zero construction. No
further averaging/grafting of this pair is selected.

## Conditional-port compatibility: bounded panel (2026-09-13; AUDITED)

[S21](../theory/CUMULATIVE_CAPABILITY_ACQUISITION.md) locked four real-encoder
conditional rectangles after two independent design-dialogue rounds.
[Runner](run_conditional_port_probe.py), [result](results/conditional_port_v1/result.json).
Eight CPU forward calls, 32 rows, no fitting, 60.27 seconds including startup.
Isolated minimum worst-edge corrections range 0.000207–0.028143 nats; all four
are below primary 0.1 and sensitivity 0.05/0.2 tolerances. Repeat/shared-input
checks agree exactly, which does not bound systematic numerical error.
Audited scope: no isolated incompatibility exceeding declared tolerances;
no global consistency, reasoning, biological quality or acquisition claim.
Fresh real Codex CLI audit e1021 upheld conditioning/arithmetic and identified
a post-hoc additive null: one constant preference per site fits all observed
edges within 0.0722242 nats, below primary tolerance. The fixed-reference null
instead has maximum error 0.1444484. These are calculations on recorded data,
not prospective held-out predictors. Practical significance remains unvalidated.
No expanded search selected; next define a task requiring useful updates.
Session apparatus/artifact: 217/0 code lines, two probe rounds, zero construction.

## Strategy checkpoint: contribution and inference contracts (2026-09-13; NO MODEL RUN)

Canonical S16–S20 record analytical examples, primary-source reading and bounded
external artifact eligibility checks. Pinned onboarding source lacks a matched
receiver baseline and replacement-contribution controls. Three inspected result
JSONs have no per-example messages/labels for an alternative-reader replay.
No external script, model or tensor was loaded; these are source assessments,
not a reproduction or a new model result. Added the current decision map and
narrowed next work to at most two actual interfaces. Session apparatus/artifact
remains 98/0 lines; one integrity probe, zero construction rounds. Explicit
strategy-only mandate governs. References and precise scope live in S18.

## Strategy refined: interchange and extension (2026-09-13; NO MODEL RUN)

Canonical strategy S11–S15 records operational port contracts, conditional bounds
and independent approach dialogues. Scene/assignment runs were not selected;
structured interfaces remain serious alternatives. No further compute or
capability result. Session apparatus/artifact remains 98/0 lines, one integrity
probe and zero construction rounds; the explicit strategy-only mandate governs.
Integrity audit references corrected to e1011–e1012 after a stale-writer collision;
the append-only ledger preserves the correction and root now serializes writes.

## Acquisition cache eligibility: retained exports unsuitable (2026-09-13)

Small read-only CPU probe of two legacy 55 MB latent bundles; no models loaded,
no fitting, no capability evaluation. [Runner](run_acquisition_probe.py),
[result](results/acquisition_cache_integrity_v1/result.json), predeclared question
and interpretation in canonical strategy notebook S6. Unsuffixed bundle has
2,906 wholly nonfinite query rows and 13,464/13,464 wholly nonfinite plan rows.
The suffixed bundle is finite, but 13,386/13,464 plan rows are the zero vector.
Both score arrays match byte-for-byte. Missing row IDs/provenance prevent a
strong pairing claim. These exports are ineligible for the proposed cross-model
plan-transfer probe; this says nothing about the underlying models' capabilities.
Sources preserved. Independent bit-level/schema audit e1011–e1012 upheld the
scoped finding. Runner pins reviewed metadata/shape/byteorder; final scan ~1.23s.

## Strategy discovery opened (2026-09-13; NO MODEL RUN)

User authorized sustained autonomous strategy research at 19:45 UTC, including
bounded informative CPU probes. Initial approach review separates contribution
from deployment; donor removal is a branch, not a mandatory ladder. Definitions,
an analytical access-ambiguity counterexample, extension obligations and primary
source reading live in the canonical direction's strategy notebook. No empirical
cross-model result, training or model calls. Initial added code apparatus/artifact
is 0/0 (undefined ratio); theory/reading blocks are strategy work, not construction.

## Direction update — cumulative capability acquisition (2026-09-13; NO RUN)

User-directed documentation update, not an experimental result. The enduring
native-mathematics philosophy is preserved; the current exploration direction
is [cumulative capability acquisition](../theory/CUMULATIVE_CAPABILITY_ACQUISITION.md).
Initial progression proposed A, acquire B, retain A, compose A+B, donor removal
and further donors. The subsequent strategy review makes donor removal a separate
assimilation branch. No donors, tasks, thresholds or runs are selected. Earlier
experiments remain historical supporting evidence, and old next-step language
does not override [current state](../STATE.md).

Reverse chronological. Only gate-passed conclusions are stated as confirmed.
Program opened 2026-08-27; prior program's log is at `legacy/experiments/EXPERIMENTS.md`.

---

## Block 3 reading: length-only rival conditionally rejected at the 11-bin channel; the equal-length-comment reference scale is confounded (2026-09-06, ZERO-CALL READING, evidence gate REVISE)

Distance from claim: reading of existing audited evidence; no model calls.
Source: `results/svb_qwen3_composition_v2_multifiller/result.json`
(Qwen3-1.7B-Base, float32 CPU, 2026-09-04; sha256 898aff900864d3d2…).
Reading script `internal/q5_reading_multifiller.py` (git-ignored; sha256
e21aac988fcedd86…); exact rows in `internal/q5_reading_multifiller_out.json`.

**Interface of the archived table.** 24 roots (depth 4; x, y, z; values
1–8), surface A `# {var} is unchanged.` (5 tokens) and surface M
`# Reassigning {var} now.` (7 tokens), three independent filler pairs
(F_A: 5 tokens; F_M: 7 tokens; Qwen3 tokenizer, verified locally), eight
two-suffix arms per root: AM, MA, AF_M, F_MA, MF_A, F_AM, F_AF_M, F_MF_A.
Token counts and hence positions are matched by construction. Channel:
11-bin reduced law. No single-suffix arms, so the initial closeness of the
input pairs is unknown; numerical floor assumed (deterministic replay was
recorded for this stack in other runs, not in this table).

**The registered length-only rival (CR-15.10.6):** on identical input
pairs, token-length- and position-matched H and filler F produce successor
discrepancies differing by at most ε_R = 0.002. Read here with input pair
(r·A, r·F_A) and successors M versus F_M (order 1), and input pair
(r·M, r·F_M) with successors A versus F_A (order 2):

| Filler pair | Order 1: roots with |b_M − b_F| > 0.002 | median b_M / b_F | sign of b_M − b_F (positive roots) | Order 2: roots with |b_A − b_FA| > 0.002 | median b_A / b_FA |
|---|---|---|---|---|---|
| pair1 | 24/24 | 0.136 / 0.119 | 21/24 | 24/24 | 0.070 / 0.072 |
| pair2 | 24/24 | 0.034 / 0.033 | 8/24 | 24/24 | 0.077 / 0.052 |
| pair3 | 23/24 | 0.041 / 0.016 | 24/24 | 24/24 | 0.069 / 0.134 |

**Reference scale: between-filler spread (evidence gate: CONFOUNDED).** For
each root, the three filler PAIRS give three values B(i,i) of the filler-
successor discrepancy; each pair changes both the input pair's filler_A (i)
and the successor filler_M. Their range mixes input-pair variation with
successor variation (B(i,i) − B(k,k) = [B(i,i) − B(k,i)] + [B(k,i) − B(k,k)])
and does not isolate successor-filler variation; the needed off-diagonal
endpoints B(i,j), i ≠ j, are absent from the archive. The numbers below are
therefore a mixed spread, not the scale of "an arbitrary equal-length
comment on a fixed input".

| Order | median between-filler spread | median content-vs-filler deviation (pair1, pair2, pair3) | roots where content deviation exceeds the filler spread |
|---|---|---|---|
| 1 (successor M vs F_M) | 0.0975 | 0.0167, 0.0263, 0.0272 | 0, 0, 1 of 24 |
| 2 (successor A vs F_A) | 0.0840 | 0.0247, 0.0329, 0.0689 | 0, 2, 6 of 24 |

**Licensed sentence (evidence gate):** on the archived 24-root panel, using
11-bin immediate-read TV and assuming an adequate numerical floor, H and its
matched successor filler violate the 0.002 length-only prediction on 24,
24, and 23 roots across three filler/input-pair variants; this conditionally
rejects the restricted channel-specific rival (counts survive even the
largest admissible guard, threshold 0.0022), but does not determine whether
H is exceptional among equal-length successors. Under the locked interface
the unqualified verdict is INCONCLUSIVE (floor unmeasured; revisions
unpinned; single-suffix endpoints absent, so initial closeness unknown;
successor W_H profiles incomplete; reduced-TV differences do not bound
full-law differences; the affine rival cannot be tested without initial
d_i). Order 2 compares surface A, which is neither the registered H nor N.
The source preregistered the residual-order contrast, not this reading.
Withdrawn: "H is not distinguishable from an arbitrary equal-length
comment"; "content-specific H is disfavored"; "fillers vary more on
identical inputs".

**What they do not support.** Initial-pair closeness (unknown; no single-
suffix arms), so no tolerance-preservation or witness verdict; full-law
statements (differences of reduced TVs do not bound full-law differences);
any mechanism; any claim about the neutral comment N (not in this table).

**Program decision (evidence gate: AMENDED).** For any future panel: use
several response-independently selected SUCCESSOR fillers on each identical
input pair (fixed input, varied successor), with a prospectively specified
comparison; three is a practical minimum, not a threshold this archive
establishes, and hand-chosen fillers are not independent population
samples. Generic readout sensitivity remains unexcluded, not confirmed. The
deferred panel's discrimination against that rival is unestablished (no
calibrated rival envelope, no reserved disagreement set). Decision-relevant:
it changes which controls must be acquired (the missing fixed-input filler
control) and prevents a misleading interpretation.

**Ruling (evidence gate):** stop zero-call rotation pending compute approval;
the one prioritized acquisition is a small, time-capped CPU position-matched,
full-law, eligibility-first panel on repaired strong-assert pairs spanning
at least five task blocks: first measure empty and H profiles and
reproducibility; if five robust W_H-close pairs exist, acquire the N/H
successor profiles and three matched successor fillers on those same pairs;
if coverage fails, stop without relaxing the threshold. Decision it answers:
do genuinely close real-model histories exist here, and which matched
continuations separate them?

---

## Q4 reading: tolerance preservation under the neutral and heavy comments on the archived congruence panel (2026-09-06, ZERO-CALL READING, evidence gate REVISE)

Distance from claim: reading of existing audited evidence; no model calls.
Source: `results/svb_qwen3_congruence/result.json` (Qwen3-1.7B-Base,
float32 CPU, 2026-09-04; sha256 54d2b20afdf496db…). Reading script
`internal/q4_reading_congruence.py` (git-ignored; sha256 806971037a766d7d…);
exact rows in `internal/q4_reading_congruence_out.json`.

**Interface of the archived table (its original limits apply).** 24 cells
(depth 4; variables x, y, z; outer values 1–8), three presentation pairs
(a, a′) of the same semantic role, actions c0 = neutral comment
`# No changes.` and c1 = heavy comment `# Reassigning {var} now.`, in
forward order (surface then composer). Channel: the 11-bin reduced law
(digits 0–9 + OTHER), NOT the full next-token law. Reduced-channel TV is a
lower bound on full-law TV: a split at 11 bins establishes post-action
FULL-LAW SEPARATION, but not a counterexample to full-law tolerance
preservation (initial full-law closeness is unknown); closeness at 11 bins
does not imply full-law closeness. τ = 0.01 (the
table's eps_eq); guard 1e-6 ASSUMED (this table has no replay or cached-
versus-full data; CR-15.10.6 requires a measured floor; all counts survive
at the minimum locked guard 2e-6).

CR-15.10 quantities per pair: D0 = TV(a, a′) after the surface (the W_0
question); DN = TV(a·c0, a′·c0); DH = TV(a·c1, a′·c1). A pair is W_0-close
if D0 ≤ τ − guard; a witness against tolerance preservation under action
c is a close pair with D_c > τ + guard.

| Pair type | W_0-close cells / 24 | Under N: witnesses, max DN on close | Under H: witnesses, max DH on close |
|---|---|---|---|
| strong_assert (`# State unchanged.` vs `# {var} is unchanged.`) | 24 (all x, y, z) | 0, 0.0064 | 23, 0.1162 |
| moderate_assert (`# Values preserved.` vs `# {var} is still intact.`) | 8 | 0, 0.0100 | 6, 0.0705 |
| misleading (`# Changing {var}.` vs `# Updating to a new value.`) | 1 | 1, 0.0678 | 1, 0.1180 |

Conditional numerical classifications at the archived 11-bin interface
(evidence gate: under the actual locked CR-15.10.6 interface the verdict is
INCONCLUSIVE, because the two surfaces of each pair have different token
counts, 4/5, 4/6, 4/7, so composer and query positions are unmatched, and
the numerical floor is unmeasured; the archive also lacks pinned tokenizer
and model revisions and token IDs):
- strong_assert, N: preservation on 24 eligible task blocks (max DN
  0.0064 < 0.01). Reduced channel, unmatched positions, assumed guard.
- strong_assert, H: **WITNESSED** (23 of 24 close pairs split; DH up to
  0.116; the one non-witness is d4_x_1; d4_z_1 splits marginally at
  DH = 0.0121). Post-H full-law separation established; initial full-law
  closeness unknown.
- moderate_assert, N: preservation on 8 eligible task blocks (max DN
  0.009990811, i.e. 9.19e-6 below τ, 7.19e-6 after the minimum locked guard:
  sensitive to the eventual measured floor; not "within the guard").
- moderate_assert, H: WITNESSED (6 of 8).
- misleading: 1 close pair (d4_z_1), split under both N (0.068) and H
  (0.118): WITNESSED for both actions; coverage 1, so no preservation
  statement is possible for this pair type.

**Licensed sentence (evidence gate):** on the archived unequal-position
prompt panel, using 11-bin TV and an assumed numerical guard, N keeps all
32 initially close assert-type pairs below 0.01, while H separates 29; this
supports candidate selection, not a matched-position full-law preservation
verdict. Competence: on all 24 baselines and inspected roots the correct
digit exceeds every other bin including OTHER (correct full-vocabulary top-
one reads for those stored laws), which does not complete the deferred
competence protocol. Conventional medians (strong_assert): DN 0.00449, DH
0.0821 (the script's original median was the upper-middle value; fixed).

**What they do not support:** full-law preservation under N (reduced
channel); anything about carrier erasure; global moduli; any mechanism;
reverse-order (composer-first) questions, which are a different family and
were not read here; length-matched filler comparisons (the multifiller
table uses different arms and was not read).

**Program decision (evidence gate: AMENDED):** prioritize strong-assert as
the candidate family for a NEW position-matched, full-law eligibility
panel; retain moderate-assert as a secondary boundary diagnostic; treat
N-preservation and H-separation as W_0-level hypotheses. Do NOT replace the
deferred panel: CRC-0 lacked W_H-close pairs, while Q4 supplies W_0-close
pairs, a different eligibility criterion. At the minimum locked guard this
archive supplies only three W_H-close pairs (strong d4_x_1; moderate
d4_x_3, d4_z_1) over two task blocks, below the five-block requirement, and
the subsequent N-then-H and H-then-H endpoints are absent. A full-law
acquisition could eliminate initial eligibility altogether. Q4 is decision-
relevant (candidate prioritization), not a discriminating experiment.
Never say: N preserves full-law tolerance; H refutes full-law preservation;
replay-validated or matched-position witnesses; exact closure refuted;
presentation information erased; forgetting safe indefinitely; H-specific
semantics; these roots solve the locked W_H coverage requirement.

---

## CRC-0 witness reading: tolerance preservation under the heavy comment rejected on two pairs (2026-09-05, ZERO-CALL READING)

Distance from claim: reading of existing evidence; no new run.

Question (CR-15.10): is the immediate-response tolerance τ = 0.01 preserved
under the heavy comment H ("# Reassigning {var} now.") on CRC-0's panel?
The CRC-0 artifact (`results/crc_0/result.json`) stores only summaries (the
runner computes but does not save the 36 × 13 response laws), so the
enlarged-family closure test cannot be run from it. A one-sided witness
check on the retained summaries is possible: `q0.all_cliques=true` proves
every Q_0 split-event pair had d_0 < 0.01; a same-variable event with
`worst_action="H"` and post-action full-law TV > 0.01 rejects preservation.

| Pair | Recorded input bound | Recorded post-H full-law TV |
|---|---:|---:|
| d4_y_3_P0 vs d4_y_3_P1 | < 0.01 | 0.0280 |
| d4_y_5_P0 vs d4_y_5_P1 | < 0.01 | 0.0271 |

**Licensed conclusion (Codex):** CRC-0's recorded full-law witnesses violate
preservation of the 0.01 immediate-response tolerance under the heavy
comment on two same-variable presentation pairs. This is a reading of
existing numerical evidence, not a newly run test, and not a refutation of
exact profile closure. Never say: "the immediate-read family is closed under
the neutral comment"; "the model erased the presentation difference";
estimated global Lipschitz constants for N or H. Cheapest rival: ordinary
lexical/position-dependent readout sensitivity. Design of the enlarged-
family test \(W_H=\{(\epsilon,\mathrm{read}),(H,\mathrm{read})\}\) and its
decision rule: theory/CONTINUATION_REFINEMENT.md CR-15.10.5.

---

## Commit-Seal Certificate: INCONCLUSIVE; post-hoc contraction-premise failure on the fixed cube (2026-09-05, evidence gate REVISE)

Distance from claim: conditional (audit #39). Compute: one CPU retrain (130 s) + 0.3 s certificate + seconds of diagnostic. Design gate: analytic round 4; eligibility ruled by audit #40.

Question: does a TRAINED rung-1 sealed update cell (17,195 params) satisfy the
CR-15.3 contraction certificate on a pre-fixed domain, giving a proved
history-uniform overwrite bound for a learned artifact?

Protocol (fixed before the run): seed 11, sealed arm, identical training to
rung 1; weights saved inside result.json before certification; domain
D = [-H,H]^16 with H = max(1, ||m0||_inf) = 1; infinity norm; 128-bit
outward interval arithmetic (mpmath.iv); no seed/norm/domain search.

| Step | Result |
|---|---|
| 1 interface | write(0), write(1), read; bypass zero; proposals enclosed to width 3e-38 |
| 2 domain | H = 1.0 (||m0||_inf = 0.22) |
| 3 invariance U_v(D) ⊆ D | PASS (gates in [0,1], candidate in [-1,1], hull in box) |
| 4 contraction | Reported interval-Jacobian estimate 5.88; no contraction certificate. The row-maximum implementation required endpoint-safe comparison (fixed after the run, not rerun). |
| 5 fixed points | After 64 float32 iterations, reported interval residual bounds approximately 3.9e-8 and 2.7e-8. No certified location or uniqueness of a nearby fixed point. |
| 6 | Initialization and tube closure skipped: contraction was not established. |
| 7 reader | Reported reader estimate 78.3; not accepted as a certified bound because fixed offsets were computed outside interval arithmetic (fixed after the run, not rerun). |
| 8–9 | Calibration and usefulness skipped. The saved beta_hat=0.0 is a placeholder. No history-uniform response bound established. |

Diagnostics (not certificate): two-write accuracy 1.000; paired erasure max
TV 1.4e-8 on the panel.

The registered certificate remains CERTIFICATE_INCONCLUSIVE. Separately, the
post-hoc search reports interval-confirmed expansion for both writes on
D=[-1,1]^16 in the infinity norm. The confirmation logic is sound and
supports failure of that specific contraction premise. The reported 1.11 and
1.17 ratios are float32 search statistics. Expansion does not establish
tightness of the global interval estimate. Exact witness coordinates and
witness-specific provenance were not saved in this run (the runner now saves
them).

On the eight factorial training cells, recorded two-write accuracy is 1.000
and maximum immediate-reader TV across obsolete-first-write counterfactuals
is approximately 1.4e-8. These are finite-panel floating-point diagnostics,
not evidence of carrier erasure or arbitrary-history fidelity. Jacobian norms
0.987 and 0.539 concern 64-step numerical candidates; they do not certify
fixed-point neighborhoods or switching between values. No behavioral
overwrite failure is established.

**What it does not license:** any statement about address selection, two
registers, unseen wording, seal advantage, any uniform guarantee for
floating-point inference, or a pretrained model.

**Evidence gate (Codex): REVISE wording; no follow-up run on this cell.**
Apparatus/model line ratio 451:79 = 5.71:1 in this lane; the certificate-
search lane closes. Licensed sentence: on the eight factorial training cells,
the saved run reports perfect two-write accuracy and maximum immediate-reader
paired TV about 1.4e-8; its sound post-hoc interval-confirmation procedure
reports contraction-premise failure for both writes on the unit cube in the
infinity norm, while arbitrary-history behavioral overwrite remains
unestablished and unrefuted.

Config: `config/commit_seal_certificate.json`. Result: `results/commit_seal_certificate/result.json` (weights inside). Runner: `run_commit_seal.py` (mode `single_cell_certificate`, flag `--witness-search`).

---

## WB-1: Write-Boundary Construction v1 (2026-09-04, COMPLETE — F4 / TASK TOO SIMPLE)

Distance from claim: 0. First constructed artifact.

Can an explicit overwrite mechanism produce compositional compression that a
matched append-only carrier cannot? Two custom GRU models (~25K params, CPU),
same fact-tracking task (2 registers × 4 values, 5-step write sequences, 32768
possible histories).

**Theory:** CR-10 through CR-14 in theory/CONTINUATION_REFINEMENT.md. Selective
write-boundary laws (L1-L4), last-write normal form theorem, store cardinality
bound |Q_H| ≤ |S| = 16. Five pre-declared falsifiers.

**Results:**

| | Overwrite | Append-only |
|---|---|---|
| Params | 21,338 | 27,844 |
| Val accuracy | 100% | 100% |
| |Q_H| | **16** (exact bound) | 20 |
| Compression | 2048:1 | 1638:1 |
| Store alignment | **1.000** | 0.750 |
| Gen len=7 | 100% | 100% |
| Gen len=10 | 100% | 100% |
| Gen len=15 | 100% | 99.9% |

**Verdict: F4 — both compress; task may be too simple.** The append-only GRU
with attention learns to extract last-write information from its log almost as
well as the hard-masked register file. Quotient ratio 1.25x < 1.5x threshold.

**What survives:**
1. Overwrite model hits the theoretical bound EXACTLY (16 classes, perfect store
   alignment). The normal form theorem is confirmed by construction.
2. Append-only model leaks 25% of history (20 classes, 0.750 alignment). For 4
   of 16 store states, different write histories produce different behavior. The
   architectural difference IS visible, just not dramatic on this task.
3. Both generalize perfectly to 3× longer sequences on simple fact-tracking.

**What F4 means:** Simple fact-tracking (last-write recall) is too easy — a GRU
with attention can learn it without needing an explicit overwrite boundary. The
decisive test requires tasks where the answer depends on RELATIONSHIPS between
registers or on compositional operations, not just individual register reads.

**Next:** WB-2 — compositional task with cross-register operations.

---

## CEG-1: Causal Erasure Graph v1 (2026-09-04, COMPLETE — REVISE / NO_INTERFACE)

Distance from claim: 0.

Where does dead information live when a transformer retains overwritten facts?
Custom 4D attention mask interventions on Qwen3-1.7B-Base (eager attention,
full forward passes, no KV cache). 16 fixed-slot histories × 4 identifier sets,
6 intervention arms. 914 CPU forwards, 259s.

**Verdict (Codex Evidence Gate, audit #38): REVISE.** The original RELAY headline
does not pass. Competence preflight failed (83.6% < 95%); binding design required
NO_INTERFACE. RELAY-CUT Δ=0 is tautological (downstream positions see only
pair-identical tokens). The residual under QUERY-CUT is channel-nonspecific.

**Licensed sentence (Codex, verbatim):** In this fixed-slot Qwen3-1.7B-Base
panel, changing overwritten assignments altered normalized binary response
profiles, and after direct query-to-early-span attention was cut an indirect,
identifier-consistent excess over matched dummy cuts remained (mean max-channel
TV excess 0.101); however, the residual was channel-nonspecific and observed
after a failed competence preflight, so CEG-1 identifies bounded
attention-mediated context dependence, not semantic relay of dead information
in competent computation.

**Apparatus validation:**
- Fixture: OPEN mask reproduces standard inference exactly (TV=0.00)
- Mask effectiveness: extreme cut TV=0.856 (masks change output)
- Isochrony: 0 failures across 256 rendered histories
- Tokenization: clean concatenation for all histories

**Key results (post-audit):**
- OPEN Δ=0.406 (accuracy 83.6%; r0=71.9%, r1=95.3%)
- QUERY-CUT Δ=0.194 (accuracy 64.1%; channel specificity disappears)
- RELAY-CUT Δ=0.000 (accuracy 100%; tautological by mask design)
- LIVE-CUT Δ=0.934 (positive control)
- SYNCHRONIZER Δ=0.267 (accuracy 89.1%)
- Dummy OPEN Δ=0.105 (accuracy 96.1%)
- Paired excess (QUERY-CUT − DUMMY-CUT): 0.101, bootstrap CI [0.050, 0.152]
- Target vs orthogonal excess: 0.062 vs 0.061 (nonspecific)

**Audit corrections (adopted verbatim):**
1. RELAY-CUT=0 is a tautology: downstream positions see pair-identical inputs.
2. "52/48 pathway split" used arm-specific channel maxima and is invalid.
   Dummy-adjusted ratio is 33/67.
3. Under QUERY-CUT, register-specific effect disappears: target and orthogonal
   channels show equal excess. Residual is broadcast context sensitivity.
4. Four "renderings" = one syntax with four identifier pairs (name robustness,
   not template generalization).
5. SYNCHRONIZER rejects one string, not synchronization or irreversibility.
6. Design omitted target/orthogonal effects, CDFs, clustered bootstrap,
   full-vocab accuracy, replay noise, and model revision binding.
7. Mean erasure ratio (−24%) is an unstable ratio statistic driven by 3 extreme
   outliers at small OPEN denominators; "blocking direct access amplifies relay"
   is not identified. Ratio of arm means shows 52.3% reduction.
8. Measurement-to-artifact ratio: 506:0 (infinite).

**Never say:** "Dead information persists through attention relay into live-write
representations; roughly 48% is relayed, and the failed synchronizer shows that
this relay is irreversible."

**Real non-tautological finding:** RELAY-CUT raises accuracy from 83.6% to 100%.
A hard attention reset at the first live write erases earlier interference and
recovers every binary answer. This is a strong positional effect in the tested
interface; it does not by itself prove that the append-only carrier lacks an
endogenous behavioral overwrite boundary (audit #39 correction, 2026-09-05).

**Next:** CEG-1R (terminal pretrained-Qwen localization test) or close Qwen path.

Config: `config/ceg_1.json`. Result: `results/ceg_1/result.json`.

---

## CRC-1: Overwrite/Erasure Semigroup (2026-09-04, COMPLETE — INVALID / BOUNDED NEGATIVE)

Distance from claim: 0.

Does a transformer garbage-collect overwritten facts, or do dead histories
continue haunting its future behavior? Codex Architecture Theorist design.
Two binary registers (a,b), 32 roots (8 per final store in {0,1}²), 4
literal actions (a=0, a=1, b=0, b=1). Task-aligned {0,1} normalized binary
observer (primary), full-vocab diagnostic. Exhaustive words through length 2.
Cross-fitted 2-fold closure. Non-oracle NN baseline. Algebraic relation
verification (idempotence, absorption, commutation).

1508 forwards, 357s CPU, Qwen3-1.7B-Base float32.

**Preflight**: Tokenization clean. Competence 56/64 = **87.5%** — below
95% threshold, rendering experiment INVALID per preregistered criteria.

**Q_0**: 31 classes (task-channel). Store purity 1.0. Min TV = 0.011.

**Q_1**: **32 classes — identity partition.** Zero compression at every
tolerance level (0.005–0.05). Min D_1 = 0.099, far above any reasonable eps.
No multi-member classes. Every root is behaviorally distinct under even the
coarsest {0,1} observer.

**Q_2**: 32 classes. Tower trivially stable. Q_1→Q_2 frag = 0.000.

**Closure (cross-fitted)**: 52/256 = 0.203. (Bug fix: halves trained
separately, test roots assigned to nearest train class.)

**Algebraic relations**: Overall 25.9% pass rate (need >=95%).
- Idempotence: 37.9% (mean TV=0.054). a=0;a=0 ≢ a=0.
- Absorption: 20.7% (mean TV=0.102). a=0;a=1 ≢ a=1. Dead writes visible.
- Commutation: 19.1% (mean TV=0.110). a=u;b=v ≢ b=v;a=u. Not independent.

**Baselines**: NN (non-oracle, D_1-selected) TV ≈ 0.10. Quotient = 1.0
(all singletons). Store-matched = 1.0 (all held out in LOSO).

**Mode F**: TV = 1e-6. Cache perfectly faithful.

**Verdict**: INVALID (competence below 95%). But even ignoring validity:
comprehensive FAIL — identity partition, massive algebraic relation failure,
no compression at any tolerance. The model does NOT forget overwritten
facts; dead history remains operationally visible.

**Bug fixes from CRC-0**: (1) Closure now cross-fitted (2-fold). (2) NN
baseline selects neighbor by D_1 metric, not oracle.

**Binding finding**: Dead histories haunt the transformer. a=0;a=0 does
not behave like a=0. Order of independent register writes matters. The
model's state is path-dependent even for semantically equivalent programs.

Config: `experiments/config/crc_1.json`. Runner: `experiments/run_crc_1.py`.
Results: `experiments/results/crc_1/result.json`.

---

## CRC-0: Continuation-Refinement Calculus (2026-09-04, COMPLETE — FAIL / NO COMPRESSION)

Distance from claim: 0.

Decisive test of the CRC framework: does Qwen3-1.7B-Base admit a compact
predictive refinement tower on the SVB task family? Codex-designed (design
gate: REVISE then GO). Full next-token distribution (151,643 vocab), not
11-bin projection. 3 typed actions (N=neutral comment, H=heavy comment,
R=reassignment), 36 roots (3 variables × 4 values × 3 presentation
templates), exhaustive words through length 2 (13 per root). Clique-verified
equivalence classes. Cross-fitted transfer testing. Multi-tolerance stability.

544 forwards, 210s CPU, Qwen3-1.7B-Base float32.

**Preflight PASS**: Tokenization clean. Competence 36/36 = 100%.

**Q_0**: 33 classes (3 pairs merged at immediate response: x_1_P1≡y_1_P0,
y_3_P0≡y_3_P1, y_5_P0≡y_5_P1). Some presentation equivalence at horizon 0.

**Q_1**: 36 classes — **no compression**. Every root distinct. All Q_0 pairs
split. Min pairwise D_1 = 0.0165. Witnesses: R splits cross-var pair (TV=0.111),
H splits presentation pairs (TV~0.028).

**Q_2**: 36 classes. Tower stabilizes trivially. Q_1→Q_2 frag = 0.000.

**Closure**: vacuous (0/0, no multi-member Q_1 classes). Coverage: 0%.

**Transfer**: FAIL all folds (all singletons — no class members to predict from).
Text baseline TV = 0.039–0.057; nearest-neighbor = 0.016–0.031.

**Mode F**: TV = 7.0e-7. Cache perfectly faithful.

**Verdict stability**: Same FAIL at eps = 0.005, 0.01, 0.02.

**VERDICT: FAIL — no compression (Q_1 ~ one per state).**

**What we learned:**
1. The model maintains 36 distinct behavioral states (every var×val×pres triple
   produces a unique full-distribution signature surviving action application).
2. Q_0 compression exists (33/36) but shatters after one action.
3. Min D_1 = 0.0165, just 65% above tolerance — model is ALMOST compressible.
4. The CRC apparatus works cleanly; the model doesn't compress on this task.

Config: `config/crc_0.json`. Results: `results/crc_0/result.json`.
Runner: `run_crc_0.py`. Codex design gate: `.codex_crc_design.txt`.

---

## Curvature Control: Isochronous Position-Matched (2026-09-04, COMPLETE — FAIL / LINE CLOSED)

Distance from claim: 0.

Confound-killing control for the d3→d4 curvature signal. Codex-designed
(session 01a06cea), implementing exact token-position matching (two padding
families), varied inner literals (repeat8/repeat9), two shadow readouts (sd8/sd9),
6 template instances (2 train / 4 holdout), 5 surface pairs (2 train / 3 holdout),
position-gain null model (OLS), and 24-call deterministic replay calibration.

768 model calls, 249s CPU, Qwen3-1.7B-Base float32.

**G0 PASS**: All token counts verified, all boundaries clean.
**G1 PASS** (threshold 0.80): 85.4% accuracy. repeat8 profile causes digit-8 competition.
**G2 PASS**: Mean assert-minus-mislead gap = 2.16 nat.
**G3 PASS**: Zero replay noise (deterministic CPU).
**G4 FAIL**: All 4 position-gain null models fail holdout validation (train R²=0.82-0.96,
holdout MAE=0.16-0.25 > 0.05 threshold). Scalar position gain doesn't transport.
**G5 FAIL**: 9/10 checks pass. Mean κ = -0.542 nat, CI [-0.698, -0.362], 12/12 blocks
negative. Only failure: pad_diff_ok — PAD_BARE (-0.377) vs PAD_WORD (-0.706), diff
0.329 > 0.10 threshold. Individual cells: 93/96 negative, 3 PAD_BARE cells positive.
**G6 FAIL**: No valid null models from G4.

**Codex adjudication: Option (b) — real but qualified, LINE CLOSED.**
- The behavioral interaction is real but padding-sensitive, not intrinsic curvature.
- G5 is a genuine refutation of the preregistered invariant claim.
- The authorized "exactly one control round" has been spent and did not pass G4-G6.
- Most of κ comes from ASSERT becoming more beneficial at d4 (ρ_A=+0.375), not
  just MISLEAD strengthening (ρ_M=-0.167).
- "Still the R^n trap": scalar logit differences and OLS, not native math.

**Transferable residue:**
- Position matching did not produce representative-invariant magnitude.
- Scalar position-gain law fails to transport across template families.
- The response is presentation-conditioned (padding content modulates magnitude).

**Licensed sentence:** In this configured Qwen3-1.7B SVB control, the held-out
d3-to-d4 change in the MISLEAD-minus-ASSERT correct/shadow logit contrast was
negative on all 12 template-by-surface block averages under two exact-position
padding controls (mean −0.542 nat; clustered CI [−0.698, −0.362]); its magnitude
differed substantially between bare and word padding, and the registered
position-gain null did not validate, so this is a padding-sensitive behavioral
interaction rather than confirmed intrinsic curvature.

Config: `config/curvature_control_qwen3.json`.
Results: `results/curvature_control_qwen3_v1/result.json`.
Runner: `run_curvature_control.py`. Codex adjudication: `codex_curvature_control_adjudication.txt`.

---

## Curvature Test: Depth × Suffix Interaction (2026-09-04, COMPLETE — SUB_THRESHOLD / CONTROL REQUIRED)

Distance from claim: 0.

Codex-designed interventional curvature test. κ = [z(d_high, MISLEAD) - z(d_high, ASSERT)]
- [z(d_low, MISLEAD) - z(d_low, ASSERT)]. Under additive logit model κ=0.

288 calls, 98.3s CPU. 3 variables × 8 values × 3 depths × 3 suffixes.
Split-half validation: train on {x,y}, hold out {z}.

**Gate 0 (Competence):** PASS — 100% accuracy (72/72).

**d2→d3:** Heterogeneous (x positive, y negative — cancel in aggregate). Not null; not coherent.

**d3→d4:** κ=-0.284 nat for the tested pair, 24/24 cells same sign.
Sub-threshold: 0.28 < 0.5 pre-registered gate. SNR gate also never computed.

**Codex adjudication (REVISE / CONTROL REQUIRED):**
- Descriptively nonzero (17% of suffix contrast, exp(-0.28)=0.75). Scientifically sub-threshold.
- Gate 1b cross-surface check: only 65.5% of 168 (assert,mislead) pairs show negative κ.
  MISLEADING surfaces range −0.235 to +0.155 (mixed signs). NOT representative-invariant.
- Position is the strongest rival: absolute token position changes with depth (38→58→79).
- 24 deterministic cells ≠ 24 independent replications.
- Half the separation comes from ASSERT becoming MORE beneficial at d4.

**What it advances:** The fixed suffix contrast is context-dependent and cannot be
represented by the simplest globally additive logit-shift model. Candidate obstruction
and falsifier generator, not yet a law.

**Next:** One confound-killing control round (position-matched, token-matched,
multi-surface, varied shadow). If residual curvature doesn't survive, close this line.

Config: `config/curvature_qwen3.json`. Results: `results/curvature_qwen3/result.json`.
Runner: `run_curvature.py`.

---

## S3 Permutation Composition (2026-09-04, COMPLETE — MODEL_CANNOT_COMPOSE)

Distance from claim: 1 (task family competence test).

Codex-designed experiment testing whether Qwen3-1.7B-Base computes S3 permutation
group composition with genuine algebraic structure. 36 ordered pairs (6x6),
6-way candidate scoring via teacher forcing, 4 pre-registered gates.

**Result:** Gate 1 FAIL (4/36 correct, threshold 34). Model has degenerate c021
output bias — predicts the same permutation regardless of input. All 4 "correct"
predictions are lucky matches. Gate 2 also FAIL (additive null beats model).

**What we learned:** Qwen3-1.7B-Base cannot trace Python function calls for tuple
permutation operations. This is a COMPETENCE failure, not an algebra failure. The
task is beyond the model's code-tracing capability at 1.7B parameters. Future
composition tests must use tasks within the model's proven competence (e.g.,
SVB-style variable retrieval with simple single-digit output).

Config: `config/s3_qwen3.json`. Results: `results/s3_qwen3/result.json`.

---

## Congruence Test (2026-09-04, COMPLETE — NO_CONGRUENCE)

Distance from claim: 0.

768 calls, 218s CPU. Codex-designed decisive test for native algebra.

**Purpose:** Test whether behavioral equivalence (surfaces with matching δ_L)
is preserved under suffix composition — the algebraic congruence property.

**Design:** 3 equivalence pairs (cross-role surfaces with δ_L diff < 0.006),
composed with 2 composers (neutral: "# No changes.", heavy: "# Reassigning {var} now."),
in both orders (a;c and c;a), across 24 contexts.

**Results:**

| Pair | Single TV | Neutral comp | Heavy comp | Ratio (heavy) |
|---|---|---|---|---|
| strong_assert | 0.006 | 0.004-0.011 | 0.019-0.074 | 3.4-13.2 |
| moderate_assert | 0.013 | 0.008-0.012 | 0.038-0.048 | 3.0-3.8 |
| misleading | 0.025 | 0.023-0.043 | 0.051-0.064 | 2.1-2.6 |

**Verdict:** NO_CONGRUENCE. Behavioral equivalence holds in isolation but is
destroyed by composition with semantically heavy suffixes. The δ_L-based
quotient is NOT an algebraic congruence.

**Additional finding:** δ_L is predicted by text features at R²=0.88.
Dominant features: has_preserve (-1.86), is_rewrite (+0.71), has_change (-0.57).

Config: `experiments/config/svb_qwen3_composition_v2.json` (shared)
Results: `experiments/results/svb_qwen3_congruence/result.json`
Runner: `experiments/run_congruence.py`

---

## Quotient Closure + Held-Out Prediction (2026-09-04, COMPLETE — K_s DEAD)

Distance from claim: 0.

Pure analysis on Gate 1b data (3,384 obs, no new model calls).

**Purpose:** Test if K_s (3×3 stochastic matrix on CLR simplex) generalizes
on held-out data and beats baselines.

**Results:** K_s: 0/43 surfaces beat ANY baseline. Mean err 0.290 vs best
baseline (logit-shift) 0.039. The actual operator is a constant logit shift.
Shadow logit shift δ_L is depth-invariant (CV=0.036 ASSERT). Response
collision closure 100% for ASSERT roles. Cross-role transfer fails.

Runner: `experiments/analyze_quotient_closure.py`

---

## Gate 2 v2 Multi-Filler Robustness (2026-09-04, COMPLETE — POSITION_WITH_NONLINEAR_GAIN)

Distance from claim: 0.

288 new calls (filler pairs 2 & 3), reusing 192 from v2 (AM/MA + pair 1).
3 independently selected token-matched filler pairs. 2.4 min CPU.

**Purpose:** Test whether residual order contrast (the correct statistic) is
consistent across filler pairs and survives scalar gain fit.

**Results:**

| Filler pair | Resid. median | Gain-fit median | Contexts > 0.01 (gain) |
|---|---|---|---|
| pair1 | 0.067 | 0.007 | 6/24 |
| pair2 | 0.040 | 0.003 | 0/24 |
| pair3 | 0.089 | 0.009 | 10/24 |

Cross-filler correlations: pair1-2 r=0.82, pair1-3 r=0.92, pair2-3 r=0.69.

**Verdict:** POSITION_WITH_NONLINEAR_GAIN. The order effect is real, consistent
across fillers, but fully absorbed by a scalar gain fit. Position with
context-dependent magnitude is the correct explanation. Gate 2 composition is
a structured negative.

Config: `experiments/config/svb_qwen3_composition_v2_multifiller.json`
Results: `experiments/results/svb_qwen3_composition_v2_multifiller/result.json`
Runner: `experiments/run_multifiller.py`

---

## Gate 2 v2 — Filler-Based Noncommutativity (2026-09-04, CLOSED — subsumed by multi-filler)

Distance from claim: 0 (this IS the native math claim).

216 calls, 8 arms per context, 24 contexts (3 vars x 8 vals), depth 4.
135s CPU on Qwen3-1.7B-Base. Deterministic (seed 42).

**Purpose:** Test whether suffix operators A and M compose non-trivially
(noncommutatively) after absorbing all additive position/recency effects.
Uses Codex-designed token-length-matched neutral fillers and algebraic
subtraction: predicted_phi(AM) = phi(AF_M) + phi(F_AM) - phi(F_AF_M).

**Design:** 8 arms: AM, MA, AF_M, F_MA, MF_A, F_AM, F_AF_M, F_MF_A.
A = "# {var} is unchanged.\n" (5 tok), M = "# Reassigning {var} now.\n" (7 tok).
F_A = "# Code block follows.\n" (5 tok), F_M = "# Looking at the next part.\n" (7 tok).

**Results:**

| Metric | Value |
|---|---|
| Direct TV(AM,MA) median | 0.068 |
| Direct TV(AM,MA) mean | 0.072 |
| Excess (prob-space) mean | 0.033 |
| Excess (log-ratio) mean | 0.038 |
| Gate 1 (direct > 0.01) | PASS |
| Gate 2 (excess > 0.01) | PASS |
| Verdict | NONCOMMUTATIVE_CONTROLLED |

Per-variable (log-ratio): x direct=0.040/excess=0.038, y direct=0.112/excess=0.026,
z direct=0.063/excess=0.050.

**Codex evidence gate: PROVISIONAL.** Gate 1 (direct order) strong. Gate 2
(excess) disputed: implementation computed armwise prediction error, not
pre-registered residual order contrast. Filler-predicted order vectors align
(cosine=0.996) but over-predict magnitude. Nonlinear position gain is the
strongest rival. Repair: frozen residual-order statistic, multi-filler
robustness, held-out prefix family.

**What we learned:**
1. Direct order effect is robust (median TV=0.068, all 24 contexts above threshold).
2. Filler-based controls confirm direction but not independence from position.
3. Nonlinear position gain (scalar fit) reduces 18/24 contexts below threshold.
4. K_A correction rate L->C is depth-invariant at ~0.72 (separate finding).

Config: `experiments/config/svb_qwen3_composition_v2.json`
Results: `experiments/results/svb_qwen3_composition_v2/result.json`
Runner: `experiments/run_composition_v2.py`

---

## Gate 2 v1 — Operator-Level Composition (2026-09-04, COMPLETE — RECENCY_DOMINATED)

Distance from claim: 0.

504 calls, 4 pairs (2 cross-role, 2 same-role controls), depth 4. 550s CPU.
Verdict: RECENCY_DOMINATED. Cross-role TV median=0.081 < same-role max=0.130.
Codex correction: same-role controls are the wrong null model.

Config: `experiments/config/svb_qwen3_composition.json`
Results: `experiments/results/svb_qwen3_composition/result.json`

---

## Gate 1b — Confound Control (2×2 ANOVA + lumpability + surface equivalence) (2026-09-04, COMPLETE)

**Interpretation qualification, 2026-09-13:** the content factor is the tested
assertion-versus-misleading text contrast, not an intervention on meaning with
every surface feature controlled. Its 6.3× ratio excludes variable mention alone
on this panel. Historical "proves" and native-coordinate language below is also
superseded by the later failed composition and scalar logit-bias findings.

Distance from claim: 0 (this IS the native math claim).

3,384 calls across 8 roles (ASSERT, ASSERT_VAR, MISLEADING_ASSERT,
MISLEADING_ASSERT_NOVAR, REWRITE, OBSERVE, BOUNDARY, BASELINE), 3 vars,
8 outer_values, 3 depths (2-4). ~100 min CPU on Qwen3-1.7B-Base.

**Purpose:** Disentangle content vs variable-mention effects from Gate 1.
Test whether K_u is a well-defined operator (lumpability) and whether
surfaces within the same semantic class are interchangeable (equivalence).

**Results:**

| Analysis | Verdict | Key number |
|---|---|---|
| 2×2 ANOVA | CONTENT DOMINATES | 6.3x (content +0.092 vs var-mention +0.015) |
| Lumpability (F8) | STRONG | R2 = 0.79–0.98 across all roles |
| Multiplicative char CV | HIGH | Within-class CV 0.33–0.51 |
| Surface equiv (class A) | 1 pair passes | max TV = 0.009 |
| Surface equiv (class M) | 0 pairs pass | best max TV = 0.022 |
| K_a vs logit-bias | INCONCLUSIVE | No holdout for cross-validation |
| R-preservation | Tentatively K_a | ASSERT dR=−0.007, MISLEADING dR=+0.003 |
| F6 Baseline | MARGINAL | +0.006 TV over cheap features |

**What we learned:**
1. The content/misleading distinction is real and dominant (6.3x over var-mention).
2. The (C,L,R) simplex IS approximately the right coordinate system — lumpability
   proves K_u is well-defined on it.
3. But the mapping surface→operator is finer than semantic role. Different surfaces
   within the same class produce different K_u operators (CV 0.33–0.51).
4. Gate 2 has one validated class-A pair but zero class-M pairs at eps_eq=0.01.

Runner: `experiments/run_semantic_descent.py`. Config: `experiments/config/svb_qwen3_gate1b.json`.
Result: `experiments/results/svb_qwen3_gate1b/result.json`.
Analyzers: `analyze_multiplicative_character.py`, `analyze_lumpability.py`,
`analyze_surface_equivalence.py`. Ledger: `gate1b_confound_control`.

---

## Decode-versus-full diagnostic — faithful query continuation localized (2026-09-03, COMPLETE; INTERPRETATION CORRECTED)

Distance from claim: 2 (runtime-instrument validation).

Four fixed 27-cell Falcon-H1 conditions compare:

- prefix prefill followed by the 12-token query in one multi-token cached call;
- prefix prefill followed by the same query one token at a time;
- the same complete text in one no-cache forward pass.

**Results:** condition-mean TV between tokenwise cached and full-forward 11-bin
response laws was `5.269e-7` to `1.2107523675e-6`. Multi-token cached versus
full-forward TV was `0.546–0.885`. The prefetched `s1` condition reduced mean
correct-digit probability relative to `s0` at both tested depths.

**Licensed interpretation:** the multi-token cached query path changes the
tested hybrid runtime's response law, while tokenwise cache advancement agrees
closely with full-forward execution on this projected aggregate diagnostic.

**Limits:** the runner inserts `s0`/`s1` before prefill and changes only query
chunking, so it does not rerun the original prefix-cache → suffix-plus-query
boundary. The result JSON retains condition aggregates rather than per-cell
response laws. It does not establish full-vocabulary equivalence, architecture
separation, an attention-versus-SSM mechanism, continuous-vector reasoning,
efficiency, or future acquisition.

Runner: `experiments/verify_svb_decode_mode.py`. Result:
`experiments/results/svb_decode_verification.json`. Audit correction:
`svb_decode_verification_audit_correction` in the append-only ledger.

---

## Execution-Mode Invariance — four modes, 9 core words (2026-09-03, COMPLETE: ALL DIVERGENT)

Distance from claim: 2 (measurement of the artifact).

Tests whether the suffix action algebra is invariant across execution modes
of the Mamba-Transformer hybrid. Codex design gate: CONDITIONAL GO.

**Four modes:**
- L (legacy joint): prefix cached → [word + query] — reproduces existing evidence
- W (whole then query): prefix cached → [word] → [query] — isolates query boundary
- G (generator by generator): prefix cached → [gen1] → [gen2] → ... → [query]
- F (full text): [prefix + word + query], no cache — tests theory's claimed Z

**Pre-registered:**
- Mode tolerance: eps_mode = max(0.01, 5 * delta_replay), calibrated from replay
- Defect re-adjudication per mode (BAND2, SAT3, right-continuation) at eps_TV=0.06
- 100K bootstrap for mode comparisons, 10K for defects, seed 42
- A post-hoc simultaneous-band artifact was produced but is nondecision-grade:
  its maximum statistic was uncentered and cannot validate the independent intervals.

**Decision rules:**
- L≠W: query-boundary artifact in existing evidence
- L=W but W≠G: generator packaging matters, no partition-independent action
- L=W=G but ≠F: cached-state action system valid, Z = cache states not texts
- All agree: execution-mode blocker clears

**Results (2079 calls, 5568s CPU, Stage 1 only — Stage 2 not triggered):**
- Token identity gate: 420 checks, zero mismatches.
- Replay calibration: delta_replay=0.0 (perfectly deterministic), eps_mode=0.01.
- Consistency check: Mode L CPC bit-for-bit identical to band2 historical data.
- All 54 word×mode-pair point comparisons exceed the registered mode tolerance
  under the original reducer. The committed simultaneous reanalysis is not a
  valid confidence-band repair and carries no additional decision weight.
- L vs W: TV 0.33-0.42 — MASSIVE query-boundary artifact.
- W vs G: TV 0.05-0.18 — generator packaging changes state.
- All vs F: TV 0.36-0.78 — cached vs non-cached fundamentally different.
- Mode F collapses ALL defects to NEAR (TV<0.016). The algebra vanishes
  without KV caching.
- Defect re-adjudication: L matches historical. W shifts some edges. G makes
  all FAR. F makes all NEAR.
- Spearman structural invariance: L/W/G moderately correlated (rho 0.54-0.69),
  F uncorrelated with all (rho -0.007 to 0.189, p>0.27).

**Interpretation (REVISED 2026-09-03):** Source inspection localizes a
HuggingFace implementation gap: Falcon-H1 `torch_forward` starts the Mamba SSM
layers from ZERO state for multi-token continuation (line 819:
`previous_states = torch.zeros_like(states[:, :1])`). Both
`mamba_chunk_scan_combined` and `causal_conv1d_fn` accept `initial_states`
parameters; this path does not pass them for `seq_len > 1`.

Verification: one-shot vs two-shot TV = 0.118, max logit diff = 6.527.
One-shot vs single-token decode TV = 0.000000 (exact match).
Cross-prefix two-shot response convergence TV = 0.794, consistent with lost
direct recurrent history but not proof of zero prefix influence through attention.

The affected path omits direct cached recurrent and convolutional history, and
its registered response law differs sharply from full-forward and tokenwise
execution. This makes the recorded Falcon suffix algebra non-claim-bearing as
faithful state dynamics; it does not prove that the omission causes every
suffix contrast. All prior Falcon-H1 SVB/suffix experiments used this path.
SVB-Qwen3 does not use the Falcon/Mamba branch.

**Status:** COMPLETE runtime diagnosis; no architecture or mechanism claim.

---

<details>
<summary><strong>Historical affected-runtime algebra experiments — superseded</strong></summary>

> **Terminal correction:** The following measurements were made through the
> defective multi-token cached continuation. Their arithmetic remains a runtime
> diagnostic, but their native-object, composition, and faithful-model
> interpretations are not claim-bearing.

## Historical Right-Continuation Test (2026-09-03)

Distance from claim: 2 after runtime audit.

Tests whether length-3 words are genuine terminal elements (right absorption)
or merely single-step near-returns. 4 arms (CPC, CPCC, PCPP, CPCPC), 27 cells,
135 forward passes, 2654s. TV metric, eps_TV=0.06.

**Results:**
- CPC absorbs C? TV(CPC,CPCC)=0.060 [0.051,0.067] → INCONCLUSIVE
- PCP absorbs P? TV(PCP,PCPP)=0.072 [0.066,0.077] → REFUTE (LB>ε)
- Len-5 SAT?    TV(CPCP,CPCPC)=0.064 [0.056,0.072] → INCONCLUSIVE
- CPC terminal? TV(CPC,CPCPC)=0.099 [0.086,0.110] → REFUTE

Key findings:
- The monoid does NOT collapse to a finite quotient. Words keep evolving past
  length 3. CPCPC is genuinely different from CPC (TV=0.099, well above ε).
- H-SAT3 near-return (CPCP≈CPC) is a one-step property, not genuine saturation.
- Sigma degrades monotonically with suffix length: CPC=0.486 → CPCC=0.439 →
  CPCPC=0.393. Performance worsens, contradicting terminal-zone interpretation.
- PCP definitively does NOT absorb P (LB=0.066 > ε_TV).
- CPC absorbing C remains inconclusive (point estimate exactly at threshold).
- CPC arm bit-for-bit consistent with band2 run (consistency check PASS).

Historical diagnostic object: a truncated response-labelled continuation table
for the affected runtime.

---

## Historical Band2 Decisive Test (2026-09-03)

Distance from claim: 2 after runtime audit.

Tests three competing hypotheses after H-LRB refutation:
H-BAND2 ((CP)^2~CP), H-SAT3 (CPCP~CPC), H-GEN-IDEM (all grow).
12 arms, 27 cells, 351 forward passes, 925s. TV metric, eps_TV=0.06.

**Results:**
- H-BAND2 REFUTED: TV(CPCP,CP)=0.107 [0.098,0.117], TV(PCPC,PC)=0.122 [0.108,0.136]
- H-SAT3 BEST FIT, FORMALLY INCONCLUSIVE: CP arm passes (TV=0.050, UB=0.056 ≤ 0.06);
  PC arm borderline (TV=0.052, UB=0.064 > 0.06). Registered joint rule requires both.
- H-GEN-IDEM REFUTED: SAT3(CP) UB <= 0.06

Key findings:
- Generator idempotence does NOT propagate to products: CPP-CP formally inconclusive
  (CI [0.056,0.070] crosses ε_TV); PCC-PC marginal (LB=0.062). Not a congruence.
- Branch coalescence: TV(CPC,CPP)=0.039 — CP's children converge, but both remain
  separated from CP itself. NOT right absorption.
- Overlapping maximal cliques at ε=0.06, not a single equivalence class.
- Exploratory contraction (unregistered): C distance-reducing (mean r=0.67),
  P mixed (mean r=1.03). Post-hoc, incomplete pair selection.
- No semigroup variety licensed. The affected-runtime table can be represented
  as a truncated response-labelled continuation automaton.
Fidelity: cached-vs-full TV=0.533; all conclusions scoped to cached-continuation.
Codex evidence gates (two rounds): REVISE verdicts adopted. LRB-era defects
reproduce exactly.

Runner: experiments/run_band2_decisive.py. Results: experiments/results/band2_decisive/.
Predictions frozen: commit f95dfb7.

---

## Historical Suffix Action Algebra construction (2026-09-03; interpretation superseded)

Distance from claim: 2 after runtime audit — formal diagnostic, not native-model evidence.
Files: `theory/closed/SUFFIX_ACTION_ALGEBRA.md`, `theory/suffix_algebra.py`, `theory/frozen_predictions_CP.txt`.
Ledger: `suffix_action_algebra_construction`.

**What was built:**
- SVB transition world as D1 specialization (raw states, suffix actions, 11-bin response laws)
- Behavioral places Q = Z/~ (D5 quotient by future-response identity)
- Suffix transition monoid S = A*/≡_Q (behaviorally distinct suffix sequences)
- Defect measures (idempotence I, residuation R, commutator N) with binding thresholds
- Accessibility languages L_θ(q) and directed cost c_θ(q)

**Hypotheses stated:**
- H-LRB: S is approximately a left-regular band (a² ≈ a, aba ≈ ab)
- H-Truth: suffix actions conjugate under truth-reversal involution J

**Theorem proved:** Truth-congruence reversal obstruction (on raw states Z).

**Symbolic normalizer (frozen predictions before experiment):**
- 2 generators {C, P}: 5 normal forms, 2 decisive predictions (CPC ≈ CP, PCP ≈ PC)
- 4 generators {C, P, U, V}: 65 normal forms, 13 decisive predictions

**Codex audit:** FAIL → 7 corrections adopted (D1 convention bridge, finite-access,
theorem retyping on Z, notation clashes, R^n claim narrowed, LRB caveats, metric scale).

</details>

---

## Historical Falcon content screen — affected by the Mamba continuation gap (2026-09-03)

Distance from claim: 1 (settling mechanism characterization).
Runner: `scratchpad/semantic_content_probe.py`. Results: `experiments/results/semantic_content_result.json`.
Ledger: `semantic_content_probe`.

**Method:** 8 suffix conditions on SVB-2 template (d3, Falcon-H1, 27 measurements each).
Tested: "# No changes.", "# TODO", "# Lorem ipsum dolor sit amet", "#", "# ", "pass", "\n", no suffix.

**Results:**
| Condition | Mean σ | Gain |
|-----------|--------|------|
| # No changes. | 0.4299 | +53.5% |
| # (space) | 0.3295 | +17.7% |
| # TODO | 0.2963 | +5.8% |
| # (bare) | 0.2855 | +1.9% |
| baseline | 0.2801 | — |
| pass | 0.2410 | -13.9% |
| Lorem ipsum | 0.2351 | -16.1% |
| bare newline | 0.2230 | -20.4% |

**Current interpretation:** these contrasts are reproducible inside the defective
multi-token cached runtime but are not faithful Falcon-state evidence. The
registered strings must be rerun through the corrected interface before any
content, semantic, or computation mechanism is promoted.

---

## Historical Falcon order screen — affected by the Mamba continuation gap (2026-09-03)

Distance from claim: 1 (structural prediction test).
Runners: `scratchpad/order_independence_probe.py` (v1), `scratchpad/order_independence_svb2.py` (v2).
Results: `experiments/results/order_independence_result.json` (v1),
`experiments/results/order_independence_v2_result.json` (v2). Ledger: `order_independence_v1`, `order_independence_v2`.

**Method:** 7 suffix conditions (s0, s1_comment, s1_pass, s2_comment+pass, s2_pass+comment,
s2_comment+comment, s2_pass+pass) at d3 on Falcon-H1. V1 used simplified template, V2 used SVB-2 template.

**V1:** Commutativity appeared to hold (diff 0.011) but comment was near ceiling (0.90).
**V2 historical observation:** comment→pass produces 0.084 higher σ than
pass→comment, same sign in all 27 paired cases, inside the defective execution
regime. No faithful-state commutativity or idempotence claim survives.

**Codex evidence gate (REVISE):** Order effect is real but projection/subspace mechanism language
not earned. "Pass hurts" is heterogeneous across variables. Prompt-history sensitivity is an equally
valid explanation. Licensed claim: behavioral order sensitivity on the fixed panel, not a latent
projection or consolidation law.

---

## Historical SVB-2 on Falcon-H1 — affected by the Mamba continuation gap (2026-09-03)

Distance from claim: 2 (historical runtime diagnostic; not mechanism identification).
Runner: `experiments/run_svb_0.py experiments/config/svb_2.json`. Config: `experiments/config/svb_2.json`.
Results: `experiments/results/svb_2/`. Ledger: `svb_2_result`.

**Method:** Suffix counts [0,1,2,3,4,6,8] at depths 1-4. Neutral suffix = "# No changes.\n".
Fine-grained resolution to determine if d4 s1 peak is genuine.

**Results:** The observed maximum was at s1 for d2 and d4 and at s2 for d3;
the d3 s1/s2 difference was unresolved (p=0.86).
Recorded s1-relative gain increased across d2-d4 in the affected runtime:
d2 +29.6%, d3 +53.5%, d4 +87.9%.
The numerical profile is retained as historical diagnostic evidence only; it
does not confirm a faithful one-shot trigger.

---

## SVB-Qwen3-Formal — bounded suffix-conditioned response effect (2026-09-03; FIXED-PANEL WITNESS)

Distance from claim: 1 (bounded depth-dependent response intervention).
Runner: `experiments/run_svb_0.py experiments/config/svb_qwen3_formal.json`.
Results: `experiments/results/svb_qwen3_formal/`. Ledger: `svb_qwen3_formal`.

**Method:** Full SVB on Qwen3-1.7B-Base (pure transformer, no recurrence). 621 calls, 3.3 min CPU.

**Results:** On this Qwen fixed panel, the comment intervention increases mean
correct-digit probability by 0.0118, 0.0487, 0.1087, and 0.1263 at depths 1–4.
The paired change is positive in 24/27 cells at d1 and 27/27 at d2–d4. Among
tested suffix counts `{0,1,2,4}`, s1 has the highest mean at d2–d4.

**Corrected interpretation:** this is one prompt-family-bounded extra-text
response effect. The affected Falcon measurements cannot provide a clean
cross-architecture replication. Added computation, token position, lexical
content, syntax, and attention state change together. No universal settling
law, prior answer encoding, architecture mechanism, or efficiency claim is
licensed.

---

## Qwen suffix-response probes — descriptive mechanism screen (2026-09-03)

Distance from claim: 1 (bounded response characterization; mechanism unresolved).
Runners: various probe scripts in scratchpad. Results in STATE.md and NOTEBOOK.md.

Series of quick probes on Qwen3-1.7B-Base:
1. Double comment (+11.8%) is below single comment (+13.0%); this rejects a simple monotone token-count account, not all computation-time accounts.
2. Python `#` (+13.0%) exceeds C++ `//` (+2.9%) inside the Python prompt family; syntax, lexical familiarity, and position remain entangled.
3. Docstrings exceed comments (+17.5% at d3), while a competing value lowers the score by 26–34% on tested cells.
4. The best tested task-relevant reminder at d4 reaches σ=0.940, above raw d2 (0.897); because it supplies relevant content, it is not latent-only computation evidence.

---

## Historical SVB-1 Falcon depth curve — affected by the Mamba continuation gap (2026-09-03)

Distance from claim: 2 (archived measurement from a defective cached-state interface).
Runner: `experiments/run_svb_0.py experiments/config/svb_1.json`. Config: `experiments/config/svb_1.json`.
Results: `experiments/results/svb_1/`. Ledger: `svb_1_launch`, `svb_1_result`.

**Method:** Extends SVB-0 to depths 1-4, single-var only (two-var skipped — already
known to fail). Same model, same task family, same suffix profile [0,1,2,4]. Runner
parameterized with CLI config path and `single_var_only` flag. Pre-registered
predictions: geometric decay r≈0.73, σ_d3≈0.36, σ_d4≈0.26.

**Depth capacity curve:**
| Depth | σ | CI | κ | CI |
|-------|-------|-------------|-------|-------------|
| 1 | 0.681 | [0.637, 0.725] | 0.712 | [0.691, 0.733] |
| 2 | 0.497 | [0.437, 0.557] | 0.501 | [0.472, 0.531] |
| 3 | 0.280 | [0.213, 0.350] | 0.243 | [0.218, 0.267] |
| 4 | 0.229 | [0.182, 0.280] | 0.190 | [0.174, 0.206] |

Recorded decay ratios are d1→d2=0.73, d2→d3=0.56, d3→d4=0.82. They reject the preregistered geometric fit **inside the affected runtime only**.

**Recorded suffix profile across depths:**
| Depth | s0 | s1 | s2 | s4 | Peak | Gain |
|-------|-------|-------|-------|-------|------|------|
| 1 | 0.681 | 0.669 | 0.560 | 0.396 | s0 | — |
| 2 | 0.497 | 0.645 | 0.597 | 0.516 | s1 | +30% |
| 3 | 0.280 | 0.430 | 0.441 | 0.338 | s2 | +57% |
| 4 | 0.229 | 0.431 | 0.401 | 0.298 | s1 | +88% |

**Historical interpretation, superseded:** the recorded curve is non-geometric
and its largest tested suffix-conditioned gain rises from d2 to d4. Because the
multi-token continuation discarded cached recurrent state, it cannot establish
a phase transition, stored-but-inaccessible bindings, a recurrent settling
mechanism, or a native cost law. It remains a diagnostic profile of that runtime.

**Evidence gate:** superseded by the 2026-09-03 Mamba continuation audit and the
append-only correction `svb_decode_verification_audit_correction`.

---

## Historical SVB-0 on Falcon-H1-1.5B-Instruct (2026-09-03; AFFECTED BY MAMBA GAP)

Distance from claim: 2 (the task remains useful, but the recorded cached-state
result is not faithful hybrid-state evidence).
Runner: `experiments/run_svb_0.py`. Config: `experiments/config/svb_0.json`.
Results: `experiments/results/svb_0/`. Ledger: `svb_0_launch`, `svb_0_result`.
Spec: `theory/closed/SVB_0.md`.

**Method:** Python lexical scoping as the task family: outer assignment → inner
function call (shadows variable) → scope closure → query. Falcon-H1-1.5B-Instruct
selected for code capability + recurrent state (DynamicCache save/restore for
state injection). 11-bin response law: {digit 0-9, OTHER}. 3 variables (x,y,z),
9 outer values (1-9), depths 1-2, suffix counts [0,1,2,4] neutral comment lines.
Competence staircase: direct assignment → single-var depth-1 → two-var depth-1.
Single-var and two-var science observations with suffix profiles. 3 observables:
σ (scope binding fidelity = P(correct digit)), κ (path contrast = TV between
different outer values), ι (entity interaction = variable-specific vs global).
Null ladder: uniform, inner_value, mean_dist, identity. Bootstrap CIs (1000 resamples).

**Competence:** Rung 1 PASS (27/27 = 100.0%, gate 90%). Rung 2 PASS (27/27 = 100.0%,
gate 85%). Rung 3 FAIL (23/30 = 76.7%, gate 80%).

**Verdict: INSUFFICIENT_SCOPE_BINDING.** Two-var depth-1 competence (76.7%) below
80% gate. Formal adjudication stops at the competence check. However, all observables
computed from 1188 observations (1854 model calls, 4385s/73min CPU).

**Terminal correction:** all suffix/state observables below were captured through
the defective multi-token cached continuation. They remain historical runtime
measurements and cannot establish faithful state, binding, access, or mechanism.

**Observables:**
- σ_d1 = 0.6811 (CI: [0.637, 0.725]), σ_d2 = 0.4975 (CI: [0.437, 0.557])
- κ_d1 = 0.7123 (CI: [0.691, 0.733]), κ_d2 = 0.5009 (CI: [0.472, 0.531])
- ι = 0.3593 (CI: [0.343, 0.376])
- Depth-1 suffix profile: s0=0.681, s1=0.669, s2=0.560, s4=0.396
- Depth-2 suffix profile: s0=0.497, s1=0.645, s2=0.597, s4=0.516

**Null ladder:**
- d1: uniform=0.627, inner_value=0.833, mean_dist=0.597, identity=0.712
- d2: uniform=0.609, inner_value=0.636, mean_dist=0.555, identity=0.501

**Historical profile:** the affected path produced registered contrasts across
depth, entity, and suffix conditions. Its two-variable competence rung failed.
The later runtime audit supersedes the causal-control, variable-specific binding,
recurrent breathing-room, and “strongest positive signal” interpretations.

---

## PMO-0R — Path-Memory Observability Revised on Finch-3B (2026-09-02; TASK_POPULATION_VOID)

Distance from claim: 1 (bounded continuation-distinguishability witness).
Runner: `experiments/run_pmo_0r.py`. Config: `experiments/config/pmo_0r.json`.
Results: `experiments/results/pmo_0r/`. Ledger: `pmo_0r_launch`, `pmo_0r_result`.
Spec: `theory/closed/PMO_0R.md` (Codex R1+R2 locked).

**Method:** Corrects two PFC-0 defects (symmetric endpoints, 3-logit renormalization).
Asymmetric panels (entities end at different locations), 4-bin response law
{kitchen, garden, office, OTHER} — full next-token softmax, no renormalization.
9 roots × 3 panels × 6 extensions = 162 history states. 2 matched pairs per
configuration for commutation defect κ. Suffix injection via saved recurrent
states at 0/1/2/4 macro repetitions. Competence staircase: direct facts →
2-action → 4-action → suffixes. 9-method cross-fitted null ladder.

**Verdict: TASK_POPULATION_VOID.** Direct fact competence failed at rung 1
(0/36, 0.0% accuracy). Model puts ~65% probability on continuation tokens
("answer", "question") vs ~35% on all location tokens combined. Competence
gate requires correct location > P(OTHER) — unreachable with this template.
Even relaxed (argmax among 3 locations), accuracy only 55.6% due to kitchen bias.

**What was learned:**
(1) PFC-0's 3-logit renormalization was hiding model incompetence. After
renormalization, kitchen at 68.5% looked like clear discrimination; the raw
model allocates only 24% to kitchen (its strongest location) and 6%/5% to
garden/office. Codex was correct to flag renormalization as a defect.
(2) Finch-3B's entity-location query interface is fundamentally limited for
path-memory measurement. The model discriminates among locations but doesn't
produce location-dominated responses with natural-language prompts.
(3) Kitchen calibration bias persists: kitchen is the default regardless of
correct answer, causing 44% of relaxed errors.

**Closure:** This response interface (4-bin + explicit-choice template) on
Finch-3B is closed as unusable. Competence failure does NOT close path memory
as a concept, entity discrimination (TV=0.43), or bit-exact state replay.

**Codex evidence gate: DEFERRED (Codex credits exhausted until 2026-09-06).**

---

## PFC-0 — Path-Fiber Calculus v1 on Finch-3B (2026-09-02; TASK_POPULATION_VOID)

Distance from claim: 2 (Codex evidence gate correction — K is researcher-imposed
3×3 stochastic matrix on softmax simplex; composition is ordinary matrix
multiplication; imported probability algebra, not native math).
Runner: `experiments/run_path_fiber_v1.py`. Config: `experiments/config/path_fiber_v1.json`.
Results: `experiments/results/path_fiber_v1/`. Ledger: `pfc_0_launch`, `pfc_0_result`.

**Method:** Four-corner path square (p00, pL, pR, pLR) with washout tail.
9 roots × 3 panels × 6 paths × 2 queries = 324 endpoint specs + 72 replay calls.
Cross-fitted stochastic transports K_L, K_R learned from 3 training corners,
predict held-out pLR. SLSQP with ridge 1e-3. 3-fold cross-fitting on roots.
7 baselines: parser, last-1, last-2, multiset, discounted (LOO lambda), primitive
composition (Adam 3000 steps, 6 channel types), causal 1-NN.

**Verdict: TASK_POPULATION_VOID.** Competence gate failed (washed 0.79, raw 0.72,
need 0.95). PFC transport clean (coverage 1.0, coherence 0.002) but loses to all
baselines except causal kNN. Parser best at TV 0.019 vs PFC at 0.038.
Defects below gate (0.03 < 0.05 required). Advantages all negative.

**What was learned (Codex evidence gate corrected):**
(1) Competence failure is garden-specific (all failing arms target garden;
kitchen/office are 1.000) — answer-token calibration, not sequence-length capacity.
(2) "Washout erases signal" unsupported — raw-to-washed decrease only 0.006 TV
(CI crosses zero); one panel increases. At most attenuation, not erasure.
(3) Individual K edges improve ~0.006 TV over identity (significant), but composed
K_L·K_R is worse than identity by 0.008 TV (CI [−0.015, −0.002]). Composition
destroys individual edge gains. Panel-additive residual (0.020) beats K (0.024).
(4) Stochastic-K vehicle closed. Next: PMO-0 (path-memory observability via
common-suffix distinguishability, no washout, competence staircase).

---

## PSR-v2 — Corrected Predictive-State Adjudication on Finch-3B (2026-09-02; INVALID_PSR_V2 per Codex evidence gate)

Distance from claim: 0 (the refined quotient and transition law ARE native math).
Runner: `experiments/run_psr_v2.py`. Results: `experiments/results/psr_v2/`.
Ledger: `psr_v2_launch`, `psr_v2_complete`, `psr_v2_evidence_gate`.

**Method:** Attempted corrected adjudication addressing PSR-v1 evidence gate
issues. Frozen train/test split (d0+d1=63, d2=54 eval). Argmax quantization
(deviated from approved Q16). Construction-only transition table. Full null
ladder: parser, kNN, last-action, shuffled. 702 model calls.

**Raw result:** 21 classes, 6 suffixes, right congruence reported 0 violations.
Coverage: 19/54=35.2%. kNN TV=0.1090 vs quotient TV=0.1183 on 19 covered rows.
Action descent: 0/0 (untestable). Within-class TV: 0.1542 (mean), 0.4998 (max).

**Codex evidence gate: INVALID_PSR_V2 — not a valid scientific negative.**
Multiple deviations invalidate confirmatory adjudication: (1) horizon safety
violation (d1 suffixes query d2 behavior); (2) argmax instead of approved Q16;
(3) right congruence "0 violations" VACUOUS — all 9 d0 anchors singletons, no
eligible pairs; (4) action descent 0/0 = untestable; (5) coverage 35% vs
required 90%; (6) no actual recurrent-state substitution; (7) kNN comparison
target-informed; (8) wrong statistical test.

**Licensed sentence (Codex, verbatim):** PSR-v2 is an invalid confirmatory
adjudication, not a valid negative result: the implemented argmax quotient used
out-of-horizon response laws, its zero-violation right-congruence diagnostic was
vacuous because all depth-0 anchors were singleton classes, composition covered
19/54 histories, and action descent and causal substitution were not tested; RCQ
is therefore closed on this task by the bounded-round protocol, not empirically
falsified.

**What we learned:** (1) Horizon-custody law required. (2) Nonvacuity certificate
essential: 0/0 = N/A. (3) Argmax cannot define predictive equivalence. (4) Fair
evaluation needs common denominator and task-clustered inference. (5) Response
similarity ≠ causal substitution. (6) Need multiple histories per operational
state. RCQ closed on entity-location tracking by protocol, not falsified.

---

## PSR-v1 — Predictive-State Refinement on Finch-3B (2026-09-02; EVIDENCE-GATE NO-GO)

Distance from claim: 0 (the refined quotient and transition law ARE native math).
Runner: `experiments/run_psr.py`. Results: `experiments/results/psr_v1/`.
Ledger: `psr_v1_launch`, `psr_v1_complete`, `psr_v1_evidence_gate`.

**Method:** Nerode-style counterexample-guided refinement. Start with Γ = {2
direct queries}. Build behavioral quotient (greedy clustering, mean TV ≤ 0.10).
Check right congruence: for each class, verify all members' post-action
successors land in the same class. On violation, add action+query suffixes
to Γ, recompute.

**Budget:** 3 rounds, 2000 calls max, 80 states max. Used 1638 calls.

| Round | Γ size | Classes | Violations | New suffixes |
|-------|--------|---------|------------|--------------|
| 1     | 2→12   | 50      | 9          | 10           |
| 2     | 12→14  | 41      | 9          | 0 (exhausted)|
| 3     | 14     | 41      | 9          | 0            |

**Raw result:**
- 41 classes from 117 histories (2.9:1 compression, 14 singletons)
- Selected-coverage composition: 12/19 (35 histories uncovered)
- Common-denominator: 12/54=22.2% vs parser 8/54=14.8% (p=0.229, NS)
- Right congruence: 9 violations remain

**Codex evidence gate (NO-GO on headline claim):**
1. ~~48.3pp surplus~~ WITHDRAWN — denominator mismatch (12/19 vs 8/54)
2. Not valid Nerode refinement — classes merged (50→48→41) due to greedy TV
   averaging with new low-distance suffixes diluting existing differences
3. Comparison unfair — quotient uses 14 suffixes + learned lookup vs parser
   using only ground-truth abstract state
4. Coverage selection bias — 19 testable cases are mechanically easy ones
5. Simpler controls not run (kNN, memorization, last-action, shuffled)
6. Proposed full-table fix trains on test answers — unsafe

**Licensed sentence:** "In a transductive selected-coverage screen, a
thresholded response-signature partition correctly predicted 12 of 19
covered two-action class labels; no predictive surplus or quotient action
law is established."

**What we learned:**
1. Path-conditioned response signatures contain reusable transition info
2. 41 distinguishable behavioral states exist (vs 9 abstract), but greedy
   order-dependent clustering ≠ genuine equivalence relation
3. The composition test must use identical denominators and proper train/test
4. Greedy exemplar clustering is non-transitive — need proper refinement
5. Positive exploratory evidence justifies ONE corrected adjudication

---

## RCQ-0 — Real Causal Quotient on Finch-3B (2026-09-02; IN PROGRESS)

Distance from claim: 0 (the quotient and action law ARE the native math).
Runner: `experiments/run_rcq0.py`. Config: `experiments/config/rcq0_v1.json`.
Spec: `theory/closed/REAL_CAUSAL_QUOTIENT.md`. Ledger: `rcq0_substrate_confirmed`,
`rcq0_v1_composition_fail`.

**Substrate selection:**
Tested RWKV-4 (169M, 430M, 1.5B), RWKV-6 Finch (1.6B, 3B). Entity
discrimination gate (asymmetric-state Q0-Q1 TV > 0.10) confirmed only on
Finch-3B (TV=0.43). RWKV-4 at all sizes tracks location dominance, not
entity-specific state (TV=0.03). Finch-1.6B marginal (TV=0.08). State replay
bit-exact (TV=0.000000). Entity swap via state injection confirmed (TV=0.41/0.46).

**Task:** 2 entities × 3 locations = 9 joint states. 6 macro-actions. 2 probes.
Teacher-forced log-likelihood scoring.

**8 iterations, varying quotient construction:**

| Run | Method | Classes | Comp Top-1 | Trans Cons | Sub Same |
|-----|--------|---------|------------|------------|----------|
| 1 | δ=0.05 hard | 35/36 | 0/0 | n/a | n/a |
| 2 | δ=0.10 hard | ~35 | 0/0 | n/a | n/a |
| 3 | TV 0.10 + moved | 26 | 0.25 | ~0.85 | ~0.08 |
| 4 | TV 0.10 direct | 17 | 0.47 | ~0.88 | ~0.06 |
| 5 | TV 0.15 phrased | 15 | 0.34 | 0.88 | 0.07 |
| 6 | TV 0.05 word-order | 16 | 0.43 | 1.00* | 0.02 |
| 7 | GT 9-state (4 phr) | 9 | 0.51 | 0.86 | 0.13 |
| 8 | GT 9-state (1 phr) | 9 | 0.65 | 1.00* | n/a |

(*) Trivially 1.0 due to single member per class.

**Key finding: path dependence.** Within-state word-order TV: min=0.0004,
max=0.2948, median=0.09. Post-action distributions differ from direct-statement
distributions by TV~0.25. The model encodes HOW information was presented.

**What we learned:**
1. Entity discrimination emerges between 1.6B and 3B in RWKV-6
2. The model has genuine entity state (discrimination + injection + substitution)
3. The 9-state quotient has real structure but cannot compose (gate: 90%)
4. Path dependence prevents discrete quotient composition
5. No parser surplus: quotient TV=0.24 vs parser TV=0.19
6. Affine behavioral dynamics under investigation

---

## HANDLE-mu Rung 1 — Causal Handle Algebra, distance-1 (2026-09-01; PIPELINE-INVALID)

Distance from claim: 1 (designed latent world). 7x7 key-lock grid, 5 causal handles
(2 keys, 2 locks, goal; agent excluded), partial visibility (Manhattan r=2), observation
identity permuted per step. Five architectures: dense typed slots (6x32, all-pairs
messaging, 20,927 params), learned-sparse slots (top-2, 20,927), flat GRU (19,930),
historyless (10,207), direct-state oracle. Training: next-obs MSE + next-event CE,
40 epochs, 3 seeds (42, 137, 2026), 2048 train / 512 val / 1024 test trajectories.
Seven pre-registered gates. Runner: `experiments/run_handle_mu.py`.
Spec: `theory/closed/HANDLE_MU.md`. Ledger: `handle_mu_rung1`.

**Prediction metrics (event macro-F1 / status macro-F1):**

| Model | Seed 42 | Seed 137 | Seed 2026 |
|-------|---------|----------|-----------|
| Oracle | 0.990 / - | 0.996 / - | 0.996 / - |
| Dense slots | 0.991 / 0.337 | 0.992 / 0.348 | 0.996 / 0.330 |
| Sparse slots | 0.988 / 0.330 | 0.991 / 0.345 | 0.994 / 0.335 |
| Flat GRU | 0.682 / 0.343 | 0.383 / 0.218 | 0.386 / 0.245 |
| Historyless | 0.655 / 0.299 | 0.814 / 0.297 | 0.915 / 0.297 |

**Gate results:**

| Gate | Seed 42 | Seed 137 | Seed 2026 |
|------|---------|----------|-----------|
| Eligibility | FAIL | FAIL | FAIL |
| Causal consumption | FAIL | FAIL | FAIL |
| Shielding | PASS | PASS | PASS |
| Timing | PASS | PASS | PASS |

**Overall verdict: PIPELINE-INVALID (Codex R3).** Not a scientific negative — five
protocol bugs invalidate the run. Eligibility never passes (status F1 ~0.33, need
>=0.90). Causal consumption improvement = 0 or negative across all seeds. Bounded
spec/runner repair justified.

**What we learned:**
1. **MSE loss drowns status signal.** Status is 4/27 dims of the observation record,
   trained with MSE. Status changes are sparse events. Model predicts majority class
   (status_acc ~0.69, macro-F1 ~0.34). Needs separate CE head for status.
2. **Flat GRU bottleneck.** Parameter-matching forces flat GRU to hidden_dim=33 vs
   slots' effective 192-dim state. Flat GRU is WORSE than historyless in 2/3 seeds
   (0.38 vs 0.81-0.91). The "within 3 points" gate is structurally impossible.
3. **Events are too predictable.** Historyless achieves 0.91 event F1 (seed 2026) —
   events can be predicted from current observation alone. Undermines need for temporal
   causal tracking.
4. **Slot swap targets wrong slot 88% of the time.** Identity permuted independently per
   trajectory means donor slot N and recipient slot N usually carry different handles.
   Codex R3 diagnostic: same_numeric_slot = 11/93 pairs.
5. **Shared suffix not actually shared.** The paired history finder uses donor actions as
   suffix but recipient trajectory has different actions. zero full-suffix matches.
6. **Events identical at contact 94% of the time.** Only 5/88 contacted pairs have
   different events at the contact step. Most events are "none."
7. **Only keys (handles 0-1) have paired histories.** Locks and goal never independently
   differ between same-level trajectories.

---

## FBA-0 — Factored Bottleneck Architecture campaign (2026-09-01; FAIL)

Distance from claim: 1 (engineered-factorization control). Synthetic Z/8×Z/4 POMDP, 16 opaque actions, T=3 episodes, 85/15 confusion matrix. Six architectures: FBA (16/16, 33,664 params), flat GRU (40K), flat-matched (33K), asymmetric split (24/8, 34,688), modular (4×8, 34,688), flat bottleneck (37,760). Three seeds (42, 137, 2026). 2000 epochs, cosine LR (1e-3→1e-5), batch 256. Kill gates: K4 (all train≥0.90), K6 (FBA>best flat≥20pp), K7a (FBA>asym≥15pp), K7b (branch interchange>historyless null with Bonferroni CI), paired effects. Response-class equivalence split: 21/3/8 (train/val/test) from 32 classes. Oracle ceilings: historyless ~72-73%, recurrent ~95-96%.

Runner: `experiments/run_fba_0.py`. Theory: `theory/closed/FBA_BRIDGE.md`. Ledger: `fba0_full_campaign`.

**Test accuracies:**

| Model | Seed 42 | Seed 137 | Seed 2026 |
|-------|---------|----------|-----------|
| FBA (16/16) | 0.847 | **0.915** | 0.897 |
| Flat GRU (40K) | 0.802 | 0.832 | 0.847 |
| Flat matched (33K) | 0.796 | 0.804 | 0.775 |
| Asym split (24/8) | **0.924** | 0.828 | 0.924 |
| Modular (4×8) | 0.891 | 0.805 | 0.847 |
| Flat BN (37K) | 0.884 | 0.812 | **0.926** |

**K7b branch interchange (cross-accuracy, null=0.722):**

| Seed | Cross | CI | Orient | Verdict |
|------|-------|----|--------|---------|
| 42 | 0.630 | [0.571, 0.687] | A=place,B=fiber | FAIL |
| 137 | 0.884 | [0.846, 0.923] | A=fiber,B=place | PASS |
| 2026 | 0.414 | [0.355, 0.481] | A=fiber,B=place | FAIL |

**Overall verdict: FAIL.** Joint predicate 0/3 seeds. K7b passes for seed 137 only.

**What we learned:**
1. Architecture matters: structured > flat consistently, but WHICH structure wins is seed-dependent.
2. The 16/16 symmetric split is not robustly optimal — asymmetric 24/8 wins 2/3 seeds on test accuracy.
3. Branch interchange is possible (seed 137: cross=0.884) but does not emerge consistently across seeds. The response-class split (which classes land in train vs test) determines whether factored representations develop.
4. Pre-registered kill gates (20pp, 15pp gaps) were too aggressive — actual effects are 4-9pp.
5. Wrong-channel controls (same-factor preservation) consistently pass (0.93-0.98 range) — the branches carry factor-relevant information even when cross-episode hybrids fail.
6. Flat bottleneck (37K) can match or beat FBA (seed 2026: 0.926 vs 0.897), suggesting bottleneck constraint alone, without independent branch updates, is sometimes sufficient.

**Transferable residue:** Width allocation (how many dims per branch) matters more than product factorization. Alignment between architectural priors and task structure determines whether factored representations emerge. This informs next-direction choices about typed vs untyped architectural constraints.

---

## OCI/RAC — Real-entity activation steering (2026-09-01; CLOSED per Codex round 10)

Qwen3-0.6B-Base, frozen, CPU-only. Real capital-city entities (Tokyo/Japan, Berlin/Germany, London/UK, etc.). Hidden-state transplant and function-vector composition at B20. Distance-from-claim: 1 (activation steering is R^n scaffolding, not native math; signals inform the next constructive program).

- **OCI-001 through OCI-003** — Operational positional-carrier confirmation. Cross-template routing follows ordinal clause position (96.5%), not token position. Sentence-position routing at B20 confirmed across 3 disjoint panels, 192+ transplant rows.

- **RAC-0 — POSITIVE (bounded)** (`results/rac_0/verdict.json`; `run_rac_0.py`). First successful composition in 75+ experiments. Two orthogonal function vectors (routing=position rv, relation=capital/language rlv; cos=-0.046) independently and jointly control retrieval. Composition generalizes 100% to held-out entity pairs. Survives 5 layers of separated computation (B15-B20). Layer-phase deformation: works B15-B21, FAILS at B22 (content commitment transition).

- **RAC-1 — FAIL (Codex round 9; algebra program CLOSED)** (`results/rac_1/verdict_affine.json`; `run_rac_1.py`). Affine coordinate-overwrite setters: S_v(h) = h + direction * (tau - c_dim(h)). True mathematical idempotence (48/48). correct_top1 composition 21/36 (58%) — binding metric per spec. Metric substitution error caught by Codex: comp metric (32/36, 89%) was p_target>p_start, not top-1. Same-layer commutativity tautological (affine construction, rv perp rlv). Transport: ALL DEFORM. Codex ruling: "useful local coordinate actuator, not a response quotient carrying a product algebra."

- **Codex direction round 10 — END FROZEN-RESIDUAL MEASUREMENT LANE.** Jacobian rank experiment REJECTED (already run as CPD-001, R^n trap; full rank 40/40, effective rank 6-28). Terminal diagnostic prescribed: Gate I (logit-additive null) + Gate F (response-fiber stability). Outcome table governs closure characterization.

- **Terminal diagnostic — CLOSE AS DECODER/LOGIT COMPETITION** (`results/terminal_diagnostic/verdict.json`). Gate I: logit-additive null l_a+l_b-l_0 explains 99.1% of composition effect (sqrt(JSD) = 0.0001 between predicted and actual composed distributions; 12/12 top-1 match). The "composition" in RAC-0/RAC-1 is ordinary logit addition through a linear unembedding — decoder competition, not nonlinear response geometry. Gate F: NO_INTERFACE — no template pair has baseline sqrt(JSD) < 0.15 (smallest gap: 0.19). The model's response law is presentation-dependent; no template-invariant response fiber exists. Per Codex outcome table: CLOSE AS DECODER/LOGIT COMPETITION. Frozen-residual measurement lane ends.

- **Codex direction round 11 — DIRECTION DIALOGUE (round 1/3).** Codex recommends stopping the frozen-model discovery program (agreed). Provides "genuinely different" criteria table: agent state (not residual vector), native actions (not external edits), behavioral identity (not cosine/PCA), persistent environments (not prompt templates), composition surviving time and relabeling. D1-D9 should be design contracts. Bar: "not induced by or dependent upon chosen vector coordinates." Three options: (a) archive, (b) falsification-methods paper, (c) new stateful-substrate design project. Codex leans toward (b) then (c). Pushback: option (c) IS the FBA constructive program per AGENTS.md second lens.

- **Codex direction round 12 — CONDITIONAL YES (round 2/3).** FBA IS option 3. Four additions required: (1) partial observability/recurrence, (2) response-law definitions for place/fiber, (3) four-way comparison (flat, FBA, scrambled FBA, modular), (4) strengthened K7 ablation. Tautology risk: architecture and world share factorization. Falsifier genuine. Correct claim: "behaviorally grounded structural prior enables compositional generalization." One more spec-hardening round before compute.

- **Codex direction round 13 — SPEC HARDENING (round 3/3, in progress).** All four additions addressed. Architecture updated with independent encoding, four baselines designed, partial observability via noisy observations, K7 strengthened. GO/NO-GO decision pending.

---

## Phase 3 — Co-designed dynamical carrier (2026-09-01 →)

Build a learned carrier whose internal states compose under causal transplantation. Distance-from-claim: 0 (the carrier IS the artifact).

- **endogenous_action_carrier_v1 — STOPPED (registered PASS, Codex ruling: architecturally tautological)** (`results/endogenous_action_carrier_v1/evidence.json`, `verdict.json`; Codex ruling: `scratchpad/codex_eac1_next_rung.txt`). 150K CPU forwards, 62 seconds, 359K params. Structured world encoding (key-value transition triples) + differentiable attention lookup + context-dependent readout. All 7 registered gates PASS across 3 seeds: acc 99.4-99.8%, self-patch 0.0, same-place 100%, descent 95.6-100%, three-way 100%. **Codex review determined result is tautological:** carrier IS the next-state embedding; host/donor share the same world; no donor action representation is produced or transplanted; a symbolic table lookup achieves identical results. The causal gates validate the differentiable associative-memory construction, not a compositional action carrier or native mathematics. **EAC/LAC line STOPPED.** A non-tautological continuation would require hard access control, no raw successor embeddings, independent relabelings, held-out compositions, and matched controls — a new preregistration.

- **learned_action_carrier_v0 — PORTABLE_NOT_COMPOSABLE (terminal)** (`results/lac0_run_20260901_064645/`; Codex design gate: `scratchpad/codex_init_gate_response.txt`). 739K-param typed neural machine: WorldWriter→M, ActionWriter→A, Composer(GRU)→μ(a1,a2), gated 2-block Executor(M,A)→M', Renderer(M',legend)→logits. 730K-param untyped transformer control. 5000 steps × batch 128 × 3 seeds, CPU only. **Claim-bearing run (Xavier init):** seed 42: prim=100%, held_comp=33.83%, portability F=1.0, sequential=33.8% FAIL; seed 137: prim=100%, held_comp=14.82%, sequential=11.6% FAIL. Untyped: 12.5-13.4% (chance). **Cross-eval of provisional checkpoint (default init, Codex):** held_comp=96.11% BUT sequential_agreement=0.8% FAIL (gate ≥90%). **Critical finding: circuit selection.** Default init learns endpoint composition but NOT sequential execution (E(E(M,a1),a2) = 0%). Xavier learns sequential execution (94.8%) but NOT composed carrier (34%). These are different optimization basins. Neither passes complete Gate 4. Stop condition §14.7(b): portability without composition is terminal. **Residue:** typed architecture → perfect primitives + portability; untyped cannot; endpoint vs sequential composition are distinct targets; GRU shortcuts ≠ algebraic composition; successor must define composition via response-law quotients.

---

## Phase 5 — PSQ-1: Predictive-State Quotient (2026-09-01, killed at eligibility)

Test whether frozen Qwen3-1.7B-Base can track state in a 64-state two-dial world (Z8×Z8) via Python-completion prompts. Distance-from-claim: 0 (predictive-state quotient geometry IS native math). Codex R1-R3 direction dialogue.

- **psq1_v1_capability — NO_INTERFACE** (`results/psq1_v1/capability_screen.json`). 128 CPU forwards, 217s. 4-shot Python-completion, 2-8 operations per test, separate x/y queries. Overall 53.91% (gate ≥95%). Per-cell: x_0=50.0%, x_1=59.4%, y_0=9.4%, y_1=96.9%. Model almost always predicts "0" for y queries; at chance for x. No state tracking. Stop condition: NO_INTERFACE. No repair per Codex R3. **Note:** the predeclared fallback (Qwen3-8B-Base) was already tested in the earlier PSQ-1 round (see Phase 5 PSQ-1 ALL SUBSTRATES section below) and also failed (55.5%, NO_INTERFACE). Qwen3-4B-Base smoke test (56.2%, 16 items, this session, informal) further confirms: all sizes 0.6B–8B fail.

---

## Phase 4 attempt — KV command-slot transplant (2026-09-01, killed at eligibility)

Test whether frozen Qwen3-0.6B-Base admits modular KV-level command execution. Distance-from-claim: 1 (modular execution is prerequisite for native algebra). Codex direction R1+R2: engineering falsification, not native math.

- **permutation_eligibility_v1 — INELIGIBLE (kills 4-launch program)** (`results/permutation_eligibility_v1/eligibility.json`, `cases.json`). 200 CPU forwards, 318s. Three-item permutation (rotate left/right/reverse) with fixed 3-shot prompt, 8-symbol pool. Overall accuracy 48.0% (gate >=95%), termination 0.5% (gate >=95%). Per-op: rotate left 13.4%, rotate right 34.3%, reverse 97.0%. Model can reverse but cannot track cyclic rotation. Kill condition fires. No Launch 2-4. Direction dead.

---

## Phase 2 — Non-R^n behavioral algebra (2026-08-31 →)

Instrument: logit lens (apply model's final layernorm + unembedding at each layer) + sqrt(JSD) as behavioral distance. Model: Qwen3-0.6B (28 layers, 1024 hidden dim), CPU-only. Synthetic fact-worlds (2- or 3-entity, controlled assignments).

- **query_port_composition_v1 — INCONCLUSIVE_ALLOCATION_STOP** (`results/query_port_composition_v1/evidence.json`, `verdict.json`). 144 CPU forwards (36 cells × 4 arms: clean + self-patch + L21 donor + L25 donor). Theory Section 12. Three families (ZOG/MIP, PLIM/KROT, DREN/VORN), 3 values each (three-way clash: s=v_r, t=v_{r+1}, d=v_{r+2} all distinct). Exact query-position hidden-state row-copy from donor to host at L21/L25. Instrument PASS: carrier ok, self-patch max=0.0, clean 94.4% (cluster LB=0.86), L21 viability 94.4%. **Composition gates all FAIL:** F(21)=0.306 [0.139] (at 3-way chance), C(21)=-0.062 [-0.201, 0.077], W(21)=0.125 [-0.024, 0.273]. Donor wins 58% at L21 (D21=0.583), source only 11%. L25: C(25)=-0.701 (pure donor-verbalizer copy). Localization L=0.639 [0.548] PASS — L21 is NOT the late verbalizer, yet still fails to compose. No named nonpass verdict reached (not enough for DONOR_VERBALIZER_COPY, not enough for HOST_INERTIA). The transplanted query state disrupts the host but does not direct toward the target answer — it scatters across the triplet with partial donor lean. **Terminal per Codex allocation ruling. Ends the Qwen prompt micro-world.**

- **observational_selectivity_quotient_v1 — OBSERVATIONAL_SELECTIVITY_VERBALIZER_SUFFICIENT** (`results/observational_selectivity_quotient_v1/evidence.json`, `verdict.json`). 48 CPU forward passes, 28-layer logit-lens profile, 12 undirected edges, 10,000 bootstrap resamples. Theory Section 11 (Codex design gate). Purely observational --- no bypass, no intervention. Measures layer-resolved selectivity contrast S(l) = E[d_rel(l) - d_irr(l)] using sqrt(JSD/ln2) at all 28 post-block layers. All 11 core gates PASS: integrity (d=0.00023), material (12/12), early null (B=0.008), window emergence (Gw=0.239 [0.137]), anchor G(25)=0.633 [0.554], R(25)=0.785 [0.712], OSQ(25)=0.706 [0.612], peak=L25, onset=L24, persistence S(27)=0.240 [0.184], presentation (std=0.644 [0.506]/rev=0.638 [0.512]), family (ZOG_MIP=0.664, PLIM_KROT=0.696, HESK_VORN=0.563). **Verbalizer null NOT rejected (V=1.01):** coarse-graining to 3-bin (v0, v1, rest) accounts for 100% of the selectivity signal (S^pi(25)=0.647 vs S(25)=0.641). The selectivity is entirely ordinary answer-token routing. R^n trap: escapes (no cosine/PCA/Euclidean in estimand). Claim ceiling: observational logit-lens measurements exhausted. Terminal.

- **endogenous_response_quotient_v1 — INVALID_OR_NO_PROPAGATION_CONTROL** (`results/endogenous_response_quotient_v1/evidence.json`, `verdict.json`). 144 CPU forward passes (48 cells x 3 arms). Theory Section 10. Compares block 25's native computation with identity bypass (skip block, pass pre-block state unchanged). No donor transplant --- purely endogenous. Three families, 4 worlds, 2 queries, 2 declaration orders. Instrument PASS (noop d=0.0). Material PASS (12/12). **Bypass viability FAIL (7/48)** --- identity bypass too destructive; off-manifold lesion. No scientific verdict. Descriptive: O-endpoint shows query-selective block-25 action (A_O=0.254 [0.198], C_O=0.174 [0.086], Sigma_O=0.428 [0.304]). F-endpoint amplification survives (A_F=0.159 [0.111]), compression does not (C_F=0.013 [-0.017]). Stability clean (ub=0.038). Terminal.

- **commitment_hysteresis_v1 — INCONCLUSIVE_ALLOCATION_STOP (Codex evidence gate: REVISE, corrections adopted)** (`4593ab0`; `results/commitment_hysteresis_v1/evidence.json`, `verdict.json`). 456 CPU forward passes (24 clean + 48x9 intervention). Tests whether prefix transplant at L21 leaves a causal trace that survives restoring the original host prefix after the commitment bottleneck (L25). Three families (ZOG/MIP, PLIM/KROT, HESK/VORN), 4 worlds each, 48 directed edges, 12 clusters. Controls perfect: self-patch 0.00, C25 (full restore) 0.00, all 12 clusters eligible. Corrected metrics (Codex independent reduction): **M=0.438 [0.383, 0.504] PASS**, **T=0.200 [0.144, 0.247] FAIL**, **H=0.403 [0.322, 0.512] PASS**, **U=0.092 [-0.005, 0.169] FAIL**, **L=0.162 [0.091, 0.222] FAIL**. Original reducer averaged all queries instead of using changed-entity query for T/U/L per Section 9. Under this exact whole-prefix intervention, an L21 transplant produced a response-law shift that remained after the clean host prefix was restored at L25. The corrected preregistered reduction cleared the descriptive M and H thresholds but did not clear the donor-directed T/U or localization L lower-bound gates. It does not establish absence of donor-directed transfer, generic disruption, or commitment-specific hysteresis. High H is compatible with ordinary causal propagation from replacing an entire upstream prefix. Terminal — do not mine high-T edges or advance to held-out names.

- **signature_restatement Phase 4d terminal anti-echo factorial — NO_INTERFACE, allocation stop (Codex evidence gate UPHOLD)** (`experiments/run_signature_restatement_v1.py --phase4d-only`; `results/signature_restatement_v1/phase4d_results.json`). 2304 forward passes (2208 factorial + 96 base-signature). Integrity passed. **Gate 1 (interface) FAIL:** confirming records followed on 47/48 and 44/48 coordinates; contradicting records on only 21/48 and 8/48. **Commitment-congruence asymmetry** (bounded discovery): append is nearly identity-like when congruent, unreliable overwrite otherwise. **Gates 2-4 unreached** with descriptive results: crossed-order recency rule did not pass (recency generally not excluded); alias logit shifts 3.62/2.39 nats establish keyed alias-content sensitivity below reliable argmax control. Terminal: prompt-renderer tuning ends; S^G receives no semantic upgrade. Anti-echo question remains inconclusive.

- **signature_restatement_v1** (`b70aac2`; `results/signature_restatement_v1/results.json`). Tests O1: does representative-independent restatement S^G_g from observable greedy signatures exist? **O1 RESOLVED: YES.** Idempotence: 100% greedy (96/96), JSD 0.077/0.071. Descent: 12/12 (100%) / 15/15 (100%) — fixes S^W's one failure. Place preservation 100%. **Fixed cyclic shuffle 0/32 (pairing-sensitive; textual echo unresolved).** Non-naturality persists with S^G: JSD 0.193/0.188. **NEW FINDING:** Correction does NOT descend to G (58%/80%). Typed square is pointwise K(Cx) vs C(Kx), not quotient-level. Pattern: order-dependent — `_rev` representatives often ignore correction while `_std` accepts it. Correction effectiveness is itself presentation-path dependent. **Anti-echo alias control (Phase 4c, Codex REVISE): no anti-echo evidence.** Faithful alias did not exceed comparators by predeclared 30pp margin. Direct R(g) recovered 47.9%, ruling out deterministic latest-explicit-assignment. Implementation defects in shuffled arm. Literal signature renderer preserved 32/32 greedy signatures — do not call semantic or latent-space invariance.

- **predictive_fiber_action_v2** (`dc7d4ae`; `results/predictive_fiber_action_v2/results.json`). Codex v6 corrected experiment. Fixes v1 construction error: typed square S_{p'}.C vs C.S_p with both paths ending at corrected world. Full greedy signatures. Descent tested. **CONFIRMS COUPLING**: idempotence 100% greedy (JSD 0.070); typed square JSD 0.208 (registered 89.6% greedy, held-out 70.8% greedy). Held-out commutativity WORSE than v1's invalid test. Coupling is genuine. Descent: empty 100%, restatement 92-100%.

- **predictive_fiber_action_v1** (`e4e8ba7`; `results/predictive_fiber_action_v1/results.json`). *(Superseded by v2.)* Construction error: old-world restatement on both paths. Idempotence valid; square test invalid. Codex v6 review identified the error.

- **predictive_fiber_v1** (`4978a85`; `results/predictive_fiber_v1/results.json`). Codex-directed (v4). Tests whether distributional residual inside greedy fibers is predictive or presentation. Three pair classes, six continuations. **MIXED**: history>benign at baseline+corrections (3/3), history<benign after restatement (0/3). Cross-world smallest. Residual = predictive commitment + resetable presentation. Canonical restatement = synchronization.

- **rebroadening_test_v1** (`915988c`; `results/rebroadening_test_v1/results.json`). Re-broadened distribution is **MEANINGFUL**: 5-7/10 divergent tokens are history-related entity values. Model leaks entire fact-world into output. Repetition narrows 2-3x.

- **entropy_structure_v1** (`cc83d06`; `results/entropy_structure_v1/results.json`). **COMMITMENT BOTTLENECK**: entropy 0.05-0.30 bits at L24-25 (top-1: 0.999), re-broadens to 5.5-7.7. Explains greedy congruence (97%) + distributional failure (0%).

- **distributional_congruence_v1** (`048bae1`; `results/distributional_congruence_v1/results.json`). Full-distribution congruence (sqrt(JSD), threshold 0.05). **0/96 congruent**. JSD 0.07-0.45.

- **position_contribution_v1**, **three_fact_resolution_v1**, **mlp_decomposition_v1**, **attention_control_v1**, **continuation_congruence_v1**, **logit_lens_resolution_v1** — see STATE.md for details. Key findings: resolution (L21-25, up to 62x selectivity), not attention routing (r<0.25), whole-sequence value-space operation, generalizes to 3 facts, 97% greedy congruence.

- **fusion_fission_v1-v2b** — compositional structure series. Cosine blind; behavioral transplant sees structure; whole-state property; K-matrix instrument has defects (Codex v2b review).

- **causal_resolution_v1** — UNINFORMATIVE (full-state injection trivial).

- **predictive_setter_algebra_v1** — FAIL (16 types, primacy bias).

---

## Phase 1 record (2026-08-27 → 2026-08-31; PROGRAM REOPENED 2026-08-31; Phase 1 experiments below are closed results; Phase 2 begins with non-R^n approaches; audits #27–#50 adopted; orientation document `docs/HANDOFFS.md` is historical)

- **Theory (restart 2026-08-30; PROGRAM REOPENED 2026-08-31).** Adopted foundation: `theory/AXIOMS.md` D1–D9, Theorems 1/4/7/8, Propositions 2/6, native bridge definition (mathematics audits #42 and #44; D7 audit #48 REVISE adopted; licensed wording in `STATE.md`). Phase 1 Codex review verdict was STOP, but closure was not authorized by the project owner; native math existence is axiomatic (see `feedback_native_math_is_axiom.md` in memory). Five transferable insights deposited; Phase 2 begins with genuinely non-R^n approaches. The bullets below are retained as Phase 1's closed record; their "program remains paused / on restart" language is historical.
- **`register_bridge_preflight_v1` — PREFLIGHT PASS — EXPLICIT-LEGEND STATE LINEARLY DECODABLE (audit #39: UPHOLD; noncausal feasibility result, not a code-level or causal bridge)** (locked `8beb8e9`; result `d2acfe8`; ledger `register_bridge_preflight_v1_lock`, `register_bridge_preflight_v1_result`, `register_bridge_preflight_v1_audit39`; config `experiments/config/register_bridge_preflight_v1.json`; runner `experiments/run_register_bridge_preflight.py`; results `experiments/results/register_bridge_preflight_v1/`). `register_bridge_preflight_v1` — PREFLIGHT PASS: EXPLICIT-LEGEND STATE LINEARLY DECODABLE. In frozen Qwen3-1.7B-Base, cross-fitted rank-≤8 Ridge decoders evaluated on held-out entities × templates × a disjoint balanced permutation bank achieved 0.815 accuracy (entity-bootstrap LB 0.779; folds 0.828/0.852/0.766; minimum state recall 0.615), versus input-embedding 0.125, categorical 0.135, paired reassigned-legend original-state 0.016, and shuffle-null p99 0.204; the intact decoders followed the paired legend’s newly denoted state at 0.852, ruling out fixed tag identity. This is a noncausal explicit-legend lookup signal, not a code-level or causal bridge. The program remains paused; on restart, conduct the required direction dialogue and then one held-out causal residual-to-writer-centroid injection test, with no synthetic staircase advance or further decoder sweep first. Entry below.

- **NLM-007 — CLOSED** under the program's terminal allocation rule, not by a
  scientific null (audit #22 closing statement, verbatim in `STATE.md`; ledger
  `nlm007_closed_audit22`). Terminal ladder: 34a raw CONTINUE -> 34a static
  CONTINUE -> 34b INCONCLUSIVE (terminal rung) -> 34c not run. No operational
  state, native law, composition, representation-level hostile hole, or
  independent replication was identified. Every bullet under the NLM-007
  heading below is the closed record; its queue/order language is historical.
- **Toy quotient-world program (Rounds 36–37) — ENDED 2026-08-29** under the
  governance amendment in `AGENTS.md` (exact certificates are diagnostics
  only; one audit per result, which must answer "should this continue"; real
  models only; ratio tripwire). Rounds 36 v1 / 36b / 36c / 36d and Round 37
  are closed results with one licensed reading each (audits #23–#26 and the
  Round 37 audit; verbatim in `STATE.md` "Closed toy program — licensed
  wording"). No learned artifact passed the complete exact reducer (only the
  oracle fixture); Round 37: `NO ARCHITECTURAL WIN`. Runners and configs were
  retired from the active path (`f6dac0e`; git history); verdicts retained
  under `experiments/results/operational_quotient_*/` and
  `experiments/results/presentation_quotient_v1_*/`. Entries below.
- **Real-model line (2026-08-29) — frozen-residual-stream constructions
  STOPPED as an allocation pivot (audit #28), not a scientific conclusion.**
  `coordinate_v1` UNINTERPRETABLE — INVALID POLARITY BASELINE; `coordinate_v2`
  KILLED at its baseline gate; `coordinate_v3` mechanics reproduced but
  audited (#27) into a narrow late lexical-control effect; `interchange_v1`
  closed by its locked raw-sign baseline, no swap arm ran. Verbatim
  replacement wording (audit #28): "Under the current apparatus budget, this
  sequence of frozen residual-stream constructions is not yielding a
  native-mathematics artifact, so open-ended layer, task, and decision-rule
  repair stops here. This is an allocation pivot, not evidence that frozen
  pretrained residual streams lack usable native structure." Entries below.
- **`state_bus_v1r1` — REGISTERED FAIL, CLOSED (audit #29; ledger
  `state_bus_v1r1_result`, `state_bus_v1r1_audit_stage`,
  `state_bus_v1r1_audit29`); no bus v2.** Licensed sentence (verbatim):
  **`state_bus_v1r1` is a fixed-construction FAIL: a 98,400-parameter supervised interface repeatedly injected a 16-dimensional code into frozen Qwen3-1.7B-Base, and across three seeds held-out same-state donors preserved every four-way categorical choice but failed the training-derived confidence-signature tolerance on 15–16/16 rows, while fixed-cycle cross codes changed taxonomy choice on 7/16 rows—always cat→dog in 3/4 and cow→horse in 4/4—and the complete registered gate vector also failed heldout taxonomy and the cross-consequence third/first movement criterion; taxonomy verbalizers were absent from the bus loss, but the all-pair sensitivity was donor-verbalizer-specific rather than state-general, so the licensed residue is a repeatedly maintained supervised response controller with pair-specific lexical/semantic steering, not autonomous persistence, abstraction, general interchangeability, or native latent mathematics.**
- **`interchange_v2` — REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED** (audit
  #30 adopted verbatim `fe3c541`; ledger `interchange_v2_lock`,
  `interchange_v2_result`); no v3. Licensed sentence and never-say list
  verbatim in `STATE.md` "Current statement".
- **`control_cost_v1` — REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED** (audit
  #31 adopted verbatim `17b3d90`; ledger `control_cost_v1_result`,
  `control_cost_v1_audit31_correction`). Cost-law, rank, ratio and
  asymmetry readings withdrawn as censoring artefacts. Entry below.
- **Frozen-residual line STOPPED (Codex direction round 10, ledger
  `direction_r10_program_ruling`)** — an allocation stop point, not a proof
  that pretrained models lack latent mathematics.
- **`onewrite_state_v1` — KILLED PRE-LOCK** (ledger
  `onewrite_state_v1_killed_prelock`): the base model cannot apply a stated
  rule to visible tags; no state hypothesis was tested. Entry below.
- **`onewrite_recall_v1` — REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED**
  (unanimous, seeds 11/23/37; audit #32 adopted verbatim `3a47fd8`; ledger
  `onewrite_recall_v1_lock`, `onewrite_recall_v1_seed11`,
  `direction_r18_ruling`, `onewrite_recall_v1_result`,
  `onewrite_recall_v1_audit32`). Entry below.
- **`necessity_navigator_v1` — BUILT; audit-#32 blockers FIXED; post-fix
  smoke only; DEFERRED, not cancelled** (round 19; ledger
  `navigator_smoke_r32fixes`, `direction_r19_ruling`). Entries below.
- **`onewrite_recall_rung1` — staircase RUNG 1 — REGISTERED
  CONSTRUCTION-LEVEL FAIL, CLOSED** (unanimous, seeds 11/23/37; audit #33
  adopted verbatim `25f65ac`; ledger `onewrite_recall_rung1_lock`,
  `onewrite_recall_rung1_result`, `onewrite_recall_rung1_audit33`,
  `onewrite_recall_rung1_rung0_diag`, `onewrite_recall_rung1_z_localization`).
  The one-write E/J interface stops under its predeclared rule. Entry below.
- **`oracle_actuator_rung0` — staircase RUNG 0 — REGISTERED FAIL
  (`FAIL — ORACLE ACTUATOR/SITE/RETRIEVAL CONSTRUCTION`), unanimous; audit #34
  ADOPTED (licensed sentence and never-say verbatim in `STATE.md`)** (Codex round 20 design; ledger `direction_r20_rung0_design`,
  `oracle_actuator_rung0_preflight`, `oracle_actuator_rung0_lock`,
  `oracle_actuator_rung0_seed11`, `oracle_actuator_rung0_result`,
  `oracle_actuator_rung0_diag`). Nothing beyond the ledger row is licensed.
  Entry below.
- **Direction rounds 19–20** (ledger `direction_r19_ruling`,
  `direction_r20_rung0_design`): round 19 = navigator comparator rule (self
  = ceiling), navigator deferred, rung 1 built and locked; round 20 = oracle-
  code actuator rung 0 designed as the one next locked artifact, rung 0b
  (encoder on tag-token positions) only after a bounded rung-0 pass. NEXT:
  audit #34 -> Codex round 21.
- **`necessary_register_rung1` — QUALIFIED PASS — ANSWER-SUPERVISED EIGHT-SYMBOL WRITER INTO A FROZEN SYNTHETIC CONSUMER, seeds 11/23/37 (audit #38 adopted `d4b82b7`; wording in `STATE.md`)** (locked
  `ea2e831`; ledger `necessary_register_rung1_lock`,
  `necessary_register_rung1_result`; config
  `experiments/config/necessary_register_rung1.json`; results
  `experiments/results/necessary_register_rung1/run_result.json`). Own
  0.966/1.0/1.0; counterfactual 0.964/1.0/1.0; shuffled-donor 0.967/1.0/1.0;
  telemetry cosine 0.72/0.42/0.49. Entry below.
- **`necessary_register_v1` rung 0 — QUALIFIED INSTRUMENT PASS —
  PERMUTATION-GENERALIZING ORACLE-CODE LABEL SELECTOR** (audit #37 adopted
  verbatim `abee584`; result `ff6c42b`; lock `be28305`; results
  `experiments/results/necessary_register_v1/run_result.json`). Own-code
  first-label accuracy 0.955/1.000/1.000; counterfactual following
  0.962/1.000/1.000; not compositional. Licensed sentence and never-say list
  verbatim in `STATE.md` "Current statement". Entry below.
- **Audits #34–#36 adopted** (`oracle_actuator_rung0` FAIL wording; `site_oracle_v1`
  REGISTERED CONSTRUCTION-LEVEL FAIL, exact recipe closed as an allocation stop;
  `reachability_v1` MEASUREMENT, `NO SLOT-SPECIFIC GEOMETRY CONCLUSION`); the
  frozen-model line is closed; write-up `docs/STRUCTURED_NEGATIVE_2026_08_29.md`;
  licensed sentences and never-say lists verbatim in `STATE.md`.

## PSQ-3α — terminal experiment: task-trained Qwen3-1.7B-Base intervention (2026-08-31; CLOSED NO_INTERFACE)

Per Codex direction round 6 (CONTINUE ONCE, TERMINAL, ARTIFACT-FIRST): trained one seed of Qwen3-1.7B-Base on the Z_8×Z_8 two-dial world (5,000 steps, LoRA r=16, training loss 0.0154), then ran the micro intervention staircase (x-channel panel, 256 calls).

- **Result:** 69.14% accuracy (177/256), gate >=95%. **NO_INTERFACE.** Stopped at stage 1 per predeclared stop rule.
- **Progression:** frozen 0.6B 26.56% (PSQ-3μ) -> trained 1.7B 69.14% (PSQ-3α). Training substantially improves behavioral accuracy but the registered interface gate is not met.
- **Disposition:** per direction round 6, any outcome other than PASS closes the program as currently constituted.
- Runner: `experiments/run_psq3.py`. Config: `experiments/config/psq3_alpha.json`. Result: `experiments/results/psq3_alpha/result.json`. Adapter: `experiments/results/psq3_alpha/adapter_seed42/` (deleted; LoRA weights gitignored). Ledger: `psq3_alpha_result`.

## PSQ-3μ — frozen-model necessary-condition micro-test (2026-08-31; CLOSED NO_INTERFACE)

One-action necessary-condition intervention test on frozen Qwen3-0.6B-Base (CPU only, x-channel probes, S_μ = Z_8 × {0,2,4,6}).

- **Result:** 26.56% accuracy (68/256), gate >=95%. **NO_INTERFACE.** Stopped at stage 1 (256/1,152 calls, 132.6s CPU).
- **Disposition:** per predeclared stop rule, no repair; does not adjudicate full PSQ-3. Audit #49 adopted.
- Runner: `experiments/run_psq3.py`. Config: `experiments/config/psq3_micro_cpu.json`. Result: `experiments/results/psq3_micro_cpu/result.json`. Ledger: `psq3_micro_cpu`.

## PSQ-2 — LoRA fine-tuning for modular state tracking + d_4 geometry measurement (2026-08-30; CLOSED — Codex direction review: STOP hyperparameter staircase)

Two-dial world Z_8×Z_8 (64 states, 4 actions, 2 binary observations). H*=4 (Moore partition saturates at 64 = all states distinguishable). d_4 = d_∞ for this finite world (Corollary 3).

- **PSQ-2 v1** (single-step training only, 512 examples, lr=5e-5, 3 epochs): 60.2% overall, **NO_INTERFACE**. Model learned "not zero" bias (87.5% of training had answer=0). Per-step degradation: 81.2% at 2-step → 40.0% at 8-step. Config: `psq2_v1.json`. Result: `psq2_v1/finetune_and_gate.json`.
- **PSQ-2 v2** (class-balanced 1-3 step, 512 examples, lr=3e-5, 5 epochs, test OOD 4-8 step): 73.4% overall, **NO_INTERFACE**. Per-cell: x_0=65.6%, x_1=84.4%, y_0=62.5%, y_1=81.2%. Length degradation: 86.4% at 4-step → 56.0% at 8-step. Composition doesn't generalize from short to long. Config: `psq2_v2.json`. Result: `psq2_v2/finetune_and_gate.json`.
- **PSQ-2 v3** (class-balanced ALL 1-8 step, 1536 examples, lr=2e-5, 5 epochs, in-distribution test): 75.0% overall, **NO_INTERFACE**. Per-step: 1-step 100%, 2-step 100%, 3-step 78.3%, 4-step 85.7%, 5-step 77.8%, 6-step 46.7%, 7-step 60.0%, 8-step 63.2%. Model learns basic modular operations perfectly but can't compose 6+ steps. Worse on long sequences than v2's OOD (lower lr + diluted data). Config: `psq2_v3.json`. Result: `psq2_v3/finetune_and_gate.json`.
- **Exploratory d_4** (psq2_v3_d4): COMPLETE (DIAGNOSTIC ONLY per Codex direction review). d_4 measurement on v3 adapter at 75% accuracy: rho_model_oracle=0.579, rho_hidden_d4 L6/L12/L18 = 0.259/0.277/0.505, quasiconvex violations L18 = 4.6%. Sub-gate exploratory, not a scientific claim. Config: `psq2_v3_d4.json`. Result: `psq2_v3_d4/d4_measurement.json`.
- **Codex direction review (PSQ-2 round 1)**: Identified metric mismatch (max √JS degenerate for binary oracle), tautological causal test, invalid quasiconvexity surrogate, structured noise. Verdict: STOP hyperparameter staircase. Redesign as PSQ-3 with product metric d_{2,4}, shared Procrustes operators, calibration/held-out split.
- All runs: Qwen3-1.7B-Base (revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`), LoRA r=16 alpha=32. Runner: `experiments/run_psq2_finetune.py`. d_4 runner: `experiments/run_psq1_d4.py`.

## PSQ-1 — few-shot capability screen: ALL SUBSTRATES NO_INTERFACE (2026-08-30; CLOSED)

Two-dial world Z_8×Z_8 with 4-shot Python-completion template. All three substrates failed:
- **Qwen3-1.7B-Base**: 50.0% (always predicts "1"). NO_INTERFACE.
- **Qwen3-8B-Base**: 55.5% (same bias). NO_INTERFACE.
- **Qwen3-8B-Instruct**: 50.0% (4-shot) / 64.1% (balanced 2-shot). NO_INTERFACE.
Root cause: models cannot do multi-step modular wrap-around arithmetic from few-shot prompting. This motivated PSQ-2 (fine-tuning approach). Runner: `experiments/run_psq1_capability.py`. Configs: `psq1_*.json`.

## register_bridge_preflight_v1 — noncausal real-model feasibility measurement: PREFLIGHT PASS — EXPLICIT-LEGEND STATE LINEARLY DECODABLE (audit #39: UPHOLD; noncausal feasibility result, not a code-level or causal bridge) (2026-08-30; audit #39 adopted verbatim)

- **Licensed sentence (audit #39, verbatim).** In frozen Qwen3-1.7B-Base revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, a predeclared rank-≤8 cross-fitted linear decoder read the explicitly legend-denoted state from the two-token record-tag residual under held-out entities, two held-out templates, and a disjoint balanced permutation bank at 0.815 accuracy (folds 0.828/0.852/0.766, entity-bootstrap lower bound 0.779, minimum state recall 0.615), versus 0.125 input-embedding, 0.135 categorical, and 0.016 paired reassigned-legend original-state controls and a 0.204 shuffle-null p99; on the paired reassigned legends the unchanged-tag decoder followed the newly denoted state at 0.852, establishing a noncausal, prompt-family-bounded explicit-legend state signal—not a code-level or causal bridge, persistent register, synthetic-consumer capability, or native latent mathematics.
- **Never say (audit #39).** “Qwen learned a register.” “This establishes a causal bridge.” “The Qwen residual already contains the constructed consumer’s code.” “The state survived, persisted, or was remembered.” “The result establishes an eight-dimensional state subspace.” “The decoder reads tag identity.” “The destroyed arm contains no state information.” “The destroyed context failed.” “Template and permutation transfer were independently established.” “The result generalizes to arbitrary templates, legends, or permutations.” “The shuffle null reran the entire selection pipeline.” “The bootstrap includes decoder-training uncertainty.” “Every layer contains the signal.” “The legend-occurrence reference is a gated control.” “The residuals can drive the frozen synthetic consumer.” “This demonstrates semantic facts, an autonomous state, or native latent mathematics.”
- **Restart ruling (audit #39, verbatim).** **Continue conditionally; the predeclared PASS branch should apply.** The highest-leverage next move is the required 2–3-round direction dialogue followed by one held-out causal bridge test: map Qwen record-span residuals into the successful writer centroids and inject them into the frozen constructed consumer. The program should not continue through another synthetic rung, decoder characterization round, layer sweep, or prompt repair. The current pause remains appropriate until the dialogue produces a locked causal test. The program is still tunnel-visioned around an eight-symbol explicit-lookup micro-world. The PASS justifies one causal discriminator; it does not justify extending the ladder indefinitely.
- Evidence: `run_result.json`, `run_rows.json` (per-row prompts, ids, spans, predictions), `run_features.npz` (local only; sha256 in the ledger result row).

## necessary_register_rung1 — constructed substrate, rung 1 (zero-delay training-entity source writer; replayed rung-0 consumers): QUALIFIED PASS — ANSWER-SUPERVISED EIGHT-SYMBOL WRITER INTO A FROZEN SYNTHETIC CONSUMER (audit #38 adopted `d4b82b7`; wording in `STATE.md`) (2026-08-30; locked `ea2e831`; runner `experiments/run_necessary_register.py`, config `experiments/config/necessary_register_rung1.json`, results `experiments/results/necessary_register_rung1/`; ledger `necessary_register_rung1_lock`, `necessary_register_rung1_result`)

Audit #38's exact licensed sentence, qualifications, and never-say list are reproduced in `STATE.md`'s Current statement. The synthetic staircase stopped after this rung; the only subsequent action was the separately locked noncausal preflight, now governed by audit #39.

## necessary_register_v1 — constructed substrate, rung 0 (oracle write, hard-masked register, unseen label permutations): QUALIFIED INSTRUMENT PASS — PERMUTATION-GENERALIZING ORACLE-CODE LABEL SELECTOR (2026-08-30; audit #37 adopted verbatim `abee584`; runner `experiments/run_necessary_register.py`, config `experiments/config/necessary_register_v1.json`, results `experiments/results/necessary_register_v1/`; commits `805bce2`, `e35b836`, `be28305`, `ff6c42b`, `abee584`)

STATE.md wording, verbatim:

- `necessary_register_v1` rung 0 — QUALIFIED INSTRUMENT PASS. The locked scalar gates mechanically passed in seeds 11/23/37, with own-code first-label accuracy 0.955/1.000/1.000 and same-prompt all-seven counterfactual following 0.962/1.000/1.000 on globally unseen state-to-label permutations. The result establishes a learned synthetic permutation-generalizing oracle-code-to-visible-label consumer, not general composition, source writing, persistence, or pretrained-model structure. Qualification: state/template/panel evaluation was marginally rather than factorially balanced; zero/random results are assigned-target accuracies only; zero-hook is decoded-state identity; and termination/raw rows were not retained for every arm. NEXT: one zero-delay training-entity writer rung, with fully crossed evaluation and complete evidence retention, alongside a cheap real-model source-span bridge-feasibility preflight; then audit once and stop on failure.
  Licensed sentence (audit #37): In this locked synthetic task, three independently initialized two-layer causal transformers trained from scratch used fixed orthonormal vectors injected at a dedicated register position to select the intended abstract state under globally unseen visible state-to-label permutations, with own-code first-label accuracy 0.955/1.000/1.000 and all-seven same-prompt counterfactual-code following 0.962/1.000/1.000; this establishes a learned permutation-generalizing oracle-code-to-visible-label consumer for the tested schedule, not general compositional reasoning, learned writing or persistence, a fully crossed presentation-invariance result, or structure in a pretrained model.
  Never say (audit #37): “The model learned compositional reasoning.” “No lookup strategy can pass the held-out gate.” “The evaluation fully crossed every state, panel, and template.” “Zero and random codes had no systematic effect.” “Zero/random-code behavior was at chance” without “assigned-target accuracy.” “Zero-hook proved exact logit-level identity” or “zero-hook equals the zero-vector arm.” “The mask proved that source information flows only through the register.” “Seed 11 was uniformly robust across presentations.” “The register learned to write, retain, or retrieve state.” “This establishes an abstract state representation in a pretrained model.” “The constructed ladder will transfer to Qwen.” “Instrument valid” without the synthetic, first-label, schedule, and evidence-retention qualifications.
- NEXT (historical; superseded by the rung-1 entry above): `necessary_register_rung1` (round 26 spec + audit #37 requirements: replay/freeze the rung-0 consumers, answer-only source writer at zero configured filler, fully crossed evaluation, raw rows, checkpoints, per-arm termination, logit-level identity, paired clustered uplift, lookup baselines) → one audit → direction dialogue on moving `register_bridge_v1` earlier vs short delay. Alongside (not staircase advancement): a cheap CPU real-model source-span linear-decodability preflight with token/context and shuffled-coordinate controls. Failure closes the constructed architecture; a lookup-only pass also stops advancement. Moves, algebra, effort, maps, long delay, larger models, the navigator and any block-12 actuator stay off.

## oracle_actuator_rung0 — positive-control staircase rung 0 (oracle code, no encoder; actuator/site/retrieval control): REGISTERED FAIL, unanimous; audit #34 ADOPTED (2026-08-30; ledger `direction_r20_rung0_design`, `oracle_actuator_rung0_preflight`, `oracle_actuator_rung0_lock`, `oracle_actuator_rung0_seed11`, `oracle_actuator_rung0_result`, `oracle_actuator_rung0_diag`; commits `2939d39`, `2ed1f5a`, `3a66506`, `0576717`, `c986668`, `aaba738`, `df8827e`)

- **Why (audit #33 + rung-0 localization; Codex round 20).** Rung 1 failed
  and its failure was not localized among encoder, actuator, cap, site,
  propagation and decoding; the localization check found only weak tag
  information at the chosen source anchor. Round 20 removed the encoder
  entirely: can a *known* hidden code, written once through a trainable
  linear map, make the frozen model recall its matching tag?
- **Lock (`0576717`; ledger `oracle_actuator_rung0_lock`).** Runner
  `experiments/run_oracle_actuator.py` (70 nonblank lines; cap telemetry
  added to the shared write hook), config
  `experiments/config/oracle_actuator_rung0.json` (config/module/machinery
  sha256 in the ledger row). Frozen Qwen3-1.7B-Base, block 12, final-token
  `Internal record:` slot; fixed immutable hashed codebook of eight unit
  centred-simplex codes c_k = sqrt(8/7)(e_k − 1/8, 0_8) (hash `3f2381fe…`);
  only a zero-init biasless J (16→2048) trained 400 balanced entity×code
  steps, lr 3e-3, seeds 11/23/37; 0.25×slot-norm cap in primary with per-step
  and per-row telemetry; one uncapped replay of the frozen J as diagnostic.
  Eval per seed: 24 training entities × 8 codes (own code = the entity's
  assigned code; the other 7 = wrong codes) + cue + zero-hook + 8 fixed
  off-code unit random vectors; training query wording, zero configured
  filler (~36 prompt tokens; never "zero delay"); entities = bootstrap
  clusters. Gates per seed: completion >= 0.95 (code/wrong/random);
  code-follow >= 0.85 (LB > 0.75; >= 0.75 per code); own-code >= 0.85;
  wrong-code follow >= 0.85 over 168 rows with uplift >= 0.65 over the cue
  matching rate (LB > 0.50); cue and random true-tag <= 0.20; own − random
  >= 0.60 (LB > 0.50); zero-hook = cue row-for-row; every gate in >= 2 of 3
  seeds. Statuses: `BOUNDED ACTUATOR PASS — ORACLE CODE` / `CAP-LIMITED
  ACTUATOR` (uncapped passes, capped fails) / `FAIL — ORACLE
  ACTUATOR/SITE/RETRIEVAL CONSTRUCTION` (both fail; this J/site/retrieval
  line stops). No repair run.
- **Preflight (eight zero-J rows only; `experiments/results/onewrite_recall_rung1/oracle_preflight.log`): PASS** —
  codebook Gram diag 1.0, off-diagonal −0.143; hook fires once per hooked
  prefill; pre-cap ‖Jc‖ = 0 on all rows; zero-hook = cue on 8/8. (A first
  pass reported FAIL only because it read the hook counter after a cue decode
  had reset it; the corrected run is the record.)
- **Result (`experiments/results/oracle_actuator_rung0/run_result.json`;
  `J_seed{11,23,37}.pt`; 2903 s; ledger `oracle_actuator_rung0_result`).**
  `FAIL — ORACLE ACTUATOR/SITE/RETRIEVAL CONSTRUCTION`; capped passes 0/3,
  uncapped passes 0/3. Cap never active in any seed (pre-cap ‖Jc‖ <= 14 vs
  threshold 43), so capped and uncapped evaluations are identical. Code-follow
  0.219/0.245/0.245; per code: code 0 0.96/0.96/0.96, code 6 0.75/0.96/0.96,
  codes 1–5 0.0, code 7 0.04 in every seed; own-code 0.25 in every seed;
  wrong-code follow 0.21/0.24/0.24 (uplift 0.10/0.125/0.125 over the cue
  matching rate); cue true-tag 0.125 (the cue decode is the base prior
  `fask`, code 0's tag); random-vector true-tag 0.13/0.13/0.125; completion
  0.96 in code/random/cue arms; zero-hook = cue row-for-row; training loss
  flat 1.5–1.75.
- **Read-only diagnostic on the saved J's (ledger
  `oracle_actuator_rung0_diag`; `diag_logit_lens.log`).** The eight write
  vectors J c_k are near-orthogonal (mean pairwise cosine −0.14) with norms
  12–23; the raw logit lens of each write does not point at the code's tag
  (own-tag first-token rank ~10^5 of 151k in all codes and seeds);
  late-training loss by code: code 0 1.20–1.22, code 6 1.41–1.56, all others
  1.59–2.10; every tag is two tokens with distinct first tokens. Diagnostic
  only; feeds audit #34.
- **Status.** Registered FAIL; audit #34 (fresh, unprimed) ADOPTED — licensed sentence and never-say list verbatim in `STATE.md` (ledger `oracle_actuator_rung0_audit34`).

## onewrite_recall_rung1 — positive-control staircase rung 1 (training entities, training wording, zero configured filler): REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED (2026-08-29 night; unanimous seeds 11/23/37; ledger `direction_r19_ruling`, `onewrite_recall_rung1_lock`, `onewrite_recall_rung1_result`, `onewrite_recall_rung1_rung0_diag`, `onewrite_recall_rung1_audit33`, `onewrite_recall_rung1_z_localization`; commits `c71d44a`, `15758aa`, `9c14c7b`, `25f65ac`, `844676b`)

- **Why (governance amendment 8, `AGENTS.md`; audit #32).** The common
  explanation of the day's closures is that bespoke instruments and actuators
  were changed before the proximal mechanism was shown learnable (the v1
  interface never learned tag identity even on training entities: own-write
  4/24, own vs counterfactual outputs identical 24/24). The loop now keeps one
  cumulative artifact and climbs a staircase: (i) own-write vs same-entity
  counterfactual specificity on training items at zero delay; (ii) zero/short
  delay; (iii) held-out names; (iv) unseen wording; (v) long delay — one
  difficulty per rung, each locked, run and audited once. Replaces "one locked
  artifact per day".
- **Lock (Codex round 19).** Same runner `experiments/run_onewrite_recall.py`
  and one-write machinery as v1 (frozen Qwen3-1.7B-Base, E/J interface,
  block-12 slot write, strict decoding, 8 tags, same-entity counterfactual
  donors); config `experiments/config/onewrite_recall_rung1.json`. Every later
  difficulty removed: 24 TRAINING entities, training wording
  `TAG OF {NAME}:`, zero intervening tokens (target = slot + VALID TAGS line +
  query); 24 entities x 2 training source phrasings = 48 cases per arm per
  seed; arms write/cue/zero-hook/wrong/random/visible; gates unchanged from v1
  (write >= 0.75; write − cue >= 0.50, LB > 0.30; write − random >= 0.50, LB
  > 0.30; random <= 0.20; recovery >= 0.60; per-source >= 0.70; counterfactual
  follow >= 0.60 and >= 0.40 over the cue-cf rate; completion >= 0.95 in
  write/wrong/random/visible; zero-hook = cue row-for-row).
- **Pre-lock validation PASS** (`experiments/results/onewrite_recall_rung1/validate_result.json`):
  visible 1.0, cue 0.125 (chance), visible completion 1.0, cue completion
  0.96, zero-hook = cue. No outcome inspected before the lock.
- **Rule.** If rung 1 fails, this interface stops without testing later rungs;
  if it passes, the next rung changes exactly one difficulty.
- **Result (`experiments/results/onewrite_recall_rung1/train_result.json`;
  `iface_seed{11,23,37}.pt`; 1820 s; ledger `onewrite_recall_rung1_result`,
  `15758aa`).** Unanimous FAIL. Visible 1.0; cue 0.125 (chance); own write
  0.167/0.188/0.292 (8/48, 9/48, 14/48) vs same-entity counterfactual write
  0.167/0.188/0.271 (8/48, 9/48, 13/48) vs fixed random write
  0.167/0.208/0.333 (8/48, 10/48, 16/48); counterfactual follow
  0.083/0.167/0.167; completion 1.0 under any write, 0.96 cue; zero-hook =
  cue row-for-row; single-example loss 2.2/1.6/2.2 -> ~1.0. The interface
  stops under the round-19 rule.
- **Audit #33 (fresh, unprimed; `.codex_audit33.md`; adopted verbatim
  `25f65ac`; ledger `onewrite_recall_rung1_audit33`).** FAIL upheld. My
  "learns no fact-specific control" was withdrawn as unqualified (own and
  counterfactual writes changed the greedy tag on 6/144 paired rows; no pair
  followed both intended directions; own write never beat the fixed random
  write; each seed emitted only 4 of 8 tags); "zero delay" withdrawn (36
  prompt tokens of VALID TAGS/instruction block separate slot and query — say
  "zero configured filler"); the loss near 1.0 is the two-token
  chance-first-token null (1.04), not learned tag identity; evaluation used 2
  of 3 training source templates; failure NOT localized among encoder,
  actuator, cap, site, propagation, decoding. Program: stop this interface
  family; run one no-training localization check on the saved encoders; keep
  the navigator deferred.
- **Licensed sentence (audit #33, verbatim):** `onewrite_recall_rung1` is a
  unanimous registered construction-level FAIL: in frozen Qwen3-1.7B-Base, on
  24 training entities evaluated under two of the three training source
  templates, the training query wording, and zero configured filler,
  own-write true-tag accuracy was 0.167/0.188/0.292 across seeds 11/23/37,
  versus 0.167/0.188/0.271 for same-entity counterfactual-tag writes and
  0.167/0.208/0.333 for one fixed random write; own and counterfactual writes
  changed the greedy tag on 6/144 paired rows, but no pair simultaneously
  followed the true and counterfactual tags, so this exact 65,552-parameter
  linear E/J, generic block-12 slot, 0.25-norm-capped, 400-step construction
  did not establish reliable control-relative tag recall and stops under its
  predeclared rule.
- **Never say (audit #33, verbatim):** “The interface learns no fact-specific
  control,” without specifying the registered construction and greedy
  readout. “Correct and counterfactual writes are identical row-for-row.”
  “There is no tag-specific effect at all.” “The loss implies about 35–38%
  correct-tag probability.” “The training loss proves that `E` learned tag
  identity.” “This was a literal zero-delay or immediate-query test.” “All
  training source phrasings were evaluated.” “The fixed random arm rules out
  random interventions generally.” “The 0.25 cap was adequate” or “the cap
  caused the failure.” “The encoder failed,” “the writer failed,” or “the
  state was written but could not be retrieved.” “A linear one-write channel,
  block 12, a 16-dimensional state, or frozen-model memory is unlearnable.”
  “This establishes that current pretrained latent spaces are hostile to
  structured reasoning.” “This closes the real-model program.”
- **Rung-0 localization on the saved encoders (read-only; ledger
  `onewrite_recall_rung1_rung0_diag`, `onewrite_recall_rung1_z_localization`;
  `z_localization.log`; `9c14c7b`, `844676b`).** Entity-grouped,
  source-template-held-out nearest-centroid tag accuracy: raw block-12
  residual at the final ` Internal record:` source token 0.340 (shuffled
  95th pct 0.181; chance 0.125); learned z = E(LN(h)) 0.333/0.153/0.194
  (random 16-d projection 0.17/0.23/0.19); earlier leave-one-out on training
  sources 0.36/0.14/0.12; write lift of the correct tag +1.36/+1.42/+1.56
  nats vs counterfactual-tag lift +1.27/+1.31/+1.29; pre-cap |Jz|
  ~1900–2100 against the 0.25×slot-norm cap. Recorded branch (ledger, not
  audited wording): the tag signal at the chosen source anchor is weak and
  the encoder recovers at most that signal; downstream stages untested. Led
  to Codex round 20 (oracle-code actuator rung 0; entry above).

## necessity_navigator_v1 — audit-#32 blockers fixed; post-fix smoke (code-path validation, NOT a result); comparator rule; DEFERRED (2026-08-29 night; ledger `navigator_smoke_r32fixes`, `direction_r19_ruling`; commits `faa7a64`, `c71d44a`)

- **Fixes (`faa7a64`).** Swap rollout continues from the donor state with
  the NEXT step's input (no duplicated step); manifest hash binds goal words,
  permutation triples, times, walk actions and poses; untrained controls on
  identical inputs; comparator per the lock text.
- **Smoke (2000 steps, seed 11, reduced readouts;
  `experiments/results/necessity_navigator_v1/smoke_result.json`; NOT a
  result).** Held-out top-1 in A* 0.879 vs historyless control 0.484 (valid).
  Causal swap: donor recurrent state inserted into a recipient episode under a
  different symbol permutation moves behaviour to the donor place — swap 0.90
  vs noswap 0.47 / wrong-place 0.41 / random 0.44; self reference 0.9025;
  decision-4 0.90. Structural readouts fail: moves R 0.24 (untrained
  same-input control 0.48); composition order accuracy 0.51 (chance),
  commuting ratio 2.1; inverse ratio 0.63 (untrained 0.90); distance Spearman
  0.30. With self in the comparator the uplift is −0.0025 by construction;
  referred to round 19.
- **Round 19 comparator rule (ledger `direction_r19_ruling`; config
  `experiments/config/necessity_navigator_v1.json`, `c71d44a`).** `self` is
  the oracle ceiling, excluded from both uplift comparators; locked swap rule:
  swap >= 0.75; decision-4 >= 0.65; swap − max(noswap, wrong-place, random)
  >= 0.25; mass uplift >= 0.50 over the same three; swap/self >= 0.80; self
  reported as ceiling. All other gates, controls, seeds (3 x 4000 steps) and
  status rules unchanged. Resolves an ambiguity in the lock; rescues nothing.
- **Status: DEFERRED, not cancelled** — governance amendment 8 puts the
  real-model proximal mechanism (rung 1) first; runs only after a direction
  dialogue; no v2. Licensed user sentence (round 19, verbatim): "In a
  non-registered 2,000-step smoke, the navigator's recurrent state
  transferred across a new symbol permutation and drove donor-place behavior
  at 0.90 accuracy—essentially matching the 0.9025 self ceiling and exceeding
  no-swap, wrong-place, and random controls—but this validates only the
  causal-swap instrument, not learned latent algebra or a navigator result,
  because composition remained at chance at 0.51 and the four structural
  readouts failed."

## onewrite_recall_v1 — REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED (2026-08-29 night; unanimous seeds 11/23/37; ledger `direction_r17_amendment`, `onewrite_recall_v1_lock`, `onewrite_recall_v1_seed11`, `direction_r18_ruling`, `onewrite_recall_v1_result`, `onewrite_recall_v1_audit32`; audit #32 adopted verbatim `3a47fd8`)

- **Lock (rounds 15–17; `1f2fcba`).** Round 17 amended the pre-lock protocol
  gate (cue completion diagnostic with no minimum; zero-hook must reproduce
  cue row-for-row else INVALID — NO VERDICT); validation PASS: visible 1.0
  (A 1.0, B 1.0), visible completion 1.0, cue 0.0, cue completion 0.031.
  Three seeds 11/23/37, 400 single-example steps each; 64 cases per arm per
  seed (16 heldout entities x 2 source phrasings x 2 target wordings); arms
  write/cue/zero-hook/wrong/random/visible.
- **Result (`experiments/results/onewrite_recall_v1/train_result.json`;
  4094 s).** Per seed, held-out raw recall after the one write
  0.031/0.031/0.0625 = counterfactual-write = random-write; cue 0.0;
  visible-copy 1.0; completion under any learned write 0.44/0.53/0.42
  (content-independent tag-shaped nudge), cue/zero-hook 0.03; counterfactual
  follow 0.16/0.13/0.16; zero-hook reproduces cue row-for-row in all seeds
  (valid). Audit #32 replay: correct/counterfactual/random writes produced
  identical raw text on 64/64 rows in seeds 11 and 23 and identical choices
  on 64/64 (raw 62/64) in seed 37; no correct-tag signal below the parser.
  Post-hoc seed-11 training-slice diagnostic: own-write 4/24, cue 2/24, own
  vs counterfactual-source outputs identical 24/24 (diagnostic, not a
  result).
- **Licensed sentence (audit #32, verbatim):** Under the round-17-amended
  behavioral readout—visible-copy accuracy 1.00 and cue accuracy
  0.00—`onewrite_recall_v1` was a unanimous construction-level FAIL: across
  seeds 11, 23, and 37, the correct-source intervention achieved held-out tag
  accuracy 0.031, 0.031, and 0.0625, respectively, exactly matching the
  same-entity counterfactual-tag and fixed-random arms within each seed,
  while valid-tag emission rose nonspecifically from 0.031 in cue to 0.438,
  0.531, and 0.422; therefore this 65,552-parameter, 16-dimensional
  encoder/injector, trained for 400 single-example steps and applied once at
  the norm-capped block-12 slot before the registered
  filler/instruction/query sequence, did not establish a held-out causal
  tag-recall channel.
- **Never say (audit #32, verbatim):** “The fact was written but did not
  survive.” “The experiment proved that the model cannot store a fact in
  hidden state.” “Block 12 cannot support persistent memory.” “A
  16-dimensional or 65,552-parameter interface is insufficient in principle.”
  “The 71–73 filler tokens are the complete write-to-query delay.” “All three
  seeds scored 0.031.” “The training loss proves that the train facts or tag
  identities were learned.” “Correct and counterfactual writes had identical
  raw text in all three seeds.” Their choices did; seed 37 had two non-tag
  raw-text differences. “Random controls rule out every useful
  intervention.” Only one fixed random state per seed was tested. “The
  intervention had no effect.” It strongly changed valid-tag emission. “The
  write was content-independent” without the construction qualifier; only its
  saved downstream choices were nonspecific under these arms. “The original
  preregistered instrument passed.” It passed the outcome-aware but
  pre-training round-17 amendment. “More steps, another layer, a different
  slot, a maintained state, or another objective would also fail.” “Frozen
  language models lack native state or latent mathematics.” “This closes the
  real-model route.”
- **Rounds 18–19.** Round 18 (ledger `direction_r18_ruling`) read the result
  as a clean negative on a valid instrument for the registered construction
  only and licensed one navigator calibration run; its sentence is superseded
  by audit #32 above. Round 19 (ledger `direction_r19_ruling`) deferred the
  navigator and moved the one-write line to the staircase (rung 1 above).
  Design and pre-lock history: next entry.

## onewrite_recall_v1 — design and pre-lock record (2026-08-29; ledger `direction_r15_ruling`, `onewrite_recall_v1_validation`, `direction_r16_recall_gates`, `onewrite_recall_v1_validation_r16`; result entry above)

- **Design (Codex direction round 15; gates and amendments round 16).**
  Runner `experiments/run_onewrite_recall.py`, config
  `experiments/config/onewrite_recall_v1.json` (`95fb53e`, amendments
  `c1486f1`). Source `The private tag assigned to {name} is {tag}.` -> one
  write of a 16-d code at block 12 at the slot's final token -> unseen
  target wording (`PRIVATE TAG FOR {NAME}:`) after two fixed fillers of >=
  64 tokens; endpoint = the raw decoded tag; 8 nonce tags (3 train + 2
  heldout facts per tag; chance 0.125); canonical VALID TAGS line in every
  target; arms visible-copy / cue-only / correct-write / wrong-donor
  (counterfactual tag by balanced derangement) / random / zero-hook;
  byte-identical nonvisible target across arms. A positive would license
  only: "a co-designed interface can write one factual value once into a
  frozen real model and recover it after unseen wording" — a persistent
  causal memory channel, not abstraction, transfer, native state, or latent
  mathematics. Result gates (round 16): write >= 0.75; write − cue >= 0.50
  (LB > 0.30); write − random >= 0.50 (LB > 0.30); random <= 0.20; recovery
  >= 0.60; per-phrasing/wording write >= 0.70 with gaps <= 0.15; wrong-state
  follows its counterfactual tag >= 0.60 and exceeds the cue rate by >=
  0.40; completion >= 0.95 in all six arms; zero-hook must reproduce cue
  row-for-row else INVALID — NO VERDICT; no repair run.
- **Pre-lock validation 1 (provisional gates; no state;
  `experiments/results/onewrite_recall_v1/validate.log`):** 16 heldout
  entities x 2 wordings, 16 tags: visible-copy 0.906 (A 0.875, B 0.938),
  visible completion 0.906; cue accuracy 0.0, cue completion 0.0.
- **Pre-lock validation 2 (round-16 locked gate; no state;
  `experiments/results/onewrite_recall_v1/validate_result.json`):**
  visible-copy accuracy 1.0 (A 1.0, B 1.0), visible completion 1.0; cue
  accuracy 0.0 (both wordings); visible − cue = 1.0; cue completion 0.031
  (without the fact the model emits `100`/`123`, not a listed tag, despite
  the VALID TAGS line). By the letter of the gate (`cue completion >= 0.95`)
  this is a pre-lock FAIL on a protocol criterion orthogonal to the state
  hypothesis while the substantive instrument (visible 1.0 vs cue 0.0) is
  perfect. **Not locked; referred to Codex round 17** (kill by the letter, or
  amend the protocol criterion pre-lock). No training run; nothing licensed.

## onewrite_state_v1 — one-write persistent state on frozen Qwen3-1.7B-Base: KILLED PRE-LOCK (2026-08-29; ledger `direction_r13_onewrite_design`, `onewrite_state_v1_smoke`, `direction_r14_instrument`, `onewrite_state_v1_killed_prelock`)

- **Design (Codex direction round 13, locked; `experiments/run_onewrite_state.py`,
  `experiments/config/onewrite_state_v1.json`, `b7991dc`).** Train only E
  (2048->16) and J (16->2048), 65,552 params; one write at block 12 to the
  final token of an early `Internal record:` slot, clamp ||Jz|| <= 0.25
  ||h_slot||; 40 invented entities over three binary nonce attributes;
  trained consequence families PORT/VAULT, NORTH/SOUTH, RING/STAR; heldout
  families H1 (XOR -> CEDAR/QUARTZ) and H2 (attr2 x attr3 -> four labels)
  with labels and templates absent from training; arms correct-write /
  cue-only / wrong-state / random / visible-text ceiling; raw decoded
  choices only.
- **Smoke (60 steps, seed 11; `experiments/results/onewrite_state_v1/smoke.log`;
  not a result):** behavioural instrument invalid before any state
  question — visible-text ceiling 0.27 (gate >= 0.80), termination 0.17;
  cue 0.20; write 0.39 = wrong 0.39 = random 0.39 (a non-specific format
  effect). Referred to Codex round 14, which allowed ONE instrument repair
  (terse FORMAT EXAMPLE / TEST CASE / NEW ITEM format, strict first-item
  parsing) and pre-declared the kill sentence.
- **Sole pre-lock validation (round-14 format; no state; 64 heldout cases;
  `experiments/results/onewrite_state_v1/validate_result.json`):**
  visible-tag accuracy 0.344 (H1 0.50, H2 0.19; wording A 0.31, B 0.38) =
  cue accuracy 0.344; completion 1.0 in both arms — the model always emits
  an allowed label but ignores whether the tags are shown. Gate (visible >=
  0.80, visible − cue >= 0.30) FAILED. **Pre-declared ruling (Codex round 14,
  verbatim):** onewrite_state_v1 is killed pre-lock because Qwen3-1.7B-Base
  could not support the registered behavioral instrument even when the
  facts were visible; no state hypothesis was tested. Side probe
  (`instrument_probe.log`, `instrument_probe_4b.log`): Qwen3-4B-Base is also
  at chance on the H1/H2 two-variable tables (0.44/0.31 terse; 0.50/0.56
  one-shot) while managing a one-attribute rule at 0.81. No training run;
  no prompt iteration or model substitution (pre-declared). Untracked
  smoke artifacts (`iface_seed11.pt`, `smoke_result.json`) are local only.
- **What we learned:** the discovered constraint is instrument validity —
  the tested Qwen3 bases through 4B cannot reliably apply a two-variable
  table to visible facts, so rule-dependent behavioral readouts cannot
  adjudicate hidden-state interventions at this scale (round 15).

## necessity_navigator_v1 — from-scratch navigator on Z_11^2 ⋊ C_4 with aliased, per-episode-permuted observations: BUILT, smoke-tested; optional one-round calibration control; UNRUN (2026-08-29; ledger `direction_r11_navigator_design`, `direction_r12_ruling`)

- **Design (Codex direction round 11).** GRU hidden 64 (~17.7k params);
  8 observation classes over 121 positions with per-episode S_8 symbol
  permutation from disjoint banks; goal-word episodes with BFS-optimal
  action sets; historyless Bayes control; behavioural validity = held-out
  top-1 membership >= 0.85 and >= 0.20 above the control; five approximate
  readouts on held-out permutations (moves, composition/noncommutativity,
  inverses, reachability distance, causal state swap); statuses BOUNDED
  POSITIVE / PARTIAL / FAIL; no v1r2. Runner
  `experiments/run_necessity_navigator.py` (196 nonblank lines), config
  `experiments/config/necessity_navigator_v1.json` (`ab758ec`).
- **Smoke (2000 steps, seed 11; code-path validation, not a result;
  `experiments/results/necessity_navigator_v1/smoke_result.json`, untracked):**
  held-out top-1 in A* 0.879 vs historyless control 0.484; readouts execute
  — moves R 0.243 (untrained-GRU control 0.383), composition order accuracy
  0.509, inverse ratio 0.629, distance Spearman 0.296; the swap pairing
  needs same-goal episodes across permutations if it is ever run.
- **Status (round 12 amendment, round 15):** optional one-round calibration
  control, not the central artifact; do not run before a real-model result;
  runs once only if direct recall is valid but donor-specific persistence
  fails; if run and >= 2 seeds are behaviourally valid but composition and
  swap both fail, close the hypothesis that task necessity alone yields a
  readable causally portable algebra; no repair of group/GRU/alias/optimizer/readouts.

## control_cost_v1 — minimum-energy span control, Qwen3-1.7B-Base block 12: REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED (2026-08-29; ledger `direction_r9_control_cost_design`, `control_cost_v1_lock`, `control_cost_v1_result`, `control_cost_v1_audit31_correction`)

- **Design (Codex direction round 9; runner `experiments/run_control_cost.py`,
  config `experiments/config/control_cost_v1.json`, lock `f5aaf9f`).**
  One-time sigma-scaled field broadcast over the 23-token prefix span at
  block 12; first-order minimum-energy solve `v* = J^T (J J^T)^+ r` through
  the three-probe signature Jacobian at v=0; alpha grid {0, .25, .5, 1, 2,
  4}; A readout (cat/dog probes) and B readout (unoptimized consequence);
  prompt-specific and shared fields; random-field and lexical controls.
- **Result (`experiments/results/control_cost_v1/result.json`; `ff361bc`):**
  status `FAIL — FIXED BLOCK-12 SPAN CONTROL CONSTRUCTION`. Native A
  validity 23/24; native B validity 17/24 (cat 7/12, dog 10/12) -> B gates
  void. Prompt-specific first-order fields realized the A target in 1/8
  recipients by alpha <= 4 (7 right-censored); shared fields attained A in
  2/4 recipients per direction at alpha = 2.
- **Audit #31 (fresh, unprimed; adopted verbatim `17b3d90`).** FAIL upheld;
  interpretation corrected: "the first-order minimum-energy law does not
  hold", Spearman 0.76 (censored costs stored as 4x predicted, ranks
  inherited), median ratio 4.0 (a censoring boundary), cross-vs-within 8/8
  (only 2 orderings identified), the semantic-B advantage (B invalid),
  "random fields move B just as much", and the directional cost asymmetry
  are all withdrawn as evidence. Licensed residue: the registered A readout
  responds causally and directionally to the constructed fields (mean
  dog->cat movement 0.28/0.40/0.54/0.67 vs cat->dog 0.04/0.11/0.14/0.16 at
  alpha .25/.5/1/2) — an actuator/readout-specific response asymmetry, not
  an effort geometry. **Licensed sentence (verbatim):** At fixed block 12 in Qwen3-1.7B-Base, the registered sigma-scaled uniform 23-token prefix-span field derived from the v=0 three-probe Jacobian attained the joint A endpoint in 1/8 held-out recipients by α≤4, while B was not a native-valid readout; because seven local costs and six within-class costs were right-censored, the saved data do not establish a realized-cost rank law, cross-versus-within effort geometry, semantic transfer, or directional cost asymmetry, so this closes the registered actuator/solver/readout/budget construction—not first-order control, span reachability, or latent effort generally.
- **Never say (audit #31):** “The first-order minimum-energy law does not hold.” “The model’s response is strongly nonlinear at the magnitudes needed” without naming the unresolved solver, scaling, actuator, and censoring alternatives. “Predicted cost ranked realized cost.” “The method underpredicted realized cost fourfold.” “Cross-class moves cost more than presentation changes.” “The semantic field transferred to unoptimized consequences.” “The semantic field beat lexical steering.” “`p=.008` establishes semantic structure.” “Random fields moved B just as much.” “Cat→dog is intrinsically cheaper than dog→cat.” “Span control failed” or “reachability failed” without the complete registered scope. “The preregistration was executed completely.” “Five independent interventions prove the substrate lacks native structure.” “Frozen residual streams cannot support structured reasoning.” “The next latent space must be trained” as a scientific conclusion rather than an allocation hypothesis.
- **What we learned:** construction family closed; no alpha/rcond/layer/
  span/probe/optimizer repair; ratio ~6.4:1; the broader question continues
  only after a substrate-level pivot (direction round 10).

## Direction rounds 10–17 (Codex direction dialogue, 2026-08-29 evening) — rulings

- **Round 10** (ledger `direction_r10_program_ruling`, `44c5507`): STOP
  POINT for the frozen-pretrained residual-stream line as constituted (six
  measurement rounds to one build round); not a proof that pretrained
  models lack latent mathematics; next artifact class = a constructive
  state-bearing substrate.
- **Round 11** (ledger `direction_r11_navigator_design`, `17b3d90`):
  `necessity_navigator_v1` locked design (entry above).
- **Round 12** (ledger `direction_r12_ruling`, `17b3d90`/`ab758ec`):
  reconciliation with audit #31's alternative — build the real-model
  one-write state artifact first; navigator demoted to an optional
  one-round calibration control.
- **Round 13** (ledger `direction_r13_onewrite_design`, `5934693`):
  `onewrite_state_v1` locked design (entry above).
- **Round 14** (ledger `direction_r14_instrument`, `9a2a54a`): one
  instrument repair for `onewrite_state_v1`, sole pre-lock validation, and
  the pre-declared kill sentence.
- **Round 15** (ledger `direction_r15_ruling`, `a31235e`; verbatim in
  NOTEBOOK): keep the kill; dominant problem = semantic instrumentation;
  next artifact `onewrite_recall_v1`; navigator not now.
- **Round 16** (ledger `direction_r16_recall_gates`, `95fb53e`):
  `onewrite_recall_v1` locked amendments and gates (entry above).
- **Round 17** (ledger `direction_r17_amendment`): amended the
  `onewrite_recall_v1` pre-lock cue-completion criterion (cue completion
  diagnostic only; zero-hook must reproduce cue row-for-row); locked
  `1f2fcba` (result entry above).

## interchange_v2 — bias-controlled operational interchangeability, Qwen3-1.7B-Base block 12: REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED (2026-08-29; ledger `interchange_v2_lock`, `interchange_v2_result`; audit #30 adopted `fe3c541`)

- **Design (Codex direction round 8, `d069955`; runner
  `experiments/run_interchange.py` + config
  `experiments/config/interchange_v2.json`, locked `f7e3d18` with no outcome
  inspected; contexts re-matched `7beeb2b`).** Same Qwen3-1.7B-Base
  revision, CPU fp32, block 12, final anchor ` The animal`, coefficient-one
  replacement; fresh cat/dog calibration and held-out paraphrases, all
  exactly 25 tokens; fresh length-matched cow/horse contexts as on-manifold
  third-state donors; calibration-centred standardized probe statistic
  (removes the verbalizer bias that killed `interchange_v1`); fractional
  donor movement with squared-separation denominator; exact 2^8 recipient
  sign-flip specificity test. One construction; no v3.
- **Result** (`experiments/results/interchange_v2/result.json`; `65680e5`):
  native centred validity passed (cat 11/12, dog 12/12; gate met), same-state donors within tolerance 8/8 (median distance 0.35, tau 3.38), but replacing the block-12 anchor residual with the opposite class's anchor residual moved the three-probe signature by a median fractional 0.009 of the class separation with 0/8 recipients flipping two decisions; on-manifold cow/horse third-state donors moved recipients |T| 0.10–0.21 and the cross-vs-third specificity test was null (4/8 paired positive; exact sign-flip p = 0.47). Pre-declared status, verbatim: `FAIL — FIXED BLOCK-12 SINGLE-ANCHOR INTERCHANGE CONSTRUCTION` (480 forwards, 105 s). Gates: same_state True, cross_state False,
  specificity False.
- **Audit #30 (fresh, unprimed; ledger `interchange_v2_audit30_correction`;
  adopted verbatim `fe3c541`): REGISTERED CONSTRUCTION-LEVEL FAIL upheld;
  distributed-state and toward-neutral readings withdrawn.** The same-state
  gate is non-discriminating (every cross- and third-state arm also lies
  within tolerance); `p=.4727` is a descriptive sign-symmetry sensitivity,
  not an exact randomized test. **Licensed sentence (verbatim):** In Qwen3-1.7B-Base, with calibration-relative three-probe validity on 23/24 decisions, coefficient-one replacement of the block-12 residual at only the final generic ` The animal` token by matched opposite-class donors moved the standardized probe signature a median 0.0087 class separations and changed none of 24 probe signs across eight recipients, so `interchange_v2` fails its preregistered fixed single-anchor construction; because same-, cross-, and cow/horse third-state donor perturbations all fell within the same-state tolerance and the unchanged prefix remained causally available, this result does not establish absence of class information at the anchor, a distributed animal state, or failure of frozen-residual interchangeability outside this site, layer, coefficient, donor pairing, and readout.
- **Never say (audit #30):** “There is no class state at the anchor.” “The animal state lives across the prefix.” “The probes read the class from earlier tokens through attention” as an established mechanism. “Later layers overwrote the donor state.” “Block 12 is the wrong layer” as a result rather than a hypothesis. “Same-state interchangeability passed.” “Same-state signatures were preserved.” “Cow and horse donors moved both classes toward neutral.” “The exact `p=.47` proves no specificity.” “Interchangeability failed in Qwen3-1.7B-Base.” “Frozen residual streams lack native state or native mathematics.” “Persistent state must be trained.” “Native animal classification was 23/24” without “calibration-relative.” “Twenty-four independent decisions” or “three independent probes.” “The preregistration was executed completely”; the short decodes are absent.

## state_bus_v1r1 — result, audit-stage re-adjudication and audit #29: REGISTERED FAIL, CLOSED (2026-08-29; ledger `state_bus_v1r1_result`, `state_bus_v1r1_audit_stage`, `state_bus_v1r1_audit29`)

- **Registered result (`7beeb2b`; unanimous, seeds 11/23/37).** Stored
  status `FAIL — same_swap, persistence`; `summary.fails` also records
  `heldout_consequence` in every seed. Per seed: same-swap outside the
  training-derived tau (0.268/0.377/0.228) on 16/15/16 rows while self and
  same-donor raw choice accuracy is 1.0 on all three decisions; cross
  uplift-consistent ≥2/3 on 15/16/16; heldout-taxonomy uplift-consistent
  9/9/8 of 16 (gate ≥10), gain +0.34/+0.38/+0.19; taxonomy raw choice
  follows the donor on 7/16 in every seed; cross-consequence movement
  12.2→9.8→2.9, 13.9→8.7→3.0, 11.9→7.3→3.1 nats; constrained own-choice
  rollouts all-donor on 15/16/15. Artifacts
  `experiments/results/state_bus_v1r1/result.json` (checkpoints and logs
  tracked at `bd1d9de`).
- **Audit-stage re-adjudication (`556b47c`; eval-only sensitivity beside the
  registered FAIL; 0.5-nat floor declared pre-result).** All-pair raw
  taxonomy donor choice 15/48, 14/48, 7/48; on-manifold wrong-code control
  1/96, 1/96, 2/96; per-state recipients moved cat/dog/cow/horse =
  1/0/2/0, 2/0/1/0, 0/0/0/0; class `FAIL` in all seeds
  (`experiments/results/state_bus_v1r1/audit_result.json`). Not a rescue.
- **Audit #29 (fresh, unprimed; adopted verbatim `4087742`).** FAIL upheld
  with corrections both ways: the status string omitted the registered
  heldout-taxonomy failure (control-flow defect, fixed post hoc in
  `experiments/run_state_bus.py` `verdict()`; stored results unchanged);
  categorical same-state behaviour transferred while fine-grained
  confidence signatures did not; the raw 7/16 is always cat→dog 3/4 and
  cow→horse 4/4 and every all-pair taxonomy success targets canine or
  equine only; the movement ratio is not an identified temporal decay.
  Pre-declared round-8 reading ("sequential semantic controller", "partial
  taxonomy transfer", "76% decay") withdrawn. **Licensed sentence
  (verbatim):** **`state_bus_v1r1` is a fixed-construction FAIL: a 98,400-parameter supervised interface repeatedly injected a 16-dimensional code into frozen Qwen3-1.7B-Base, and across three seeds held-out same-state donors preserved every four-way categorical choice but failed the training-derived confidence-signature tolerance on 15–16/16 rows, while fixed-cycle cross codes changed taxonomy choice on 7/16 rows—always cat→dog in 3/4 and cow→horse in 4/4—and the complete registered gate vector also failed heldout taxonomy and the cross-consequence third/first movement criterion; taxonomy verbalizers were absent from the bus loss, but the all-pair sensitivity was donor-verbalizer-specific rather than state-general, so the licensed residue is a repeatedly maintained supervised response controller with pair-specific lexical/semantic steering, not autonomous persistence, abstraction, general interchangeability, or native latent mathematics.**
- **Never say (audit #29):** “Partial taxonomy transfer” without “pair-specific” and the cat→dog/cow→horse concentration. “`7/16` beat chance” or “`9/16` is statistically significant.” “Three independent semantic replications.” “Same-state interchangeability failed.” “Same-state confidence signatures generalized.” “The state decayed by 76%.” “The state survived through generation.” “The bus learned autonomous persistence.” “The rollouts freely generated all donor words.” “The heldout consequence was unseen”; only its verbalizers were absent from the bus loss. “All four states” or “all donor pairs” transferred. “The controls rule out lexical or output-space steering.” “Token length explains the whole effect.” “Reader MSE proves a persistent readable state.” “Trained-consequence accuracy was 1.0” without saying this is the self-code statistic. “Qwen learned the bus”; Qwen was frozen. “The bus establishes abstraction, interchangeability, or native latent mathematics.” “This FAIL refutes co-developed interfaces or persistent state generally.”
- **What we learned:** the bus construction and budget are closed; no bus
  v2. Strongest unified alternative for the day's real-model results (audit
  #29): the frozen model supplies a biased lexical-semantic response
  geometry within which late interventions and trained injection vectors
  move candidate-word likelihoods, with visible selected words mediating
  later decisions — no persistent interchangeable state is needed. Ratio
  about 3.6:1 (runner-only) to 6.9:1 (with config and audit stage) exceeds
  both tripwires; pivot to reachability / control cost mandated.

## state_bus_v1r1 — design, lock and pre-outcome record (historical; 2026-08-29; ledger `state_bus_v1r1_lock`, `audit28`)

- **Design (Codex direction round 7, ledger `direction_r7_ruling`; runner
  `experiments/run_state_bus.py` `c64cd67`; config
  `experiments/config/state_bus_v1.json`).** Frozen Qwen3-1.7B-Base; block-12
  16-d bus with <100k trainable parameters (encoder E, four prototypes,
  injector J at every continuation position, reader R); four states
  cat/dog/cow/horse, 8 training + 4 held-out paraphrases each; trained
  consequences sound + young; held-out consequence taxonomic adjective
  (verbalizers absent from the bus loss); loss = prototype + native +
  same-swap + cross-swap + persistence; three seeds × 600 AdamW steps,
  4-hour wall cap, no sweeps; arms none/self/same-donor/cross-donor/
  shuffled/random; kill rule and POSITIVE/CONTROLLER/FAIL status language
  registered before any result.
- **Lock (`d9d1513`; Codex Tier-1 review `.codex_statebus_review.md`, "not
  audit-ready" → fixes applied before any result):** identical
  recipient-conditioned history for every arm; held-out gain = donor-label
  fraction vs mean(shuffled, random); candidate-word-token-only summed LL
  with matched-baseline uplift and donor-vs-recipient DiD, raw argmax
  reported separately; tau = max(1e-3, Q95 of training-context same-vs-self
  distances); persistence loss at both decision boundaries; movement gate
  m1>0, m3>0, m3>=0.5·m1; LayerNorm before E and R. The earlier unreviewed
  run was stopped with no evaluation inspected and its outputs discarded
  (its smoke/train log directory `experiments/results/state_bus_v1/` was
  deleted in the 2026-08-29 sweep). Results: `experiments/results/state_bus_v1r1/`.
- **Pre-outcome licensed sentence (audit #28, verbatim):** `state_bus_v1r1`
  is a fixed three-seed training run of a 98,400-parameter supervised
  interface attached to frozen Qwen3-1.7B-Base. Training and held-out
  paraphrases are index-separated, and taxonomy verbalizers are absent from
  its loss. Its registered hard gates nevertheless use donor-directed
  relative-likelihood uplift rather than actual behavioral choice, lack a
  nontrivial magnitude floor, compare against off-manifold controls, and
  permit unsafe incomplete-seed aggregation; therefore no “persistent
  interchangeable state bus” claim is licensed from the registered status
  alone.
- **Post-outcome ceiling (audit #28):** positive → at most "descriptive
  evidence for a supervised semantic steering interface"; fail → closes this
  construction and budget only. Eval-only addendum after the run (actual
  choice change, magnitude floor, all-three-seeds rule, on-manifold
  wrong-state control, all donor pairs, per-state minimum) reported alongside
  the registered status. Full wordings and the state-bus never-say list:
  `STATE.md` "Real-model line". Audit #28: "Finish and audit the current run
  once; do not scale it." Outcome and audit #29: entry above.

## interchange_v1 — operational interchangeability, Qwen3-1.7B-Base block 12, cat/dog paraphrases: closed by the locked raw-sign baseline; no swap arm (2026-08-29; ledger `interchange_v1_baseline`, `direction_r7_ruling`, `audit28`)

- **Design (Codex direction round 6; all text frozen before results).**
  `experiments/run_interchange.py`, `experiments/config/interchange_v1.json`
  (`0f252f8`); block-12 residual at a fixed anchor (` The animal`); three
  probe margins; native held-out probe decisions by raw margin sign, gate
  `>=20/24` and `>=9` per class before any donor-swap arm; context lengths
  23–27 tokens (approximately, not exactly, matched — a deviation).
- **Result** (`experiments/results/interchange_v1/result.json`): cat 8/12,
  dog 7/12 (15/24) → BASELINE FAIL; no swap arm ran. Diagnostic only: the
  native signatures are strongly class-separated on all three probes
  relative to a calibration midpoint (24/24), but two probes carry a
  constant lexical bias, so raw sign confounds class with lexical frequency.
- **Ruling (direction round 7, `b9e4581`):** raw sign was the intended
  native-behaviour statistic; calibration-centring would change the estimand
  post hoc; also exact length matching missed and `cross_toward_other`
  divides by separation rather than its square; no corrected gate, no swap
  arm, artifact DEAD.
- **Licensed sentence (audit #28, verbatim):** The locked raw-zero baseline
  failed and closes `interchange_v1`; no swap arm ran. Calibration-relative
  class separation is a diagnostic design finding, not a rescue and not
  evidence for interchangeability. Audit #28: kill UPHELD for the locked
  construction only; evidence against interchangeability: NONE.
- **Never say (audit #28):** "interchangeability failed in Qwen3-1.7B-Base";
  "three independent causal interventions failed". Cheapest moot-maker: a
  bias-controlled `interchange_v2` with calibration-midpoint/paired
  contrasts preregistered, on-manifold wrong-donor control, clustered
  reporting — authorized by direction round 8 and run (entry above;
  REGISTERED CONSTRUCTION-LEVEL FAIL under audit #30).

## coordinate_v3 — prediction-site two-bit coordinate (tense × number), Qwen3-1.7B-Base: gates met mechanically; audit #27 = narrow late lexical-control effect (2026-08-29; ledger `coordinate_v3_result`, `coordinate_v3_audit27`)

- **Design (Codex direction round 5).** Base model rev `ea980cb0…`; native
  next-token readout {is, was, are, were} at a fixed final-token prediction
  site; two mean residual differences estimated from 12 calibration families
  using only states 00/10/01 (11 never prompted); LOFO layer rule at
  coefficient 1; held-out 8 noun/complement families × 4 corner transports;
  zero arm and three seeded norm-matched random-axis-pair arms. Config
  `experiments/config/coordinate_v3.json` (`c8beee9`).
- **Result** (`experiments/results/coordinate_v3/baseline_result.json`,
  `full_result.json`): baseline 12/12 ×3; blocks 8/12/16 pass number 12/12
  but tense 0/12; block 20 passes all four signed single-axis transports →
  frozen (|v_T| 214, |v_S| 281); corner transports 8/8, 8/8, 8/8, 8/8; zero
  0/8; three random arms 0/8 each. Logit lens: W_U·(v_T+v_S) top token =
  ` were`; dose response is a threshold (0.25 → 0/32, 0.5 → 1/32, 1.0 →
  32/32).
- **Audit #27 (`2799efb`/`d44b1de`): NEGATIVE for the coordinate artifact;
  positive only as a narrow lexical-control diagnostic.** Design error: every
  corner transport flips the number bit without changing the visible
  subject, so the grammar gate is 0/32 (not ~30/32) and `test_gate` was
  never enforced. **Licensed sentence, verbatim:** At a fixed final-token
  prediction site in Qwen3‑1.7B‑Base, a coefficient‑1 block‑20 patch built
  by adding or subtracting two mean residual differences estimated from 12
  calibration families using only states 00/10/01 changed the
  full-vocabulary greedy next token to the predeclared member of {is, was,
  are, were} in all 32 held-out noun/complement-template cases, versus 0/32
  for the zero arm and each of three seeded random-axis-pair arms, but the
  vectors project directly onto those verb logits and every corner token
  disagrees in number with the unchanged visible subject, so the result is
  a narrow late lexical-control effect rather than a grammatical,
  persistent, or general latent coordinate.
- **Never say (audit #27):** “A two-dimensional latent grammatical coordinate
  was discovered.” / “Two hidden grammatical states composed to produce an
  unseen state.” / “The intervention generated grammatical native
  continuations.” / “The state persisted through generation.” / “The result
  generalizes to held-out tasks, verbs, templates, models, or layers.” /
  “Random controls prove the learned direction is uniquely meaningful.” /
  “A small perturbation produced the effect.” / “Number is represented
  abstractly earlier than tense.” / “State 11 was unseen by the model”; only
  the experiment’s calibration and selection omitted it.
- **What we learned:** the cheapest explanation (late lexical readout
  steering living in the unembedding) is strongly supported; scoped
  measurement-to-artifact ratio 6.7:1 exceeded the governance threshold and
  was raised to the user.

## Direction rounds 5–7 (Codex direction dialogue, 2026-08-29) — rulings

- **Round 5** (`.codex_direction_r5`, not committed; no separate ledger row —
  the round-5 rule is recorded in ledger `coordinate_v3_result`): after the
  two instruct-model baseline kills, move to a base model's native
  next-token readout at the prediction site; a positive requires all four
  corner transports with zero and random controls at 0, plus one fresh audit
  before any claim.
- **Round 6** (`.codex_direction_r6`, not committed; no separate ledger row —
  design recorded in ledger `interchange_v1_baseline`): pivot from Cartesian
  coordinates to operational interchangeability (donor code swap between
  paraphrases at block 12), native-behaviour gate by raw probe sign before
  calibration-scale standardization.
- **Round 7** (ledger `direction_r7_ruling`, `b9e4581`; verbatim in
  NOTEBOOK): `interchange_v1` dead; frozen-residual mining ruled "not
  working" (audit #28 reclassified this as an allocation pivot, not a
  scientific conclusion); central artifact becomes the co-developed latent
  interface `state_bus_v1`; frozen-model interventions become controls.

## coordinate_v2 — tense × grammatical number, Qwen3-1.7B: KILLED at the pre-declared baseline gate (2026-08-29; ledger `coordinate_v2_baseline`)

- **Design (Codex direction round 4; wording frozen before any result).**
  Qwen3-1.7B rev `70d244cc…`; states `00` present-singular, `10`
  past-singular, `01` present-plural, `11` past-plural; intervention =
  persistent-current-position (final prompt position at prefill, then the
  sole current position at every decoding step); baseline-first kill gate:
  W0 only, no `11` prompted, each of `00/10/01` must reach `>=14/16`
  normalized exact single-sentence outputs with `16/16` termination; on any
  failure the artifact is killed with no further prompt repair. Config
  `experiments/config/coordinate_v2.json` (`97504bf`).
- **Result** (`experiments/results/coordinate_v2/baseline_result.json`,
  `baseline.log`; 78 s): `00` 16/16, `10` 16/16, `01` 12/16, termination
  48/48 -> `passed: false`; **artifact KILLED** by the pre-declared rule. No
  hidden capture, no intervention run. Misses on `01`: two items pluralized
  the definite object along with the subject (canonical-form ambiguity), two
  ignored the instruction.
- **What we learned:** the second candidate two-bit task also fails its
  capability baseline on the model at hand; next step is a Codex decision.

## coordinate_v1 — two-bit causal coordinate (tense × polarity), Qwen3-0.6B: UNINTERPRETABLE — INVALID POLARITY BASELINE (2026-08-29; ledger `coordinate_v1_result`)

- **Registration.** NOTEBOOK re-contextualization #27; Codex direction
  dialogue rounds 1–4 (`.codex_direction_r1..r4`, not committed); runner
  `experiments/run_coordinate.py` (`7f66f55`; matched explicit baseline /
  held-out single axes / fixed random directions / hash-stamped results
  `ddb5eee`), config `experiments/config/coordinate_v1.json`. Design: two
  moves `v_T`, `v_N` estimated leave-one-family-out from single-axis
  calibration states `00/10/01` only (state `11` never used adaptively);
  causal layer rule at coefficient one on the final prompt token; free decode
  primary; chance 1/4 among canonical forms; sham + norm-matched random
  controls; explicit-instruction baseline.
- **Demo outcome** (`experiments/results/coordinate_v1/demo.log`,
  `result.json`; Qwen3-0.6B rev `c1899de2…`, CPU): calibration hidden states
  captured `(16, 28, 1024)`; no block cleared the calibration rule —
  `acc_T = acc_N = 0.0` with termination `1.0` at every block 0–27 — so the
  run stopped before any held-out transport; `layer: null`.
- **Licensed headline and sentence (Codex direction round 4; verbatim):**
  `UNINTERPRETABLE — INVALID POLARITY BASELINE`; "stopped at calibration" is
  the execution subfinding. **Coordinate-v1 is uninterpretable:** Qwen3-0.6B
  failed the explicit polarity capability gate, so tense × polarity was not a
  valid two-bit task; independently, no block cleared the coefficient-one
  final-prompt-token calibration rule (`0/16` for both LOFO axes at every
  block, termination `1.0`), so no held-out transport was run. The valid
  tense subtask therefore supplies a bounded negative for that exact
  one-shot intervention, not for residual-stream coordinates generally.
- **Diagnostics (read-only, calibration families; not results):** explicit
  polarity instruction fails on Qwen3-0.6B (`01` 0/8 in every wording tried);
  final-token and 9-token-tail patches inert at blocks 6–18, coefficients
  1–3; all-position injection inert at blocks 4–12, degrading at 16/20 with
  one tense hit (L20 ×2). Qwen3-1.7B: tense 8/8, polarity `01` 5/8 under a
  repaired wording — below the gate.
- **Never say:** the model re-reads instruction tokens; no additive
  instruction state is present; a stable tense direction has been found.

## Round 37 — presentation-duplicated 32->16 quotient world: NO ARCHITECTURAL WIN; last toy-world round (2026-08-29; ledger `round37_lock`, `round37_result`, `round37_audit`)

- **Lock** (`round37_lock`, `eb61470`; config sha256 `5a419bea…`, module
  sha256 `6786b67b…`; registered by Codex in `theory/EXPERIMENTS.md`):
  32 presentations, true 16-class quotient; quotient-factored `z=(q,p)`
  carrier vs unrestricted carrier × 2 presentation roles × 5 seeds; rolled
  interchangeability primary; non-gating horizon/role diagnostics; Tier-1
  review passed (fixture 46.3 s, 529.5 MiB peak). Runner/config retired from
  the active path after the result (`f6dac0e`; git history).
- **Result** (`round37_result`, `b5bef1b`; artifacts
  `experiments/results/presentation_quotient_v1_{factored,unrestricted}/`
  `verdict.json` + `manifest.json` + `config.json`; evidence/weights retired,
  sha256-pinned; 35 min CPU wall): both carriers `FAIL — BEHAVIOR UNDERFIT OR
  BASE SIGNATURE UNSUPPORTED`; comparison `NO ARCHITECTURAL WIN`; both
  verdict files carry one factored-primary world verdict. Diagnostic-only:
  H2 held-out supported-truthful cells factored 867–1184/1184 vs unrestricted
  795–1184/1184; H3 factored 365–754/1056 vs unrestricted 485–992/1056; first
  divergence predominantly at step 3 of H3.
- **Licensed sentence (Round 37 audit, `round37_audit`, `d9ca753`; verbatim):**
  Under the frozen exact toy reducer, neither carrier reached
  behavior-qualified held-out presentation transfer or rolled
  interchangeability; descriptively the unrestricted carrier had higher
  transfer and interchangeability rates in 9 of 10 paired seed × role units,
  while failures were predominantly—but not exclusively—future-signature
  failures and H3 first divergence was predominantly at step 3, so the
  imposed factorization showed no benefit in this setup.
- **Never say:** every failure was future-signature / terminal responses
  never failed (128 factored and 69 unrestricted failure cells involve a
  terminal error; four cells are terminal-only) / the factorization
  constraint is harmful / unrestricted is the architectural winner / the
  Round 36d mechanism reproduced (only the horizon localization recurred) /
  the hole is a property of behaviour-supervised learning (hypothesis only) /
  anything about real residual streams based on Round 37.
- **What we learned:** the toy program ends; transferable residue = identity
  by causal interchangeability, and never use exact certificates as primary
  evidence for learned continuous systems.

## Rounds 36 v1 / 36b / 36c / 36d — minimal operational-quotient world (closed; audits #23–#26 govern)

- **Round 36 v1 — valid registered FAIL** of its frozen `0.10/0.90` exact
  reducer, every gate (`073037f`; adjudicated `e69ac72`; config
  `experiments/config/operational_quotient_v1.json` (retired; git history); artifacts
  `experiments/results/operational_quotient_v1/`). Only licensed
  reading = audit #23, verbatim in `STATE.md`: a behavior-, calibration-, and
  exactness-confounded non-certification of the registered operational
  quotient — not evidence of no approximate composable structure; the
  confidence-free replay numbers are DIAGNOSTIC only. Never say "did not
  supply ... composable action algebra", "0/176 cross-seed agreement" as a
  cell count, or "FIT BUT NON-CONGRUENT" for a support-only failure.
- **Round 36b — complete; all four cells behaviour-ineligible** (lock V3,
  review #2 RUN-READY, `61e2430`; results `abef6cf`; ledger
  `round36b_ladder`; audit #24 `round36b_audit24`, adopted `57b0961`;
  configs `experiments/config/operational_quotient_36b_{S16,S64,LR64,W64}.json` (retired; git history);
  artifacts `experiments/results/operational_quotient_36b_*/`).
  Every cell `FAIL — BEHAVIOR UNDERFIT; QUOTIENT INELIGIBLE`; no eligible
  cell, no PASS. Only licensed reading = audit #24, verbatim in `STATE.md`:
  the ladder did not reach exact held-out eligibility, while the
  reachability of that exact learned precondition remains unvalidated — not
  proven unsatisfiable; W64's cross-seed-stable canonical one-step skeleton
  is informational only, not a certified operational quotient/action
  algebra. Never say "unsatisfiable by construction", "the exact structural
  gates are unreachable", "the latent is unorganized", or "organized"
  without the local/canonical scope.
- **Round 36c — complete; both cells FAIL (POSITIVE-CONTROL scope; audit
  #25 governs).** Explicitly transition-supervised learned positive control
  (behavioural BCE + `1.0 *` MSE to the stop-gradient true-successor
  encoding; unchanged exact reducer; lock V2, review #2 RUN-READY,
  `dd699e2`; configs `experiments/config/operational_quotient_36c_{w32,w64}.json`, retired; git history).
  w32 (`3742df8`; ledger `round36c_w32_positive_control`): FAIL on every
  exact gate in every seed, action-table truth 0/5, cross-seed table FAIL.
  w64 (`d5975c1`; ledger `round36c_w64_positive_control`): FAIL — swap/toggle
  table 4/5, held-out depth-2 closure 3/5, all other gates 0/5. Only licensed
  reading = audit #25 (`round36c_audit25`, adopted `0f61280`; verbatim in
  `STATE.md`) and the w32 adjudication's licensed sentence
  (`round36c_w32_adjudication`): the registered joint moving-target
  learned-target recipe did not reach the certificate; learned gate
  reachability unresolved — not proof that the carrier or exact gates are
  unreachable; no behaviour-only interpretation; W64 stays informational
  under audit #24; only the oracle fixture has passed the reducer. Never say
  "not reachable even with direct supervision", "certification-regime
  problem" as the cause, "the auxiliary objective caused it" (leading
  hypothesis only), or "W64 beat the control". No further moving-target cells.
- **Round 36d — complete; joint FAIL — INTERCHANGEABILITY (POSITIVE-CONTROL,
  frozen chart; audit #26 governs).** The one permitted frozen-target
  head-only calibration (audit #25 rank 1): hash-pinned behaviour-only W64
  encoder/readout frozen, fresh width-64 transition head trained on the
  stationary 176-cell successor table for 16,000 steps, BCE as an evaluation
  gate, unchanged exact reducer; ONE capped cell. Registered before any
  outcome (ledger `round36d_37_registered` -> `round36d_impl_discrepancy` ->
  lock V2 `round36d_lock_v2` -> review #1 RUN-READY `round36d_run_ready`;
  runner + config `6a95c29`,
  `experiments/config/operational_quotient_36d_w64.json`, retired; git history). Run `7a5ce35`
  (ledger `round36d_frozen_chart_positive_control_produce` / `_reduce`,
  `round36d_frozen_chart_control`; `118.454 s` of a `480 s` wall; artifacts
  `experiments/results/operational_quotient_36d_w64/` — `config.json`,
  `manifest.json`, `verdict.json` committed; `evidence.json`, `weights.npz`
  git-ignored, sha256-pinned): exact behaviour in all five seeds
  (`21,184/21,184` train, `2,240/2,240` held out); exact PASS on quotient
  availability, well-definedness, toggle involution, swap/toggle table, H2
  closure, H3 closure, canonical action-table truth, and the cross-seed
  table; interchangeability misses `16/5/28/98/0` of `132,160` cells
  (`147/660,800`; all depth-2 rolled histories followed by H3; 89
  confidence-only, 58 with a future-probe truth error; no immediate
  endpoint error), so `1/5` seeds pass and the joint status is
  `FAIL — INTERCHANGEABILITY`. Only licensed reading = audit #26
  (`round36d_audit26`, adopted `cad85ef`; verbatim in `STATE.md`): narrow
  individual-gate reachability under privileged full-table supervision, not
  quotient discovery or a complete learned certificate; no complete learned
  artifact has passed the joint reducer (only the oracle fixture has); the
  adequacy ratio is diagnostic and does not identify cause. Never say
  "reached the exact certificate", "the learned-pass gap is closed", "the
  quotient/action algebra was learned", "eight independent gates", or
  "interchangeability generally fails". Round 36 calibration is closed; no
  further Round 36 cells.

## NLM-007 — LM residual-stream dynamics; middle-depth ridge lead withdrawn under the identity baseline; displacement ladder adjudicated (audit #8 wording); forward-time move adjudicated NOT MET = nonpass, not a kill (Round 20, audit #9); within-style null = diagnostic only (both arms); LOCO A/B within-family positive, bounded (audit #10 wording; adjudicated Round 22); equalized addendum defect-affected, descriptive only (audit #11); unseen-word runs A/B mechanical pass, formal gate pending (Round 23, audit #12); corrected equalized reruns contract-correct (A adjudicated Round 25; B pending); residualization A-static contract-valid, adjudicated Round 26 as corrected by audit #14 (registered-presentation sensitivity + surviving X-linked residual predictability; neither state nor presentation-independence); A-augmented (`P_aug-score4`) adjudicated Round 27 as corrected by audit #15 (nested sensitivity on the same sentinel-A cells, not a replication; outcome-clean but transductive within carrier); B-static adjudicated Round 28 as corrected by audit #16 (F4–F20 pass, F0 fails; a correlated two-sentinel check, not replication); B-augmented (`P_aug-score4`) scored on the third launch after two losses in the F8 grammar block, adjudicated Round 32 as amended-implementation and SVD-telemetry-incomplete (F4–F20 pass, F0 fails); the sentinel {A,B} × {P_static, P_aug-score4} table is complete only for the residual-versus-four-word-only-null mechanical gate — within-decoder, within-population condition robustness, not replication (audit #17); patched A-static `resSA2` complete and ADJUDICATED (Evidence gate 2026-08-28, PASS qualified: the four-cell table is complete on one common K=13/four-null/crossed-bootstrap scale; F4–F20 pass in all four correlated cells; F0 non-qualifying in three cells with a weak pooled A-score4 exception; audit #18: F0 is a model-class-sensitive diagnostic, not an all-field dead end); contextual-prefix screens `ctxscr_A/B` = point-only screens that did not triage the X-conditioned hypothesis out at F4–F20 (ctx effective df ~42.7 vs state ridge ~210–406; no state-reading gate passed; audit #18); SVD telemetry gate PARKED and unpassed by allocation (repair cap = global CLAUDE.md §2.7, not an AGENTS.md rule); Round 33 consequence instrument implemented, UNRUN, PARKED and unpassed after Tier-1 re-review #4 NOT-READY (joint-key rule closed; provenance/parity blockers open; repair cap applied as an allocation decision, ledger `nlm007_consequence_instrument_parked`; branch `conseq-instrument`, main analyzer at HEAD); contextual-prefix completions `ctx_A/ctx_B` (unresidualized) COMPLETE, scored pending adjudication, audit #19 wording upheld by audit #20: descriptive higher-EDF predictor comparisons — at F4–F20 the higher-EDF state predictor retained a positive held-out score difference from the registered `token_ids_v1` context-only pair with positive crossed lower bounds, 8/8 point-positive keys, no family collapse, support 1.0; state ridge ~5–10× the contextual ridge's effective df; F0 non-qualifying; NOT evidence that context failed, that capacity is the sole confound, or that operational state is identified (the phrase "did not close the gap" is withdrawn); Round 34 capacity-matched audit registered (`d493cf2`) and implemented (`9eb1301`; producer RUN-READY, joint reducer flagged) but HELD pending the preregistered Round 34a matched-EDF core screen (`f97a533`; K=13 KL-rank diagnostic, raw continuous KL confirmatory; audit #19 staging); `ctxS_A` and `ctxS_B` (P_static-residualized) COMPLETE, scored pending adjudication, audit #20 wording (sentinel B mirrors A: context cosine ~0.04–0.08 / nerr ~1.00 vs residual ridge cosine ~0.52–0.58 / nerr 0.82–0.86): the registered token-context ridge/kernel falls to held-out cosine ~0.04–0.07 and normalized error ~1.00 while the residual `X_perp` ridge keeps cosine ~0.56–0.62 and normalized error 0.78–0.83; raw context performance is highly non-robust to the registered `P_static` residualization and therefore `P_static`-aligned in this fitted design — NEVER "beyond template metadata", "presentation removed", "largely by construction", or "same feature space"; licensed positive wording: a higher-capacity predictor from `X_perp` carries held-out predictive information beyond the registered `P_static` nuisance projection and this fixed `token_ids_v1` context field; Round 34a matched-EDF core screen RUN-READY (`6b93ff1`, four Tier-1 rounds; six runs queued: `ctxcapA_raw`, `ctxcapB_raw`, `ctxcap_raw_joint`, `ctxcapA_static`, `ctxcapB_static`, `ctxcap_static_joint`; both estimands required, no cross-estimand verdict); Round 34b (P/C partial-overlap screen) and Round 34c (item-embedding-by-P_static X-free comparator) preregistered (`3b49321`, `ff69d82`), implemented on the main analyzer, UNRUN and under Tier-1 repair (review #1 NOT-READY on four items), to run after Round 34a and before the full Round 34 (held); Round 35 typed truth-evaluable world preregistered docs-only (`c74bfab`; authors nothing until the 34a/34b/34c ladder resolves); Round 33 parked; chain running on the frozen analyzer copy (`6b93ff1` blob): parity check (`parity_head` A/B done, `parity_ref` A running, then B and a parity verdict), then the six Round 34a runs; capture extension committed (`4137258`); contextual-prefix X-free baseline committed (`eab0a68`); populations: v1 design-void (audit #16), v2 and v3 voided by the independent linguistic adversary, v4 approved 48/48 and frozen — a bounded mentioned-string instruction micro-world (audit #17); operation-verb update = declared-operation-verb context intervention; Round 31 chain and X-free chain disarmed; Freedman–Lane conditional on one A-static cell; audit #17 allocation ruling in force (Round 33); retention marker non-commensurate (audit #13); **PROGRAM CONTINUATION RULING (2026-08-29, ledger `nlm007_program_continuation_ruling`; supersedes every queue item before it in this header): NLM-007 STOPPED as an open-ended program (infrastructure drift 6:1); terminal closeout ladder only — Round 34a raw (DONE, CONTINUE) -> Round 34a static (RUNNING) -> Round 34b (conditional on both 34a CONTINUE + final bounded repair RUN-READY) -> Round 34c (conditional on a 34b CONTINUE); first STOP/MOOT/REDUNDANT/INCONCLUSIVE rung ends it; all-CONTINUE = one narrow measurement claim then closure; CUT: full Round 34, Round 33 (branch archived as tag `archive/conseq-instrument-parked`), the parity check as a gate, the random-weight null, a second decoder; parity verdict IDENTICAL A/B (`analysis_parity_*.json`; evidence only); Round 34a RAW CONTINUE at F4–F20 in both sentinels with capacity-matched cosine margins +0.04 to +0.08 (LBs 0.02–0.05), 8/8 keys, F0 INCONCLUSIVE/diagnostic, joint COMPLETE/SCREEN-ONLY CONTINUE — capacity matching removed most of the unmatched raw gap; a narrow survival, not a strong one, not a state claim; Round 35 = requirements envelope only; Round 36 minimal operational-quotient world = the constructive program (design registered `c26eee4`; implementation in progress); producer/reducer separation mandatory** (2026-08-29); **CLOSED 2026-08-29 at the Round 34b INCONCLUSIVE rung — see the program status above and the 34a-static / 34b bullets below**

- **Lock.** Round 13, documentation-only (ledger `nlm007_round13_lock`;
  design `theory/dialogue/003.md`, `theory/EXPERIMENTS.md`); Round 14
  amendment `097e2df`; Round 16 correction (completed law read at the
  substituted slot; final pair uses `head(Yhat)` on the post-norm state,
  ledger `nlm007_round16_corrected_rerun_predeclared`). Qwen3-0.6B (28
  layers), 80 one-token words × 16 carriers, four carrier-block folds; six
  layer pairs; law ladder word-mean / kNN / ridge / low-rank affine / kernel
  ridge; per-carrier oracle; within-word carrier permutations; two-way
  cluster bootstrap. Decision: ≥0.05 lead over the best static chart with
  lower bound >0 on successor cosine and both completed-law readouts in ≥2
  layer pairs. CPU only.
- **Capture.** `experiments/run_lm_dynamics.py` →
  `experiments/results/lm_dyn_v1/manifest.json` (model revision c1899de2…,
  batch 16, batched-vs-single nulls ≤ 6.1e-5, 79 s). `states.npz` is
  git-ignored; sha256 `6ec9520845811bbd…` recorded in the manifest.
- **Artifacts (`experiments/results/lm_dyn_v1/`; all kept).**
  - `analysis.json` — fallback run, pairs L0→1 / L8→9 / L27→28, 20 shuffles,
    500 boot (ledger `nlm007_fallback_declared`, `nlm007_v1_fallback`; 1427 s,
    19% over the 20-min cap). Successor-endpoint numbers valid; completed-law
    numbers read at the last token — **secondary only, invalid for the lock**
    (Tier-3 audit #5).
  - `analysis_ext.json` — extension, pairs L4→5 / L12→13 / L20→21 (ledger
    `nlm007_ext_predeclared`, `nlm007_ext_v1`; 1100 s). Same validity split:
    successor valid, completed-law secondary/invalid for the lock.
  - `analysis_slot.json` — **canonical slot-endpoint result**: corrected rerun
    over all six pairs, 20 shuffles, 500 boot, seed 13007 (ledger
    `nlm007_slot_v1`; 2145 s of a 3300 s budget; reload check unchanged).
    Exploratory at the reduced 20/500 budget; its L8/L12 qualification is
    withdrawn below.
  - `analysis_basesmoke.json` — moot-maker smoke at L8→L9 only, 2 shuffles /
    20 boot, point estimates (ledger `nlm007_baselines_smoke_L8`; 796 s).
    Pipeline validation; superseded by `analysis_base.json`.
  - `analysis_base.json` — predeclared six-pair moot-maker run
    (identity-plus-residual and per-carrier affine; ledger
    `nlm007_baselines_v1`). Took 4540.8 s against the predeclared 3300 s
    budget: **budget-incomplete exploratory artifact** — measured values
    retained, no planned full-budget gate earned; the null-making withdrawal
    still applies (Round 18, audit #7).
  - `identity_check.json` — stored-true-successor identity test of the slot
    completion at every pair and carrier (ledger `nlm007_identity_check_v1`;
    audit #6 action 3). **Valid**: routing validated to measured precision
    (per-pair max KL 1.9e-6 to 6.2e-6 over 16 × 80 cells); no per-carrier
    error profile or fresh-float32 comparison was stored.
  - `analysis_deltasmoke.json` — `--target delta` pipeline smoke at L8→L9
    (1 shuffle / 10 boot; ledger `nlm007_delta_smoke_L8`). **Not a result.**
  - `analysis_delta.json` — **valid, adjudicated (Round 19, audit #8)**:
    five-pair displacement ladder (ledger `nlm007_delta_predeclared`,
    `nlm007_delta_v1`; 1750.3 s of the 5700 s wall; support 1.0). Reading
    below.
  - `forward_manifest_A.json` / `forward_manifest_B.json` — forward-time
    captures, sentinel A = '.' and B = ',' (ledger
    `nlm007_forward_predeclared`, `nlm007_forward_locality_control`);
    `forward_states_A/B.npz` git-ignored. Locality control passes under the
    Round 20 corrected clause (ledger `nlm007_forward_locality_ruling`);
    A/B unappended q-states and laws identical bit-exactly (ledger
    `nlm007_forward_AB_equality`).
  - `analysis_fwdsmoke.json` — `--source forward` pipeline smoke at F8, A
    (1 shuffle / 10 boot; ledger `nlm007_forward_smoke_F8A`). **Not a
    result.**
  - `analysis_fwdA.json` — forward-time move, sentinel A = '.', layers
    0/4/8/12/20, 20 shuffles / 500 boot (ledger `nlm007_forward_fwdA`;
    2220 s). **Valid; adjudicated Round 20 + audit #9**: the primary arm
    did not meet the preregistered two-layer same-sentinel criterion (only
    `F20` qualifies) — a nonpass under the historical contract, not a kill.
    Oracle field meaningless (ledger `nlm007_oracle_defect_forward`).
  - `analysis_fwdB.json` — sentinel B = ',' arm (registered as the
    control/replication arm; a correlated same-population check), same
    settings (ledger `nlm007_forward_fwdB`; 1823 s). **Valid; adjudicated
    Round 20**: `F12` and `F20` qualify (ridge); cannot rescue the period
    arm. Oracle field meaningless. Reading below.
  - `analysis_stylesmoke.json` — `--style-null` + KL-rank pipeline smoke at
    F8, A (2 shuffles / 10 boot; ledger `nlm007_stylenull_smoke_F8A`).
    **Not a result.**
  - `analysis_styleA.json` — within-style-family target null, sentinel A,
    layers 0/4/8/12/20, 20 shuffles / 500 boot (ledger
    `nlm007_stylenull_predeclared`, `nlm007_stylenull_styleA`; 2213 s;
    support 1.0). **Diagnostic only (audit #9)**: the null is an
    alignment-destruction diagnostic, not a clean style null; its KL-rank
    endpoint ranked K = 7 candidates instead of the preregistered 10 —
    labelled, **not contract-valid on that endpoint**. No claim. Oracle
    field meaningless.
  - `analysis_styleB.json` — sentinel B arm of the same control (ledger
    `nlm007_stylenull_styleB`; 2238 s; support 1.0): `F8/F12/F20` pass the
    historical style gate mechanically, `F4` misses, `F0` fails. **Diagnostic
    only**, same K = 7 label, no claim (Round 21). Oracle field meaningless.
  - `analysis_locoA.json` — within-family leave-one-carrier-out control,
    sentinel A, layers 0/4/8/12/20, 500 word-clustered boot (ledger
    `nlm007_loco_predeclared`, `nlm007_loco_locoA`; 2902 s of the 4500 s
    wall; support 1.0). **Scored under the Round 21 rule; adjudicated Round
    22.** Reading below (audit #10 wording). Oracle field meaningless.
    The LOCO smoke (`nlm007_loco_smoke_F8A`) crashed before writing a JSON;
    log numbers only.
  - `analysis_locoB.json` — sentinel B arm of the LOCO control (ledger
    `nlm007_loco_locoB`; 3091 s; support 1.0). **Scored; adjudicated Round
    22** (audit #11 precision). Reading below. Oracle field meaningless.
  - `analysis_locoeqA.json` — Round 22 equalized-baseline LOCO addendum,
    sentinel A (word-only one-hot ridge; shrunk word mean; ledger
    `nlm007_loco_addendum_predeclared`, `nlm007_loco_locoeqA`; 2911 s;
    support 1.0). **Defect-affected (audit #11; ledger
    `nlm007_locoeq_defect_inner_centre`): outer margins descriptive only**,
    inner-selection claim invalid. The equalized smoke
    (`nlm007_locoeq_smoke_F8A`) artifact was deleted; log numbers only.
  - `analysis_locoeqB.json` — sentinel B arm of the addendum (ledger
    `nlm007_loco_locoeqB`; 2977 s; support 1.0). **Defect-affected (audit
    #11): descriptive only**; `F12/F20` mechanical, `F4/F8` miss on
    skill/KL-rank lower bounds, `F0` fails.
  - `analysis_unseenA.json` / `analysis_unseenB.json` — Round 22 unseen-word
    runs, sentinel A = '.' (ledger `nlm007_unseen_unseenA`; 2239 s) and
    B = ',' (ledger `nlm007_unseen_unseenB`; 2256 s; predeclared
    `nlm007_unseen_predeclared`); support 1.0, eight block × word-fold keys.
    **Mechanical pass at `F4/F8/F12/F20`, `F0` fails; formal gate pending
    (audit #12)** — status "mechanical pass under the recorded reduction;
    formal gate pending a contract-correct bootstrap". Reading below.
  - `analysis_locoeq2A.json` — corrected equalized addendum, sentinel A
    (analyzer `d10fc66`: inner two-carrier centre; comparator frozen by
    calibration score; ledger `nlm007_loco_locoeq2A`; 3753 s of the 4500 s
    wall; support 1.0). **Contract-correct; adjudicated Round 25** (audit
    #13 wording). Reading below.
  - `analysis_locoeq2B.json` — sentinel B arm of the corrected addendum
    (ledger `nlm007_loco_locoeq2B`; 4196 s; support 1.0).
    **Contract-correct; Codex adjudication pending** (required before any
    combined A/B equalized reading). Reading below.
  - `analysis_residsmoke.json` — smoke of the audit #12 bootstrap repair,
    the stronger unseen-word lexical nulls, `--residualize static`, and the
    Round 24 raw four-null shadow arm (sentinel A, F8, 1 shuffle / 10 boot;
    ledger `nlm007_resid_smoke_F8A`, `nlm007_resid_shadow_smoke_F8A`;
    design `nlm007_residualization_predeclared`). **Not a result**; meets
    the Round 25 raw-shadow launch prerequisite at pipeline level.
  - `analysis_resSA.json` — residualization, sentinel A, `P_static`
    (Round 25 launch ruling, ledger `nlm007_residualization_budget_amended`:
    120-minute wall, K = 13, five layers, two unseen-word folds, 20 shuffles
    / 500 boot; ledger `nlm007_resid_resSA`; 4405.7 s; support 1.0).
    **Contract-valid for the primary residual-vs-null question; scored and
    adjudicated Round 26 as corrected by audit #14** (ledger
    `nlm007_audit14_adopted`). Pre-patch analyzer: retention reported only
    as "the predeclared robustness marker is mechanically met" (ledger
    `nlm007_retention_marker_defect`,
    `nlm007_retention_common_scale_predeclared`). Reading below.
  - `analysis_resAA.json` — A-augmented (`P_aug-score4`: `P_static` plus at
    most four scores obtained by projecting a leave-calibration-word-pool
    carrier mean of `X` into a basis learned from calibration carriers; the
    full carrier-mean vector is not appended — audit #15), same contract,
    common-scale retention field present (ledger `nlm007_resid_resAA`;
    4737.8 s of the 7200 s wall; support 1.0; adjudicated
    `nlm007_resid_resAA_adjudicated`, Round 27, as corrected by audit #15).
    **Outcome-clean but transductive within carrier; not unqualifiedly
    contract-valid** until the pre-result meaning of the lock's
    carrier-mean clause is resolved. Reading below.
  - `analysis_resSB.json` — B-static (`P_static`, sentinel ','), same
    contract, common-scale field present (ledger `nlm007_resid_resSB`;
    4598.4 s; support 1.0). **Contract-valid; adjudicated Round 28.**
    Reading below.
  - `analysis_resAB.json` — B-augmented (`P_aug-score4`, sentinel ','),
    same contract, common-scale field present (ledger `nlm007_resid_resAB`;
    5073.8 s of the 7200 s wall; support 1.0). Third launch: the first two
    launches were lost in the F8 grammar block (ledger
    `nlm007_resid_resAB_crash`, `nlm007_resid_resAB_crash2`; erratum
    `nlm007_erratum_resAB_crash_localization`: the first loss occurred while
    entering grammar_w0 with no traceback, only the second is localized to
    torch `linalg.svd` non-convergence on the fitted low-rank coefficient
    matrix at grammar_w1 — "the F8 grammar block", not "the same fold"); the
    third ran with a numpy float64 LAPACK SVD fallback. **Adjudicated Round
    32 (`f8a2c48`, ledger `nlm007_round32_labels`) with the labels
    amended-implementation and SVD-telemetry-incomplete**: F4–F20 pass the
    residual-vs-null gate, F0 fails; ridge-only cosine and skill margins are
    mechanically reportable, the K = 13 KL-rank endpoint and every low-rank
    interpretation are amendment-qualified until the SVD telemetry gate
    (per-fit provider/exception/shape/finite/spectrum/rank telemetry plus a
    float64 NumPy shadow-backend agreement check) passes Tier-1 numerical
    review. Reading below.
  - `analysis_resSA2.json` — patched A-static rerun (identical design to
    `resSA`, common-scale retention block; ledger
    `nlm007_resid_resSA2_predeclared`, result `nlm007_resid_resSA2`).
    **Complete (5825 s, committed analyzer): F4, F8, F12, F20 pass the
    residual-vs-strongest-null gate (block-first LBs cos >= 0.46, skill >=
    0.18, KL-rank >= 0.20; 7-8/8 keys; retention held on all three
    endpoints); F0 fails (2/8 full-gate keys, negative skill/KL-rank
    margins).** This fills the {A,B} x {P_static, P_aug-score4} table on one
    common scale. **Adjudicated (Evidence gate 2026-08-28, ledger
    `nlm007_fourcell_adjudication`; PASS, qualified):** strict full-gate
    keys 7/8, 7/8, 6/8, 8/8 (the "7-8/8 keys" above are jointly positive
    keys); minimum crossed block-first lower bounds cos 0.458, skill 0.175,
    K=13 KL-rank 0.197; all 48 F4–F20 layer × endpoint common-scale ratio
    medians exceed 0.5 (estimator/null competition ratios, not retained
    signal). Licensed wording is the audit #17 sentence in STATE.md
    (condition robustness within one decoder and population, not
    replication, state, or a native law). Provenance erratum: ledger row
    `nlm007_resid_resSA2` says sentinel `2`; the artifact records
    `sentinel_tag: A` (`nlm007_erratum_resSA2_sentinel_label`).
  - `analysis_ctxscr_A.json`, `analysis_ctxscr_B.json` — contextual-prefix
    X-free **point-only screens** (`--ctx-screen`, `token_ids_v1`, committed
    analyzer copy; ledger `nlm007_ctxprefix_ctxscr_A`,
    `nlm007_ctxprefix_ctxscr_B`). Not gate adjudications. Audit #18 wording:
    at F4–F20 the cell-state ridge exceeds the strongest `token_ids_v1`
    field by approximately 0.11–0.20 cosine and 0.11–0.20 normalized-error
    reduction in both sentinels, so the screen did not triage the
    X-conditioned hypothesis out; it does not establish that the
    state-reading gate is live (skill, continuous-KL, crossed intervals,
    joint key count, collapse checks, and a capacity-matched comparison are
    unscored). Contextual ridge effective df ~42.7 vs state ridge ~210–406
    across F4–F20. At F0 the contextual field nearly closes ridge
    direction (cosine gaps ~0.019/0.018) but not magnitude (normalized
    error ~1.00 vs 0.97) — not proof that "prefix IDs explain F0".
  - `analysis_ctx_A.json` (ledger `nlm007_ctxprefix_ctx_A`; 5783 s) and
    `analysis_ctx_B.json` (`nlm007_ctxprefix_ctx_B`; 4476 s) — unresidualized
    contextual-prefix completions (`--contextual-prefix-xfree`, 20 shuffles /
    500 boot; committed-analyzer copy): **complete, scored pending
    adjudication; audit #19 wording governs.** Each is a completed
    unresidualized, outer-held-out predictor comparison. At F4–F20 the
    higher-EDF cell-state ridge retained a positive held-out score difference
    from the registered `token_ids_v1` context-only pair on displacement
    cosine, normalized error, frozen completion skill and continuous KL — A:
    cosine +0.15 to +0.20 (LB ≥ 0.13), normalized error +0.14 to +0.20, skill
    +0.34 to +0.46 (LB ≥ 0.25), continuous KL +0.27 to +0.45 (LB ≥ 0.17); B:
    cosine +0.11 to +0.18 (LB ≥ 0.09), normalized error +0.11 to +0.16, skill
    +0.33 to +0.41, continuous KL +0.24 to +0.40 — with positive crossed lower
    bounds, all eight outer keys point-positive, no carrier family collapse,
    support 1.0. F0: cosine +0.019 (A) / +0.018 (B) while skill and
    continuous-KL lower bounds cross zero and the continuation family
    collapses — non-qualifying, model-class-sensitive. The state ridge has
    approximately 5–10 times the selected contextual ridge's effective
    degrees of freedom and a different feature class; inner tuning is
    calibration-only (no held-out-outcome double use); the endpoints are
    correlated functionals of one prediction. Licensed reading: a higher-EDF
    state predictor has a positive held-out score difference from this fixed
    context-only pair — a descriptive predictor comparison, not evidence that
    context failed, capacity is the sole confound, or operational state has
    been identified. Never "did not close the gap"; never "gate live".
    Audit #20 upholds this reading for `ctx_B`.
  - `analysis_ctxS_A.json` (ledger `nlm007_ctxprefix_ctxS_A`; 5655 s;
    `--residualize static`, 20 shuffles / 500 boot; committed-analyzer copy)
    — **complete, scored pending adjudication; audit #20 wording governs.**
    The corresponding sentinel-A comparison on the `P_static`-residualized
    relation. At F4–F20, the registered token-context ridge/kernel falls to
    held-out cosine approximately 0.04–0.07 and normalized error
    approximately 1.00, while the residual `X_perp` ridge has cosine
    approximately 0.56–0.62 and normalized error 0.78–0.83; the crossed
    cosine, normalized-error, skill, and continuous-KL margins are positive
    (cosine +0.51 to +0.58, LB ≥ 0.46; normalized error +0.17 to +0.23;
    skill +0.32 to +0.49; continuous KL +0.26 to +0.48; 8/8 keys), with
    support 1.0 and no family collapse. The contextual ridge is already at
    approximately 47 EDF in every F4–F20 key and the contextual kernel is
    approximately 48 EDF at F8–F20, so the collapse is not a
    low-selected-EDF artefact. F0: cosine margin positive, normalized-error,
    skill, and continuous-KL margins negative — model-class-sensitive
    diagnostic. `P_static` is a ten-column block/length/position nuisance
    design; `token_ids_v1` is a distinct approximately 205–222-column
    carrier/POS token-context design (at most 48 distinct training rows,
    omitting the item token and cell `X`); only `X` and `Delta` are
    residualized. Licensed reading: the registered context field's raw
    predictive signal is highly non-robust to `P_static` residualization
    and is therefore `P_static`-aligned within this fitted design; a
    higher-capacity predictor from `X_perp` carries held-out predictive
    information beyond the registered `P_static` nuisance projection and
    this fixed `token_ids_v1` context field. Never "largely by
    construction", "beyond template metadata", "presentation removed",
    "same feature space", a quantified presentation share, mediation, or
    causal attribution; not an identified state contribution, native law,
    or representation-level hostile hole.
  - `analysis_ctxS_B.json` (ledger `nlm007_ctxprefix_ctxS_B`; 4784 s;
    `--residualize static`, 20 shuffles / 500 boot; committed-analyzer copy)
    — **complete, scored pending adjudication; the same audit #20 wording
    governs.** Sentinel B mirrors sentinel A: at F4–F20 the registered
    `token_ids_v1` ridge/kernel falls to held-out cosine approximately
    0.04–0.08 and normalized error approximately 1.00, while the residual
    `X_perp` ridge keeps cosine approximately 0.52–0.58 and normalized error
    0.82–0.86; residual ridge minus the strongest contextual arm: cosine
    +0.46 to +0.51 (LB ≥ 0.42), normalized error +0.14 to +0.19, skill
    +0.34 to +0.45, continuous KL +0.24 to +0.41; 8/8 keys, support 1.0, no
    family collapse. F0: cosine margin positive, normalized-error, skill,
    and continuous-KL margins negative. Licensed reading (audit #20): raw
    context performance is highly non-robust to the registered `P_static`
    residualization and therefore `P_static`-aligned in this fitted
    design; not identified as presentation; not by construction; not a
    state contribution; the residual predictor separation is descriptive
    and unmatched in capacity. Same never-say list as `ctxS_A`.
  - **Program continuation ruling (2026-08-29; ledger
    `nlm007_program_continuation_ruling`, commit `6e74798`; supersedes the
    queue described in the bullets below):** NLM-007 is STOPPED as an
    open-ended program (infrastructure drift 6:1 by the constitution's
    tripwire) and closes via one terminal ladder — 34a raw (once) -> 34a
    static (once, separately; no cross-estimand pooling) -> 34b (only if both
    34a estimands CONTINUE and its final bounded repair is RUN-READY without
    scope expansion) -> 34c (only after a 34b CONTINUE). First
    STOP/MOOT/REDUNDANT or INCONCLUSIVE rung ends the ladder; all-CONTINUE
    records "the predictor separation survived these registered controls"
    and NLM-007 closes anyway. CUT: full six-arm Round 34; the Round 33
    consequence instrument (branch `conseq-instrument` archived as tag
    `archive/conseq-instrument-parked`, never run); the parity check as a
    gate; the random-weight architecture null; a second decoder. Round 35 is
    a requirements envelope only. Governance: mandatory producer/reducer
    separation.
  - `analysis_parity_head_A/B.json`, `analysis_parity_ref_A/B.json` —
    HEAD-vs-refactor CPU parity check (ledger `nlm007_parity_verdict`,
    commit `cbed1ee`; contextual-prefix static screens on the committed
    analyzer copy vs the parked branch's refactored analyzer; decision JSON
    scrubbed of timing/SVD/shadow fields): **IDENTICAL for A and B.** The
    gate is cut by the continuation ruling; kept as evidence only.
  - `analysis_ctxcapA_raw.json` / `analysis_ctxcapB_raw.json` /
    `analysis_ctxcap_raw_joint.json` (+ hash-bound sidecars
    `round34a_evidence_ctxcapA_raw.npz` / `round34a_evidence_ctxcapB_raw.npz`;
    frozen analyzer copy `analyze_r34a_frozen.py` = blob `6b93ff1`; 291 s /
    254 s, tokenizer only; ledger `nlm007_round34a_raw`, commit `60d06f7`)
    — **Round 34a RAW: CONTINUE at F4–F20 in both sentinels.** Strongest
    matched margin per layer (F4/F8/F12/F20): A cosine +0.072/+0.057/+0.045/
    +0.042 (crossed LBs 0.034/0.024/0.019/0.024), normalized error
    +0.073/+0.047/+0.040/+0.054; B cosine +0.082/+0.064/+0.047/+0.043 (LBs
    0.049/0.034/0.023/0.022), nerr +0.088/+0.054/+0.042/+0.067; 8/8 keys
    jointly positive at every F4–F20 layer; strongest contextual arm = the
    token-id kernel at most layers; F0 INCONCLUSIVE (matched EDF undefined;
    diagnostic only). Joint: COMPLETE/SCREEN-ONLY, CONTINUE, common layers
    F4/F8/F12/F20 (one reducer NaN-replay defect fixed on the main analyzer,
    producer untouched; ledger `nlm007_round34a_reducer_nan_fix`). Reading:
    capacity matching removed most of the unmatched raw gap (`ctx_A`/`ctx_B`
    cosine margins +0.11 to +0.20); a +0.04 to +0.08 separation survives
    (audit #21 withdrew "lower bounds just above the 0.02 threshold": the
    registered rule is point margin >= 0.02 with LB > 0; smallest raw point
    0.0397, smallest LB 0.0146) — a small-magnitude but systematic
    within-design survival, not a state claim.
  - `analysis_ctxcapA_static.json` / `analysis_ctxcapB_static.json` /
    `analysis_ctxcap_static_joint.json` (+ `round34a_evidence_*_static.npz`)
    — **Round 34a STATIC: CONTINUE at F4–F20 in both sentinels** (ledger
    `nlm007_round34a_static`, commit `850414c`; 314 s / 304 s; joint re-run
    on the main analyzer after the NaN-replay reducer fix: COMPLETE/
    SCREEN-ONLY, CONTINUE, common layers F4/F8/F12/F20). Strongest matched
    margins (F4/F8/F12/F20): A cosine +0.306/+0.383/+0.373/+0.435 (LBs
    0.227/0.315/0.305/0.353), nerr +0.047/+0.089/+0.084/+0.115; B cosine
    +0.329/+0.352/+0.337/+0.367 (LBs 0.262/0.278/0.275/0.264), nerr
    +0.065/+0.082/+0.077/+0.100; 8/8 keys; F0 INCONCLUSIVE (diagnostic).
    Selected state ridge 202–384 EDF; contextual ridge target ~47; kernel
    target ~48 at F8–F20 but ~4.36 in 4/8 A and 2/8 B F4 keys. Licensed
    wording (audit #21): the residual predictor separation was not
    eliminated by the registered EDF match within these fixed feature
    classes — never "not a capacity artefact"; not a state claim.
  - `analysis_ctxoverlap_A.json` (444 s) / `analysis_ctxoverlap_B.json`
    (595 s) / `analysis_ctxoverlap_joint.json` — **Round 34b (P/C
    partial-overlap screen, static estimand): INCONCLUSIVE in both
    sentinels and in the joint — the terminal rung** (ledger
    `nlm007_round34b_sentinels`, `nlm007_round34b_joint`; commits `b285945`,
    `21ecb3f`). `P+C − P` cosine A +0.0178..+0.0373, B +0.0238..+0.0355
    (A/F4 point just below 0.02 but its upper interval exceeds 0.02) — the
    redundancy STOP fails; residual-context (`C⊥→Δ⊥`) cosine ~+0.019..+0.089
    with residual normalized-error gain negative in every ridge/kernel,
    sentinel, F4–F20 cell — retention fails. Joint: COMPLETE/SCREEN-ONLY,
    INCONCLUSIVE, no common retaining layer, no common stop layer. The joint
    reducer's EDF<=rank bound was producer-inconsistent by ~3e-5 at excluded
    F0 fits only; repair = post-outcome but not outcome-selective (audit
    #22); producers, sidecars, and gate functions unchanged; no rerun. The
    positive `P+C − P` increments are evidence against the registered
    strict-redundancy account, not evidence for operational state. Under
    the continuation ruling an INCONCLUSIVE rung is an allocation stop:
    Round 34c (`itemctx_*`; implemented, never run) does not run and
    NLM-007 closes (audit #22).
  - **Round 36 — minimal operational-quotient world (the constructive
    program; design registered in `theory/EXPERIMENTS.md` at `c26eee4`,
    ledger `round36_design_registered`; run-ready `a383a45`):** runnable
    CPU-only latent transition system on the 16 four-bit states with
    toggle/swap/no-op, trained from behaviour only; identity = equality of
    future response signatures under allowed actions (bisimulation);
    falsifiers with registered thresholds (quotient well-definedness,
    involution, non-commutation table, held-out 2/3-step closure,
    interchangeability, cross-seed action-table invariance). One module
    `experiments/run_operational_quotient.py` (`produce` / `reduce` /
    `fixture`; retired `f6dac0e`), config `experiments/config/operational_quotient_v1.json` (retired);
    producer/reducer separated; fixture before any produce.
    **v1 first run — FAIL every gate** (commit `073037f`; ledger
    `round36_first_run`; adjudication `e69ac72`, `round36_adjudication1`;
    audit #23 `round36_audit23`). Artifacts
    `experiments/results/operational_quotient_v1/`: `config.json`,
    `manifest.json`, `evidence.json`, `verdict.json`, `weights.npz`.
    `produce` (seeds 11/23/37/53/71, CPU, one process) 52.6 s (train 41.7 s,
    evidence 11.0 s; wall 900 s); `reduce` FAIL on quotient availability,
    quotient well-definedness, toggle involution, swap/toggle table,
    held-out depth-2/3 closure, interchangeability, action-table truth
    (0/5 seeds; 14–56% of 176 cells at `0.10/0.90`), cross-seed whole-table
    gate. Behaviour: train 96.563–98.546%, held-out 97.009–98.259%, depth-3
    93.8–96.3%, loss still falling at step 4,000. Licensed reading = the
    audit #23 paragraph (verbatim in `STATE.md`): behavior-, calibration-,
    and exactness-confounded non-certification; no fit-but-non-congruent
    claim. DIAGNOSTIC only (confidence-free `p>0.5` replay, read-only): 148–
    174/176 truthful one-step action cells per seed (84.1–98.9%); cells
    identical across all five seeds 11/176 at the registered thresholds,
    112/176 at `p>0.5`; 175/176 bitwise-majority table; every exact gate
    still fails.
  - **Round 36b — behaviour-fit ladder (preregistered `f9dea33`;
    COMPLETE; audit #24):** four cells `S16` / `S64` / `LR64` / `W64`
    (16k/.003/w32; 64k/.003/32; 64k/.001/32; 64k/.003/64; configs
    `experiments/config/operational_quotient_36b_S16.json`, `_S64.json`, (retired `f6dac0e`)
    `_LR64.json`, `_W64.json`; walls 8/20/20/30 min; every cell run and
    visible; no pooling or best-cell). Locks before any outcome: V1
    `f95ff01` (`round36b_lock`) -> review #1 NOT-READY `70b58a7`
    (`round36b_review1`: eligibility from producer aggregates) -> audit #23
    amendment `9edb892` (three-stage status tree, DIAGNOSTIC `p>0.5` table,
    cellwise cross-seed accounting, depth traces) -> row-level logit replay,
    lock V3 `ff8eaa7` (`round36b_lock_v3`) -> review #2 RUN-READY
    (`round36b_run_ready`); runner + configs `61e2430`. Run (ledger
    `round36b_ladder`; `abef6cf`; walls 174/606/618/696 s): every cell
    `FAIL — BEHAVIOR UNDERFIT; QUOTIENT INELIGIBLE`. Fit (train / 21,184;
    held-out / 2,240; five seeds): S16 20,894–21,184 (1 exact) / 2,179–
    2,218; S64 21,078–21,184 (2 exact) / 2,184–2,226; LR64 21,088–21,184
    (4 exact) / 2,198–2,225; W64 21,184 on all five seeds / 2,216–2,239
    (98.9–99.96%, none exact; all misses H3, none common to all seeds).
    Artifacts `experiments/results/operational_quotient_36b_*/`:
    `config.json`, `manifest.json`, `verdict.json` committed;
    `evidence.json` (165–177 MB per cell) and `weights.npz` git-ignored
    (`.gitignore`: remote size limit), retained locally, sha256-pinned in
    manifest/verdict/ledger. Licensed reading = audit #24 (verbatim in
    `STATE.md`): eligibility not reached; reachability of the exact learned
    precondition unvalidated, not unsatisfiable. INFORMATIONAL only (W64,
    `p>0.5`): all 16 encoder identities and the truthful 176/176 canonical
    action table identical across five seeds; well-definedness 71–94%,
    involution 46–84%, H2 closure 98.6–99.9%, H3 closure 61–92%,
    interchangeability 39–77% — a cross-seed-stable canonical one-step
    skeleton, not a certified quotient/action algebra. A prospectively
    locked, post-outcome, outcome-informed successor: exploratory, not
    confirmatory; it cannot rescue or overturn v1.
  - **Round 36c — quotient-trained positive control (COMPLETE; both
    cells FAIL; audit #25):** same carrier/seeds/representatives as 36b
    with explicit transition supervision (BCE + `1.0 *` MSE to the
    stop-gradient true-successor encoding over the 176 canonical
    transitions), `result_scope = POSITIVE-CONTROL`, unchanged exact
    reducer. Locks before any outcome: `round36c_registered_locked` ->
    review #1 NOT-READY / lock V2 (`round36c_review1_lock_v2`) -> review #2
    RUN-READY (`round36c_run_ready`); runner + configs `dd699e2`. w32
    (produce 819.8 s / 1800 s wall): FAIL on every exact gate in every
    seed, action-table truth 0/5, cross-seed table FAIL. w64 (conditional
    cell, precondition met; produce 987.6 s / 2400 s wall): FAIL —
    swap/toggle 4/5, depth-2 closure 3/5, all else 0/5. Artifacts
    `experiments/results/operational_quotient_36c_w32/` and `_w64/`:
    `config.json`, `manifest.json`, `verdict.json` committed;
    `evidence.json` and `weights.npz` git-ignored, retained locally,
    sha256-pinned. Adjudication (`round36c_w32_adjudication`): the
    combined loss trace deteriorates in seeds 11/53/71 after early minima,
    plateaus in 37, near-converges only in 23; paired regression vs
    width-matched behaviour-only S64 train counts localizes the regression
    to the added joint objective. Licensed reading = audit #25 (verbatim in
    `STATE.md`); the FAIL registers as "this reachability control did not
    reach the certificate", nothing stronger. Next: Round 36d (frozen-chart
    transition control, one capped cell), then Round 37 (presentation-
    duplicated 32->16 quotient world) — both registering, neither run.
  - Round 34 capacity-matched state-versus-context audit (registered
    `d493cf2`, `theory/EXPERIMENTS.md`; implemented on the main analyzer as
    `--context-capacity-audit round34_v1`, commit `9eb1301`; ledger
    `nlm007_round34_registered`, `_impl_review1..3`,
    `nlm007_round34_producer_run_ready`): producer path RUN-READY; the joint
    claiming reducer is flagged for one further review (repair rounds on it:
    3, cap reached). **Full six-arm run CUT by the continuation ruling
    (never run)**; previously HELD (audit #19, ledger
    `nlm007_audit19`): its primary estimand is `P_static`-residualized and
    cannot retroactively capacity-match `ctx_A`/`ctx_B`; its K=13 KL-rank
    endpoint must become diagnostic (raw continuous KL confirmatory) or the
    parked SVD gate must reopen before any outcome. **Round 34a — matched-EDF
    core screen** (Codex preregistration `f97a533`; tags `ctxcapA_raw` /
    `ctxcapB_raw` for the unresidualized estimand, `ctxcapA_static` /
    `ctxcapB_static` separately; token ridge/kernel only, state ridge
    bisected to the selected contextual EDF and to the 47/48 rank ceiling,
    same outer folds, cosine + normalized error with paired block-first
    crossed intervals, no completion): **RUN-READY at `6b93ff1`** after four
    Tier-1 rounds (ledger `nlm007_round34a_registered_implemented`,
    `_review1..3`, `_fix1`, `nlm007_round34a_run_ready`; sentinel artifacts
    non-claiming, joint reducer read-only, hash-bound per-cell evidence
    sidecars `round34a_evidence_<tag>.npz`). Audit #20: raw and static are
    separate required screens — raw stages the `ctx_A`/`ctx_B` comparison,
    static stages the `ctxS` and future consequence estimand; neither
    substitutes for the other and no cross-estimand verdict is permitted.
    Six runs: `ctxcapA_raw`, `ctxcapB_raw`, `ctxcap_raw_joint` (DONE,
    CONTINUE — see above), `ctxcapA_static`, `ctxcapB_static`,
    `ctxcap_static_joint` (RUNNING). Under the continuation ruling a
    surviving margin does not reopen full Round 34, Round 33, or a completion
    comparison; it only licenses the conditional 34b rung.
  - **Round 34b / 34c (audit #20; preregistered by Codex in
    `theory/EXPERIMENTS.md`, `3b49321` / `ff69d82`, ledger
    `nlm007_round34bc_registered`):** before interpreting the static
    collapse as contextual redundancy and before the full Round 34, a cheap
    same-fold `P_static`/context partial-overlap screen (`P`, `C`, `P+C`,
    `C_perp -> Delta_perp`, same-EDF `X_perp` reference, context-to-`P_static`
    alignment; 34b, tags `ctxoverlap_A` / `ctxoverlap_B` / `ctxoverlap_joint`)
    and an item-embedding-by-`P_static` X-free comparator (`P_static` + 16
    calibration-only item-embedding PCs + 160 interactions + boundary/POS
    floor; 34c, tags `itemctx_A` / `itemctx_B` / `itemctx_joint`).
    **Implemented on the main analyzer (`round34b_overlap_analysis`,
    `round34c_itemctx_analysis`; ledger `nlm007_round34bc_implemented`),
    UNRUN, under Tier-1 repair:** review #1 NOT-READY on four items —
    leakage provenance validated by counts rather than exact fold
    identities, clamped global `cos_rows` admitting undefined-cosine cells,
    NaN able to win inner selections with feature-dimension telemetry not
    locked, walls checked only per outer key (ledger
    `nlm007_round34bc_review1`; repair round 1 of 3; fix pass 2 + re-review
    #3 = repair round 3 of 3 in flight, ledger `nlm007_round34bc_fix2`).
    Under the continuation ruling both are CONDITIONAL rungs of the terminal
    ladder: 34b only if both 34a estimands CONTINUE and this final repair is
    RUN-READY without scope expansion; 34c only after a 34b CONTINUE.
    Full Round 34 and Round 33 are cut.
  - **Round 35 — typed truth-evaluable world (docs-only design,
    `c74bfab`; ledger `nlm007_round35_design_registered`):** four-bit finite
    world (toggle/swap/no-op), population and linguistic-adversary
    contracts, frozen forced-choice yes/no log-odds, wrapper and same-length
    controls, causal patches, inherited X-free ladder, involution and one
    non-commuting composition, CPU-only budget. Authors nothing. Under the
    continuation ruling it is a **requirements envelope only** (right
    direction, wrong first artifact); the constructive program is Round 36.
  - SVD telemetry / shadow-backend gate — **parked and unpassed** after
    re-review #4 (ledger `nlm007_svd_telemetry_review4`): a discretionary
    allocation decision under the global CLAUDE.md §2.7 repair-round cap,
    not an AGENTS.md rule (audit #18). Every low-rank / K=13 KL-rank claim
    keeps its amendment qualification; the analyzer diff stays uncommitted.
  - Round 33 multi-position consequence instrument (runner
    `capture_forward_consequence`, analyzer `--source forward_consequence`;
    ledger `nlm007_consequence_impl_pending_review`, `_review2`, `_review3`):
    **implemented, unrun, PARKED and unpassed** (ledger
    `nlm007_consequence_instrument_parked`): re-review #3 found the reducer
    did not enforce six keys jointly positive across both horizons (closed);
    re-review #4 NOT-READY on base-compat schema mirror / preflight,
    serialized-fit fingerprints, wall rechecks, and a real CPU parity run —
    the fourth consecutive repair round, so the global CLAUDE.md §2.7 repair
    cap was applied as an allocation decision (instrument lives on branch
    `conseq-instrument`; main analyzer stays at HEAD; raised to the user;
    parked, not killed). No consequence run is authorized; even after repair
    a pass licenses only persistence of downstream predictive accuracy under
    frozen tails (audit #18). Audit #19 upheld the parking as allocation, not
    a kill: review #4 still found a real legacy-manifest crash, unproved
    exact-fit reuse, incomplete preflight/binding, wall gaps, and missing CPU
    parity; the branch is salvageable only if a later matched-capacity result
    makes the consequence test worth reopening. **CUT by the continuation
    ruling:** branch archived as tag `archive/conseq-instrument-parked`, not
    deleted; its legacy-parity question was answered (IDENTICAL) but no
    CONTINUE reopens it.
  - `analysis_xf{SA,SB,AA,AB}.json` — Round 27 comparator 2, the
    **registered X-free field** (`--xfree-field`: calibration-only
    residual-space field from `P_static` + the rank-4 carrier-summary scores
    + 16 frozen-embedding PCs + 64 interactions, no cell-level `X⊥`, with a
    df-matched state ridge; ledger `nlm007_xfree_field_predeclared`,
    `nlm007_xfree_comparator_implemented`, frozen at analyzer `cddcd47`
    with the literal command in `nlm007_xfree_comparator_frozen`).
    Four cells, 7200 s each; "registered", not "fair" (audit #15: the fixed
    rank-4/full-prefix omissions remain substantive). **Disarmed** (ledger
    `nlm007_xfree_chain_disarmed`): the armed chain was killed under the
    Round 29 order and sits behind the external-axis probes. No artifact.
  - `analysis_fl{SA,SB,AA,AB}.json` — Round 27 comparator 1, the fully
    refitted Freedman–Lane residual-geometry null (`--fl-null 20`:
    layer-level exact test, common cell mask, ridge-only inner grid; ledger
    `nlm007_freedman_lane_predeclared`,
    `nlm007_flnull_comparator_implemented`). **Ready, not armed**: Round 29
    (adopting audit #15) limits it to one conditional A-static cell, run
    only if the external-axis probes leave the state reading live. No
    artifact.
  - `experiments/config/lexical_probe_fresh_v1.json` — Round 29 probe-2
    population (ledger `nlm007_fresh_population_frozen`): four families
    question / instruction / comparison / enumeration, 8 matched
    presentation pairs + 4 operational control pairs, same 80 words; ` not`
    (id 537) is a single token on every prefix. Prospectively authored and
    committed before any new capture or score, not independently blind to
    prior results; the declared digest `c6edaa92…` is not the raw file
    SHA-256 (`12c72401…`). **Design-void for confirmatory probes 2–4
    (audit #16)**: no pair establishes presentation-only equivalence across
    all four word classes, and several change syntactic licensing, modality,
    definiteness, degree, or quantification. Retained unchanged as an
    exploratory mixed-frame stress set only; no post hoc subset rescue is
    confirmatory. No capture. Successors: v2 and v3 (voided), v4 (frozen)
    below.
  - `experiments/config/lexical_probe_fresh_v2.json` — Round 31 population
    (`79c8628`; Codex as outcome-blind author; `Please/Kindly |
    For reference,/For clarity, … plan to {repeat|omit|capitalize|reverse}
    the word <X>`; tokenization pre-check passed). **Voided** by the
    independent linguistic adversary (ledger `nlm007_fresh_v2_voided`,
    erratum `nlm007_erratum_v2_void_count`): all 16 pair-2 cells fail —
    "For reference" vs "For clarity" introduce distinguishable discourse
    purposes that can scope over the operation; the 16 pair-1 cells and the
    16 control cells pass. Retained unchanged as a pragmatic-purpose stress
    set (audit #17: not a dead file). No capture.
  - `experiments/config/lexical_probe_fresh_v3.json` — Round 31 population
    (`a8b14a8`, fresh session; Please/Kindly; ASCII vs typographic
    apostrophe). **Voided** on control edit-magnitude (ledger
    `nlm007_fresh_v3_voided`): all 32 presentation pair cells passed
    (apostrophe rated near-degenerate), but the 8 controls under the
    orthographic wrapper fail clause 6 (a whole-word operation swap vs a
    one-glyph presentation edit) — a control-design failure only (audit
    #17). Retained unchanged, descriptively. No capture.
  - `experiments/config/lexical_probe_fresh_v4.json` — Round 31 population
    (`afd6fcc`, fresh outcome-blind author): `{Please|Kindly} plan to OP the
    word <X>` and `{Hello,|Hi,} please plan to OP the word <X>`, OP ∈
    {repeat, omit, capitalize, reverse}; aligned surface-word edit distance
    = 1 for every pair and control; frozen `operation_updates` block.
    **Approved 48/48 by a separate fresh adversary session (outcome-blind
    procedural approval: grammaticality, preservation of the explicit
    string-edit instruction, matched surface-word distance under the common
    mention frame — not 48 independent linguistic observations); tokenization
    pass; frozen** (`3a70890`, ledger `nlm007_fresh_v4_frozen`): raw sha256
    `f813f9b2cb96546726412b55857e79324ac23b47a2cb6418f8569ce47bbc5d33`, git
    blob `8845f75c89c27d8db9c5f5cc8a11cfd109b4756b`; captures must pass
    `--expected-config-sha256`; any edit voids the approval. The config's
    top-level "not approved for capture" note is historical authoring-time
    text superseded by the structured approval/hash fields (erratum
    `nlm007_erratum_v4_config_note`). Audit #17: the approval licenses a
    bounded mentioned-string instruction micro-world (every item in the
    autonymic `the word <X>` frame), not presentation inertness across
    ordinary noun, verb, adjective, and function-word uses; v4 sentinel
    results are a fresh-population stress test of the same append
    construction, never pooled with `lm_dyn_v1`. No capture; chain
    `run_v4.cmd` not armed (requires the operation-update and bridge code to
    pass Tier-1 review).
  - Contextual-prefix X-free baseline (`eab0a68`, Round 31; analyzer
    `--contextual-prefix-xfree` / `--ctx-screen`, token_ids_v1, point-only
    screen; ledger `nlm007_ctxprefix_implemented`): screens scored
    (`ctxscr_A/B` above); completions running/queued.
    Operation-verb update capture stage (`capture_op_update`) and the
    no-model acceptance fixture `op_update_fixture.py` committed
    (`d9a6cca`); the analyzer side (`--source op_update`) and the
    bridge-ladder patch: uncommitted, under Tier-1 review. Audit #17: the operation-verb
    update is a declared-operation-verb context intervention, not yet a
    denizen-enacted operational move (source and recipient are separate
    prefix encodings; no execution consequence is measured).
  - `run_r31.cmd` (probe-1 screens, `P_aug-full` cell A, contextual-prefix
    screens and completions on `lm_dyn_v1`): **disarmed** (ledger
    `nlm007_r31_chain_disarmed_pending_svd_gate`) — Round 32 forbids further
    low-rank output before the SVD telemetry / shadow-backend gate passes
    Tier-1 numerical review. No artifact.
  - Capture extension (`4137258`, Round 30; no artifact): `run_lm_dynamics.py`
    gains a config provenance guard (`--expected-config-sha256`), the ` not`
    operator-insertion capture, repeat-noise arrays, and population-void
    hard controls. RUN-READY; nothing captured under it yet.
  - `analysis_unseensmoke.json` — `--unseen-words 2` pipeline smoke at F8, A
    (1 shuffle / 10 boot; ledger `nlm007_unseen_smoke_F8A`, overwritten by
    `nlm007_unseen_smoke2_F8A` with the audit #10 lexical nulls and the
    K = 11 rank universe). **Not a result**; the full run awaits Codex
    predeclaration.
- **Successor endpoint (valid in all runs).** L0→L1: word-mean = ridge =
  kernel = 0.949, shuffled null 0.95 — lexical persistence, no law beyond
  word identity. From L4 on, full-dimensional ridge beats word-mean and the
  best static chart at every depth (ridge/chart/word-mean: L4 0.927/0.884/
  0.886; L8 0.941/0.860/0.861; L12 0.977/0.898/0.888; L20 0.965/0.901/0.897;
  L27 0.976/0.883/0.864, the last on normed vectors). Shuffle penalty grows
  with depth.
- **Slot-endpoint gate reading (Round 17, superseded at L8/L12 by Round 18).**
  On `analysis_slot.json` the pairs L8→L9, L12→L13, L27→L28 cleared every
  locked gate mechanically (support 1.0); L4→L5 and L20→L21 cleared both slot
  readouts and the word-mean gate but missed the all-fold +0.05
  successor-cosine lead (a stricter convention than the original lock, audit
  #6); L0→L1 fails every lead gate. Word-mean slot skill decays with depth
  (0.95, 0.84, 0.78, 0.70, 0.43, 0.40) while ridge holds 0.92–0.98 and the
  chart collapses late (0.50, 0.51). Round 16 scorecard: five of six
  predictions held; the L27→L28 attenuation prediction failed.
- **Withdrawal at L8→L9 and L12→L13 (Round 18 + audit #7; ledger
  `nlm007_baselines_v1`).** Pooled ridge − identres on successor cosine /
  slot skill / slot ordering: L8 −0.008/−0.021/−0.020; L12 −0.007/−0.009/
  −0.013 (only slot skill and ordering are completed-law slot metrics). On
  shared words and held-out carrier blocks, identity-plus-shared-displacement
  is at least as good as full ridge within a post-hoc one-sided 0.02 pooled
  margin on the three recorded comparison metrics at L8→L9 and L12→L13; the
  finite-ladder ridge wording is withdrawn as a conservative policy. The
  intervals support "no demonstrated positive ridge advantage under this
  margin", not "no lead" or equivalence. The measured relation is consistent
  with identity plus a calibration-mean displacement under this design; the
  experiment does not determine whether the displacement is carrier-, state-,
  or word-dependent. The Round 17 two-pair criterion does not survive as a
  claim. Identity-plus-shared-displacement does not meet the chosen margin at
  L0 (+0.46), L4 (+0.033/+0.019/+0.022), L20 (+0.018/+0.034/+0.032), or L27;
  L4 and L20 remain non-qualifying but live, while L27 is not a valid
  raw-residual persistence comparison. Per-carrier affine is far below the
  cross-carrier field everywhere (within-carrier diagnostic only).
- **Displacement ladder (Round 19 + audit #8; `analysis_delta.json`).**
  Only `L20->L21` passes the predeclared three-endpoint gate (kernel;
  positive clustered lower bounds on displacement cosine, slot skill, slot
  ordering) — retained as one bounded qualifying pair under the registered
  displacement-and-slot-law gate. `L0` is lexical persistence. `L4` has a
  small live remainder but fails the gate. `L8/L12` separate strongly from
  the word-conditioned displacement mean on displacement coordinates, with
  kernel minimal among the tested ladder, but slot-ordering leads are only
  0.003–0.022 and slot-skill lower bounds are mixed — the gate fails. Adopted
  wording: held-out-carrier evidence for predictable displacement variation
  beyond a word-conditioned mean, with a kernel as the minimal tested
  predictor; carrier/template versus state dependence remains unresolved. The
  carrier shuffle is a carrier-alignment diagnostic, not a state-independence
  null (shuffled field reported for ridge/low-rank only). "The slot law
  barely registers it" is a readout fact, not a world fact.
- **Forward-time move (Round 20 + audit #9; `analysis_fwdA.json`,
  `analysis_fwdB.json`).** Sentinel '.': `F0` token-identity dominated
  (shared mean = word-conditioned mean = 0.67 ≈ field 0.69). `F4/F8/F12`:
  displacement cosine ridge/kernel 0.71–0.78 vs word-conditioned mean
  0.48–0.53; law skill at the sentinel position 0.39–0.57 vs 0.01–0.02;
  carrier-shuffled field 0.12–0.32 vs 0.67–0.81; but ordering leads
  0.00–0.08 with lower bounds ≤ 0 in half the folds — three-endpoint gate
  fails. `F20` qualifies (ridge: +0.16–0.23 / +0.50–0.61 / +0.020–0.058,
  all LBs > 0). Sentinel ',': same shape; `F12` and `F20` qualify (ridge),
  `F8` misses one ordering LB by −0.002, `F4` one skill LB. Token-identity
  control: the '.'-fitted predictor on the ',' target scores 0.43–0.54 vs
  0.26–0.30 for the shared mean. Adopted wording: the period sentinel did
  not meet the preregistered two-layer, three-endpoint qualification
  criterion — a nonpass under the historical contract, not a kill of forward
  transport; in the shared-word, held-out-carrier design, sentinel
  displacement is predictably improved over the word-conditioned mean from
  F4 onward and the response law registers that variation in cosine and
  skill; the ordering endpoint was later diagnosed as insensitive/saturated,
  so the qualification failure is not a substantive null result. The comma
  arm falsifies "token identity or position prevents any qualifying layer".
  Carrier/template presentation versus state dependence remains unresolved
  (audit #8). Ordering is replaced prospectively by KL-to-truth candidate
  rank (K = 10); no existing run is reclassified.
- **Within-style-family null, sentinel '.' (`analysis_styleA.json`;
  diagnostic only).** Mechanically `F4/F8/F20` beat both the word-conditioned
  mean and the null on cosine, skill, KL-rank (ridge KL-rank 0.82–0.90 vs
  word-mean 0.31–0.41); `F12` misses one fold's KL-rank LB; `F0` fails. The
  null collapses below the shared mean (0.16–0.50 vs 0.47–0.62). Audit #9:
  a field refit on a broken carrier pairing predicts the wrong carrier's
  displacement, so "beats the within-style null" is not informative evidence
  for a state-linked component; "style-robust" is withdrawn as a claim. The
  KL-rank endpoint here ranked K = 7 (kNN-1/5/20 omitted; fixed in
  `269e46c`) and is not contract-valid. Sentinel ',' (`analysis_styleB.json`)
  has the same shape: `F8/F12/F20` mechanical, same label, same verdict.
- **LOCO control, sentinel '.' (`analysis_locoA.json`; Round 21 rule;
  adjudicated Round 22).** Pooled ridge − per-word block mean: `F4`
  +0.126 / +0.313 / +0.300, `F8` +0.118 / +0.232 / +0.292 (cosine / law
  skill / K = 4 KL-rank), `F12` and `F20` in the same range, all lower
  bounds > 0.08, 11–15 of 16 held-out carriers passing all three; `F0`
  no pass (block mean ≥ ridge). Audit #10 wording: on already-seen words,
  within a style family, X predicts a held-out carrier's displacement and
  response-law consequence better than the three-carrier per-word family
  mean at F4–F20 — not a presentation-independent state or a native law.
  The baseline is variance-disadvantaged; equalized X-free lexical baselines
  (word-only ridge, shrunk word mean) are required before interpretation;
  LOCO does not separate state from a smooth carrier/style code; the pooled
  16-carrier bootstrap is secondary. `F0` = "no detected conditional gain".
- **LOCO control, sentinel ',' (`analysis_locoB.json`; Round 22, audit
  #11).** `F12/F20` pass (pooled ridge − block-word mean: cosine +0.07–0.10,
  skill +0.15–0.20, KL-rank +0.20–0.26, lower bounds > 0; 12–13 of 16
  carriers); `F4` misses skill and KL-rank; `F8` misses skill only (KL-rank
  LB +0.021); `F0` fails. Run-level positive (2/5); weaker in breadth than
  A — a sentinel-specific instrument result, not evidence that B carries
  less state information. Same audit #10 wording as A.
- **Equalized LOCO addendum, sentinel '.' (`analysis_locoeqA.json`;
  defect-affected, audit #11).** All 80 folds selected maximal shrinkage
  (`lam_wordonly = 100`, `alpha_shrunk = 1`), i.e. the equalized baselines
  equal the shared mean; ridge − equalized baseline at `F4–F20`: cosine
  +0.09–0.13, skill +0.23–0.30, KL-rank +0.26–0.34 (11–14/16 carriers);
  `F0` negative. **Descriptive only:** the inner centre included the
  validation carrier and the comparator was chosen on held-out outcomes, so
  "the data selected maximal shrinkage" is invalid as implemented; whether
  maximal shrinkage persists under the corrected centre is unknown. Withdrawn
  (audit #11): "no per-word lexical signal", "variance objection answered",
  "context not content", "the state-conditioned component is large". Adopted
  wording: the word-conditioned component captured by the tested estimators
  is negligible for the measured forward displacement in this design; the
  positive object is X-conditioned residual predictability.
- **Corrected equalized LOCO addendum (Round 25 + audit #13;
  `analysis_locoeq2A.json`, `analysis_locoeq2B.json`).** With the inner
  two-carrier centre and the comparator frozen by calibration score, the
  equalized baselines no longer collapse onto the shared mean: A's
  calibration-selected equalized comparator sits roughly 0.002–0.009 above
  it, B's 0.002–0.007. A: `F4/F8/F12/F20` pass against the equalized
  comparator (cos +0.09–0.13, skill +0.23–0.31, KL-rank +0.30–0.43, lower
  bounds > 0.08; 11–14/16 carriers), `F0` fails — Round 25: a valid
  mechanical positive for the bounded sentinel-A seen-word within-family
  diagnostic; audit #13: audit #11's inner-centre *defect* concern is
  resolved by the corrected sentinel-A data (not "audit #11 is resolved");
  the pooled equalized interval is secondary. B: `F12/F20` pass, `F4/F8`
  miss on skill/KL-rank lower bounds (cosine leads hold), `F0` fails;
  run-level positive (2/5); adjudication pending. Both arms agree with the
  defect-affected runs' numbers (baselines moved by ≤0.01, no verdict
  changed). Maximum wording (audit #13): "On already-seen words, within
  sentinel A's style-family design, the context-bearing X field predicts
  the held-out carrier's forward displacement and response-law consequence
  beyond the properly nested, calibration-selected X-free lexical
  comparator at F4–F20." Still withdrawn: "no per-word lexical signal",
  "context rather than content", "the state-conditioned component is
  large", any presentation-independent or native-law reading.
- **Unseen-word runs (Round 23 + audit #12; `analysis_unseenA.json`,
  `analysis_unseenB.json`).** Calibration and held-out word identities
  disjoint; the class-mean and frozen-input-embedding `wordonly_knn` nulls
  sit at the shared mean at every layer. Block-first pooled ridge − stronger
  X-free lexical null: A cos +0.14–0.19, skill +0.33–0.47, K = 11 KL-rank
  +0.35–0.57 (lower bounds > 0.12; full-gate keys 7/8, 7/8, 8/8, 8/8 at
  F4/F8/F12/F20); B cos +0.11–0.17, skill +0.31–0.41, KL-rank +0.31–0.52
  (lower bounds > 0.09; 5/8, 6/8, 8/8, 8/8). `F0`: A's continuation block
  collapses, B's cosine lead 0.018 is below the 0.02 point gate — audit #12
  wording "non-qualifying, with the continuation held-out block providing the
  strongest local failure pattern". Round 23 adjudication: the state-linked
  prediction is held only as X-conditioned residual predictability,
  generalizing across unseen word identities; the tested
  lexical-interpolation prediction fails; the presentation/style nuisance
  prediction remains live. Audit #12: the predeclared class-preserving word
  bootstrap was not implemented (words resampled without class strata and
  nested within blocks) and the lexical null family is weak, so the status
  is **mechanical pass under the recorded reduction; formal gate pending a
  contract-correct bootstrap** and stronger nulls (nested
  frozen-embedding→Δ ridge, nested embedding-conditioned kernel, k ladder).
  Adopted wording: "not exact held-out-word lookup and not the tested lexical
  interpolator" — never "not word lookup" or "not lexical" unqualified; "the
  tested lexical nulls fail", not "lexical content is absent"; the ~0.06
  seen→unseen drop is a point comparison at F8 only; positive object =
  X-conditioned residual predictability transferring across the held-out
  word fold and held-out block. Strongest alternative: `X` contains smooth
  lexical and presentation coordinates along which the later displacement
  varies; the coarse nulls collapse for coarseness, not because the
  variation is operational state.
- **Residualization A-static (Round 26 as corrected by audit #14;
  `analysis_resSA.json`).** Sentinel A, `P_static` (centred block one-hot,
  tokenized lengths, slot/sentinel positions) cross-fitted out of both `X`
  and `Δ`; two unseen-word folds; K = 13; crossed class-preserving
  bootstrap. `F4/F8/F12/F20` pass the residual-vs-null gate: `X⊥` ridge
  residual cosine 0.56–0.62 vs 0.06–0.07 for the strongest residualized
  X-free null; block-first margins cos +0.50–+0.56, skill +0.31–+0.48,
  KL-rank +0.40–+0.61 with positive lower bounds; full-gate keys 7/8, 7/8,
  6/8, 8/8 (misses family-localized in gloss/association; the four
  checkpoints are correlated measurements, not replications); no block
  collapse. `F0` fails (negative pooled skill; association-block collapse)
  — "no qualifying conditional gain at F0 under this instrument", a genuine
  negative control. Not a cosine-geometry mirage (audit #14): the ridge
  cosine falls from raw 0.65–0.76 to residual 0.56–0.62 while the nulls
  fall to ~0.06; shuffled q95 ≤ 0.13; residual normalized error 0.78–0.83.
  Registered-static-metadata arm (`P_static→Δ`): held-out cosine 0.43–0.63
  by layer — a cosine, never a variance share, fraction, or "explains 42%";
  not a pure presentation component (audit #16). Adopted
  ruling: "`P_static` took the non-collapse branch. Locally, the result
  establishes registered-presentation sensitivity and survival of X-linked
  residual predictability after cross-fitted removal of those registered
  coordinates. It identifies neither the surviving field as operational
  state nor the result as presentation-independent." Withdrawn (audit #14):
  Round 26's "much of the raw lead may have been presentation-mediated" —
  the overlap between presentation and the raw ridge lead is not
  identified. Retention: "the predeclared robustness marker is mechanically
  met" only (audit #13; pre-patch analyzer; patched rerun `resSA2` queued).
  The gate is too easy for a state claim: a fully refitted Freedman–Lane
  residual-geometry null and a flexible calibration-only presentation/
  lexical comparator are to be preregistered before any state reading.
  `P_static` pass plus `P_aug` collapse remains the predeclared "static
  coordinates incomplete, not state" branch.
- **Residualization A-augmented (Round 27 as corrected by audit #15;
  `analysis_resAA.json`).** Sentinel A, `P_aug-score4` cross-fitted out of
  both `X` and `Δ`; same folds, K = 13, bootstrap and raw-shadow arm as
  A-static; common-scale retention block present. All five correlated
  checkpoints meet the registered aggregate residual-vs-null gate. F4–F20:
  `X⊥` ridge residual cosine 0.56–0.62 vs 0.06–0.07 for the strongest
  residualized X-free null; block-first margins cos +0.50–+0.56, skill
  +0.35–+0.46, KL-rank +0.43–+0.56 with positive lower bounds; full-gate
  keys 8/8, 7/8, 6/8, 8/8; no block collapse. F0 is qualitatively weaker —
  residual cosine 0.34 vs −0.01, skill +0.16 [LB 0.02], KL-rank +0.30
  [0.12], only 2/8 keys clear the full per-key gate — and is not an
  independent confirmation of the F4–F20 profile; the raw F0 transition
  remains identity/token dominated (raw ridge exceeds the raw null by only
  ~0.019 cosine). `P_aug` carrier-summary nuisance arm (`P_aug → Δ`)
  0.45–0.64 cosine by layer; because its scores are derived from
  carrier-level `X`, this is not a presentation-only estimate or a variance
  share. Common-scale retention: the paired bootstrap median of the
  reassembled residual-model margin over the raw-ridge margin exceeds 0.5 in
  every layer-endpoint cell; 14 of 15 interval lower bounds exceed 0.5 (F4
  continuous KL 0.495), so no uniform 95% claim; ratios above one do not show
  strengthening. Reading (audit #15): A-static and A-aug are nested
  sensitivities on the same sentinel-A cells, not independent replications;
  the registered static and rank-4-score nuisance fits do not absorb the
  `X⊥–Δ⊥` association; broader presentation, carrier-geometry, and
  prefix-fingerprint explanations remain fully live. Neither result
  identifies operational state, presentation independence, fresh-style
  transfer, composition, or a native law. Strongest alternative missed by
  both queued comparators: a high-dimensional prefix/carrier fingerprint
  (aligned, cell-level, compatible with unseen-word transfer and law
  improvement).
- **Residualization B-static (Round 28; `analysis_resSB.json`).** Sentinel
  ',', `P_static`, same contract as A-static, common-scale field present.
  `F4/F8/F12/F20` pass the residual-vs-null gate: `X⊥` ridge residual
  cosine 0.52–0.58 vs 0.06–0.09 for the strongest residual null;
  block-first margins cos +0.45–+0.50, skill +0.35–+0.42, KL-rank
  +0.40–+0.58; 8/8 positive keys at every passing layer; no collapse. `F0`
  fails (cosine lead +0.27 but pooled skill negative — the −7.5 skill is
  driven by two association folds near −30, locally ill-conditioned
  normalization, not uniform failure — and KL-rank lower bound < 0; gloss and
  association collapse), as under A-static. Registered-static-metadata arm
  (`P_static→Δ`) 0.41–0.63 cosine by layer (a cosine, not a variance share
  or a pure presentation component).
  Common-scale retention passes at the bootstrap median for every F4–F20
  endpoint; F4 continuous-KL lower bound 0.426 blocks a uniform interval
  claim; joint static retention awaits `resSA2`; the ratios are
  robustness ratios, not retained signal, state, or mediation. Ruling
  (Round 28 as corrected by audit #16): across the correlated A/B static
  runs, registered block/length/position metadata predict raw
  displacement, and `X⊥` retains predictive association with `Δ⊥` beyond
  four X-free lexical nulls at F4–F20. This is a two-sentinel robustness
  result within one decoder and authored population — a correlated
  second-sentinel check, not independent replication, state, presentation
  independence, mediation, or a native law.
- **Residualization B-augmented (Round 32; `analysis_resAB.json`;
  amended-implementation, SVD-telemetry-incomplete).** Sentinel ',',
  `P_aug-score4`, same contract as A-augmented; third launch with the SVD
  fallback. `F4/F8/F12/F20` pass the residual-vs-null gate: `X⊥` ridge
  residual cosine 0.52–0.57 vs 0.06–0.09 for the strongest residual null;
  block-first margins cos +0.46–+0.51, skill +0.40–+0.44, KL-rank
  +0.45–+0.54; 6–8/8 full keys; no collapse. `F0` fails (cosine +0.33 but
  skill lower bound −0.04; 4/8 full keys; gloss collapse).
  Registered-static-metadata + carrier-summary nuisance arm (`P_aug→Δ`)
  0.42–0.64 by layer (not a presentation-only component). Same-run
  common-scale ratios exceed 0.5 at the median at F4–F20 (F0 wide). Only
  the ridge-only cosine and skill margins are mechanically reportable; the
  K = 13 KL-rank endpoint and every low-rank interpretation remain
  amendment-qualified (Round 32). Reading (audit #17 wording): the sentinel
  {A,B} × {P_static, P_aug-score4} table is complete only for the
  residual-versus-four-word-only-null mechanical gate: F4–F20 pass in all
  four correlated cells, while F0 fails except for a weak pooled A-score4
  association with only 2/8 full-gate keys. This is consistent
  within-decoder, within-population condition robustness, not replication.
- **Oracle defect (ledger `nlm007_oracle_defect_forward`).** The per-carrier
  oracle read the stored states directly; in forward and delta mode it
  predicted X from X, so the ~0.98 oracle values in `analysis_fwdA/B`,
  `analysis_styleA/B`, `analysis_locoA/B` are meaningless. Fixed
  prospectively; diagnostic only, no result changes.
- **What we learned.** Identity is the null for residual-stream transport.
  The present data support persistence plus a calibration-average
  displacement as a competitive finite-design description at L8 and L12,
  retain small unresolved remainders at L4 and L20, and do not yet establish
  a native or generally reusable affine law. The forward step is a bounded
  held-out-carrier displacement-forecasting result that does not yet
  distinguish a state-space regularity from a carrier/template-conditioned
  nuisance law. A permutation null that a flexible model trivially beats is
  not a control. Within one style family the state carries predictive
  variation beyond the family's per-word mean for seen words (LOCO A), which
  narrows but does not remove the carrier/template alternative; the
  positive object is X-conditioned residual predictability (audit #11). On
  words never seen in calibration the same object transfers across the
  held-out word fold and held-out block against the tested X-free lexical
  nulls — a mechanical pass whose formal gate awaits a contract-correct
  bootstrap and stronger lexical nulls (audit #12). The corrected equalized
  addendum removes the audit #11 defect without changing any verdict (Round
  25, audit #13). Residualization runs may state only that the predeclared
  robustness marker is mechanically met until the common-scale retention
  marker is scored (audit #13). After cross-fitted removal of the registered
  static presentation coordinates, `X⊥` still predicts `Δ⊥` and its
  reassembled consequence beyond the residual X-free lexical nulls at
  F4–F20 — registered-presentation sensitivity plus surviving X-linked
  residual predictability, with the presentation/raw-lead overlap
  unidentified (Round 26 as corrected by audit #14). The same survival holds
  for the second sentinel under `P_static` (Round 28) and, on the same
  sentinel-A cells, under the nested rank-4-score `P_aug` fit (Round 27 as
  corrected by audit #15) — correlated sensitivities, not replications; the
  registered nuisance fits do not absorb the association, and presentation,
  carrier-geometry, and prefix-fingerprint accounts remain live. Bounded to
  one decoder and one authored template population. The fourth cell
  (B-augmented, Round 32) completes the sentinel {A,B} × {P_static,
  P_aug-score4} table for the mechanical residual-vs-null gate only — F4–F20
  pass in all four correlated cells, F0 fails except for a weak A-score4
  association — and licenses within-decoder, within-population condition
  robustness, not replication; its low-rank content is amended-implementation
  and SVD-telemetry-incomplete until the telemetry gate passes (audit #17).
  Audit #17's allocation ruling (adopted Round 33; supersedes the Round 29
  and Round 31 orders): protected work (`resSA2`, the SVD telemetry gate) →
  the contextual-prefix X-free baseline → one bounded multi-position
  consequence test (next k ∈ {4, 8} tokens, teacher-forced) BEFORE the v4
  bridge/interchangeability probes and before a second decoder; do not
  author a v5 — the next population, when one is authored, is a typed
  use-frame task. Behind it, disarmed or conditional: the Round 31 chain
  (probe-1 screens, `P_aug-full` A cell, contextual-prefix screens), the
  registered X-free field ×4, Freedman–Lane on A-static only, the second
  pinned decoder. Further lessons: a numerical instrument that fails to
  converge is a finding about the instrument until diagnostics say
  otherwise (B-aug) — and every low-rank result now sits behind a telemetry
  gate; a template population must pass a predeclared linguistic contract
  before any capture, or the presentation axis it is meant to test is not
  defined (audit #16); the v1–v3 loop showed that an all-inventory
  ordinary-use presentation contract had not been achieved — v4 obtains
  grammatical core-operation equivalence by placing every item in the same
  autonymic `the word <X>` frame, which licenses a bounded mentioned-string
  instruction micro-world and nothing wider (audit #17); an instruction-verb
  edit without a measured execution consequence is a declared-operation-verb
  context intervention, not an operational move (audit #17).

## Round 12 closure — frozen-encoder program closed; pivot to worlds with dynamics (2026-08-27)

- Ledger `nlm006b_round12_adjudication`; dialogue `theory/dialogue/003.md`;
  commit `3294718`. NLM-006b corrected to non-diagnostic under its own
  label-preservation gate (below); frozen-encoder closeness/map work closes
  as scope management.
- **Residue (narrow, this encoder/dataset).** Training supplies a
  task-effective chart metric, affine-path smoothness, and graceful chart
  degradation under identity-destroying moves; no native construct tested
  (substitutability profiles, Fisher pullback, their transported variants)
  competes with it. Not a general claim about native constructs.
- Next program: causal-LM residual streams, where the forward pass is the
  world's own transport (NLM-007).

## NLM-006b — calibrated transport audit; chart survives, NON-DIAGNOSTIC under lock (2026-08-28)

- **Design.** Locked Round 11 (`nlm006b_prereg_transport_audit`): independent
  candidate strata (20 same-/20 cross-fine-label per anchor), transported-pair
  predictors F_T / R_T vs cosine_T / euclid_T on (T_e x, T_e y), true
  fine-label endpoint, label-preservation gate p_e ≥ 0.80, calibrated
  displacement gate. Ledger `nlm006b_v1`; artifact
  `experiments/results/nlm006b_v1/analysis.json`; transports
  `experiments/results/vision_cifar100_dinov2s_edits_v2/` (edits.npz
  git-ignored, sha256 9cc0e7c0…; displacement.json committed). 471 s, CPU.
- **Chart survives every displaced transport.** Support 400/400; displacement
  gate passes for crop50/invert/mix50/occlude50 (0.98–1.0 above control q95).
  TT chart lead over best native: crop50 +0.208, invert +0.227, occlude50
  +0.222, mix50 +0.090 (paired CIs exclude 0).
- **Non-diagnostic (Round 12).** Label preservation 0.19–0.46 for all four
  displaced families vs the 0.80 gate (controls hflip 0.77, shift 0.76): every
  family is OOD under the identity gate, so chart survival is descriptive
  only and no native/chart verdict is issued.
- **Order effect.** ST−TS cosine ≈ 0.035 for all displaced families (CIs
  exclude 0); ≈ 0 for hflip. Real, small, outside the invariance class only.

## NLM-006 v1 — transports outside the invariance class; EXPLORATORY (cosine-selected negatives) (2026-08-28)

- **Design.** Six transport families re-encoded by the frozen encoder
  (`experiments/results/vision_cifar100_dinov2s_edits_v2/edits.npz`, keyed
  `test_emb_<family>`: hflip, shift1px, crop50, invert, mix50, occlude50;
  `displacement.json` alongside), stratified candidates, true fine-label
  endpoint. Relabeled **exploratory** by Tier-3 audit #3 before results were
  read: hard negatives were cosine-selected, so the pool is adversarial to any
  chart-like ranking. Ledger `nlm006_v1_exploratory`; artifact
  `experiments/results/nlm006_v1/analysis.json`.
- **Uninterpretable for the primitive contest.** Every predictor scores below
  0.5 (cosine 0.411, Euclid 0.402, F 0.486, R_no_coarse 0.477) — cosine's
  "collapse" is manufactured by selecting negatives with the tested metric.
- **Exploratory signal.** Support 400/400 (stratification fixes NLM-005's
  support failure). Order sensitivity appears only outside the invariance
  class: ST−TS cosine 0.05–0.10 with CIs excluding 0 for crop50/invert/mix50/
  occlude50; 0.00 for hflip/shift1px. Displacement mean cos: hflip 0.96,
  shift 0.98 vs crop 0.63, invert 0.49, mix 0.43, occlude 0.66.
- **Next.** NLM-006b (ledger `nlm006b_prereg_transport_audit`): independent
  candidate strata, transported-pair predictors, label-preservation and
  calibrated displacement gates. Lesson: candidate pools must never be
  selected by the metric under test.

## Round 10 closure — frozen-encoder closeness/map line closed (2026-08-27, narrowed by audit #3)

- Ledger `round10_frozen_chart_closure`. The NLM-003 R-over-F claim is
  withdrawn (coarse-taxonomy leak, see diagnostics below); NLM-005 is void on
  support; no native construct built so far (substitutability profiles, Fisher
  pullback) competes with the trained chart metric on this artifact.
- **Residue as narrowed by Tier-3 audit #3:** training creates a
  task-effective chart and affine-path smoothness *in this encoder/dataset*
  (cosine 0.946 trained vs 0.575 random-init; same-class chart-line flicker
  12.7% vs 95%). Not a general claim that native constructs are dominated, and
  not proof of intrinsic geometry or of "straight routes inherited from
  training" beyond this encoder and dataset.
- Replacement line: NLM-006/006b — stratified transports outside the trained
  invariance class.

## NLM-005 — composed transport/substitution; VOID on support (2026-08-27)

- **Design.** Locked `a12aad4` (artifact lock `aab0f69`). hflip and 1-px-shift
  transports re-encoded by the frozen encoder, composed with random
  substitutions in both orders (ST, TS), true fine-label endpoint. Ledger
  `nlm005_v1_composition`; artifact `experiments/results/nlm005_v1/analysis.json`.
  Transport families now live in
  `experiments/results/vision_cifar100_dinov2s_edits_v2/edits.npz`
  (`test_emb_hflip`, `test_emb_shift1px`; byte-identical to the original
  NLM-005 file, which was removed as superseded).
- **Void by kill condition 3:** support 129/400 (32%) < 80%. Order gaps
  non-diagnostic: ST−TS cosine ≤ 0.006 (hflip 0.006 [−0.003, 0.017], shift
  0.004 [−0.003, 0.013]); shift1px R_no_coarse 0.027 [−0.003, 0.057] on a
  sensitivity row. Cosine leads native candidates by ≈0.32 on every order.
- **Lessons.** hflip/1-px shift are augmentations DINOv2 was trained to be
  invariant to, so they are near-identity moves in its world — transports must
  lie outside the trained invariance class. 40 random candidates over 100
  classes cannot reach 80% support — candidate sampling must be stratified.

## NLM-003 v2 diagnostics — R's win was a coarse-head leak (2026-08-27)

- **Design.** Same lock, artifact, endpoint as NLM-003; new anchor sample; audit
  #2 diagnostics (tie accounting, R without coarse head, cheap-baseline ladder,
  kNN k-sensitivity). Ledger `nlm003_v2_diagnostics` (Round 9: sensitivity
  accounting, not new evidence); artifact
  `experiments/results/nlm003_v2_diagnostics/analysis.json`.
- **Leak.** `R_no_coarse` 0.586 < `F` 0.667 (R with coarse 0.762; fine labels
  nest inside coarse classes). The NLM-003 R-over-F directional claim is
  withdrawn. Δ_{F−R} on this resample −0.095 [−0.142, −0.049]. R ties on
  22–33% of comparisons.
- **Ladder.** cosine 0.934, PCA-32 cosine 0.941, Euclid 0.933; pixel-stat
  Euclid 0.624, raw-pixel cosine 0.622. kNN same-class flicker 0.18/0.13/0.10
  vs cross-class 0.41/0.38/0.37 at k = 8/32/128 — world-path contrast robust to k.

## NLM-004 — random-init null world; SUPPORTED (2026-08-27)

- **Design.** Preregistered in ledger (`nlm004_prereg_null_world`) before
  scoring: random-init DINOv2-small chart
  (`experiments/results/vision_cifar100_randinit/`), true fine-label endpoint.
  Ledger `nlm004_v1_null_world`; adjudication `nlm004_round9_adjudication`
  (supported, exploratory — bootstrap CIs not in artifact); artifact
  `experiments/results/nlm004_v1/analysis.json`. CPU, 230 s.
- **Supported.** Cosine 0.575 in the null chart vs 0.946 trained (gap 0.371;
  gates ≤ 0.70 and ≥ 0.20). Embedding-kNN fine accuracy 0.069 vs 0.761.
  Same-class chart-line kNN flicker 95% (null) vs 12.7% (trained). Semantic
  heads collapse (coarse 0.21) while pixel-statistic heads stay strong (rgb
  0.83, luma 0.82) — cheap-baseline confound noted.
- **Reading.** The chart's task-effective metric and affine-path smoothness are
  created by training in this encoder/dataset; the null chart has neither.

## NLM-003 — R beats F on the true fine-label endpoint; cosine dominates both (2026-08-27) — R-over-F WITHDRAWN (see v2 diagnostics)

- **Design.** Locked at `e2a1fb2` (`theory/EXPERIMENTS.md`, NLM-003). Same
  frozen CIFAR-100/DINOv2-small artifact and runner as NLM-002, endpoint
  switched to the true fine label (no head is trained on it):
  `python experiments/run_nlm002_vision.py --cache experiments/results/vision_cifar100_dinov2s --out nlm003_v1 --endpoint fine_label`.
  Ledger `nlm003_v1_true_fine_endpoint`; artifact
  `experiments/results/nlm003_v1/analysis.json`.
- **Directional gate met.** Profile-continuity `R` 0.734 vs Fisher pullback `F`
  0.630, Δ_{F−R} = −0.104 [−0.148, −0.058] over 6,199 scored pairs; support
  thin (130/400 anchors had a same-fine-class candidate among 40 draws).
- **Chart metrics dominate.** Cosine 0.946 and Euclidean 0.935 on the same
  anchors beat both native constructs by 20–30pp.
- **Tier-3 audit #2 reclassification (adopted).** NLM-003 is a **narrow
  instrument comparison** — "these implementations lose to cosine on this
  endpoint" — not evidence that native geometry is generally dominated (one
  encoder, one endpoint, one-step random substitutions, one seed, 130 supported
  anchors). `R` takes five values with 0.5 tie credit and includes the coarse
  head (fine nested in coarse), so tie accounting and an `R`-without-coarse
  rerun are required. Next gate: random-init null (NLM-004,
  `nlm004_prereg_null_world`), cheap-baseline ladder, kNN k-sensitivity,
  nonlinear re-charting, composed / out-of-distribution moves.

## NLM-002 — non-LM branch (CIFAR-100/DINOv2): endpoint killed, chart-path structure found (2026-08-27)

- **Design.** CIFAR-100 → DINOv2-small CLS, 6000 train / 2000 test, raw pixels
  stored (`experiments/results/vision_cifar100_dinov2s/`, built by
  `experiments/build_vision_cache.py`). Runner `experiments/run_nlm002_vision.py`
  (default endpoint `rawpixel_knn`). Ledger `nlm002_v1_nonlm_branch`; artifact
  `experiments/results/nlm002_v1/analysis.json`. CPU, 133 s.
- **M2 kill condition met.** Raw-pixel k=32 kNN fine label is nearly
  uninformative (0.115 accuracy; 0.12 agreement with embedding kNN, which
  scores 0.761), so the locked endpoint is invalid and M3 (`F` 0.601 vs `R`
  0.605, Δ = −0.004 [−0.034, +0.026], 16,660 pairs) is a tie on noise, not a
  primitive verdict. Lesson: an endpoint must be independent of both candidates
  *and* informative — the true fine label is both (→ NLM-003).
- **M1 chart-path structure (informative, audit-qualified).** Along straight
  lines between same-class embeddings the coarse-semantic readout flickers on
  only 2% of paths; between classes the fine-label kNN flickers on 38% and
  any-readout on 78%. Audit #2: the 2% figure is weak evidence (affine argmax
  is near-monotone by construction) and kNN flicker is at k=32 only — a
  k-sensitivity analysis is required before any world-path claim. Pixel-stat
  heads are weak (52–59% test acc), so their 21–24% flicker is partly head noise.
- **Implementation decisions flagged at lock:** pixel statistics of
  interpolated points are approximated (no pixels exist off the data), and the
  fine-label head is never trained.

## NLM-001 — verdict: negative on predictive novelty (2026-08-27)

- **Design.** Analysis-preregistered at `fea3a8f` over sequestered raw
  matrices (`experiments/results/nlm001_v1/manifest.json`); three CPU systems
  (Qwen3-0.6B, gemma-3-270m, SmolLM2-360M); primary = 72 calibration-unseen
  words, all 80 as sensitivity; `--rule pooled --scale-normalize`. Command and
  metrics: ledger `nlm001_v1_primary_72`; artifacts
  `experiments/results/nlm001_v1/analysis_primary_72.json`,
  `analysis_sensitivity_80.json`.
- **Central bet fails.** Native calibration-KL closeness does not beat a learned
  diagonal Mahalanobis metric on the model's own contextual hidden states for
  held-out orderings: Qwen Δ = −0.058 [−0.222, +0.034]; gemma Δ = +0.017
  [−0.02, +0.06]. Every predictor scores 0.95–1.00 — the robust held-out labels
  are large-gap easy pairs (instrument limitation). Post-verdict reading
  (ledger `nlm001_v1_postverdict_note`): unlearned centered contextual cosine at
  layer 14 reaches 1.000 vs native 0.954, and the preregistered selection rule
  chose an overfit metric (calib 1.000, held-out 0.947), so the reported Δ
  understates the native loss.
- **Context reversals exceed the paraphrase null** in Qwen (Q = 2.12
  [1.70, 2.56], R = 0.18) and SmolLM (Q = 17.1 but W ≈ 0.005, so Q is not
  interpretable there); not in gemma (Q = 1.40 [0.90, 2.55]).
- **Directedness absent.** Robust (≥2-of-4) asymmetric pairs: 1.5% Qwen, 9.2%
  SmolLM. Cross-system transfer τ_b: 0.14 (qwen|gemma), 0.47 (qwen|smollm),
  0.14 (gemma|smollm).
- **Kill conditions 3 (predictive novelty), 6 (coordinate confound), 8
  (instrument metadata recorded post hoc) apply.** Tier-3 fresh audit adopted:
  T2 is geometrically vacuous, κ is an invariant of the probe table not the
  space, B>W may recover the hand-authored block taxonomy, and no NLM-001
  outcome could have earned "cosine is the wrong object".
- **What we learned.** The substitutability/KL primitive on lexical embedding
  rows adds nothing over a symmetric learned metric on contextual states. Do not
  run NLM-002 on more words; next is a competition among primitives (see
  `STATE.md`). Runners must record tokenizer revision, library versions, thread
  and batch settings at run time.

## NLM-001 — instrument calibration, pre-verdict (2026-08-27)

- **NLM-001 — contextual substitutability, context rank, and transfer.** Frozen
  theory contract: `theory/EXPERIMENTS.md`. One CPU entrypoint,
  `experiments/run_lexical_closeness.py` (using the existing substitution-probe
  helper); frozen slice:
  `experiments/config/lexical_probe_v1.json`. The 12-word smoke and disclosed
  eight-item full-pipeline validation are calibration only. The latter
  invalidated the MAD robustness rule and put asymmetry signs at chance; H1 is
  exploratory. Primary analysis uses the 72 calibration-unseen words, with all
  80 reported only as sensitivity. Three-system raw matrices were acquired
  concurrently before the Round-2b amendment and stayed sequestered until the
  amended contract was committed at `fea3a8f`; the verdict entry above is the
  first outcome analysis. Ledger: `substitution_probe_smoke_qwen3_0p6b`,
  `nlm001_pipeline_smoke_8`; artifacts `experiments/results/pipeline_smoke_8/`.

S53 follow-up: two-round consumer attribution review and bounded official-repository
access check, recorded as `s53_consumer_attribution_access_20260914`. Reference graph
queries verify answers, not neural-reader competence. Three matched feature arms
are proposed for one late request. Box inspection failed and revision API returned
504; endpoint absence is not inferred. Execution deferred, no new empirical block.

S54: one primary sequential-merging source inspected after two-round direction
review. Endpoint order robustness and stage-relative retention are different;
classification-only evidence does not establish joint composition. Ledger
`s54_admission_order_source_20260914`; zero new empirical blocks or model forwards.

S55: checked historical vision cache directories contain manifests only. A bounded
packet search and concept-consumer source check selected no execution. The source
measures global policy editing, not the proposed local attribute correction.
Ledger `s55_resource_and_editable_interface_20260914`; six local probe blocks and
fifteen scored model forwards unchanged. No new runner, package or GPU work.

## BALANCED KILL EXPERIMENT v3 — 2026-09-19

**Purpose:** Operator-balanced cell set (R21/R23 mandate) to resolve effective-n
limitation of the original 40-cell set, where operator explained 34-74% of fragility
variance and effective n ≈ 6 strata.

**Design (Fable-reviewed, R24):**
- 69 cells: 60 primary (+: 15, -: 15, //: 15, %: 15) + 9 appendix (*: 6, **: 3)
- ALL operands 1-9 (no double-digit tokenization confound)
- Identity filter: "answer readable without an arithmetic step" — zero identities
- 24 perturbations (groups A-D, same as original 40-cell), 2 surfaces
- 3312 forward passes per model, fp32 CPU
- Pre-registered: primary 60-cell core for main analysis; 69-cell full and
  residual-after-stratum-means as secondary views
- Power ~0.72 for rho=0.3 at n=69

**Models:** (same panel as original + Qwen3-0.6B-Base per R23 C1)
Qwen3-1.7B-Base, SmolLM2-1.7B, SmolLM2-1.7B-Instruct, SmolLM2-360M,
SmolLM2-135M, Qwen3-0.6B-Base, Qwen3-0.6B-Instruct, Qwen3-1.7B-Instruct,
StableLM-2-1.6B, CodeGen-350M, Phi-1.5

**Key questions:**
1. Does the gap survive with operator-balanced cells?
2. Does residual gap (after stratum means) match raw gap?
3. Does cross-model fragility transfer change from -0.022?
4. Does eta² for operator drop from 34-74%?

**Runner:** `experiments/run_balanced_kill.py`
**Results:** `experiments/results/transfer_spectrum/balanced_kill_*.json`
**Ledger:** `trho_balanced_kill_v3_launch`

**Results (5 models complete, StableLM running):**

| Model | Acc | Gap | CI | ResGap | eta2_op | Admissible |
|-------|-----|-----|----|----|---------|-----------|
| Qwen3-1.7B-Base | 99.1% | 0.500 | [0.411, 0.588] | 0.462 | 0.394 | Yes |
| Qwen3-0.6B-Base | 97.2% | 0.271 | [0.187, 0.354] | 0.427 | 0.653 | Yes |
| SmolLM2-1.7B | 87.6% | 0.275 | [0.202, 0.362] | 0.389 | 0.687 | Yes |
| SmolLM2-360M | 69.2% | 0.244 | [0.186, 0.312] | 0.471 | 0.854 | NO |
| Phi-1.5 | 65.0% | 0.034 | [0.006, 0.067] | 0.190 | 0.834 | NO |
| StableLM-2-1.6B | 72.0% | 0.215 | [0.161, 0.285] | 0.173 | 0.608 | NO |

**6-model cross-analysis — R26 REVIEW COMPLETE, MAJOR RETRACTION:**

~~TWO-CLUSTER STRUCTURE~~ **RETRACTED (R26).** The "two clusters" were a group-ordering
artifact: Qwen3 has D>A mean distance, all 4 non-Qwen3 have D<A. This single bit
reproduces the partition. Group-mean-only null gives HIGHER rho for non-Qwen pairs.
Within-group pert ranks (group ordering removed): SmolLM2 within collapses +0.739 ->
+0.014; SmolLM2-StableLM +0.642 -> +0.129; Phi-StableLM +0.457 -> -0.029.
BUT Qwen3-Phi RISES +0.253 -> +0.614. Qwen3 within-family SURVIVES: +0.943.

**LEVEL-OF-ANALYSIS DISSOCIATION (R26 key finding):**
- Pert-level rho: Qwen3 within = +0.937. Cell-level rho: Qwen3 within = **+0.015 (p=0.90)**.
- Models agree on which perturbation types are damaging but NOT which cells are vulnerable.
- SmolLM2 within-family: pert +0.739, cell +0.496 (moderate agreement at both levels).
- S1.7-Phi: pert +0.543, cell **-0.576** (anti-correlated at cell level!).
- Split-half cell reliability 0.76-0.98 (real signal, not noise).

~~Admissible-only gap~~ **RETRACTED (R26).** n=1 within pair, n=2 cross pairs — no power.
Cell-level rho was zero WITHIN Qwen3 anyway. The gap is perturbation-magnitude ordering.

~~H1 (accuracy vs eta2_op)~~ **NOT ESTABLISHED.** rho=-0.771, p=0.072, n=6.
Without Phi (p_digit=0.0004): p=0.19. eta2 driven by per-op accuracy dispersion.

**Confound tests:**
- TOKENIZER: REFUTED for SmolLM2/StableLM/Phi. NEW: Qwen3 c_newline_before merges
  `\n\n` into one token. Qwen3-0.6B's single largest pert. Without it, C>A drops.
- OPERAND MAGNITUDE: REFUTED — cell residuals uncorrelated with max(a,b)
- GROUP C POSITION: CONFIRMED — relabeled as prefix-position inert
- GROUP ORDERING: CONFIRMED (R26) — the dominant confound in all cross-model pert rhos

**~~Surviving after R26~~ — FURTHER RETRACTED BY R27:**

**R27 ADVERSARIAL REVIEW — GAP IS D-COLUMN RELIABILITY ARTIFACT:**
- ~~"Gap real for all 6 models"~~ → Phi fails permutation test (p=0.31). Q0.6/StLM marginal.
- ~~"A-D gap measures inert vs semantic"~~ → AB=AA in 5/6 models (B identical to A). A not special.
- ~~"D perturbations shift cell vulnerability"~~ → Disattenuated AD = 0.94-0.98 for S1.7/S360/Phi/StLM.
  Only Qwen3 genuine (0.82).
- DD reliability (within-group rho): Q1.7=0.134, Q0.6=0.392, S1.7=0.531, Phi=0.824.
  D columns are unreliable; AA-AD gap = reliability artifact.
- The measured d[cell,pert] IS header sensitivity (logit distance with vs without `# Python code\n`).

**Surviving after R27:**
- Qwen3 within-group pert rho = +0.893 (robust under all analysis variants)
- Reliable-within/absent-across cell dissociation (split-half 0.85-0.95, cross ≈ 0)
- Fragile ~0.5 cross-family pert cluster (Qwen3↔Phi/StLM; dies under exclusions)
- Qwen3-only genuine cell-ranking shift (disattenuated AD = 0.82, CI excludes 1)
- IT > scale for SmolLM2 pert ordering (base↔instruct +0.624 vs base↔360M +0.157)

**SmolLM2-1.7B-Instruct COMPLETE (7th model, 4th admissible):**

| Model | Acc | Gap | CI | ResGap | eta2_op | Admissible |
|-------|-----|-----|----|----|---------|-----------|
| SmolLM2-1.7B-Instruct | 93.4% | 0.307 | [0.231, 0.397] | 0.452 | 0.500 | Yes |

IT comparison (SmolLM2-1.7B): acc 87.6%→93.4%, gap 0.275→0.307, eta2_op 0.687→0.500.
Base↔Instruct: pert-beyond-group rho +0.624 (p=0.001), cell-beyond-op +0.572 (p<0.001).

**Status:** 7/10 models complete. R27 demolished the gap as a semantic measure. 2×2
D-reliability diagnostic RUNNING on Qwen3-0.6B to determine if D unreliability is
structured (semantic) or token noise.

(33) **R27 2×2 D-reliability diagnostic — Qwen3-0.6B (2026-09-20).** DONE.
3312 forward passes, 364s CPU, 69 cells × 24 perts × 2 surfaces.
Result: D unreliability is STRUCTURED (5/6 ops have within-op rho 0.65-0.89). ADJACENCY
is the driver: O-O (+0.660) ≈ A-A (+0.696), both >> D-D (+0.392). temp_var is token noise.
Data: `experiments/results/transfer_spectrum/r27_2x2_qwen3_0.6b_base.json`

(34) **R27 2×2 D-reliability diagnostic — SmolLM2-360M replication (2026-09-20).** DONE.
3312 forward passes, 237s CPU, 69 cells × 24 perts × 2 surfaces. Accuracy: 55.9% (caveat).
**ADJACENCY REPLICATES:** O-O (+0.609) >> D-D (+0.412). All 6 ops structured (within-op 0.82-0.98).
**NUANCE:** In 360M, O is BETWEEN A and D (not ≈ A like Qwen3). A-O (0.409) < A-D (0.536) —
own-line ops probe different patterns from inert despite higher reliability. Adjacency affects
reliability universally but pattern identity is model-specific. temp_var (+0.821) is structured
in 360M vs anti-correlated in Qwen3 (-0.346).
Data: `experiments/results/transfer_spectrum/r27_2x2_smollm2_360m.json`

(35) **Answer identity confound test (4 models × 2 surfaces, 2026-09-20).** DONE → RETRACTION.
No new forward passes (reanalysis of isomorphic same-family data). D-D ≤ E-E-same-sign in ALL
8 tests. Isomorphic meaning effect RETRACTED — fully explained by answer identity confound.
Sign matching is the dominant correlation factor (R² increment 0.183). Meaning adds R² = 0.003.
See NOTEBOOK entry 62h-62i.
**R29 CORRECTION:** The 62h comparison was biased (D-D averaged cross-sign pairs, EE only
same-sign). Corrected DDsame vs EEsame: SmolLM2-1.7B +0.105 [0.025, 0.199] FL, +0.231
[0.116, 0.350] 11b (both excl 0). Meaning is small+model-dependent, not zero.

(36) **R29 adversarial review (2026-09-20).** Corrections to entries 62h/62i/adjacency.
Key findings: (1) meaning retraction overclaimed; (2) adjacency confounded with locus;
(3) disattenuated AD not D-specific; (4) 62i regression non-identifiable; (5) Phi failure
is inadmissible data. See NOTEBOOK entry 67.

(37) **R29 inside-line inert experiment — Qwen3-0.6B (2026-09-20).** DONE → DECISIVE.
4140 forward passes, 69 cells × 30 perts × 2 surfaces. Adds group I (in-line inerts:
`x = e  # c`, `x = e;`, `x = e ;`, `x = e  `, `x = e\t`, `x  = e`) alongside A/D/V/O.
Tests whether D unreliability = adjacency-to-expression or locus (in-line vs after-line).
**RESULT:** I-I = +0.708, A-A = +0.696 → line position effect +0.012 (ZERO).
D-D = +0.392, O-O = +0.660. A-I cross = +0.728. Expression modification effect = -0.316.
ADJACENCY-TO-EXPRESSION CONFIRMED, LOCUS HYPOTHESIS REFUTED. Per-pert: 5/6 I perts
reliable (0.757-0.799); anomaly: i_trail_space = 0.367. D-V structuredness replicates R27.
Data: `experiments/results/transfer_spectrum/r29_inline_inert_qwen3_0.6b_base.json`

(38) **R29 inline-inert SmolLM2-360M replication (2026-09-20).** DONE → REPLICATES.
4140 forward passes, 69 cells × 30 perts × 2 surfaces. Accuracy: 57.6% (below 80% caveat).
I-I = +0.814, A-A = +0.780, D-D = +0.412. Line position effect: +0.034 (near zero).
Corrected (excl i_trail_space/a_whitespace duplicate): A-A = +0.746, I-I = +0.789, diff = +0.044.
i_trail_space NOT anomalous in 360M (0.863 vs Qwen3's 0.367) — model-specific token sensitivity.
Data: `experiments/results/transfer_spectrum/r29_inline_inert_smollm2_360m.json`

(39) **R30 adversarial review of inline-inert (2026-09-20).** PROVISIONAL verdict.
15 concerns, 4 SERIOUS: (1) duplicate pert found (i_trail_space = a_whitespace),
(2) inert token reproduces D-sized drop → expression modification NOT necessary, only sufficient,
(5) I perts don't touch operand span → adjacency vs semantics still confounded,
(10) I group selection bias toward safe edits. Key recommendations: in-expression inerts,
larger pert groups, whitespace-token census. See NOTEBOOK entry 69.

(40) **R30b in-expression inert — Qwen3-0.6B-Base (2026-09-20).** DONE → CORRECTED post-Fable review.
69 cells × 36 perts × 2 surfaces, 1031s wall, 89.4% accuracy.
6 groups: A (after-line inert), I (in-line inert outside expression), J (in-expression
inert — NEW), D (in-line semantic), V (lexical variants of D), O (own-line semantic).
Pooled J-J = +0.413, but 3 SERIOUS defects found by Fable adversarial review:
(1) `//`-comment artifact on 15 cells dominates pooled rho (10x distance inflation),
(2) j_spaces_both is pseudo-replicate of j_space_before_op (rho 0.957),
(3) j_unary_first misclassified (adds AST node, belongs in D).
**Op-stratified (correct): J-J = 0.53-0.59 vs I-I = 0.87, D-D = 0.29. J is INTERMEDIATE.**
Neither adjacency nor semantic change predicts the ordering; "rarity fits" WITHDRAWN
post-R31 Fable review (pooling artifact — op-stratified loadings flat at 0.53-0.60
despite 3x surprisal range). NEW FINDING: `a  // b` read as C-comment on bare surface;
header restores floor division (surface disambiguates language, 10x effect). See NOTEBOOK 73.

(41) **R30b replication — SmolLM2-360M (2026-09-20).** DONE → UNRELIABLE (accuracy below threshold).
69 cells × 36 perts × 2 surfaces, 55.96% accuracy (below 80% threshold).
A-A=+0.669, I-I=+0.812, J-J=+0.621, D-D=+0.412. J-J > midpoint(A,D)=0.540,
consistent with J intermediate/inert-leaning. Not used as evidence — accuracy too low.
Fable review correctly warned against replicating this design. See NOTEBOOK 74.

(42) **R30b within/between decomposition — Stage 0 (2026-09-20).** DONE — ZERO COST.
Computed strat4 (op-stratified A-loading), betwOp (between-operator block correlation),
per-operator loadings with 10000-sample bootstrap CIs for all 36 R30b perts.
Saved: experiments/results/transfer_spectrum/r30b_decomposition.json.
Three-level plateau confirmed: A/I strat4 0.79-0.95, J 0.50-0.87, D/V -0.01 to 0.66.
j_space_after_op is outlier at 0.87 (nearly A-level, betwOp=1.00). Revised plateau:
in-expression AST-preserving edits cluster at 0.50-0.66 except space-after-op.
d_temp_var: strat4=-0.007 but betwOp=0.657 (preserves block structure, destroys cell ranking).
d_cond_true: strat4=0.581 but betwOp=0.086 (opposite: preserves cell ranking, destroys block).
See NOTEBOOK 79.

(43) **R31 Stage 1 — 5 new spacing/paren perts (2026-09-20).** DONE. CORRECTED post-Fable review.
69 cells × 7 perts (5 new + 2 anchors) × 3 passes (bare, header, surprisal).
Qwen3-0.6B-Base, fp32 CPU. 226s.
Fable adversarial review: 1/6 claims survive, 5 withdrawn.
SURVIVING: p_double < d_parens, cell-level paired bootstrap diff -0.376 CI[-0.60,-0.15].
WITHDRAWN: per-operator mechanism (within 1 SE), betwOp inversion (p=0.24), token-context
refutation (`%` artifact), rarity dead (untested as registered), CIs wrong (operator-level).
LEADING ALTERNATIVE: header shielding (depth-monotone attenuation). `%`-disambiguation
artifact confirmed. Spacing perts near-inert once artifact excluded. See NOTEBOOK 80.

(44) **R32 Nesting depth test (2026-09-20).** DONE.
69 cells × 2 perts (p_triple + a_empty anchor). Qwen3-0.6B-Base, fp32 CPU. ~30s.
Tests M-shield prediction: does strat4 decrease monotonically with paren depth?
Depth series: expr(0.924) → (expr)(0.655) → ((expr))(0.279) → (((expr)))(0.287).
Result: FLOOR at depth 2 (p_triple ≈ p_double, diff +0.007). NOT monotonic.
Per-operator mean distance: M-shield holds d0→d2 on header-sensitive ops (`-`, `%`),
but ALL 4 operators show small mean distance INCREASE at depth 3.
Tokenization: all depths ≥1 have 8 tokens (` (((` and `)))\n` are single tokens).
See NOTEBOOK 80 (R32 section).

(45) **R33 Zero-cost re-analysis — artifact exclusion + cell-level CIs (2026-09-20).** DONE.
Zero cost (reads saved R30b/R31/R32 data). Implements all Fable review recommendations.
Artifact exclusion rule (median > 3x A-median): catches j_space_before_op/`//`,
j_spaces_both/`//`, s_no_spaces/`%`, s_space_after_only/`%`, d_temp_var/`+,//,%`.
Cell-level stratified bootstrap (2000 iterations) for all 42 forms.
CLEAN LANDSCAPE: spacing perts near-inert (0.66-0.80), single-paren forms 0.58-0.66,
double/triple parens uniquely at 0.28. "Three-level plateau" was artifact of
disambiguation inflation. Paired contrasts confirmed. See NOTEBOOK 81.

(46) **R33b H_rarity as registered — 13-form test (2026-09-20).** DONE — INCONCLUSIVE.
69 cells × 8 R30b forms × 1 forward pass = 552 forward passes, 63s CPU.
Computed line surprisal for 8 R30b in-expression forms + 5 R31 forms = 13 total.
Saved: experiments/results/transfer_spectrum/r33b_r30b_surprisal.json.
H_RARITY (13 forms, raw Spearman): rho=-0.196, p=0.522. NOT SIGNIFICANT.
Partial on |token delta|: rho=-0.151. Registered threshold (rho ≤ -0.6, CI excl 0) FAILS.
Sign correct (higher surprisal → lower strat4) but effect weak and underpowered at n=13.
See NOTEBOOK 82.

(47) **Fable R34 design review — R34 ABANDONED (2026-09-20).**
Review of proposed M-shield test. Verdict: DO NOT RUN AS DESIGNED.
Key finding from existing C-group data: "Python evidence" is NOT the operative variable.
`pass` (Python keyword) RAISES header distance by 50%; `# start` (comment) COLLAPSES it.
Prefix effect tracks "first-line-is-a-comment", not P(Python|prefix).
Paren floor (~14) is 3× ABOVE comment-prefix floor (~4) — M-shield prediction fails.
p_double is genuinely REORDERED (Pearson vs A = 0.105), not "A scaled down."
M-SHIELD AS STATED IS REFUTED by existing C-group data + Pearson check.
RECOMMENDED REPLACEMENT: (a) C-group re-analysis as prefix family, (b) 2×2 factorial,
(c) sign-flip control (`# C code\n`). See NOTEBOOK 83.
