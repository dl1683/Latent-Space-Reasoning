# NOTEBOOK

## Entry 119: Theorem — quadratic superposition optimum is sparse-exact storage; power laws need retrieval geometry or dynamics (2026-09-18)

**Context.** Cycle HP (v2, theory/HD_POWER_LAW_ALLOCATION.md): why do trained
representations have power-law eigenspectra? First design (v1) was gated
NO-GO by the first headless Droid review (ledger hp1_design_gate_droid) with
five fixes and a strategic steer: analytic foundation cycle first. HP-A ran.

**A1 result.** The random-mixing (mean-field) stationarity analysis of the
quadratic superposition autoencoder predicts a bulge-with-cutoff norm
profile. L-BFGS multi-init optima refute it decisively (fit R² 0.00–0.08).
The true optimum: store the top-d usage tokens orthonormally, drop the tail,
loss = Σ_{i>d} p_i exactly (matched to 6 decimals, 6/6 configs), frame
operator = I_d, perfectly FLAT spectrum.

**Theorem (with one-line proof).** L = tr((G−I)P(G−I)), G PSD of rank ≤ d;
on ker G the factor G−I acts as −I, so L ≥ tr(P restricted to any
(n−d)-dim kernel); choose the kernel = tail coordinates. Attained by
orthonormal top-d storage. ∎

**Why this matters (HD-native reading).** In high dimensions the quadratic
reconstruction loss has NO superposition premium: when d ≪ n, interference is
avoided by dropping tokens, and the optimum is an uninteresting identity
frame. The cost of superposition — and therefore any power-law allocation —
only bites when every token must remain RETRIEVABLE, which is the softmax
world (margins vs. interference), i.e., the regime real embeddings train in.
This cleanly splits the "why power law" question:
1. dynamics branch: GD transients on ANY loss manufacturing steep spectra
   before converging to flat (P-E predicts QSAE GD endpoints flat);
2. geometry branch: softmax equilibrium genuinely power-law (XE arm, HP-B).
Both branches are now pre-registered with frozen predictions (P-A′..P-E).

**Process note.** First full loop under the standing orders: Droid gate →
redesign → analytic cycle → theorem. The gate's five fixes were all
material; the NO-GO was correct (v1's mean-field proposition was wrong in
both coefficient and conclusion — the optimum is not mixed at all).

Data: experiments/results/hpa_random_frame/a1_meanfield.json
Theory: theory/HD_POWER_LAW_ALLOCATION.md §4b
Ledger: hpa_a1_result

## Entry 118: Controlled eigenspace vulnerability INVERTS entry 92 — top eigenspaces are MORE sensitive (2026-09-18)

Entry 92's "tail vulnerability" claimed perturbation vectors projecting onto
low-eigenvalue subspaces cause MORE behavioral disruption. But with only 6
natural code-edit perturbation types (confounded magnitude and eigenspace
distribution), this correlation was ambiguous. This entry tests it with
CONTROLLED perturbations: fixed-magnitude injection into individual gate_proj
eigenspaces, post-layer-norm.

**Setup:** SmolLM2-135M. For each of 576 gate_proj eigenspaces at layers 0, 15,
29: inject perturbation eps * h_norm * v_k (eps=0.05) into the MLP input
(post-LN, so layer norm is controlled for). Measure output cosine distance
and KL divergence. Linearity check at 5 sampled eigenspaces with eps in
{0.01, 0.05, 0.1} confirms approximately quadratic scaling (expected for
cosine distance).

**Result — OVERTURNS entry 92 interpretation:**

| Layer | rho(eigenvalue, cos_dist) | beta (S ~ lambda^beta) | tail/top ratio |
|-------|---------------------------|------------------------|----------------|
| L0    | +0.357 (p<1e-18)          | +0.246 (R2=0.08)       | 0.35           |
| L15   | +0.625 (p<1e-63)          | +0.408 (R2=0.30)       | 0.12           |
| L29   | +0.399 (p<1e-23)          | +0.290 (R2=0.11)       | 0.07           |

All three layers: POSITIVE beta. Higher eigenvalue = MORE sensitivity per unit
perturbation. Top-20 eigenspaces are 3-14x more sensitive than tail-20. The
SNR prediction (beta=-0.5) is completely wrong.

**Mechanism:** The gate operator amplifies perturbations proportionally to
sqrt(lambda_k). Top-eigenspace perturbations get amplified MORE through the
gate projection. This amplification dominates the sensitivity pattern.

**Reconciliation with entry 92:** Entry 92 measured NATURAL perturbations (code
edits) propagated through ALL layers. The model's multi-layer processing
ABSORBS perturbation energy from top eigenspaces through compensatory dynamics
in layers 0-28. By the time a perturbation reaches L29, its top-eigenspace
component is suppressed (compensated), while its tail component survives. Entry
92's correlation was between "what the model failed to compensate" (tail energy)
and "how much damage remains" (output sensitivity). This is a PROCESSING
artifact, not an eigenspace property.

**Reframed interpretation of the power-law spectrum:**
- Old (entry 92): "Power-law spectrum IS the vulnerability gradient" — WRONG
- New: Power-law spectrum creates an AMPLIFICATION gradient. Top eigenspaces
  are both more powerful (more signal) and more dangerous (more amplification
  of noise). The model compensates by developing DEFENSE IN DEPTH: earlier
  layers absorb top-eigenspace perturbations through learned dynamics. The
  tail is not inherently fragile — it's just undefended because it's less
  critical. Entry 92's tail vulnerability is the leftover from multi-layer
  compensation, not an eigenspace property.

**Implication for entry 94 (information-theoretic priority):** Entry 94 found
rare tokens allocated to top eigenspaces. Combined with this finding: rare
tokens get the most-amplified (most-dangerous) eigenspaces but also the
most-defended ones. The model invests both spectral energy AND defensive
depth in critical information. The tail tokens (common, context-dependent)
are in less-amplified, less-defended eigenspaces — they're cheap to process
and cheap to lose.

Data: experiments/results/frame_analysis/eigenspace_vulnerability.json,
      eigenspace_vulnerability_full.json
Script: scratchpad/eigenspace_vulnerability.py

## Entry 117: General HD math — interference hierarchy and representation-computation gap (2026-09-18)

**Pivot:** From model-specific experiments to general mathematical exploration
of HD spaces. Scope: pure math, stat phys, info theory, RMT, kernel methods.

**Analytical result 1: Interference hierarchy.**
For k features superposed in R^d, processed by a degree-p polynomial map:
- 1-body terms (individual features): O(1)
- 2-body terms (pairwise interference): O(1/√d) RMS
- t-body terms: O(d^{-(t-1)/2}) RMS

Proven for quadratic case (p=2). For symmetric M ∈ R^{d×d}, x = Σ c_i f_i:
- Signal: f_i^T M f_i has E = tr(M)/d = O(1)
- Interference: f_i^T M f_j (i≠j) has E = 0, RMS = ||M||_F/d = O(1/√d)

This is the HD analogue of the cluster expansion (stat phys), 1/N expansion
(QFT), and Hoeffding decomposition (statistics). Dimension d plays the role
of the large parameter. The "mean field" is independent feature processing.

**Analytical result 2: Representation-computation gap.**
- Representation capacity: O(exp(d)) features via packing bounds
- Linear computation capacity: O(√d) simultaneous features (coherence-limited)
- Bilinear computation capacity: O(1) — dimension-INDEPENDENT

Key insight for bilinear: both signal terms (g_{ij} u_{ij}) and interference
terms (g_{ij} u_{il}) have identical distributions for random features.
ISR_bilinear = √(k-1), independent of d. Reliable bilinear computation: k ≤ 2.

Training closes the gap by STRUCTURING weights: gating selects neurons,
encoding/decoding create dedicated slots. Number of structured computations
scales with m (intermediate dim), not d (representation dim).

**Analytical result 3: Rate-distortion optimal allocation.**
For features queried with Zipf(s) probabilities, optimal eigenvalue allocation:
λ_i* ∝ √p_i ∝ i^{-s/2}. Optimal exponent α* = s/2.

Proof: Lagrange optimization of E = σ² Σ p_i/λ_i subject to Σ λ_i = P.
Observed model exponents α = 0.35-0.90 are steeper than optimal — gradient
descent over-allocates to frequent features.

**Analytical result 4: Computation capacity bounds.**
For incoherent interference (random features): k < √d.
For coherent interference (worst case): k < d^{1/4}.
Real features: between these extremes.
Depth requirement: n/√d layers for n total features.

**Open threads:** Formal proof for general degree-p; connection to Wiener chaos;
spectral calculus (evolution equations for eigenvalue distributions); phase
transitions in superposition capacity; HD calculus (gradient concentration).

Theory notes: scratchpad/interference_hierarchy.md, scratchpad/hd_computation_capacity.md

## Entry 113: Hidden state spectra flatten with depth — complementary specialization (2026-09-18)

**Question:** WHY do MLP weight spectra steepen with depth (entry 112)?
- Hypothesis A (gradient-driven): later layers get stronger gradients → more structure
- Hypothesis B (residual-driven): residual stream accumulates structure → weights adapt

**Method:** Forward pass of SmolLM2-135M on 677 tokens of real text (CPU). Extract hidden states at each layer, compute covariance eigenspectrum, fit power law.

**Key result: Hidden state spectra FLATTEN with depth (rho=+0.871, p<0.001).**

| Layer | Hidden state exp. | Weight up_exp | Norm mean |
|-------|-------------------|---------------|-----------|
| Emb   | -5.42            | —             | 2.7       |
| L0    | -4.26            | -0.68         | 45.2      |
| L5    | -2.33            | -0.65         | 90.5      |
| L10   | -2.22            | -0.68         | 137.0     |
| L25   | -2.09            | -0.81         | 538.4     |
| L29   | -2.21            | -0.79         | 51.0      |

Weights steepen (rho=-0.89); hidden states flatten (rho=+0.87). **Anti-correlated.**

**This rules out Hypothesis B.** If weights were adapting to a steepening input spectrum, they'd steepen WITH the input. Instead, the input spectrum flattens while weights steepen — they move in opposite directions.

**Interpretation: Complementary specialization.**
- Early layers: steep hidden states (few dominant directions) + flat weights (general-purpose MLP)
- Late layers: flat hidden states (diverse representation) + steep weights (selective MLP)
- The residual stream ACCUMULATES diverse information (flattening), so later MLPs NEED to be more selective (steepening) to extract relevant components

This is an efficiency principle: MLP weight specialization compensates for representation diversity. The model doesn't waste weight selectivity where the input is already concentrated, and increases selectivity where the input is diffuse.

**CONFOUND RESOLVED: Layer norm.**
The MLP sees layer_norm(residual), not the raw residual. Does LN equalize
spectra across layers, killing the anti-correlation? Test: captured
post_attention_layernorm outputs via hooks.
- Post-LN exponent depth trend: rho=+0.446 (p=0.014) — still flattens
- LN reduces exponent variability by 60% (raw std=0.139, LN std=0.055)
- But the depth gradient SURVIVES normalization
- Post-LN erank DECREASES with depth (rho=-0.77) — complex spectral reshaping
- Complementary specialization interpretation is ROBUST to the LN confound

**Remaining caveats:**
- Erank crashes to 1-3 at L15-L25 (sink token dominance)
- 325-677 tokens may underestimate covariance; needs cross-input verification
- Single model (SmolLM2-135M); needs cross-architecture check

**Emerging depth story (caveated per Fable review):**
1. Embeddings: power-law from Zipf (entry 101)
2. Post-LN hidden states: flatten through depth (rho=+0.45, p=0.01) — confirmed robust to LN confound
3. MLP up_proj weights: steepen through depth (rho=-0.89) — gate steepening is model-specific
4. Gate-up pairing: zero in bulk, strong positive at near-output layers (entry 111)
5. Attention: depth trend unresolvable at current resolution (entry 112)

## Entry 116: Neuron-eigenspace coverage — weak gradient, depth-dependent (2026-09-18)

**Question:** Does the spectral structure explain tail vulnerability (entry 92) by concentrating fewer neurons onto tail eigenspaces?

**Method:** For each eigenspace k (from embedding eigenbasis), count gate neurons with above-uniform projection. SmolLM2-135M, all 30 layers, weight-only.

**Result:** Coverage ~ eigenspace rank: rho = -0.773 (p < 1e-115). Top eigenspaces have MORE neuron coverage. BUT the absolute difference is only ~3%: top-10 eigenspaces = 502 neurons, bottom-10 = 491. The coverage is nearly UNIFORM despite the power-law weight spectrum.

**Depth dependence of coverage gradient:**
- L0: rho = +0.27 (tail has slightly MORE coverage)
- L10: rho = -0.14
- L20: rho = -0.37
- L29: rho = -0.88 (top has MUCH more coverage)

The coverage gradient steepens with depth, mirroring the weight spectral gradient (entry 112). Late layers concentrate neurons onto top eigenspaces.

**Partial explanation for tail vulnerability:** The ~3% average coverage deficit for tail eigenspaces is statistically robust but too small to fully explain the strong sensitivity gradient (rho = -0.66 to -1.00) from entry 92. Other factors likely dominate: eigenvalue-based norm weighting, gating threshold effects, or attention error correction.

## Entry 115: Spectral exponent does NOT predict layer sensitivity — NEGATIVE (2026-09-18)

**Prediction tested:** Layers with steeper up_proj spectra (late layers) should be more sensitive to ablation, because spectral specialization means less redundancy.

**Method:** Zero-ablation of up_proj at each layer of SmolLM2-135M. Measure output divergence (cosine distance and KL divergence from baseline) across 42 tokens of real text.

**Result: PREDICTION FAILS.**
- Exponent ~ cos_dist: rho = +0.275, p=0.14 (not significant, wrong sign)
- Sensitivity is U-SHAPED: L0 cos_dist=0.39, L29=1.07, middle layers 0.004-0.12
- Position (first/last bottleneck) dominates over spectral structure
- The partial correlation (controlling for norm) gives rho=+0.43 (p=0.02) but wrong direction

**Interpretation:** The spectral depth gradient is a TRAINING phenomenon — it describes how gradient descent organizes weights at different depths. It does NOT predict functional sensitivity. First and last layers are critical bottlenecks regardless of spectral properties; middle layers are functionally redundant.

**This limits the theory:** The spectral steepening (entry 112), complementary specialization (entry 113), and scale invariance (entry 114) are all DESCRIPTIVE — they characterize training's organizational fingerprint. But spectral structure alone does not predict which layers matter for computation. Behavioral relevance requires different tools.

**Data:** experiments/results/frame_analysis/spectral_sensitivity.json

## Entry 114: Scale test — steepening confirmed at 360M, ~scale-invariant (2026-09-18)

**Test:** Does the up_proj steepening prediction (entry 112) hold at larger scale?

SmolLM2-360M (d=960, 32 layers): up rho = -0.862 (p<0.001). Prediction confirmed.

| Model | d | Layers | up rho | Normalized rate | Early | Late |
|-------|---|--------|--------|-----------------|-------|------|
| SmolLM2-135M | 576 | 30 | -0.891 | -0.162 | -0.675 | -0.792 |
| SmolLM2-360M | 960 | 32 | -0.862 | -0.142 | -0.664 | -0.764 |
| Qwen3-0.6B | 1024 | 28 | -0.649 | — | -0.632 | -0.726 |

SmolLM2 rates are similar (-0.162 vs -0.142, 12% reduction at 2.7x parameters). Qwen3 shows weaker steepening (rho=-0.649) — possibly architecture-specific (different normalization, tokenizer, training). The prediction now has 3/3 confirmations across 2 model families.

**Data:** experiments/results/frame_analysis/scale_steepening.json

## Entry 112: Spectral depth gradient — generalization to specialization (2026-09-18)

**Question:** Do spectral properties change systematically with layer depth, and if so, how do all components (MLP, attention, pairing) fit together?

**Method:** Combined per-layer analysis of MLP Gram eigenspectrum exponents (gate_proj, up_proj), attention QK/OV effective ranks, and gate-up pairing z-scores. Spearman correlations with depth, cross-component correlations, first-5 vs last-5 layer comparisons. Both SmolLM2-135M (30 layers) and Qwen3-0.6B (28 layers).

**Core finding: MLP spectral steepening with depth.**

| Metric | SmolLM2 rho (p) | Qwen3 rho (p) |
|--------|-----------------|----------------|
| up_exp vs depth | -0.891 (p<0.001) | -0.649 (p<0.001) |
| gate_exp vs depth | -0.698 (p<0.001) | -0.278 (p=0.15) |
| pair_nsig vs depth | +0.264 (p=0.16) | +0.657 (p<0.001) |
| qk_erank vs depth | -0.295 (p=0.11) | -0.347 (p=0.07) |

The up_proj power-law exponent steepens monotonically with depth in BOTH models. Gate follows but less universally. First-5 vs last-5 layer difference: up_exp shifts from -0.67 to -0.79 (SmolLM2) and -0.63 to -0.73 (Qwen3), both highly significant.

**Interpretation: Generalization-to-specialization gradient.**
- Early layers: flatter MLP spectra → more uniform singular value distribution → "general purpose" computation using many spectral directions
- Late layers: steeper MLP spectra → concentrated on top directions → "specialized" computation targeting fewer spectral components
- The pairing dipole (entry 111) sits at the extreme end of this gradient, where spectral concentration peaks

**Attention circuits are NOT part of this gradient.**
- QK effective rank: 80-82% of d_head, no significant depth trend
- OV effective rank: 93-96% of d_head, no significant depth trend
- The steepening is specific to MLP weight matrices, not attention

**Cross-component correlations:**
- gate_exp ~ up_exp: rho = +0.79 (SmolLM2), +0.77 (Qwen3) — gate and up steepen together
- gate_exp ~ qk_erank: rho = +0.41 (SmolLM2, p=0.02), +0.40 (Qwen3, p=0.04) — layers with steeper MLP spectra have lower QK rank (mild coupling)

**Gate-up asymmetry (gate steeper than up):**
- Mean: gate -0.10 steeper than up in both models
- SmolLM2: constant through depth (rho=-0.06 ns)
- Qwen3: asymmetry INCREASES with depth (rho=+0.58, p=0.001)

**Testable prediction:** In ANY SwiGLU transformer, the up_proj Gram eigenspectrum steepens monotonically with depth (rho < -0.5). Verifiable with weight analysis alone, no inference needed. NOTE: gate_proj steepening is model-specific (rho=-0.70 SmolLM2 vs -0.28 ns Qwen3), so the prediction is restricted to up_proj.

**Attention null caveat (Fable review):** The attention QK/OV "no trend" is a weak null — only ~64 nonzero singular values per head, 28-30 layers. Absence of evidence is not evidence of absence; a depth trend could be undetectable at this resolution.

**Open questions:**
1. Is the steepening driven by gradient magnitude or residual accumulation? (Partially answered by entry 113: post-LN input flattens, supporting input-driven mechanism.)
2. Does the rate of steepening predict model quality?
3. Does the steepening emerge early in training or gradually?

**Data:** experiments/results/frame_analysis/attention_spectra.json, real_model_pairing.json

## Entry 111: Real-model pairing verification — near-output anomaly (2026-09-18)

**Question:** Does the additive anti-pairing (entry 110, z=-7.79) or any residual spectral correlation appear in real gated models?

**Method:** Signed-product pairing statistic S_k = Σ_i (gate_i · v_k)(up_i · v_k) for SmolLM2-135M across all 30 layers, using the embedding eigenbasis. 200 shuffles per layer. Pure weight analysis, no inference.

**Results — confirmation:**
- Aggregate: top-5 z = -0.11 ± 0.66, t-test p=0.375 (not significant)
- 28 of 30 layers: z ≈ 0, consistent with toy gated condition
- No depth trend (Spearman rho = -0.022, p=0.908)
- This confirms: gating destroys spectral anti-pairing in real models

**Results — anomaly at layer 28 (penultimate):**
- top-5 z = +2.48, top-10 z = +2.95, all z = +1.05
- 146/576 eigenvectors significant (25% vs expected 4.5%)
- 137 POSITIVE, 9 negative — strong directional effect
- max z = 11.5 for individual eigenvectors
- Under N(0, 1/sqrt(5)), top5_z=2.48 is 5.5σ; Bonferroni-corrected for 30 layers: p ≈ 1e-6

**Interpretation:**
- The penultimate layer develops POSITIVE pairing: gate and up co-project onto the SAME embedding eigenvectors with the SAME sign
- This is the opposite of additive anti-pairing — it's spectral amplification
- In SwiGLU, same-sign projection means: when gate detects eigenvector k is active (positive dot product → silu passes), up also has energy along k, so the product amplifies k
- Each MLP neuron becomes a "spectral switch" — selectively amplifying specific embedding eigenvectors
- Layer 27 also shows mild elevation (78 significant, max z = 6.9), but the effect peaks at L28
- Layer 29 (final) returns to baseline — possibly because L29's MLP doesn't feed unembed directly (goes through attention + norm first)

**Leading hypothesis:** The loss gradient is most concentrated at the penultimate layer. Gate and up receive correlated gradient components that push them toward the same spectral directions. This gradient-driven co-adaptation creates the pairing signal that is absent in earlier layers where gradients are more diffuse.

**Cross-model verification (Qwen3-0.6B, 28 layers, d=1024):**
- Aggregate: top-5 z = -0.16, p=0.293 (not significant, same as SmolLM2)
- L27 (final): top5_z = +1.80, 291/1024 significant (240 positive), max z = 14.3
- L26 (penultimate): top5_z = -2.35, 142/1024 significant (122 NEGATIVE), max z = 5.6
- **SPECTRAL DIPOLE at output**: L26 anti-paired (negative z) → L27 pro-paired (positive z)

**Both models show the same pattern:**
| Model       | Layers | Peak+ layer     | Peak z | Sig fraction | Preceding layer pattern |
|-------------|--------|-----------------|--------|--------------|------------------------|
| SmolLM2-135M| 30     | L28 (penult.)   | +2.48  | 25%          | L27 mixed, L29 mild -  |
| Qwen3-0.6B  | 28     | L27 (final)     | +1.80  | 28%          | L26 strongly negative  |

Both: ~25-28% of eigenvectors significantly paired at the peak layer, overwhelmingly positive. Confirmed cross-architecture.

**Refined interpretation (post-Fable review):** Near-output layers (last 1-2) develop strong positive gate-up spectral pairing. The positive peak concentrates gate-up projections along the same eigenvector directions, creating "spectral switches" that selectively amplify specific embedding spectral components.

**Fable corrections applied:**
- Peak location is "near-output" (last 1-2 layers), not specifically "penultimate" — SmolLM2 peaks at L28 (penultimate), Qwen3 at L27 (final)
- The "spectral dipole" (negative preceding positive) is QWEN-SPECIFIC: SmolLM2 L27 is mixed (37+/41-), not the strong negative the dipole narrative requires. SmolLM2's pattern is more pulse than dipole. Drop dipole as universal claim until a third model confirms.

**Open questions:**
1. Does the pairing peak magnitude correlate with model quality?
2. Does the pairing peak deepen (span more layers) in larger models?
3. What's the causal relationship to the output projection — does tying/untying weights matter?
4. Is a third model's near-output pattern more SmolLM2-like (pulse) or Qwen3-like (dipole)?

## Entry 110: Spectral-composition bridge — NEGATIVE, with surprise (2026-09-18)

**Hypothesis tested:** Does training on Zipfian data jointly organize spectral
structure (embedding eigenspectrum) AND compositional pairing (gate-up neuron
correspondence)? Specifically: do the most important gate-up pairs align with
the top eigenvectors of the embedding's frame operator?

**Design (Fable-reviewed v2):** Minimal model (n=1024, d=64, h=128, untied
unembed) trained 5000 steps on Zipfian marginal data (s=0.5, index-decoupled).
Pair-sensitive signed-product statistic S_k = Σ_i (gate_i·v_k)(up_i·v_k)
compared to 100 shuffled-pair nulls. Four conditions × 5 seeds:
1. Gated MLP on Zipf data (main)
2. Frozen Zipf embedding + random MLP (arithmetic control)
3. Additive MLP on Zipf data (gating control)
4. Gated MLP on uniform data (frequency control)

**Results:**

| Condition | Top-5 z (mean ± std) | All z (mean ± std) |
|-----------|---------------------|---------------------|
| Gated (main) | -0.35 ± 0.90 | -0.24 ± 0.61 |
| Frozen-embed | -0.05 ± 0.40 | -0.07 ± 0.33 |
| **Additive** | **-7.79 ± 0.35** | **-5.92 ± 0.03** |
| Uniform | +0.10 ± 0.68 | +0.08 ± 0.08 |

Gated top-5 z: t=-0.78, p=0.76 (one-sided, Bonferroni). NOT significant.
Gated > Frozen: p=0.70. NOT significant.
Gated > Additive: p<0.0001. SIGNIFICANT (but in the wrong direction —
gated is LESS anti-paired, not more paired).
Norm-partialled importance-alignment rho: +0.021, p=0.35. NOT significant.

**VERDICT: Bridge hypothesis REJECTED.**

Training does not create spectrally-organized gate-up pairing in the gated
MLP. Neither the pairing statistic nor the importance-alignment correlation
reaches significance.

**SURPRISE FINDING: Additive composition creates strong spectral anti-pairing.**

The additive MLP (gate + up) develops z=-7.79 anti-correlation in the
signed product along top eigenvectors. This means gate and up rows
systematically project with OPPOSITE signs onto the embedding's dominant
eigenvectors. The gated MLP has z≈0 — gating DISRUPTS this anti-pairing.

**Mechanism (theoretical):** In additive composition (down @ (gate + up)),
if gate_i and up_i both project positively onto the same eigenvector, they
are redundant — their contributions to the sum are collinear. SGD pushes
them to opposite signs for information efficiency. In gated composition
(down @ (gate * up)), the gradient couples gate and up through their
product, making anti-pairing suboptimal — the product of opposite signs
is negative, which after silu (always ≥0 in the gate) wastes capacity.

**Implication for composition sensitivity:** This EXPLAINS the gating
amplification finding from entry 106 (1.46x vs additive). Additive MLPs
develop spectrally-determined pairing (gate and up in complementary roles
along each eigenvector). Gated MLPs prevent this — the pairing structure
is freed from spectral constraints and becomes input-dependent. The
"holographic" characterization from entry 105 is confirmed: gating pushes
compositional structure OUT of the spectral domain.

**Caveat:** The embedding barely develops spectral structure (exponent -0.261
vs random -0.25). In the embedding-only model (entry 101), n=1024, d=64
reached exponent -0.80. The full model spreads gradient across 5 parameter
groups, limiting embedding organization at 5000 steps. The bridge test may
have limited power because the spectral hierarchy is weak.

**What this changes:**
- Spectral structure does NOT predict composition → they are independent
- The predictive rewiring challenge (Droid-prescribed) becomes the right
  next step — find what DOES predict composition behavior
- Additive anti-pairing is a new structural insight about why gating helps

Script: scratchpad/spectral_composition_bridge.py (v2)
Data: experiments/results/frame_analysis/spectral_composition_bridge.json

## Entry 107b: Effective Zipf s=0.5 and quantitative prediction grid (2026-09-18, fork)

**Measured effective Zipf exponent for real subword tokenizers:**
- SmolLM2-135M: s = 0.55
- Qwen3-0.6B: s = 0.50
- Phi-1.5: s = 0.53

All three are ≈0.5, far below the word-level s≈1.0. BPE subword tokenization
flattens the frequency distribution. This explains why real model eigenvalue
exponents (-0.35 to -0.57) are much flatter than our toy model at s=1.0 (-1.75).

**Grid search (5×5 grid, d=64, 5000 training steps):**

| n/d\s | 0.3   | 0.4   | 0.5   | 0.6   | 0.7   |
|-------|-------|-------|-------|-------|-------|
| 8     | -0.55 | -0.93 | -1.29 | -1.53 | -1.70 |
| 16    | -0.28 | -0.44 | -0.81 | -1.15 | -1.39 |
| 32    | -0.18 | -0.20 | -0.39 | -0.75 | -1.04 |
| 64    | -0.13 | -0.13 | -0.17 | -0.42 | -0.72 |
| 128   | -0.09 | -0.09 | -0.09 | -0.19 | ~-0.5 |

**Predictions at real model parameters:**
- Phi-1.5 (s=0.53, n/d=25): predicted -0.68, observed -0.57 (21% off)
- SmolLM2 (s=0.55, n/d=85): predicted -0.24, observed -0.52 (2x too flat)
- Qwen3 (s=0.50, n/d=148): predicted -0.09, observed -0.35 (3.7x too flat)

The toy model UNDERPREDICTS steepness at high n/d because 5000 steps is
insufficient for full directional organization in highly overcomplete settings.
Real models train for millions of steps.

**Conclusion: three variables control the eigenvalue exponent:**
1. Zipf s of training data frequency distribution (~0.5 for BPE tokenizers)
2. Overcompleteness ratio n/d (larger = flatter)
3. Training convergence (more training = steeper, especially at high n/d)

**Testable prediction:** eigenvalue exponents should steepen over the course
of training, with the rate of steepening inversely proportional to n/d.

Files: scratchpad/effective_zipf.py, scratchpad/predict_real_exponents.py
Data: experiments/results/frame_analysis/effective_zipf.json

## Entry 102: Droid adversarial review — mixed moment program (2026-09-18)

**Verdict: narrow result survives, broad claims must be withdrawn.**

Droid reviewed entries 97-100 and the formal theorem. Key findings:

**1. Theorem is incorrect as stated.** Multiple proof errors:
- Statement says D1 constant at α=1/2 but proof finds log divergence
- For α<1/2, shared rank contribution is O(1) nonzero, not d^{2α-1}→0
- H(s,δ) diverges for s>1; derivation assumes α<1 but claims all α>0
- Tail sum error; missing concentration argument
- Measured exponents (1.1-1.8) are outside the α<1 derivation range

**2. Row-permutation killer argument.** Permuting only up_proj rows preserves
S_up = W_u^T W_u, all D_k, all marginal spectra — but changes which gate and
up neurons are multiplied. Network function changes drastically while every
proposed invariant is fixed. D_k CANNOT be sufficient for SwiGLU composition.

**3. Experimental methodology issues:**
- Only tested α=0.8 synthetically; never tested critical transition at α=0.5
- Z-score is non-standard (effect/pooled SD, not inferential)
- Free probability comparison uses wrong prediction (not free convolution)
- No spectrum-preserving null for real models
- "Effective sharing fraction" divides by unmatched synthetic benchmark

**4. Constant D1 is not yet surprising.** Three confounded checkpoints
(depth, training, and width all vary). Post hoc compensation explanation.
Calling 0.21 a universal budget is premature.

**What survives:** Gate and up input-side Gram operators exhibit positive
centered Frobenius alignment, and three SmolLM2 checkpoints have similar
layer-averaged alignment. This is valid but not yet surprising or predictive.

**Prescribed next step:** Row-permutation challenge. Permute up_proj rows
(preserving all D_k exactly), measure held-out MLP outputs and downstream
loss. Include function-preserving control (common permutation with
compensating down_proj permutation). If behavior changes while D_k is fixed,
abandon sufficiency claim and derive row-aware composition invariant.

**Assessment:** The row-permutation argument is correct and fatal to the
sufficiency claim. The theorem needs withdrawal and correct rederivation.
The descriptive finding (positive alignment exists) stands but needs to be
connected to actual network behavior, not just weight statistics.

Source: scratchpad/droid_mixed_moment_review.md

## Entry 104: Row-wise cosines are zero — the pairing paradox (2026-09-18)

**Result: Individual gate-up row pairs show NO directional alignment, yet
row pairing determines the function.**

SmolLM2-135M, all 30 layers, 1536 gate/up neuron pairs per layer.

| Metric | Observed | Random baseline |
|--------|----------|-----------------|
| Mean row cosine cos(g_i, u_i) | -0.0009 | 0.0 |
| Fraction cosines > 0 | 0.499 | 0.5 |
| Cosine range across layers | -0.007 to +0.004 | ~0 |
| Depth trend | None | N/A |
| Weighted cosine (by down_proj norm) | -0.007 to +0.004 | ~0 |

Cross-matrix diagonal dominance |C_ii|/max_j|C_ij| = 0.41 mean, vs ~0.21
for iid random (2x above random, moderate).

**The paradox:** 
- D_k (Gram-level statistics) show significant shared structure (D1_norm=0.22)
- Row-wise cosines show NO individual-pair alignment (cos ≈ 0)
- Yet row permutation causes 139% MLP output error (entry 103)

**Resolution:** In d=576, ANY two random vectors have near-zero cosine.
The 1536 gate rows and 1536 up rows all live in a space where pairwise
cosines are ~0 by concentration of measure. But the IDENTITY of each vector
matters — which specific high-dimensional direction gates which specific
value direction. No scalar summary of individual pairs captures this; the
information is in the full combinatorial assignment.

**Implication for invariants:** Neither Gram operators (too coarse — entry 103)
nor row-wise statistics (too fine, washed out by high-d concentration) capture
the compositional structure. The relevant information exists at an INTERMEDIATE
level: perhaps the joint spectrum of W_gate @ W_up^T (the full cross-matrix),
or the action of the paired (gate_i, up_i) on specific input distributions.

This is a genuine high-dimensional phenomenon: summary statistics at every 
natural level are insufficient, because the space of possible pairings is
combinatorially large and the "correct" pairing is not distinguished by any
per-pair measure.

Script: experiments/row_aware_analysis.py
Data: experiments/results/row_permutation/row_aware_analysis.json

## Entry 105: Cross-matrix SVD — still blind to composition (2026-09-18)

**Result: The singular value spectrum of W_gate^T @ W_up is nearly identical
for original and randomly permuted row pairings.**

SmolLM2-135M, all 30 layers. Computed G = W_gate^T @ W_up (d x d) for
original pairing and one random permutation (seed 42), then compared SVD.

| Metric | Original/Permuted ratio | Interpretation |
|--------|------------------------|----------------|
| Nuclear norm (total coupling) | 1.004 | Identical |
| Frobenius norm | 1.038 | Nearly identical |
| Spectral norm (top SV) | 1.44 (range 0.93-5.87) | Sporadic, inconsistent |
| Effective rank | 410 / 413 | Identical |
| Power-law exponent | 0.372 / 0.370 | Identical |

**Three levels of invariant tested, all insufficient:**

| Level | Invariant | Captures row pairing? |
|-------|-----------|----------------------|
| Gram operators (D_k) | W^T W | No (entry 103, 0% change) |
| Row-wise statistics | cos(g_i, u_i) | No (entry 104, ≈ random) |
| Cross-matrix SVD | svd(W_gate^T @ W_up) | No (entry 105, ≈ permuted) |

Yet row permutation causes 139% MLP output error (entry 103).

**Theoretical significance:** The computational structure of SwiGLU MLPs
is HOLOGRAPHIC with respect to the weight matrices: the information about
which gate row works with which up row is not concentrated in any spectral
component, summary statistic, or low-rank approximation. It is distributed
across the full joint structure of the entries and manifests only through
computation on inputs.

This has implications for:
- Pruning: you cannot identify "important" gate-up pairings from weight
  statistics — importance is input-dependent
- Transfer: aligning gate and up subspaces via spectral methods will not
  preserve compositional behavior
- Interpretability: the "meaning" of an MLP neuron (its gate-up pairing)
  is not readable from weight norms, cosines, or spectral position

**Next question:** Is there ANY weight-only test that distinguishes the
correct pairing? Or is composition fundamentally data-dependent — requiring
activation statistics to characterize?

Script: experiments/cross_matrix_svd.py
Data: experiments/results/row_permutation/cross_matrix_svd.json

## Entry 106: Gating drives pairing sensitivity, not silu (2026-09-18)

**Result: The multiplicative gating (elementwise multiply) is the primary
source of row-permutation sensitivity, not the silu nonlinearity.**

Three computation modes under the same row permutation:

| Mode | Formula | Mean rel error | vs SwiGLU |
|------|---------|---------------|-----------|
| SwiGLU | down @ (silu(g) * u) | 1.39 | 1.00x |
| Linear gated | down @ (g * u) | 1.28 | 0.92x |
| Additive | down @ (g + u) | 0.88 | 0.63x |

SwiGLU/linear ratio = 1.09: silu adds only 9% extra sensitivity.
Gated/additive ratio = 1.46: the elementwise multiply adds 46% sensitivity.
All three modes show substantial error, but the multiplicative coupling
between gate and up vectors is the dominant amplifier.

**Why gating amplifies:** In the gated case, error for neuron i is:
  gate_i * (up_{perm(i)} - up_i)
The gate acts as a POSITION-DEPENDENT AMPLIFIER. Neurons where gate_i is
large amplify the permutation error more. This creates input-dependent
sensitivity that additive composition lacks.

**Why additive is still ~0.88:** Even without gating, permuting 1536 rows
in d=576 creates a different m-dimensional projection of the input. The
relative error of ~0.88 reflects the near-orthogonality of random vectors
in high dimensions: up@x and up_perm@x are nearly independent.

**Theoretical implication:** The "native math" of gated MLPs is the
MULTIPLICATIVE coupling — the elementwise product in the intermediate space.
This creates a tensor (not matrix) structure that is:
1. Stronger than additive composition (1.46x amplification)
2. NOT further amplified by activation nonlinearity (1.09x)
3. Not captured by any matrix decomposition of individual weight matrices
4. Fundamentally input-dependent (gate amplifies differently for each input)

Script: experiments/bilinear_vs_gated.py
Data: experiments/results/row_permutation/bilinear_vs_gated.json

## Entry 107: Droid strategic review — tensor overclaim, gauge symmetry (2026-09-18)

**Verdict: Entry 103 is a clean falsification. Entries 104-106 overclaim.**

Key corrections from Droid:

**1. Entry 105 has a mathematical error.** The m×m cross-matrix C = GU^T
has SVD EXACTLY INVARIANT under row permutation (C' = CP^T, P orthogonal).
The script computed G^T U (d×d version), which can differ but happened to
be similar empirically — likely because row statistics are approximately iid.
The "finding" is a near-tautology, not an empirical discovery.

**2. The 139% error ≈ sqrt(2) (random decorrelation).** This is the expected
relative distance between uncorrelated vectors of comparable norm. The result
shows random permutation decorrelates the output, not that composition has
special "structured destruction." Need output cosine and norm ratio to interpret.

**3. Entry 106 doesn't identify causality.** The 0.88 additive error shows
most sensitivity comes from breaking the DU product, not gate-up tensor
coupling specifically. The 1.46x gated/additive ratio compares functions
with different scales — apples to oranges.

**4. The correct framing is GAUGE SYMMETRY, not tensor phenomenon.** Entry 103
demonstrates hidden-unit permutation symmetry: you must permute consistently
across all incident weights. This is well-known and holds in any dimension.
The high-d aspect (cosine concentration) explains entry 104 but doesn't make
entry 103 a HD phenomenon.

**5. Overclaims to WITHDRAW:**
- "Composition is fundamentally a tensor phenomenon"
- "Holographic" information distribution
- "No weight-only invariant captures composition"
- "Genuine high-dimensional phenomenon" (gauge symmetry is dimension-free)
- Implications for pruning, transfer, interpretability
- "Gating drives the sensitivity" (additive has 63% of the effect)

**6. What survives:**
- D_k sufficiency is cleanly killed (entry 103) — real contribution
- Diagonal dominance 0.41 vs ~0.21 random may be a real row-level signature
  (needs proper null, not investigated)
- The correct characterization: "Any invariant factoring only through G^T G
  and U^T U cannot characterize a SwiGLU block."

**7. Prescribed next step: Held-out predictive rewiring challenge.**
Compare predictors of permutation damage: random-decorrelation null, DU
change, row norms/cosines, spectral summaries, static ||ΔT||, activation-
conditioned metrics. Predict held-out MLP output MSE, downstream KL, across
unseen prompts and layers. Decision rule:
- Static T predicts → continue weight-only calculus
- Only activation metrics predict → pivot to data-dependent analysis
- Neither beats DU/decorrelation nulls → close this line

Source: scratchpad/droid_tensor_review.md

## Entry 108: Predictive rewiring challenge — gate info is essential (2026-09-18)

**Result: Gate-aware tensor predictor robustly predicts permutation damage
on held-out data (R2 = 0.90), while gate-blind DU predictor fails (R2 = -2.4).**

SmolLM2-135M, 5 layers (0,7,15,22,29), 60 partial permutations per layer
(10% to 100% row corruption), train on 10 prompts, evaluate on 5 held-out.

Four predictors compared:

| Predictor | Formula | Mean R2 | Mean corr |
|-----------|---------|---------|-----------|
| Decorrelation null | sqrt(2)*norm(y) | 0.00 | 0.00 |
| DU change (no gate) | norm(D*(Ux-PUx)) | -2.42* | 0.97 |
| Tensor (gate-aware) | norm(D*(Gx)*(Ux-PUx)) | 0.90 | 0.98 |
| Activation-weighted | norm(D*silu(Gx)*(Ux-PUx)) | 1.00** | 1.00 |

*DU has high correlation but catastrophic scale mismatch (L7: R2=-15.2)
**Activation predictor is TAUTOLOGICALLY PERFECT — it computes the same
formula as the actual error. This is a design flaw, not a finding.

Per-layer tensor R2: L0=0.98, L7=0.86, L15=0.95, L22=0.72, L29=0.96.
Consistently strong. DU R2: L0=0.66, L7=-15.2(!), L15=0.91, L22=0.65,
L29=0.89. Catastrophically unstable.

**Why DU fails:** The DU predictor ||D(Ux - PUx)|| ignores which neurons
are gated strongly. When a permutation affects neurons that gate projects
strongly onto the input, the damage is amplified. The DU predictor can't
capture this input-specific amplification, so it gets the magnitude wrong
(especially at L7 where gate structure apparently matters most).

**Why tensor works:** The tensor predictor includes Gx as a position-dependent
weight. Neurons with large |g_i^T x| contribute more to the output and thus
more to the error when permuted. This gate-value coupling IS the essential
compositional structure.

**Decision per Droid's rule:**
The tensor predictor requires BOTH weights (G) AND input (x). It's not
purely weight-only. But the gate WEIGHTS are what make prediction possible.
This is a weight-data interaction: the weights determine which inputs are
gated; the inputs determine which neurons are active.

**Conclusion:** Neither purely weight-only nor purely data-dependent analysis
suffices. The compositional structure lives in the gate-value coupling through
specific inputs. A useful compositional invariant must characterize HOW the
gate modulates the value space conditional on input patterns.

**Next:** A truly weight-only predictor (e.g., ||ΔT||_F for the full tensor)
vs the input-conditioned tensor predictor. If the weight-only version is
competitive, gate structure in the weights alone is sufficient.

Script: experiments/predictive_rewiring.py
Data: experiments/results/row_permutation/predictive_rewiring.json

## Entry 109: Weight-only test — DU suffices, gate adds nothing (2026-09-18)

**Result: The weight-only tensor predictor ||ΔT||_F (which includes gate info)
performs identically to the simpler DU predictor ||DΔU||_F (which ignores gate).**

Three predictors, R2 for predicting average MLP output error across 5 prompts:

| Predictor | Mean R2 | Range | Gate needed? |
|-----------|---------|-------|-------------|
| Weight-only tensor ||ΔT||_F | 0.94 | 0.90-0.96 | Has gate |
| Weight-only DU ||DΔU||_F | 0.95 | 0.90-0.97 | No gate |
| Input-conditioned tensor | 0.96 | 0.92-0.99 | Gate + input |

All three are near-identical. The gate information in the weights does NOT
improve prediction of permutation damage.

**Resolution of the entry 108 discrepancy:** In entry 108, DU had R2 = -2.4
while tensor had R2 = 0.90. The difference was CALIBRATION TRANSFER: entry 108
fit a scaling factor on train prompts and applied it to test prompts. The DU
scaling didn't transfer well (L7: R2 = -15.2). But the DU CORRELATION was
still 0.97 — it ranked permutations correctly, just predicted the wrong
absolute magnitude.

Entry 109 uses in-sample scaling, showing that all predictors rank damage
equally well. The gate information only helps with CALIBRATION across different
input distributions, not with RANKING which permutation is more damaging.

**By Droid's decision rule:** "Neither beats DU or decorrelation nulls → close
this line." The DU predictor (R2 = 0.95) beats the decorrelation null (R2 ≈ 0)
handily. But the tensor predictor does NOT beat DU. Gate structure in the weights
is irrelevant for predicting permutation damage.

**Implication:** The sensitivity of gated MLPs to row permutation is primarily
about how the down projection (D) amplifies changes in the up weights (U).
The gate's role is only in input-specific modulation — selecting which neurons
are active for a given input — not in the weight-level coupling between gate
and up matrices.

**This closes the "tensor composition" line.** The D_k mixed moment program
started in entry 97, pivoted through the row-permutation challenge (103),
and now terminates: there is no weight-only gate-value invariant to discover
because the gate structure doesn't predict damage beyond what DU provides.

Script: experiments/weight_only_predictor.py
Data: experiments/results/row_permutation/weight_only_predictor.json

## Entry 103: Row-permutation challenge — D_k sufficiency killed (2026-09-18)

**Result: Droid's objection confirmed conclusively.**

SmolLM2-135M, 10 prompts, all 30 layers, random row permutation (seed 42).

Three conditions per layer:
1. **Up-only permutation:** permute up_proj rows, keep gate and down unchanged
2. **Function-preserving control:** same permutation on gate+up rows, compensating
   column permutation on down_proj
3. **D_k verification:** compare Gram operators before/after permutation

| Metric | Up-only perm | Control | Separation |
|--------|-------------|---------|------------|
| Relative MLP output error | 1.39 (139%) | 2.5e-07 | 5,547,709x |
| D1 change | < 5e-08 | N/A | Preserved |
| Gram operator max diff | < 2e-08 | N/A | Preserved |

**Per-layer output errors (up-only permutation):**
L0=1.04, L5=1.46, L10=1.38, L15=1.31, L20=1.56, L25=1.46, L29=1.33
Range: 1.04-1.64. All layers show massive disruption. No layer is robust.

**Interpretation:** Permuting which gate neuron gates which up neuron — while
preserving all weight statistics, all Gram operators, all D_k, all marginal
spectra — produces output that is 139% different on average. The computation
is completely destroyed while every proposed invariant remains perfect.

The function-preserving control (same permutation applied consistently to
gate, up, and inverse to down) produces output error at machine precision.
This confirms that the issue is specifically the row correspondence between
gate and up, not any other structural property.

**What this kills:** Any compositional invariant based solely on Gram
operators W^T W. This includes D_k mixed moments, spectral analysis of
frame operators, Frobenius alignment, and any statistic that treats the
weight matrix only through its correlation structure.

**What survives:** The descriptive finding that gate/up Gram operators are
aligned (positive D1) remains valid. But D1 measures correlation structure,
not computational structure. A useful invariant must be ROW-AWARE.

**What comes next:** Derive a row-aware composition measure that captures
the actual SwiGLU pairing: how row i of gate_proj relates to row i of
up_proj. The simplest candidate: row-wise cosine similarity or the
diagonal of gate @ up^T (which IS sensitive to row permutation).

Script: experiments/row_permutation_challenge.py
Data: experiments/results/row_permutation/challenge_result.json

## Entry 101: First-principles confirmation — Zipf + cross-entropy → power-law eigenspectra (2026-09-18)

**Core result**: A single-layer dot-product embedding trained with SGD on
Zipfian token distribution + cross-entropy loss produces power-law eigenspectra.
No multi-layer architecture needed. The power law is a CONSEQUENCE OF TRAINING
ON ZIPFIAN DATA, not an architectural feature.

**Control**: Uniform token distribution (Zipf s=0) → flat spectrum (exp=-0.25,
matching random init). The power law requires non-uniform frequency distribution.

Zipf exponent sweep (n=1024, d=64, 3000 steps):
| Zipf s | Eigenvalue exponent | Condition |
|--------|-------------------|-----------|
| 0.0    | -0.25 (flat)      | 3         |
| 0.5    | -0.80             | 97        |
| 1.0    | -1.75             | 213       |
| 1.5    | -1.79             | 290       |

Steepness increases with Zipf s but saturates around s>1.0.

n/d sweep (d=64, Zipf s=1.0):
| n/d  | Exponent | Condition |
|------|----------|-----------|
| 2    | -2.39    | 1783      |
| 8    | -1.99    | 378       |
| 32   | -1.50    | 107       |
| 128  | -0.97    | 27        |

**Higher overcompleteness → flatter spectrum.** This CONFIRMS the n/d hypothesis
from entry 95: the controlling variable is overcompleteness ratio, not d alone.
Phi-1.5 (n/d=25) has steeper spectrum than Qwen3 (n/d=148) because lower
overcompleteness allows more directional organization.

Toy model exponents (-0.97 to -2.39) are steeper than real model exponents
(-0.35 to -0.57). The gap likely comes from: real data has effective Zipf s<1
(subword tokenization flattens frequency distribution), multi-layer composition
redistributes spectral energy, longer training with better optimization.

**Theoretical framework now complete**:
1. Tokens follow Zipf → frequent tokens get more gradient updates
2. Gradient descent concentrates common tokens in shared directions
3. Rare tokens get unique directions with less refinement (tail)
4. Power-law eigenspectrum is the steady-state of this dynamics
5. Exponent controlled by (Zipf s, n/d ratio, training convergence)
6. The information-theoretic priority (entry 94) is the SEMANTIC
   manifestation: rare morphological fragments = high information =
   unique directions = top eigenspaces
7. Tail vulnerability (entry 92) is the COST: poorly organized tail
   directions are fragile under perturbation

This closes the "why power law?" question for embeddings. The remaining open
question: does this extend to MLP weight matrices, where the "tokens" are
features rather than vocabulary entries?

Files: scratchpad/fast_training_spectrum.py
Data: experiments/results/frame_analysis/training_dynamics_spectrum.json

## Entry 100: D-scaling — constant cross-component coupling across dimensions (2026-09-18)

Tested D1_norm across the SmolLM2 scale ladder (d=576, 960, 2048) to check
the theorem's prediction D1 ~ d^{2alpha-1}.

**RESULT: D1_norm is constant at ~0.21 across a 3.5x range of dimensions.**

| Model        | d    | D1 mean | D1 late | D1 max | tail alpha |
|-------------|------|---------|---------|--------|------------|
| SmolLM2-135M | 576  | +0.214  | +0.297  | +0.537 | 1.4-1.8    |
| SmolLM2-360M | 960  | +0.204  | +0.246  | +0.409 | 1.4-1.8    |
| SmolLM2-1.7B | 2048 | +0.215  | +0.255  | +0.400 | 1.1-1.4    |

Measured scaling exponent: +0.009 (essentially zero).
Predicted (fixed alpha=0.8): +0.600.
Predicted (fixed alpha=0.6): +0.200.

**This is NOT a failure of the theorem.** The theorem holds for fixed alpha.
What happens in practice: alpha DECREASES with d. The 1.7B has tail exponents
1.1-1.4 vs the 135M's 1.4-1.8. Flatter spectra at larger d. The decrease
in alpha exactly compensates the d^{2alpha-1} growth, yielding constant D1.

**FINDING 1: Constant cross-component coupling budget.**

Training maintains a constant level of gate-up eigenvector alignment
(D1_norm ~ 0.21) regardless of model width. Extra capacity goes to
independent features, not to extending the shared subspace.

**FINDING 2: Alpha decreases with d (spectral democratization).**

Larger models have flatter eigenspectra — less concentration in top
eigenvalues. This is consistent with entry 88's finding that embedding
condition number decreases with d (11,971 → 1,175 → 640).

The co-adaptation of alpha and d to maintain constant D1 is a training-
level constraint: the model optimizes the AMOUNT of cross-component
coupling, not just its existence.

**FINDING 3: Depth profiles flatten with scale.**

Early-late D1 gap: 135M = 0.18, 360M = 0.10, 1.7B = 0.11.
Larger models distribute coupling more evenly across depth. The late-layer
spike (entry 98) is largest for the smallest model.

**FINDING 4: Overlap increases with d despite constant D1.**

Top-4 eigenvector overlap: 135M = 0.35, 360M = 0.37, 1.7B = 0.42.
More eigenvectors align (overlap increases) but each contributes less
(because eigenvalues are flatter). The two effects cancel in D1.

**Synthesis of entries 97-100:**

The full picture:
1. Pairwise geometry is insufficient for power-law frames (entry 97, theorem)
2. Real models have shared eigenvector structure, D1 > 0 everywhere (entry 98)
3. The critical exponent alpha=0.5 separates two regimes (entry 97)
4. But training co-adapts alpha and d to maintain constant D1 ~ 0.21 (entry 100)
5. This constant represents an optimized cross-component coupling budget

The "native HD math" operating in trained networks is not just the geometric
phase transition (theorem) — it's the INTERACTION between geometry and
optimization. The model sits at a specific operating point on the
alpha-vs-d curve, maintaining constant coupling.

Data: experiments/results/frame_composition/d_scaling_test.json

## Entry 99: Directional structure analysis — hierarchical eigenspace clustering (2026-09-18)

The directional structure that creates power-law eigenspectra (entry 95) is
a HIERARCHICAL CLUSTERING of tokens in eigenspace.

Per-token effective eigenspace rank (corrected normalization):
- SmolLM2 (d=576): p10=12, p50=32, p90=101 — median uses 5.6% of dimensions
- Qwen3 (d=1024): p10=224, p50=326, p90=422 — median uses 31.8% of dimensions

SmolLM2 has 5.7x stronger directional concentration — each token effectively
lives in ~32 dimensions out of 576. Qwen3 tokens spread across ~326 of 1024.
The concentration ratio (median_erank/d) inversely tracks the power-law
steepness: low concentration ratio = steep power law.

Most concentrated tokens (erank ~3-4) sit almost entirely in eigenvector 0
with 77-83% alignment. These are rare vocabulary entries — consistent with
entry 94's information-theoretic priority. Most spread tokens (erank ~265 for
SmolLM2, ~536 for Qwen3) have negligible alignment with any single eigenvector.

The alignment-rank power law matches the eigenvalue power law:
- SmolLM2: alignment exponent -0.516 vs eigenvalue exponent -0.517
- Qwen3: alignment exponent -0.361 vs eigenvalue exponent -0.350
This is expected (eigenvalue = sum of squared alignments) but confirms
self-similar structure.

Norm-alignment correlation reveals different representational strategies:
- SmolLM2: rho(norm, top-1 alignment) = 0.67 (high norm = top eigenspace)
- Qwen3: rho = -0.08 (norm and eigenspace are decoupled)

Also in entry 95: Phi-1.5 calibration FALSIFIES monotone alpha(d). Alpha values
(SmolLM2 0.50, Qwen3 0.32, Phi-1.5 0.52) suggest n/d dependence rather than
d dependence, but 3 points with confounded training distributions is insufficient.

Files: scratchpad/directional_structure_analysis.py, scratchpad/fix_token_erank.py
Data: experiments/results/frame_analysis/directional_structure.json

## Entry 98: Real models have shared eigenvector structure (2026-09-18)

Computed normalized mixed moments D1_norm for MLP gate_proj and up_proj frame
operators across all layers of SmolLM2-135M (d=576) and Qwen3-0.6B (d=1024).
This bridges entry 97 (synthetic proof) to real models.

Normalization: S_norm = S * d/tr(S) so mean eigenvalue = 1, making D1
scale-invariant and directly comparable across layers and models.

**FINDING 1: Gate and up share eigenvector structure at ALL layers.**

D1_norm is positive everywhere (random baseline = 0.0 +/- 0.01):

| Model        | D1_norm range     | D1_norm mean |
|-------------|-------------------|-------------|
| SmolLM2-135M | [+0.07, +0.54]   | +0.22       |
| Qwen3-0.6B  | [+0.05, +1.32]   | +0.22       |

The directions gate considers "important" overlap with up's important
directions. SwiGLU operates on partially aligned information streams.

**FINDING 2: Sharing increases monotonically through depth.**

SmolLM2-135M D1_norm by depth zone:
  L0-9:   0.107 (early)
  L10-19: 0.226 (middle)
  L20-29: 0.317 (late, peak L28=0.54)

Qwen3-0.6B D1_norm by depth zone:
  L0-9:   0.112
  L10-19: 0.244
  L20-27: 0.350 (peak L27=1.32, 10x early-layer baseline)

Interpretation: early layers afford independent feature detection. Later
layers need tight gate-up coordination for correct output production.
The last-layer spike is the most coordinated computation in the network.

**FINDING 3: This is invisible to pairwise spectral analysis.**

Entry 88 reported condition numbers, effective rank, coherence per component.
None detect eigenvector alignment between components. Two frame operators
with identical spectra but different D1 (entry 97) look identical under
pairwise analysis. D1_norm is the missing cross-component statistic.

**FINDING 4: Effective sharing fraction is 2-15%.**

vs synthetic fully-shared (entry 97, r=d/4):
- SmolLM2 mean: 0.22/5.81 = 3.8%
- Qwen3 mean: 0.22/8.54 = 2.6%
- Qwen3 last layer: 1.32/8.54 = 15.5%

Not all eigenvectors align, but the aligned ones are high-eigenvalue
(high-importance), contributing disproportionately to D1.

**FINDING 5: Implications.**

The model doesn't keep gate and up as independent channels. Shared
eigenstructure means gating interacts with value computation through the
shared subspace. This is not visible from analyzing components separately.

Connects to formal superposition theory gap: tight-frame analysis assumes
independent feature directions. Real models violate this with structured
cross-component alignment that grows through depth.

**Combined significance of entries 97-98:**
- Entry 97 proved synthetically: pairwise geometry is insufficient for
  power-law frames. Mixed moments D1 capture what spectra miss.
  Critical exponent alpha=0.5 separates two regimes.
- Entry 98 proves empirically: real models ARE in the "insufficient"
  regime, with D1_norm=0.05-1.32 (vs 0 for random).
  The structure increases through depth.
Together: the first concrete demonstration that real neural networks
contain native high-dimensional structure inaccessible to standard tools.

Data: experiments/results/frame_composition/real_model_mixed_moments.json

## Entry 95: Rate-distortion prediction — directional structure, not norms, creates power laws (2026-09-18)

Three models of eigenspectrum origin tested:
1. **Greedy packing** (idealized perfect structure): predicts exponent ~-0.93
2. **Monte Carlo random directions + information-weighted norms**: exponent ~-0.1 (flat)
3. **Monte Carlo random directions + uniform norms** (MP baseline): exponent ~-0.1 (flat)

Observed: SmolLM2 -0.517, Qwen3 -0.350.

**Critical result**: Models 2 and 3 produce nearly identical flat spectra
(condition ~1.5, R2~0.78 for power law fit). Information-weighted norms with
random directions are indistinguishable from uniform norms with random
directions. Norm scaling alone does NOT explain the observed power laws.

**Conclusion**: The power-law eigenspectrum comes from DIRECTIONAL STRUCTURE
created by training. Tokens don't just get different norms — they get organized
into correlated directions. The top eigenspaces represent coherent directions
that many tokens align with. Training creates a low-dimensional manifold
structure where the alignment strength distribution follows a power law.

Directional organization parameter alpha = (obs - random) / (greedy - random):
- SmolLM2 (d=576):  alpha = 0.50 (50% of theoretical maximum structure)
- Qwen3 (d=1024):   alpha = 0.32 (32% of theoretical maximum structure)

alpha decreases with d — higher-dimensional models need less directional
organization because they have more capacity. Prediction: alpha(d) is monotone
decreasing; very high d embeddings approach random (flat spectrum).

Zipf exponent sensitivity (greedy model, SmolLM2 config):
- Zipf s=0.50: predicted exponent -0.46 (close to observed -0.52)
- Zipf s=1.00: predicted exponent -0.93 (too steep)
This suggests effective Zipf exponent of training data is ~0.5, not 1.0.

**Connection to entry 94**: The information-theoretic priority (rare morphological
fragments in top eigenspaces, common words in tail) is the SEMANTIC consequence
of directional structure. Training doesn't just weight tokens by information —
it ORGANIZES them into coherent directions, and the semantically informative
tokens get the most-aligned directions.

**Theoretical framework emerging**:
- Formal superposition theory assumes tight frames (alpha=1.0)
- Real models have alpha~0.3-0.5 (partial directional organization)
- The degree of organization is capacity-constrained (d-dependent)
- Power-law spectrum = optimal allocation under Zipfian distribution + finite d
- The tail vulnerability (entry 92) is the cost of partial organization:
  low-eigenvalue directions are poorly organized and thus fragile

**Phi-1.5 calibration FALSIFIES monotone alpha(d):**
- SmolLM2 (d=576, n/d=85):  alpha = 0.50
- Qwen3 (d=1024, n/d=148):  alpha = 0.32
- Phi-1.5 (d=2048, n/d=25): alpha = 0.52

Phi-1.5 has HIGHER alpha than both smaller models despite 3.5x higher d.
The monotone alpha(d) prediction is wrong. Alternative: alpha depends on
n/d (overcompleteness ratio). Ordering by n/d: Phi(25) > SmolLM2(85) >
Qwen3(148) matches alpha ordering: 0.52 > 0.50 > 0.32. More overcomplete
vocabularies have less directional organization — harder to organize more
tokens per dimension. But 3 points with confounded training distributions
(Phi was trained on curated "textbook" data) is insufficient to distinguish
n/d-dependence from training-distribution effects.

Phi-1.5 exponent (-0.565) is STEEPER than SmolLM2 (-0.517) despite 3.5x
higher d. The condition number is much lower (640 vs 11,971) because d is
higher, but spectrum shape (exponent) and dynamic range (condition) are
measuring different geometric properties.

Files: scratchpad/rate_distortion_prediction.py, scratchpad/phi_eigenspectrum.py
Data: experiments/results/frame_analysis/rate_distortion_prediction.json,
      experiments/results/frame_analysis/phi_eigenspectrum_alpha.json
Next: controlled tiny-transformer sweep (fixed n, varying d) to isolate
d-dependence from training/architecture confounds; test n/d hypothesis.

## Entry 97: Frame composition — pairwise geometry IS insufficient (2026-09-18)

First mathematical experiment in the HD math program. Tests whether marginal
spectral properties (eigenspectrum, coherence) suffice to predict how frame
operators compose, or whether mixed moments (encoding eigenvector alignment)
carry essential information.

**Setup:** 5 ensembles × {d=64,128,256,512,1024} × {γ=2,4} × 30 trials.
For each pair (S,T) of frame operators from the same ensemble:
- Marginal spectra: eigenvalues of S and T
- Mixed moments: D_k(S,T) = (1/d)tr[(S-τI)(T-τI)]^k, k=1,2,3
- Additive composition deviation: W2 distance between eig((S+T)/2) and
  average of eig(S), eig(T)

Key comparison: power-law frames (α=0.8, matching entry 89) with SHARED top-r
eigenvectors (r=d/4) vs RANDOM eigenvectors. Same marginal spectrum. Different
alignment.

**FINDING 1: Mixed moments perfectly separate shared from random frames.**

| d    | D1_shared | D1_random | ratio    | z-score |
|------|-----------|-----------|----------|---------|
| 64   | +1.733    | -0.002    | 694x     | 49.6    |
| 128  | +2.637    | +0.008    | 335x     | 62.1    |
| 256  | +3.937    | -0.003    | 1378x    | 201.4   |
| 512  | +5.815    | +0.000    | 31166x   | 460.4   |
| 1024 | +8.538    | +0.000    | 27094x   | 667.9   |

D1 alone suffices. No marginal statistic can detect this — marginal spectra
are identical by construction.

**FINDING 2: Separation GROWS with dimension as d^{2α-1}.**

D1_shared scales as d^{0.575} (measured) vs d^{0.6} (predicted from α=0.8).
The theoretical prediction: for power-law spectrum λ_k ~ k^{-α} and shared
rank fraction r/d:

  D1(shared) ~ (1-α)² · ζ(2α) · d^{2α-1}

Critical boundary at α=0.5: below this, D1→0 (pairwise geometry suffices at
large d). Above this, D1→∞ (pairwise geometry fails increasingly at large d).
Real models have α ∈ [0.6, 0.9] — firmly in the "fails" regime.

**FINDING 3: Higher mixed moments carry genuinely new information.**

| d    | D2_shared    | D2_random | D2/D1² ratio |
|------|-------------|-----------|--------------|
| 64   | 71.99       | 0.065     | 24.0         |
| 256  | 1212.07     | 0.066     | 78.2         |
| 1024 | 19863.41    | 0.079     | 272.3        |

D2 scales as d^{2.03}, faster than D1²~d^{1.15}. Each mixed moment order
captures distinct structural information. Predicted: D_k ~ d^{2kα-1}.

**FINDING 4: Power-law spectra break free-probability composition at high d.**

Additive composition deviation (W2 from averaged marginals):

| d    | Gaussian | PL random | PL shared | Sparse |
|------|----------|-----------|-----------|--------|
| 64   | 0.215    | 0.478     | 0.044     | 0.004  |
| 256  | 0.212    | 0.721     | 0.039     | 0.001  |
| 1024 | 0.212    | 1.047     | 0.035     | 0.000  |

Gaussian: stable ~0.21 (finite-size correction to free convolution).
Power-law random: GROWS from 0.48 to 1.05 — composition becomes LESS
predictable at higher d. Power-law breaks universality.
Power-law shared: DECREASES from 0.044 to 0.035 — shared structure makes
composition MORE predictable. The mixed moments carry the prediction.

**FINDING 5: The critical exponent α=0.5 separates two regimes.**

For power-law frame operators with spectrum λ_k ~ k^{-α}:
- α < 0.5: D_k → 0 as d → ∞. Pairwise geometry suffices asymptotically.
  Free probability applies. Composition is predictable from marginals.
- α > 0.5: D_k → ∞ as d → ∞. Pairwise geometry fails increasingly.
  Free probability breaks. Mixed moments are the native information.

Real models (α = 0.6-0.9) are in the "fails" regime. The gap GROWS with model
dimension. This means: as models get wider, the standard tools (spectrum,
coherence) become LESS adequate for predicting composition behavior. You NEED
the mixed moments.

**Theoretical derivation:**

For frame operator S = UΛU^T with power-law spectrum λ_i = C·i^{-α} (mean-
normalized), and two operators sharing top-r eigenvectors:

  D_k(S,T) = (1/d) tr[(S_c T_c)^k]

  where S_c = S - (tr S/d)I is the centered operator.

The shared subspace contributes the dominant term:
  D_k ≈ (1/d) · sum_{i=1}^r (λ_i - 1)^{2k}
      ≈ (1-α)^{2k} · d^{2kα-1} · sum_{i=1}^r i^{-2kα}

The sum converges for 2kα > 1, i.e., k > 1/(2α). For the first moment
(k=1), this requires α > 0.5.

All 6 pre-registered predictions confirmed. Data: experiments/results/
frame_composition/composition_test.json

**What this means for HD math:**

This is the first concrete result in the program. It establishes that:
1. Pairwise spectral statistics are provably insufficient for predicting
   composition of power-law frame operators
2. Mixed moments are the native high-d information for structured systems
3. There is a critical exponent (α=0.5) separating two qualitative regimes
4. Real models are in the regime where pairwise tools fail increasingly

**Next:** Compute D1 for REAL model frame operators. Do adjacent layers share
eigenvector structure (non-zero D1)? If yes: mixed moments predict real model
composition behavior that no pairwise statistic can capture.

## Entry 96: Droid strategic review — frame composition program (2026-09-18)

Droid reviewed the expanded HD-math direction ("what is the native math operating
in high dimensions?"). Key assessments:

**Best wedge for a 1-person lab:** Develop a predictive calculus for composition
and interference in structured high-dimensional systems, including finite-d
corrections and explicit universality failures. Narrow enough for one person,
broad enough to matter across pure math, kernels, random features, and neural
representations.

**Four most promising threads:**
1. Structured universality — which HD laws survive dependence? What invariant
   predicts when universality breaks?
2. Interference and composition of nearly-orthogonal systems — when can combined
   behavior be predicted from individual invariants? Free probability handles
   the free case; the frontier is non-free (structured, learned) systems.
3. Moderate-dimensional crossover laws — what happens at d=100-5000 vs asymptotic
   theory? Phase transition widths, convergence rates, noncommuting limits.
4. Observable-relative geometry — effective geometry depends on allowed
   transformations and observations, not just Euclidean distance.

**Small-lab advantage:** Exact counterexamples, moderate-d phase diagrams,
finite-size theorems, universality stress tests, proof + CPU simulation,
negative classification results. Large labs optimize systems; we isolate
mechanisms across thousands of controlled ensembles.

**Concrete first proof targets:**
1. Derive Gaussian baseline and finite-d concentration for frame operators.
2. Solve one structured deformation (e.g., shared low-rank component).
3. Construct two ensembles with identical (d, coherence, spectrum) but different
   composition behavior. This is the killer result — proves pairwise geometry
   is insufficient.
4. Identify the lowest-order mixed invariant separating them.

**Synthesis with entries 88-89:** The fork found real models are 100x-10,000x
away from tight-frame optimality (entry 88) and follow power-law spectra with
architecture-specific exponents (entry 89). This is exactly the "structured
universality failure" Droid identifies — the math operating in trained networks
deviates massively from what random matrix theory or ETF theory predicts. The
right starting point is: can we characterize composition behavior for power-law
frame operators? Do the empirical exponents (-0.7 to -1.8) predict composition
differently than Marchenko-Pastur or tight-frame baselines?

**Existing perturbation data (Droid assessment):** NOT evidence for HD math
(no dimension variation). BUT valuable methodological evidence: "a high-d
substrate can produce a very low-d, probe-imposed observable, and scalar
behavioral geometry cannot identify the substrate's native mathematics." The
86 experiments are closed for HD inference. Future experiments should follow
mathematical predictions derived on dimension-indexed ensembles.

**Breakthrough path:** A universality theorem covering structured ensembles;
a finite-d law predicting interference/phase-transition width from measurable
invariants; a composition calculus where marginal spectra + mixed cumulants
predict combined behavior; an impossibility theorem proving existing invariants
are insufficient + identifying the replacement.

Review: scratchpad/droid_hd_direction.md

## Entry 94: Eigenspace semantics — information-theoretic priority (2026-09-18)

Analyzed which tokens occupy which eigenspaces in SmolLM2-135M embedding.
The result reveals an information-theoretic priority hierarchy in the frame
structure.

**Top eigenspace (eigenvalue 0, 50.6% of trace):**
Dominated by WORD FRAGMENTS of rare formal/academic words:
  strugg-, dissemin-, enthus-, affili-, comprom-, authent-, interpre-,
  necessit-, miscon-, psychiat-, dismant-, interspers-, acknow-, elig-
These tokens have:
  - High norms: 6.3-7.5 (mean ~3.0)
  - 70-78% of energy in eigenspace 0 alone
  - 89-93% of energy in top 10 eigenspaces
  - Mean eigenindex: 15-22 (very top of spectrum)

**Lowest-norm tokens (embedding tail):**
PROPER NAMES: Michael, John, David, Robert, Paul, Sam, James, George, Alex
  - Low norms: 1.75-1.90
  - Only 17-45% of energy in top 10
  - Mean eigenindex: 150-200 (mid-to-tail)

**Category summary (embedding eigenbasis):**
  Category          ||W||   Top10%  MeanIdx
  punctuation       3.50    49%     137
  short_letters     2.97    47%     130
  special/ws        3.15    46%     158
  other tokens      2.93    31%     161
  common_words      2.65    30%     190
  digits            2.66    30%     240

**The priority hierarchy is INFORMATION-THEORETIC, not frequency-based.**
- TOP: Tokens with high intrinsic information (rare morphological fragments
  that narrow continuation space dramatically: "strugg-" -> struggle/struggling/...)
- MID: Structurally important tokens (punctuation, short letters)
- TAIL: Context-dependent tokens (common words like "the" carry no information
  alone; proper names are syntactically interchangeable)

**This explains tail vulnerability (entries 92-93):**
Our code perturbations (comments, parens, identity operations) are STRUCTURAL
modifications — they change context, not rare-word content. Structural tokens
live in mid-spectrum eigenspaces. So perturbation vectors naturally project
into mid-to-tail eigenspaces where the model is fragile.

The top eigenspace — allocated to morphologically distinctive fragments —
is robust but IRRELEVANT to our perturbation types. The model's "defense
budget" is allocated by information content, not by token frequency.

**Connection to information theory:**
- Information per token ~ -log P(token | preceding_tokens)
- Word fragments like "authent-" have high self-information and low ambiguity
- Common words like "the" have low self-information, high contextual dependence
- The embedding allocates frame energy in proportion to intrinsic information
- This is a form of optimal coding in high-d: high-information tokens get the
  "best" (top eigenspace) directions; low-information tokens share the tail

Data: experiments/results/frame_analysis/eigenspace_semantics.json

## Entry 93: H1b cross-model validation — early-layer universal, late-layer divergent (2026-09-18)

Replicated the eigenprojection test (entry 92) on Qwen3-0.6B (d=1024, 28 layers)
to check if tail vulnerability is universal or model-specific.

**Cross-model comparison of correlations (output_L1 vs top10 energy):**

| Layer position | SmolLM2-135M | Qwen3-0.6B |
|----------------|-------------|------------|
| First (L0) | rho=-0.43 | rho=-0.83 (p=0.04) |
| Early-mid | rho=-0.31 (L7) | rho=-0.09 (L7) |
| Mid | rho=-0.66 (L15) | rho=+0.60 (L14) |
| Late-mid | rho=+0.89 (L22) | rho=+0.14 (L21) |
| Last | rho=-1.00 (L29) | rho=+0.49 (L27) |

**What transfers across architectures (universal):**
1. Early-layer negative correlation: BOTH models show rho < 0 at L0. High-
   sensitivity perturbations avoid top eigenspaces at the input end. Qwen3's
   effect is statistically significant (rho=-0.83, p=0.042).
2. Low top-10 energy: 1-5% at all layers in both models. Perturbation vectors
   are broadly distributed, never concentrated in a few eigenspaces.
3. Median eigenindex past midpoint: perturbations skew toward the spectral
   tail in both models. Qwen3: 490-854/1024 (48-83%). SmolLM2: 253-433/576 (44-75%).

**What diverges (architecture-specific):**
1. Mid-to-late layer correlation signs differ: SmolLM2 negative at L15/L29,
   Qwen3 positive at L14/L27. Internal compensation strategies differ.
2. SmolLM2's perfect -1.0 at the last layer is NOT reproduced in Qwen3.

**Revised understanding:**
The EARLY-LAYER tail vulnerability is the robust finding: at the input end,
the model's gate_proj frame creates a spectral gradient where perturbations
that most disrupt output avoid the top eigenspaces. This may reflect how the
gate decides which input features to amplify — perturbations in low-eigenvalue
directions bypass the gate's main "attention" and sneak through to later layers.

The late-layer divergence suggests different architectures develop different
strategies for handling perturbation energy that has propagated to later layers.
This is consistent with H3 (layer composition matters).

Data: experiments/results/frame_analysis/eigenprojection_h1_qwen3.json

## Entry 92: H1 test — perturbation sensitivity vs eigenspace (2026-09-18)

Tested whether perturbation sensitivity correlates with eigenspace position in
SmolLM2-135M. Ran 6 perturbation types through the model, extracted hidden
states at 5 layers, projected perturbation vectors (delta = h_pert - h_base)
into the gate_proj frame operator's eigenbasis.

**H1 as stated: REJECTED.** High-sensitivity perturbations do NOT project more
onto top eigenspaces. In fact, the opposite pattern dominates.

**What we found instead — vulnerability inversion:**

Correlations between output sensitivity and top-10 eigenspace energy:
  Layer 0:  rho = -0.43 (high sensitivity -> LESS top eigenspace)
  Layer 7:  rho = -0.31
  Layer 15: rho = -0.66
  Layer 22: rho = +0.89 (p=0.019 — FLIPS positive at this late layer)
  Layer 29: rho = -1.00 (p<0.001 — perfect negative at output)

The sign flip at layer 22 and the perfect negative at layer 29 are notable.

**Revised hypothesis (H1b): Tail vulnerability.**
Perturbation vectors that project onto LOW-eigenvalue subspaces cause MORE
behavioral disruption. Interpretation: top eigenspaces are well-represented
(high energy, many features) and therefore robust — the model can absorb
perturbations there. Tail eigenspaces are poorly represented (low energy,
few features) and vulnerable — perturbations there are not averaged out.

This reframes superposition's fragility story: the model doesn't need tight
frames everywhere. It needs the TOP of the spectrum to be robust (which it
achieves by allocating capacity there). The tail is sacrificed — low-eigenvalue
directions are "where the model saves capacity" and therefore where it's
fragile. The power-law spectrum IS the vulnerability gradient.

**Quantitative details:**
- Perturbation energy is broadly distributed: top-10 holds only 1-5%
- Median eigenindex is 270-430 out of 576 (past the middle, in the tail)
- At layer 29: median eigenindex jumps to 372-433 (more tail-concentrated)
- The model's final layer pushes perturbation differences deeper into tail

**Caveat:** Only 6 perturbation types — limited statistical power. The
qualitative pattern is clear but needs validation with more perturbation
types and a second model.

Data: experiments/results/frame_analysis/eigenprojection_h1_test.json

## Entry 91: Theory-practice gap synthesis — why do models work? (2026-09-18)

Theoretical synthesis of entries 87-90 plus deep reads of arXiv:2608.27540
(formal superposition theory) and arXiv:2602.02224 (spectral superposition).

**The gap, precisely stated:**

Formal theory (arXiv:2608.27540) proves: ReLU(W^T W x + b) recovers sparse
binary features from an overcomplete dictionary W when:
  (a) W is nearly tight (condition ~ 1)
  (b) W has low coherence (near Welch bound)
  (c) Expected sparsity is up to O(d / log n)

Spectral theory (arXiv:2602.02224) shows: at capacity saturation, features
organize into tight frames, collapse onto single eigenspaces, and become
classifiable via association schemes.

Our measurements show: real models have:
  (a) Condition numbers 640-12,000 (vs 1 for tight)
  (b) Coherence 0.75-1.00 (vs Welch bound 0.02-0.04)
  (c) Power-law eigenspectra (vs uniform for tight frames)
  (d) Architecture-specific layer profiles (spike, plateau, bottleneck)

The models work anyway. The theory-practice gap is 2-4 orders of magnitude
on every metric. This is not noise — it is a qualitative difference in the
mathematical structure.

**Three hypotheses for why models work despite non-tight geometry:**

H1: **Soft priority ordering.** The power-law spectrum creates a natural
priority hierarchy: features aligned with top eigenspaces are reliably encoded
(strong projection, low interference), while features in the tail are
approximate (weak projection, high interference). Models tolerate this
because not all features need the same fidelity. If true: perturbation
sensitivity should correlate with eigenspace projection strength.

H2: **Structured sparsity.** The formal theory analyzes worst-case support
(any k features could be active). Real feature activations have correlations
— certain features co-occur, others exclude. If co-active features are in
different eigenspaces, interference stays low despite high global coherence.
If true: feature co-activation patterns should align with spectral structure.

H3: **Layer composition.** Single-layer analysis misses how the SEQUENCE of
frames through the network works. Even if each layer's frame is non-tight,
the composition across 24-30 layers might create an effective pipeline where
early layers encode broadly and later layers specialize. The layer-dependent
spectral profiles (steepening power laws through depth) support this.
If true: the product of frame operators across layers should be "tighter"
than any single layer.

**Testable next steps (ranked by leverage):**

1. **Eigenprojection of perturbation types** (tests H1): compute which
   eigenspaces our known perturbation types project into, and check if
   perturbation sensitivity hierarchy correlates with eigenspace position.
   Cost: reuse existing perturbation data + frame eigendecompositions. CPU.

2. **Cross-layer frame product** (tests H3): compute the composition of
   frame operators across layers and check if it's "tighter" (lower condition)
   than individual layers. Cost: matrix multiplications, CPU.

3. **Feature co-activation analysis** (tests H2): extract activations on a
   set of inputs, compute pairwise co-activation, and check spectral alignment.
   Cost: forward passes on small model, CPU.

**Distance from central artifact:** This analysis is at distance 0 — it IS
the mathematics of high-dimensional spaces. We're measuring how HD geometry
manifests in trained representations and discovering that formal theory doesn't
explain real model behavior.

## Entry 90: Cross-model frame comparison — d-dependent condition number (2026-09-18)

Added Phi-1.5 (d=2048, 24 layers, UNTIED embedding/LM-head) to the frame analysis.
Cross-model comparison across 3 architecture families.

**Core comparison (embedding frame operator):**

| Model | d | vocab/d | Condition | Erank% | Coherence | Welch |
|-------|---|---------|-----------|--------|-----------|-------|
| SmolLM2-135M | 576 | 85x | 11,971 | 87.6% | 0.973 | 0.041 |
| Qwen3-0.6B | 1024 | 148x | 1,175 | 96.7% | 1.000 | 0.031 |
| Phi-1.5 | 2048 | 25x | 640 | 95.1% | 0.750 | 0.022 |

**Finding: Embedding condition DECREASES with d.**
11,971 → 1,175 → 640 as d goes 576 → 1024 → 2048.
Higher-dimensional embeddings are more uniform (closer to tight frame).
This is confounded by overcomplete ratio (vocab/d), but the trend is clear.

**Finding: Coherence tracks overcomplete ratio, not d.**
Qwen3 (vocab/d = 148) has coherence 1.0; Phi (vocab/d = 25) has 0.75.
Higher packing pressure → higher coherence. The Welch bound sets the floor;
the ratio to Welch is similar across models (24x, 32x, 35x).

**Finding: MLP frame properties are architecture-specific.**
- SmolLM2 MLP middle layers: condition ~569, erank 91.3%, coherence 0.72
- Qwen3 MLP: condition ~5000 (10x higher), with dramatic spike at layers 15-20
- Phi MLP layer 0: condition 3.8M (pathological), rest 1400-2200
- The MLP structure does NOT obviously track d_model.

**Finding: Phi-1.5 has UNTIED embedding and LM head.**
Embedding condition: 640 (good). LM head condition: 12,322 (poor).
The output projection is much less uniform than the input embedding.
This separation of input/output geometry is unique to Phi among our 3 models.

**What this means for HD math:**
The d-dependent improvement in embedding condition is a genuine HD phenomenon:
as d increases, the embedding has more room to spread out, and it does.
But the overcomplete ratio (how many things you're packing in) matters too.
The interaction between d and vocab_size determines frame quality.

MLP frame structure is architecture-specific, not dimension-dependent.
This suggests MLP representations are shaped more by training and
architecture design than by the abstract geometry of the ambient space.

Data: experiments/results/frame_analysis/frame_phi_1.5.json

## Entry 89: Eigenspectrum shape — power law, not tight frame (2026-09-18)

Full eigenspectrum analysis of frame operators in SmolLM2-135M and Qwen3-0.6B.
Tests whether spectra follow Marchenko-Pastur (random matrix prediction), power
law, or tight frame (uniform) distribution.

**Finding 1: Spectra are approximately power law.**
- MLP gate_proj: λ_k ~ k^(-0.7 to -0.9), R²=0.70-0.78 (full spectrum)
- MLP up_proj: λ_k ~ k^(-0.6 to -0.8), R²=0.59-0.73 (full spectrum)
- Tails are steeper: exponents -1.3 to -1.8 for bottom 80%
- NOT Marchenko-Pastur (no bulk+edge structure). NOT tight frame (not uniform).
- This is a trained-specific regularity that neither random matrix theory nor
  toy-model superposition theory predicts.

**Finding 2: SmolLM2 embedding is effectively ~4-dimensional.**
- Participation ratio: 3.9/576 (0.7%!)
- Top eigenvalue holds 50.6% of total trace alone
- Top 10 hold 55.6%
- Shannon effective rank: 41.6/576 (7.2%)
- The 28x gap between λ_1 and λ_2 is extreme
- Almost all 49K token embeddings point in roughly the same direction

**Finding 3: Qwen3 embedding is dramatically less concentrated.**
- Participation ratio: 73.5/1024 (7.2%)
- Top eigenvalue holds only 11.3% of trace
- Shannon effective rank: 626.8/1024 (61.2%)
- 18x more effective dimensions than SmolLM2 (73.5 vs 3.9)

**Finding 4: Power law exponents steepen through depth.**
Gate_proj tail exponent by layer (Qwen3):
  L0: -1.41, L7: -1.66, L14: -1.79, L21: -1.77, L27: -1.46
Pattern: steepens from early to middle layers, flattens again at the end.
SmolLM2 shows the same trend: L0: -1.44, L7: -1.61, L15: -1.72, L22: -1.80, L29: -1.71

**Finding 5: Gate is consistently steeper than Up (both models).**
- Gate participation ratio: 43-56% vs Up: 58-74%
- Gate Shannon erank: 59-75% vs Up: 75-84%
- Gate has fewer effective dimensions — sharper feature selection
- This asymmetry is universal across layers and architectures

**What this means for HD math:**
The eigenspectrum of trained networks follows a power law with architecture-
specific exponents. This is NOT what either random matrix theory (Marchenko-
Pastur) or toy superposition theory (tight frames) predicts. The power law
is an emergent regularity of training in high dimensions.

Questions this opens:
1. Is the power law exponent universal? (Test more architectures)
2. Does the exponent relate to model capability?
3. What loss landscape geometry produces power-law spectra?
4. Does the exponent change during training, and if so, how?
5. Can we derive the power law from first principles?

The SmolLM2 embedding result (effectively 4-d in 576-d space) is particularly
striking — it implies massive redundancy in the embedding, with most information
carried by a few directions. Connection to concentration of measure: in high-d,
random vectors concentrate on a thin shell, but trained embeddings concentrate
on a low-d subspace. Training creates structure that's OPPOSITE to what random
high-d geometry predicts.

Data: experiments/results/frame_analysis/eigenspectrum_analysis.json

## Entry 88: Frame-theoretic analysis of real models (2026-09-18)

Tested whether trained transformer weight matrices approximate the tight frame /
equiangular tight frame (ETF) structure assumed by formal superposition theory
(arXiv:2608.27540). Analyzed SmolLM2-135M (d=576, 30 layers) and Qwen3-0.6B
(d=1024, 28 layers). Measured: frame operator eigenspectrum, condition number
(B/A; =1 for tight frame), effective rank, coherence vs Welch bound.

**Finding 1: Real models are massively far from tight frames.**
- SmolLM2 embedding condition: 11,970 (tight = 1.0)
- Qwen3 embedding condition: 1,175
- MLP condition numbers: 180-25,888 across layers
- The Aug 2026 formal theory assumes "nearly tight, low-coherence dictionaries."
  Real models satisfy neither assumption. Yet they work.

**Finding 2: Coherence is 15-30x the Welch bound.**
- SmolLM2 embedding: max cos = 0.97, Welch bound = 0.04 (24x excess)
- Qwen3 embedding: max cos = 1.00 (!), Welch bound = 0.03 (33x excess)
- MLP layers: coherence 0.47-0.95, all far above Welch (~0.03)
- Some Qwen3 token pairs share effectively identical embedding directions.

**Finding 3: Effective rank uses ~88-96% of available dimensions.**
- SmolLM2 embedding: 504.7/576 (87.6%)
- Qwen3 embedding: 990.7/1024 (96.7%)
- MLP gate_proj: ~89-93% across both models
- MLP up_proj: ~91-96% across both models
- NOT all dimensions are equally used, but most are non-negligible.

**Finding 4: Gate vs Up asymmetry is universal.**
- Gate_proj has higher condition numbers than up_proj in BOTH models
- Gate has lower effective rank (more concentrated)
- Gate has higher coherence in late layers
- Interpretation: gate selects features (needs sharp directions), up computes
  values (needs broader coverage). Different mathematical regimes in the SAME
  layer.

**Finding 5: Qwen3 has a dramatic middle-layer condition spike.**
- Gate condition: 3771 (L0) → 18,937 (L20) → 1587 (L21)
- Up condition: 591 (L0) → 25,888 (L20) → 1902 (L21)
- 10x spike and sharp drop at layer 20-21 boundary
- SmolLM2 has NO comparable spike (range 180-1440 throughout)
- This is an architecture-specific "compression bottleneck" pattern.

**What this means for HD math:**
The gap between formal ETF-based superposition theory and real model geometry
is enormous. Models operate 100x-10,000x away from tight-frame optimality.
The theory-practice gap has rich structure:
- Layer-dependent (condition changes through depth)
- Component-dependent (gate vs up in same layer)
- Architecture-specific (Qwen3 bottleneck vs SmolLM2 stability)

This gap is itself a research target. The question is not "are models tight
frames?" (answer: no, not remotely) but "what alternative mathematical
structure enables superposition in non-tight, high-coherence regimes?"

Data: experiments/results/frame_analysis/frame_smollm2_135m.json,
experiments/results/frame_analysis/frame_qwen3_0.6b.json

## Entry 87: HD math landscape survey (2026-09-18)

Web research survey across 10 areas of high-dimensional mathematics to identify
where a 1-person CPU lab has the highest leverage for field-changing work.

**Areas surveyed:** concentration of measure, almost-orthogonal packing &
superposition, random matrix theory, random projections (JL), curse/blessing of
dimensionality, manifold hypothesis, information geometry, topological data
analysis, high-dimensional statistics, computation in superposition.

**Key recent papers found:**
- "Towards a mathematical theory of superposition" (arXiv:2608.27540, Aug 2026):
  frame theory + compressed sensing framework for superposition. Sharp recovery
  thresholds for equiangular tight frames. Sparsity up to O(d/log n).
- "Spectral Superposition" (arXiv:2602.02224, Feb 2026): frame operator WW^T
  gives spectral measure of feature geometry. Capacity saturation → tight frames.
  "One-over-width" law from Welch bound confirmed empirically at NeurIPS 2025.
- Active debate: "Compressed Computation ≠ Computation in Superposition" (Jun 2026)
  vs counter-paper arguing it IS (Jul 2026). No resolution.
- Complexity: O(n²/log n) features for n neurons (nearly tight bounds).
- JL: optimal non-asymptotic bounds for sparse projections close exponential gap
  between theory and practice (Jun 2026, Springer ML).

**Tier 1 — highest leverage for us:**

1. **Computation in superposition** — youngest field, active disputes, our
   perturbation data directly relevant, CPU-testable. No settled theory for what
   computations can be performed on superposed representations.

2. **Superposition geometry (frame theory)** — mathematical foundation for #1.
   The Aug 2026 paper gives a formal framework we can test: do our small models'
   weight matrices approach equiangular tight frame structure? Where they deviate,
   what does it predict about behavior?

3. **Curse/blessing transition** — directly answers "how does math change at
   high d?" Controlled-d tiny transformers (d=16-512) on CPU can map transition
   boundaries. No one doing this with our precision.

**Recommended first moves:**
1. Characterize frame-theoretic structure in our small models (extract W, compute
   WW^T eigenvalues, coherence, proximity to tight frame)
2. Controlled-d sweep with tiny transformers to map transition boundaries
3. Model C-group interference using frame theory (spectral localization?)
4. Deep read of arXiv:2608.27540 to connect our data to the formal theory

Full survey: scratchpad/hd_math_landscape.md

## Direction expansion: mathematics of high-dimensional spaces (2026-09-18)

The research program expands from LLM-specific latent space analysis to the
mathematics of high-dimensional spaces broadly. How does math change when d >> 1?
Scope: pure HD math, all ML (not just neural nets), deep learning, HD statistics.
Dimension range is open (d=100 to arbitrarily large). LLM representations are one
motivating example among many.

Prior entry (below) narrowed this to d=100-4096 and LLM-focused HD math. That
framing was too narrow. The math itself is the subject.

## HISTORICAL pivot note: native high-dimensional mathematics (2026-09-18)

Direction change. The project reorients from behavioral measurement of specific
neural network phenomena (transfer rho, omega modulus) to the native mathematics
of high-dimensional spaces (d=100–4096).

**Why:** 20 days of transfer-rho work produced one defensible finding (granularity-
dependent transfer: coarse properties transfer across models, fine properties
don't) but nothing field-changing. The omega modulus program produced extensive
measurements but unresolved mechanism. Both are data, not breakthroughs.

**What we explored before deciding:**
1. High-D math tools — develop algebra for almost-orthogonal packings, interference,
   superposition. Real gaps exist (confirmed by literature: Luisto 2025 survey,
   Spectral Superposition 2602.02224, adversarial interference 2510.11709).
2. Lipschitz-concentration-transfer bridge — connect Lévy's lemma to inter-model
   transfer. Fable review (fable_hd_math_review.md) demolished this as already
   known under different terminology (Platonic Representation Hypothesis, Huh
   et al. ICML 2024; "Convergence Without Understanding" 2605.23315).
3. Structured divergence vs PRH — use our data to challenge PRH. Rejected as
   commentary, not capability.

**What we're doing:** Exploring, discovering, characterizing mathematical structure
native to high-dimensional spaces. Not proving one theorem. Not building one tool.
The subject of study IS the mathematics. Prior behavioral data becomes evidence
within this broader frame.

**Key gaps identified from literature review:**
- No algebra for almost-orthogonal packings (Luisto: "no inherent algebraic organization")
- No interference composition rules (2510.11709: geometry but no algebra)
- Computation in superposition bounded but not characterized (2408.05451: O(n²/log n))
- Spectral superposition works in toy models only (2602.02224)
- The d=512-4096 regime is a theoretical blind spot between JL and exact packing

**Prior results reframed as high-D data:**
- Within-op >> cross-op (2×2 diagnostic) = different operations project onto different
  nearly orthogonal directions in vulnerability space
- Granularity-dependent transfer = possible concentration-of-measure signature
- Adjacency effect (own-line behaves like inert) = interference geometry of attention
- Multiplicative sensitivity = first-order Taylor in small-angle regime

Next: begin mathematical exploration. Fable reviews every 5 hours, Droid for
back-and-forth strategic thinking.

## Jacobian derivation — KILLED (Fable R4, 2026-09-18)

Theory-first attempt: derive the observed perturbation sensitivity hierarchy from
the transformer attention Jacobian. Decomposes ∂h_t/∂x_p into value channel (α·V)
and routing channel (∂α/∂x). Claims causal masking → after-expression perts have
zero value channel at expression positions → explains A/B >> D hierarchy.

**Fable R4 verdict: KILL.** Review at scratchpad/fable_jacobian_r4.md.

Key findings (5 CRITICAL, 5 MAJOR, 2 MINOR):
1. Value/routing dichotomy is backwards: at expression positions (t < p), BOTH
   channels are zero (trivially, by causal masking definition). At readout T,
   BOTH are nonzero. The stated dichotomy doesn't exist.
2. Perturbations not in Jacobian domain: A/B/C/I change sequence length → δ = x' - x
   undefined. Prefix mechanism wrong under RoPE (used by all 7 models).
3. **d_add_zero falsification (new empirical finding):** d_add_zero ("x=1+1+0")
   appends tokens touching zero original positions → causal masking predicts A-like
   behavior. FAILS in 6/7 models on existing balanced kill data:

   | Model | rho(add_zero,A) | AA baseline | Gap |
   |---|---|---|---|
   | Qwen3-1.7B | +0.062 | +0.726 | -0.664 |
   | SmolLM2-1.7B | +0.487 | +0.935 | -0.448 |
   | SmolLM2-Instruct | +0.534 | +0.875 | -0.341 |
   | Qwen3-0.6B | +0.506 | +0.639 | -0.133 |
   | SmolLM2-360M | +0.615 | +0.746 | -0.131 |
   | StableLM | +0.855 | +0.930 | -0.075 |
   | Phi-1.5 | +0.867 | +0.848 | +0.019 (uninformative) |

   Position of the edit does NOT determine behavior; whether appended tokens
   enter the computation the readout must perform DOES. d_add_zero extends the
   expression (model must evaluate "+0") while A-perts are decorative.

4. Internal contradiction: multiplicativity needs J isotropic; hierarchy needs J
   anisotropic. Can't have both on the same page.
5. Not novel: value/routing = QK/OV circuit split (Kobayashi 2020, Elhage 2021).
   d ≈ ||Jδ|| = adversarial examples (Goodfellow 2015). Nothing uses d (not HD math).
6. token_direction_test.py has 8 flaws: measures embedding not Jδ, mean-pools away
   position, wrong perturbation set, wrong cell set, correlation never computed.

**R3 kill criteria satisfied:** 3/4 met — prediction fails (d_add_zero), theory IS
the simple baseline (position count), and that baseline is itself refuted.

**Salvage finding (Tier 3):** d_add_zero is a clean dissociator between positional
and computational accounts. Position dominates among tokens the readout treats as
inert, but NOT among tokens inside the computation. This sharpens the hierarchy
from "position" to "computational relevance to the readout."

**HD Math direction status:** This is the 3rd theory-first cycle killed since R3
(concentration/routing, weight spectral bridge, Jacobian). R3 criterion #1 (3
cycles, no surviving testable prediction) is triggered. Next attempt must state
a numeric/ordinal prediction absent from existing data BEFORE examining data,
or the direction closes.

## Fable strategic assessment: d-blindness diagnosis (2026-09-18)

Review: scratchpad/fable_strategy_direction.md.

Core argument: every instrument the project built is d-blind — no surviving
finding changes value as a function of the dimension of the space. The three
HD-math theory kills and the tautological prereg are one instrument failure
seen from four sides, not four failures. All bridges from d-dependent math
to d-blind data have nowhere to land.

**Four structural negatives (N1-N4):**
N1. Scalar behavioral distance is a d-blind instrument (task rank, not dim rank).
N2. Bridge theories to this data produce only prior art (because data can't discriminate).
N3. Cross-model fine structure never transfers; universal = coarse = PRH territory.
N4. No design gate ever asked "does this observable depend on d?"

**Decisive datum:** behavioral rank (PC1 share) is non-monotone in d across
the scale ladder and moves 35 points at FIXED d=1024 (Qwen3-0.6B base 88%
vs instruct 53%). Rank is set by training history, not ambient geometry.

**Recommendation:** close both programs, record structured negative, reopen
only under stated d-varying prediction. One admissible reopening test:
controlled-d tiny transformers (2-layer, d swept 16-512, 3 seeds, ~1 CPU-day)
with pre-registered P-blind vs P-geometry predictions. Prior ~80% on P-blind.

**Assessment of this review:** thorough and well-argued for the HD-math
direction, which is already KILLED. But the perturbation sensitivity hierarchy
(position >> meaning) is a separate research question that does not claim
d-dependence. Testing whether it generalizes across domains (arithmetic →
strings/boolean/type) is about transformer computation, not about d. The
review's N1 applies to HD math; domain universality is a different axis.

---

## Computational-relevance test — REJECTED (Fable design review, 2026-09-18)

Pre-registered a "syntactic context determines behavioral sensitivity" test with
6 new perturbation types: same-token-different-context pairs (expression extension
vs comment). Fable design review at scratchpad/fable_prereg_review.md.

**VERDICT: REJECT.** Do not run. Do not re-open the behavioral program under a
new name.

Key findings:
1. **Not pre-registered.** The predictions are already confirmed by existing R29/R30
   inline-inert data (i_comment at A-level 0.777 vs expression edits at 0.39) and
   R31 `//`-comment disambiguation (10x effect). Six comment perturbations already
   exist in the kill set. The 0.85 threshold is fitted to the d_add_zero table.
2. **Confirmatory outcome is tautological.** "Comment tokens cluster with inert
   appends" = "tokens the model can ignore matter less" with "ignore" spelled `#`.
   The readout IS a comment the model evaluates — the task requires knowing what
   comments are.
3. **Disconfirmatory outcome uninterpretable.** Three competing explanations (rare
   code form, answer-annotation slot, commented-out code) that the design cannot
   separate.
4. **Missed confounds:** d_mult_one changes AST attachment on +/- cells (30/69);
   two-space-before-# is a known Qwen3 anomaly; wrong control (A_mean vs i_comment).
5. **"D-like" premise wrong.** 5/7 models: d_add_zero correlates MORE with A than D.
   Qwen3-1.7B: 0.06/0.007 (correlates with nothing). "Below inert ceiling" ≠ "D-like."
6. **Breakthrough ceiling = zero.** Positive = model knows Python. Negative = curiosity
   with 3 explanations. Does not touch HD math (acknowledged in pre-registration).

**Side finding:** R29/R30 scripts (r29_inline_inert.py, r30_inexpr_inert.py) were
never committed to git — exact i_comment string is unrecoverable. Reproducibility
defect to repair.

**Implication:** This was not counted as the R3 required "novel HD prediction"
because it's behavioral sensitivity analysis, not HD math, and its predictions
were already in the data. R3 criterion #1 remains triggered. The behavioral
sensitivity program is closed.

---

## HD Math exploration day 1 (2026-09-18)

### Geometry probes: SmolLM2-1.7B (d=2048)

**Probe 1** (100 arithmetic expressions with shared suffix "print(x)\n# Output: "):
Showed dramatic "spread-then-concentrate" pattern — cos drops from 1.0 (L0) to 0.57
(L20) then back to 0.87 (L24). Effective dim peaks at 25 (L23), norms peak at 1245.

**Probe 2** (150 diverse: 50 arith + 50 NL + 50 code, bare text, last-token extraction):
NO hunchback. All classes show monotonic concentration. Combined: cos rises from 0.11
(L0) to 0.79 (L23), eff_dim stays 1-3 throughout. Cross-class: mid-layers CONVERGE
all domains (cos ≈ 0.55), final layer RE-SEPARATES NL from symbolic domains
(arith-code cos=0.70, arith-NL cos=0.37, NL-code cos=0.35).

**Probe 3** (4 controlled-similarity sets: identical prefix, paraphrases, same-domain,
cross-domain): NO hunchback for ANY set. All monotonically concentrate. Initial cos
ranges from 0.03 (cross-domain) to 0.11 (identical prefix), all converge to 0.76-0.87.

**Critical finding**: The hunchback in Probe 1 was an ARTIFACT of the shared suffix
making all last tokens identical (cos=1.0 at L0). The "spread" was context propagation
from earlier (varied) tokens to the (identical) last token via attention. When last
tokens vary (Probes 2-3), no spread phase occurs.

**Fable review** demolished: eff_dim=14 expected (ID hunchback literature), anisotropy
confound, input-output cosine ≠ feature superposition (category error), interference
tensor has extensive prior art (5+ papers 2024-2026).

**What survives the review:**
1. Cross-class mid-layer convergence zone → final-layer domain separation is clean data
2. The context-propagation mechanism (why identical-last-token produces the hunchback)
   is a concrete finding about how attention propagates information through depth
3. Per-class final-layer cosine reflects next-token distribution similarity
   (arith-code close because both predict symbolic tokens)

### Mathematical investigations 1-5 (scratchpad)
- Inv 1: Interference under linear maps (correct math, obvious result)
- Inv 2: Corrected geometric growth, interference tensor concept (prior art exists)
- Inv 3: Cosine-dimensionality dissociation (artifact of Probe 1's identical-token bug)
- Inv 4: From phenomenology to prediction (framework for deriving L* from architecture)
- Inv 5: Differentiation threshold hypothesis (partially undermined by artifact finding)

### Fable strategic review R2 (2026-09-18)

Recommended B+C hybrid: MP-deviation spectrum of behavioral Gram matrices,
testing whether spikes above Tracy-Widom predict transfer hierarchy. Droid
repeatedly stalled (3 attempts, API rate limit), so single-reviewer convergence.

### MP-Deviation Analysis (2026-09-18, zero-cost reanalysis)

Tested Fable's recommendation on existing 7-model balanced kill data (69×24).

**Part 1 (spectrum characterization):**
- All 7 models have 1-3 eigenvalues above Tracy-Widom threshold
- Dominant spike loads on structural/semantic perturbations (c+d groups)
- Second spike (when present) loads on surface perturbations (a+b groups)

**Part 2 (held-out prediction test):**
- H1 (MP deviations exist): CONFIRMED
- H3 (spike overlap predicts held-out transfer): REJECTED (rho near 0, all splits)
- Cross-model spike count ~ gap: ARTIFACT (p=0.034 on interleaved split, p=0.562 on full data)

**Comparison to standard methods:**
- MP spike count ≈ parallel analysis component count ≈ Kaiser criterion
- RMT framework adds NO predictive power over standard PCA
- The null model (MP) is interesting but the deviations are just PCA components

**Kill criterion triggered:** #2 — deviations exist but don't predict. Rotating.

**Key learning:** At p=24 (behavioral readout dimensionality), the data isn't 
high-dimensional enough for RMT to add value over PCA. The genuinely high-d
math (BBP transition, Tracy-Widom edge statistics) becomes interesting only when
BOTH n and d are large. Testing on hidden-state Gram matrices (d=2048) would be
the real test, but that's already published territory (arXiv:2410.17770 for
weights, arXiv:2608.07921 for attention).

### Transitivity / Interference Composition (2026-09-18, zero-cost)

Tested Option C from Fable strategy: does the product rule r(i,k) ~ r(i,j)*r(j,k)
hold for perturbation correlations? If not, the deviation is a composition law.

**Raw transitivity test (Qwen3-1.7B-Base first, then all 7 models):**
- Spearman(product prediction, actual) = 0.73 (moderate)
- Mean residual = +0.14 (actual EXCEEDS product prediction — superadditive)
- Consistent across all 7 models: raw residual +0.13 to +0.20
- Within-group residuals (0.18) > cross-group (0.11)

**Factor decomposition (the kill):**
- A 3-factor model (PCA) captures 79-97% of variance across models
- After removing 3 factors, transitivity residual FLIPS NEGATIVE (-0.02 to -0.12)
- The superadditivity was ENTIRELY EXPLAINED by shared factors
- Perturbation-specific effects are actually SUBADDITIVE after factoring

**What the 3 factors are:**
- F1 (60-93% var): general cell vulnerability, loads architecture-specifically
  (SmolLM2: surface a+b dominant; Qwen3: c_newline_before + d_temp_var dominant)
- F2-F3 (2-16% each): model-specific secondary structure
- Cross-model subspace alignment: 0.37-0.82 (Qwen3-StableLM cluster at 0.77-0.82;
  SmolLM2 cluster at 0.63-0.74; cross-cluster 0.37-0.58)
- Factors are architecture-family-specific, NOT universal

**Kill criterion triggered:** Composition law is trivially explained by ~3 common
vulnerability factors. No genuine superadditivity remains after factoring.

**CORRECTION — Behavioral rank is model-intrinsic, not dataset artifact:**
Bootstrap test: random perturbation subsets (p=6..24) give rank-growth matching
a synthetic rank-k process (not random). The rank IS intrinsic to each model.

BUT rank varies dramatically across models:
| Model                  | k(80%) | k(90%) | F1 var  |
|------------------------|--------|--------|---------|
| SmolLM2-360M           | 1      | 1      | 92.6%   |
| SmolLM2-1.7B           | 1      | 2      | 86.7%   |
| Phi-1.5                | 1      | 2      | 88.6%   |
| StableLM-2-1.6B        | 2      | 2      | 75.8%   |
| Qwen3-1.7B-Base        | 3      | 4      | 65.1%   |
| Qwen3-0.6B-Base        | 3      | 5      | 60.9%   |
| SmolLM2-1.7B-Instruct  | 4      | 6      | 59.9%   |

Key: rank does NOT correlate with d or H (Phi-1.5 and Qwen3 have same d/H,
different rank). It depends on training data + instruction tuning.
SmolLM2-1.7B base→instruct: rank jumps 1→4. Instruction tuning quadruples
behavioral sensitivity diversity.

**Theoretical framing (concentration of measure):**
In d>>1, random perturbation directions are nearly equivalent (concentration).
Only perturbations aligned with the top singular directions of the behavioral
Jacobian produce distinguishable effects. Behavioral rank = number of resolvable
Jacobian singular values above the noise floor. Models with more capacity or
more diverse training resolve more directions from superposition.

### Weight-Space Spectral Bridge (2026-09-18, CPU only, KILLED)

Tested Fable R3 Path B: can weight matrix spectral properties predict
behavioral participation ratio?

**Method:** Loaded all 7 models' weights (CPU, float16), computed SVD of every
attention + MLP weight matrix (168-224 per model), extracted spectral summaries
(PR, heavy-tail alpha, stable rank, spectral norm).

**Result: NO bridge.**
- Spearman(weight PR, behavioral PR) = 0.393, p = 0.383
- Spearman(weight alpha, behavioral PR) = 0.071, p = 0.879
- No weight spectral quantity predicts behavioral PR (all p > 0.25)

**Decisive evidence:** SmolLM2-1.7B base vs instruct have IDENTICAL weight
spectra (PR: 1482.1 vs 1482.9, alpha: 6.02 vs 6.03) but 100% different
behavioral PR (1.32 vs 2.62). Weight spectra encode model CAPACITY, not
behavioral ROUTING.

**Delta weight analysis (instruct - base):** The instruction tuning delta is
high-rank (~38% of full rank for attention, ~66% for MLP) but small magnitude
(~6% relative change). Instruction tuning broadly reorganizes weight directions
without changing spectral properties. Only 2/168 matrices have low-rank deltas.

**Fable R3 kill criteria status:** 3+ focused cycles with no testable prediction.
Factor analysis, bootstrap/rank-scaling, weight-spectral bridge — all produced
measurements but no theory. Direction meets the kill threshold.

### Assessment: HD Math Direction Day 1 Complete

**What was found (6 analyses, all zero-cost except weight loading):**
1. Behavioral data is rank-k (model-intrinsic, bootstrap-confirmed)
2. k varies 1-4+ across models, perfectly predicted by PR (rho=0.954, p=0.001)
3. Instruction tuning doubles PR (1.32 → 2.62 for SmolLM2-1.7B)
4. Weight spectra don't predict behavioral PR (no bridge exists)
5. Instruct delta is high-rank, small magnitude (reorganization, not capacity)

**What was NOT found:**
- Any genuinely high-d-specific mathematics
- Any theoretical prediction derived before testing
- Any result that goes beyond standard PCA / SVD
- Anything field-changing

**Fable R3 recommendation (accepted):** Go theory-first. The project has data
and no theory. Every attempt was "apply tool to data, hope something emerges."
Reverse the flow: derive a mathematical prediction from first principles, then test.

### Theory-first: Concentration-of-measure bound on behavioral rank

**Setup.** A transformer maps input x ∈ R^d through L layers. The behavioral
effect of a perturbation δ at the output is Δf ≈ J·δ where J is the end-to-end
Jacobian. For the 11-bin or full-logit readout, J: R^d → R^V.

**Theorem sketch (from architecture).**
1. The Jacobian J decomposes as a product of per-layer Jacobians.
2. Each attention layer contributes rank ≤ H·d_k = d per layer (H heads, d_k = d/H per head).
3. With residual stream: J_l = I + J_attn_l + J_mlp_l·(I + J_attn_l), so
   the update at each layer adds ≤ H effective directions.
4. Over L layers: rank(J) ≤ L·H (loose upper bound, typically hundreds).
5. By concentration of measure on S^{d-1}: for random δ, ||J·δ||² concentrates
   around (||δ||²/d)·||J||_F². Most directions produce the SAME behavioral
   effect. Only the top singular directions of J create distinguishable effects.

**Prediction.** The behavioral rank k = #{σ_i(J) > σ_bulk · √(d/n)}, where
the BBP threshold determines resolvability. This gives k << d (concentration
bound) but the specific value of k depends on how many attention heads are
"input-discriminating" — producing different routing patterns for different inputs.

**Why weight spectra don't predict k:** The routing diversity is input-dependent
(depends on attention softmax, which is a function of both weights AND inputs).
Two models with identical weight spectral properties can have different routing
diversity (base vs instruct).

**Why instruction tuning increases k:** Instruct training forces the model to
attend to diverse input features (intent, format, constraints). This creates
more input-discriminating heads → higher effective Jacobian rank → higher k.

**Testable consequence (requires model runs):** The number of attention heads
whose output changes significantly between original and perturbed input should
correlate with behavioral k. This count should be:
- ~1 for SmolLM2-360M (rank-1 model: all perturbations route through same heads)
- ~4 for SmolLM2-1.7B-Instruct (rank-4 model: different perturbation types
  activate different heads)

**Test result: PREDICTION KILLED.**
SmolLM2-1.7B base vs instruct have identical attention routing:
- Heads changed per perturbation: 387-749 (base) vs 398-752 (instruct)
- Jaccard overlap between perturbation types: 0.71-0.90 (base) vs 0.71-0.89 (instruct)
- 50-97% of heads change for every perturbation (not sparse)
- Routing change is POSITIONAL (token insertion shifts attention positions),
  not content-based re-routing

The behavioral rank difference (1.32 → 2.62) does NOT come from attention
routing diversity. The mechanism must be in value/MLP computation — distributed
through specific weight values, not capturable by spectral or routing metrics.

**Direction assessment (end of day 1):**
8 analyses completed. 3 positive findings (rank is model-intrinsic, PR predicts k
perfectly, instruct tuning increases diversity). 5 structured negatives
(MP deviation, transitivity, weight bridge, theoretical prediction, routing).
No genuinely high-d mathematics discovered. Fable R3 kill criteria met.

---

## Kill experiment program — session 2026-09-18/19 (continued)

**Completed this session (2026-09-19 continuation):**
1. R19 adversarial review READ: "2/5 transfer" claim DEMOLISHED. StableLM outlier-driven,
   Qwen3 method-dependent, 2/5 split was covariate composition artifact. Full 12-method
   instability table. Defensible statement: fragility dominates, 0/5 transfer after PC1+PC2.
2. Scale ladder COMPLETE: SmolLM2-135M (gap=0.121, p=0.004), SmolLM2-360M (gap=0.318,
   Perm Z=6.0), Qwen3-0.6B (gap=0.489, Perm Z=8.8, 100% accuracy).
3. **NON-MONOTONIC SCALING:** SmolLM2 gap peaks at 360M (0.318) > 1.7B (0.281). Qwen3-0.6B
   gap (0.489) > 1.7B (0.452). Accuracy is monotonic. Gradient exists at 135M/52.5% accuracy.
4. Two-way interaction test (Tukey + variance decomposition): 16-44% additive-frame interaction,
   Tukey p < 1e-14 all models. BUT: fork agent found MULTIPLICATIVE model R²=1.000 for all 5.
   The 16-44% was cross-term artifact. d[i,j] = fragility[i] × magnitude[j]. No real interaction.
5. **Expanded-D late-twin (3 models, 6 D-D pairs):** MEANING DEFINITIVELY DEAD.
   11-bin: 0/3 positive meaning. Qwen3 REVERSES (D-D-DE=-0.111, CI excl zero negative).
   FL: Qwen3 meaning VANISHES (was CI [+0.082,+0.382], now [-0.050,+0.148]).
   FL: SmolLM2 only clear (1/3). Operator-family decomposition: within same family,
   meaning on 11-bin = +0.002/+0.008/-0.090 (ALL ZERO across 3 architectures).
6. All canonical records updated (STATE.md, EXPERIMENTS.md, ledger.jsonl).

**Previous session (2026-09-18):**
1. Phi Q1 40-cell DEFINITIVE: gap=0.217 [0.123, 0.355], p=5.26e-11. 5/5 universality.
2. R17 review incorporated: copy-vs-compute RETRACTED, per-pert profiles downgraded.
3. Tokenization-covariate: R^2≈0 (NOT tokenizer artifact).
4. Operator-class test: H1 failed all 5 (operator = common fragility).

**Central result (post-R20 + direction analysis, 2026-09-19):**
Sensitivity is predominantly multiplicative: d[i,j] ~ fragility[i] * magnitude[j].
Log-space additive R^2=0.82-0.88 (12-25% structured residual). **R^2=1.000 RETRACTED**
(raw-vs-log domain mixing). NOT type-blind: 7/8 models show cell x group interaction
(nested F + group-label perm). Gradient REAL (7+ models, 135M-1.7B, Perm Z>2, audit-clean).

**R21 ADVERSARIAL REVIEW (2026-09-19) — CORRECTIONS APPLIED:**
1. Direction meta-r (0.51-0.83) is TAUTOLOGICAL — algebraically expected from the
   log-additive decomposition, not a mechanism. Downgraded to consistency check.
2. Cross-model fragility "completely model-specific" → underpowered (n=40, power ~50%).
3. "Double non-transfer" → all cross-model comparisons NS at n=15 pairs.
4. SmolLM2-360M residual > raw → suppressor artifact at PC1=90.5%.
5. Non-monotonic residual scaling → underpowered at n=3, CIs overlap.
SURVIVING: 7/8 position-driven interaction, D-spread (3/4 perm-valid), locus confound.

**DIRECTION DIVERGENCE (descriptive, NOT mechanism, 2026-09-19):**
1. D perturbations push logits in genuinely DIFFERENT DIRECTIONS, not just different amounts
   (3/4 models significant, Z=2.9-5.8, permutation-validated against constraint null).
2. Direction-transfer meta-r (0.51-0.83) is algebraically expected (R21 tautology),
   not an independent predictive finding. The underlying D-spread IS real.
3. Cross-model direction prediction: all NS at n=15 (underpowered, not interpretable).
All within-D (same code position), residual-based, and permutation-validated.

**COPY CONTROL — GRADIENT IS SURFACE, NOT COMPUTATION (2026-09-19):**
Qwen3-1.7B on literal assignments (x=5): gap=0.403 [0.288, 0.516], Perm Z=7.9.
Expression gap=0.452. H_surface PASSES (diff=0.049, within 0.15 threshold).
H_compute FAILS (ratio=0.89, not < 0.50). The gradient does NOT require
computation — removing all arithmetic preserves 89% of the effect. This is
pure SURFACE/POSITION sensitivity. Running on remaining 4 models.

**LOCUS CONFOUND DISCOVERY (2026-09-19):**
Cross-group interaction (A/B/C/D) is primarily POSITIONAL (5/8 negative type effect).
Matched-pair test: same operation at different position = same as random cross (diff < 0.06).
BUT within-D direction structure IS genuine (position constant within D).

**Non-monotonic scaling:** Peak fragility at medium scale. Scale-dependent D-coherence:
SmolLM2 135M D=0.71 -> 360M D=0.18 -> 1.7B D=0.44. int_cast is the primary diverger.

**R21 ADVERSARIAL REVIEW — MAJOR CORRECTIONS (2026-09-19):**
1. ATTENUATION RATIO 0.79-1.01: single cell factor reproduces A-D gap. Inert appends
   predict rewrite orderings as well as other rewrites do (8/8). "Not predictable" → "less reliable".
2. INTERACTION EXISTS AMONG INERTS ALONE: AB vs C (no D) significant in 7/8. Comment-vs-
   statement within A+B significant in 7/8. Declared ABCD not the natural axes.
3. D-COHERENCE OVERCLAIMED: "anti-correlated" → uncorrelated (CIs include 0); "arithmetic
   identities" → cluster includes parens; "cond_true = control flow" → SmolLM2-1.7B only,
   driven by *1/÷1 identity cells. parens/int_cast coupling may be space-token deletion.
4. MULTIPLICATIVE = first-order Taylor (small-angle regime). Content is in the 12-25% residual.
5. Phi residual IS organized (comment-vs-statement p=0.002), just not along ABCD.
6. No rho on n=40 without CI; no cluster label without partition test (process rule).

**After R21 — what survives cleanly:**
- Gradient is REAL (7+ models, perm-validated, copy-control confirmed)
- Gradient is about surface context features, not computation or semantic type
- Cell fragility is completely model-specific (cross-model rho ~ 0)
- Residual direction structure is real (3/4) but small (12-25% of total)
- Direction divergence predicts pairwise transfer failure (5/6)
- Cross-model prediction fails (direction patterns model-specific)

**This session — new analyses (2026-09-19 continuation 2):**
1. **D-GROUP CROSS-MODEL STRUCTURE (8 models):** Within-D rho 0.169-0.712. Cross-model D-pair
   consistency: 0/28 significant. No D-perturbation consistently most anomalous. D-coherence does
   NOT predict residual gap (rho=+0.05, p=0.91). Raw gap does NOT predict residual (rho=-0.11).
2. **Tokenization verification:** Confirmed parens/int_cast both delete space token before operand.
   Qwen3-0.6B has parens↔int_cast rho=0.18 (outlier vs 0.53-0.80 for all others) despite same
   tokenizer. Instruct tuning may explain the decorrelation.
3. **Cell rank distribution:** Mean rank change = 13.3 = random expectation (13.3 for n=40).
   Max rank change = 39/40 (`1**5`). Even most stable cells change by 14+ positions.
4. **Fable R21 adversarial review** (launched, 3-round): 2/6 tautological, 2/6 underpowered,
   1/6 inflated, 1/6 real (locus confound). All corrections applied to STATE.md.
5. **Qwen3-0.6B-Base kill experiment RUNNING** (fp32 CPU, ~10 min) to resolve instruct confound.
6. **Locus-crossed design specified** (scratchpad): 6 types × 3 positions × 40 cells.

**Continuation 3 — zero-cost analyses and copy control (2026-09-19):**

7. **D-PAIR COUPLING CHECK (fixed, 8 models):** parens-int_cast top-3 in 5/8 confirmed
   (rank 1-7/15). BUT coupling is NOT space-token-specific: in 3/8 models, int_cast-add_zero
   is stronger than parens-int_cast (despite add_zero preserving the space token). The coupling
   tracks syntactic proximity (lightweight wrapping), not tokenization artifacts.

8. **CROSS-MODEL CELL FRAGILITY ANALYSIS (8 models, zero-cost):**
   - Cross-model rho = -0.022 [range: -0.494 to +0.348]. CONFIRMED model-specific.
   - ANTI-CORRELATED PREDICTORS: double-operator expressions (// % **) are maximally fragile
     in SmolLM2-1.7B (rho=0.71) but maximally robust in Qwen3-0.6B (rho=-0.56) and
     CodeGen (rho=-0.49). Same feature, opposite effect.
   - Answer digit: Qwen3-1.7B rho=-0.48 (low digits fragile), Phi rho=+0.48 (high digits fragile).
   - Within-digit ICC: 0.06 (SmolLM2-1.7B, expr-driven) to 0.44 (Qwen3-1.7B, digit-driven).

9. **TOKENIZER SEGMENTATION CHECK (zero-cost):** SmolLM2 and Qwen3 have IDENTICAL
   tokenization of all 40 expressions (7 tokens each, same token splits). Yet SmolLM2-1.7B
   has rho=+0.71 with double-operators while Qwen3-0.6B has rho=-0.56. StableLM, CodeGen,
   Phi produce CONSTANT token counts (all expressions same length). **Fragility is a property
   of learned weights, not the tokenizer.** Same tokens in → opposite fragility out.

10. **COPY CONTROL — SmolLM2-1.7B:** Gap=0.386 [0.261, 0.527], Perm Z=7.5. Accuracy 100%.
    H_surface PASS (diff=0.105). H_compute FAIL (ratio=1.37 — LARGER than expression gap).
    Removing arithmetic makes SmolLM2-1.7B MORE sensitive to headers. 2/2 models confirm
    gradient is surface sensitivity. StableLM running.

11. **POSITION vs CONTENT (zero-cost, 7 models):** Perturbations at the SAME POSITION
    but with different tokens (mean rho=0.705) are MORE correlated than perturbations with
    IDENTICAL tokens at different positions (mean rho=0.530). Position wins 6/7, content 0/7.
    Gap = +0.175. Behavioral sensitivity is organized by WHERE modifications are, not WHAT they
    contain. After-pairs (0.803) > before-pairs (0.548) — position effect strongest near assignment.

12. **SAME-LINE LOCUS (zero-cost, 7 models):** Same-line inerts (a_whitespace, b_semicolon)
    are +0.070 more D-correlated than different-line inerts. 7/7 positive, 2/7 significant.
    Small but consistent. The dominant axis is before/after, not same-line/different-line.

13. **COPY CONTROL — StableLM-2-1.6B:** Gap=0.193 [0.094, 0.315], Perm Z=3.9. Accuracy 100%.
    H_surface PASS (diff=0.106). H_compute FAIL (ratio=0.65). Third admissible model.
    Copy/expr ratio=65% — lowest, but still well above H_compute 50% threshold.
    3/3 admissible models from 3 families confirm gradient persists on literals.

14. **INSTRUCTION TUNING AMPLIFIES THE GRADIENT (Qwen3-0.6B base vs instruct):**
    IT selectively reshapes correlations: A<->C UP (0.410->0.503, learns inerts are "same kind"),
    A<->D DOWN (0.586->0.301, becomes more type-discriminative), D<->D DOWN (0.394->0.169).
    Gap AMPLIFIED 1.5x (0.324->0.489). IT makes models MORE sensitive to perturbation type.
    CAUTION: single model-pair, needs replication.

15. **POSITION SHIFT MECHANISM TEST — NULL (zero-cost, 7 models):** Absolute token-count
    shift does NOT predict C-group pair correlation. Grand r=0.010, p=0.918 (105 pooled pairs).
    5/7 models negative. SmolLM2-1.7B highest C pair (rho=0.875) has LARGEST shift diff (4
    tokens) — opposite of prediction. The position effect is QUALITATIVE (before vs after in
    causal attention window), not QUANTITATIVE (how far the assignment shifts). Mechanism:
    causal attention asymmetry — before-perturbations are attended to by the assignment,
    after-perturbations are invisible to it.

16. **SPACE-TOKEN CONFOUND — DISMISSED (zero-cost, 7 models):** R21 flagged parens/int_cast
    share space-token deletion, top-3 D pair in 5/8 models. Test: Jaccard similarity of
    token-level changes vs behavioral D-pair rho. Grand r=0.094, p=0.340 (n=105). NULL.
    Critical control: add_zero/zero_plus have IDENTICAL token changes (Jaccard=1.0) but
    behavioral rho ranges 0.023-0.777 across models. If tokenization drove D-coherence, this
    pair should always top-correlate. It doesn't. One exception: Qwen3-1.7B r=0.523 (p=0.045).
    Space-token confound is model-specific at most, not a general threat.

**Continuation 4 — R22 corrections and magnitude asymmetry (2026-09-19):**

17. **FABLE R22 ADVERSARIAL REVIEW (3 round):** Claim 1 (fragility-is-learned) CONFIRMED
    WITH QUALIFIER — narrow to "token segmentation doesn't predict." Claim 2 (copy=surface)
    CONFIRMED WITH DESIGN CONCERN — ratio > 1 needs addressing. Claim 3 (syntactic wrapping)
    NEEDS REVISION — "wrapping" is unoperationalized. 4 corrections applied below.

18. **R22 CORRECTION 1 — TOKEN IDs VERIFIED:** SmolLM2 (49K vocab) and Qwen3 (152K vocab)
    are DIFFERENT tokenizers with only 26% vocab overlap and different IDs. But all 40
    expressions produce identical decoded segmentation. Claim sharpened: "same token boundaries,
    different token IDs, opposite fragility." (Post-hoc flag: double-operator feature was
    data-mined, not pre-specified.)

19. **R22 CORRECTION 2 — RATIO BOOTSTRAP CIs:** All 3 models: ratio CI includes 1.0.
    Qwen3: 0.89 [0.64, 1.14]. SmolLM2: 1.37 [0.93, 1.87]. StableLM: 0.65 [0.31, 1.05].
    Gaps INDISTINGUISHABLE at 95%. SmolLM2 ratio>1 is NOT significant — "computation-as-
    shielding" interpretation unsupported. Binary H_surface/H_compute thresholds DROPPED
    in favor of CI overlap.

20. **R22 CORRECTION 3 — D COUPLING:** "syntactic wrapping" replaced with precise statement:
    space-token deletion ruled out (3/8 violate, 6/8 wrong-direction boundary). Replacement
    mechanism untested — could be edit distance, identity-operation semantics, or modification
    position. DO NOT name a replacement without evidence.

21. **PERTURBATION MAGNITUDE ASYMMETRY (zero-cost, 8 models):** Cross-model rho for
    perturbation magnitudes = +0.397 [range: -0.17 to +0.77]. SUBSTANTIALLY SHARED.
    Compare cell fragility: -0.022. The multiplicative model d = f[i] * m[j] decomposes
    into one partially universal factor (which change types matter — shared across models)
    and one completely idiosyncratic factor (which cells are vulnerable). Most consistent
    pert: c_desc_comment (rank 23-24 in all 8). Least consistent: d_temp_var (rank 1-21).
    Group ordering: C (before-position) always weakest. Qwen3 has D > A (rewrites more
    disruptive than inerts); all others have A > D.

22. **FEATURE REGRESSION (zero-cost, 8 models):** Linear regression R2=0.41-0.84 within
    each model. Zero features have consistent sign across all 8 models. Not one. The
    fragility landscape is well-structured WITHIN each model but completely unpredictable
    ACROSS models. Same-family consistency also absent: SmolLM2-135M vs 360M rho=-0.12,
    Qwen3-0.6B vs 1.7B rho=+0.03.

23. **COPY CONTROL — Phi-1.5 (SERIES COMPLETE):** Gap=0.197 [0.115, 0.281], Z=3.83.
    Accuracy 100% but P(digit)=0.001 → inadmissible by confidence criterion. H_surface PASS
    (diff=0.020). H_compute FAIL (ratio=0.91). 5/5 models tested, ALL H_surface PASS.
    3 admissible from 3 families. Copy control series COMPLETE.

24. **DIRECT TRANSFER PREDICTION TEST (zero-cost, 8 models, 28 pairs):**
    - Cell vulnerability transfer: rho = -0.022 +/- 0.227. ZERO.
    - Perturbation profile transfer: rho = +0.326 +/- 0.171. PARTIAL.
    - Within-family best: Qwen3 0.6B->1.7B profile rho=0.52, SmolLM2 within-family 0.47-0.49.
    - Cross-family worst: Qwen3-1.7B->CodeGen rho=-0.05 (ANTI-correlated).
    - CodeGen is a profile outlier: high transfer with SmolLM2 (0.39-0.51) but near-zero
      with Qwen3 (-0.05 to +0.005). Architecture-specific profile clusters.
    ALIGNMENT IMPLICATION: you can partially predict WHICH TYPES of changes matter
    (~33% shared perturbation hierarchy), but CANNOT predict WHICH INPUTS are affected
    (~0% cell fragility transfer). This is why behavioral transfer fails.

25. **CodeGen-350M COPY CONTROL (supplementary):** Gap=0.241 [0.163, 0.337], Z=4.5.
    Accuracy 70%, P(digit)=0.000 → NOT admissible. Gradient exists but model struggles
    with literal digit prediction. H_surface PASS (diff=0.077). 5/5 copy control done.

26. **SCALING ANALYSIS: WITHIN-FAMILY vs CROSS-FAMILY TRANSFER (zero-cost, 8 models):**
    - Within-family magnitude transfer: rho = 0.678 ± 0.091 (4 pairs: 3 SmolLM2, 1 Qwen3)
    - Cross-family magnitude transfer: rho = 0.350 ± 0.260 (24 pairs)
    - Within-family fragility transfer: rho = 0.028 ± 0.095 (near ZERO, same as cross-family)
    - Cross-family fragility transfer: rho = -0.030 ± 0.241 (ZERO)
    - No scaling relationship: param distance does NOT predict transfer rho (r=-0.165, p=0.40)
    - A-group and D-group profile transfer nearly identical (0.042 vs 0.035) — no group effect
    - Per-model factor asymmetry (mag_xfer - frag_xfer) is UNIVERSAL: 0.13-0.59 across all 8
    - Qwen3-1.7B is magnitude transfer outlier (lowest at 0.174, all others > 0.29)
    IMPLICATION: Architecture determines perturbation hierarchy (within-family 0.68 >> cross 0.35)
    but does NOT determine input vulnerability even within the same architecture. The fragility
    fingerprint is learned independently at each model scale, not inherited from architecture.

27. **SmolLM2-1.7B-Instruct KILL EXPERIMENT (running):** Replicating Qwen3-0.6B instruction
    tuning effect. SmolLM2-1.7B base gap = 0.281. If instruct shows gap >> 0.281, instruction
    tuning amplification generalizes beyond Qwen3.

28. **QWEN3 IS A MAGNITUDE-TRANSFER OUTLIER (zero-cost, existing data):**
    - Qwen3-0.6B mean cross-family magnitude transfer = 0.289
    - Qwen3-1.7B mean cross-family magnitude transfer = 0.084 (LOWEST of all 8)
    - Within-Qwen3 family: rho = 0.714 (highly consistent internally)
    - Qwen3's perturbation ranking is qualitatively DIFFERENT:
      - d_temp_var: ranked #1-2 in Qwen3, #16 median in others
      - b_dead_var: ranked #18 in Qwen3-1.7B, #2 median in others
      - c_pass_before: ranked #3 in Qwen3-1.7B, #19 median in others
    - Both Qwen3 models are instruction-tuned. All other models are base.
    - PREDICTION (now tested): If SmolLM2-Instruct magnitude ranking shifts toward Qwen3's
      pattern, then instruction tuning reorganizes perturbation sensitivity.
    - **RESULT (item 29):** Qwen3-0.6B-Base has rho=0.905 with Qwen3-1.7B-Instruct.
      ARCHITECTURE drives magnitude ranking, NOT instruction tuning.

29. **QWEN3 BASE vs INSTRUCT MAGNITUDE TEST (zero-cost, decisive):**
    - Qwen3-0.6B-Base vs Qwen3-1.7B-Instruct: rho = 0.905 (near-perfect agreement)
    - Qwen3-0.6B-Base vs Qwen3-0.6B-Instruct: rho = 0.737 (strong)
    - Qwen3-0.6B-Base vs non-Qwen3 bases: mean rho = 0.186 (low, same as Instruct's 0.285)
    - CONCLUSION: Qwen3's perturbation magnitude hierarchy is ARCHITECTURE-DETERMINED.
      Instruction tuning has two separable effects:
      (a) Magnitude ranking: UNCHANGED by instruction tuning (architecture sets it)
      (b) A-D gap: AMPLIFIED 5.6x by instruction tuning (separate mechanism)
    - Top rank discrepancies between Base and Instruct: c_pass_before (±13 ranks),
      c_dead_var_before (±13), d_add_zero (±11). These are the perturbations most
      affected by instruction tuning, but they're small relative to the architecture
      signature.

30. **INSTRUCTION TUNING EFFECT ON MULTIPLICATIVE FACTORS (zero-cost, Qwen3-0.6B):**
    Group-specific amplification from Base→Instruct:
    - A (inert): 0.98x (UNCHANGED)
    - B (inert append): 0.94x (~unchanged)
    - C (before-position): 1.42x (INCREASED — more sensitive)
    - D (rewrites): 0.69x (REDUCED — less sensitive)
    - Overall: 0.77x (distances shrink)
    Fragility (cell) correlation Base→Inst: rho=0.333 (p=0.036). Weak but significant.
    Magnitude (pert) correlation Base→Inst: rho=0.737. Strong (architecture).
    KEY INSIGHT: Instruction tuning reduces D absolute sensitivity by 31% while
    INCREASING the profile-based gap. The Spearman metric captures CONSISTENCY of
    differentiation across cells, not absolute magnitude.

31. **GAP MECHANISM DECOMPOSITION (zero-cost, Qwen3-0.6B Base vs Instruct):**
    | Component | Base | Instruct | Change |
    |---|---|---|---|
    | Within-A rho | 0.910 | 0.791 | -0.119 |
    | A-D cross rho | 0.586 | 0.301 | -0.284 |
    | Within-D rho | 0.394 | 0.169 | -0.225 |
    | Gap (AA-AD) | 0.324 | 0.489 | +0.165 |
    70% of gap increase from A-D cross-rho dropping, 30% from within-A dropping.
    MECHANISM: Instruction tuning makes D perturbations behave MORE IDIOSYNCRATICALLY.
    Each rewrite affects cells in its own distinct pattern (within-D coherence drops 2.3x).
    The gap increases NOT because inerts become more consistent but because rewrites
    become more DIVERSE in their cell-level effects.

32. **GAP IS FUNDAMENTALLY A D-COHERENCE MEASURE (zero-cost, all 9 model variants):**
    Across all 9 models (7 base + 2 instruct), what drives the gap?
    - Gap vs Within-A: rho=-0.467 (NS). Within-A is always high (>0.67). DOESN'T predict gap.
    - Gap vs A-D cross: rho=-0.850 (p=0.004). Strong negative. AD DRIVES the gap.
    - Gap vs Within-D: rho=-0.867 (p=0.002). Strong negative. DD DRIVES the gap.
    Within-group ordering is nearly universal: A ≈ B > C ≈ D. Inerts always cohere.
    Base models: mean DD=0.482, mean gap=0.268.
    Instruct models: mean DD=0.197, mean gap=0.471. 2.5x DD collapse.
    INTERPRETATION: The gap measures how DIVERSELY meaning-preserving rewrites affect
    different cells. High gap = each rewrite is an individually distinct operation on the
    model's behavior landscape. Low gap = all rewrites do roughly the same thing.
    Instruction tuning AMPLIFIES rewrite diversity, making each semantic rewrite type
    individually distinct. This means alignment properties measured under one type of
    rewrite (e.g., paraphrasing) won't predict behavior under a different type.

33. **FABLE R23 ADVERSARIAL REVIEW — DEVASTATING (2026-09-19):**
    VERDICT: "Nothing added since R22 survives as a new finding."
    Key corrections applied:
    - Profile transfer (0.326) carries +0.008 cell-specific content → NOT a separate test
    - "ZERO transfer" → "small mean, large spread" (CI [-0.09, +0.14])
    - Magnitude +0.397 → 0.170 after group means → 0.100 without C → mostly C-group floor
    - "Alignment testing cannot generalize" → OUT OF SCOPE (was a 4-token header effect)
    - 5.6x instruct effect → 1.5x like-for-like (definition artifact)
    - "FRAGILITY IS LEARNED NOT TOKENIZED" → "Token segmentation does not predict ordering"
    - Bold "ALL H_compute FAIL" contradicted by deprecation → removed
    - Effective n ≈ 6 operator strata, not 40 cells
    DIRECTIVE: Stop slicing existing 40×24 matrices. Build the balanced cell set (R21
    precondition: n≥60, no identity expressions, operator-balanced) before any new
    cross-model claim.

34. **SmolLM2-1.7B-Instruct COMPLETE — INSTRUCTION TUNING DOES NOT GENERALIZE:**
    - Gap = 0.251 [0.170, 0.339], Z=3.2. Accuracy 94.4%, P(digit)=0.931. ADMISSIBLE.
    - SmolLM2-1.7B Base gap = 0.281. Instruct gap = 0.251. **Ratio: 0.89x (DECREASES).**
    - CIs heavily overlap. Instruction tuning shows NO gap amplification in SmolLM2.
    - Group amplification: A=1.92x, B=2.11x, C=2.75x, D=2.65x — ALL increase uniformly.
    - Contrast with Qwen3: A=0.98x, D=0.69x (selective D reduction → gap increase).
    - Cross-variant correlations: fragility rho=0.689 (strong), magnitude rho=0.883 (very strong).
    - Qwen3 cross-variant fragility: 0.333 (weak). SmolLM2 preserves its fingerprint 2x better.
    CONCLUSION: Instruction tuning's effect on the gap is ARCHITECTURE-DEPENDENT.
    Qwen3 amplification (1.5x) does NOT generalize to SmolLM2 (0.89x).
    | | Qwen3-0.6B | SmolLM2-1.7B |
    |---|---|---|
    | Base gap | 0.324 | 0.281 |
    | Instruct gap | 0.489 | 0.251 |
    | Ratio | 1.5x | 0.89x |
    | D amplification | 0.69x | 2.65x |
    | Frag rho B/I | 0.333 | 0.689 |

35. **MEANING-CHANGE TRANSFER TEST (Qwen3-1.7B, Group E, fp32 CPU):**
    Group E = genuinely meaning-CHANGING perturbations: e_plus_one, e_minus_one, e_times_two,
    e_negate, e_reassign, e_wrong_const. Same 40 cells, two surfaces.
    Does sensitivity to inert additions predict sensitivity to meaning changes?
    - Inert (A+B) → E: rho=0.261 (p=0.104) — NOT SIGNIFICANT.
    - D (rewrites) → E: rho=0.536 (p<0.001) — SIGNIFICANT.
    - A → E: rho=0.345 (p=0.029) — marginal.
    - Inert → D: rho=0.568 (p<0.001) — still strong (consistency check).
    KEY: D sensitivity predicts meaning-change sensitivity but inert sensitivity DOES NOT.
    The A-D gap captures proximity to genuine behavioral vulnerability. D is qualitatively
    closer to actual semantic changes. e_wrong_const (x=42) has rho=nan (constant cross-surface
    distance — no cell variation when output is fixed).
    LIMITATION: Single model. Needs replication on SmolLM2, StableLM before claiming generality.
    NOTE: This is the first test where group types are COMPARED across a semantic boundary.
    All prior work stayed within meaning-preserving perturbations.

36. **MEANING-CHANGE TRANSFER REPLICATION (SmolLM2-1.7B, Group E, fp32 CPU):**
    REPLICATION PARTIALLY SUCCEEDS — pattern differs from Qwen3:
    | Transfer | Qwen3-1.7B | SmolLM2-1.7B |
    |---|---|---|
    | Inert (A+B) → E | 0.261 (NS) | **0.514 (p<0.001)** |
    | D → E | 0.536 (p<0.001) | **0.727 (p<0.001)** |
    | A → E | 0.345 (p=0.029) | 0.504 (p<0.001) |
    | Inert → D | 0.568 (p<0.001) | 0.800 (p<0.001) |
    UNIVERSAL: D→E > Inert→E in both models (Δ=0.213 SmolLM2, Δ=0.275 Qwen3).
    NOT UNIVERSAL: Inert→E is NS in Qwen3 but sig in SmolLM2. SmolLM2 has higher
    baseline correlations everywhere (Inert→D 0.800 vs 0.568), so inert and D are less
    separable as predictors. The A-D gap is sharper in Qwen3 (higher discrimination).
    Per-pert: additive shifts (e_plus_one, e_minus_one) and reassignment well predicted
    by inert (rho=0.518-0.727); multiplicative (e_times_two, 0.215 NS) and negation
    (e_negate, -0.247 NS) are not.
    CONCLUSION: D is always a better predictor of meaning-change sensitivity than inert.
    Whether inert itself reaches significance depends on the model's baseline A-D correlation.

37. **BALANCED CELL SET v3 — FABLE-REVIEWED, RUNNING (2026-09-19):**
    R21/R23 mandated: build operator-balanced set before any new cross-model claims.
    **Design (v3, post-review):**
    - 60 primary cells: + (15), - (15), // (15), % (15)
    - 9 appendix cells: * (6), ** (3) — not in stratum-mean comparisons
    - ALL operands 1-9 (no double-digit tokenization confound)
    - Identity filter: "answer readable without arithmetic step" — excludes a-a, a//a,
      a%b with a<b, a//1, a%1, a%a, etc. Zero identities in final set.
    - // floor-zero (a<b) capped at 3, labeled sub-stratum
    - % answer-0 capped at 3
    - 24 perturbations (same A/B/C/D groups as original 40-cell)
    - 3312 forward passes per model (~4.4 min fp32 CPU)
    **Fable review findings (R24):**
    - v2 identity filter incomplete (% with a<b was identity in old set)
    - Double-digit operands create tokenization confound (Qwen3=2 tokens, SmolLM2=1)
    - Both fixed in v3. Pre-registered: primary 60-cell core for main analysis,
      69-cell all + 60-cell primary + residual-after-stratum-means as three views.
    - Power ~0.72 for rho=0.3 at n=69. Estimate with CI, not test.
    - Commuted pairs kept as within-operation error term.
    **Running:** Qwen3-1.7B-Base first (reference), then sequential models.
    Script: `experiments/run_balanced_kill.py`
    **Qwen3-1.7B-Base COMPLETE (2026-09-19):**
    - Gap = 0.500, CI [0.411, 0.588], Perm Z = 4.4 (old 40-cell: 0.452)
    - Residual gap (stratum means removed) = 0.462 — 92% of gap survives
    - Primary-only gap (60 cells) = 0.524
    - eta2_operator = 0.394, eta2_answer_digit = 0.121
    - Multiplicative model R^2 = 0.85 (pert=67%, cell=18%, interaction=15%)
    - Cell effective dimensionality = 1.6, perturbation = 3.1
    - Fragility inversely correlates with gap (rho=-0.316, p=0.008)
    - d_zero_plus is anti-correlated with group A (gap=0.933); d_cond_true is A-like (gap=0.178)
    - Per-operator A-to-D profile transfer: //=+0.623, *=+0.229, +=−0.013
    - Perturbation magnitudes old vs new: rho=0.989 (instrument validated)
    **SmolLM2-1.7B COMPLETE:**
    - Gap = 0.275, CI [0.202, 0.362], Perm Z = 4.3
    - Old 40-cell gap was 0.281 (delta -0.006, stable)
    - Residual gap = 0.389 (141% of raw — INCREASES after removing operator means)
    - eta2_operator = 0.687, eta2_answer_digit = 0.408
    - Accuracy = 87.6%, admissible
    - Stratum means 15-80x larger than Qwen3 (SmolLM2 much more perturbation-sensitive)
    **CROSS-MODEL (Qwen3 vs SmolLM2, balanced 69-cell):**
    - Cell fragility rho = +0.108 (p=0.38, NS) — model-specific (old: +0.125 same pair)
    - Pert magnitude rho = +0.017 (p=0.94) — was +0.035 this pair (old +0.397 was 5-model mean)
    - Residual fragility rho = +0.183 (p=0.13, NS)
    - Multiplicative residual rho = -0.008 (NS)
    - Only Group C (semantic) magnitude transfers: rho = +0.943, p=0.005 (n=6, same in old+new!)
    - SmolLM2 residual gap INCREASES (+41%) vs Qwen3 DECREASES (-8%)
    **FULL 10-MODEL REANALYSIS (old 40-cell) reveals hierarchical transfer structure:**
    - Magnitude within-family mean +0.700 (n=9), cross-family +0.306 (n=36), p<0.0001 (MW)
    - Fragility within-family mean +0.134 (n=9, NS vs cross), cross-family -0.007 (n=36)
    - Same-scale base/instruct: mag +0.810, frag +0.511 (IT partially preserves fragility)
    - Diff-scale same family: mag +0.669, frag +0.027 (SCALING DESTROYS FRAGILITY)
    - Qwen3 family mag: +0.714 to +0.905. SmolLM2 family mag: +0.770 to +0.883
    - Phi-SmolLM2-1.7B fragility: -0.491 (p=0.001, ANTI-correlated across families)
    **ALIGNMENT IMPLICATION:** Red-teaming types-of-attacks transfer across model scales;
    specific failure-input identity does NOT. IT preserves vulnerable inputs partially; scaling wipes them.
    SmolLM2-360M balanced: gap=0.244, acc=69.2% (NON-ADMISSIBLE), residual=0.471 (+93% increase, same SmolLM2 pattern)
    eta2_operator=0.854 (scales inversely with model size: 360M→0.854, 1.7B→0.687, Qwen3-1.7B→0.394)
    **3-MODEL BALANCED CROSS-ANALYSIS:**
    - Within-family fragility (SmolLM2 1.7B vs 360M): raw +0.496, residual +0.189 (operator-driven)
    - Cross-family (Qwen3 vs SmolLM2-360M): raw -0.257 (ANTI-correlated!), residual +0.161
    - ALL pairs converge to residual ~+0.17 after operator mean removal → small universal component
    **FOUR-LEVEL TRANSFER HIERARCHY (old 10-model + balanced 3-model):**
    L1: Operator type — partially shared across families (removed by stratum means)
    L2: Within-operator cell difficulty — weakly shared (rho~0.21, ONLY universal component)
    L3: Perturbation magnitude ordering — family-specific (within +0.740, cross ~0)
    L4: Cell x perturbation interaction — unique per model (within +0.187, cross ~0)
    Magnitude profile dimensionality: 2.1 (PC1=68%, separates Qwen3 from SmolLM2)
    Fragility profile dimensionality: 4.1 (no dominant structure)
    **FORMAL 4-LEVEL VARIANCE DECOMPOSITION (log-space, balanced 69-cell):**
    Qwen3-1.7B: operator 6.7%, perturbation 67.0%, cell-within-op 11.6%, interaction 14.7%
    SmolLM2-1.7B: operator 34.4%, perturbation 40.5%, cell-within-op 10.3%, interaction 14.9%
    SmolLM2-360M: operator 35.0%, perturbation 39.8%, cell-within-op 7.0%, interaction 18.3%
    → Qwen3 = perturbation-dominated (HOW you perturb matters); SmolLM2 = operator-dominated (WHAT you compute matters)
    → Cross-family transfer fails because models organize computation differently
    → Operator rho: within +0.600, cross -0.657 to +0.086 (families DISAGREE on operator difficulty)
    → Cell-within-op rho ~+0.21 for ALL pairs = arithmetic-difficulty universality
    → Per-stratum universality: subtraction +0.482, addition +0.307; complex operators (%, //) no agreement
    **Qwen3-0.6B-Base COMPLETE (balanced 69-cell):**
    - Gap = 0.271, CI [0.187, 0.354], Z=2.3
    - Residual gap = 0.427 (+57% increase — same pattern as SmolLM2 models)
    - Primary-only gap = 0.283
    - eta2_operator = 0.653, eta2_answer_digit = 0.093
    - Accuracy = 97.2%, admissible
    - Variance: operator=5.9%, perturbation=75.3%, cell=5.7%, interaction=13.1%
    → Even MORE perturbation-dominated than Qwen3-1.7B (75% vs 67%)
    **UPDATED 4-MODEL CROSS-ANALYSIS (2 within-family + 4 cross-family pairs):**
    | Pair | Op | Cell | Pert | Int | Family |
    |---|---|---|---|---|---|
    | Qwen3-0.6B vs Qwen3-1.7B | -0.257 | +0.072 | **+0.937** | +0.133 | WITHIN |
    | SmolLM2-1.7B vs SmolLM2-360M | +0.600 | +0.211 | **+0.740** | +0.187 | WITHIN |
    | Within mean | +0.171 | +0.142 | **+0.838** | +0.160 | |
    | Cross mean (4 pairs) | -0.029 | +0.135 | **-0.072** | +0.039 | |
    KEY: Perturbation magnitude is SHARPLY family-specific (+0.838 within, -0.072 cross).
    Cell-within-operator ~+0.14 for BOTH within and cross → universal arithmetic difficulty.
    Qwen3 within-family pert rho = +0.937 (near-identical perturbation profiles across scales).
    SCALING FINDING: Residual gap INCREASES for all models except largest (Qwen3-1.7B):
    Qwen3-1.7B: -8% | Qwen3-0.6B: +57% | SmolLM2-1.7B: +41% | SmolLM2-360M: +93%
    → Smaller models have more operator-level confound masking the underlying gap.
    eta2_op scales inversely: 360M=0.854, SmolLM2-1.7B=0.687, Qwen3-0.6B=0.653, Qwen3-1.7B=0.394
    **FABLE R25 ADVERSARIAL REVIEW — COMPLETE (2026-09-19):**
    WITHDRAWN:
    - F2 "magnitude rho collapsed +0.397→+0.017": FALSE COMPARISON. Pair was always ~0
      (+0.035 old, +0.017 new). The +0.397 was a panel mean driven by Group C floor (R23).
    - F3 "only semantic (Group C) transfers": MISLABELED. Group C is PREFIX-POSITION INERT
      (comment_before, pass_before, etc.), NOT semantic. Semantic perts are in Group D.
      The rho=+0.943 is a token-adjacency/header-buffering effect, not semantic transfer.
    MERGED: F1 + F4 are ONE observation (high eta2_op → compressed raw gap; mechanically linked).
    DOWNGRADED: F5 between-model gap difference vanishes on residual gaps (CI [-0.09, +0.23]).
    SURVIVING: Both gaps positive and replicated to 3 decimals across cell sets. Gap is REAL.
    NEW FINDINGS from review:
    - N3: Fragility tracks UNCERTAINTY in SmolLM2 (rho=-0.50 with p_answer) but NOT Qwen3 (+0.28).
      SmolLM2's high eta2_op is a COMPETENCE effect (weak on //, % strata), not "modularity".
    - N4: Group C is structurally different from A/B (prefix position vs suffix). Should not be
      pooled as "inert" in any summary — a<->c mixes position with perturbation type.
    - N6: Three D perturbations (d_add_zero, d_zero_plus, d_cond_true) reintroduce literal 0 tokens.
      Cell set purged zeros but 3/6 D perts put them back. Partial contamination, not fatal.
    ACTION ITEMS: (1) Add per-cell predictions to runner output, (2) report gap on argmax-preserving
    cells only, (3) relabel Group C everywhere, (4) pre-register Group C ordering for remaining panel.
    **Phi-1.5 balanced kill RUNNING** (cell 60/69, 64% accuracy — NON-ADMISSIBLE but diagnostic).
    **R26 TOKENIZER CONFOUND TEST — REFUTED:**
    Token-count difference (verbose - minimal) is CONSTANT at 4 for ALL 24 perturbations
    across ALL 5 tokenizers (Qwen3, SmolLM2, Phi). One exception: c_newline_before = 3 for Qwen3.
    Family-specific perturbation magnitude profiles CANNOT be explained by tokenizer differences —
    all models see the same token-level input change. The perturbation magnitude hierarchy is
    genuinely about how models process identical inputs differently.
    **R26 PRE-REGISTRATION (BEFORE Phi-1.5 result seen):**
    H1: eta2_op tracks operator competence (predict Phi eta2_op > 0.5 given 64% accuracy)
    H2: Pert magnitude family-specific (predict Phi pert rho < +0.2 vs all — TOKENIZER RULED OUT)
    H3: Cell-within-op universal ~+0.14 — OPERAND-MAGNITUDE CONFOUND REFUTED:
      Per-model residual-vs-operand_max: ALL near zero (-0.116 to +0.004, p>0.34).
      Partial correlations identical to raw (controlling for max(a,b) changes nothing).
      The universal component is NOT "bigger numbers are harder."
    H4: Group C ordering shared (comment < statement < whitespace, effective df ~3)
    H5/H6 merged into H1 (mechanical corollaries of competence gradient)
    Per-cell records added to runner (R25 N5 action). Hypotheses pre-registered in scratchpad.

    **Phi-1.5 COMPLETE (balanced 69-cell, NON-ADMISSIBLE):**
    - Accuracy: 65.0%, P(digit) median: 0.0004 (barely assigns probability to correct digit)
    - Gap: 0.034, CI [0.006, 0.067] — near zero (old 40-cell was 0.055)
    - Residual gap: 0.190 (5.6x increase — most dramatic in entire panel)
    - eta2_operator: 0.834 (comparable to SmolLM2-360M's 0.854)
    - Variance: op=33.5%, pert=49.6% (intermediate between Qwen3 and SmolLM2)
    - All group rhos very high (0.69-0.85) — Phi treats all perturbation types similarly
    - A-D gap nearly vanishes (0.851 vs 0.818) — no semantic vs inert distinction
    **PRE-REGISTERED H1 CONFIRMED (3 families):** eta2_op inversely scales with accuracy:
    | Model | Family | Accuracy | eta2_op |
    | Qwen3-1.7B | Qwen3 | 99.1% | 0.394 |
    | Qwen3-0.6B | Qwen3 | 97.2% | 0.653 |
    | SmolLM2-1.7B | SmolLM2 | 87.6% | 0.687 |
    | SmolLM2-360M | SmolLM2 | 69.2% | 0.854 |
    | Phi-1.5 | Phi | 65.0% | 0.834 |
    **H2 PARTIALLY CONFIRMED (nuance):** Phi pert rho:
    - vs Qwen3: +0.155 to +0.238 (NS) — as predicted
    - vs SmolLM2-1.7B: +0.484 (p=0.016) — SIGNIFICANT! Unexpected cross-family similarity
    - CAVEAT: Both Phi and SmolLM2 have low accuracy — shared pert profile may be
      competence confound (confused models produce similar distance patterns regardless of family).
      Need admissible cross-family model (StableLM) to resolve.
    StableLM-2-1.6B RUNNING. late_twins.py qwen3 consuming 7.5GB (stale, from prior agent).

    **StableLM-2-1.6B COMPLETE (balanced 69-cell, NOT ADMISSIBLE):**
    - Accuracy: 72.0%, P(digit) median: 0.621 (reasonable but acc<80%)
    - Gap: 0.215, CI [0.161, 0.285], Z=2.34 — positive, significant
    - Residual gap: 0.173
    - eta2_operator: 0.608 (BREAKS H1 monotonicity — lower than SmolLM2-360M despite similar accuracy)
    - Operator difficulty: ** > // > + > - > * > % (shares // fragility with SmolLM2)

    **TWO-CLUSTER FINDING (6 models, PROVISIONAL pending R26 review):**
    Perturbation-level cross-correlations reveal TWO clusters, NOT family-specific profiles:
    - QWEN3 CLUSTER (2 models): +0.937 within, near-zero with everything else (+0.017 to +0.042 vs SmolLM2)
    - NON-QWEN3 CLUSTER (4 models, 3 families): SmolLM2-StableLM rho=+0.642/+0.654,
      Phi-SmolLM2-1.7B=+0.543, Phi-StableLM=+0.457. Moderate-high mutual rho ACROSS families.
    This is NOT a competence confound: StableLM (72%) has rho=+0.642 with SmolLM2-1.7B (88%).
    The original "cross-family = zero" was a QWEN3-SPECIFIC effect, not universal.
    RISK: Only 3 admissible models (2 Qwen3 + 1 SmolLM2). Cannot test admissible non-Qwen3 cross-family.
    Fable R26 review IN PROGRESS.

    **H1 UPDATE (6 models):** rho(accuracy, eta2_op) = -0.771 (p=0.072).
    StableLM anomaly: 72% acc but eta2=0.608 (lower than 360M at 0.854 with 69% acc).
    Monotonic H1 WEAKENED to "strong negative tendency with exceptions."

    **Admissible-only transfer (unchanged):** within +0.937, cross +0.029. Gap +0.907.
    But this is Qwen3-vs-SmolLM2 only (n=3 models, n=1 within pair, n=2 cross pairs).

**Next:** SmolLM2-1.7B-Instruct (IT comparison), then Qwen3-0.6B-Instruct. R26 review pending.

38. **HEADER RELOCATION TEST (Qwen3-1.7B-Base, R22 E1, fp32 CPU, 2026-09-18):**
    Moved `# Python code\n` from BEFORE code to AFTER assignment line.
    Original verbose: `# Python code\n{before}{assign}\n{after}{query}`
    Relocated verbose: `{before}{assign}\n# Python code\n{after}{query}`
    | Group Rho | Original | Relocated | Delta |
    |---|---|---|---|
    | A_A | 0.786 | 0.584 | -0.202 |
    | A_B | 0.812 | 0.575 | -0.238 |
    | A_C | 0.611 | 0.338 | -0.273 |
    | A_D | 0.333 | 0.350 | +0.016 |
    | B_B | 0.837 | 0.739 | -0.098 |
    | B_C | 0.617 | 0.162 | -0.456 |
    | C_C | 0.589 | 0.453 | -0.135 |
    | D_D | 0.224 | 0.239 | +0.015 |
    | Gap | 0.453 | 0.234 | -0.219 |
    HEADER ADJACENCY HYPOTHESIS: REJECTED AS SOLE MECHANISM. Simple adjacency predicted
    C_C should INCREASE (freed from header); instead it DECREASED (-0.135). The header in
    the MIDDLE disrupts ALL correlations, not selectively the adjacent ones. B_C dropped
    the most (-0.456) because the header now physically separates before and after blocks.
    KEY: D_D and A_D barely changed (~+0.015). D perturbations rewrite the assignment itself
    and are independent of header position. The gap halved from 0.453 to 0.234: roughly 50%
    header effect, 50% genuine A-D structural difference. The residual gap (0.234) matches
    the 11-bin gap direction (0.359) and the balanced-cell gap (0.500), confirming a real
    structural component beyond header artifacts.

39. **FABLE R24 ADVERSARIAL REVIEW — MAJOR CORRECTIONS (2026-09-18):**
    Two-round review of meaning-change transfer findings. Key corrections:
    (a) 16/40 cells (40%) are identity expressions — dominant confound in all prior results.
    (b) SmolLM2 Inert->E (0.514) RETRACTED: drops to 0.143 (NS) on non-identity cells.
    (c) D->E transfer driven by 2 of 6 D members (add_zero + cond_true) in BOTH models on
        non-identity cells. Other 4 D members contribute near-zero.
    (d) Token-prefix sharing explains the correlation structure: pairs that share more prefix
        tokens (add_zero/plus_one) correlate highly; prefix-breaking pairs (temp_var/plus_one)
        do not.
    (e) Full-logit cosine vs 11-bin disconnect: in Qwen3, cell-profile cross-readout rho is
        0.099 (inert) and -0.084 (D). The full-logit cosine measures formatting geometry, not
        answer sensitivity, in Qwen3.
    (f) BUT: the gap SURVIVES on 11-bin in all 4 models (0.19-0.37). Not just formatting.
    (g) E-group has no coherence beyond form twins. Partial E-E rho after removing twins ~0.
    RETRACTIONS: SmolLM2 Inert->E; "depends on model" difference (NS); "additive transfers
    multiplicative doesn't" (prefix confound); "D captures behavioral vulnerability" (no
    behavioral quantity measured).
    SURVIVING: D->E > Inert->E on non-identity cells (0.46/0.39 vs 0.07/0.14), driven by
    add_zero + cond_true. Consistent with token-prefix sharing, not yet separable from it.

40. **SmolLM2 balanced kill v3 COMPLETE (2026-09-19).**
    Gap=0.275, CI [0.202, 0.362], perm z=4.33. 69 cells, 87.6% accuracy, admissible.
    Group rhos (full-logit): AA=0.944, AB=0.929, AC=0.757, AD=0.669, DD=0.531.
    DD < AD again (universal). Residual gap 0.389 (HIGHER than raw — operator strata absorb
    some of the gap, so partialling them out increases the A-D difference).
    Eta-squared: operator=0.687, answer_digit=0.408.
    Compares to Qwen3 balanced (0.500): SmolLM2 lower, consistent with original 40-cell
    (SmolLM2 0.281, Qwen3 0.452). Both survive on balanced cells.

41. **Prefix-form crossover COMPLETE — Qwen3 (2026-09-19).**
    2x2 design: D members (add_zero=`e+0`, zero_plus=`0+e`) x E members (plus_one=`e+1`,
    one_plus=`1+e`). 80 new forward passes for one_plus, rest from existing data.
    Non-identity cells (n=24):
    - add_zero -> plus_one: rho=0.634 (same prefix `e + N`, STRONG)
    - add_zero -> one_plus: rho=-0.301 (diff prefix, ANTI-correlation)
    - zero_plus -> one_plus: rho=0.063 (same prefix `N + e`, WEAK)
    - zero_plus -> plus_one: rho=0.428 (diff prefix, MODERATE)
    - Diagonal mean: 0.349, off-diagonal mean: 0.063
    - E-E rho (plus_one vs one_plus): -0.143 (independent)
    **R25 CORRECTION:** The 2×2 is a COLUMN main effect (plus_one transfers, one_plus doesn't),
    not a diagonal/prefix effect. Interaction NS in both models. `one_plus` is a garden-path
    column (precedence confound on `* // %` cells). Operator strata reverse the diagonal in
    Qwen3. "E-E independence" RETRACTED (CI spans zero in Qwen3). See entry 46.

42. **SmolLM2 prefix crossover COMPLETE — signs replicate Qwen3 (2026-09-19).**
    Non-identity cells (n=24):
    | D \ E | plus_one (e+1) | one_plus (1+e) |
    |---|---|---|
    | add_zero (e+0) | **0.676** | -0.405 |
    | zero_plus (0+e) | 0.390 | -0.159 |
    Diagonal mean: 0.258, off-diagonal: -0.007. E-E rho: **-0.470 (p=0.021, SIG)**.
    All four signs replicate Qwen3. E-E anti-correlation is stronger and significant.

43. **VARIANCE ANALYSIS KILLS SIMPLE PREFIX STORY (zero-cost, 2026-09-19).**
    add_zero has 3.55x (Qwen3) / 4.27x (SmolLM2) more cross-cell variance than zero_plus.
    The "diagonal > off-diagonal" is dominated by one high-variance entry, not a symmetric
    prefix-form mechanism. Full D x E table shows d_temp_var REVERSES the pattern in both
    models. Mean D -> plus_one: +0.311 (p=0.139, NS). Mean D -> one_plus: -0.277 (p=0.191, NS).
    Neither prefix nor gradient hypothesis survives aggregation over all D members.
    ROBUST FINDINGS:
    (a) E-E anti-correlation (sig in SmolLM2, -0.470). Sensitivity to `e+1` and `1+e` are
        anti-correlated. Operand position matters.
    (b) add_zero specifically transfers to plus_one (0.634/0.676 cross-model) and anti-transfers
        to one_plus (-0.301/-0.405). This is additive-position-specific, not a general D property.
    (c) Transfer is perturbation-member-specific, not perturbation-group-specific. Individual
        D member identity determines which E members are predicted, not "D as a group."

44. **11-BIN CROSSOVER REANALYSIS — TOKEN-PREFIX NULL REFUTED (Qwen3, 2026-09-19).**
    CRITICAL: The E-E anti-correlation and position-dependent transfer are FULL-LOGIT-SPECIFIC.
    On 11-bin (answer distribution), the picture completely reverses:
    | Cell | Full-logit | 11-bin |
    |---|---|---|
    | add_zero->plus_one [DIAG] | +0.634 | +0.610 |
    | add_zero->one_plus [OFF] | -0.301 | **+0.531** |
    | zero_plus->plus_one [OFF] | +0.428 | **+0.710** |
    | zero_plus->one_plus [DIAG] | +0.063 | **+0.423** |
    | E-E rho | -0.143 | **+0.758 (p<0.001)** |
    ALL four entries positive and significant on 11-bin. E-E flips from -0.14 to +0.76.
    Off-diagonal HIGHER than diagonal (0.620 > 0.517) — gradient hypothesis WINS on 11-bin.
    IMPLICATIONS:
    (a) The R24 "token-prefix null" (transfer follows prefix, not meaning) is REFUTED on
        11-bin. On the answer distribution, all additive perturbations transfer regardless of
        operand position. The token-prefix pattern existed only in formatting geometry.
    (b) Entries 41-43 full-logit findings (member-specificity, E-E anti-correlation, operand
        position) are RETRACTED as general claims. They describe full-logit formatting effects,
        not answer-distribution behavior.
    (c) The 11-bin picture is simpler: cells have a common "sensitivity" factor. Both D and E
        perturbations load on this factor. The gap (AA-AD) measures D-DIVERSITY (how differently
        each D member modulates this factor), not D-DISCONNECTION from E.
    (d) REFRAMING: The gap is real. D->E transfer is also real (on 11-bin). The gap measures
        internal D diversity, not lack of cross-group transfer.
    **R25 QUALIFICATION:** This 11-bin result was on old 40-cell set (violating balanced-v3
    mandate). R25 shows the 2×2 column effect (one_plus anomaly) confounds the crossover.
    SmolLM2 partial replication (entry 45) weakens the "broad transfer" claim. Qwen3 11-bin
    may also be operator-stratum driven. Needs rerun on balanced v3 with within-operator control.

45. **SmolLM2 11-BIN CROSSOVER REPLICATION — PARTIAL (2026-09-19).**
    Core artifact claim CONFIRMED: E-E flips from -0.470 (p=0.021) to +0.124 (NS, p=0.563).
    Anti-correlation is a full-logit formatting effect in both models.
    BUT broad positive transfer does NOT replicate:
    | Cell | Full-logit | 11-bin |
    |---|---|---|
    | add_zero->plus_one [DIAG] | +0.676 | **+0.564** |
    | add_zero->one_plus [OFF] | -0.405 | +0.089 (NS) |
    | zero_plus->plus_one [OFF] | +0.390 | +0.170 (NS) |
    | zero_plus->one_plus [DIAG] | -0.159 | +0.129 (NS) |
    | E-E rho | -0.470 | +0.124 (NS) |
    Only 1/4 entries significant on 11-bin (add_zero→plus_one, the dominant D member).
    Qwen3 had 4/4 significant and E-E=+0.758. SmolLM2 has 1/4 and E-E=+0.124.
    IMPLICATIONS:
    (a) Anti-correlation-as-artifact: CONFIRMED across both models. The sign flip is universal.
    (b) Broad positive transfer on 11-bin: QWEN3-SPECIFIC, not universal.
    (c) add_zero dominance: The only surviving cross-readout transfer in BOTH models is
        add_zero→plus_one. This is the highest-variance D member (3.5-4.3x).
    (d) The "D-diversity not D-disconnection" reframing needs qualification: on SmolLM2,
        D→E transfer on 11-bin is weak outside add_zero. The gap still measures D-diversity,
        but D-diversity doesn't guarantee D→E transfer on the answer distribution.

46. **FABLE R25 ADVERSARIAL REVIEW — DEVASTATING (2026-09-19).**
    The 2×2 is a COLUMN main effect, not a diagonal/prefix effect.
    Column effect (plus_one − one_plus): +0.65 / +0.82 (CI excl 0 in both models).
    Row effect: NS. Interaction: NS (p=0.085 / 0.052, CI includes 0).
    `one_plus` (1+e) is a GARDEN-PATH column: anti-correlates with 17/18 inerts in Qwen3
    (mean rho -0.44). `1 + 8 // 2` is parsed as precedence override.
    Operator strata REVERSE Qwen3 diagonal: within-stratum ad→plu = -0.23 (sign flip!).
    SmolLM2 ad→plu survives within-operator (+0.52) but interaction collapses to 0.04.
    RETRACTIONS: "prefix form partial mediator" → downgrade; "E-E independence" → retract;
    "meaning change determines cells" → withdraw; "token-prefix sharing" → narrow to
    late-edit locality only.
    MOST HONEST READING: Edit locality relative to `e` + precedence artifact in `1+e`.
    Append-after-e edits cohere (rho 0.43-0.66). Meaning change has NO demonstrated role.
    CRITICAL GAP: No meaning-preserving late twin (e-0, e*1) ever measured. If these
    correlate with e+1 as much as e+0 does, the D→E narrative is dead.
    EXPERIMENT DESIGNED: Late-twin disambiguation (6 perts on balanced v3, both readouts).

47. **WITHIN-OPERATOR STRATUM ANALYSIS — GAP SURVIVES AND GROWS (zero-cost, 2026-09-19).**
    On balanced v3 (69 cells), tested whether the gap (AA-AD) is operator-stratum artifact.
    | Metric | Qwen3 | SmolLM2 |
    |---|---|---|
    | Raw gap | 0.500 | 0.275 |
    | Within-operator residual gap | **0.515** | **0.365** |
    | DD < AD within strata | 5/5 | 3/5 |
    The residual gap is HIGHER than raw in BOTH models. Operator strata SUPPRESS the gap.
    Per-stratum: `-` has largest gap (0.711/0.221), `*` smallest (0.272/0.043).
    CRITICAL: d_add_zero and d_zero_plus (the crossover D members) have the LOWEST AD
    correlations of all D members (0.068/-0.171 Qwen3). They're the LEAST representative
    D members. R25's "operator strata reverse the diagonal" was about these anomalous members
    on the old 40-cell set — not representative of the gap as a whole.
    d_cond_true has the HIGHEST AD (0.584/0.839). d_temp_var, d_parens, d_int_cast: 0.31-0.39
    (Qwen3) / 0.70-0.82 (SmolLM2). The gap is driven by the majority of D members, not outliers.

48. **LATE-TWIN DISAMBIGUATION — MEANING IS DEAD ON 11-BIN (Qwen3, 2026-09-19).**
    R25's #1 experiment. 6 append-after-e perts on balanced v3 (69 cells), both readouts.
    D (meaning-preserving): e+0, e-0. E (meaning-changing): e+1, e-1, e+2, e+3.
    Sanity: kill-data add_zero vs fresh e+0 rho=1.0000 (deterministic).
    **Full-logit:** D-D=+0.658, D-E=+0.424, E-E=+0.405. D-D > D-E by +0.234.
    Meaning-preserving twins more similar in formatting geometry.
    **11-bin:** D-D=+0.591, D-E=+0.644, E-E=**+0.781**. D-D < D-E by -0.053.
    **Meaning is IRRELEVANT on answer distribution.** E-E is HIGHEST. Meaning-changing perts
    are MORE internally coherent. e+0 correlates MORE with E (0.73-0.83) than e-0 (0.59) on
    11-bin. INTERPRETATION: all late-edit perts load on common "surface sensitivity" factor.
    Meaning changes AMPLIFY this (E-E > D-E > D-D) via answer-uncertainty increase.
    The D->E narrative is DEAD on 11-bin.
    **Full-logit bootstrap CI on D-D minus D-E: [+0.082, +0.382]** (excludes zero: meaning
    matters on formatting geometry, confirmed). 11-bin bootstrap pending.

49. **AMPLIFICATION GRADIENT ANALYSIS (Qwen3 late-twin reanalysis, zero-cost, 2026-09-19).**
    E-E coherence is STRUCTURED by answer-shift direction:
    Same-sign E-E (both shift answer same way): mean rho=0.836 (11-bin), 0.543 (FL).
    Opposite-sign E-E: mean rho=0.725 (11-bin), 0.267 (FL).
    By |value_diff|: 1=>0.847, 2=>0.771, 3=>0.700, 4=>0.749 (nearly monotonic).
    Not just "E is noisier" — the noise is structured by answer-shift direction and magnitude.
    Confirms amplification mechanism: cells where model is uncertain become more uncertain for
    BOTH same-direction perts, creating correlated distance profiles.

50. **SmolLM2 LATE-TWIN REPLICATION — CONFIRMED (2026-09-19).**
    Same 828 fwd passes on SmolLM2-1.7B. Sanity: rho=1.0000 vs kill-data add_zero.
    **Full-logit:** D-D=+0.704, D-E=+0.468, E-E=+0.541. D-D minus D-E = +0.236.
    Bootstrap CI [+0.106, +0.357] (excludes zero: meaning matters on FL, confirmed).
    **11-bin:** D-D=+0.365, D-E=+0.349, E-E=+0.415. D-D minus D-E = **+0.015**.
    Bootstrap CI **[-0.171, +0.200]** (SPANS ZERO: meaning is IRRELEVANT on answer dist).
    REPLICATION TABLE:
    | | Qwen3 FL | Qwen3 11b | SmolLM2 FL | SmolLM2 11b |
    |---|---|---|---|---|
    | D-D | +0.658 | +0.591 | +0.704 | +0.365 |
    | D-E | +0.424 | +0.644 | +0.468 | +0.349 |
    | E-E | +0.405 | +0.781 | +0.541 | +0.415 |
    | D-D minus D-E | +0.234 | -0.053 | +0.236 | +0.015 |
    Both models: meaning matters on FL (CI excludes zero), meaning irrelevant on 11-bin.
    SmolLM2 amplification weaker (E-E=0.415 vs Qwen3 0.781) but ordering preserved.

51. **CROSS-POSITION TRANSFER (zero-cost reanalysis, both models, 2026-09-19).**
    Reanalyzed balanced kill data: A (after), B (novel after), C (before), D (rewrites).
    Qwen3 (full-logit): A-A=0.762, C-C=0.458, A-C=0.557, A-D=0.262, C-D=0.276.
    SmolLM2 (full-logit): A-A=0.944, C-C=0.805, A-C=0.757, A-D=0.669, C-D=0.626.
    KEY: A-A > A-C in both (bootstrap CI excludes zero). "After" perts cohere within position.
    BUT: C-C < A-C in Qwen3 (!), C-C marginal over A-C in SmolLM2. "Before" perts less cohesive.
    A-D ≈ C-D in both: rewrites are equally low from both positions. Gap doesn't depend on
    reference position. D diversity (DD << AD) is confirmed as the gap's structural cause.

52. **TRIVIAL PERT INFLATION TEST — FALSIFIED (zero-cost, 2026-09-19).**
    Dropped trivial A (empty, whitespace, newline) from gap computation.
    Qwen3: gap 0.500 => 0.451 (10% drop). SmolLM2: 0.275 => 0.242 (12% drop).
    A-D unchanged (0.264 vs 0.260 Qwen3). Gap shrinkage comes from A-A dropping,
    not A-D changing. B group gap is similar (0.522 Qwen3, 0.261 SmolLM2).
    The gap is ROBUST to perturbation selection within inerts.

53. **ADVERSARIAL SELF-REVIEW — KEY ATTACKS ON LATE-TWIN (2026-09-19).**
    (a) D-group has ONLY 1 D-D pair (e+0 ~ e-0). And e+0/e-0 are nearly identical (both add
    zero). With 1 pair vs 8 D-E and 6 E-E pairs, D-D estimate has massive variance.
    RESPONSE: Bootstrap CI for SmolLM2 11-bin spans zero [-0.171, +0.200], confirming low power.
    Correct claim: "we FAIL TO DETECT meaning effect on 11-bin" (null), not "meaning is dead."
    FIX: Expand D to 4 append-after-e perts: e+0, e-0, e*1, e**1 => 6 D-D pairs.
    (b) Amplification gradient (E-E > D-E > D-D) may be trivial: E perts inject more variance,
    so pairs with more answer-shift naturally correlate more. FIX: regress out answer-shift.
    (c) Finding 3 (cross-position) rated SOLID.
    (d) Finding 4 (trivial inflation): 10-12% is "mostly robust," not "falsified."

54. **StableLM-2-1.6B LATE-TWIN — EDIT LOCALITY ON BOTH READOUTS (2026-09-19).**
    3rd architecture, 828 fwd passes, 237s. Sanity: rho=1.0000 vs kill-data.
    **Full-logit:** D-D=+0.579, D-E=+0.596, E-E=+0.713. D-D - D-E = **-0.017**.
    Bootstrap CI [-0.170, +0.129] (spans zero). EDIT LOCALITY EVEN ON FULL-LOGIT.
    **11-bin:** D-D=+0.414, D-E=+0.458, E-E=+0.650. D-D - D-E = **-0.044**.
    Bootstrap CI [-0.201, +0.122] (spans zero).
    StableLM doesn't encode +/- operator distinction in formatting geometry.
    "Meaning matters on FL" is MODEL-SPECIFIC (Qwen3, SmolLM2), not universal.
    **Edit locality on 11-bin: NOW 3/3 MODELS.**
    | Model | FL D-D-D-E | FL CI | 11b D-D-D-E | 11b CI |
    |---|---|---|---|---|
    | Qwen3 | +0.234 | [+0.082,+0.382]* | -0.053 | pending |
    | SmolLM2 | +0.236 | [+0.106,+0.357]* | +0.015 | [-0.171,+0.200] |
    | StableLM | -0.017 | [-0.170,+0.129] | -0.044 | [-0.201,+0.122] |
    * = CI excludes zero (meaning matters on FL for these two only)
    E-E amplification strongest in StableLM: e+2~e+3 = 0.922 (FL), 0.785 (11-bin).

**Next:** Expanded-D experiment (e*1, e**1, e*2, e**2 added) for 6 D-D pairs.
Addresses the single strongest attack: 1-pair D-D estimate had massive variance.

55. **FABLE R26 ADVERSARIAL REVIEW — LEVEL-OF-ANALYSIS DEMOLITION (2026-09-18).**
    Reviewed 6-model balanced kill cross-analysis. **Independently verified all claims.**

    **F1: Two-cluster structure DEMOLISHED.** The "two clusters" were a group-ordering
    artifact. Qwen3 has D>A mean distance; all 4 non-Qwen3 models have D<A. This single
    bit reproduces the partition. Group-mean-only null gives HIGHER rho than actual data
    for non-Qwen pairs (S1.7-S360: null +0.800 > actual +0.739). Within-group ranks
    (group ordering removed): SmolLM2 within-family collapses +0.739 -> +0.014.
    SmolLM2-StableLM: +0.642 -> +0.129. StLM-Phi: +0.457 -> -0.029.
    BUT Qwen3-Phi RISES: +0.253 -> +0.614. Qwen3 within-family SURVIVES: +0.943.
    The "Qwen3 outlier" was group ordering disagreement, not perturbation-level.

    **F3: Cell-level vs perturbation-level DISSOCIATION.** Independently verified:
    Qwen3 within-family: pert rho = +0.937, cell rho = **+0.015 (p=0.90)**.
    Models agree on WHICH PERTURBATION TYPES are damaging but NOT which CELLS are
    vulnerable. Split-half cell reliability 0.76-0.98 (not noise — real disagreement).
    Cell-level cross-model rho exists WITHIN SmolLM2 (+0.496), but NOT within Qwen3.
    S1.7-Phi: pert +0.543, cell **-0.576** (ANTI-correlated at cell level!).

    **F2: H1 weakened.** rho=-0.771, p=0.072, n=6. Without Phi (p_digit=0.0004): p=0.19.
    eta2 driven by per-operator accuracy dispersion, not mean accuracy.

    **F4: Qwen3 tokenizer merge.** c_newline_before merges `\n\n` into one token in
    Qwen3 tokenizer. This is Qwen3-0.6B's single largest perturbation. Without it,
    Qwen3's C>A ordering drops to match other models. Does not change cluster rhos.

    **RETRACTIONS (R26):**
    - ~~Two-cluster structure~~ -> GROUP ORDERING ARTIFACT (one bit: D>A vs D<A)
    - ~~SmolLM2 within-family pert rho +0.739~~ -> Drops to +0.014 within-group
    - ~~Non-Qwen3 cluster~~ -> COLLAPSES within-group (all pairs near zero)
    - ~~Qwen3 as outlier~~ -> Only in group ordering; within-group Qwen3-Phi = +0.61
    - ~~"Cross-family = zero" (admissible, pert-level)~~ -> Level artifact; cell rho
      was already near-zero WITHIN Qwen3 family
    - H1 monotonicity -> NOT ESTABLISHED (p=0.19 without Phi)

    **SURVIVING (R26-verified):**
    - Within-model gap (AA - AD) real for all 6 models
    - Qwen3 within-family within-group pert rho = +0.943 (genuine fine-grained)
    - Cell profiles reliable (split-half 0.76-0.98)
    - Level-of-analysis dissociation itself (pert vs cell rho diverge)
    - Qwen3 D>A group ordering (genuine, possibly tokenizer-driven)

    **NEW FINDING: LEVEL-OF-ANALYSIS DISSOCIATION.**
    Pert-level rho measures "which perturbation types are more damaging overall."
    Cell-level rho measures "which arithmetic problems are more vulnerable."
    These are INDEPENDENT. Two Qwen3 models agree perfectly on perturbation type
    ordering (+0.937) but zero on cell vulnerability (+0.015). This means:
    (a) Post-training that targets perturbation types would transfer within-family
    (b) Post-training that targets specific problem vulnerabilities would NOT
    (c) The "transfer gap" depends entirely on WHAT is transferred (type vs item)

    **R26 RECOMMENDATION:** Re-run all 6 models with current runner (per_cell for
    all), pre-register within-group analysis. Add 7th model from 4th family >80%
    accuracy. Cell-level analysis is the correct level for alignment transfer claims.

56. **DISTANCE MATRIX DECOMPOSITION — FAMILY-SPECIFIC VARIANCE STRUCTURE (zero-cost, 2026-09-18).**
    Decomposed each model's 69x24 distance matrix into: Op×Group block means, cell-beyond-
    operator, pert-beyond-group, and cell×pert residual. Variance explained:
    | Model | OpGrp | CellBeyOp | PertBeyGrp | Residual |
    |---|---|---|---|---|
    | Q1.7 | 14.6% | 10.4% | **48.2%** | 26.8% |
    | Q0.6 | 10.5% | 4.1% | **67.1%** | 18.2% |
    | S1.7 | **63.1%** | 14.6% | 2.2% | 20.1% |
    | S360 | **67.9%** | 7.7% | 4.8% | 19.5% |
    | Phi | **50.2%** | 7.5% | 24.2% | 18.1% |
    | StLM | **47.6%** | 18.0% | 12.0% | 22.4% |

    **KEY FINDING: The distance matrix has family-specific variance structure.**
    - Qwen3: dominated by PERT-BEYOND-GROUP (48-67%). "Which specific perturbation is severe."
    - SmolLM2: dominated by OP×GROUP (63-68%). "Which operator×group combos are hard."
    - Phi/StableLM: mixed (~50% OpGrp, 12-24% PertBeyGrp).

    Cross-model pert-beyond-group (the clean level):
    - Qwen3 within: +0.893 (genuine fine-grained). SmolLM2 within: +0.157 (NS!!!).
    - Qwen3-Phi: +0.534 (p=0.007). Qwen3-StLM: +0.483. GENUINE cross-family.
    Cell-beyond-operator: ALL near zero (Qwen3 within: +0.003). Q1.7-Phi: +0.412 (exception).
    Residual (cell×pert): ALL tiny (max +0.263). Model-specific fingerprint.

    **IMPLICATION: "Perturbation profile transfer" measures DIFFERENT THINGS per family.**
    For Qwen3: shared within-group perturbation ordering (genuine structure).
    For SmolLM2: shared group ordering + Op×Group (artifact, collapses within-group).
    The high pert rho within SmolLM2 (+0.739) was OpGroup structure, not pert specificity.
    SmolLM2 within-family pert-beyond-group = +0.157 (NS) — NO genuine pert agreement.

    **TRANSFER DEPENDS ON GRANULARITY:**
    - Pert-type level: transfers within Qwen3 (+0.89), Qwen3->Phi (+0.53). NOT SmolLM2.
    - Cell level: transfers NOWHERE (all near zero beyond operator).
    - Operator level: trivially shared (dominant structure in non-Qwen3 models).
    - Residual: unique to each model (max +0.26).

57. **EXPANDED-D LATE-TWIN — SmolLM2: EDIT LOCALITY CONFIRMED WITH 6 D-D PAIRS (2026-09-19).**
    Addresses adversarial attack #1 (entry 53a): D-group had only 1 D-D pair.
    Added 4 new perts: e*1 (D), e**1 (D), e*2 (E), e**2 (E). Now 4 D + 6 E = 10 perts.
    6 D-D pairs, 24 D-E pairs, 15 E-E pairs. 552 new fwd passes (reused 6 existing perts).
    **Full-logit:** D-D=+0.449, D-E=+0.269, E-E=+0.209. D-D - D-E = **+0.180**.
    Bootstrap CI [+0.074, +0.285] => **MEANING MATTERS** (CI excludes zero).
    **11-bin:** D-D=+0.310, D-E=+0.225, E-E=+0.201. D-D - D-E = **+0.085**.
    Bootstrap CI **[-0.017, +0.189]** => **MEANING NOT DETECTED** (CI spans zero).
    INTERPRETATION: With 6x more D-D pairs, the finding HOLDS. Edit locality on 11-bin is
    not an artifact of 1-pair D-D variance. D-D means dropped (from 0.704 to 0.449 FL, 0.365
    to 0.310 11-bin) because e*1/e**1 correlate less with e+0/e-0 than they do with each other
    (e*1~e**1 = 0.662 FL, 0.521 11-bin vs e+0~e*1 = 0.250 FL, 0.283 11-bin).
    NOTABLE: e*1/e**1 correlate HIGH with e*2/e**2 (same operator family, different meaning):
    e*1~e*2 = 0.663 FL, 0.262 11b. e**1~e**2 = 0.659 FL, 0.510 11b.
    OPERATOR FAMILY > MEANING. Pairs sharing an operator (regardless of D/E) correlate more
    than pairs sharing meaning-preservation status but different operators.
    StableLM expanded-D RUNNING.

**Next:** StableLM expanded-D (reuses existing JSON), then Qwen3 (all 10 perts from scratch).

58. **SVD DECOMPOSITION — QWEN3 CELL ANTI-TRANSFER (zero-cost, 2026-09-18).**
    Rank-1 SVD of centered 69x24 distance matrices. 
    V1 (pert severity): Q1.7-Q0.6 = +0.914 (shared). U1 (cell fragility): Q1.7-Q0.6 =
    **-0.421** (p=0.0003, ANTI-correlated!). Since V1 same-sign and U1 opposite, the 
    rank-1 interaction is anti-correlated: cells fragile for Q1.7 are robust for Q0.6.
    **Mechanism:** Different operator vulnerability rankings across Qwen3 scales.
    Q1.7 top fragile: % and // (7 of top 10). Q0.6 top fragile: - (8 of top 10).
    Cell-beyond-operator still +0.003 — even within same operator, cell orderings differ.
    Rank-1 accounts for 68-83% of variance; rank-2 reaches 84-94% (low-rank matrix).
    Multiplicative model (d = f(cell)*g(pert)) better than additive for 5/6 models.
    Q1.7-Phi cell-beyond-op agreement (+0.412) driven by % operator (rho=+0.675, p=0.006).
    "Universal" cell fragility is SmolLM2/StableLM-specific (+0.63), not Qwen3/Phi (0).
    R27 adversarial review RUNNING.

59. **MULTIPLICATIVE RESIDUAL STRUCTURE — GAP SURVIVES AND GROWS (zero-cost, 2026-09-18).**
    After log-additive residualization (removing cell_fragility × pert_severity):
    | Model | Gap_raw | Gap_residual | AA_res | AD_res | DD_res |
    |---|---|---|---|---|---|
    | Q1.7 | +0.500 | **+0.532** | +0.255 | -0.277 | +0.140 |
    | Q0.6 | +0.271 | **+0.475** | +0.277 | -0.198 | +0.028 |
    | S1.7 | +0.275 | **+0.647** | +0.449 | -0.198 | +0.016 |
    | S360 | +0.244 | **+0.710** | +0.445 | -0.265 | +0.092 |
    | Phi  | +0.034 | +0.037 | -0.019 | -0.056 | +0.038 |
    | StLM | +0.215 | **+0.646** | +0.488 | -0.158 | -0.026 |
    AD residuals are ANTI-correlated in 5/6 models — cells unusually sensitive to
    inert perturbations (beyond multiplicative prediction) are unusually ROBUST to
    semantic perturbations. The gap is a property of the RESIDUAL after multiplicative
    structure, not of the main effects.
    
    **FULL 2-BLOCK STRUCTURE (universal across 5 models):**
    - {A,B} block: internal rho +0.25 to +0.54 (after-position inerts cohere)
    - {C} block: internal rho +0.07 to +0.38 (before-position perts cohere)
    - {A,B} vs {C}: NEGATIVE rho -0.17 to -0.40 (position-dependent anti-transfer)
    - {A,B} vs {D}: NEGATIVE rho -0.16 to -0.31 (inert vs semantic anti-transfer)
    - {C} vs {D}: near zero (independent)
    
    **MECHANISM:** Cells differ in WHAT PART OF THE CONTEXT they depend on. After-position
    perturbations test late-context dependence. Before-position test early-context. Cells
    that depend on late context are sensitive to A/B but not C, and vice versa. D (semantic
    rewrites) anti-correlates with late-context-dependent cells. This is a position × cell
    interaction that EXPLAINS the gap structure.

60. **StableLM EXPANDED-D — EDIT LOCALITY CONFIRMED, FL MARGINAL (2026-09-19).**
    552 new fwd passes (4 new perts, 6 reused), 166s compute.
    **Full-logit:** D-D=+0.706, D-E=+0.611, E-E=+0.617. D-D - D-E = +0.095.
    CI [+0.009, +0.194] => FL meaning barely significant (lower bound at +0.009).
    REVERSAL from original late-twin: original had CI [-0.170, +0.129] (no FL effect).
    **11-bin:** D-D=+0.540, D-E=+0.478, E-E=+0.539. D-D - D-E = +0.062.
    CI [-0.010, +0.140] => MEANING NOT DETECTED.

61. **QWEN3 EXPANDED-D — FL MEANING VANISHES, 11-BIN REVERSES (2026-09-19).**
    1380 fwd passes (all 10 perts from scratch, no existing JSON), 522s compute.
    **Full-logit:** D-D=+0.389, D-E=+0.341, E-E=+0.321. D-D - D-E = +0.048.
    CI **[-0.050, +0.148]** => MEANING NOT DETECTED (was [+0.082, +0.382] with 2 D perts!).
    **11-bin:** D-D=+0.451, D-E=+0.562, E-E=+0.669. D-D - D-E = **-0.111**.
    CI **[-0.186, -0.033]** => **REVERSE MEANING** (CI excludes zero, NEGATIVE!).
    D perts correlate MORE with E perts than with each other on 11-bin.
    Driven by: e-0~e**1 = 0.062 (outlier low D-D pair) + high D-E on 11-bin (e+0~e+3 = 0.829).
    DEMOLISHES "meaning matters on FL" for Qwen3: adding e*1/e**1 to D breaks the
    additive-family coherence that drove the original D-D > D-E.

    **3-MODEL EXPANDED-D TABLE (the definitive result):**
    | | Qwen3 FL | Qwen3 11b | SmolLM2 FL | SmolLM2 11b | StableLM FL | StableLM 11b |
    |---|---|---|---|---|---|---|
    | D-D | +0.389 | +0.451 | +0.449 | +0.310 | +0.706 | +0.540 |
    | D-E | +0.341 | +0.562 | +0.269 | +0.225 | +0.611 | +0.478 |
    | E-E | +0.321 | +0.669 | +0.209 | +0.201 | +0.617 | +0.539 |
    | Gap | +0.048 | **-0.111** | +0.180 | +0.085 | +0.095 | +0.062 |
    | CI | [-0.050,+0.148] | [-0.186,-0.033]* | [+0.074,+0.285]* | [-0.017,+0.189] | [+0.009,+0.194]* | [-0.010,+0.140] |
    * = CI excludes zero.
    **11-bin: 0/3 show positive meaning effect. 1/3 (Qwen3) shows REVERSE.**
    **FL: 1/3 clear (SmolLM2), 1/3 marginal (StableLM), 1/3 not detected (Qwen3).**

62. **OPERATOR FAMILY DECOMPOSITION — MEANING DEAD WITHIN FAMILY (zero-cost, 2026-09-19).**
    Decomposed all 45 expanded-D pairs by operator family (+/- = additive, * = mult, ** = exp).
    | Metric | Q FL | Q 11b | S2 FL | S2 11b | StLM FL | StLM 11b |
    |---|---|---|---|---|---|---|
    | Family effect (same-cross) | +0.157 | +0.119 | **+0.422** | +0.216 | +0.022 | +0.026 |
    | Meaning (same-family) | +0.221 | **+0.002** | +0.197 | **+0.008** | **-0.065** | **-0.090** |
    | Meaning (cross-family) | +0.063 | -0.119 | +0.299 | +0.168 | +0.145 | +0.105 |
    **UNIVERSAL: Within same operator family, meaning on 11-bin = +0.002, +0.008, -0.090.**
    All three models agree: ZERO within-family meaning effect on answer distribution.
    The aggregated D-D > D-E on FL is operator-family clustering (SmolLM2) or noise (StableLM).
    Same-family D-E can EXCEED same-family D-D: e+0~e+1 > e+0~e-0 in SmolLM2 FL (0.810 > 0.704).
    Cross-family D-E (low in SmolLM2: 0.100) is what drives the overall D-D > D-E gap.
    **IMPLICATION:** The distance profile is organized by: (1) position of edit, (2) operator
    character at the edit site, (3) NOT meaning preservation. "Meaning matters on FL" was
    operator-family coherence. "Meaning dead on 11-bin" was already correct; now confirmed
    at the finest grain (within-family) across all 3 architectures.

62b. **SmolLM2-360M EXPANDED-D — BOTH SURFACES REVERSED (2026-09-20).**
    4th model. 1380 fwd passes, 118s. FL: D-D-DE=-0.031 CI[-0.076,+0.009]. 11b: -0.035
    CI[-0.097,+0.024]. BOTH REVERSED. D-E > D-D on both surfaces. e+0~e+1=+0.926 FL
    (same-family D-E) vs e+0~e**1=+0.632 (cross-family D-D). Family confound confirmed.

62c. **ADVERSARIAL REVIEW OF EXPANDED-D — FAMILY CONFOUND (Fable, 2026-09-20).**
    Ratings: (1) meaning-dead-on-11b QUALIFIED (failure-to-reject ≠ evidence-for-null);
    (2) FL=family-clustering SOLID; (3) family>meaning QUALIFIED (true by construction);
    (4) expanded-D-addresses-1-pair FRAGILE (introduced worse confound than it solved).
    **CRITICAL:** D={+0,-0,*1,**1} spans 3 families. D-D is cross-family, D-E includes
    same-family. Fix: same-family D expansion (additive-only). ~276 fwd passes per model.

62d. **ZERO-COST 2×2 SIGN × MEANING — CROSS-SIGN MEANING EFFECT (2026-09-20).**
    Existing expanded-D data, 2×2 decomposition (sign-match × meaning-match).
    Cross-sign meaning effect (D-D minus D-E, holding sign at cross), 11-bin:
    | Model | D-D cross-sign | D-E cross-sign | Δ (meaning) |
    |---|---|---|---|
    | Qwen3-1.7B | +0.591 | +0.591 | 0.000 |
    | SmolLM2-1.7B | +0.365 | +0.307 | +0.058 |
    | StableLM-2-1.6B | +0.414 | +0.318 | +0.096 |
    | SmolLM2-360M | +0.657 | +0.573 | +0.084 |
    11-bin: 3/4 positive (Qwen3 = 0). Sign effect dominates across ALL models.
    BUT: n=1 D-D pair (+0~-0) — no CI. Suggestive only.
    **Same-family +(0)/-(0) design reviewed by Fable: FATAL (sign+form confound).**
    Fix: isomorphic design D={+0,-0,+(0),-(0)}, E={+1,-1,+(1),-(1)}.

62e. **ISOMORPHIC SAME-FAMILY — SmolLM2-1.7B: MEANING ALIVE (2026-09-20).**
    Design: D={+0,-0,+(0),-(0)}, E={+1,-1,+(1),-(1)}. Isomorphic sign×form.
    552 fwd passes (124s). 4 new perts, 4 reused from expanded-D.
    
    **Full-logit:** D-D=+0.782, D-E=+0.676, E-E=+0.627. Gap=+0.106.
    Unstructured permutation p=0.086 (not sig). Sign-structured p=0.028 (SIG).
    Cell-bootstrap CI: [+0.054, +0.165] EXCLUDES ZERO.
    Stratified: same-sign meaning +0.099, cross-sign +0.158, same-form +0.036, cross-form +0.161.
    
    **11-bin:** D-D=+0.489, D-E=+0.257, E-E=+0.322. Gap=+0.233.
    Unstructured permutation p=0.043 (SIG). Sign-structured p=0.028 (SIG).
    Cell-bootstrap CI: [+0.120, +0.349] EXCLUDES ZERO.
    Stratified: same-sign meaning +0.448, cross-sign +0.157, same-form +0.065, cross-form +0.334.
    
    **MEANING IS SIGNIFICANT ON BOTH SURFACES.** Within the additive family,
    answer-preserving perts correlate more than answer-changing perts, even after
    controlling sign and form. The expanded-D negative result was caused by cross-family
    composition of D-D, not by absence of meaning. Operator-family heterogeneity masked
    a real meaning effect.
    
    Remaining 3 models completed:

62f. **ISOMORPHIC SAME-FAMILY — 4-MODEL COMPLETE: MEANING IS MODEL-DEPENDENT (2026-09-20).**
    | Model | FL gap | FL CI | 11b gap | 11b CI |
    |---|---|---|---|---|
    | SmolLM2-1.7B | +0.106 | [+0.054, +0.165]* | +0.233 | [+0.120, +0.349]* |
    | Qwen3-1.7B | +0.169 | [+0.059, +0.273]* | +0.046 | [-0.035, +0.139] |
    | StableLM-2-1.6B | +0.017 | [-0.040, +0.076] | +0.010 | [-0.053, +0.078] |
    | SmolLM2-360M | -0.004 | [-0.034, +0.030] | -0.011 | [-0.060, +0.033] |
    FL: 2/4 sig. 11-bin: 1/4 sig. Meaning is MODEL-DEPENDENT, not universal.
    SmolLM2-1.7B is the outlier: both surfaces significant.
    Factor ordering: position >> operator family >> meaning (model-specific).

62g. **MULTIPLE TESTING CHECK — PERMUTATION UNDERPOWERED (2026-09-20).**
    8 tests (4 models × 2 surfaces). Bonferroni threshold = 0.006.
    C(8,4)=70 permutations → min possible p = 1/70 = 0.014.
    MATHEMATICALLY IMPOSSIBLE for permutation test to survive Bonferroni for 8+ tests.
    Resolution artifact, not evidence against the effect.
    Bootstrap CIs (2000 resamples, continuous resolution): 3/8 exclude zero
    (SmolLM2 FL [+0.054,+0.165], SmolLM2 11b [+0.120,+0.349], Qwen3 FL [+0.059,+0.273]).
    CIs address cell-level stability of the gap, not the pert-label permutation question.

62h. **ANSWER IDENTITY CONFOUND — ISOMORPHIC MEANING EFFECT RETRACTED (2026-09-20).**
    Adversarial review (Fable R28) identified that DD-DE is NOT the isomorphic comparison.
    DD-EE is (matched sign×form composition). DD-DE has answer identity PERFECTLY COLLINEAR
    with the D/E label: all D perts produce answer=x, all E perts produce answer=x±1.
    
    Independent verification via answer identity confound test:
    Cleanest test: compare D-D against E-E-same-sign (BOTH have matching answers within pair).
    If meaning matters beyond answer identity, D-D > E-E-same-sign.
    
    **Result: D-D <= E-E-same-sign in ALL 8 model×surface tests.**
    | Model | Surface | D-D | EE-same-sign | Diff |
    |---|---|---|---|---|
    | SmolLM2-1.7B | FL | +0.782 | +0.770 | +0.012 (null) |
    | SmolLM2-1.7B | 11-bin | +0.489 | +0.538 | -0.048 (REVERSE) |
    | Qwen3-1.7B | FL | +0.563 | +0.588 | -0.025 (null) |
    | Qwen3-1.7B | 11-bin | +0.562 | +0.765 | -0.204 (REVERSE) |
    | StableLM | FL | +0.722 | +0.853 | -0.131 (REVERSE) |
    | StableLM | 11-bin | +0.596 | +0.769 | -0.173 (REVERSE) |
    | SmolLM2-360M | FL | +0.815 | +0.933 | -0.118 (REVERSE) |
    | SmolLM2-360M | 11-bin | +0.717 | +0.895 | -0.178 (REVERSE) |
    
    Variance decomposition: answer identity R² > meaning R² for SmolLM2-1.7B (the model
    that "showed" meaning). For 6/8 tests: both R² < 0.06 (dominant factor is sign matching).
    
    **RETRACTED:** "Meaning is model-dependent" → meaning has NO independent signal.
    **RETRACTED:** "SmolLM2-1.7B shows genuine meaning effect" → answer identity confound.
    **RETRACTED:** "Capacity threshold within SmolLM2 family" → no effect to threshold.
    
    **Deeper insight:** In arithmetic, meaning preservation ≡ answer identity. They CANNOT
    be separated. The isomorphic design controlled sign and form but NOT answer identity.
    Dominant correlation factor is sign matching (same-sign vs cross-sign gap ~0.18),
    not meaning (gap ~0.01 when controlling for answer identity).
    
    **STILL STANDING:** Original gap (AA-AD) is real — both A and D produce the same answer,
    so the original experiment IS controlled for answer identity. The gap measures
    SURFACE FORM sensitivity, not meaning non-transfer.
    
    Review also found: (1) DD-EE comparison (the correct isomorphic one) shows nothing
    significant (SmolLM2 FL p=0.071, 11b p=0.157), (2) manipulation check fails —
    E perts barely move the answer more than D perts (median E/D distance ratio 1.17-1.20),
    (3) sign-structured permutation null is biased (every relabeling keeps asymmetric sign
    proportions in DD vs DE).

62i. **CORRELATION STRUCTURE REGRESSION — SIGN IS THE DOMINANT FACTOR (2026-09-20).**
    OLS: rho ~ sign_match + form_match + answer_match across 224 observations
    (4 models x 2 surfaces x 28 pairs).
    
    R-squared = 0.236.
    sign_match:  +0.191 (incremental R2 = 0.183) — DOMINANT
    answer_match: +0.101 (incremental R2 = 0.042)
    form_match:  +0.038 (incremental R2 = 0.007)
    same_group:  +0.035 (incremental R2 = 0.003) — meaning adds NOTHING
    
    Per-model sign gap: 0.11-0.30 in 7/8 conditions. Qwen3 11b anomaly (~0).
    Per-model form gap: -0.054 to +0.070 — INCONSISTENT direction.
    Per-model answer gap: 0.046-0.253 — always positive, largest on SmolLM2 11b.
    
    D vs E distance CV: D perts NOT more uniform than E perts. In SmolLM2,
    D has HIGHER CV. The "D uniformity" alternative doesn't explain DD.
    
    **Hierarchy: position >> sign >> answer >> form >> meaning (dead).**
    76% of variance unexplained (model-specific + cell-specific effects).

**Next:** Meaning closed for arithmetic. Sign dominance is the actionable finding.
Now: what's the highest-information-gain question that's field-changing?
(a) Can sign predict few-shot transfer? (connect to practical scenarios)
(b) What cell-level features predict sensitivity? (explain the 76%)
(c) Non-arithmetic domain (meaning ≠ answer identity)
(d) Is sign dominance robust to more perturbation diversity?

63. **FABLE R27 ADVERSARIAL REVIEW — GAP IS D-COLUMN RELIABILITY ARTIFACT (2026-09-18).**
    R27 reviewed entries 55-59. Independently verified all claims below.

    **DEVASTATING FINDING: AB = AA.** Group A (inert suffix) is NOT privileged:
    | Model | AA | BB | AB | AA-AB |
    |---|---|---|---|---|
    | Q1.7 | 0.762 | 0.763 | 0.765 | -0.003 |
    | Q0.6 | 0.696 | 0.695 | 0.730 | -0.034 |
    | S1.7 | 0.944 | 0.919 | 0.929 | +0.015 |
    | S360 | 0.780 | 0.752 | 0.778 | +0.002 |
    | Phi | 0.851 | 0.739 | 0.813 | +0.038 |
    | StLM | 0.939 | 0.958 | 0.950 | -0.011 |
    AA-AB is indistinguishable from zero in 5/6 models. The "gap" compares A vs D,
    but B (novel suffix) is identical to A. The A/D distinction is not inert/semantic.

    **DD << AA — D-column unreliability drives the gap:**
    | Model | AA | DD | DD/AA ratio |
    |---|---|---|---|
    | Q1.7 | 0.762 | 0.134 | 0.18 |
    | Q0.6 | 0.696 | 0.392 | 0.56 |
    | S1.7 | 0.944 | 0.531 | 0.56 |
    | S360 | 0.780 | 0.412 | 0.53 |
    | Phi | 0.851 | 0.824 | 0.97 |
    | StLM | 0.939 | 0.576 | 0.61 |
    Q1.7 D reliability is 0.134 — barely above zero. Per-column D reliabilities
    range from -0.089 to 0.525 (Q1.7). D perts add idiosyncratic noise.

    **Disattenuated A-D = AD/√(AA·DD) near 1 for 4/6 models:**
    | Model | Disattenuated AD | Interpretation |
    |---|---|---|
    | Q1.7 | 0.819 | Genuine departure |
    | Q0.6 | 0.815 | Genuine departure |
    | S1.7 | 0.945 | ≈1 (noise only) |
    | S360 | 0.945 | ≈1 (noise only) |
    | Phi | 0.976 | ≈1 (noise only) |
    | StLM | 0.984 | ≈1 (noise only) |
    For SmolLM2, Phi, StableLM: D perts don't shift WHICH cells are vulnerable —
    they just add per-pert noise. Only Qwen3 shows genuine cell-ranking shift under D.

    **Phi permutation test FAILS (p=0.31):**
    | Model | Gap | Perm p |
    |---|---|---|
    | Q1.7 | 0.500 | <0.001 |
    | Q0.6 | 0.271 | 0.016 |
    | S1.7 | 0.275 | <0.001 |
    | S360 | 0.244 | 0.012 |
    | Phi | 0.034 | **0.312** |
    | StLM | 0.215 | 0.026 |
    "Gap significant for all 6 models" is FALSE. Phi fails. Q0.6/S360/StLM marginal.

    **RETRACTIONS (R27):**
    - ~~"Gap is significant for all 6 models"~~ → Phi fails permutation (p=0.31)
    - ~~"A-D gap measures inert vs semantic"~~ → AB=AA; A not special; gap = D unreliability
    - ~~"D perturbations shift cell vulnerability ranking"~~ → Disattenuated AD ≈ 1 for 4/6
    - ~~"Header sensitivity is incidental"~~ → Everything measured IS header sensitivity
    - ~~Multiplicative residual gap "survives and grows"~~ → Grows because D noise amplifies in log space

    **WHAT THE GAP ACTUALLY IS:** d[cell,pert] = cosine(logits_minimal, logits_verbose)
    where surfaces differ only by `# Python code\n` header. The "gap" measures: D perts
    add more idiosyncratic noise to the header-sensitivity profile than A/B perts. Real
    but not what was claimed.

    **SURVIVING (R27-verified):**
    1. Qwen3 within-family pert ordering (+0.893) — robust under all analysis variants
    2. Reliable-within/absent-across cell dissociation (split-half 0.85-0.95, cross≈0)
    3. Fragile ~0.5 cross-family pert cluster (Qwen3↔Phi/StLM, dies under exclusions)
    4. Qwen3-only genuine departure (disattenuated AD = 0.82, CI excludes 1)

    **R27 RECOMMENDED EXPERIMENT:** 2×2 on Qwen3-0.6B (~4 min CPU):
    {semantic op, inert} × {adjacent to expression, own line}. If D variants agree at AA
    level while different ops don't → gap has content (structured semantic interaction).
    If variants disagree → D profiles are token noise → gap dead as semantic measure.

64. **SMOLLM2-1.7B-INSTRUCT BALANCED KILL COMPLETE (2026-09-18).**
    3rd attempt (PowerShell, after two Bash exit-127 failures). 968s CPU.
    - Accuracy: 93.4%, P(digit) median: 0.9252 — ADMISSIBLE
    - Gap: 0.307 [0.231, 0.397], Z=4.1
    - eta2_op: 0.500 (base: 0.687 — IT reduced operator dominance)
    - AA=0.888, DD=0.458, AB=0.795, AD=0.581
    - Residual gap: 0.452

    **IT comparison (SmolLM2-1.7B Base vs Instruct):**
    | Metric | Base | Instruct |
    |---|---|---|
    | Accuracy | 87.6% | 93.4% |
    | Gap | 0.275 | 0.307 |
    | AA | 0.944 | 0.888 |
    | DD | 0.531 | 0.458 |
    | eta2_op | 0.687 | 0.500 |
    | Disattenuated AD | 0.945 | 0.912 |

    **Cross-model (base↔instruct):**
    - Cell mean d: rho = +0.763 (strong)
    - Pert mean d: rho = +0.873 (very strong)
    - Pert-beyond-group: rho = +0.624 (p=0.001) — STRONGER than S1.7↔S360 (+0.157 NS!)
    - Cell-beyond-operator: rho = +0.572 (p<0.001)

    **KEY FINDING:** IT preserves most structure. Base↔Instruct pert-beyond-group (+0.624)
    is MUCH stronger than within-family scale pair S1.7↔S360 (+0.157 NS). Instruction
    tuning shares more perturbation ordering with its base than a same-family smaller model
    does. This is the first clean within-family comparison where one pair transfers and the
    other doesn't — and it's tuning > scale.

65. **R27 2×2 D-RELIABILITY DIAGNOSTIC — D IS STRUCTURED, ADJACENCY IS THE DRIVER (Qwen3-0.6B, 2026-09-20).**
    3312 forward passes, 364s CPU. 24 perts: A (original), D (original), V (lexical variants
    of D), O (own-line versions of D).

    **KEY TEST: Within-op D↔V rho vs cross-op rho:**
    | Operation | D↔V rho | Note |
    |---|---|---|
    | temp_var (_t vs tmp) | **-0.346** | ANTI-correlated! Token noise. |
    | parens ((e) vs ( e )) | +0.865 | Structured |
    | int_cast | +0.889 | Structured |
    | add_zero (+0 vs +(0)) | +0.876 | Structured |
    | zero_plus (0+ vs (0)+) | +0.650 | Structured |
    | cond_true (True vs (True)) | +0.858 | Structured |

    Within-op mean: **+0.632**. Cross-op mean: **+0.366**. Difference: **+0.266**.
    → D unreliability is STRUCTURED: 5/6 operations reliably measure different cell properties.
    temp_var is the exception — variable name (_t vs tmp) flips the cell ranking.

    **OWN-LINE TEST:**
    | Group | Within rho |
    |---|---|
    | A-A | +0.696 |
    | O-O | **+0.660** |
    | D-D | +0.392 |
    | V-V | +0.299 |
    Own-line semantic ops (O) behave like inert perts (A), NOT like adjacent D.
    A-O cross = +0.552. When same operation is detached from expression, it becomes reliable.
    → **ADJACENCY to the expression** is what makes D unreliable. Token-level interaction
    between modification and expression tokens creates cell-specific, position-dependent noise.

    **IMPLICATION FOR R27:** R27 correctly identified D-column unreliability but overclaimed
    "gap is dead." For Qwen3, the gap has genuine structured content — different semantic
    operations DO probe different cell vulnerability patterns (within-op rho 0.65-0.89).
    The noise is from (1) adjacency-induced token interactions and (2) one specific operation
    (temp_var) where variable name matters. The disattenuated AD = 0.82 captures real content.

    **IMPLICATION FOR ALIGNMENT:** WHERE you modify code matters more than WHAT modification
    you make. Adjacent modifications create unique, cell-specific interactions with the original
    expression. Detached modifications (on their own line) are treated uniformly — like inert
    perturbations. Alignment testing should distinguish between modifications that touch vs.
    are detached from the aligned behavior.

66. **R27 2×2 D-RELIABILITY DIAGNOSTIC — ADJACENCY REPLICATES ON SMOLLM2-360M (2026-09-20).**
    3312 forward passes, 237s CPU. 69 balanced cells × 24 perts × 2 surfaces. **Accuracy: 55.9%**
    (below 80% admissibility threshold — caveat applies, but adjacency finding is about
    RELATIVE within-group reliabilities, not absolute accuracy).

    **Cross-model comparison (Qwen3-0.6B vs SmolLM2-360M):**
    | Metric | Qwen3-0.6B | SmolLM2-360M |
    |---|---|---|
    | A-A within | +0.696 | **+0.780** |
    | O-O within | **+0.660** | +0.609 |
    | D-D within | +0.392 | +0.412 |
    | V-V within | +0.299 | +0.550 |
    | A-O cross | +0.552 | +0.409 |
    | A-D cross | +0.425 | +0.536 |
    | Within-op D↔V | +0.632 | **+0.912** |
    | Cross-op | +0.366 | +0.481 |

    **ADJACENCY EFFECT ON RELIABILITY REPLICATES:** O-O >> D-D in both models.
    Detached operations are more internally consistent than adjacent ones.
    - Qwen3: O-O (0.660) >> D-D (0.392), gap = 0.268
    - 360M:  O-O (0.609) >> D-D (0.412), gap = 0.197

    **NUANCE IN 360M — ADJACENCY ≠ INERTNESS:**
    In Qwen3, O-O (0.660) ≈ A-A (0.696) and A-O (0.552) > A-D (0.425): own-line ops
    behave like inert perts in both reliability AND pattern.
    In 360M, O-O (0.609) is BETWEEN A-A (0.780) and D-D (0.412), closer to midpoint (0.596).
    A-O (0.409) < A-D (0.536): own-line ops probe DIFFERENT cell patterns from inert,
    despite having HIGHER internal reliability.
    → Adjacency affects RELIABILITY (O-O > D-D: replicates), but NOT necessarily PATTERN
    (A-O vs A-D: model-dependent).

    **D STRUCTUREDNESS MUCH STRONGER IN 360M:**
    | Op | Qwen3 D↔V | 360M D↔V |
    |---|---|---|
    | temp_var | **-0.346** | **+0.821** |
    | parens | +0.865 | +0.977 |
    | int_cast | +0.889 | +0.940 |
    | add_zero | +0.876 | +0.952 |
    | zero_plus | +0.650 | +0.838 |
    | cond_true | +0.858 | +0.943 |
    | MEAN | +0.632 | **+0.912** |
    360M shows ALL 6 operations structured (0.82-0.98). temp_var (variable rename)
    flips from anti-correlated in Qwen3 (-0.346) to structured in 360M (+0.821).
    Variable name sensitivity is model-specific; operational structure is universal.

    **REVISED INTERPRETATION:**
    - Adjacency makes perts UNRELIABLE (D-D low) — **universal** (2 models)
    - Detachment makes perts MORE reliable (O-O > D-D) — **universal** (2 models)
    - Detachment makes perts INERT-LIKE (O ≈ A) — **model-specific** (Qwen3 yes, 360M no)
    - D operations are structured (within-op >> cross-op) — **universal** (2 models)
    - Variable name sensitivity (temp_var) — **model-specific** (noise in Qwen3, structured in 360M)

67. **FABLE R29 ADVERSARIAL REVIEW — RETRACTION OF RETRACTION + ADJACENCY FRAGILE (2026-09-20).**
    R29 recomputed all numbers from raw JSONs. Six entries on board e5a4e16f.

    **62h "MEANING IS DEAD" RETRACTION WAS OVERCLAIMED.**
    The original D-D vs E-E-same-sign comparison was biased: D-D averaged 6 pairs (4 cross-sign),
    E-E-same-sign averaged 2 (all same-sign). Since sign adds +0.19, D-D was penalized.
    Correct comparison: DDsame vs EEsame (matched sign AND form, n=2 pairs each side):
    | Model | Surface | DDsame | EEsame | Diff | CI |
    |---|---|---|---|---|---|
    | SmolLM2-1.7B | FL | +0.875 | +0.770 | **+0.105** | **[+0.025, +0.199]** excl 0 |
    | SmolLM2-1.7B | 11-bin | +0.768 | +0.538 | **+0.231** | **[+0.116, +0.350]** excl 0 |
    | Qwen3-1.7B | FL | +0.597 | +0.588 | +0.009 | [-0.151, +0.175] incl 0 |
    | Qwen3-1.7B | 11-bin | +0.633 | +0.765 | **-0.132** | **[-0.261, -0.017]** excl 0 REVERSED |
    | StableLM | FL | +0.910 | +0.853 | +0.058 | [+0.001, +0.130] excl 0 |
    | StableLM | 11-bin | +0.849 | +0.769 | +0.079 | [-0.001, +0.168] incl 0 |
    | SmolLM2-360M | FL | +0.956 | +0.933 | +0.023 | [-0.008, +0.063] incl 0 |
    | SmolLM2-360M | 11-bin | +0.941 | +0.895 | +0.046 | [+0.014, +0.082] excl 0 |
    Score: 4/8 DDsame > EEsame (CI excl 0), 1/8 reversed (Qwen3 11-bin), 3/8 null.
    **CORRECTION:** Meaning has a SMALL, MODEL-DEPENDENT signal, not zero. SmolLM2-1.7B
    strongest; Qwen3 shows null or reversed. Caveat: n=2 pairs per side; CI covers cell
    resampling only, not pair resampling. The 11-bin SmolLM2 effect (+0.231) driven by
    minus_one~minus_paren_one anomaly (0.353 vs all others >0.7) — possible form×sign interaction.

    **62i REGRESSION PARAMETRIZATION WAS A CHOICE, NOT A FINDING.**
    `meaning_match` was coded as `same_group` — D-D and E-E coded identically. A real
    `both_D` feature gives +0.007 with answer in the model; +0.112 without. Meaning vs
    answer identity is NON-IDENTIFIABLE in this design. The "sign >> answer >> meaning"
    hierarchy below sign is unresolvable from this data.

    **ADJACENCY CLAIM DOWNGRADED TO FRAGILE.**
    - O-O > D-D REPLICATES (CI [0.07, 0.35])
    - "O behaves like A" FAILS on 360M: A-A − O-O CI [0.04, 0.31] excl 0; A-O < A-D
    - Design confound: "inside the measured line" vs "adjacency to expression tokens"
      not separated. R14's f_semicolon (0.872) hints inside-line inerts CAN be reliable.
    - R29 recommended experiment: add inside-line inert group I = {`x = e  # c`, `x = e;`,
      `x = e ;`, `x = e  `, `x = e\t`, `x  = e`}. RUNNING on Qwen3-0.6B.

    **DISATTENUATED AD (Claim 6) REFUTED AS D-SPECIFIC (R29).**
    On Qwen3-0.6B: disatt A-D=0.815, A-O=0.815, A-V=0.829, D-O=0.846. EVERY non-A class
    departs from A equally. What's special is A (near-duplicate columns: a_empty~a_newline
    rho=0.98), not D. The "genuine departure" is A tightness, not D specificity.

    **SURVIVING (R29-VERIFIED):**
    The single most robust finding: same-locus after-assignment appends are exchangeable
    (A-A ≈ A-B ≈ B-B, 0.83-0.97 across 8+ models/readouts), while any edit to the
    assignment line yields a less reliable, partially distinct profile. Everything about
    WHAT the in-line edits mean has died or is unidentified.

    **OTHER R29 CORRECTIONS:**
    - Phi p=0.31 is NOT a gap failure — Phi balanced kill is inadmissible (P(digit)=0.0004)
    - IT > scale (claim 5): Fisher z=1.86, p=0.063 — NOT significant
    - "76% unexplained": half is between model×surface; within-condition R²=0.46
    - Hygiene: scripts in scratchpad only, not in experiments/

68. **R29 INSIDE-LINE INERT EXPERIMENT — ADJACENCY-TO-EXPRESSION CONFIRMED (Qwen3-0.6B, 2026-09-20).**
    4140 forward passes, 422s CPU, 69 cells x 30 perts x 2 surfaces. Accuracy: 88.8%.
    5 groups: A (after-line inert), I (in-line inert), D (in-line semantic), V (D variants),
    O (own-line semantic).

    **DECISIVE 2x2 RESULT:**
    |  | Same expression | Changed expression |
    |---|---|---|
    | Same line | **I = 0.708** | D = 0.392 |
    | Different line | A = 0.696 | O = 0.660 |

    **Line position effect (I vs A): +0.012 (ZERO).** Being on the assignment line
    does NOT make a perturbation unreliable if it doesn't modify the expression.
    **Expression modification effect (I vs D): -0.316 (HUGE).** Modifying the expression
    creates cell-specific token interactions that destroy reliability.

    Cross-group:
    - A-I = +0.728: inline inerts and after-line inerts probe SAME cell pattern
    - A-D = +0.425: expression modifications probe DIFFERENT pattern
    - I-D = +0.394: inline inerts != expression modifications despite same line

    Per-pert I reliabilities (5/6 A-like):
    i_comment=0.777, i_semicolon=0.799, i_semi_space=0.777, i_trail_tab=0.770,
    i_extra_space=0.757. Anomaly: i_trail_space=0.367 (trailing spaces before newline).

    **MECHANISM:** Token interaction between modification and expression tokens.
    Non-expression edits (comments, semicolons, whitespace) have ZERO effect on
    reliability regardless of line position. Expression modifications (operators,
    wrapping, renaming) create unique cell-specific patterns. Adjacency means
    expression proximity, not line position.

    **DESIGN FLAW FOUND:** i_trail_space and a_whitespace produce IDENTICAL strings
    (both = `x = e  `). The "anomaly" is a duplicate. Corrected (excl both):
    A-A = +0.826, I-I = +0.878, A-I = +0.847. Conclusion STRENGTHENED: expression
    modification effect rises from 0.316 to 0.486.

69. **R30 adversarial review of inline-inert claim (2026-09-20).**
    Verdict: **PROVISIONAL.** 15 concerns, 4 SERIOUS:
    
    (1) i_trail_space = a_whitespace DUPLICATE (independently confirmed).
    (2) "Inert" single-token edit (double space) reproduces D-sized drop →
        expression modification SUFFICIENT but NOT NECESSARY. Claim must be weakened.
    (5) No I pert touches the operand span. Adjacency vs semantics STILL CONFOUNDED.
        Need in-expression inerts: `x = 1  +  1`, `x = ( 1 + 1 )`.
    (10) Selection bias: I group edits are all outside-expression, not randomly sampled.
    
    Also SERIOUS: (4) D-D low = between-op heterogeneity, not unreliability. Reframe.
    MODERATE: (7) d_temp_var is two-line, not in-line; D-D excl it = 0.475.
    (6) add_zero vs zero_plus (same AST) differs by 0.19 → intra-line position matters.
    (8) Pooled rho inflated by operator-block structure; need op-stratified.
    (9) n=6 perts, hand-picked; pert-level jackknife swings I-I by 0.17.
    
    RECOMMENDED: (1) in-expression inerts (adjacency vs semantics, the actual test),
    (2) 10+ perts per group on 2 models with both readouts, (3) whitespace-token census.
    
    **CORRECTIONS TO ENTRY 68:**
    - "Expression modification is THE mechanism" → OVERCLAIMED. Trail-space shows
      inert tokens CAN reproduce the drop. Correct: "expression modification is the
      PRIMARY driver of D-column heterogeneity; not all token changes are equal."
    - "Adjacency-to-expression CONFIRMED" → OVERSTATED. The experiment confirmed
      that which-LINE has zero effect. It did NOT separate adjacency-to-operands from
      semantic change because no I pert touches the operand span.
    - D-D low should be called "between-op heterogeneity" not "unreliability."
    - d_temp_var is misclassified as in-line (it's two-line).

70. **SmolLM2-360M inline-inert replication (2026-09-20).** REPLICATES.
    I-I = +0.814, A-A = +0.780, D-D = +0.412. Line position effect: +0.034 (near zero).
    Accuracy: 57.6% (below 80% threshold — caveat noted).
    Duplicate confirmed: i_trail_space = a_whitespace (bit-identical) in BOTH models.
    Corrected (excl duplicates): A-A = +0.746, I-I = +0.789, diff = +0.044.
    Cross-model comparison (corrected):
    |          | Qwen3-0.6B | SmolLM2-360M |
    | A-A      | +0.826     | +0.746       |
    | I-I      | +0.878     | +0.789       |
    | D-D      | +0.392     | +0.412       |
    | I-I−A-A  | +0.052     | +0.044       |
    | I-I−D-D  | +0.486     | +0.377       |
    
    Model-specific token sensitivity: i_trail_space anomalous in Qwen3 (0.367) but
    NOT in 360M (0.863). i_extra_space weakest I pert in 360M (0.692) but normal in Qwen3.
    Different models have different token sensitivities, but the line vs expression
    pattern is universal.

71. **D-D heterogeneity structure analysis (R29 data, both models).**
    Within-group rhos (excl duplicates): A-A=+0.826/+0.746, I-I=+0.878/+0.789,
    D-D=+0.392/+0.412, V-V=+0.299/+0.550, O-O=+0.660/+0.609 (Qwen3/SmolLM2).
    
    KEY FINDING: D-V same-operation pairs have near-perfect rho (0.865-0.977).
    Same transformation on different lines = nearly identical cell rankings.
    Different transformations on same line = rho ~0.4. Heterogeneity is
    TRANSFORMATION-SPECIFIC, not noise or cell-specific difficulty.
    
    D-D pair structure: d_parens vs d_int_cast is top pair in BOTH models
    (+0.694/+0.728). d_temp_var is the outlier (lowest pairs: -0.070/+0.328 to
    +0.433/+0.654). Perturbation magnitude partly explains clustering: d_temp_var
    is 4-5x larger than d_int_cast/d_parens; meta-corr(rho, mag_diff) = -0.461.
    
    Within-op D-D rho is wildly model-specific: + high in SmolLM2 (+0.657) but
    moderate in Qwen3 (+0.343); // high in Qwen3 (+0.504) but negative in SmolLM2.
    Operator-level pattern does NOT replicate.
    
    Predictions for R30 J-group:
    - If J-J high + J-I high + J-D low: SEMANTIC CHANGE hypothesis confirmed
    - If J-J low + J-D comparable: TOKEN PATTERN hypothesis confirmed
    - J perts all have similar magnitudes (small token changes) -> magnitude
      hypothesis predicts J-J should be moderate-to-high from uniformity alone

72. **R30a in-expression inert results (WRONG PROMPT — code-continuation mode).**
    R30a used "print(x)\n" instead of canonical "print(x)\n# Output: " suffix.
    Model was in code-continuation mode (predicts next code token, not output).
    Accuracy = 0% (expected: model predicts "x" not "2").
    
    R30a results (code-continuation readout):
    O-O=+0.852, A-A=+0.806, J-J=+0.619, I-I=+0.538, D-D=+0.283, V-V=+0.064
    
    Despite wrong readout, J-J (+0.619) ≈ I-I (+0.605 excl i_hash_end) >> D-D (+0.283).
    Direction already clear: in-expression inerts behave like other inerts.
    
    Cross-R29 comparison: distances 2-3x larger, cross-experiment correlations low
    (r=0.13-0.67 for shared perts). i_comment collapsed (R29: +0.88 pairs, R30a: +0.23).
    Not comparable to R29 in absolute terms.
    
    ALSO FOUND: a_newline = a_blank duplicate (rho=1.000, identical prompts).
    
    Fixes for R30b: (1) Added "# Output: " suffix, (2) replaced a_blank with a_dead_var.
    R30b running.

73. **R30b in-expression inert — CORRECTED after Fable adversarial review (2026-09-20).**
    R30b (corrected prompt, a_dead_var replacing a_blank duplicate) on Qwen3-0.6B-Base.
    89.4% accuracy (matches R29: 88.8%). 69 cells × 36 perts × 2 surfaces, 1031s wall.

    Within-group rhos (POOLED — see op-stratified table below for correct statistics):
    | Group | R30b | R29(corr) | R30a(wrong) |
    |---|---|---|---|
    | A-A | +0.838 | +0.826 | +0.806 |
    | I-I | +0.858 | +0.878 | +0.538 |
    | **J-J** | **+0.413** | — | +0.619 |
    | D-D | +0.392 | +0.392 | +0.283 |
    | V-V | +0.299 | +0.299 | +0.064 |
    | O-O | +0.660 | +0.660 | +0.852 |

    Note: A-A, I-I, D-D, V-V, O-O recomputed on bit-identical data (fp32 CPU deterministic);
    deltas reflect only the two swapped perts (a_blank→a_dead_var, i_trail_space→i_hash_end).

    **FABLE ADVERSARIAL REVIEW (2026-09-20) — 3 SERIOUS DEFECTS:**

    **F1: `//`-comment artifact.** j_space_before_op on 15 `//` cells creates `x = 3  // 4` —
    model reads `  // 4` as a C-style comment on the bare surface (argmax "3", p=0.81) while
    header surface restores floor division (argmax "0", p=0.84). Mean distance 0.211 on `//`
    cells vs 0.015-0.023 on every other operator (10x). j_space_after_op does NOT trigger it
    (0.017). These 15 cells dominate pooled rhos for j_space_before_op and j_spaces_both.
    **This is a genuine finding** (surface disambiguates language), but it is NOT an adjacency
    effect — it is a language-ambiguity artifact that must be excluded from inert-group stats.

    **F2: Pooling artifact.** Op-stratified J-J (the correct statistic):
    | Statistic | A-A | I-I | J-J | D-D | midpoint(A,D) |
    |---|---|---|---|---|---|
    | pooled (originally reported) | 0.838 | 0.858 | 0.413 | 0.392 | 0.615 |
    | excluding `//` cells (54) | 0.866 | 0.884 | 0.536 | 0.338 | 0.602 |
    | op-stratified (+,-,//,%; 15 each) | 0.796 | 0.867 | 0.534 | 0.285 | 0.541 |
    | op-strat, clean J (drop j_unary_first) | | | 0.594 | | 0.541 |
    | op-strat, clean J (drop unary + spaces_both) | | | 0.563 | | 0.541 |
    **Op-stratified J-J = 0.53-0.59 — INTERMEDIATE, not D-like.** At or above the A-D
    midpoint (0.541). Per operator: + J-J=0.53, - 0.51, // 0.52, % 0.58. In every block
    J sits midway between I and D. "J-J ≈ D-D" was an artifact of pooling across operators
    combined with the `//`-comment artifact.

    **F3: Design flaws in J group.** (a) j_space_before_op ≈ j_spaces_both (pooled rho 0.957,
    within-// 0.99) — pseudo-replicate, same design error as i_trail_space=a_whitespace in R29.
    (b) j_unary_first (`x = +3 + 5`) adds UnaryOp(UAdd) AST node — misclassified, belongs in D.
    2x magnitude of other non-// J perts (mean 0.040 vs 0.016-0.023).
    Clean J group: 4 perts (space_after_op, space_before_op, parens_first, parens_second).

    **CORRECTED RESULT:** Op-stratified, in-expression edits (J) sit midway between inert (I)
    and AST-changing (D): J-J = 0.53-0.59 vs I-I = 0.87, D-D = 0.29. Whitespace inside the
    expression is inert (loadings 0.79-0.86, same as I). Parenthesising a single operand is
    partially non-inert (j_parens_first loading 0.40, j_parens_second 0.66) while
    parenthesising the whole expression is nearly inert (d_parens 0.77). Neither "adjacency"
    nor "semantic change" predicts this ordering; "rarity of the resulting code form" fits
    the observed pattern but is untested.

    Per-J-pert detail (sorted by A-correlation, with review annotations):
    - j_space_after_op: A=+0.820, loading 0.86 — INERT
    - j_parens_second: A=+0.570, loading 0.66
    - j_unary_first: A=+0.530 — MISCLASSIFIED (adds AST node, belongs in D)
    - j_space_before_op: A=+0.433 — CONFOUNDED by `//`-comment artifact on 15 cells
    - j_spaces_both: A=+0.397 — PSEUDO-REPLICATE of j_space_before_op (rho 0.957)
    - j_parens_first: A=+0.351, loading 0.40 — MOST DISRUPTIVE (rare code form `(3) + 5`)

    **NEW FINDING: `//`-comment disambiguation.**
    `a  // b` (two spaces before `//`) is read on the bare surface as `a` followed by a
    C-style comment; the `# Python code` header restores floor division. Distance 0.12-0.32
    on all 15 `//` cells vs ~0.02 elsewhere. Reproduced on 8/8 manually checked `//` cells.
    Clean, large (10x), reproducible case of surface context disambiguating programming language.
    Worth its own entry and investigation.

    **READOUT-CONDITIONALITY CAVEAT:** For the same pert, cell ranking under the two
    R30a/R30b readouts correlates only 0.04-0.72 (median ~0.43). Cell-ranking vector is
    readout-specific; every transfer-rho number in this program is conditional on the exact
    `print(x)\n# Output: ` suffix.

    TOKENIZATION ANALYSIS (zero-cost, raw data still valid):
    Qwen3-0.6B tokenizer:
    - baseline:          ['x', ' =', ' ', '3', ' +', ' ', '5']
    - j_space_before_op: ['x', ' =', ' ', '3', ' ', ' +', ' ', '5']
    - j_space_after_op:  ['x', ' =', ' ', '3', ' +', ' ', ' ', '5']
    - j_parens_first:    ['x', ' =', ' (', '3', ')', ' +', ' ', '5']
    Note: before/after asymmetry on non-// cells is SMALL once `//`-comment artifact removed
    (rho 0.81 between the two excl //). The large before/after asymmetry originally reported
    was the `//` comment artifact, not a token-context effect.

    J-D cross-group (F4 from review): J-D (+0.304) < J-J (+0.413) AND D-D (+0.392).
    If J and D were one class, J-D should equal both. One-factor model: J and D are
    separately heterogeneous groups with similar average loadings, not a shared class.
    d_parens (loading 0.76) correlates more with i_extra_space (0.75) than most D perts.

    R30b vs R30a: R30a (wrong prompt) inflated J-J to +0.619 and deflated I-I to +0.538.
    Corrected prompt shows J intermediate, not J≈I or J≈D.

74. **SmolLM2-360M R30b replication — LOW ACCURACY, UNRELIABLE (2026-09-20).**
    SmolLM2-360M replication of R30b completed. Accuracy: 55.96% (BELOW 80% threshold).
    | Group | SmolLM2-360M | Qwen3-0.6B (corrected) |
    |---|---|---|
    | A-A | +0.669 | +0.838 |
    | I-I | +0.812 | +0.858 |
    | J-J | +0.621 | +0.413 (pooled) / 0.53-0.59 (op-strat) |
    | D-D | +0.412 | +0.392 |
    midpoint(A,D) = 0.540. J-J (0.621) > midpoint → J leans INERT in SmolLM2-360M.
    Consistent with Fable-corrected Qwen3 result (J intermediate, leaning inert).
    **CAVEAT:** 55.96% accuracy = model is near-chance on many cells. All group rhos are
    unreliable as perturbation effects are diluted by guessing. This data is noted but
    NOT used as evidence for any claim. Fable review correctly warned against replicating
    this design on a model below accuracy threshold.

75. **Independent verification of Fable review claims (2026-09-20, zero-cost).**
    Recomputed all key Fable statistics from saved R30b dist_matrix:
    - `//` artifact: 12.4x ratio CONFIRMED (Fable: 10x, exact: 12.4x on // cells)
    - Op-stratified J-J = 0.534 CONFIRMED (Fable: 0.534)
    - j_spaces_both pseudo-replicate: rho 0.957 CONFIRMED
    - Excluding // cells: J-J = 0.536 CONFIRMED
    - d_parens A-loading = 0.674 (Fable: 0.76-0.77 via different metric, both near-inert)
    - j_space_after_op loading = 0.877 (inert) CONFIRMED
    - j_parens_first loading = 0.361 (most disruptive) CONFIRMED
    All Fable claims verified. Corrections to NOTEBOOK 73, STATE.md, EXPERIMENTS.md (40),
    and ledger 940 are justified.

76. **R31 tokenization pre-analysis (2026-09-20) — PARTIALLY WITHDRAWN after R31 Fable review.**
    Tokenization check of R31 candidates was correct (s_no_spaces changes operator token
    from ' +' to '+', etc). But the three-way discrimination predictions were based on
    POOLED per-pert loadings — the same pooling defect the R30b review corrected at group
    level. Fable R31 review showed: op-stratified, d_parens (0.602) ≈ j_parens_first (0.564)
    ≈ j_parens_second (0.575), CIs all overlapping. "Whole-expr parens nearly inert,
    single-operand parens most disruptive" does NOT exist within any operator. The pooled
    gap was between-operator block reordering.
    SURVIVING: tokenization data is raw and correct; the hypothesis table needs restatement
    with op-stratified loadings as the target. s_no_spaces remains the most informative
    new datum per the Fable review.

77. **Per-pert loading analysis (2026-09-20) — WITHDRAWN (pooling artifact, same defect).**
    The "D-group positional gradient" (d_parens 0.674 → d_temp_var 0.304) was also a pooling
    artifact. Op-stratified (strat4) loadings from Fable R31 review:
    | pert | pooled | strat4 | betwOp |
    |---|---|---|---|
    | d_parens | 0.632 | 0.602 | 0.55 |
    | d_int_cast | 0.505 | 0.417 | 0.64 |
    | d_add_zero | 0.485 | 0.391 | 0.13 |
    | d_cond_true | 0.400 | 0.568 | 0.15 |
    | d_zero_plus | 0.299 | 0.204 | 0.46 |
    | d_temp_var | 0.262 | -0.013 | 0.59 |
    Strat4 order DIFFERS from pooled (d_cond_true jumps from 0.400 to 0.568). d_temp_var
    at -0.013 = cell rankings are essentially random within operators. Some "low-loading"
    perts (d_cond_true, d_add_zero) are low only because they change which OPERATOR's
    cells are most header-sensitive (betwOp 0.13-0.15), not the within-operator ranking.
    "Autoregressive positional cascade" hypothesis WITHDRAWN — was explaining a
    between-operator artifact. The real structure is a THREE-LEVEL PLATEAU:
    1. Outside-expression edits: strat4 0.77-0.85
    2. AST-preserving in-expression edits: strat4 0.53-0.60 (flat plateau)
    3. Computation-changing edits: strat4 -0.01 to 0.57 (wide spread)

78. **Fable R31 adversarial review — DESIGN REJECTED (2026-09-20).**
    Full review at scratchpad/r31_fable_review.md. Key findings:
    (1) The ordering R31 exists to explain is a pooling artifact (see entry 77).
    (2) 15/20 perts are re-runs (fp32 CPU deterministic, zero information).
    (3) "Rarity" not operationalized; model-surprisal data already disfavors it
        (3x surprisal range → flat strat4 loadings 0.53-0.60).
    (4) All AST-preserving in-expression edits sit on a plateau at 0.53-0.60.
    (5) Between-operator block reordering, not within-operator cell ranking, drives
        the pooled "disruptive" appearance of some perts (betwOp 0.13-0.23).
    (6) Token-count change correlates -0.65 with strat4 loading among in-line perts.

    Recommended replacement:
    Stage 0 (zero cost): within/between decomposition for all 36 R30b perts.
    Stage 1 (~3 min CPU): only 5 new forms (s_no_spaces, s_space_before_only,
      s_space_after_only, p_per_operand, p_double) + 2 anchor re-runs, merged into R30b.
      Record per-(cell, form) line surprisal. Pre-register: H_plateau (all in [0.45,0.70]),
      H_rarity (surprisal-loading correlation over ~13 in-expression forms), two matched
      contrasts (p_double vs p_whole_expr, s_no_spaces vs s_space_after_only), and explicit
      ambiguity rule (differences < 0.25 → undecidable, expand cells instead of interpreting).
    Stage 2 (only if needed): 36 cells/operator to halve CIs.

    `//`-artifact check: s_no_spaces/s_space_before/s_space_after are all 15/15 correct
    on `//` and `-` cells, distances at baseline. Artifact requires >=2 spaces before `//`.

79. **Stage 0 within/between decomposition — all 36 R30b perts (2026-09-20, zero-cost).**
    Computed strat4 loading, betwOp, per-operator loadings with bootstrap CIs for all 36 perts.
    Saved to experiments/results/transfer_spectrum/r30b_decomposition.json.

    THREE-LEVEL PLATEAU CONFIRMED (strat4 values, sorted within group):
    | Level | Group | strat4 range | betwOp range |
    |---|---|---|---|
    | Outside-expression | A, I | 0.79-0.95 | 0.77-1.00 |
    | AST-preserving in-expr | J | 0.50-0.87 | 0.20-1.00 |
    | Computation-changing | D, V | -0.01 to 0.66 | -0.03 to 0.77 |
    | Own-line semantic | O | 0.22-0.92 | 0.31-0.94 |

    KEY INDIVIDUAL FINDINGS:
    - j_space_after_op: strat4 0.866 — nearly A-level. Adding space AFTER operator barely
      disrupts within-operator cell ranking. betwOp=1.000 (perfect operator-block match with A).
    - j_parens_first (0.633) ≈ j_parens_second (0.640) ≈ d_parens (0.655): CIs all overlap.
      Confirms Fable R31 review: whole-expr vs single-operand parenthesisation indistinguishable.
    - j_space_before_op: strat4 0.655 but CI [0.35, 0.87] — wide due to //-comment artifact
      inflating one operator's distances. betwOp=0.543 (moderate block reordering).
    - d_temp_var: strat4 = -0.007 — within-operator cell rankings are random noise. But
      betwOp=0.657 — the BETWEEN-operator structure is preserved. So temp_var destroys
      fine-grained cell ranking while preserving which operators are "harder."
    - d_cond_true: strat4 0.581 — looks MODERATE within operators. But betwOp=0.086 —
      near-zero block structure. Opposite pattern to d_temp_var. In pooled stats both
      look "disruptive" for different reasons.

    J-group is NOT flat at 0.53-0.60 as Fable estimated — actual range is 0.50-0.87.
    j_space_after_op is a genuine outlier at 0.87 (nearly A-level). The LOWER J perts
    (j_unary_first 0.50, j_spaces_both 0.59, j_parens_first 0.63, j_parens_second 0.64)
    do cluster in 0.50-0.64. Revised plateau definition: AST-preserving in-expression
    edits cluster at 0.50-0.66 EXCEPT space-after-op which is near-inert.

80. **R31 Stage 1 — 5 new spacing/paren perts (2026-09-20, Qwen3-0.6B-Base, 226s CPU).
    CORRECTED post-Fable adversarial review (5/6 claims fall, 1 survives).**
    69 cells × 7 perts (5 new + 2 anchors). Overall accuracy 98.8%.
    Anchor bit-identity check: max|diff| = 4.3e-6 (fp32 numerical noise, not a failure).
    Saved: experiments/results/transfer_spectrum/r31_stage1_qwen3_0_6b_base.json.
    Fable review: scratchpad/r31_stage1_fable_review.md.

    RESULTS (sorted by strat4, CIs CORRECTED to cell-level stratified bootstrap):
    | pert | strat4 | cell-CI | betwOp | surp | acc | +  | -  | // | %  |
    |---|---|---|---|---|---|---|---|---|---|
    | a_empty_anchor | 0.924 | [0.79,0.97] | 1.000 | 2.41 | 1.00 | .94 | .98 | .96 | .82 |
    | s_space_before_only | 0.671 | [0.47,0.82] | 0.886 | 3.62 | 1.00 | .97 | .73 | .88 | .10 |
    | s_no_spaces | 0.670 | [0.45,0.81] | 0.657 | 3.24 | 0.99 | .74 | .87 | .78 | .29 |
    | d_parens_anchor | 0.655 | [0.41,0.80] | 0.486 | 2.47 | 1.00 | .70 | .53 | .64 | .76 |
    | p_per_operand | 0.606 | [0.35,0.76] | 0.200 | 2.90 | 0.99 | .58 | .47 | .62 | .75 |
    | s_space_after_only | 0.469 | [0.22,0.65] | 0.829 | 3.28 | 0.94 | .71 | .63 | .65 | -.10 |
    | p_double | 0.279 | [0.01,0.48] | -0.600 | 3.48 | 1.00 | .21 | .03 | .59 | .30 |

    ANCHOR CEILING (corrected): a_empty 0.924 is self-inflated (it's in its own A-group
    reference). Leave-one-out: 0.881. True A-A pairwise within-operator mean: ~0.80
    (+0.81, -0.88, //0.86, %0.64). All "plateau vs inert" gaps should be measured against ~0.80.

    ORIGINAL CIs (operator-level bootstrap, 35 distinct outcomes) were WRONG — both too
    narrow when operators agreed and too wide when one operator was an artifact outlier.
    Cell-level CIs above are from Fable's recomputation.

    PRE-REGISTERED HYPOTHESES:
    - H_plateau (all 5 in [0.45,0.70]): **REJECTED** — p_double at 0.279, and with `%`
      artifact removed s_space_before_only reaches 0.86 (above plateau ceiling).
    - H_rarity: **NOT TESTED AS REGISTERED.** The registered test was 13 forms with
      partialling on |token delta|; only 5 forms were used, no partialling. n=5 Spearman
      cannot reach p<0.017 even at rho=±1. The two registered matched contrasts both went
      RARITY'S WAY: p_double < d_parens (P=1.000), s_no_spaces > s_space_after_only
      (strat3 diff +0.14, CI [-0.01,+0.32]). Evidence: MIXED, UNTESTED. Not dead.
    - H_token_context Contrast 1 (p_double vs d_parens): **CONFIRMED** — paired cell-level
      bootstrap diff -0.376, CI [-0.60,-0.15], P(diff≤0)=1.000.
    - H_token_context Contrast 2 (s_no_spaces vs s_space_after_only): **UNDECIDABLE** on
      strat4; on strat3 (clean, excluding `%`): diff +0.14, CI [-0.01,+0.32].
    - Contrast 3 (s_no_spaces vs s_space_before_only): **NOT PRE-REGISTERED.** Executed
      post-hoc. On clean operators: -0.06, CI [-0.26,+0.10]. Indistinguishable.

    SURVIVING FINDING: p_double `((expr))` has lower within-operator loading than d_parens
    `(expr)` and p_per_operand `(a)op(b)`. Survives cell-level paired bootstrap and artifact
    removal (strat3 diff -0.35, CI [-0.61,-0.08]).

    WITHDRAWN (5 claims):
    1. "Destroys +/- while preserving //" — per-operator loadings have n=15 each; Fisher-z
       CIs: + [-.37,.67], - [-.52,.56], // [.07,.85], % [-.29,.72]. The //-vs-+ and //-vs-
       differences are within 1 SE. Ordering of four noisy numbers read as structure.
    2. "// resilience from disambiguation, robust to nesting" — `-` is the MOST header-
       sensitive operator (34.6 mean dist) with zero disambiguation. When disambiguation IS
       the effect (`%` artifact), loadings are LOWEST, not preserved. No `//` effect to explain.
    3. "Token-context refuted: s_no_spaces ≈ s_space_before_only" — the 0.670≈0.671 equality
       is manufactured by the `%` artifact. Clean (strat3): 0.80 vs 0.86, both near-inert
       ceiling. Can't conclude either way. Contrast was not pre-registered.
    4. "betwOp = -0.600 inverts operator structure" — Spearman over n=6 points. Exact
       permutation p=0.24. Non-significant.
    5. "Rarity is dead" — n=5 test had no power. Registered test never run. Two matched
       contrasts went rarity's way. Evidence is mixed, not conclusive in either direction.

    LEADING ALTERNATIVE (from Fable review): HEADER SHIELDING (M-shield).
    Parens shield the expression from the header, depth-monotonically. Per-operator mean
    distance and SD for `-`: A=33.6(10.0), (e)=24.4(5.5), ((e))=14.0(2.1). This is the
    header effect being REMOVED on the operator that had the most of it, not cell rankings
    being "reordered." The loading drop follows from SD compression — when all cells have
    similar (low) distances, the ranking is whatever noise remains near the floor.

    `%`-DISAMBIGUATION ARTIFACT (confirmed, same class as `//`-comment artifact):
    `<digit>%` with no leading space triggers percent-literal reading ("four percent") on
    bare surface. Header cures it. Contaminates s_no_spaces and s_space_after_only on `%`
    cells (7-15x distance inflation). Must be excluded. With `%` excluded: s_no_spaces
    strat3=0.80, s_space_after_only strat3=0.66, s_space_before_only strat3=0.86 — spacing
    perts sit at or near the inert ceiling. The "plateau" for spacing perts was an artifact.

    R32 NESTING DEPTH FOLLOW-UP (zero-cost except p_triple, ~30s CPU):
    Depth series: expr(0.924) → (expr)(0.655) → ((expr))(0.279) → (((expr)))(0.287).
    FLOOR at depth 2 (p_triple 0.287 ≈ p_double 0.279, diff +0.007). NOT monotonic.
    M-shield prediction (mean distance decreases monotonically) holds through depth 2 on
    header-sensitive operators (`-`: 33.6→24.4→14.0; `%`: 24.7→15.6→11.8) but REVERSES
    slightly at depth 3 (`-`: 14.0→15.9; `%`: 11.8→13.0; `+`: 14.4→16.7; `//`: 14.9→15.5).
    All 4 operators show small mean distance INCREASE from d2→d3. The floor is real but
    it is not pure monotonic compression — something additional happens at depth 3.
    Tokenization: ` (((` and `)))\n` are each single Qwen3 tokens (same as ` ((` and `))\n`),
    so all depths ≥1 have the same token count (8). The floor is not a tokenization artifact.

81. **R33 zero-cost re-analysis — artifact exclusion, cell-level CIs, clean landscape (2026-09-20).**
    Implements all Fable review recommendations. Zero cost (reads saved data only).
    Saved: experiments/results/transfer_spectrum/r33_reanalysis.json.

    ARTIFACT EXCLUSION RULE: exclude (pert,op) blocks where median distance > 3x A-median.
    Catches: j_space_before_op/`//`, j_spaces_both/`//`, s_no_spaces/`%`, s_space_after_only/`%`,
    d_temp_var/`+,//,%`. Nothing unexpected. Rule catches exactly the known disambiguation artifacts.

    A-A PAIRWISE CEILING (corrected): +0.808, -0.880, //0.861, %0.635. Mean ~0.80.
    LOO anchor: a_empty 0.881 (self-included 0.924).

    CLEAN LANDSCAPE (artifact-excluded strat4):
    Level 1 INERT (>0.80): all A-group, I-group, j_space_after_op (0.866),
      j_space_before_op (0.810, was 0.655 with `//` artifact), s_no_spaces (0.798, was 0.670
      with `%` artifact). ALL spacing perts are near-inert once artifacts excluded.
    Level 2 PARTIAL (0.55-0.80): a_identity (0.790), o_parens (0.745), j_spaces_both (0.714),
      s_space_before_only (0.671), s_space_after_only (0.660, was 0.469), d_parens (0.655),
      j_parens_first/second (0.63-0.64), p_per_operand (0.606), v_parens (0.587), d_cond_true (0.581).
    Level 3 BELOW PLATEAU (<0.55): p_double (0.279), p_triple (0.287). ONLY double/triple parens.

    KEY INSIGHT: The "three-level plateau" from R30b was an artifact of the `//` and `%`
    disambiguation artifacts inflating pooled statistics. With artifacts removed:
    - Spacing changes are INERT (whitespace doesn't matter)
    - Single-paren forms span 0.58-0.66 (the real "plateau")
    - Only double/triple parens (~0.28) are genuinely below
    - The organizing variable is parenthesisation depth, not "AST-preserving in-expression edit"
    - Fable review's "adds bracket tokens vs changes whitespace" distinction is confirmed

    PAIRED CONTRASTS (cell-level bootstrap, 2000 iterations):
    p_double - d_parens: -0.364 CI[-0.601,-0.156] P(≤0)=0.999 (confirmed)
    p_double - d_parens (strat3, excl %): -0.338 CI[-0.600,-0.079] P(≤0)=0.995
    p_triple - p_double: +0.013 CI[-0.226,+0.255] (floor confirmed, CI spans 0)
    p_triple - d_parens: -0.355 CI[-0.612,-0.078] P(≤0)=0.992
    s_space_before - s_space_after (strat3): +0.207 CI[+0.031,+0.425] P(≤0)=0.009
    s_no_spaces - s_space_before (strat3): -0.063 CI[-0.256,+0.113] (indistinguishable)

    H_RARITY (PARTIAL): n=5 R31 forms only, rho=0.100, p=0.873. STILL INCOMPLETE — need
    R30b J-group + extra surprisal (~1 min CPU) for the registered 13-form test.

82. **R33b H_rarity as registered — 13-form test (2026-09-20, ~63s CPU).**
    Computed line surprisal for 8 R30b in-expression forms (j_space_before_op, j_space_after_op,
    j_spaces_both, j_parens_first, j_parens_second, j_unary_first, d_parens, v_parens) + 5 R31
    forms = 13 total. 69 cells × 8 forms × 1 forward pass = 552 forward passes, 63s CPU.
    Saved: experiments/results/transfer_spectrum/r33b_r30b_surprisal.json.

    H_RARITY (13 forms, raw Spearman): rho=-0.196, p=0.522. NOT SIGNIFICANT.
    H_RARITY (13 forms, partial on |token delta|): rho=-0.151. WEAKENED further.
    Registered threshold: rho ≤ -0.6 with CI excluding 0. FAILS.

    The sign is correct (negative: higher surprisal → lower strat4, consistent with rarity
    predicting that rarer forms have lower loadings). But the effect is weak and not significant.
    Rarity is INCONCLUSIVE: evidence is genuinely mixed.
    - Rarity-supportive: p_double (high surprisal 3.48, lowest strat4 0.28) fits.
    - Rarity-contradictory: j_unary_first (highest surprisal 3.99, strat4 0.50 not lowest).
    - j_space_after_op (lowest surprisal 3.07, highest strat4 0.87) fits rarity direction.
    - Scatter is wide. n=13 is underpowered for rho=-0.20.
    The registered H_rarity test is now COMPLETE. Verdict: INCONCLUSIVE.

83. **Fable R34 design review — R34 ABANDONED (2026-09-20).**
    Review: scratchpad/r34_design_review.md. Verdict: DO NOT RUN AS DESIGNED.
    KEY FINDINGS FROM EXISTING C-GROUP DATA:
    (1) "Python evidence" is NOT the operative variable. `pass` (Python keyword) RAISES header
        distance by 50% (33.6→49.6 on `-`). `# start` (comment, ambiguous language) COLLAPSES
        it hardest (4.5). The prefix effect tracks "first-line-is-a-comment", not P(Python|prefix).
    (2) The paren floor (~14) is 3× ABOVE the comment-prefix floor (~4). If parens "shielded"
        the header, the paren floor should match the redundancy floor. It does not.
    (3) p_double is NOT a compressed copy of A. Pearson(p_double, A-mean) = 0.105 over 60 cells.
        d_parens = 0.716. p_double is genuinely REORDERED, not "A scaled down."
    (4) M-shield as stated (compression → noise → low strat4) FAILS quantitative check:
        simulation with compression ratio 0.4 + constant noise predicts strat4≈0.62; observed 0.28.
        p_double within-`-` SD (2.1) < A-A noise SD (3.3) → noise compressed WITH signal →
        Spearman should not have dropped at all. p_double's low strat4 is genuine reordering.
    (5) `c_comment_before` at mean distance ~4 (3× below p_double) keeps strat4 0.54.
        "Near floor → noise" directly contradicted.
    (6) Before-line edits are NOT inert — every C-group pert has strat4 0.37-0.61.
    (7) The R34 hypothesis labels were INVERTED.

    M-SHIELD AS STATED IS REFUTED by the existing C-group data + the Pearson check.
    Need a new account of why p_double has low strat4. Leading candidates:
    - Bracket-token identity effect (` ((`/`))\n` are rare tokens that change representation)
    - Genuine representation change from double parenthesisation
    - Some combination of compression + reordering specific to bracket tokens

    RECOMMENDED REPLACEMENT (from Fable):
    (a) Zero-cost C-group re-analysis as prefix family
    (b) 2×2 factorial: {none, `# start\n`} × {e, ((e))} — 3 of 4 cells exist, one new (~30s)
        M-redundancy predicts: prefix+parens ≈ prefix alone (parens redundant under prefix)
        Repr-change predicts: prefix+parens ≠ prefix alone (parens still shift cell pattern)
    (c) Sign-flip control: `# C code\n` — language-evidence predicts distance UP;
        first-line-comment predicts COLLAPSE. Maximally diagnostic.

**Next:** Zero-cost C-group re-analysis (Pearson vs A, per-op mean distance ratio). Then
    implement 2×2 factorial + sign-flip control (~60s CPU).

84. **Zero-cost C-group re-analysis — prefix family characterization (2026-09-18).**
    6 models, balanced kill data. Script: scratchpad/cgroup_reanalysis.py. No model runs.

    C-GROUP IS NOT INERT. CC transfer rho (within-group Spearman):
    Qwen3-1.7B: 0.458, SmolLM2-1.7B: 0.805, SmolLM2-360M: 0.640,
    Qwen3-0.6B: 0.173, Phi-1.5: 0.686, StableLM-2: 0.496.
    Compare AA: 0.696-0.944 across same models. CC << AA in 5/6 models.
    AC cross-group: 0.271-0.757. Prefix perturbations probe different cell structure than suffix.

    TWO-TIER SPLIT by prefix type (C/A distance ratio, cross-model median):
    Tier 1 — COLLAPSE header distance:
      c_desc_comment (`# Description\n`): 0.12-0.37x A (most collapsing)
      c_comment_before (`# start\n`): 0.12-0.66x A
      c_assert_before (`assert True\n`): 0.17-0.79x A
    Tier 2 — MATCH or RAISE header distance:
      c_dead_var_before (`_z = 0\n`): 0.27-1.24x A
      c_pass_before (`pass\n`): 0.25-1.77x A
      c_newline_before (`\n`): 0.35-3.73x A (most amplifying)

    ORGANIZING VARIABLE: "first line is a comment" → collapse; "first line is code" → no collapse.
    This confirms entry 83(1): the operative variable is comment-vs-not-comment, not
    P(Python|prefix). A bare newline (no Python evidence) RAISES distance 1-3.7x in 4/6 models;
    a `pass` statement (strong Python keyword) raises it 1.4-1.8x in Qwen3 models.

    PEARSON VS A-MEAN (cell-ranking preservation):
    Comment prefixes have variable Pearson with A-mean: 0.17-0.93 (model-dependent).
    c_dead_var_before has consistently high Pearson (0.31-0.96): preserves A-group cell ranking.
    c_newline_before: high Pearson in 4/6 models (0.48-0.92): same cells are fragile.
    Interpretation: comment prefixes can REORDER cell vulnerability (change which cells are
    fragile); code prefixes preserve the ranking but may scale the magnitude.

    MODEL HETEROGENEITY:
    Phi-1.5: most stable — all C Pearson>0.63, CC rho=0.686. C behaves like scaled A.
    Qwen3-0.6B: most heterogeneous — CC rho=0.173, c_desc_comment Pearson=-0.228.
      C-group probes genuinely different structure in this model.
    SmolLM2-1.7B: highest CC (0.805) and AC (0.757) rho — best C→A transfer.

    PER-OPERATOR PATTERNS:
    The `//` (integer division) operator consistently shows the largest C/A distance ratios
    across models (especially for c_newline_before and c_dead_var_before). The `%` operator
    is similarly amplified. This is the same operator-specificity seen in the main kill data.

    IMPLICATION FOR PERTURBATION HIERARCHY: C-group sits BETWEEN A-group (inert suffix) and
    D-group (expression modification). CC rho < AA rho confirms C is more heterogeneous than A.
    The two-tier split means position alone doesn't determine C-group behavior — what the
    prefix IS (comment vs code) matters. This is the first clear case where CONTENT of a
    non-expression perturbation affects behavior above noise level. The hierarchy is:
    A (inert suffix) → C-comment (comment prefix, collapses) → C-code (code prefix, variable)
    → D (expression modification, heterogeneous). Position is necessary but not sufficient.

85. **Fable adversarial review of domain universality design — REVISE (2026-09-18).**
    Review: scratchpad/fable_universality_v2.md. VERDICT: REVISE, not proceed as designed.

    KEY FINDINGS:
    - CRITICAL-1: 4/6 predictions are tautological. AA >> DD is a property of perturbation
      position (causal masking), not domain. Will trivially generalize.
    - CRITICAL-2: Breakthrough ceiling for generalization is LOW. "Position matters for
      transformers processing code" is expected since Vaswani 2017, not field-changing.
    - MAJOR-2: D-group is not domain-portable. Identity operations differ fundamentally
      across domains (int() on int vs str() on string vs roundtrip vs bool cast).
    - MAJOR-3: C-group two-tier split (entry 84) IS the diagnostic prediction — it tests
      content distinction (comment vs code), not just position.
    - MAJOR-4: Test alone doesn't meet "breakthrough or nothing." Needs pairing with
      theoretical advance (formal model of why comment/code processing differs).

    REVISED RECOMMENDATIONS:
    1. Primary question: "does C-group comment/code split generalize?" (non-tautological)
    2. Reduce to 20 cells × 2 domains × 3 models (~2880 forward passes, ~5-8 min CPU)
    3. Keep only portable D-perts (d_parens, d_cond_true, d_temp_var, d_double_parens)
    4. Pre-register C-group split: comment < 0.5x A-mean, code > 0.5x A-mean
    5. Run G0 first as separate diagnostic
    6. State breakthrough path: universality → C-group → formal model → testable predictions

86. **Droid strategic review — construct-validity kill test (2026-09-18).**
    Review: scratchpad/droid_strategic_r1.md. 

    CENTRAL CONCERN: The hierarchy may be a property of the `# Python code\n` probe,
    not of perturbations in general. The measured quantity is distance(bare, header)
    under each perturbation — NOT distance(unperturbed, perturbed). The C-group
    comment/code split could be an interaction with the header: comment prefixes make
    bare and header conditions more similar (both start with #), code prefixes make
    them structurally different.

    RECOMMENDATION: Run neither domain universality nor sign-flip as the main experiment.
    First run a CONSTRUCT-VALIDITY KILL TEST:
    - Measure perturbation distance DIRECTLY: cosine(unperturbed, perturbed)
    - Do this on both bare and header-present surfaces separately
    - Use small representative set: inert suffix, comment prefix, `# C code` prefix,
      code prefix, blank-line prefix, expression rewrite
    - Full-logit and answer-level readouts, multiple model families

    DECISION RULE:
    - If hierarchy FAILS direct measurement: reframe as "contexts modulate response to
      a Python-language header." Correct public claim. Close program.
    - If hierarchy SURVIVES: evidence for probe-independent susceptibility structure.
      Only then run domain test.

    BREAKTHROUGH ASSESSMENT: None of these experiments is field-changing alone. Only path
    is a predictive invariant: rule specified before observation that predicts unseen
    perturbation effects across domains/readouts/models, beating token frequency, length,
    position, and syntax baselines.

    SYNTHESIS OF BOTH REVIEWS:
    Fable says: C-group split is the diagnostic prediction, not A/D gap (tautological).
    Droid says: Before testing universality, test construct validity — is the hierarchy
    probe-independent? Both agree: breakthrough ceiling of current designs is low.
    Both point toward the same deeper question: is there a probe-independent perturbation
    sensitivity structure, and can it be predicted from first principles?

    NEXT: Revise design to address construct validity first (Droid), then C-group
    universality (Fable), with breakthrough path stated explicitly.

---

### User-requested paused closeout — 2026-09-14

Research is paused through S65; no new experiment is queued. Start at
[docs/HANDOFFS.md](docs/HANDOFFS.md) on return. S58 supplies a narrow teaching
positive; S62 does not justify richer feature access on its panel; S63/S64 separate
temporal contributor evidence from consumer/memory effects. S65 completes a
three-case supplied-revision diagnostic, but zero capital prerequisites are correct
(18/18 responses terminate). All outputs and first-attempt failure are preserved.
The natural-question prerequisite remedy is unrun. Mission/philosophy remain
unchanged and unfinished; no cumulative acquisition or native law is established.


### Strategy routing consolidation — 2026-09-14

S53's consumer/access distinction, S54's admission-history analysis and S55's
global-editing/resource check are complete in canonical theory and the ledger.
The current next-work boundary now indexes remaining empirical decisions and
completed arguments rather than repeating chronological routing. No result or
closure is changed; no new computation is counted.

## Strategy/discovery checkpoint and history — 2026-09-14

S53 selects reaction-sensitive structural requests as a concrete candidate. Source
study keeps mixed inputs, recorded-reference limits and MIT's missing stereochemistry
explicit. A four-case source illustration is manually checked, with no parser/model
run. The reviewer corrected query-count economics: more queries can favor a graph
decoded once. Next define a limited consumer and attribution comparison; no training.

S52 checks assimilation through independent ports and the existing per-request
consumer loss. KL decomposition identifies its marginal target; two-round review
chooses objectively assessable later uses instead of imitating donor randomness.
A reusable sufficient-record predictor is the cheap comparator; two anticipated
answers alone do not show operation acquisition. No empirical count changes.

Fresh periodic review around 15:30 UTC followed62f85fe and separate entropy checking.
It corrected a possible conflation of unchanged connected addition with shared
updates that can forget old uses. S50 replay is conditional on modifying relevant
dependencies and is not necessary if induced-input component replay already suffices.
S51 works one depth-boundary error and a direct repair hypothesis; it supplies no
measured reason for shared adaptation. Next choose a specific testable benefit,
not another model merely because it is downloadable. Independent subagent review,
not a real CLI audit. No empirical count changes.

S51 defines a useful category/depth pixel selection and verifies one raw depth
interface. No learned alignment is required to express the direct route. A limited
source check did not bind the actual trained AB endpoint, so local execution of
that exact acquisition episode closes. Keep source findings; pivot the approach.

S50 now supplies the concrete reference episode: semantic acquisition into a
spatial model, old-task evaluations and useful joint behavior. Attribution is
limited by unmatched guidance, semantic comparator and resolution. The next
question was retaining the achieved interaction during another admission; S50's
worked replay proposal now records that dependency, and S51 closes this exact
local episode after its endpoint check. Original-input retention alone does not
settle interaction retention. No local reproduction.

**Latest user restart, 2026-09-14:** mission fixed, approach flexible. S37 and its
unavailable CLI audit are parked. S38 now separates donor target construction,
student intervention training, pretrained contribution and mechanism interpretation.
Primary sources and independent strategy reviews inform the comparison. S39
then found actual cross-model communication implementations and a concrete
receiver-prompt confound. S40 identifies a constructive separately prepared/joint
use precedent, with unbound exact joint-run provenance in the checked release.
S41 one-read attention algebra is independently checked but kept diagnostic; it
does not justify a new measurement program. S42 separates response-generator,
embedding-teacher and predictor roles; conditional prefix reuse is checked but
not implemented. S43 read32 published answers on16 matched queries, finding one
clean correction amid shared errors and unfinished targets. Slice closed without
expansion; no training or model run selected.
S44-S46 clarified conditional evidence selection and direct teaching. S47 ran a
107-line lexical diagnostic on four usable pairs among five selected reports:
7/8 adequate whole-abstract selections,8/8 with RESULTS filtering, one mixed-quality
tie remaining. Closed without model probe or expansion. No acquired artifact.
S48 checks the distinction between portable directions and read/write operations;
shared calibration evidence must also benefit the direct-learning comparator.
Readout transfer without new task calibration does not establish reusable updates.
S49 verifies the missing new-question input in a reconstruction path and a
separate prepared memory consumer with actual later-prompt conditioning. Their
combination remains untested; source multi-span reuse required extra training.
Continuation identifies a trained reasoning comparator and a no-alignment control
that retains rationale access. Shared-model targets remain dynamic despite
detachment; a small-checkpoint download was rejected as low leverage after
two-round review. Consumer teaching must exercise intended later requests.
No commutator apparatus or model run is selected. Older checkpoints below are
historical where they conflict with this current allocation.

**Mission-coverage consolidation, September 14:** current routing now requires
supported prior A, useful new B and a joint use; repeated requests over one memory
are not successive acquisition. Recent interface findings change implementation
choices without supplying that episode. Information novelty is not required for
useful contribution. The compact canonical map separates precedents from missing
local evidence and supersedes the accumulated routing diary above.

The enduring question is what a denizen of latent space needs for identity,
movement, effort, maps and reusable laws. Cumulative capability acquisition is
one direction within that philosophy. Connected growth and assimilation after
donor removal are separate branches. User authorization at 19:45 UTC permits
sustained strategy, source research and bounded CPU probes, not system-building.
No cumulative artifact, causal graft, encoder-reasoning or cost advantage is
established. Closed prior lines stay closed; no run follows from documentation.

**Earlier model-forward checkpoint:** S26's single CPU continuation run is complete, with
independent audit complete: e1044-e1045. Primary and independent pair recovery were both 0/16;
second-site recovery was 2/16 for both, versus cyclic content control 3/16.
The first prediction was correct on only 1/16, capping unrevised pair recovery
below the locked .10 gain threshold even with a perfect second step. Close this
procedure on this panel; this weak-proposal result does not exclude composition
with competent inputs or establish that a learned exporter helps. No rerun,
threshold change or new selection follows. [Raw result](experiments/results/encoder_continuation_v1/result.json)
retains probabilities, counts and hashes; the completed audit receipt is in the ledger.

**Current priority reassessment:** S36's compatibility-first selection is
superseded. Fresh review first preferred S28's evidence-selection comparison;
the second round accepted S37's objectively scored causal-teaching preflight.
That round could not read local files because its shell failed. The third review
read the lock and runner and issued conditional GO after supervisor enforcement,
complete tokenizer/alignment assertions, per-batch checkpoints and reporting fixes.
S37 then made two bounded execution attempts, both with zero scored forwards:
a 180 s compute-phase timeout and an HTTP-compression download bug at metadata
alignment. Raw attempts and diagnostic hashes are retained. Downloader and owned
cache metadata are corrected offline; no third run occurs in this allocation.
No donor competence, recipient headroom or intervention result exists. Two fresh real CLI audit attempts could not read files; their audits are incomplete.
Separate collaboration custody checks verified raw counts and hashes; ordinary/text-counterfactual teaching remains
the required comparator if a future eligible preflight funds teaching. Receipts
live in internal/codex_transcripts, exact failures in the ledger and canonical S37.
S34 and previously closed scientific probes remain deferred/closed.

**Accounting and custody:** 1065 apparatus / 0 artifact-bearing code lines
(98 + 119 + 204 + 207 + 330 + 107), six probe blocks / zero construction rounds;
S37 has two failed attempts and zero scored forwards; 15 total scored forwards,
no GPU or training. Strategy authorization supersedes construction to repair the
ratio. Root alone writes boards after stale-session ID collisions; integrity
entries restored as e1011–e1012. Mission synthesis retains its hang workaround.
Fresh real CLI result audits and design dialogues have distinct receipts below;
fourth periodic review completed around 02:40 UTC on September 14 (e1086-e1087)
after consolidation at e72f83d and a separate entropy review. Strategy map review completed around 03:34 UTC
(e1095–e1096); retain the two-hour cadence during subsequent substantive research. The user-requested S37 reassessment supersedes S36 priority; the canonical
next-work boundary controls execution readiness.
Detailed mathematics/source readings live in
[canonical theory](theory/CUMULATIVE_CAPABILITY_ACQUISITION.md); exact results,
corrections and receipt hashes remain in the [ledger](experiments/ledger.jsonl)
and [experiment log](experiments/EXPERIMENTS.md).

### 23:03–23:34 UTC: typed-contract closure and one continuation probe

The second review closed S24's purchasing contract: authoritative scoped import
and its specified calculator already supply fresh consequences and B retention.
Producer reliability/rechecking would need an actual scoped source and semantic
port; do not invent a learned receiver or extend this hypothetical case.
Two fresh CLI direction rounds then selected S26's one-model self-composition
probe, with independent design review and locked controls for mask-count scaling.
Seven forwards / 112 rows over four reference constructs completed in 10.1455s;
repeat probability drift was zero. Root read all 16 decoded pairs. These are
reference recoveries, not biological quality, held-out generalization or acquisition.
Receipts: ledger `autonomous_continuation_design_20260913` (fresh CLI sessions
01a09d0d-bee5-7ee2-b682-f5a9a17c59b8 and 01a09d11-6b92-7fe3-96ee-c260c44cf46c),
`encoder_continuation_v1_intent` and `encoder_continuation_v1_result`.

### Later strategy: procedure reuse and predicting revision consequences

Two-round dialogue specified S2.2's reusable acquisition procedure: its recipe,
evidence permissions and budget rule persist across donors, while adapters and
receiver weights may learn. New hand-engineered recipes are method revisions;
fresh inputs and new operation combinations distinguish capability from cached
annotations. S12's tied-head calculation saves no unique weights when the input
embedding remains; contextual encoding can avoid generation without becoming
small. This is conditional bookkeeping, not a proprietary or runtime claim.

S25 incorporates lifelong editing rather than excluding text exports: MEMOIR's
learned correction and mask routing, RippleEdits' consequence tests, and
StableEdit's qualified stabilization evidence. None is replicated. Dependency
rules need explicit semantics/correction policies; types alone do not establish
entailment, and orthogonal updates alone do not preserve reads. Answer import
and acquisition of useful consequences are different legitimate branches.

Two fresh real CLI design rounds then sharpened S24's hypothetical 12-to-10
carton correction: the 30-item answer stays fixed, the 36-item answer changes.
Possible dependency, correction applicability and actual answer change differ.
One trusted applicability distinction resolves the stipulated two-world case;
a typed field suffices. A later answer used to infer scope is acquisition evidence,
not an unqueried consequence test. This instantiates S1/S23, not a new theorem.
Caller dependency routing supplies integration behavior, not a discovered learned
law. The dialogue rejected another generic impossibility or renamed program slice.

Geiger2021's existing causal-abstraction bridge separates a symbolic test target
from consequences computed by the intervened neural model. Its MQNLI models
receive task and phrase-relation supervision; the source is not evidence of a
new method or pretrained cumulative graft. Root and independent reader verified
this boundary; no exhaustive pair/clique procedure was imported.
Receipts: ledger events `acquisition_reuse_revision_precedents_20260913`,
`revision_consequence_design_dialogue_20260913` (fresh sessions
01a09ced-ed65-7ab0-809e-c9c8353d18f7 and 01a09cf2-0c79-77b2-8c58-03082c4518e0),
and `causal_abstraction_strategy_bridge_20260913`. Both CLI sessions kept boards
read-only; design dialogue does not replace the periodic approach audit.

### After 21:42 UTC: task eligibility, worked episode and attribution

The task-first PFN1 candidate closed before its 90-minute limit. Test row 125
and training rows 375/440/910/936 share the exact question/disease answer despite
verified sequence edits. Exact-question lookup suffices for this endpoint;
there is no whole-benchmark score or claim that the trained model ignores DNA.
The final formatter omits nucleotide changes, so these pairs do not instantiate
the earlier contradictory-text swap concern. Shared labels make the proposed
functional contrast unmeasurable. The loader combines 146 test and 144 validation
rows, explaining 290 without proving historical selection protocol. No usable
original receiver was verified on scoped official surfaces. Missing resources:
independent answer-changing effect labels, receiver role competence and its
artifact. S8 and ledger retain revisions, row locators, hashes and independent
checks; no replacement system was trained to manufacture eligibility.

S24's actual released table illustration assigns text/locations to OCR, boxes
to the structure donor, binding to geometric postprocessing, and late queries
to an explicit executor. Two-round dialogue and pinned source inspection rejected
crediting fixed binding rules to the donor or weakening the structured comparator
to a hard table. No OCR/pipeline/training ran. The conditional route to an answer
is explicit, but superiority over grouping or direct source extraction is
unmeasured. Token identity, crop coordinates and header conventions can require
translation/revalidation. Coordinate repair is caller work, not a latent hole;
a later two-round dialogue declined crop replacement because wrapper independence
would supply its laws. This is an on-paper episode, not a document product.

The full S3 stitching source replaced its abstract-only placeholder: class-centre
noise is informative and supervised adapters can learn correspondence. Useful
acquisition does not establish preservation of donor computation; learning is
valid when attributed. README's historical 6.3× contrast was qualified without
changing its value; the experiment log preserves the historical interpretation.
Receipts: `table_growth_episode_20260913` and
`graft_interpretation_revision_contract_20260913` in the ledger.

### 21:42 UTC: first periodic approach review

Fresh CLI 01a09cb3-77b6-7840-a52e-a5b0099e7822 reviewed consolidated HEAD 0a93fb7
and retained evidence, with audit-board synthesis; root relayed e1027. Continue
the direction, but availability-led diagnostics are not working as route selection.
Adopt one specialist-encoder/KEGG eligibility block of at most 90 minutes: trace
an actual answer, roles, coupling, ports, later growth and an old use to preserve;
return a populated contract or exact missing resource, not another diagnostic.
Independent two-round dialogue also corrected two restrictive assumptions:
role competence need not mean standalone end-task superiority, and frozen-receiver
interchange is narrower than acquisition. A failed average does not rule out richer
useful distinctions. No architectural hole or acquisition was established.

### Before 21:42 UTC: three bounded probes and their consequences

| Probe | Preserved outcome and qualification | Corrected evidence / review |
|---|---|---|
| S6, retained exports | Legacy plan exports nonfinite or overwhelmingly zero; ineligible for proposed plan-transfer analysis, not a test of model capability | [result](experiments/results/acquisition_cache_integrity_v1/result.json); restored audit e1011–e1012 |
| S21, conditional port, 20:59 checkpoint | Four fixed rectangles below all registered tolerances; eight CPU forwards, 60.27 seconds, no fitting. Post-hoc additive preferences also fit primary tolerance: no useful updating mechanism or global law | [result](experiments/results/conditional_port_v1/result.json); fresh CLI audit e1021, exact arithmetic and scope in S21 |
| S22, cached output screen | Locked two-column equal-rank average loses to retrospectively better donor on four assays / 8,610 matched finite variants; 23.86 seconds including retrieval, no model calls. No absence-of-complementarity or graft-feasibility conclusion | [result](experiments/results/cached_donor_screen_v1/result.json); fresh CLI audit e1022 required integer rank sums for exact ties. Offline correction independently checked with SciPy, maximum correlation change 0.00001343, direction unchanged; original source/result history and byte-preserving Git attributes retained |

Two design dialogues selected S21 after video resources lacked matched observation
access and a fixed receiver; the diagnostic closed without searching for dramatic
cases. S23's independent analytical dialogue separated pairwise from joint
compatibility and compatible completion from identified answers. Joint use needs
coupling evidence or declared assumptions/uncertainty, not necessarily one global
world model. The three probes remain closed, with no automatic panel expansion.

At the 20:45 checkpoint, S16–S20 separated ordinal messages, calibrated quantities,
information novelty and usable computation. Pinned onboarding source lacked an
initial/replacement-content control and compared pooled with single receivers;
its speed headline cannot establish donor contribution, but donor nonuse is not
proved. Three JSON summaries lacked frozen example-level messages/labels for an
alternative-reader replay. Encoder use also needs an inference schedule and
cross-donor calibration. No external model or training script was executed.

Earlier S11–S15 distinguished observability/writeability, exports, conditional
value, interchange and extension. Scene/assignment runs were declined; a fixed
hub can preserve donor-specific information, so alignment is not itself a
bottleneck proof. S10.3's approximate export count includes recipient side
information and is not a model-size lower bound. S8 initially flagged that a
sequence-only swap may contradict variant-identifying text; the later PFN1
formatter inspection above resolves that concern for its specific example.

### Opening and consolidation continuity

Initial review placed access, identifiability and complementary usefulness before
architecture; access ambiguity and conservative extension framed the first questions.
Independent sweeps consolidated current routing without deleting historical
sections. Scoped methodological residue went to the closed capability-translation
sibling (commits 57f8623 and 41aaab7) and the LSR row/lesson in the unversioned
portfolio index; no other portfolio claims or sibling closure were reassessed.

Reverse-chronological running log. Newest first. Each entry: what was done, what
was learned, what's next. Canonical state lives in STATE.md.


### README propagation and theory status index (2026-09-06)

README still claimed "well-defined operators on a probability simplex ...
the coordinate system native to the model's computation" (K_A framework
recorded DEAD on 2026-09-04) and "operator algebra finer than semantic
role". Propagated the audited wording: no stable operator algebra
(congruence failure 2.6-13x; logit-bias rival); surface-specific
responses; the filler reading (equal-length blocks unequal); the neutral-
vs-reassignment reading with its reduced-readout, unmatched-position
limits; a "current direction" paragraph with conditional mathematical
results and the open empirical question; repository structure updated to
the six runners. Added a status index at the top of the theory document.
Codex quota at 92% (resets 2026-09-10): no audits possible; no new theory
claims will be added until then.

---

### Evidence gate on block 3: REVISE; reference scale confounded; rotation ruling (2026-09-06)

Arithmetic reproduced exactly (72 rows; 576 concatenations token-matched;
480 laws valid). Length-only rival conditionally rejected at the 11-bin
W_0 channel (24/24/23; survives the largest admissible guard). My reference
scale was confounded: each filler PAIR changes both the input pair's
filler_A and the successor filler_M, so the "between-filler spread" mixes
input-pair and successor variation; the fixed-input successor range
B(i,j) over j is what is needed and its off-diagonal endpoints are absent.
Withdrawn: "H not distinguishable from an arbitrary comment", "content-
specific H disfavored". Order 2 compares A (neither H nor N). No affine
rival test possible (no initial d_i). Decision amended: fixed-input,
response-independent successor fillers, prospectively specified.

Ruling: rotation blocks 1-3 produced useful decisions (corrected
architectural claims; strong-assert prioritized; the missing fixed-input
filler control exposed) but no artifact; stop zero-call rotation pending
compute approval. Named acquisition: position-matched full-law
eligibility-first panel on repaired strong-assert pairs (five blocks; floor
measured; five robust W_H-close pairs; N/H successors; three fixed-input
successor fillers). Devansh's standing rule is never idle and rotate: so
the next block is analytical preparation that does not reopen closed lines.

---

### Block 3 reading: fillers differ from each other more than H differs from a filler (2026-09-06)

Zero model calls. Multifiller table (Gate 2 v2 repair data): surfaces and
fillers token-matched by construction (5/5 and 7/7). Length-only rival
(|b_H - b_F| <= 0.002 on identical input pairs) violated in >= 23/24 roots
for all three fillers and both orders; sign of the deviation filler-
dependent. Reference scale: between-filler spread median 0.098 (order 1),
0.084 (order 2), versus content-vs-filler deviation medians 0.017-0.027
and 0.025-0.069; content exceeds the filler spread in 0/0/1 roots (order
1) and 0/2/6 (order 2). Reading: any 7-token comment perturbs pair
discrepancies about as much as the "reassignment" comment; the content-
specific H hypothesis is disfavored; the deferred panel needs >= 3 fillers
and the between-filler spread as reference. Limits: initial closeness
unknown, 11-bin, floor assumed. Evidence gate next.

---

### Evidence gate on Q4: REVISE; decision amended; next block = multifiller (2026-09-06)

Codex recovered the deleted runner from git (c2465ba^) to confirm arm
semantics (_c0/_c1 = surface -> composer -> query; _c0r/_c1r reversed),
reproduced all 72 rows with zero discrepancy, and found: (i) "full-law
witnesses" overstated: an 11-bin split proves post-H full-law separation,
not a counterexample to full-law preservation (initial full-law closeness
unknown); (ii) the guard is assumed (no replay data in this table); (iii)
the two surfaces of each pair differ in token count (4/5, 4/6, 4/7), so
composer/query positions are unmatched: under the locked interface the
verdict is INCONCLUSIVE; (iv) my positive-verdict branch was too weak in
general (must bound every potentially close pair) though the counts stand;
(v) median bug (upper-middle), fixed. Decision amended: strong-assert as
candidate family for a NEW position-matched full-law eligibility panel; do
not replace the deferred panel (CRC-0 lacked W_H-close pairs; Q4 supplies
W_0-close pairs; only three W_H-close pairs exist here over two blocks).
Q4 is decision-relevant. Next: multifiller reading, TV(AM, F_AM) vs
TV(AF_M, F_AF_M) on the same roots, beyond 0.002, after verifying token
matching; if non-actionable, stop rotation pending compute approval.

---

### Q4 reading: N preserves, H witnesses, on the archived congruence panel (2026-09-06)

Zero model calls. The svb_qwen3_congruence table (24 cells x 3 presentation
pairs x composers N/H, 11-bin laws) contains exactly the CR-15.10
quantities D0, DN, DH for forward order. At tau = 0.01: strong_assert
24/24 close, N preserves on all (max 0.0064), H witnesses 23/24 (max
0.116); moderate_assert 8/24 close, N preserves (max 0.00999, borderline),
H witnesses 6/8; misleading 1 close pair, split under both. Reduced
channel: H witnesses are valid full-law witnesses; N's preservation is
reduced-channel only. Decision candidate: the deferred panel's root family
should be assert-type presentation pairs (>= 24 close pairs vs CRC-0's
zero), N as the plausible preserved action, H as the witnessing action.
Evidence gate next (one Codex call).

---

### Audit #42 on CR-16: RESTATEMENT; rotate to Q4 (2026-09-06)

Audit #42: RESTATEMENT; CR-16 requires corrections. Fixed-token causal site invariance and, with consecutive positions, the appended-token offset identity are retained. The exact two-route decomposition, content-independent prefix re-weighting, filler-effect subtraction and identification with CEG-1 RELAY-CUT are withdrawn. CR-15.10.6 already requires matched positions, valid tokenization boundaries and the registered H-versus-F rival. Close Q1 and rotate to one bounded Q4 reading of existing audited response tables under their original interface and claim limits, with no runs or new acquisition.

Why the two-route decomposition failed: tokenization can change prefix
IDs; 'exactly k' needs consecutive position IDs; the new keys' softmax
contribution is content-dependent; the final query state is new and
content-dependent through earlier layers; residual/MLP paths at the read
position contribute even with zero attention output projections; new-site
information changes the later query which changes its weighting of old
sites, so the routes overlap causally. Historical RELAY-CUT blocks
live_start.. -> dead_span attention, not query -> appended block. Program:
continue; 'not working as a route to a real-model artifact'; Q4 is one
bounded evidence-to-decision map, then a decision or close.

---

### Question rotation block 1: CR-16 architectural restrictions (2026-09-06)

Devansh: never stop; rotate questions; spare the machine. Proposed five
compute-free questions; Codex ranked Q1 (forced identifications of the
transformer world) first with a strict gate: it counts only if it yields a
nontrivial consequence for W_0/W_H/W_2, else rotate to Q4 (existing
evidence). Codex corrections to my five statements: (1) rotary symmetry =
origin relabeling of all position IDs, not prefix-length changes; cite
RoFormer; (2) causality exact but site-local; appending changes the read
position; (3) RMSNorm eps > 0 is NOT scale-invariant; pre-norm residual
breaks ray invariance; a pure pre-LayerNorm stack under the stated final-reader conditions has a whole-stack
shift symmetry; my ray quotient withdrawn; (4) multiset identification
withdrawn (I had already withdrawn it); (5) parity is in TC0, Hahn's rate
is for non-causal attention; scope separately.

Drafted CR-16 with the one consequence I can prove from 16.1 + 16.2: the
appended-block offset effect. Any appended block of length k increases the
final query's offset to EVERY frozen prefix site by exactly k, content-
free; so the B-question differs from the immediate read through exactly
two routes: (a) re-weighting of frozen sites (present for fillers too) and
(b) the new sites' prefix-conditioned representations. Consequences: the
length-only filler is mandatory, not a straw man; W_H-splits of W_0-close
pairs can be offset effects; RELAY-CUT geometry is derived; question
families should be stratified by appended length. One audit next.

---

### Audit #41 on CR-15.10.7: REVISE; analytical cycle stops (2026-09-05)

Fresh audit of the seven draft items plus my self-critique. Retained: (2)
finite-panel enclosures, (5) refinement identity, corrected moduli (1) and
conditional routes (4). Withdrawn or replaced: the obstruction theorem
(a depth-one transcript can force omega(0)=1 in EVERY consistent world, so
the "zero-modulus completion" claim was false; replaced by a finite
response-transcript obstruction that asserts only the unit-modulus
completion); the cyclic parity claim (two consecutive cosines determine the
phase for every m >= 3 via sin(j a) = (cos(j a)cos a - cos((j+1)a))/sin a;
gain bounded by 1 + 2|cos a| <= 3, not unbounded); the coordinate-
replacement row (needs a product-max response metric; counterexample
given); (4b) needs FULL-PROFILE Lipschitz control, not immediate readers;
the saturation impossibility (lambda >= e^G/L_g on same-sided domains
survives as a conditioning statement; an arbitrarily saturated reader can
still give L kappa lambda < e/4 < 1); the rank proposition (local
necessary condition K >= d survives; the global s_min bound is false, with
a circle counterexample; Jacobian rank is not PSR dimension; W_H is two
FULL-LAW questions, not two scalars, so no rank deficiency follows; the
rung-1 reader is rank one on an open 16-dim domain, which rules out
carrier-metric inverse observability there but not behavioral guarantees).
Also: the saved certificate result stores mean probabilities, not per-cell
logits, so |g| ~ 18 was not established; "order-one carrier separation" is
compatible, not indicated.

My self-critique was only partly resolved (4 of 6), and one of my "fixes"
(parity) was itself wrong. Lesson recorded in memory: hour-by-hour solo
derivations without review accumulate errors; audit before extending.

Ruling: one correction-and-consolidation pass (this), then STOP the
analytical cycle and wait for approval of an empirical step that can change
a program decision. "This is not working as a route to a real-model
artifact." Program remains open.

---

### Inverse observability is a rank condition (2026-09-05, heartbeat addendum)

Continuing (6): premise (4b) needs the question family's response map to
be injective on the invariant set, hence at least d questions for a
d-dimensional carrier, with a well-conditioned question Jacobian; lambda
<= 1 / inf s_min(J_P), and saturation multiplies s_min by e^{-G}. The
relevant d is the behaviorally effective dimension (rank of carrier ->
all-future responses), i.e. the PSR dimension; a family attaining it is a
core test set (PSR/OOM prior art, used not claimed). Consequences: the
deferred W_H panel (K = 2) can never give an upper bound on omega beyond
the panel (rank-deficient by construction; witnesses and panel moduli
only); the rung-1 reader was rank ONE on a 16-dim carrier, so no
immediate-read family could have supplied inverse observability there,
a structural reason visible before any run.

---

### Saturation destroys inverse observability (2026-09-05, heartbeat addendum)

Item 6 of my self-critique, derived: for a sigmoid/softmax channel, on the
region |g| >= G the response compresses carrier differences by e^{-G}, so
the inverse-observability constant lambda >= e^G / L_g and premise (4b)
cannot hold from saturated immediate reads. Applied analytically to the
recorded certificate run: correct-slot probabilities within 1e-8 of 1 mean
|g| ~ 18, so the 1.4e-8 erasure TV is compatible with ORDER-ONE carrier
separation between obsolete-write counterfactuals; the panel erasure may
be entirely reader-side. Pre-registered seconds-long check on the saved
weights written down but NOT run (evidence gate: no follow-up on that
cell). Design consequence for the deferred real-model panel: choose
continuations by response entropy so that some questions are unsaturated.

---

### CR-15.10.7 draft: the analytical deliverables, written solo (2026-09-05)

No Codex (quota). Delivered in theory/ as DRAFT: (1) moduli for seven
world/family rows; the cyclic counter shows both failure modes in one
world (consistency failure under immediate reads; exact closure with
amplification once one extra question is added). (2) Finite-panel
bounding proposition with proof; the panel modulus lower-bounds the
domain modulus and never upper-bounds it. (3) Obstruction theorem:
any finite transcript is consistent with worlds having omega(0) = 0 and
omega(0) = 1 (countdown branch attached beyond the transcript horizon).
(4) Three premises for an upper bound: finite orbit (exact enumeration
at depth N - |Q_0|); contraction + reader Lipschitz + INVERSE
OBSERVABILITY (rho <= lambda d_{W*}) giving omega <= L kappa lambda t,
which reproduces the CR-15.9 world's t/2 exactly and names the half the
certificate lane lacked; closed invariant relations (bounds only pairs in
a common realization set). Proposition: a denizen certifies lower bounds,
panel moduli, and refutations; never an upper bound. (5) Refinement
modulus rho_h(t) = max(sup_{eligible} d_{W_h}, max_a omega_a^{W_h}(t)) (corrected per audit #41).

Review pending (fresh audit when quota allows). Nothing here is a real-
model finding.

---

### Direction locked: CR-15.10.6, analytical cycle on finite-question certification (2026-09-05)

Round 3 (locking). Interface AMENDED (actual token positions and typed
context, not character counts; W_H successors need execution-order tuples
(a,) and (a,H); P2's `func` vs `f` malformation; dense laws must be saved).
Guarantee AMENDED (tau = 0.01 locked; five pairs = coverage over five task
blocks, not statistics; CRC-0's 7e-7 was a 4-sentinel cached-vs-full mean,
not a replay floor). Prediction REPLACED (P and R are not exclusive; the
doubling world already shows exact sufficiency with amplification; safety
prediction P on reserved endpoints + a prospectively reserved disagreement
set). Rival rule REPLACED (my CV rule is unsound: b/d = g + c/d; use
interval-aware affine infeasibility with eps_R = 0.002, and a separate
length-only rival with a position-matched filler). Evidence route REPLACED:
analytical first; the historical panel has ZERO eligible W_H-close
same-variable pairs, so the proposed positive gate could not succeed on it;
1.7B panel deferred and needs explicit approval.

Codex independently derived the same four moduli as my worked-examples
note (countdown 1; doubling min{1,2t}; CR-15.5 theta t; CR-15.9
half min{t, delta}); my note adds the cyclic counter's two-mode case and
the refinement modulus rho_h(t). The substantive analytical gap named:
turning finitely many questions into an UPPER bound on omega needs an
observability premise in the other direction (small d_W => small carrier
separation), which neither CR-15.2 nor CR-15.3 supplies.

Codex weekly quota at 75%; no further Codex calls today. Next: deliver
the four analytical items of CR-15.10.6 solo as a DRAFT in theory/, for
review when quota allows.

---

### Direction dialogue round 2: propagation inequality kept, "iff" replaced by a modulus (2026-09-05)

Codex round 2 on my rule attempt. Kept: d_W(T_a x, T_a y) = d_{aW}(x,y)
exactly; the propagation inequality with corrected display. Replaced: my
"propagates iff closed" with the modulus omega_a(t); exact profile closure
preserves exact equality but need not preserve positive tolerances (T(x) =
min(1,2x) doubles small discrepancies); sufficient fixed-tolerance
condition L_a tau + 2 delta_a <= tau. Correspondence with CR-15.2 is exact
at eta = 0 on an invariant set with all immediate channels in W, with NO
L <= 1 requirement; not exact in general (abstract state may remember
distinctions absent from W). My countdown example had a factor-of-two
error (radius vs diameter). Angluin: I had misnamed consistency as
closedness. No mathematical novelty; value must come from real-model
predictions.

Real-model: my "immediate-read family closed under N, not under H" reading
is NOT licensed; licensed sentence is continuation-dependent amplification
at 11-bin resolution. CRC-0 never saved its 36 x 13 laws (runner computes,
saves summaries; config description mismatches), so the enlarged-family
test needs a rerun; but two recorded witnesses already reject preservation
of the 0.01 tolerance under H (d4_y_3, d4_y_5 presentation pairs).

Adopted as CR-15.10. Direction: interface-relative approximate
interchangeability with omega_a(tau) as organizing quantity. Round 3 must
settle interface, guarantee, one consequential prediction, cheapest rival
(lexical/position readout sensitivity), evidence route. Codex weekly quota
being rationed.

---

### Re-contextualization round 1: register-free theorem adopted; direction reset (2026-09-05)

Codex counter to my re-contextualization note. Adopted: CR-15.9, the
register-free CR-15.8 (least congruence generated by eps-accurate word
equations; endpoint loop-erasure; (N-1) eps; no factor two). Corrected: I
had promoted finite N into an organizing necessity; it is one sufficient
premise among three (counting, contraction, closed invariant relations),
with an explicit infinite-orbit contracting counterexample. "Local" means a
short equation whose eps bounds all futures, not a short horizon.

Trap ruling: the von Neumann diagnosis is substantially right (engineered
addresses/values/readers certified as discovery), but coordinate-free
wording is not an escape by itself; a human-chosen interface is not native
because it avoids coordinates. Four explanations of the certificate result
remain live (reader hides history; expansion off the useful region;
expanding directions irrelevant; short-horizon shortcut).

Direction: option 4 + new option 5 (interface-relative erasure and
preservation): produce one explicit rule about when an erased distinction
stays irrelevant under subsequent actions, anchored to CR-9's real-model
pairs, with a counterexample and an intervention consequence. Discrete
carrier deferred (finiteness is designed in; BFS counts concrete not
behavioral states; quantization changes the certified system). No run.

Codex's mandatory answer: continue the program; the selective-reset line is
"not working yet" as a route to the central real-model artifact. Honest
accounting today: zero new real-model-building cycles; one 130 s tiny-cell
retrain; two theory blocks adopted; two audits + one evidence gate.

---

### Certificate check: INCONCLUSIVE, premise fails on the fixed cube (2026-09-05)

Design gate GO (round 4) with a ten-step protocol; my Claim 1 (CR-15.8
vacuous on continuous carriers) refuted in its strong form by a dense-orbit
example with packing number 1; my Claim 3 corrected: a bare positive per-step foreign-write drift bound
does not prove arbitrary-history preservation. Contraction with a sufficient
radius is one remedy; direct invariant-tube containment is another and does
not require foreign-write contraction.

Implemented the certificate in run_commit_seal.py (mode
single_cell_certificate; mpmath.iv 128-bit outward arithmetic; PyTorch
GRUCell gate convention; SiLU reader gradient enclosed). Smoke test on a
one-epoch cell exercised the inconclusive path only; it did not validate the
general PASS path. The evidence gate later identified endpoint-comparison,
reader-offset, fixed-point-enclosure, and serialization defects (fixed after
the run; not rerun).

Real run: seed 11 sealed, 200 epochs, 130 s. acc 1.000, erasure 1.4e-8.
Invariance PASS. kappa_hat over the full cube = 5.88 -> INCONCLUSIVE.
Witness search (refutation direction only, same domain/norm): confirmed
expansion pairs for both values (1.11, 1.17). So this is a genuine premise
failure on [-1,1]^16 in the inf-norm, not interval conservatism. Sampled
Jacobian inf-norm at fixed points: 0.987 (v=0), 0.539 (v=1).

Evidence gate (REVISE): the finite-panel result coexists with reported
interval-confirmed expansion on the fixed cube. Candidate cancellation is
compatible with these observations but was not demonstrated. Expansion does
not eliminate conservatism in the reported 5.88 estimate. A box enclosing
short histories, even with a margin, is not an invariant-domain certificate.
This attempt closes without another run; the wider behavioral-realization
question remains open. Apparatus ratio in this lane 5.71:1 -> lane closed.

---

### Audit #40 on CR-15.8: ADOPT core, REVISE guards and attribution (2026-09-05)

All proofs verified (loop cutting, right-to-left count k(n-k)+C(k,2),
endpoint chain with no factor two, fidelity corollary, both lower-bound
constructions, 3^k vs 2^k, packing theorem, irrational floor strengthened
to exact attainment D = N tau, ultrametric remark). Corrections applied:
parameter guards (N>=2, eps>0 for the sharpness examples; zero cases
separate), the binary example keeps its live bit exact (nuisance error
only), prior art (quantitative algebra: Mardare-Panangaden-Plotkin; metric
rewriting: Gavazzo-Di Florio; Moore; Girard-Pappas), status line, STATE
summary and next obligation.

Program ruling: continue; the next step is one useful behavioral error
bound on one fixed learned endogenous interface. A rigorous interval-
arithmetic contraction certificate on one trained ~17K-param GRU cell is a
legitimate "derivation needs a concrete check", becoming a campaign only if
it searches seeds/models/norms. Conditions: actual learned transition and
native reader, verified invariant domain and initialization coverage,
derivative-to-distance passage, reader calibration, every registered
action; proving kappa < 1 alone is insufficient; a failed certificate is
inconclusive, not evidence against overwrite. Pointer: Bonassi-Farina-
Scattolini GRU stability conditions (Lemma 1, Thm 2) to check first.

---

### CR-15.8: state budgets bound accumulated write-law error (2026-09-05)

Round 3 on audit #39's question. My attempt: loop-cut in the joint system
Q x P, then rewrite; I first counted the rewrites as quadratic, then
claimed linear via "each write crossed at most once per register". Codex
verified the loop-cutting lemma, refuted the unqualified crossing claim
(word r r s r: eliminating the second r first makes s cross two dead
writes), repaired it with a right-to-left procedure (rho <= k n), and then
found the stronger theorem: shorten the ENDPOINT CHAIN of the rewrite
proof, not the execution history. All chain states lie in K_alpha, so
cutting repeated vertices leaves <= |K_alpha| - 1 edges of size eps:
diam K_alpha <= min{1, (N-1) eps}. No |R| or |P| factor, no factor 2.
Lower bound Omega(min{1, N eps}) from the cyclic phase example with
faithful margin 1; exact (N-1) eps attained by two singleton registers on
a line. Tolerance version via packing number. Three suppliers of history-
uniformity: finite exact state, contraction, closed invariant relations.

My secondary lean was corrected on both counts: c_tau = 0 is not a
characterization of overwrite, and append-only carriers can have exact
last-write readers. Neither survives; recorded so they are not retried.

Codex round 3 first attempt hung 90+ min on bb_synthesis of the mission
board (too large); killed and relaunched with synthesis skipped. Memory
note written. Codex weekly quota at 66% used.

---

### Audit #39 on CR-15: REVISE, corrections propagated (2026-09-05)

Fresh unprimed audit. Core retained (15.0, 15.2, 15.3, 15.5, 15.7 ADOPT).
Revised: 15.1 (untouched-value example typing; the factorization is a
set-with-action isomorphism, not a metric product; it is a very-well-
behaved lens, Foster), 15.4 (zero-radius paragraph; calibration at mixed
centers; "selective forgetting" sentence softened), 15.6 (paired-pool lemma
now has explicit hypotheses; state-tracking prior art added: Merrill-
Sabharwal, Liu et al., Mamba, Illusion of State). Also fixed CR-4 proof
opening, CR-5 cardinality caveat, CR-9 exact-equivalence overreading,
CR-12 "TESTED" -> "REQUIRED EVIDENCE".

Withdrawn from every surface: "distance from claim 0" and "pretrained
transformers lack an endogenous overwrite boundary" (CEG-1 failed its
competence preflight; retained raw history is not retained behavioral
distinction). STATE, theory CR-10/CR-14, EXPERIMENTS CEG-1 edited.

Program ruling: continue selective behavioral reset; generic contraction
certification is not the highest-leverage next step. Next question: sharp
state-budget/error tradeoff turning eps-accurate local write laws into a
history-uniform store abstraction with no product coordinates or tubes.
My attempt (loop-cutting in Q x P then rewriting) gives C = O((N|P|)^2),
lower bound Omega(N) from the cyclic counter; round 3 asks for the sharp
exponent.

---

### CR-15 adopted: selective behavioral reset, representation and certificates (2026-09-05)

Compute-constrained analytic mode began today (directive: derivations and
proofs, tiny models only, recurring entropy sweeps). Two-round dialogue.

Round 1 lean (mine): make CEG-1/CRC-1 a theorem by bounding absorption
defect below by attention mass on dead tokens; gated overwrite needs a
saturated gate. Both refuted with explicit constructions: values and
projections can annihilate dead mass, and a scalar GRU with gates at 1/2
overwrites exactly on {-a,a} by candidate cancellation (verified
numerically). Adopted instead: common-kernel representation Q_sat = B x S,
and the invariant-set 2-eta theorem.

Counter-round (mine): the invariant-set theorem has no supplier; contraction
supplies it (CR-15.3); fibered contraction is the approximate shadow of
B x S (CR-15.4); conjecture that a shared gate cannot do two registers.

Round 2 verdicts: CR-15.3 repaired (the interval [-a,a] is NOT invariant for
the scalar construction despite |U'| <= 0.11, so a contraction certificate
needs a verified invariant domain and initialization coverage); CR-15.4
false as stated (consistent fixed points do not give commutation; defect
kappa(u-y)(x-v)), replaced by a selective-tube certificate; keyed-gate
conjecture false (constant gate, address in the candidate). Partial writes
reconcile the two CR-11 bounds. Corollary: selective memory requires
selective forgetting.

What this changes: the rung-1 experiment's claim-bearing object is now a
certificate (invariant domain + contraction + reader margin), not a paired-
erasure table. Erasure tables can pass for reasons that do not survive
history length (cyclic-counter counterexample).

Repo entropy sweep also completed today: 98 scripts, 47 configs, 134 result
dumps removed (manifest kept); STATE 3477->192 lines, NOTEBOOK 14614->845,
closed theory under theory/closed/, handoffs merged.

---

### CR-14 revised to commit-seal carrier (2026-09-04)

Codex design gate corrected the original CR-14 hard-register experimental
design as EAC tautological: hard-wired overwrite guarantees |Q_H|=16 by
construction, so testing whether it erases is circular. WB-1 validates the
math (|Q_H| matches ∏|V_r| = 4^2 = 16, the store cardinality for initialized
registers) but is architecturally tautological — a single-seed diagnostic,
not L1-L4 validation (Codex: learned write heads unused, overwrite arm retains
history-bearing GRU path, arms have unequal parameters).

Revised CR-14 uses commit-seal carrier: hard SEAL (information bottleneck
erasing pre-commit activations) + LEARNED update U(m, proposal) → m+.
The model CAN fail (encode both old/new values, entangle addresses, ignore
carrier). The learned update is the falsifiable object.

Also corrected: L1 renamed "absorption" (not "idempotence"). Quotient bound:
∏|V_r| for initialized registers, ∏(|V_r|+1) only when registers have a
distinct unwritten state.

Positive-control staircase: rung 1 = 1 address, 2 values, 2 commits,
counterfactual pairs, self-transplant + no-seal ablation + value-proposal
shuffle (not scrambled-address — impossible with 1 address).

Codex WB-2 design gate (compositional): strongest candidate is chained Z4
operations (A(x,y)=(x+y mod 4,y), B(x,y)=(x,x+y mod 4)) but it should run
at rung 3, after rungs 1 and 2 pass. Z4 is imported algebra (scaffolding),
not native latent math — the native candidate is the behaviorally established
congruence and induced transformation semigroup of the learned carrier.

---

### WB-1 complete: F4 — task too simple, overwrite hits exact bound (2026-09-04)

First constructed artifact. Two GRU models (~25K params, CPU), 2 registers × 4
values, 5-step sequences. Overwrite model: |Q_H|=16 EXACTLY (theoretical bound),
store alignment 1.000, 100% generalization to length 15. Append-only model:
|Q_H|=20, alignment 0.750, 99.9% generalization. Both compress ~2000:1.

F4 fires: the quotient ratio (1.25x) doesn't clear the 1.5x threshold. Simple
fact-tracking is too easy — a GRU+attention learns last-write extraction without
needing the overwrite boundary.

**Signal:** The 25% history leakage in append-only IS the architectural
difference showing through. 4 of 16 store states have history-dependent behavior
in the append-only model, vs zero in the overwrite model. The difference exists
but doesn't bite on simple tasks.

**Next:** WB-2 with compositional cross-register operations. The task must
require reasoning ACROSS registers, not just individual register reads.

---

### Pivot to construction: Selective Write-Boundary program (2026-09-04)

Codex direction dialogue: skip CEG-1R, close Qwen path-dependence line, pivot
to constructing a latent substrate with an explicit write boundary.

**Decision logic:** CEG-1R's outcomes both lead to the same next step
(construction), so running it is measurement entropy. Measurement-to-artifact
ratio was infinite (506:0), governance requires halt above 5:1. CEG-1R is not
literally the R^n trap but remains an outside-in measurement tunnel — it
characterizes what a surgeon can cut, not a move available to a denizen.

**Critical pushback adopted:** The write boundary alone is NOT the artifact
(repeats EAC tautology). It is a declared substrate axiom. The scientific
question is whether a learned denizen can USE it for portable, compositional
predictive states.

**Theory (CR-10 through CR-14):**
- Laws: overwrite idempotence (L1), cross-register commutativity (L2),
  preservation of unrelated distinctions (L3), write fidelity (L4)
- Theorem: every write history has a unique last-write normal form; predictive
  quotient bounded by store cardinality ∏(|V_r|+1), not history length
- Five pre-declared falsifiers (F1-F5), each testable in one round
- Primary gate (F1): overwrite model must produce a smaller quotient than
  matched append-only ablation

**Narrative:** "Give a model an internal act that genuinely means 'replace this
fact,' then test whether compositional reasoning becomes possible because dead
history is truly dead."

**Next:** Codex design gate on CR-14 experimental design, then implement the
smallest model that can be wrong (2 registers × 4 values, 5-step sequences).

---

### CEG-1 Evidence Gate: REVISE — bounded context dependence, not relay (2026-09-04)

Codex Evidence Gate (audit #38) returned REVISE / formal NO_INTERFACE. The
RELAY headline does not pass. Corrections adopted verbatim.

**Key corrections:**
1. RELAY-CUT Δ=0 is tautological — downstream positions see only pair-identical
   tokens. Not an empirical finding.
2. Under QUERY-CUT, register-specific dead-value effect disappears: target and
   orthogonal channels show equal excess (0.062 vs 0.061). Residual is broadcast
   context sensitivity, not semantic relay.
3. "52/48 pathway split" is invalid — used arm-specific channel maxima.
4. Competence preflight failed (83.6% < 95%); binding design required NO_INTERFACE.
5. Measurement-to-artifact ratio: infinite (506:0).

**What survives:** Paired excess (QUERY-CUT − DUMMY-CUT) = 0.101, bootstrap CI
[0.050, 0.152], p=0.001 at block level. A hard attention reset at first live
write recovers 100% accuracy — the append-only carrier lacks an endogenous
overwrite boundary. One canonical restatement is not an ε-synchronizer.

**Next:** CEG-1R (terminal Qwen localization test) or close Qwen path entirely.

---

### CEG-1 RESULT: RELAY (pre-audit) — dead info relayed through attention (2026-09-04)

914 forwards, 259s. Custom 4D attention masks, eager attention, full forward
passes. Fixture perfect (TV=0.00), mask effectiveness confirmed (TV=0.856).
Original RELAY verdict superseded by Evidence Gate above.

---

### CEG-1 design: Causal Erasure Graph v1 (2026-09-04)

**Direction pivot (Codex strategic review):** CRC closed on Qwen3-1.7B-Base.
Next instrument: position-controlled causal-erasure via attention mask
interventions. Binding question: when an assignment becomes semantically dead,
does its influence survive through direct KV access, relay into later states,
or because no legal continuation synchronizes the model?

**Design:** 16 fixed-slot histories (4 dead combos × 4 live stores) ×
4 rendering templates (2 calibration, 2 held-out). 6 intervention arms:
OPEN, QUERY-CUT, RELAY-CUT, DUMMY-CUT (on dummy-register histories),
LIVE-CUT (positive control), SYNCHRONIZER (append live-store restatement).
Custom 4D attention masks with eager attention, complete forward passes
(no KV cache). Fixture validation: OPEN mask must match standard inference
within TV 1e-5. ~900 CPU forwards.

**Predeclared interpretations:**
- QUERY-CUT collapses → dead info directly reread from old KV cells
- Only RELAY-CUT collapses → dead info relayed into later states
- SYNCHRONIZER collapses → denizen-executable reset word exists (→ reuse CRC)
- Neither works, LIVE-CUT does → distributed/redundant (stop Qwen line)

**Codex adversarial corrections adopted for STATE.md:** "Dead histories haunt"
replaced with licensed sentence; competence-failure overclaim corrected;
relation rates marked as confounded.

---

### CRC-1 result: INVALID / bounded negative (2026-09-04)

CRC-1 complete. 1508 forwards, 357s. Two-register overwrite semigroup on
Qwen3-1.7B-Base. Task-aligned {0,1} observer — the coarsest possible
observer that captures task semantics.

**Result:** INVALID (competence 87.5%). Identity partition (Q_1=32, zero
compression). Algebraic relations: idempotence 37.9%, absorption 20.7%,
commutation 19.1%. Min D_1 = 0.099 — not even close to merging.

**The binding finding:** Dead histories haunt the transformer. The model
treats `a=0; a=0` differently from `a=0` alone. It treats `a=0; a=1` differently
from just `a=1`. It treats `a=0; b=1` differently from `b=1; a=0`. The
overwritten fact, the redundant repetition, and the execution order of
independent operations — all are visible in the model's next-token prediction,
even through the coarsest binary observer.

**What this closes:** CRC on Qwen3-1.7B-Base. Two task families (SVB
scoping, binary register overwrite), two observers (full-vocab, binary).
Both identity partitions. The Codex stop condition is met.

**What this opens:** The path-dependence IS a structural finding. It's not
just "no compression" — it's a positive characterization: the model's
computation is history-sensitive in specific, measurable ways. The algebra
of actions (idempotence, absorption, commutation) fails at specific rates
that characterize HOW the model processes history.

**Competence failure analysis (post-hoc):** All 8/64 competence failures
share the same structure: the queried register's last write is NOT the final
line of the history. When it IS the last line: 32/32 = 100% correct. When
it's not: 24/32 = 75% correct. In every failure, the model predicts the
EARLIER (overwritten) value. This is a clean structural result: positional
recency dominates semantic state tracking. Intervening writes to the OTHER
register interfere with recall of the queried register's value.

**Cross-framework convergence:** The prior `predictive_setter_algebra_v1`
experiment (2026-08-31, natural language fact appending) found last_write_wins
= 27%, commutation = 56%. CRC-1 (2026-09-04, Python register assignments,
full CRC framework) found absorption = 20.7%, commutation = 19.1%. Two
completely independent methods, same model, same finding: **primacy bias +
non-commutation are structural properties of Qwen3-1.7B-Base**, not
artifacts of any single measurement approach.

**Next:** Codex strategic analysis for what to do after CRC closure.

---

### RE-CONTEXTUALIZATION: Two campaigns, zero positive results (2026-09-04)

**Project**: Latent-Space-Reasoning.
**Live question**: Does native mathematics exist in latent spaces, and what does it look like?

**Pattern across campaigns:**
1. SVB campaign (scalar projections): 8+ experiments, all structured negatives. Root cause: R^n trap — scalar logit projections discard structure.
2. CRC campaign (full distributions): escaped R^n, clean apparatus, but identity partition. The model is too competent — every state is unique, no room for compression.

**Alternative interpretations (anti-tunnel):**
1. **Task exhaustion, not framework failure.** CRC works mechanically. The SVB variable-scoping task at 100% competence simply has no degeneracy. A task where the model naturally confuses states (near capacity limit) would be the right substrate.
2. **"Almost compressible" is the signal.** Min D_1 = 0.0165 at eps=0.01. At eps=0.02, one pair merges. The model is on the boundary — presentations are almost equivalent. Better-matched presentations (same token count, same structure) might cross it.
3. **Compact quotient is the wrong shape.** Native math might not be a finite state machine. It could be continuous (differential structure on the probability simplex), topological (connected components under action), or information-geometric (Fisher metric on response distributions).
4. **Different model.** Qwen3-1.7B-Base may be too large / too capable for this task. A smaller model, or a model at its capacity limit on a harder task, would have forced compression.
5. **Different response channel.** We used next-token prediction only. Hidden-state similarity, attention pattern structure, or gradient-based probes would access different structure — but those are the R^n tools the project explicitly moved away from.
6. **The result IS the math.** The identity partition at horizon 1 with Q_0 compression at horizon 0 IS a mathematical fact about this model. The refinement tower Q_0(33) → Q_1(36) → Q_2(36) with specific split witnesses is a complete structural characterization. It's just not the compact structure we hoped for.

**What still holds:**
- The CRC framework is clean, decisive, and interpretable.
- Full-distribution analysis (not scalar projections) is the right approach.
- Mode F control (TV = 7e-7) confirms transformer KV cache is perfectly faithful.
- The model genuinely maintains 36 distinct behavioral states through depth-4 nested scope.

**What's reframed:**
- The SVB task family is doubly exhausted — even escaping R^n, it doesn't compress.
- The "almost compressible" boundary is a structural fact worth understanding, not just a near-miss.

**Awaiting:** Codex Architecture Theorist strategic analysis for next direction.

---

### CRC-0 result: FAIL — no compression (2026-09-04)

CRC-0 complete. 544 forwards, 210s CPU. Codex-designed: 36 roots (3 vars ×
4 vals × 3 presentations), 3 typed actions (N/H/R), exhaustive A^2 = 13
words per root, full 151K-vocab distribution, clique-verified classes.

Result: Q_0 = 33, Q_1 = 36, Q_2 = 36. No compression at any tolerance.
The model maintains a unique behavioral signature for every (var, val, pres)
triple. Three Q_0 pairs merged (presentation variants with TV < 0.01) but
all split at Q_1: the H action distinguishes presentations at TV~0.028,
the R action distinguishes cross-variable pairs at TV=0.111.

Transfer vacuously fails (all Q_1 classes are singletons). Mode F control
shows TV = 7e-7 (cache perfectly faithful). Verdict stable at all tolerances.

Key observation: min D_1 = 0.0165, just 65% above eps=0.01. The model is
ALMOST compressible — presentation variants nearly merge — but not quite.
This "almost" is itself interesting: the model treats different presentations
as nearly but not exactly equivalent, and even a neutral comment reveals the
difference.

This closes the SVB task family as a CRC substrate. The framework works
(clean, interpretable, decisive) but the task doesn't yield compression.
Next: Codex consultation on what task family WOULD yield nontrivial structure.

---

### PIVOT: Continuation-Refinement Calculus (2026-09-04)

Codex Architecture Theorist analysis (codex_post_svb_pivot.txt, ~12KB):
All SVB experiments projected Q_M onto scalars. The structured negatives
trace to discarding structure that later operations reveal. New direction:
study the calculus of distinguishing futures, not scalar distances.

Key objects: Sep(x,y) = set of continuations that prove states different.
Refinement tower Q_0 ← Q_1 ← ... (bounded-horizon equivalence classes).
Brzozowski derivative: how an action transforms separating languages.

Codex corrections to IDC framework:
1. Replace strict chain (Δ ⊂ ≡_M ⊂ ≡_E) with kernel comparison
2. Replace one-object monoid with graded typed process category
3. Make response-tree equality primary, demote D7 metric/Finsler geometry
4. Add finite-access refinement tower
5. Require finite predictive presentation for promotion

Important: congruence experiment didn't test Q_M (congruence by construction).
It tested a coarse measured quotient via δ_L and 11-bin projection.

Theory: theory/CONTINUATION_REFINEMENT.md (CR-1 through CR-9).
Next: zero-call reanalysis of congruence data, then one bounded CPU test.


### CURVATURE CONTROL: line closed — R^n trap (2026-09-04)

Confound-killing isochronous control round. 768 calls, 249s CPU. 6 templates,
2 shadow digits, 2 padding families. Position-matched via padding prepended to d3.

Results: G0-G3 PASS, G4-G6 FAIL. κ=−0.542 nat, 12/12 blocks negative, but
padding sensitivity (0.329 nat gap) exceeds 0.10 tolerance. 93/96 cells negative,
3 PAD_BARE positive. Position-gain null failed transport across template families.

Codex adjudication: real but qualified behavioral interaction, not intrinsic
curvature. Line CLOSED per one-round authorization.

Critical insight (Codex): "This line is not working as a route to native
latent-space mathematics: it remains scalar logit finite differences and OLS —
the R^n trap." All SVB-campaign experiments (congruence, S3 composition,
noncommutativity, curvature) produced structured negatives. The suffix effects
are learned semantic control — useful diagnostics, but R^n projections of
inherently non-R^n structure.

What's needed: a fundamentally different approach that works IN the latent space
natively, not by projecting it onto scalars and fitting linear models.


### CURVATURE TEST: sub-threshold, surface-specific interaction (2026-09-04)

Codex-designed 2×2 interventional test (depth × suffix). 288 calls, 98s.

d2→d3: heterogeneous (x positive, y negative, z positive — cancel in aggregate).
d3→d4: κ=-0.28 for tested pair, 24/24 cells same sign. Sub-threshold (0.28 < 0.5).

Codex adjudication (REVISE / CONTROL REQUIRED):
- Do NOT lower threshold retroactively. Descriptively nonzero, scientifically sub-threshold.
- Position is strongest rival: absolute token position changes with depth (38→58→79).
- Gate 1b cross-surface check: MISLEADING surfaces range -0.235 to +0.155 nat (mixed signs).
  Only 65.5% of 168 (assert, mislead) pairs show negative κ. NOT quotient-level.
- Variable-mentioning mislead surfaces (75% negative) vs non-var (52% negative).
- 24 deterministic cells ≠ 24 independent replications. One holdout variable insufficient.
- The SNR gate was never computed — "only magnitude failed" is formally false.
- Half the separation comes from ASSERT becoming MORE beneficial at d4.

Licensed: "On one preregistered SVB surface pair, the d3→d4 MISLEAD-minus-ASSERT
logit contrast was -0.284 nat in all 24 crossed cells; it missed the 0.5-nat gate
and remains confounded by position, shadow-token exposure, and surface choice."

What it advances: the fixed suffix contrast is context-dependent and cannot be
represented by the simplest globally additive logit-shift model. Candidate
obstruction and falsifier generator, not yet a law.

NEXT: One confound-killing control round. Position-matched, token-matched,
multi-surface, varied shadow digit. If residual curvature doesn't survive, close.

### S3 PERMUTATION COMPOSITION: model cannot compose (2026-09-04)

Codex-designed S3 experiment. Gate 1 FAIL: 4/36 correct (degenerate c021 bias).
The model cannot trace Python function calls for tuple permutations at 1.7B.
This is a competence failure, not an algebra failure. Behavioral testing of
compositional algebra requires tasks within the model's proven competence.

Key insight: behavioral bottleneck. Even if algebraic structure exists internally,
the model can't express it through code-tracing behavior. Two paths forward:
(a) compositional tasks with simple output (extend SVB), or (b) probe hidden
states directly (short GPU burst, bypasses behavioral bottleneck).

### CONGRUENCE TEST: behavioral equivalence is NOT algebraic (2026-09-04)

768 calls, 218s CPU. Codex-designed decisive test for native algebra.

3 equivalence pairs (surfaces with delta_L diff < 0.006):
- strong_assert: "# State unchanged." ~ "# {var} is unchanged." (TV=0.006)
- moderate_assert: "# Values preserved." ~ "# {var} is still intact." (TV=0.013)
- misleading: "# Changing {var}." ~ "# Updating to a new value." (TV=0.025)

Composed with neutral suffix "# No changes.":
- Equivalence PRESERVED (ratios 0.62-1.90x). Trivial.

Composed with misleading suffix "# Reassigning {var} now.":
- Equivalence DESTROYED (ratios 2.6-13.2x). Non-trivial.

Verdict: NO CONGRUENCE. The behavioral equivalence defined by delta_L is
surface-level. Composition with a semantically heavy suffix reveals differences
that our (C,L,R) quotient cannot see. The model internally distinguishes
surfaces that our scalar metric says are equivalent.

IMPLICATION: This is the Codex-predicted failure mode. The delta_L landscape
is learned semantic control, not native algebra. Per Codex: "close this line
as a useful hole — the latent space exposes stable semantic biases but no
context-independent navigation law."

ADDITIONAL FINDING: delta_L is predictable from text features (R2=0.88).
Simple binary features (has_preserve, has_change, is_rewrite, is_observe)
explain 88% of delta_L variance. This strengthens the "trained behavior"
interpretation.

WHAT THE SVB CAMPAIGN ESTABLISHED (positive):
1. Suffix effects are real and content-driven (6.3x content over confound)
2. The effect is a depth-invariant logit bias on shadow probability
3. ASSERT and MISLEADING are cleanly separable by delta_L (no overlap)
4. Quotient closure holds for single suffixes (100% ASSERT)
5. The quotient is NOT preserved under composition -> not algebra

NEXT: Switch task families. SVB is squeezed dry. Need a task where the
model's internal structure demands richer algebraic operations.

---

### QUOTIENT CLOSURE TEST: K_s stochastic DEAD, logit-shift WINS (2026-09-04)

Held-out prediction test on Gate 1b data (3,384 obs, 43 surfaces, no new model calls).

K_s (3×3 stochastic matrix on CLR simplex): 0/43 surfaces beat ANY baseline.
Mean held-out error: 0.290 vs best baseline 0.039. The stochastic matrix
overfits in probability space because the actual transformation is nonlinear.

Logit-shift baseline (constant δ in log-probability space): best model,
mean error 0.02 cross-depth. This IS the operator — not a matrix on the simplex.

KEY FINDING — depth-invariant shadow logit shift:
- δ_L (shadow digit logit shift) is depth-invariant: CV=0.036 for ASSERT, 0.053 for ASSERT_VAR
- δ_C (correct digit logit shift) is NOT depth-invariant: CV=0.39
- Each surface has a specific δ_L value, constant across depths d2-d4
- ASSERT: δ_L ranges from -0.94 to -2.54 across surfaces
- MISLEADING_ASSERT: δ_L ranges from +0.27 to -0.84

Response collision closure: 100% for ASSERT/ASSERT_VAR (quotient exists),
99% for MISLEADING roles, 62-70% for OBSERVE/REWRITE.

Cross-role transfer: FAILS. K trained on ASSERT→ASSERT_VAR: 0.32 vs
logit-shift baseline 0.02. Different surfaces within same role have
different δ_L values.

HONEST ASSESSMENT: The suffix effect is a surface-specific, depth-invariant
logit bias on shadow probability. This is structurally real but mathematically
simple — a lookup table, not an algebra. The 3×3 K_A operator was an artifact
of the wrong coordinate system (probability vs logit space).

---

### GATE 2 v2 MULTI-FILLER: CLOSED — POSITION_WITH_NONLINEAR_GAIN (2026-09-04)

Multi-filler robustness test with 3 independently selected filler pairs (all
token-matched). 288 new calls (pairs 2 & 3), reusing AM/MA and pair 1 from v2.
2.4 min CPU.

Results:
- Residual order contrast (pre-gain): all 3 pairs pass (medians 0.040, 0.067, 0.089)
- Cross-filler consistency: corr 0.69–0.92 (the order effect is real, not filler-specific)
- Scalar gain fit: ALL fail (post-gain medians 0.003, 0.007, 0.009 < 0.01)
- Pair 2 has 0/24 contexts above threshold post-gain

Verdict: The order effect is real and consistent across fillers, but it is fully
explained by position with context-dependent magnitude (nonlinear gain). There is
no genuine content interaction between A and M beyond position effects.

Gate 2 composition is now a STRUCTURED NEGATIVE. The surfaces A and M don't interact
noncommutatively — they contribute additive position-dependent effects whose magnitude
varies by context.

Strongest remaining finding: K_A correction operator (depth-invariant L→C ≈ 0.72).
This is independent of composition and describes a genuine operator equation.

Next: Codex strategy session on K_A as headline finding. Design held-out prediction
test for the atomic operator equation.

---

### CODEX EVIDENCE GATE: Gate 2 v2 downgraded to PROVISIONAL (2026-09-04)

Codex identified critical Gate 2 issue: implementation computed armwise prediction
error (mean=0.033), not the pre-registered residual order contrast. Corrected
diagnostics show filler-predicted order vectors align almost perfectly with observed
(cosine=0.996) but over-predict magnitude. Fitting one scalar gain per context
reduces mean residual TV to 0.008 (below threshold in 18/24 contexts).

Strongest rival: nonlinear position/recency gain or saturation. The order effect
is real (Gate 1 strong) but may be fully positional with nonlinear gain profile.

Repair plan: (1) freeze exact residual-order statistic and clustered lower bound,
(2) test nonlinear-gain rival with context-feature prediction, (3) multi-filler
robustness (multiple mechanically selected filler pairs), (4) held-out prefix family.

Also found: depth-invariant K_A correction rate (L->C ~= 0.72 across d2-d4).
This is a stable operator equation independent of the noncommutativity claim.

---

### GATE 2 v2: PROVISIONAL — direct order effect confirmed, excess disputed (2026-09-04)

216 calls, 8 arms per context (Codex filler-based design), 24 contexts, depth 4. 135s CPU.
Token-length-matched fillers: F_A="# Code block follows.\n" (5tok), F_M="# Looking at the next part.\n" (7tok).

**Results:**
- Direct TV(AM,MA): mean=0.072, median=0.068, max=0.141
- Excess beyond filler null (prob-space): mean=0.033
- Excess beyond filler null (log-ratio): mean=0.038
- Both pre-registered gates PASS (threshold 0.01)
- Verdict: NONCOMMUTATIVE_CONTROLLED

**Per-variable breakdown (log-ratio excess):**
- x: direct=0.040, excess=0.038 — nearly all genuine interaction
- y: direct=0.112, excess=0.026 — large effect but mostly position-dependent
- z: direct=0.063, excess=0.050 — mostly genuine interaction

**What we learned:**
1. A and M genuinely interact — the effect of applying A is modulated by M's presence
   and vice versa, beyond any additive position/recency model.
2. The interaction is variable-dependent: x and z contexts produce more genuine
   noncommutativity, while y contexts have a larger raw effect that's more position-dominated.
3. This is the first controlled evidence that suffix operators compose non-trivially.
   K_A o K_M != K_M o K_A even after subtracting position effects.
4. Robust across probability-space and log-ratio coordinate predictions.

**What's next:** Codex evidence gate running. Then: what algebraic relations do these
operators satisfy? Path from "operators don't commute" to "here is the native algebra."
Gate 3 (causal relay-state composition) remains the decisive experiment but needs GPU.

---

### GATE 2 v1: RECENCY_DOMINATED — MISLEADING class too variable (2026-09-04)

504 calls, 4 pairs (2 cross-role, 2 same-role controls), depth 4 only. 550s CPU.

**Results:**
- Cross-role TV: mean=0.085, median=0.081
- Same-role ASSERT: mean=0.006 (nearly commutative, despite 3.8× a_u range!)
- Same-role MISLEADING: mean=0.097 (highly noncommutative, 2.9× a_u range)
- Verdict: RECENCY_DOMINATED (cross-role median 0.081 < same-role max 0.130)

**What we learned:**
1. ASSERT class is remarkably orderly — "State unchanged" (a_u=0.10) and "No changes"
   (a_u=0.38) nearly commute (TV=0.006). The ASSERT operator is content-stable.
2. MISLEADING class is chaotic — "Changing {var}" (a_u=0.50) and "{var} gets reassigned"
   (a_u=1.44) are highly order-dependent (TV=0.097). The class is too heterogeneous.
3. Cross-role noncommutativity (0.081) is real but CANNOT be distinguished from the
   extreme within-MISLEADING order effects.

**Codex correction (received before experiment):** Same-role controls are the wrong
null model. They don't properly absorb the discounted-scalar rival s'=λs+δ_u.
Need token-length-matched neutral fillers and an additive interaction test.
Next: redesigned 192-call experiment with Codex's filler-based null model.

---

### GATE 1b COMPLETE: CONTENT DOMINATES, LUMPABILITY STRONG (2026-09-04)

3,384 calls across 8 roles × 3 depths × 3 vars × 8 values. ~100 min CPU.

**Key results:**
1. **2×2 ANOVA: content dominates at 6.3x** (content effect +0.0915 vs var-mention
   +0.0145). Per depth: d2=32x, d3=6x, d4=4.9x. Replicates on holdout. The
   +0.092 ASSERT→MISLEADING difference is overwhelmingly content-driven.

2. **Lumpability: STRONG (R2 0.79–0.98).** Contexts with similar baseline (C,L,R)
   produce similar post-suffix (C,L,R). K_u is approximately a well-defined
   operator on the simplex. BOUNDARY (0.98) and MISLEADING_ASSERT (0.96) are
   near-perfect. This means the (C,L,R) projection captures most of the information
   the model uses to determine post-suffix behavior.

3. **Surface equivalence: WEAK.** Only 1 cross-role pair in class A passes at
   eps_eq=0.01 (ASSERT:"State unchanged" ↔ ASSERT_VAR:"{var} is unchanged",
   max TV=0.009). Class M has zero qualifying pairs. The multiplicative character
   a_u is surface-specific (within-class CV 0.33–0.51), not class-invariant.

4. **K_a vs logit-bias: INCONCLUSIVE.** Cross-fit has no holdout data. R-preservation
   tentatively favors K_a (ASSERT dR=−0.007, MISLEADING dR=+0.003, both small).

**Implications for Gate 2:** Content dominance unblocks noncommutativity testing
but surface equivalence constrains it. For class A we have one validated pair.
For class M we need either a relaxed threshold or different surfaces. The
lumpability result is the strongest positive: it means the (C,L,R) simplex IS
the right coordinate system for studying suffix operators.

---

### CODEX NATIVE MATH THEORY FOUNDATIONS (2026-09-04)

Codex Architecture Theorist (session 01a06c67) on what native latent-space
math IS and what noncommutativity actually proves.

**Central verdict: "Noncommutativity is cheap; stable equations are expensive."**
The transformation monoid exists for ANY sequential transducer. The native prize
is model-discovered, representative-independent algebra with causal realization.

**What noncommutativity proves (and doesn't):**
- PROVES: obstruction to specified commutative factorizations (K_a, logit-bias)
- DOES NOT PROVE: dimension >1, nonlinearity, lumpability failure, compression,
  proof-relevance, topological holes, transformer-specificity, or compositional
  prediction. Cross-role > same-role TV doesn't even rule out recency — position-
  matched singleton+filler arms are required.

**The right framework:** Typed causal process category C_M = Q_M ⋊ I. Objects =
predictive causal states at layer-position cuts. Arrows = model computations.
External denotation is a functor. EqWit = vertical arrows invisible to denotation.
Commuting/failing squares between layer transports and behavioral operations are
where architecture enters.

**The decisive experiment: "causal relay-state composition"**
1. Multiple surfaces per role, assigned to nonce codes (so physical code can't
   predict role)
2. Define relay-state equivalence by causal substitution at declared layer cut
3. Mask original tokens downstream — test compiled state vs rereading
4. Predict unseen compositions from atomic transition table
5. Transplant relay-token state causally — must follow class + composition table
6. Practical control: use predicted class to correct degraded scope answer

**Directed homotopy:** Still viable but EqWit ≅ π₁↑ NOT justified. Needs:
verified execution squares (empirical 2-cells), directed graph/cubical complex,
demonstrated inequivalent parallel paths with same endpoints. AB≠BA alone
supplies neither equal endpoints nor absence of homotopy.

**Native math manifesto axioms:** Operational identity (future-experiment
substitutability), causal action (admissible model computations), presentation
invariance (surface/coordinate changes), architecture locality (layer-position
causal graph). First theorems: predictive-state minimality, kernel/fiber,
causal-realization, independence (commutation from disjoint causal cones),
obstruction (failed square forbids coarse factorization).

**Implication for current program:** Gate 1b → Gate 2 composition path is valid
but NARROW. The composition test can reject K_a/logit-bias but cannot establish
native math alone. The full program needs the causal relay-state experiment
(Gate 3+), which requires GPU and internal-state access.

### CODEX LOGIT-BIAS RIVAL REVIEW + COMPOSITION TEST DESIGN (2026-09-04)

Codex Architecture Theorist follow-up (session 01a06c48) on how to defeat the
logit-bias rival. Key findings:

**1. Content dominance ≠ rival defeat.** Even if Gate 1b shows content >> var-mention,
a content-specific logit bias δ_true vs δ_misleading is fully consistent. Content
dominance rejects the narrow "mentions-variable" hypothesis but not the logit bias.

**2. Analyzer Step 4 was tautological.** The K_a vs logit-bias comparison fitted a_u
and δ_u per cell and "predicted" the same cell. FIXED: now cross-fits one parameter
per role on train split, predicts holdout. Reports full (C,L,R) prediction errors.

**3. R-preservation is testable.** K_a predicts ΔC=-ΔL, ΔR=0. Logit-bias predicts
all non-shadow probs scale by 1/Z where Z=1+(b-1)L, so R'=R/Z≠R and R'/C'=R/C preserved.
For b≈4, L≈0.05, R≈0.25: expected |ΔR| ≈ 0.026. Gate 1b data can adjudicate.

**4. Noncommutativity criterion — the strongest discriminator.** K_a matrices ALSO
commute (upper-triangular with same diagonal). So noncommutativity defeats BOTH logit-bias
AND K_a simultaneously. If α(u)∘α(v) ≠ α(v)∘α(u), the monoid is genuinely non-scalar.

**Actions:** Analyzer Step 4 rewritten with cross-fitting. Noncommutativity criterion
added to theory/closed/INTENSIONAL_DESCENT.md. Composition runner (run_composition) added to
run_semantic_descent.py. Config svb_qwen3_composition.json created: 120 calls, ~2.5 min.
Gate 1b ~20 min from completion; composition test runs after.

### CODEX NATIVE MATH ASSESSMENT: χ demoted, logit-bias rival identified (2026-09-04)

Two independent Codex reviews (Architecture Theorist + Novelty Challenger) on whether
the multiplicative character constitutes native latent-space math.

**Verdict: NOT YET NATIVE.** K_a is ordinary linear algebra on a coarse output simplex
(in (S,L,R) coordinates it's just diag(1,a,1)). The multiplicative character χ is
standard representation theory. Both are R^n scaffolding, not native structure.

**Key corrections adopted:**
1. Bayesian interpretation was WRONG: E(u)=id does not imply likelihood ratio=1.
   A denotationally null operation can still be evidence about the answer/task.
   Likelihood ratios multiply ODDS, not raw probability L.
2. **Strongest rival: constant shadow-logit bias.** A suffix producing additive
   logit shift δ_u gives L'/C' = e^{δ_u} × L/C exactly, and L'/L ≈ e^{δ_u} when
   L is small. Setting δ_u ≈ log(4) reproduces the 4x ratio with no EqWit needed.
3. Discriminating tests: K_a predicts R'=R; logit-bias predicts R changes via
   normalization. K_a predicts stable raw ratio; logit-bias predicts stable log-odds.

**What IS potentially native:** The operational quotient — denotationally null histories
becoming different moves because they alter future possibilities. The behavioral action
α and its path algebra, not K_a or χ by themselves.

**Actions taken:** Theory Bayesian section corrected. K_a vs logit-bias comparison added
to character analyzer. χ demoted to diagnostic candidate. Idempotent config created.

### CODEX FOLLOW-UP: Idempotent test + multiplicative character theory (2026-09-04)

Codex Architecture Theorist follow-up on two questions from Gate 1 results.

**1. Idempotent test — newline FAILS:**
Zero-call screen from existing Gate 1 data: b="\n", b²="\n\n", b³="\n\n\n".
TV(b,b²)=0.003 (near-equivalent), TV(b²,b³)=0.027 (clearly separated).
True idempotence b²=b ⟹ b³=b²=b, but b³ departs. Transient plateau,
not stable monoid relation. Cheapest next test: `pass\n` at d4 only,
48 calls. If that fails, `type(x)` for another 48.

**2. Multiplicative 4x ratio — observable constraint, not operator:**
The stable MISLEADING/ASSERT L ratio (~4x across depths) constrains only
ℓ∘K_M and ℓ∘K_A on three aggregated inputs. Does NOT imply K_M = 4·K_A.
Codex derives the consistent operator family: K_a(C,L,R) = (C+(1-a)L, aL, R),
composing as K_a·K_b = K_{ab}. If confirmed, χ: EqWit → (R≥0, ×) would be
a multiplicative character (one-dimensional representation of the monoid).

**Promotion ladder** (each step requires the previous):
1. Stable cellwise ratio (not just depth means)
2. R-preservation + S-preservation (coordinate action)
3. Surface descent (paraphrases in same class yield same a_u)
4. Composition: a_{uv} ≈ a_u·a_v on unseen compositions

Gate 1b BASELINE role enables step 1 testing. Steps 3-4 need further experiments.

Theory updated: transformation monoid + denotational grading + multiplicative
character section added to INTENSIONAL_DESCENT.md. Committed and pushed.

### GATE 1 COMPLETE + CONFOUND DIAGNOSTIC + GATE 1b LAUNCHED (2026-09-04)

Gate 1 re-run completed (2835 calls, 31.6 min — slower due to duplicate process contention).
Full analysis including holdout set, per-depth breakdown, and confound diagnostic.

**Decisive test (all data):** ASSERT L=0.032, MISLEADING_ASSERT L=0.124, **+0.092 diff (t=21.23)**.
**Holdout replication:** +0.121 diff. Effect is LARGER on held-out surfaces.
**Per-depth scaling:** d2 +0.047, d3 +0.105, d4 +0.126. Content effect grows with depth.

**Confound diagnostic:** MIXED. Content effect (misleading_novar - assert) = +0.061.
Var-mention effect (misleading_var - misleading_novar) = +0.051. Both comparable (~1.2:1).
"Updating the value" (no var mention) has L=0.097 vs assert 0.036 = genuine content effect.
But var-mentioning surfaces average L=0.148, adding ~50% more.

**Gate 1b launched** (svb_qwen3_gate1b.json): 2x2 content × var-mention crossover with
BASELINE role for F8 lumpability. ~3096 calls, ~19 min on CPU. Running now.

### GATE 1 PRELIMINARY: Content sensitivity confirmed, form/content confound identified (2026-09-04)

Gate 1 experiment completed all 3 depths (2835 calls, ~17 min on CPU).

Train-set results (n per role: ASSERT=486, MISLEADING_ASSERT=324, REWRITE=486,
OBSERVE=324, BOUNDARY=324):

| Role               | C_mean | L_mean | R_mean |
|---|---|---|---|
| ASSERT             | 0.9395 | 0.0342 | 0.0263 |
| MISLEADING_ASSERT  | 0.8493 | 0.1138 | 0.0369 |
| REWRITE            | 0.5235 | 0.4244 | 0.0521 |
| OBSERVE            | 0.6473 | 0.3292 | 0.0235 |
| BOUNDARY           | 0.8191 | 0.1505 | 0.0304 |

DECISIVE TEST: MISLEADING_ASSERT L=0.114 vs ASSERT L=0.034 = +0.080 difference.
The model follows CONTENT, not just form: misleading comments (that say they're
rewriting but denotationally don't) get 3.3x higher shadow leakage than true
assertions of invariance.

Between/within ratio = 1.17 (weak overall). Codex follow-up identified why:
only ASSERT vs MISLEADING_ASSERT genuinely crosses form and content. REWRITE,
OBSERVE, BOUNDARY are confounded with their statement types (assignment,
expression, delimiter). The five-class null-witness claim isn't supportable
from this design alone.

Codex V2 corrections applied to theory/closed/INTENSIONAL_DESCENT.md:
- D10 composition order matched to D1 convention
- D11 congruence proof: triangle decomposition added
- Removed residual |I|<|P|
- F7 revised: what may be artifactual is algebraic structure, not coarse distinctions

Codex F6/F8 findings:
- F6 needs nested ladder: M_state < M_cheap < M_cheap+role (variance comparison inadequate)
- F8 (lumpability): Gate 1 data alone insufficient; need empty-suffix baseline responses
- Exclude val=9 from all (C,L,R) analyses (shadow/correct digit collision)
- Current design supports narrow comment-content result, not five-class claim
- Need balanced role-by-realization crossover for stronger Gate 1

Re-running experiment with Unicode fix (crashed on Windows cp1252 arrow character).
Awaiting full results including holdout set and per-depth breakdown.

### STRATEGIC PIVOT: Intensional Descent Criterion (2026-09-04)

Codex Architecture Theorist synthesis delivered a clear verdict: the coefficient
cartography (suffix counts, attenuation estimates, simplex analysis) is standard
R^n mathematics — necessary scaffolding but NOT the native math. The native math
is the EQUALITY-WITNESS ALGEBRA: the structure of how denotationally null moves
(things that "do nothing" extensionally) have nontrivial, compositional, causal
model actions.

Formal framework written: theory/closed/INTENSIONAL_DESCENT.md
- D10-D15 extend D1-D7 with external denotational category
- P → I → D factorization: model factors through intensional quotient but NOT
  through extensional denotation
- Equality witnesses: classes of "doing nothing" the model distinguishes
- Witness transport: context-indexed operators on denotational fibers
- Five exact falsifiers (F1-F5)

Biggest credible claim: "Pretrained transformers implement proof-relevant causal
semantics — they compress execution histories into equality-witness classes
strictly finer than denotational state, compositional, and causally controlling."

Gate 1 (Semantic Descent) experiment designed and launched:
- 5 intensional roles × ~7 surface realizations × 3 vars × 9 vals × 3 depths
- Critical test: MISLEADING_ASSERT (comments describing rewrites) vs true ASSERT
- If content > form: intensional descent. If form > content: lexical cueing.

Codex reviewing the theorem attempt in parallel.

### DEPTH-5 REVEALS L SATURATION: shadow leakage has finite capacity (2026-09-04)

Depth-5 prediction experiment (P4) shows the exponential depth model BREAKS:
- L(4)=0.234, L(5)=0.244 — only 4% increase (was x3.0, x2.2, x1.2 at d1-d4)
- L saturates at ~24%. Model has finite scope-confusion capacity.
- R grows instead (0.035→0.047) — general noise replaces scope confusion.
- a_c weakens to 0.43 (from 0.38); R invariance breaks (|dR|=0.007).
- Recovery fraction stable at 0.49 (consistent d2-d5).

Operator family formalized: R-preserving subfamily K_a forms abelian monoid.
Semantic binding signal beta orders suffixes on a one-dimensional scale.
Fisher distance increases with depth (operator does more work where needed).
K_a * K_b = K_{ab} algebraically, but empirically one-shot (not iterable).

New predictions P5-P8 registered (semantic equivalence, semantic opposition,
cross-model content/boundary decomposition, attention pattern selectivity).

### CONTENT-SPECIFICITY CONFIRMED (P2): the suffix is a semantic intervention, not a boundary signal (2026-09-04)

Suffix ablation experiment (6 conditions, 756 calls, 237s CPU) tests whether
any code-like token provides the ~62% shadow suppression, or if "# No changes.\n"
is special. Result: content IS specific.

Two-component model:
- Generic boundary signal (~17%): any code token (pass, newline, short comment)
  provides modest suppression (a_c ~ 0.82). This is the "boundary appeared" cue.
- Content-specific "no changes" signal (~45% additional): the semantic content
  carries the dominant disambiguation signal.
- Self-assignment "x = x" actively INTERFERES (a_c > 1.6, mass flows C -> L).

R-preservation hierarchy: "# No changes.\n" is the ONLY suffix that is both
strongly suppressive AND R-preserving (|dR| < 0.002). Other suffixes partially
disturb R. This selects a distinguished kernel from the operator family.

Codex review (switched controlled transducer formalization): GO on controlled
fiberwise response-law synchronization. NO-GO on declaring native coordinates
or full predictive-state synchronizer yet. Key corrections adopted:
- (C,L,R) is R^3 subset; K is standard stochastic linear map
- The native contribution is the operationally induced quotient, not the matrix
- (C,L,R) equality is not predictive-state equality (R collapses bins)
- Call it "one-probe fiber synchronizer" or "registered response-law synchronizer"
- Obs. 6 "fresh readout" / "look twice" interpretations marked SUPERSEDED

Gossip version (Codex): "A comment doesn't give the transformer more time —
it tells it which overwritten value to stop believing."

Next: depth-5 prediction (P4), nested vs flat control (P3), Codex on the
theoretical implications of content-specificity and the operator family.

### SHADOW TRACKING CONFIRMED: boundary-conditioned attenuation is a genuine scope mechanism (2026-09-04)

Reanalysis of per-cell 11-bin distributions revealed the suffix effect is NOT a
"fresh readout pathway" or "mixture model." It is specific suppression of the
dominant shadow-binding competitor. The shadow relabeling experiment (3 conditions:
shadow9/2/5, 324 calls, 95s CPU) confirms the mechanism tracks the actual shadow
digit under relabeling — not hardcoded to digit 9.

Key results:
- The attenuation law: L(d,1) = a_c · L(d,0), a_c ≈ 0.38 at d≥2
- R (non-shadow, non-correct mass) is invariant (<0.002 change)
- Scope-relevant mass (C+L) is invariant — the model already knows the
  answer is {target, shadow}; the suffix resolves the disambiguation
- Scope-pair entropy reduction: 0.08-0.33 bits (42% at d≥2)
- The operator is one-shot, not iterable (s>1 adds interference)
- Within-binding variation contracts 25-59% (synchronization signature)

Mathematical object: T_suffix: (C,L,R) → (C+(1-a_c)L, a_c·L, R) is a linear
operator on the response-law simplex. One-parameter family indexed by a_c.
Preserves scope-relevant mass, contracts shadow leakage, invariant residual.

Next: suffix content sensitivity (P2), nested vs flat control (P3), Codex
follow-up on predictive-state transducer framework.

Older entries (2026-08-27 through 2026-09-03) were moved verbatim to docs/NOTEBOOK_ARCHIVE_2026_08_27_to_09_03.md.
