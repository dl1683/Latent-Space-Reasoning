# Cross-Domain Research Synthesis: Mechanisms Beyond AI

## Purpose
Identify mechanisms from physics, neuroscience, signal processing, computational chemistry, dynamical systems, and control theory that could inspire the next evolution of latent-space reasoning — or fundamentally challenge it.

## Codex Review Status

### Round 1 (2026-04-15)
- Analogy ratings assigned: 3 MISLEADING (VR, OGY, Phase Transitions), 4 SUGGESTIVE, 1 RIGOROUS-as-tool
- Missing domain: inference-time search / best-of-N is the primary comparison
- Paper framing warning: analogies are idea generators, NOT paper framing

### Round 2 (2026-04-15) — CONVERGENT
- Domains 8-14 "mostly honest," specific overreach corrections applied
- **Paper formal frame**: Ensemble theory (diversity-quality tradeoff, error correlation, selector ceiling)
- **Priority reordered**: Temperature comparison FIRST (fastest novelty check), gated attention SECOND
- **Position-control ablation**: MANDATORY (not "if feasible")
- **Single remaining gap**: Causal decomposition of gain into position-shift / sink-disruption / trajectory-diversity / selector-quality
- **Pre-implementation measurement contract**: Added below

---

## CRITICAL THREAT: Gated Attention Eliminates Attention Sinks (NeurIPS 2025 Best Paper)

### The Finding
Qwen team (Alibaba) won NeurIPS 2025 Best Paper for "Gated Attention for Large Language Models." A simple sigmoid gate after SDPA eliminates attention sinks entirely. Already deployed in **Qwen3-Next** (the next generation of the exact model family we use).

### Why This Threatens Our Mechanism
Our strongest effect — attention sink rescue in planning (14 words → 650 words) — directly depends on attention sinks existing. If Qwen3-Next eliminates sinks architecturally:
- The "rescue" mechanism disappears
- Our non-monotonic dose-response may flatten (if the peak at 2 tokens is sink-disruption-dependent)
- The entire framing of "perturbation breaks pathological attractor" becomes invalid for next-gen models

### Adversarial Assessment
**Best case**: Our mechanism has TWO components: (1) attention sink disruption + (2) trajectory diversification. If (2) is independent of (1), perturbation still works on sink-free models by diversifying greedy trajectories. Direction-agnostic improvement would persist.

**Worst case**: The effect is ENTIRELY attention-sink-dependent. On sink-free models, random prefix = random degradation. Our 19.6pp improvement becomes null. All cross-domain results (legal, planning) also null.

**Required experiment** (Phase 5.5 — Strategic Probe): Run Phase A on a gated-attention model (Qwen3-Next if available, or train a small gated-attention model). If the effect vanishes, our mechanism has a shelf life and we must pivot to trajectory diversification that doesn't depend on architectural pathology.

**Timeline urgency**: Qwen3-Next is already deployed. If others test our mechanism on it and find null results, our paper's framing collapses before publication. We MUST test this before submitting.

Sources:
- [NeurIPS 2025 Best Paper: Gated Attention](https://arxiv.org/abs/2505.06708)
- [Attention Sink Survey (180+ papers)](https://arxiv.org/abs/2604.10098)
- [Attention Sinks as Hallucination Signals](https://arxiv.org/html/2604.10697)

---

## Domain 1: Stochastic Resonance (Neuroscience/Physics)

### The Analogy
Stochastic resonance (SR) is the phenomenon where optimal noise ENHANCES signal detection in nonlinear systems. Our non-monotonic dose-response (peak at 2 tokens, degradation at 3+) is the hallmark of SR: too little noise = no effect, optimal noise = maximum enhancement, too much noise = masking.

### Key Insights for Our System
1. **SR requires a threshold nonlinearity**: In biological systems, it's the neural firing threshold. In our system, the threshold may be softmax attention concentration — when attention on the first tokens exceeds a critical fraction, the model locks into a degenerate trajectory. Noise near this threshold breaks the lock.

2. **Optimal noise level is system-dependent**: Different neurons/sensors have different SR optima. Our finding that 2-token peak is model-specific (4B vs 8B vs DeepSeek all behave differently) is consistent with SR theory.

3. **Multi-frequency SR (stochastic multiresonance)**: Recent 2024 research found FOUR local maxima at FOUR optimal noise levels in small-world neural networks. This suggests our dose-response may have additional peaks at higher token counts or different RMS scales that we haven't tested.

4. **Colored noise outperforms white noise**: SR with colored (correlated) noise outperforms white noise in some biological systems. Our random Gaussian prefix is white noise. Correlated noise (e.g., autoregressive noise, or noise drawn from the model's own embedding distribution) might work better.

### Transcranial Random Noise Stimulation (tRNS)
The closest biological parallel to our mechanism: random electrical noise applied externally to the brain enhances neural processing. tRNS works via:
- Summation between endogenous and exogenous fluctuations
- Neuronal synchronization enhancement
- Prolonging sodium channel opening → altered excitability

**Analogy to our system**: Random embedding prefix = exogenous noise that sums with endogenous embedding patterns, altering the model's "excitability" (tendency to explore vs. converge).

### Adversarial Check
SR requires a threshold nonlinearity and a subthreshold signal. Is our system truly subthreshold? If the model can already solve the problem (answer-anywhere = 80% for Qwen3-4B), the "signal" isn't subthreshold — it's already detected but poorly routed. This would make our mechanism NOT stochastic resonance but rather noise-induced basin selection, which is a different (though related) phenomenon.

### Novel Ideas
- **Adaptive noise based on SR theory**: Calibrate RMS to the model's "threshold" — the attention concentration level at which trajectories bifurcate. This is model-specific and could be measured.
- **Colored prefix noise**: Draw prefix vectors from the model's own embedding covariance matrix, not from isotropic Gaussian. Test whether structured noise outperforms white noise.
- **Multi-token SR mapping**: Test token counts 1-16 with finer RMS gradation to check for additional SR peaks.

Sources:
- [Stochastic Resonance in Sensory Systems](https://www.sciencedirect.com/science/article/pii/S1388245724002025)
- [SR Married to Neuroscience](https://link.springer.com/chapter/10.1007/978-3-032-00815-2_28)
- [Stochastic Multiresonance in Neural Networks](https://www.nature.com/articles/s41598-024-55997-4)
- [tRNS: Using Noise for the Better](https://www.sciencedirect.com/science/article/pii/S0149763422001919)

---

## Domain 2: Vibrational Resonance (Physics)

### The Analogy
Vibrational resonance (VR) is the deterministic cousin of SR: a weak low-frequency signal is amplified by a high-frequency auxiliary drive in a nonlinear (typically bistable) system. Unlike SR (random noise), VR uses a structured, deterministic high-frequency signal.

### Key Insight
Our evolved soft prompts may be closer to VR than SR: the evolution loop tries to find structured (not random) perturbations that amplify the model's weak internal signal (latent knowledge it can't access). Random perturbation = SR. Evolved perturbation = VR.

But our empirical finding is that random = evolved (p=1.0). This suggests either:
- The system is not bistable enough for VR to outperform SR
- Our evolution loop is too weak to find the VR signal
- The regime we're operating in is genuinely noise-dominated (SR), not signal-dominated (VR)

### Novel Ideas
- **Deterministic prefix patterns**: Instead of random noise, test periodic patterns (sinusoidal in embedding space), chirps (frequency-sweeping), or impulses. If VR is operative, structured signals will outperform random noise.
- **Bistability mapping**: Map the model's generation landscape to identify genuinely bistable regions (two distinct trajectory classes with a barrier between them). If the landscape is multi-modal (not just bistable), VR theory doesn't directly apply.

### Adversarial Check
VR requires the system to be genuinely bistable with a clear barrier. If the model's trajectory landscape has dozens of attractors (not two), VR is the wrong frame. Our data shows at least 19/25 "sensitive" tasks (different seeds → different outcomes) — this looks more like a multi-attractor landscape than a bistable system.

Sources:
- [Vibrational Resonance via Single-Ion Phonon Laser](https://journal.hep.com.cn/fop/EN/10.15302/frontphys.2025.012203)
- [Weak Signal Enhancement by Nonlinear Resonance](https://www.nature.com/articles/s41467-020-15827-3)

---

## Domain 3: Basin Hopping (Computational Chemistry)

### The Analogy
Basin hopping (BH) is a global optimization method: random perturbation + local minimization, iterated. It finds global minima on complex energy landscapes by hopping between local basins. Our mechanism is inference-time BH: random prefix = perturbation, greedy decoding = "local minimization" (deterministic descent from perturbed initial conditions).

### Key Insights
1. **Adaptive step size**: Modern BH adjusts perturbation magnitude based on acceptance rate. If too many perturbations are rejected (output quality degrades), reduce energy. If too few, increase energy. We don't do this — we use fixed RMS.

2. **Parallel evaluation**: 2025 advances show near-linear speedup with 8 concurrent candidates. Our Phase A pipeline already evaluates N=10 in parallel.

3. **Basin revisit prevention**: Equality-graph-based BH prevents revisiting known basins. We have no mechanism to ensure diversity among our N candidates — different seeds could produce the same trajectory class.

4. **Adaptive BH with RL**: 2025 work combines BH with reinforcement learning for temperature scheduling. Could learn the optimal RMS schedule per model/task.

### Novel Ideas
- **Diversity-enforced prefix set**: Instead of N independent random seeds, use seeds that are enforced to be dissimilar in early trajectory (e.g., reject any candidate whose first 16 tokens overlap >80% with an already-generated candidate). This addresses the "basin revisit" problem.
- **Adaptive RMS via acceptance rate**: Track what fraction of N candidates produce "acceptable" outputs (not truncated, not degenerate). If <30% are acceptable, reduce RMS. If >80%, increase RMS. This auto-calibrates the perturbation energy.

### Adversarial Check
Basin hopping requires local minimization between hops. In our system, greedy decoding IS the "local minimization" — it deterministically descends from the perturbed initial state. But there's no ability to iteratively improve within a trajectory (it's one-shot). This means we're doing BH without the "hopping" — just "basin sampling." The improvement over random search depends on the landscape topology.

Sources:
- [Adaptive Basin Hopping (2025)](https://arxiv.org/html/2510.25938)
- [Global Optimization Review (2025)](https://onlinelibrary.wiley.com/doi/10.1002/jcc.70243)

---

## Domain 4: OGY Chaos Control (Nonlinear Dynamics)

### The Analogy
The OGY method stabilizes unstable periodic orbits in chaotic systems using small, precisely-timed perturbations. Our system applies a perturbation at time 0 (the prefix) and hopes it lands the trajectory near a good orbit. OGY would instead wait for the model to naturally approach a good trajectory, then nudge it.

### Key Insight
OGY is CLOSED-LOOP: observe the system state, compute the intervention. Our system is OPEN-LOOP: apply random perturbation, hope for the best. Phase B (Observer-Router) is trying to add the closed-loop element — observe early trajectory, then decide whether to continue. But even Phase B is "observe then decide to continue/abort," not "observe and intervene mid-trajectory."

### Novel Idea: Mid-Generation Intervention
Instead of only prepending prefix tokens:
1. Start generation normally (no prefix)
2. Monitor attention sink mass at each step
3. When sink mass exceeds a threshold (the model is about to lock), inject a perturbation into the residual stream at that specific layer/step
4. This is "OGY for transformers" — targeted intervention at the moment of trajectory bifurcation

### Adversarial Check
- Autoregressive generation is sequential — there's no simple way to "inject" a perturbation mid-generation without breaking the KV cache or restarting generation
- OGY requires the system's Jacobian (linearization near the orbit), which we don't have for an LLM's internal dynamics
- The practical implementation would require hooks into the model's forward pass at each generation step, which is model-specific and fragile

### But: We Already Have the Hooks
`decode/steering.py` has `IntermediateLayerSteering` that injects vectors into the residual stream at specific layers. This IS mid-generation intervention capability. It's just not used in the validated pipeline.

Sources:
- [OGY Method and Chaos Control](https://en.wikipedia.org/wiki/Control_of_chaos)
- [Data-Driven Stabilization of Unstable Orbits](https://arxiv.org/html/2507.08630)
- [Attractor Metadynamics in Slow-Fast Systems](https://ar5iv.labs.arxiv.org/html/1611.00174)

---

## Domain 5: Dithering in Signal Processing

### The Analogy
Dithering adds small random noise before quantization to decorrelate quantization error from the signal. Our finding that quantization modulates the perturbation effect (8-bit works, 4-bit doesn't for Qwen3-8B) directly maps: the soft prefix may be acting as a dither signal for the quantized model.

### Key Insight
In audio dithering: noise at approximately 1 LSB (least significant bit) optimally decorrelates quantization artifacts. Below 1 LSB = no effect. Above = noise dominates signal.

In our system: RMS ≈ 0.022 may be approximately 1 "LSB" of the embedding representation at 8-bit quantization. At 4-bit, the quantization steps are larger, so our fixed RMS is below 1 LSB and can't dither effectively.

### Novel Idea: Quantization-Adaptive Dithering
- Measure the effective "step size" of the quantized embedding representation
- Set prefix RMS proportional to this step size (1× for optimal dithering)
- Predict: 4-bit models need LARGER RMS than 8-bit models to achieve the same effect
- This is directly testable and would explain the quantization interaction

### Adversarial Check
Dithering works for static quantization of a signal. But transformer inference is a dynamic process — each step's activations flow through quantized weights differently. The "LSB" of the system isn't a fixed quantity. Also, in BitsAndBytes quantization, the embedding layer is typically NOT quantized (float16), so the prefix isn't dithering the embeddings — it's influencing how quantized attention layers process the embedding.

Sources:
- [Dithering and Noise Shaping in Digital Audio](https://digitalsoundandmusic.com/5-3-7-the-mathematics-of-dithering-and-noise-shaping/)
- [Quantization Noise Reduction Methods](https://blog.sivo.it.com/signal-processing/how-to-reduce-quantization-noise/)

---

## Domain 6: Noise-Induced Phase Transitions (Statistical Mechanics)

### The Analogy
In far-from-equilibrium systems, noise can CREATE ordered phases that don't exist without noise. This goes beyond stochastic resonance: noise doesn't just amplify an existing signal — it creates qualitatively new behavior.

### Key Insight
Our finding that evolved soft prompts surface "qualitatively different knowledge" (honeypots, MITRE ATT&CK, HSM integration — concepts baseline NEVER produces) looks like a noise-induced phase transition: the model enters a qualitatively different computation mode that doesn't exist in the unperturbed system. This is not just "better output" — it's "different knowledge accessed."

### Novel Idea: Phase Diagram Mapping
Treat (token_count × RMS², quantization, model_size) as control parameters and map:
- Order parameter 1: P(answer_anywhere_correct)
- Order parameter 2: P(converged | answer_anywhere)
- Order parameter 3: trajectory class distribution entropy
- Order parameter 4: knowledge novelty (concepts in output that don't appear in baseline)

Phase boundaries where these order parameters change sharply are the most scientifically interesting regions — and potentially the most practically useful.

### Adversarial Check
Phase transitions require large systems (thermodynamic limit). A single LLM with 4B parameters is "one system" — it doesn't have the ensemble statistics that phase transitions require. We're mapping behavior over 10 seeds × 25 tasks, which is a finite sample from a deterministic system, not a thermodynamic ensemble. The "phase diagram" framing is metaphorical, not rigorous statistical mechanics.

Sources:
- [Noise-Induced Phase Transitions (Springer)](https://link.springer.com/book/10.1007/3-540-36852-3)
- [Dissipation-Induced Non-Equilibrium Phases](https://www.nature.com/articles/s42005-025-02113-1)

---

## Domain 7: Symbolic Regression / Automated Discovery

### Why This Matters
Instead of hand-designing the perturbation → outcome relationship, use symbolic regression to DISCOVER the mathematical relationship automatically from data.

### Novel Idea: Discover the Perturbation Law
Once Phase A collects the atlas (spec, biomarkers, outcome), feed it to a symbolic regression system (PySR, eggp) to discover:
- `P(correct) = f(RMS, token_count, quantization, attention_sink_mass, ...)`
- The function f is discovered, not assumed

This could reveal whether the relationship is polynomial, exponential, threshold-based, or something unexpected.

### Adversarial Check
Symbolic regression on 100-250 task groups with 10-dimensional features and binary labels is likely to overfit. Need held-out validation and simplicity pressure. Also, if the true relationship is stochastic (same inputs → different outcomes with some probability), symbolic regression will find the expected relationship, not the generative mechanism.

Sources:
- [Improving GP for Symbolic Regression with Equality Graphs](https://arxiv.org/abs/2501.17848)
- [Neuro-Evolutionary Physics-Aware Symbolic Regression](https://arxiv.org/html/2504.16503)

---

## Domain 8: Kramers Escape Theory (Statistical Physics)

### The Analogy
Kramers escape rate theory describes how a Brownian particle trapped in a potential well escapes over an energy barrier with the help of thermal noise. The escape rate depends exponentially on the barrier height and the noise intensity (temperature). Too little noise = particle stays trapped. Optimal noise = efficient escape. Too much noise = random diffusion without directed escape.

This is a suggestive quantitative analog of our mechanism (Codex R2: NOT claimed as "most direct" — Kramers rate increases monotonically with noise; our non-monotonicity comes from escape *quality*, not escape probability):
- **Potential well** = the model's default greedy decoding trajectory (the attractor basin it naturally falls into)
- **Energy barrier** = the "cost" of deviating from the default trajectory early enough to reach a different basin
- **Thermal noise** = our random embedding prefix (RMS scale)
- **Escape event** = the model switching to a qualitatively different trajectory class

### Key Insights
1. **Exponential sensitivity to barrier/noise ratio**: Kramers' rate is `r ~ exp(-ΔE/kT)`. In our system, this predicts that the probability of trajectory switching depends exponentially on the ratio of "trajectory barrier height" to RMS scale. Small changes in RMS near the critical ratio produce large changes in switching probability — consistent with our non-monotonic dose-response.

2. **Non-Gaussian noise enhances escape**: Recent work shows non-Gaussian noise (e.g., Lévy flights, heavy-tailed distributions) can enhance escape rates by orders of magnitude compared to Gaussian noise. Our isotropic Gaussian prefix may be suboptimal — heavy-tailed or structured noise distributions could produce dramatically better trajectory switching.

3. **Optimal correlation time**: For active (correlated) noise, there exists an optimal correlation time that maximizes transition rate. In our context: temporal correlation in the prefix (structured sequences vs. i.i.d. random tokens) might matter. The optimal "prefix structure" may be neither fully random nor fully optimized.

4. **Barrier height estimation from escape data**: Algorithms exist to estimate the potential barrier height from noise-induced escape time data. We could estimate "trajectory barrier heights" from our experimental data — how often does each RMS level cause trajectory switching? The distribution reveals the barrier structure.

### Novel Ideas
- **Heavy-tailed prefix noise**: Replace Gaussian with controlled Student-t (df=3-5) or Gaussian-mixture distributions first (Codex R2: use controlled variants before raw Cauchy). If the trajectory landscape has high barriers between basins, heavy-tailed noise would be dramatically more effective.
- **Barrier mapping**: For each task, measure the fraction of seeds that produce trajectory switching at each RMS level. Fit the Kramers escape model to estimate the effective barrier height. Tasks with lower barriers are "easier to perturb" — this could predict where our mechanism works best.
- **Temperature sweep protocol**: Instead of fixed RMS, test a geometric progression (0.005, 0.01, 0.02, 0.04, 0.08) to map the full Kramers curve. The inflection point reveals the effective barrier height.

### Adversarial Check
Kramers theory assumes a well-defined barrier in a low-dimensional energy landscape. The model's trajectory space is extremely high-dimensional — there's no single "barrier" to cross, but rather a complex manifold of trajectory boundaries. The exponential sensitivity prediction may not hold in high dimensions where the "barrier" is actually a ridge with many saddle points. In high-dimensional landscapes, barriers are typically saddled (have escape routes) — which would mean even low noise suffices for escape, contradicting our non-monotonic finding.

Sources:
- [Kramers Escape Rate Theory](https://home.icts.res.in/~abhi/notes/kram.pdf)
- [Barrier Height Estimation from Escape Data](https://link.springer.com/article/10.1007/s10955-020-02574-4)
- [Non-Markovian Kramers Theory](https://www.sciencedirect.com/science/article/abs/pii/S0003491619303008)
- [Kramers Barrier Crossing with Two Noises](https://pubmed.ncbi.nlm.nih.gov/34241384/)

---

## Domain 9: Constructive Neutral Evolution (Evolutionary Biology)

### The Analogy
Constructive Neutral Evolution (CNE) describes how biological complexity can increase through neutral (random) processes — genetic drift — without any positive selection. A "ratchet" mechanism means each neutral step is more likely to be followed by further complexity than to reverse. The key insight: **you don't need selection (optimization) to produce complexity (quality). Random drift on a ratcheted landscape suffices.**

### Why This Matters for Our System
Our empirical finding is that random perturbation ≈ evolved perturbation (p=1.0). The evolution loop (which tries to optimize) doesn't outperform random noise. CNE explains this: if the model's trajectory landscape has ratchet-like properties — once you leave the default basin, autoregressive generation "locks in" the new trajectory — then random perturbation is just as effective as directed perturbation.

The autoregressive ratchet: once the model generates the first few tokens down a new trajectory, the KV cache and conditional probabilities commit it to that path. There's no "undoing" the trajectory choice mid-generation. Random drift into a good basin is locked in just as firmly as optimized navigation into it.

### Key Insights
1. **Drift enables exploration that selection cannot**: Small populations (few candidates) with high drift (random perturbation) explore more of the fitness landscape than large populations under strong selection. Our N=10 random seeds may explore more trajectory classes than N=10 evolved seeds because evolution converges toward a single "good" direction while drift samples broadly.

2. **The complexity ratchet**: In biology, neutral mutations accumulate dependencies (presuppression) that make reversal unlikely. In our system, the first few tokens of a trajectory create dependencies (via attention) that make trajectory switching impossible mid-generation. The ratchet is the autoregressive structure itself.

3. **Weak selection, strong drift**: CNE is most powerful when selection is weak relative to drift — when the fitness differences between alternatives are small. In our system: if most trajectory classes produce similar-quality outputs (the oracle surface is relatively flat), then random selection among them is nearly as good as optimized selection.

### Novel Ideas
- **Population size effect**: Test whether N=3 seeds produces different trajectory class diversity than N=10 or N=50. In CNE, smaller populations have more drift effect. If our mechanism is drift-dominated, N=3 might already capture most of the diversity.
- **Ratchet measurement**: For each candidate, measure at which token position the trajectory becomes "committed" (divergence from other candidates becomes irreversible). This is the ratchet engagement point. If it's early (tokens 5-10), the prefix perturbation only needs to create initial divergence — the ratchet does the rest.

### Adversarial Check
CNE operates over evolutionary timescales (generations). Our system has no generations — it's a single-shot perturbation. The "ratchet" analogy is suggestive but the mechanism is fundamentally different: biological ratchets depend on gene duplication and functional redundancy; our "ratchet" is just autoregressive conditional dependence. Calling this CNE stretches the analogy beyond its formal conditions. It's more accurately described as "path dependence in a deterministic system seeded by noise."

Sources:
- [Constructive Neutral Evolution 20 Years Later](https://link.springer.com/article/10.1007/s00239-021-09996-y)
- [How a Neutral Evolutionary Ratchet Can Build Cellular Complexity](https://pubmed.ncbi.nlm.nih.gov/21698757/)
- [The Constructive Neutral Evolution of Behaviour (2025)](https://onlinelibrary.wiley.com/doi/10.1002/ece3.71736)
- [The Generality of CNE](https://link.springer.com/article/10.1007/s10539-018-9614-6)

---

## Domain 10: Percolation Theory (Network Science)

### The Analogy
In percolation theory, adding random connections to a network produces a sharp phase transition: below the percolation threshold, only local clusters exist. Above it, a single giant connected component spans the entire network. The transition is extremely sharp — a tiny increase in connection probability near the threshold produces a discontinuous jump in global connectivity.

### Connection to Our System
Consider the transformer's attention pattern as a network. At each layer, attention heads create "connections" between token positions. The model's default attention pattern may create a network that is below the percolation threshold for certain information pathways — relevant knowledge exists in the model's parameters but can't "flow" to the output because the attention connections don't span the right path.

Our prefix perturbation adds 2 random "nodes" (tokens) with random attention connections. If the default attention network is near the percolation threshold for some information pathway, these 2 additional connections could push it over the threshold — suddenly enabling information flow that was disconnected before.

### Key Insights
1. **Threshold sensitivity**: Percolation transitions are sharp. Near the threshold, tiny perturbations have enormous effects; far from the threshold, perturbations do nothing. This could explain our task sensitivity: "sensitive" tasks (19/25 in arithmetic) are near the attention percolation threshold; "frozen" tasks (5/25) are far from it.

2. **Critical window**: The percolation transition has a critical window that narrows with system size. In larger models (more layers, more heads), the critical window might be narrower — meaning the perturbation needs to be more precisely calibrated. This could explain why different models have different optimal RMS levels.

3. **Non-trivial interconnectivity optimum**: In interdependent networks, there's an optimal intermediate level of interconnection that maximizes robustness. Too few connections = fragmentation. Too many = cascading failures. Our non-monotonic dose-response (2 tokens optimal, 3+ degrades) parallels this: too few prefix tokens = no percolation benefit; too many = the additional connections create interference (attention dilution) that cascades through the network.

### Novel Ideas
- **Attention network analysis**: For a given input, compute the effective attention graph at each layer. Measure its connectivity (largest connected component, average path length). Compare with vs. without prefix. If the prefix consistently pushes the graph past a connectivity threshold, that's the percolation mechanism.
- **Task difficulty as percolation distance**: Classify tasks by how "close" their default attention network is to the percolation threshold. Predict: tasks near the threshold benefit most from perturbation.

### Adversarial Check
Transformer attention is not a random graph — it's a learned, structured graph with strong priors (attention to nearby tokens, sink tokens, etc.). Percolation theory applies to random or semi-random graphs. Also, the attention graph is FULLY connected (softmax over all positions) — there's no binary "connected/not connected." Every token attends to every other token. The percolation analogy only works if we threshold the attention weights to define "significant connections," which introduces an arbitrary parameter. The analogy is suggestive but not rigorous without a formal definition of the attention network's percolation properties.

Sources:
- [Percolation on Complex Networks](https://www.sciencedirect.com/science/article/abs/pii/S0370157320304269)
- [Dynamic Percolation with Triadic Interactions](https://www.nature.com/articles/s41467-023-37019-5)
- [Percolation-Like Phase Transitions in Protein Dynamics](https://pmc.ncbi.nlm.nih.gov/articles/PMC4457657/)
- [Chapter 8: Network Science](https://networksciencebook.com/chapter/8)

---

## Domain 11: SGD Noise and Loss Landscape Navigation (ML Theory)

### The Analogy
During training, SGD's gradient noise helps networks find flat (generalizable) minima by escaping sharp (overfitting) minima. The noise acts as implicit regularization — it biases the optimization toward regions of the loss landscape with better generalization properties. Recent work (January 2026) shows anisotropic noise reshapes the loss landscape itself, creating an "effective potential" that systematically favors flatter solutions.

### Transfer to Inference Time
This is the training-time analog applied to inference-time:
- **Training landscape** → **Inference trajectory landscape** (the space of possible greedy decodings from different initial conditions)
- **Sharp minima** → **Degenerate trajectories** (repetitive, attention-sink-dominated, truncated outputs)
- **Flat minima** → **Robust trajectories** (diverse, knowledge-rich, complete outputs)
- **SGD noise** → **Prefix embedding noise** (our random perturbation)

The prediction: just as SGD noise biases training toward flat minima, our prefix noise biases inference toward "flat" (robust) trajectory classes — ones that don't depend sensitively on exact initial conditions and produce consistently good output.

### Key Insights
1. **Heavy-tailed noise is more effective**: SGD with heavy-tailed gradient noise (not Gaussian) eliminates sharp minima more effectively. This parallels the Kramers escape finding and suggests our Gaussian prefix noise is suboptimal.

2. **Noise conditions the computation**: The 2026 "transient learning dynamics" paper shows noise creates an effective potential during training. In our system: the prefix conditions the model's computation from step 1 by altering the KV cache and hidden state initialization. (Codex R2: "reshapes the inference landscape" is too strong — the prefix conditions computation, it does not reshape a learned loss landscape in the SGD sense.)

3. **Anticorrelated noise**: 2022 work showed that anticorrelated noise injection improves generalization more than i.i.d. noise. In our context: structured (anticorrelated across embedding dimensions) prefix noise might outperform i.i.d. Gaussian noise.

### Adversarial Check
The training/inference analogy is loose. During training, noise acts over millions of gradient steps, gradually biasing toward flat minima. During inference, our noise acts ONCE (at the initial prefix). There's no iterative refinement — just a single perturbation that shifts the starting point. The "flat minima" analogy requires that trajectory classes have different "widths" in initial-condition space (robust trajectories are accessible from a wider range of starting conditions). This is plausible but unverified.

Sources:
- [Transient Learning Dynamics Drive Escape from Sharp Valleys (2026)](https://arxiv.org/html/2601.10962)
- [Eliminating Sharp Minima from SGD with Heavy-Tailed Noise](https://arxiv.org/abs/2102.04297)
- [Anticorrelated Noise Injection for Improved Generalization](https://proceedings.mlr.press/v162/orvieto22a/orvieto22a.pdf)
- [Global Dynamics of Heavy-Tailed SGDs](https://arxiv.org/html/2510.20905)

---

## CRITICAL COMPETITOR: Inference-Time Prompt Perturbation (Feb 2025)

### The Finding
A February 2025 paper "On the Effect of Sampling Diversity in Scaling LLM Inference" (arXiv:2502.11027) systematically studies **diversified prompt perturbations within the Best-of-N framework**. They test:
- Task-level perturbations (rephrasing the task description)
- Query-level perturbations (adding random "idea injection" to the prompt)
- Sampling temperature diversity

### Results
- **+10.8% EM@100 on reasoning** (MMLU-Pro)
- **+9.6% on mathematics** (MATH)
- **+9.5% Pass@100 on code** (HumanEval)
- They found "moderately relevant perturbations" outperform both no-perturbation and high-perturbation

### Why This Threatens Us
This is **literally what we're doing** — perturbing the input to get diverse outputs from deterministic models, then selecting the best. The differences:
- They perturb in TOKEN space (rephrasing prompts). We perturb in EMBEDDING space (sub-token continuous vectors).
- They use large models (GPT-4 class). We use small quantized models (4B-8B).
- They require N full generations. We hypothesize N cheap scorer evaluations + 1 full generation (once Phase B works).

### Our Differentiating Position (Codex R2 Mandated)
> "If temperature sampling matches prefix perturbation under equal N and equal output-token budget, our claim is not better best-of-N accuracy; it is that continuous prefix perturbations are a competitive, analyzable diversity operator that exposes hidden trajectory sensitivity."

Without a validated selector/router, the efficiency story is aspirational. If temperature best-of-N matches our oracle and Phase B is weak, the paper becomes mechanistic/diagnostic, not performance-superiority work. That is still publishable but requires honest positioning.

### What This Means for Our Contribution
1. **Trajectory diversification is a known technique.** We cannot claim novelty for "perturbing inputs improves best-of-N." That's been published.
2. **Our novelty MUST be**: (a) perturbation in continuous embedding space below token granularity, (b) the non-monotonic dose-response and its theoretical explanation, (c) the connection to attention sinks and model pathologies, (d) the efficiency story (prefix perturbation is cheaper than prompt rephrasing for the same diversity).
3. **Reframing urgency**: The paper narrative must acknowledge this prior art and clearly differentiate.

### Adversarial Assessment
The worst case: a reviewer says "this is just best-of-N with random prefix noise instead of prompt rephrasing — what's new?" We need a strong answer. Candidate answers:
- Sub-token granularity accesses trajectory classes that no token-space perturbation can reach (because the embedding space is continuous, not discrete)
- Our non-monotonic dose-response reveals the underlying mechanism (stochastic resonance / Kramers escape), which prompt rephrasing cannot measure
- We show the effect on models small enough that prompt rephrasing might not create sufficient diversity (the vocabulary "resolution" may be too coarse for small models)

Sources:
- [On the Effect of Sampling Diversity in Scaling LLM Inference](https://arxiv.org/html/2502.11027v3)
- [Awesome Test Time LLMs](https://github.com/dereck0602/awesome_test_time_llms)

---

## Domain 12: Noise-Induced Bimodality (Systems Biology)

### The Analogy
In bistable cell regulatory systems, noise doesn't just help switch between existing states — it can CREATE bimodality (two distinct stable states) that doesn't exist in the deterministic (noise-free) system. This goes beyond stochastic resonance: the noise doesn't amplify a subthreshold signal, it creates entirely new dynamical states.

### Connection to Our System
Our finding that perturbation surfaces "qualitatively different knowledge" (honeypots, MITRE ATT&CK, tiered credential rotation) — concepts the baseline NEVER produces — looks like noise-induced bimodality. The default model has ONE observed mode of generation under greedy decoding. The perturbation reveals a SECOND mode that generates different knowledge entirely. This second mode is not observed under the deterministic baseline. (Codex R2: need cluster evidence before claiming true bimodality.)

### Key Insight: Critical System Size
Noise-induced states are characterized by a critical group/system size at which collective properties qualitatively change. In our system: there may be a critical model size or attention head count below which noise creates new trajectory modes (our 4B/8B results) and above which the model is robust to perturbation (DeepSeek's resistance, or very large models). This would predict that our mechanism is most powerful for mid-range models — too small to have stable trajectories (always random), too large to be perturbed (trajectories are locked by many redundant attention heads).

### Adversarial Check
Noise-induced bimodality in biology requires specific enzymatic network topologies (positive feedback loops with cooperativity). We have no evidence that the transformer's attention pattern has the topology required for noise-induced bistability. The analogy is intriguing but has no formal grounding in transformer architecture. To validate, we'd need to show that the model's trajectory landscape has exactly two (or few) distinct modes that are noise-dependent.

Sources:
- [In Search of Noise-Induced Bimodality](https://link.springer.com/article/10.1186/1741-7007-10-89)
- [Noise-Induced Effects in Collective Dynamics](https://royalsocietypublishing.org/doi/10.1098/rstb.2019.0381)
- [A Genetic Timer Through Noise-Induced Stabilization](https://www.pnas.org/doi/10.1073/pnas.0806349105)
- [Noise in Biology (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4033672/)

---

## Domain 13: Ensemble Theory (Machine Learning)

### Why Codex Flagged This as Missing
Codex: "Diversity, error correlation, oracle coverage, and selector quality are central to your actual system." This is the most directly relevant framework — not a metaphor, but the literal mathematical structure our system operates in.

### The Framework
Our system IS an ensemble: N=10 candidate generations, combined via oracle selection (or future routing). Ensemble theory provides the formal tools:

1. **Ambiguity decomposition**: Ensemble error = mean individual error - diversity. For a fixed individual error (same model, same task), improvement comes ENTIRELY from diversity. Our prefix perturbation is a diversity-generation mechanism.

2. **Error correlation**: If all N candidates make correlated errors (fail on the same tasks), oracle selection can't help. If errors are uncorrelated, oracle coverage grows as 1 - p^N where p is individual error rate. Our empirical oracle coverage (100% on some tasks) suggests low error correlation — different seeds fail on different tasks.

3. **Oracle ceiling**: With N=10 and individual accuracy p=0.52 (Qwen3-4B), if errors were independent: oracle coverage = 1 - (1-0.52)^10 = 99.95%. We observe 100%. If errors were perfectly correlated: oracle = 52%. We observe 100%. This is *consistent with* low error correlation, but 100% oracle on 25 tasks does NOT prove approximate independence — it only suggests correlation is low enough. Need pairwise error correlation matrix with CIs to make this claim. (Codex R2: tighten math, prove with correlation data.)

4. **Selector quality ceiling (CRITICAL)**: The "Inference Scaling FLaws" paper (Stroebl, Kapoor, Narayanan 2024) proves: with an imperfect verifier that has false positive rate ε, accuracy is bounded by 1/(1+ε) regardless of N. Our scorer/judge has non-zero false positive rate → our system has a ceiling that more seeds cannot overcome. The only way to raise the ceiling is to improve the verifier.

### Key Insights for Our System

1. **Diversity is the mechanism**: We don't need cross-domain analogies to explain why random perturbation helps. Ensemble theory says: any mechanism that decorrelates errors across candidates improves oracle coverage. Random prefix is one such mechanism. The interesting question is: does it decorrelate errors MORE than temperature sampling? More than prompt rephrasing?

2. **Self-certainty as selector**: The self-certainty metric (divergence of token distribution from uniform, measured from the model's own logits — no additional model needed) is closely related to our Phase B routing score. Self-certainty at generation time ≈ mean_logprob component of our routing_score. This suggests our Phase B design is on the right track, but we should compare against the published self-certainty metric directly.

3. **The verifier ceiling**: If our judge (LLM-as-judge for legal/planning, exact-match for arithmetic) has false positive rate ε, then:
   - For arithmetic: ε ≈ 0 (exact match is perfect for correctly formatted answers). Ceiling is very high.
   - For legal/planning: ε > 0 (judge can prefer a worse response). Ceiling is lower.
   - This predicts: arithmetic oracle gains should scale better with N than legal/planning oracle gains.

### Novel Ideas
- **Error correlation matrix**: For existing N=10 arithmetic data, compute the 10×10 error correlation matrix (does seed i fail on the same tasks as seed j?). If correlation is high, increasing N won't help. If low, increasing N will.
- **Compare diversity sources**: Run N=10 with (a) random prefix, (b) temperature=0.6, (c) prompt rephrasing. Measure error decorrelation for each. The mechanism that produces the lowest error correlation is the best diversity source for best-of-N.
- **Self-certainty comparison**: Implement the self-certainty metric (token distribution divergence from uniform) and compare against our preregistered routing_score for Phase B.

### Adversarial Check
Ensemble theory assumes individual learners are "different enough" to have uncorrelated errors. But our N=10 candidates are the SAME model with the SAME weights — only the 2-token prefix differs. Whether this creates sufficient diversity for ensemble benefits is an empirical question that our data partially answers (100% oracle coverage suggests yes). But ensemble theory also says: optimal diversity maximization may conflict with individual quality. Adding more noise increases diversity but hurts individual accuracy. Our non-monotonic dose-response is exactly this tradeoff.

Sources:
- [Inference Scaling FLaws: Limits of Resampling with Imperfect Verifiers](https://arxiv.org/abs/2411.17501v2)
- [Scalable Best-of-N via Self-Certainty](https://arxiv.org/abs/2502.18581)
- [On the Effect of Sampling Diversity in Scaling LLM Inference](https://arxiv.org/abs/2502.11027)
- [Self-Error Adjustment: Balancing Performance and Diversity](https://arxiv.org/html/2508.04948v1)
- [Boosting Ensemble Accuracy by Revisiting Diversity Metrics (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Boosting_Ensemble_Accuracy_by_Revisiting_Ensemble_Diversity_Metrics_CVPR_2021_paper.pdf)

---

## Domain 14: Lyapunov Analysis / Random Dynamical Systems

### Why Codex Flagged This
Codex: "Random dynamical systems / Lyapunov analysis — more directly relevant than OGY." OGY requires closed-loop control (we don't have it). Lyapunov analysis characterizes the sensitivity of a dynamical system to initial perturbation — which is EXACTLY what we're measuring.

### The Framework
A deterministic dynamical system with Lyapunov exponent λ > 0 (positive) is chaotic: nearby initial conditions diverge exponentially at rate e^(λt). For our system:
- **State**: the model's hidden state at each layer
- **Dynamics**: the transformer forward pass (layer by layer)
- **Initial perturbation**: the 2-token prefix (δ₀)
- **Divergence**: δ(t) = δ₀ · e^(λt) where t is layer depth

If the transformer has λ > 0 for the relevant hidden-state trajectories, a small initial perturbation (RMS ~0.022) grows exponentially through layers and can produce macroscopically different output by the final layer.

### Key Insights

1. **Lyapunov exponent predicts perturbation sensitivity**: If we could measure λ for a given model on a given input, we could predict: (a) which tasks are sensitive to perturbation (high λ → sensitive), (b) how much RMS is needed to shift trajectory class (lower λ → more RMS needed), (c) how quickly the effect saturates with additional prefix tokens.

2. **Finite-time Lyapunov exponents (FTLE)**: In high-dimensional systems, the Lyapunov exponent varies in time and across the state space. The FTLE at each layer measures how much that layer amplifies the perturbation. Some layers may be strongly amplifying (high FTLE), others suppressing (negative FTLE). The model's attention mechanism could create FTLE patterns that are input-dependent.

3. **Lyapunov dimension**: In a chaotic attractor, the Lyapunov dimension tells you the effective dimensionality of the strange attractor. If the model's trajectory space has a strange attractor with dimension d ≪ embed_dim, then perturbation in d directions matters and perturbation in the remaining directions is absorbed. This could explain why random Gaussian noise (which projects onto all dimensions equally) works: it's guaranteed to have a nonzero component in the d important directions.

4. **Critical perturbation amplitude**: For systems near the edge of chaos (λ ≈ 0), there's a critical perturbation amplitude below which trajectories converge back together and above which they diverge. The threshold may be related to local FTLE, but finite-amplitude basin boundaries and argmax discontinuities may dominate even when local FTLE is negative. (Codex R2: "the threshold IS λ = 0" is overconfident — drop from paper. Lyapunov is an analysis tool, not paper theory yet.)

### Novel Ideas
- **Empirical Lyapunov estimation**: Use our existing `IntermediateLayerSteering` hooks to measure how perturbation in the residual stream at layer 0 grows or shrinks through subsequent layers. Run this for each of 25 arithmetic tasks. Tasks with higher effective Lyapunov exponent should be more sensitive to prefix perturbation.
- **Layer-specific sensitivity map**: Measure ||hidden_state_perturbed - hidden_state_baseline|| at each layer. The growth rate of this divergence IS the finite-time Lyapunov exponent. If it's negative at early layers but positive at later layers, the model's computation amplifies perturbations only in specific processing stages.

### Adversarial Check
LLMs are discrete-token systems with autoregressive structure — not continuous dynamical systems. The Lyapunov framework applies to the continuous hidden states within a single forward pass, but the autoregressive generation step (argmax → embed → next forward pass) is a discontinuous, non-differentiable operation. Lyapunov theory requires differentiability. The argmax at each step means the system is piecewise-linear with sharp boundaries, not smooth. FTLE can still be estimated numerically, but the theoretical guarantees of smooth dynamical systems don't directly transfer.

Additionally, transformer forward passes are finite (L layers), not infinite-time. Lyapunov exponents are defined as t → ∞ limits. The finite-time analog (FTLE) is the right tool, but it's noisier and harder to interpret.

Sources:
- [Lyapunov Exponents and Dynamical Systems (Nature)](https://www.nature.com/research-intelligence/nri-topic-summaries/lyapunov-exponents-and-dynamical-systems-micro-183081)
- [Chaos, Lyapunov Exponents, and Sensitivity (AIMS 2025)](https://www.aimspress.com/article/doi/10.3934/math.20251019)
- [Control of Spatiotemporal Chaos by Stochastic Resetting](https://arxiv.org/html/2412.21043)
- [Stability Analysis of Chaotic Systems from Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC10076397/)

---

## Codex Analogy Ratings (Round 1)

| Domain | Rating | Codex Assessment |
|--------|--------|-----------------|
| Stochastic Resonance | SUGGESTIVE | Non-monotonic dose-response fits superficially. Formal SR conditions (threshold, subthreshold signal) not met — Qwen3-4B already 80% answer-anywhere. Keep as "reminiscent," not explanatory. |
| Vibrational Resonance | **MISLEADING** | VR needs deterministic high-frequency drive + bistability. Random=evolved undermines the analogy. **DROP from paper.** |
| Basin Hopping | SUGGESTIVE | Useful frame: "multi-start basin sampling" (random prefix → basin, greedy decode descends). No objective, no acceptance rule, no iterative hopping. |
| OGY Chaos Control | **MISLEADING** (current) | OGY is closed-loop. We are open-loop. Useful only as future inspiration for mid-generation control. **Do not claim in paper.** |
| Dithering | SUGGESTIVE | Quantization interaction makes it worth testing. But "1 LSB" is overconfident — BitsAndBytes step sizes are layer/group/dynamic, not uniform. |
| Phase Transitions | **MISLEADING** | No thermodynamic limit, no ensemble. Use "regime map" language only. **"Phase transition" invites reviewer pushback.** |
| Symbolic Regression | RIGOROUS (as tool) | Not a mechanism analogy. Valid analysis if held-out and simplicity-penalized. Current sample sizes make overfit likely. |
| Kramers Escape | SUGGESTIVE (new) | Most direct classical analog. Exponential sensitivity prediction needs high-dimensional correction. |
| CNE / Genetic Drift | SUGGESTIVE (new) | Autoregressive ratchet analogy is suggestive. Formally different from biological CNE (no generations). |
| Percolation | SUGGESTIVE (new) | Requires threshold definition on a fully-connected softmax graph — arbitrary parameter needed. |
| SGD Noise / Flat Minima | SUGGESTIVE (new) | Training-to-inference transfer is loose (one shot vs. millions of steps). But "trajectory class width" prediction is testable. |
| Noise-Induced Bimodality | SUGGESTIVE (new) | Intriguing but requires specific network topology we haven't verified. |

**Paper-safe analogies**: Multi-start basin sampling (SUGGESTIVE, honest), Dithering (SUGGESTIVE, testable), Kramers escape (SUGGESTIVE, quantitative predictions).
**Idea generators only**: SR, CNE, Percolation, SGD noise, Bimodality.
**Drop entirely**: VR, OGY (for current system), Phase Transitions.

---

## The Simplest Explanation (Codex-Approved Framing)

> Two OOD embedding-scale prefix tokens perturb early hidden states and/or positions in a deterministic greedy transformer. This routes generation into different reasoning trajectories. Some trajectories are shorter, less formal, and more likely to terminate on the right answer; others ramble and regress. The optimum balances trajectory diversity against coherence and token-budget exhaustion. Best-of-N plus a judge extracts the upside.

This explanation does not need SR, VR, OGY, or phase transitions. Attention sinks explain why early positions are high leverage, but they are not yet proven to be the necessary mechanism. Cross-domain analogies serve as idea generators for future work, not as claimed mechanisms.

---

## Synthesis: Revised Priority Ranking (Post-Codex Round 2)

### Codex-Approved Implementation Order:

1. **Head-to-head vs temperature / prompt perturbation** — FASTEST EXISTENTIAL NOVELTY CHECK. Run temperature sampling (T=0.3, 0.6, 0.9) best-of-N=10 on same 25 arithmetic tasks with same Qwen3-4B Q4. Also run prompt rephrasing (synonym substitution) at N=10. Compare oracle coverage + mean accuracy vs our prefix perturbation N=10. If temperature matches our oracle, our novelty shrinks to efficiency/mechanism only.
   - **Key sentence (Codex-mandated)**: "If temperature sampling matches prefix perturbation under equal N and equal output-token budget, our claim is not better best-of-N accuracy; it is that continuous prefix perturbations are a competitive, analyzable diversity operator that exposes hidden trajectory sensitivity."

2. **Qwen3.5-4B gated-attention transfer probe** — SUBMISSION-GATING. 25 arithmetic + 1 planning task. 4 MANDATORY conditions: (a) baseline, (b) zero 2-token soft prefix, (c) random 2-token N=10 RMS-matched, (d) position-shift control (no prefix, position_ids start at 2).
   - **Position-control is MANDATORY** (Codex R2: "not 'if feasible'"). Also test random-vocab-token prefix as ablation.
   - Feasibility: Qwen3.5-4B fits on 24GB GPU. Same family as Qwen3-4B → cleanest comparison.

3. **Error correlation matrix + trajectory clustering** — VALIDATES ENSEMBLE FRAMING. Compute pairwise 10×10 error correlation across existing N=10 arithmetic data. Cluster outputs by first 16-32 tokens. Report diversity metrics with bootstrap CIs over tasks. This is needed BEFORE the paper can claim ensemble-theoretic framing.

4. **Selector/self-certainty comparison** — PAPER LIVES OR DIES ON SELECTOR REALISM. Implement self-certainty metric (token distribution divergence from uniform). Compare against preregistered routing_score for Phase B. Oracle results are not enough; if the selector can't identify the best candidate, efficiency story collapses.

5. **Position-control ablation battery** — Full ablation: (a) zero prefix, (b) random prefix with corrected position IDs, (c) random vocab-token prefix, (d) shifted-position control. This is the causal decomposition: how much from position shift vs embedding noise vs attention disruption.

6. **RMS sweep on Q4 vs Q8** — Tests dithering hypothesis. Run RMS multipliers {0.25, 0.5, 1.0, 2.0, 4.0} on Qwen3-8B at both Q4 and Q8. If Q4's optimum shifts to higher RMS, dithering gains support.

7. **Heavy-tailed prefix noise** — After baselines established. Student-t (df=3-5), Gaussian mixture, or norm-clipped before raw Cauchy. Tests Kramers barrier structure prediction.

8. **Ratchet engagement / Lyapunov probes** — Secondary mechanism analysis. Token-by-token divergence from baseline to find commitment point. Empirical FTLE estimation via residual hooks.

### Demoted / Future Work:
- **Colored prefix noise** → Lower priority. Try random vocabulary embedding sampling first.
- **Mid-generation OGY** → Future work only.
- **Percolation analysis** → Weakest domain (Codex R2). Future attention-graph analysis only.

---

## Pre-Implementation Measurement Contract (Codex R2 Mandate)

All experiments must:
1. **Equalize budget**: Same N, same max_new_tokens, comparable wall-clock or generated-token budget across conditions
2. **Report standard metrics**: Oracle accuracy, selected accuracy (from selector), individual candidate accuracy (mean), diversity (trajectory class count), error correlation (pairwise)
3. **Bootstrap CIs**: Over tasks (not seeds), minimum 1000 iterations
4. **Separate evaluation types**: Exact-match tasks (arithmetic) reported separately from judge-scored tasks (legal/planning)
5. **Audit judge false positives**: For legal/planning, measure judge FP rate against human review on held-out subset
6. **Define comparison outcomes**:
   - "Temperature matches us" = oracle coverage within 5pp AND mean accuracy within 3pp
   - "Prefix beats" = oracle coverage > 10pp advantage OR mean accuracy > 5pp
   - "Prefix is mechanistically interesting" = different trajectory classes accessed, but similar oracle/accuracy

---

## Single Remaining Theoretical Gap (Codex R2)

**Causal decomposition of the gain:**

How much improvement comes from:
- (a) **Position shift** — does shifting position IDs by 2 alone produce trajectory diversity?
- (b) **Attention-sink disruption** — does the prefix reduce attention sink mass, and does that correlate with improvement?
- (c) **Embedding-noise trajectory diversity** — conditional on (a) and (b), how much additional diversity comes from the random embedding content?
- (d) **Selector quality** — how much of the final gain depends on having a good oracle/selector vs. the diversity source?

**Until this decomposition is measured, the theory is still underdetermined.** The position-control ablation battery (priority #5) directly addresses this.

---

## The Big Picture: Where the Science Goes

**If gated attention kills the effect on Qwen3-Next:**
- The attention-sink rescue mechanism has a shelf life. Our contribution is historical — first to document the phenomenon on pre-gated models.
- Pivot to: is there a gated-attention-compatible perturbation that achieves trajectory diversification through a different mechanism? (Position perturbation, layer-specific residual injection, etc.)

**If the effect persists on gated-attention models:**
- The phenomenon is deeper than attention sinks. Trajectory diversification in deterministic greedy decoding is a fundamental property.
- Paper framing: "multi-start basin sampling" in continuous embedding space, with the simplest explanation as primary narrative. Cross-domain analogies move to a "Related Work / Theoretical Context" section, clearly labeled as suggestive, not claimed.
- The science goes toward:
  - Regime mapping (NOT "phase diagram") over (RMS, token_count, quantization, model_size)
  - Head-to-head efficiency comparison vs temperature sampling and prompt rephrasing
  - Symbolic discovery of the perturbation-outcome relationship
  - Integration with inference-time compute scaling

**If best-of-N temperature sampling matches our oracle:**
- Our novelty reduces to: (a) mechanism insight (sub-token continuous perturbation), (b) efficiency story (prefix perturbation + early routing is cheaper than N full generations), (c) theoretical framework for understanding when perturbation helps.
- This is still publishable but requires honest positioning against best-of-N literature.
