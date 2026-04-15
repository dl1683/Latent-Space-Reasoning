**Verdict**

Phase 2 is directionally right, but not ready for Phase 3/5 as a system design. It has replaced “latent evolution” with “trajectory control” in language, but the central mechanism is still unvalidated: an early Observer-Router is assumed to predict final correctness before the project has measured that signal.

The design needs another Phase 2 revision. The next step should be one decisive measurement experiment, not architecture expansion.

**Critical Findings**

1. **The 32-token Observer assumption is existential and unsupported.** Phase 2 says trajectory class is measured in the first 16-64 tokens and defaults to 32 in the primary flow [phase2_mental_assembly.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase2_mental_assembly.md:25), [phase2_mental_assembly.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase2_mental_assembly.md:264). The experiment record supports “early dynamics matter,” but not “32 tokens predict correctness.” Word problems failed from token-cap/truncation effects [EXPERIMENTS.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/EXPERIMENTS.md:256), and legal/planning quality can emerge or collapse hundreds of tokens later.

2. **`attention_sink_mass` is a hypothesis, not a validated biomarker.** The planning result says perturbation overcomes attention sinks [EXPERIMENTS.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/EXPERIMENTS.md:117), but there is no measured sink-mass-to-correctness correlation. Phase 2 even admits this remains unvalidated [phase2_mental_assembly.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase2_mental_assembly.md:352). At best it may predict collapse/length, not correctness.

3. **The six-feature router is unjustified.** The listed features are not six clean scalar observables: `entropy_trajectory` is a sequence, `answer_position_forecast` and `length_forecast` are themselves learned predictors, and `trajectory_class` is a latent label with no labeling rule [phase2_mental_assembly.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase2_mental_assembly.md:77). A 6→32→1 MLP is architectural theater until mutual information and calibration are measured.

4. **Component B is implementable only with new generation plumbing.** The package orchestrator still decodes through `encoder.decode()` [orchestrator.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/src/latent_reasoning/orchestrator/orchestrator.py:583), while the validated path is raw `inputs_embeds` soft-prefix generation in experiments [run_latent_sensitivity.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_latent_sensitivity.py:49), [run_latent_sensitivity.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_latent_sensitivity.py:135). `encoder.decode()` still uses latent statistics to seed/adjust generation [encoder.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/src/latent_reasoning/core/encoder.py:621), [encoder.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/src/latent_reasoning/core/encoder.py:652). The observer needs token IDs, logits, logprobs, attentions, hidden states, prefix length, and KV cache continuation. That contract is missing.

5. **The “degraded mode = current system behavior” fallback is unsafe.** Phase 2 says degraded mode falls back to current behavior [phase2_mental_assembly.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/tesla_session/phase2_mental_assembly.md:333). But current public behavior is exactly the gap Phase 1 flagged. If fallback means `encoder.decode()`, regressions are hidden. If judge fallback means length/format heuristic, it violates the project’s own rule that numeric/heuristic scores are not valid quality evidence [CLAUDE.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/CLAUDE.md:5), [CLAUDE.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/CLAUDE.md:13).

**Assumption Audit**

32 tokens: not valid yet. The correct minimum window is empirical: the earliest window where `I(features_w; final_correct)` saturates. If correct and incorrect traces share the first 64 tokens, token features have near-zero usable information. Hidden/logit features may carry earlier signal, but that has not been shown.

`attention_sink_mass`: plausible for collapse detection, not correctness. It may be a length/truncation proxy. It also may invert by model/layer: high attention to soft prefix could be constructive, not pathological.

Six biomarkers: insufficiently justified. Missing features include first divergence time, cumulative logprob slope, EOS hazard, token-cap risk, answer-extraction stability, repetition/loop markers, verifier partials, task difficulty/base confidence, and relative candidate features.

Other fragile assumptions: task class is known; judge is available and reliable; partial generation can cheaply continue without rerun; atlas labels are unbiased; multi-surface interventions compose; energy is captured by `token_count × rms_scale²`; and unpromoted candidates need not be labeled. Several are likely false.

**Primitive Inheritance**

A inherits the right primitive only for `input_prefix`: calibrated random RMS-matched soft tokens. The phase diagram idea is right, but `E = token_count × rms²` is under-specified because embedding dimension, layer norm, prompt position, quantization, and surface all change effective energy.

B is the correct new measurement primitive, but it does not inherit enough implementation detail. Existing probes can do hidden-state forward passes [run_activation_probe.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_activation_probe.py:168), and existing soft prompt decode works through `inputs_embeds` [encoder.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/src/latent_reasoning/core/encoder.py:927), [encoder.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/src/latent_reasoning/core/encoder.py:943). But there is no real autoregressive observer interface yet.

C does not eliminate the scorer problem. It moves it from “latent scorer” to “early trace scorer.” That is better only if early traces contain enough information. Otherwise it destroys the 92% legal oracle effect by pruning the winning candidate before output validation [EXPERIMENTS.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/EXPERIMENTS.md:61).

D is necessary. Keep only `input_prefix` in mainline for now. Residual steering exists as hooks [steering.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/src/latent_reasoning/decode/steering.py:137), [steering.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/src/latent_reasoning/decode/steering.py:177), but it is not validated as part of PGRMS/TCA.

E is necessary and aligned with `CLAUDE.md`, but its fallback must not be allowed to make quality claims.

F is necessary for science, but must log all candidates during router training, not only promoted candidates, or it trains on selection bias.

**Component Failures**

A fails by generating harmful specs or invalid atlas defaults. Then every downstream component optimizes garbage.

B fails by producing false negatives. That is catastrophic: the best candidate never reaches the validator.

C fails by over-pruning. This converts oracle lift into mean regression.

D fails by silently using the wrong conditioning path. This is the current highest implementation risk.

E fails by selecting fluent hallucinations, especially in legal/planning.

F fails by contaminating the atlas with stale model/tokenizer/prompt-template/judge versions.

Data flow failure: partial generation followed by full generation is not specified as continuation from the same KV cache. If full generation reruns from scratch, the system pays extra cost and risks mismatch. If it continues, the `ModifiedInput` contract must carry cache, generated prefix tokens, attention masks, and decode offsets.

**Cross-Domain Challenge**

Information theory: the key quantity is not correlation; it is `I(F_w; Y)` where `F_w` is early features at window `w` and `Y` is final correctness or convergence class. If this is below about 0.1 bits at 64 tokens, the router has no meaningful signal. For a useful low-error binary router, you likely need materially more than that, especially under class imbalance.

Dynamical systems: trajectory labels can switch. “Exploring” can converge; “collapsing” can recover; legal outputs can look coherent early and hallucinate later. Treat classes as metastable states with transition probabilities, not fixed attractor labels after 32 tokens.

Neuroscience: closed-loop feedback systems trade latency for fidelity. Very early signals are fast but coarse; high-fidelity routing requires recurrent evidence accumulation. Prediction: early biomarkers will detect collapse/length better than semantic correctness.

Statistical mechanics: the phase diagram framing is valid, but order parameters must be explicit: final correctness, answer-anywhere correctness, convergence rate, truncation rate, entropy slope, divergence time, sink mass, oracle diversity, and regression rate. Phase boundaries should be learned response surfaces, not hand-set thresholds like `attention_sink_mass > 0.6`.

**Alternative**

Replace the online Observer-Router with **offline validated sequential routing**:

1. Fix the package path so raw `inputs_embeds` input-prefix generation is the default.
2. Generate N random 2-token RMS-calibrated candidates fully.
3. Validate in output space.
4. Log all candidates with early features at 1/4/8/16/32/64/128 tokens.
5. Only deploy a router after it proves it can retain >90% of oracle winners while promoting a small fraction of candidates.

The 80/20 architecture is: `InputPrefixSpecGenerator → RawSoftPrefixApplicator → FullGeneration → OutputValidator → AtlasLogger`. No multi-surface interventions. No learned router. No latent evolution.

Minimum viable next-gen system: validated raw soft-prefix package API, deterministic arithmetic verifier, legal/planning judge rubric, candidate-level logging, and offline MI analysis.

**Priority Directive**

Before any TCA code, run this experiment:

Measure `I(early_trajectory_features_w; final_correct)`, `I(features_w; answer_anywhere_correct)`, and `I(features_w; convergence_correct | answer_anywhere_correct)` for `w ∈ {1, 4, 8, 16, 32, 64, 128}` across existing arithmetic, word problem, planning, and legal datasets. Include Qwen3-4B Q4, Qwen3-8B 8-bit, and DeepSeek where data exists.

Acceptance gate: if cheap features at 64 tokens cannot retain at least 90% of oracle winners while promoting at most 40-50% of candidates, or if MI stays below 0.1 bits, kill the online Observer-Router. Keep it as an offline analysis tool only.

**Gate Decision**

Do not proceed to Phase 5. Phase 2 needs revision before hypotheses/predictions are formalized.

Proceed only to a narrowed Phase 3 stress test whose sole purpose is validating the early-biomarker premise. Required revisions:

1. Mark `attention_sink_mass`, 32-token classification, and six-feature routing as hypotheses.
2. Remove multi-surface interventions from mainline.
3. Define the observer’s real tensor/cache/logit/attention contract.
4. Define fallback as “validated random raw-prefix + full output validation,” not current package behavior.
5. Require unbiased atlas logging of promoted and unpromoted candidates.
6. Add MI/AUROC/recall-of-oracle thresholds as hard gates.

Current design is promising as a research agenda. It is not yet an architecture.