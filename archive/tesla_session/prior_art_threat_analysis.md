# Prior Art Threat Analysis

## Status: COMPLETE — Critical new paper found

---

## THREAT LEVEL: HIGH — "Soft Reasoning" (ICML 2025 Spotlight)

### Paper: "Soft Reasoning: Navigating Solution Spaces in LLMs through Controlled Embedding Exploration"
- **Authors**: Zhu, Zhao, Yan, He, Chen, Gui
- **Venue**: ICML 2025 Spotlight
- **arXiv**: 2505.24688

### What they do (VERY close to our work)
- Add Gaussian noise to the embedding of the **first generated token** (not prefix tokens)
- Use greedy decoding after perturbation (same as us)
- Observe different perturbations activate different reasoning paths (same claim as us)
- Use Bayesian optimization (GP + Expected Improvement) to find good perturbation directions
- Require a verifier to select among candidates
- Models: LLaMA-3.1-8B, Qwen2-7B, Qwen2-70B, Mistral-8B (all 7B+, none quantized)
- Results: +4.9pp on GSM8K (LLaMA-8B), +8.8pp on SVAMP

### What preserves our novelty (MUST differentiate clearly)
1. **Random noise sufficiency**: They need Bayesian optimization + verifier. We show RANDOM noise alone works. This is a stronger claim about the underlying phenomenon.
2. **Prefix position vs output-token position**: We prepend tokens BEFORE the input (attention sink disruption). They perturb the first OUTPUT token (distribution shift). Architecturally different.
3. **Verifier-free plurality voting**: They require a verifier. Our minority-correct plurality voting (32%→72%) needs NO verifier.
4. **Small quantized models**: They use 7B+. We show effects on 4-bit 4B (~2B effective).
5. **Non-monotonic dose-response**: We find 2 tokens optimal, 3+ hurts. Not reported elsewhere.
6. **Quantization-dependent sensitivity**: 4-bit null vs 8-bit +16pp on same model. Novel.
7. **Cross-domain**: Legal reasoning, planning, attention-sink rescue. Not in their paper.

### Honest positioning
> "Soft Reasoning (Zhu et al., ICML 2025) demonstrates that embedding perturbation improves reasoning via Bayesian-optimized search with verifier guidance. We show a complementary and arguably more surprising result: purely random perturbation with no optimization and no verifier achieves strong gains on small quantized models, with a sharp non-monotonic dose-response and effective verifier-free plurality voting."

---

## THREAT LEVEL: MODERATE — Wang et al. (2502.11027)

### Paper: "On the Effect of Sampling Diversity in Scaling LLM Inference"
- **Authors**: Wang, Liu, Chen, Light, Liu, Chen, Zhang, Cheng

### What they do
- **Discrete text-level** prompt rewording (NOT continuous embedding perturbation)
- Five perturbation styles: Role, Strategic Instruction, Jabberwocky, RandIdeaInj, RandQReph
- Frontier models: GPT-4o-mini, Claude-4-Sonnet, DeepSeek-V3
- Best-of-N with ground-truth verification + LLM-as-Judge
- Formal theorem: diversified sampling always improves Pass@N
- **Explicitly warn**: majority voting does NOT benefit from diversity (Theorem 3.3)

### Overlap with our work
- Diversity helps Best-of-N (theoretical grounding we can cite)
- Diversity-fidelity tradeoff (related to our dose-response)
- Majority voting warning (aligns with our majority vote catastrophe finding)

### What's different
- All discrete text-level, no continuous embeddings
- Frontier models, not small/quantized
- N=100 samples, not N=5-16
- No minority-correct regime analysis
- No plurality voting mechanism

### Our position
Cite for theoretical grounding. Different mechanism entirely (discrete vs continuous).

---

## THREAT LEVEL: LOW — Other Papers

### COCONUT (arXiv:2412.06769, Meta)
- Feeds hidden state back as input (continuous recurrence). Requires training.
- Different approach entirely. Cite as related continuous-space reasoning.

### Scaling Test-Time Compute (arXiv:2408.03314, Snell et al., ICLR 2025)
- Optimal compute allocation between parallel sampling and sequential refinement.
- Our work is about increasing coverage, orthogonal to their selection/revision focus.

### Soft Thinking (arXiv:2505.15778, NeurIPS 2025)
- Soft aggregation over vocabulary during generation. No perturbation.
- Different mechanism. Cite as related.

---

## Summary: Novelty Assessment

### Anticipated by prior art (must NOT claim as novel)
- Diversity helps Best-of-N (Wang et al.)
- Embedding perturbation can diversify reasoning paths (Soft Reasoning)
- Greedy decoding + noise = deterministic diversity (Soft Reasoning)

### Genuinely novel in our work
1. Random perturbation sufficiency (no optimization needed)
2. Non-monotonic dose-response with sharp 2-token optimum
3. Prefix position (before input, not output token)
4. Verifier-free plurality voting in minority-correct regime (32%→72%)
5. Quantization-dependent sensitivity (4-bit null vs 8-bit +16pp)
6. Small model regime (4-bit 4B, ~2B effective)
7. Cross-domain validation (arithmetic + legal + planning + attention-sink rescue)
8. Oracle coverage (100% at N=10 from 10 random directions)

### Required honest positioning
The paper CANNOT claim "first to show embedding perturbation helps reasoning" — Soft Reasoning got there first (ICML 2025 Spotlight). The paper CAN claim:
- First to show RANDOM perturbation (no optimization) suffices
- First to show it works on small quantized models
- First to discover the non-monotonic dose-response
- First to demonstrate verifier-free plurality voting in minority-correct regime
- First to show quantization precision modulates perturbation sensitivity
