> **Note (2026-08-27):** this internal dialogue predates the correction of the
> nested-arithmetic perturbation claims. Figures such as "32%→72%" and
> "perturbation beats scaling/temperature" quoted below are withdrawn — see
> [docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md](../docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md).

Most dangerous: **1 and 3 together**.

If perturbation only recovers outputs that temperature sampling already reaches at matched compute, and those outputs already have non-negligible baseline probability, then the thesis mostly collapses. It becomes “continuous prompting is another sampler,” not “there is a structured accessibility landscape.”

Quantization is second-most dangerous. It can demote the thesis to “quantization roughens routing.” That is narrower but still potentially interesting. Temperature/tail equivalence is worse because it kills the claim that embedding-space access is a distinct interface.

**One Experiment**
Run a **precision-matched token-vs-embedding accessibility census**.

Use 500-1,000 verifiable tasks, filtered into baseline-wrong and baseline-correct controls. Run the same tasks on:

- Same model in fp16, 8-bit, 4-bit
- At least one non-Qwen model family
- Same prompt, same answer extractor, same compute budget

For each task/model/precision, compare four arms:

1. Greedy baseline
2. Temperature sampling grid, matched total samples: `T = 0.2, 0.5, 0.8, 1.0, 1.3`
3. Embedding perturbation sampling, same number of samples
4. Embedding perturbation plus answer-blind selector, trained only on non-answer features

Log: output, correctness, answer string, reasoning trace, baseline logprob/rank of successful answers, hidden trajectories, perturbation vector, local-neighbor perturbations.

This single census answers the core objections:

- If temperature finds the same correct answers, perturbation is not special.
- If fp16 loses the effect, quantization is probably central.
- If perturbation-only answers were already high-probability tails, “dark knowledge” is inflated language.
- If successful perturbations have locality, transfer, low-rank predictability, or answer-blind selectability, the landscape is real even if not discrete.

**Probability**
Strong thesis survives rigorous testing: **30%**.

By strong thesis I mean: structured, navigable accessibility landscape, not reducible to token sampling or quantization damage, visible beyond one small quantized model family.

Weaker thesis survives: **55-60%**. Namely: perturbation reveals real routing fragility and useful hidden competence in some models/tasks, but the effect may be partly precision-dependent and not universal.

Breathtaking universal version: **10-15%** right now.

**Negative Version**
If it is all quantization noise plus token-tail sampling, that still tells us something useful:

- Small quantized reasoners have unstable routing surfaces.
- Correctness under sampling may be more about tail coverage than hidden computation.
- Claims about “model knows but cannot access” need precision and sampling controls.
- Quantization may create artificial competence islands: occasionally useful, scientifically non-fundamental.

That negative result is not worthless. It becomes a warning about overinterpreting perturbation wins. But it is not the breakthrough thesis.

**Exact Thresholds**
Push harder if the census shows all of these:

- Perturbation finds correct answers on **>=15%** of baseline-wrong tasks that matched temperature does not.
- Perturbation oracle@K beats best temperature@K by **>=5 absolute points** on fp16 and quantized models.
- Among perturbation-only successes, median baseline probability of the correct answer is **<=1e-8**, or median critical answer-token rank is **>1,000**.
- fp16 perturbation-exclusive unlock rate is **>=8%** and at least **40%** of the 4-bit unlock rate.
- Cross-precision task unlock correlation is **r >= 0.35**.
- Locality lift: neighbors of successful perturbations succeed at **>=2x** random perturbation rate.
- Answer-blind selector top-5 beats random top-5 by **>=2x** on held-out tasks.
- Low-rank predictor with rank `<=32` gets held-out AUC **>=0.65**.

Pivot if these happen:

- Temperature recovers **>=80%** of perturbation successes at matched samples.
- Perturbation oracle@K advantage over best temperature@K is **<=2 points**.
- Median baseline probability of perturbation-success answers is **>=1e-5**, or critical answer-token rank is **<=100**.
- fp16 perturbation-exclusive unlock rate is **<2%**, or **<15%** of the 4-bit rate.
- Cross-precision unlock correlation is **r < 0.10**.
- Locality lift is **<1.25x**.
- Answer-blind selector lift is **<1.25x**.
- Low-rank held-out AUC is **<0.56**.

Yellow zone: perturbation beats temperature, but only in 4-bit and without locality/selector transfer. That means “quantization accessibility artifact”: maybe worth one short investigation, not months of thesis-building.

