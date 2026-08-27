# Legal Domain Selector Analysis

## Status: COMPLETE — No viable verifier-free selector found

---

## 1. The Problem

For arithmetic, plurality voting works because:
- There is ONE correct answer (extractable integer)
- Wrong answers are diverse (many distinct wrong values)
- Correct answer forms the largest cluster → plurality selects it

For legal reasoning, plurality voting is **fundamentally inapplicable** because:
- There is NO single correct answer (free-text responses)
- "Quality" is a multi-dimensional continuous variable
- No extraction step reduces responses to comparable tokens
- Every response is textually unique (Jaccard similarity 0.04-0.11)

**Question**: Can any verifier-free signal select the best perturbation seed for legal tasks?

---

## 2. Tested Selectors

### 2.1 Word Count Selector (pick longest response)

**Hypothesis**: Longer responses are more thorough and higher quality.

**Results** (9 tasks with score data):
- Correlation between word count and Codex quality score: r = 0.138 (nearly zero)
- WC selector beats baseline: 5/9 (56%)
- Random perturbation beats baseline: 4/9 (44%)
- WC picks actual best B seed: 4/9 (44%)

**Verdict**: Slightly better than random, but unreliable. Correlation too weak to be a real signal.

### 2.2 Consensus Selector (pick response most similar to other seeds)

**Hypothesis**: The "wisdom of crowds" — the most typical response across seeds represents the central, best reasoning.

**Method**: Compute mean pairwise Jaccard word-overlap between each B seed and all other B seeds. Pick seed with highest mean similarity.

**Results** (11 tasks):
- Consensus picks actual best B seed: 1/11 (9%)
- Random baseline: 1/5 = 20%
- **Consensus is WORSE THAN RANDOM**

**Why it fails**: The best legal responses are the ATYPICAL ones. In task 11, B3 uniquely identified the "real business" defense for corporate veil piercing — an argument no other seed produced. In task 08 (negotiation), B2 provided uniquely detailed leverage analysis. The value of perturbation for legal IS the diversity — the outlier seed that finds the novel argument.

This is the **opposite of arithmetic**: in arithmetic, the correct answer is the common answer (it clusters because there's only one right value). In legal, the best answer is the UNCOMMON one (it diverges because it found a novel insight).

---

## 3. Structural Analysis

### Text diversity
- Jaccard similarity between B seeds: 0.04-0.11 (very low — highly diverse)
- Every perturbation seed produces genuinely different legal analysis
- This is qualitatively different from evolution outputs, where broken scorer produces 5 identical copies

### Word count distribution
- Perturbation seeds: 181-890 words (high variance)
- Task 02: 4/5 B seeds produced 0 words (perturbation destroyed output for GDPR task)
- Task 12: All B seeds 718-790 words (low variance, high output)
- Variance itself doesn't predict quality

---

## 4. Why This Is Hard (Fundamental Analysis)

### Arithmetic vs Legal: Selector geometry

**Arithmetic answer space**: 1-dimensional (integers). One correct point. Many wrong points. Correct point is modal → plurality works.

**Legal answer space**: Effectively infinite-dimensional (free text). No single "correct" point. Quality is a continuous manifold. "Better" responses are scattered across diverse locations in text space → no clustering signal.

### What WOULD work for legal

1. **LLM-as-judge** (expensive but reliable): Use a stronger model or self-evaluation to rank candidates. This IS a verifier, just a soft one.
2. **Trained reward model**: Like Soft Reasoning's Bayesian optimization + verifier. Requires training data.
3. **Human evaluation**: Gold standard but not scalable.
4. **Structural completeness scoring**: Parse prompt for sub-questions, check if response addresses each. Viable but domain-specific.
5. **Hallucination detection**: Legal responses with fewer fabricated citations may be better. Requires knowledge base.

### What definitely DOESN'T work
- Word count (r=0.138)
- Text centrality / consensus (9% < random 20%)
- Any method that assumes "correct answer clusters" — in legal, it doesn't

---

## 5. Implications for the Paper

### What we can honestly claim for legal:
1. **Oracle gap exists**: In 11/12 tasks, best perturbation seed beats baseline (avg +1.6 on 10-point scale)
2. **The model has latent legal knowledge** it can't access via greedy decoding
3. **Perturbation produces genuinely diverse legal analysis** (Jaccard 0.04-0.11)
4. **Best responses are atypical** — the value is in rare insights, not common patterns

### What we CANNOT claim:
1. "Verifier-free selection works for legal" — it does NOT
2. "Plurality voting generalizes beyond arithmetic" — it does NOT (fundamentally different geometry)
3. "CDE provides deployed legal improvement" — only with oracle or LLM-as-judge

### Paper framing:
> "Plurality voting is effective for extractable-answer domains (arithmetic) where error dispersion creates favorable voting conditions. For open-ended domains (legal reasoning), the oracle gap demonstrates that embedding perturbation activates latent knowledge, but *selection* requires quality-aware methods (LLM-as-judge or trained reward models). The open challenge is efficient verifier-free selection in free-text domains."

This is honest and scientifically valuable — it identifies the boundary condition for the plurality mechanism.

---

## 6. Future Work

### Viable research directions for legal selector:
1. **Conclusion extraction + voting**: Many legal tasks have binary conclusions (e.g., "unfairness found" vs "not found"). Extract these and vote. Mid-complexity, might work.
2. **Sub-question decomposition**: Parse legal prompts into sub-questions, count how many each seed addresses. Structural completeness as proxy for quality.
3. **Self-evaluation**: Feed the model its own outputs and ask it to rank. Requires one additional inference pass but no training.
4. **Cross-seed synthesis**: Rather than selecting one seed, combine insights from multiple seeds (like "ensemble of reasoning paths" rather than "select best path").

### None of these are needed for the current paper
The paper's contribution is the arithmetic mechanism (prefix → diverse errors → plurality wins). Legal is supporting evidence for oracle gap only.
