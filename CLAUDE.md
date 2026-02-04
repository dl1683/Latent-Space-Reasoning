# CRITICAL: Evaluation Guidelines

## Automated Scorer Scores Are IRRELEVANT

**The numeric scores from our latent scorer (the trained judge) DO NOT MATTER for assessing quality.**

Why the automated scores are unreliable:
- The latent scorer was barely trained and may score style rather than actual reasoning quality
- Scores can go wrong in many ways - higher score doesn't mean better response
- The scorer operates in latent space and has no understanding of actual text quality
- These scores are ONLY useful as evolutionary guidance (helping latent mutations), not as quality metrics

## The ONLY Valid Evaluation: LLM-as-Judge with Manual Review

**Every time you test changes or validate improvements, you MUST:**

1. **Save decoded outputs** - Store the actual text outputs (before and after changes)

2. **Use LLM-as-judge subagent** - Launch a Claude Code subagent to manually review:
   - Read the actual decoded text outputs (not scores)
   - Evaluate coherence, correctness, and quality of reasoning
   - Compare before vs after to determine if changes actually improved quality
   - Provide detailed assessment with specific examples

3. **Codex review for cross-validation** - Use Codex for independent verification:
   ```bash
   codex exec "Review these outputs for quality. Be brutally honest about whether the reasoning is sound, coherent, and correct:

   [paste decoded outputs here]" 2>&1
   ```

## What to Evaluate

When reviewing outputs, the subagent should check:
- **Coherence**: Does the response flow logically? Is it well-structured?
- **Correctness**: Is the reasoning accurate? Are conclusions valid?
- **Completeness**: Does it address the full query?
- **Clarity**: Is it understandable and well-explained?
- **Specificity**: Does it provide concrete details rather than vague statements?

## NEVER Trust Numeric Scores Alone

Do NOT make decisions based on:
- "Score improved from 0.6 to 0.7" - This means nothing about actual quality
- "Diversity increased 400x" - Diversity in latent space doesn't equal better outputs
- Any automated metric without manual output verification

The ONLY valid statement is: "I manually reviewed the outputs and the response quality improved because [specific reasons]"
