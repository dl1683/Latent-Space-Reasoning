# STATE

Canonical current state. Updated every session; history in NOTEBOOK.md.

## Direction (set 2026-08-27)

**Native mathematics of latent spaces.** Build the mathematics of latent/embedding
spaces from axioms upward — what a latent space *is*, which relations and
operations are actually meaningful — and derive definitions and propositions from
there, Euclid-style. Not porting ℝⁿ mathematics (vector arithmetic, cosine, Bayes
over coordinates) onto embedding vectors. Classical goals carry over (measure
closeness, update on evidence, compose, infer, prove); the constructs may be
unrecognizable. Scope: existing embedding spaces, constructed latent spaces under
our own axioms, and maps between latent spaces. Axioms may be discovered from
measurement or posited and tested.

**Mode.** Codex (real CLI) drives theorems and thoughts; Claude challenges,
curates, runs cleanup, keeps the loop going. Autonomous, continuous.

## Live question

What is the smallest axiom set under which a latent space has a well-defined
native notion of (a) closeness, (b) evidence/update, (c) composition — and which
of those notions, when instantiated on a real embedding space, predicts something
ℝⁿ-imported math gets wrong?

## Constraints

- No GPU runs without explicit user approval (laptop hard-crashes under sustained
  load). Theory and CPU work unconstrained. Qwen3-0.6B embeddings for local probes.
- Evidence gates in the local process file apply to any empirical claim.

## Prior program (closed 2026-08-27)

LLM perturbation line: nested-arithmetic claims withdrawn
(`docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md`). Standing results and
hypotheses are in `README.md`. Residue carried forward: the hardware-dependent
decoding-determinism finding; the termination/direct-control/null-model gates.

## Next

Align the analyzer with the locked Round-2b instrument: per-paraphrase scale
normalization, pooled magnitude threshold, four-of-four sign rule, 72-word
primary exclusion, H2 anchor-bootstrap ratio using both halves, and genuine
per-replicate baseline reselection. Raw three-system matrices already exist and
must remain outcome-sequestered until the amendment is committed; analyze them
rather than rerunning measurement. Directed asymmetry is exploratory; primary
claims are context non-collapse and held-out transfer. Exact predictions and
kill conditions are in `theory/EXPERIMENTS.md`.
