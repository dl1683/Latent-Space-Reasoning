# RAC-1: Response-Algebra Composition — Formal Specification

Registered per Codex round 8 ruling. RAC-0 earned exactly one confirmatory round.

## 1. Definitions

### 1.1 Response law

For a prompt P and hidden state h at layer b, the **response law** R(h) is the
full next-token probability distribution:

    R(h) = softmax(W_u · LayerNorm(F_{b→28}(h)))

where F_{b→28} is the model's forward computation from layer b to the output.

### 1.2 Response tolerance

Two response laws are ε-equivalent:

    R(h) ~_ε R(h')  iff  sqrt(JSD(R(h), R(h'))) < ε

Threshold: ε = 0.05 (corresponding to JSD < 0.0025). This is the noise floor
observed in deterministic fp32 forward passes with identical inputs.

### 1.3 Quotient space Q_b

    Q_b = H_b / ~_ε

The quotient of the hidden state space at layer b by response-law equivalence.
Two states are in the same equivalence class iff they produce ε-equivalent
response laws. Identity is defined by executable future, not coordinates.

### 1.4 Fixed absolute setters

A **fixed absolute setter** S_v for value v is a constant additive perturbation
applied to the hidden state:

    S_v(h) = h + δ_v

where δ_v is fixed (independent of h). The key difference from RAC-0's
transition directions: in RAC-0, the sign of rv/rlv was chosen based on
knowing the source and target cells. A fixed absolute setter uses the same
δ_v regardless of starting state.

**Construction:** For each variable (position, relation) and each value:
- S_pos1 = mean hidden state of all pos1 conditions − grand mean
- S_pos2 = mean hidden state of all pos2 conditions − grand mean
- S_cap = mean hidden state of all capital conditions − grand mean
- S_lang = mean hidden state of all language conditions − grand mean

These are **centred** around the grand mean, making them fixed directions
that should steer toward the target value from any starting state.

### 1.5 Quotient descent

A setter S descends to the quotient iff:

    h ~_ε h'  ⟹  S(h) ~_ε S(h')

Test: For multiple representatives of the same response class (different
prompts that produce the same answer distribution), apply S and check that
the resulting distributions are ε-equivalent.

### 1.6 Overwrite / idempotence

An overwrite setter satisfies:

    S_v(S_v(h)) ~_ε S_v(h)    (idempotence)
    S_v(S_w(h)) ~_ε S_v(h)    (last-writer-wins, for same variable)

### 1.7 Response-law commutativity

Two setters S_a, S_b for different variables commute iff:

    R(S_a(S_b(h))) ~_ε R(S_b(S_a(h)))

Measured by: sqrt(JSD) between the two response laws < ε_comm = 0.10.

### 1.8 Transport defect

The transport defect of setter S under model computation F_{b1→b2}:

    D(S, b1, b2) = sqrt(JSD(R(F(S_{b1}(h))), R(S_{b2}(F(h)))))

where S_{b1} applies S at layer b1, and S_{b2} applies the same S at layer b2.

- D < 0.05: INTERTWINE (setter is a natural transformation)
- 0.05 ≤ D < 0.15: PARTIAL (direction preserved, magnitude differs)
- D ≥ 0.15: DEFORM (setter is not natural at this boundary)

## 2. Gates

### Gate A: Fixed setter efficacy
For each of the 4 setters (S_pos1, S_pos2, S_cap, S_lang), applied to states
from ALL cells (not just the "wrong" cell):
- The setter steers the response toward the target value
- Measured by: Δ_top1 = (target answer becomes top-1) OR Δ_rank (target rises)
- PASS: ≥ 10/12 cells correct top-1 per setter

### Gate B: Specificity
S_pos1 changes position (pos1 answers rise) but does NOT change relation
(capital/language ratio preserved). And vice versa for S_cap.
- Measured by: for each setter, the off-target variable's answer ratio
  changes by < 0.10 absolute.
- PASS: ≥ 10/12 cells specific per setter

### Gate C: Composition
S_pos1 ∘ S_cap = applying both setters steers to the pos1_cap cell.
- Test all 4 composed targets from all 4 starting cells (4×4 = 16 tests,
  minus 4 identity = 12 non-trivial).
- Held-out: leave-one-pair-out extraction.
- PASS: ≥ 10/12 held-out cells correct top-1

### Gate D: Overwrite / idempotence
- S_pos1(S_pos1(h)): response ~_ε S_pos1(h)
- S_pos1(S_pos2(h)): response ~_ε S_pos1(h)
- PASS: all JSD < 0.05

### Gate E: Commutativity
- JSD(R(S_pos1(S_cap(h))), R(S_cap(S_pos1(h)))) at same layer
- JSD(R(S_pos1@b1(S_cap@b2(h))), R(S_cap@b1(S_pos1@b2(h)))) at separated layers
- PASS: all sqrt(JSD) < 0.10

### Gate F: Quotient descent
- Multiple prompts producing the same answer → apply S → same answer
- PASS: all JSD < 0.05 between resulting distributions

### Gate G: Transport defect
- For each setter, measure D across layer pairs: (B16,B20), (B18,B20),
  (B19,B20), (B20,B21), (B20,B22)
- PASS: D < 0.05 within [B16, B21], D > 0.15 at B22

### Gate H: Random direction control
- Generate 20 random unit vectors in R^1024, scale to ||rv|| and ||rlv||
- Apply as single setters and composed pairs
- PASS: random setters achieve ≤ 2/12 correct top-1 (chance is ~0/12)

### Gate I: Logit-additive baseline
- Instead of adding vectors at B20, add them directly to the logits
  (after final layernorm + unembedding)
- PASS: logit-additive achieves lower composition success than B20 injection

## 3. Entity design

### Training entities (vector extraction)
- Tokyo/Japan/Japanese + Rome/Italy/Italian
- Berlin/Germany/German + Paris/France/French
- London/UK/English + Cairo/Egypt/Arabic

### Held-out entities (NOT used in extraction)
- Seoul/South Korea/Korean + Madrid/Spain/Spanish
- Moscow/Russia/Russian + Athens/Greece/Greek

### Templates
- Template A: "{E1} is the capital of {V1}. {E1} speaks {L1}. {E2}..."
- Template B: "The language of {E1} is {L1}. The capital of {E1} is {V1}. {E2}..."
- Template C: "Here, {E1} is the capital of {V1} and speaks {L1}. Also, {E2}..."

## 4. Measurement

All results serialized in one canonical JSON artifact with:
- Clean baselines (no intervention) for every cell
- Single-setter results (4 setters × all cells)
- Double-setter results (all compositions × all cells)
- Full response distributions (top-10 tokens with probabilities)
- JSD values for all commutativity/idempotence/quotient tests
- Transport defect matrix (setters × layer pairs)
- Random direction control results
- Model revision hash, tokenizer hash, git commit

## 5. Stop conditions

- Any gate A-C FAIL with ≥ 4 failures: RAC-1 FAIL, close the algebra program
- Gate H FAIL (random directions work equally well): the effect is not
  directionally specific, close
- All gates PASS: the larger algebra program is earned

## 6. Measurement-to-artifact ratio

RAC-1 is ONE experiment file producing ONE result artifact. The ratio is 1:1.
