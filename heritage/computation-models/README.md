# New Computation Model

**The goal.** Do what Turing did in 1936: define a genuinely new model of computation,
prove what it can do (a capability theorem) and what it provably cannot do (an
impossibility theorem), and show it is better in some useful way on some class of
tasks. The bet is that Turing's quiet axioms (finite symbols, finite states, local
steps) are dials, and that every model that later stuck made a previously free resource
first-class and came with a "cannot" theorem. Full statement: [AGENTS.md](AGENTS.md).

**Where it stands (2026-10-08).** The project is ACTIVE: an autonomous exploration loop (three AI engines plus an orchestrator; see `ROUNDTABLE/`) has been searching for a new model since 2026-10-07. It has produced
reviewed, scoped theorems, but **no new model of computation and no verified capability/impossibility pair that survives translation into a known model and beats the standard one.** Every result is
either partly known in the literature, a calibration, or a component. The reviewers are AI agents, not humans, and the same tool families produced the work.

What exists:

- **Erasure and statistical contexts (October 3-4).** Exact thresholds and work laws for computing AND (and general functions) on a reversible-plus-RESET circuit under a cap on per-step entropy loss. The cost meter is a
  logical ensemble quantity; the attempt to tie it to a real reset device failed.
- **Endurance-limited storage (October 4).** Universal lifetime ceilings, a query-depth law, a checked finite certificate. The construction did not survive a workspace-matched incumbent (October 7).
- **Lease theorems (October 7-8, side results).** For reversible circuits with borrowed bits that must be restored within a window (a "lease"): the computable functions have algebraic degree at most lease + 1 at any gate count
  (proved at scope, independently attacked), and for group-valued outputs the lease constrains only the abelian quotient (verified upper bound; constructions for five named groups; the general claim is refuted by a counterexample).
  They are Reed-Muller degree classes plus Barrington-style commutator mechanisms, and ordinary circuits run every witness verbatim: not a new model.
- **A large negative record.** Over 40 July candidates and roughly 170 pre-registered units in October (storage, circuits, catalytic space, proof complexity, scheduling, task-semantics models, calibrated cost-spectrum censuses
  over dozens of models) were killed by equal-resource translation. The pattern: the gadget side is occupied by an incumbent and the impossibility side is an elementary fact.

## Read in this order

1. [docs/STATUS.md](docs/STATUS.md) — where we are and what is running
2. [docs/RESULTS.md](docs/RESULTS.md) — every established result, with scope and labels
3. [docs/ATTEMPTS.md](docs/ATTEMPTS.md) — everything tried and why each direction ended
4. [docs/DECISIONS.md](docs/DECISIONS.md) — the direction-setting decisions
5. [docs/LESSONS.md](docs/LESSONS.md) — what the record teaches
6. [docs/MAP.md](docs/MAP.md) — where everything is

## Layout

| Folder | Contents |
|---|---|
| `docs/` | The synthesis above, the claim ledger, an old-path to new-path map |
| `research/` | Definitions, theorems, constructions and reviews, grouped by line of work |
| `experiments/` | Registered finite CPU controls, ledger |
| `results/` | Raw outputs and receipts for the controls |
| `literature/` | Papers read and source-comparison notes |
| `archive/` | Logs, review prompts and dialogue transcripts (provenance only) |

## Conventions

Every claim carries a rigor label: PROVED / EMPIRICAL / CONJECTURE / HEURISTIC /
SPECULATION. Nothing is recorded PROVED until a fresh independent reviewer has
attacked the proof and failed. Failures and negative results are kept, not deleted.
Rules and scope: [AGENTS.md](AGENTS.md). Branch: `codex/can-cannot-research-2026-10-03`.
