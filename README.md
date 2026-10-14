# Latent Space Reasoning

**Intelligence that keeps its understanding—and learns to compute differently.**

We investigate persistent latent memory, the mathematics of high-dimensional spaces
and neural networks, and new ways to construct, train, and execute intelligent
systems. The ambition is to make learned knowledge reusable, revisable and cheaper
to compute with, rather than reconstructing every problem through repeated language
generation.

*Can a machine keep an unfinished thought, change part of it, combine it with new
knowledge, and continue reasoning without rebuilding everything?*

This is the sole research home for the former Latent Space Reasoning, Matrix Native
Math, LLM Genome / Neural Dissection Lab, and New Computation Model programs.
Their evidence and failures now inform one program. Its organizing tracks are:

| Track | Central question |
|---|---|
| **Latent memory and long-context computation** | What must a persistent state preserve to answer unexpected later questions, simulate consequences, and repair understanding without replaying the whole history? |
| **New model construction, training and inference** | Can new representations, layer operations and learning procedures reuse pretrained knowledge while reducing the complete cost of capability? |

High-dimensional mathematics, neural dissection, matrix-native arithmetic,
representation-changing algorithms, capability interoperability and computational
model theory are shared foundations. They remain substantive research ambitions,
including beyond AI. A new computational framework may emerge from this work; it is
not an established result or a required starting assumption.

## The approach

Build representations that preserve useful relationships and support operations:
addressing, composition, selective revision, branching and continued computation.
Compare several carriers rather than assuming one giant embedding is sufficient.
Keep exact evidence recoverable and measure when compressed state needs reopening.

Reuse pretrained encoders, embeddings and model computations where useful. Explore
new learned readers, updaters and layer operations. Ask which expensive computation
they actually replace and which capabilities survive. Mathematical theory,
construction and experiments develop together.

**Construction, training, conversion, memory, retrieval, precision, repair and
fallback all count.** Fewer generated tokens or model calls alone is not a win.

## What bringing the programs together gives us

Memory becomes a question about **which future operations a state preserves**:
reading, acting, revising, reopening history and simulating have distinct contracts.
Learning becomes a question about the complete transition—including optimizer state
and supervision—and whether its objective preserves the later distinctions we need.
Representation becomes an algorithmic choice: compact descriptors, retained factors
and approximations can coexist, with different interaction and continued-error costs.

Those connections now have [worked source-grounded implications](docs/UNIFIED_EVIDENCE.md#what-studying-the-programs-together-changes),
[concrete starting choices](theory/RESEARCH_PROGRAM.md#starting-choices-not-a-launch-queue)
and an [operation-directed reuse guide](heritage/README.md#borrow-by-operation-with-its-scope-attached).
The next question can concern persistent memory, a new pretrained-layer computation,
independent mathematics or cooperating components; each has an informed starting route.

## What the evidence currently supports

- Conditional mathematics of future distinctions, selective writes and composition;
  scoped high-dimensional interference and allocation results.
- Concrete neural-dissection observations about formation, readout, forgetting and
  maintenance, with architecture/optimizer/scale limitations.
- Matrix-representation growth, closure and approximation constraints, alongside
  engineering experiments whose claims changed under stronger specialist baselines.
- Scoped resource/restoration theorems and a substantial negative record showing
  why equal-access conventional implementations must be taken seriously.

**We have not established a general persistent latent reasoning architecture,
arbitrary capability transfer, a new model of computation, or an end-to-end
efficiency advantage.** The latest compressed-state constructions failed their
registered admission criteria; that constrains those constructions, not the mission.
See the [corrected evidence map](docs/UNIFIED_EVIDENCE.md).

## Start here

1. [Research program](theory/RESEARCH_PROGRAM.md): full ambition, two tracks,
   operational contracts, shared foundations and construction questions.
2. [Current state](STATE.md): execution status, inherited verdicts and resume boundaries.
3. [Unified evidence](docs/UNIFIED_EVIDENCE.md): usable results, corrections, failed
   approaches and what a cross-foundation transfer must establish.
4. [Research library](heritage/README.md): selected original mathematics, reference
   code and source records with provenance.
5. [Assets and recovery](docs/ASSETS.md): datasets, checkpoints, historical evidence
   and how to recover selected local artifacts.
6. [Experiment record](experiments/EXPERIMENTS.md): existing results and negatives;
   new work continues in this repository.
7. [Research history](docs/RESEARCH_HISTORY.md): earlier arcs, breakpoint vocabulary
   and the provenance of the shared foundations.

## Repository map

```text
theory/                  Unified program and mathematical work
experiments/             Active experiment code, configurations and evidence ledger
heritage/                Historical source library; reference-only, no launch queues
docs/                    Evidence synthesis, custody, handoffs and research history
tools/                   Repository utilities, including selective asset recovery
assets/                  Local recovered data/checkpoints; ignored by Git
legacy/                  Earlier LSR program, preserved as historical evidence
STATE.md                 Current authority for status and allocation
NOTEBOOK.md              Research journal; historical entries retain their scope
```

Existing LSR code and evidence paths are retained. Historical scripts and old
"READY" instructions are not automatically adopted experiments. Preparation of the
unified program does not launch compute or restore retired automation.

Licensed under [MIT](LICENSE) for repository-owned material. External data, model
assets and third-party dependencies retain their own terms.
