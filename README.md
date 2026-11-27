# Latent Space Reasoning

Build one model that reasons over a changing multimodal corpus in a persistent latent space. Swarms maintain that same space before future questions arrive. Memory is optional in the same call. The model should return grounded findings, conflicts, and consequential unknowns, with exact sources available when needed. [Goal and anti-drift rules](GOAL.md)

## Current status

**Learned evaluation is blocked.** The four-case independent ordinary-reader gate is frozen and unrun. Resume requires explicit one-time owner authorization and the custodian's input-only JSON. The grader stays with the independent custodian. The existing Linux/Qwen controller profile needs requalification before launch. Confirm a policy-compliant backend, the 12 GiB free-RAM floor on the actual host, and a hard process deadline before any run. [Independent protocol](docs/INDEPENDENT_ORDINARY_GATE_AND_SCALAR_BOUNDARY.md#four-fresh-ordinary-reader-cases-frozen-out-of-band) · [Controller contract](docs/CONTEXT_TOOL.md#running-the-four-case-ordinary-calibration)

| Evidence | Scope |
|---|---|
| Versioned memory publication, same-pin operation removal and restoration, and source-linked task receipts work in deterministic fixtures. | Learned task addressing, grounded answer gains, and full lifecycle cost remain unmeasured. [Operator receipt results](docs/RESULTS.md#binding-composed-operator-uses-into-text-task-receipts-27-november-2026) |
| The ordinary CPU runner verifies frozen input bytes and seals first-answer records. | The earlier full-text diagnostic scored 0/18 complete answers. The independent reviewer verified four feasible scripted routes. Independent generated answers remain unrun. [Earlier diagnostic](docs/RESULTS.md#fresh-06b-reader-competence-screen-27-november-2026) · [Reviewer feasibility](docs/INDEPENDENT_ORDINARY_GATE_AND_SCALAR_BOUNDARY.md#offline-access-feasibility-not-competence) |
| Partial linear evidence can bound a declared downstream scalar. Stronger cold baselines removed most of the apparent codec reuse advantage. On public package metadata, a mixed cache saved source reads at one budget and lost at another. | These controls use declared linear laws, codec interfaces or supplied query masks. They constrain the research design. [Composition bounds](docs/RESULTS.md#bounding-a-downstream-decision-from-partial-operator-evidence) · [Closed codec line](docs/RESULTS.md#closed-codec-equivalence-controls) · [Residual read costs](docs/DUAL_READOUT_RESIDUAL_COST.md) |

The latest public symbolic diagnostic found that shared vocabulary can suggest a bridge while leaving the conclusion unresolved. Treat bridge discovery and decisive evidence selection as separate steps in the next proposed study. [Shared-vocabulary witness](docs/SHARED_VOCABULARY_WITNESS.md#decision)

After authorization and input-only handoff, qualify all four ordinary-reader cases once. Independent grading must pass all four first answers. Qualify the planner and complete the integration preflight before opening the separate held comparison. That comparison must hold the source pin and reasoner fixed, remove and restore the maintained computation, include memory omitted, and charge the full lifecycle against a competent text reader and warm exact control. [Evaluation sequence](docs/LATENT_WORLD_SYSTEM_DESIGN.md#7-research-sequence-and-decision-gates) · [Whole-world evaluation](docs/WHOLE_WORLD_EVALUATION.md#blinded-revision-to-task-experiment)

Rank 1 is a separate paused line. Its repair cap and resource reconciliation still apply. [Rank 1 resume rules](docs/RESULTS.md#rank-1-state-and-resume-requirements)

## Where things live

Read only the section relevant to the next decision. The ledgers preserve detailed evidence and older failures.

| Need | Start here |
|---|---|
| Owner goal and what falsifies a proposed slice | [GOAL.md](GOAL.md) |
| System direction and candidate reasoning mechanisms | [One-page design recommendation](docs/LATENT_WORLD_SYSTEM_DESIGN.md#recommendation-in-one-page) · [Canonical architecture](docs/ARCHITECTURE.md) |
| Measured outcomes, refusals and earlier evidence | [Results](docs/RESULTS.md) · [Failure ledger](docs/FAILURES.md) |
| Current evaluation sequence, source custody and runner interfaces | [Evaluation gates](docs/LATENT_WORLD_SYSTEM_DESIGN.md#7-research-sequence-and-decision-gates) · [Context and runner protocols](docs/CONTEXT_TOOL.md) |
| Source-only application | [Public interfaces](docs/CONTEXT_TOOL.md) · [Application entry point](experiments/latent_context_app.py) |
| Current research code | [Task episodes](experiments/memory/task_episode.py) · [Retained operations](experiments/memory/exact_latent_linear.py) · [Ordinary-reader controller on hold](experiments/memory/run_independent_ordinary_gate.py) |

Local operating instructions apply in addition to this README. Keep private process material out of Git and sealed evaluators outside the checkout. Verify current authorization before starting any experiment.

## Running the tests

From the repository root, use Python 3.11 or newer. For a fresh environment, see the [pinned CPU dependencies and tokenizer setup](https://github.com/dl1683/Latent-Space-Reasoning/blob/a67277de262c80ac6d726fe55046a35f71002691/README.md#running-the-tests). The following runs a focused offline software check.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -B -m pytest -q -p no:cacheprovider experiments/memory/test_exact_latent_linear.py experiments/memory/test_independent_ordinary_input.py experiments/memory/test_iterative_text_competence.py experiments/memory/test_native_operator_inventory.py
```

## Run the application

The source-only context tool is a navigation slice. With NumPy, psutil and MCP installed, inspect its interface with:

```powershell
python -B -m experiments.latent_context_app --help
```

[Source-only consumer guide](docs/LATENT_MCP_PROJECT_GUIDE.md) · [Application and service interfaces](docs/CONTEXT_TOOL.md)

Closed calibrations and retired runners are linked from the [evidence ledger](docs/RESULTS.md#evidence-and-recovery). Use `git show <archive-commit>:<path>` to recover a pinned file locally.
