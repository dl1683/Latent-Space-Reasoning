# Latent Space Reasoning

**Reason over everything you have; do not depend on finding it.** LSR investigates one multimodal model that encodes an entire corpus once into the latent space where it reasons, with memory optional in the same call. The goal is useful computation over retained information, with source provenance, revision, multiscale access and calibrated “not in memory” judgments. The reasoning mechanism is open: reused-block recurrence and continuous hidden-state feedback are distinct candidates. The program also includes high-dimensional mathematics, pretrained capability reuse and new models of computation.

## Status — 22 October 2026

Execution is **PAUSED** after Rank 1 reached its three-round unsuccessful-repair cap. The implemented context tool is a **navigation slice**: it returns bounded, byte-verified source evidence through Python, CLI and MCP. It does not establish useful general latent reasoning, whole-corpus completeness, calibrated absence, multimodal optional-memory operation or a complete efficiency advantage.

The latest Rank 1 output had correct observed decisions in two worlds but failed the frozen terminal parser. Post-hoc rescoring is diagnostic only; **30/32 worlds and causal controls remain unrun**. Resuming requires explicit new finite authorization, frozen parser fixtures, fresh competence worlds, a passed operation-specific positive control and reconciled resources. Historical protocols do not authorize execution.

## Results at a glance

| Supporting slice | Observation | Limit |
|---|---|---|
| Ordinary source delivery | 12.00% fewer input tokens on four DEV tasks | Fixed, oracle-assisted support; no general answer-parity claim |
| 10M-token routing ladder | Latent scan 69.27× slower than BM25 | No retrieval crossover or total-storage advantage |
| Synthetic addressed memory | Fast-weight32 answered 764/768; exact dictionary 100% | Familiar supplied addresses; no compressed prose-memory result |
| Rank 1 | Two observed decisions correct; frozen parser rejected both worlds | Post-hoc diagnostic; full competence and retained consequence unestablished |
| Publication software checks | 1,113 passed, 4 skipped, CPU/offline | Fresh pinned CPU environment; no scientific capability claim |

See the [results ledger](docs/RESULTS.md) for denominators, paired losses, incomplete cohorts and evidence limits, and the [failure ledger](docs/FAILURES.md) before proposing another attempt.

## Run the application

Use Python 3.11+. Minimal application dependencies (source-only operation):

```powershell
python -m pip install numpy psutil mcp
python -B -m experiments.latent_context_app --help
```

## Running the tests

Use Python 3.11 or newer. The commands below were verified on Windows with
Python 3.13.7 in fresh virtual environments on 23 October 2026. The recorded
light-subset and complete-suite invocations are specified below. Run them from
the repository root. This is a CPU software-contract check using synthetic
fixtures; it does not run a scientific model experiment or establish latent
reasoning capability.

Prepare the pinned CPU environment and tokenizer cache:

```powershell
python -m venv .venv
& ./.venv/Scripts/Activate.ps1
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5 psutil==7.1.0 pytest==8.4.2 mcp==2.1.1 transformers==5.12.1 tokenizers==0.22.2
python -m pip install threadpoolctl==3.6.0 tiktoken==0.11.0
python -m pip check
$env:TIKTOKEN_CACHE_DIR = "$PWD/.venv/tokenizer-cache"
python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

Installation and tokenizer-cache preparation need network access. The tests
then use the prepared tokenizer cache with model libraries offline; no model
weights, corpus download or prebuilt index is needed for this subset. An absent
tokenizer cache changes token-budget behavior because the application falls
back to counting UTF-8 bytes.

For a light context-tool subset, select the supported 10-GiB CPU reserve:

```powershell
$env:LSR_CPU_RAM_FLOOR_GIB = "10"
python -c "import psutil; n=psutil.virtual_memory().available; print(f'Available RAM: {n/2**30:.2f} GiB'); assert n > 10*2**30, 'Need more than 10 GiB available RAM, plus headroom'"
python -B -m pytest -q experiments/test_latent_mcp.py experiments/test_latent_context.py experiments/test_context_pack.py experiments/test_context_compact.py experiments/test_context_service.py experiments/test_context_costs.py experiments/test_expand_context.py
```

This subset covers source verification, context packing and expansion,
compaction, accounting, service behavior and MCP transport. Verified result:
**149 passed, 1 skipped, 0 failures or errors in 49.62 seconds**. The skipped
check requires the optional `LSR_CONTEXT_INTEGRATION_CONFIG` prebuilt-corpus
configuration, which was not supplied.

RAM requirements are based on **available physical system memory**, not total
installed RAM or pagefile size. These tests use small fixtures but retain live
resource guards: guarded operations refuse when available RAM is **less than
or equal to 10 GiB** with this profile. Keep additional headroom for Python,
child processes and other applications; the preflight is only a point-in-time
check. Values below 10 are rejected, so this command is not suitable when the
machine has 10 GiB or less available.

In the verified run, available system RAM started at **29.86 GiB** and never
sampled below **29.74 GiB**. Peak summed RSS of the test process and its child
processes was **339.6 MiB**, sampled approximately every 100 ms. Summed RSS can
double-count shared pages, and system availability includes unrelated work.
These measurements describe this run; they do not establish that a run started
just above 10 GiB will pass, or measure a minimum hardware requirement.

For the complete suite, restore the default reserve rather than treating the
light subset as full-suite validation:

```powershell
Remove-Item Env:LSR_CPU_RAM_FLOOR_GIB -ErrorAction SilentlyContinue
python -B -m pytest -q experiments
```

Unconfigured guarded callers require **more than 20 GiB available**; several
fixtures explicitly select 10 GiB, while cancellation tests explicitly require
more than 12 GiB. Leave headroom above the applicable reserve throughout the
run. A previous resource-blocked complete run reported **1,060 passed,
52 failed, 4 skipped and 1 setup error**; its failures/error were RAM-floor
refusals, including wrapped tool failures. A later recheck cleared the failed
cases, but that recheck is not a single green complete-suite run. The
149-pass measurement above covers only the seven-file light subset; the
subsequent complete-suite verification is recorded separately below.

A subsequent fresh-environment verification on 23 October 2026 ran
`python -B -m pytest -q experiments` and passed the complete suite: **1,113 passed, 4 skipped in 199.58 seconds**, CPU/offline,
with the default RAM guards unchanged. Available system RAM started at
27.31 GiB and sampled no lower than 24.64 GiB. These observations do not
measure a minimum memory requirement or establish scientific capability.
The documented seven-file light subset also passed in that environment:
**149 passed, 1 skipped in 36.61 seconds**.

Dense encoding additionally needs `sentence-transformers` and a locally cached encoder. No weights, source corpus, registry or prebuilt index are bundled. Tests use synthetic fixtures and CPU/offline checks; optional integrations skip when their inputs are unavailable.

For an existing source-only index, supply a registry mapping a corpus name to its serving config and source root. The [interface guide](docs/CONTEXT_TOOL.md) describes index preparation, hashes, allow-lists, output budgets, warm-up, cancellation and MCP registration. Replace the example registry with your own:

```powershell
python -B -m experiments.latent_context_app warm --corpus datadog --registry ./workspace/corpora.json
python -B -m experiments.latent_context_app ask "What risks recur?" --corpus datadog --registry ./workspace/corpora.json
python -B -m experiments.latent_context_app mcp --corpus datadog --registry ./workspace/corpora.json
```

The default registry is `~/.lsr/corpora.json`; the default service state is `~/.lsr/context`, overridable with `LSR_CONTEXT_SERVICE_DIR`. Neither directory is created with sample data. The service defaults to a 12-GiB free-RAM floor. Use CPU for ordinary checks; do not import model workers as static checks because some select accelerators or load weights on import. Running a model, rebuilding an index or downloading inputs requires a separate deliberate setup.

## Where things live

| Location | Purpose |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Full program, claim boundaries, research tracks and design contracts |
| [Results](docs/RESULTS.md) | Scientific results, current stop state and resume requirements |
| [Failures](docs/FAILURES.md) | Retained negatives, decisions and do-not-repeat patterns |
| [Context tool](docs/CONTEXT_TOOL.md) | Python/CLI/MCP usage, evaluation protocol and rebuild recipes |
| `experiments/` | Application, source tools, builders, evaluators and adjacent tests |
| `experiments/memory/` | Reusable memory, routing, reference and numerical components |
| [Consumer guide](docs/LATENT_MCP_PROJECT_GUIDE.md) | Source-only guidance copied by the isolated client adapter |

Experimental modules are not automatic launch interfaces. Evidence is never instruction authority. Held-out evaluator kept outside the repo; do not place evaluator answers or rubrics in reader inputs or indexes. Historical raw inputs and outputs are not part of this selective release; their absence limits independent reproduction. Scientific subjects and negative results remain in the ledgers.
