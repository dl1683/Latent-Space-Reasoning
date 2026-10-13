# Reference: neuro-ai-lab (sibling project, permanent idea source)

Source: `<LOCAL_PROJECTS>\neuro-ai-lab` (public mirror:
github.com/dl1683/neuro-ai-lab). Distilled 2026-09-24 by an exploration agent;
re-mine this doc (not the repo) during hourly reviews. Status when distilled:
mostly closed (May 18 – Aug 31 2026), nothing running. Key files if a deep
re-read is ever needed: `STATUS.md`, `docs/CIRCUIT_VIABILITY_PRUNING_REPORT.md`,
`docs/UNIFIED_ERROR_SPACE.md`, `experiments/EXPERIMENTS.md`,
`results/pilot_suite_summary.json`.

## What it is

A solo, agent-driven lab testing neuroscience-inspired claims about artificial
networks with hard evidence discipline: every headline number backed by a
checked-in JSON artifact and recomputed by an audit script. Same hardware as
us (one RTX 5090, ~2.5 GPU-h cap per run).

## Mechanisms implemented AND tested (with results)

- **Synaptic pruning** (magnitude / SynFlow / gradient-saliency masks) + **homeostatic
  capacity reserves** (anti-silencing) — the headline constructive result. At
  98-99% sparsity global SynFlow deletes the whole first classifier bridge
  (fc1) in 3/3 CNNs; a capacity-reserve repair beats magnitude pruning by up to
  **+6.57 pts (4/4 seeds)** on CIFAR-10 ResNet-20-style nets.
- **Feature-subspace preservation beats liveness for transformers:** on
  TinyViT, residual-stream feature cosine predicted post-prune recovery
  (r = 0.583) better than keeping units alive; all-liveness repair
  *underperformed* magnitude. Liveness-at-all-costs FAILED on pretrained
  ResNet-18/TinyImageNet (feature-preserving repair restored parity).
- **Degeneracy / multi-route redundancy:** diversity-penalized route optimizer
  beats magnitude at 99% sparsity on TinyResNet.
- **Variable-T (variable compute-window) training:** sampling solver depth
  T∈[4,16] per batch made an anytime solver — off-window T=32 accuracy
  88.5%→99.9%; mechanism = contraction-rate suppression; replicated 48/51
  across depth/task/architecture. Their cleanest positive.
- **Margin-based early exit:** 99.05% accuracy at 10.4% of compute — but
  accuracy-vs-depth was FLAT (R²=0.018), killing the drift-diffusion story.
- **Reconsolidation pilot (retrieval-gated labilization):** +3.14 pts vs EWC on
  sequential digit tasks — but the EWC baseline equaled naive (weak evidence).
- **Sleep-cycle training pilot:** NREM=decay+prune, REM=noisy-replay →
  generalization gap 0.0174→0.0153 and 40% sparsity at −1.59 pts accuracy
  (a trade, not a win).
- **Self-consistency energy / attractor dynamics (UESD D38-D40): FALSIFIED** —
  stronger self-consistency loss monotonically lowered residual AND accelerated
  accuracy collapse; the only converged run decoded to ~100% wrong attractors.
- **Use-dependent stabilization: FAILED** (narrowed to pure liveness).
- **Grokking/representation-velocity prediction pilot: FAILED** (memorized,
  never generalized — early-warning signals may simply not exist on
  under-trained models).

Written about but never tested: predictive coding, free-energy principle,
Nishimori criticality, morphogenesis/bioelectric framing.

## Process rules worth adopting outright

1. **Evidence architecture:** immutable result JSONs + audit scripts that
   recompute every headline number from checked-in artifacts.
2. **Denominator discipline ("vacuous metric" rule):** every rate must state
   its denominator — their "0% wrong attractors" was vacuous because the
   converged fraction was 0.
3. **Kills are fuel:** ≥3 new hypotheses per kill; failure synthesis after
   3-5 kills in a line.
4. **Matched-budget comparisons** always; many small hypothesis tests over
   scaled single runs.
5. **Three-gate causal-state criterion** (from the LSR deposit — same as our
   prior HANDLE-0 era): present ≠ addressable ≠ composable; most
   interpretability claims stop at gate 1.
6. **Negative map (do not retry as-is):** self-consistency energies as
   correctness, liveness-at-all-costs on pretrained nets, critic selection
   without matched populations, task suites an encoder-only ablation can pass.

## Standing dissection questions for THIS lab (ranked mappings)

1. **MLP-row death vs homeostatic liveness:** does magnitude/SynFlow pruning
   of our tiny LM delete whole MLP rows/organs (cf. our E001 keystone MLP-0),
   and does minimal liveness repair change circuit behavior more than loss?
2. **Feature-subspace preservation as recovery predictor:** does pre-repair
   residual-stream cosine predict post-edit performance in a small LM — a
   cheap predictor for surgery outcomes?
3. **Bridge/cutset collapse:** which transformer route families (embeddings,
   W_O rows, final head) are the cutsets at 98-99% sparsity?
4. **Reconsolidation vs EWC** on sequential tasks in a small LM, with a REAL
   EWC baseline this time (their EWC == naive).
5. **Sleep cycles:** periodic NREM-prune + REM-noisy-replay phases in LM
   continual learning at matched compute.
6. **Variable-depth training:** sample layer-count during training → robustness
   to layer-drop / early exit at inference (their best replicated result; we
   can test the LM + lesion version, which dovetails with our E001 finding
   that late attention layers are near-dead).
7. **Margin early-exit:** is small-LM accuracy flat-in-depth, and can logit
   margin buy 10x compute savings?
8. **Verifier-gated Best-of-N switch accounting** (harmful vs beneficial answer
   switches with explicit denominators).
