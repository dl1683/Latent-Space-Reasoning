# Tesla Session Status — 2026-04-15

## Blueprint: CONVERGED (Codex Round 8)
- `phase6_blueprint.md` — Phase A implementation blueprint, ready for engineering handoff
- All 5 Round 7 bugs fixed and verified
- Two cosmetic items remain (stale "KV cache" comments, duplicate sentence) — engineer can handle
- **Verdict**: "Design is converged. Blueprint is ready for engineering handoff."

## Cross-Domain Research: CONVERGED (Codex Round 2)
- `cross_domain_research.md` — 14 domains, critical competitor analysis, ensemble theory frame
- Codex approved ensemble theory as paper's formal frame, replacing cross-domain analogies
- All overreach corrections applied (Kramers, SGD, bimodality, Lyapunov)
- Pre-implementation measurement contract defined
- **Single remaining gap**: Causal decomposition of gain

## Gated Attention Probe: DESIGNED
- `gated_attention_probe_design.md` — experimental design for Qwen3.5-4B vs Qwen3-4B
- Feasibility confirmed: Qwen3.5-4B fits on 24GB GPU
- 4 mandatory conditions + interpretation criteria pre-registered
- Needs Codex review before execution

## Implementation Priority (Codex-Approved Order)
1. Temperature/prompt-perturbation head-to-head (fastest novelty check)
2. Qwen3.5-4B gated-attention transfer probe (submission-gating)
3. Error correlation matrix + trajectory clustering (validates ensemble frame)
4. Selector/self-certainty comparison (paper viability)
5. Position-control ablation battery (causal decomposition)
6. Q4/Q8 RMS sweep (dithering hypothesis)
7. Heavy-tailed prefix noise (after baselines)
8. Ratchet/Lyapunov probes (secondary mechanism)

## Key Documents Produced This Session
| File | Purpose | Codex Status |
|------|---------|-------------|
| phase1_decomposition.md | Component breakdown | Reviewed R1 |
| phase2_mental_assembly.md | Original TCA design | Superseded |
| phase2_revised.md | Phase A/B split | Reviewed R3 |
| phase2_1_corrections.md | RMS fix, 7 corrections | Reviewed R4 |
| phase3_stress_test.md | 9 stress tests | Reviewed R5 |
| phase5_hypotheses.md | H1-H6 preregistration | Reviewed R5 |
| phase6_blueprint.md | **THE BLUEPRINT** | **CONVERGED R8** |
| cross_domain_research.md | 14-domain synthesis | **CONVERGED R2** |
| gated_attention_probe_design.md | Probe experimental design | Needs review |

## Codex Review Summary
| Round | Session | Gate Result |
|-------|---------|-------------|
| 1 | Phase 1 | "Wrong unit of control" |
| 2 | Phase 2 | "Observer assumption existential; blocked" |
| 3 | Phase 2b | "RMS bug; blocked" |
| 4 | Phase 2.1 | "Gate opened for Phase 3/5" |
| 5 | Phase 3/5 | "Gate opened for Phase 6" |
| 6 | Phase 6 | "Architecture converged; boundary_mode fix needed" |
| 7 | Phase 6 final | "5 code-level bugs; not yet handoff-ready" |
| 8 | Phase 6 R8 | **"CONVERGED. Blueprint ready for engineering handoff."** |
| X-1 | Cross-domain | "3 MISLEADING, 4 SUGGESTIVE; missing best-of-N comparison" |
| X-2 | Cross-domain R2 | **"CONVERGENT. Ensemble theory = paper frame."** |

## Next Steps to Start Implementation
1. Run `experiments/probe_inputs_embeds.py` on Qwen3-4B Q4 (Component 0)
2. Implement temperature best-of-N baseline (Priority #1)
3. Download + probe Qwen3.5-4B for gated attention test (Priority #2)
4. Compute error correlation matrix from existing N=10 data (Priority #3)
