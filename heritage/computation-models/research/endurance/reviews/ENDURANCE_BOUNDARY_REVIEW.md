**DESIGN-CONTRACT: PASS.**  
**ENDPOINT-RELAXATION: VALID.**

These are scoped design-review verdicts. The repaired wording closes the previous liveness gap; I found no further essential infinity, advice, controller or wear loophole.

The [termination clause](/<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_BOUNDARY_MODEL_CANDIDATE.md:41) requires initialization to terminate for every \(x\), every QUERY to terminate correctly at every reachable boundary through depth \(H\), and every FLIP to terminate correctly before depth \(H\). Promised invocations must respect \(S\), wear limits and canonical boundary clearing. Divergence therefore cannot avoid a promised endpoint or answer. No termination obligation is needed for unreachable states or unpromised commands.

The endpoint coverage survives independent attack:

- **Stationary decoding.** For every persistent state reached through depth \(H\), define \(d(s)_i\) using QUERY(i). Canonical entry, immutable code, read-only queries and absence of wear feedback make this answer independent of history and hidden wear. Correctness forces histories sharing \(s\) to share their logical vector.
- **Initialization.** Termination supplies \(I(x)\) for every \(x\), with \(d(I(x))=x\). Starting from unworn zeros, actual initialization wear \(w_0\) satisfies \(I(x)\le w_0\) coordinatewise. No preferred encoding or \(I(0)=0\) assumption is required.
- **Stationary updates.** Define \(T_i(s)\) wherever \(s\) occurs before depth \(H\). Identical \(s,i\) yield identical executions across histories and depths: every promised execution avoids exhausted opposite-value writes, so differing hidden wear cannot supply feedback or alter the endpoint.
- **Wear relaxation.** If a command makes actual change-count vector \(c_t\), then \(s_t\oplus s_{t+1}\le c_t\). Along each initial input and flip sequence,
  \[
  v_0=I(x)\le w_0,\qquad
  v_{t+1}=v_t+(s_t\oplus T_i(s_t))
  \le w_t+c_t=w_{t+1}\le L\mathbf1.
  \]
  This covers initialization and all intermediate changes, including undone writes. Arbitrarily interleaved queries preserve cells, wear and canonical entry. Every reachable wear record must remain represented; one representative per persistent state would be insufficient.
- **Domain and horizon.** Extend \(d\) arbitrarily outside reachable boundary states, and \(T_i\) outside states occurring before \(H\). Exploration from every \(I(x)\) follows actual endpoints through \(H\), so extensions add no required histories. States occurring only at depth \(H\) require decoding only. States also occurring earlier retain their stationary update maps.

This argument allows arbitrary nonlinear decoders and initialization, and multi-bit updates. It uses no symmetry, one-bit-transition or linear-decoder assumption.

The [accounting clauses](/<LOCAL_PROJECTS>/new-computation-model/research/ENDURANCE_BOUNDARY_MODEL_CANDIDATE.md:8) charge program generation, installation, code size/access, PC/controller storage, routing, arithmetic and clearing. Parameter-only compiled code cannot serve as input-dependent advice. Offline solver memory cannot establish compliance with runtime \(S\).

Formula completeness, checked UNSAT certificates, and compiled SAT instruction-level execution remain separate gates. This review admits no positive lifetime, journal construction, source priority or completed CAN/CANNOT pair.

Reviewed candidate SHA256: `f0ab692cf629fd1f3c4f35d7e6600c3a09f5a6505b98f9ff776649b3c8d7ec97`.

Read-only throughout; no agents, GPU, solver or implementation. Git status remained clean.