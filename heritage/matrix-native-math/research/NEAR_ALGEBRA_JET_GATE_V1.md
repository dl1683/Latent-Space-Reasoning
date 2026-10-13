# Near-algebra jet gate v1 (unit near_algebra_jet_gate_v1)

Distance from claim: 1 (the object the mission needs: a compact operator descriptor that composes natively with a certified tail).
Headline if it passes: "a laptop replaced giant physics matrices with arithmetic on dozens of numbers, reproduced the public simulation, and certified the answer". Verdict today: WEAK (no workload evidence).
Origin: 2026-10-02 meditation (Stateful Codex T1 proposal; Droid and Antigravity agreed on the gate; Droid supplied the window inequality). Closest prior art: Dyson/Volterra series, Frechet derivatives (Al-Mohy/Higham), Krylov action, certified reduced basis (Rozza/Patera). Gap to show: a small composition-closed first-order descriptor, compact on held-out operators, with a cheap certified tail.

## PRE-REGISTRATION
Family: fixed-mesh 2D variable-coefficient diffusion step A(theta) = A0 + eta*E(theta) (A0 = constant-coefficient Laplacian-type step on an m x m grid, exact small algebra; E(theta) = sparse coefficient perturbation of spectral norm 1 built from theta, NO normalisation of eta). Training thetas: 16; holdout thetas: 16 (disjoint draws). Grid m in {16, 32}; eta in {1e-4, 1e-3, 1e-2}; depths L in {2, 4, 8}.
Objects: F0 = A0-part words; F_q = span of words with exactly q insertions of E-type perturbation, stacked over training thetas; numerical dimension d_q at relative singular-value threshold 1e-8 of the leading value (scale preserved).
H0 (boring): d_1 and d_2 grow with the ambient dimension (saturate toward n^2 or n*|train|), the held-out first-order jet error exceeds the drop-E error, and the plain tail certificate is vacuous by L=8.
H1: d_1 <= 0.1 * n (m=32: n=1024) and d_2 <= 0.25 * n at L=8, holdout application error of the first-order jet <= 1.0x its computed tail certificate and < 0.1 x the drop-E error, certificate <= 20x the true error at L=8.
Kill rule (any one): d_1 or d_2 above the H1 bound at m=32; holdout error above its certificate; certificate > 20x true at L=8 for eta=1e-3; or no regime with eps_jet <= tau < eps_drop (Droid window) showing lifecycle win (setup + Q x jet apply) over sequential sparse apply and drop-E for Q >= 100.
Baselines: drop-E (A0 only), sequential sparse apply, Krylov/expm-action where applicable. All on CPU; GPU stays idle for this unit.
Controls: negative (random dense E: d_1 must saturate), positive (E in the algebra: d_1 = 0).

## AMENDMENT 1 (2026-10-02, before any run; the hashed registration above is untouched)
Source: Droid strategy review 22:17 (owner default model), findings accepted by the lead.
1. d_q is the dimension of F_q modulo span(F_0) (the quotient), not of the raw span; the positive control (E inside the algebra) must give d_1 = 0 only under this definition.
2. Output contract: action on a probability/state vector, relative l2 error <= tau, computed (not observed) certificate; the dense operator is NOT an output. Setup costs (basis discovery, F_q construction, certification) are charged and amortised over Q in {1, 10, 100, 1000} queries.
3. Added baseline and weaker-null correction: a priced first-order Frechet/Dyson method (apply A0 and E sequentially with first-order correction, no stored descriptor) is the decisive comparator; 'jet error > drop-E error' is demoted to a sanity check since Taylor theory predicts the opposite. The jet counts as new only if its descriptor beats that method on the full lifecycle.
4. If this diffusion family loses to its specialist, vary the algebra/perturbation structure and re-test the same proposition; do not return to certificate tuning as theory.
