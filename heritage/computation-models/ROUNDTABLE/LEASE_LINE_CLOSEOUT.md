# Lease line closeout (Claude, 2026-10-07 night) — status: ABSORBED at the tested contracts
Evidence: note_3/4/5, r3/r4/r5 reports (Codex, Droid), grunt verifications, FACTCHECK_gen1.md.
Result: for one borrowed bit with a restoration lease ell, immutable inputs, NCT gates: every computable f has F2-degree <= ell+1 (any gate count); ell=2,3 => exactly degree<=3
(CONJECTURE with proof; independent attack by Droid found no gap; 1,978,827-map enumeration independently reproduced twice). Droid E6: increment span = RM(3,n) (leases 2,3; n=4..8).
Brittleness: double-star AND4 at lease 3; one mutable input gives AND_n at lease 2; alphabet changes the class. Variants (Codex E7): S3 output => C3^(2^n) x| RM(3,n) (degree statement,
no deadline needed); quantum borrowed qubit => RM(4,n) phases (degree again; remaining wall is the known information-disturbance tradeoff); probabilistic restoration => no function impossibility.
Prior art: Reed-Muller->Toffoli synthesis (Gupta-Agrawal-Jha 2006), dirty-ancilla commutators (Barenco 1995, Gidney 2017), group programs (Barrington-Therien 1988; BST 1990).
Not absorbed (open, but weak): the exact 'maximum consecutive disturbance' meter has no verified prior treatment (absence of search results is not evidence of novelty).
Verdict: not a new model of computation and no advantage over an equal-resource ordinary reversible circuit. Keep as a documented side result. Do not promote.
Second line closed by evidence this session after endurance (workspace-matched flat log dominated the compact construction; retained cache removed its edge).

## Addendum: restoration-window catalytic logspace (Droid round 6, 2026-10-07 night) — constant window ABSORBED into L
CONJECTURE with an elementary proof (not independently reviewed): catalytic machine with polynomial tape, O(log n) clean space and a restoration window of ell steps. Because correctness must hold for
EVERY initial tape, run on the all-zero tape; at most ell cells differ from zero since the last whole-tape return; an ordinary machine stores those cells and addresses in O(log n + ell log n) space.
So constant ell gives exactly L; in general class <= SPACE(log n + ell log n) (the window is just a sparse-dirty-set space charge). Control: 10,002 addressed-tape programs, 40,008 executions
matched the sparse simulation step by step; finite census (2 data bits + 1 borrowed bit): windows 1,2 compute 8 affine functions, windows 3,4 all 16 functions. Citations verified (Cook-Mertz CCC 2022;
Buhrman-Koucky-Loff-Speelman TOCS 2018). Droid tally: 15/15.

## Addendum: Codex round 6 (job reaped for low system memory before its final report; artifacts read directly)
Codex independently reached the same conclusion for the global restoration-window catalytic TM: a sparse zero-tape simulation puts WCL[ell] inside ordinary space O(s+ell) (sequential head,
interval buffer) or O(s + ell log c) (random access); by Williams (2025) this yields a larger-L plateau for windows up to the ordinary-space budget. Finite checks: 17,520 scalar owner trajectories,
1,065,536 source steps, 2,670 certified zero-input blocks, 33 synthesis certificates, brute-word census 22,621 words matching BFS at ell=1,2,4,8 (same author, not independent review).
Two independent explorers (Droid, Codex) now agree: restoration window = space charge; no hierarchy. Third line closed by evidence (endurance, lease, restoration-window catalytic logspace).
Note: system memory was ~2.9 GB free of 63 GB with six linker (link.exe) processes at 3-4 GB each, apparently from another session; background jobs were reaped. Not restarted.

## Definition-first scoping (Droid, scope_droid.md, 2026-10-08; research only, no compute)
Six classical-boundary candidates, self-rated equal-resource survival 0.10 / 0.08 / 0.06 / 0.04 / 0.025 / 0.015: (1) lemma-interface throughput in proof complexity (Frege vs bounded-depth Frege on grid Tseitin;
Hastad 2020, Hastad-Risse 2025) with a possible non-counting CANNOT via switching lemmas; (2) scale-certified spectral circuits (Ailon 2014 well-conditioned DFT bound); (3) dynamic certificate traffic (dynamic
connectivity gap (loglog n)^2); (4) composition-debt tensor machine (omega < 2.371177, Dupont et al. 2026; flagged as the likeliest trap); (5) signed-component flow (connectivity depth, Karchmer-Wigderson);
(6) cancellation-budget matching. Droid could not fetch some discovered URLs; Claude verified Hastad 2020 (JACM Sept 2020) and arXiv 2608.16884. TOP: lemma-interface throughput, CONFIDENCE 0.10.
Reading: even the best candidate is a 10% shot; all need an equal-resource comparison before any claim.

## Round 8 (Codex, 2026-10-08): proof-complexity lemma-interface throughput KILLED
Pre-registered (r8_PREREG.md); 12 grid/proof combinations; 16 valid certificates accepted and 36 corruptions rejected by a separate checker; 18,240 exhaustive semantic cases; 68 sealed hashes verified;
sweep 9.3 s, 62 MB. An ordinary interpreter matches the exposed-lemma traces under every weighting. Citations verified by Claude: Nordstrom 2013 (LMCS 9(3)); Hastad 2020 (JACM). Others reported by
Codex (Cook-Reckhow 1979; Ben-Sasson-Wigderson 2001; Beame et al. 1996; Cook-Coullard-Turan 1987) were not independently fetched. FIFTH line closed. Per the repair-cap rule the orchestrator has stopped launching
and has escalated twice; no further rounds without direction.

## S1 prior-art check (Droid, s1_droid.md, 2026-10-08): PARTLY-OCCUPIED
Computing with correlated oracle errors is a structured instance of Bayesian active learning with correlated noisy tests (Golovin-Krause-Ray 2010; Chen-Hassani-Krause 2017, ECED). Calibration CAN/CANNOT is elementary
(diversity over independent error blocks vs repetition; KL bound). Only an open slice remains: a tight geometry-(metric)-dependent call lower bound for the binary-oracle task (no verified source found; absence of search is not proof).
Citations verified by Claude: Chen-Hassani-Krause 2017 (PMLR v54), Kohli 2026 (arXiv 2605.29800). Droid tally: 21/21 including the earlier set. Verdict: not a new model; at most a corollary-level theorem.

## Grunt 3 (agy 3.8 Flash, 2026-10-08): scaling check of Droid's toy-scale kills (work/agy_grunt_3_out.md; Claude re-ran scale_tests.py: exit 0, 31 'mismatches=0' lines, none non-zero, 59 s)
Of 54 round-10 units, only 17 have a size parameter in the code; 37 are hard-coded toy instances (3-vertex triangle, 2 qubits, 1-bit commitment, 4-cell tables...) and would need redesign to scale: those kills carry NO evidence about scale.
Of the 10 scalable 'matched the incumbent' units re-run at 39 larger configurations (e.g. U01 N=4..8, U04 n=4..10, U30 B=4..10, U49 n=7..50), NO gap appeared: the candidate equals the equal-access incumbent exactly.
Caveat: several equalities hold by construction (the incumbent is given the same descriptor/access), so scaling confirms the kill logic, not a deeper claim about unexplored models.

## Round 12 (Droid deep unit, 2026-10-08): lease x output-group — Claude's solvable/non-solvable dichotomy REFUTED; the real divide is the abelianization
Conjecture tested (mine): lease classes are degree-bounded for solvable output groups and collapse for non-solvable. FALSE. At lease 2 the solvable groups S3, A4, S4 and the non-solvable A5 all localize a non-identity output on AND_n
for every n (witness gate counts 4 / 16 / 40 for n = 3 / 4 / 5, all at peak lease 2). The surviving any-length restrictions are on the ABELIAN QUOTIENT: S3 and S4 sign component has Boolean degree <= 3; A4/V4 = C3 component is affine; A5 (perfect)
unrestricted; C2 (abelian) degree <= 3; the commutator-subgroup component escapes the lease (Barrington-style commutator mechanism). Classes unchanged at leases 3, 4; lease 1 only constants. A4 has derived length 2 (Claude's grouping with S4 at length 3 was wrong).
CONJECTURE with proof supplied; independent enumerator completed C2/S3 (<=6 gates) and A4 (<=4) searches; S4/A5 used exact orbit calculations, full map enumeration not done; 24 AND_n witnesses (n=3..8) replayed across 16,128 initial states, 224 minterm programs, all at peak lease 2.
Prior art: Barrington 1989; Barrington-Therien 1988; Barrington-Straubing-Therien 1990; Peladeau-Therien (nilpotent groups). An ordinary machine with identical registers runs each witness verbatim: no advantage. Side result only.
