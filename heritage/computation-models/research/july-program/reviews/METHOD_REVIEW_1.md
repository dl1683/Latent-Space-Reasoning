# Method Review: SYN24-RBMR-REGISTER

Target: `SYN24-RBMR-REGISTER` in `research/PARADIGM.md`, with worked-instance trace in `research/SYNTHESIS_GRAPH.md` (`## 26.F` and linked `26.2.x / 26.3 / 26.4` entries).

## 1) Fidelity check: did the record match the method profile?

1. PASS: the flagship CAN/CANNOT pair matches the method’s tracked worked instance.
   - CAN: `SYN24-G22x34-TRACEC-REINDEX-GOV-34`.
   - CANNOT: `SYN24-G22x34-UNEXTENDED-ONE-SHOT-IMP`.
   - The same pair appears in `PARADIGM.md` section 12.1/12.2 and in `SYNTHESIS_GRAPH.md` sections 26.4 and 26.F.
2. PASS: the arc explicitly performs the required route checks as a paired profile rather than as one isolated theorem.
   - `26.2.2` records the four pre-run vectors (`V1`–`V4`).
   - `26.2.1` and the paired dependencies (`24.R`, `24.R2`, `20.2`, `SYN_REVIEW_5`) record strict counter-omission and failure modes needed for staple/anti-circularity confirmation.
   - `13.2.3` and `26.2.x` together provide trace anti-circularity and non-portability constraints.
3. PASS: debt propagation is explicit at the closure boundary.
   - `26.4` and `26.F` carry forward open debt set (`DEBT:G22x34-PORTABILITY`, `DEBT:G19-G09-IDENTITY`, `DEBT:WG24-ALT-CARRIER`, `DEBT:WG24-BOUNDARY`) and close the route via explicit debt list refresh.
4. PASS with one qualification: the record does not always keep the method tuple `R=(CAN, extensions, contracts, debts)` as a single formal object in each section, but the same data is present across the linked rows.

## 2) Checkability assessment: can a stranger apply the method to a new pair?

1. FAIL: `ReplayC` is not defined in the method section and is not consistently defined in the adjacent history excerpts.
   - It appears in `PARADIGM.md` 13 pre-run protocol (`V4`) but without a compact semantic contract in the method block.
   - This blocks pure, stand-alone execution of the protocol.
2. FAIL: `Reject(E)`/`Debt(E)` are defined by reference to prior governed-proposal sections, not the method registration block itself.
   - A new operator must cross-read other sections (`10.1`/governed bridge rules) to run a complete pass.
3. FAIL: the method block is missing a minimal machine-readable schema for the “extension-residue map.”
   - The record has a `W/F` and `?W/?F` discipline and mentions dependent-row debt, but not a canonical field-by-field output format for new `E`.
4. PASS: leaky channels are testable in-practice in this repo by comparing `U_fixed`/omitted-field rows, staple matrices, and debt labels, but this relies on those prerequisite conventions.

## 3) Non-theorem labeling audit

1. PASS: no theorem-grade universality claim is made in the method block.
   - `PARADIGM.md` marks `SYN24-RBMR-REGISTER` as a registered method, **not** a theorem, with `RIGOR: HEURISTIC`.
2. PASS: the pair and closure claims are explicitly labeled as registered artifacts (`PROVED (after review)`) at the row level, while the method label itself remains non-theorem.

## Verdict

- METHOD-UPHELD-WITH-REPAIR

Repairs required for reusable standalone use:
1. Add one paragraph in the method block giving a concrete `ReplayC` contract and allowed output format for the `extension-residue map` per extension `E`.
2. Add a compact schema line for the registered route tuple `R` (CAN, extensions, contracts, debts) so new pairs can be evaluated without prior file navigation.
