# Root design questions before any synthesis implementation

2026-10-04 HEURISTIC / design obligations, not a theorem or chosen strategy.
Actual source/design pre-test and genuine architect round1 still run.

1. Exhausted-write semantics must be explicit. A strict forbidden next change,
   a saturating no-op, and an observable hardware error are different machines.
   Our proposed stationary endpoint-policy relaxation covers algorithms only
   when all writes on guaranteed histories succeed with ordinary bit semantics,
   without wear-dependent feedback. If probing failed writes can reveal latent
   wear, equal visible bit states can yield different endpoint transitions.
   UNSAT for stationary T_i(s) then does not cover all allowed programs.
   Chee etal Definition1's programming operation includes saturation; translate
   it rather than assuming it matches a strict-budget machine.
   [Primary definition](https://arxiv.org/pdf/2109.09932).

2. From known unworn zeros and strict changed-bit accounting, the hidden wear
   vector's parity equals the visible cell vector. A finite verifier can use
   reachable wear vectors; the algorithm cannot read them. Initialization ones
   already consume wear. Endpoints omit intermediate changes, so their wear is
   a relaxation until a compiled bit procedure is checked.

3. All decoder partitions and all stationary transition maps must be admitted.
   Requiring exactly one changed bit, a parity decoder, a fixed initializer or
   an unproved symmetry reduction can turn a claimed universal finite UNSAT
   into a restricted-family failure. Unreachable states may be assigned arbitrary
   extensions only after proving those assignments add no obligations on valid
   guaranteed histories.

4. Finite SAT witnesses must pay their table/program generation and command
   interpretation cost. Arbitrary parameter-specific finite tables supply
   calibration; they are not efficient uniform scalable constructions. Tiny
   tuples outside the target's L range do not falsify that asymptotic target.

5. Boundary-reset persistence and crash atomicity inside a command differ.
   Preserve the initial contract during first source/design translation; a new
   crash guarantee needs its own registered implementation/metadata/wear costs.
   It cannot be silently added or attributed to completed-command semantics.

These questions are inputs to the actual dialogue and independent design gate.
No code, solver invocation, result promotion or additional repair family here.
