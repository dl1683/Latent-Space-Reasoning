# Audit correction: prefix parity update accounting

**Found after all `r23_PREREG3`/`PREREG4` runs.** Those protocols
say a logical FLIP costs `wt(column_i(U))` physical **writes**. A
generic read/write bit-probe controller cannot flip an unknown
physical bit by a blind WRITE; in the explicit implementation
without cached copies it must read the old value before writing
its complement. Thus the results are exact for a *native
atomic bit-TOGGLE probe* at unit cost (plus free local XOR on already
read query values), or for a write-only accounting that omits
update reads. They are **not** read+write cell-probe totals of
2 writes per update.

Under ordinary charged READ then WRITE at each affected location,
the explicit linear algorithms take at most `2*wt(column)` total
update probes. The n=6 witness is therefore an upper bound of
4 update probes and 2 query reads; the n=7 no-(2,2) matrix
factorization only excludes schemes with **at most two affected
cells per update**, not arbitrary adaptive bit-probe structures
with update total <=4. A conventional equal-access controller with
the same native TOGGLE primitive matches the source exactly; one
without TOGGLE and without cached old values needs the explicit
reads. This correction does not
create an advantage, only tightens the cost contract.
With endpoint-only observation and no other side effects, each
native TOGGLE has an ordinary READ/WRITE lowering of two charged
probes, so even the missing primitive has only constant overhead
in this restricted bit-probe task. A hostile mid-update observer
would change H3 and is treated separately in `r23_atomic.py`.

No frozen preregistration, script, receipt or earlier negative
result has been rewritten. Update any summary calling it generic
"bit-probe" to "linear native-toggle probe" or explicitly qualify
the update metric. The finite algebra is unaffected; the
operational meaning of the write count is narrower. The matrix
description and initial encoding are also free in this toy probe
measure; a uniform practical controller would have to charge them,
as the R23 report does not claim to provide.

**EMPIRICAL READ/WRITE replay (`r23_PREREG27.md`,
`r23_rw_cost.json`).** At n=4,5,6 the known (2,2) encodings cost at
most 4 physical probes per update (2 READ + 2 WRITE) and 2 query
READ probes. Thus the max-update-plus-max-query score is 6 at
each size, versus 6,7,8 for direct storage respectively; every
state/query/flip was replayed. The n=5 entire linear-encoding
census also has minimum re-priced score 6. Such a small gain
against *direct storage* is an ordinary data-structure choice,
not a new model or an end-to-end benefit; control/layout/setup
remain unpriced, and the unrestricted ordinary target can
select the same encoding.
