# Question-loop pretest: a cold clock does not suffice for deterministic gap routing

Date: 2026-10-04. Rigor: HEURISTIC design rejection / candidate adversarial
trace, not an independently admitted theorem or all-wear-leveling lower bound.
No model change, construction repair, numerical run or new positive unit.

An alternative positive thought was to keep k data bits, one gap and k-1
counter bits inside N=2k. Two finite Johnson counters could provide a cold
phase clock without an external invocation ordinal, while a Start-Gap-style
rotation redistributes the stored values. This would aim to remove per-record
key labels and their logarithmic space penalty. Cheap clock recovery alone
does not establish the necessary deterministic all-history guarantee.

Candidate obstruction for the SPECIFIED simple schedule: one nearest-neighbor
gap relocation per FLIP, otherwise a direct physical owner-bit complement.
Before L consecutive requests, the gap trajectory touches at most L+1 physical
locations. If k+1>L+1, choose an occupied location outside that trajectory and
initialize its owner bit to one. Request that same owner bit L times. No scheduled
relocation touches its physical location, so it has INIT wear one plus L direct
changes, violating L. Logical ownership of untouched physical locations stays
fixed for this placement schedule. A paid cold phase clock does not alter those
changes. If a proposed scheme changes routing/relocation in response to this
history, it is outside this exact toy schedule and requires its own certificate.

This does NOT attack arbitrary nonlinear codes, response-dependent relabeling,
redundancy sharing, log records, or the active crit-bit family. It does not
establish that all deterministic wear leveling is impossible. A high-endurance
finite toy case in which the gap reaches all cells before exhaustion does not
rescue the primary family L around log-squared N, where L<k eventually.

Targeted live primary check: Devanny–Goodrich–Irani, *A competitive analysis for
the Start-Gap algorithm for online memory wear leveling*, published IPL2021.
Read model, contribution and Theorem1 sections (PDF pages1–3). Their placement
is one-to-one on word/block cells with wear-free registers. The deterministic
lower is for M=N-1. Their positive analysis uses initial randomization and a
request concentration restriction; the hostile-workload extension uses an
unlimited-endurance cache. Those premises neither transfer automatically nor
exclude arbitrary nonlinear encodings at N=2k. They explain why a recovered
clock or ordinary deterministic rotation alone is not the selected positive
mechanism. [Primary PDF](https://ics.uci.edu/~goodrich/pubs/J-89.pdf).

Disposition: retain as a falsification note, no CPU run needed to investigate
this explicit untouched-cell trace. The SINGLE active compact construction,
its ONE repair quota, fair adaptive-incumbent comparison and mission standing
remain unchanged. New clock/indexing advances would need fresh useful positive
hypotheses rather than a retroactive claim that this rejected schedule passed.
