**NO-BINDING at the registered WINDOW-DESIGN-1 scope.** The bounded pass produces an explicit RESET reference construction and symbolic work/error bounds. It does not bind the averaging response, reversible controls, source provision and controller cleanup into a certified circuit. No single bounded repair closes those independent gaps; the registered **PIVOT condition applies to this operational attempt**.

All new derivations below are **CONJECTURE / derived candidates pending fresh independent proof review**, not newly admitted PROVED results. The registered \(r,n,b,\rho\), reference controls and correctness target remain unchanged.

**Response first.** During a constant-gap hold, the primary model specifies
\[
p(t)=g_E+(p^--g_E)e^{-\lambda t},
\qquad g_E=\frac1{1+2^E}.
\]
Writing energy in units \(k_BT_{\rm bath}\ln2\), its expected heat-release rate into the bath is
\[
j_E(t)=\lambda E(p^--g_E)e^{-\lambda t}.
\]
Choosing reference tick duration \(\Delta\) gives the registered half-swap when \(\lambda\Delta=\ln2\). This specifies occupation relaxation and heat exchange. It supplies **no calibrated receiver or averaging kernel** that turns heat into the zero-extended rectangular two-gate window. Heat can also flow into the memory when \(p^-<g_E\). The relaxation constant cannot certify a window cap or residual overhead. These are inferences from the [primary detailed-balance model](https://arxiv.org/html/2106.00580).

**Concrete RESET construction.** Fix
\[
s\in\{r-1,r,r+2\},\quad h\in\{8,32,128\},\quad
A=2^{-h},\quad L=41-s.
\]
Start with a degenerate memory and occupation \(p\). With the bath disconnected, raise its excited level to \(s\); hold for \(h\) ticks; then raise by one and hold at every level through 40. Finally disconnect the bath and return the gap to zero.

Holding at the initial level makes the schedule explicit. Finite diagonal ramps preserve occupation in the ideal reference model; switching, isolation and ramp-control costs still require certificates. This construction extends the [partial-swap recurrence in Appendix B](https://arxiv.org/html/1311.7612#A2), rather than importing its balanced-input work guarantee.

The exact rational updates are
\[
u_s=Ap+(1-A)g_s,\qquad
u_E=A u_{E-1}+(1-A)g_E,
\]
hence
\[
u_{40}=A^Lp+(1-A)\sum_{E=s}^{40}A^{40-E}g_E.
\]
They apply separately to the registered rare interval
\(p\in[7\beta/8,\beta]\), \(\beta=2^{-r}\), and to pure inputs \(p=0,1\).

For example, assigning one reference tick to each isolated ramp gives duration
\[
d_{\rm reset}=[Lh+L+1]\Delta.
\]
This is finite reference bookkeeping, **not certified physical timing**.

The dimensionless expected cyclic work is
\[
w_{s,h}(p)
=sp+\sum_{E=s}^{39}u_E-40u_{40}.
\]
The initial quench \(sp\) and final work recovery \(-40u_{40}\) are included. The expected bath heat is
\[
q_{\rm bath}
=\sum_{E=s}^{40}E(u_{E-1}-u_E)=w_{s,h}(p),
\]
where \(u_{s-1}=p\). Equality follows because the memory Hamiltonian starts and ends at zero. It does not identify instantaneous heat with work spikes or guarantee a realization-wise heat cap.

**Useful positive result: uniform RESET error.** For every input occupation \(p\in[0,1]\),
\[
u_{40}
\le A^L+\frac{1-A}{1-2A}\,2^{-40}
\le2^{-208}+\frac{255}{254}2^{-40}
<2^{-39}.
\]
Here \(L\ge27\) and \(A\le2^{-8}\), throughout the frozen reference choices. Thus the reference channel has a uniform error bound, including pure inputs; rare-source assumptions are unnecessary for this particular bound.

Assuming fresh memoryless bath interactions and exact reversible gates, a hybrid coupling gives
\[
\Pr(\text{incorrect or dirty circuit output})
\le M\,2^{-39},
\]
where \(M\) counts RESETs. Static inspection gives, at the frozen \(n=r,2r\),
\[
M_{\rm masked}=\frac nr(2^r-1),\qquad
M_{\rm context}=n2^r<2^{17}.
\]
Consequently their **RESET-only** contribution is below \(2^{-22}\), leaving part of the common \(2^{-20}\) budget for other errors. This does not admit either full operational circuit. For a much longer Gray construction, an insufficient union bound would mean **UNRESOLVED**, not exclusion.

Finite-temperature reset remains noisy: the recurrence gives \(u_{40}>0\). Its output error cannot be recorded as merely a cost-approximation tolerance.

**Concrete cost mismatch.** Integer-gap equilibrium
\[
g_s=\frac1{1+2^s}
\]
is not any registered dyadic design \(a/2^b\): its reduced denominator is odd. Exact implementation of a design \(q\) would instead require gap
\[
E(q)=\log_2\frac{1-q}{q},
\]
with a separately certified control precision.

Even unlimited holding time leaves discrete-step work
\[
w_\infty(p)=sp+\sum_{E=s}^{39}g_E-40g_{40},
\]
rather than scalar crossentropy. In particular,
\[
w_\infty(p)-C(p,g_s)
=
\underbrace{\sum_{E=s}^{39}g_E-\int_s^{40}g(E)\,dE}_{\text{discrete-step excess}}
-\log_2(1+2^{-40})-40g_{40}.
\]
The bracket lies between zero and \(g_s-g_{40}\). Longer holds reduce relaxation error; they do not remove this discretization term.

A symbolic finite-hold bound is
\[
|w_{s,h}(p)-w_\infty(p)|
\le
\frac{A}{1-A}\bigl(|p-g_s|+g_s-g_{40}\bigr)
+40\left(A^L|p-g_s|+\frac{A}{1-2A}2^{-40}\right).
\]
Any comparison to registered \(C(p,q)\) must additionally charge design mismatch, discrete-step excess and terminal corrections. A uniform per-RESET cost discrepancy \(\epsilon\) contributes \(M\epsilon\); it is distinct from \(M\delta_{\rm reset}\).

The [2021 paper’s Eq. 5](https://arxiv.org/html/2106.00580) assumes an initially thermalized, balanced bit. Our rare flag at a degenerate Hamiltonian does not satisfy that premise. The construction above retains its actual occupation and quench work instead of substituting the balanced theorem.

**The remaining resource ledger is substantive.**

| Required binding | What this pass supplies—and what remains missing |
|---|---|
| Averaging response | A reference heat-release profile; no certified averaging kernel, boundary response or window implication. |
| Reversible gates and padding | Logical words; no finite control/duration/error certificate establishing an attainable overhead envelope. Neither registered \(\rho\) value is calibration. |
| Source and clean ancillas | Explicit source promises; no charged preparation/authentication certificate covering both product and correlated classes. |
| Bath provision | Reference thermal channels; no finite reservoir capacity/refresh certificate. A collision realization must account for fresh Gibbs ancillas and any swap randomness. |
| Controller cleanup | A finite schedule; no cyclic controller, clock, switching and record-cleanup certificate. Fixed scheduling need not create measurement records, but its implementation still needs accounting. |

These costs enter alongside RESET work, rather than being replaced by an independently chosen \(\rho G\).

Noise also changes the theorem interface. A hybrid error bound can establish probabilistic correctness without asserting that conditioning on successful previous resets preserves the original source law. After faults, the ideal rare-flag interval need not hold. Actual occupation changes must therefore enter work/window bounds separately from \(M\epsilon\). The reviewed exact-circuit lower bounds do not automatically become lower bounds for these noisy channels.

**Repair disposition.** The concrete RESET-local correction is to replace the scalar-cost identification with the cyclic channel and work/heat ledger above. It is useful, but it does not resolve the averaging, reversible-control, source or controller certificates. Calling their simultaneous invention “one repair” would conceal several new obligations. Therefore I do **not** return REPAIR-NEEDED for the complete binding.

Under the [registered decision rule](<LOCAL_PROJECTS>/new-computation-model/research/WINDOW_DESIGN_EXPERIMENT.md:97), further RESET sensitivity tables would not establish CONTINUE. This is a scoped failure to supply the operational binding, **not global physical impossibility or DEMOTE of every admissible construction**.

**HEURISTIC narrative:** finite noisy RESETs are constructible in the reference dynamics; the claimed circuit advantage remains unbound to an attainable operational contract. This is a component result, with no novelty or mission-completion admission.

Read-only throughout. No agents, GPU, registered grid, FULL execution or numerical theory probes were used.