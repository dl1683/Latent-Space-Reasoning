**L1 PASS. L2 PASS at the stated gate-work scope. V1 PASS. V2 PASS.** The claims in [METER_CANDIDATE.md](<LOCAL_PROJECTS>/new-computation-model/research/METER_CANDIDATE.md) survived independent attack under the finite, fixed-schedule, clean-AND contract.

First shell: `C:/Windows/System32/cmd.exe`, `login=false`. Read-only CPU review; no edits, agents, GPU use, or FULL execution. I read AGENTS, the full dual-loop skill, current model, reviewed contracts, and elementary Gray source.

| Claim | Verdict | Exact accepted scope |
|---|---|---|
| L1 | **PASS** | For \(n\ge3,r\ge0,\eta=1/8,\beta=2^{-r}\), exact feasibility floor \(h_2(\beta/8)\) |
| L2 | **PASS** | Every feasible \(K<\theta_r\): optimal elementary work \(\Theta(2^{n+r})\). At \(K\ge\theta_r\): \(\Theta(n)\) for fixed \(r\), with charged upper \(O(n2^r+n+r)\) |
| V1 | **PASS** | For \(n\ge4\), every fixed finite circuit and finite cap has an admitted finite dyadic counterprior |
| V2 | **PASS** | For \(n=3\), cap one is impossible for every finite \(r\); charged Gray cleanup admits symbolic cap \(\log_2 7\) |

**L1 and L2 lower bounds.** At a fixed RESET, the target-bit probability is a linear function of the *original* X prior. Its attainable values form an interval. For \(K<1\), that interval cannot cross between the two allowed entropy tails without containing a probability whose entropy exceeds K. Thus each gate has one globally rare value. Its orientation may change between gates, but cannot depend on the realized prior.

Any original \((X,R)\) label on that side can receive mass \(\eta\beta\), giving the L1 lower bound. Below \(\theta_r\), the rare probability is bounded by \(a<\beta\). If at least eight distinct X labels were marked, placing mass \(1/8\) on eight of them would force rare probability at least \(\beta\). Hence fewer than eight are marked.

Assigning \(1/8\) to **all** those marked X labels simultaneously is valid: the remaining labels have enough capacity to complete normalization. Consequently,

\[
N_{\rm rare}\eta\beta\le a,\qquad
b=\left\lfloor\frac{a}{\eta\beta}\right\rfloor<8.
\]

Each binary RESET collision uses a distinct rare predecessor class, containing at least one distinct original rare label. Earlier coalescence cannot invalidate this count. The necessary class reduction is \(2^r(2^n-2)\), so

\[
G\ge\left\lceil\frac{2^r(2^n-2)}b\right\rceil.
\]

For \(b=0\), cleanup is impossible. Physical recoding of R does not alter original-label counting.

**The incoming-singleton charged upper works.** I independently built the elementary macro and full cleanup circuits. After clearing the old dependent prefix suffix, the common predicate excludes the changed bit. Marking the incoming b singleton, moving it to a, resetting the flag, then applying `CNOT(common, changed_bit)` moves the joined class forward to b. The common predicate remains valid throughout and can be uncomputed.

Inductively, unvisited vertices remain original singletons. At every genuine merge the reset flag therefore marks exactly one original label in one context, giving probability at most \(\beta/8\). At the final AND-one vertex, the Y-zero incoming class is absent: the flag is identically zero, and the advance is reversible. Y-one input cleanup and every scratch bit also clear.

The modification adds constant gate overhead to each existing merge. Charged Gray suffix maintenance satisfies

\[
\sum_{\text{edges}}(n-j)<2^{n+1},
\]

so the upper is \(O(2^{n+r}+n+r)\), with width \(2n+3r+4\). No arbitrary predicate or retained erased distinction is free.

At \(K\ge\theta_r\), context-conditioned cleanup flags have probability at most \(\beta\), yielding \(L\le h_2(\beta)\) for \(r\ge1\); at \(r=0\), every bit entropy is at most one. This supplies the charged fast upper.

Acceptance covers **elementary gate work**. Program encoding, generation and source preparation remain separately charged. **Growing-r critical optimal work remains OPEN**: the entropy lower \(\Omega(n/\theta_r+n+r)\) does not match the available \(O(n2^r+n+r)\) upper.

**L is not joint conditional loss.** Immediately before RESET, write the complete state as \((B,T)\). Then

\[
J=H(B\mid T),\qquad L=H(B).
\]

The actual existing Gray source marks the accumulated class. Under uniform \(n=3,r=0\), its RESET flag counts are \(1,2,\ldots,7\): local entropy reaches one. Its final move has **joint loss zero but local cost \(h_2(7/8)>0\)**. The incoming orientation has counts \(1,1,1,1,1,1,0\). This decisively rejects transplanting the old compiler’s J bound directly into L.

**V1 telescoping is exact after previous merges.** Along any positive original input’s trajectory,

\[
\sum_g V_g
=\log_2\frac{\Pr(S_G=s_G)}{\Pr(S_0=s_0)}.
\]

Intermediate class masses cancel, including masses enlarged by earlier RESETs. Exact cleanup prevents classes from crossing final \((f,R)\) fibres.

For the submitted nine-label counterprior, all positive X labels have AND zero. Initial mass is \(\beta\epsilon\); final mass is \(\beta\). Thus the sum is exactly \(\log_2(1/\epsilon)\), independently of r. For G gates and finite K, choose finite integer \(m>\max(3,GK)\) and \(\epsilon=2^{-m}\). Some positive-predecessor branch must exceed K.

This uses a **family of finite descriptions with unbounded allowed preparation size**. Every selected source remains finite and its preparation must be charged. A fixed description/preparation limit or minimum-positive-atom promise changes the source class; V1 supplies no impossibility for that restricted class and no infinite-precision capability.

**V2 survives.** At \(n=3\), eight atoms capped at \(1/8\) force uniformity. For positive child masses u and v, cap one requires both

\[
(u+v)/u\le2,\qquad (u+v)/v\le2,
\]

hence \(u=v\). Starting from singleton leaves, all classes retain power-of-two leaf counts. Each final AND-zero/R fibre has seven leaves, so cleanup is impossible. Returned R cannot provide cross-slice merges because merged complete states cannot later return different original labels.

Singleton Gray absorption instead has branch ratios \(2,3,\ldots,7\); the final empty-vertex move has ratio one. This proves the finite \(\log_2 7\) capability with charged predicates and cleanup. **No optimal uniform V threshold is approved.**

The requested comparisons have exact certificates:

\[
\begin{aligned}
h_2(1/32)>3/16&\iff31^{31}<2^{154},\\
h_2(1/4)>3/4&\iff27<32,\\
h_2(1/8)<3/4&\iff7^7>2^{18}.
\end{aligned}
\]

All registered predictions pass: at \(K=3/4\), J permits linear work with one fair bit; L requires exponential work at r=1 and r=2, and permits linear work at r=3. At r=2, \(K=3/16\), L is infeasible while J is feasible with exponential work.

**Bounded reviewer evidence:** 25,488 standalone macro encodings across both orientations; 24 independently built cleanup circuits with 3,695,656 transitions; 1,590 RESET label checks; 45 final empty-vertex checks; 36 weighted traces with exact branch telescoping; and five actual-source circuits. Missing advance, dirty predicate and retained-erased-flag variants were rejected. These are **EMPIRICAL reviewer probes**, not registered FULL outcomes or concave-maximum certificates.

Occupied foundations include deterministic information-loss additivity ([Geiger–Kubin](https://arxiv.org/pdf/1303.6409)), singleton partition costs ([Meilă, equation 28](https://sites.stat.washington.edu/mmp/Papers/compare-jmva-revised.pdf)), adversarial collision priors ([Frank](https://arxiv.org/pdf/1806.10183)), protected reversible controls ([Demaine et al.](https://arxiv.org/pdf/1605.08448)), and statistical operating contexts ([compositional GRC](https://arxiv.org/html/2511.02425v2)). These occupy substantial ingredients; the inspected passages do not establish equivalence to this complete frontier or certify priority.

The useful result is a precise demonstration that meter choice changes feasibility and required work. L shifts the context threshold; V removes universal finite-cap cleanup for the unrestricted small-atom source class. Physical calibration, preparation/authentication costs, useful-world advantage and new-paradigm standing remain unresolved. **This review passes no mission-completion or terminal gate.**

Another process changed status/evidence files during review. I reread the final model; the meter candidate and elementary Gray source remained unchanged.