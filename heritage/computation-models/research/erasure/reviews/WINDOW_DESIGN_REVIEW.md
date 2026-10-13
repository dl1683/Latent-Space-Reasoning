**Product LOWER: PASS. Product charged UPPER: PASS. Correlated LOWER: PASS. Correlated charged UPPER: PASS. Total tradeoff: PASS. General-window reduction: PASS at the repaired scope.** No remaining mathematical repair or counterexample emerged.

For the [current candidate](<LOCAL_PROJECTS>/new-computation-model/research/WINDOW_DESIGN_CANDIDATE.md:7), acceptance requires \(n\ge r\ge8\), pointwise clean AND and returned R, fixed rational designs chosen before the workload, and universal zero-extended window admissibility.

| Claim | Verdict | Exact surviving conclusion |
|---|---|---|
| Product LOWER | **PASS** | \(G=\Omega(n+r+n2^r/r)\) |
| Product charged UPPER | **PASS** | One word attains matching work, linear width and \(E_q\le\lceil n/r\rceil(r+2)\) |
| Product total cost | **PASS** | Worst-source optimum \(\Theta(n+r)\), attained simultaneously with optimal work |
| Correlated LOWER | **PASS** | \(G\ge D/(2\beta)=\Omega(n2^r)\) |
| Correlated charged UPPER | **PASS** | \(G=O(n2^r+n+r)\), linear width and \(E_q\le n(r+2)\) |
| Correlated total tradeoff | **PASS** | \(G\le C_0n2^r\Rightarrow E_q=\Omega_{C_0}(nr)\) on uniform input |
| General-window reduction | **PASS** | The width-indexed inequalities below, for padding-preserved crossentropy costs |

Here \(\beta=2^{-r}\), \(\theta=h_2(\beta)\), and \(D=n-h_2(2^{-n})\). Generation, serialization, source preparation and rational descriptions remain separately charged; these are elementary-gate-work optimality statements.

**Product attack.** Uniform joint input belongs to both source classes. Entropy telescoping gives \(E_q\ge D\), while disjoint measured windows give
\[
E_q\le2\theta\lceil G/2\rceil\le\theta(G+1).
\]
Zero extension handles odd length and one-gate schedules. In the masked compiler, an incoming pattern fixes at most one original R label per original X; previous chunk coalescence cannot change the untouched chunk or returned R. Thus \(\mu_g\le\beta\). Fixed \(q_g=\beta\) gives \(c_g\le\theta\), establishing both upper bounds for the same word.

**Correlated attack.** The endpoint-inclusive certificate survives:
\[
h_2(3\beta)-2h_2(\beta)
\ge\beta\!\left[r-3\log_2 3-2\log_2 e\right]
>\beta(r-39/5)>0.
\]
Convexity therefore fixes one globally rare orientation per RESET. Three active original-R columns would admit a dyadic witness assigning \(\beta\) to each selected cell—even within one X row, since \(3\beta\le1/8\). Remaining row capacities fill all other columns. Hence at most two columns are active.

Pointwise return makes original R recoverable from every intermediate complete state, including after RESET. Inactive slices lose no entropy, giving \(J_g\le2\beta\). The context-linear upper uses flags confined to one R slice and satisfies the stronger \(c_g\le\theta\).

**Lambda repair: PASS.** The [repaired distinction](<LOCAL_PROJECTS>/new-computation-model/research/WINDOW_DESIGN_CANDIDATE.md:92) is necessary and sufficient:
\[
\lambda_g=\beta(a_{g,1}+a_{g,2}),\qquad
\mu_g\in\{\lambda_g,1-\lambda_g\}.
\]
Thus \(h_2(\mu_g)=h_2(\lambda_g)\ge(r-1)\lambda_g\), including \(\lambda_g=0\). Jensen gives
\[
h_2(\bar a)\ge\frac{D}{2\beta T}\ge\frac1{4C_0},
\]
so \(\bar a\ge\alpha_{C_0}\), even when some conditional probabilities exceed \(1/2\). Consequently
\[
E_q\ge(r-1)\alpha_{C_0}D=\Omega_{C_0}(nr).
\]
The matching upper requires a sufficiently large fixed allowance; neither existence for every prescribed \(C_0\) nor a lower bound for arbitrarily slow programs follows.

**Width and padding repair: PASS.** The [precise scope](<LOCAL_PROJECTS>/new-computation-model/research/WINDOW_DESIGN_CANDIDATE.md:117) is
\[
W_1(wK;b)\le W_w(K;b),\qquad
W_w(K;b+2)\le wW_1(wK;b).
\]
Padding preserves original probabilities and fixed designs. A CNOT on two additional initialized-zero scratch bits supplies a reachable-state identity. Padded length is \(w(G-1)+1\le wG\), including \(G=1\). This grants neither the same-width converse nor an arbitrary history-dependent-meter theorem. The unit-reset crossover is sufficient for ordinary RESET padding, without excluding other unmasked constructions below it.

Read AGENTS, the full dual-loop skill, required orientation files, candidate/review reports and implementing macros. **EMPIRICAL controls passed:** 2,604 macro state cases, six sampled complete circuits with 1,051,300 label–gate transitions, five exact entropy comparisons, 18 dyadic transportation witnesses and 108 boundary-padding checks. Universal acceptance rests on the analytic arguments.

First shell was the specified `cmd.exe`, `login=false`. No private memory, edits, agents, GPU or FULL execution. **No physical, priority or mission admission.**