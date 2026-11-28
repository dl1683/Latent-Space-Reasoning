# Continuous latent-worker admission has a lifetime error budget

28 November 2026. Classification: necessary mathematical foundation for auditability, self-maintenance and edge economics. Base `35775d42`. This note is a scoped statistical calculation, not a learned experiment, modification of the frozen gate, or permission to access independent held inputs.

## Decision

Separate an inventor's targeted challenge stream from a probabilistic admission stream. The [latent library proposal](COUNTEREXAMPLE_DRIVEN_LATENT_LIBRARY.md) already requires separate validation and independently witnessed outcomes. An unbounded swarm introduces an additional problem: continually proposing, repairing and retesting modules can eventually publish a bad module even if every individual admission test looks reasonable. A world that improves itself cannot let every worker reset the statistical error budget.

A targeted, independently verified counterexample can refute a scoped universal claim. A sequence of unsuccessful attempts to find one cannot generally establish a small rate of errors on later operands. If every challenge samples a convenient subset, a module can be wrong everywhere outside it and pass indefinitely. Repeated queries to a deterministic witness about the same operand are not new independent evidence. Exact proofs of stated conditions, when available, avoid the particular statistical sampling burden below, but cannot certify the fidelity of an ungrounded semantic abstraction.

This is an application of established adaptive-data-analysis and sequential-testing ideas, not a novel statistical theorem: [Dwork et al., 2015](https://arxiv.org/abs/1506.02629) study adaptive holdout reuse; [Howard et al., time-uniform Chernoff bounds](https://arxiv.org/abs/1808.03204) give the general nonnegative-supermartingale framework.

## The cost of always searching for a passing module

Consider a fixed candidate whose independently sampled audit operands have probability epsilon of a witnessed error, and require n consecutive clean independent draws. Its false-admission probability is (1-epsilon)^n. If M such bad candidates have independent audit blocks, the probability that at least one passes is exactly

    1 - (1-(1-epsilon)^n)^M.

For epsilon=0.1, n=20 and M=100 this is 0.999997653681698. This is a deliberately bad-candidate example, not an estimated LSR error rate. Even without independence across candidates the union bound gives a useful upper limit on risk if each candidate test is conditionally valid. To control the probability of any false admission at 0.05 across 1,000 preallocated tests of this kind, a sufficient per-candidate clean count is ceil(log(0.05/1000)/log(0.9)) = 94. A swarming design that requires thousands of such independent semantic witnesses may lose its edge economics before inference begins.

Independence is a substantive assumption: 94 correlated views of one verified source event are not 94 independent random operands. Nor is a single proof with 94 lines equivalent to 94 independent validation draws.

## A scoped anytime-valid alternative

For one immutable candidate/version, let a declared audit distribution D describe eligible operands, edit families and a fixed world/source/rule scope. Let X_t=1 when an independently assessed prediction is materially wrong (include false closure or invalid provenance if in scope) and 0 otherwise. Freeze the loss rule and candidate before the audit stream. Under the null that conditional error probability given all prior revealed history is at least epsilon, for each draw choose a predictable betting alternative a_t in [0,epsilon), and set

    E_0 = 1
    E_t = E_(t-1) * (a_t/epsilon)^X_t
                    * ((1-a_t)/(1-epsilon))^(1-X_t).

Use the corresponding factor directly when a_t=0, so no ambiguous 0^0 is needed. Conditional on past history and null error probability p >= epsilon, the factor has expectation

    p*a_t/epsilon + (1-p)*(1-a_t)/(1-epsilon) <= 1,

because it decreases with p and equals one at p=epsilon. Thus E is a nonnegative supermartingale under the null. Ville's inequality bounds the probability of ever crossing 1/alpha by alpha. For iid fresh D draws and a fixed candidate, the null includes candidates with population error at least epsilon. More general adaptive sampling requires the stated conditional null; arbitrary adversarially selected operands do not justify a population-error interpretation.

Allocate a predictable alpha_j to every new candidate, including repaired candidates, such that their sum never exceeds alpha. With a fixed null definition per candidate, fresh conditional evidence and a common accounting of attempts, the union bound controls the lifetime probability of admitting at least one null candidate by alpha, regardless of dependence between admission events. For example alpha_j=alpha/(j(j+1)) telescopes for an unbounded number of attempts. With epsilon=.1, alpha=.05 and a zero-error betting alternative, this requires 36, 74, 116 and 160 consecutive clean independent draws for attempts 1, 10, 100 and 1000 respectively. This is sufficient, not an optimal test or cost lower bound. Alpha is a procedure-level error probability, not a posterior probability that a particular admitted module is correct.

An exact checker can instead admit a module with a proof under explicit conditions. Use the probabilistic lease for empirical generalization claims only, not to certify an individual answer or substitute for source citations. A deterministic falsifier invalidates a universal claim regardless of its empirical lease.

## Why revision is a separate issue

Even a properly admitted candidate is certified only for its declared distribution, immutable operator, and interpretation scope. New evidence, changing operand distributions, rule edits and coordinate changes can invalidate the target. A precise mathematical transfer condition illustrates the limit: if the binary error function is genuinely unchanged and the total variation distance between old and new operand distributions is at most d, then new population error is at most old population error + d (capped at 1), by the definition of total variation. Without such a justified transfer bound or exact preservation witness, carry no statistical guarantee across the change. An unchanged source hash alone does not establish an unchanged error function. A change to the operator or target semantics may invalidate even this inequality's premise.

Do not equate many locally valid operations with a certified end-to-end reasoning episode: adaptive routing changes the distribution of operands used, and chains of operations have a different failure event. Audit consequential composed routes or supply an exact composition argument. Independent sample counts do not replace evidence of causal use at the same source pin.

## Reproducible arithmetic checks

The following standard-library check calculates the examples and exactly sums first-crossing probability for all Bernoulli histories up to 20 draws using an absorbing-state dynamic program, without Monte Carlo. It checks mass conservation and the bound after each time step. The null threshold in this check is .3 and the betting alternative is .05.

```python
from fractions import Fraction as F
from math import log, ceil
print(1-(1-.9**20)**100)
print(ceil(log(.05/1000)/log(.9)))
for j in [1,10,100,1000]:
    a=.05/(j*(j+1))
    print(j,a,ceil(log(a)/log(.9)))
for alpha in [F(1,5),F(1,20)]:
    for p in [F(3,10),F(1,2),F(9,10)]:
        states={0:F(1)}
        hit=F(0)
        eps=F(3,10)
        alt=F(1,20)
        for t in range(1,21):
            new={}
            for k,w in states.items():
                for x,prob in [(0,1-p),(1,p)]:
                    j=k+x
                    mass=w*prob
                    e=(alt/eps)**j*((1-alt)/(1-eps))**(t-j)
                    if e>=1/alpha:
                        hit+=mass
                    else:
                        new[j]=new.get(j,F(0))+mass
            states=new
            assert sum(states.values())+hit==1
            assert hit<=alpha
        print(alpha,p,float(hit))
```

Executed: all 120 mass and 120 bound checks passed. Crossing probabilities for p=.3/.5/.9 were 0.1509170787190806 / 0.016411781311035156 / 0.00000100000540004617 at alpha=.2; 0.0382174040708803 / 0.0010528564453125 / 0.0000000001000009 at alpha=.05. This verifies the arithmetic under independent fixed-p draws, not the general supermartingale argument or the validity of any real semantic audit distribution.

## Consequence for the persistent world

- Keep exploratory counterexamples, exact proof/grounding witnesses, and independent randomized audit results distinct in the operation's provenance. Candidate predictions should be committed before audit labels are revealed. Reusing a public development set is training unless a justified reusable-holdout protocol applies.
- Allocate evidence effort by consequential scope and likely reuse, not raw link count. Prefer compact exact preservation witnesses or genuinely independent scoped audits. An unqualified operation can remain a hypothesis guiding discovery; it should not silently publish factual conclusions.
- Compare against a warm symbolic or exact system with identical evidence rights and charge the cost of proposal, failed admission, random audits, proof checking, requalification and task-time composition. If full lifetime costs erase savings, stop the economic claim. There is no evidence yet that a learned maintenance swarm can earn this cost.
- This is a future design recommendation, not a new mandatory gate, code change, or reason to defer the current four-case ordinary-reader qualification. No private material, model, grader, or authorization was touched. The actual first experiment remains the [registered sequence](LATENT_WORLD_SYSTEM_DESIGN.md#7-research-sequence-and-decision-gates).
