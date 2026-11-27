# Retain operation equivalences, not necessarily coordinates

27 November 2026. Base 83e168af. Classification: mathematical foundation and a public learned-system-identification component experiment. Independent of the frozen ordinary reader and held operation panels. No pretrained model, grader or held input used.

## Decision and failure that motivated it

The previous [transition repair result](PREDICTIVE_TRANSITION_REPAIR.md) established that a common predictive basis exists for strict UTF-8 versus surrogatepass and that a single rank-2 transition change suffices in that fitted basis. That did not establish a cheap way to discover it.

I attempted to localize the edit prospectively from a public development pool: intersect the byte sets of every word for which the old model and revised CPython oracle disagree. Under a same-coordinate single-transition-change hypothesis (unchanged boundaries and all other transitions), every changed word must contain that transition symbol. This is a necessary condition, not sufficient evidence of locality. The intersection singled out byte 237 in both directions (62 changed development words). But a bounded anchor family built from old QR pivots and failing prefixes/suffixes had sampled revised ranks 10 and 11, respectively, instead of 12. Two development-only attempts to find a new cross pivot (1,200 random pairs from the larger old/failure pool; then 3,876 short observed split compositions containing the suspected byte) did not restore rank. The respective final attempts cost 46,041 and 48,910 new oracle reads and still refused. Dense edits strict-to-replace and surrogatepass-to-ignore had empty intersections. These failed exploratory anchor policies are not offered as a qualified method; the previous post-hoc rank-12 witness remains valid. A rank-deficient sampled probe family is not proof that the true state dimension has fallen.

Instead, retain **structural equivalences among operations**, which are not tied to a particular invertible coordinate basis. A word computation has the form u A[w1] ... A[wn] v. If A[a]=A[b], replacing a by b anywhere leaves its predicted output invariant. Equality survives a shared invertible coordinate transformation. If revised operations remain equal within the old classes, the old alphabet quotient remains valid even if the revised model has different state dimension and coordinates. If a class splits, the quotient can fail even on short words. This is ordinary matrix algebra, not a new theorem, and it is about modeled operations, not an unconditional assertion about every future source or interpretation.

Treat these equivalences as explicit, revocable hypotheses with witness histories in WorldPin, distinct from the transition parameters themselves. Preserve a useful quotient across revision if fresh checks support it; if a prior class collapses consequentially different meanings, split it or refuse and rebuild. The worker should eventually discover these equivalences for semantic role-bound operations without being supplied the classes. A coordinate shift should not by itself erase learned computational structure.

## Executed public experiment

Independent executable oracle: CPython UTF-8 codec strict, surrogatepass, replace and ignore, with scalar output decoded character count + 1 or zero on UnicodeDecodeError. Same public domain-aware generated development and test pools as the previous pilot; deduplicated 2,903 development words. The opening 12-dimensional models are learned via the previous spectral Hankel procedure, no codec byte classes or hidden coordinates supplied.

Infer old operation classes by equality of learned transition matrices rounded at 7 decimals (normalize negative zero). Split off any byte in the intersection of failing development words. Compare original and canonicalized word oracle values across the entire development pool. If any differs, refuse rather than silently apply the old quotient. Otherwise relearn a predictive model over the representative alphabet, starting from empty prefix/suffix, then expand its transitions and recheck the original words. This is a refit in new coordinates **using an old learned structural abstraction**, not a successful sparse patch of old weights.

Controls: the previous full 256-byte cold fit; and a cold candidate quotient inferred using 64 fixed-seed insertion contexts from the same public development words, querying all 256 symbols per context, followed by the same quotient validation/refit. The second is one simple cold inference policy, not a claim against best active automata learning. All unique revised-oracle calls through localization, mapping validation and fitting are charged; the test readout is scored separately. Old model fitting is already paid on the warm path (66,131 or 73,588 opening reads). No claim on wall time, energy, memory or total lifecycle crossover. The exact codec remains the practical incumbent.

| Revision | Warm classes | Warm new reads | Warm result, errors/20,695 | Full cold reads | Cold-signature new reads / outcome |
|---|---:|---:|---|---:|---|
| strict to surrogatepass | 12 | 17,717 | 0 | 73,588 | 21,505 / refused (875 mapping failures) |
| surrogatepass to strict | 12 | 18,851 | 0 | 66,131 | 21,519 / refused (829) |
| strict to replace | 12 | 7,254 | 0 | 27,085 | 21,546 / refused (644) |
| surrogatepass to ignore | 11 | 5,529 | refused (98 mapping failures) | 31,458 | 21,536 / refused (1,441) |

All full cold models had zero disagreements on the same 20,695 unique public test words. Each of the three successful warm refits also scored 0/20,686 test words that never appeared in that arm's acquisition queries. Revised fitted dimensions were 12, 12 and 7. Thus the third successful refit preserved useful operation classes even when the state rank changed and the single-edit hypothesis was not localized. Revised reads fell approximately 76%, 71%, 73% versus the full cold fits, respectively. The fourth warm arm made no test predictions; refusal is not a correct answer or a win. The cold-signature arm found 6, 7, 9 and 8 classes respectively and all were inadequate on development. If falling back to the full cold algorithm after a refusal, charge the failed attempt too (at most the sum of unique counts, with possible overlap).

The inputs are public exploratory probes from a reused generator and development seed, not a blind independent semantic task panel; previous work already examined some of the evaluation cases. This result is a scoped observation about exact codec functions and learned linear realizations, not a natural-language result or population-level accuracy estimate. Class agreement on finite probes is not proof of global equivalence. Numerical rounding can merge close but distinct operators. The test generator is domain aware. The stronger baseline is active equivalence-class discovery and exact incremental algorithms. No production or integration tests run; direct checkout DNS unavailable. The frozen ordinary competence gate is unchanged.

## Architecture consequence

The next interface should separate (a) learned operation equivalence classes, (b) their current coordinate realizations, (c) uncertainty and counterexamples about the classes, and (d) provenance/invalidation. This suggests hierarchical maintenance: change representation or refit a local dynamic module while reusing independently witnessed semantic equivalences. The learned worker's job is to propose the equivalence and find the best counterexample, not to receive a canonicalization dictionary. In the real corpus, an example is learning that several suppliers' acknowledgment forms represent the same contractual event, then discovering that one supplier's variant is no longer equivalent after a new rule. Source evidence and the raw alternatives must remain recoverable; never discard them merely because the present module treats them as equivalent.

Next test: an adaptive cold and warm class-splitting learner that chooses a context to distinguish members of a class, versus a generic active automata learner and the exact incumbent. Use fresh opaque executable systems with edit families that split, merge and leave classes invariant; compare lifecycle cost with fallback included. Only then take the primitive to corpus-derived role binding and query-blind maintenance with held late tasks. Do not substitute this codec result for the registered 0.6B ordinary-reader qualification.

## Reproduction of the final quotient experiment

Save the following as a Python file next to experiments/memory/predictive_transition_revision_pilot.py and run with OPENBLAS_NUM_THREADS=1. The executed local base was the oracle, pool and non-residual fit from that adjacent script (same seed, same observed full-cold counts). NumPy 2.3.5, SciPy 1.17.0. The three unsuccessful exploratory anchor approaches above are reported as negative diagnostics, not part of this reproduction.

```python
from functools import lru_cache
import json, random
import numpy as np
from scipy.linalg import qr
from predictive_transition_revision_pilot import oracle, pool, fit


def reduced_fit(mode, dev, old=None):
    seen=set()
    @lru_cache(None)
    def ask(w):
        seen.add(w)
        return oracle(w,mode)
    classes={}; mapping=[]
    if old is not None:
        bad=[w for w in dev if round(old['pred'](w))!=ask(w)]
        candidates=sorted(set.intersection(*(set(w) for w in bad))) if bad else []
        for a,T in enumerate(old['T']):
            R=np.round(T,7);R[R==0]=0
            key=(R.tobytes(),a if a in candidates else -1)
            mapping.append(classes.setdefault(key,a))
        info=dict(candidates=candidates,old_classes=len({tuple(np.round(T,7).ravel()) for T in old['T']}))
    else:
        # Cold comparator: infer candidate operation classes from 64 fresh
        # scalar insertion contexts. No old model or old labels.
        rng=random.Random(421)
        contexts=[]
        for w in rng.sample([w for w in dev if w],64):
            i=rng.randrange(len(w));contexts.append((w[:i],w[i+1:]))
        for a in range(256):
            key=tuple(ask(p+bytes([a])+s) for p,s in contexts)
            mapping.append(classes.setdefault(key,a))
        info=dict(contexts=len(contexts))
    reps=sorted(set(mapping)); trans=bytes(mapping)
    mapped=[w.translate(trans) for w in dev]
    mismatches=sum(ask(w)!=ask(v) for w,v in zip(dev,mapped))
    info.update(classes=len(reps),mapping_dev_errors=mismatches)
    def stop(reason):return None,dict(info,reason=reason,new_reads=len(seen)),seen
    if mismatches:return stop('operation quotient rejected')
    P,E=[b''],[b''];rounds=[]
    for _ in range(41):
        H=np.array([[ask(p+e) for e in E] for p in P],float)
        d=np.linalg.matrix_rank(H,tol=1e-8)
        ix=qr(H.T,pivoting=True,mode='economic')[2][:d];S=[P[i] for i in ix]
        jx=qr(H[ix,:],pivoting=True,mode='economic')[2][:d];F=[E[j] for j in jx]
        M=H[ix,:][:,jx]
        Hx=np.array([[[ask(p+bytes([a])+e) for e in F] for p in S] for a in reps],float)
        T=np.linalg.solve(np.broadcast_to(M,Hx.shape),Hx)
        lookup={a:T[i] for i,a in enumerate(reps)}
        full=np.stack([lookup[a] for a in mapping])
        init=np.array([ask(e) for e in F],float)
        out=np.linalg.solve(M,np.array([ask(p) for p in S],float))
        def pred(w):
            z=init.copy()
            for a in w:z=z@full[a]
            return float(z@out)
        badword=next((w for w in mapped if abs(pred(w)-ask(w))>1e-5),None)
        rounds.append(int(d))
        if badword is None:
            if any(abs(pred(w)-ask(w))>1e-5 for w in dev):return stop('original alphabet replay failed')
            return pred,dict(info,reason='provisional reduced-alphabet refit',rank=int(d),rounds=rounds,new_reads=len(seen)),seen
        pnew=list(dict.fromkeys(P+[badword[:i] for i in range(len(badword)+1)]));enew=list(dict.fromkeys(E+[badword[i:] for i in range(len(badword)+1)]))
        if pnew==P and enew==E:return stop('stalled')
        P,E=pnew,enew
    return stop('round cap')


def main():
    dev=list(dict.fromkeys(pool(1742,3000)));ds=set(dev)
    test=list(dict.fromkeys(w for w in pool(108282,22000) if w not in ds))
    models={m:fit(m,dev) for m in ('strict','surrogatepass')}
    for opening,revised in [('strict','surrogatepass'),('surrogatepass','strict'),('strict','replace'),('surrogatepass','ignore')]:
        cold=models.get(revised) or fit(revised,dev)
        for arm,old in [('warm_quotient',models[opening]),('cold_signature',None)]:
            pred,info,seen=reduced_fit(revised,dev,old)
            info.update(arm=arm,opening=opening,revised=revised,opening_reads=models[opening]['queries'],cold_reads=cold['queries'],cold_errors=sum(abs(cold['pred'](w)-oracle(w,revised))>1e-5 for w in test),n_test=len(test))
            if pred is not None:
                err=[w for w in test if abs(pred(w)-oracle(w,revised))>1e-5]
                info.update(test_errors=len(err),unqueried_n=sum(w not in seen for w in test),unqueried_errors=sum(w not in seen for w in err))
            print(json.dumps(info),flush=True)
if __name__=='__main__':main()
```
