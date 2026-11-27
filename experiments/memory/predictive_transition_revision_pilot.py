import random, json, time, sys
from functools import lru_cache
import numpy as np
from scipy.linalg import qr
from sympy import Matrix, ilcm

A=[bytes([i]) for i in range(256)]
def oracle(w, mode):
    try: return len(w.decode('utf-8', errors=mode))+1
    except UnicodeDecodeError: return 0

def pool(seed,n):
    rng=random.Random(seed)
    for i in range(n):
        if i%4==0:
            yield rng.randbytes(rng.randrange(1,24))
            continue
        # Mix valid and near-valid inputs. This is domain-aware sampling, not a supplied state or transition.
        chars=[chr(rng.randrange(*rng.choice([(0,128),(128,2048),(2048,65536),(65536,0x110000)]))) for _ in range(rng.randrange(1,14))]
        w=''.join(chars).encode('utf-8','surrogatepass')
        if i%4==2: w=w[:rng.randrange(len(w)+1)]
        if i%4==3:
            j=rng.randrange(len(w)); w=w[:j]+rng.randbytes(1)+w[j+1:]
        yield w

def fit(mode, dev, warm=None, base=None):
    def target(w):
        v=oracle(w,mode)
        if base is None:return v
        b=base["pred"](w); rb=round(b)
        if abs(b-rb)>1e-5:raise RuntimeError("noninteger base")
        return v-rb
    ask=lru_cache(None)(target)
    P,E=([b''],[b'']) if warm is None else (list(warm[0]),list(warm[1]))
    rounds=[]; start=time.perf_counter()
    while True:
        H=np.array([[ask(p+e) for e in E] for p in P],dtype=float)
        rank=np.linalg.matrix_rank(H,tol=1e-8)
        if rank==0:
            T=np.zeros((256,0,0));init=out=np.zeros(0);M=np.eye(1)
        else:
            _,_,ix=qr(H.T,pivoting=True,mode='economic'); S=[P[i] for i in ix[:rank]]
            Hs=H[ix[:rank],:]; _,_,jx=qr(Hs,pivoting=True,mode='economic'); F=[E[j] for j in jx[:rank]]
            M=Hs[:,jx[:rank]]
            Hx=np.array([[[ask(p+a+e) for e in F] for p in S] for a in A],dtype=float)
            T=np.linalg.solve(np.broadcast_to(M,Hx.shape),Hx)
            init=np.array([ask(e) for e in F],dtype=float)
            out=np.linalg.solve(M,np.array([ask(p) for p in S],dtype=float))
        def pred(w):
            z=init.copy()
            for a in w:z=z@T[a]
            return float(z@out)
        bad=None
        # Recheck the entire fixed development pool every round; cache counts unique oracle calls.
        for w in dev:
            if abs(pred(w)-ask(w))>1e-5:bad=w;break
        rounds.append([int(rank),ask.cache_info().misses,None if bad is None else bad.hex()])
        if bad is None:break
        pnew=list(dict.fromkeys(P+[bad[:i] for i in range(len(bad)+1)]))
        enew=list(dict.fromkeys(E+[bad[i:] for i in range(len(bad)+1)]))
        if pnew==P and enew==E:raise RuntimeError('stalled')
        P,E=pnew,enew
        if len(rounds)>40:raise RuntimeError('round cap')
    return dict(pred=pred,basis=(P,E),rank=int(rank),queries=ask.cache_info().misses,cache_hits=ask.cache_info().hits,rounds=rounds,secs=time.perf_counter()-start,T=T,init=init,out=out,cond=float(np.linalg.cond(M)))

def score(model,mode,test):
    errors=[]; maxerr=0.
    for w in test:
        e=abs(model['pred'](w)-oracle(w,mode));maxerr=max(maxerr,e)
        if e>1e-5:errors.append(w.hex())
    return dict(n=len(test),wrong=len(errors),maxerr=maxerr,examples=errors[:3])

def modrank(H, prime=1009):
    # Nonzero minor mod a prime implies nonzero integer minor.
    h=[[int(x)%prime for x in row] for row in H]; r=0
    for j in range(len(h[0])):
        pivot=next((i for i in range(r,len(h)) if h[i][j]),None)
        if pivot is None:continue
        h[r],h[pivot]=h[pivot],h[r];v=pow(h[r][j],-1,prime)
        h[r]=[(x*v)%prime for x in h[r]]
        for i in range(r+1,len(h)):
            v=h[i][j]
            if v:h[i]=[(a-v*b)%prime for a,b in zip(h[i],h[r])]
        r+=1
        if r==len(h):break
    return r

def coordinate_audit(models,test):
    P=list(dict.fromkeys(p for m in models for p in m['basis'][0]));E=list(dict.fromkeys(e for m in models for e in m['basis'][1]))
    old=lru_cache(None)(lambda w:oracle(w,'strict'));new=lru_cache(None)(lambda w:oracle(w,'surrogatepass'))
    X=np.array([[old(p+e) for e in E] for p in P]);Y=np.array([[new(p+e) for e in E] for p in P])
    ranks=[modrank(H) for H in [X,Y,Y-X]]
    print(json.dumps(dict(exact_minor_ranks=ranks,shape=X.shape,prime=1009)))
    # Greedy common feature selection. No codec state/class names provided.
    cols=[]
    for i in range(len(E)):
        trial=cols+[i]
        if np.linalg.matrix_rank(X[:,trial])==np.linalg.matrix_rank(Y[:,trial])==len(trial):cols=trial
    if len(cols)!=ranks[0] or len(cols)!=ranks[1]:
        print(json.dumps(dict(common_columns=len(cols),aligned='failed')));return
    F=[E[j] for j in cols];Ts=[];initial=[];outputs=[]; exact=[]; exact_out=[]; exact_in=[]
    for H,ask in [(X,old),(Y,new)]:
        _,_,ix=qr(H[:,cols].T.astype(float),pivoting=True,mode='economic');S=[P[i] for i in ix[:len(cols)]]
        M=H[ix[:len(cols)],:][:,cols]
        Hx=np.array([[[ask(p+a+e) for e in F] for p in S] for a in A],dtype=int)
        mf=M.astype(float);Ts.append(np.linalg.solve(np.broadcast_to(mf,Hx.shape),Hx))
        initial.append(np.array([ask(e) for e in F],dtype=float));outputs.append(np.linalg.solve(mf,np.array([ask(p) for p in S],dtype=float)))
        # Rational check: verify the sampled rank upper bound and exact transition differences.
        inv=Matrix(M.tolist()).inv(); den=int(ilcm(*[v.q for v in inv])); adj=np.array((inv*den).tolist(),dtype=object)
        assert np.array_equal(H[:,cols].astype(object)@adj@H[ix[:len(cols)],:].astype(object),den*H.astype(object))
        exact.append((den,adj@Hx.astype(object)));exact_out.append((den,adj@np.array([ask(p) for p in S],dtype=object)));exact_in.append([ask(e) for e in F])
    changed=[i for i in range(256) if np.max(abs(Ts[1][i]-Ts[0][i]))>1e-8]
    d0,t0=exact[0];d1,t1=exact[1];diff=d0*t1-d1*t0
    exact_changed=[i for i in range(256) if np.any(diff[i]!=0)]
    exact_ranks=[Matrix(diff[i].tolist()).rank() for i in exact_changed]
    assert exact_changed==changed and exact_in[0]==exact_in[1]
    assert np.array_equal(exact_out[0][1]*exact_out[1][0],exact_out[1][1]*exact_out[0][0])
    print(json.dumps(dict(exact_rational_changed=exact_changed,exact_ranks=[int(r) for r in exact_ranks],exact_sampled_ranks_old_new=[len(cols),len(cols)],same_readout=True)))
    patched=Ts[0].copy();patched[changed]=Ts[1][changed]
    def pred(w):
        z=initial[1].copy()
        for a in w:z=z@patched[a]
        return float(z@outputs[1])
    print(json.dumps(dict(common_columns=len(cols),changed_byte_operators=changed,changed_ranks=[int(np.linalg.matrix_rank(Ts[1][i]-Ts[0][i],tol=1e-8)) for i in changed],initial_diff=float(np.max(abs(initial[0]-initial[1]))),output_diff=float(np.max(abs(outputs[0]-outputs[1]))),audit_queries=[old.cache_info().misses,new.cache_info().misses],score=score(dict(pred=pred),'surrogatepass',test))))

def main():
    dev=list(pool(1742,3000)); devset=set(dev);test=list(dict.fromkeys(w for w in pool(108282,22000) if w not in devset))
    opening=fit('strict',dev); warm=fit('surrogatepass',dev,opening['basis']); cold=fit('surrogatepass',dev)
    delta=fit('surrogatepass',dev,base=opening)
    dp=delta['pred']; bp=opening['pred']; delta['pred']=lambda w:dp(w)+bp(w)
    # Byte-mode reversal is an exact old snapshot restore, not a new model fit.
    for name,m,mode in [('opening',opening,'strict'),('warm',warm,'surrogatepass'),('cold',cold,'surrogatepass'),('delta',delta,'surrogatepass')]:
        print(json.dumps(dict(name=name,rank=m['rank'],queries=m['queries'],cache_hits=m['cache_hits'],rounds=m['rounds'],secs=m['secs'],cond=m['cond'],storage=m['T'].nbytes+m['init'].nbytes+m['out'].nbytes,distinct_ops=len({np.round(x,7).tobytes() for x in m['T']}),score=score(m,mode,test))))
    targeted=[chr(c).encode('utf-8','surrogatepass') for c in range(0xd800,0xe000)]
    print(json.dumps(dict(targeted={name:score(m,'surrogatepass',targeted) for name,m in [('warm',warm),('cold',cold),('delta',delta)]}, targeted_stale=score(opening,'surrogatepass',targeted))))
    coordinate_audit([opening,warm,cold,delta],test)
    print(json.dumps(dict(stale=score(opening,'surrogatepass',test),changed=sum(oracle(w,'strict')!=oracle(w,'surrogatepass') for w in test),valid_strict=sum(oracle(w,'strict')>0 for w in test),valid_pass=sum(oracle(w,'surrogatepass')>0 for w in test),python=sys.version)))
if __name__=='__main__':main()
