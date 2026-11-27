import base64,binascii,random,json
from collections import defaultdict
A=tuple(range(128))
def oracle(w,mode):
    try:
        if mode.startswith('64'):
            binascii.a2b_base64(w,strict_mode=mode=='64strict')
        else:
            base64.b32decode(w,casefold=mode!='32strict',map01=b'I' if mode=='32map' else None)
        return 1
    except (binascii.Error,ValueError):return 0

def pool(family,seed,n):
    r=random.Random(seed); out={b''}
    enc=base64.b64encode if family=='64' else base64.b32encode
    for _ in range(n):
        w=enc(r.randbytes(r.randrange(0,18)));out.add(w)
        for k in range(3):
            x=bytearray(w)
            for _ in range(1+k):
                i=r.randrange(len(x)+1); op=r.randrange(4)
                if op==0 or not x:x[i:i]=bytes([r.randrange(128)])
                elif i<len(x):
                    if op==1:del x[i]
                    elif op==2:x[i]=r.randrange(128)
                    else:x[i]=ord(chr(x[i]).swapcase())
            out.add(bytes(x))
        out.add(w.lower());out.add(w.replace(b'O',b'0').replace(b'I',b'1'))
        out.add(bytes(r.choices(A,k=r.randrange(25))))
    return sorted(out,key=lambda w:(len(w),w))

def learn(ask,alphabet,dev):
    S=[b'']; E=[b'']; rounds=0
    def row(w):return tuple(ask(w+e) for e in E)
    while True:
        rounds+=1
        rows={row(s):s for s in S}; extra=next((s+bytes([a]) for s in S for a in alphabet if row(s+bytes([a])) not in rows),None)
        if extra is not None:S.append(extra);continue
        split=None; seen={}
        for s in S:
            rr=row(s)
            if rr in seen:
                t=seen[rr]
                for a in alphabet:
                    x,y=row(s+bytes([a])),row(t+bytes([a]))
                    if x!=y:
                        split=bytes([a])+E[next(i for i in range(len(E)) if x[i]!=y[i])];break
                if split is not None:break
            else:seen[rr]=s
        if split is not None:E.append(split);continue
        rows={row(s):s for s in S}; init=row(b''); trans={rr:{a:row(s+bytes([a])) for a in alphabet} for rr,s in rows.items()}
        def pred(w):
            state=init
            for a in w:state=trans[state][a]
            return state[0]
        bad=next((w for w in dev if pred(w)!=ask(w)),None)
        if bad is None:return pred,trans,rounds
        S=list(dict.fromkeys(S+[bad[:i] for i in range(len(bad)+1)]))
        if rounds>1000:raise RuntimeError('round bound')

def run(mode,dev,test,initial=None,full=False):
    cache={}
    def ask(w):
        if w not in cache:cache[w]=oracle(w,mode)
        return cache[w]
    mapping=list(A if full else initial if initial is not None else [0]*len(A)); splits=0
    if not full:
        # Disagreement-driven class splitting; no oracle state or grammar.
        while True:
            table=bytes(mapping+list(range(128,256))); bad=next((w for w in dev if ask(w)!=ask(w.translate(table))),None)
            if bad is None:break
            x=bytearray(bad)
            for i,a in enumerate(bad):
                b=mapping[a]
                if a==b:continue
                before=ask(bytes(x));x[i]=b
                if before==ask(bytes(x)):continue
                p,s=bytes(x[:i]),bytes(x[i+1:]); groups={}
                for c in A:
                    if mapping[c]==b:
                        val=ask(p+bytes([c])+s);mapping[c]=groups.setdefault(val,c)
                assert len(groups)==2;splits+=1;break
            else:raise AssertionError('no witness')
    table=bytes(mapping+list(range(128,256)));reps=sorted(set(mapping)); mapped=sorted({w.translate(table) for w in dev},key=lambda w:(len(w),w))
    pred,T,rounds=learn(ask,reps,mapped)
    # Merge discovered equal transitions only for the exported prior; model remains unchanged.
    groups={}; exported=[]
    for a in A:
        sig=tuple(T[r][mapping[a]] for r in T);exported.append(groups.setdefault(sig,a))
    errors=sum(pred(w.translate(table))!=oracle(w,mode) for w in test if w not in cache and w not in dev)
    total=sum(w not in cache and w not in dev for w in test)
    allerrors=sum(pred(w.translate(table))!=oracle(w,mode) for w in test)
    return exported,dict(reads=len(cache),classes=len(reps),export_classes=len(set(exported)),splits=splits,states=len(T),rounds=rounds,errors=errors,eligible=total,allerrors=allerrors,alltest=len(test))

def main():
    for family,modes in [('64',['64strict','64loose']),('32',['32strict','32fold','32map'])]:
        dev=pool(family,314,600);test=pool(family,2718,2500)
        priors={}
        for mode in modes:
            prior,stats=run(mode,dev,test,full=True);priors[mode]=prior;print(mode,'full',json.dumps(stats),flush=True)
            _,stats=run(mode,dev,test);print(mode,'cold',json.dumps(stats),flush=True)
        for a in modes:
            for b in modes:
                if a!=b:
                    _,stats=run(b,dev,test,priors[a]);print(a+'->'+b,'warm',json.dumps(stats),flush=True)
if __name__=='__main__':main()
