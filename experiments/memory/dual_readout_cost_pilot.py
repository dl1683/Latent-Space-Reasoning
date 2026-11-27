import numpy as np
import collections, hashlib, json, random, re, subprocess
raw=subprocess.check_output(['dpkg-query','-W','-f=${binary:Package}\\t${db:Status-Status}\\t${Installed-Size}\\t${Depends}\\t${Pre-Depends}\\n'],text=True)
rows=sorted(line.split('\t') for line in raw.splitlines() if line.split('\t')[1]=='installed' and line.split('\t')[2].isdigit())
names=[r[0] for r in rows]; idx=collections.defaultdict(list)
for i,name in enumerate(names):idx[name.split(':')[0]].append(i)
x=[int(r[2]) for r in rows]; n=len(rows)
def bits(m):
    while m:
        b=m&-m; yield b.bit_length()-1; m-=b
D=[]
for r in rows:
    refs=re.findall(r'(?:^|[,|])\s*([a-z0-9][a-z0-9+.-]*(?::[a-z0-9-]+)?)',','.join(r[3:]))
    D.append(sum(1<<j for j in {j for s in refs for j in idx.get(s.split(':')[0], [])}))
E=[]
for m in D:
    t=m
    for j in bits(m):t|=D[j]
    E.append(t)
train=[i for i,s in enumerate(names) if int(hashlib.sha256(s.encode()).hexdigest(),16)%4!=0]; held=[i for i in range(n) if i not in set(train)]
Q=[m for i in train for m in (D[i],E[i]) if m]; R=list(sorted(set(Q))); rng=random.Random(37)
T={'direct':[D[i] for i in held if D[i]],'two_hop':[E[i] for i in held if E[i]]}
pairs=[rng.sample(held,2) for _ in range(500)]
T['union']=[E[i]|E[j] for i,j in pairs if E[i]|E[j]];T['difference']=[E[i]&~E[j] for i,j in pairs if E[i]&~E[j]]
T['novel_union']=[q for q in T['union'] if q not in set(Q)];T['novel_difference']=[q for q in T['difference'] if q not in set(Q)]
freq=collections.Counter(j for m in Q for j in bits(m)); order=sorted(range(n),key=lambda i:(-freq[i],i))
# Each stored row uses 8 bytes value + 2 bytes length + 2 bytes per member; coordinate is row of length one.
def cost(m):return 10+2*m.bit_count()
def fit(budget,share):
    k=min(n,int(budget*share)//12); C=sum(1<<i for i in order[:k]); remaining=budget-12*k
    U=[m&~C for m in Q]; base=np.array([m.bit_count() for m in U],dtype=np.int16); bank=[]; candidates=[(m,m&~C,cost(m)) for m in R if m.bit_count()>1 and m&~C]
    dist=np.array([[(u^v).bit_count() for m,v,c in candidates] for u in U],dtype=np.int16)
    costs=np.array([c for m,v,c in candidates]); active=np.ones(len(candidates),dtype=bool)
    while remaining>=14 and active.any():
        scores=np.maximum(0,base[:,None]-dist).sum(axis=0)/costs
        scores[~active | (costs>remaining)]=0
        i=int(scores.argmax())
        if scores[i]<=0:break
        m,v,c=candidates[i];bank.append(m);remaining-=c;base=np.minimum(base,dist[:,i]);active[i]=False
    return C,bank,budget-remaining,int(sum(base.astype(np.int64)))
def assess(C,bank,qs,values,two=False):
    coords={j:values[j] for j in bits(C)}; cached=[sum(values[j] for j in bits(m)) for m in bank]; total=0; worst=0
    for q in qs:
        b=(q&~C).bit_count(); choice=-1
        for i,m in enumerate(bank):
            d=((q^m)&~C).bit_count()
            if d<b:b=d;choice=i
        m=bank[choice] if choice>=0 else 0
        pos=q&~m; neg=m&~q
        residual={j:1 for j in bits(pos)}|{j:-1 for j in bits(neg)}
        value=cached[choice] if choice>=0 else 0
        if two:
            support=(pos|neg)&~C; p=pos&~C; g=neg&~C; second=None
            for i,a in enumerate(bank):
                v=a&~C; extra=(v&~support).bit_count()
                for sign,hit in [(1,(v&p).bit_count()),(-1,(v&g).bit_count())]:
                    d=support.bit_count()+extra-hit
                    if d<b:b=d;second=(i,sign)
            if second is not None:
                i,sign=second;value+=sign*cached[i]
                for j in bits(bank[i]):residual[j]=residual.get(j,0)-sign
        assert b==sum(c!=0 and j not in coords for j,c in residual.items())
        value+=sum(c*(coords[j] if j in coords else values[j]) for j,c in residual.items() if c)
        assert value==sum(values[j] for j in bits(q))
        total+=b;worst=max(worst,b)
    return [total,worst]
print('snapshot',hashlib.sha256(raw.encode()).hexdigest(),n,len(train),len(held),len(Q),len(R),{k:len(v) for k,v in T.items()})
for budget in [256,1024,4096]:
    for share in [0,0.5,1]:
        C,B,used,trainreads=fit(budget,share)
        out={k:assess(C,B,v,x) for k,v in T.items()}
        out2={k:assess(C,B,v,x,two=True) for k,v in T.items()}
        fanout=sum(m.bit_count() for m in B)+C.bit_count()
        # Same-pin value revisions: exact update rule checked for all retained rows, no q metadata changes.
        revised=x.copy(); change=random.Random(99).sample(range(n),100)
        for j in change:revised[j]+=1
        for m in B:assert sum(x[j] for j in bits(m))+sum(bool(m&(1<<j)) for j in change)==sum(revised[j] for j in bits(m))
        assert all(assess(C,B,v,revised)==out[k] and assess(C,B,v,revised,two=True)==out2[k] for k,v in T.items())
        edits=sum(C>>j&1 for j in change)+sum(sum(bool(m&(1<<j)) for j in change) for m in B)
        print(json.dumps({'budget':budget,'share':share,'used':used,'coordinates':C.bit_count(),'views':len(B),'trainreads':trainreads,'held':out,'two':out2,'fanout':fanout,'edits':edits}))
print('cold', {k:[sum(q.bit_count() for q in v),max(q.bit_count() for q in v)] for k,v in T.items()})
