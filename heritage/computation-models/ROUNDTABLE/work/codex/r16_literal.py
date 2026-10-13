"""Independent scalar-machine/Boolean-face checks, no older code imported."""
import json,itertools,random,time,argparse,hashlib
import numpy as np
from r16_groups import make,PairGroup
from r16_filtration import filtration

# Gate format: A(input-support mask), B(q-support mask,group-label), O(label).
def inverse(G,W):
    return [(g[0],g[1]) if g[0]=='A' else ('O',int(G.inv[g[1]])) if g[0]=='O' else ('B',g[1],int(G.inv[g[2]])) for g in reversed(W)]
def predicate(n,S):return np.array([int(x&S==S) for x in range(1<<n)],dtype=np.int32)
def execute(G,n,W,full_regular=False):
    m=1<<n;offset=np.zeros(m,dtype=np.int32);mult=np.full((2,m),G.e,dtype=np.int32);age=peak=0
    preds={S:predicate(n,S) for S in range(m) if S.bit_count()<=2}
    for gate in W:
        if gate[0]=='A':offset^=preds[gate[1]]
        elif gate[0]=='O':mult=G.mul[gate[1],mult]
        else:
            for b in range(2):
                active=(offset^b)&preds[gate[1]]
                mult[b]=np.where(active,G.mul[gate[2],mult[b]],mult[b])
        age=age+1 if np.any(offset) else 0;peak=max(peak,age)
    if full_regular:
        # Entire regular action is verified independently, starting with every z.
        for b in range(2):
            for x in range(m):
                for z in range(G.size):
                    bit=b;v=z
                    for gate in W:
                        if gate[0]=='A':bit^=int(x&gate[1]==gate[1])
                        elif gate[0]=='O':v=int(G.mul[gate[1],v])
                        elif bit and x&gate[1]==gate[1]:v=int(G.mul[gate[2],v])
                    assert bit==int(b^offset[x]);assert v==G.mul[mult[b,x],z]
    return mult,peak,bool(np.any(offset))

def echo_offset(terms,q,g):return [('A',p) for p in terms]+[('B',q,g)]+[('A',p) for p in reversed(terms)]
def polarization(G,terms,q,g):
    r=len(terms);W=[]
    for k in range(1,r+1):
        for J in itertools.combinations(range(r),k):
            label=G.power(g,(-1)**(r-k));W+=echo_offset([terms[j] for j in J],q,label)
    W.append(('B',q,G.power(g,(-1)**r)))
    return W
def face_coefficients(G,values):
    n=(len(values)-1).bit_length();order=sorted(range(1<<n),key=lambda S:(S.bit_count(),S));out={}
    for T in order:
        v=G.e
        for S in order:
            if S==T:break
            if S&T==S:v=int(G.mul[v,out[S]])
        out[T]=int(G.mul[G.inv[v],values[T]])
    return out
def pair_coefficients(P,values):
    n=(len(values)-1).bit_length();order=sorted(range(1<<n),key=lambda S:(S.bit_count(),S));out={}
    for T in order:
        v=P.e
        for S in order:
            if S==T:break
            if S&T==S:v=P.mul(v,out[S])
        out[T]=P.mul(P.inv(v),values[T])
    return out
def normalize(G,W):
    active=set();out=[]
    for gate in W:
        if gate[0]=='A':
            p=gate[1]
            if p in active:active.remove(p)
            else:active.add(p)
        elif gate[0]=='O':out.append(gate)
        else:out+=echo_offset(sorted(active),gate[1],gate[2])
    assert not active
    return out

def random_word(G,n,ell,rng,blocks=12):
    polys=[S for S in range(1<<n) if S.bit_count()<=2];qs=[0]+[1<<i for i in range(n)];W=[]
    for unused in range(blocks):
        sub=[];active=set()
        for k in range(rng.randint(1,ell+1)):
            options=[('O',rng.randrange(G.size)),('B',rng.choice(qs),rng.randrange(G.size))]
            p=rng.choice(polys);nactive=active^{p}
            if len(sub)+1+len(nactive)<=ell+1:options.append(('A',p))
            gate=rng.choice(options)
            if gate[0]!='A' and len(sub)+1+len(active)>ell+1:break
            sub.append(gate)
            if gate[0]=='A':active^={gate[1]}
        sub += [('A',p) for p in sorted(active)]
        W+=sub
    return W

def check_all():
    rng=random.Random(16008);out={'complete':False,'normalization':[],'polarization':[],'face_products':[],'inverse':[]}
    names=['D8','Q8','SL2F3','UT3F2','UT4F2','C4xC2','S3xC2','D16','Q16','UT3F3']
    for name in names:
        G=make(name)
        for ell in [2,3,4,6]:
            n=5;P,N,K,gens,atoms=filtration(G,ell,n);count=0
            for run in range(80):
                W=random_word(G,n,ell,rng);V=normalize(G,W)
                values,peak,dirty=execute(G,n,W);other,p2,d2=execute(G,n,V)
                assert not dirty and not d2 and peak<=ell and p2<=ell and np.array_equal(values,other)
                assert len(V)<=(ell+1)*len(W)
                iv,ip,idirty=execute(G,n,inverse(G,W));assert not idirty and ip==peak
                assert np.array_equal(iv,G.inv[values])
                coeff=pair_coefficients(P,[int(values[0,x])*G.size+int(values[1,x]) for x in range(1<<n)])
                assert all(g in N[S.bit_count()] for S,g in coeff.items());count+=1
            out['normalization'].append({'group':name,'lease':ell,'n':n,'words':count,'all_two_slices':True,'coefficient_memberships':count*(1<<n)})
            out['inverse'].append({'group':name,'lease':ell,'words':count,'inverse_peak_matches':True})
        for r in [1,2,3]:
            n=2*r+1;terms=[(1<<(2*j))|(1<<(2*j+1)) for j in range(r)];q=1<<(2*r);count=0
            for g in range(G.size):
                W=polarization(G,terms,q,g);values,peak,dirty=execute(G,n,W,full_regular=(r==1))
                h=G.power(g,(-2)**(r-1));expected=np.full_like(values,G.e);expected[0,-1]=h;expected[1,-1]=G.inv[h]
                assert np.array_equal(values,expected) and not dirty and peak<=2*r
                assert len(W)==(r+1)*2**r;count+=1
            out['polarization'].append({'group':name,'r':r,'n':n,'labels':count,'all_inputs_and_borrowers':True,'full_regular_direct_replay':r==1,'gates':len(W)})
        # Independently generate closed products of the predicted face labels.
        n=5;P,N,K,gens,atoms=filtration(G,4,n);count=0
        for run in range(100):
            f=np.full(1<<n,G.e,dtype=np.int32)
            for ignored in range(30):
                S=rng.randrange(1<<n);g=rng.choice(sorted(K[S.bit_count()]))
                f=np.where(predicate(n,S),G.mul[f,g],f)
            coeff=face_coefficients(G,f)
            assert all(g in K[S.bit_count()] for S,g in coeff.items());count+=1
        out['face_products'].append({'group':name,'lease':4,'n':n,'products':count,'factors_per_product':30,'all_coefficients_in_filtration':True})
        json.dump(out,open('r16_LITERAL_CHECKS.json','w'),indent=2)
        print('LITERAL',name,'complete',flush=True)
    out['complete']=True;json.dump(out,open('r16_LITERAL_CHECKS.json','w'),indent=2)

if __name__=='__main__':check_all()
