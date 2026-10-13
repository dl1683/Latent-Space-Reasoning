"""Fresh finite groups; does not import any earlier-round implementation."""
import itertools
from collections import deque
import numpy as np

class Group:
    def __init__(self,name,items,multiply):
        self.name=name; self.items=list(items); self.size=len(self.items)
        index={a:i for i,a in enumerate(self.items)}
        self.mul=np.array([[index[multiply(a,b)] for b in self.items] for a in self.items],dtype=np.int32)
        self.e=next(i for i in range(self.size) if np.array_equal(self.mul[i],np.arange(self.size)) and np.array_equal(self.mul[:,i],np.arange(self.size)))
        self.inv=np.array([next(j for j in range(self.size) if self.mul[i,j]==self.e and self.mul[j,i]==self.e) for i in range(self.size)],dtype=np.int32)
        self.generators=self.minimal_generators(range(self.size))
    def closure(self,seeds):
        seeds=list(set(map(int,seeds))); known={self.e}; todo=deque([self.e])
        while todo:
            a=todo.popleft()
            for b in seeds:
                c=int(self.mul[a,b])
                if c not in known: known.add(c); todo.append(c)
        return known
    def minimal_generators(self,values):
        gens=[]; known={self.e}
        for a in sorted(values):
            if a not in known: gens.append(a); known=self.closure(gens)
        return gens
    def power(self,g,k):
        if k<0: g=int(self.inv[g]); k=-k
        out=self.e
        while k:
            if k&1: out=int(self.mul[out,g])
            g=int(self.mul[g,g]); k>>=1
        return out
    def comm(self,a,b):
        return int(self.mul[self.mul[self.mul[self.inv[a],self.inv[b]],a],b])
    def normal_closure(self,seeds):
        gens=self.minimal_generators(seeds); known=self.closure(gens)
        while True:
            changed=False
            for a in list(gens):
                for p in self.generators:
                    c=int(self.mul[self.mul[self.inv[p],a],p])
                    if c not in known:
                        gens.append(c); known=self.closure(gens); changed=True
            if not changed: return known,gens
    def comm_subgroup(self,H,K):
        return self.normal_closure([self.comm(a,b) for a in self.minimal_generators(H) for b in self.minimal_generators(K)])[0]
    def lower_central(self):
        terms=[set(range(self.size))]
        while True:
            v=self.comm_subgroup(terms[-1],terms[0]); terms.append(v)
            if v==terms[-2]: break
        return terms
    def check(self):
        # Every associativity triple, not just generator identities.
        ar=np.arange(self.size)
        for a in ar:
            for b in ar:
                assert np.array_equal(self.mul[self.mul[a,b],ar],self.mul[a,self.mul[b,ar]])
        return {'order':self.size,'associativity_triples':self.size**3,'identity':self.e,'generators':self.generators,'lower_central_sizes':[len(v) for v in self.lower_central()]}

def cyclic(m): return Group('C'+str(m),range(m),lambda a,b:(a+b)%m)
def dihedral(order):
    m=order//2
    return Group('D'+str(order),itertools.product(range(m),range(2)),lambda a,b:((a[0]+(-1 if a[1] else 1)*b[0])%m,a[1]^b[1]))
def quaternion(order):
    m=order//2
    return Group('Q'+str(order),itertools.product(range(m),range(2)),lambda a,b:((a[0]+(-1 if a[1] else 1)*b[0]+(m//2 if a[1] and b[1] else 0))%m,a[1]^b[1]))
def direct(G,H,name=None):
    return Group(name or G.name+'x'+H.name,itertools.product(range(G.size),range(H.size)),lambda a,b:(int(G.mul[a[0],b[0]]),int(H.mul[a[1],b[1]])))
def symmetric3():
    return Group('S3',itertools.permutations(range(3)),lambda a,b:tuple(a[b[i]] for i in range(3)))
def sl2_3():
    items=[a for a in itertools.product(range(3),repeat=4) if (a[0]*a[3]-a[1]*a[2])%3==1]
    return Group('SL2F3',items,lambda a,b:((a[0]*b[0]+a[1]*b[2])%3,(a[0]*b[1]+a[1]*b[3])%3,(a[2]*b[0]+a[3]*b[2])%3,(a[2]*b[1]+a[3]*b[3])%3))
def unitriangular(d,p):
    positions=[(i,j) for i in range(d) for j in range(i+1,d)]; idx={v:k for k,v in enumerate(positions)}
    def multiply(a,b):
        return tuple((a[idx[i,j]]+b[idx[i,j]]+sum(a[idx[i,k]]*b[idx[k,j]] for k in range(i+1,j)))%p for i,j in positions)
    return Group('UT'+str(d)+'F'+str(p),itertools.product(range(p),repeat=len(positions)),multiply)
def make(name):
    if name in ('S4','A4','A5'):
        d=5 if name=='A5' else 4
        def even(a):return sum(a[i]>a[j] for i in range(d) for j in range(i+1,d))%2==0
        items=[a for a in itertools.permutations(range(d)) if name=='S4' or even(a)]
        return Group(name,items,lambda a,b:tuple(a[b[i]] for i in range(d)))
    if name in ('D16xC4','D8xC3','M32xC3'):
        a,b=name.split('x');return direct(make(a),make(b),name)
    if name=='W64C4':
        def rotate(a,k):return ((a<<k)|(a>>(4-k)))&15 if k else a
        return Group(name,itertools.product(range(16),range(4)),lambda a,b:(a[0]^rotate(b[0],a[1]),(a[1]+b[1])%4))
    if name=='W64V4':
        def shift(a,k):return sum(((a>>i)&1)<<(i^k) for i in range(4))
        return Group(name,itertools.product(range(16),range(4)),lambda a,b:(a[0]^shift(b[0],a[1]),a[1]^b[1]))
    if name=='M32':
        return Group(name,itertools.product(range(8),range(4)),lambda a,b:((a[0]+pow(5,a[1],8)*b[0])%8,(a[1]+b[1])%4))
    if name=='M64':
        return Group(name,itertools.product(range(16),range(4)),lambda a,b:((a[0]+pow(5,a[1],16)*b[0])%16,(a[1]+b[1])%4))
    if name=='T128':
        return Group(name,itertools.product(range(16),range(8)),lambda a,b:((a[0]+pow(3,a[1],16)*b[0])%16,(a[1]+b[1])%8))
    if name=='M16':
        return Group(name,itertools.product(range(4),range(4)),lambda a,b:((a[0]+(-1 if a[1]%2 else 1)*b[0])%4,(a[1]+b[1])%4))
    if name=='F21':
        return Group(name,itertools.product(range(7),range(3)),lambda a,b:((a[0]+pow(2,a[1],7)*b[0])%7,(a[1]+b[1])%3))
    if name=='M27':
        return Group(name,itertools.product(range(9),range(3)),lambda a,b:((a[0]+pow(4,a[1],9)*b[0])%9,(a[1]+b[1])%3))
    if name=='U81':
        U=unitriangular(4,3)
        a=U.items.index((1,0,0,0,0,0));b=U.items.index((0,0,0,1,0,1))
        values=sorted(U.closure([a,b]));assert len(values)==81
        return Group(name,values,lambda a,b:int(U.mul[a,b]))
    if name=='X32':
        # Extraspecial plus type, central product of two D8s. Coordinates u,v,w.
        return Group(name,itertools.product(range(4),range(4),range(2)),lambda a,b:(a[0]^b[0],a[1]^b[1],a[2]^b[2]^((a[0]&b[1]).bit_count()%2)))
    if name.startswith('D'): return dihedral(int(name[1:]))
    if name.startswith('Q'): return quaternion(int(name[1:]))
    if name=='SL2F3': return sl2_3()
    if name=='S3xC2': return direct(symmetric3(),cyclic(2),'S3xC2')
    if name=='C4xC2': return direct(cyclic(4),cyclic(2),'C4xC2')
    if name.startswith('UT'): return unitriangular(int(name[2:name.index('F')]),int(name.split('F')[1]))
    if name.startswith('C'): return cyclic(int(name[1:]))
    raise ValueError(name)

class PairGroup:
    """No |G|^4 multiplication table is allocated."""
    def __init__(self,G):
        self.G=G; self.size=G.size**2; self.e=G.e*G.size+G.e
        self.generators=[g*G.size+G.e for g in G.generators]+[G.e*G.size+g for g in G.generators]
    def mul(self,a,b):
        m=self.G.size
        return int(self.G.mul[a//m,b//m])*m+int(self.G.mul[a%m,b%m])
    def inv(self,a):
        m=self.G.size
        return int(self.G.inv[a//m])*m+int(self.G.inv[a%m])
    def comm(self,a,b): return self.mul(self.mul(self.mul(self.inv(a),self.inv(b)),a),b)
    def closure(self,seeds):
        seeds=list(set(seeds)); known={self.e}; todo=deque([self.e])
        while todo:
            a=todo.popleft()
            for b in seeds:
                c=self.mul(a,b)
                if c not in known: known.add(c); todo.append(c)
        return known
    def minimal_generators(self,values):
        gens=[]; known={self.e}
        for a in sorted(values):
            if a not in known: gens.append(a); known=self.closure(gens)
        return gens
    def normal_closure(self,seeds):
        gens=[]; known={self.e}
        for a in seeds:
            if a not in known: gens.append(a); known=self.closure(gens)
        while True:
            changed=False
            for a in list(gens):
                for p in self.generators:
                    c=self.mul(self.mul(self.inv(p),a),p)
                    if c not in known: gens.append(c); known=self.closure(gens); changed=True
            if not changed: return known,gens
