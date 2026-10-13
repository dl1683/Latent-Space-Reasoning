"""Exact returned-block subgroup census using independent permutation orbits.

No weighted-filtration ingredients enter the enumerated generating group.
Diagonal acceptance is implemented as point stabilizers in a two-sided action.
"""
import argparse,itertools,json,time,math,hashlib,os,sys,threading,datetime
import numpy as np
from sympy.combinatorics import Permutation,PermutationGroup
from r16_groups import make
from r16_filtration import filtration

def predicates(n,s):
    m=1<<n; full=(1<<m)-1
    polys=[sum(1<<x for x in range(m) if x&S==S) for S in range(m) if S.bit_count()<=2]
    offsets={0}
    for k in range(1,min(s,len(polys))+1):
        for terms in itertools.combinations(polys,k):
            a=0
            for p in terms:a^=p
            offsets.add(a)
    qs=[full]+[sum(1<<x for x in range(m) if x>>i&1) for i in range(n)]
    pairs=sorted({(a&q,(full^a)&q) for a in offsets for q in qs})
    return pairs,len(offsets)

def canonical_orbits(n,pairs):
    """Input renaming only; orbit expansion must exactly recover the source list."""
    m=1<<n; pairs=set(pairs); images=[]
    for p in itertools.permutations(range(n)):
        images.append([sum(((x>>i)&1)<<p[i] for i in range(n)) for x in range(m)])
    def rename(a,ix):return sum(1<<ix[x] for x in range(m) if a>>x&1)
    unseen=set(pairs); sizes=[]; reps=[]
    while unseen:
        a,b=min(unseen); orbit={(rename(a,ix),rename(b,ix)) for ix in images}
        assert orbit<=pairs; unseen-=orbit; sizes.append(len(orbit));reps.append(min(orbit))
    assert sum(sizes)==len(pairs)
    return {'orbits':len(sizes),'sizes':sizes,'representatives':reps,'input_permutations':len(images)}

def action(G,f0,f1):
    """Faithful regular pair action plus an auxiliary left/right action.

    For each x, use three disjoint |G|-point blocks. The first block is
    z -> f0*z*f1^-1. The other two give faithful regular slice actions.
    Fixing every identity point in the first blocks means f0=f1 pointwise.
    """
    k=G.size;m=len(f0);out=np.empty(3*m*k,dtype=np.int32);ar=np.arange(k)
    for x in range(m):
        out[x*k:(x+1)*k]=G.mul[G.mul[f0[x],ar],G.inv[f1[x]]]+x*k
        out[(m+x)*k:(m+x+1)*k]=G.mul[f0[x],ar]+(m+x)*k
        out[(2*m+x)*k:(2*m+x+1)*k]=G.mul[f1[x],ar]+(2*m+x)*k
    return Permutation(out.tolist())

def census(name,n,ell):
    start=time.time();G=make(name);m=1<<n;pairs,n_offsets=predicates(n,ell//2)
    canonical=canonical_orbits(n,pairs) if n<=4 else {'performed':False,'reason':'factorial renaming guard; group orbit census still exact'}
    generators=[];e=np.full(m,G.e,dtype=np.int32)
    for g in G.generators:generators.append(action(G,np.full(m,g),np.full(m,g)))
    for a,b in pairs:
        for g in G.generators:
            f0=np.array([g if a>>x&1 else G.e for x in range(m)],dtype=np.int32)
            f1=np.array([g if b>>x&1 else G.e for x in range(m)],dtype=np.int32)
            if np.any(f0!=G.e) or np.any(f1!=G.e):generators.append(action(G,f0,f1))
    construction=time.time()-start
    print('START',name,n,ell,'degree',3*m*G.size,'generators',len(generators),flush=True)
    H=PermutationGroup(generators); points=[x*G.size+G.e for x in range(m)]
    base,strong=H.schreier_sims_incremental(base=points)
    assert base[:m]==points
    Horder=int(H.order())
    # Strong generators fixing this initial base prefix generate its stabilizer.
    diag_gens=[g for g in strong if all(g(p)==p for p in points)]
    if not diag_gens:diag_gens=[Permutation(list(range(3*m*G.size)))]
    accepted=PermutationGroup(diag_gens); actual=int(accepted.order())
    P,N,K,kgen,atoms=filtration(G,ell,n)
    predicted=math.prod(len(K[S.bit_count()]) for S in range(m))
    rawpred=math.prod(len(N[S.bit_count()]) for S in range(m))
    checked=0
    for S in range(m):
        for g in G.minimal_generators(K[S.bit_count()]):
            f=np.array([g if x&S==S else G.e for x in range(m)],dtype=np.int32)
            assert accepted.contains(action(G,f,f));checked+=1
    assert actual==predicted,(name,n,ell,actual,predicted)
    assert Horder==rawpred,(name,n,ell,Horder,rawpred)
    # Complete finite inclusion both ways follows from generator membership + equal order.
    receipt={'group':name,'n':n,'lease':ell,'complete':True,'method':'exact Schreier--Sims; identity-point stabilizer; generator inclusion and equal orders',
             'input_assignments':m,'permutation_degree':3*m*G.size,'offsets':n_offsets,'predicate_pairs':len(pairs),'renaming':canonical,
             'raw_generators':len(generators),'strong_generators':len(strong),'base_length':len(base),
             'raw_order':str(Horder),'predicted_raw_order':str(rawpred),'accepted_order':str(actual),'predicted_accepted_order':str(predicted),
             'all_predicted_face_generator_memberships':checked,'generation_seconds':construction,'seconds':time.time()-start,
             'rss_bytes':__import__('psutil').Process().memory_info().rss}
    print('DONE',name,n,ell,'accepted',actual,'seconds',round(receipt['seconds'],2),flush=True)
    return receipt

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--group',default='all');ap.add_argument('--n',type=int,default=2);ap.add_argument('--lease',type=int,default=0);args=ap.parse_args()
    # Bound every new process, even while a library call is in progress.
    def guard():
        import psutil
        deadline=datetime.datetime.fromisoformat(os.environ.get('R16_RESEARCH_DEADLINE_UTC','2026-10-08T15:30:00+00:00'))
        assert deadline.tzinfo is not None
        process=psutil.Process();peak=0
        while True:
            rss=process.memory_info().rss;peak=max(peak,rss)
            now=datetime.datetime.now(datetime.timezone.utc)
            reason='RAM cap' if rss>6*1024**3 else 'round deadline' if now>=deadline else None
            if reason:
                json.dump({'complete':False,'reason':reason,'argv':sys.argv,'peak_rss_bytes':peak,'utc':now.isoformat()},open('r16_PARTIAL_'+args.group+'_n'+str(args.n)+'_ell'+str(args.lease)+'.json','w'),indent=2)
                print('STOP',reason,'partial receipt preserved',flush=True);os._exit(3)
            time.sleep(2)
    threading.Thread(target=guard,daemon=True).start()
    names=['D8','Q8','SL2F3','UT3F2','UT4F2','C4xC2','S3xC2','D16','Q16'] if args.group=='all' else [args.group]
    for name in names:
        for ell in ([2,3,4] if args.lease==0 else [args.lease]):
            r=census(name,args.n,ell)
            with open('r16_ORBITS_'+name+'_n'+str(args.n)+'_ell'+str(ell)+'.json','w') as f:json.dump(r,f,indent=2)
