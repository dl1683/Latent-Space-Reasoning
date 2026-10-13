"""Weighted structural prediction, independent of any gate/truth-table enumeration."""
import argparse,json,time,hashlib
from r16_groups import make,PairGroup

def filtration(G,ell,depth):
    assert ell>=2
    P=PairGroup(G); s=ell//2; atoms=[]
    for r in range(1,s+1):
        seeds=[]
        for g in range(G.size):
            h=G.power(g,2**(r-1)); seeds.append(h*G.size+int(G.inv[h]))
        H,gens=P.normal_closure(seeds); atoms.append((2*r+1,H,gens))
    N=[set(range(P.size)),set(range(P.size))]; ngens=[P.generators,P.generators]
    for d in range(2,depth+1):
        seeds=[h for w,H,gens in atoms if w>=d for h in gens]
        for i in range(1,d):
            seeds.extend(P.comm(a,b) for a in ngens[i] for b in ngens[d-i])
        H,gens=P.normal_closure(seeds); assert H<=N[-1]
        N.append(H); ngens.append(gens)
    K=[{g for g in range(G.size) if g*G.size+g in H} for H in N]
    # Check central-filtration inclusions independently on every element, in batches.
    for i in range(1,depth):
        for j in range(1,depth-i+1):
            for a in N[i]:
                for b in ngens[j]: assert P.comm(a,b) in N[i+j]
    return P,N,K,ngens,atoms

def receipt(G,ell,depth):
    start=time.time(); P,N,K,gens,atoms=filtration(G,ell,depth)
    return {'group':G.name,'lease':ell,'depth':depth,'seconds':time.time()-start,
            'pair_sizes':[len(H) for H in N],'diagonal_sizes':[len(H) for H in K],
            'diagonal_elements':[sorted(H) for H in K], 'pair_generators':gens,
            'atoms':[{'weight':w,'size':len(H),'generators':g} for w,H,g in atoms],
            'pair_hashes':[hashlib.sha256(json.dumps(sorted(H)).encode()).hexdigest() for H in N]}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--group',default='all');ap.add_argument('--depth',type=int,default=12);args=ap.parse_args()
    names=['D8','Q8','SL2F3','UT3F2','UT4F2','C4xC2','S3xC2','D16','Q16','UT3F3'] if args.group=='all' else [args.group]
    out={'method':'weighted pair filtration, fresh group tables','complete':False,'groups':[],'cases':[]}
    path='r16_FILTRATION_'+args.group+'.json'
    for name in names:
        G=make(name); v=G.check();out['groups'].append({'name':name,**v})
        for ell in [2,3,4]:
            r=receipt(G,ell,args.depth);out['cases'].append(r)
            print(name,ell,r['pair_sizes'],r['diagonal_sizes'],round(r['seconds'],2),flush=True)
            with open(path,'w') as f:json.dump(out,f,indent=2)
    out['complete']=True
    with open(path,'w') as f:json.dump(out,f,indent=2)
