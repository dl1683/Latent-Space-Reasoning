// Query-blind partial transport composition control, 2026-11-26.
// Run: node experiments/memory/partial_transport_composition_control.js
//
// MATHEMATICAL FOUNDATION, not a learned or integrated receiver result.
// A source law can identify only P F = K, rather than the full latent
// map F: A -> B. A later law G: B -> C identifies a downstream row g F
// iff g lies in row(P), for unrestricted linear completions F.
// If g=hP, gF=hK. Otherwise choose v in ker(P) with gv != 0 and
// F' = F + v w^T: the observations are unchanged but a late nonzero
// x with w^T x != 0 has a different consequence. Thus composing
// individually underdetermined operators can yield exact consequences.
// Rows are admitted independently, not as an all-or-nothing vector.
// Scope/authority/coverage and exact source versions must be separately
// established. This canary supplies those labels and exact numeric laws,
// does not learn a frame or process text, and uses independent P rows.
// The +/-1 pivot elimination is intentionally restricted to the fixture.
// An industrial example: a calibration gives total load across two buses
// but not its allocation; aggregate capacity depends on the sum and can be
// checked, whereas a later per-bus limit cannot until allocation is known.
// A worker can observe the calibration projections and downstream sensitivity
// before any task arrives. The exact indexed arm must receive these same
// projections, downstream laws, eligibility, generations and budget.
// A byte-identical restore gets a new version key. The in-memory snapshots
// are not journal replay. No physical memory, latency, acquisition, model
// calibration, receiver integration or held-panel claim is made.
// Direct oracle enumerates free coordinates in these constructed worlds.
// Three samples per free dimension identify constancy of affine functions
// here; that is not a general finite-sampling proof for nonlinear maps.
// Eager/lazy counters count preparations and row factorization calls,
// not arithmetic FLOPs, actual indexing costs, latency or total economics.
// A lazy warm exact arm ties correctness; sparse eager maintenance loses.
//
function run(){
  const ok=(v,m)=>{if(!v)throw Error(m)}, eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
  const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
  const add=(a,b,k=1)=>a.map((v,i)=>v+k*b[i]);
  const mul=(a,k)=>a.map(v=>v*k);
  const mv=(a,x)=>a.map(r=>dot(r,x));
  let seed=926471;
  function rand(n){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed%n;}
  // Find h such that P^T h = g. All fixture elimination pivots are
  // +/-1; throw rather than silently use floating-point rank decisions.
  function factor(P,g){
    const m=P[0].length,n=P.length;
    const a=Array.from({length:m},(_,i)=>[...P.map(r=>r[i]),g[i]]);
    let rank=0;const piv=[];
    for(let j=0;j<n;j++){
      let k=rank;while(k<m&&!a[k][j])k++;
      if(k===m)continue;
      [a[k],a[rank]]=[a[rank],a[k]];
      const p=a[rank][j];ok(Math.abs(p)===1,"non-unit pivot outside fixture");
      a[rank]=mul(a[rank],p);
      for(let i=0;i<m;i++)if(i!==rank)a[i]=add(a[i],a[rank],-a[i][j]);
      piv.push(j);rank++;
    }
    for(let i=rank;i<m;i++)if(a[i][n])return null;
    const h=Array(n).fill(0);piv.forEach((j,i)=>h[j]=a[i][n]);
    ok(eq(P[0].map((_,i)=>dot(h,P.map(r=>r[i]))),g),"bad factor");
    return h;
  }
  // Each measurement is a governing versioned observation of P F = K.
  // Downstream laws contain a G row per independently readable consequence.
  // Scope and authority are supplied, not learned from prose.
  function compile(s,c){
    c.compile++;
    const relevant=s.sources.filter(r=>r.scope===s.scope);
    if(s.sources.some(r=>r.scope===null))return {rows:[null,null],deps:relevant.map(r=>r.id)};
    const upstream=relevant.filter(r=>r.kind==="measurement");
    const downstream=relevant.filter(r=>r.kind==="law");
    const deps=relevant.map(r=>r.id);
    if(!upstream.length||downstream.length!==1)return {rows:[null,null],deps};
    const P=upstream.map(r=>r.p),K=upstream.map(r=>r.k);
    // The test family only admits independent measurement rows. This
    // avoids claiming completeness for inconsistent/redundant measurements.
    for(let j=0;j<P.length;j++){
      const basis=P.slice(0,j);
      if(j&&factor(basis,P[j])!==null)return {rows:[null,null],deps};
    }
    const rows=downstream[0].g.map(g=>{
      c.factor++;
      const h=factor(P,g);
      return h===null?null:K[0].map((_,j)=>dot(h,K.map(r=>r[j])));
    });
    return {rows,deps};
  }
  function read(op,x,off=false){
    return {values:op.rows.map(r=>off||r===null?null:dot(r,x)),deps:op.deps};
  }
  function fixture(i){
    const perm=[0,1,2,3];for(let j=3;j>0;j--){const k=rand(j+1);[perm[j],perm[k]]=[perm[k],perm[j]];}
    const axes=perm.map(j=>Array.from({length:4},(_,k)=>k===j?(rand(16)<8?1:-1):0));
    // Coordinates in the intermediate space need not match either endpoint.
    const [a,b,c,d]=axes,p=add(a,c),q=add(b,d);
    const k=Array.from({length:3},()=>rand(13)-6),l=Array.from({length:3},()=>rand(13)-6),
      t=Array.from({length:3},()=>rand(13)-6);
    const scope=`site-${i}`,r=(id,kind,props,sc=scope)=>({id:`${i}:${id}`,kind,scope:sc,...props});
    const m=[r("sum-a:1","measurement",{p,k}),r("sum-b:1","measurement",{p:q,k:l})];
    const old=r("law:1","law",{g:[add(p,q,2),add(mul(p,2),q,-1)]});
    const changed=r("law:2","law",{g:[c,old.g[1]]});
    const extra=r("individual:1","measurement",{p:c,k:t});
    const extra2=r("individual:2","measurement",{p:c,k:add(t,[1,0,0])});
    const extra3=r("individual:3","measurement",{p:c,k:t});
    const decoy=r("decoy:1","law",{g:changed.g},"other-site");
    const rival=r("rival:1","law",{g:changed.g});
    const unknown=r("unresolved:1","law",{g:changed.g},null);
    const restored=r("law:3","law",{g:old.g});
    const states=[
      {sources:[...m,old],law:old,individual:null},
      {sources:[...m,old,decoy],law:old,individual:null},
      {sources:[...m,old,rival],law:null,individual:null},
      {sources:[...m,changed],law:changed,individual:null},
      {sources:[...m,changed,extra],law:changed,individual:extra},
      {sources:[...m,changed,extra2],law:changed,individual:extra2},
      {sources:[...m,changed],law:changed,individual:null},
      {sources:[...m,changed,extra3],law:changed,individual:extra3},
      {sources:[...m,changed,extra3,unknown],law:null,individual:extra3},
      {sources:m,law:null,individual:null},
      {sources:[...m,restored],law:restored,individual:null},
    ].map((s,generation)=>({...s,scope,generation}));
    return {states,axes,k,l};
  }
  // Independent direct reference: enumerate explicit intermediate
  // completions. For this linear fixture, three values per free
  // coordinate suffice to distinguish a constant affine function.
  // It does not use the row-space solver or the compiled operator.
  function oracle(f,s,x){
    if(!s.law)return [null,null];
    const [a,b,c,d]=f.axes,A=dot(f.k,x),B=dot(f.l,x);
    const U=s.individual?[dot(s.individual.k,x)]:[-1,0,1],V=[-1,0,1];
    const sets=[new Set(),new Set()];
    for(const u of U)for(const v of V){
      const y=add(add(mul(a,A-u),mul(b,B-v)),add(mul(c,u),mul(d,v)));
      mv(s.law.g,y).forEach((z,j)=>sets[j].add(z));
    }
    return sets.map(set=>set.size===1?[...set][0]:null);
  }
  const count=()=>({compile:0,factor:0,reads:0});
  const eager=count(),lazy=count(),sparseE=count(),sparseL=count();
  let reads=0,resolved=0,open=0,partial=0,off=0,history=0,unsafe=0,revisionFlips=0,wrongScope=0,freshRestores=0;
  for(let i=0;i<128;i++){
    const f=fixture(i);
    // Publication occurs before any task vectors are generated.
    const published=f.states.map(s=>compile(s,eager)),cache=new Map();
    const roots=Array.from({length:4},()=>[1+rand(7),rand(11)-5,rand(11)-5]);
    for(const [j,s] of f.states.entries()){
      if(!cache.has(j))cache.set(j,compile(s,lazy));
      for(const x of roots){
        eager.reads++;lazy.reads++;
        const a=read(published[j],x),b=read(cache.get(j),x),gold=oracle(f,s,x);
        ok(eq(a,b)&&eq(a.values,gold),"oracle mismatch");
        ok(eq(a.deps,s.sources.filter(r=>r.scope===s.scope).map(r=>r.id)),"provenance");
        reads++;resolved+=gold.filter(z=>z!==null).length;open+=gold.filter(z=>z===null).length;
        if(gold.filter(z=>z===null).length===1)partial++;
        if(gold.some(z=>z!==null)){
          ok(read(published[j],x,true).values.every(z=>z===null),"operator removal");off++;
        }
        // Zero-filling unresolved intermediate allocations would
        // always produce a numeric result when a unique law exists.
        if(s.law)unsafe+=gold.filter(z=>z===null).length;
      }
      if(j===1){ok(eq(published[0],published[1]),"wrong-scope decoy changed publication");wrongScope++;}
      if(j===5){ok(read(published[5],roots[0]).values[0]!==read(published[4],roots[0]).values[0],"edit did not reach consequence");revisionFlips++;}
      if(j===7){ok(eq(published[4].rows,published[7].rows)&&!eq(published[4].deps,published[7].deps),"restoration provenance");freshRestores++;}
    }
    for(const x of roots){
      ok(eq(read(published[0],x).values,oracle(f,f.states[0],x)),"history changed");history++;
    }
    // Sparse lifecycle: same 11 installations, only the final state of
    // eight regions receives a query. Lazy exact has identical information,
    // same factorization and a version-keyed compiled cache.
    const installed=f.states.map(s=>compile(s,sparseE));
    if(i<8){
      const last=f.states.length-1,op=compile(f.states[last],sparseL);
      for(const x of roots){
        sparseE.reads++;sparseL.reads++;
        ok(eq(read(installed[last],x),read(op,x)),"sparse mismatch");
      }
    }
  }
  return {worlds:128,reads,resolvedScalarConsequences:resolved,unresolvedScalarConsequences:open,
    partialReads:partial,operatorOffChecks:off,historicalChecks:history,
    zeroFillWouldFalselyClose:unsafe,changedMeasurementChecks:revisionFlips,wrongScopeChecks:wrongScope,
    freshRestorationChecks:freshRestores,dense:{eager,lazy},sparse:{eager:sparseE,lazy:sparseL}};
}
console.log(JSON.stringify(run(), null, 2));
