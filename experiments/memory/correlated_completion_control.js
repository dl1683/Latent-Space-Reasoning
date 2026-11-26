// Standalone development control: node experiments/memory/correlated_completion_control.js
// No model calls, receiver integration, durable replay or lifecycle timings.
// The supplied uncertainty is a Euclidean residual ball in orthonormal coordinates;
// the structured oracle also holds for the corresponding sphere (dimension >=2).
// Taylor remainder: tanh'' is bounded by 4/(3 sqrt(3)); at zero use
// |tanh(t)-t| <= |t|^3/3, integrating tanh(t)^2 <= t^2.
// For each axis a, linearize jointly: sum a_j sech^2(c_j) M[:,j].
// Its residual support is r times the norm of that combined vector, not
// the sum of separate column supports. Add absolute Taylor error, intersect
// with the box numerator, and divide by the signed norm interval.
// Real-arithmetic argument only; epsilon here is a heuristic numerical guard.
function run() {
  const assert=(x,m)=>{if(!x)throw Error(m)};
  const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
  const norm=a=>Math.sqrt(dot(a,a));
  const unit=a=>a.map(x=>x/norm(a));
  const trans=a=>a[0].map((_,i)=>a.map(r=>r[i]));
  const mul=(a,b)=>a.map(row=>trans(b).map(col=>dot(row,col)));
  const eps=1e-10, floor=.95;
  let seed=812731; const rand=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
  const signed=()=>2*rand()-1;
  function enclosure(c,M,r,staged=null) {
    const radii=trans(M).map(col=>r*norm(col));
    const spread=staged||radii;
    const lo=c.map((x,i)=>Math.tanh(x-spread[i])-eps);
    const hi=c.map((x,i)=>Math.tanh(x+spread[i])+eps);
    const nlo=norm(lo.map((x,i)=>Math.max(0,x,-hi[i])))-eps;
    const nhi=norm(lo.map((x,i)=>Math.max(Math.abs(x),Math.abs(hi[i]))))+eps;
    const center=c.map(Math.tanh);
    // Exact real-arithmetic remainder: max |tanh''| = 4/(3 sqrt(3)).
    // At c=0, |tanh(t)-t| <= |t|^3/3 is tighter.
    const error=radii.map((x,i)=>c[i]===0?Math.min(x*x*x/3,2*x*x/(3*Math.sqrt(3))):2*x*x/(3*Math.sqrt(3)));
    const slopes=center.map(x=>1-x*x);
    function score(axis,affine=false) {
      if(nlo<=0)return [-1,1];
      let a=dot(axis,lo.map((x,i)=>axis[i]>=0?x:hi[i]));
      let b=dot(axis,hi.map((x,i)=>axis[i]>=0?x:lo[i]));
      if(affine) {
        const weights=axis.map((x,i)=>x*slopes[i]);
        const d=r*norm(M.map(row=>dot(row,weights)))+
          dot(axis.map(Math.abs),error)+eps;
        const mid=dot(axis,center);
        a=Math.max(a,mid-d); b=Math.min(b,mid+d);
      }
      return [Math.max(-1,a/(a>=0?nhi:nlo)-eps),
              Math.min(1,b/(b>=0?nlo:nhi)+eps)];
    }
    return {score};
  }
  const closure=(t,v)=>t[0]>=floor&&v[1]<floor;
  const report={structured:0,oracleSupported:0,stagedSupported:0,composedSupported:0,affineSupported:0,pointUnsafe:0,falseClosure:0,randomScoreContainments:0};
  // A fixed known preactivation plus two opposite hidden coordinates. The
  // factors have an exact cancellation in the second residual coordinate.
  const L=[[1,1],[1,-1]], R=[[0,.5,-.5],[0,.5,-.5]], M=mul(L,R);
  const target=[1,0,0];
  for(let k=0;k<128;k++) {
    const c=[.85+.35*rand(),0,0], a=.91+.025*rand(), b=Math.sqrt(1-a*a);
    for(const wide of [false,true])for(const rivalReachable of [false,true]) {
      const r=wide?.44+.10*rand():.105+.015*rand();
      const rival=[a,b/Math.sqrt(2),(rivalReachable?-1:1)*b/Math.sqrt(2)];
      const stage=trans(R).map(col=>r*dot(col.map(Math.abs),trans(L).map(norm)));
      const staged=enclosure(c,M,r,stage), composed=enclosure(c,M,r);
      const s=closure(staged.score(target),staged.score(rival));
      const d=closure(composed.score(target),composed.score(rival));
      const f=closure(composed.score(target,true),composed.score(rival,true));
      // Independent exact real-arithmetic extrema: h=(tanh(c0),tanh(t),-tanh(t))
      // with t in [-r,r], which the residual circle realizes as well.
      const z=Math.sqrt(2)*Math.tanh(r)/Math.tanh(c[0]);
      const effectiveB=rivalReachable?b:0;
      const zmax=Math.min(z,effectiveB/a);
      const maxRival=(a+effectiveB*zmax)/Math.sqrt(1+zmax*zmax);
      const exact=1/Math.sqrt(1+z*z)>=floor&&maxRival<floor;
      report.structured++;report.oracleSupported+=+exact;report.stagedSupported+=+s;
      report.composedSupported+=+d;report.affineSupported+=+f;report.pointUnsafe+=+(!exact&&a<floor);
      report.falseClosure+=+(!exact&&(s||d||f));
      assert(!(!exact&&(s||d||f)),"false closure");
      assert(!d||f,"intersection should preserve box closure");
      assert(exact===(!wide&&!rivalReachable),"fixture degeneracy");
    }
  }
  // Generic dense case: numerical containment checks are sampling, not proof.
  for(let k=0;k<128;k++){
    const c=Array.from({length:8},()=>signed()*2);
    const M=Array.from({length:3},()=>Array.from({length:8},signed));
    const r=.05+.55*rand(), e=enclosure(c,M,r);
    const axes=Array.from({length:2},()=>unit(c.map(signed)));
    for(let n=0;n<128;n++){
      const u=unit(Array.from({length:3},signed)).map(x=>x*r*(n%2?1:rand()));
      const h=unit(c.map((v,i)=>Math.tanh(v+dot(u,M.map(row=>row[i])))));
      for(const axis of axes){
        const v=dot(h,axis);
        for(const affine of [false,true]){
          const [a,b]=e.score(axis,affine);
          assert(v>=a-eps&&v<=b+eps,"sample escaped bound");
          report.randomScoreContainments++;
        }
      }
    }
  }
  // Equal-information lazy control: the exact same composed projected factors
  // and norms can be cached at first use. Revisions are synthetic immutable
  // descriptors; keys include region/version, not question or outcome.
  // The common prefix c and radius r are task-specific and not precomputed.
  const regions=128, revisions=2, queries=64, visited=4;
  let eager=new Map(), lazy=new Map(), eagerPrep=0,lazyPrep=0, reads=0;
  const key=(i,v)=>`${i}:${v}`;
  function prepare(i,v) {
    // Version-dependent fitted map, supplied to both arms.
    const l=[[1,1],[1,-1]], rr=[[0,.5,-.5],[0,.5,-.5]];
    if((i+v)%2)rr[0]=rr[0].map(x=>-x);
    const m=mul(l,rr);
    return {m,dependency:key(i,v)};
  }
  for(let v=0;v<revisions;v++) {
    for(let i=0;i<regions;i++){eager.set(key(i,v),prepare(i,v));eagerPrep++;}
    for(let n=0;n<queries/revisions;n++) {
      const i=n%visited,k=key(i,v);
      if(!lazy.has(k)){lazy.set(k,prepare(i,v));lazyPrep++;}
      const c=[1,0,0], r=.1+(n%5)*.01;
      const score=x=>[x.dependency,enclosure(c,x.m,r).score(target,true)];
      assert(JSON.stringify(score(eager.get(k)))===JSON.stringify(score(lazy.get(k))),"cache mismatch");
      reads++;
    }
  }
  report.cache={eagerPreparations:eagerPrep,lazyPreparations:lazyPrep,pairedReads:reads,
    composedScalarProductsPerPreparation:12,
    eagerCompositionMultiplications:eagerPrep*12,lazyCompositionMultiplications:lazyPrep*12};
  return report;
}
console.log(JSON.stringify(run(), null, 2));
