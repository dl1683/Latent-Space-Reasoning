// Standalone CPU mathematical control, not a production matcher or model run.
// Run: node experiments/memory/regional_bridge_control.js
// Exact integer keys have squared norm 4225. The band is cosine .95 +/- .025:
// a pair is plausible iff integer dot >= 3909, definite iff dot > 4119.
// Compare candidate generation only; source-level ambiguity admission is shared.
// The box bound is conservative in exact arithmetic. Float64 production bounds
// would need an outward rounding margin. The pooled mean is a negative control.
// Both safe arms cache exact dot products by key pair across revisions, including
// inactive pairs; cache space/eviction, source grouping, box reconstruction and
// admission costs are NOT included in dot counts. SourceRows/boxKeyVisits/bounds/
// keyVisits are separate work receipts, not a total runtime or edge-cost model.
// The fixtures test source-level bridge admission, not downstream recurrence.

function regionalBridgeControl() {
  "use strict";
  // Exact integer-grid surrogate for the Float64 unit-key matcher.
  const D = 16, SCALE = 65, LOW = 3909, HIGH = 4119;
  const point = (x, y, z=0) => [x,y,z,...Array(D-3).fill(0)];
  const keys = [point(65,0),point(63,16),point(60,25),point(0,65),point(0,-65),point(-65,0),point(-63,16),point(0,0,65),point(0,0,-65)];
  const row = (id,k,region) => ({id,key:keys[k],region});
  const keyId = k => k.join(",");
  const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
  const check = (ok,msg) => { if (!ok) throw Error(msg); };
  const eq = (a,b,msg) => check(JSON.stringify(a)===JSON.stringify(b),msg);
  function group(rows) {
    const m=new Map();
    for (const r of rows) {
      check(r.key.length===D && dot(r.key,r.key)===SCALE*SCALE,"bad fixture key");
      const id=keyId(r.key);
      if (!m.has(id)) m.set(id,{key:r.key,ids:[]});
      m.get(id).ids.push(r.id);
    }
    return m;
  }
  function regionTable(rows) {
    const m=new Map();
    for (const r of rows) {
      if (!m.has(r.region)) m.set(r.region,new Map());
      m.get(r.region).set(keyId(r.key),r.key);
    }
    return [...m.values()].map(g=>{
      const lo=Array(D).fill(Infinity), hi=Array(D).fill(-Infinity), sum=Array(D).fill(0);
      for(const k of g.values()) for(let i=0;i<D;i++) {
        lo[i]=Math.min(lo[i],k[i]); hi[i]=Math.max(hi[i],k[i]); sum[i]+=k[i];
      }
      return {g,lo,hi,sum};
    });
  }
  // The admission rule is shared: plausible competitors count on BOTH
  // sides, including duplicate source IDs with identical key coordinates.
  function admit(A,B,edges) {
    const left=new Map(),right=new Map(), definite=[];
    for(const [a,b,s] of edges) {
      if(s<LOW) continue;
      left.set(a,(left.get(a)||0)+B.get(b).ids.length);
      right.set(b,(right.get(b)||0)+A.get(a).ids.length);
      if(s>HIGH) definite.push([a,b]);
    }
    const bridges=[];
    for(const [a,b] of definite) if(left.get(a)===1 && right.get(b)===1)
      bridges.push([A.get(a).ids[0],B.get(b).ids[0]]);
    const matched=new Set(bridges.flat());
    return {
      bridges:bridges.sort((a,b)=>JSON.stringify(a).localeCompare(JSON.stringify(b))),
      unresolved:[...A.values(),...B.values()].flatMap(g=>g.ids).filter(id=>!matched.has(id)).sort(),
    };
  }
  function read(left,right,method,cache) {
    const A=group(left),B=group(right), regions=method==="join"?[]:regionTable(right);
    const counts={sourceRows:left.length+right.length,boxKeyVisits:regions.reduce((s,r)=>s+r.g.size,0),
      bounds:0,keyVisits:0,newDots:0,cacheHits:0,rawPairs:left.length*right.length};
    const edges=[];
    for(const [a,ag] of A) {
      let candidates;
      if(method==="join") candidates=B.keys();
      else {
        const selected=new Set();
        for(const r of regions) {
          counts.bounds++;
          // Exact safe maximum over a coordinate envelope (integer arithmetic).
          // Pooled comparator is deliberately unsafe, using mean distinct key.
          const bound=method==="box"
            ? ag.key.reduce((s,x,i)=>s+x*(x>=0?r.hi[i]:r.lo[i]),0)
            : dot(ag.key,r.sum)/r.g.size;
          if(bound<LOW) continue;
          for(const b of r.g.keys()) selected.add(b);
        }
        candidates=selected;
      }
      for(const b of candidates) {
        counts.keyVisits++;
        const id=a+"|"+b;
        let s;
        if(cache.has(id)){counts.cacheHits++;s=cache.get(id);}
        else {counts.newDots++;s=dot(ag.key,B.get(b).key);cache.set(id,s);}
        if(s>=LOW) edges.push([a,b,s]);
      }
    }
    return {result:admit(A,B,edges),counts};
  }
  // Independent source-level oracle; no bucketing or region filtering.
  function oracle(left,right) {
    const pairs=[];
    for(const a of left) for(const b of right) {
      const s=dot(a.key,b.key);
      if(s>=LOW)pairs.push([a.id,b.id,s]);
    }
    const out=pairs.filter(([a,b,s])=>s>HIGH &&
      pairs.filter(p=>p[0]===a).length===1 && pairs.filter(p=>p[1]===b).length===1)
      .map(p=>p.slice(0,2)).sort((a,b)=>JSON.stringify(a).localeCompare(JSON.stringify(b)));
    const matched=new Set(out.flat());
    return {bridges:out,unresolved:[...left,...right].map(r=>r.id).filter(id=>!matched.has(id)).sort()};
  }
  const left=[row("entry",0,"local")];
  const strong=row("strong",0,"local"),weak=row("weak",1,"remote");
  const noise=[...Array.from({length:64},(_,i)=>row("noise+"+i,3,"remote")),
    ...Array.from({length:32},(_,i)=>row("noise-"+i,4,"remote"))];
  const snapshots=[
    ["opening",[strong,...noise]],
    ["add marginal competitor",[strong,weak,...noise]],
    ["withdraw strong",[weak,...noise]],
    ["promote remaining competitor",[row("weak",0,"remote"),...noise]],
    ["restore opening",[strong,...noise]],
  ];
  const caches={join:new Map(),box:new Map(),pool:new Map()};
  const revisionResults=snapshots.map(([name,right])=>{
    const j=read(left,right,"join",caches.join),b=read(left,right,"box",caches.box),
      p=read(left,right,"pool",caches.pool);
    eq(j.result,oracle(left,right),"factored join mismatch: "+name);
    eq(b.result,j.result,"envelope mismatch: "+name);
    return {name,join:j.counts,box:b.counts,exactBridges:j.result.bridges,
      pooledBridges:p.result.bridges,exactUnresolved:j.result.unresolved.length};
  });
  eq(revisionResults.map(x=>x.exactBridges),[[["entry","strong"]],[],[],[["entry","weak"]],[["entry","strong"]]],"revision sequence");
  eq(revisionResults[1].pooledBridges,[["entry","strong"]],"pooled false singleton must reproduce");
  const denseL=Array.from({length:128},(_,i)=>row("L"+i,0,"dense"));
  const denseR=Array.from({length:128},(_,i)=>row("R"+i,0,"dense"));
  const denseCaches={join:new Map(),box:new Map()};
  const denseResults=[denseR,[...denseR.slice(0,-1),row("R127",1,"dense")],denseR].map((right,i)=>{
    const j=read(denseL,right,"join",denseCaches.join),b=read(denseL,right,"box",denseCaches.box);
    eq(j.result,b.result,"dense mismatch");
    eq(j.result.bridges,[],"dense ambiguity");
    return {revision:i,join:j.counts,box:b.counts};
  });
  // Revisions of mixed sparse/dense regional worlds; all three methods see
  // identical rows, and both safe methods retain exact score caches.
  let seed=92761;
  const random=n=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed%n;};
  let a=[],b=[]; const jc=new Map(),bc=new Map();
  for(let t=0;t<1000;t++){
    const side=random(7)<3?a:b, prefix=side===a?"a":"b";
    const id=prefix+random(11),idx=side.findIndex(r=>r.id===id);
    if(idx>=0)side.splice(idx,1);
    if(random(5)!==0)side.push(row(id,random(keys.length),"region"+random(5)));
    const j=read(a,b,"join",jc),q=read(a,b,"box",bc),o=oracle(a,b);
    eq(j.result,o,"random join "+t);eq(q.result,o,"random envelope "+t);
  }
  return {dimension:D,low:LOW,high:HIGH,scale:SCALE,revisionResults,denseResults,randomRevisions:1000};
}

console.log(JSON.stringify(regionalBridgeControl(), null, 2));
