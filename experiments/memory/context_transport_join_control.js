// Standalone mathematical control; no model calls or production receiver claims.
// Run: node experiments/memory/context_transport_join_control.js
// Context observations and authority metadata are supplied. Eager maintenance
// and lazy incremental join share input information, operator family, subpath
// products and provenance. Counts are logical operations, not timings or
// complete lifecycle costs. Revisions are replacement snapshots in memory,
// not durable journal replay. The final restore uses the initial source object.
function run() {
  // All arithmetic is exact on small integer rows. Two orthogonal maps have
  // the same on-axis entry/exit, but different off-axis actions.
  const assert=(ok,msg)=>{if(!ok)throw Error(msg)};
  const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
  const actions={R:[[0,-1],[1,0]],F:[[0,1],[1,0]]};
  const apply=(m,x)=>m.map(r=>r[0]*x[0]+r[1]*x[1]);
  const product=(a,b)=>a.map(r=>[r[0]*b[0][0]+r[1]*b[1][0],r[0]*b[0][1]+r[1]*b[1][1]]);
  const evidence=(id,action)=>({id,observed:apply(actions[action],[9,2])});
  const region=(i)=>({id:i,base:evidence(`base:${i}:v1`,i%2?"F":"R"),alternative:evidence(`alt:${i}:v1`,i%2?"R":"F"),orders:[]});
  const order=(id,issuer="board")=>({id,issuer,start:10,end:20});
  // Both arms receive identical source-backed context, issuer whitelist,
  // scope and half-open effective interval. No task-time roots enter fitting.
  function choose(r,t,counts) {
    counts.select++;
    const active=r.orders.filter(o=>o.issuer==="board"&&o.start<=t&&t<o.end);
    if(active.length>1)return {open:"authority conflict",deps:active.map(o=>o.id)};
    const e=active.length?r.alternative:r.base;
    const deps=[...active.map(o=>o.id),...(e?[e.id]:[])];
    if(!e)return {open:"missing context",deps};
    const matches=Object.keys(actions).filter(k=>eq(apply(actions[k],[9,2]),e.observed));
    if(matches.length!==1)return {open:"unidentified transport",deps};
    return {action:matches[0],deps};
  }
  // Numeric subpath products can be shared across regions; evidence receipts
  // cannot. A lazy join caches by region generation and effective horizon.
  class World {
    constructor(eager,times) {this.eager=eager;this.times=times;this.regions=new Map();this.cache=new Map();this.products=new Map();this.counts={select:0,compose:0,reads:0,apply:0};}
    install(r) {
      this.regions.set(r.id,r);
      this.cache.delete(r.id);
      if(this.eager)for(const t of this.times)this.resolve(r.id,t);
    }
    resolve(id,t) {
      if(!this.cache.has(id))this.cache.set(id,new Map());
      const byTime=this.cache.get(id);
      if(!byTime.has(t)){
        const s=choose(this.regions.get(id),t,this.counts);
        if(s.action&&!this.products.has(s.action)){
          this.products.set(s.action,product(actions.R,actions[s.action]));this.counts.compose++;
        }
        byTime.set(t,s);
      }
      return byTime.get(t);
    }
    read(id,t,x,off=false) {
      this.counts.reads++;
      const s=this.resolve(id,t);
      if(!s.action||off)return {status:"open",deps:s.deps};
      this.counts.apply++;
      const h=apply(this.products.get(s.action),x);
      return {status:"resolved",hidden:h,terminal:h[1]>0?"positive":"negative",deps:s.deps};
    }
  }
  // Independently specified oracle: R followed by R is -I, F followed by
  // R is diag(-1,1). Terminal bytes and the second stage do not change.
  function oracle(action,x,deps) {
    if(!action)return {status:"open",deps};
    const h=[-x[0],action==="R"?-x[1]:x[1]];
    return {status:"resolved",hidden:h,terminal:h[1]>0?"positive":"negative",deps};
  }
  const times=[9,10,19,20], roots=[[11,3],[13,-4],[17,5],[19,-6]];
  const n=128;
  const eager=new World(true,times),lazy=new World(false,times);
  let checks=0,offChecks=0,forcedWrong=0,resolved=0,open=0;
  function test(r,t,action,deps) {
    for(const x of roots) {
      const a=eager.read(r.id,t,x),b=lazy.read(r.id,t,x),expected=oracle(action,x,deps);
      assert(eq(a,b)&&eq(a,expected),"paired receipt mismatch");
      checks++;
      if(action) {
        resolved++;
        if(oracle("R",x,deps).terminal!==expected.terminal)forcedWrong++;
        assert(eager.read(r.id,t,x,true).status==="open","operator-off did not open");
        offChecks++;
      } else open++;
    }
  }
  for(let i=0;i<n;i++) { const r=region(i);eager.install(r);lazy.install(r); }
  for(let i=0;i<n;i++) {
    const r=region(i),base=i%2?"F":"R",alt=i%2?"R":"F",good=order(`order:${i}:v1`),bad=order(`untrusted:${i}:v1`,"visitor");
    const scenarios=[
      [r,base,base,[r.base.id],[r.base.id]],
      [{...r,orders:[bad]},base,base,[r.base.id],[r.base.id]],
      [{...r,orders:[good]},base,alt,[r.base.id],[good.id,r.alternative.id]],
      [{...r,orders:[good],alternative:null},base,null,[r.base.id],[good.id]],
      [{...r,orders:[good,order(`rival:${i}:v1`)]},base,null,[r.base.id],[good.id,`rival:${i}:v1`]],
      [{...r,base:null},null,null,[],[]],
      [r,base,base,[r.base.id],[r.base.id]],
    ];
    for(const [state,outside,inside,outdeps,indeps] of scenarios) {
      // The peer is a geometry twin with independent source provenance.
      const peer=i^1,peerBefore=eager.read(peer,10,roots[0]);
      eager.install(state);lazy.install(state);
      for(const t of times)test(state,t,t>=10&&t<20?inside:outside,t>=10&&t<20?indeps:outdeps);
      assert(eq(peerBefore,eager.read(peer,10,roots[0])),"unrelated region changed");
      assert(eq(peerBefore,lazy.read(peer,10,roots[0])),"unrelated join changed");
    }
  }
  // Separate sparse workload: same revisions, same declared context,
  // no closure/accuracy advantage; lazy selection does not scan cold regions.
  const E=new World(true,times),L=new World(false,times);
  for(let i=0;i<n;i++){const r=region(i);E.install(r);L.install(r);}
  for(let i=0;i<n;i++){const r={...region(i),orders:[order(`order:${i}:v2`)]};E.install(r);L.install(r);}
  for(const i of [0,1,6,7])for(const t of times)for(const x of roots)
    assert(eq(E.read(i,t,x),L.read(i,t,x)),"sparse mismatch");
  assert(L.counts.select<E.counts.select,"lazy did not avoid cold selection");
  return {regions:n,checks,resolved,open,forcedWrong,offChecks,sparse:{eager:E.counts,lazy:L.counts}};
}
console.log(JSON.stringify(run(), null, 2));
