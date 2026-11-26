// Source-scoped support rails and history routing control, 2026-11-26.
// Run: node experiments/memory/scoped_history_rails_control.js
// Standalone supplied binary facts; not integrated Python or trained latent inference.
// Scope and hard/unknown/reference authority are external; no parser, source-byte
// verifier, journal SHA, model run, or hidden panel. Groups use a convenience
// eager scan; count group rebuilds and matching rows, NOT actual scans/time.
// The exact comparator has its own indexed lazy group caches; no cold replay.
// Generated history changes scheduling only; a wrong eligible-order hint cannot
// add evidence or close a partially visited route set. Scoped gaps are sound only
// relative to a complete, correct registry and scope recognizer. Wrong scope
// recognition remains an acquisition problem. Historical maps are in-memory pins.
// Two rails are max/OR support witnesses, not posterior probabilities; opposed
// support and incomplete coverage are independent. No claim of latent advantage.

function run() {
  // Standalone, deterministic control; no held tasks, no learned model.
  const N=128;
  let seed=0x17c0ffee;
  const rng=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed;};
  const check=(v,msg)=>{if(!v)throw Error(msg);};
  const eq=(a,b,msg)=>check(JSON.stringify(a)===JSON.stringify(b),msg+":"+JSON.stringify([a,b]));
  const k=x=>x.kind+":"+x.scope;
  const verdict=(p,n,g)=>[+!!p,+!!n,+!!g];
  const stats={worlds:N,fullChecks:0,historyChecks:0,budgetChecks:0,oldPinChecks:0,
    wrongRouteFalseClosures:0,meanCollisionWorlds:0,unrelatedGapChecks:0,
    eagerGroupRebuilds:0,lazyGroupRebuilds:0,eagerRowsScanned:0,lazyRowsScanned:0,
    railRouteReads:0,fixedRouteReads:0};
  // Shared input facts: controlled origin routing and scoped relay observations.
  // The authority labels and candidate identity universe are supplied externally.
  // The two numeric rails record EXISTENCE, not confidence or proof of truth.
  function rails(rows,kind,routeNames) {
    const hard=rows.filter(x=>x.authority==="hard");
    const gap=rows.some(x=>x.authority==="unknown");
    if(kind==="origin")return {a:routeNames.map(v=>+hard.some(x=>x.route===v)),gap:gap||!hard.length};
    return {a:[+hard.some(x=>x.sign===1),+hard.some(x=>x.sign===-1)],gap:gap||!hard.length};
  }
  // Raw, independent exact enumerator, scanning all eligible rows.
  function oracle(rows,q) {
    const routes=new Set();let p=false,n=false,g=false;
    for(const x of rows.values())if(x.kind==="origin"&&x.scope===q){
      if(x.authority==="unknown")g=true;
      if(x.authority==="hard")routes.add(x.route);
    }
    if(!routes.size)g=true;
    for(const route of routes){
      let found=false;
      for(const x of rows.values())if(x.kind==="relay"&&x.scope===route){
        if(x.authority==="unknown")g=true;
        if(x.authority==="hard"){found=true;if(x.sign===1)p=true;else n=true;}
      }
      if(!found)g=true;
    }
    return verdict(p,n,g);
  }
  // Eager numeric bank: only the groups affected by a source event rebuild.
  // Keep immutable snapshot maps, with untouched group objects shared.
  function revise(bank,rows,changes,names) {
    const next=new Map(bank), affected=new Set();
    for(const [id,x] of changes){const old=rows.get(id);if(old)affected.add(k(old));
      if(x){rows.set(id,x);affected.add(k(x));}else rows.delete(id);}
    for(const key of affected){
      const group=[...rows.values()].filter(x=>k(x)===key);
      stats.eagerGroupRebuilds++;stats.eagerRowsScanned+=group.length;
      next.set(key,rails(group,key.split(":")[0],names));
    }
    return next;
  }
  function read(bank,q,names,hint=null,budget=Infinity,count=true){
    const origin=bank.get("origin:"+q)||{a:names.map(()=>0),gap:true};
    const eligible=names.filter((_,i)=>origin.a[i]===1);
    // Generated history is a scheduling HINT, not a source of eligibility.
    if(eligible.includes(hint)){eligible.splice(eligible.indexOf(hint),1);eligible.unshift(hint);}
    let p=0,n=0,g=origin.gap||!eligible.length;
    for(const route of eligible.slice(0,budget)){
      const relay=bank.get("relay:"+route)||{a:[0,0],gap:true};
      p=Math.max(p,relay.a[0]);n=Math.max(n,relay.a[1]);g=g||relay.gap;
      if(count)stats.railRouteReads++;
    }
    return verdict(p,n,g||eligible.length>budget);
  }
  // Independent strong baseline: incrementally invalidate indexed group caches.
  // The query's fixed-width accumulator is a two-bit mask plus one gap bit;
  // no history rows. Group caches retain the same supplied facts and scopes.
  function exactIndex(){
    const rows=new Map(),groups=new Map(),cache=new Map();
    function apply(changes){
      for(const [id,x] of changes){
        const old=rows.get(id);
        if(old){groups.get(k(old)).delete(id);cache.delete(k(old));}
        if(x){rows.set(id,x);if(!groups.has(k(x)))groups.set(k(x),new Map());groups.get(k(x)).set(id,x);cache.delete(k(x));}
        else rows.delete(id);
      }
    }
    function get(key){
      if(cache.has(key))return cache.get(key);
      const group=groups.get(key)||new Map(),routes=new Set();
      let mask=0,gap=false,hard=0;
      stats.lazyGroupRebuilds++;stats.lazyRowsScanned+=group.size;
      for(const x of group.values()){
        if(x.authority==="unknown")gap=true;
        if(x.authority==="hard"){hard++;if(x.kind==="origin")routes.add(x.route);else mask|=x.sign===1?1:2;}
      }
      const value={routes,mask,gap:gap||!hard};cache.set(key,value);return value;
    }
    function query(q){
      const o=get("origin:"+q);let mask=0,g=o.gap;
      for(const route of o.routes){const b=get("relay:"+route);mask|=b.mask;g=g||b.gap;stats.fixedRouteReads++;}
      return verdict(mask&1,mask&2,g);
    }
    return {apply,query};
  }
  for(let w=0;w<N;w++){
    const names=[0,1,2,3].map(i=>w*20+i+1);
    for(let i=names.length-1;i>0;i--){const j=rng()%(i+1);[names[i],names[j]]=[names[j],names[i]];}
    const [a,b,c,d]=names,q=w*20+12,q2=w*20+13,sg=(rng()>>>29)&1?1:-1;
    const o=(scope,route,authority="hard")=>({kind:"origin",scope,route,authority});
    const t=(scope,sign,authority="hard")=>({kind:"relay",scope,sign,authority});
    const rows=new Map(),index=exactIndex();let bank=new Map();
    const edit=changes=>{bank=revise(bank,rows,changes,names);index.apply(changes);};
    const expect=(p,n,g)=>verdict(p,n,g);
    const single=expect(sg===1,sg===-1,false), opposite=expect(sg===-1,sg===1,false);
    function test(expected,tag){
      const raw=oracle(rows,q);eq(raw,expected,tag+" oracle");
      eq(read(bank,q,names),raw,tag+" rails");eq(index.query(q),raw,tag+" fixed");
      stats.fullChecks++;
      for(const hint of [...names,999999,null]){
        eq(read(bank,q,names,hint,Infinity,false),raw,tag+" history");
        stats.historyChecks++;
      }
      for(const budget of [0,1]){
        const partial=read(bank,q,names,b,budget,false);
        check(partial[0]<=raw[0]&&partial[1]<=raw[1],tag+" fabricated support");
        if(!partial[2])eq(partial,raw,tag+" premature closure");
        stats.budgetChecks++;
      }
    }
    edit([["root",o(q,a)],["real",t(a,sg)],["decoy",t(b,-sg)],["noise",t(c,1)],
      ["otherRoot",o(q2,d)],["other",t(d,-1)]]);
    const historical=bank;test(single,"opening");
    // Separate source and history channels alone do not stop a wrong generated route:
    // the unsafe reader consults an entirely real, but inapplicable, decoy relay.
    const wrong=bank.get("relay:"+b);
    eq(verdict(...wrong.a,wrong.gap),opposite,"wrong route");
    stats.wrongRouteFalseClosures++;
    edit([["rival",t(a,-sg)]]);test(expect(1,1,0),"competing hard authority");
    const conflictMean=(sg+(-sg))/2;
    edit([["rival",null],["real",null]]);test(expect(0,0,1),"withdrawal");
    const missingMean=0;check(conflictMean===missingMean,"pooled collision");stats.meanCollisionWorlds++;
    edit([["rival",t(a,-sg,"reference")]]);test(expect(0,0,1),"reference is not hard");
    edit([["real",t(a,sg)]]);test(single,"restored with reference");
    edit([["qualified",t(a,-sg,"unknown")]]);test(expect(sg===1,sg===-1,1),"relevant authority gap");
    edit([["qualified",null],["unrelatedUnknown",t(c,-1,"unknown")]]);
    test(single,"unrelated authority gap");stats.unrelatedGapChecks++;
    edit([["root2",o(q,b)]]);test(expect(1,1,0),"second eligible route");
    edit([["decoy",null]]);test(expect(sg===1,sg===-1,1),"missing second premise");
    edit([["root2",null],["unknownRoot",o(q,b,"unknown")]]);test(expect(sg===1,sg===-1,1),"unresolved origin scope");
    edit([["unknownRoot",null],["root",o(q,b)],["decoy",t(b,-sg)]]);test(opposite,"source changes relay");
    edit([["root",o(q,a)]]);test(single,"same content restoration");
    eq(read(historical,q,names,b,Infinity,false),single,"historical snapshot");stats.oldPinChecks++;
    // Final unqueried change illustrates workload-sensitive eager preparation cost.
    edit([["other",t(d,1)]]);
  }
  return stats;
}

console.log(JSON.stringify(run(), null, 2));
