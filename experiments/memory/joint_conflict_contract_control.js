// Standalone typed joint-conflict action canary: node experiments/memory/joint_conflict_contract_control.js
// Fixed registered three-slot scope; simple prose grammar; binary support.
// Version counters and exact string matching stand in for source byte verification.
// No model, Python host integration, durable journal, SHA verification, authorization or timing.
// Reuse counts are pair preparations only; different operators have different costs.
function run() {
  const assert=(p,m)=>{if(!p)throw Error(m)};
  const scope=["anchor","left","late"], full=255;
  const stats={worlds:0,paired_receipts:0,closed_receipts:0,unknown_receipts:0,refused_contracts:0,causal_swaps:0,unary_twins:0,compiled_source_versions_per_arm:0,latent_pair_preparations:0,eager_exact_pair_preparations:0,lazy_exact_pair_preparations:0,verified_span_opens:0};
  const refuse=fn=>{let ok=false;try{fn()}catch(e){ok=true}assert(ok,"expected refusal");stats.refused_contracts++};
  const parse=(text,labels)=>{
    const m=/^Setting ([A-Z]+) must be (the same as|different from) setting ([A-Z]+)\.$/.exec(text);
    if(!m||!labels.includes(m[1])||!labels.includes(m[3]))throw Error("uncompiled");
    return [labels.indexOf(m[1]),labels.indexOf(m[3]),Number(m[2]==="different from")];
  };
  const prose=(a,b,p)=>"Setting "+a+" must be "+(p?"different from":"the same as")+" setting "+b+".";
  const bits=f=>{let m=0;for(let x=0;x<8;x++)if((((x>>f[0])^(x>>f[1]))&1)===f[2])m|=1<<x;return m};
  const projection=m=>{let s=0;for(let x=0;x<8;x++)if(m&(1<<x))s|=1<<(((x&1)<<1)|((x>>2)&1));return s};
  const verdict=m=>m?"compatible_in_registered_scope":"conflict";
  const pop=x=>{let n=0;while(x){n+=x&1;x>>>=1}return n};
  const unary=m=>[pop(m&3),pop(m&12),pop(m&5),pop(m&10)];
  function workspace(labels) {
    let seq=0,versions={},prior={};
    const parsed=new Map(),latentCache=new Map(),exactCache=new Map(),lazyCache=new Map();
    const key=(rows,id)=>rows[id]?id+":"+rows[id].version:id+":absent";
    function publish(changes,{complete=true,budget=true}={}) {
      const rows={...prior};
      for(const [id,text] of Object.entries(changes)){
        versions[id]=(versions[id]||0)+1;
        if(text===null)delete rows[id];else rows[id]=Object.freeze({id,text,version:versions[id]});
      }
      prior=rows;
      const fs={},masks={};
      for(const id of scope)if(rows[id]){
        const k=key(rows,id);
        if(!parsed.has(k)){
          const factor=parse(rows[id].text,labels);
          parsed.set(k,{factor,mask:bits(factor)});
          stats.compiled_source_versions_per_arm++;
        }
        const row=parsed.get(k);fs[id]=row.factor;masks[id]=row.mask;
      }
      const pairKey=key(rows,"left")+"|"+key(rows,"late");
      if(!latentCache.has(pairKey)){
        // Eliminate the middle slot before the task. Only the retained
        // separator support, not raw constraints, reaches latent readout.
        let m=0;
        for(let a=0;a<2;a++)for(let c=0;c<2;c++)for(let b=0;b<2;b++){
          const v=[a,b,c];
          if(["left","late"].every(id=>!fs[id]||(v[fs[id][0]]^v[fs[id][1]])===fs[id][2]))m|=1<<(2*a+c);
        }
        latentCache.set(pairKey,m);stats.latent_pair_preparations++;
      }
      if(!exactCache.has(pairKey)){
        exactCache.set(pairKey,(masks.left??full)&(masks.late??full));
        stats.eager_exact_pair_preparations++;
      }
      return Object.freeze({seq:++seq,rows:Object.freeze(rows),complete,budget,
        left:latentCache.get(pairKey),right:projection(masks.anchor??full),
        exact:exactCache.get(pairKey)&(masks.anchor??full),pairKey,masks:Object.freeze(masks)});
    }
    function lazy(pin) {
      if(!lazyCache.has(pin.pairKey)){
        lazyCache.set(pin.pairKey,(pin.masks.left??full)&(pin.masks.late??full));
        stats.lazy_exact_pair_preparations++;
      }
      return lazyCache.get(pin.pairKey)&(pin.masks.anchor??full);
    }
    // Independent direct-bundle oracle, explicit enumeration and no messages.
    function bundle(pin) {
      const fs=scope.filter(id=>pin.rows[id]).map(id=>parse(pin.rows[id].text,labels));
      let n=0;
      for(const a of [0,1])for(const b of [0,1])for(const c of [0,1]){
        const x=[a,b,c];if(fs.every(([i,j,p])=>(x[i]!==x[j])===Boolean(p)))n++;
      }
      return verdict(n);
    }
    return {publish,lazy,bundle};
  }
  // Host-owned scoped action. The policy cannot select latent coordinates,
  // provide a likelihood, or manufacture its own receipt.
  function episode(pin,need,hidden=pin.left){
    const reads=[],receipts=[];
    function open(id,version=pin.rows[id]?.version,text=pin.rows[id]?.text,forNeed=need){
      const current=pin.rows[id];
      if(!current||version!==current.version||text!==current.text)throw Error("stale source");
      reads.push(Object.freeze({id,version,text,need:forNeed,pin:pin.seq,ordinal:0}));
      stats.verified_span_opens++;
      return reads.length-1;
    }
    function check(request){
      if(Object.keys(request).sort().join(",")!=="need_id,ordinal,read")throw Error("unexpected action fields");
      const {need_id,read,ordinal}=request,a=reads[read];
      if(need_id!==need||!Number.isInteger(read)||!a||a.need!==need||a.id!=="anchor"||ordinal!==a.ordinal)
        throw Error("unbound anchor");
      const status=pin.complete&&pin.budget?verdict(hidden&pin.right):"unknown";
      const deps=scope.filter(id=>pin.rows[id]).map(id=>Object.freeze({id,version:pin.rows[id].version,ordinal:0}));
      const receipt=Object.freeze({pin:pin.seq,need_id,status,deps:Object.freeze(deps)});
      receipts.push(receipt);
      return {receipt:receipts.length-1,...receipt};
    }
    function submit({need_id,receipt,claim,citations,open_question}){
      const r=receipts[receipt];
      if(!Number.isInteger(receipt)||!r||r.pin!==pin.seq||r.need_id!==need_id||need_id!==need||claim!==r.status)
        throw Error("unbound conclusion");
      if(claim==="unknown"){
        if(!open_question)throw Error("missing open question");
        return true;
      }
      if(open_question)throw Error("cannot close and leave same scope open");
      if(!r.deps.every(d=>citations.some(i=>{
        const s=reads[i];return Number.isInteger(i)&&s&&s.need===need&&s.pin===pin.seq&&s.id===d.id&&s.version===d.version&&s.ordinal===d.ordinal;
      })))throw Error("missing exact same-need dependencies");
      return true;
    }
    return {open,check,submit};
  }
  const names=["ALPHA","BETA","GAMMA","DELTA","EPSILON","ZETA","ETA","THETA"];
  for(let w=0;w<64;w++){
    const labels=[names[w%8],names[(w+1)%8],names[(w+3+Math.floor(w/8)%4)%8]];
    const [a,b,c]=labels,p=w&1,q=(w>>1)&1,r=p^q,need="joint-"+w;
    const world=workspace(labels);
    const start=world.publish({anchor:prose(a,c,r),left:prose(a,b,p),late:prose(b,c,q),
      unrelated:"Other equipment maintenance record."});
    const conflict=world.publish({late:prose(b,c,1-q)});
    assert(JSON.stringify(unary(start.left))===JSON.stringify(unary(conflict.left))&&start.left!==conflict.left,"unary twins");
    stats.unary_twins++;
    const unrelated=world.publish({unrelated:"Other equipment maintenance record revised."});
    const withdrawn=world.publish({late:null});
    const restored=world.publish({late:prose(b,c,1-q)});
    const limited=world.publish({}, {budget:false});
    const incomplete=world.publish({}, {complete:false});
    world.publish({late:prose(b,c,q)}); // never queried: lazy arm skips work
    const snapshots=[start,conflict,unrelated,withdrawn,restored,start,conflict,limited,incomplete];
    const expected=["compatible_in_registered_scope","conflict","conflict","compatible_in_registered_scope","conflict","compatible_in_registered_scope","conflict","unknown","unknown"];
    for(let j=0;j<snapshots.length;j++){
      const pin=snapshots[j],ep=episode(pin,need),root=ep.open("anchor");
      const rec=ep.check({need_id:need,read:root,ordinal:0});
      assert(rec.status===expected[j],"revision expected");
      if(rec.status!=="unknown"){
        assert(world.bundle(pin)===rec.status&&verdict(pin.exact)===rec.status&&verdict(world.lazy(pin))===rec.status,"all exact arms");
        stats.closed_receipts++;
      }else stats.unknown_receipts++;
      assert(!rec.deps.some(d=>d.id==="unrelated"),"unrelated not dependency");
      const citations=[root,...rec.deps.filter(d=>d.id!=="anchor").map(d=>ep.open(d.id))];
      const report={need_id:need,receipt:rec.receipt,claim:rec.status,citations,open_question:rec.status==="unknown"};
      assert(ep.submit(report),"valid receipt");stats.paired_receipts++;
      refuse(()=>ep.check({need_id:need,read:root,ordinal:0,weights:[0,1]}));
      if(rec.status==="unknown"){
        refuse(()=>ep.submit({...report,open_question:false}));
        refuse(()=>ep.submit({...report,claim:"conflict"}));
      }else{
        refuse(()=>ep.submit({...report,citations:[root]}));
        refuse(()=>ep.submit({...report,need_id:"wrong"}));
        refuse(()=>ep.submit({...report,claim:rec.status==="conflict"?"compatible_in_registered_scope":"conflict"}));
      }
      if(j===4){
        refuse(()=>ep.open("late",conflict.rows.late.version,conflict.rows.late.text));
        const other=ep.open("late",undefined,undefined,"other-need");
        refuse(()=>ep.submit({...report,citations:citations.slice(0,-1).concat(other)}));
        refuse(()=>ep.check({need_id:need,read:root,ordinal:1}));
        refuse(()=>ep.open("late",restored.rows.late.version,"wrong bytes"));
      }
    }
    const altered=episode(conflict,need,start.left);
    const rec=altered.check({need_id:need,read:altered.open("anchor"),ordinal:0});
    assert(rec.status==="compatible_in_registered_scope"&&world.bundle(conflict)==="conflict","causal swap");
    const recovered=episode(conflict,need);
    assert(recovered.check({need_id:need,read:recovered.open("anchor"),ordinal:0}).status==="conflict","restore");
    // The provenance gate alone cannot detect a corrupted hidden computation.
    // The intervention is diagnostic, never a production receipt.
    stats.causal_swaps++;stats.worlds++;
  }
  return stats;
}
console.log(JSON.stringify(run(),null,2));
