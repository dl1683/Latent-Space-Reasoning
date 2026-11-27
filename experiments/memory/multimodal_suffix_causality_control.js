/*
26 November 2026 — Mathematical foundation / mixed-modal causal diagnosis.
Reviewed 4fb674adb48a903e8bbe62607b8e1e73d70d38f5; responds to issue #6
comment 7209040198. Run: node experiments/memory/multimodal_suffix_causality_control.js

The existing test_binary_media_supplies_a_necessary_bridge_to_the_same_reader
uses an image to connect L01 to L02 or L03, and ordinary name rows L02->L07
and L03->L11. The deterministic _CausalControl has a route direction in the
first two coordinate blocks and terminal labels in the third. A later
ordinary observation can supply the terminal after a derived value routes
the reader. native_latent_recurrence._continue reads ordinary observations
at every step; value withholding fixes selection only at the first matched
ready step, not the later observation path.

Thus a predicted alternate answer under a directional value substitution
can establish content-sensitive influence without establishing that the
terminal consequence was already computed and retained. This is not a
defect in the narrower claim that an acquired relation is useful. It is
an important distinction when deciding whether we have a reusable latent
computation or merely a pointer to a later ordinary read.

This independent finite mixed-modal control has actual P6 image byte edits,
text source clauses, a source-only row-wise discovery worker, a separate
column-wise checker, withdrawal/restoration, serialized snapshots reopened
in fresh objects, two late fields per origin and an unchanged-region control.
It deliberately supplies a grammar and exact image interpretation.
It is NOT an integrated native-reader test, independent held-task panel,
OS-process restart, model inference, or evidence of learned acquisition.
No evaluator gold or pretrained model was used. The warm exact comparator
may materialize the same complete join, and the source-aware comparator
receives the same text and image; both can retain/cache the parsed inputs.
No economics claim: this is an identifiability witness, not a timed or
matched-compute trial. Source-aware here is a deterministic parser, not a
qualified text model. The checker is an independent algorithm over the
same evidence, not an independent source of truth.

The route-only and terminal readers both pass ordinary pin/revision,
omission/restoration and direction-specific swap checks. The alternate
value alone changes the route-only reader's suffix observation. Holding
that suffix to the intact observation (or withholding the terminal
observation with appropriate abstention scoring) separates the two
mechanisms in the constructed case. Both routes remain valuable if they
save full cost versus the warm exact control; do not require terminal
storage for a claim of useful link maintenance.

For the real candidate: record the full post-intervention source and
derived selection traces, actual input precision and timing. First run
normal rerouting; classify whether directional consequences are mediated
by new ordinary evidence. Only if claiming stored terminal computation,
run a read-only matched suffix-source clamp/terminal-source omission
diagnostic, with unrelated perturbations, and report off-manifold and
nonselection failures separately. Do not treat a mixed-pin or clamped
read as a valid answer/citation. Preserve the existing registered
comparison and ordinary competence gate; this is a later separately
authorized mechanism diagnostic, not a reason to block the held test.

Executed counts (exhaustive relabelings of one constructed family, not
independent questions or observed native model rates):
{
  "revisionPairs": 552,
  "changedLateNeeds": 3456,
  "unchangedLateNeeds": 960,
  "ordinaryParity": 4416,
  "omissionAndRestoration": 4416,
  "actualRevisionAndHistory": 4416,
  "directionalSwapBoth": 3456,
  "suffixClampSeparates": 3456,
  "unchangedSwapParity": 960,
  "checkerRejectsTamper": 552
}
*/
function mixedModalCausalControl() {
  // A P6 4x4 wiring diagram: one white pixel per row/column. Rows are inputs,
  // columns outputs. The image is genuine raw PPM; semantics are supplied.
  const N=4, header=[80,54,10,52,32,52,10,50,53,53,10], range=[0,1,2,3];
  const assert=(c,msg)=>{if(!c)throw Error(msg)};
  const perms=a=>a.length? a.flatMap((v,i)=>perms(a.filter((_,j)=>i!==j)).map(t=>[v,...t])):[[]];
  const image=p=>header.concat(range.flatMap(r=>range.flatMap(c=>Array(3).fill(p[r]===c?255:0))));
  const text="origin A row 0\norigin B row 1\norigin C row 2\norigin D row 3\n"+range.map(i=>`port ${i} duty D${i} zone Z${(i+1)%4}`).join("\n");
  const parseText=t=>{
    const origins={}, ports={};
    for(const l of t.split("\n")){const a=l.split(" ");if(a[0]==="origin")origins[a[1]]=Number(a[3]);else ports[Number(a[1])]=[a[3],a[5]]}
    return {origins,ports};
  };
  const pixel=(raw,r,c)=>raw[header.length+3*(r*N+c)];
  function discover(raw,t) {
    if(raw===null)return {};
    assert(raw.length===header.length+3*N*N && header.every((v,i)=>raw[i]===v),"image format");
    const {origins,ports}=parseText(t), out={};
    for(const [a,r] of Object.entries(origins)){
      const hits=range.filter(c=>pixel(raw,r,c)===255);
      assert(hits.length===1,"ambiguous wire");
      const c=hits[0];out[a]={route:c,terminal:ports[c].slice()};
    }return out;
  }
  // Second implementation enumerates columns first and binds the text via
  // direct string lookup, independently of the worker's row-wise parser.
  function check(raw,t,proposal) {
    if(raw===null)return Object.keys(proposal).length===0;
    let expected={};
    for(const c of range){
      const hits=range.filter(r=>pixel(raw,r,c)===255);
      if(hits.length!==1)return false;
      const r=hits[0], origin=String.fromCharCode(65+r);
      if(!t.split("\n").includes(`origin ${origin} row ${r}`))return false;
      const record=t.split("\n").find(l=>l.startsWith(`port ${c} `)).split(" ");
      expected[origin]={route:c,terminal:[record[3],record[5]]};
    }
    return Object.keys(expected).length===Object.keys(proposal).length&&Object.entries(expected).every(([a,v])=>JSON.stringify(v)===JSON.stringify(proposal[a]));
  }
  const publish=(revision,raw,proposal)=>{
    assert(check(raw,text,proposal),"checker refused");
    return JSON.stringify({revision,source:{text,image:raw},proposal});
  };
  // In the two implementations, the first ready selected contribution is
  // represented by one hot route or by two terminal labels. Conceptually
  // both are equal-norm alternatives within their respective representations.
  // The route-only implementation obtains answer content from a later
  // ordinary observation. The terminal implementation consumes it directly.
  function read(task,world,mode,{alternate=null,omit=false,clamp=false}={}) {
    if(omit||!world||!world.proposal[task.origin])return {answer:null,suffix:null};
    const current=world.proposal[task.origin], value=alternate||current;
    const {ports}=parseText(world.source.text);
    const suffix=mode==="route"&&!clamp?value.route:current.route;
    return {answer:mode==="route"?ports[suffix][task.field]:value.terminal[task.field],suffix:mode==="route"?suffix:null};
  }
  const exact=(task,world)=>world.proposal[task.origin]?.terminal[task.field]??null;
  const sourceAware=(task,world)=>{
    if(world.source.image===null)return null;
    const {origins,ports}=parseText(world.source.text);
    return ports[range.find(c=>pixel(world.source.image,origins[task.origin],c)===255)][task.field];
  };
  let counts={revisionPairs:0,changedLateNeeds:0,unchangedLateNeeds:0,ordinaryParity:0,omissionAndRestoration:0,actualRevisionAndHistory:0,directionalSwapBoth:0,suffixClampSeparates:0,unchangedSwapParity:0,checkerRejectsTamper:0};
  const all=perms(range);
  for(const p of all)for(const q of all){
    if(p.every((x,i)=>x===q[i]))continue;
    const rawP=image(p),rawQ=image(q);
    const initial=discover(rawP,text),edited=discover(rawQ,text);
    const pin0=publish(0,rawP,initial),pin1=publish(1,rawQ,edited);
    const pin2=publish(2,null,discover(null,text)),pin3=publish(3,rawP,discover(rawP,text));
    const tamper=JSON.parse(JSON.stringify(initial));tamper.A.route=(tamper.A.route+1)%N;
    assert(!check(rawP,text,tamper),"tamper admitted");counts.checkerRejectsTamper++;
    // Tasks are enumerated only after publications. The fixed finite family
    // is not an independently authored or held semantic benchmark.
    const tasks=Object.keys(initial).flatMap(origin=>[0,1].map(field=>({origin,field})));
    const old=JSON.parse(pin0),now=JSON.parse(pin1),missing=JSON.parse(pin2),restored=JSON.parse(pin3);
    counts.revisionPairs++;
    for(const task of tasks){
      const a=sourceAware(task,old),b=sourceAware(task,now),alt=now.proposal[task.origin];
      assert(a===exact(task,old)&&b===exact(task,now),"warm exact mismatch");
      for(const mode of ["route","terminal"]){
        assert(read(task,old,mode).answer===a&&read(task,now,mode).answer===b,"ordinary mismatch");
        assert(read(task,old,mode,{omit:true}).answer===null&&read(task,old,mode).answer===a,"omission/restoration");
        assert(read(task,missing,mode).answer===null&&sourceAware(task,missing)===null&&read(task,restored,mode).answer===a&&read(task,JSON.parse(pin0),mode).answer===a,"revision/history");
      }
      counts.ordinaryParity++;counts.omissionAndRestoration++;counts.actualRevisionAndHistory++;
      const r=read(task,old,"route",{alternate:alt}), t=read(task,old,"terminal",{alternate:alt});
      assert(r.answer===b&&t.answer===b,"directional intervention");
      if(a===b){counts.unchangedLateNeeds++;assert(r.suffix===old.proposal[task.origin].route,"irrelevant route changed");counts.unchangedSwapParity++;continue}
      counts.changedLateNeeds++;counts.directionalSwapBoth++;
      const rc=read(task,old,"route",{alternate:alt,clamp:true}),tc=read(task,old,"terminal",{alternate:alt,clamp:true});
      assert(r.suffix!==rc.suffix&&rc.answer===a&&tc.answer===b,"suffix clamp did not separate");
      counts.suffixClampSeparates++;
    }
  }
  return counts;
}

console.log(JSON.stringify(mixedModalCausalControl(), null, 2));
