// Visible source-acquisition and nonlinear order control, 2026-11-26.
// Run: node experiments/memory/visible_order_revision_control.js
// Classification: mathematical foundation + visible calibration fixture.
// Response to issue #6 comment 7193268396, upstream 403e0761.
// Independently authored from the existing supplied-coordinate fixture, but
// the same author wrote the prose, restricted parser, and manual oracle
// labels. NOT a blind semantic evaluation, model run, or held panel.
// The experiment is a proposal for comparing reused pretrained feedback,
// recurrent associative state, and an equally informed warm interpreter.
// All arms must get these exact source revisions and authority registry;
// freeze acquisition before task selection, score independent semantic
// adjudication in a subsequent run, and keep the held panel closed.
// The parser here is intentionally a narrow transparent reference, not a
// learned extractor. It recognizes exact wording, refuses a qualified
// installation notice, and ignores an explicitly untrusted crew note.
// The key counterexample: a+b=s>0 does imply a positive sum after tanh in
// real arithmetic, but cannot establish ANY uniform positive admission
// margin without bounding nuisance. With |a|,|b|<=2, |s|>=1/2 gives a
// lower bound >1/32. A bulletin moving an existing limiter from Birch to
// Alder changes that consequence before any command is selected. An
// independent trim then identifies an individual channel. A correction
// flips the aggregate; withdrawal and same-byte restoration exercise scope.
// No general floating-point error guarantee: saturated tanh can give zero
// for positive real sums. Use a certified backend/roundoff bound before
// numerical admission. The finite samples are checks, not the proof.
// Version counters + literal strings are only a stand-in for actual journal
// pins and digests. The direct arm has the same parsed law, not an oracle
// answer injected into a model. No physical memory/time or economic claim.
// Expected output on this version: {"parse":14,"eager":9,"lazy":2,"reads":45,"signs":45,"margin":24,"channels":8,"off":45,"history":9,"sample":790,"denseLazy":9,"sparseLazy":2}

function run(){
  const assert=(v,m)=>{if(!v)throw Error(m)};
  const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
  // Visible author-written prose, not independently graded or held out.
  // The authority registry is supplied; documents do not grant themselves authority.
  const initial={
    cal:"Calibration record: In normal service at Alder, the sum of the two normalized channels equals the commanded offset.",
    relay:"Relay handbook: At Alder in normal service, add the hyperbolic tangents of the two normalized channels. Enable above a positive margin of 1/64; inhibit below its negative. Between those limits leave the decision open.",
    limiter:"Portable limiter specification: When approved for a station in normal service, each normalized channel is between -2 and 2 inclusive.",
    approval:"Installation bulletin: The portable limiter is approved for Birch in normal service.",
    trim:null,
    decoy:"Crew notebook: The Alder limiter is probably installed. Treat the readings as equal if the panel is quiet."
  };
  const revisedCal="Calibration record: In normal service at Alder, the sum of the two normalized channels equals the negative of the commanded offset.";
  const alderApproval="Installation bulletin: The portable limiter is approved for Alder in normal service.";
  const trim="Trim certificate: In normal service at Alder, channel A is held at zero.";
  const edits=[
    {},
    {approval:alderApproval},
    {approval:"Installation bulletin: The portable limiter may be approved for Alder in normal service."},
    {approval:alderApproval},
    {cal:revisedCal},
    {trim},
    {approval:null},
    {trim:null,cal:initial.cal},
    {approval:alderApproval}
  ];
  // Separately hand-adjudicated expected meanings per pin; not passed to parser or worker.
  const adjudicated=[
    [1,false,false],[1,true,false],[1,false,false],[1,true,false],
    [-1,true,false],[-1,true,true],[-1,false,true],[1,false,false],[1,true,false]
  ];
  const authorities=["cal","relay","limiter","approval","trim"];
  const versions={}, snapshots=[];
  let sources={};
  for(let i=0;i<edits.length;i++){
    for(const [id,raw] of Object.entries(i===0?initial:edits[i])){
      versions[id]=(versions[id]||0)+1;
      sources={...sources,[id]:{raw,generation:versions[id]}};
    }
    snapshots.push(sources);
  }
  const counts={parse:0,eager:0,lazy:0,reads:0,signs:0,margin:0,channels:0,off:0,history:0,sample:0};
  const parsed=new Map();
  function acquire(pin){
    const out={};
    for(const id of authorities){
      const rec=pin[id], key=id+":"+rec.generation;
      if(!parsed.has(key)){
        counts.parse++;
        let meaning=null, m;
        const t=rec.raw;
        if(id==="cal"&&t&&(m=/^Calibration record: In normal service at Alder, the sum of the two normalized channels equals the (negative of the )?commanded offset\.$/.exec(t)))meaning=m[1]?-1:1;
        if(id==="relay"&&t===initial.relay)meaning=true;
        if(id==="limiter"&&t===initial.limiter)meaning=2;
        if(id==="approval"&&t&&(m=/^Installation bulletin: The portable limiter is approved for (Alder|Birch) in normal service\.$/.exec(t)))meaning=m[1];
        if(id==="trim"&&t===trim)meaning=true;
        parsed.set(key,meaning);
      }
      out[id]=parsed.get(key);
    }
    return out;
  }
  function compile(pin,kind){
    counts[kind]++;
    const facts=acquire(pin);
    assert(facts.cal!==null&&facts.relay,"unrecognized governing calibration/relay");
    const bounded=facts.limiter===2&&facts.approval==="Alder";
    const fixed=facts.trim===true;
    // Coverage includes potentially eligible authorities, even a currently
    // inapplicable or absent bulletin. Positive evidence is kept separate.
    const coverage=authorities.map(id=>[id,pin[id].generation,pin[id].raw]);
    const witnesses=["cal","relay",...(bounded?["limiter","approval"]:[]),...(fixed?["trim"]:[])];
    return {k:facts.cal,bounded,fixed,coverage,witnesses};
  }
  // Real arithmetic theorem: tanh(a)+tanh(b) has sign(a+b).
  // If |a|,|b|<=2 and |a+b|>=1/2, MVT and cosh(2)<4 give
  // |sum| > 1/32. Thus the 1/64 margin is justified.
  // Without a finite bound the infimum of |sum| is zero for fixed nonzero a+b.
  // This does not promise nonzero FLOAT64 output for unbounded inputs.
  const commands=[-1,-.5,0,.5,1];
  const status=v=>v>1/64?"ENABLE":v< -1/64?"INHIBIT":"OPEN";
  function retained(pub,x){
    if(!pub)return {sign:"UNKNOWN",relay:"OPEN",a:"OPEN",b:"OPEN"};
    const s=pub.k*x;
    return {sign:Math.sign(s),relay:(pub.bounded||pub.fixed)?status(s):"OPEN",
      a:"OPEN",b:pub.fixed?status(s):"OPEN"};
  }
  // Separate analytic oracle on manually adjudicated facts: optimize
  // f(a)=tanh(a)+tanh(s-a) over the feasible interval. Its only interior
  // extremum for nonzero s is a=s/2; both infinite endpoint limits are 0.
  // Individual channels use monotonicity. At s=0 the sum is identically 0.
  function oracle([k,bounded,fixed],x){
    const s=k*x;
    let lo=bounded?Math.max(-2,s-2):-Infinity;
    let hi=bounded?Math.min(2,s+2):Infinity;
    if(fixed)lo=hi=0;
    assert(lo<=hi,"infeasible fixture");
    const f=a=>Number.isFinite(a)?Math.tanh(a)+Math.tanh(s-a):0;
    const vals=[f(lo),f(hi)];
    if(lo<=s/2&&s/2<=hi)vals.push(f(s/2));
    const classify=(l,h)=>l>1/64?"ENABLE":h< -1/64?"INHIBIT":"OPEN";
    return {sign:Math.sign(s),relay:classify(Math.min(...vals),Math.max(...vals)),
      a:classify(Math.tanh(lo),Math.tanh(hi)),
      b:classify(Math.tanh(s-hi),Math.tanh(s-lo))};
  }
  // Query-blind build: no commands are passed to this function.
  const pubs=snapshots.map(pin=>compile(pin,"eager"));
  const valid=(pub,pin)=>same(pub.coverage,authorities.map(id=>[id,pin[id].generation,pin[id].raw]));
  const cache=new Map();
  function lazy(pin,i,x){
    if(!cache.has(i))cache.set(i,compile(pin,"lazy"));
    const p=cache.get(i);
    // Exact warm comparator uses full same acquired law, not retained verdict.
    return oracle([p.k,p.bounded,p.fixed],x);
  }
  for(let i=0;i<pubs.length;i++){
    const p=pubs[i];
    assert(valid(p,snapshots[i]),"publication failed coverage");
    assert(same([p.k,p.bounded,p.fixed],adjudicated[i]),"prose acquisition disagrees with manual label");
    if(i)assert(!valid(pubs[i-1],snapshots[i]),"stale generation accepted");
    for(const x of commands){
      const expected=oracle(adjudicated[i],x);
      const answer=retained(p,x);
      assert(same(answer,expected),"retained/oracle mismatch "+i+":"+x);
      assert(same(lazy(snapshots[i],i,x),expected),"warm mismatch");
      counts.reads++; counts.signs++;
      if(answer.relay!=="OPEN")counts.margin++;
      if(answer.b!=="OPEN")counts.channels++;
      const off=retained(null,x);
      assert(off.relay==="OPEN"&&off.sign==="UNKNOWN","operator-off");
      counts.off++;
      // Sample concrete completions only as a check, not proof.
      const s=p.k*x;
      for(const a of p.fixed?[0]:p.bounded?Array.from({length:33},(_,j)=>Math.max(-2,s-2)+j*(Math.min(2,s+2)-Math.max(-2,s-2))/32):[-100,-20,-2,0,s/2,2,20,100]){
        const value=Math.tanh(a)+Math.tanh(s-a);
        if(answer.relay==="ENABLE")assert(value>1/64,"false enable");
        if(answer.relay==="INHIBIT")assert(value< -1/64,"false inhibit");
        counts.sample++;
      }
    }
  }
  assert(!valid(pubs[1],snapshots[3]),"same-byte restoration reused old receipt");
  assert(snapshots[1].approval.raw===snapshots[3].approval.raw,"not same-byte restoration");
  assert(!pubs[0].witnesses.includes("approval")&&pubs[1].witnesses.includes("approval"),"newly applicable premise not discovered");
  assert(retained(pubs[0],.5).relay==="OPEN"&&retained(pubs[1],.5).relay==="ENABLE","revision no effect");
  assert(retained(pubs[4],.5).relay==="INHIBIT","correction did not propagate");
  assert(retained(pubs[6],.5).b==="INHIBIT","individual consequence lost");
  assert(Math.tanh(100)+Math.tanh(.5-100)===0,"float saturation demonstration");
  for(let i=0;i<pubs.length;i++){
    assert(same(retained(pubs[i],.5),oracle(adjudicated[i],.5)),"historical read");
    counts.history++;
  }
  // Sparse demand: on the identical revisions only two pins are queried.
  // Acquisition is cached independently; count preparations, not latency.
  const fullLazy=counts.lazy; cache.clear();counts.lazy=0;
  for(const i of [0,8])for(const x of commands)lazy(snapshots[i],i,x);
  return {...counts,denseLazy:fullLazy,sparseLazy:counts.lazy};
}
console.log(JSON.stringify(run(),null,2));
