/*
Whole-loop information boundary negative controls (visible, synthetic).
Run: node experiments/memory/whole_loop_boundary_control.js
No learned model, real journal, hidden vectors, or performance claim.
The answer-table arm is a legitimate conventional control, not a leak.
See Issue #6 for interpretation and admission requirements.
*/
function control() {
  const assert = (v, msg) => { if (!v) throw new Error(msg); };
  let state = 0x87a55b32;
  const random = n => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) % n; };
  const shuffle = a => { a = [...a]; for (let i=a.length-1;i>0;i--) {const j=random(i+1); [a[i],a[j]]=[a[j],a[i]];} return a; };
  // Independent oracle: enumerate all valid three-edge sequences, not the indexed transition.
  function oracle(rows, start, first=0) {
    let paths = [{at:start, ids:[]}];
    for(let s=first;s<3;s++) paths=paths.flatMap(p=>rows.filter(r=>r.stage===s&&r.from===p.at).map(r=>({at:r.to,ids:[...p.ids,r.id]})));
    assert(paths.length<=1,"fixture must be functional");
    return paths.length ? {answer: paths[0].at%2 ? "opposed":"supported",route:paths[0].ids} : {answer:"open",route:[]};
  }
  const freezeRows = rows => Object.freeze(rows.map(r=>Object.freeze({...r})));
  // A deterministic source-only publication. No supplied coordinates or route, no task argument.
  const publish = rows => Object.freeze(Object.fromEntries(rows.map(r=>[`${r.stage}:${r.from}`,r])));
  const serial = p => JSON.stringify(p);
  function reason(pub, start, intervention, hint) {
    let at=start, writes=[];
    for(let stage=0;stage<3;stage++){
      if(intervention&&intervention.stage===stage) at=intervention.at;
      // A malicious supplied-route arm fetches the next row by the hint instead
      // of its current state. Off/restoration still work, but the state is irrelevant.
      const row=pub&&(hint?Object.values(pub).find(r=>r.id===hint[stage]):pub[`${stage}:${at}`]);
      if(!row) return {answer:"open",writes};
      at=row.to; writes.push({stage,at,id:row.id,generation:row.generation});
    }
    return {answer:at%2?"opposed":"supported",writes};
  }
  // Evaluation boundary: task input is not a path, and final verification
  // may reject a proposal but must not replace it with its own answer.
  function taskInput(t) {
    assert(Object.keys(t).length===1&&Object.hasOwn(t,"start")&&Number.isInteger(t.start)&&t.start>=0&&t.start<4,"task input carries extra information");
    return Object.freeze({start:t.start});
  }
  function checkOnly(proposal, gold) {
    return proposal.answer===gold ? proposal.answer : "open";
  }
  const out={worlds:64,pins:0,lateReads:0,resolved:0,sourceOnlyCorrect:0,routeLeakCorrect:0,taskVisibleCorrect:0,oracleGateCorrect:0,answerTableCorrect:0,sourceOnlyCausal:0,routeLeakCausal:0,taskVisibleCausal:0,oracleGateCausal:0,answerTableCausal:0,routeFieldsRejected:0,pairedPublicationDifferences:0,sourceOnlyPublicationDifferences:0,rawOracleGateResolved:0,checkOnlyResolved:0,midpointTrials:0,sourceOnlyMidpointCorrect:0,routeLeakMidpointCorrect:0,historyReads:0,changedAnswers:0,withdrawnAnswers:0,restoredAnswers:0};
  for(let w=0;w<out.worlds;w++){
    let rows=[];
    for(let stage=0;stage<3;stage++){
      const targets=shuffle([0,1,2,3]), ids=shuffle([0,1,2,3]);
      for(let from=0;from<4;from++)rows.push({stage,from,to:targets[from],id:`w${w}-s${stage}-r${ids[from]}`,generation:1});
    }
    const old=freezeRows(shuffle(rows));
    const other=[1,2,3].find(x=>oracle(old,x).answer!==oracle(old,0).answer);
    assert(other!==undefined,"opposite donor");
    const a=old.find(r=>r.stage===0&&r.from===0), b=old.find(r=>r.stage===0&&r.from===other);
    const edit=freezeRows(old.map(r=>r.id===a.id?{...r,to:b.to,generation:2}:r.id===b.id?{...r,to:a.to,generation:2}:r));
    const missing=edit.find(r=>r.stage===1&&r.from===b.to);
    const withdrawn=freezeRows(edit.filter(r=>r.id!==missing.id));
    const restored=freezeRows(shuffle([...withdrawn,{...missing,generation:4}]));
    assert(restored.find(r=>r.id===missing.id).generation!==missing.generation,"fresh restoration generation");
    const oldPub=publish(old), oldBytes=serial(oldPub);
    for(const [pin,src] of [old,edit,withdrawn,restored].entries()){
      // All source-only publishing completes before the four late tasks are selected.
      const pub=publish(src), before=serial(pub);
      const table=Object.freeze(Object.fromEntries([0,1,2,3].map(x=>[x,reason(pub,x).answer])));
      const order=shuffle([0,1,2,3]);
      out.pins++;
      const visiblePublications=[];
      for(const x of order){
        const t=taskInput({start:x}), g=oracle(src,x);
        const route=Object.freeze(g.route);
        const acquired=reason(pub,t.start);
        const leaked=reason(pub,t.start,null,route);
        const taskPub=publish(src.filter(r=>route.includes(r.id)));
        visiblePublications.push(serial(taskPub));
        const visible=reason(taskPub,t.start);
        const raw={answer:"open"};
        const gateFilled=pub?g.answer:raw.answer; // intentionally invalid oracle-style postprocessor
        const tableRead=table[t.start];
        out.lateReads++;
        out.sourceOnlyCorrect+=acquired.answer===g.answer;
        out.routeLeakCorrect+=leaked.answer===g.answer;
        out.taskVisibleCorrect+=visible.answer===g.answer;
        out.oracleGateCorrect+=gateFilled===g.answer;
        out.answerTableCorrect+=tableRead===g.answer;
        const off=reason(null,t.start).answer;
        if(g.answer!=="open"){
          out.resolved++;
          out.sourceOnlyCausal+=acquired.answer===g.answer&&off==="open"&&reason(pub,t.start).answer===g.answer;
          out.routeLeakCausal+=leaked.answer===g.answer&&off==="open"&&reason(pub,t.start,null,route).answer===g.answer;
          out.taskVisibleCausal+=visible.answer===g.answer&&off==="open"&&reason(taskPub,t.start).answer===g.answer;
          out.oracleGateCausal+=gateFilled===g.answer&&raw.answer==="open"&&g.answer===gateFilled;
          out.answerTableCausal+=tableRead===g.answer&&off==="open"&&table[t.start]===g.answer;
        }
        out.rawOracleGateResolved+=raw.answer!=="open";
        out.checkOnlyResolved+=checkOnly(raw,g.answer)!=="open";
        try{taskInput({start:x,route});}catch(e){out.routeFieldsRejected++;}
        out.sourceOnlyPublicationDifferences+=serial(pub)!==before;
        if(pin===1&&g.answer!==oracle(old,x).answer)out.changedAnswers++;
        if(pin===2&&g.answer!==oracle(edit,x).answer)out.withdrawnAnswers++;
        if(pin===3&&g.answer!==oracle(withdrawn,x).answer)out.restoredAnswers++;
        if(pin===0||pin===1||pin===3){
          const donor=[0,1,2,3].find(y=>oracle(src,y).answer!==g.answer);
          const at=reason(pub,donor).writes[0].at;
          const target=oracle(src,at,1).answer;
          assert(target!==g.answer,"answer-changing same-pin donor");
          const intervention={stage:1,at};
          out.midpointTrials++;
          out.sourceOnlyMidpointCorrect+=reason(pub,t.start,intervention).answer===target;
          out.routeLeakMidpointCorrect+=reason(pub,t.start,intervention,route).answer===target;
        }
      }
      // Two different tasks at the same source pin must not change an alleged
      // pre-task publication. This detects the explicit task-visible worker,
      // not a malicious worker whose task leak happens to be output-invariant.
      for(let i=1;i<visiblePublications.length;i++)out.pairedPublicationDifferences+=visiblePublications[i]!==visiblePublications[0];
    }
    assert(serial(oldPub)===oldBytes,"history mutated");
    for(let x=0;x<4;x++){
      assert(reason(oldPub,x).answer===oracle(old,x).answer,"historical answer");
      out.historyReads++;
    }
  }
  for(const k of ["sourceOnlyCorrect","routeLeakCorrect","taskVisibleCorrect","oracleGateCorrect","answerTableCorrect","routeFieldsRejected"])assert(out[k]===out.lateReads,k);
  for(const k of ["sourceOnlyCausal","routeLeakCausal","taskVisibleCausal","oracleGateCausal","answerTableCausal"])assert(out[k]===out.resolved,k);
  assert(out.sourceOnlyPublicationDifferences===0&&out.pairedPublicationDifferences===out.pins*3,"publication boundary");
  assert(out.rawOracleGateResolved===0&&out.checkOnlyResolved===0,"oracle gate attribution");
  assert(out.sourceOnlyMidpointCorrect===out.midpointTrials&&out.routeLeakMidpointCorrect===0,"state intervention");
  assert(out.changedAnswers===128&&out.withdrawnAnswers===64&&out.restoredAnswers===64,"revision counts");
  return out;
}

console.log(JSON.stringify(control(), null, 2));
