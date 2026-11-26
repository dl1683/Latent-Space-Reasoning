// Public development control; not independent evaluator material.
// Run: node experiments/memory/raw_source_order_control.js
// See accompanying Issue #6 discussion for evidence boundaries.
"use strict";

function rawSourceOrderControl() {
  // Development-only synthetic prose. No learned model, no held-out evaluator.
  const assert = (ok, label) => { if (!ok) throw Error(label); };
  const words = s => s.toLowerCase().match(/[a-z]+[0-9]*|[.!?]/g) || [];
  const grams = (s, k) => {
    const w = words(s), counts = new Map();
    for (let i=0; i+k<=w.length; i++) {
      const key = w.slice(i,i+k).join(" ");
      counts.set(key,(counts.get(key)||0)+1);
    }
    return JSON.stringify([...counts].sort(([a],[b])=>a.localeCompare(b)));
  };
  // Generic identifier co-occurrence per sentence; no role labels or task.
  // This only proposes candidates, not semantic entailment.
  const compile = records => {
    const links = new Map();
    for (const [source, raw] of Object.entries(records)) {
      for (const [ordinal, span] of raw.split(/[.!?]/).entries()) {
        const ids = [...new Set(words(span).filter(w=>/^id[0-9]+$/.test(w)))];
        for (const a of ids) for (const b of ids) if (a!==b) {
          if (!links.has(a)) links.set(a, []);
          links.get(a).push({peer:b,source,ordinal});
        }
      }
    }
    return links;
  };
  const proposal = (links, id) => {
    const candidates = links.get(id)||[];
    return {status:new Set(candidates.map(c=>c.peer)).size===1?"candidate":"unresolved",candidates};
  };
  let rng = 174907;
  const next = () => (rng=(Math.imul(1664525,rng)+1013904223)>>>0);
  const sentence = (a,b) => `The applicable approval for ${a} after all required checks of the operating team and the independent review of the documented process have been completed is recorded under ${b} in the separate ledger retained for audit and later verification by the responsible office.`;
  const logs = (a,b,c) => ({
    log1:`The current decision for ${a} is clearance.`,
    log2:`The current decision for ${b} is withholding.`,
    log3:`The current decision for ${c} is clearance.`
  });
  const corpus = (a,b,c,x,y,z,reverse) => ({
    routes:[sentence(a,reverse?y:x),sentence(b,reverse?x:y),sentence(c,z)].join(" "),
    ...logs(x,y,z)
  });
  let comparisons=0, lateFindings=0, forcedErrors=0, withdrawals=0, unrelated=0, restored=0, ambiguity=0, negation=0;
  for (let trial=0;trial<128;trial++) {
    const names = new Set(); while(names.size<6) names.add("id"+next());
    const [a,b,c,x,y,z]=names;
    const left=corpus(a,b,c,x,y,z,false),right=corpus(a,b,c,x,y,z,true);
    const before=compile(left),after=compile(right);
    // Raw source IDs, terminal records, sentence count and token counts fixed.
    // Compare exact multisets, not hash collisions or floating point tolerance.
    for(let k=1;k<=16;k++){
      assert(grams(left.routes,k)===grams(right.routes,k),"k-gram collision");
      comparisons++;
    }
    assert(left.routes!==right.routes,"raw bytes differ");
    for(const log of ["log1","log2","log3"]) assert(left[log]===right[log],"terminal changed");
    // All candidate links compiled before selection of either late task.
    const tasks=next()%2?[a,b]:[b,a];
    const answers=new Map([[x,"clearance"],[y,"withholding"],[z,"clearance"]]);
    for(const task of tasks) {
      const first=proposal(before,task),second=proposal(after,task);
      assert(first.status==="candidate"&&second.status==="candidate","unique candidates");
      const p=first.candidates[0],q=second.candidates[0];
      assert(p.source==="routes"&&q.source==="routes","provenance");
      assert(p.peer===(task===a?x:y)&&q.peer===(task===a?y:x),"wrong candidate");
      assert(answers.get(p.peer)!==answers.get(q.peer),"opposite decisions");
      // Any deterministic forced verdict based only on identical pooled
      // features and the same task is wrong at least once in this pair.
      forcedErrors++;
      lateFindings+=2;
    }
    const {routes,...withoutRoute}=right;
    assert(proposal(compile(withoutRoute),a).status==="unresolved","withdrawal not reflected");
    withdrawals++;
    assert(proposal(before,c).candidates[0].peer===z&&proposal(after,c).candidates[0].peer===z,"unrelated region changed");
    unrelated++;
    assert(JSON.stringify(proposal(compile(left),a))===JSON.stringify(proposal(before,a)),"restoration");
    restored++;
    const ambiguous={...left,routes:left.routes+" "+sentence(a,y)};
    assert(proposal(compile(ambiguous),a).status==="unresolved","competing candidate not preserved");
    ambiguity++;
    // Span co-occurrence itself cannot establish a relation under negation.
    const negative={...left,routes:left.routes.replace("is recorded under "+x,"is not recorded under "+x)};
    assert(proposal(compile(negative),a).candidates[0].peer===x,"negation changed candidate");
    negation++;
  }
  return {twin_pairs:128,exact_pooled_kgram_equalities:comparisons,k_range:[1,16],
    late_candidate_checks:lateFindings,minimum_forced_binary_errors:forcedErrors,
    withdrawal_checks:withdrawals,unrelated_region_checks:unrelated,restoration_checks:restored,
    ambiguous_candidate_checks:ambiguity,negation_false_entailment_controls:negation,
    qualification:"Candidate generation only; exact indexed sentence co-occurrence is the same algorithm. No learned discovery, semantic entailment, or efficiency claim."};
}

console.log(JSON.stringify(rawSourceOrderControl(), null, 2));
