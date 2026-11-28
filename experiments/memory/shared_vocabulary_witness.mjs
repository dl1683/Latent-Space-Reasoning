// Public paired-annotation propositional witness diagnostic; no model or held gate.
// node shared_vocabulary_witness.mjs MALLS.jsonl FOLIO.jsonl
import { readFileSync } from "node:fs";
function audit(data) {
  function parse(s,mode){
    s=s.trim();
    if(mode==="malls"){
      if(!/^∀x\s*\(/u.test(s))throw Error("scope");
      s=s.replace(/^∀x\s*/u,"");
      if(!s.startsWith("(")||!s.endsWith(")"))throw Error("scope");
      let d=0;for(let i=0;i<s.length;i++){if(s[i]==="(")d++;if(s[i]===")")d--;if(d===0&&i<s.length-1)throw Error("scope");if(d<0)throw Error("scope");}if(d!==0)throw Error("scope");
    }
    const atoms=new Set();let pos=0;
    function ws(){while(pos<s.length&&/\s/u.test(s[pos]))pos++}
    function take(t){ws();if(s.startsWith(t,pos)){pos+=t.length;return true}return false}
    function unary(){if(take("¬"))return ["not",unary()];if(take("(")){const a=expr(0);if(!take(")"))throw Error("syntax");return a}
      ws();const m=s.slice(pos).match(/^([\p{L}_][\p{L}\p{N}_]*)\s*\(\s*([\p{L}\p{N}_]+(?:\s*,\s*[\p{L}\p{N}_]+)*)\s*\)/u);if(!m)throw Error("syntax");
      const args=m[2].split(",").map(x=>x.trim());if(mode==="malls"&&(args.length!==1||args[0]!=="x"))throw Error("scope");if(mode==="folio"&&args.some(x=>/^[xyzuvw]$/.test(x)))throw Error("possible free variable");
      pos+=m[0].length;const key=m[1]+"("+args.join(",")+")";atoms.add(key);return ["atom",key];
    }
    const prec={"↔":1,"→":2,"⊕":3,"∨":4,"∧":5};
    function expr(min){let a=unary();while(true){ws();const op=s[pos],p=prec[op];if(!p||p<min)break;pos++;a=[op,a,expr(p+(op==="→"?0:1))];}return a}
    const t=expr(0);ws();if(pos!==s.length)throw Error("syntax");return {tree:t,atoms:[...atoms].sort()};
  }
  function evalTree(t,v){if(t[0]==="atom")return v[t[1]];if(t[0]==="not")return !evalTree(t[1],v);const a=evalTree(t[1],v),b=evalTree(t[2],v);switch(t[0]){case "∧":return a&&b;case "∨":return a||b;case "⊕":return a!==b;case "→":return !a||b;case "↔":return a===b;default:throw Error("bad tree");}}
  function pair(old,neu,mode){
    const a=parse(old,mode),b=parse(neu,mode);const common=a.atoms.filter(x=>b.atoms.includes(x));const priv=[...new Set([...a.atoms,...b.atoms])].filter(x=>!common.includes(x)).sort();const keys=[...common,...priv];if(keys.length>12)throw Error("size");
    const n=keys.length,k=common.length,full=1<<n,shared=1<<k;let cells=Array(shared).fill(0),hist=[0,0,0,0],labels=[];
    for(let m=0;m<full;m++){const v=Object.fromEntries(keys.map((x,i)=>[x,Boolean(m&(1<<i))]));const z=2*Number(evalTree(a.tree,v))+Number(evalTree(b.tree,v));hist[z]++;labels.push(z);cells[m&(shared-1)]|=1<<z;}
    const oldOnly=4,newOnly=2,disagree=6;const robust=allowed=>cells.filter(c=>(c&~allowed)===0).length;
    const possible=allowed=>cells.filter(c=>c&allowed).length;
    function minCube(allowed,dimension,arr){
      let best=null;
      for(let mask=0;mask<(1<<dimension);mask++){
        const count=mask.toString(2).replace(/0/g,"").length;if(best!==null&&count>=best)continue;
        const bad=new Set();for(let i=0;i<arr.length;i++)if(arr[i]&~allowed)bad.add(i&mask);
        for(let x=mask;;x=(x-1)&mask){if(!bad.has(x)){best=count;break}if(x===0)break}
      }return best;
    }
    const result={n,k,hist,possible:possible(disagree),robust:robust(disagree),oldRobust:robust(oldOnly),newRobust:robust(newOnly),oldMinShared:minCube(oldOnly,k,cells),newMinShared:minCube(newOnly,k,cells),oldMinFull:minCube(oldOnly,n,labels.map(x=>1<<x)),newMinFull:minCube(newOnly,n,labels.map(x=>1<<x))};
    for(let j=0;j<shared;j++){let c=0;for(let m=j;m<full;m+=shared)c|=1<<labels[m];if(c!==cells[j])throw Error("projection")}
    for(const [d,ms,mf] of [[oldOnly,result.oldMinShared,result.oldMinFull],[newOnly,result.newMinShared,result.newMinFull]]){if((ms!==null)!==Boolean(robust(d))||(mf!==null)!==Boolean(hist[d===4?2:1])||(ms!==null&&(mf===null||ms<mf)))throw Error("witness check")}
    return result;
  }
  const report={};
  for(const [mode,rows] of Object.entries(data)){
    const eligible=rows.filter(x=>x.is_original_correct==="no"&&!x.ambiguity&&x.FOL_sentence_new!==x.FOL_sentence_old&&(mode!=="folio"||String(x.id).startsWith("concl")));
    const evaluated=[],skipped=[];
    for(const x of eligible){try{evaluated.push({id:x.id,...pair(x.FOL_sentence_old,x.FOL_sentence_new,mode)})}catch(e){if(!["scope","syntax","possible free variable","size"].includes(e.message))throw e;skipped.push({id:x.id,reason:e.message})}}
    const diff=evaluated.filter(x=>x.hist[1]+x.hist[2]>0), directions=diff.flatMap(x=>[[x.oldMinFull,x.oldMinShared],[x.newMinFull,x.newMinShared]]).filter(x=>x[0]!==null);
    report[mode]={eligible:eligible.length,evaluated:evaluated.length,skipped,different:diff.length,allNoShared:diff.filter(x=>x.oldRobust+x.newRobust===0).length,sharedSome:diff.filter(x=>x.oldRobust+x.newRobust>0).length,possibleDirections:directions.length,sharedDirections:directions.filter(x=>x[1]!==null).length,sharedMulti:directions.filter(x=>x[1]!==null&&x[1]>=2).length,assignmentChecks:evaluated.reduce((s,x)=>s+(1<<x.n),0),rows:evaluated};
  }
  const p=pair("∀x (A(x) → B(x))","∀x (B(x) → A(x))","malls");
  const q=pair("∀x (A(x) → B(x))","∀x (A(x) → C(x))","malls");
  const z=pair("∀x (A(x) → B(x))","∀x (¬A(x) ∨ B(x))","malls");
  if(p.oldMinShared!==2||p.newMinShared!==2||q.oldMinShared!==null||q.newMinShared!==null||q.oldMinFull!==3||z.hist[1]+z.hist[2]!==0)throw Error("analytic controls");
  // Explicitly conditional bridge: not an automatically discovered alias.
  const f=data.folio.find(x=>x.id==="concl_20_13");
  const before=pair(f.FOL_sentence_old,f.FOL_sentence_new,"folio");
  const after=pair(f.FOL_sentence_old.replace(/\bTop10\(/g,"InTop10("),f.FOL_sentence_new,"folio");
  if(before.hist[1]+before.hist[2]!==2||after.hist[1]+after.hist[2]!==0)throw Error("alias control");
  report.aliasControl={id:f.id,before,after};
  return report;
}
const read = path => readFileSync(path,"utf8").trim().split("\n").map(JSON.parse);
if(process.argv.length!==4)throw Error("provide pinned MALLS and FOLIO JSONL");
console.log(JSON.stringify(audit({malls:read(process.argv[2]),folio:read(process.argv[3])}),null,2));
