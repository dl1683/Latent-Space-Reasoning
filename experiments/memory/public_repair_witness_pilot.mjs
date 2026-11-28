// Public development diagnostic, not the frozen independent gate.
 // Data: Brunello et al., Fixing FOLIO and MALLS (arXiv:2606.02837v2), CC BY 4.0.
 // Pin: dslab-uniud/Fixing-FOLIO-and-MALLS@bac88aa1ca01c4b2fcf7ca05dad0a2c5623b2588
 // Usage: node experiments/memory/public_repair_witness_pilot.mjs FOLIO.jsonl MALLS.jsonl
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
function pilot(data){
  // Public candidate ranking: candidate generation is privileged; no gold-derived inputs.
  const stem=w=>w.toLowerCase().replace(/(?:ing|ed|es|s)$/,"");
  const words=s=>(s.match(/[A-Za-z]+/g)||[]).map(stem);
  const identifiers=s=>(s.match(/[A-Za-z][A-Za-z0-9_]*/g)||[]).filter(x=>x.length>1).flatMap(x=>words(x.replace(/([a-z])([A-Z])/g,"$1 $2").replace(/_/g," ")));
  const count=(s,re)=>(s.match(re)||[]).length;
  const ops=[/∀/g,/∃/g,/¬/g,/∧/g,/∨/g,/→/g,/↔/g,/⊕/g];
  function parens(s){let d=0,min=0,max=0;for(const ch of s){if(ch==="(")d++;if(ch===")")d--;min=Math.min(d,min);max=Math.max(d,max)}return [d,min,max]}
  const cues=[/\b(all|every|each|everyone|any)\b/gi,/\b(some|exists|someone|there is|there are)\b/gi,/\b(not|no|never|neither|cannot|without)\b/gi,/\b(and|both)\b/gi,/\b(or|either)\b/gi,/\b(if|when|implies|unless)\b/gi,/\b(if and only if|exactly when|equivalent)\b/gi,/\b(either|exclusive|not both|exactly one)\b/gi];
  const stop=new Set(words("a an the of to in on for by is are be been was were has have had do does that this those these with from at as who which will would can could their they them it he she his her its all every some not no and or if then only either neither both person people one any everyone someone there must may"));
  function features(nl,f,mode){
    const n=words(nl).filter(w=>w&&!stop.has(w)), p=identifiers(f).filter(w=>w&&!stop.has(w));
    const ns=new Set(n),ps=new Set(p),oc=ops.map(re=>count(f,re)),nc=cues.map(re=>count(nl,re)),[d,min,max]=parens(f);
    const b=[Math.log1p(f.length),Math.log1p(p.length),Math.log1p(ps.size),Math.abs(d),-min,max,...oc.map(Math.log1p)];
    if(mode==="structure") return b;
    const nmatch=[...ns].filter(w=>ps.has(w)).length,pmatch=[...ps].filter(w=>ns.has(w)).length;
    // For a single implication, compare surface phrase order on the two sides.
    // These are cheap diagnostic features, not a semantic parser.
    const m=nl.toLowerCase().match(/\bif\b/);let align=0;
    if(m&&oc[5]===1){const cut=m.index,left=new Set(words(nl.slice(0,cut))),right=new Set(words(nl.slice(cut+2)));
      const [ant,cons]=f.split("→").map(s=>new Set(identifiers(s)));
      const overlap=(a,b)=>[...a].filter(x=>!stop.has(x)&&b.has(x)).length;
      align=overlap(ant,right)+overlap(cons,left)-overlap(ant,left)-overlap(cons,right);
    }
    return [...b,nmatch/Math.max(1,ns.size),pmatch/Math.max(1,ps.size),n.filter(w=>ps.has(w)).length/Math.max(1,n.length),p.filter(w=>ns.has(w)).length/Math.max(1,p.length),Math.log((f.length+1)/(nl.length+1)),align,...oc.map((v,i)=>Math.log1p(v)*Math.log1p(nc[i])),...oc.map((v,i)=>Math.abs(Math.log1p(v)-Math.log1p(nc[i])))];
  }
  const eligible=(r)=>r.is_original_correct==="no"&&r.ambiguity!==true&&typeof r.FOL_sentence_old==="string"&&typeof r.FOL_sentence_new==="string"&&r.FOL_sentence_old!==r.FOL_sentence_new;
  const balanced=f=>{const [d,min]=parens(f);return d===0&&min===0};
  const corp={};for(const [k,rs] of Object.entries(data))corp[k]=rs.filter(eligible).map(r=>({r,group:k==="folio"?String(r.id).match(/\d+/)[0]:String(r.id),balanced:balanced(r.FOL_sentence_old)&&balanced(r.FOL_sentence_new),unchanged:["True","False","Uncertain"].includes(r.label_z3)&&r.label_z3===r.label_old_z3}));
  function fit(train,mode){
    const ds=train.map(({r})=>{const a=features(r.NL_sentence,r.FOL_sentence_old,mode),b=features(r.NL_sentence,r.FOL_sentence_new,mode);return b.map((v,i)=>v-a[i])});
    const dim=ds[0].length,scale=Array.from({length:dim},(_,i)=>Math.max(.1,Math.sqrt(ds.reduce((s,v)=>s+v[i]*v[i],0)/ds.length))),xs=ds.map(d=>d.map((v,i)=>v/scale[i])),w=Array(dim).fill(0);
    // Pairwise logistic, balanced candidate order; no intercept, fixed ridge.
    for(let k=0;k<1000;k++){let g=w.map(v=>.1*v);for(const x of xs){const s=x.reduce((z,v,i)=>z+v*w[i],0),a=1/(1+Math.exp(Math.max(-40,Math.min(40,s))));for(let i=0;i<dim;i++)g[i]-=x[i]*a/xs.length}for(let i=0;i<dim;i++)w[i]-=.05*g[i]}
    return (r)=>{const a=features(r.NL_sentence,r.FOL_sentence_old,mode),b=features(r.NL_sentence,r.FOL_sentence_new,mode);return b.reduce((s,v,i)=>s+(v-a[i])*w[i]/scale[i],0)};
  }
  function score(rows,predict){const pred=rows.map(x=>predict(x.r));let wins=0,ties=0,loss=0;for(const v of pred){if(!Number.isFinite(v))throw Error("nonfinite");if(v>1e-10)wins++;else if(v< -1e-10)loss++;else ties++}return [wins,ties,loss]}
  const out={};
  for(const k of ["folio","malls"]){const other=k==="folio"?"malls":"folio",a=corp[k],tr=corp[other];out[k]={n:a.length,groups:new Set(a.map(x=>x.group)).size,balanced:a.filter(x=>x.balanced).length,unchanged:a.filter(x=>x.unchanged).length,train:tr.length};
    for(const mode of ["structure","joint"]){const predict=fit(tr,mode);out[k][mode]={all:score(a,predict),balanced:score(a.filter(x=>x.balanced),predict),unchanged:score(a.filter(x=>x.unchanged),predict)}}
    const shorter=r=>r.FOL_sentence_old.length-r.FOL_sentence_new.length;
    const moreBalanced=r=>{const p=s=>{const[d,min]=parens(s);return Math.abs(d)-min};return p(r.FOL_sentence_old)-p(r.FOL_sentence_new)};
    out[k].shorter=score(a,shorter);out[k].paren=score(a,moreBalanced);
    for(const mode of ["structure","joint"]){const p=fit(tr,mode);for(const {r} of a){const q={...r,FOL_sentence_old:r.FOL_sentence_new,FOL_sentence_new:r.FOL_sentence_old};if(Math.abs(p(r)+p(q))>1e-10)throw Error("order leak")}}
    for(const mode of ["structure","joint"]) for(const [subset,n] of [["all",a.length],["balanced",out[k].balanced],["unchanged",out[k].unchanged]])if(out[k][mode][subset].reduce((a,b)=>a+b,0)!==n)throw Error("denominator mismatch");
  }
  const rr=data.folio,valid=x=>["True","False","Uncertain"].includes(x);
  out.endpoint={human:rr.filter(r=>r.is_original_correct==="no"&&r.label&&r.label_old&&r.label===r.label_old).length,solver:rr.filter(r=>r.is_original_correct==="no"&&valid(r.label_z3)&&r.label_z3===r.label_old_z3).length,conclusions:rr.filter(r=>r.label&&r.label_old).length,incorrectConclusions:rr.filter(r=>r.is_original_correct==="no"&&r.label&&r.label_old).length,incorrectValidSolver:rr.filter(r=>r.is_original_correct==="no"&&valid(r.label_z3)&&valid(r.label_old_z3)).length};
  return out;
}
const paths=process.argv.slice(2);
if(paths.length!==2)throw Error('supply pinned FOLIO and MALLS JSONL paths');
const data={};
const expected={"folio":"e5428ca9a4efbe9460bc8a833d9e5f01ff3d92d9","malls":"c42ef1680ce6b21cbaf33a429faa93644e011c37"};
for(const [i,c] of ['folio','malls'].entries()){
  const b=readFileSync(paths[i]);
  const sha=createHash('sha1').update('blob '+b.length+'\0').update(b).digest('hex');
  if(sha!==expected[c])throw Error('input Git blob mismatch: '+c);
  data[c]=b.toString('utf8').trim().split('\n').map(JSON.parse);
}
console.log(JSON.stringify(pilot(data),null,2));
