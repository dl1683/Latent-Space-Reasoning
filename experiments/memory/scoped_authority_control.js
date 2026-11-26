// Development-only supplied grammar and authority policy; no learned model.
function control(){
  const counts={worlds:128,checks:0,scanParity:0,historical:0,newestErrors:0,scopeBlindErrors:0,timeBlindErrors:0,incrementalBytes:0,rebuildBytes:0};
  function assert(v,m){if(!v)throw Error(m)}
  function parse(d){
    let pos=0;
    return d.text.split("\n").map(line=>{
      const ptr=[d.id,d.gen,pos,pos+line.length];pos+=line.length+1;
      let m=/^(\w+) (allows|denies) (\w+) from (\d+) until (\d+)\.$/.exec(line);
      if(m)return {kind:"assert",id:m[1],verdict:m[2],scope:m[3],lo:+m[4],hi:+m[5],ptr};
      m=/^For (\w+) from (\d+) until (\d+), (\w+) replaces (\w+)\.$/.exec(line);
      if(m)return {kind:"order",scope:m[1],lo:+m[2],hi:+m[3],winner:m[4],loser:m[5],issuer:d.issuer,ptr};
      throw Error("unrecognized fixture clause");
    });
  }
  function resolve(rows,s,t,mode="exact"){
    const applies=r=>r.lo<=t&&t<r.hi;
    const a=rows.filter(r=>r.kind==="assert"&&r.scope===s&&applies(r));
    const orders=rows.filter(r=>r.kind==="order"&&r.issuer==="board"&&
      (mode==="scopeBlind"||r.scope===s)&&(mode==="timeBlind"||applies(r)));
    const ids=new Set(a.map(r=>r.id)),suppressed=new Set(),edges=new Map(),used=[];
    assert(ids.size===a.length,"duplicate assertion");
    function receipt(status,active=a,removed=[]){return {status,active:active.map(r=>r.ptr).sort(),suppressed:removed.map(r=>r.ptr).sort(),orders:used.slice().sort()}}
    for(const o of orders){
      used.push(o.ptr);
      if(!ids.has(o.winner)||!ids.has(o.loser)||o.winner===o.loser)return receipt("unresolved");
      suppressed.add(o.loser);
      if(!edges.has(o.winner))edges.set(o.winner,new Set());
      edges.get(o.winner).add(o.loser);
    }
    const visiting=new Set(),done=new Set();
    function cyclic(k){
      if(visiting.has(k))return true;
      if(done.has(k))return false;
      visiting.add(k);for(const v of edges.get(k)||[])if(cyclic(v))return true;
      visiting.delete(k);done.add(k);return false;
    }
    for(const k of ids)if(cyclic(k))return receipt("unresolved");
    const live=a.filter(r=>!suppressed.has(r.id)),dead=a.filter(r=>suppressed.has(r.id));
    const v=new Set(live.map(r=>r.verdict));
    return receipt(v.size===1?live[0].verdict:v.size===0?"unresolved":"conflict",live,dead);
  }
  class Index{
    constructor(){this.docs=new Map();this.rows=new Map();this.buckets=new Map()}
    update(k,d){
      const old=this.rows.get(k)||[],next=d?parse(d):[];
      if(d)counts.incrementalBytes+=d.text.length;
      for(const s of new Set([...old,...next].map(r=>r.scope))){
        const retained=(this.buckets.get(s)||[]).filter(r=>r.ptr[0]!==k);
        this.buckets.set(s,retained.concat(next.filter(r=>r.scope===s)));
      }
      if(d){this.docs.set(k,d);this.rows.set(k,next)}else{this.docs.delete(k);this.rows.delete(k)}
    }
    pin(){return new Map(this.buckets)}
  }
  let seed=21721;
  function id(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return "k"+seed.toString(16)}
  for(let n=0;n<counts.worlds;n++){
    const x=id(),y=id(),a=id(),b=id(),da=id(),db=id(),ord=id(),fake=id(),rival=id();
    const doc=(id,gen,issuer,text)=>({id,gen,issuer,text});
    const A=doc(da,2,"ledger",`${a} allows ${x} from 0 until 30.\n${a} allows ${y} from 0 until 30.`);
    const B=doc(db,1,"ledger",`${b} denies ${x} from 0 until 30.\n${b} denies ${y} from 0 until 30.`);
    const first=n%2===0;
    function order(flip){
      const p=first!==flip?[a,b]:[b,a];
      return `For ${x} from 10 until 20, ${p[0]} replaces ${p[1]}.\nFor ${y} from 10 until 20, ${p[1]} replaces ${p[0]}.`;
    }
    const idx=new Index(),history=[];
    idx.update(da,A);idx.update(db,B);
    const expected=(flip,s)=>((s===x)===(first!==flip)?"allows":"denies");
    function check(stage,inside,outside="conflict"){
      const snapshot=idx.pin(),docs=[...idx.docs.values()];
      const rows=docs.flatMap(d=>{counts.rebuildBytes+=d.text.length;return parse(d)});
      for(const s of [x,y])for(const t of [5,10,15,19,20,25]){
        const exp=10<=t&&t<20?inside(s):outside;
        const got=resolve(snapshot.get(s)||[],s,t),full=resolve(rows,s,t);
        assert(got.status===exp,`${stage}: ${got.status} vs ${exp}`);
        assert(JSON.stringify(got)===JSON.stringify(full),"receipt parity");
        for(const ptr of [...got.active,...got.suppressed,...got.orders]){
          const d=docs.find(d=>d.id===ptr[0]&&d.gen===ptr[1]);
          assert(d&&d.text.slice(ptr[2],ptr[3]).endsWith("."),"invalid source span");
        }
        counts.checks++;counts.scanParity++;
        const newest=rows.filter(r=>r.kind==="assert"&&r.scope===s&&r.lo<=t&&t<r.hi).sort((u,v)=>v.ptr[1]-u.ptr[1])[0];
        if((newest?.verdict||"unresolved")!==exp)counts.newestErrors++;
        if(resolve(rows,s,t,"scopeBlind").status!==exp)counts.scopeBlindErrors++;
        if(resolve(rows,s,t,"timeBlind").status!==exp)counts.timeBlindErrors++;
        history.push({snapshot,s,t,receipt:JSON.stringify(got)});
      }
    }
    check("opening",()=>"conflict");
    idx.update(ord,doc(ord,3,"board",order(false)));check("discovery",s=>expected(false,s));
    idx.update(ord,doc(ord,4,"board",order(true)));check("correction",s=>expected(true,s));
    idx.update(fake,doc(fake,99,"untrusted",order(false)));check("untrusted",s=>expected(true,s));idx.update(fake,null);
    idx.update(rival,doc(rival,6,"board",order(false)));check("cycle",()=>"unresolved");idx.update(rival,null);
    idx.update(da,null);check("missing endpoint",()=>"unresolved","denies");idx.update(da,{...A,gen:7});
    idx.update(ord,null);check("withdrawal",()=>"conflict");
    idx.update(ord,doc(ord,8,"board",order(true)));check("restoration",s=>expected(true,s));
    for(const p of history){assert(JSON.stringify(resolve(p.snapshot.get(p.s)||[],p.s,p.t))===p.receipt,"historical pin changed");counts.historical++}
  }
  return counts;
}
console.log(JSON.stringify(control(), null, 2));
