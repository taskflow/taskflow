'use strict';
// .tfp parser worker (inlined in index.html as #src-parser — keep in sync).
//
// Besides decoding the file it precomputes everything the UI needs to stay
// fast on traces with millions of tasks:
//
//   nameId / nameStr  every distinct task name is decoded once; each task stores
//                     a small integer id (0 = anonymous) instead of a string.
//   pdur              prefix sum of task durations in index order, so the busy
//                     time of any contiguous run of tasks is one subtraction.
//   lod*              per-row level-of-detail clusters (see buildLod below).
//   topIdx            indices of the longest tasks, longest first.

function u8(v,o){return v.getUint8(o);}
function u16(v,o){return v.getUint16(o,true);}
function u32(v,o){return v.getUint32(o,true);}

// LEB128 varint — matches _tfp_write_varint in C++.
function readVarint(v, o) {
  let r=0, s=1, b;
  do { b=v.getUint8(o++); r+=(b&0x7f)*s; s*=128; } while(b&0x80);
  return {val:r, o};
}

function parse(buf){
  const v=new DataView(buf); let o=0;
  const dc=new TextDecoder();

  // ── File header (12 bytes): magic(4) version(u16) flags(u16) num_exec(u32)
  const mg=String.fromCharCode(u8(v,0),u8(v,1),u8(v,2),u8(v,3));
  if(mg!=='TFPX') throw new Error('Not a .tfp file');
  o=4; u16(v,o); o+=2; o+=2;   // version, flags (skip)
  const ne=u32(v,o); o+=4;      // num_exec

  // ── Pass 1: count total segments for typed array allocation ────────────
  let tot=0;
  { let p=o;
    for(let e=0;e<ne;e++){
      // Executor header (24 bytes): uid(8) origin_us(8) str_table_len(4) num_wl(4)
      p+=16;
      const sl=u32(v,p); p+=4;  // str_table_len
      const nw=u32(v,p); p+=4;  // num_wl
      p+=sl;                     // skip string table
      for(let w=0;w<nw;w++){
        p+=8;                    // worker_id(4) + level(4)
        const ns=u32(v,p); p+=4; tot+=ns;
        for(let s=0;s<ns;s++){
          let b; do{b=v.getUint8(p++);}while(b&0x80); // delta_beg varint
          do{b=v.getUint8(p++);}while(b&0x80);         // duration varint
          p+=5;                  // name_off(4) + type|nlen(1)
        }
      }
    }
  }

  // ── Allocate output arrays ─────────────────────────────────────────────
  const beg=new Float64Array(tot), end=new Float64Array(tot),
        tp=new Uint8Array(tot), nid=new Uint32Array(tot);
  const nameStr=[null];          // id 0 = anonymous
  const byString=new Map();      // name -> id (shared across executors)
  const ws=[]; let c=0, minU=Infinity, maxU=-Infinity;

  // ── Pass 2: decode each executor block ────────────────────────────────
  for(let e=0;e<ne;e++){
    o+=8;                         // uid (skip)
    o+=8;                         // origin_us (skip — timestamps are origin-relative)
    const sl=u32(v,o); o+=4;      // str_table_len
    const nw=u32(v,o); o+=4;      // num_wl

    // Per-executor string table; names are interned by (offset, length).
    const sb=new Uint8Array(buf, o, sl); o+=sl;
    const local=new Map();
    const intern=(no,nl)=>{
      if(nl===0) return 0;
      const key=no*32+nl;
      let id=local.get(key);
      if(id===undefined){
        const s=dc.decode(sb.subarray(no,no+nl));
        id=byString.get(s);
        if(id===undefined){ id=nameStr.length; nameStr.push(s); byString.set(s,id); }
        local.set(key,id);
      }
      return id;
    };

    // Worker-level blocks
    for(let w=0;w<nw;w++){
      const wi=u32(v,o); o+=4;    // worker_id
      const lv=u32(v,o); o+=4;    // level
      const ns=u32(v,o); o+=4;    // num_segs
      const so=c; let prevBeg=0;
      for(let s=0;s<ns;s++){
        let r=readVarint(v,o); const deltaBeg=r.val; o=r.o;
        r=readVarint(v,o);     const dur=r.val;      o=r.o;
        const b=prevBeg+deltaBeg, en=b+dur; prevBeg=b;
        const no=u32(v,o); o+=4;
        const packed=u8(v,o); o+=1;
        beg[c]=b; end[c]=en; tp[c]=(packed>>5)&0x07; nid[c]=intern(no,packed&0x1F);
        if(b<minU)minU=b; if(en>maxU)maxU=en; c++;
      }
      ws.push({key:`E${e}.W${wi}.L${lv}`,executorId:e,workerIdx:wi,level:lv,offset:so,length:ns});
    }
  }

  if(tot===0){minU=0;maxU=0;}

  const pdur=new Float64Array(tot+1);
  for(let i=0;i<tot;i++) pdur[i+1]=pdur[i]+(end[i]-beg[i]);

  const lod=buildLod(beg,end,tp,ws);
  const topIdx=longestTasks(beg,end);

  return{ok:true,meta:{numWorkers:ws.length,totalSegs:tot,minUs:minU,maxUs:maxU},
         workers:ws,beg,end,type:tp,nameId:nid,nameStr,pdur,topIdx,
         lodI0:lod.i0,lodEnd:lod.end,lodType:lod.type};
}

// ── Level-of-detail clustering ──────────────────────────────────────────────
// Within one (worker, level) row, tasks are sorted and do not overlap. Merging
// every pair of neighbours whose gap is below a threshold r gives a set of
// clusters; raising r only ever merges more, so the clusterings form a
// hierarchy. For each row we store a few levels of that hierarchy, each with at
// most LOD_RATIO times the clusters of the level below it. At draw time the UI
// picks the coarsest stored level whose threshold is still below the gap it
// wants to merge (half a pixel) and finishes merging on the fly. Drawing cost
// is therefore bounded by a small multiple of the pixels on screen, whatever
// the trace size, and the stored levels add at most a third of the row size.
//
// A cluster is the contiguous task range [i0[k], i0[k+1]) of its row (the
// last one ends at the row end); we store its first task, its end time and its
// type (MIXED if it contains several types).
const LOD_RATIO=0.25, LOD_MIN=256, NB=64, R0=1/1024, MIXED=255;

function gapBucket(g){
  if(!(g>=R0)) return 0;                        // includes 0 and overlaps
  let k=Math.floor(Math.log2(g/R0))+1;          // R0·2^(k-1) <= g < R0·2^k
  while(k>1 && g<R0*Math.pow(2,k-1)) k--;       // guard against log2 rounding
  while(g>=R0*Math.pow(2,k)) k++;
  return k<NB ? k : NB-1;
}

function buildLod(beg,end,tp,rows){
  // Plan: per row, count gaps per power-of-two bucket. Merging at threshold
  // r_k = R0·2^k joins every gap in buckets 0..k, so the cluster count at each
  // threshold is known before building anything.
  let total=0;
  const hist=new Float64Array(NB);
  for(const w of rows){
    w.lod=[];
    const n=w.length;
    if(n<LOD_MIN) continue;
    hist.fill(0);
    let maxEnd=end[w.offset];
    for(let s=w.offset+1;s<w.offset+n;s++){
      hist[gapBucket(beg[s]-maxEnd)]++;
      if(end[s]>maxEnd) maxEnd=end[s];
    }
    let merged=0, last=n;
    for(let k=0;k<NB;k++){
      merged+=hist[k];
      const cnt=n-merged;
      if(cnt<=last*LOD_RATIO){ w.lod.push({r:R0*Math.pow(2,k), off:0, n:cnt}); last=cnt; total+=cnt; }
      if(cnt<=1) break;
    }
  }

  const i0=new Uint32Array(total), cend=new Float64Array(total), ctype=new Uint8Array(total);
  let out=0;
  for(const w of rows){
    let src=null;                                // null = raw tasks
    for(const L of w.lod){
      L.off=out;
      const r=L.r;
      let cI=-1, cE=0, cT=0;
      const take=(ti0,te,tt)=>{
        if(cI>=0 && beg[ti0]-cE<r){ if(te>cE) cE=te; if(tt!==cT) cT=MIXED; }
        else{
          if(cI>=0){ i0[out]=cI; cend[out]=cE; ctype[out]=cT; out++; }
          cI=ti0; cE=te; cT=tt;
        }
      };
      if(!src) for(let s=w.offset;s<w.offset+w.length;s++) take(s,end[s],tp[s]);
      else     for(let k=src.off;k<src.off+src.n;k++) take(i0[k],cend[k],ctype[k]);
      i0[out]=cI; cend[out]=cE; ctype[out]=cT; out++;
      L.n=out-L.off;                             // equals the planned count
      src=L;
    }
  }
  return {i0, end:cend, type:ctype};
}

// ── Longest tasks ───────────────────────────────────────────────────────────
// The UI's "top tasks" chart needs the longest tasks inside a time window.
// For wide windows it walks this list (longest first) and stops after enough
// hits; for narrow windows it scans the window directly.
function longestTasks(beg,end){
  const N=beg.length;
  if(N===0) return new Uint32Array(0);
  const T=Math.min(N, Math.max(5000, Math.ceil(N/100)));
  const d=new Float64Array(N);
  for(let i=0;i<N;i++) d[i]=end[i]-beg[i];
  const thr=nthLargest(d.slice(), T);
  const idx=new Uint32Array(T);
  let n=0;
  for(let i=0;i<N && n<T;i++) if(d[i]>thr) idx[n++]=i;
  for(let i=0;i<N && n<T;i++) if(d[i]===thr) idx[n++]=i;
  const out=idx.subarray(0,n);
  out.sort((a,b)=>d[b]-d[a] || a-b);
  return out.slice();
}

// k-th largest value (1-based) via iterative quickselect; reorders a.
function nthLargest(a,k){
  let lo=0, hi=a.length-1;
  const target=a.length-k;                       // index in ascending order
  while(hi>lo){
    const mid=(lo+hi)>>1;
    let x=a[lo], y=a[mid], z=a[hi];
    const p = x<y ? (y<z ? y : (x<z ? z : x)) : (x<z ? x : (y<z ? z : y));
    let i=lo, j=hi;
    while(i<=j){
      while(a[i]<p) i++;
      while(a[j]>p) j--;
      if(i<=j){ const t=a[i]; a[i]=a[j]; a[j]=t; i++; j--; }
    }
    if(target<=j) hi=j; else if(target>=i) lo=i; else break;
  }
  return a[target];
}

self.onmessage=e=>{
  try{
    const r=parse(e.data.buffer);
    self.postMessage(r,[r.beg.buffer,r.end.buffer,r.type.buffer,r.nameId.buffer,r.pdur.buffer,
                        r.topIdx.buffer,r.lodI0.buffer,r.lodEnd.buffer,r.lodType.buffer]);
  }
  catch(er){self.postMessage({ok:false,error:er.message});}
};
