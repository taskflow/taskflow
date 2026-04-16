'use strict';
function u8(v,o){return v.getUint8(o);}
function u16(v,o){return v.getUint16(o,true);}
function u32(v,o){return v.getUint32(o,true);}

// LEB128 varint — matches _tfp_write_varint in C++.
function readVarint(v, o) {
  let val=0, shift=0, b;
  do { b=v.getUint8(o++); val+=(b&0x7F)*Math.pow(2,shift); shift+=7; } while(b&0x80);
  return {val, o};
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
        tp=new Uint8Array(tot), na=new Array(tot);
  const ws=[]; let c=0, minU=Infinity, maxU=-Infinity;

  // ── Pass 2: decode each executor block ────────────────────────────────
  for(let e=0;e<ne;e++){
    // Executor header
    o+=8;                         // uid (skip)
    o+=8;                         // origin_us (skip — timestamps are origin-relative)
    const sl=u32(v,o); o+=4;      // str_table_len
    const nw=u32(v,o); o+=4;      // num_wl

    // Per-executor string table
    const sb=new Uint8Array(buf, o, sl); o+=sl;
    const nm=(no,nl)=>nl===0?null:dc.decode(sb.subarray(no,no+nl));

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
        beg[c]=b; end[c]=en; tp[c]=(packed>>5)&0x07; na[c]=nm(no,packed&0x1F);
        if(b<minU)minU=b; if(en>maxU)maxU=en; c++;
      }
      ws.push({key:`E${e}.W${wi}.L${lv}`,executorId:e,workerIdx:wi,level:lv,offset:so,length:ns});
    }
  }

  if(tot===0){minU=0;maxU=0;}
  return{ok:true,meta:{numWorkers:ws.length,totalSegs:tot,minUs:minU,maxUs:maxU},
         workers:ws,beg,end,type:tp,names:na};
}
self.onmessage=e=>{
  try{const r=parse(e.data.buffer);self.postMessage(r,[r.beg.buffer,r.end.buffer,r.type.buffer]);}
  catch(er){self.postMessage({ok:false,error:er.message});}
};
