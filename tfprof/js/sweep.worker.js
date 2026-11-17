'use strict';
let B=null,E=null,W=null,physicalWorkers=null;
// Pre-computed full-resolution curves (over entire [minUs,maxUs] range)
let fullWC=null, fullTC=null, fullBins=0, fullMin=0, fullMax=0;

const FULL_BINS = 4000;  // high resolution pre-computed curve

function buildFullCurves(minUs, maxUs) {
  const bins=FULL_BINS, range=maxUs-minUs;
  if(range<=0){fullWC=new Float32Array(bins);fullTC=new Float32Array(bins);return;}
  const bu=range/bins;
  const wc=new Float32Array(bins), td=new Float32Array(bins+1);

  for(let pi=0;pi<physicalWorkers.length;pi++){
    const wa=new Uint8Array(bins);
    const wlist=physicalWorkers[pi];
    for(let wi=0;wi<wlist.length;wi++){
      const{offset:off,length:len}=W[wlist[wi]];
      for(let s=0;s<len;s++){
        const i=off+s,b=B[i],e=E[i];
        if(e<=minUs||b>=maxUs)continue;
        const fb=Math.floor((Math.max(b,minUs)-minUs)/bu);
        const lb=Math.min(bins-1,Math.floor((Math.min(e,maxUs)-minUs-1e-9)/bu));
        if(fb>lb)continue;
        for(let bn=fb;bn<=lb;bn++)wa[bn]=1;
        td[fb]+=1;td[lb+1]-=1;
      }
    }
    for(let bn=0;bn<bins;bn++)if(wa[bn])wc[bn]+=1;
  }
  const tc=new Float32Array(bins);let r=0;
  for(let bn=0;bn<bins;bn++){r+=td[bn];tc[bn]=r;}
  fullWC=wc; fullTC=tc; fullBins=bins; fullMin=minUs; fullMax=maxUs;
}

// Slice the pre-computed full curves to the query window [x0,x1]
// and downsample to px output bins — O(FULL_BINS) not O(segments)
function sliceCurves(x0, x1, px) {
  const outBins=Math.max(1,Math.floor(px));
  const wc=new Float32Array(outBins), tc=new Float32Array(outBins);
  if(!fullWC||fullMax<=fullMin)return{wc,tc};

  const range=fullMax-fullMin;
  const bu=range/fullBins;  // microseconds per full bin

  // Map x0,x1 to full bin indices
  const fb0=Math.max(0,Math.floor((x0-fullMin)/bu));
  const fb1=Math.min(fullBins-1,Math.floor((x1-fullMin)/bu));
  const nFull=fb1-fb0+1;
  if(nFull<=0)return{wc,tc};

  // Average/max full bins into output bins
  const ratio=nFull/outBins;
  for(let ob=0;ob<outBins;ob++){
    const bStart=fb0+Math.floor(ob*ratio);
    const bEnd=fb0+Math.min(nFull-1,Math.floor((ob+1)*ratio));
    let maxWC=0, maxTC=0;
    for(let b=bStart;b<=bEnd;b++){
      if(fullWC[b]>maxWC)maxWC=fullWC[b];
      if(fullTC[b]>maxTC)maxTC=fullTC[b];
    }
    wc[ob]=maxWC; tc[ob]=maxTC;
  }
  return{wc,tc};
}

self.onmessage=e=>{
  const m=e.data;
  if(m.type==='init'){
    B=m.beg;E=m.end;W=m.workers;
    const map=new Map();
    for(let wi=0;wi<W.length;wi++){
      const key=W[wi].executorId+'_'+W[wi].workerIdx;
      if(!map.has(key))map.set(key,[]);
      map.get(key).push(wi);
    }
    physicalWorkers=[...map.values()];
    // Pre-compute full curves in background
    buildFullCurves(m.minUs, m.maxUs);
    self.postMessage({type:'ready'});
    return;
  }
  if(m.type==='query'){
    const{wc,tc}=sliceCurves(m.x0,m.x1,m.px);
    self.postMessage({type:'result',wc,tc},[wc.buffer,tc.buffer]);
  }
};
