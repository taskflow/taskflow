// Program: tfprof
// Author: twhuang
'use strict';

const tfp = {
  
  // main timeline svg
  dom : null,
  svg : null,
  width: null,
  height: null,
  topMargin: 30,
  bottomMargin: 40,
  leftMargin: 100,
  rightMargin: 40,
  innerH: 30,
  innerW: 20,
  segMaxH: 18,
  segH: null,
  segHOfst: null,
 
  // timeline chart
  tlG: null,
  tlW: null,
  tlH: null,
  tlXScale: d3.scaleLinear(),
  tlWScale: d3.scaleBand(),  
  tlEScale: d3.scaleOrdinal(),
  tlZScale: d3.scaleOrdinal(),
  tlXAxis: d3.axisBottom(),
  tlWAxis: d3.axisLeft(),
  tlEAxis: d3.axisLeft(),
  tlBrush: d3.brushX(),
  tlBrushTimeout: null,

  // load chart
  loadG: null,
  loadW: null,
  loadH: null,
  loadXScale: d3.scaleLinear(),
  loadEScale: d3.scaleOrdinal(),
  loadXAxis: d3.axisBottom(),
  loadEAxis: d3.axisLeft(),

  // overview chart
  ovG: null,
  ovXScale: d3.scaleLinear(),
  ovXSel: null,
  ovXDomain: null,
  ovXAxis: d3.axisBottom(),
  ovBrush: d3.brushX(),
  ovH: 20,

  // rank chart
  rankG: null,
  rankH: 300,
  rankW: null,
  rankXScale: d3.scaleBand(),
  rankYScale: d3.scaleLinear(),
  rankXAxis: d3.axisBottom(),
  rankYAxis: d3.axisLeft(),

  // legend
  zColorMap: new Map([
    ['static', '#4682b4'],
    ['subflow', '#ff7f0e'],
    ['cudaflow', '#6A0DAD'],
    ['condition', '#32CD99'],
    ['module', '#0000FF'],
    ['async', '#292b2c'],
    ['clustered', '#999DA0']
  ]),
  zScale: null,

  // segmenet  
  disableHover: false,
  
  // transition
  transDuration: 700,

  // data field
  zoomXs: [],    // scoped time data
  zoomY : null,  // scoped worker
  data: null,
    
  timeFormat : function(d) {
    if(d >= 1e9) return `${(d/1e9).toFixed(2)}G`;
    if(d >= 1e6) return `${(d/1e6).toFixed(1)}M`;
    if(d >= 1e3) return `${(d/1e3).toFixed(0)}K`;
    return d;
  }
};

//const simple_file = "js/simple.json";
//const simple_file = "js/wb_dma.json";

async function fetchTFPData(file) {
  const response = await fetch(file);
  const json = await response.json();
  return json;
}

class Database {

  constructor (rawData, maxSegments=500) {

    this.data = [];
    this.maxSegments = maxSegments;
    this.indexMap = new Map();

    let numSegs = 0, minX = null, maxX = null, k=0;

    const begParse = performance.now();

    for (let i=0, ilen=rawData.length; i<ilen; i++) {
      const E = rawData[i].executor;
      for (let j=0, jlen=rawData[i].data.length; j<jlen; j++) {
        
        let slen = rawData[i].data[j].data.length;
        const W = rawData[i].data[j].worker;
        const L = rawData[i].data[j].level;

        this.data.push({
          executor: `${E}`,
          worker  : `E${E}.W${rawData[i].data[j].worker}.L${L}`,
          level   : `${L}`,
          segs    : rawData[i].data[j].data,
          range   : [0, slen]
        });
        
        if(slen > 0) {
          let b = rawData[i].data[j].data[0].span[0];
          let e = rawData[i].data[j].data[slen-1].span[1];
          if(minX == null || b < minX) minX = b;
          if(maxX == null || e > maxX) maxX = e; 
          numSegs += slen;
        }

        this.indexMap.set(`${this.data[this.data.length-1].worker}`, k);
        k = k+1;
      }
    }

    this.numSegs = numSegs;
    this.minX = minX;
    this.maxX = maxX;
  }

  query(zoomX = null, zoomY = null) {

    //fetch(`/query`, {
    //  method: 'post', 
    //  body: `{zoomX: "${zoomX}", zoomY: "${zoomY}"}`
    //}).then(function(response) {
    //  console.log(response);
    //  return response.json();
    //}).then(function(data) {
    //  console.log(data);
    //});
    //response.text().then(function(text) {
    //    console.log(text);
    //  });
    //});

    // default selection is the entire region
    if(zoomX == null) {
      zoomX = [this.minX, this.maxX];
    }
    
    if(zoomY == null) {
      //zoomY = [...Array(this.data.length).keys()]
      zoomY = d3.range(0, this.data.length);
    }
    else {
      zoomY = zoomY.map(d => this.indexMap.get(d));
    }

    //console.log(zoomY)

    console.assert(zoomX[0] <= zoomX[1]);

    let R = 0;
    let S = [];
    let G = [];
    
    // find out the segs in the range
    for(let y=0; y<zoomY.length; ++y) {

      const w = zoomY[y];

      const slen = this.data[w].segs.length;
            
      let l = null, r = null, beg, end, mid;

      // r = maxArg {span[0] <= zoomX[1]}
      beg = 0, end = slen;
      while(beg < end) {
        mid = (beg + end) >> 1;
        if(this.data[w].segs[mid].span[0] <= zoomX[1]) {
          beg = mid + 1;
          r = (r == null) ? mid : Math.max(mid, r);
        }
        else {
          end = mid;
        }
      }

      if(r == null) {
        this.data[w].range = [0, 0];
        continue;
      }

      // l = minArg {span[1] >= zoomX[0]}
      beg = 0, end = slen;
      while(beg < end) {
        mid = (beg + end) >> 1;
        if(this.data[w].segs[mid].span[1] >= zoomX[0]) {
          end = mid;
          l = (l == null) ? mid : Math.min(mid, l);
        }
        else {
          beg = mid + 1;
        }
      };

      if(l == null || l > r) {
        this.data[w].range = [0, 0];
        continue;
      }

      // range ok
      this.data[w].range = [l, r+1];
      R += (r+1-l);
      //console.log(`  ${this.data[w].worker} has ${r+1-l} segs`);

      for(let s=l; s<=r; s++) {
        if(s != r) {
          G.push({w:w, s:s, d: this.data[w].segs[s+1].span[0] - this.data[w].segs[s].span[1]});
        }
        this.data[w].segs[s].cluster = [s, s];
      }
    }

    G.sort((a, b) => a.d - b.d);

    let g = 0, remain = R;
    while(remain > this.maxSegments && g < G.length) {

      const w = G[g].w;
      const s = G[g].s;
      const a = this.data[w].segs[s].cluster;
      const b = this.data[w].segs[s+1].cluster;

      this.data[w].segs[a[0]].cluster[1] = b[1];
      this.data[w].segs[b[1]].cluster[0] = a[0];
      
      g++;
      remain--;
    }
    
    //console.log(this.data);

    let numDrawn = 0;

    for(let y=0; y<zoomY.length; ++y){
      
      let T=0, st=0, dt=0, gt=0, ct=0, mt=0, at=0;

      const w = zoomY[y];
      const N = this.data[w].range[1]-this.data[w].range[0];
      
      let b = this.data[w].range[0];
      let s = [];

      while(b < this.data[w].range[1]) {

        const e = this.data[w].segs[b].cluster[1];

        console.assert(b == this.data[w].segs[b].cluster[0]);
        console.assert(e <  this.data[w].segs.length);
        
        if(b == e) {
          s.push(this.data[w].segs[b]);
        }
        else {
          s.push({
            span: [this.data[w].segs[b].span[0], this.data[w].segs[e].span[1]],
            name: "-",
            type: "clustered",
            cluster: [b, e]
          });
        }

        for(let i=b; i<=e; i++) {
          const t = this.data[w].segs[i].span[1] - this.data[w].segs[i].span[0];
          T += t;
          // cumulate data
          switch(this.data[w].segs[i].type) {
            case "static"   : st += t; break;
            case "subflow"  : dt += t; break;
            case "cudaflow" : gt += t; break;
            case "condition": ct += t; break;
            case "module"   : mt += t; break;
            case "async"    : at += t; break;
            default         : console.assert(false); break;
          }
        }
        numDrawn++; 
        b = e+1;
      }

      let load = [], x=0;

      load.push({type: "static",    span: [x, x+st], ratio: (st/T*100).toFixed(2)}); 
      x += st;
      load.push({type: "subflow",   span: [x, x+dt], ratio: (dt/T*100).toFixed(2)}); 
      x += dt;
      load.push({type: "cudaflow",  span: [x, x+gt], ratio: (gt/T*100).toFixed(2)}); 
      x += gt;
      load.push({type: "condition", span: [x, x+ct], ratio: (ct/T*100).toFixed(2)}); 
      x += ct;
      load.push({type: "module",    span: [x, x+mt], ratio: (mt/T*100).toFixed(2)}); 
      x += mt;
      load.push({type: "async",     span: [x, x+at], ratio: (at/T*100).toFixed(2)}); 
      x += at;
      
      S.push({
        executor: this.data[w].executor,
        worker: this.data[w].worker,
        tasks: N,
        segs: s,
        load: load,
        totalTime: T
      });
    }
    return S;
  }

  minX() { return this.minX; }

  maxX() { return this.maxX; }

  numSegs() { return this.numSegs; }
}

function _adjustDim() {

  tfp.width = tfp.dom.style('width').slice(0, -2)
    - tfp.dom.style('padding-right').slice(0, -2)
    - tfp.dom.style('padding-left').slice(0, -2);
  
  tfp.segH = tfp.segMaxH * 0.8;
  tfp.segHOfst = (tfp.segMaxH - tfp.segH) / 2;

  const w = tfp.width - tfp.leftMargin - tfp.rightMargin - tfp.innerW;

  tfp.tlW = w*0.80;
  tfp.tlH = tfp.data.length * tfp.segMaxH;
  tfp.loadW = w - tfp.tlW;
  tfp.loadH = tfp.tlH;
  tfp.ovW = tfp.tlW;
  tfp.rankW = tfp.width - tfp.leftMargin - tfp.rightMargin;

  tfp.height = tfp.topMargin + tfp.tlH + 2*tfp.innerH + tfp.ovH + tfp.rankH + tfp.bottomMargin;

  tfp.svg.attr('width', tfp.width).attr('height', tfp.height);
}

function _render_tlXAxis() {

  tfp.tlXScale
    .domain(tfp.zoomXs[tfp.zoomXs.length - 1])
    .range([0, tfp.tlW]).clamp(true);

  tfp.tlXAxis.scale(tfp.tlXScale)
    .tickSizeOuter(0)
    .tickSize(-tfp.tlH)
    .tickFormat(tfp.timeFormat);
    //.tickFormat(d3.format('.2s'))
    //.ticks(numXTicks(tfp.tlW));
  
  tfp.tlG.select('g.tfp-tl-x-axis')
    .attr('transform', `translate(0, ${tfp.tlH})`)
    .transition().duration(tfp.transDuration)
      .call(tfp.tlXAxis)
      .attr('font-size', 16);
}

function _render_tlWAxis() {


  tfp.tlWScale
    //.domain(tfp.data.map(d=>`${d.executor}+&+${d.worker}`))
    .domain(tfp.data.map(d=>d.worker))
    .range([0, tfp.tlH]);

  tfp.tlWAxis.scale(tfp.tlWScale)
    .tickSizeOuter(0)
    //.tickFormat(d => {
    //  const wl = d.split('.L');
    //  return +wl[1] == 0 ? wl[0] : `↳L${wl[1]}`;
    //});
    .tickFormat(d => d);
  
  tfp.tlG.select('g.tfp-tl-w-axis')
    //.attr('transform', `translate(${tfp.tlW}, 0)`)
    .attr('transform', `translate(0, 0)`)
    .transition().duration(tfp.transDuration)
      .call(tfp.tlWAxis)
      .attr('font-size', 14);
  
  //tfp.tlG.select('g.tfp-tl-w-axis').selectAll('text')
  //  .on('click', d => console.log(d));
}

function _render_tlEAxis() {

  tfp.tlEScale.domain(tfp.eMeta.map(d=>d.executor));

  let cntLines = 0;

  tfp.tlEScale.range(tfp.eMeta.map(d => {
    const pos = (cntLines + d.workers.length/2)*tfp.segMaxH;
    cntLines += d.workers.length;
    return pos;
  }));
  
  //tfp.tlEAxis.scale(tfp.tlEScale).tickSizeOuter(0);
  
  //tfp.tlG.select('g.tfp-tl-e-axis')
  //  .transition().duration(tfp.transDuration)
  //    .call(tfp.tlEAxis)
  //    .attr('font-size', 14)
  //
  //tfp.tlG.select('g.tfp-tl-e-axis').selectAll('text')
  //  .on('click', d => console.log(d));
  
  // rect
  var ed1 = tfp.tlG.select('g.tfp-tl-e-rect').selectAll('rect').data(tfp.eMeta);

  ed1.exit().transition().duration(tfp.transDuration)
    .style('stroke-opacity', 0)
    .style('fill-opacity', 0)
    .remove();
  
  ed1 = ed1.merge(ed1.enter().append('rect')
    .attr('x', 0).attr('y', 0).attr('height', 0).attr('width', 0)
    //.on('mouseover', tfp.executorTooltip.show)
    //.on('mouseout', tfp.executorTooltip.hide);
  );

  ed1.transition().duration(tfp.transDuration)
    .attr('width', tfp.tlW)
    .attr('height', d => tfp.segMaxH * d.workers.length)
    .attr('y', d => tfp.tlEScale(d.executor) - d.workers.length/2*tfp.segMaxH);
}

function _render_tlZAxis() {

  let zGroup = tfp.tlG.select('g.tfp-tl-legend');
  zGroup.attr('transform', `translate(0, ${-tfp.segMaxH})`);

  tfp.tlZScale.domain(Array.from(tfp.zColorMap.keys()));
  tfp.tlZScale.range(Array.from(tfp.zColorMap.values()));

  const zW = tfp.width - tfp.leftMargin - tfp.rightMargin;
  const zH = tfp.segMaxH;
  const binW = zW / tfp.tlZScale.domain().length;

  let slot = zGroup.selectAll('g').data(tfp.tlZScale.domain());

  slot.exit().remove();

  const newslot = slot.enter().append('g')
    .attr('transform', (d, i) => `translate(${binW * i}, -5)`);

  newslot.append('rect');
  newslot.append('text');
  //  .on('click', d => console.log("click legend", d));
  
  slot = slot.merge(newslot);

  slot.select('rect')
    .attr('width', binW)
    .attr('height', zH)
    .attr('fill', d => tfp.tlZScale(d));

  slot.select('text')
    .text(d => d).attr('x', binW*0.5).attr('y', zH*0.5)
    .style('font-size', 14);
}

function _render_tlSegs() {

  var sd1 = tfp.tlG.select('g.tfp-tl-graph').selectAll('g').data(tfp.data)
  sd1.exit().remove();
  sd1 = sd1.merge(sd1.enter().append('g'));

  sd1.attr('transform', (d, i) => `translate(0, ${i*tfp.segMaxH})`);

  var sd2 = sd1.selectAll('rect').data(d => d.segs);
  sd2.exit().remove();

  sd2 = sd2.merge(sd2.enter().append('rect')
    .attr('rx', 1)
    .attr('ry', 1)
    .attr('x', tfp.tlW/2)    // here we initialize the rect to avoid
    .attr('y', tfp.tlH/2)    // NaN y error during transition
    .attr('width', 0)
    .attr('height', 0)
    .style('fill-opacity', 0)
    .on('mouseover.tlTooltip', tfp.tlTooltip.show)
    .on('mouseout.tlTooltip', tfp.tlTooltip.hide)
    .on("mouseover", function(d) {

      if (tfp.disableHover) return;

      const r = tfp.segHOfst; // enlarge ratio

      d3.select(this).transition().duration(250).style('fill-opacity', 1)
        .attr("width", d => d3.max([1, tfp.tlXScale(d.span[1])-tfp.tlXScale(d.span[0])]) + r)
        .attr('height', tfp.segH+r)
        .attr('x', d => tfp.tlXScale(d.span[0])-r/2)
        .attr('y', d => tfp.segHOfst-r/2);
    })
    .on("mouseout", function(d) {
      d3.select(this).transition().duration(250).style('fill-opacity', .8)
        .attr("width", d => d3.max([1, tfp.tlXScale(d.span[1])-tfp.tlXScale(d.span[0])]))
        .attr('height', tfp.segH)
        .attr('x', d => tfp.tlXScale(d.span[0]))
        .attr('y', d => tfp.segHOfst)
        .style('fill', d => tfp.zColorMap.get(d.type));
    })
    .on('click', d=>{
      const zoomX = d.span;
      //console.log("zoom to ", zoomX);
      tfp.zoomXs.push(zoomX);
      _onZoomX(zoomX, true);
    })
  );

  sd2.transition().duration(tfp.transDuration)
    .attr('x', d => tfp.tlXScale(d.span[0]))
    .attr('width', d => d3.max([1, tfp.tlXScale(d.span[1])-tfp.tlXScale(d.span[0])]))
    .attr('y', tfp.segHOfst)
    .attr('height', tfp.segH)
    .style('fill-opacity', .8)
    .style('fill', d => tfp.zColorMap.get(d.type));
}

function _render_tlBrush() {
  tfp.tlG.select('g.tfp-tl-brush')
    .call(tfp.tlBrush.extent([[0, 0], [tfp.tlW, tfp.tlH]]));
}

function _render_tl() {

  tfp.tlG.attr('transform', `translate(${tfp.leftMargin}, ${tfp.topMargin})`);

  _render_tlXAxis();  // x-axis
  _render_tlWAxis();  // w-axis
  _render_tlEAxis();  // e-axis
  _render_tlZAxis();  // z-axis
  _render_tlSegs();   // s-rect
  _render_tlBrush();  // brush
  
}

function _render_ovXAxis() {
  tfp.ovXScale.domain(tfp.ovXDomain).range([0, tfp.ovW]).clamp(true);

  tfp.ovXAxis.scale(tfp.ovXScale)
    .tickSizeOuter(0)
    .tickSize(-tfp.ovH)
    .tickFormat(tfp.timeFormat);

  tfp.ovG.select('g.tfp-ov-x-axis')
    .attr('transform', `translate(0, ${tfp.ovH})`)
    .transition().duration(tfp.transDuration)
      .call(tfp.ovXAxis)
      .attr('font-size', 16);

  tfp.ovG.select('.tfp-ov-bg').attr('width', tfp.ovW).attr('height', tfp.ovH);
}

function _render_ovBrush() {
  tfp.ovG.select('g.tfp-ov-brush')
    .call(tfp.ovBrush.extent([[0, 0], [tfp.ovW, tfp.ovH]]))
    .call(tfp.ovBrush.move, tfp.ovXSel.map(tfp.ovXScale))
}

function _render_ovInfo() {
  tfp.ovG.select('.tfp-ov-info')
    .attr('x', tfp.ovW + tfp.innerW)
    .attr('y', tfp.ovH/2)
    .attr('font-size', 16)
    .html(`&#10144; ${d3.sum(tfp.data, d=>d.tasks)} tasks`);
}

function _render_ov() {
  tfp.ovG.attr('transform', 
    `translate(${tfp.leftMargin}, ${tfp.topMargin + tfp.tlH + tfp.innerH})`
  );
  console.assert(tfp.ovXDomain && tfp.ovXSel);
  _render_ovXAxis();
  _render_ovBrush();
  _render_ovInfo();
}

function _render_loadXAxis() {

  tfp.loadXScale
    .domain([0, d3.max(tfp.data, d=>d.totalTime)])
    .range([0, tfp.loadW]).clamp(true);

  tfp.loadXAxis.scale(tfp.loadXScale)
    .tickSizeOuter(0)
    .tickSize(-tfp.loadH)
    .tickFormat(tfp.timeFormat)
    .ticks(numXTicks(tfp.loadW));

  tfp.loadG.select('g.tfp-load-x-axis')
    .attr('transform', `translate(0, ${tfp.loadH})`)
    .transition().duration(tfp.transDuration)
      .call(tfp.loadXAxis)
      .attr('font-size', 16);
}

function _render_loadEAxis() {

  tfp.loadEScale.domain(tfp.eMeta.map(d=>d.executor));

  let cntLines = 0;

  tfp.loadEScale.range(tfp.eMeta.map(d => {
    const pos = (cntLines + d.workers.length/2)*tfp.segMaxH;
    cntLines += d.workers.length;
    return pos;
  }));
  
  //tfp.loadEAxis.scale(tfp.loadEScale).tickSizeOuter(0);
  //
  //tfp.loadG.select('g.tfp-load-e-axis')
  //  .transition().duration(tfp.transDuration)
  //    .call(tfp.loadEAxis)
  //    .attr('font-size', 14)
  //
  //tfp.loadG.select('g.tfp-load-e-axis').selectAll('text')
  //  .on('click', d => console.log(d));
  
  // rect
  var ed1 = tfp.loadG.select('g.tfp-load-e-rect').selectAll('rect').data(tfp.eMeta);

  ed1.exit().transition().duration(tfp.transDuration)
    .style('stroke-opacity', 0)
    .style('fill-opacity', 0)
    .remove();
  
  ed1 = ed1.merge(ed1.enter().append('rect')
    .attr('x', 0).attr('y', 0).attr('height', 0).attr('width', 0)
    //.on('mouseover', tfp.executorTooltip.show)
    //.on('mouseout', tfp.executorTooltip.hide);
  );

  ed1.transition().duration(tfp.transDuration)
    .attr('width', tfp.loadW)
    .attr('height', d => tfp.segMaxH * d.workers.length)
    .attr('y', d => tfp.loadEScale(d.executor) - d.workers.length/2*tfp.segMaxH);
}

function _render_loadGraph() {
  
  var sd1 = tfp.loadG.select('g.tfp-load-graph').selectAll('g').data(tfp.data)
  sd1.exit().remove();
  sd1 = sd1.merge(sd1.enter().append('g'));

  sd1.attr('transform', (d, i) => `translate(0, ${i*tfp.segMaxH})`);

  var sd2 = sd1.selectAll('rect').data(d => d.load);
  sd2.exit().remove();

  sd2 = sd2.merge(sd2.enter().append('rect')
    .attr('x', 0)    // here we initialize the rect to avoid
    .attr('y', 0)    // NaN y error during transition
    .attr('width', 0)
    .attr('height', 0)
    .style('fill-opacity', 0)
    .on('mouseover.loadTooltip', tfp.loadTooltip.show)
    .on('mouseout.loadTooltip', tfp.loadTooltip.hide)
    .on("mouseover", function(d) {

      if (tfp.disableHover) return;

      const r = tfp.segHOfst; // enlarge ratio

      d3.select(this).transition().duration(250).style('fill-opacity', 1)
        .attr("width", d => d3.max([1, tfp.loadXScale(d.span[1])-tfp.loadXScale(d.span[0])]) + r)
        .attr('height', tfp.segH+r)
        .attr('x', d => tfp.loadXScale(d.span[0])-r/2)
        .attr('y', d => tfp.segHOfst-r/2);
    })
    .on("mouseout", function(d) {
      d3.select(this).transition().duration(250).style('fill-opacity', .8)
        .attr("width", d => d3.max([1, tfp.loadXScale(d.span[1])-tfp.loadXScale(d.span[0])]))
        .attr('height', tfp.segH)
        .attr('x', d => tfp.loadXScale(d.span[0]))
        .attr('y', d => tfp.segHOfst)
        .style('fill', d => tfp.zColorMap.get(d.type));
    })
  );

  sd2.transition().duration(tfp.transDuration)
    .attr('x', d => tfp.loadXScale(d.span[0]))
    .attr('width', d => d3.max([1, tfp.loadXScale(d.span[1])-tfp.loadXScale(d.span[0])]))
    .attr('y', tfp.segHOfst)
    .attr('height', tfp.segH)
    .style('fill-opacity', .8)
    .style('fill', d => tfp.zColorMap.get(d.type));
}

function _render_load() {

  tfp.loadG.attr('transform', 
    `translate(${tfp.width-tfp.rightMargin-tfp.loadW}, ${tfp.topMargin})`
  );

  _render_loadXAxis();  
  _render_loadEAxis();
  _render_loadGraph();
}

function _render_rankGraph() {
    
  var from = Math.max(0, Math.round($('#tfp_menu_rank_from').val()) - 1);
  var to   = Math.round($('#tfp_menu_rank_to').val());
  var limit = (from >= to) ? [0, 50] : [from, to]

  // process data
  let rank = [];
  for(let w=0; w<tfp.data.length; w++) {
    for(let s=0; s<tfp.data[w].segs.length; s++) {
      rank.push([w, s, tfp.data[w].segs[s].span[1] - tfp.data[w].segs[s].span[0]]);
      //for(let c=tfp.data[w].segs[s].cluster[0]; c<=tfp.data[w].segs[s].cluster[1]; c++) {
      //  rank.push([w, c, tfp.db.data[w].segs[c].span[1] - tfp.db.data[w].segs[c].span[0]]);
      //}
    }
  }

  rank = rank.sort((a, b) => b[2] - a[2]).slice(limit[0], limit[1]);
  
  // x-axis
  tfp.rankXScale.domain(rank).range([0, tfp.rankW]).padding(0.2);

  tfp.rankXAxis.scale(tfp.rankXScale).tickFormat('');

  tfp.rankG.select('g.tfp-rank-x-axis')
    .attr('transform', `translate(0, ${tfp.rankH})`)
    .transition().duration(tfp.transDuration)
      .call(tfp.rankXAxis);
  
  // y-axis
  const maxY = rank.length ? rank[0][2] : 0;

  tfp.rankYScale.domain([0, maxY]).range([tfp.rankH, 0]);

  tfp.rankYAxis.scale(tfp.rankYScale)
    .tickFormat(tfp.timeFormat)
    .tickSize(-tfp.rankW);

  tfp.rankG.select('g.tfp-rank-y-axis')
    //.transition().duration(tfp.transDuration)
      .call(tfp.rankYAxis)
      .attr('font-size', 14);

  var bars = tfp.rankG.select('g.tfp-rank-graph').selectAll('rect').data(rank);

  bars.exit().remove();

  let newBars = bars.enter().append('rect')
    .attr('x', 0)
    .attr('y', 0)
    .attr('width', 0)
    .attr('height', 0)
    .style('fill-opacity', 0)
    .on('mouseover.rankTooltip', tfp.rankTooltip.show)
    .on('mouseout.rankTooltip', tfp.rankTooltip.hide)
    .on("mouseover", function(d) {
      if (tfp.disableHover) return;
      d3.select(this).transition().duration(250).style('fill-opacity', 1)
        .attr('x', tfp.rankXScale)
        .attr('y', d=>tfp.rankYScale(d[2]))
        .attr('width', tfp.rankXScale.bandwidth())
        .attr('height', d=>tfp.rankH-tfp.rankYScale(d[2]))
        .style('fill', d => tfp.zColorMap.get(tfp.data[d[0]].segs[d[1]].type));
    })
    .on("mouseout", function(d) {
      d3.select(this).transition().duration(250).style('fill-opacity', .8)
        .attr('x', tfp.rankXScale)
        .attr('y', d=>tfp.rankYScale(d[2]))
        .attr('width', tfp.rankXScale.bandwidth())
        .attr('height', d=>tfp.rankH-tfp.rankYScale(d[2]))
        .style('fill', d => tfp.zColorMap.get(tfp.data[d[0]].segs[d[1]].type));
    });

  bars = bars.merge(newBars);
  
  bars.transition().duration(tfp.transDuration)
    .attr('x', tfp.rankXScale)
    .attr('y', d=>tfp.rankYScale(d[2]))
    .attr('width', tfp.rankXScale.bandwidth())
    .attr('height', d=>tfp.rankH-tfp.rankYScale(d[2]))
    .style('fill-opacity', .8)
    .style('fill', d => tfp.zColorMap.get(tfp.data[d[0]].segs[d[1]].type));

  // xlabel
  tfp.rankG.select('text.tfp-rank-label')
    .html(`Top ${limit[0]+1}-${limit[0]+rank.length} Critical Tasks`);
}

function _render_rank() {

  tfp.rankG.attr('transform', 
    `translate(${tfp.leftMargin}, ${tfp.topMargin + tfp.tlH + 2*tfp.innerH + tfp.ovH})`
  );

  tfp.rankG.select('text.tfp-rank-label')
    .attr('transform', `translate(${tfp.rankW/2}, ${tfp.rankH + tfp.innerH/2})`);

  _render_rankGraph();
}

function _onZoomX(zoomX, refreshBrush) {

  queryData(zoomX, tfp.zoomY);
  _render_tlXAxis();
  _render_tlSegs();
  _render_loadXAxis();
  _render_loadGraph();
  _render_ovInfo();
  _render_rankGraph();

  if(refreshBrush) {
    tfp.ovXSel = zoomX;
    _render_ovBrush();
  }
}

function queryData(zoomX, zoomY) {

  tfp.data = tfp.db.query(zoomX, zoomY);

  let eMeta = tfp.data.reduce((res, item) => {
    res[item.executor] = [...res[item.executor] || [], item.worker];
    return res;
  }, {});

  tfp.eMeta = Object.keys(eMeta).map(e => { return {executor: e, workers: eMeta[e]} });
}

function numXTicks(W) {
  return Math.max(2, Math.min(12, Math.round(W * 0.012)));
}

async function main() {
  
  // initialize static field
  tfp.dom = d3.select('#tfp');
  tfp.svg = tfp.dom.append('svg').attr('width', 0).attr('height', 0);

  tfp.tlG = tfp.svg.append('g').attr('class', 'tfp-tl-group');
  tfp.tlG.append('g').attr('class', 'tfp-tl-legend');
  tfp.tlG.append('g').attr('class', 'tfp-tl-x-axis');
  tfp.tlG.append('g').attr('class', 'tfp-tl-w-axis');
  tfp.tlG.append('g').attr('class', 'tfp-tl-e-axis');
  tfp.tlG.append('g').attr('class', 'tfp-tl-e-rect');  // layer 1
  tfp.tlG.append('g').attr('class', 'tfp-tl-brush');   // layer 2
  tfp.tlG.append('g').attr('class', 'tfp-tl-graph');  // layer 3

  tfp.ovG = tfp.svg.append('g').attr('class', 'tfp-ov-group');
  tfp.ovG.append('rect').attr('class', 'tfp-ov-bg');
  tfp.ovG.append('g').attr('class', 'tfp-ov-x-axis');
  tfp.ovG.append('g').attr('class', 'tfp-ov-brush');
  tfp.ovG.append('text').attr('class', 'tfp-ov-info');
  
  tfp.loadG = tfp.svg.append('g').attr('class', 'tfp-load-group');
  tfp.loadG.append('g').attr('class', 'tfp-load-x-axis');
  tfp.loadG.append('g').attr('class', 'tfp-load-e-rect');  // layer 1
  tfp.loadG.append('g').attr('class', 'tfp-load-graph');   // layer 2

  tfp.rankG = tfp.svg.append('g').attr('class', 'tfp-rank-group');
  tfp.rankG.append('g').attr('class', 'tfp-rank-x-axis');
  tfp.rankG.append('g').attr('class', 'tfp-rank-y-axis');
  tfp.rankG.append('g').attr('class', 'tfp-rank-graph');
  tfp.rankG.append('text').attr('class', 'tfp-rank-label');

  // tl brush event
  tfp.tlBrush.on('brush', function() {
    const s = d3.event.selection;
    if(s) {
      tfp.ovXSel = s.map(x=>tfp.tlXScale.invert(x));
      _render_ovBrush();
    }
  })
  .on('end', function() { 

    if(!d3.event.sourceEvent) return;

    const s = d3.event.selection;
    
    // Consume the brush action
    if (s) {
      const zoomX = s.map(tfp.tlXScale.invert);
      //console.log("zoom to ", zoomX);
      tfp.zoomXs.push(zoomX);
      _onZoomX(zoomX, false);

      tfp.tlG.select("g.tfp-tl-brush").call(tfp.tlBrush.move, null);
    }
    // double-click to the previous cache
    else { 
      if (!tfp.tlBrushTimeout) {
        return (tfp.tlBrushTimeout = setTimeout(() => {
          tfp.tlBrushTimeout = null;
        }, 350));
      }
      
      if(tfp.zoomXs.length > 1) {
        let zoomX = tfp.zoomXs.pop();
        while(tfp.zoomXs.length > 1) {
          if(zoomX[0] == tfp.zoomXs[tfp.zoomXs.length-1][0] &&
             zoomX[1] == tfp.zoomXs[tfp.zoomXs.length-1][1]) {
            tfp.zoomXs.pop();
          }
          else break;
        }
        _onZoomX(tfp.zoomXs[tfp.zoomXs.length-1], true);
      }
    }
  });

  // ov brush event
  tfp.ovBrush.on('end', function() {
    if(d3.event.sourceEvent && d3.event.selection) {
      if(d3.event.sourceEvent.type === "mouseup") {
        const zoomX = d3.event.selection.map(tfp.ovXScale.invert);
        //console.log("ovBrush fires zoomX", zoomX);
        tfp.zoomXs.push(zoomX);
        _onZoomX(zoomX, false);
      }
    }
  });
  
  // tl tooltips
  tfp.tlTooltip = d3.tip()
    .attr('class', 'tfp-tooltip')
    .direction('s')
    .offset([10, 0])
    .html(d => {
      return `Type: ${d.type}<br>
              Name: ${d.name}<br>
              Span: [${d.span}]<br>
              Time: ${d.span[1]-d.span[0]}`;
    });
  
  tfp.svg.call(tfp.tlTooltip);

  // load tooltip
  tfp.loadTooltip = d3.tip()
    .attr('class', 'tfp-tooltip')
    .direction('s')
    .offset([10, 0])
    .html(d => {
      //const p = ((d[1]-d[0]) * 100 / (d.data.totalTime)).toFixed(2);
      return `Type: ${d.type}<br>
              Total Time: ${d.span[1]-d.span[0]}<br>
              Ratio: ${d.ratio}%`;
    });

  tfp.svg.call(tfp.loadTooltip);

  // rank tooltip
  tfp.rankTooltip = d3.tip()
    .attr('class', 'tfp-tooltip')
    .direction('s')
    .offset([10, 0])
    .html(d=> {
      return `Worker: ${tfp.data[d[0]].worker}<br>
              Type: ${tfp.data[d[0]].segs[d[1]].type}<br>
              Name: ${tfp.data[d[0]].segs[d[1]].name}<br>
              Time: ${d[2]}`
    });

  tfp.svg.call(tfp.rankTooltip);
  
  //let begFetch = performance.now();
  //const res = await fetchTFPData(simple_file);
  //let endFetch = performance.now();

  render_simple();
}

function _adjust_menu() {
  
  // worker menu
  var wmenu = d3.select('#tfp_menu_workers').selectAll('a').data(tfp.db.data);

  wmenu.selectAll('input').remove();
  wmenu.selectAll('label').remove();

  wmenu.exit().remove();
  
  wmenu = wmenu.merge(wmenu.enter().append('a')
    .attr('class', 'dropdown-item')
    //.attr('data-value', d => d.worker)
    //.attr('tabIndex', '-1')
    //.on('mouseover', tfp.executorTooltip.show)
    //.on('mouseout', tfp.executorTooltip.hide);
  );
  
  wmenu.append('input')
    .attr('type', 'checkbox')
    .attr('class', 'mr-2')
    .attr('value', d=>d.worker)
    .attr('id', d=>d.worker)
    .property('checked', true)
    .attr('name', 'worker');

  wmenu.append('label').attr('for', d=>d.worker).text(d => {
    const wl = d.worker.split('.');
    return `${d.worker} (Executor ${wl[0]} / Worker ${wl[1]} @ Level ${wl[2]})`
  });
  
  // rank menu
  document.getElementById('tfp_menu_rank_from').value = 1;
  document.getElementById('tfp_menu_rank_to').value = 50;
}

function feed(input) {

  // database wide
  tfp.db = new Database(input);
  tfp.ovXDomain = [tfp.db.minX, tfp.db.maxX];
  tfp.ovXSel = [tfp.db.minX, tfp.db.maxX];
  tfp.zoomXs = [[tfp.db.minX, tfp.db.maxX]];  // clear cached data
  tfp.zoomY  = Array.from(tfp.db.indexMap.keys())

  _adjust_menu();
  
  // data wide
  queryData(tfp.zoomXs[tfp.zoomXs.length-1], tfp.zoomY);
  _adjustDim();
  _render_tl();
  _render_load();
  _render_ov();
  _render_rank();
}

function render_simple() {
  $('#tfp_textarea').text(JSON.stringify(simple));
  feed(simple);
}

function render_composition() {
  $('#tfp_textarea').text(JSON.stringify(composition));
  feed(composition);
}

function render_inference() {
  $('#tfp_textarea').text(JSON.stringify(inference))
  feed(inference);
}

function render_dreamplace() {
  $('#tfp_textarea').text(JSON.stringify(dreamplace))
  feed(dreamplace);
}

main();

// ---- jquery ----

$('#tfp_composition').on('click', function() {
  render_composition();
})

$('#tfp_inference').on('click', function() {
  render_inference();
})

$('#tfp_dreamplace').on('click', function() {
  render_dreamplace();
})

// textarea changer event
$('#tfp_textarea').on('input propertychange paste', function() {

  if($(this).data('timeout')) {
    clearTimeout($(this).data('timeout'));
  }

  $(this).data('timeout', setTimeout(()=>{
    
    var text = $('#tfp_textarea').val().trim();
    
    $('#tfp_textarea').removeClass('is-invalid');

    if(!text) {
      return;
    }
    
    try {
      var json = JSON.parse(text);
      //console.log(json);
      feed(json);
    }
    catch(e) {
      $('#tfp_textarea').addClass('is-invalid');
      console.error(e);
    }

  }, 2000));
});


$('#tfp_menu_workers').on('click', function( event ) {
  
  event.stopPropagation();  // keep dropdown alive

  //console.log("dropdown")

  var zoomY = [];
  $.each($("input[name='worker']:checked"), function(){
    zoomY.push($(this).val());
  });

  //console.log(zoomY.join(', '))
  
  //tfp.ovXDomain = [tfp.db.minX, tfp.db.maxX];
  tfp.ovXSel = tfp.zoomXs[tfp.zoomXs.length-1];
  //tfp.zoomXs = [[tfp.db.minX, tfp.db.maxX]];  // clear cached data
  tfp.zoomY  = zoomY;

  queryData(tfp.zoomXs[tfp.zoomXs.length-1], tfp.zoomY);

  //console.log(tfp.data)

  _adjustDim();
  _render_tl();
  _render_load();
  _render_ov();
  _render_rank();
});

$('#tfp_menu_rank').on('input', function (event) {
  
  event.stopPropagation();
  
  if($(this).data('timeout')) {
    clearTimeout($(this).data('timeout'));
  }

  // get selected option and change background
  $(this).data('timeout', setTimeout(()=>{
    _render_rankGraph();
  }, 1000));
})

$('#tfp_menu_reset_zoom').on('click', function() {
  tfp.zoomXs = [[tfp.db.minX, tfp.db.maxX]];  // clear cached data
  _onZoomX(tfp.zoomXs[tfp.zoomXs.length-1], true);
})






