// Program: tfprof
// Author: Dr. Tsung-Wei Huang
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
    ['subflow', '#F1A42B'],
    ['condition', '#42B649'],
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
  zoomXs: [],       // scoped time data
  zoomY : null,     // scoped worker
  view  : "Cluster",  // default view type
  limit : 256,
  maxLimit: 512,
  data: null,
  tfpFile: null,
  numTasks: null,
  numExecutors: null,
  numWorkers: null, 
  
  timeFormat : function(d) {
    if(d >= 1e9) return `${(d/1e9).toFixed(2)}G`;
    if(d >= 1e6) return `${(d/1e6).toFixed(1)}M`;
    if(d >= 1e3) return `${(d/1e3).toFixed(0)}K`;
    return d;
  }
};

async function fetchTFPData(file) {
  const response = await fetch(file);
  const json = await response.json();
  return json;
}

async function queryInfo() {

  const response = await fetch(`/queryInfo`, {
    method: 'put'
  });
  
  let info = await response.json();

  tfp.tfpFile = info.tfpFile;
  tfp.numTasks = info.numTasks;
  tfp.numExecutors = info.numExecutors;
  tfp.numWorkers = info.numWorkers;
  tfp.maxLimit = Math.max(1, Math.min(tfp.maxLimit, tfp.numTasks));
  tfp.limit = Math.min(tfp.limit, tfp.maxLimit)
}

async function queryData(zoomX, zoomY, view, limit) {

  //console.log(zoomY);
  //$('#tfp_tb_loader').css("display", "block");

  const response = await fetch(`/queryData`, {
    method: 'put', 
    body: JSON.stringify({
      zoomX: zoomX,
      zoomY: zoomY,
      view : view,
      limit: limit
    })
  });

  tfp.data = await response.json();

  //console.log(tfp.data);
  
  let eMeta = tfp.data.reduce((res, item) => {
    res[item.executor] = [...res[item.executor] || [], item.worker];
    return res;
  }, {});

  tfp.eMeta = Object.keys(eMeta).map(e => { 
    return {executor: e, workers: eMeta[e]} 
  });
  
  //$('#tfp_tb_loader').css("display", "none");
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
    .tickFormat(tfp.timeFormat)
    //.tickFormat(d3.format('.2s'))
    .ticks(numXTicks(tfp.tlW));
  
  tfp.tlG.select('g.tfp-tl-x-axis')
    .attr('transform', `translate(0, ${tfp.tlH})`)
    .transition().duration(tfp.transDuration)
      .call(tfp.tlXAxis)
      .attr('font-size', 16);
}

function _render_tlWAxis() {

  tfp.tlWScale
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
    .on('click', async function(d) {
      const zoomX = d.span;
      //console.log("zoom to ", zoomX);
      tfp.zoomXs.push(zoomX);
      await _onZoomX(zoomX, true);
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
    .tickFormat(tfp.timeFormat)
    .ticks(numXTicks(tfp.tlW));

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
    .html(`${d3.sum(tfp.data, d=>d.tasks)} tasks`);
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
    
  //var from = Math.max(0, Math.round($('#tfp_tb_rank_from').val()) - 1);
  //var to   = Math.round($('#tfp_tb_rank_to').val());
  //var limit = (from >= to) ? [0, 50] : [from, to]

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

  //rank = rank.sort((a, b) => b[2] - a[2]).slice(limit[0], limit[1]);
  rank = rank.sort((a, b) => b[2] - a[2]);
  
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
    .html(`Top-${rank.length} Critical Tasks`);
}

function _render_rank() {

  tfp.rankG.attr('transform', 
    `translate(${tfp.leftMargin}, ${tfp.topMargin + tfp.tlH + 2*tfp.innerH + tfp.ovH})`
  );

  tfp.rankG.select('text.tfp-rank-label')
    .attr('transform', `translate(${tfp.rankW/2}, ${tfp.rankH + tfp.innerH/2})`);

  _render_rankGraph();
}

async function _onZoomX(zoomX, refreshBrush) {

  await queryData(zoomX, tfp.zoomY, tfp.view, tfp.limit);
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

function numXTicks(W) {
  return Math.max(2, Math.min(12, Math.round(W * 0.012)));
}


function _adjust_tb() {
  
  // flattern the executors and workers
  let flat = [];

  for(let l=0; l<tfp.eMeta.length; l++) {
    flat.push(`E${tfp.eMeta[l].executor}`)
    flat.push(...tfp.eMeta[l].workers)
  }

  // worker tb
  var wtb = d3.select('#tfp_tb_workers').selectAll('a').data(flat);

  wtb.selectAll('input').remove();
  wtb.selectAll('label').remove();

  wtb.exit().remove();
  
  wtb = wtb.merge(wtb.enter().append('a')
    .attr('class', 'dropdown-item')
  );
  
  wtb.append('input')
    .attr('type', 'checkbox')
    .attr('class', d=>{ return d.split('.').length == 1 ? 'mr-2' : 'mr-2 ml-4'})
    .attr('value', d=>d)
    .attr('id', d=>d)
    .property('checked', true)
    .attr('name', d=>{ return d.split('.').length == 1 ? 'executor' : 'worker'});

  wtb.append('label').attr('for', d=>d).text(d => {
    const wl = d.split('.');
    if(wl.length == 1) {
      return `Executor ${wl[0]}`;
    }
    else {
      return `${d} (Worker ${wl[1]} @ Level ${wl[2]})`;
    }
  });

  // limit tb
  let numTasks = d3.sum(tfp.data, d=>d.segs.length);
  let limit = document.getElementById('tfp_tb_limit');
  limit.min = 1;
  limit.max = tfp.maxLimit;
  limit.value = tfp.limit;
  
  // rank tb
  //document.getElementById('tfp_tb_rank_from').value = 1;
  //document.getElementById('tfp_tb_rank_to').value = 50;
}

function _render(adjustDim) {
  if(adjustDim) {
    _adjustDim();
  }
  _render_tl();
  _render_load();
  _render_ov();
  _render_rank();
}

async function main() {
  
  await queryInfo();
  
  $('#tfp_tb_finfo').text(tfp.tfpFile.split(/(\\|\/)/g).pop()); // filename
  $('#tfp_tb_tinfo').text(`${tfp.numTasks} tasks`);
  $('#tfp_tb_einfo').text(`${tfp.numExecutors} executors`);
  $('#tfp_tb_winfo').text(`${tfp.numWorkers} workers`);

  await queryData(null, null, tfp.view, tfp.limit);

  let minX = null, maxX = null;

  for(let i=0; i<tfp.data.length; i++) {
    let l = tfp.data[i].segs.length;
    if(l > 0) {
      if(minX == null || tfp.data[i].segs[0].span[0] < minX) {
        minX = tfp.data[i].segs[0].span[0];
      }
      if(maxX == null || tfp.data[i].segs[l-1].span[1] > maxX) {
        maxX = tfp.data[i].segs[l-1].span[1];
      }
    }
  }
  
  tfp.ovXDomain = [minX, maxX];
  tfp.ovXSel = [minX, maxX];
  tfp.zoomXs = [[minX, maxX]];
  tfp.zoomY = tfp.data.map(d=>d.worker);
  
  // adjust the margin to the current nav
  tfp.topMargin += $('nav').outerHeight(true);
  
  // clean-up the loader 
  $('#tfp_loader').css("display", "none");
  //$('#tfp_toolbar').css("display", "block")
  $('#tfp').css("display", "block");
  
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
  .on('end', async function() { 

    if(!d3.event.sourceEvent) return;

    const s = d3.event.selection;
    
    // Consume the brush action
    if (s) {
      const zoomX = s.map(tfp.tlXScale.invert);
      //console.log("zoom to ", zoomX);
      tfp.zoomXs.push(zoomX);
      await _onZoomX(zoomX, false);

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
        await _onZoomX(tfp.zoomXs[tfp.zoomXs.length-1], true);
      }
    }
  });

  // ov brush event
  tfp.ovBrush.on('end', async function() {
    if(d3.event.sourceEvent && d3.event.selection) {
      if(d3.event.sourceEvent.type === "mouseup") {
        const zoomX = d3.event.selection.map(tfp.ovXScale.invert);
        //console.log("ovBrush fires zoomX", zoomX);
        tfp.zoomXs.push(zoomX);
        await _onZoomX(zoomX, false);
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
  
  _adjust_tb();
  _render(true)
  
  // set up the jquery
  $("#tfp_tb_workers").on('click', function(event){
    event.stopPropagation();
  });

  $("#tfp_tb_workers input[name='executor']").on('click', async function(event) {
    event.stopPropagation();
    //console.log("executor click!!", $(this).val(), $(this).is(':checked'));
    
    let v = $(this).is(':checked');
    let e = $(this).val();
    
    $.each($("#tfp_tb_workers input[name='worker']"), function(){
      if($(this).val().split('.')[0] == e) {
        $(this).prop('checked', v);
      }
    });
    
    var zoomY = [];
    $.each($("input[name='worker']:checked"), function(){
      zoomY.push($(this).val());
    });
    
    tfp.ovXSel = tfp.zoomXs[tfp.zoomXs.length-1];
    tfp.zoomY  = zoomY;

    await queryData(tfp.zoomXs[tfp.zoomXs.length-1], tfp.zoomY, tfp.view, tfp.limit);
    
    _render(true);
  });
  
  $("#tfp_tb_workers input[name='worker']").on('click', async function(event) {
    event.stopPropagation();
    //console.log("worker click!!", $(this).val(), $(this).is(':checked'));
    
    let v = $(this).is(':checked');
    let e = $(this).val().split('.')[0];
    
    // check or uncheck the executor
    let m = true;
    $.each($("#tfp_tb_workers input[name='worker']"), function(){
      if($(this).val().split('.')[0] == e && !$(this).is(':checked')) {
        m = false;
      }
    });
    $(`#${e}`).prop('checked', m);
  
    var zoomY = [];
    $.each($("input[name='worker']:checked"), function(){
      zoomY.push($(this).val());
    });
    
    tfp.ovXSel = tfp.zoomXs[tfp.zoomXs.length-1];
    tfp.zoomY  = zoomY;

    await queryData(tfp.zoomXs[tfp.zoomXs.length-1], tfp.zoomY, tfp.view, tfp.limit);
    
    _render(true);
  });

  $('#tfp_tb_view a').on('click', async function() {
  
    //event.stopPropagation();  // keep dropdown alive
  
    tfp.view = $(this).text();
  
    //console.log($(this).siblings('.hidden').text());
    $(this).parent().siblings('.btn').text($(this).text());
  
    tfp.ovXSel = tfp.zoomXs[tfp.zoomXs.length-1];
    
    await queryData(tfp.zoomXs[tfp.zoomXs.length-1], tfp.zoomY, tfp.view, tfp.limit);
  
    _render(false);
  });
  
  $('#tfp_tb_reset_zoom').on('click', async function() {
    tfp.zoomXs = [tfp.ovXDomain];  // clear cached data
    await _onZoomX(tfp.zoomXs[tfp.zoomXs.length-1], true);
  });
  
  $('#tfp_tb_limit').on('input change', function (){
  
    $(this).siblings('.btn').text($(this).val());
  
    if($(this).data('timeout')) {
      clearTimeout($(this).data('timeout'));
    }
  
    // get selected option and change background
    $(this).data('timeout', setTimeout(async ()=>{
      tfp.limit = +$(this).val();
      tfp.ovXSel = tfp.zoomXs[tfp.zoomXs.length-1];
    
      await queryData(tfp.zoomXs[tfp.zoomXs.length-1], tfp.zoomY, tfp.view, tfp.limit);
      
      _render(false);
    }, 1000));
  });
}

main();


