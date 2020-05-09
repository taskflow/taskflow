// Program: Generate timelines
// Author: twhuang
//
// This program stands on shoulders of the following giants:
//   - d3 javascript facility
//   - timeline chart by Vasco Asturiano.

'use strict';

var state = {
  
  // DOMAIN (data) -> RANGE (graph)

  // main timeline svg
  dom : null,
  svg : null,                     // svg block
  graph: null,                    // graph block
  graphW: 0,
  graphH: 0,
  zoomX: [null, null],            // scoped time data
  zoomY: [null, null],            // scoped line data
 
  // main graph
  width: window.innerWidth,
  height: 650,
  maxHeight: Infinity,
  maxLineHeight: 20,
  leftMargin: 100,
  rightMargin: 100,
  topMargin: 26,
  bottomMargin: 30,
  
  // overview element
  overviewAreaSvg: null,
  overviewAreaScale: d3.scaleLinear(),
  overviewAreaSelection: [null, null],
  overviewAreaDomain: [null, null],
  overviewAreaBrush: null,
  overviewAreaTopMargin: 1,
  overviewAreaBottomMargin: 30,
  overviewAreaXGrid: d3.axisBottom().tickFormat(''),
  overviewAreaXAxis: d3.axisBottom().tickPadding(0),
  overviewAreaBrush: d3.brushX(),

  // bar chart
  barSvg : null,
  barXScale: d3.scaleBand(),
  barYScale: d3.scaleLinear(),
  barXAxis: d3.axisBottom(),
  barYAxis: d3.axisLeft(),
  barHeight: 400,
  barWidth: window.innerWidth,
  barLeftMargin: 100,
  barRightMargin: 100,
  barTopMargin: 30,
  barBottomMargin: 100,

  // axes attributes
  xScale: d3.scaleLinear(),
  yScale: d3.scalePoint(),  
  grpScale: d3.scaleOrdinal(),
  xAxis: d3.axisBottom(),
  xGrid: d3.axisTop(),
  yAxis: d3.axisRight(),
  grpAxis: d3.axisLeft(),
  minLabelFont: 2,

  // legend
  zColorMap: new Map([
    ['static task', '#4682b4'],
    ['dynamic task', '#ff7f0e'],
    ['cudaflow', '#6A0DAD'],
    ['condition task', '#41A317'],
    ['module task', '#0000FF']
  ]),
  zScale: null,
  zGroup: null,
  zWidth: null,
  zHeight: null,

  // date marker line
  dateMarker: null,
  dateMarkerLine: null,
  
  // segmenet  
  minSegmentDuration: 0, // ms
  disableHover: false,
  minX: null,
  maxY: null,
  
  // transition
  transDuration: 700,

  // data field
  completeStructData: [],       // groups and lines
  completeFlatData: [],         // flat segments with gropu and line
  structData: null,
  flatData: null,
  totalNLines: 0,
  nLines: 0
};

d3.selection.prototype.moveToFront = function() {
      return this.each(function(){
        this.parentNode.appendChild(this);
      });
    };

d3.selection.prototype.moveToBack = function() {
    return this.each(function() {
        var firstChild = this.parentNode.firstChild;
        if (firstChild) {
            this.parentNode.insertBefore(this, firstChild);
        }
    });
};

// Procedure: make_tfp_structure
function make_tfp_structure(dom_element) {
  
  //console.log("timeline chart created at", dom_element);
  
  state.dom = d3.select('#tfp').attr('class', 'tfp');

  // main svg
  state.svg = state.dom.append('svg');
  
  // overview svg
  state.overviewAreaSvg = state.dom.append('div').append('svg').attr('class', 'brusher');

  // bar svg


  //console.log("_make_tfp_structure");
    
  state.yScale.invert = _invertOrdinal;
  state.grpScale.invert = _invertOrdinal;
  
  _make_tfp_gradient_field();
  _make_tfp_axes();
  _make_tfp_legend();
  _make_tfp_graph();
  _make_tfp_date_marker_line();
  _make_tfp_overview();
  _make_tfp_bar();
  _make_tfp_tooltips();
  _make_tfp_events();
}

// Procedure: feed()
function feed(rawData) {

  // clear the previous state
  state.zoomX = [null, null];
  state.zoomY = [null, null];
  state.minX  = null;
  state.maxX  = null;
  state.completeStructData = [];
  state.completeFlatData = [];
  state.completeTableData = [];
  state.totalNLines = 0;

  // iterate executor
  for (let i=0, ilen=rawData.length; i<ilen; i++) {
    const group = rawData[i].group;

    state.completeStructData.push({
      group: group,
      lines: rawData[i].data.map(d => d.label)
    });
    
    // iterate worker
    for (let j= 0, jlen=rawData[i].data.length; j<jlen; j++) {
      var total_time=0, stime=0, dtime=0, gtime=0, ctime=0, mtime=0;
      // iterate segment
      for (let k= 0, klen=rawData[i].data[j].data.length; k<klen; k++) {
        const { timeRange, val, name } = rawData[i].data[j].data[k];

        state.completeFlatData.push({
          group: group,
          label: rawData[i].data[j].label,
          timeRange: timeRange,
          val: val,                             // legend value
          name: name
        });

        if(state.minX == null || timeRange[0] < state.minX) {
          state.minX = timeRange[0];
        }

        if(state.maxX == null || timeRange[1] > state.maxX) {
          state.maxX = timeRange[1];
        }
        
        const elapsed = timeRange[1] - timeRange[0];
        total_time += elapsed;

        switch(val) {
          case "static task":
            stime += elapsed;
          break;

          case "dynamic task":
            dtime += elapsed;
          break;

          case "cudaflow":
            gtime += elapsed;
          break;

          case "condition task":
            ctime += elapsed;
          break;

          case "module task":
            mtime += elapsed;
          break;

          default:
            console.assert(false);
          break;
        }
      }

      state.completeTableData.push({
        "group": group,
        "label": rawData[i].data[j].label,
        "tasks": rawData[i].data[j].data.length,
        "static task": stime,
        "dynamic task": dtime,
        "cudaflow": gtime,
        "condition task": ctime,
        "module task": mtime,
        "busy": total_time
      });

      state.totalNLines++;
    }
  }

  //console.log("total", state.totalNLines, " lines");
  console.log(state.completeTableData)
  
  // static data fields
  state.overviewAreaDomain = [state.minX, state.maxX];

  // update all dynamic fields
  update([state.minX, state.maxX], [null, null]);
  
  
  // update the bar chart fields
  var options = d3.select("#tfp-sel-executor").selectAll("option")
		.data(['default (all)', ...rawData.map(d=>d.group)])

  options.exit().remove();
  options = options.merge(options.enter().append('option'))
              .text(d=>d);

  state.barXScale
    .padding(0.5)
    .domain(state.completeTableData.map(d=>d.label))
    .range([state.barLeftMargin, state.barWidth-state.barRightMargin]);
  
  state.barYScale
    .domain([0, d3.max(state.completeTableData, d=>d.busy)])
    .range([state.barHeight - state.barBottomMargin, state.barTopMargin]);

  console.log("barYScale", state.barYScale.domain(), state.barYScale.range());

  state.barXAxis.scale(state.barXScale).tickSizeOuter(0);
  state.barYAxis.scale(state.barYScale).tickSize(-state.barWidth+state.barLeftMargin +state.barRightMargin);

  state.barSvg.select('g.tfp-bar-x-axis')
    .attr('transform', `translate(0, ${state.barHeight - state.barBottomMargin})`)
    .transition().duration(state.transDuration)
      .call(state.barXAxis)
    .selectAll("text")
      .attr("y", 0)
      .attr("x", -40)
      .attr("dy", ".35em")
      .attr("transform", "rotate(-90)");

  state.barSvg.select('g.tfp-bar-y-axis')
    .attr('transform', `translate(${state.barLeftMargin}, 0)`)
    .transition().duration(state.transDuration)
      .call(state.barYAxis)

  var keys = ['static task', 'dynamic task', 'cudaflow', 'condition task', 'module task'];
  var stacked_data = d3.stack().keys(keys)(state.completeTableData);

  console.log(stacked_data)

  var l1 = state.barSvg.select('g.tfp-bar-graph')
    .selectAll('g')
    .data(stacked_data);

  l1.exit().remove();
  l1 = l1.enter().append('g').merge(l1).attr("fill", d => state.zColorMap.get(d.key));

  var l2 = l1.selectAll("rect").data(d=>d);

  l2.exit().remove(); 
  //l2.enter().append("rect").merge(l2)
  //  .transition().duration(state.transDuration)
  //  .attr('rx', 1)
  //  .attr('ry', 1)
  //  .attr('x', d => state.barXScale(d.data.label))
  //  .attr('y', d => state.barYScale(d[1]))
  //  .attr('height', d => state.barYScale(d[0]) - state.barYScale(d[1]))
  //  .attr('width', state.barXScale.bandwidth());

  var newbars = l2.enter().append("rect")
    .attr('width', 0)
    .attr('height', 0)
    .attr('x', 0)
    .attr('y', 0)
    .style('fill-opacity', 0.8)
    .on('mouseover.barTooltip', state.barTooltip.show)
    .on('mouseout.barTooltip', state.barTooltip.hide);
    
  newbars
    .on('mouseover', function() {

      if (state.disableHover) { return; }

      //MoveToFront()(this);
      d3.select(this).moveToFront();

      //const hoverEnlarge = state.lineHeight*hoverEnlargeRatio;

      const hoverEnlarge = state.barXScale.bandwidth()*0.01;

      //  const x = state.barXScale(d.data.label);
      //  const y = state.barYScale(d[1]);
      //  const w = state.barXScale.bandwidth();
      //  const h = state.barYScale(d[0]) - state.barYScale(d[1]);
      d3.select(this).moveToFront()
        .transition().duration(250)
        .attr('x', function(d) {
          return state.barXScale(d.data.label)-hoverEnlarge/2; 
        })
        .attr('width', state.barXScale.bandwidth() + hoverEnlarge)
        .attr('y', function(d) {
          return state.barYScale(d[1]) - hoverEnlarge/2;
        })
        .attr('height', function(d) {
          return state.barYScale(d[0]) - state.barYScale(d[1]) + hoverEnlarge;
        })
        .style('fill-opacity', 1);
    })
    .on('mouseout', function() {
      d3.select(this).moveToBack()
        .transition().duration(250)
        .attr('width', d => state.barXScale.bandwidth())
        .attr('height', d => state.barYScale(d[0]) - state.barYScale(d[1]))
        .attr('x', d => state.barXScale(d.data.label))
        .attr('y', d => state.barYScale(d[1]))
        .style('fill-opacity', 0.8);
    })


  l2.merge(newbars)
    .transition().duration(state.transDuration)
    .attr('rx', 1)
    .attr('ry', 1)
    .attr('x', d => state.barXScale(d.data.label))
    .attr('y', d => state.barYScale(d[1]))
    .attr('height', d => state.barYScale(d[0]) - state.barYScale(d[1]))
    .attr('width', state.barXScale.bandwidth());
}

// Procedure: update
function update(zoomX, zoomY) {
  
  // if the successive change is small, we don't update;
  // this also avoids potential infinite loops caused by cyclic event updates
  if((state.zoomX[0] == zoomX[0] && state.zoomX[1] == zoomX[1] &&
      state.zoomY[0] == zoomY[0] && state.zoomY[1] == zoomY[1]) ||
    (Math.abs(state.zoomX[0] - zoomX[0]) < Number.EPSILON && 
     Math.abs(state.zoomX[1] - zoomX[1]) < Number.EPSILON &&
     Math.abs(state.zoomY[0] - zoomY[0]) < Number.EPSILON && 
     Math.abs(state.zoomY[1] - zoomY[1]) < Number.EPSILON)) {
    //console.log("skip update", state.zoomX, state.zoomY, zoomX, zoomY);
    return;
  }
  
  // we use zoomX and zoomY to control the update
  state.zoomX = zoomX;
  state.zoomY = zoomY;
  state.overviewAreaSelection = state.zoomX;

  //console.log("update");

  _apply_filters();
  _adjust_dimensions();
  _adjust_xscale();
  _adjust_yscale();
  _adjust_grpscale();
  _adjust_legend();
    
  _render_axes()
  _render_groups();
  _render_timelines();
  _render_overview_area();
}

// ----------------------------------------------------------------------------
// private function definitions
// ----------------------------------------------------------------------------

// Procedure: _invertOrdinal 
// perform interpolation
function _invertOrdinal(val, cmpFunc) {

  cmpFunc = cmpFunc || function (a, b) {
      return (a >= b);
    };

  const scDomain = this.domain();
  let scRange = this.range();

  if (scRange.length === 2 && scDomain.length !== 2) {
    // Special case, interpolate range vals
    scRange = d3.range(scRange[0], scRange[1], (scRange[1] - scRange[0]) / scDomain.length);
  }

  const bias = scRange[0];
  for (let i = 0, len = scRange.length; i < len; i++) {
    if (cmpFunc(scRange[i] + bias, val)) {
      return scDomain[Math.round(i * scDomain.length / scRange.length)];
    }
  }

  return this.domain()[this.domain().length-1];
}
  
function _make_tfp_gradient_field() {  
  //console.log("making gradient ...");
  state.groupGradId = `areaGradient${Math.round(Math.random()*10000)}`;
  const gradient = state.svg.append('linearGradient');

  gradient.attr('y1', '0%')
          .attr('y2', '100%')
          .attr('x1', '0%')
          .attr('x2', '0%')
          .attr('id', state.groupGradId);
  
  const color_scale = d3.scaleLinear().domain([0, 1]).range(['#FAFAFA', '#E0E0E0']);
  const stop_scale = d3.scaleLinear().domain([0, 100]).range(color_scale.domain());
  
  let color_stops = gradient.selectAll('stop')
                      .data(d3.range(0, 100.01, 20)); 

  color_stops.exit().remove();
  color_stops.merge(color_stops.enter().append('stop'))
    .attr('offset', d => `${d}%`)
    .attr('stop-color', d => color_scale(stop_scale(d)));
}

// Procedure: _make_tfp_date_marker_line
function _make_tfp_date_marker_line() {
  //console.log("making date marker ...");
  state.dateMarkerLine = state.svg.append('line').attr('class', 'x-axis-date-marker');
}

// Procedure: _make_tfp_overview
function _make_tfp_overview() {
  //console.log("making the overview ...");

  state.overviewAreaBrush
    .handleSize(24)
    .on('end', function() {
      
      //console.log("ON 'end': brush ends by source", d3.event.sourceEvent);

      if (!d3.event.sourceEvent) {
        return;
      }

      //console.log("    -> type:", d3.event.sourceEvent.type);

      const selection = d3.event.selection ? 
        d3.event.selection.map(state.overviewAreaScale.invert) : 
        state.overviewAreaScale.domain();

      // avoid infinite event loop
      if(d3.event.sourceEvent.type === "mouseup") {
        state.svg.dispatch('zoom', { detail: {
          zoomX: selection,
          zoomY: state.zoomY
        }});
      }
    });

  // Build dom
  const brusher = state.overviewAreaSvg.append('g').attr('class', 'brusher-margins');
  brusher.append('rect').attr('class', 'grid-background');
  brusher.append('g').attr('class', 'x-grid');
  brusher.append('g').attr('class', 'x-axis');
  brusher.append('g').attr('class', 'brush');
        
  //state.svg.on('zoomScent', function() {

  //  const zoomX = d3.event.detail.zoomX;

  //  if (!zoomX) return;

  //  // Out of overview bounds > extend it
  //  if (zoomX[0]<state.overviewArea.domainRange()[0] || zoomX[1]>state.overviewArea.domainRange()[1]) {
  //    console.log("can this happen?");
  //    state.overviewArea.domainRange([
  //      new Date(Math.min(zoomX[0], state.overviewArea.domainRange()[0])),
  //      new Date(Math.max(zoomX[1], state.overviewArea.domainRange()[1]))
  //    ]);
  //  }

  //  state.overviewArea.currentSelection(zoomX);

  //  console.log("on ZoomScent");
  //});
}

// Procedure: _make_tfp_bar
function _make_tfp_bar() {
  
  const barDiv = state.dom.append('div');

  state.barSvg = barDiv.append('svg')
    .attr('width', state.barWidth)
    .attr('height', state.barHeight);

  barDiv.append('div').attr('style', 'ml-4')
    .append('select').attr('id', 'tfp-sel-executor');

  state.barSvg.append('g').attr('class', 'tfp-bar-x-axis');
  state.barSvg.append('g').attr('class', 'tfp-bar-y-axis');
  state.barSvg.append('g').attr('class', 'tfp-bar-graph');
}

// Procedure: _make_tfp_axes
function _make_tfp_axes() {  
  //console.log("making the axes ...");
  const axes = state.svg.append('g').attr('class', 'axes');
  axes.append('g').attr('class', 'x-axis');
  axes.append('g').attr('class', 'x-grid');
  axes.append('g').attr('class', 'y-axis');
  axes.append('g').attr('class', 'grp-axis');

  state.yAxis.scale(state.yScale).tickSize(0);
  state.grpAxis.scale(state.grpScale).tickSize(0);
}

// Procedure: _make_tfp_legend
function _make_tfp_legend() {

  //console.log("making the legend ...");

  // add a reset text
  state.resetBtn = state.svg.append('text')
    .attr('class', 'reset-zoom-btn')
    .text('Reset Zoom')
    .on('click' , function() {
      //console.log("ON 'click': reset btn");
      state.svg.dispatch('resetZoom');
    });
  
  // add a legend group
  state.zScale = d3.scaleOrdinal()
    .domain(['static', 'dynamic', 'cudaflow', 'condition', 'module'])
    .range(['#4682b4', '#FF7F0E', '#6A0DAD', '#41A317', '#0000FF']);

  state.zGroup = state.svg.append('g')
                   .attr('class', 'legend');
  state.zWidth = (state.width-state.leftMargin-state.rightMargin)*3/4;
  state.zHeight = state.topMargin*0.8;

  const binWidth = state.zWidth / state.zScale.domain().length;

  //console.log(binWidth)

  let slot = state.zGroup.selectAll('.z-slot')
    .data(state.zScale.domain());

  slot.exit().remove();

  const newslot = slot.enter()
    .append('g')
    .attr('class', 'z-slot');

  newslot.append('rect')
    .attr('y', 0)
    .attr('rx', 0)
    .attr('ry', 0)
    .attr('stroke-width', 0);

  newslot.append('text')
    .style('text-anchor', 'middle')
    .style('dominant-baseline', 'central');

  // Update
  slot = slot.merge(newslot);

  slot.select('rect')
    .attr('width', binWidth)
    .attr('height', state.zHeight)
    .attr('x', (d, i) => binWidth*i)
    .attr('fill', d => state.zScale(d));

  slot.select('text')
    .text(d => d)
    .attr('x', (d, i) => binWidth*(i+.5))
    .attr('y', state.zHeight*0.5)
    .style('fill', '#FFFFFF');
}

// Procedure: _make_tfp_graph
function _make_tfp_graph() {

  //console.log("making the graph ...");

  state.graph = state.svg.append('g');

  state.graph.on('mousedown', function() {

    //console.log("ON 'mousedown'");

    if (d3.select(window).on('mousemove.zoomRect')!=null) // Selection already active
      return;

    const e = this;

    if (d3.mouse(e)[0]<0 || d3.mouse(e)[0] > state.graphW || 
        d3.mouse(e)[1]<0 || d3.mouse(e)[1] > state.graphH)
      return;

    state.disableHover=true;

    const rect = state.graph.append('rect')
      .attr('class', 'chart-zoom-selection');

    const startCoords = d3.mouse(e);

    d3.select(window)
      .on('mousemove.zoomRect', function() {

        //console.log("ON 'mousemove'");

        d3.event.stopPropagation();
        const newCoords = [
          Math.max(0, Math.min(state.graphW, d3.mouse(e)[0])),
          Math.max(0, Math.min(state.graphH, d3.mouse(e)[1]))
        ];

        rect.attr('x', Math.min(startCoords[0], newCoords[0]))
          .attr('y', Math.min(startCoords[1], newCoords[1]))
          .attr('width', Math.abs(newCoords[0] - startCoords[0]))
          .attr('height', Math.abs(newCoords[1] - startCoords[1]));

        state.overviewAreaSelection = [startCoords[0], newCoords[0]]
                                        .sort(d3.ascending)
                                        .map(state.xScale.invert);
        _render_overview_area();
        //state.svg.dispatch('zoomScent', { detail: {
        //  zoomX: [startCoords[0], newCoords[0]].sort(d3.ascending).map(state.xScale.invert),
        //  zoomY: [startCoords[1], newCoords[1]].sort(d3.ascending).map(d =>
        //    state.yScale.domain().indexOf(state.yScale.invert(d))
        //    + ((state.zoomY && state.zoomY[0])?state.zoomY[0]:0)
        //  )
        //}});
      })
      .on('mouseup.zoomRect', function() {

        //console.log("ON 'mouseup'");

        d3.select(window).on('mousemove.zoomRect', null).on('mouseup.zoomRect', null);
        d3.select('body').classed('stat-noselect', false);
        rect.remove();
        state.disableHover=false;

        const endCoords = [
          Math.max(0, Math.min(state.graphW, d3.mouse(e)[0])),
          Math.max(0, Math.min(state.graphH, d3.mouse(e)[1]))
        ];

        if (startCoords[0]==endCoords[0] && startCoords[1]==endCoords[1]) {
          //console.log("no change");
          return;
        }

        //console.log("coord", endCoords);

        const newDomainX = [startCoords[0], endCoords[0]].sort(d3.ascending).map(state.xScale.invert);

        const newDomainY = [startCoords[1], endCoords[1]].sort(d3.ascending).map(d =>
          state.yScale.domain().indexOf(state.yScale.invert(d))
          + ((state.zoomY && state.zoomY[0])?state.zoomY[0]:0)
        );
        
        state.svg.dispatch('zoom', { detail: {
          zoomX: newDomainX,
          zoomY: newDomainY
        }});
      }, true);

    d3.event.stopPropagation();
  });
}

// Procedure: _make_tfp_tooltips
function _make_tfp_tooltips() {

  //console.log("making the tooltips ...");
  
  // group tooltips 
  state.groupTooltip = d3.tip()
       .attr('class', 'tfp-tooltip')
       .direction('w')
       .offset([0, 0])
       .html(d => {
         const leftPush = (d.hasOwnProperty('timeRange') ?
                          state.xScale(d.timeRange[0]) : 0);
         const topPush = (d.hasOwnProperty('label') ?
                          state.grpScale(d.group) - state.yScale(d.group+'+&+'+d.label) : 0 );
         state.groupTooltip.offset([topPush, -leftPush]);
         return d.group;
       });

  state.svg.call(state.groupTooltip);

  // label tooltips
  state.lineTooltip = d3.tip()
       .attr('class', 'tfp-tooltip')
       .direction('e')
       .offset([0, 0])
       .html(d => {
         const rightPush = (d.hasOwnProperty('timeRange') ? 
                            state.xScale.range()[1]-state.xScale(d.timeRange[1]) : 0);
         state.lineTooltip.offset([0, rightPush]);
         return d.label;
       });

  state.svg.call(state.lineTooltip);
  
  // segment tooltips
  state.segmentTooltip = d3.tip()
    .attr('class', 'tfp-tooltip')
    .direction('s')
    .offset([5, 0])
    .html(d => {
      return `Type: ${d.val}<br>
              Name: ${d.name}<br>
              Time: [${d.timeRange}]<br>
              Span: ${d.timeRange[1]-d.timeRange[0]}`;
    });

  state.svg.call(state.segmentTooltip);
  
  // bar tooltips
  state.barTooltip = d3.tip()
    .attr('class', 'tfp-tooltip')
    .direction('w')
    .offset([0, -5])
    .html(d => {
      const t = d[1] - d[0];
      const p = ((t / d.data.busy)*100).toFixed(2);
      return `Total Time: ${t}<br>
              Percentage: ${p}%`;
    });

  state.svg.call(state.barTooltip);
}
      
// Proecedure: _make_tfp_events      
function _make_tfp_events() {

  //console.log("making dom events ...");

  state.svg.on('zoom', function() {

    const evData = d3.event.detail;   // passed custom parameters 
    const zoomX = evData.zoomX;
    const zoomY = evData.zoomY;
    //const redraw = (evData.redraw == null) ? true : evData.redraw;
    
    console.assert((zoomX && zoomY));
    //console.log("ON 'zoom'");

    update(zoomX, zoomY);
    
    // exposed to user
    //if (state.onZoom) {
    //  state.onZoom(state.zoomX, state.zoomY);
    //}
  });

  state.svg.on('resetZoom', function() {
    //console.log("ON resetZoom");
    update([state.minX, state.maxX], [null, null]);
    //if (state.onZoom) state.onZoom(null, null);
  });
}

// Procedure: _apply_filters
function _apply_filters() {

  // Flat data based on segment length
  //state.flatData = (state.minSegmentDuration>0
  //  ? state.completeFlatData.filter(d => (d.timeRange[1]-d.timeRange[0]) >= state.minSegmentDuration)
  //  : state.completeFlatData
  //);
  //state.flatData = state.completeFlatData;
  
  console.assert(state.zoomY);

  // zoomY
  //if (state.zoomY == null || state.zoomY==[null, null]) {
  if(state.zoomY == null || (state.zoomY[0] == null && state.zoomY[1] == null)) {
    //console.log("use all y");
    state.structData = state.completeStructData;
    state.nLines = state.totalNLines;
    //for (let i=0, len=state.structData.length; i<len; i++) {
    //  state.nLines += state.structData[i].lines.length;
    //}
    //console.log(state.nLines, state.totalNLines);
    return;
  }

  //console.log("filtering struct Data on ", state.zoomY[0], state.zoomY[1]);

  state.structData = [];
  const cntDwn = [state.zoomY[0] == null ? 0 : state.zoomY[0]]; // Initial threshold
  cntDwn.push(Math.max(
    0, (state.zoomY[1]==null ? state.totalNLines : state.zoomY[1]+1)-cntDwn[0])
  ); // Number of lines

  state.nLines = cntDwn[1];
  for (let i=0, len=state.completeStructData.length; i<len; i++) {

    let validLines = state.completeStructData[i].lines;

    //if(state.minSegmentDuration>0) {  // Use only non-filtered (due to segment length) groups/labels
    //  if (!state.flatData.some(d => d.group == state.completeStructData[i].group)) {
    //    continue; // No data for this group
    //  }

    //  validLines = state.completeStructData[i].lines
    //    .filter(d => state.flatData.some(dd =>
    //      dd.group == state.completeStructData[i].group && dd.label == d
    //    )
    //  );
    //}
    if (cntDwn[0]>=validLines.length) { // Ignore whole group (before start)
      cntDwn[0]-=validLines.length;
      continue;
    }
    const groupData = {
      group: state.completeStructData[i].group,
      lines: null
    };
    if (validLines.length-cntDwn[0]>=cntDwn[1]) {  // Last (or first && last) group (partial)
      groupData.lines = validLines.slice(cntDwn[0],cntDwn[1]+cntDwn[0]);
      state.structData.push(groupData);
      cntDwn[1]=0;
      break;
    }
    if (cntDwn[0]>0) {  // First group (partial)
      groupData.lines = validLines.slice(cntDwn[0]);
      cntDwn[0]=0;
    } else {  // Middle group (full fit)
      groupData.lines = validLines;
    }

    state.structData.push(groupData);
    cntDwn[1]-=groupData.lines.length;
  }

  state.nLines-=cntDwn[1];
  //console.log("filtered lines:", state.nLines);
}


function _adjust_dimensions() {
  //console.log("adjusting up dimensions ... nLines =", state.nLines);
  state.graphW = state.width - state.leftMargin - state.rightMargin;
  state.graphH = state.nLines*state.maxLineHeight;
  state.height = state.graphH + state.topMargin + state.bottomMargin;
  //console.log("transition to", state.width, state.height, " graph", state.graphH, state.graphW);
  state.svg//.transition().duration(state.transDuration)
    .attr('width', state.width)
    .attr('height', state.height);

  state.graph.attr('transform', `translate(${state.leftMargin}, ${state.topMargin})`);
}

function _adjust_xscale() {
  console.assert(state.zoomX[0] && state.zoomX[1]);
  //console.log("adjusting xscale to", state.zoomX);
  state.xScale.domain(state.zoomX)
              .range([0, state.graphW])
              .clamp(true);
}

// Procedure: _adjust_yscale
function _adjust_yscale() {

  let labels = [];
  for (let i= 0, len=state.structData.length; i<len; i++) {
    labels = labels.concat(state.structData[i].lines.map(function (d) {
      return `${state.structData[i].group}+&+${d}`
    }));
  }

  //console.log("adjusting yscale to", labels);
  state.yScale.domain(labels);
  //console.log(state.graphH/labels.length*0.5, state.graphH*(1-0.5/labels.length));
  state.yScale.range([state.graphH/labels.length*0.5, state.graphH*(1-0.5/labels.length)]);
}
    
// Procedure: _adjust_grpscale
function _adjust_grpscale() {
  //console.log("adjusting group domain", state.structData.map(d => d.group));
  state.grpScale.domain(state.structData.map(d => d.group));

  let cntLines = 0;

  state.grpScale.range(state.structData.map(d => {
    const pos = (cntLines + d.lines.length/2)/state.nLines*state.graphH;
    cntLines += d.lines.length;
    return pos;
  }));
}

// Procedure: _adjust_legend
function _adjust_legend() {
  //console.log("adjusting legend ...");
  state.resetBtn.transition().duration(state.transDuration)
    .attr('x', state.leftMargin + state.graphW*.99)
    .attr('y', state.topMargin *.8);
  
  state.zGroup.transition().duration(state.transDuration)
    .attr('transform', `translate(${state.leftMargin}, ${state.topMargin * .1})`);
}

// Procedure: _render_axes
function _render_axes() {

  state.svg.select('.axes')
    .attr('transform', `translate(${state.leftMargin}, ${state.topMargin})`);

  // X
  const nXTicks = num_xticks(state.graphW);

  //console.log("rendering axes nXTicks =", nXTicks);

  state.xAxis
    .scale(state.xScale)
    .ticks(nXTicks);

  state.xGrid
    .scale(state.xScale)
    .ticks(nXTicks)
    .tickFormat('');

  state.svg.select('g.x-axis')
    .style('stroke-opacity', 0)
    .style('fill-opacity', 0)
    .attr('transform', 'translate(0,' + state.graphH + ')')
    .transition().duration(state.transDuration)
      .call(state.xAxis)
      .style('stroke-opacity', 1)
      .style('fill-opacity', 1);

  /* Angled x axis labels
   state.svg.select('g.x-axis').selectAll('text')
   .style('text-anchor', 'end')
   .attr('transform', 'translate(-10, 3) rotate(-60)');
   */

  state.xGrid.tickSize(state.graphH);
  state.svg.select('g.x-grid')
    .attr('transform', 'translate(0,' + state.graphH + ')')
    .transition().duration(state.transDuration)
    .call(state.xGrid);

  if (
    state.dateMarker &&
    state.dateMarker >= state.xScale.domain()[0] &&
    state.dateMarker <= state.xScale.domain()[1]
  ) {
    state.dateMarkerLine
      .style('display', 'block')
      .transition().duration(state.transDuration)
        .attr('x1', state.xScale(state.dateMarker) + state.leftMargin)
        .attr('x2', state.xScale(state.dateMarker) + state.leftMargin)
        .attr('y1', state.topMargin + 1)
        .attr('y2', state.graphH + state.topMargin)
  } else {
    state.dateMarkerLine.style('display', 'none');
  }

  // Y
  const fontVerticalMargin = 0.6;
  const labelDisplayRatio = Math.ceil(
    state.nLines*state.minLabelFont/Math.SQRT2/state.graphH/fontVerticalMargin
  );
  const tickVals = state.yScale.domain().filter((d, i) => !(i % labelDisplayRatio));
  let fontSize = Math.min(14, state.graphH/tickVals.length*fontVerticalMargin*Math.SQRT2);
  let maxChars = Math.ceil(state.rightMargin/(fontSize/Math.SQRT2));

  state.yAxis.tickValues(tickVals);
  state.yAxis.tickFormat(d => reduceLabel(d.split('+&+')[1], maxChars));
  state.svg.select('g.y-axis')
    .transition().duration(state.transDuration)
      .attr('transform', `translate(${state.graphW}, 0)`)
      .attr('font-size', `${fontSize}px`)
      .call(state.yAxis);

  // Grp
  const minHeight = d3.min(state.grpScale.range(), function (d, i) {
    return i>0 ? d-state.grpScale.range()[i-1] : d*2;
  });

  fontSize = Math.min(14, minHeight*fontVerticalMargin*Math.SQRT2);
  maxChars = Math.ceil(state.leftMargin/(fontSize/Math.SQRT2));
  
  //console.log(minHeight, maxChars, fontSize);

  state.grpAxis.tickFormat(d => reduceLabel(d, maxChars));
  state.svg.select('g.grp-axis')
    .transition().duration(state.transDuration)
    .attr('font-size', `${fontSize}px`)
    .call(state.grpAxis);

  //// Make Axises clickable
  //if (state.onLabelClick) {
  //  console.log("register callback")
  //  state.svg.selectAll('g.y-axis,g.grp-axis').selectAll('text')
  //    .style('cursor', 'pointer')
  //    .on('click', function(d) {
  //      const segms = d.split('+&+');
  //      //state.onLabelClick(...segms.reverse());
  //      console.log("click on", d);
  //    });
  //}

  function reduceLabel(label, maxChars) {
    return label.length <= maxChars ? label : (
      label.substring(0, maxChars*2/3)
      + '...'
      + label.substring(label.length - maxChars/3, label.length
    ));
  }
}

// Procedure: _render_groups
function _render_groups() {

  let groups = state.graph.selectAll('rect.series-group').data(state.structData, d => d.group);
  //console.log("rendering groups", groups);
      
  groups.exit()
    .transition().duration(state.transDuration)
    .style('stroke-opacity', 0)
    .style('fill-opacity', 0)
    .remove();

  const newGroups = groups.enter().append('rect')
    .attr('class', 'series-group')
    .attr('x', 0)
    .attr('y', 0)
    .attr('height', 0)
    .style('fill', `url(#${state.groupGradId})`)
    .on('mouseover', state.groupTooltip.show)
    .on('mouseout', state.groupTooltip.hide);

  newGroups.append('title')
    .text('click-drag to zoom in');

  groups = groups.merge(newGroups);

  groups.transition().duration(state.transDuration)
    .attr('width', state.graphW)
    .attr('height', function (d) {
      return state.graphH*d.lines.length/state.nLines;
    })
    .attr('y', function (d) {
      return state.grpScale(d.group)-state.graphH*d.lines.length/state.nLines/2;
    });
}

// procedure: _render_timelines
function _render_timelines(maxElems) {

  //console.log("rendering timelines ...");

  if (maxElems == undefined || maxElems < 0) {
    maxElems = null;
  }

  const hoverEnlargeRatio = .4;

  const dataFilter = (d, i) =>
    (maxElems == null || i<maxElems) &&
    (state.grpScale.domain().indexOf(d.group)+1 &&
     d.timeRange[1]>=state.xScale.domain()[0] &&
     d.timeRange[0]<=state.xScale.domain()[1] &&
     state.yScale.domain().indexOf(d.group+'+&+'+d.label)+1);

  state.lineHeight = state.graphH/state.nLines*0.8;

  let timelines = state.graph.selectAll('rect.series-segment').data(
    //state.flatData.filter(dataFilter),
    state.completeFlatData.filter(dataFilter),
    d => d.group + d.label + d.timeRange[0]
  );

  timelines.exit()
    .transition().duration(state.transDuration)
    .style('fill-opacity', 0)
    .remove();

  const newSegments = timelines.enter().append('rect')
    .attr('class', 'series-segment')
    .attr('rx', 1)
    .attr('ry', 1)
    .attr('x', state.graphW/2)    // here we initialize the rect to avoid
    .attr('y', state.graphH/2)    // NaN y error during transition
    .attr('width', 0)
    .attr('height', 0)
    .style('fill-opacity', 0)
    .style('fill', d => state.zColorMap.get(d.val))
    .on('mouseover.groupTooltip', state.groupTooltip.show)
    .on('mouseout.groupTooltip', state.groupTooltip.hide)
    .on('mouseover.lineTooltip', state.lineTooltip.show)
    .on('mouseout.lineTooltip', state.lineTooltip.hide)
    .on('mouseover.segmentTooltip', state.segmentTooltip.show)
    .on('mouseout.segmentTooltip', state.segmentTooltip.hide);

  newSegments
    .on('mouseover', function() {

      if (state.disableHover) { return; }

      //MoveToFront()(this);

      const hoverEnlarge = state.lineHeight*hoverEnlargeRatio;

      d3.select(this)
        .transition().duration(250)
        .attr('x', function (d) {
          return state.xScale(d.timeRange[0])-hoverEnlarge/2;
        })
        .attr('width', function (d) {
          return d3.max([1, state.xScale(d.timeRange[1])-state.xScale(d.timeRange[0])])+hoverEnlarge;
        })
        .attr('y', function (d) {
          return state.yScale(`${d.group}+&+${d.label}`)-(state.lineHeight+hoverEnlarge)/2;
        })
        .attr('height', state.lineHeight+hoverEnlarge)
        .style('fill-opacity', 1);
    })
    .on('mouseout', function() {
      d3.select(this)
        .transition().duration(250)
        .attr('x', function (d) {
          return state.xScale(d.timeRange[0]);
        })
        .attr('width', function (d) {
          return d3.max([1, state.xScale(d.timeRange[1])-state.xScale(d.timeRange[0])]);
        })
        .attr('y', function (d) {
          return state.yScale(`${d.group}+&+${d.label}`)-state.lineHeight/2;
        })
        .attr('height', state.lineHeight)
        .style('fill-opacity', .8);
    })
    .on('click', function (s) {
      if (state.onSegmentClick)
        state.onSegmentClick(s);
    });

  timelines = timelines.merge(newSegments);

  timelines.transition().duration(state.transDuration)
    .attr('x', function (d) {
      return state.xScale(d.timeRange[0]);
    })
    .attr('width', function (d) {
      return d3.max([1, state.xScale(d.timeRange[1])-state.xScale(d.timeRange[0])]);
    })
    .attr('y', function (d) {
      return state.yScale(`${d.group}+&+${d.label}`)-state.lineHeight/2;
    })
    .attr('height', state.lineHeight)
    .style('fill-opacity', .8);
}

function _render_overview_area()  {

  //console.log("rendering overview...")
  
  // domain is not set up yet
  if (state.overviewAreaDomain[0] == null || state.overviewAreaDomain[1] == null) {
    return;
  }

  const brushWidth = state.graphW;
  const brushHeight = 20;
  const nXTicks = num_xticks(brushWidth);

  //console.log("brush ", brushWidth, brushHeight);

  state.overviewAreaScale
    .domain(state.overviewAreaDomain)
    .range([0, brushWidth]);

  state.overviewAreaXAxis
    .scale(state.overviewAreaScale)
    .ticks(nXTicks);

  state.overviewAreaXGrid
    .scale(state.overviewAreaScale)
    .tickSize(-brushHeight);

  state.overviewAreaSvg
    .attr('width', state.width)
    .attr('height', brushHeight + state.overviewAreaTopMargin
                                + state.overviewAreaBottomMargin);

  state.overviewAreaSvg.select('.brusher-margins')
    .attr('transform', `translate(${state.leftMargin}, ${state.overviewAreaTopMargin})`);

  state.overviewAreaSvg.select('.grid-background')
    //.attr('transform', `translate(${state.leftMargin},${})`)
    .attr('width', brushWidth)
    .attr('height', brushHeight);

  state.overviewAreaSvg.select('.x-grid')
    .attr('transform', `translate(0, ${brushHeight})`)
    .call(state.overviewAreaXGrid);

  state.overviewAreaSvg.select('.x-axis')
    .attr("transform", `translate(0, ${brushHeight})`)
    .call(state.overviewAreaXAxis)
    .selectAll('text').attr('y', 8);

  state.overviewAreaSvg.select('.brush')
    .call(state.overviewAreaBrush.extent([[0, 0], [brushWidth, brushHeight]]))
    .call(state.overviewAreaBrush.move, state.overviewAreaSelection.map(state.overviewAreaScale));
}


// ----------------------------------------------------------------------------
// Helper functions
// ----------------------------------------------------------------------------
function num_xticks(W) {
  return Math.max(2, Math.min(12, Math.round(W * 0.012)));
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// Example: Matrix multiplication
$('#tfp_matmul').on('click', function() {
  tfp_render_matmul();
})

$('#tfp_kmeans').on('click', function() {
  tfp_render_kmeans();
})

$('#tfp_inference').on('click', function() {
  tfp_render_inference();
})

$('#tfp_dreamplace').on('click', function() {
  tfp_render_dreamplace();
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

function tfp_render_simple() {
  feed(simple);
  $('#tfp_textarea').text(JSON.stringify(simple, null, 2));
}

function tfp_render_matmul() {
  feed(matmul);
  $('#tfp_textarea').text(JSON.stringify(matmul));
}

function tfp_render_kmeans() {
  feed(kmeans);
  $('#tfp_textarea').text(JSON.stringify(kmeans));
}

function tfp_render_inference() {
  feed(inference);
  $('#tfp_textarea').text(JSON.stringify(inference))
}

function tfp_render_dreamplace() {
  feed(dreamplace);
  $('#tfp_textarea').text(JSON.stringify(dreamplace))
}

// ----------------------------------------------------------------------------



// DOM objects
make_tfp_structure();

tfp_render_simple();

