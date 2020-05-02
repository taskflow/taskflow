import Kapsule from 'kapsule';
import { min, max, range, ascending } from 'd3-array';
import { axisBottom, axisTop, axisRight, axisLeft } from 'd3-axis';
import { scaleSequential, scaleOrdinal, scalePoint, scaleLinear, scaleUtc, scaleTime } from 'd3-scale';
import { event, select, mouse } from 'd3-selection';
import { utcFormat, timeFormat } from 'd3-time-format';
import d3Tip from 'd3-tip';
import { interpolateRdYlBu, schemeCategory10, schemeSet3 } from 'd3-scale-chromatic';
import { gradient, moveToFront } from 'svg-utils';
import { fitToBox } from 'svg-text-fit';
import ColorLegend from 'd3-color-legend';
import { brushX } from 'd3-brush';

function _toConsumableArray(arr) {
  return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
}

function _arrayWithoutHoles(arr) {
  if (Array.isArray(arr)) return _arrayLikeToArray(arr);
}

function _iterableToArray(iter) {
  if (typeof Symbol !== "undefined" && Symbol.iterator in Object(iter)) return Array.from(iter);
}

function _unsupportedIterableToArray(o, minLen) {
  if (!o) return;
  if (typeof o === "string") return _arrayLikeToArray(o, minLen);
  var n = Object.prototype.toString.call(o).slice(8, -1);
  if (n === "Object" && o.constructor) n = o.constructor.name;
  if (n === "Map" || n === "Set") return Array.from(o);
  if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
}

function _arrayLikeToArray(arr, len) {
  if (len == null || len > arr.length) len = arr.length;

  for (var i = 0, arr2 = new Array(len); i < len; i++) arr2[i] = arr[i];

  return arr2;
}

function _nonIterableSpread() {
  throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}

function styleInject(css, ref) {
  if (ref === void 0) ref = {};
  var insertAt = ref.insertAt;

  if (!css || typeof document === 'undefined') {
    return;
  }

  var head = document.head || document.getElementsByTagName('head')[0];
  var style = document.createElement('style');
  style.type = 'text/css';

  if (insertAt === 'top') {
    if (head.firstChild) {
      head.insertBefore(style, head.firstChild);
    } else {
      head.appendChild(style);
    }
  } else {
    head.appendChild(style);
  }

  if (style.styleSheet) {
    style.styleSheet.cssText = css;
  } else {
    style.appendChild(document.createTextNode(css));
  }
}

var css_248z = ".timelines-chart {\n\n  text-align: center;\n\n  /* Cancel selection interaction */\n  -webkit-touch-callout: none;\n  -webkit-user-select: none;\n  -khtml-user-select: none;\n  -moz-user-select: none;\n  -ms-user-select: none;\n  user-select: none;\n}\n\n  .timelines-chart .axises line, .timelines-chart .axises path {\n      stroke: #808080;\n    }\n\n  .timelines-chart .axises .x-axis {\n      font: 12px sans-serif;\n    }\n\n  .timelines-chart .axises .x-grid line {\n      stroke: #D3D3D3;\n    }\n\n  .timelines-chart .axises .y-axis line, .timelines-chart .axises .y-axis path, .timelines-chart .axises .grp-axis line, .timelines-chart .axises .grp-axis path {\n        stroke: none;\n      }\n\n  .timelines-chart .axises .y-axis text, .timelines-chart .axises .grp-axis text {\n        fill: #2F4F4F;\n      }\n\n  .timelines-chart line.x-axis-date-marker {\n    stroke-width: 1;\n    stroke: #293cb7;\n    fill: \"none\";\n  }\n\n  .timelines-chart .series-group {\n    fill-opacity: 0.6;\n    stroke: #808080;\n    stroke-opacity: 0.2;\n  }\n\n  .timelines-chart .series-segment {\n    stroke: none;\n  }\n\n  .timelines-chart .series-group, .timelines-chart .series-segment {\n    cursor: crosshair;\n  }\n\n  .timelines-chart .legend {\n    font-family: Sans-Serif;\n  }\n\n  .timelines-chart .legend .legendText {\n      fill: #666;\n    }\n\n  .timelines-chart .reset-zoom-btn {\n    font-family: sans-serif;\n    fill: blue;\n    opacity: .6;\n    cursor: pointer;\n  }\n\n.brusher .grid-background {\n    fill: lightgrey;\n  }\n\n.brusher .axis path {\n    display: none;\n  }\n\n.brusher .tick text {\n    text-anchor: middle;\n  }\n\n.brusher .grid line, .brusher .grid path {\n      stroke: #fff;\n    }\n\n.chart-zoom-selection, .brusher .brush .selection {\n  stroke: blue;\n  stroke-opacity: 0.6;\n  fill: blue;\n  fill-opacity: 0.3;\n  shape-rendering: crispEdges;\n}\n\n.chart-tooltip {\n  color: #eee;\n  background: rgba(0,0,140,0.85);\n  padding: 5px;\n  border-radius: 3px;\n  font: 11px sans-serif;\n  z-index: 4000;\n}\n\n.chart-tooltip.group-tooltip {\n    font-size: 14px;\n  }\n\n.chart-tooltip.line-tooltip {\n    font-size: 13px;\n  }\n\n.chart-tooltip.group-tooltip, .chart-tooltip.line-tooltip {\n    font-weight: bold;\n  }\n\n.chart-tooltip.segment-tooltip {\n     text-align: center;\n  }";
styleInject(css_248z);

var TimeOverview = Kapsule({
  props: {
    width: {
      "default": 300
    },
    height: {
      "default": 20
    },
    margins: {
      "default": {
        top: 0,
        right: 0,
        bottom: 20,
        left: 0
      }
    },
    scale: {},
    domainRange: {},
    currentSelection: {},
    tickFormat: {},
    onChange: {
      "default": function _default(selectionStart, selectionEnd) {}
    }
  },
  init: function init(el, state) {
    state.xGrid = axisBottom().tickFormat('');
    state.xAxis = axisBottom().tickPadding(0);
    state.brush = brushX().handleSize(24).on('end', function () {
      if (!event.sourceEvent) return;
      var selection = event.selection ? event.selection.map(state.scale.invert) : state.scale.domain();
      state.onChange.apply(state, _toConsumableArray(selection));
    }); // Build dom

    state.svg = select(el).append('svg').attr('class', 'brusher');
    var brusher = state.svg.append('g').attr('class', 'brusher-margins');
    brusher.append('rect').attr('class', 'grid-background');
    brusher.append('g').attr('class', 'x grid');
    brusher.append('g').attr('class', 'x axis');
    brusher.append('g').attr('class', 'brush');
  },
  update: function update(state) {
    if (state.domainRange[1] <= state.domainRange[0]) return;
    var brushWidth = state.width - state.margins.left - state.margins.right,
        brushHeight = state.height - state.margins.top - state.margins.bottom;
    state.scale.domain(state.domainRange).range([0, brushWidth]);
    state.xAxis.scale(state.scale).tickFormat(state.tickFormat);
    state.xGrid.scale(state.scale).tickSize(-brushHeight);
    state.svg.attr('width', state.width).attr('height', state.height);
    state.svg.select('.brusher-margins').attr('transform', "translate(".concat(state.margins.left, ",").concat(state.margins.top, ")"));
    state.svg.select('.grid-background').attr('width', brushWidth).attr('height', brushHeight);
    state.svg.select('.x.grid').attr('transform', 'translate(0,' + brushHeight + ')').call(state.xGrid);
    state.svg.select('.x.axis').attr("transform", "translate(0," + brushHeight + ")").call(state.xAxis).selectAll('text').attr('y', 8);
    state.svg.select('.brush').call(state.brush.extent([[0, 0], [brushWidth, brushHeight]])).call(state.brush.move, state.currentSelection.map(state.scale));
  }
});

function alphaNumCmp(a, b) {
  var alist = a.split(/(\d+)/),
      blist = b.split(/(\d+)/);
  alist.length && alist[alist.length - 1] == '' ? alist.pop() : null; // remove the last element if empty

  blist.length && blist[blist.length - 1] == '' ? blist.pop() : null; // remove the last element if empty

  for (var i = 0, len = Math.max(alist.length, blist.length); i < len; i++) {
    if (alist.length == i || blist.length == i) {
      // Out of bounds for one of the sides
      return alist.length - blist.length;
    }

    if (alist[i] != blist[i]) {
      // find the first non-equal part
      if (alist[i].match(/\d/)) // if numeric
        {
          return +alist[i] - +blist[i]; // compare as number
        } else {
        return alist[i].toLowerCase() > blist[i].toLowerCase() ? 1 : -1; // compare as string
      }
    }
  }

  return 0;
}

var timelines = Kapsule({
  props: {
    data: {
      "default": [],
      onChange: function onChange(data, state) {
        parseData(data);
        state.zoomX = [min(state.completeFlatData, function (d) {
          return d.timeRange[0];
        }), max(state.completeFlatData, function (d) {
          return d.timeRange[1];
        })];
        state.zoomY = [null, null];

        if (state.overviewArea) {
          state.overviewArea.domainRange(state.zoomX).currentSelection(state.zoomX);
        } //


        function parseData(rawData) {
          state.completeStructData = [];
          state.completeFlatData = [];
          state.totalNLines = 0;

          for (var i = 0, ilen = rawData.length; i < ilen; i++) {
            var group = rawData[i].group;
            state.completeStructData.push({
              group: group,
              lines: rawData[i].data.map(function (d) {
                return d.label;
              })
            });

            for (var j = 0, jlen = rawData[i].data.length; j < jlen; j++) {
              for (var k = 0, klen = rawData[i].data[j].data.length; k < klen; k++) {
                var _rawData$i$data$j$dat = rawData[i].data[j].data[k],
                    timeRange = _rawData$i$data$j$dat.timeRange,
                    val = _rawData$i$data$j$dat.val,
                    labelVal = _rawData$i$data$j$dat.labelVal,
                    name = _rawData$i$data$j$dat.name;
                state.completeFlatData.push({
                  group: group,
                  label: rawData[i].data[j].label,
                  timeRange: timeRange.map(function (d) {
                    return new Date(d);
                  }),
                  val: val,
                  labelVal: labelVal !== undefined ? labelVal : val,
                  data: rawData[i].data[j].data[k],
                  name: name !== undefined ? name : k
                });
              }

              state.totalNLines++;
            }
          }
        }
      }
    },
    width: {
      "default": window.innerWidth
    },
    maxHeight: {
      "default": 640
    },
    maxLineHeight: {
      "default": 12
    },
    leftMargin: {
      "default": 90
    },
    rightMargin: {
      "default": 100
    },
    topMargin: {
      "default": 26
    },
    bottomMargin: {
      "default": 30
    },
    useUtc: {
      "default": false
    },
    xTickFormat: {},
    dateMarker: {},
    timeFormat: {
      "default": '%Y-%m-%d %-I:%M:%S %p',
      triggerUpdate: false
    },
    zoomX: {
      // Which time-range to show (null = min/max)
      "default": [null, null],
      onChange: function onChange(zoomX, state) {
        if (state.svg) state.svg.dispatch('zoom', {
          detail: {
            zoomX: zoomX,
            zoomY: null,
            redraw: false
          }
        });
      }
    },
    zoomY: {
      // Which lines to show (null = min/max) [0 indexed]
      "default": [null, null],
      onChange: function onChange(zoomY, state) {
        if (state.svg) state.svg.dispatch('zoom', {
          detail: {
            zoomX: null,
            zoomY: zoomY,
            redraw: false
          }
        });
      }
    },
    minSegmentDuration: {},
    zColorScale: {
      "default": scaleSequential(interpolateRdYlBu)
    },
    zQualitative: {
      "default": false,
      onChange: function onChange(discrete, state) {
        state.zColorScale = discrete ? scaleOrdinal([].concat(_toConsumableArray(schemeCategory10), _toConsumableArray(schemeSet3))) : scaleSequential(interpolateRdYlBu); // alt: d3.interpolateInferno
      }
    },
    zDataLabel: {
      "default": '',
      triggerUpdate: false
    },
    // Units of z data. Used in the tooltip descriptions
    zScaleLabel: {
      "default": '',
      triggerUpdate: false
    },
    // Units of colorScale. Used in the legend label
    enableOverview: {
      "default": true
    },
    // True/False
    enableAnimations: {
      "default": true,
      onChange: function onChange(val, state) {
        state.transDuration = val ? 700 : 0;
      }
    },
    segmentTooltipContent: {
      triggerUpdate: false
    },
    // Callbacks
    onZoom: {},
    // When user zooms in / resets zoom. Returns ([startX, endX], [startY, endY])
    onLabelClick: {},
    // When user clicks on a group or y label. Returns (group) or (label, group) respectively
    onSegmentClick: {} // When user clicks on a segment. Returns (segment object) respectively

  },
  methods: {
    getNLines: function getNLines(s) {
      return s.nLines;
    },
    getTotalNLines: function getTotalNLines(s) {
      return s.totalNLines;
    },
    getVisibleStructure: function getVisibleStructure(s) {
      return s.structData;
    },
    getSvg: function getSvg(s) {
      return select(s.svg.node().parentNode).html();
    },
    zoomYLabels: function zoomYLabels(state, _) {
      if (!_) {
        return [y2Label(state.zoomY[0]), y2Label(state.zoomY[1])];
      }

      return this.zoomY([label2Y(_[0], true), label2Y(_[1], false)]); //

      function y2Label(y) {
        if (y == null) return y;
        var cntDwn = y;

        for (var i = 0, len = state.completeStructData.length; i < len; i++) {
          if (state.completeStructData[i].lines.length > cntDwn) return getIdxLine(state.completeStructData[i], cntDwn);
          cntDwn -= state.completeStructData[i].lines.length;
        } // y larger than all lines, return last


        return getIdxLine(state.completeStructData[state.completeStructData.length - 1], state.completeStructData[state.completeStructData.length - 1].lines.length - 1); //

        function getIdxLine(grpData, idx) {
          return {
            'group': grpData.group,
            'label': grpData.lines[idx]
          };
        }
      }

      function label2Y(label, useIdxAfterIfNotFound) {
        useIdxAfterIfNotFound = useIdxAfterIfNotFound || false;
        var subIdxNotFound = useIdxAfterIfNotFound ? 0 : 1;
        if (label == null) return label;
        var idx = 0;

        for (var i = 0, lenI = state.completeStructData.length; i < lenI; i++) {
          var grpCmp = state.grpCmpFunction(label.group, state.completeStructData[i].group);
          if (grpCmp < 0) break;

          if (grpCmp == 0 && label.group == state.completeStructData[i].group) {
            for (var j = 0, lenJ = state.completeStructData[i].lines.length; j < lenJ; j++) {
              var cmpRes = state.labelCmpFunction(label.label, state.completeStructData[i].lines[j]);

              if (cmpRes < 0) {
                return idx + j - subIdxNotFound;
              }

              if (cmpRes == 0 && label.label == state.completeStructData[i].lines[j]) {
                return idx + j;
              }
            }

            return idx + state.completeStructData[i].lines.length - subIdxNotFound;
          }

          idx += state.completeStructData[i].lines.length;
        }

        return idx - subIdxNotFound;
      }
    },
    sort: function sort(state, labelCmpFunction, grpCmpFunction) {
      if (labelCmpFunction == null) {
        labelCmpFunction = state.labelCmpFunction;
      }

      if (grpCmpFunction == null) {
        grpCmpFunction = state.grpCmpFunction;
      }

      state.labelCmpFunction = labelCmpFunction;
      state.grpCmpFunction = grpCmpFunction;
      state.completeStructData.sort(function (a, b) {
        return grpCmpFunction(a.group, b.group);
      });

      for (var i = 0, len = state.completeStructData.length; i < len; i++) {
        state.completeStructData[i].lines.sort(labelCmpFunction);
      }

      state._rerender();

      return this;
    },
    sortAlpha: function sortAlpha(state, asc) {
      if (asc == null) {
        asc = true;
      }

      var alphaCmp = function alphaCmp(a, b) {
        return alphaNumCmp(asc ? a : b, asc ? b : a);
      };

      return this.sort(alphaCmp, alphaCmp);
    },
    sortChrono: function sortChrono(state, asc) {
      if (asc == null) {
        asc = true;
      }

      function buildIdx(accessFunction) {
        var idx = {};

        var _loop = function _loop(i, len) {
          var key = accessFunction(state.completeFlatData[i]);

          if (idx.hasOwnProperty(key)) {
            return "continue";
          }

          var itmList = state.completeFlatData.filter(function (d) {
            return key == accessFunction(d);
          });
          idx[key] = [min(itmList, function (d) {
            return d.timeRange[0];
          }), max(itmList, function (d) {
            return d.timeRange[1];
          })];
        };

        for (var i = 0, len = state.completeFlatData.length; i < len; i++) {
          var _ret = _loop(i);

          if (_ret === "continue") continue;
        }

        return idx;
      }

      var timeCmp = function timeCmp(a, b) {
        var aT = a[1],
            bT = b[1];
        if (!aT || !bT) return null; // One of the two vals is null

        if (aT[1].getTime() == bT[1].getTime()) {
          if (aT[0].getTime() == bT[0].getTime()) {
            return alphaNumCmp(a[0], b[0]); // If first and last is same, use alphaNum
          }

          return aT[0] - bT[0]; // If last is same, earliest first wins
        }

        return bT[1] - aT[1]; // latest last wins
      };

      function getCmpFunction(accessFunction, asc) {
        return function (a, b) {
          return timeCmp(accessFunction(asc ? a : b), accessFunction(asc ? b : a));
        };
      }

      var grpIdx = buildIdx(function (d) {
        return d.group;
      });
      var lblIdx = buildIdx(function (d) {
        return d.label;
      });
      var grpCmp = getCmpFunction(function (d) {
        return [d, grpIdx[d] || null];
      }, asc);
      var lblCmp = getCmpFunction(function (d) {
        return [d, lblIdx[d] || null];
      }, asc);
      return this.sort(lblCmp, grpCmp);
    },
    overviewDomain: function overviewDomain(state, _) {
      if (!state.enableOverview) {
        return null;
      }

      if (!_) {
        return state.overviewArea.domainRange();
      }

      state.overviewArea.domainRange(_);
      return this;
    },
    refresh: function refresh(state) {
      state._rerender();

      return this;
    }
  },
  stateInit: {
    height: null,
    overviewHeight: 20,
    // Height of overview section in bottom
    minLabelFont: 2,
    groupBkgGradient: ['#FAFAFA', '#E0E0E0'],
    yScale: null,
    grpScale: null,
    xAxis: null,
    xGrid: null,
    yAxis: null,
    grpAxis: null,
    dateMarkerLine: null,
    svg: null,
    graph: null,
    overviewAreaElem: null,
    overviewArea: null,
    graphW: null,
    graphH: null,
    completeStructData: null,
    structData: null,
    completeFlatData: null,
    flatData: null,
    totalNLines: null,
    nLines: null,
    minSegmentDuration: 0,
    // ms
    transDuration: 700,
    // ms for transition duration
    labelCmpFunction: alphaNumCmp,
    grpCmpFunction: alphaNumCmp
  },
  init: function init(el, state) {
    var elem = select(el).attr('class', 'timelines-chart');
    state.svg = elem.append('svg');
    state.overviewAreaElem = elem.append('div'); // Initialize scales and axes

    state.yScale = scalePoint();
    state.grpScale = scaleOrdinal();
    state.xAxis = axisBottom();
    state.xGrid = axisTop();
    state.yAxis = axisRight();
    state.grpAxis = axisLeft();
    buildDomStructure();
    addTooltips();
    addZoomSelection();
    setEvents(); //

    function buildDomStructure() {
      state.yScale.invert = invertOrdinal;
      state.grpScale.invert = invertOrdinal;
      state.groupGradId = gradient().colorScale(scaleLinear().domain([0, 1]).range(state.groupBkgGradient)).angle(-90)(state.svg.node()).id();
      var axises = state.svg.append('g').attr('class', 'axises');
      axises.append('g').attr('class', 'x-axis');
      axises.append('g').attr('class', 'x-grid');
      axises.append('g').attr('class', 'y-axis');
      axises.append('g').attr('class', 'grp-axis');
      state.yAxis.scale(state.yScale).tickSize(0);
      state.grpAxis.scale(state.grpScale).tickSize(0);
      state.colorLegend = ColorLegend()(state.svg.append('g').attr('class', 'legendG').node());
      state.graph = state.svg.append('g');
      state.dateMarkerLine = state.svg.append('line').attr('class', 'x-axis-date-marker');

      if (state.enableOverview) {
        addOverviewArea();
      } // Applies to ordinal scales (invert not supported in d3)


      function invertOrdinal(val, cmpFunc) {
        cmpFunc = cmpFunc || function (a, b) {
          return a >= b;
        };

        var scDomain = this.domain();
        var scRange = this.range();

        if (scRange.length === 2 && scDomain.length !== 2) {
          // Special case, interpolate range vals
          scRange = range(scRange[0], scRange[1], (scRange[1] - scRange[0]) / scDomain.length);
        }

        var bias = scRange[0];

        for (var i = 0, len = scRange.length; i < len; i++) {
          if (cmpFunc(scRange[i] + bias, val)) {
            return scDomain[Math.round(i * scDomain.length / scRange.length)];
          }
        }

        return this.domain()[this.domain().length - 1];
      }

      function addOverviewArea() {
        state.overviewArea = TimeOverview().margins({
          top: 1,
          right: 20,
          bottom: 20,
          left: 20
        }).onChange(function (startTime, endTime) {
          state.svg.dispatch('zoom', {
            detail: {
              zoomX: [startTime, endTime],
              zoomY: null
            }
          });
        }).domainRange(state.zoomX).currentSelection(state.zoomX)(state.overviewAreaElem.node());
        state.svg.on('zoomScent', function () {
          var zoomX = event.detail.zoomX;
          if (!state.overviewArea || !zoomX) return; // Out of overview bounds > extend it

          if (zoomX[0] < state.overviewArea.domainRange()[0] || zoomX[1] > state.overviewArea.domainRange()[1]) {
            state.overviewArea.domainRange([new Date(Math.min(zoomX[0], state.overviewArea.domainRange()[0])), new Date(Math.max(zoomX[1], state.overviewArea.domainRange()[1]))]);
          }

          state.overviewArea.currentSelection(zoomX);
        });
      }
    }

    function addTooltips() {
      state.groupTooltip = d3Tip().attr('class', 'chart-tooltip group-tooltip').direction('w').offset([0, 0]).html(function (d) {
        var leftPush = d.hasOwnProperty('timeRange') ? state.xScale(d.timeRange[0]) : 0;
        var topPush = d.hasOwnProperty('label') ? state.grpScale(d.group) - state.yScale(d.group + '+&+' + d.label) : 0;
        state.groupTooltip.offset([topPush, -leftPush]);
        return d.group;
      });
      state.svg.call(state.groupTooltip);
      state.lineTooltip = d3Tip().attr('class', 'chart-tooltip line-tooltip').direction('e').offset([0, 0]).html(function (d) {
        var rightPush = d.hasOwnProperty('timeRange') ? state.xScale.range()[1] - state.xScale(d.timeRange[1]) : 0;
        state.lineTooltip.offset([0, rightPush]);
        return d.label;
      });
      state.svg.call(state.lineTooltip);
      state.segmentTooltip = d3Tip().attr('class', 'chart-tooltip segment-tooltip').direction('s').offset([5, 0]).html(function (d) {
        if (state.segmentTooltipContent) {
          return state.segmentTooltipContent(d);
        }

        var normVal = state.zColorScale.domain()[state.zColorScale.domain().length - 1] - state.zColorScale.domain()[0];
        var dateFormat = (state.useUtc ? utcFormat : timeFormat)("".concat(state.timeFormat).concat(state.useUtc ? ' (UTC)' : ''));
        return '<strong>' + d.labelVal + ' </strong>' + state.zDataLabel + (normVal ? ' (<strong>' + Math.round((d.val - state.zColorScale.domain()[0]) / normVal * 100 * 100) / 100 + '%</strong>)' : '') + '<br>' + '<strong>Name: </strong>' + d.name + '<br>' + '<strong>From: </strong>' + dateFormat(d.timeRange[0]) + '<br>' + '<strong>To: </strong>' + dateFormat(d.timeRange[1]);
      });
      state.svg.call(state.segmentTooltip);
    }

    function addZoomSelection() {
      state.graph.on('mousedown', function () {
        if (select(window).on('mousemove.zoomRect') != null) // Selection already active
          return;
        var e = this;
        if (mouse(e)[0] < 0 || mouse(e)[0] > state.graphW || mouse(e)[1] < 0 || mouse(e)[1] > state.graphH) return;
        state.disableHover = true;
        var rect = state.graph.append('rect').attr('class', 'chart-zoom-selection');
        var startCoords = mouse(e);
        select(window).on('mousemove.zoomRect', function () {
          event.stopPropagation();
          var newCoords = [Math.max(0, Math.min(state.graphW, mouse(e)[0])), Math.max(0, Math.min(state.graphH, mouse(e)[1]))];
          rect.attr('x', Math.min(startCoords[0], newCoords[0])).attr('y', Math.min(startCoords[1], newCoords[1])).attr('width', Math.abs(newCoords[0] - startCoords[0])).attr('height', Math.abs(newCoords[1] - startCoords[1]));
          state.svg.dispatch('zoomScent', {
            detail: {
              zoomX: [startCoords[0], newCoords[0]].sort(ascending).map(state.xScale.invert),
              zoomY: [startCoords[1], newCoords[1]].sort(ascending).map(function (d) {
                return state.yScale.domain().indexOf(state.yScale.invert(d)) + (state.zoomY && state.zoomY[0] ? state.zoomY[0] : 0);
              })
            }
          });
        }).on('mouseup.zoomRect', function () {
          select(window).on('mousemove.zoomRect', null).on('mouseup.zoomRect', null);
          select('body').classed('stat-noselect', false);
          rect.remove();
          state.disableHover = false;
          var endCoords = [Math.max(0, Math.min(state.graphW, mouse(e)[0])), Math.max(0, Math.min(state.graphH, mouse(e)[1]))];
          if (startCoords[0] == endCoords[0] && startCoords[1] == endCoords[1]) return;
          var newDomainX = [startCoords[0], endCoords[0]].sort(ascending).map(state.xScale.invert);
          var newDomainY = [startCoords[1], endCoords[1]].sort(ascending).map(function (d) {
            return state.yScale.domain().indexOf(state.yScale.invert(d)) + (state.zoomY && state.zoomY[0] ? state.zoomY[0] : 0);
          });
          var changeX = newDomainX[1] - newDomainX[0] > 60 * 1000; // Zoom damper

          var changeY = newDomainY[0] != state.zoomY[0] || newDomainY[1] != state.zoomY[1];

          if (changeX || changeY) {
            state.svg.dispatch('zoom', {
              detail: {
                zoomX: changeX ? newDomainX : null,
                zoomY: changeY ? newDomainY : null
              }
            });
          }
        }, true);
        event.stopPropagation();
      });
      state.resetBtn = state.svg.append('text').attr('class', 'reset-zoom-btn').text('Reset Zoom').style('text-anchor', 'end').on('mouseup', function () {
        state.svg.dispatch('resetZoom');
      }).on('mouseover', function () {
        select(this).style('opacity', 1);
      }).on('mouseout', function () {
        select(this).style('opacity', .6);
      });
    }

    function setEvents() {
      state.svg.on('zoom', function () {
        var evData = event.detail,
            zoomX = evData.zoomX,
            zoomY = evData.zoomY,
            redraw = evData.redraw == null ? true : evData.redraw;
        if (!zoomX && !zoomY) return;
        if (zoomX) state.zoomX = zoomX;
        if (zoomY) state.zoomY = zoomY;
        state.svg.dispatch('zoomScent', {
          detail: {
            zoomX: zoomX,
            zoomY: zoomY
          }
        });
        if (!redraw) return;

        state._rerender();

        if (state.onZoom) state.onZoom(state.zoomX, state.zoomY);
      });
      state.svg.on('resetZoom', function () {
        var prevZoomX = state.zoomX;
        var prevZoomY = state.zoomY || [null, null];
        var newZoomX = state.enableOverview ? state.overviewArea.domainRange() : [min(state.flatData, function (d) {
          return d.timeRange[0];
        }), max(state.flatData, function (d) {
          return d.timeRange[1];
        })],
            newZoomY = [null, null];

        if (prevZoomX[0] < newZoomX[0] || prevZoomX[1] > newZoomX[1] || prevZoomY[0] != newZoomY[0] || prevZoomY[1] != newZoomX[1]) {
          state.zoomX = [new Date(Math.min(prevZoomX[0], newZoomX[0])), new Date(Math.max(prevZoomX[1], newZoomX[1]))];
          state.zoomY = newZoomY;
          state.svg.dispatch('zoomScent', {
            detail: {
              zoomX: state.zoomX,
              zoomY: state.zoomY
            }
          });

          state._rerender();
        }

        if (state.onZoom) state.onZoom(null, null);
      });
    }
  },
  update: function update(state) {
    applyFilters();
    setupDimensions();
    adjustXScale();
    adjustYScale();
    adjustGrpScale();
    renderAxises();
    renderGroups();
    renderTimelines();
    adjustLegend(); //

    function applyFilters() {
      // Flat data based on segment length
      state.flatData = state.minSegmentDuration > 0 ? state.completeFlatData.filter(function (d) {
        return d.timeRange[1] - d.timeRange[0] >= state.minSegmentDuration;
      }) : state.completeFlatData; // zoomY

      if (state.zoomY == null || state.zoomY == [null, null]) {
        state.structData = state.completeStructData;
        state.nLines = 0;

        for (var i = 0, len = state.structData.length; i < len; i++) {
          state.nLines += state.structData[i].lines.length;
        }

        return;
      }

      state.structData = [];
      var cntDwn = [state.zoomY[0] == null ? 0 : state.zoomY[0]]; // Initial threshold

      cntDwn.push(Math.max(0, (state.zoomY[1] == null ? state.totalNLines : state.zoomY[1] + 1) - cntDwn[0])); // Number of lines

      state.nLines = cntDwn[1];

      var _loop3 = function _loop3(_i, _len) {
        var validLines = state.completeStructData[_i].lines;

        if (state.minSegmentDuration > 0) {
          // Use only non-filtered (due to segment length) groups/labels
          if (!state.flatData.some(function (d) {
            return d.group == state.completeStructData[_i].group;
          })) {
            return "continue"; // No data for this group
          }

          validLines = state.completeStructData[_i].lines.filter(function (d) {
            return state.flatData.some(function (dd) {
              return dd.group == state.completeStructData[_i].group && dd.label == d;
            });
          });
        }

        if (cntDwn[0] >= validLines.length) {
          // Ignore whole group (before start)
          cntDwn[0] -= validLines.length;
          return "continue";
        }

        var groupData = {
          group: state.completeStructData[_i].group,
          lines: null
        };

        if (validLines.length - cntDwn[0] >= cntDwn[1]) {
          // Last (or first && last) group (partial)
          groupData.lines = validLines.slice(cntDwn[0], cntDwn[1] + cntDwn[0]);
          state.structData.push(groupData);
          cntDwn[1] = 0;
          return "break";
        }

        if (cntDwn[0] > 0) {
          // First group (partial)
          groupData.lines = validLines.slice(cntDwn[0]);
          cntDwn[0] = 0;
        } else {
          // Middle group (full fit)
          groupData.lines = validLines;
        }

        state.structData.push(groupData);
        cntDwn[1] -= groupData.lines.length;
      };

      _loop2: for (var _i = 0, _len = state.completeStructData.length; _i < _len; _i++) {
        var _ret2 = _loop3(_i);

        switch (_ret2) {
          case "continue":
            continue;

          case "break":
            break _loop2;
        }
      }

      state.nLines -= cntDwn[1];
    }

    function setupDimensions() {
      state.graphW = state.width - state.leftMargin - state.rightMargin;
      state.graphH = min([state.nLines * state.maxLineHeight, state.maxHeight - state.topMargin - state.bottomMargin]);
      state.height = state.graphH + state.topMargin + state.bottomMargin;
      state.svg.transition().duration(state.transDuration).attr('width', state.width).attr('height', state.height);
      state.graph.attr('transform', 'translate(' + state.leftMargin + ',' + state.topMargin + ')');

      if (state.overviewArea) {
        state.overviewArea.width(state.width * 0.8).height(state.overviewHeight + state.overviewArea.margins().top + state.overviewArea.margins().bottom);
      }
    }

    function adjustXScale() {
      state.zoomX[0] = state.zoomX[0] || min(state.flatData, function (d) {
        return d.timeRange[0];
      });
      state.zoomX[1] = state.zoomX[1] || max(state.flatData, function (d) {
        return d.timeRange[1];
      });
      state.xScale = (state.useUtc ? scaleUtc : scaleTime)().domain(state.zoomX).range([0, state.graphW]).clamp(true);

      if (state.overviewArea) {
        state.overviewArea.scale(state.xScale.copy()).tickFormat(state.xTickFormat);
      }
    }

    function adjustYScale() {
      var labels = [];

      var _loop4 = function _loop4(i, len) {
        labels = labels.concat(state.structData[i].lines.map(function (d) {
          return state.structData[i].group + '+&+' + d;
        }));
      };

      for (var i = 0, len = state.structData.length; i < len; i++) {
        _loop4(i);
      }

      state.yScale.domain(labels);
      state.yScale.range([state.graphH / labels.length * 0.5, state.graphH * (1 - 0.5 / labels.length)]);
    }

    function adjustGrpScale() {
      state.grpScale.domain(state.structData.map(function (d) {
        return d.group;
      }));
      var cntLines = 0;
      state.grpScale.range(state.structData.map(function (d) {
        var pos = (cntLines + d.lines.length / 2) / state.nLines * state.graphH;
        cntLines += d.lines.length;
        return pos;
      }));
    }

    function adjustLegend() {
      state.svg.select('.legendG').transition().duration(state.transDuration).attr('transform', "translate(".concat(state.leftMargin + state.graphW * 0.05, ",2)"));
      state.colorLegend.width(Math.max(120, state.graphW / 3 * (state.zQualitative ? 2 : 1))).height(state.topMargin * .6).scale(state.zColorScale).label(state.zScaleLabel);
      state.resetBtn.transition().duration(state.transDuration).attr('x', state.leftMargin + state.graphW * .99).attr('y', state.topMargin * .8);
      fitToBox().bbox({
        width: state.graphW * .4,
        height: Math.min(13, state.topMargin * .8)
      })(state.resetBtn.node());
    }

    function renderAxises() {
      state.svg.select('.axises').attr('transform', 'translate(' + state.leftMargin + ',' + state.topMargin + ')'); // X

      var nXTicks = Math.max(2, Math.min(12, Math.round(state.graphW * 0.012)));
      state.xAxis.scale(state.xScale).ticks(nXTicks).tickFormat(state.xTickFormat);
      state.xGrid.scale(state.xScale).ticks(nXTicks).tickFormat('');
      state.svg.select('g.x-axis').style('stroke-opacity', 0).style('fill-opacity', 0).attr('transform', 'translate(0,' + state.graphH + ')').transition().duration(state.transDuration).call(state.xAxis).style('stroke-opacity', 1).style('fill-opacity', 1);
      /* Angled x axis labels
       state.svg.select('g.x-axis').selectAll('text')
       .style('text-anchor', 'end')
       .attr('transform', 'translate(-10, 3) rotate(-60)');
       */

      state.xGrid.tickSize(state.graphH);
      state.svg.select('g.x-grid').attr('transform', 'translate(0,' + state.graphH + ')').transition().duration(state.transDuration).call(state.xGrid);

      if (state.dateMarker && state.dateMarker >= state.xScale.domain()[0] && state.dateMarker <= state.xScale.domain()[1]) {
        state.dateMarkerLine.style('display', 'block').transition().duration(state.transDuration).attr('x1', state.xScale(state.dateMarker) + state.leftMargin).attr('x2', state.xScale(state.dateMarker) + state.leftMargin).attr('y1', state.topMargin + 1).attr('y2', state.graphH + state.topMargin);
      } else {
        state.dateMarkerLine.style('display', 'none');
      } // Y


      var fontVerticalMargin = 0.6;
      var labelDisplayRatio = Math.ceil(state.nLines * state.minLabelFont / Math.sqrt(2) / state.graphH / fontVerticalMargin);
      var tickVals = state.yScale.domain().filter(function (d, i) {
        return !(i % labelDisplayRatio);
      });
      var fontSize = Math.min(12, state.graphH / tickVals.length * fontVerticalMargin * Math.sqrt(2));
      var maxChars = Math.ceil(state.rightMargin / (fontSize / Math.sqrt(2)));
      state.yAxis.tickValues(tickVals);
      state.yAxis.tickFormat(function (d) {
        return reduceLabel(d.split('+&+')[1], maxChars);
      });
      state.svg.select('g.y-axis').transition().duration(state.transDuration).attr('transform', 'translate(' + state.graphW + ', 0)').style('font-size', fontSize + 'px').call(state.yAxis); // Grp

      var minHeight = min(state.grpScale.range(), function (d, i) {
        return i > 0 ? d - state.grpScale.range()[i - 1] : d * 2;
      });
      fontSize = Math.min(14, minHeight * fontVerticalMargin * Math.sqrt(2));
      maxChars = Math.floor(state.leftMargin / (fontSize / Math.sqrt(2)));
      state.grpAxis.tickFormat(function (d) {
        return reduceLabel(d, maxChars);
      });
      state.svg.select('g.grp-axis').transition().duration(state.transDuration).style('font-size', fontSize + 'px').call(state.grpAxis); // Make Axises clickable

      if (state.onLabelClick) {
        state.svg.selectAll('g.y-axis,g.grp-axis').selectAll('text').style('cursor', 'pointer').on('click', function (d) {
          var segms = d.split('+&+');
          state.onLabelClick.apply(state, _toConsumableArray(segms.reverse()));
        });
      } //


      function reduceLabel(label, maxChars) {
        return label.length <= maxChars ? label : label.substring(0, maxChars * 2 / 3) + '...' + label.substring(label.length - maxChars / 3, label.length);
      }
    }

    function renderGroups() {
      var groups = state.graph.selectAll('rect.series-group').data(state.structData, function (d) {
        return d.group;
      });
      groups.exit().transition().duration(state.transDuration).style('stroke-opacity', 0).style('fill-opacity', 0).remove();
      var newGroups = groups.enter().append('rect').attr('class', 'series-group').attr('x', 0).attr('y', 0).attr('height', 0).style('fill', 'url(#' + state.groupGradId + ')').on('mouseover', state.groupTooltip.show).on('mouseout', state.groupTooltip.hide);
      newGroups.append('title').text('click-drag to zoom in');
      groups = groups.merge(newGroups);
      groups.transition().duration(state.transDuration).attr('width', state.graphW).attr('height', function (d) {
        return state.graphH * d.lines.length / state.nLines;
      }).attr('y', function (d) {
        return state.grpScale(d.group) - state.graphH * d.lines.length / state.nLines / 2;
      });
    }

    function renderTimelines(maxElems) {
      if (maxElems < 0) maxElems = null;
      var hoverEnlargeRatio = .4;

      var dataFilter = function dataFilter(d, i) {
        return (maxElems == null || i < maxElems) && state.grpScale.domain().indexOf(d.group) + 1 && d.timeRange[1] >= state.xScale.domain()[0] && d.timeRange[0] <= state.xScale.domain()[1] && state.yScale.domain().indexOf(d.group + '+&+' + d.label) + 1;
      };

      state.lineHeight = state.graphH / state.nLines * 0.8;
      var timelines = state.graph.selectAll('rect.series-segment').data(state.flatData.filter(dataFilter), function (d) {
        return d.group + d.label + d.timeRange[0];
      });
      timelines.exit().transition().duration(state.transDuration).style('fill-opacity', 0).remove();
      var newSegments = timelines.enter().append('rect').attr('class', 'series-segment').attr('rx', 1).attr('ry', 1).attr('x', state.graphW / 2).attr('y', state.graphH / 2).attr('width', 0).attr('height', 0).style('fill', function (d) {
        return state.zColorScale(d.val);
      }).style('fill-opacity', 0).on('mouseover.groupTooltip', state.groupTooltip.show).on('mouseout.groupTooltip', state.groupTooltip.hide).on('mouseover.lineTooltip', state.lineTooltip.show).on('mouseout.lineTooltip', state.lineTooltip.hide).on('mouseover.segmentTooltip', state.segmentTooltip.show).on('mouseout.segmentTooltip', state.segmentTooltip.hide);
      newSegments.on('mouseover', function () {
        if ('disableHover' in state && state.disableHover) return;
        moveToFront()(this);
        var hoverEnlarge = state.lineHeight * hoverEnlargeRatio;
        select(this).transition().duration(70).attr('x', function (d) {
          return state.xScale(d.timeRange[0]) - hoverEnlarge / 2;
        }).attr('width', function (d) {
          return max([1, state.xScale(d.timeRange[1]) - state.xScale(d.timeRange[0])]) + hoverEnlarge;
        }).attr('y', function (d) {
          return state.yScale(d.group + '+&+' + d.label) - (state.lineHeight + hoverEnlarge) / 2;
        }).attr('height', state.lineHeight + hoverEnlarge).style('fill-opacity', 1);
      }).on('mouseout', function () {
        select(this).transition().duration(250).attr('x', function (d) {
          return state.xScale(d.timeRange[0]);
        }).attr('width', function (d) {
          return max([1, state.xScale(d.timeRange[1]) - state.xScale(d.timeRange[0])]);
        }).attr('y', function (d) {
          return state.yScale(d.group + '+&+' + d.label) - state.lineHeight / 2;
        }).attr('height', state.lineHeight).style('fill-opacity', .8);
      }).on('click', function (s) {
        if (state.onSegmentClick) state.onSegmentClick(s);
      });
      timelines = timelines.merge(newSegments);
      timelines.transition().duration(state.transDuration).attr('x', function (d) {
        return state.xScale(d.timeRange[0]);
      }).attr('width', function (d) {
        return max([1, state.xScale(d.timeRange[1]) - state.xScale(d.timeRange[0])]);
      }).attr('y', function (d) {
        return state.yScale(d.group + '+&+' + d.label) - state.lineHeight / 2;
      }).attr('height', state.lineHeight).style('fill-opacity', .8);
    }
  }
});

export default timelines;
