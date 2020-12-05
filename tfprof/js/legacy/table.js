
// ==> inside feed()

// table
const total_span = state.maxX - state.minX;
for(let i=0; i<state.completeTableData.length; i++) {
  state.completeTableData[i].busy = (
    (state.completeTableData[i].busy/total_span*100).toFixed(2)
  );
}

// https://stackoverflow.com/questions/32871044/how-to-update-d3-table
var tr = state.table.select("tbody").selectAll("tr")
  .data(state.completeTableData);

tr.exit().remove();        // remove surplus tr (tr > data)
tr = tr.merge(tr.enter().append('tr'))
       .style("cursor", "pointer");

var td = tr.selectAll("td")
  .data(d => Object.values(d));

td.exit().remove();
td.merge(td.enter().append("td"))
  .text(d=>d);
