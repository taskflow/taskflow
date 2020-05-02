// DOM objects
const tfp_timeline = document.getElementById('tfp_timeline');

//const valColorScale = d3.scaleOrdinal()
//  .domain(['static task', 'condition task', 'cudaflow', 'dynamic task', 'module task'])
//  .range(['red', 'green', 'blue', 'black', 'yellow']);

const valColors = {
  'static task': 'red',
  'condition task': 'green',
  'cudaflow': 'blue',
  'dynamic task' : 'red',
  'module task': 'blue'
};

// timeline objects
tfp_render_simple();

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
      console.log(json);
      tfp_render_timeline(json);
    }
    catch(e) {
      $('#tfp_textarea').addClass('is-invalid');
      console.log(e);
    }


  }, 2000));
});

// render the timeline from a parsed json
function tfp_render_timeline(json) {
    
  // clear the existing timeline
  //document.getElementById("MyDiv").innerHTML = "";
  $('#tfp_timeline').html('');

  delete timeline;
  timeline = TimelinesChart();
  timeline.xTickFormat(n => +n);
  timeline.zQualitative(true);
  timeline.timeFormat('%Q');
  timeline.data(json);
  timeline.maxHeight(Infinity);
  timeline.maxLineHeight(16);
  //timeline.zColorScale(val => valColors[val]);
  timeline(tfp_timeline);
}

// render default data

function tfp_render_simple() {
  tfp_render_timeline(simple);
  $('#tfp_textarea').text(JSON.stringify(simple, null, 2));
}

function tfp_render_matmul() {
  tfp_render_timeline(matmul);
  $('#tfp_textarea').text(JSON.stringify(matmul));
}

function tfp_render_kmeans() {
  tfp_render_timeline(kmeans);
  $('#tfp_textarea').text(JSON.stringify(kmeans));
}

function tfp_render_inference() {
  tfp_render_timeline(inference);
  $('#tfp_textarea').text(JSON.stringify(inference))
}

function tfp_render_dreamplace() {
  tfp_render_timeline(dreamplace);
  $('#tfp_textarea').text(JSON.stringify(dreamplace))
}

