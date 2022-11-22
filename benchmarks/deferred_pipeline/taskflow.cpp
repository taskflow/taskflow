#include "deferred_pipeline.hpp"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

struct Frame {
  bool processed = false;
  size_t fid;
  char ftype;
  std::vector<int> depend_on;
};

std::chrono::microseconds measure_time_taskflow(
  size_t  num_threads, std::string type, size_t num_frames) {
  
  std::vector<Frame> video(num_frames);

  for (size_t i = 0; i < num_frames; ++i) {
    // video_1 frame pattern
    if (type == "1") {
      video[i].fid = i;
      video[i].ftype = video_1[i%128];
    }
    // video_2 frame pattern
    else {
      video[i].fid = i;
      video[i].ftype = video_2[i%300];
    }
  }
  // construct the frame dependency
  for (size_t i = 0; i < num_frames; ++i) {
    if (video[i].ftype == 'I') {
      continue; 
    }
    else if (video[i].ftype == 'P') {
      int p_idx = i-1;
      while(p_idx >= 0) {
        if (video[p_idx].ftype == 'I' || video[p_idx].ftype == 'P') {
          video[i].depend_on.push_back(p_idx);
          break; 
        }
        --p_idx;
      }
    }
    else {
      int p_idx = i-1, l_idx = i+1;
      while(p_idx >= 0) {
        if (video[p_idx].ftype == 'I' || video[p_idx].ftype == 'P') {
          video[i].depend_on.push_back(p_idx);
          break;
        }
        --p_idx;
      }
      while(l_idx < static_cast<int>(num_frames)) {
        if (video[l_idx].ftype == 'I' || video[l_idx].ftype == 'P') {
          video[i].depend_on.push_back(l_idx);
          break;
        }
        ++l_idx;
      }
    }
  }

  std::chrono::microseconds elapsed;
  
  auto beg = std::chrono::high_resolution_clock::now();
  
  tf::Taskflow taskflow;
  
  tf::Executor executor(num_threads);
   
  tf::Pipeline pl(num_threads,
    tf::Pipe{tf::PipeType::SERIAL, [num_frames, &video](tf::Pipeflow& pf) {
      if(pf.token() == num_frames) {
        pf.stop();
      }

      else {
        if (video[pf.token()].ftype == 'I') {
          // no dependency
          // proceed to the next pipe
        }
        else if (video[pf.token()].ftype == 'P') {
          switch(pf.num_deferrals()) {
            case 0:
              {
                int depend_on_id = video[pf.token()].depend_on[0];
                pf.defer(depend_on_id);
              }
            break;

            default:
              // dependency is resolved
              // proceed to the next pipe
            break;
          }
        }
        else {
          switch(pf.num_deferrals()) {
            case 0:
              {
                int depend_on_id = video[pf.token()].depend_on[0];
                pf.defer(depend_on_id);

                // one more dependency
                if (video[pf.token()].depend_on.size() > 1) {
                  depend_on_id = video[pf.token()].depend_on[1];
                  pf.defer(depend_on_id);
                }
              }
            break;

            default:
              // dependency is resolved
              // proceed to the next pipe
            break;
          }
        }
      }
    }},
    
    // second parallel pipe
    tf::Pipe{tf::PipeType::PARALLEL, [&video](tf::Pipeflow& pf){
      if (video[pf.token()].ftype == 'I') {
      work_I();
      }
      else if (video[pf.token()].ftype == 'P') {
        work_P();
      }
      else {
        work_B();
      }
    }}
  );

  taskflow.composed_of(pl);
  executor.run(taskflow).wait();
  
  auto end = std::chrono::high_resolution_clock::now();

  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
 
  return elapsed;
}


