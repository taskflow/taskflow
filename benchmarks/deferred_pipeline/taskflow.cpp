#include "deferred_pipeline.hpp"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

struct frame {
  size_t fid;
  char ftype;
  std::vector<int> depend_on;
  frame() = default;

  frame(const size_t id, char t, const std::vector<int>& vec):
    fid{id}, ftype{t}, depend_on{vec} {}
};

void construct_video(std::vector<frame>& video, const size_t num_frames, std::string type) {
  for (size_t i = 0; i < num_frames; ++i) {
    frame f;
    // video_1 frame pattern
    if (type == "1") {
      f.fid = i;
      f.ftype = video_1[i%128];
    }
    // video_2 frame pattern
    else {
      f.fid = i;
      f.ftype = video_2[i%300];
    }
    video.emplace_back(std::move(f));
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

  //for (size_t i = 0; i < video.size(); ++i) {
  //  std::cout << "frame[" << i << "] is " << video[i].ftype << ", depends on ";

  //  for (const auto& d : video[i].depend_on) {
  //    std::cout << d << ' ';
  //  }
  //  std::cout << '\n';
  //}
}

std::chrono::microseconds measure_time_taskflow(
  size_t num_threads, std::string type, size_t num_frames) {

  // declare a x264 format video
  std::vector<frame> video;
  construct_video(video, num_frames, type);
  
  std::chrono::microseconds elapsed;

  auto beg = std::chrono::high_resolution_clock::now();
  
  tf::Taskflow taskflow;
  static tf::Executor executor(num_threads);

  tf::Pipeline pl(
    num_threads,
    tf::Pipe{tf::PipeType::SERIAL, [num_frames, &video](auto& pf) mutable {
      if(pf.token() == num_frames) {
        pf.stop();
        return;
      }
      else {
        if (video[pf.token()].ftype == 'I') {
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
              
                if (video[pf.token()].depend_on.size() > 1) {
                  depend_on_id = video[pf.token()].depend_on[1];
                  pf.defer(depend_on_id);
                }
              }
            break;

            default:
              // proceed to the next npipe
            break;
          }
        }
      }
    }},

    tf::Pipe{tf::PipeType::PARALLEL, [&video](auto& pf) {
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

  taskflow.composed_of(pl).name("module_of_pipeline");

  executor.run(taskflow).wait();
  
  auto end = std::chrono::high_resolution_clock::now();

  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
  return elapsed;
}


