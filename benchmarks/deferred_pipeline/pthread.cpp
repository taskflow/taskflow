#include "deferred_pipeline.hpp"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cassert>

class Frame {
public:
  bool processed = false;
  size_t fid;
  char ftype;
  std::mutex m;                                                                                          
  std::condition_variable  cv;
  std::vector<int> depend_on;

  Frame(size_t i, char t) : fid{i}, ftype{t} {}
};

std::vector<std::unique_ptr<Frame>> video;

std::vector<int> vec1;
std::mutex global_m;

void encode_frame(const size_t idx, const size_t num_threads,
  const size_t num_frames) {
   
  size_t fid;
  size_t iterations = 1+(num_frames-1)/num_threads;

  for (size_t i = 0; i < iterations; ++i) {
    fid = num_threads*i+idx;
    if (fid >= num_frames) {
      break;
    }

    // I frame does not have any depency
    if (video[fid]->ftype == 'I') {
      std::unique_lock lk(video[fid]->m);  
      work_I();
      video[fid]->processed = true;
      video[fid]->cv.notify_all();
      continue;
    }

    else if (video[fid]->ftype == 'P') {
      // wait on the dependency 
      {
        int depend_on_id = video[fid]->depend_on[0];

        std::unique_lock lk(video[depend_on_id]->m);
        video[depend_on_id]->cv.wait(lk, [&]{
          return video[depend_on_id]->processed; 
        });
      }
      // dependency is resolved
      {
        std::unique_lock lk(video[fid]->m);
        work_P();
        video[fid]->processed = true;
        video[fid]->cv.notify_all();
      }
    }

    else {
      // wait on the first dependency
      int depend_on_id = video[fid]->depend_on[0];
      {
        std::unique_lock lk(video[depend_on_id]->m);
        video[depend_on_id]->cv.wait(lk, [&]{
          return video[depend_on_id]->processed; 
        });
      }

      // wait on the second dependency
      if (video[fid]->depend_on.size() > 1) {
        depend_on_id = video[fid]->depend_on[1];
      }
      
      {
        std::unique_lock lk(video[depend_on_id]->m);
        video[depend_on_id]->cv.wait(lk, [&]{
          return video[depend_on_id]->processed; 
        });
      }
      
      // dependency is resolved
      {
        std::unique_lock lk(video[fid]->m);
        work_B();
        video[fid]->processed = true;
        video[fid]->cv.notify_all();
      }
    }
  }
}

std::chrono::microseconds measure_time_pthread(
  size_t  num_threads, std::string type, size_t num_frames) {

  std::chrono::microseconds elapsed;
  
  std::vector<std::thread> threads;
  
  for (size_t i = 0; i < num_frames; ++i) {
    // x264 frame pattern is viedo_1
    // video_1 has 128 frames in total
    if (type == "1") {
      std::unique_ptr<Frame> p(new Frame(i, video_1[i%128]));
      video.emplace_back(std::move(p));
    }
    // x264 frame pattern is video_2.
    // video_2 has 300 frames in total.
    else {
      std::unique_ptr<Frame> p(new Frame(i, video_2[i%300]));
      video.emplace_back(std::move(p));
    }
  }

  // construct the depend_on vector of each frame
  for (size_t i = 0; i < num_frames; ++i) {
    // I frames do not depend on other frames
    if (video[i]->ftype == 'I') {
      continue;
    }
    // P frames depend on its previous I or P frame
    // have one dependency
    else if (video[i]->ftype == 'P') {
      int p_idx = i-1;
      while(p_idx >= 0) {
        if (video[p_idx]->ftype == 'I' || video[p_idx]->ftype == 'P') {
          video[i]->depend_on.push_back(p_idx);
          break; 
        }
        --p_idx;
      }
    }
    // B frames depend on its previous I or P frame and its later I or P frame
    // have up to two dependencies
    else {
      int p_idx = i-1, l_idx = i+1;
      while(p_idx >= 0) {
        if (video[p_idx]->ftype == 'I' || video[p_idx]->ftype == 'P') {
          video[i]->depend_on.push_back(p_idx);
          break;
        }
        --p_idx;
      }
      while(l_idx < static_cast<int>(num_frames)) {
        if (video[l_idx]->ftype == 'I' || video[l_idx]->ftype == 'P') {
          video[i]->depend_on.push_back(l_idx);
          break;
        }
        ++l_idx;
      }
    }
  }

  auto beg = std::chrono::high_resolution_clock::now();
  
  for (size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back(
      std::thread(encode_frame, i, num_threads, num_frames));
  }
  
  // join all threads
  for (size_t i = 0; i < num_threads; ++i) {
    threads[i].join();
  }

  auto end = std::chrono::high_resolution_clock::now();
  
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);

  
  return elapsed;
}
