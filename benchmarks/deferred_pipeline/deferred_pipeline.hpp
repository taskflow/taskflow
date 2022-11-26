#include <chrono>
#include <thread>
#include <vector>
#include <iterator>
#include <algorithm>
#include <cassert>
#include <iostream>




// video_1 has the frame patterns extracted from PARSEC's "simlarge" input video.
static std::vector<char> video_1 = {'I','P','P','P','P','P','P','P','P','I','P','I','P','I','P','I','P','I','P','P','P','I','P','I','P','I','P','P','P','P','P','P','P','P','P','I','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','I','P','P','P','P','P','P','P','P','P','P','P','I','P','P','P','P','P','P','I','P','P','P','P','P','P','P','P','I','I','P','P','P','P','P','P','P','I','P','P','P','P','P','P','P','I','P','P','P','I','P','P','I','P','P','I','I','I','I','I','I','I','P','P','P'};


// video_2 has the general frame patterns from an online x.264 video
static std::vector<char> video_2 = {'I','B','B','B','P','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','P','B','B','B','B','P','B','B','B','P','B','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','B','B','P','B','B','B','P','B','B','B','B','P','B','B','B','P','B','B','B','B','P','I','B','B','B','B','P','B','B','B','P','B','B','B','B','P','B','B','B','P','B','B','B','B','P','B','B','B','P','B','B','B','B','P','B','B','B','P','B','B','B','B','P','B','B','B','P','B','B','B','P'};





// time for I frame processing in PARSEC is 0.011941 sec
// ~ 12 ms
inline void work_I() {
  size_t N = 4;
  size_t mat1[N][N];
  size_t mat2[N][N];
  size_t res[N][N];

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      mat1[i][j] = i+1;
      mat2[i][j] = j+1;
    }
  }

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      res[i][j] = 0;
      for (size_t k = 0; k < N; k++) {
        res[i][j] += mat1[i][k] * mat2[k][j];
      }
    }
  }
   
  std::this_thread::sleep_for(std::chrono::microseconds(12));

}

// time for P frame processing in PARSEC is 0.0094 sec
// ~ 9 ms
inline void work_P() {
  size_t N = 4;
  size_t mat1[N][N];
  size_t mat2[N][N];
  size_t res[N][N];

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      mat1[i][j] = i+1;
      mat2[i][j] = j+1;
    }
  }

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      res[i][j] = 0;
      for (size_t k = 0; k < N; k++) {
        res[i][j] += mat1[i][k] * mat2[k][j];
      }
    }
  }
   
  std::this_thread::sleep_for(std::chrono::microseconds(9));

}

// time for B frame processing in PARSEC is not available
// use 11 ms
inline void work_B() {
  size_t N = 4;
  size_t mat1[N][N];
  size_t mat2[N][N];
  size_t res[N][N];

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      mat1[i][j] = i+1;
      mat2[i][j] = j+1;
    }
  }

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      res[i][j] = 0;
      for (size_t k = 0; k < N; k++) {
        res[i][j] += mat1[i][k] * mat2[k][j];
      }
    }
  }
   
  std::this_thread::sleep_for(std::chrono::microseconds(11));

}

/*
inline bool verify(
  size_t num_threads, size_t frequency, int deferred, size_t num_frames,
  std::vector<int>& vec) {

  assert(vec.size() == num_frames);

  for (auto& fid : vec) {
    if (fid != 0 && fid%frequency != 0) {
      continue;
    }
    if (fid+deferred < 0 || 
        fid+deferred >= static_cast<int>(num_frames) || 
        (fid+1)%num_threads == 0) {
      continue;
    }

    auto it  = std::find(vec.begin(), vec.end(), fid);
    auto dit = std::find(vec.begin(), vec.end(), fid+deferred);
    
    if (std::distance(dit, it) > 0) {
      return true;
    }
    else {
      return false;
    }
  }
}
*/

std::chrono::microseconds measure_time_taskflow(size_t, std::string, size_t);
std::chrono::microseconds measure_time_pthread(size_t, std::string, size_t);

