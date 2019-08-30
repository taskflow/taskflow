#include "universe.h"
#include <taskflow/taskflow.hpp> 


void seismic_taskflow(unsigned num_threads, unsigned num_frames, Universe& u) { 

  tf::Executor executor(num_threads);
  tf::Taskflow taskflow;

  int UniverseHeight = u.UniverseHeight;

  auto stress_tasks = taskflow.dynamic_parallel_for(0, UniverseHeight-1, 1, [&](int i) mutable {
    u.UpdateStress(Universe::Rectangle(0, i, u.UniverseWidth-1, 1));
  }, executor.num_workers());

  auto velocity_tasks = taskflow.dynamic_parallel_for(1, UniverseHeight, 1, [&](int i) mutable {
    u.UpdateVelocity(Universe::Rectangle(1, i, u.UniverseWidth-1, 1));
  }, executor.num_workers());

  std::get<1>(stress_tasks).precede(std::get<0>(velocity_tasks));
  std::get<0>(stress_tasks).work([&](){ u.UpdatePulse(); });

  executor.run_n(taskflow, num_frames).get();
}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads, unsigned num_frames, Universe& u) {
  auto beg = std::chrono::high_resolution_clock::now();
  seismic_taskflow(num_threads, num_frames, u);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

