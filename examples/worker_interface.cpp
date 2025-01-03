// This program demonstrates how to change the worker behavior
// upon the creation of an executor.

#include <taskflow/taskflow.hpp>


// ----------------------------------------------------------------------------
// Affinity
// ----------------------------------------------------------------------------
#if defined(__linux__)
  #include <sched.h>
  #include <pthread.h>
#elif defined(_WIN32)
  #include <windows.h>
#elif defined(__APPLE__)
  #include <mach/mach.h>
  #include <mach/thread_policy.h>
#endif

// affine the given thread to a specific core
bool affine(std::thread& thread, size_t core_id) {
#if defined(__linux__)
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  pthread_t native_handle = thread.native_handle();
  return pthread_setaffinity_np(native_handle, sizeof(cpu_set_t), &cpuset) == 0;
#elif defined(_WIN32)
  return SetThreadAffinityMask(thread.native_handle(), 1ULL << core_id) != 0;
#elif defined(__APPLE__)
  thread_port_t native_handle = pthread_mach_thread_np(thread.native_handle());
  thread_affinity_policy_data_t policy = {static_cast<integer_t>(core_id)};
  return thread_policy_set(
    native_handle, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, 1
  ) == KERN_SUCCESS;
#else
  // Unsupported platform
  return false;
#endif
}

// ----------------------------------------------------------------------------

class CustomWorkerBehavior : public tf::WorkerInterface {

  public:
  
  // to call before the worker enters the scheduling loop
  void scheduler_prologue(tf::Worker& w) override {
    printf("worker %zu prepares to enter the work-stealing loop\n", w.id());
    
    // now affine the worker to a particular CPU core equal to its id
    if(affine(w.thread(), w.id())) {
      printf("successfully affines worker %zu to CPU core %zu\n", w.id(), w.id());
    }
    else {
      printf("failed to affine worker %zu to CPU core %zu\n", w.id(), w.id());
    }
  }

  // to call after the worker leaves the scheduling loop
  void scheduler_epilogue(tf::Worker& w, std::exception_ptr) override {
    printf("worker %zu left the work-stealing loop\n", w.id());
  }
};

int main() {
  tf::Executor executor(4, tf::make_worker_interface<CustomWorkerBehavior>());
  return 0;
}


