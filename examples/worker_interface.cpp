// This program demonstrates how to change the worker behavior
// upon the creation of an executor.

#include <taskflow/taskflow.hpp>

class CustomWorkerBehavior : public tf::WorkerInterface {

  public:
  
  // to call before the worker enters the scheduling loop
  void scheduler_prologue(tf::Worker& w) override {
    printf("worker %lu prepares to enter the work-stealing loop\n", w.id());
    
    // now affine the worker to a particular CPU core equal to its id
    if(tf::affine(w.thread(), w.id())) {
      printf("successfully affines worker %lu to CPU core %lu\n", w.id(), w.id());
    }
    else {
      printf("failed to affine worker %lu to CPU core %lu\n", w.id(), w.id());
    }
  }

  // to call after the worker leaves the scheduling loop
  void scheduler_epilogue(tf::Worker& w, std::exception_ptr) override {
    printf("worker %lu left the work-stealing loop\n", w.id());
  }
};

int main() {
  tf::Executor executor(4, tf::make_worker_interface<CustomWorkerBehavior>());
  return 0;
}
