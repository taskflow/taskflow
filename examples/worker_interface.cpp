// This program demonstrates how to change the worker behavior
// upon the creation of an executor.

#include <taskflow/taskflow.hpp>

class CustomWorkerBehavior : public tf::WorkerInterface {

  public:
  
  // to call before the worker enters the scheduling loop
  void scheduler_prologue(tf::Worker& w) override {
    std::cout << tf::stringify(
      "worker ", w.id(), " (native=", w.thread()->native_handle(), ") enters scheduler\n"
    ); 
  }

  // to call after the worker leaves the scheduling loop
  void scheduler_epilogue(tf::Worker& w, std::exception_ptr) override {
    std::cout << tf::stringify(
      "worker ", w.id(), " (native=", w.thread()->native_handle(), ") leaves scheduler\n"
    ); 
  }
};

int main() {

  tf::Executor executor(4, std::make_shared<CustomWorkerBehavior>());
  
  return 0;
}
