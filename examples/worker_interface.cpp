// This program demonstrates how to change the worker behavior
// upon the creation of an executor.

#include <taskflow/taskflow.hpp>

class CustomWorkerBehavior : public tf::WorkerInterface {

  public:
  
  // to call before the worker enters the scheduling loop
  void scheduler_prologue(tf::WorkerView wv) override {
    std::cout << tf::stringify("worker ", wv.id(), " enters the scheduler loop\n"); 
  }

  // to call after the worker leaves the scheduling loop
  void scheduler_epilogue(tf::WorkerView wv, std::exception_ptr) override {
    std::cout << tf::stringify("worker ", wv.id(), " leaves the scheduler loop\n"); 
  }

};

int main() {

  tf::Executor executor(4, std::make_shared<CustomWorkerBehavior>());
  
  return 0;
}
