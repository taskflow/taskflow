#include <taskflow/taskflow.hpp>
#include <taskflow/sanitizer/sanitizer.hpp>

int main() {

  tf::Taskflow taskflow;

  // create tasks in taskflow ...
  
  tf::Sanitizer sanitizer;

  sanitizer.check_nonreachable(taskflow, std::cout);


  return 0;
}
