#include <taskflow/taskflow.hpp>
#include <taskflow/sanitizer/sanitizer.hpp>

int main() {

  tf::Taskflow taskflow;

  // create tasks in taskflow ...
  
  tf::Sanitizer sanitizer;
    
  // nonreachable 
  std::vector<tf::Task> res1 = sanitizer.check_nonreachable(taskflow, std::cout);
  
  // 
  //std::vector<std::vector<tf::Task>> res2 = sanitizer.check_deadlock(taskflow, std::cout);

  std::vector<std::vector<tf::Task>> res3 = sanitizer.check_infinite_loop(taskflow, std::cout);

  return 0;
}
