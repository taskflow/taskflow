#include <taskflow/taskflow.hpp>  // the only include you need

int main(){

  tf::Executor executor;

  auto A = executor.silent_dependent_async("A", [](){ std::cout << "A\n"; });
  auto B = executor.silent_dependent_async("B", [](){ std::cout << "B\n"; });
  executor.silent_dependent_async("C", [](){ std::cout << "C\n"; }, A, B);

  executor.wait_for_all();

  return 0;
}
