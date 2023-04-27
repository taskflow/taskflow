#include <taskflow/taskflow.hpp>  // the only include you need

int main(){

  tf::Executor executor;

  std::vector<tf::AsyncTask> range;

  auto A = executor.silent_dependent_async("A", [](){ std::cout << "A\n"; });
  
  auto B = executor.silent_dependent_async("B", [](){ std::cout << "B\n"; });

  range.push_back(A);
  range.push_back(B);

  auto C = executor.silent_dependent_async(
    "C", [](){ std::cout << "C\n"; }, range.begin(), range.end()
  );

  executor.wait_for_all();

  return 0;
}
