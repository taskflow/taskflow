/**
  This program demonstrates how to use dependent async tasks to create
  dependent algorithm tasks.
*/

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/algorithm/transform.hpp>
#include <taskflow/algorithm/reduce.hpp>

int main(){

  const size_t N = 65536;

  tf::Executor executor;
  
  int sum{1};
  std::vector<int> data(N);

  // for-each
  tf::AsyncTask A = executor.silent_dependent_async(tf::make_for_each_task(
    data.begin(), data.end(), [](int& i){ i = 1; }
  ));

  // transform
  tf::AsyncTask B = executor.silent_dependent_async(tf::make_transform_task(
    data.begin(), data.end(), data.begin(), [](int& i) { return i*2; }
  ), A);

  // reduce
  tf::AsyncTask C = executor.silent_dependent_async(tf::make_reduce_task(
    data.begin(), data.end(), sum, std::plus<int>{}
  ), B);
  
  // wait for all async task to complete
  executor.wait_for_all();
  
  // verify the result
  if(sum != N*2 + 1) {
    throw std::runtime_error("INCORRECT RESULT");
  }
  else {
    std::cout << "CORRECT RESULT\n";
  }
  
  return 0;
}




