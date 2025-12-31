// This example demonstrates how to use Taskflow's recursive tasking features,
// using the famous Fibonacci recursion as an example.
#include <taskflow/taskflow.hpp>

tf::Executor& get_executor() {
  static tf::Executor executor;
  return executor;
}

size_t spawn_async(size_t N) {
  
  if (N < 2) {
    return N; 
  }
  
  size_t res1, res2;
  
  // create a task group
  tf::TaskGroup tg = get_executor().task_group();
  
  tg.silent_async([N, &res1](){ res1 = spawn_async(N-1); });

  // tail optimization
  res2 = spawn_async(N-2);

  tg.corun();

  return res1 + res2;
}

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./fibonacci N\n";
    std::exit(EXIT_FAILURE);
  }

  size_t N = std::atoi(argv[1]);

  auto& executor = get_executor();

  auto tbeg = std::chrono::steady_clock::now();
  printf("fib[%zu] = %zu\n", N, executor.async([N](){ return spawn_async(N); }).get());
  auto tend = std::chrono::steady_clock::now();

  std::cout << "elapsed time: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg).count()
            << " ms\n";

  return 0;
}









