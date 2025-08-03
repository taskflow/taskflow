// This example demonstrates how to use Taskflow's subflow and runtime tasking features
// to create recursive parallelism, using the famous Fibonacci recursion as an example.
#include <taskflow/taskflow.hpp>

tf::Executor& get_executor() {
  static tf::Executor executor;
  return executor;
}

size_t spawn_async(size_t N, tf::Runtime& rt) {

  if (N < 2) {
    return N; 
  }
  
  size_t res1, res2;

  rt.silent_async([N, &res1](tf::Runtime& rt1){ res1 = spawn_async(N-1, rt1); });
  
  // tail optimization
  res2 = spawn_async(N-2, rt);

  // use corun to avoid blocking the worker from waiting the two children tasks to finish
  rt.corun();

  return res1 + res2;
}

size_t fibonacci_async(size_t N) {
  size_t res;
  get_executor().async([N, &res](tf::Runtime& rt){ res = spawn_async(N, rt); }).get();
  return res;
}

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./fibonacci N\n";
    std::exit(EXIT_FAILURE);
  }

  size_t N = std::atoi(argv[1]);

  auto tbeg = std::chrono::steady_clock::now();
  printf("fib[%zu] = %zu\n", N, fibonacci_async(N));
  auto tend = std::chrono::steady_clock::now();

  std::cout << "elapsed time: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg).count()
            << " ms\n";

  return 0;
}









