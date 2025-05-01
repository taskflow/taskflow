// This example demonstrates how to use Taskflow's subflow and runtime tasking features
// to create recursive parallelism, using the famous Fibonacci recursion as an example.
#include <taskflow/taskflow.hpp>

tf::Executor& get_executor() {
  static tf::Executor executor;
  return executor;
}

// ------------------------------------------------------------------------------------------------
// implementation using subflow (slower)
// ------------------------------------------------------------------------------------------------

size_t spawn_subflow(size_t n, tf::Subflow& sbf) {
  
  if (n < 2) {
    return n;
  }

  size_t res1, res2;

  sbf.emplace([&res1, n] (tf::Subflow& sbf_n_1) { res1 = spawn_subflow(n - 1, sbf_n_1); } );
  sbf.emplace([&res2, n] (tf::Subflow& sbf_n_2) { res2 = spawn_subflow(n - 2, sbf_n_2); } );

  sbf.join();
  return res1 + res2;
}

size_t fibonacci_subflow(size_t N) {

  size_t res;  // result

  tf::Taskflow taskflow("fibonacci");

  taskflow.emplace([&res, N] (tf::Subflow& sbf) {
    res = spawn_subflow(N, sbf);
  }).name(std::to_string(N));

  get_executor().run(taskflow).wait();
  
  return res;
}

// ------------------------------------------------------------------------------------------------
// implementation using async (faster)
// ------------------------------------------------------------------------------------------------

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

  if(argc != 3) {
    std::cerr << "usage: ./fibonacci N [subflow|async]\n";
    std::exit(EXIT_FAILURE);
  }

  size_t N = std::atoi(argv[1]);

  auto tbeg = std::chrono::steady_clock::now();
  if(std::strcmp(argv[2], "subflow") == 0) {
    printf("fib[%zu] (with subflow) = %zu\n", N, fibonacci_subflow(N));
  }
  else if(std::strcmp(argv[2], "async") == 0) {
    printf("fib[%zu] (with async) = %zu\n", N, fibonacci_async(N));
  }
  else {
    std::cerr << "unrecognized method " << argv[2] << '\n';
    std::exit(EXIT_FAILURE);
  }
  auto tend = std::chrono::steady_clock::now();

  std::cout << "elapsed time: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg).count()
            << " ms\n";

  return 0;
}









