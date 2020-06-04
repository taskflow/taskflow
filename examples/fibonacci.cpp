#include <taskflow/taskflow.hpp>

int spawn(int n, tf::Subflow& sbf) {
  if (n < 2) return n;
  int res1, res2;
  sbf.emplace([&res1, n] (tf::Subflow& sbf) { res1 = spawn(n - 1, sbf); } );
  sbf.emplace([&res2, n] (tf::Subflow& sbf) { res2 = spawn(n - 2, sbf); } );
  sbf.join();
  return res1 + res2;
}

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./fibonacci N\n";
    std::exit(EXIT_FAILURE);
  }

  int N = std::atoi(argv[1]);

  if(N < 0) {
    throw std::runtime_error("N must be non-negative");
  }
  
  int res;  // result

  tf::Executor executor;
  tf::Taskflow taskflow;

  taskflow.emplace([&res, N] (tf::Subflow& sbf) { 
    res = spawn(N, sbf);  
  });

  executor.run(taskflow).wait();

  std::cout << "Fib[" << N << "]: " << res << std::endl;

  return 0;
}









