#include <taskflow/taskflow.hpp>

void loop_body() {  }

int main(int argc, const char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./for_loop num_iterations" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const size_t N = std::atoi(argv[1]);
  
  // expressing loop using Taskflow's control taskflow graph programming model
  tf::Executor executor;
  tf::Taskflow taskflow;

  auto X = taskflow.emplace([](){});
  auto Y = taskflow.emplace([&, i=size_t(0)]() mutable { 
    loop_body();
    return (i++ < N) ? 0 : 1;
  });
  auto Z = taskflow.emplace([](){});
  X.precede(Y);
  Y.precede(Y, Z);
  
  auto tbeg = std::chrono::steady_clock::now();
  executor.run(taskflow).wait();
  auto tend = std::chrono::steady_clock::now();
  std::cout << "in-graph control flow time    : " << std::chrono::duration_cast<std::chrono::microseconds>(tend-tbeg).count()
                             << " us\n";

  // expressing the loop without using cointrol taskflow graph
  tf::Taskflow taskflow_x, taskflow_y, taskflow_z;
  taskflow_x.emplace([](){});
  taskflow_y.emplace(loop_body);
  taskflow_z.emplace([](){});

  tbeg = std::chrono::steady_clock::now();
  executor.run(taskflow_x).wait();
  for(size_t i=0; i<N; ++i) {
    executor.run(taskflow_y).wait();
  }
  executor.run(taskflow_z).wait();
  tend = std::chrono::steady_clock::now();
  std::cout << "out-of-graph control flow time: " << std::chrono::duration_cast<std::chrono::microseconds>(tend-tbeg).count()
                             << " us\n";


  return 0;
}
