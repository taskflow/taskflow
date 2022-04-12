// This program demonstrates how to create a simple SAXPY
// ("single-precision AX+Y") task graph using syclFlow.

#include <taskflow/taskflow.hpp>
#include <taskflow/sycl/syclflow.hpp>

constexpr size_t N = 1000000;

int main() {

  tf::Executor executor;
  tf::Taskflow taskflow("saxpy example");

  sycl::queue queue;

  // allocate shared memory
  auto X = sycl::malloc_shared<float>(N, queue);
  auto Y = sycl::malloc_shared<float>(N, queue);

  // create a syclFlow to perform the saxpy operation
  taskflow.emplace_on([&](tf::syclFlow& sf){

    tf::syclTask fillX = sf.fill(X, 1.0f, N).name("fillX");
    tf::syclTask fillY = sf.fill(Y, 2.0f, N).name("fillY");

    tf::syclTask saxpy = sf.parallel_for(sycl::range<1>(N),
      [=] (sycl::id<1> id) {
        X[id] = 3.0f * X[id] + Y[id];
      }
    ).name("saxpy");

    saxpy.succeed(fillX, fillY);

  }, queue).name("syclFlow");

  // dump the graph without detailed syclFlow connections
  taskflow.dump(std::cout);
  
  // run the taskflow
  executor.run(taskflow).wait();

  // dump the graph with all syclFlow details (after executed)
  taskflow.dump(std::cout);

  // verify the result
  for(size_t i=0; i<N; i++) {
    if(std::fabs(X[i]-5.0f) >= 1e-4) {
      throw std::runtime_error("incorrect saxpy result (expected 5.0f)");
    }
  }

  std::cout << "correct saxpy result\n";

  // free the memory
  sycl::free(X, queue);
  sycl::free(Y, queue);

  return 0;
}




