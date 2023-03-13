// This program performs a simple single-precision Ax+Y operation
// using cudaFlow and verifies its result.

#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>

// Kernel: saxpy
__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a*x[i] + y[i];
  }
}

// Function: main
int main() {
  
  const unsigned N = 1<<20;

  tf::Taskflow taskflow ("saxpy-flow");
  tf::Executor executor;

  std::vector<float> hx, hy;

  float* dx {nullptr};
  float* dy {nullptr};
  
  // allocate x
  auto allocate_x = taskflow.emplace([&]() {
    std::cout << "allocating host x and device x ...\n";
    hx.resize(N, 1.0f);
    cudaMalloc(&dx, N*sizeof(float));
  }).name("allocate_x");

  // allocate y
  auto allocate_y = taskflow.emplace([&]() {
    std::cout << "allocating host y and device y ...\n";
    hy.resize(N, 2.0f);
    cudaMalloc(&dy, N*sizeof(float));
  }).name("allocate_y");
  
  // saxpy cudaFlow
  auto cudaflow = taskflow.emplace([&]() {
    
    std::cout << "running cudaflow ...\n";

    tf::cudaFlow cf;
    auto h2d_x = cf.copy(dx, hx.data(), N).name("h2d_x");
    auto h2d_y = cf.copy(dy, hy.data(), N).name("h2d_y");
    auto d2h_x = cf.copy(hx.data(), dx, N).name("d2h_x");
    auto d2h_y = cf.copy(hy.data(), dy, N).name("d2h_y");
    auto kernel = cf.kernel((N+255)/256, 256, 0, saxpy, N, 2.0f, dx, dy)
                    .name("saxpy");
    kernel.succeed(h2d_x, h2d_y)
          .precede(d2h_x, d2h_y);
    
    std::cout << "launching cudaflow ...\n";
    tf::cudaStream stream;
    cf.run(stream);
    stream.synchronize();
    
    // visualize this cudaflow
    cf.dump(std::cout);

  }).name("saxpy");

  cudaflow.succeed(allocate_x, allocate_y);

  // Add a verification task
  auto verifier = taskflow.emplace([&](){
    float max_error = 0.0f;
    for (size_t i = 0; i < N; i++) {
      max_error = std::max(max_error, abs(hx[i]-1.0f));
      max_error = std::max(max_error, abs(hy[i]-4.0f));
    }
    std::cout << "saxpy finished with max error: " << max_error << '\n';
  }).succeed(cudaflow).name("verify");

  // free memory
  auto deallocate_x = taskflow.emplace([&](){
    std::cout << "deallocating device x ...\n";
    cudaFree(dx);
  }).name("deallocate_x");
  
  auto deallocate_y = taskflow.emplace([&](){
    std::cout << "deallocating device y ...\n";
    cudaFree(dy);
  }).name("deallocate_y");

  verifier.precede(deallocate_x, deallocate_y);

  executor.run(taskflow).wait();

  std::cout << "dumping the taskflow ...\n";
  taskflow.dump(std::cout);

  return 0;
}

