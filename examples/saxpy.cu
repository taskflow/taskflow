#include <taskflow/taskflow.hpp>

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

  tf::Taskflow taskflow;
  tf::Executor executor;

  std::vector<float> hx, hy;

  float* dx {nullptr};
  float* dy {nullptr};
  
  // allocate x
  auto allocate_x = taskflow.emplace([&]() {
    hx.resize(N, 1.0f);
    cudaMalloc(&dx, N*sizeof(float));
  }).name("allocate_x");

  // allocate y
  auto allocate_y = taskflow.emplace([&]() {
    hy.resize(N, 2.0f);
    cudaMalloc(&dy, N*sizeof(float));
  }).name("allocate_y");
  
  // saxpy
  auto cudaflow = taskflow.emplace([&](tf::cudaFlow& cf) {
    auto h2d_x = cf.copy(dx, hx.data(), N);
    auto h2d_y = cf.copy(dy, hy.data(), N);
    auto d2h_x = cf.copy(hx.data(), dx, N);
    auto d2h_y = cf.copy(hy.data(), dy, N);
    auto kernel = cf.kernel(
      {(N+255)/256, 1, 1}, {256, 1, 1}, 0, saxpy, N, 2.0f, dx, dy
    );
    kernel.succeed(h2d_x, h2d_y)
          .precede(d2h_x, d2h_y);
  });

  cudaflow.succeed(allocate_x, allocate_y);

  executor.run(taskflow).wait();

  return 0;
}

