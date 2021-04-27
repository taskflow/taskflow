// This program performs a simple single-precision Ax+Y operation
// using a cudaFlow capturer and verifies its result.

#include <taskflow/cudaflow.hpp>
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

  tf::Taskflow taskflow ("saxpy-flow");
  tf::Executor executor;

  std::vector<float> hx(N, 1.0f), hy(N, 2.0f);

  auto dx = tf::cuda_malloc_device<float>(N);
  auto dy = tf::cuda_malloc_device<float>(N);

  taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
    std::cout << "running cudaflow capturer ...\n";
    auto h2d_x = cf.copy(dx, hx.data(), N).name("h2d_x");
    auto h2d_y = cf.copy(dy, hy.data(), N).name("h2d_y");
    auto d2h_x = cf.copy(hx.data(), dx, N).name("d2h_x");
    auto d2h_y = cf.copy(hy.data(), dy, N).name("d2h_y");
    auto kernel = cf.kernel((N+255)/256, 256, 0, saxpy, N, 2.0f, dx, dy)
                    .name("saxpy");
    kernel.succeed(h2d_x, h2d_y)
          .precede(d2h_x, d2h_y);
  }).name("saxpy");
  
  executor.run(taskflow).wait();

  float max_error = 0.0f;
  for (size_t i = 0; i < N; i++) {
    max_error = std::max(max_error, abs(hx[i]-1.0f));
    max_error = std::max(max_error, abs(hy[i]-4.0f));
  }
  std::cout << "saxpy finished with max error: " << max_error << '\n';

  // free memory
  tf::cuda_free(dx);
  tf::cuda_free(dy);

  std::cout << "dumping the taskflow ...\n";
  taskflow.dump(std::cout);

  return 0;
}

