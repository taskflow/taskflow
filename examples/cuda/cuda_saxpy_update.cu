// This program performs a simple single-precision Ax+Y operation
// using cudaGraph and showcase how to update its kernel parameters.

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

  std::vector<float> hx, hy;

  float* dx {nullptr};
  float* dy {nullptr};
  
  // allocate x
  hx.resize(N, 1.0f);
  cudaMalloc(&dx, N*sizeof(float));

  // allocate y
  hy.resize(N, 2.0f);
  cudaMalloc(&dy, N*sizeof(float));
  
  // saxpy cudaGraph: y[i] = 2*1 + 2
  tf::cudaGraph cg;
  auto h2d_x = cg.copy(dx, hx.data(), N);
  auto h2d_y = cg.copy(dy, hy.data(), N);
  auto d2h_x = cg.copy(hx.data(), dx, N);
  auto d2h_y = cg.copy(hy.data(), dy, N);
  auto kernel = cg.kernel((N+255)/256, 256, 0, saxpy, N, 2.0f, dx, dy);
  kernel.succeed(h2d_x, h2d_y)
        .precede(d2h_x, d2h_y);
  
  tf::cudaStream stream;
  tf::cudaGraphExec exec(cg);
  stream.run(exec)
        .synchronize();
  
  // visualize this cudaflow
  cg.dump(std::cout);

  // verify x[i] = 1, y[i] = 2
  float max_error = 0.0f;
  for (size_t i = 0; i < N; i++) {
    max_error = std::max(max_error, abs(hx[i]-1.0f));
    max_error = std::max(max_error, abs(hy[i]-4.0f));
  }
  std::cout << "saxpy finished with max error: " << max_error << '\n';

  // now update the parameters: y[i] = 3*1 + 4
  exec.copy(h2d_x, dy, hy.data(), N);  // dy[i] = 4
  exec.copy(h2d_y, dx, hx.data(), N);  // dx[i] = 1
  exec.kernel(kernel, (N+255)/256, 256, 0, saxpy, N, 3.0f, dx, dy);
  exec.copy(d2h_x, hy.data(), dy, N);  // hy[i] = 7
  exec.copy(d2h_y, hx.data(), dx, N);  // hx[i] = 1

  stream.run(exec)
        .synchronize();
  
  // visualize this cudaflow
  cg.dump(std::cout);
  
  // verify
  max_error = 0.0f;
  for (size_t i = 0; i < N; i++) {
    max_error = std::max(max_error, abs(hx[i]-1.0f));
    max_error = std::max(max_error, abs(hy[i]-7.0f));
  }
  std::cout << "updated saxpy finished with max error: " << max_error << '\n';

  // free memory
  cudaFree(dx);
  cudaFree(dy);

  return 0;
}

