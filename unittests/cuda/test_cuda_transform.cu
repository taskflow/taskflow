#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>
#include <taskflow/cuda/algorithm/transform.hpp>

constexpr float eps = 0.0001f;

void run_and_wait(tf::cudaGraphExec& exec) {
  tf::cudaStream stream;
  stream.run(exec).synchronize();
}

// ----------------------------------------------------------------------------
// cudaflow transform 1
// ----------------------------------------------------------------------------

template <typename T>
void transform1() {

  tf::Taskflow taskflow;
  tf::Executor executor;
  
  for(int n=1; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){

      T v1 = ::rand() % 100;
      T v2 = ::rand() % 100;

      std::vector<T> hx, hy;

      T* dx {nullptr};
      T* dy {nullptr};
      
      // allocate x
      hx.resize(n, v1);
      REQUIRE(cudaMalloc(&dx, n*sizeof(T)) == cudaSuccess);

      // allocate y
      hy.resize(n, v2);
      REQUIRE(cudaMalloc(&dy, n*sizeof(T)) == cudaSuccess);
      
      // axpy
      tf::cudaGraph cg;
      auto h2d_x = cg.copy(dx, hx.data(), n);
      auto h2d_y = cg.copy(dy, hy.data(), n);
      auto d2h_x = cg.copy(hx.data(), dx, n);
      auto d2h_y = cg.copy(hy.data(), dy, n);
      auto kernel = cg.transform(dx, dx+n, dy, 
        [] __device__ (T x) { return x + 2;  }
      );
      kernel.succeed(h2d_x, h2d_y)
            .precede(d2h_x, d2h_y);

      tf::cudaGraphExec exec(cg);
      run_and_wait(exec);

      // verify the result 
      for (int i = 0; i < n; i++) {
        REQUIRE(std::fabs(hx[i] - v1) < eps);
        REQUIRE(std::fabs(hy[i] - (hx[i] + 2)) < eps);
      }

      // update the kernel and run the cf again
      exec.transform(kernel, dy, dy+n, dx,
        [] __device__ (T y) { return y - 4; }
      );
      
      run_and_wait(exec); 
      
      // verify the result 
      for (int i = 0; i < n; i++) {
        REQUIRE(std::fabs(hx[i] - (v1 - 2)) < eps);
        REQUIRE(std::fabs(hy[i] - (v1 + 2)) < eps);
      }

      // free memory
      REQUIRE(cudaFree(dx) == cudaSuccess);
      REQUIRE(cudaFree(dy) == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cudaGraph.transform1.int" * doctest::timeout(300)) {
  transform1<int>();
}

TEST_CASE("cudaGraph.transform1.float" * doctest::timeout(300)) {
  transform1<float>();
}

TEST_CASE("cudaGraph.transform1.double" * doctest::timeout(300)) {
  transform1<double>();
}

// ----------------------------------------------------------------------------
// cudaGraph transform2
// ----------------------------------------------------------------------------

template <typename T>
void transform2() {

  tf::Taskflow taskflow;
  tf::Executor executor;
  
  for(int n=1; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){

      T v1 = ::rand() % 100;
      T v2 = ::rand() % 100;
      T v3 = ::rand() % 100;

      std::vector<T> hx, hy, hz;

      T* dx {nullptr};
      T* dy {nullptr};
      T* dz {nullptr};
      
      // allocate x
      hx.resize(n, v1);
      REQUIRE(cudaMalloc(&dx, n*sizeof(T)) == cudaSuccess);

      // allocate y
      hy.resize(n, v2);
      REQUIRE(cudaMalloc(&dy, n*sizeof(T)) == cudaSuccess);
      
      // allocate z
      hz.resize(n, v3);
      REQUIRE(cudaMalloc(&dz, n*sizeof(T)) == cudaSuccess);
      
      // axpy
      tf::cudaGraph cg;
      auto h2d_x = cg.copy(dx, hx.data(), n);
      auto h2d_y = cg.copy(dy, hy.data(), n);
      auto h2d_z = cg.copy(dz, hz.data(), n);
      auto d2h_x = cg.copy(hx.data(), dx, n);
      auto d2h_y = cg.copy(hy.data(), dy, n);
      auto d2h_z = cg.copy(hz.data(), dz, n);
      auto kernel = cg.transform(dx, dx+n, dy, dz,
        [] __device__ (T x, T y) { return x + y;  }
      );
      kernel.succeed(h2d_x, h2d_y, h2d_z)
            .precede(d2h_x, d2h_y, d2h_z);

      tf::cudaGraphExec exec(cg);

      run_and_wait(exec);

      // verify the result 
      for (int i = 0; i < n; i++) {
        REQUIRE(std::fabs(hx[i] - v1) < eps);
        REQUIRE(std::fabs(hy[i] - v2) < eps);
        REQUIRE(std::fabs(hz[i] - v1 - v2) < eps);
      }

      // update the kernel and run the exec again
      // dz = v1 + v2
      // dx = v1
      // dy = v2
      exec.transform(kernel, dz, dz+n, dx, dy,
        [] __device__ (T z, T x) { return z + x + T(10); }
      );
      
      run_and_wait(exec); 
      
      // verify the result 
      for (int i = 0; i < n; i++) {
        REQUIRE(std::fabs(hy[i] - (v1 + v2 + v1 + T(10))) < eps);
      }

      // free memory
      REQUIRE(cudaFree(dx) == cudaSuccess);
      REQUIRE(cudaFree(dy) == cudaSuccess);
      REQUIRE(cudaFree(dz) == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cudaGraph.transform2.int" * doctest::timeout(300)) {
  transform2<int>();
}

TEST_CASE("cudaGraph.transform2.float" * doctest::timeout(300)) {
  transform2<float>();
}

TEST_CASE("cudaGraph.transform2.double" * doctest::timeout(300)) {
  transform2<double>();
}

