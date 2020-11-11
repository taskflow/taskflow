#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cudaflow.hpp>

constexpr float eps = 0.0001f;

// --------------------------------------------------------
// Testcase: add2
// --------------------------------------------------------
template <typename T, typename F>
void add2() {

  //const unsigned N = 1<<20;

  for(size_t N=1; N<=(1<<20); N <<= 1) {

    tf::Taskflow taskflow;
    tf::Executor executor;

    T v1 = ::rand() % 100;
    T v2 = ::rand() % 100;
    T v3 = v1 + v2;

    std::vector<T> hx, hy;

    T* dx {nullptr};
    T* dy {nullptr};
    
    // allocate x
    auto allocate_x = taskflow.emplace([&]() {
      hx.resize(N, v1);
      REQUIRE(cudaMalloc(&dx, N*sizeof(T)) == cudaSuccess);
    }).name("allocate_x");

    // allocate y
    auto allocate_y = taskflow.emplace([&]() {
      hy.resize(N, v2);
      REQUIRE(cudaMalloc(&dy, N*sizeof(T)) == cudaSuccess);
    }).name("allocate_y");
    
    // axpy
    auto cudaflow = taskflow.emplace([&](F& cf) {
      auto h2d_x = cf.copy(dx, hx.data(), N).name("h2d_x");
      auto h2d_y = cf.copy(dy, hy.data(), N).name("h2d_y");
      auto d2h_x = cf.copy(hx.data(), dx, N).name("d2h_x");
      auto d2h_y = cf.copy(hy.data(), dy, N).name("d2h_y");
      //auto kernel = cf.add(dx, N, dx, dy);
      auto kernel = cf.transform(
        dx, dx+N, [] __device__ (T& v1, T& v2) { return v1 + v2; }, 
        dx, dy
      );
      kernel.succeed(h2d_x, h2d_y)
            .precede(d2h_x, d2h_y);
    }).name("saxpy");

    cudaflow.succeed(allocate_x, allocate_y);

    // Add a verification task
    auto verifier = taskflow.emplace([&](){
      for (size_t i = 0; i < N; i++) {
        REQUIRE(std::fabs(hx[i] - v3) < eps);
      }
    }).succeed(cudaflow).name("verify");

    // free memory
    auto deallocate_x = taskflow.emplace([&](){
      REQUIRE(cudaFree(dx) == cudaSuccess);
    }).name("deallocate_x");
    
    auto deallocate_y = taskflow.emplace([&](){
      REQUIRE(cudaFree(dy) == cudaSuccess);
    }).name("deallocate_y");

    verifier.precede(deallocate_x, deallocate_y);

    executor.run(taskflow).wait();
  }
}

TEST_CASE("add2.int" * doctest::timeout(300)) {
  add2<int, tf::cudaFlow>();
}

TEST_CASE("add2.float" * doctest::timeout(300)) {
  add2<float, tf::cudaFlow>();
}

TEST_CASE("add2.double" * doctest::timeout(300)) {
  add2<double, tf::cudaFlow>();
}

TEST_CASE("capture_add2.int" * doctest::timeout(300)) {
  add2<int, tf::cudaFlowCapturer>();
}

TEST_CASE("capture_add2.float" * doctest::timeout(300)) {
  add2<float, tf::cudaFlowCapturer>();
}

TEST_CASE("capture_add2.double" * doctest::timeout(300)) {
  add2<double, tf::cudaFlowCapturer>();
}

// --------------------------------------------------------
// Testcase: add3
// --------------------------------------------------------

template <typename T, typename F>
void add3() {

  //const unsigned N = 1<<20;

  for(size_t N=1; N<=(1<<20); N <<= 1) {

    tf::Taskflow taskflow;
    tf::Executor executor;

    T v1 = ::rand() % 100;
    T v2 = ::rand() % 100;
    T v3 = ::rand() % 100;
    T v4 = v1 + v2 + v3;

    std::vector<T> hx, hy, hz;

    T* dx {nullptr};
    T* dy {nullptr};
    T* dz {nullptr};
    
    // allocate x
    auto allocate_x = taskflow.emplace([&]() {
      hx.resize(N, v1);
      REQUIRE(cudaMalloc(&dx, N*sizeof(T)) == cudaSuccess);
    }).name("allocate_x");

    // allocate y
    auto allocate_y = taskflow.emplace([&]() {
      hy.resize(N, v2);
      REQUIRE(cudaMalloc(&dy, N*sizeof(T)) == cudaSuccess);
    }).name("allocate_y");
    
    // allocate z
    auto allocate_z = taskflow.emplace([&]() {
      hz.resize(N, v3);
      REQUIRE(cudaMalloc(&dz, N*sizeof(T)) == cudaSuccess);
    }).name("allocate_y");
    
    // saxpy
    auto cudaflow = taskflow.emplace([&](F& cf) {
      auto h2d_x = cf.copy(dx, hx.data(), N).name("h2d_x");
      auto h2d_y = cf.copy(dy, hy.data(), N).name("h2d_y");
      auto h2d_z = cf.copy(dz, hz.data(), N).name("h2d_z");
      auto d2h_x = cf.copy(hx.data(), dx, N).name("d2h_x");
      auto d2h_y = cf.copy(hy.data(), dy, N).name("d2h_y");
      auto d2h_z = cf.copy(hz.data(), dz, N).name("d2h_z");

      //auto kernel = cf.add(dx, N, dx, dy, dz);
      auto kernel = cf.transform(
        dx, dx+N, [] __device__ (T& v1, T& v2, T& v3) { return v1 + v2 + v3; }, 
        dx, dy, dz
      );
      kernel.succeed(h2d_x, h2d_y, h2d_z)
            .precede(d2h_x, d2h_y, d2h_z);
    }).name("saxpy");

    cudaflow.succeed(allocate_x, allocate_y, allocate_z);

    // Add a verification task
    auto verifier = taskflow.emplace([&](){
      for (size_t i = 0; i < N; i++) {
        REQUIRE(std::fabs(hx[i] - v4) < eps);
      }
    }).succeed(cudaflow).name("verify");

    // free memory
    auto deallocate_x = taskflow.emplace([&](){
      REQUIRE(cudaFree(dx) == cudaSuccess);
    }).name("deallocate_x");
    
    auto deallocate_y = taskflow.emplace([&](){
      REQUIRE(cudaFree(dy) == cudaSuccess);
    }).name("deallocate_y");
    
    auto deallocate_z = taskflow.emplace([&](){
      REQUIRE(cudaFree(dz) == cudaSuccess);
    }).name("deallocate_z");

    verifier.precede(deallocate_x, deallocate_y, deallocate_z);

    executor.run(taskflow).wait();
  }
}

TEST_CASE("add3.int" * doctest::timeout(300)) {
  add3<int, tf::cudaFlow>();
}

TEST_CASE("add3.float" * doctest::timeout(300)) {
  add3<float, tf::cudaFlow>();
}

TEST_CASE("add3.double" * doctest::timeout(300)) {
  add3<double, tf::cudaFlow>();
}

TEST_CASE("capture_add3.int" * doctest::timeout(300)) {
  add3<int, tf::cudaFlowCapturer>();
}

TEST_CASE("capture_add3.float" * doctest::timeout(300)) {
  add3<float, tf::cudaFlowCapturer>();
}

TEST_CASE("capture_add3.double" * doctest::timeout(300)) {
  add3<double, tf::cudaFlowCapturer>();
}

// --------------------------------------------------------
// Testcase: multiply2
// --------------------------------------------------------
template <typename T, typename F>
void multiply2() {

  //const unsigned N = 1<<20;

  for(size_t N=1; N<=(1<<20); N <<= 1) {

    tf::Taskflow taskflow;
    tf::Executor executor;

    T v1 = ::rand() % 100;
    T v2 = ::rand() % 100;
    T v3 = v1 * v2;

    std::vector<T> hx, hy;

    T* dx {nullptr};
    T* dy {nullptr};
    
    // allocate x
    auto allocate_x = taskflow.emplace([&]() {
      hx.resize(N, v1);
      REQUIRE(cudaMalloc(&dx, N*sizeof(T)) == cudaSuccess);
    }).name("allocate_x");

    // allocate y
    auto allocate_y = taskflow.emplace([&]() {
      hy.resize(N, v2);
      REQUIRE(cudaMalloc(&dy, N*sizeof(T)) == cudaSuccess);
    }).name("allocate_y");
    
    // saxpy
    auto cudaflow = taskflow.emplace([&](F& cf) {
      auto h2d_x = cf.copy(dx, hx.data(), N).name("h2d_x");
      auto h2d_y = cf.copy(dy, hy.data(), N).name("h2d_y");
      auto d2h_x = cf.copy(hx.data(), dx, N).name("d2h_x");
      auto d2h_y = cf.copy(hy.data(), dy, N).name("d2h_y");

      //auto kernel = cf.multiply(dx, N, dx, dy);
      auto kernel = cf.transform(
        dx, dx+N, [] __device__ (T& v1, T& v2) { return v1 * v2; }, 
        dx, dy
      );
      kernel.succeed(h2d_x, h2d_y)
            .precede(d2h_x, d2h_y);
    }).name("saxpy");

    cudaflow.succeed(allocate_x, allocate_y);

    // Add a verification task
    auto verifier = taskflow.emplace([&](){
      for (size_t i = 0; i < N; i++) {
        REQUIRE(std::fabs(hx[i] - v3) < eps);
      }
    }).succeed(cudaflow).name("verify");

    // free memory
    auto deallocate_x = taskflow.emplace([&](){
      REQUIRE(cudaFree(dx) == cudaSuccess);
    }).name("deallocate_x");
    
    auto deallocate_y = taskflow.emplace([&](){
      REQUIRE(cudaFree(dy) == cudaSuccess);
    }).name("deallocate_y");

    verifier.precede(deallocate_x, deallocate_y);

    executor.run(taskflow).wait();
  }
}

TEST_CASE("multiply2.int" * doctest::timeout(300)) {
  multiply2<int, tf::cudaFlow>();
}

TEST_CASE("multiply2.float" * doctest::timeout(300)) {
  multiply2<float, tf::cudaFlow>();
}

TEST_CASE("multiply2.double" * doctest::timeout(300)) {
  multiply2<double, tf::cudaFlow>();
}

TEST_CASE("capture_multiply2.int" * doctest::timeout(300)) {
  multiply2<int, tf::cudaFlowCapturer>();
}

TEST_CASE("capture_multiply2.float" * doctest::timeout(300)) {
  multiply2<float, tf::cudaFlowCapturer>();
}

TEST_CASE("capture_multiply2.double" * doctest::timeout(300)) {
  multiply2<double, tf::cudaFlowCapturer>();
}

// ----------------------------------------------------------------------------
// for_each
// ----------------------------------------------------------------------------

template <typename T, typename F>
void for_each() {

  for(int n=1; n<=123456; n = n*2 + 1) {

    tf::Taskflow taskflow;
    tf::Executor executor;
    
    T* cpu = nullptr;
    T* gpu = nullptr;

    auto cputask = taskflow.emplace([&](){
      cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
      REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);
    });

    tf::Task gputask;
    
    gputask = taskflow.emplace([&](F& cf) {
      auto d2h = cf.copy(cpu, gpu, n);
      auto h2d = cf.copy(gpu, cpu, n);
      auto kernel = cf.for_each(
        gpu, gpu+n, [] __device__ (T& val) { val = 65536; }
      );
      h2d.precede(kernel);
      d2h.succeed(kernel);
    });

    cputask.precede(gputask);
    
    executor.run(taskflow).wait();

    for(int i=0; i<n; i++) {
      REQUIRE(std::fabs(cpu[i] - (T)65536) < eps);
    }

    std::free(cpu);
    REQUIRE(cudaFree(gpu) == cudaSuccess);
  }
}

TEST_CASE("for_each.int" * doctest::timeout(300)) {
  for_each<int, tf::cudaFlow>();
}

TEST_CASE("for_each.float" * doctest::timeout(300)) {
  for_each<float, tf::cudaFlow>();
}

TEST_CASE("for_each.double" * doctest::timeout(300)) {
  for_each<double, tf::cudaFlow>();
}

TEST_CASE("capture_for_each.int" * doctest::timeout(300)) {
  for_each<int, tf::cudaFlowCapturer>();
}

TEST_CASE("capture_for_each.float" * doctest::timeout(300)) {
  for_each<float, tf::cudaFlowCapturer>();
}

TEST_CASE("capture_for_each.double" * doctest::timeout(300)) {
  for_each<double, tf::cudaFlowCapturer>();
}

// --------------------------------------------------------
// Testcase: for_each_index
// --------------------------------------------------------

template <typename T, typename F>
void for_each_index() {

  for(int n=10; n<=123456; n = n*2 + 1) {

    tf::Taskflow taskflow;
    tf::Executor executor;
    
    T* cpu = nullptr;
    T* gpu = nullptr;

    auto cputask = taskflow.emplace([&](){
      cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
      REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);
    });

    auto gputask = taskflow.emplace([&](F& cf) {
      auto d2h = cf.copy(cpu, gpu, n);
      auto h2d = cf.copy(gpu, cpu, n);
      //auto kernel = cf.for_each_index(gpu, n, [] __device__ (T& value){ value = 17; });
      auto kernel1 = cf.for_each_index(
        0, n, 2, 
        [gpu] __device__ (int i) { gpu[i] = 17; }
      );
      auto kernel2 = cf.for_each_index(
        1, n, 2, 
        [=] __device__ (int i) { gpu[i] = -17; }
      );
      h2d.precede(kernel1, kernel2);
      d2h.succeed(kernel1, kernel2);
    });

    cputask.precede(gputask);
    
    executor.run(taskflow).wait();

    for(int i=0; i<n; i++) {
      if(i % 2 == 0) {
        REQUIRE(std::fabs(cpu[i] - (T)17) < eps);
      }
      else {
        REQUIRE(std::fabs(cpu[i] - (T)(-17)) < eps);
      }
    }

    std::free(cpu);
    REQUIRE(cudaFree(gpu) == cudaSuccess);
  }
}

TEST_CASE("for_each_index.int" * doctest::timeout(300)) {
  for_each_index<int, tf::cudaFlow>();
}

TEST_CASE("for_each_index.float" * doctest::timeout(300)) {
  for_each_index<float, tf::cudaFlow>();
}

TEST_CASE("for_each_index.double" * doctest::timeout(300)) {
  for_each_index<double, tf::cudaFlow>();
}

TEST_CASE("capture_for_each_index.int" * doctest::timeout(300)) {
  for_each_index<int, tf::cudaFlowCapturer>();
}

TEST_CASE("capture_for_each_index.float" * doctest::timeout(300)) {
  for_each_index<float, tf::cudaFlowCapturer>();
}

TEST_CASE("capture_for_each_index.double" * doctest::timeout(300)) {
  for_each_index<double, tf::cudaFlowCapturer>();
}

// ----------------------------------------------------------------------------
// transform
// ----------------------------------------------------------------------------

struct TransformFunc {
  __device__ int operator () (int& v1, float& v2, double& v3) {
    v1 = 1;
    v2 = 3.0f;
    v3 = 5.0;
    return 17;
  }
};

template <typename F>
void transform() {

  for(unsigned n=1; n<=123456; n = n*2 + 1) {

    tf::Taskflow taskflow;
    tf::Executor executor;
    
    int* htgt = nullptr;
    int* tgt = nullptr;
    int* hsrc1 = nullptr;
    int* src1 = nullptr;
    float* hsrc2 = nullptr;
    float* src2 = nullptr;
    double* hsrc3 = nullptr;
    double* src3 = nullptr;

    auto htgttask = taskflow.emplace([&](){
      htgt = static_cast<int*>(std::calloc(n, sizeof(int)));
      hsrc1 = static_cast<int*>(std::calloc(n, sizeof(int)));
      hsrc2 = static_cast<float*>(std::calloc(n, sizeof(float)));
      hsrc3 = static_cast<double*>(std::calloc(n, sizeof(double)));
      REQUIRE(cudaMalloc(&tgt, n*sizeof(int)) == cudaSuccess);
      REQUIRE(cudaMalloc(&src1, n*sizeof(int)) == cudaSuccess);
      REQUIRE(cudaMalloc(&src2, n*sizeof(float)) == cudaSuccess);
      REQUIRE(cudaMalloc(&src3, n*sizeof(double)) == cudaSuccess);
    });

    auto gputask = taskflow.emplace([&](F& cf) {
      auto d2h = cf.copy(htgt, tgt, n);
      auto d2h3 = cf.copy(hsrc3, src3, n);
      auto d2h2 = cf.copy(hsrc2, src2, n);
      auto d2h1 = cf.copy(hsrc1, src1, n);
      auto kernel = cf.transform(
        tgt, tgt+n, 
        [] __device__ (int& v1, float& v2, double& v3) -> int {
          v1 = 1;
          v2 = 3.0f;
          v3 = 5.0;
          return 17;
        }, 
        src1, src2, src3
      );
      auto h2d = cf.copy(tgt, htgt, n);
      h2d.precede(kernel);
      kernel.precede(d2h, d2h1, d2h2, d2h3);
    });

    htgttask.precede(gputask);
    
    executor.run(taskflow).wait();

    for(unsigned i=0; i<n; ++i) {
      REQUIRE(htgt[i] == 17);
      REQUIRE(hsrc1[i] == 1);
      REQUIRE(std::fabs(hsrc2[i] - 3.0f) < eps);
      REQUIRE(std::fabs(hsrc3[i] - 5.0) < eps);
    }

    std::free(htgt);
    std::free(hsrc1);
    std::free(hsrc2);
    std::free(hsrc3);
    REQUIRE(cudaFree(tgt) == cudaSuccess);
    REQUIRE(cudaFree(src1) == cudaSuccess);
    REQUIRE(cudaFree(src2) == cudaSuccess);
    REQUIRE(cudaFree(src3) == cudaSuccess);
  }
}

TEST_CASE("transform" * doctest::timeout(300)) {
  transform<tf::cudaFlow>();
}

TEST_CASE("capture_transform" * doctest::timeout(300) ) {
  transform<tf::cudaFlowCapturer>();
}



/*// ----------------------------------------------------------------------------
// row-major transpose
// ----------------------------------------------------------------------------

// Disable for now - better to use cublasFlowCapturer

template <typename T>
__global__
void verify(const T* din_mat, const T* dout_mat, bool* check, size_t rows, size_t cols) {
  
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  size_t size = rows * cols;
  for(; tid < size; tid += gridDim.x * blockDim.x) {
    if(din_mat[tid] != dout_mat[tid / cols + (tid % cols) * rows]) {
      *check = false;
      return;
    }
  }
}

template <typename T>
void transpose() {
  tf::Executor executor;

  for(size_t rows = 1; rows <= 7999; rows*=2+3) {
    for(size_t cols = 1; cols <= 8021; cols*=3+5) {

      tf::Taskflow taskflow;
      std::vector<T> hinput_mat(rows * cols);

      std::generate_n(hinput_mat.begin(), rows * cols, [](){ return ::rand(); });

      T* dinput_mat {nullptr};
      T* doutput_mat {nullptr};
      bool* check {nullptr};
      
       //allocate
      auto allocate = taskflow.emplace([&]() {
        REQUIRE(cudaMalloc(&dinput_mat, (rows * cols) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMalloc(&doutput_mat, (rows * cols) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMallocManaged(&check, sizeof(bool)) == cudaSuccess);
        *check = true;
      }).name("allocate");

       //transpose
      auto cudaflow = taskflow.emplace([&](tf::cudaFlow& cf) {
        auto h2d_input_t = cf.copy(dinput_mat, hinput_mat.data(), rows * cols).name("h2d");

        auto kernel_t = tf::cudaBLAF(cf).transpose(
          dinput_mat,
          doutput_mat,
          rows,
          cols
        );

        auto verify_t = cf.kernel(
          32,
          512,
          0,
          verify<T>,
          dinput_mat,
          doutput_mat,
          check,
          rows,
          cols
        );

        h2d_input_t.precede(kernel_t);
        kernel_t.precede(verify_t);
      }).name("transpose");


       //free memory
      auto deallocate = taskflow.emplace([&](){
        REQUIRE(cudaFree(dinput_mat) == cudaSuccess);
        REQUIRE(cudaFree(doutput_mat) == cudaSuccess);
      }).name("deallocate");
      

      allocate.precede(cudaflow);
      cudaflow.precede(deallocate);

      executor.run(taskflow).wait();
      REQUIRE(*check);
    }
  }
}

TEST_CASE("transpose.int" * doctest::timeout(300) ) {
  transpose<int>();
}

TEST_CASE("transpose.float" * doctest::timeout(300) ) {
  transpose<float>();
}


TEST_CASE("transpose.double" * doctest::timeout(300) ) {
  transpose<double>();
}

// ----------------------------------------------------------------------------
// row-major matrix multiplication
// ----------------------------------------------------------------------------

template <typename T>
void matmul() {
  tf::Taskflow taskflow;
  tf::Executor executor;
  
  std::vector<T> a, b, c;

  for(int m=1; m<=1992; m=2*m+1) {
    for(int k=1; k<=1012; k=2*k+3) {
      for(int n=1; n<=1998; n=2*n+8) {

        taskflow.clear();

        T* ha {nullptr};
        T* hb {nullptr};
        T* hc {nullptr};
        T* da {nullptr};
        T* db {nullptr};
        T* dc {nullptr};
      
        T val_a = ::rand()%5-1;
        T val_b = ::rand()%7-3;

        auto hosta = taskflow.emplace([&](){ 
          a.resize(m*k);
          std::fill_n(a.begin(), m*k, val_a);
          ha = a.data();
          REQUIRE(cudaMalloc(&da, m*k*sizeof(T)) == cudaSuccess);
        }).name("ha");

        auto hostb = taskflow.emplace([&](){ 
          b.resize(k*n);
          std::fill_n(b.begin(), k*n, val_b);
          hb = b.data();
          REQUIRE(cudaMalloc(&db, k*n*sizeof(T)) == cudaSuccess);
        }).name("hb");

        auto hostc = taskflow.emplace([&](){
          c.resize(m*n);
          hc = c.data();
          REQUIRE(cudaMalloc(&dc, m*n*sizeof(T)) == cudaSuccess);
        }).name("hc");

        auto cuda = taskflow.emplace([&](tf::cudaFlow& cf){
          auto pa = cf.copy(da, ha, m*k);
          auto pb = cf.copy(db, hb, k*n);

          auto op = tf::cudaBLAF(cf).matmul(
            da, db, dc, m, k, n 
          ).name("op");

          auto cc = cf.copy(hc, dc, m*n).name("cc");

          op.precede(cc).succeed(pa, pb);
        });

        cuda.succeed(hosta, hostb, hostc);

        executor.run(taskflow).wait();

        int ans = val_a*val_b*k;
        for(const auto& x : c) {
          REQUIRE((int)x == ans);
        }

        REQUIRE(cudaFree(da) == cudaSuccess);
        REQUIRE(cudaFree(db) == cudaSuccess);
        REQUIRE(cudaFree(dc) == cudaSuccess);
      }
    }
  }
}

TEST_CASE("matmul.int" * doctest::timeout(300) ) {
  matmul<int>();
}

TEST_CASE("matmul.float" * doctest::timeout(300) ) {
  matmul<float>();
}

TEST_CASE("matmul.double" * doctest::timeout(300) ) {
  matmul<double>();
}*/

