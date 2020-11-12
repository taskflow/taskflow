#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cublasflow.hpp>

// ----------------------------------------------------------------------------
// Testcase: gemv and c_gemv
// ----------------------------------------------------------------------------

template <typename T>
void gemv(
  bool row_major,
  const int M, 
  const int N, 
  const std::vector<T>& hA,
  const std::vector<T>& hx,
  const std::vector<T>& golden,
  bool trans
) {

  for(size_t d=0; d<tf::cuda_get_num_devices(); d++) {
  tf::Taskflow taskflow;
  tf::Executor executor;

    auto dA = tf::cuda_malloc_device<T>(M*N, d);
    auto dAlpha = tf::cuda_malloc_device<T>(1, d);
    auto dBeta  = tf::cuda_malloc_device<T>(1, d);

    T* hy;
    T* dx;
    T* dy;

    if(trans) {
      hy = new T[N];
      dx = tf::cuda_malloc_device<T>(M, d);
      dy = tf::cuda_malloc_device<T>(N, d);
    }
    else {
      hy = new T[M];
      dx = tf::cuda_malloc_device<T>(N, d);
      dy = tf::cuda_malloc_device<T>(M, d);
    }

    auto cudaflow = taskflow.emplace_on([=](tf::cudaFlow& cf){

      REQUIRE(tf::cuda_get_device() == d);
      
      auto copyA = cf.copy(dA, hA.data(), M*N);

      tf::cudaTask copyx;

      (trans) ? copyx = cf.copy(dx, hx.data(), M)
              : copyx = cf.copy(dx, hx.data(), N);

      auto alpha = cf.single_task([=] __device__ () { *dAlpha = 1; });
      auto beta  = cf.single_task([=] __device__ () { *dBeta  = 0; });

      tf::cudaTask gemv; 
      
      if(trans) {        
        if(row_major) {       // C = A^T * x (r-major)
          gemv = cf.capture([&](tf::cudaFlowCapturer& cap){
            cap.make_capturer<tf::cublasFlowCapturer>()->c_gemv(
              CUBLAS_OP_T,
              M, N, dAlpha, dA, N, dx, 1, dBeta, dy, 1 
            );
          });
        }
        else {
          gemv = cf.capture([&](tf::cudaFlowCapturer& cap){
            cap.make_capturer<tf::cublasFlowCapturer>()->gemv(
              CUBLAS_OP_N,
              N, M, dAlpha, dA, N, dx, 1, dBeta, dy, 1
            );
          });
        }
      }
      else {            
        if(row_major) {       // C = A * x (r-major)
          gemv = cf.capture([&](tf::cudaFlowCapturer& cap){
            cap.make_capturer<tf::cublasFlowCapturer>()->c_gemv(
              CUBLAS_OP_N,
              M, N, dAlpha, dA, N, dx, 1, dBeta, dy, 1
            );
          });
        }
        else {
          gemv = cf.capture([&](tf::cudaFlowCapturer& cap){
            cap.make_capturer<tf::cublasFlowCapturer>()->gemv(
              CUBLAS_OP_T,
              N, M, dAlpha, dA, N, dx, 1, dBeta, dy, 1
            );
          });
        }
      }
      
      tf::cudaTask copyy; 
      (trans) ? copyy = cf.copy(hy, dy, N)
              : copyy = cf.copy(hy, dy, M);

      gemv.precede(copyy)
          .succeed(copyA, copyx, alpha, beta);
    }, d);

    auto verify = taskflow.emplace([=, &golden](){
      for(size_t i=0; i<golden.size(); i++) {
        //std::cerr << hy[i] << " ";
        REQUIRE(std::fabs(hy[i]-golden[i]) < 0.0001);
      }
      //std::cerr << '\n';
      tf::cuda_free(dA);
      tf::cuda_free(dx);
      tf::cuda_free(dy);
      tf::cuda_free(dAlpha);
      tf::cuda_free(dBeta);
      delete [] hy;
    });
    
    cudaflow.precede(verify);

  executor.run(taskflow).wait();
  }

}

template <typename T>
void gemv_test(bool row_major, bool trans) {

  int M = 3, N = 4;

  const std::vector<T> hA = {
    11, 12, 13, 14,
    15, 16, 17, 18,
    19, 20, 21, 22
  };  // 3x4

  std::vector<T> hx;
  std::vector<T> golden;

  //ha.T * hx
  if(trans) {
    hx = {11, 12, 13};
    golden = {548, 584, 620, 656};
  }
  else {
    hx = {11, 12, 13, 14};
    golden = {630, 830, 1030};
  }

  gemv<T>(row_major, M, N, hA, hx, golden, trans);
}

// gemv (column-major)
TEST_CASE("gemv_n.float") {
  gemv_test<float>(false, false);
}

TEST_CASE("gemv_n.double") {
  gemv_test<double>(false, false);
}

TEST_CASE("gemv_t.float") {
  gemv_test<float>(false, true);
}

TEST_CASE("gemv_t.double") {
  gemv_test<double>(false, true);
}

// gemv (row-major)
TEST_CASE("c_gemv_n.float") {
  gemv_test<float>(true, false);
}

TEST_CASE("c_gemv_n.double") {
  gemv_test<double>(true, false);
}

TEST_CASE("c_gemv_t.float") {
  gemv_test<float>(true, true);
}

TEST_CASE("c_gemv_t.double") {
  gemv_test<double>(true, true);
}

// ----------------------------------------------------------------------------
// trsv
// ----------------------------------------------------------------------------

template <typename T>
void c_trsv_test() {

  int N = 3;

  const std::vector<T> hA = {
    2, 0, 0,
    1, 2, 0,
    1, 1, 2
  };

  const std::vector<T> hB = {
    5,
    4,
    7
  };

  const std::vector<T> sol = {
    2.5,
    0.75,
    1.875
  };

  std::vector<T> res(N, 0);

  tf::Taskflow taskflow;
  tf::Executor executor;

  auto dA = tf::cuda_malloc_device<T>(hA.size());
  auto dB = tf::cuda_malloc_device<T>(hB.size());

  taskflow.emplace([&](tf::cudaFlowCapturer& capturer){
    auto blas = capturer.make_capturer<tf::cublasFlowCapturer>();
    auto h2dA = blas->copy(dA, hA.data(), hA.size());
    auto h2dB = blas->copy(dB, hB.data(), hB.size());
    auto trsv = blas->c_trsv(CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
      N, dA, N, dB, 1
    );
    auto d2h = blas->copy(res.data(), dB, res.size());

    trsv.succeed(h2dA, h2dB)
        .precede(d2h);
  });

  executor.run(taskflow).wait();
  
  for(size_t i=0; i<res.size(); ++i) {
    //std::cout << res[i] << '\n';
    REQUIRE(std::fabs(res[i] - sol[i]) < 0.0001);
  }

}

TEST_CASE("c_trsv.float") {
  c_trsv_test<float>();
}

TEST_CASE("c_trsv.double") {
  c_trsv_test<double>();
}





