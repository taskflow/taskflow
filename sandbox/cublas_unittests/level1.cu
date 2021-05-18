#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cublasflow.hpp>

// ----------------------------------------------------------------------------
// amax, amin, and asum
// ----------------------------------------------------------------------------

template <typename T>
void amax_amin_asum() {

  int N = 11111;
  T min_v = 100000, max_v = -1;
  T sum = 0, h_sum = -1;

  std::vector<T> host(N);

  for(int i=0; i<N; i++) {
    host[i] = rand() % 100 - 50;
    min_v = std::min(min_v, std::abs(host[i]));
    max_v = std::max(max_v, std::abs(host[i]));
    sum += std::abs(host[i]);
  }

  auto gpu = tf::cuda_malloc_device<T>(N);
  auto min_i = tf::cuda_malloc_device<int>(1);
  auto max_i = tf::cuda_malloc_device<int>(1);
  auto gsum = tf::cuda_malloc_device<T>(1);
  int h_min_i = -1, h_max_i = -1;

  tf::Taskflow taskflow;
  tf::Executor executor;

  taskflow.emplace([&](tf::cudaFlow& cf){
    auto cublas = cf.capture([&](tf::cudaFlowCapturer& cap){
      auto capturer = cap.make_capturer<tf::cublasFlowCapturer>();
      auto amax = capturer->amax(N, gpu, 1, max_i);
      auto amin = capturer->amin(N, gpu, 1, min_i);
      auto vset = capturer->vset(N, host.data(), 1, gpu, 1);
      auto back = cap.single_task([min_i, max_i] __device__ () {
        (*min_i)--;
        (*max_i)--;
      });
      auto asum = capturer->asum(N, gpu, 1, gsum);
      vset.precede(amin, amax, asum);
      back.succeed(amin, amax);
    });
    auto copy_min_i = cf.copy(&h_min_i, min_i, 1);
    auto copy_max_i = cf.copy(&h_max_i, max_i, 1);
    auto copy_sum   = cf.copy(&h_sum, gsum, 1);
    cublas.precede(copy_min_i, copy_max_i, copy_sum);
  });
  
  executor.run(taskflow).wait();
  
  REQUIRE(std::abs(host[h_min_i]) == min_v);
  REQUIRE(std::abs(host[h_max_i]) == max_v);
  REQUIRE(std::abs(sum-h_sum) < 0.0001);

  taskflow.clear();
  h_min_i = -1;
  h_max_i = -1;

  // pure capturer
  
  taskflow.emplace([&](tf::cudaFlowCapturer& cap){
    auto capturer = cap.make_capturer<tf::cublasFlowCapturer>();
    auto amax = capturer->amax(N, gpu, 1, max_i);
    auto amin = capturer->amin(N, gpu, 1, min_i);
    auto vset = capturer->vset(N, host.data(), 1, gpu, 1);
    auto back = cap.single_task([min_i, max_i] __device__ () {
      (*min_i)--;
      (*max_i)--;
    });
    auto asum = capturer->asum(N, gpu, 1, gsum);
    vset.precede(amin, amax, asum);
    back.succeed(amin, amax);
    auto copy_min_i = cap.copy(&h_min_i, min_i, 1);
    auto copy_max_i = cap.memcpy(&h_max_i, max_i, sizeof(h_max_i));
    auto copy_sum   = cap.copy(&h_sum, gsum, 1);
    back.precede(copy_min_i, copy_max_i, copy_sum);
  });

  executor.run(taskflow).wait();
  
  REQUIRE(std::abs(host[h_min_i]) == min_v);
  REQUIRE(std::abs(host[h_max_i]) == max_v);
  REQUIRE(std::abs(sum-h_sum) < 0.0001);

  tf::cuda_free(gpu);
  tf::cuda_free(min_i);
  tf::cuda_free(max_i);
}

TEST_CASE("amax-amin-asum.float" * doctest::timeout(300)) {
  amax_amin_asum<float>();
}

TEST_CASE("amax-amin-asum.double" * doctest::timeout(300)) {
  amax_amin_asum<double>();
}

// ----------------------------------------------------------------------------
// axpy
// ----------------------------------------------------------------------------

template <typename T>
void axpy() {

  int N = 1745;

  std::vector<T> hx(N), hy(N), golden(N), res(N);

  for(int i=0; i<N; i++) {
    hx[i] = rand() % 100 - 50;
    hy[i] = rand() % 100 - 50;
    golden[i] = 2 * hx[i] + hy[i];
    res[i] = rand();
  }

  auto dx = tf::cuda_malloc_device<T>(N);
  auto dy = tf::cuda_malloc_device<T>(N);
  auto alpha = tf::cuda_malloc_device<T>(1);

  tf::Taskflow taskflow;
  tf::Executor executor;

  taskflow.emplace([&](tf::cudaFlow& cf){
    cf.capture([&](tf::cudaFlowCapturer& cap){
      auto capturer = cap.make_capturer<tf::cublasFlowCapturer>();
      auto vsetx = capturer->vset(N, hx.data(), 1, dx, 1);
      auto vsety = capturer->vset(N, hy.data(), 1, dy, 1);
      auto spar = cap.single_task([alpha] __device__ () {
        *alpha = 2;
      });
      auto axpy = capturer->axpy(N, alpha, dx, 1, dy, 1);
      auto vgety = capturer->vget(N, dy, 1, res.data(), 1);
      axpy.succeed(vsetx, vsety, spar)
          .precede(vgety);
    });
  });

  executor.run(taskflow).wait();

  for(int i=0; i<N; i++) {
    REQUIRE(std::abs(res[i] - golden[i]) < 0.0001);
  }
  
  tf::cuda_free(dx);
  tf::cuda_free(dy);
  tf::cuda_free(alpha);
}

TEST_CASE("axpy.float" * doctest::timeout(300)) {
  axpy<float>();
}

TEST_CASE("axpy.double" * doctest::timeout(300)) {
  axpy<double>();
}

// ----------------------------------------------------------------------------
// dot
// ----------------------------------------------------------------------------

template <typename T>
void dot() {

  int N = 1745;

  T res = -1, golden = 0;
  std::vector<T> hx(N), hy(N);

  for(int i=0; i<N; i++) {
    hx[i] = rand() % 100 - 50;
    hy[i] = rand() % 100 - 50;
    golden += hx[i] * hy[i];
  }

  auto dx = tf::cuda_malloc_device<T>(N);
  auto dy = tf::cuda_malloc_device<T>(N);
  auto dr = tf::cuda_malloc_device<T>(1);

  tf::Taskflow taskflow;
  tf::Executor executor;

  taskflow.emplace([&](tf::cudaFlow& cf){
    cf.capture([&](tf::cudaFlowCapturer& cap){
      auto capturer = cap.make_capturer<tf::cublasFlowCapturer>();
      auto vsetx = capturer->vset(N, hx.data(), 1, dx, 1);
      auto vsety = capturer->vset(N, hy.data(), 1, dy, 1);
      auto xydot = capturer->dot(N, dx, 1, dy, 1, dr);
      auto copyr = cap.memcpy(&res, dr, sizeof(T));
      xydot.succeed(vsetx, vsety)
           .precede(copyr);
    });
  });

  executor.run(taskflow).wait();
  
  REQUIRE(std::abs(res-golden) < 0.0001);
  
  tf::cuda_free(dx);
  tf::cuda_free(dy);
  tf::cuda_free(dr);
}

TEST_CASE("dot.float" * doctest::timeout(300)) {
  dot<float>();
}

TEST_CASE("dot.double" * doctest::timeout(300)) {
  dot<double>();
}

// ----------------------------------------------------------------------------
// swap
// ----------------------------------------------------------------------------

template <typename T>
void swap() {

  int N = 1745;

  std::vector<T> hx(N), hy(N), rx(N), ry(N);

  for(int i=0; i<N; i++) {
    hx[i] = rand() % 100 - 50;
    hy[i] = rand() % 100 - 50;
  }

  auto dx = tf::cuda_malloc_device<T>(N);
  auto dy = tf::cuda_malloc_device<T>(N);

  tf::Taskflow taskflow;
  tf::Executor executor;

  taskflow.emplace([&](tf::cudaFlow& cf){
    cf.capture([&](tf::cudaFlowCapturer& cap){
      auto capturer = cap.make_capturer<tf::cublasFlowCapturer>();
      auto vsetx = capturer->vset(N, hx.data(), 1, dx, 1);
      auto vsety = capturer->vset(N, hy.data(), 1, dy, 1);
      auto xyswp = capturer->swap(N, dx, 1, dy, 1);
      auto copyx = cap.memcpy(rx.data(), dx, N*sizeof(T));
      auto copyy = cap.memcpy(ry.data(), dy, N*sizeof(T));
      xyswp.succeed(vsetx, vsety)
           .precede(copyx, copyy);
    });
  });

  executor.run(taskflow).wait();

  for(int i=0; i<N; i++) {
    REQUIRE(rx[i] == hy[i]);
    REQUIRE(ry[i] == hx[i]);
  }
  
  tf::cuda_free(dx);
  tf::cuda_free(dy);
}

TEST_CASE("swap.float" * doctest::timeout(300)) {
  swap<float>();
}

TEST_CASE("swap.double" * doctest::timeout(300)) {
  swap<double>();
}

// ----------------------------------------------------------------------------
// scal
// ----------------------------------------------------------------------------

template <typename T>
void scal() {

  int N = 17;

  std::vector<T> hx(N), rx(N);

  for(int i=0; i<N; i++) {
    hx[i] = rand() % 100 - 50;
    rx[i] = rand() % 100 - 50;
  }

  auto dx = tf::cuda_malloc_device<T>(N);
  auto alpha = tf::cuda_malloc_device<T>(1);

  tf::Taskflow taskflow;
  tf::Executor executor;

  taskflow.emplace([&](tf::cudaFlow& cf){
    cf.capture([&](tf::cudaFlowCapturer& cap){
      auto capturer = cap.make_capturer<tf::cublasFlowCapturer>();
      auto vsetx = capturer->vset(N, hx.data(), 1, dx, 1);
      auto spar = cap.single_task([alpha] __device__ () {
        *alpha = 2;
      });
      auto vgetx = capturer->vget(N, dx, 1, rx.data(), 1);
      auto scal = capturer->scal(N, alpha, dx, 1);
      scal.succeed(vsetx, spar)
          .precede(vgetx);
    });
  });

  executor.run(taskflow).wait();

  for(int i=0; i<N; i++) {
    REQUIRE(std::abs(rx[i] - 2.0*hx[i]) < 0.0001);
  }
  
  tf::cuda_free(dx);
  tf::cuda_free(alpha);
}

TEST_CASE("scal.float" * doctest::timeout(300)) {
  scal<float>();
}

TEST_CASE("scal.double" * doctest::timeout(300)) {
  scal<double>();
} 


