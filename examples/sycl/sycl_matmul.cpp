// The example shows how to use syclFlow to multiply two 2D matrices.

#include <taskflow/taskflow.hpp>
#include <taskflow/sycl/syclflow.hpp>

// Matrix multiplication using GPU
auto gpu(int M, int N, int K) {

  std::vector<int> ha, hb, hc;
  int *da, *db, *dc;

  tf::Executor executor;
  tf::Taskflow taskflow("MatrixMultiplication");

  sycl::queue queue;

  // allocate the host and device storage for a
  auto allocate_a = taskflow.emplace([&](){
    ha.resize(M*N, M+N);
    da = sycl::malloc_device<int>(M*N, queue);
  }).name("allocate_a");

  // allocate the host and device storage for b
  auto allocate_b = taskflow.emplace([&](){
    hb.resize(N*K, N+K);
    db = sycl::malloc_device<int>(N*K, queue);
  }).name("allocate_b");

  // allocate the host and device storage for c
  auto allocate_c = taskflow.emplace([&](){
    hc.resize(M*K);
    dc = sycl::malloc_device<int>(M*K, queue);
  }).name("allocate_c");

  // create a syclFlow to run the matrix multiplication
  auto syclFlow = taskflow.emplace_on([&](tf::syclFlow& sf){

    // copy data to da, db, and dc
    auto copy_da = sf.copy(da, ha.data(), M*N).name("H2D_a");
    auto copy_db = sf.copy(db, hb.data(), N*K).name("H2D_b");
    auto copy_hc = sf.copy(hc.data(), dc, M*K).name("D2H_c");

    auto _M = (M % 16 == 0) ? M : (M + 16 - M % 16);
    auto _K = (K % 16 == 0) ? K : (K + 16 - K % 16);

    auto kmatmul = sf.parallel_for(
      sycl::nd_range<2>{sycl::range<2>(_M, _K ), sycl::range<2>(16, 16)},
      [=](sycl::nd_item<2> item) {
        int row = item.get_global_id(0);
        int col = item.get_global_id(1);
        if(row < M && col < K) {
          int sum = 0;
          for(int n = 0; n < N; n++) {
            sum += da[row * N + n] * db[n * K + col];
          }
          dc[row * K + col] = sum;
        }
      }
    ).name("matmul");

    // It is also possible to just use range and let the runtime decide the
    // partition of groups, but the result is less efficient.
    //
    //auto kmatmul = sf.parallel_for(
    //  sycl::range<2>(M, K),
    //  [=](sycl::id<2> id) {
    //    int row = id[0];
    //    int col = id[1];
    //    int sum = 0;
    //    for(int n = 0; n < N; n++) {
    //      sum += da[row * N + n] * db[n * K + col];
    //    }
    //    dc[row * K + col] = sum;
    //  }
    //).name("matmul");

    kmatmul.succeed(copy_da, copy_db)
           .precede(copy_hc);

  }, queue).name("syclFlow");

  auto free = taskflow.emplace([&](){
    sycl::free(da, queue);
    sycl::free(db, queue);
    sycl::free(dc, queue);
  }).name("free");

  syclFlow.succeed(allocate_a, allocate_b, allocate_c)
          .precede(free);

  executor.run(taskflow).wait();

  // You may uncomment the line below to dump the task graph
  //taskflow.dump(std::cout);

  return hc;
}

// Matrix multiplication using CPU
auto cpu(int M, int N, int K) {

  std::vector<int> a, b, c;

  tf::Executor executor;
  tf::Taskflow taskflow;

  auto ha = taskflow.emplace([&](){
    a.resize(M*N, M+N);
  }).name("allocate_a");

  auto hb = taskflow.emplace([&](){
    b.resize(N*K, N+K);
  }).name("allocate_b");

  auto hc = taskflow.emplace([&](){
    c.resize(M*K, 0);
  }).name("allocate_c");

  auto pf = taskflow.for_each_index(0, M, 1, [&] (int m) {
    for(int k=0; k<K; k++) {
      for(int n=0; n<N; n++) {
        c[m*K+k] += (a[m*N+n]*b[n*K+k]);
      }
    }
  });

  pf.succeed(ha, hb, hc);

  //taskflow.dump(std::cout);

  executor.run(taskflow).wait();

  return c;
}

// Function: main
int main(int argc, char *argv[]) {

  if(argc != 4) {
    std::cerr << "usage: matrix-multiplication M N K\n";
    std::exit(EXIT_FAILURE);
  }

  int M = std::atoi(argv[1]);
  int N = std::atoi(argv[2]);
  int K = std::atoi(argv[3]);

  std::cout << "matrix A: " << M << 'x' << N << '\n'
            << "matrix B: " << N << 'x' << K << '\n'
            << "matrix C: " << M << 'x' << K << '\n';

  // matrix multiplication using gpu
  std::cout << "running gpu matrix multiplication ... ";
  auto gbeg = std::chrono::steady_clock::now();
  auto gres = gpu(M, N, K);
  auto gend = std::chrono::steady_clock::now();
  std::cout << "completed with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(gend-gbeg).count()
            << " ms\n";

  // matrix multiplication using cpu
  std::cout << "running cpu matrix multiplication ... ";
  auto cbeg = std::chrono::steady_clock::now();
  auto cres = cpu(M, N, K);
  auto cend = std::chrono::steady_clock::now();
  std::cout << "completed with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(cend-cbeg).count()
            << " ms\n";

  // verify the result
  int64_t error = 0;
  std::cout << "verifying results ... ";
  for(int i=0; i<M*K; ++i) {
    error += abs(gres[i] - cres[i]);
  }
  std::cout << "abs-error=" << error << '\n';

  return 0;
}









