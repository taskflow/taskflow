// The example shows how to use cudaFlow to multiply two 2D matrices.

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/cuda/cudaflow.hpp>

// Kernel: matmul
__global__ void matmul(int *a, int *b, int *c, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  if(col < k && row < m) {
    for(int i = 0; i < n; i++) {
      sum += a[row * n + i] * b[i * k + col];
    }
    c[row * k + col] = sum;
  }
}

// Matrix multiplication using GPU
auto gpu(int M, int N, int K) {
  
  std::vector<int> ha, hb, hc;
  int *da, *db, *dc;

  tf::Taskflow taskflow("MatrixMultiplication");
  tf::Executor executor;

  // allocate the host and device storage for a
  auto allocate_a = taskflow.emplace([&](){
    ha.resize(M*N, M+N);
    TF_CHECK_CUDA(cudaMalloc(&da, M*N*sizeof(int)), "failed to allocate a");
  }).name("allocate_a");
  
  // allocate the host and device storage for b
  auto allocate_b = taskflow.emplace([&](){
    hb.resize(N*K, N+K);
    TF_CHECK_CUDA(cudaMalloc(&db, N*K*sizeof(int)), "failed to allocate b");
  }).name("allocate_b");
  
  // allocate the host and device storage for c
  auto allocate_c = taskflow.emplace([&](){
    hc.resize(M*K);
    TF_CHECK_CUDA(cudaMalloc(&dc, M*K*sizeof(int)), "failed to allocate c");
  }).name("allocate_c");
  
  // create a cudaFlow to run the matrix multiplication
  auto cudaFlow = taskflow.emplace([&](){

    tf::cudaFlow cf;

    // copy data to da, db, and dc
    auto copy_da = cf.copy(da, ha.data(), M*N).name("H2D_a");
    auto copy_db = cf.copy(db, hb.data(), N*K).name("H2D_b");
    auto copy_hc = cf.copy(hc.data(), dc, M*K).name("D2H_c"); 
    
    dim3 grid  ((K+16-1)/16, (M+16-1)/16);
    dim3 block (16, 16);

    auto kmatmul = cf.kernel(grid, block, 0, matmul, da, db, dc, M, N, K)
                     .name("matmul");

    kmatmul.succeed(copy_da, copy_db)
           .precede(copy_hc);
    
    tf::cudaStream stream;
    cf.run(stream);
    stream.synchronize(); 

  }).name("cudaFlow");

  auto free = taskflow.emplace([&](){
    TF_CHECK_CUDA(cudaFree(da), "failed to free da");  
    TF_CHECK_CUDA(cudaFree(db), "failed to free db");  
    TF_CHECK_CUDA(cudaFree(dc), "failed to free dc");  
  }).name("free");

  cudaFlow.succeed(allocate_a, allocate_b, allocate_c)
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









