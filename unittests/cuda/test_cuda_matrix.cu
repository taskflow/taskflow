#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>

template <typename T>
void run_and_wait(T& cf) {
  tf::cudaStream stream;
  cf.run(stream);
  stream.synchronize();
}

// ----------------------------------------------------------------------------
// Matrix Multiplication Kernel
// ----------------------------------------------------------------------------
__global__ void k_multiplication(
  int *a, int *b, int *c, int m, int n, int k
) {
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

TEST_CASE("multiply" * doctest::timeout(300)) {

  tf::Taskflow taskflow;
  tf::Executor executor;
  
  std::vector<int> a, b, c;

  const unsigned B = 16;

  for(int m=1; m<=256; m<<=1) {
    for(int n=1; n<=256; n<<=1) {
      for(int k=1; k<=256; k<<=1) {

        taskflow.clear();

        int* ha {nullptr};
        int* hb {nullptr};
        int* hc {nullptr};
        int* da {nullptr};
        int* db {nullptr};
        int* dc {nullptr};

        dim3 grid  ((k+B-1)/B, (m+B-1)/B);
        dim3 block (B, B);

        auto hosta = taskflow.emplace([&](){ 
          a.resize(m*n);
          std::fill_n(a.begin(), m*n, m+n);
          ha = a.data();
          REQUIRE(cudaMalloc(&da, m*n*sizeof(int)) == cudaSuccess);
        }).name("ha");

        auto hostb = taskflow.emplace([&](){ 
          b.resize(n*k);
          std::fill_n(b.begin(), n*k, n+k);
          hb = b.data();
          REQUIRE(cudaMalloc(&db, n*k*sizeof(int)) == cudaSuccess);
        }).name("hb");

        auto hostc = taskflow.emplace([&](){
          c.resize(m*k);
          hc = c.data();
          REQUIRE(cudaMalloc(&dc, m*k*sizeof(int)) == cudaSuccess);
        }).name("hc");

        auto cuda = taskflow.emplace([&](){
          tf::cudaFlow cf;
          auto pa = cf.copy(da, ha, m*n);
          auto pb = cf.copy(db, hb, n*k);

          auto op = cf.kernel(
            grid, block, 0, k_multiplication, da, db, dc, m, n, k
          ).name("op");

          auto cc = cf.copy(hc, dc, m*k)
                      .name("cc");

          op.precede(cc).succeed(pa, pb);
          run_and_wait(cf);
        });

        cuda.succeed(hosta, hostb, hostc);

        executor.run(taskflow).wait();

        for(const auto& x : c) {
          REQUIRE(x == (int)(m+n)*(n+k)*n);
        }

        REQUIRE(cudaFree(da) == cudaSuccess);
        REQUIRE(cudaFree(db) == cudaSuccess);
        REQUIRE(cudaFree(dc) == cudaSuccess);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Matrix Transpose
// ----------------------------------------------------------------------------
__global__ void k_transpose(int *mat_in, int *mat_out, int rows, int cols) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < cols && idy < rows) {
    unsigned int pos = idy * cols + idx;
    unsigned int trans_pos = idx * rows + idy;
    mat_out[trans_pos] = mat_in[pos];
  }
}

TEST_CASE("transpose" * doctest::timeout(300)) {
  
  std::vector<int> in, out;
  
  tf::Taskflow taskflow;
  tf::Executor executor;

  const unsigned B = 16;

  for(int m=1; m<=256; m<<=1) {
    for(int n=1; n<=256; n<<=1) {

      taskflow.clear();

      int* ptr_in {nullptr};
      int* ptr_out {nullptr};
      int* sin {nullptr};
      int* sout {nullptr};

      dim3 grid  ((n+B-1)/B, (m+B-1)/B);
      dim3 block (B, B);

      auto hin = taskflow.emplace([&](){ 
        in.resize(m*n);
        out.resize(m*n);
        for(auto& item : in) {
          item = ::rand()%100;
        }
        ptr_in = in.data();
        ptr_out = out.data();
        REQUIRE(cudaMalloc(&sin, m*n*sizeof(int)) == cudaSuccess);
        REQUIRE(cudaMalloc(&sout, m*n*sizeof(int)) == cudaSuccess);
      }).name("ha");

      auto op = taskflow.emplace([&](){
        tf::cudaFlow cf;
        auto copyin = cf.copy(sin, ptr_in, m*n);
        auto copyout = cf.copy(ptr_out, sout, m*n);
        auto trans = cf.kernel(grid, block, 0, k_transpose, sin, sout, m, n);
        trans.succeed(copyin).precede(copyout);
        run_and_wait(cf);
      });

      hin.precede(op);

      executor.run(taskflow).wait();

      for(int x=0; x<m; x++) {
        for(int y=0; y<n; ++y) {
          REQUIRE(in[x*n+y] == out[y*m+x]);
        }
      }

      REQUIRE(cudaFree(sin) == cudaSuccess);
      REQUIRE(cudaFree(sout) == cudaSuccess);
    }
  }
}

// ----------------------------------------------------------------------------
// vector product
// ----------------------------------------------------------------------------
__global__ void k_product(int *a, int *b, int *c, int N) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    c[idx] = a[idx] * b[idx];
  }
}

TEST_CASE("product" * doctest::timeout(300)) {

  tf::Taskflow taskflow;
  tf::Executor executor;

  const unsigned num_batches = 1024;
  const unsigned N = 1024;
  const unsigned B = 128;

  dim3 grid  ((N+B-1)/B);
  dim3 block (B);

  std::vector<int*> hA(num_batches);
  std::vector<int*> hB(num_batches);
  std::vector<int*> hC(num_batches);
  std::vector<int*> dA(num_batches);
  std::vector<int*> dB(num_batches);
  std::vector<int*> dC(num_batches);

  for(unsigned i=0; i<num_batches; ++i) {

    int v1 = ::rand()%10;
    int v2 = ::rand()%10;

    auto allocate = taskflow.emplace([&, i, v1, v2](){
      hA[i] = new int [N];
      hB[i] = new int [N];
      hC[i] = new int [N];
      REQUIRE(cudaMalloc(&dA[i], N*sizeof(int)) == cudaSuccess);
      REQUIRE(cudaMalloc(&dB[i], N*sizeof(int)) == cudaSuccess);
      REQUIRE(cudaMalloc(&dC[i], N*sizeof(int)) == cudaSuccess);
      for(unsigned j=0; j<N; ++j) {
        hA[i][j] = v1;
        hB[i][j] = v2;
      }
    });

    auto kernel = taskflow.emplace([&, i](){
      tf::cudaFlow cf;
      auto copyA = cf.copy(dA[i], hA[i], N);
      auto copyB = cf.copy(dB[i], hB[i], N);
      auto op = cf.kernel(grid, block, 0, k_product, dA[i], dB[i], dC[i], N);
      auto copyC = cf.copy(hC[i], dC[i], N);
      op.succeed(copyA, copyB).precede(copyC);
      run_and_wait(cf);
    });

    auto deallocate = taskflow.emplace([&, i, v1, v2](){
      for(unsigned j=0; j<N; ++j) {
        REQUIRE(hC[i][j] == v1*v2);
      }
      delete hA[i];
      delete hB[i];
      delete hC[i];
      REQUIRE(cudaFree(dA[i]) == cudaSuccess);
      REQUIRE(cudaFree(dB[i]) == cudaSuccess);
      REQUIRE(cudaFree(dC[i]) == cudaSuccess);
    });

    kernel.precede(deallocate).succeed(allocate);
  }

  executor.run(taskflow).wait();

}









