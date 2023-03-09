#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>
#include <taskflow/cuda/algorithm/transform.hpp>
#include <taskflow/cuda/algorithm/for_each.hpp>
#include <taskflow/cuda/algorithm/reduce.hpp>
#include <taskflow/cuda/algorithm/scan.hpp>
#include <taskflow/cuda/algorithm/find.hpp>
#include <taskflow/cuda/algorithm/sort.hpp>

//verify
template <typename T>
__global__
void verify(const T* a, const T* b, bool* check, size_t size) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for(;tid < size; tid += gridDim.x * blockDim.x) {
    if(a[tid] != b[tid]) {
      *check = false;
      return;
    }
  }
}

//add
template <typename T>
__global__
void add(const T* a, const T* b, T* c, size_t size) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for(;tid < size; tid += gridDim.x * blockDim.x) {
    c[tid] = a[tid] + b[tid];
  }
}

//multiply
template <typename T>
__global__
void multiply(const T* a, const T* b, T* c, size_t size) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for(;tid < size; tid += gridDim.x * blockDim.x) {
    c[tid] = a[tid] * b[tid];
  }
}

//----------------------------------------------------------------------
//offload_n
//----------------------------------------------------------------------

template <typename F>
void offload_n() {

  tf::Executor executor;
  tf::Taskflow taskflow;

  for(size_t N = 1; N < 65532; N = N * 2 + 1) {
    
    taskflow.clear();

    int* a {nullptr};
    int* ans_a {nullptr};
    bool* check {nullptr};
  
    int times = ::rand() % 7;

    //allocate
    auto allocate_t = taskflow.emplace([&]() {
      REQUIRE(cudaMallocManaged(&a, N * sizeof(int)) == cudaSuccess);
      REQUIRE(cudaMallocManaged(&ans_a, N * sizeof(int)) == cudaSuccess);

      REQUIRE(cudaMallocManaged(&check, sizeof(bool)) == cudaSuccess);
    }).name("allocate");
  
    //initialize
    auto initialize_t = taskflow.emplace([&]() {
      std::generate_n(a, N, [&](){ return ::rand() % N; });
      std::memcpy(ans_a, a, N * sizeof(int));
      *check = true;
    }).name("initialize");

    //offload
    auto offload_t = taskflow.emplace([&](F& cf) {
      cf.kernel(
        32, 512, 0,
        add<int>,
        a, a, a, N
      );

      cf.offload_n(times+1);
    }).name("offload");

    //verify
    auto verify_t = taskflow.emplace([&](F& cf) {
      auto ans_t = cf.for_each(
        ans_a, ans_a + N,
        [=] __device__(int& v) { v *= std::pow(2, (times + 1)); }
      );

      auto verify_t = cf.kernel(  
        32, 512, 0,
        verify<int>,
        a, ans_a, check, N
      );

      ans_t.precede(verify_t);

      cf.offload();
      REQUIRE(*check);
    }).name("verify");

     //free memory
    auto deallocate_t = taskflow.emplace([&]() {
      REQUIRE(cudaFree(a) == cudaSuccess);
      REQUIRE(cudaFree(ans_a) == cudaSuccess);

      REQUIRE(cudaFree(check) == cudaSuccess);
    }).name("deallocate");

    allocate_t.precede(initialize_t);
    initialize_t.precede(offload_t);
    offload_t.precede(verify_t);
    verify_t.precede(deallocate_t);

    executor.run(taskflow).wait();
  }
}

TEST_CASE("cudaFlow.offload" * doctest::timeout(300)) {
  offload_n<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.offload" * doctest::timeout(300)) {
  offload_n<tf::cudaFlowCapturer>();
}


//----------------------------------------------------------------------
//join_n
//----------------------------------------------------------------------

template <typename F>
void join() {
  tf::Executor executor;
  tf::Taskflow taskflow;

  for(size_t N = 1; N < 65532; N = N * 2 + 1) {
    
    taskflow.clear();

    int* a {nullptr};
    int* ans_a {nullptr};
    bool* check {nullptr};
  
    int times = ::rand() % 7;

    //allocate
    auto allocate_t = taskflow.emplace([&]() {
      REQUIRE(cudaMallocManaged(&a, N * sizeof(int)) == cudaSuccess);
      REQUIRE(cudaMallocManaged(&ans_a, N * sizeof(int)) == cudaSuccess);

      REQUIRE(cudaMallocManaged(&check, sizeof(bool)) == cudaSuccess);
    }).name("allocate");
  
    //initialize
    auto initialize_t = taskflow.emplace([&]() {
      std::generate_n(a, N, [&](){ return ::rand() % N; });

      std::memcpy(ans_a, a, N * sizeof(int));

      *check = true;
    }).name("initialize");

    //join
    auto join_t = taskflow.emplace([&](F& cf) {
      cf.kernel(
        32, 512, 0,
        add<int>,
        a, a, a, N
      );

      cf.offload_n(times);
    }).name("join");

    //verify
    auto verify_t = taskflow.emplace([&](F& cf) {
      auto ans_t = cf.for_each(
        ans_a, ans_a + N,
        [=] __device__(int& v) { v *= std::pow(2, (times)); }
      );

      auto verify_t = cf.kernel(  
        32, 512, 0,
        verify<int>,
        a, ans_a, check, N
      );

      ans_t.precede(verify_t);

      cf.offload();
      REQUIRE(*check);
    }).name("verify");

     //free memory
    auto deallocate_t = taskflow.emplace([&]() {
      REQUIRE(cudaFree(a) == cudaSuccess);
      REQUIRE(cudaFree(ans_a) == cudaSuccess);

      REQUIRE(cudaFree(check) == cudaSuccess);
    }).name("deallocate");

    allocate_t.precede(initialize_t);
    initialize_t.precede(join_t);
    join_t.precede(verify_t);
    verify_t.precede(deallocate_t);

    executor.run(taskflow).wait();
  }
}

TEST_CASE("cudaFlow.join" * doctest::timeout(300)) {
  join<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.join" * doctest::timeout(300)) {
  join<tf::cudaFlowCapturer>();
}

//----------------------------------------------------------------------
//update kernel
//----------------------------------------------------------------------

template <typename F, typename T>
void update_kernel() {

  tf::Executor executor;
  tf::Taskflow taskflow;

  for(size_t N = 1; N < 65529; N = N * 2 + 1) {

    taskflow.clear();

    std::vector<T*> operand(3, nullptr);
    std::vector<T*> ans_operand(3, nullptr);

    std::vector<int> ind(3);
    std::generate_n(ind.data(), 3, [&](){ return ::rand() % 3; });


    bool* check {nullptr};

    //allocate
    auto allocate_t = taskflow.emplace([&]() {
      for(int i = 0; i < 3; ++i) {
        REQUIRE(cudaMallocManaged(&operand[i], N * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMallocManaged(&ans_operand[i], N * sizeof(T)) == cudaSuccess);
      }

      REQUIRE(cudaMallocManaged(&check, sizeof(bool)) == cudaSuccess);
    }).name("allocate");

    //initialize
    auto initialize_t = taskflow.emplace([&](){
      for(int i = 0; i < 3; ++i) {
        std::generate_n(operand[i], N, [&](){ return ::rand() % N - N / 2 + i; });
        std::memcpy(ans_operand[i], operand[i], N * sizeof(T));
      }
      
      *check = true;
    }).name("initialize"); 

    
    //update_kernel
    auto add_t = taskflow.emplace([&](F& cf) {
      auto multi_t = cf.kernel(
        32, 512, 0,
        multiply<T>,
        operand[ind[0]], operand[ind[1]], operand[ind[2]], N
      );

      auto add_t = cf.kernel(
        32, 512, 0,
        add<T>,
        operand[ind[1]], operand[ind[2]], operand[ind[0]], N
      );

      multi_t.precede(add_t);

      cf.offload();

      cf.kernel(
        multi_t,
        64, 128, 0, multiply<T>,
        operand[ind[2]], operand[ind[0]], operand[ind[1]], N
      );

      cf.kernel(
        add_t,
        16, 256, 0, add<T>,
        operand[ind[1]], operand[ind[0]], operand[ind[2]], N
      );

      cf.offload();

      cf.kernel(
        multi_t,
        8, 1024, 0, multiply<T>,
        operand[ind[0]], operand[ind[2]], operand[ind[1]], N
      );

      cf.kernel(
        add_t,
        64, 64, 0, add<T>,
        operand[ind[2]], operand[ind[1]], operand[ind[0]], N
      );

      cf.offload();
    }).name("add");

    //verify
    auto verify_t = taskflow.emplace([&](F& cf) {
      auto multi1_t = cf.transform(
        ans_operand[ind[0]], ans_operand[ind[0]] + N, ans_operand[ind[1]],
        ans_operand[ind[2]],
        [] __device__ (T& v1, T& v2) { return v1 * v2; }
      );

      auto add1_t = cf.transform(
        ans_operand[ind[1]], ans_operand[ind[1]]+N, ans_operand[ind[2]],
        ans_operand[ind[0]],
        [] __device__ (T& v1, T& v2) { return v1 + v2; }
      );

      auto multi2_t = cf.transform(
        ans_operand[ind[2]], ans_operand[ind[2]] + N, ans_operand[ind[0]],
        ans_operand[ind[1]],
        [] __device__ (T& v1, T& v2) { return v1 * v2; }
      );

      auto add2_t = cf.transform(
        ans_operand[ind[1]], ans_operand[ind[1]] + N, ans_operand[ind[0]],
        ans_operand[ind[2]],
        [] __device__ (T& v1, T& v2) { return v1 + v2; }
      );

      auto multi3_t = cf.transform(
        ans_operand[ind[0]], ans_operand[ind[0]] + N, ans_operand[ind[2]],
        ans_operand[ind[1]],
        [] __device__ (T& v1, T& v2) { return v1 * v2; }
      );

      auto add3_t = cf.transform(
        ans_operand[ind[2]], ans_operand[ind[2]] + N, ans_operand[ind[1]],
        ans_operand[ind[0]],
        [] __device__ (T& v1, T& v2) { return v1 + v2; }
      );
  
      auto verify1_t = cf.kernel(
        32, 512, 0,
        verify<T>,
        operand[ind[0]], ans_operand[ind[0]], check, N
      );

      auto verify2_t = cf.kernel(
        32, 512, 0,
        verify<T>,
        operand[ind[1]], ans_operand[ind[1]], check, N
      );

      auto verify3_t = cf.kernel(
        32, 512, 0,
        verify<T>,
        operand[ind[2]], ans_operand[ind[2]], check, N
      );

      multi1_t.precede(add1_t);
      add1_t.precede(multi2_t);
      multi2_t.precede(add2_t);
      add2_t.precede(multi3_t);
      multi3_t.precede(add3_t);
      add3_t.precede(verify1_t).precede(verify2_t).precede(verify3_t);

      cf.offload();
      REQUIRE(*check);

    }).name("verify");

     //free memory
    auto deallocate_t = taskflow.emplace([&]() {
      for(int i = 0; i < 3; ++i) {
      REQUIRE(cudaFree(operand[i]) == cudaSuccess);
      REQUIRE(cudaFree(ans_operand[i]) == cudaSuccess);
      }

      REQUIRE(cudaFree(check) == cudaSuccess);
    }).name("deallocate");

    allocate_t.precede(initialize_t);
    initialize_t.precede(add_t);
    add_t.precede(verify_t);
    verify_t.precede(deallocate_t);

    executor.run(taskflow).wait();

  }

}

TEST_CASE("cudaFlow.update.kernel.int" * doctest::timeout(300)) {
  update_kernel<tf::cudaFlow, int>();
}

TEST_CASE("cudaFlow.update.kernel.float" * doctest::timeout(300)) {
  update_kernel<tf::cudaFlow, float>();
}

TEST_CASE("cudaFlow.update.kernel.double" * doctest::timeout(300)) {
  update_kernel<tf::cudaFlow, double>();
}

TEST_CASE("cudaFlowCapturer.update.kernel.int" * doctest::timeout(300)) {
  update_kernel<tf::cudaFlowCapturer, int>();
}

TEST_CASE("cudaFlowCapturer.update.kernel.float" * doctest::timeout(300)) {
  update_kernel<tf::cudaFlowCapturer, float>();
}

TEST_CASE("cudaFlowCapturer.update.kernel.double" * doctest::timeout(300)) {
  update_kernel<tf::cudaFlowCapturer, double>();
}

//----------------------------------------------------------------------
// update copy
//----------------------------------------------------------------------

template <typename F, typename T>
void update_copy() {
  
  tf::Executor executor;
  tf::Taskflow taskflow;

  for(int N = 1; N < 65459; N = N * 2 + 1) {
    
    taskflow.clear();

    std::vector<T> ha(N, N + 5);
    std::vector<T> hb(N, N - 31);
    std::vector<T> hc(N, N - 47);
    std::vector<T> hz(N);

    T* da {nullptr};
    T* db {nullptr};
    T* dc {nullptr};
    T* dz {nullptr};


    //allocate
    auto allocate_t = taskflow.emplace([&]() {
      REQUIRE(cudaMalloc(&da, N * sizeof(T)) == cudaSuccess);
      REQUIRE(cudaMalloc(&db, N * sizeof(T)) == cudaSuccess);
      REQUIRE(cudaMalloc(&dc, N * sizeof(T)) == cudaSuccess);
      REQUIRE(cudaMalloc(&dz, N * sizeof(T)) == cudaSuccess);
    }).name("allocate");


    //update_copy
    auto h2d_t = taskflow.emplace([&](F& cf) {
      auto h2d_t = cf.copy(da, ha.data(), N).name("h2d");
      cf.offload();

      cf.copy(h2d_t, db, hb.data(), N);
      cf.offload();

      cf.copy(h2d_t, dc, hc.data(), N);
      cf.offload();

    });

    auto kernel_t = taskflow.emplace([&](F& cf) {
      auto add1_t = cf.transform(
        da, da+N, db, dz,
        [] __device__ (T& v1, T& v2) { return v1 + v2; }
      );

      auto add2_t = cf.transform(
        dc, dc+N, dz, dc,
        [] __device__ (T& v1, T& v2) { return v1 - v2; }
      );

      add1_t.precede(add2_t);
    });

    auto d2h_t = taskflow.emplace([&](F& cf) {
      auto d2h_t = cf.copy(hc.data(), dc, N).name("d2h");
      cf.offload();

      cf.copy(d2h_t, hz.data(), dz, N);
      cf.offload();

    });

    //verify
    auto verify_t = taskflow.emplace([&]() {
      for(auto& c: hc) {
        REQUIRE(c == -21 - N);
      }

      for(auto& z: hz) {
        REQUIRE(z == 2 * N - 26);
      }
    });

     //free memory
    auto deallocate_t = taskflow.emplace([&]() {
      REQUIRE(cudaFree(da) == cudaSuccess);
      REQUIRE(cudaFree(db) == cudaSuccess);
      REQUIRE(cudaFree(dc) == cudaSuccess);
      REQUIRE(cudaFree(dz) == cudaSuccess);
    }).name("deallocate");

    allocate_t.precede(h2d_t);
    h2d_t.precede(kernel_t);
    kernel_t.precede(d2h_t);
    d2h_t.precede(verify_t);
    verify_t.precede(deallocate_t);

    executor.run(taskflow).wait();

  }
}

TEST_CASE("cudaFlow.update.copy.int" * doctest::timeout(300)) {
  update_copy<tf::cudaFlow, int>();
}

TEST_CASE("cudaFlow.update.copy.float" * doctest::timeout(300)) {
  update_copy<tf::cudaFlow, float>();
}

TEST_CASE("cudaFlow.update.copy.double" * doctest::timeout(300)) {
  update_copy<tf::cudaFlow, double>();
}

TEST_CASE("cudaFlowCapturer.update.copy.int" * doctest::timeout(300)) {
  update_copy<tf::cudaFlowCapturer, int>();
}

TEST_CASE("cudaFlowCapturer.update.copy.float" * doctest::timeout(300)) {
  update_copy<tf::cudaFlowCapturer, float>();
}

TEST_CASE("cudaFlowCapturer.update.copy.double" * doctest::timeout(300)) {
  update_copy<tf::cudaFlowCapturer, double>();
}

//----------------------------------------------------------------------
//update memcpy
//----------------------------------------------------------------------

template <typename F, typename T>
void update_memcpy() {
  
  tf::Executor executor;
  tf::Taskflow taskflow;

  for(int N = 1; N < 65459; N = N * 2 + 1) {
    
    taskflow.clear();

    std::vector<T> ha(N, N + 5);
    std::vector<T> hb(N, N - 31);
    std::vector<T> hc(N, N - 47);
    std::vector<T> hz(N);

    T* da {nullptr};
    T* db {nullptr};
    T* dc {nullptr};
    T* dz {nullptr};


    //allocate
    auto allocate_t = taskflow.emplace([&]() {
      REQUIRE(cudaMalloc(&da, N * sizeof(T)) == cudaSuccess);
      REQUIRE(cudaMalloc(&db, N * sizeof(T)) == cudaSuccess);
      REQUIRE(cudaMalloc(&dc, N * sizeof(T)) == cudaSuccess);
      REQUIRE(cudaMalloc(&dz, N * sizeof(T)) == cudaSuccess);
    }).name("allocate");


    //update_memcpy
    auto h2d_t = taskflow.emplace([&](F& cf) {
      auto h2d_t = cf.memcpy(da, ha.data(), sizeof(T) * N).name("h2d");
      cf.offload();

      cf.memcpy(h2d_t, db, hb.data(), sizeof(T) * N);
      cf.offload();

      cf.memcpy(h2d_t, dc, hc.data(), sizeof(T) * N);
      cf.offload();

    });

    auto kernel_t = taskflow.emplace([&](F& cf) {
      auto add1_t = cf.transform(
        da, da+N, db, dz,
        [] __device__ (T& v1, T& v2) { return v1 + v2; }
      );

      auto add2_t = cf.transform(
        dc, dc+N, dz, dc,
        [] __device__ (T& v1, T& v2) { return v1 - v2; }
      );

      add1_t.precede(add2_t);
    });

    auto d2h_t = taskflow.emplace([&](F& cf) {
      auto d2h_t = cf.memcpy(hc.data(), dc, sizeof(T) * N).name("d2h");
      cf.offload();

      cf.memcpy(d2h_t, hz.data(), dz, sizeof(T) * N);
      cf.offload();

    });

    //verify
    auto verify_t = taskflow.emplace([&]() {
      for(auto& c: hc) {
        REQUIRE(c == -21 - N);
      }

      for(auto& z: hz) {
        REQUIRE(z == 2 * N - 26);
      }
    });

     //free memory
    auto deallocate_t = taskflow.emplace([&]() {
      REQUIRE(cudaFree(da) == cudaSuccess);
      REQUIRE(cudaFree(db) == cudaSuccess);
      REQUIRE(cudaFree(dc) == cudaSuccess);
      REQUIRE(cudaFree(dz) == cudaSuccess);
    }).name("deallocate");

    allocate_t.precede(h2d_t);
    h2d_t.precede(kernel_t);
    kernel_t.precede(d2h_t);
    d2h_t.precede(verify_t);
    verify_t.precede(deallocate_t);

    executor.run(taskflow).wait();

  }
}

TEST_CASE("cudaFlow.update.memcpy.int" * doctest::timeout(300)) {
  update_memcpy<tf::cudaFlow, int>();
}

TEST_CASE("cudaFlow.update.memcpy.float" * doctest::timeout(300)) {
  update_memcpy<tf::cudaFlow, float>();
}

TEST_CASE("cudaFlow.update.memcpy.double" * doctest::timeout(300)) {
  update_memcpy<tf::cudaFlow, double>();
}

TEST_CASE("cudaFlowCapturer.update.memcpy.int" * doctest::timeout(300)) {
  update_memcpy<tf::cudaFlowCapturer, int>();
}

TEST_CASE("cudaFlowCapturer.update.memcpy.float" * doctest::timeout(300)) {
  update_memcpy<tf::cudaFlowCapturer, float>();
}

TEST_CASE("cudaFlowCapturer.update.memcpy.double" * doctest::timeout(300)) {
  update_memcpy<tf::cudaFlowCapturer, double>();
}

//----------------------------------------------------------------------
//update memset
//----------------------------------------------------------------------

template <typename F, typename T>
void update_memset() {

  tf::Executor executor;
  tf::Taskflow taskflow;

  for(size_t N = 1; N < 65199; N = N * 2 + 1) {
    
    taskflow.clear();

    T* a {nullptr};
    T* b {nullptr};

    T* ans_a {nullptr};
    T* ans_b {nullptr};
    
    bool* check {nullptr};

    //allocate
    auto allocate_t = taskflow.emplace([&]() {
      REQUIRE(cudaMallocManaged(&a, N * sizeof(T)) == cudaSuccess);
      REQUIRE(cudaMallocManaged(&b, (N + 37) * sizeof(T)) == cudaSuccess);

      REQUIRE(cudaMallocManaged(&ans_a, N * sizeof(T)) == cudaSuccess);
      REQUIRE(cudaMallocManaged(&ans_b, (N + 37) * sizeof(T)) == cudaSuccess);

      REQUIRE(cudaMallocManaged(&check, sizeof(bool)) == cudaSuccess);
    }).name("allocate");

    //initialize
    auto initialize_t = taskflow.emplace([&]() {
      std::generate_n(a, N, [&](){ return ::rand() % N - N / 2; });
      std::generate_n(b, N + 37, [&](){ return ::rand() % N + N / 2; });
      
      REQUIRE(cudaMemset(ans_a, 0, N * sizeof(T)) == cudaSuccess);
      REQUIRE(cudaMemset(ans_b, 1, (N + 37) * sizeof(T)) == cudaSuccess);

      *check = true;
    }).name("initialize"); 

    //update_memset
    auto memset_t = taskflow.emplace([&](F& cf) {
      auto memset_t = cf.memset(ans_a, 0, N * sizeof(T));
      cf.offload();

      cf.memset(memset_t, a, 0, N * sizeof(T));
      cf.offload();

      cf.memset(memset_t, b, 1, (N + 37) * sizeof(T));
      cf.offload();
    }).name("memset");

    //verify
    auto verify_t = taskflow.emplace([&](F& cf) {
      cf.kernel(
        32, 512, 0,
        verify<T>,
        a, ans_a, check, N
      );

      cf.kernel(
        32, 512, 0,
        verify<T>,
        b, ans_b, check, N + 37
      );

      cf.offload();
      REQUIRE(*check);
    }).name("verify");

    //free memory
    auto deallocate_t = taskflow.emplace([&]() {
      REQUIRE(cudaFree(a) == cudaSuccess);
      REQUIRE(cudaFree(b) == cudaSuccess);
      REQUIRE(cudaFree(ans_a) == cudaSuccess);
      REQUIRE(cudaFree(ans_b) == cudaSuccess);
      REQUIRE(cudaFree(check) == cudaSuccess);
    }).name("deallocate");

    allocate_t.precede(initialize_t);
    initialize_t.precede(memset_t);
    memset_t.precede(verify_t);
    verify_t.precede(deallocate_t);

    executor.run(taskflow).wait();
  }
}

TEST_CASE("cudaFlow.update.memset.int" * doctest::timeout(300)) {
  update_memset<tf::cudaFlow, int>();
}

TEST_CASE("cudaFlow.update.memset.float" * doctest::timeout(300)) {
  update_memset<tf::cudaFlow, float>();
}

TEST_CASE("cudaFlow.update.memset.double" * doctest::timeout(300)) {
  update_memset<tf::cudaFlow, double>();
}

TEST_CASE("cudaFlowCapturer.update.memset.int" * doctest::timeout(300)) {
  update_memset<tf::cudaFlowCapturer, int>();
}

TEST_CASE("cudaFlowCapturer.update.memset.float" * doctest::timeout(300)) {
  update_memset<tf::cudaFlowCapturer, float>();
}

TEST_CASE("cudaFlowCapturer.update.memset.double" * doctest::timeout(300)) {
  update_memset<tf::cudaFlowCapturer, double>();
}

// ----------------------------------------------------------------------------
// update algorithms
// ----------------------------------------------------------------------------

struct SetValue {
  __device__ void operator()(int& a) const { a = v; }
  int v;
};

struct SetValueOnIndex {
  __device__ void operator()(int i) const { data[i] = v; }
  int* data;
  int v;
};

struct AddOrMultiply {
  __device__ int operator()(int a, int b) const { return v ? a + b : a *b; }
  bool v;
};

struct AddScalar {
  __device__ int operator()(int a) const { return a + v; }
  int v;
};

struct MultiplyScalar {
  __device__ int operator()(int a) const { return a*v; }
  int v;
};

struct LessOrGreater {
  __device__ int operator()(int a, int b) const { return v ? a < b : a > b; }
  bool v;
};

struct IsEqual {
  int v;
  __device__ bool operator()(int a) const { return v == a; }
};

// ----------------------------------------------------------------------------
// update for_each
// ----------------------------------------------------------------------------

template <typename F>
void update_for_each() {

  F cf;

  for(int N=1; N<=100000; N += (N/10+1)) {

    cf.clear();

    auto data = tf::cuda_malloc_shared<int>(N);
    
    // for each task
    //auto task = cf.for_each(data, data+N, [] __device__ (int& a){ a = 100; });
    auto task = cf.for_each(data, data+N, SetValue{100});
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    for(int i=0; i<N; i++) {
      REQUIRE(data[i] == 100);
    }
    
    // update for each parameters
    //cf.for_each(task, data, data+N, [] __device__ (int& a){ a = 1234; });
    cf.for_each(task, data, data+N, SetValue{1234});
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    for(int i=0; i<N; i++) {
      REQUIRE(data[i] == 1234);
    }

    tf::cuda_free(data);
  }
}

TEST_CASE("cudaFlow.update.for_each" * doctest::timeout(300)) {
  update_for_each<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.update.for_each" * doctest::timeout(300)) {
  update_for_each<tf::cudaFlowCapturer>();
}

// ----------------------------------------------------------------------------
// update for_each_index
// ----------------------------------------------------------------------------

template <typename F>
void update_for_each_index() {
  F cf;

  for(int N=1; N<=100000; N += (N/10+1)) {

    cf.clear();

    auto data = tf::cuda_malloc_shared<int>(N);
    
    // for each index
    auto task = cf.for_each_index(0, N, 1, SetValueOnIndex{data, 100});
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    for(int i=0; i<N; i++) {
      REQUIRE(data[i] == 100);
    }

    // update for each index
    cf.for_each_index(task, 0, N, 1, SetValueOnIndex{data, -100});
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    for(int i=0; i<N; i++) {
      REQUIRE(data[i] == -100);
    }

    tf::cuda_free(data);
  }
}

TEST_CASE("cudaFlow.update.for_each_index" * doctest::timeout(300)) {
  update_for_each_index<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.update.for_each_index" * doctest::timeout(300)) {
  update_for_each_index<tf::cudaFlowCapturer>();
}

// ----------------------------------------------------------------------------
// update reduce
// ----------------------------------------------------------------------------

template <typename F>
void update_reduce() {

  F cf;

  for(int N=1; N<=100000; N += (N/10+1)) {

    cf.clear();

    auto data = tf::cuda_malloc_shared<int>(N);
    auto soln = tf::cuda_malloc_shared<int>(1);

    for(int i=0; i<N; i++) data[i] = -1;
    *soln = 0;
    
    // reduce
    //auto task = cf.reduce(
    //  data, data + N, soln, [] __device__ (int a, int b){ return a + b; }
    //);
    auto task = cf.reduce(data, data + N, soln, AddOrMultiply{true});
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(*soln == -N);
    
    // update reduce range
    *soln = -1;
    //cf.reduce(
    //  task, data, data + N, soln, [] __device__ (int a, int b){ return a * b; }
    //);
    cf.reduce(task, data, data + N, soln, AddOrMultiply{false});
    cf.offload();
    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(*soln == ((N&1) ? 1 : -1));

    tf::cuda_free(data);
    tf::cuda_free(soln);
  }
}

TEST_CASE("cudaFlow.update.reduce" * doctest::timeout(300)) {
  update_reduce<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.update.reduce" * doctest::timeout(300)) {
  update_reduce<tf::cudaFlowCapturer>();
}

// ----------------------------------------------------------------------------
// update uninitialized reduce
// ----------------------------------------------------------------------------

template <typename F>
void update_uninitialized_reduce() {
  F cf;

  for(int N=1; N<=100000; N += (N/10+1)) {
  
    cf.clear();

    auto data = tf::cuda_malloc_shared<int>(N);
    auto soln = tf::cuda_malloc_shared<int>(1);

    for(int i=0; i<N; i++) data[i] = -1;
    
    // uninitialized_reduce
    //auto task = cf.uninitialized_reduce(
    //  data, data + N, soln, [] __device__ (int a, int b){ return a + b; }
    //);
    auto task = cf.uninitialized_reduce(data, data + N, soln, AddOrMultiply{true});
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(*soln == -N);
    
    // update reduce range
    //cf.uninitialized_reduce(
    //  task, data, data + N, soln, [] __device__ (int a, int b){ return a * b; }
    //);
    cf.uninitialized_reduce(
      task, data, data + N, soln, AddOrMultiply{false}
    );
    cf.offload();
    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(*soln == ((N&1) ? -1 : 1));

    tf::cuda_free(data);
    tf::cuda_free(soln);
  }
}

TEST_CASE("cudaFlow.update.uninitialized_reduce" * doctest::timeout(300)) {
  update_uninitialized_reduce<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.update.uninitialized_reduce" * doctest::timeout(300)) {
  update_uninitialized_reduce<tf::cudaFlowCapturer>();
}


// ----------------------------------------------------------------------------
// update transform reduce
// ----------------------------------------------------------------------------

template <typename F>
void update_transform_reduce() {

  F cf;

  for(int N=1; N<=100000; N += (N/10+1)) {

    cf.clear();

    auto data = tf::cuda_malloc_shared<int>(N);
    auto soln = tf::cuda_malloc_shared<int>(1);

    for(int i=0; i<N; i++) data[i] = -1;
    *soln = 0;
    
    // transform_reduce
    auto task = cf.transform_reduce(
      data, data + N, soln, AddOrMultiply{true}, AddScalar{2}
      //[] __device__ (int a, int b) { return a + b; },
      //[] __device__ (int a) { return a + 2; }
    );
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(*soln == N);
    
    // update reduce range
    *soln = 8;
    cf.transform_reduce(
      task, data, data + N, soln, AddOrMultiply{false}, AddScalar{1}
      //[] __device__ (int a, int b){ return a * b; },
      //[] __device__ (int a) { return a + 1; }
    );
    cf.offload();
    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(*soln == 0);

    tf::cuda_free(data);
    tf::cuda_free(soln);
  }
}

TEST_CASE("cudaFlow.update.transform_reduce" * doctest::timeout(300)) {
  update_transform_reduce<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.update.transform_reduce" * doctest::timeout(300)) {
  update_transform_reduce<tf::cudaFlowCapturer>();
}

// ----------------------------------------------------------------------------
// update transform uninitialized reduce
// ----------------------------------------------------------------------------

template <typename F>
void update_transform_uninitialized_reduce() {

  F cf;

  for(int N=1; N<=100000; N += (N/10+1)) {

    cf.clear();

    auto data = tf::cuda_malloc_shared<int>(N);
    auto soln = tf::cuda_malloc_shared<int>(1);

    for(int i=0; i<N; i++) data[i] = -1;
    *soln = 100;
    
    // transform_reduce
    auto task = cf.transform_uninitialized_reduce(
      data, data + N, soln, AddOrMultiply{true}, AddScalar{2}
      //[] __device__ (int a, int b) { return a + b; },
      //[] __device__ (int a) { return a + 2; }
    );
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(*soln == N);
    
    // update reduce range
    *soln = 8;
    cf.transform_uninitialized_reduce(
      task, data, data + N, soln, AddOrMultiply{false}, AddScalar{0}
      //[] __device__ (int a, int b){ return a * b; },
      //[] __device__ (int a) { return a; }
    );
    cf.offload();
    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(*soln == ((N&1) ? -1 : 1));

    tf::cuda_free(data);
    tf::cuda_free(soln);
  }
}

TEST_CASE("cudaFlow.update.transform_uninitialized_reduce" * doctest::timeout(300)) {
  update_transform_uninitialized_reduce<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.update.transform_uninitialized_reduce" * doctest::timeout(300)) {
  update_transform_uninitialized_reduce<tf::cudaFlowCapturer>();
}


// ----------------------------------------------------------------------------
// transform
// ----------------------------------------------------------------------------
template <typename F>
void update_transform() {

  F cf;

  for(int N=1; N<=100000; N += (N/10+1)) {
  
    cf.clear();

    auto input1 = tf::cuda_malloc_shared<int>(N);
    auto input2 = tf::cuda_malloc_shared<int>(N);
    auto output = tf::cuda_malloc_shared<int>(N);

    for(int i=0; i<N; i++) {
      input1[i] =  i;
      input2[i] = -i;
      output[i] =  7;
    }
    
    // transform
    auto task = cf.transform(
      input1, input1+N, output, MultiplyScalar{2}
      //[] __device__ (int& a) { return 2*a; }
    );
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    for(int i=0; i<N; i++) {
      REQUIRE(output[i] == input1[i]*2);
    }

    // update transform parameters
    cf.transform(task,
      input2, input2+N, output, MultiplyScalar{10}
      // [] __device__ (int& a) { return 10*a; }
    );
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    for(int i=0; i<N; i++) {
      REQUIRE(output[i] == input2[i]*10);
    }

    tf::cuda_free(input1);
    tf::cuda_free(input2);
    tf::cuda_free(output);
  }
}

TEST_CASE("cudaFlow.update.transform" * doctest::timeout(300)) {
  update_transform<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.update.transform" * doctest::timeout(300)) {
  update_transform<tf::cudaFlowCapturer>();
}

// ----------------------------------------------------------------------------
// binary transform
// ----------------------------------------------------------------------------

// update binary_transform
template <typename F>
void update_binary_transform() {

  F cf;

  for(int N=1; N<=100000; N += (N/10+1)) {
  
    cf.clear();

    auto input1 = tf::cuda_malloc_shared<int>(N);
    auto input2 = tf::cuda_malloc_shared<int>(N);
    auto output = tf::cuda_malloc_shared<int>(N);

    for(int i=0; i<N; i++) {
      input1[i] =  i;
      input2[i] = -i;
      output[i] =  7;
    }
    
    // transform
    auto task = cf.transform(
      input1, input1+N, input2, output, AddOrMultiply{false}
      //[] __device__ (int a, int b) { return a*b; }
    );
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    for(int i=0; i<N; i++) {
      REQUIRE(output[i] == input1[i] * input2[i]);
    }

    // update transform parameters
    cf.transform(task,
      input1, input1+N, input2, output, AddOrMultiply{true}
      //[] __device__ (int a, int b) { return a+b; }
    );
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    for(int i=0; i<N; i++) {
      REQUIRE(output[i] == input1[i]+input2[i]);
    }

    tf::cuda_free(input1);
    tf::cuda_free(input2);
    tf::cuda_free(output);
  }
}

TEST_CASE("cudaFlow.update.binary_transform" * doctest::timeout(300)) {
  update_binary_transform<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.update.binary_transform" * doctest::timeout(300)) {
  update_binary_transform<tf::cudaFlowCapturer>();
}

// ----------------------------------------------------------------------------
// update scan
// ----------------------------------------------------------------------------

template <typename F>
void update_scan() {
  
  F cf;

  for(int N=1; N<=100000; N += (N/10+1)) {
  
    cf.clear();

    auto input1 = tf::cuda_malloc_shared<int>(N);
    auto input2 = tf::cuda_malloc_shared<int>(N);
    auto output1 = tf::cuda_malloc_shared<int>(N);
    auto output2 = tf::cuda_malloc_shared<int>(N);

    for(int i=0; i<N; i++) {
      input1[i] = i;
      input2[i] = -i;
      output1[i] = 0;
      output2[i] = 0;
    }
    
    // scan
    auto inclusive_scan = cf.inclusive_scan(
      input1, input1+N, output1, AddOrMultiply{true}
    );
    auto exclusive_scan = cf.exclusive_scan(
      input2, input2+N, output2, AddOrMultiply{true}
    );
    cf.offload();

    REQUIRE(cf.num_tasks() == 2);
    for(int i=1; i<N; i++) {
      REQUIRE(output1[i] == output1[i-1] + input1[i]);
      REQUIRE(output2[i] == output2[i-1] + input2[i-1]);
    }
    
    // update scan
    cf.inclusive_scan(inclusive_scan,
      input2, input2+N, output2, AddOrMultiply{true}
    );
    cf.exclusive_scan(exclusive_scan,
      input1, input1+N, output1, AddOrMultiply{true}
    );
    cf.offload();

    REQUIRE(cf.num_tasks() == 2);
    for(int i=1; i<N; i++) {
      REQUIRE(output1[i] == output1[i-1] + input1[i-1]);
      REQUIRE(output2[i] == output2[i-1] + input2[i]);
    }

    // ---------- transform_scan
    cf.clear();

    inclusive_scan = cf.transform_inclusive_scan(
      input1, input1+N, output1, AddOrMultiply{true}, MultiplyScalar{2}
      //[] __device__ (int a, int b) { return a + b; },
      //[] __device__ (int a) { return a * 2; }
    );

    exclusive_scan = cf.transform_exclusive_scan(
      input2, input2+N, output2, AddOrMultiply{true}, MultiplyScalar{10}
      //[] __device__ (int a, int b) { return a + b; },
      //[] __device__ (int a) { return a * 10; }
    );
    
    cf.offload();

    REQUIRE(cf.num_tasks() == 2);
    for(int i=1; i<N; i++) {
      REQUIRE(output1[i] == output1[i-1] + input1[i]*2);
      REQUIRE(output2[i] == output2[i-1] + input2[i-1]*10);
    }
    
    // ---------- update transform scan
    
    cf.transform_inclusive_scan(inclusive_scan,
      input2, input2+N, output2, AddOrMultiply{true}, MultiplyScalar{2}
      //[] __device__ (int a, int b) { return a + b; },
      //[] __device__ (int a) { return a * 2; }
    );

    cf.transform_exclusive_scan(exclusive_scan,
      input1, input1+N, output1, AddOrMultiply{true}, MultiplyScalar{10}
      //[] __device__ (int a, int b) { return a + b; },
      //[] __device__ (int a) { return a * 10; }
    );

    cf.offload();
    
    REQUIRE(cf.num_tasks() == 2);
    for(int i=1; i<N; i++) {
      REQUIRE(output2[i] == output2[i-1] + input2[i]*2);
      REQUIRE(output1[i] == output1[i-1] + input1[i-1]*10);
    }

    tf::cuda_free(input1);
    tf::cuda_free(input2);
    tf::cuda_free(output1);
    tf::cuda_free(output2);
  }
}

TEST_CASE("cudaFlow.update.scan" * doctest::timeout(300)) {
  update_scan<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.update.scan" * doctest::timeout(300)) {
  update_scan<tf::cudaFlowCapturer>();
}

// ----------------------------------------------------------------------------
// update merge
// ----------------------------------------------------------------------------

template <typename F>
void update_merge() {

  F cf;

  for(int N=1; N<=100000; N += (N/10+1)) {
  
    cf.clear();

    auto input1 = tf::cuda_malloc_shared<int>(N);
    auto input2 = tf::cuda_malloc_shared<int>(2*N);
    auto output1 = tf::cuda_malloc_shared<int>(3*N);
    auto output2 = tf::cuda_malloc_shared<int>(3*N);

    std::iota(input1, input1+N, 0);
    std::iota(input2, input2+2*N, 0);
    std::merge(input1, input1+N, input2, input2+2*N, output2);

    // merge
    auto merge = cf.merge(
      input1, input1+N, input2, input2+2*N, output1, tf::cuda_less<int>()
    );
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(std::is_sorted(output1, output1+3*N));

    for(int i=0; i<3*N; i++) {
      REQUIRE(output1[i] == output2[i]);
      output1[i] = output2[i] = rand();
    }

    // update merge
    cf.merge(merge,
      input1, input1+N, input2, input2+N, output2, tf::cuda_less<int>()
    );
    cf.offload();
    
    std::merge(input1, input1+N, input2, input2+N, output1);

    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(std::is_sorted(output2, output2+2*N));

    for(int i=0; i<2*N; i++) {
      REQUIRE(output1[i] == output2[i]);
    }
    
    tf::cuda_free(input1);
    tf::cuda_free(input2);
    tf::cuda_free(output1);
    tf::cuda_free(output2);
  }
}

TEST_CASE("cudaFlow.update.merge" * doctest::timeout(300)) {
  update_merge<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.update.merge" * doctest::timeout(300)) {
  update_merge<tf::cudaFlowCapturer>();
}

// ----------------------------------------------------------------------------
// update sort
// ----------------------------------------------------------------------------
 
template <typename F>
void update_sort() {
   
  F cf;

  for(int N=1; N<=100000; N += (N/100+1)) {
  
    cf.clear();

    auto input1 = tf::cuda_malloc_shared<int>(N);
    auto input2 = tf::cuda_malloc_shared<int>(N);

    for(int i=0; i<N; i++) {
      input1[i] = rand();
      input2[i] = rand();
    }
    
    // create sort
    auto sort = cf.sort(input1, input1+N, LessOrGreater{true});
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(std::is_sorted(input1, input1+N));

    // update sort
    cf.sort(sort, input2, input2+N, LessOrGreater{true});
    cf.offload();
    
    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(std::is_sorted(input2, input2+N, std::less<int>()));
    
    // update sort with a different kernel
    cf.sort(sort, input1, input1+N, tf::cuda_greater<int>());
    cf.offload();
    
    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(std::is_sorted(input1, input1+N, std::greater<int>()));
    
    // free the data 
    tf::cuda_free(input1);
    tf::cuda_free(input2);
  }
}

TEST_CASE("cudaFlow.update.sort" * doctest::timeout(300)) {
  update_sort<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.update.sort" * doctest::timeout(300)) {
  update_sort<tf::cudaFlowCapturer>();
}

// ----------------------------------------------------------------------------
// update sort_by_key
// ----------------------------------------------------------------------------
 
template <typename F>
void update_sort_by_key() {
  
  std::random_device rd;
  std::mt19937 g(rd());
   
  F cf;

  for(int N=1; N<=100000; N += (N/100+1)) {
  
    cf.clear();

    auto input1 = tf::cuda_malloc_shared<int>(N);
    auto input2 = tf::cuda_malloc_shared<int>(N);
    auto index1 = tf::cuda_malloc_shared<int>(N);
    auto index2 = tf::cuda_malloc_shared<int>(N);
    std::vector<int> index(N);

    for(int i=0; i<N; i++) {
      input1[i] = i;
      input2[i] = i;
      index1[i] = i;
      index2[i] = i;
      index [i] = i;
    }
    std::shuffle(input1, input1+N, g);
    std::shuffle(input2, input2+N, g);
    
    // create sort
    std::sort(index.begin(), index.end(), [&](auto i, auto j){
      return input1[i] < input1[j];
    });
    auto sort = cf.sort_by_key(input1, input1+N, index1, LessOrGreater{true});
    cf.offload();

    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(std::is_sorted(input1, input1+N));
    for(int i=0; i<N; i++) {
      REQUIRE(index[i] == index1[i]);
    }

    // update sort
    for(int i=0; i<N; i++) {
      index[i] = i;
    }
    std::sort(index.begin(), index.end(), [&](auto i, auto j){
      return input2[i] > input2[j];
    });
    cf.sort_by_key(sort, input2, input2+N, index2, LessOrGreater{false});
    cf.offload();
    
    REQUIRE(cf.num_tasks() == 1);
    REQUIRE(std::is_sorted(input2, input2+N, std::greater<int>()));
    for(int i=0; i<N; i++) {
      REQUIRE(index[i] == index2[i]);
    }
    
    // free the data 
    tf::cuda_free(input1);
    tf::cuda_free(input2);
    tf::cuda_free(index1);
    tf::cuda_free(index2);
  }
}

TEST_CASE("cudaFlow.update.sort_by_key" * doctest::timeout(300)) {
  update_sort_by_key<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.update.sort_by_key" * doctest::timeout(300)) {
  update_sort_by_key<tf::cudaFlowCapturer>();
}

// ----------------------------------------------------------------------------
// update find
// ----------------------------------------------------------------------------

template <typename F>
void update_find() {
   
  F cf;

  for(unsigned N=1; N<=100000; N += (N/100+1)) {
  
    cf.clear();

    auto input1 = tf::cuda_malloc_shared<int>(N);
    auto input2 = tf::cuda_malloc_shared<int>(N);
    auto index1 = tf::cuda_malloc_shared<unsigned>(1);
    auto index2 = tf::cuda_malloc_shared<unsigned>(1);

    for(unsigned i=0; i<N; i++) {
      input1[i] = i;
      input2[i] = i;
    }
    
    // create find
    auto find_if = cf.find_if(input1, input1+N, index1, IsEqual{(int)(N/2)});
    cf.offload();

    REQUIRE(*index1 != N);
    REQUIRE(input1[*index1] == N/2);

    // update find
    cf.find_if(find_if, input2, input2+N, index2, IsEqual{(int)(N/2 + 1)});
    cf.offload();
    
    REQUIRE(cf.num_tasks() == 1);

    if( N/2+1 >= N) {
      REQUIRE(*index2 == N);
    }
    else {
      REQUIRE(input2[*index2] == (N/2+1));
    }

    // free the data 
    tf::cuda_free(input1);
    tf::cuda_free(input2);
    tf::cuda_free(index1);
    tf::cuda_free(index2);
  }
}

TEST_CASE("cudaFlow.update.find" * doctest::timeout(300)) {
  update_find<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.update.find" * doctest::timeout(300)) {
  update_find<tf::cudaFlowCapturer>();
}

// ----------------------------------------------------------------------------
// update min-/max-element
// ----------------------------------------------------------------------------

template <typename F>
void update_minmax_element() {
   
  F cf;

  for(unsigned N=1; N<=100000; N += (N/100+1)) {
  
    cf.clear();

    auto input1 = tf::cuda_malloc_shared<int>(N);
    auto input2 = tf::cuda_malloc_shared<int>(N);
    auto index1 = tf::cuda_malloc_shared<unsigned>(1);
    auto index2 = tf::cuda_malloc_shared<unsigned>(1);

    for(unsigned i=0; i<N; i++) {
      input1[i] = rand();
      input2[i] = rand();
    }
    
    // create find
    auto find_min = cf.min_element(input1, input1+N, index1, tf::cuda_less<int>());
    auto find_max = cf.max_element(input2, input2+N, index2, tf::cuda_less<int>());
    cf.offload();

    REQUIRE(input1[*index1] == *std::min_element(input1, input1+N));
    REQUIRE(input2[*index2] == *std::max_element(input2, input2+N));

    // update find
    cf.min_element(find_min, input2, input2+N, index2, tf::cuda_less<int>());
    cf.max_element(find_max, input1, input1+N, index1, tf::cuda_less<int>());
    cf.offload();

    REQUIRE(cf.num_tasks() == 2);
    REQUIRE(input2[*index2] == *std::min_element(input2, input2+N));
    REQUIRE(input1[*index1] == *std::max_element(input1, input1+N));

    // free the data 
    tf::cuda_free(input1);
    tf::cuda_free(input2);
    tf::cuda_free(index1);
    tf::cuda_free(index2);
  }
}

TEST_CASE("cudaFlow.update.minmax_element" * doctest::timeout(300)) {
  update_minmax_element<tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.update.minmax_element" * doctest::timeout(300)) {
  update_minmax_element<tf::cudaFlowCapturer>();
}

