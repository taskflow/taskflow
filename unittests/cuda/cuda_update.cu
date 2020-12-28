#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/cudaflow.hpp>

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
TEST_CASE("offload" * doctest::timeout(300)) {
  tf::Executor executor;

  for(size_t N = 1; N < 65532; N = N * 2 + 1) {
    tf::Taskflow taskflow;

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
    auto offload_t = taskflow.emplace([&](tf::cudaFlow& cf) {
      cf.kernel(
        32, 512, 0,
        add<int>,
        a, a, a, N
      );

      cf.offload_n(times+1);
    }).name("offload");

    //verify
    auto verify_t = taskflow.emplace([&](tf::cudaFlow& cf) {
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

//----------------------------------------------------------------------
//join_n
//----------------------------------------------------------------------
TEST_CASE("join" * doctest::timeout(300)) {
  tf::Executor executor;

  for(size_t N = 1; N < 65532; N = N * 2 + 1) {
    tf::Taskflow taskflow;

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
    auto join_t = taskflow.emplace([&](tf::cudaFlow& cf) {
      cf.kernel(
        32, 512, 0,
        add<int>,
        a, a, a, N
      );

      cf.offload_n(times);
    }).name("join");

    //verify
    auto verify_t = taskflow.emplace([&](tf::cudaFlow& cf) {
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

//----------------------------------------------------------------------
//update kernel
//----------------------------------------------------------------------

template <typename T>
void update_kernel() {
  tf::Executor executor;

  for(size_t N = 1; N < 65529; N = N * 2 + 1) {
    tf::Taskflow taskflow;

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
    auto add_t = taskflow.emplace([&](tf::cudaFlow& cf) {
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

      cf.update_kernel(
        multi_t,
        64, 128, 0,
        operand[ind[2]], operand[ind[0]], operand[ind[1]], N
      );

      cf.update_kernel(
        add_t,
        16, 256, 0,
        operand[ind[1]], operand[ind[0]], operand[ind[2]], N
      );

      cf.offload();

      cf.update_kernel(
        multi_t,
        8, 1024, 0,
        operand[ind[0]], operand[ind[2]], operand[ind[1]], N
      );

      cf.update_kernel(
        add_t,
        64, 64, 0,
        operand[ind[2]], operand[ind[1]], operand[ind[0]], N
      );

      cf.offload();
    }).name("add");

    //verify
    auto verify_t = taskflow.emplace([&](tf::cudaFlow& cf) {
      auto multi1_t = cf.transform(
        ans_operand[ind[2]],  ans_operand[ind[2]]+ N,
        [] __device__ (T& v1, T& v2) { return v1 * v2; },
        ans_operand[ind[0]], ans_operand[ind[1]]
      );

      auto add1_t = cf.transform(
        ans_operand[ind[0]],  ans_operand[ind[0]]+ N,
        [] __device__ (T& v1, T& v2) { return v1 + v2; },
        ans_operand[ind[1]], ans_operand[ind[2]]
      );

      auto multi2_t = cf.transform(
        ans_operand[ind[1]],  ans_operand[ind[1]]+ N,
        [] __device__ (T& v1, T& v2) { return v1 * v2; },
        ans_operand[ind[2]], ans_operand[ind[0]]
      );

      auto add2_t = cf.transform(
        ans_operand[ind[2]],  ans_operand[ind[2]]+ N,
        [] __device__ (T& v1, T& v2) { return v1 + v2; },
        ans_operand[ind[1]], ans_operand[ind[0]]
      );

      auto multi3_t = cf.transform(
        ans_operand[ind[1]],  ans_operand[ind[1]]+ N,
        [] __device__ (T& v1, T& v2) { return v1 * v2; },
        ans_operand[ind[0]], ans_operand[ind[2]]
      );

      auto add3_t = cf.transform(
        ans_operand[ind[0]],  ans_operand[ind[0]]+ N,
        [] __device__ (T& v1, T& v2) { return v1 + v2; },
        ans_operand[ind[2]], ans_operand[ind[1]]
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

TEST_CASE("update.kernel.int" * doctest::timeout(300)) {
  update_kernel<int>();
}

TEST_CASE("update.kernel.float" * doctest::timeout(300)) {
  update_kernel<float>();
}

TEST_CASE("update.kernel.double" * doctest::timeout(300)) {
  update_kernel<double>();
}

//----------------------------------------------------------------------
//update copy
//----------------------------------------------------------------------
template <typename T>
void update_copy() {
  tf::Executor executor;

  for(int N = 1; N < 65459; N = N * 2 + 1) {
    tf::Taskflow taskflow;

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
    auto h2d_t = taskflow.emplace([&](tf::cudaFlow& cf) {
      auto h2d_t = cf.copy(da, ha.data(), N).name("h2d");
      cf.offload();

      cf.update_copy(h2d_t, db, hb.data(), N);
      cf.offload();

      cf.update_copy(h2d_t, dc, hc.data(), N);
      cf.offload();

    });

    auto kernel_t = taskflow.emplace([&](tf::cudaFlow& cf) {
      auto add1_t = cf.transform(
        dz,  dz + N,
        [] __device__ (T& v1, T& v2) { return v1 + v2; },
        da, db
      );

      auto add2_t = cf.transform(
        dc,  dc + N,
        [] __device__ (T& v1, T& v2) { return v1 - v2; },
        dc, dz
      );

      add1_t.precede(add2_t);
    });

    auto d2h_t = taskflow.emplace([&](tf::cudaFlow& cf) {
      auto d2h_t = cf.copy(hc.data(), dc, N).name("d2h");
      cf.offload();

      cf.update_copy(d2h_t, hz.data(), dz, N);
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

TEST_CASE("update.copy.int" * doctest::timeout(300)) {
  update_copy<int>();
}

TEST_CASE("update.copy.float" * doctest::timeout(300)) {
  update_copy<float>();
}

TEST_CASE("update.copy.double" * doctest::timeout(300)) {
  update_copy<double>();
}


//----------------------------------------------------------------------
//update memcpy
//----------------------------------------------------------------------
template <typename T>
void update_memcpy() {
  tf::Executor executor;

  for(int N = 1; N < 65459; N = N * 2 + 1) {
    tf::Taskflow taskflow;

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
    auto h2d_t = taskflow.emplace([&](tf::cudaFlow& cf) {
      auto h2d_t = cf.memcpy(da, ha.data(), sizeof(T) * N).name("h2d");
      cf.offload();

      cf.update_memcpy(h2d_t, db, hb.data(), sizeof(T) * N);
      cf.offload();

      cf.update_memcpy(h2d_t, dc, hc.data(), sizeof(T) * N);
      cf.offload();

    });

    auto kernel_t = taskflow.emplace([&](tf::cudaFlow& cf) {
      auto add1_t = cf.transform(
        dz,  dz + N,
        [] __device__ (T& v1, T& v2) { return v1 + v2; },
        da, db
      );

      auto add2_t = cf.transform(
        dc,  dc + N,
        [] __device__ (T& v1, T& v2) { return v1 - v2; },
        dc, dz
      );

      add1_t.precede(add2_t);
    });

    auto d2h_t = taskflow.emplace([&](tf::cudaFlow& cf) {
      auto d2h_t = cf.memcpy(hc.data(), dc, sizeof(T) * N).name("d2h");
      cf.offload();

      cf.update_memcpy(d2h_t, hz.data(), dz, sizeof(T) * N);
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

TEST_CASE("update.memcpy.int" * doctest::timeout(300)) {
  update_memcpy<int>();
}

TEST_CASE("update.memcpy.float" * doctest::timeout(300)) {
  update_memcpy<float>();
}

TEST_CASE("update.memcpy.double" * doctest::timeout(300)) {
  update_memcpy<double>();
}



//----------------------------------------------------------------------
//update memset
//----------------------------------------------------------------------
template <typename T>
void update_memset() {
  tf::Executor executor;

  for(size_t N = 1; N < 65199; N = N * 2 + 1) {
    tf::Taskflow taskflow;

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
    auto memset_t = taskflow.emplace([&](tf::cudaFlow& cf) {
      auto memset_t = cf.memset(ans_a, 0, N * sizeof(T));
      cf.offload();

      cf.update_memset(memset_t, a, 0, N * sizeof(T));
      cf.offload();

      cf.update_memset(memset_t, b, 1, (N + 37) * sizeof(T));
      cf.offload();
    }).name("memset");

    //verify
    auto verify_t = taskflow.emplace([&](tf::cudaFlow& cf) {
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

TEST_CASE("update.memset.int" * doctest::timeout(300)) {
  update_memset<int>();
}

TEST_CASE("update.memset.float" * doctest::timeout(300)) {
  update_memset<float>();
}

TEST_CASE("update.memset.double" * doctest::timeout(300)) {
  update_memset<double>();
}

//----------------------------------------------------------------------
//rebind kernel
//----------------------------------------------------------------------

template <typename T>
void rebind_kernel() {
  tf::Executor executor;

  for(size_t N = 1; N < 65529; N = N * 2 + 1) {
    tf::Taskflow taskflow;

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

    
    //rebind_kernel
    auto add_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
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

      cf.rebind_kernel(
        multi_t,
        64, 128, 0,
        multiply<T>,
        operand[ind[2]], operand[ind[0]], operand[ind[1]], N
      );

      cf.rebind_kernel(
        add_t,
        16, 256, 0,
        add<T>,
        operand[ind[1]], operand[ind[0]], operand[ind[2]], N
      );

      cf.offload();

      cf.rebind_kernel(
        multi_t,
        8, 1024, 0,
        multiply<T>,
        operand[ind[0]], operand[ind[2]], operand[ind[1]], N
      );

      cf.rebind_kernel(
        add_t,
        64, 64, 0,
        add<T>,
        operand[ind[2]], operand[ind[1]], operand[ind[0]], N
      );

      cf.offload();
    }).name("add");

    //verify
    auto verify_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
      auto multi1_t = cf.transform(
        ans_operand[ind[2]],  ans_operand[ind[2]]+ N,
        [] __device__ (T& v1, T& v2) { return v1 * v2; },
        ans_operand[ind[0]], ans_operand[ind[1]]
      );

      auto add1_t = cf.transform(
        ans_operand[ind[0]],  ans_operand[ind[0]]+ N,
        [] __device__ (T& v1, T& v2) { return v1 + v2; },
        ans_operand[ind[1]], ans_operand[ind[2]]
      );

      auto multi2_t = cf.transform(
        ans_operand[ind[1]],  ans_operand[ind[1]]+ N,
        [] __device__ (T& v1, T& v2) { return v1 * v2; },
        ans_operand[ind[2]], ans_operand[ind[0]]
      );

      auto add2_t = cf.transform(
        ans_operand[ind[2]],  ans_operand[ind[2]]+ N,
        [] __device__ (T& v1, T& v2) { return v1 + v2; },
        ans_operand[ind[1]], ans_operand[ind[0]]
      );

      auto multi3_t = cf.transform(
        ans_operand[ind[1]],  ans_operand[ind[1]]+ N,
        [] __device__ (T& v1, T& v2) { return v1 * v2; },
        ans_operand[ind[0]], ans_operand[ind[2]]
      );

      auto add3_t = cf.transform(
        ans_operand[ind[0]],  ans_operand[ind[0]]+ N,
        [] __device__ (T& v1, T& v2) { return v1 + v2; },
        ans_operand[ind[2]], ans_operand[ind[1]]
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

TEST_CASE("rebind.kernel.int" * doctest::timeout(300)) {
  rebind_kernel<int>();
}

TEST_CASE("rebind.kernel.float" * doctest::timeout(300)) {
  rebind_kernel<float>();
}

TEST_CASE("rebind.kernel.double" * doctest::timeout(300)) {
  rebind_kernel<double>();
}

//----------------------------------------------------------------------
//rebind copy
//----------------------------------------------------------------------
template <typename T>
void rebind_copy() {
  tf::Executor executor;

  for(int N = 1; N < 65459; N = N * 2 + 1) {
    tf::Taskflow taskflow;

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


    //rebind_copy
    auto h2d_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
      auto h2d_t = cf.copy(da, ha.data(), N).name("h2d");
      cf.offload();

      cf.rebind_copy(h2d_t, db, hb.data(), N);
      cf.offload();

      cf.rebind_copy(h2d_t, dc, hc.data(), N);
      cf.offload();

    });

    auto kernel_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
      auto add1_t = cf.transform(
        dz,  dz + N,
        [] __device__ (T& v1, T& v2) { return v1 + v2; },
        da, db
      );

      auto add2_t = cf.transform(
        dc,  dc + N,
        [] __device__ (T& v1, T& v2) { return v1 - v2; },
        dc, dz
      );

      add1_t.precede(add2_t);
    });

    auto d2h_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
      auto d2h_t = cf.copy(hc.data(), dc, N).name("d2h");
      cf.offload();

      cf.rebind_copy(d2h_t, hz.data(), dz, N);
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

TEST_CASE("rebind.copy.int" * doctest::timeout(300)) {
  rebind_copy<int>();
}

TEST_CASE("rebind.copy.float" * doctest::timeout(300)) {
  rebind_copy<float>();
}

TEST_CASE("rebind.copy.double" * doctest::timeout(300)) {
  rebind_copy<double>();
}


//----------------------------------------------------------------------
//rebind memcpy
//----------------------------------------------------------------------
template <typename T>
void rebind_memcpy() {
  tf::Executor executor;

  for(int N = 1; N < 65459; N = N * 2 + 1) {
    tf::Taskflow taskflow;

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


    //rebind_memcpy
    auto h2d_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
      auto h2d_t = cf.memcpy(da, ha.data(), sizeof(T) * N).name("h2d");
      cf.offload();

      cf.rebind_memcpy(h2d_t, db, hb.data(), sizeof(T) * N);
      cf.offload();

      cf.rebind_memcpy(h2d_t, dc, hc.data(), sizeof(T) * N);
      cf.offload();

    });

    auto kernel_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
      auto add1_t = cf.transform(
        dz,  dz + N,
        [] __device__ (T& v1, T& v2) { return v1 + v2; },
        da, db
      );

      auto add2_t = cf.transform(
        dc,  dc + N,
        [] __device__ (T& v1, T& v2) { return v1 - v2; },
        dc, dz
      );

      add1_t.precede(add2_t);
    });

    auto d2h_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
      auto d2h_t = cf.memcpy(hc.data(), dc, sizeof(T) * N).name("d2h");
      cf.offload();

      cf.rebind_memcpy(d2h_t, hz.data(), dz, sizeof(T) * N);
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

TEST_CASE("rebind.memcpy.int" * doctest::timeout(300)) {
  rebind_memcpy<int>();
}

TEST_CASE("rebind.memcpy.float" * doctest::timeout(300)) {
  rebind_memcpy<float>();
}

TEST_CASE("rebind.memcpy.double" * doctest::timeout(300)) {
  rebind_memcpy<double>();
}



//----------------------------------------------------------------------
//rebind memset
//----------------------------------------------------------------------
template <typename T>
void rebind_memset() {
  tf::Executor executor;

  for(size_t N = 1; N < 65199; N = N * 2 + 1) {
    tf::Taskflow taskflow;

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

    //rebind_memset
    auto memset_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
      auto memset_t = cf.memset(ans_a, 0, N * sizeof(T));
      cf.offload();

      cf.rebind_memset(memset_t, a, 0, N * sizeof(T));
      cf.offload();

      cf.rebind_memset(memset_t, b, 1, (N + 37) * sizeof(T));
      cf.offload();
    }).name("memset");

    //verify
    auto verify_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
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

TEST_CASE("rebind.memset.int" * doctest::timeout(300)) {
  rebind_memset<int>();
}

TEST_CASE("rebind.memset.float" * doctest::timeout(300)) {
  rebind_memset<float>();
}

TEST_CASE("rebind.memset.double" * doctest::timeout(300)) {
  rebind_memset<double>();
}


/*
//----------------------------------------------------------------------
//update transpose
//----------------------------------------------------------------------
template <typename T>
void update_transpose() {
  tf::Executor executor;

  for(size_t rows = 1; rows <= 7999; rows*=2+3) {
    for(size_t cols = 1; cols <= 8021; cols*=3+5) {

      tf::Taskflow taskflow;

      T* input_mat {nullptr};
      T* output1_mat {nullptr};
      T* output2_mat {nullptr};
      T* output3_mat {nullptr};
      bool* check {nullptr};
      
      //allocate
      auto allocate_t = taskflow.emplace([&]() {
        REQUIRE(cudaMallocManaged(&input_mat, (rows * cols) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMallocManaged(&output1_mat, (rows * cols) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMallocManaged(&output2_mat, (rows * cols) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMallocManaged(&output3_mat, (rows * cols) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMallocManaged(&check, sizeof(bool)) == cudaSuccess);
      }).name("allocate");

      //initialize
      auto initialize_t = taskflow.emplace([&]() {
        std::generate_n(input_mat, rows * cols, [](){ return ::rand(); });
        *check = true;
      }).name("initialize");


      //update_transpose
      auto transpose_t = taskflow.emplace([&](tf::cudaFlow& cf) {

        auto transpose_t = tf::cudaBLAF(cf).transpose(
          input_mat,
          output1_mat,
          rows,
          cols
        );

        cf.offload();

        tf::cudaBLAF(cf).update_transpose(
          transpose_t,
          output1_mat,
          output2_mat,
          cols,
          rows
        );

        cf.offload();

        tf::cudaBLAF(cf).update_transpose(
          transpose_t,
          output2_mat,
          output3_mat,
          rows,
          cols
        );
        cf.offload();


      }).name("transpose");

      //verify
      auto verify_t = taskflow.emplace([&](tf::cudaFlow& cf) {
        cf.kernel(
          32, 512, 0,
          verify<T>,
          input_mat, output2_mat, check,
          rows * cols
        );

        cf.kernel(
          32, 512, 0,
          verify<T>,
          output1_mat, output3_mat, check,
          rows * cols
        );

        cf.offload();

        REQUIRE(*check);

      }).name("verify");

       //free memory
      auto deallocate_t = taskflow.emplace([&]() {
        REQUIRE(cudaFree(input_mat) == cudaSuccess);
        REQUIRE(cudaFree(output1_mat) == cudaSuccess);
        REQUIRE(cudaFree(output2_mat) == cudaSuccess);
        REQUIRE(cudaFree(output3_mat) == cudaSuccess);
        REQUIRE(cudaFree(check) == cudaSuccess);
      }).name("deallocate");
      

      allocate_t.precede(initialize_t);
      initialize_t.precede(transpose_t);
      transpose_t.precede(verify_t);
      verify_t.precede(deallocate_t);

      executor.run(taskflow).wait();

    }
  }
}

TEST_CASE("update.transpose.int" * doctest::timeout(300) ) {
  update_transpose<int>();
}

TEST_CASE("update.transpose.float" * doctest::timeout(300) ) {
  update_transpose<float>();
}

TEST_CASE("update.transpose.double" * doctest::timeout(300) ) {
  update_transpose<double>();
}

//----------------------------------------------------------------------
//update matmuti c = a * b
//----------------------------------------------------------------------
template <typename T>
void update_matmul() {
  tf::Executor executor;

  for(size_t m = 1; m <= 1922; m*=2+5) {
    for(size_t k = 1; k <= 2103; k*=3+3) {
      for(size_t n = 1; n <= 1998; n*=2+7) {

      tf::Taskflow taskflow;

      T* a1 {nullptr};
      T* a2 {nullptr};
      T* b1 {nullptr};
      T* b2 {nullptr};
      T* c1 {nullptr};
      T* c2 {nullptr};
      T* ans_c1 {nullptr};
      T* ans_c2 {nullptr};
      bool* check {nullptr};
      
      //allocate
      auto allocate_t = taskflow.emplace([&]() {
        REQUIRE(cudaMallocManaged(&a1, (m * k) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMallocManaged(&a2, (m * k) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMallocManaged(&b1, (k * n) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMallocManaged(&b2, (k * n) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMallocManaged(&c1, (m * n) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMallocManaged(&c2, (m * n) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMallocManaged(&ans_c1, (m * n) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMallocManaged(&ans_c2, (m * n) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMallocManaged(&check, sizeof(bool)) == cudaSuccess);
      }).name("allocate");

      //initialize
      auto initialize_t = taskflow.emplace([&]() {
        std::generate_n(a1, m * k, [](){ return ::rand(); });
        std::generate_n(a2, m * k, [](){ return ::rand(); });
        std::generate_n(b1, k * n, [](){ return ::rand(); });
        std::generate_n(b2, k * n, [](){ return ::rand(); });
        *check = true;
      }).name("initialize");


      //update_matmul
      auto matmul_t = taskflow.emplace([&](tf::cudaFlow& cf) {

        auto matmul_t = tf::cudaBLAF(cf).matmul(
          a1,
          b1,
          c1,
          m,
          k,
          n
        );
        cf.offload();

        tf::cudaBLAF(cf).update_matmul(
          matmul_t,
          a1,
          b2,
          c2,
          m,
          k,
          n
        );
        cf.offload();

        tf::cudaBLAF(cf).update_matmul(
          matmul_t,
          a2,
          b2,
          c1,
          m,
          k,
          n
        );
        cf.offload();

      }).name("matmul");
      
      //verify
      auto verify_t = taskflow.emplace([&](tf::cudaFlow& cf) {

        auto matmul1_t = tf::cudaBLAF(cf).matmul(
          a1,
          b2,
          ans_c2,
          m,
          k,
          n
        );

        auto matmul2_t = tf::cudaBLAF(cf).matmul(
          a2,
          b2,
          ans_c1,
          m,
          k,
          n
        );

        auto verify1_t = cf.kernel(
          32, 512, 0,
          verify<T>,
          c2, ans_c2, check,
          m * n
        );

        auto verify2_t = cf.kernel(
          32, 512, 0,
          verify<T>,
          c1, ans_c1, check,
          m * n
        );

        matmul1_t.precede(verify1_t);
        matmul2_t.precede(verify2_t);

        cf.offload();

        REQUIRE(*check);

      }).name("verify");

       //free memory
      auto deallocate_t = taskflow.emplace([&]() {
        REQUIRE(cudaFree(a1) == cudaSuccess);
        REQUIRE(cudaFree(a2) == cudaSuccess);
        REQUIRE(cudaFree(b1) == cudaSuccess);
        REQUIRE(cudaFree(b2) == cudaSuccess);
        REQUIRE(cudaFree(c1) == cudaSuccess);
        REQUIRE(cudaFree(c2) == cudaSuccess);
        REQUIRE(cudaFree(ans_c1) == cudaSuccess);
        REQUIRE(cudaFree(ans_c2) == cudaSuccess);

        REQUIRE(cudaFree(check) == cudaSuccess);
      }).name("deallocate");
      

      allocate_t.precede(initialize_t);
      initialize_t.precede(matmul_t);
      matmul_t.precede(verify_t);
      verify_t.precede(deallocate_t);

      executor.run(taskflow).wait();

      }
    }
  }
}

TEST_CASE("update.matmul.int" * doctest::timeout(300) ) {
  update_matmul<int>();
}

TEST_CASE("update.matmul.float" * doctest::timeout(300) ) {
  update_matmul<float>();
}

TEST_CASE("update.matmul.double" * doctest::timeout(300) ) {
  update_matmul<double>();
}

*/
