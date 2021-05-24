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

      cf.kernel(
        multi_t,
        64, 128, 0,
        multiply<T>,
        operand[ind[2]], operand[ind[0]], operand[ind[1]], N
      );

      cf.kernel(
        add_t,
        16, 256, 0,
        add<T>,
        operand[ind[1]], operand[ind[0]], operand[ind[2]], N
      );

      cf.offload();

      cf.kernel(
        multi_t,
        8, 1024, 0,
        multiply<T>,
        operand[ind[0]], operand[ind[2]], operand[ind[1]], N
      );

      cf.kernel(
        add_t,
        64, 64, 0,
        add<T>,
        operand[ind[2]], operand[ind[1]], operand[ind[0]], N
      );

      cf.offload();
    }).name("add");

    //verify
    auto verify_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
      //auto multi1_t = cf.transform(
      //  ans_operand[ind[2]],  ans_operand[ind[2]]+ N,
      //  [] __device__ (T& v1, T& v2) { return v1 * v2; },
      //  ans_operand[ind[0]], ans_operand[ind[1]]
      //);

      auto multi1_t = cf.transform(
        ans_operand[ind[0]], ans_operand[ind[0]] + N, ans_operand[ind[1]],
        ans_operand[ind[2]],
        [] __device__ (T& v1, T& v2) { return v1*v2; }
      );

      //auto add1_t = cf.transform(
      //  ans_operand[ind[0]],  ans_operand[ind[0]]+ N,
      //  [] __device__ (T& v1, T& v2) { return v1 + v2; },
      //  ans_operand[ind[1]], ans_operand[ind[2]]
      //);

      auto add1_t = cf.transform(
        ans_operand[ind[1]], ans_operand[ind[1]]+N, ans_operand[ind[2]],
        ans_operand[ind[0]],
        [] __device__ (T& v1, T& v2) { return v1 + v2; }
      );

      //auto multi2_t = cf.transform(
      //  ans_operand[ind[1]],  ans_operand[ind[1]]+ N,
      //  [] __device__ (T& v1, T& v2) { return v1 * v2; },
      //  ans_operand[ind[2]], ans_operand[ind[0]]
      //);
      
      auto multi2_t = cf.transform(
        ans_operand[ind[2]], ans_operand[ind[2]] + N, ans_operand[ind[0]],
        ans_operand[ind[1]],
        [] __device__ (T& v1, T& v2) { return v1 * v2; }
      );

      //auto add2_t = cf.transform(
      //  ans_operand[ind[2]],  ans_operand[ind[2]]+ N,
      //  [] __device__ (T& v1, T& v2) { return v1 + v2; },
      //  ans_operand[ind[1]], ans_operand[ind[0]]
      //);
      
      auto add2_t = cf.transform(
        ans_operand[ind[1]], ans_operand[ind[1]] + N, ans_operand[ind[0]],
        ans_operand[ind[2]],
        [] __device__ (T& v1, T& v2) { return v1 + v2; }
      );

      //auto multi3_t = cf.transform(
      //  ans_operand[ind[1]],  ans_operand[ind[1]]+ N,
      //  [] __device__ (T& v1, T& v2) { return v1 * v2; },
      //  ans_operand[ind[0]], ans_operand[ind[2]]
      //);
      
      auto multi3_t = cf.transform(
        ans_operand[ind[0]], ans_operand[ind[0]] + N,  ans_operand[ind[2]],
        ans_operand[ind[1]],
        [] __device__ (T& v1, T& v2) { return v1 * v2; }
      );

      //auto add3_t = cf.transform(
      //  ans_operand[ind[0]],  ans_operand[ind[0]]+ N,
      //  [] __device__ (T& v1, T& v2) { return v1 + v2; },
      //  ans_operand[ind[2]], ans_operand[ind[1]]
      //);
      
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

TEST_CASE("cudaFlowCapturer.rebind.kernel.int" * doctest::timeout(300)) {
  rebind_kernel<int>();
}

TEST_CASE("cudaFlowCapturer.rebind.kernel.float" * doctest::timeout(300)) {
  rebind_kernel<float>();
}

TEST_CASE("cudaFlowCapturer.rebind.kernel.double" * doctest::timeout(300)) {
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

      cf.copy(h2d_t, db, hb.data(), N);
      cf.offload();

      cf.copy(h2d_t, dc, hc.data(), N);
      cf.offload();

    });

    auto kernel_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
      //auto add1_t = cf.transform(
      //  dz,  dz + N,
      //  [] __device__ (T& v1, T& v2) { return v1 + v2; },
      //  da, db
      //);
      
      auto add1_t = cf.transform(
        da, da+N, db,
        dz,
        [] __device__ (T& v1, T& v2) { return v1 + v2; }
      );

      //auto add2_t = cf.transform(
      //  dc,  dc + N,
      //  [] __device__ (T& v1, T& v2) { return v1 - v2; },
      //  dc, dz
      //);
      
      auto add2_t = cf.transform(
        dc, dc + N, dz,
        dc,
        [] __device__ (T& v1, T& v2) { return v1 - v2; }
      );

      add1_t.precede(add2_t);
    });

    auto d2h_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
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

TEST_CASE("cudaFlowCapturer.rebind.copy.int" * doctest::timeout(300)) {
  rebind_copy<int>();
}

TEST_CASE("cudaFlowCapturer.rebind.copy.float" * doctest::timeout(300)) {
  rebind_copy<float>();
}

TEST_CASE("cudaFlowCapturer.rebind.copy.double" * doctest::timeout(300)) {
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

      cf.memcpy(h2d_t, db, hb.data(), sizeof(T) * N);
      cf.offload();

      cf.memcpy(h2d_t, dc, hc.data(), sizeof(T) * N);
      cf.offload();

    });

    auto kernel_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
      //auto add1_t = cf.transform(
      //  dz,  dz + N,
      //  [] __device__ (T& v1, T& v2) { return v1 + v2; },
      //  da, db
      //);
      
      auto add1_t = cf.transform(
        da, da + N, db,
        dz,
        [] __device__ (T& v1, T& v2) { return v1 + v2; }
      );

      //auto add2_t = cf.transform(
      //  dc,  dc + N,
      //  [] __device__ (T& v1, T& v2) { return v1 - v2; },
      //  dc, dz
      //);
      
      auto add2_t = cf.transform(
        dc, dc + N, dz,
        dc,
        [] __device__ (T& v1, T& v2) { return v1 - v2; }
      );

      add1_t.precede(add2_t);
    });

    auto d2h_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
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

TEST_CASE("cudaFlowCapturer.rebind.memcpy.int" * doctest::timeout(300)) {
  rebind_memcpy<int>();
}

TEST_CASE("cudaFlowCapturer.rebind.memcpy.float" * doctest::timeout(300)) {
  rebind_memcpy<float>();
}

TEST_CASE("cudaFlowCapturer.rebind.memcpy.double" * doctest::timeout(300)) {
  rebind_memcpy<double>();
}

//----------------------------------------------------------------------
//rebind memset
//----------------------------------------------------------------------
template <typename T>
void rebind_memset() {

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

    //rebind_memset
    auto memset_t = taskflow.emplace([&](tf::cudaFlowCapturer& cf) {
      auto memset_t = cf.memset(ans_a, 0, N * sizeof(T));
      cf.offload();

      cf.memset(memset_t, a, 0, N * sizeof(T));
      cf.offload();

      cf.memset(memset_t, b, 1, (N + 37) * sizeof(T));
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

TEST_CASE("cudaFlowCapturer.rebind.memset.int" * doctest::timeout(300)) {
  rebind_memset<int>();
}

TEST_CASE("cudaFlowCapturer.rebind.memset.float" * doctest::timeout(300)) {
  rebind_memset<float>();
}

TEST_CASE("cudaFlowCapturer.rebind.memset.double" * doctest::timeout(300)) {
  rebind_memset<double>();
}

// ----------------------------------------------------------------------------
// rebind algorithms
// ----------------------------------------------------------------------------

TEST_CASE("cudaFlowCapturer.rebind.algorithms") {

  tf::cudaFlowCapturer capturer;

  auto data = tf::cuda_malloc_shared<int>(10000);
  auto res = tf::cuda_malloc_shared<int>(1);

  auto task = capturer.for_each(
    data, data+10000, []__device__(int& i) {
      i = 10;
    }
  );

  capturer.offload();

  for(int i=0; i<10000; i++) {
    REQUIRE(data[i] == 10);
  }
  REQUIRE(capturer.num_tasks() == 1);
  
  // rebind to single task
  capturer.single_task(task, [=] __device__ () {*data = 2;});

  capturer.offload();
  
  REQUIRE(*data == 2);
  for(int i=1; i<10000; i++) {
    REQUIRE(data[i] == 10);
  }
  REQUIRE(capturer.num_tasks() == 1);
  
  // rebind to for each index
  capturer.for_each_index(task, 0, 10000, 1,
    [=] __device__ (int i) {
      data[i] = -23;
    }
  );

  capturer.offload();
  
  for(int i=0; i<10000; i++) {
    REQUIRE(data[i] == -23);
  }
  REQUIRE(capturer.num_tasks() == 1);

  // rebind to reduce
  *res = 10;
  capturer.reduce(task, data, data + 10000, res, 
    []__device__(int a, int b){ return a + b; }
  );

  capturer.offload();

  REQUIRE(*res == -229990);
  REQUIRE(capturer.num_tasks() == 1);
  
  // rebind to uninitialized reduce
  capturer.uninitialized_reduce(task, data, data + 10000, res, 
    []__device__(int a, int b){ return a + b; }
  );

  capturer.offload();

  REQUIRE(*res == -230000);
  REQUIRE(capturer.num_tasks() == 1);
  
  // rebind to single task
  capturer.single_task(task, [res]__device__(){ *res = 999; });
  REQUIRE(*res == -230000);

  capturer.offload();
  REQUIRE(*res == 999);
  REQUIRE(capturer.num_tasks() == 1);

  // clear the capturer
  capturer.clear();
  REQUIRE(capturer.num_tasks() == 0);

  capturer.offload();
  REQUIRE(*res == 999);
  for(int i=0; i<10000; i++) {
    REQUIRE(data[i] == -23);
  }

  // clear the memory
  tf::cuda_free(data);
  tf::cuda_free(res);
}
