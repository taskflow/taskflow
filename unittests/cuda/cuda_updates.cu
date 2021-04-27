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

TEST_CASE("cudaFlow.update.kernel.int" * doctest::timeout(300)) {
  update_kernel<int>();
}

TEST_CASE("cudaFlow.update.kernel.float" * doctest::timeout(300)) {
  update_kernel<float>();
}

TEST_CASE("cudaFlow.update.kernel.double" * doctest::timeout(300)) {
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

TEST_CASE("cudaFlow.update.copy.int" * doctest::timeout(300)) {
  update_copy<int>();
}

TEST_CASE("cudaFlow.update.copy.float" * doctest::timeout(300)) {
  update_copy<float>();
}

TEST_CASE("cudaFlow.update.copy.double" * doctest::timeout(300)) {
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

TEST_CASE("cudaFlow.update.memcpy.int" * doctest::timeout(300)) {
  update_memcpy<int>();
}

TEST_CASE("cudaFlow.update.memcpy.float" * doctest::timeout(300)) {
  update_memcpy<float>();
}

TEST_CASE("cudaFlow.update.memcpy.double" * doctest::timeout(300)) {
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

TEST_CASE("cudaFlow.update.memset.int" * doctest::timeout(300)) {
  update_memset<int>();
}

TEST_CASE("cudaFlow.update.memset.float" * doctest::timeout(300)) {
  update_memset<float>();
}

TEST_CASE("cudaFlow.update.memset.double" * doctest::timeout(300)) {
  update_memset<double>();
}

// update for_each
TEST_CASE("cudaFlow.update.for_each" * doctest::timeout(300)) {

  int N = 100000;
  
  tf::cudaFlow cf;

  auto data = tf::cuda_malloc_shared<int>(N);
  
  // for each task
  auto task = cf.for_each(data, data+N, [] __device__ (int& a){ a = 100; });
  cf.offload();

  REQUIRE(cf.num_tasks() == 1);
  for(int i=0; i<N; i++) {
    REQUIRE(data[i] == 100);
  }
  
  // update for each index - this is illegal!
  //cf.update_for_each_index(
  //  task, 0, N, 1, [data] __device__ (size_t i){ data[i] = -100; }
  //);
  
  for(int i=0; i<N; i++) {
    data[i] = -100;
  }

  // update for each parameters
  cf.update_for_each(
    task, data, data+N/2, [] __device__ (int& a){ a = 100; }
  );
  cf.offload();

  REQUIRE(cf.num_tasks() == 1);
  for(int i=0; i<N/2; i++) {
    REQUIRE(data[i] == 100);
  }

  for(int i=N/2; i<N; i++) {
    REQUIRE(data[i] == -100);
  }

  tf::cuda_free(data);
}

// update for_each_index
TEST_CASE("cudaFlow.update.for_each_index" * doctest::timeout(300)) {

  int N = 100000;
  
  tf::cudaFlow cf;

  auto data = tf::cuda_malloc_shared<int>(N);
  
  // for each index
  auto task = cf.for_each_index(0, N, 1, [data] __device__ (int i){ data[i] = 100; });
  cf.offload();

  REQUIRE(cf.num_tasks() == 1);
  for(int i=0; i<N; i++) {
    REQUIRE(data[i] == 100);
  }
  
  for(int i=0; i<N; i++) {
    data[i] = -100;
  }

  // update for each
  cf.update_for_each_index(
    task, 0, N/2, 1, [data] __device__ (int i){ data[i] = 100; }
  );
  cf.offload();

  REQUIRE(cf.num_tasks() == 1);
  for(int i=0; i<N/2; i++) {
    REQUIRE(data[i] == 100);
  }

  for(int i=N/2; i<N; i++) {
    REQUIRE(data[i] == -100);
  }

  tf::cuda_free(data);
}

// update reduce
TEST_CASE("cudaFlow.update.reduce" * doctest::timeout(300)) {

  int N = 100000;
  
  tf::cudaFlow cf;

  auto data = tf::cuda_malloc_shared<int>(N);
  auto soln = tf::cuda_malloc_shared<int>(1);

  for(int i=0; i<N; i++) data[i] = -1;
  
  // reduce
  auto task = cf.reduce(
    data, data + N, soln,
    [] __device__ (int a, int b){ return a + b; }
  );
  cf.offload();

  REQUIRE(cf.num_tasks() == 1);
  REQUIRE(*soln == -N);
  
  // update reduce range
  cf.update_reduce(
    task, data, data + N/2, soln, 
    [] __device__ (int a, int b){ return a + b; }
  );
  cf.offload();
  REQUIRE(cf.num_tasks() == 1);
  REQUIRE(*soln == -3*N/2);

  tf::cuda_free(data);
  tf::cuda_free(soln);
}

// update uninitialized reduce
TEST_CASE("cudaFlow.update.uninitialized_reduce" * doctest::timeout(300)) {

  int N = 100000;
  
  tf::cudaFlow cf;

  auto data = tf::cuda_malloc_shared<int>(N);
  auto soln = tf::cuda_malloc_shared<int>(1);

  for(int i=0; i<N; i++) data[i] = -1;
  
  // uninitialized_reduce
  auto task = cf.uninitialized_reduce(
    data, data + N, soln,
    [] __device__ (int a, int b){ return a + b; }
  );
  cf.offload();

  REQUIRE(cf.num_tasks() == 1);
  REQUIRE(*soln == -N);
  
  // update reduce range
  cf.update_uninitialized_reduce(
    task, data, data + N/2, soln, 
    [] __device__ (int a, int b){ return a + b; }
  );
  cf.offload();
  REQUIRE(cf.num_tasks() == 1);
  REQUIRE(*soln == -N/2);

  tf::cuda_free(data);
  tf::cuda_free(soln);
}
