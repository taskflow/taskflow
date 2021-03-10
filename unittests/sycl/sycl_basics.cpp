#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taskflow/syclflow.hpp>

// task creation
TEST_CASE("syclFlow.task" * doctest::timeout(300)) {
  
  sycl::queue queue;
  tf::syclFlow flow(queue);
  
  REQUIRE(flow.empty());  
  REQUIRE(flow.num_tasks() == 0);

  auto task1 = flow.single_task([](){});

  REQUIRE(flow.empty() == false);
  REQUIRE(flow.num_tasks() == 1);

  REQUIRE(task1.num_successors() == 0);
  REQUIRE(task1.num_dependents() == 0);
  
  auto task2 = flow.single_task([](){});
  
  REQUIRE(flow.num_tasks() == 2);

  task1.precede(task2);

  REQUIRE(task1.num_successors() == 1);
  REQUIRE(task1.num_dependents() == 0);
  REQUIRE(task2.num_successors() == 0);
  REQUIRE(task2.num_dependents() == 1);
}

// USM shared memory
TEST_CASE("syclFlow.USM.shared" * doctest::timeout(300)) {
    
  sycl::queue queue;
  tf::syclFlow flow(queue);

  for(size_t N=1; N<=1000000; N = (N + ::rand() % 17) << 1) {

    flow.clear();

    int* ptr = sycl::malloc_shared<int>(N, queue);
    
    auto plus = flow.parallel_for(sycl::range<1>(sycl::range<1>(N)),
      [=](sycl::id<1> id){
        ptr[id] += 2;
      }
    );

    auto fill = flow.fill(ptr, 100, N);

    fill.precede(plus);

    flow.offload_n(3);

    for(size_t i=0; i<N; i++) {
      REQUIRE(ptr[i] == 102);
    }
    
    sycl::free(ptr, queue);
  }
}

// USM device memory
TEST_CASE("syclFlow.USM.device" * doctest::timeout(300)) {
    
  sycl::queue queue;
  tf::syclFlow flow(queue);

  for(size_t N=1; N<=1000000; N = (N + ::rand() % 17) << 1) {

    flow.clear();

    int* dptr = sycl::malloc_device<int>(N, queue);
    std::vector<int> data(N, -100);

    auto d2h = flow.memcpy(data.data(), dptr, N*sizeof(int));
    auto h2d = flow.memcpy(dptr, data.data(), N*sizeof(int));
    auto pf = flow.parallel_for(sycl::range<1>(sycl::range<1>(N)),
      [=](sycl::id<1> id){
        dptr[id] += 2;
      }
    );

    pf.succeed(h2d)
      .precede(d2h);

    flow.offload();

    for(size_t i=0; i<N; i++) {
      REQUIRE(data[i] == -98);
    }
    
    sycl::free(dptr, queue);
  }
}

