#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taskflow/sycl/syclflow.hpp>

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

  task1.name("task1");
  task2.name("task2");

  REQUIRE(task1.name() == "task1");
  REQUIRE(task2.name() == "task2");

  task1.for_each_successor([](tf::syclTask task){
    REQUIRE(task.name() == "task2");
  });

  task2.for_each_dependent([](tf::syclTask task){
    REQUIRE(task.name() == "task1");
  });
}

// USM shared memory
TEST_CASE("syclFlow.USM.shared" * doctest::timeout(300)) {

  sycl::queue queue;
  tf::syclFlow flow(queue);

  for(size_t N=1; N<=1000000; N = (N + ::rand() % 17) << 1) {

    flow.clear();

    REQUIRE(flow.empty());

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

    REQUIRE(flow.empty());

    int* dptr = sycl::malloc_device<int>(N, queue);
    std::vector<int> data(N, -100);

    auto d2h = flow.memcpy(data.data(), dptr, N*sizeof(int));
    auto h2d = flow.copy(dptr, data.data(), N);
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

// syclFlow.parallel_fills
TEST_CASE("syclFlow.parallel_fills" * doctest::timeout(300)) {

  sycl::queue queue;
  tf::syclFlow syclflow(queue);

  for(size_t N=1; N<=1000000; N = (N + ::rand() % 17) << 1) {

    syclflow.clear();

    REQUIRE(syclflow.empty());

    int* dptr = sycl::malloc_shared<int>(N, queue);

    for(size_t i=0; i<N; i++) {
      dptr[i] = 999;
    }

    size_t R = N;

    while(R != 0) {

      size_t count = R % 1024 + 1;

      if(count > R) count = R;

      auto plus = syclflow.parallel_for(sycl::range<1>(count),
        [ptr=(dptr+N-R)] (sycl::id<1> id) {
          ptr[id] += 2;
        }
      );
      auto fill = syclflow.fill(dptr + N - R, -7, count);

      fill.precede(plus);

      R -= count;
    }

    syclflow.offload();

    for(size_t i=0; i<N; i++) {
      REQUIRE(dptr[i] == -5);
    }

    sycl::free(dptr, queue);
  }
}

// syclFlow.parallel_memsets
TEST_CASE("syclFlow.parallel_memsets" * doctest::timeout(300)) {

  sycl::queue queue;
  tf::syclFlow syclflow(queue);

  for(size_t N=1; N<=1000000; N = (N + ::rand() % 17) << 1) {

    syclflow.clear();

    REQUIRE(syclflow.empty());

    int* dptr = sycl::malloc_shared<int>(N, queue);

    for(size_t i=0; i<N; i++) {
      dptr[i] = 999;
    }

    size_t R = N;

    while(R != 0) {

      size_t count = R % 1024 + 1;

      if(count > R) count = R;

      auto plus = syclflow.parallel_for(sycl::range<1>(count),
        [ptr=(dptr+N-R)] (sycl::id<1> id) {
          ptr[id] += 2;
        }
      );
      auto mems = syclflow.memset(dptr + N - R, -1, sizeof(int)*count);

      mems.precede(plus);

      R -= count;
    }

    syclflow.offload();

    for(size_t i=0; i<N; i++) {
      REQUIRE(dptr[i] == 1);
    }

    sycl::free(dptr, queue);
  }
}

// syclFlow.parallel_copies
TEST_CASE("syclFlow.parallel_copies" * doctest::timeout(300)) {

  sycl::queue queue;
  tf::syclFlow syclflow(queue);

  for(size_t N=1; N<=1000000; N = (N + ::rand() % 17) << 1) {

    syclflow.clear();

    REQUIRE(syclflow.empty());

    int* dptr = sycl::malloc_device<int>(N, queue);
    std::vector<int> host(N, -1);

    size_t R = N;

    while(R != 0) {

      size_t count = R % 1024 + 1;

      if(count > R) count = R;

      auto plus = syclflow.parallel_for(sycl::range<1>(count),
        [ptr=(dptr+N-R)] (sycl::id<1> id) {
          ptr[id] += 2;
        }
      );
      auto d2hc = syclflow.copy(host.data() + N - R, dptr + N - R, count);
      auto h2dc = syclflow.copy(dptr + N - R, host.data() + N - R, count);
      auto fill = syclflow.fill(dptr + N - R, -7, count);

      h2dc.precede(fill);
      fill.precede(plus);
      plus.precede(d2hc);

      R -= count;
    }

    syclflow.offload();

    for(size_t i=0; i<N; i++) {
      REQUIRE(host[i] == -5);
    }

    sycl::free(dptr, queue);
  }

}

// syclFlow.condition
TEST_CASE("syclFlow.condition" * doctest::timeout(300)) {

  size_t N = 10000;

  sycl::queue queue;

  tf::Executor executor;
  tf::Taskflow taskflow;

  int* dptr = sycl::malloc_shared<int>(N, queue);

  auto init = taskflow.emplace([&](){
    for(size_t i=0; i<N; i++) {
      dptr[i] = -2;
    }
  });

  auto sycl = taskflow.emplace_on([&](tf::syclFlow& sf) {
    sf.parallel_for(sycl::range<1>(N),
      [dptr] (sycl::id<1> id) {
        dptr[id] += 2;
      }
    );
  }, queue);

  auto cond = taskflow.emplace([r=5]() mutable {
    return (r-- > 0) ? 0 : 1;
  });

  init.precede(sycl);
  sycl.precede(cond);
  cond.precede(sycl);

  executor.run(taskflow).wait();

  for(size_t i=0; i<N; i++) {
    REQUIRE(dptr[i] == 10);
  }

  sycl::free(dptr, queue);
}

// syclFlow.run_n
TEST_CASE("syclFlow.run_n" * doctest::timeout(300)) {

  size_t N = 10000;

  sycl::queue queue;

  tf::Executor executor;
  tf::Taskflow taskflow;

  int* dptr = sycl::malloc_shared<int>(N, queue);

  for(size_t i=0; i<N; i++) {
    dptr[i] = 0;
  }

  auto init = taskflow.emplace([](){});

  auto sycl = taskflow.emplace_on([&](tf::syclFlow& sf) {
    sf.parallel_for(sycl::range<1>(N),
      [dptr] (sycl::id<1> id) {
        dptr[id] += 2;
      }
    );
  }, queue);

  init.precede(sycl);

  executor.run_n(taskflow, 10).wait();

  for(size_t i=0; i<N; i++) {
    REQUIRE(dptr[i] == 20);
  }

  sycl::free(dptr, queue);
}
