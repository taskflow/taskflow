#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taskflow/syclflow.hpp>

// ----------------------------------------------------------------------------
// rebind algorithms
// ----------------------------------------------------------------------------

TEST_CASE("syclFlowCapturer.rebind.algorithms") {

  sycl::queue queue;

  tf::syclFlow syclflow(queue);

  auto data = sycl::malloc_shared<int>(10000, queue);
  auto res = sycl::malloc_shared<int>(1, queue);

  auto task = syclflow.for_each(
    data, data+10000, [](int& i) { i = 10; }
  );

  syclflow.offload();

  for(int i=0; i<10000; i++) {
    REQUIRE(data[i] == 10);
  }
  REQUIRE(syclflow.num_tasks() == 1);

  // rebind to single task
  syclflow.single_task(task, [=]  () {*data = 2;});

  syclflow.offload();

  REQUIRE(*data == 2);
  for(int i=1; i<10000; i++) {
    REQUIRE(data[i] == 10);
  }
  REQUIRE(syclflow.num_tasks() == 1);

  // rebind to for each index
  syclflow.for_each_index(
    task, 0, 10000, 1, [=] (int i) { data[i] = -23; }
  );

  syclflow.offload();


  for(int i=0; i<10000; i++) {
    REQUIRE(data[i] == -23);
  }
  REQUIRE(syclflow.num_tasks() == 1);

  // rebind to reduce
  *res = 10;
  syclflow.reduce(task, data, data + 10000, res,
    [](int a, int b){ return a + b; }
  );

  syclflow.offload();

  REQUIRE(*res == -229990);
  REQUIRE(syclflow.num_tasks() == 1);

  // rebind to uninitialized reduce
  syclflow.uninitialized_reduce(task, data, data + 10000, res,
    [](int a, int b){ return a + b; }
  );

  syclflow.offload();

  REQUIRE(*res == -230000);
  REQUIRE(syclflow.num_tasks() == 1);

  // rebind to single task
  syclflow.single_task(task, [res](){ *res = 999; });
  REQUIRE(*res == -230000);

  syclflow.offload();
  REQUIRE(*res == 999);
  REQUIRE(syclflow.num_tasks() == 1);

  // rebind to on
  syclflow.on(task, [res] (sycl::handler& handler) {
    handler.single_task([=](){ *res = 1000; });
  });

  syclflow.offload();
  REQUIRE(*res == 1000);

  // clear the syclflow
  syclflow.clear();
  REQUIRE(syclflow.num_tasks() == 0);

  syclflow.offload();

  REQUIRE(*res == 1000);
  for(int i=0; i<10000; i++) {
    REQUIRE(data[i] == -23);
  }

  return;

  // clear the memory
  sycl::free(data, queue);
  sycl::free(res, queue);
}

