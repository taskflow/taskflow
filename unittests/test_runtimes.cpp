#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>


// --------------------------------------------------------
// Testcase: Runtime.Schedule.ModuleTask
// --------------------------------------------------------

TEST_CASE("Runtime.Schedule.ModuleTask" * doctest::timeout(300)) {

  tf::Taskflow tf;
  int value {0};

  auto a = tf.emplace([&]() { value = -100; }).name("A");
  auto module_task = tf.placeholder().name("module");
  auto b = tf.emplace([&]() { value++; }).name("B");
  auto c = tf.emplace([&]() { value++; }).name("C");

  a.precede(module_task);
  module_task.precede(b);
  b.precede(c);

  tf::Taskflow module_flow;
  auto m1 = module_flow.emplace([&]() { value++; }).name("m1");
  auto m2 = module_flow.emplace([&]() { value++; }).name("m2");
  m1.precede(m2);

  module_task.composed_of(module_flow);

  auto entrypoint = tf.emplace([]() { return 0; }).name("entrypoint");
  auto schedule = tf.emplace([&](tf::Runtime& runtime) {
    value++;
    runtime.schedule(module_task);
  });
  entrypoint.precede(schedule, a);

  tf::Executor executor;
  executor.run(tf).wait();

  REQUIRE(value == 5);
}

// --------------------------------------------------------
// Testcase: Runtime.ExternalGraph.Simple
// --------------------------------------------------------

TEST_CASE("Runtime.ExternalGraph.Simple" * doctest::timeout(300)) {

  const size_t N = 100;

  tf::Executor executor;
  tf::Taskflow taskflow;
  
  std::vector<int> results(N, 0);
  std::vector<tf::Taskflow> graphs(N);

  for(size_t i=0; i<N; i++) {

    auto& fb = graphs[i];

    auto A = fb.emplace([&res=results[i]]()mutable{ ++res; });
    auto B = fb.emplace([&res=results[i]]()mutable{ ++res; });
    auto C = fb.emplace([&res=results[i]]()mutable{ ++res; });
    auto D = fb.emplace([&res=results[i]]()mutable{ ++res; });

    A.precede(B);
    B.precede(C);
    C.precede(D);

    taskflow.emplace([&res=results[i], &graph=graphs[i]](tf::Runtime& rt)mutable{
      rt.corun(graph);
    });
  }
  
  executor.run_n(taskflow, 100).wait();

  for(size_t i=0; i<N; i++) {
    REQUIRE(results[i] == 400);
  }

}

