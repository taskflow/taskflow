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

    taskflow.emplace([&executor, &res=results[i], &graph=graphs[i]](tf::Runtime&) mutable {
      executor.corun(graph);
    });
  }
  
  executor.run_n(taskflow, 100).wait();

  for(size_t i=0; i<N; i++) {
    REQUIRE(results[i] == 400);
  }

}


// --------------------------------------------------------------------------------------
// Fibonacci
// --------------------------------------------------------------------------------------

size_t fibonacci(size_t N, tf::Runtime& rt) {

  if (N < 2) {
    return N; 
  }
  
  size_t res1, res2;

  rt.silent_async([N, &res1](tf::Runtime& rt1){ res1 = fibonacci(N-1, rt1); });
  
  // tail optimization
  res2 = fibonacci(N-2, rt);

  // use corun to avoid blocking the worker from waiting the two children tasks to finish
  rt.corun();

  return res1 + res2;
}

size_t fibonacci(size_t T, size_t N) {
  tf::Executor executor(T);
  size_t res;
  executor.async([N, &res](tf::Runtime& rt){ res = fibonacci(N, rt); }).get();
  return res;
}

TEST_CASE("Runtime.Fibonacci.1thread" * doctest::timeout(250)) {
  REQUIRE(fibonacci(1, 25) == 75025);
}

TEST_CASE("Runtime.Fibonacci.2threads" * doctest::timeout(250)) {
  REQUIRE(fibonacci(2, 25) == 75025);
}

TEST_CASE("Runtime.Fibonacci.3threads" * doctest::timeout(250)) {
  REQUIRE(fibonacci(3, 25) == 75025);
}

TEST_CASE("Runtime.Fibonacci.4threads" * doctest::timeout(250)) {
  REQUIRE(fibonacci(4, 25) == 75025);
}

// --------------------------------------------------------------------------------------
// Fibonacci
// --------------------------------------------------------------------------------------

size_t fibonacci_swapped(size_t N, tf::Runtime& rt) {

  if (N < 2) {
    return N; 
  }
  
  size_t res1, res2;
  
  // tail optimization
  res1 = fibonacci_swapped(N-1, rt);

  rt.silent_async([N, &res2](tf::Runtime& rt2){ res2 = fibonacci_swapped(N-2, rt2); });

  // use corun to avoid blocking the worker from waiting the two children tasks to finish
  rt.corun();

  return res1 + res2;
}

size_t fibonacci_swapped(size_t T, size_t N) {
  tf::Executor executor(T);
  size_t res;
  executor.async([N, &res](tf::Runtime& rt){ res = fibonacci_swapped(N, rt); }).get();
  return res;
}

TEST_CASE("Runtime.Fibonacci.1thread" * doctest::timeout(250)) {
  REQUIRE(fibonacci_swapped(1, 25) == 75025);
}

TEST_CASE("Runtime.Fibonacci.2threads" * doctest::timeout(250)) {
  REQUIRE(fibonacci_swapped(2, 25) == 75025);
}

TEST_CASE("Runtime.Fibonacci.3threads" * doctest::timeout(250)) {
  REQUIRE(fibonacci_swapped(3, 25) == 75025);
}

TEST_CASE("Runtime.Fibonacci.4threads" * doctest::timeout(250)) {
  REQUIRE(fibonacci_swapped(4, 25) == 75025);
}

// --------------------------------------------------------
// Testcase: Runtime.Cancel
// --------------------------------------------------------

TEST_CASE("Runtime.Cancel" * doctest::timeout(300)) {

  std::atomic<bool> reached(false);
  std::atomic<bool> cancelled(false);

  tf::Executor executor;
  tf::Taskflow taskflow;
  taskflow.emplace([&](tf::Runtime &rt) {
    reached = true;
    while (!cancelled) {
      std::this_thread::yield();
      if (rt.is_cancelled()) {
        cancelled = true;
        break;
      }
    }
  });

  auto future = executor.run(std::move(taskflow));
  
  // Need to wait until we run the runtime task or the cancel may immediately
  // cancel the entire taskflow before the runtime task starts.
  while(!reached);
  future.cancel();
  future.get();

  REQUIRE(cancelled == true);
}

// ------------------------------------------------------------------------------------------------
// implicit_join
// ------------------------------------------------------------------------------------------------

void implicit_join(size_t num_threads) {

  tf::Executor executor(num_threads);
  tf::Taskflow taskflow;
  std::atomic<size_t> counter;

  auto A = taskflow.emplace([&](){ counter.store(0, std::memory_order_relaxed); });
  auto B = taskflow.emplace([&](tf::Runtime& rt){
    // explicitly synchronize all async tasks
    for(size_t i=0; i<100; i++) {
      rt.async([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
    }
    for(size_t i=0; i<100; i++) {
      rt.silent_async([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
    }
    rt.corun();
    REQUIRE(counter.load(std::memory_order_relaxed) == 200);
    
    // implicitly synchronize all async tasks
    for(size_t i=0; i<100; i++) {
      rt.async([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
    }
    for(size_t i=0; i<100; i++) {
      rt.silent_async([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
    }
    // implicit synchronization at the end of the runtime scope
  });
  auto C = taskflow.emplace([&](){
    REQUIRE(counter.load(std::memory_order_relaxed) == 400);
  });

  B.succeed(A).precede(C);

  executor.run(taskflow).wait();
}


TEST_CASE("Runtime.ImplicitJoin.1thread" * doctest::timeout(300)) {
  implicit_join(1);
}

TEST_CASE("Runtime.ImplicitJoin.2threads" * doctest::timeout(300)) {
  implicit_join(2);
}

TEST_CASE("Runtime.ImplicitJoin.3threads" * doctest::timeout(300)) {
  implicit_join(3);
}

TEST_CASE("Runtime.ImplicitJoin.4threads" * doctest::timeout(300)) {
  implicit_join(4);
}

TEST_CASE("Runtime.ImplicitJoin.5threads" * doctest::timeout(300)) {
  implicit_join(5);
}

TEST_CASE("Runtime.ImplicitJoin.6threads" * doctest::timeout(300)) {
  implicit_join(6);
}

TEST_CASE("Runtime.ImplicitJoin.7threads" * doctest::timeout(300)) {
  implicit_join(7);
}

TEST_CASE("Runtime.ImplicitJoin.8threads" * doctest::timeout(300)) {
  implicit_join(8);
}

// ------------------------------------------------------------------------------------------------
// implicit_join
// ------------------------------------------------------------------------------------------------

void recursive_implicit_join(size_t num_threads) {

  const size_t N1 = 100;
  const size_t N2 = 10;

  tf::Executor executor(num_threads);
  tf::Taskflow taskflow;
  std::atomic<size_t> counter;

  auto A = taskflow.emplace([&](){ counter.store(0, std::memory_order_relaxed); });
  auto B = taskflow.emplace([&](tf::Runtime& rt1){
    // implicitly synchronize all async tasks
    for(size_t i=0; i<N1; i++) {
      rt1.async([&](tf::Runtime& rt2){ 
        counter.fetch_add(1, std::memory_order_relaxed); 
        // each of the N1 async task spawns N2 async task
        for(size_t j=0; j<N2; j++) {
          rt2.silent_async([&](){
            counter.fetch_add(1, std::memory_order_relaxed); 
          });
        }
      });
    }
    for(size_t i=0; i<N1; i++) {
      rt1.silent_async([&](tf::Runtime& rt2){ 
        counter.fetch_add(1, std::memory_order_relaxed); 
        // each of the N1 async task spawns N2 async task
        for(size_t j=0; j<N2; j++) {
          rt2.async([&](){
            counter.fetch_add(1, std::memory_order_relaxed); 
          });
        }
      });
    }
    // implicit synchronization at the end of the runtime scope
  });
  auto C = taskflow.emplace([&](){
    REQUIRE(counter.load(std::memory_order_relaxed) == 2*(N1+N1*N2));
  });

  B.succeed(A).precede(C);

  executor.run(taskflow).wait();
}

TEST_CASE("Runtime.RecursiveImplicitJoin.1thread" * doctest::timeout(300)) {
  recursive_implicit_join(1);
}

TEST_CASE("Runtime.RecursiveImplicitJoin.2threads" * doctest::timeout(300)) {
  recursive_implicit_join(2);
}

TEST_CASE("Runtime.RecursiveImplicitJoin.3threads" * doctest::timeout(300)) {
  recursive_implicit_join(3);
}

TEST_CASE("Runtime.RecursiveImplicitJoin.4threads" * doctest::timeout(300)) {
  recursive_implicit_join(4);
}

TEST_CASE("Runtime.RecursiveImplicitJoin.5threads" * doctest::timeout(300)) {
  recursive_implicit_join(5);
}

TEST_CASE("Runtime.RecursiveImplicitJoin.6threads" * doctest::timeout(300)) {
  recursive_implicit_join(6);
}

TEST_CASE("Runtime.RecursiveImplicitJoin.7threads" * doctest::timeout(300)) {
  recursive_implicit_join(7);
}

TEST_CASE("Runtime.RecursiveImplicitJoin.8threads" * doctest::timeout(300)) {
  recursive_implicit_join(8);
}








