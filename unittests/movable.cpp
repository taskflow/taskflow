#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taskflow/taskflow.hpp>

struct CountOnDestruction {

  CountOnDestruction(const CountOnDestruction& rhs) : counter {rhs.counter} {
    rhs.counter = nullptr;
  }

  CountOnDestruction(CountOnDestruction&& rhs) : counter{rhs.counter} {
    rhs.counter = nullptr;
  }

  CountOnDestruction(std::atomic<int>& c) : counter {&c} {}

  ~CountOnDestruction() { 
    if(counter) {
      //std::cout << "destroying\n";
      counter->fetch_add(1, std::memory_order_relaxed); 
    }
  }

  mutable std::atomic<int>* counter {nullptr};
};

TEST_CASE("moved_run") {

  int N = 10000;
  
  std::atomic<int> counter {0};

  tf::Taskflow taskflow;
  
  auto make_taskflow = [&](){  
    for(int i=0; i<N; i++) {
      taskflow.emplace([&, c=CountOnDestruction{counter}](){
        counter.fetch_add(1, std::memory_order_relaxed);
      });
    }
  };
  
  // run the moved taskflow
  make_taskflow();
  tf::Executor().run_until(
    std::move(taskflow), 
    [repeat=2]() mutable { return repeat-- == 0; }, 
    [](){}
  ).wait();

  REQUIRE(taskflow.num_tasks() == 0);
  REQUIRE(counter == 3*N);

  // run the original empty taskflow
  tf::Executor().run(taskflow).wait();
  REQUIRE(counter == 3*N);

  // remake the taskflow and run it again
  make_taskflow();
  REQUIRE(taskflow.num_tasks() == N);
  tf::Executor().run(taskflow).wait();
  REQUIRE(counter == 4*N);
  REQUIRE(taskflow.num_tasks() == N);

  // run the moved taskflow
  tf::Executor().run(std::move(taskflow)).wait();
  REQUIRE(counter == 6*N);
  REQUIRE(taskflow.num_tasks() == 0);

  // run the moved empty taskflow
  tf::Executor().run(std::move(taskflow)).wait();
  REQUIRE(counter == 6*N);
  REQUIRE(taskflow.num_tasks() == 0);

  // remake the taskflow and run it with moved ownership
  make_taskflow();
  REQUIRE(taskflow.num_tasks() == N);
  tf::Executor().run_n(std::move(taskflow), 3).wait();
  REQUIRE(counter == 10*N);
  REQUIRE(taskflow.num_tasks() == 0);

  // run the moved empty taskflow with callable
  tf::Executor().run(std::move(taskflow), [&](){ 
    counter.fetch_add(N, std::memory_order_relaxed);
  }).wait();
  REQUIRE(counter == 11*N);
  REQUIRE(taskflow.num_tasks() == 0);

  // remake the taskflow and run it with moved ownership
  make_taskflow();
  tf::Executor().run(std::move(taskflow), [&](){ 
    counter.fetch_add(N, std::memory_order_relaxed);
  }).wait();
  REQUIRE(counter == 14*N);
  REQUIRE(taskflow.num_tasks() == 0);
} 


