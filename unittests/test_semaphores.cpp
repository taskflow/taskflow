#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

// --------------------------------------------------------
// Testcase: CriticalSection
// --------------------------------------------------------

void critical_section(size_t W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);
  tf::Semaphore sema(1);
  
  REQUIRE(sema.value() == 1);
  REQUIRE(sema.max_value() == 1);

  int N = 1000;
  int counter = 0;

  for(int i=0; i<N; ++i) {
    taskflow.emplace([&](){ counter++; })
            .name(std::to_string(i))
            .acquire(sema)
            .release(sema);
  }

  executor.run(taskflow).wait();

  REQUIRE(counter == N);

  executor.run(taskflow);
  executor.run(taskflow);
  executor.run(taskflow);

  executor.wait_for_all();

  REQUIRE(counter == 4*N);
  REQUIRE(sema.value() == 1);
  REQUIRE(sema.max_value() == 1);
}

TEST_CASE("Semaphore.CriticalSection.1thread" * doctest::timeout(300)) {
  critical_section(1);
}

TEST_CASE("Semaphore.CriticalSection.2threads" * doctest::timeout(300)) {
  critical_section(2);
}

TEST_CASE("Semaphore.CriticalSection.3threads" * doctest::timeout(300)) {
  critical_section(3);
}

TEST_CASE("Semaphore.CriticalSection.7threads" * doctest::timeout(300)) {
  critical_section(7);
}

TEST_CASE("Semaphore.CriticalSection.11threads" * doctest::timeout(300)) {
  critical_section(11);
}

TEST_CASE("Semaphore.CriticalSection.16threads" * doctest::timeout(300)) {
  critical_section(16);
}

// --------------------------------------------------------
// Testcase: OverlappedSemaphore
// --------------------------------------------------------

void overlapped_semaphores(size_t W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  tf::Semaphore semaphore1(1);
  tf::Semaphore semaphore4(4);

  int N = 1000;
  int counter = 0;

  for(int i=0; i<N; i++) {
    auto task = taskflow.emplace([&](){ counter++; });
    task.acquire(semaphore1);
    task.acquire(semaphore4);
    task.release(semaphore1);
    task.release(semaphore4);
  }

  executor.run(taskflow).wait();

  REQUIRE(counter == N);
  REQUIRE(semaphore1.value() == 1);
  REQUIRE(semaphore4.value() == 4);
}

TEST_CASE("Semaphore.Overlap.1thread" * doctest::timeout(300)) {
  overlapped_semaphores(1);
}

TEST_CASE("Semaphore.Overlap.2threads" * doctest::timeout(300)) {
  overlapped_semaphores(2);
}

TEST_CASE("Semaphore.Overlap.4threads" * doctest::timeout(300)) {
  overlapped_semaphores(4);
}

TEST_CASE("Semaphore.Overlap.8threads" * doctest::timeout(300)) {
  overlapped_semaphores(8);
}

// --------------------------------------------------------
// Testcase: Conflict Graph
// --------------------------------------------------------

void conflict_graph(size_t W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  tf::Semaphore conflict_AB(1);
  tf::Semaphore conflict_AC(1);

  int counter {0};
  std::mutex mutex;

  tf::Task A = taskflow.emplace([&](){ counter++; });

  // B and C can run together
  tf::Task B = taskflow.emplace([&](){
    std::lock_guard<std::mutex> lock(mutex);
    counter++;
  });
  tf::Task C = taskflow.emplace([&](){
    std::lock_guard<std::mutex> lock(mutex);
    counter++;
  });

  // describe the conflict between A and B
  A.acquire(conflict_AB).release(conflict_AB);
  B.acquire(conflict_AB).release(conflict_AB);

  // describe the conflict between A and C
  A.acquire(conflict_AC).release(conflict_AC);
  C.acquire(conflict_AC).release(conflict_AC);

  executor.run(taskflow).wait();

  REQUIRE(counter == 3);

  for(size_t i=0; i<10; i++) {
    executor.run_n(taskflow, 10);
  }
  executor.wait_for_all();

  REQUIRE(counter == 303);
}

TEST_CASE("Semaphore.ConflictGraph.1thread" * doctest::timeout(300)) {
  conflict_graph(1);
}

TEST_CASE("Semaphore.ConflictGraph.2threads" * doctest::timeout(300)) {
  conflict_graph(2);
}

TEST_CASE("Semaphore.ConflictGraph.3threads" * doctest::timeout(300)) {
  conflict_graph(3);
}

TEST_CASE("Semaphore.ConflictGraph.4threads" * doctest::timeout(300)) {
  conflict_graph(4);
}

// ----------------------------------------------------------------------------
// Module Task 
// ----------------------------------------------------------------------------

void semaphore_in_module(unsigned W) {
  
  tf::Taskflow taskflow1;
  tf::Taskflow taskflow2;
  tf::Executor executor(W);
  tf::Semaphore semaphore(2);

  size_t N = 1024;
  size_t counter {0};  

  for(size_t i=0; i<N; i=i+1){
    auto t = taskflow1.emplace([&](){ counter++; });
    t.acquire(semaphore);
    t.release(semaphore);
  }

  auto m = taskflow2.composed_of(taskflow1);
  m.acquire(semaphore);
  m.release(semaphore);

  executor.run(taskflow2).get();
  REQUIRE(counter == N);
}

TEST_CASE("Semaphore.Module.1thread" * doctest::timeout(300)) {
  semaphore_in_module(1);
}

TEST_CASE("Semaphore.Module.2threads" * doctest::timeout(300)) {
  semaphore_in_module(2);
}

TEST_CASE("Semaphore.Module.3threads" * doctest::timeout(300)) {
  semaphore_in_module(3);
}

TEST_CASE("Semaphore.Module.4threads" * doctest::timeout(300)) {
  semaphore_in_module(4);
}

// ----------------------------------------------------------------------------
// Semahpores in Module Task 
// ----------------------------------------------------------------------------

void semaphores_in_module(unsigned W) {
  
  tf::Taskflow taskflow1;
  tf::Taskflow taskflow2;
  tf::Executor executor(W);
  std::vector<tf::Semaphore> semaphores(10);

  for(auto& sema : semaphores) {
    REQUIRE(sema.value() == 0);
    REQUIRE(sema.max_value() == 0);
    sema.reset(2);
    REQUIRE(sema.value() == 2);
    REQUIRE(sema.max_value() == 2);
  }

  size_t N = 1024;
  size_t counter {0};  

  for(size_t i=0; i<N; i=i+1){
    auto t = taskflow1.emplace([&](){ counter++; });
    t.acquire(semaphores.begin(), semaphores.end());
    t.release(semaphores.begin(), semaphores.end());
  }

  auto m = taskflow2.composed_of(taskflow1);

  m.acquire(semaphores.begin(), semaphores.end());
  m.release(semaphores.begin(), semaphores.end()); 

  executor.run(taskflow2).get();
  REQUIRE(counter == N);
}

TEST_CASE("Semaphores.Module.1thread" * doctest::timeout(300)) {
  semaphores_in_module(1);
}

TEST_CASE("Semaphores.Module.2threads" * doctest::timeout(300)) {
  semaphores_in_module(2);
}

TEST_CASE("Semaphores.Module.3threads" * doctest::timeout(300)) {
  semaphores_in_module(3);
}

TEST_CASE("Semaphores.Module.4threads" * doctest::timeout(300)) {
  semaphores_in_module(4);
}

TEST_CASE("Semaphores.Module.5threads" * doctest::timeout(300)) {
  semaphores_in_module(5);
}

TEST_CASE("Semaphores.Module.6threads" * doctest::timeout(300)) {
  semaphores_in_module(6);
}

TEST_CASE("Semaphores.Module.7threads" * doctest::timeout(300)) {
  semaphores_in_module(7);
}

TEST_CASE("Semaphores.Module.8threads" * doctest::timeout(300)) {
  semaphores_in_module(8);
}

// ----------------------------------------------------------------------------
// Linear Chain
// ----------------------------------------------------------------------------

void linear_chain(unsigned W) {

  const size_t L = 10000;

  std::vector<tf::CachelineAligned<size_t>> counters(L);
  std::vector<tf::Semaphore> semaphores(L);

  for(auto& semaphore : semaphores) {
    semaphore.reset(1);
  }
  
  tf::Executor executor(W);
  tf::Taskflow taskflow;

  for(size_t i=0; i<L; i++) {
    auto t = taskflow.emplace([i, &counters](){
      if(i) {
        counters[i-1].data++;
      }
      counters[i].data++;
    });
    
    if(i) {
      t.acquire(semaphores[i-1])
       .release(semaphores[i-1]);
    }
    t.acquire(semaphores[i])
     .release(semaphores[i]);
  }

  executor.run(taskflow).get();

  counters.back().data++;

  for(auto& c : counters) {
    REQUIRE(c.data == 2);
  }
}

TEST_CASE("Semaphore.LinearChain.1thread" * doctest::timeout(300)) {
  linear_chain(1);
}

TEST_CASE("Semaphore.LinearChain.2threads" * doctest::timeout(300)) {
  linear_chain(2);
}

TEST_CASE("Semaphore.LinearChain.3threads" * doctest::timeout(300)) {
  linear_chain(3);
}

TEST_CASE("Semaphore.LinearChain.4threads" * doctest::timeout(300)) {
  linear_chain(4);
}

TEST_CASE("Semaphore.LinearChain.5threads" * doctest::timeout(300)) {
  linear_chain(5);
}

TEST_CASE("Semaphore.LinearChain.6threads" * doctest::timeout(300)) {
  linear_chain(6);
}

TEST_CASE("Semaphore.LinearChain.7threads" * doctest::timeout(300)) {
  linear_chain(7);
}

TEST_CASE("Semaphore.LinearChain.8threads" * doctest::timeout(300)) {
  linear_chain(8);
}

// ----------------------------------------------------------------------------

/*
void deadlock(unsigned W) {

  tf::Executor exec(W);
  tf::Taskflow flow;
  tf::Semaphore semaphore(8);

  for(size_t i = 0; i < 5; ++i) {

    tf::Task commonTask = flow.emplace([i]() {
      std::string str;
      str.append("Common task for i = ").append(std::to_string(i)).append("\n");
      std::cout << str;
    }).name(std::string("Common i = ").append(std::to_string(i)));

    for(size_t j = 0; j < 8; ++j) {
      tf::Task preTask = flow.emplace([i, j]() {
        std::string str;
        str.append("Pre task for i = ").append(std::to_string(i)).append(", j = ").append(std::to_string(j)).append("\n");
        std::cout << str;
      }).name(std::string("Pre task i = ").append(std::to_string(i)).append(", j = ").append(std::to_string(j)));

      tf::Task postTask = flow.emplace([i, j]() {
        std::string str;
        str.append("Post task for i = ").append(std::to_string(i)).append(", j = ").append(std::to_string(j)).append("\n");
        std::cout << str;
      }).name(std::string("Post task i = ").append(std::to_string(i)).append(", j = ").append(std::to_string(j)));

      preTask.precede(commonTask);
      postTask.succeed(commonTask);

      preTask.acquire(semaphore);
      postTask.release(semaphore);
    }
  }

  flow.dump(std::cout);

  std::cout << "Start!" << std::endl;
  exec.run(flow).wait();
  std::cout << "End!" << std::endl;
}

TEST_CASE("Semaphore.Deadlock.8threads" * doctest::timeout(300)) {
  deadlock(8);
}
*/
