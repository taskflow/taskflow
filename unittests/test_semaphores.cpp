#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

// --------------------------------------------------------
// Testcase: Semaphores
// --------------------------------------------------------

TEST_CASE("Semaphore") {
  tf::Semaphore semaphore1;
  REQUIRE(semaphore1.count() == 0);

  tf::Semaphore semaphore2(100);
  REQUIRE(semaphore2.count() == 100);

  for(int i=0; i<200; i++) {
    if(i<100) {
      REQUIRE(semaphore2.try_acquire() == true);
    }
    else {
      REQUIRE(semaphore2.try_acquire() == false);
    }
  }

  REQUIRE(semaphore2.count() == 0);

  for(int i=0; i<200; i++) {
    semaphore2.release(2);
    REQUIRE(semaphore2.count() == 2*(i+1));
  }

  REQUIRE(tf::try_acquire(semaphore1, semaphore2) == false);
  tf::release(semaphore1, semaphore2);
  REQUIRE(semaphore1.count() == 1);
  REQUIRE(tf::try_acquire(semaphore1, semaphore2) == true);
}

// --------------------------------------------------------
// Testcase: CriticalSection
// --------------------------------------------------------

void critical_section(unsigned W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);
  tf::Semaphore semaphore(1);
  
  // We don't do large N as acquire will trigger recursive
  // corun_until which can become too deep to process
  int N = 50;
  int counter = 0;

  for(int i=0; i<N; ++i) {
    taskflow.emplace([&](tf::Runtime& rt){ 
      rt.acquire(semaphore);
      counter++; 
      rt.release(semaphore);
    });
  }

  executor.run(taskflow).wait();

  REQUIRE(counter == N);

  executor.run(taskflow);
  executor.run(taskflow);
  executor.run(taskflow);

  executor.wait_for_all();

  REQUIRE(counter == 4*N);
  REQUIRE(semaphore.count() == 1);
}

TEST_CASE("CriticalSection.1thread") {
  critical_section(1);
}

TEST_CASE("CriticalSection.2threads") {
  critical_section(2);
}

TEST_CASE("CriticalSection.3threads") {
  critical_section(3);
}

TEST_CASE("CriticalSection.7threads") {
  critical_section(7);
}

TEST_CASE("CriticalSection.11threads") {
  critical_section(11);
}

TEST_CASE("CriticalSection.16threads") {
  critical_section(16);
}

// --------------------------------------------------------
// Testcase: CriticalSectionWithAsync
// --------------------------------------------------------

void critical_section_async(unsigned W) {

  tf::Executor executor(W);
  tf::Semaphore semaphore(1);

  int N = 50;
  int counter = 0;

  for(int i=0; i<N; ++i) {
    executor.async([&](tf::Runtime& rt){ 
      rt.acquire(semaphore);
      counter++; 
      rt.release(semaphore);
    });
    
    executor.silent_async([&](tf::Runtime& rt){ 
      rt.acquire(semaphore);
      counter++; 
      rt.release(semaphore);
    });
    
    executor.dependent_async([&](tf::Runtime& rt){ 
      rt.acquire(semaphore);
      counter++; 
      rt.release(semaphore);
    });
    
    executor.silent_dependent_async([&](tf::Runtime& rt){ 
      rt.acquire(semaphore);
      counter++; 
      rt.release(semaphore);
    });
  }

  executor.wait_for_all();
  REQUIRE(counter == N*4);
  REQUIRE(semaphore.count() == 1);
}

TEST_CASE("CriticalSectionWithAsync.1thread") {
  critical_section_async(1);
}

TEST_CASE("CriticalSectionWithAsync.2threads") {
  critical_section_async(2);
}

TEST_CASE("CriticalSectionWithAsync.3threads") {
  critical_section_async(3);
}

TEST_CASE("CriticalSectionWithAsync.7threads") {
  critical_section_async(7);
}

TEST_CASE("CriticalSectionWithAsync.11threads") {
  critical_section_async(11);
}

TEST_CASE("CriticalSectionWithAsync.16threads") {
  critical_section_async(16);
}

// ----------------------------------------------------------------------------
// Testcase: MultipleCriticalSections
// ----------------------------------------------------------------------------

void multiple_critical_sections(unsigned W) {

  const size_t S = 10;
  const size_t N = 50;

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::vector<tf::Semaphore> semaphores(S);

  for(auto& semaphore : semaphores) {
    REQUIRE(semaphore.count() == 0);
    semaphore.reset(1);
    REQUIRE(semaphore.count() == 1);
  }

  int counter(0);

  for(size_t i=0; i<N; i++) {
    taskflow.emplace([&semaphores, &counter](tf::Runtime& rt){
      // critical section 1
      rt.acquire(semaphores.begin(), semaphores.end());
      counter++;
      rt.release(semaphores.begin(), semaphores.end());

      // critical section 2
      rt.acquire(semaphores[0]);
      counter++;
      rt.release(semaphores[0]);
      
      // critical section 3
      rt.acquire(semaphores.begin(), semaphores.end());
      counter++;
      rt.release(semaphores.begin(), semaphores.end());
      
      // critical section 4
      rt.acquire(semaphores[0]);
      counter++;
      rt.release(semaphores[0]);
    });
  }
  
  executor.run(taskflow).wait();
  REQUIRE(counter == N*4);
}

TEST_CASE("MultipleCriticalSections.1thread") {
  multiple_critical_sections(1);
}

TEST_CASE("MultipleCriticalSections.2threads") {
  multiple_critical_sections(2);
}

TEST_CASE("MultipleCriticalSections.3threads") {
  multiple_critical_sections(3);
}

TEST_CASE("MultipleCriticalSections.4threads") {
  multiple_critical_sections(4);
}

TEST_CASE("MultipleCriticalSections.8threads") {
  multiple_critical_sections(8);
}

TEST_CASE("MultipleCriticalSections.16threads") {
  multiple_critical_sections(16);
}

// ----------------------------------------------------------------------------
// Testcase: Linearity
// ----------------------------------------------------------------------------

void linearity(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  tf::Semaphore semaphore(1);

  int N = 50;
  int counter = 0;

  for(int i=0; i<N; i++) {

    auto f = taskflow.emplace([&](tf::Runtime& rt){ 
      rt.acquire(semaphore);
      counter++; 
    });

    auto m = taskflow.emplace([&]() {
      counter++;
    });

    auto t = taskflow.emplace([&](tf::Runtime& rt){ 
      counter++; 
      rt.release(semaphore);
    });

    f.precede(m);
    m.precede(t);
  }

  executor.run(taskflow).wait();

  REQUIRE(counter == 3*N);

}

TEST_CASE("Linearity.1thread") {
  linearity(1);
}

TEST_CASE("Linearity.2threads") {
  linearity(2);
}

TEST_CASE("Linearity.4threads") {
  linearity(4);
}

TEST_CASE("Linearity.8threads") {
  linearity(8);
}

// --------------------------------------------------------
// Testcase: Conflict Graph
// --------------------------------------------------------
/*
     A
    / \
   /   \
  B-----C
*/
void conflict_graph_1(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  tf::Semaphore AB(1), AC(1), BC(1);

  int counter(0);

  taskflow.emplace([&](tf::Runtime& rt){
    rt.acquire(AB, AC);
    counter++;
    rt.release(AB, AC);
  });
  
  taskflow.emplace([&](tf::Runtime& rt){
    rt.acquire(AB, BC);
    counter++;
    rt.release(AB, BC);
  });
  
  taskflow.emplace([&](tf::Runtime& rt){
    rt.acquire(AC, BC);
    counter++;
    rt.release(AC, BC);
  });

  executor.run(taskflow).wait();

  REQUIRE(counter == 3);

  for(size_t i=0; i<10; i++) {
    executor.run_n(taskflow, 10);
  }
  executor.wait_for_all();

  REQUIRE(counter == 303);
}

TEST_CASE("ConflictGraph1.1thread") {
  conflict_graph_1(1);
}

TEST_CASE("ConflictGraph1.2threads") {
  conflict_graph_1(2);
}

TEST_CASE("ConflictGraph1.3threads") {
  conflict_graph_1(3);
}

TEST_CASE("ConflictGraph1.4threads") {
  conflict_graph_1(4);
}

// A----C
// |    |
// B----D
void conflict_graph_2(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  tf::Semaphore AB(1), AC(1), BD(1), CD(1);

  int counter_ab(0), counter_ac(0), counter_bd(0), counter_cd(0);

  taskflow.emplace([&](tf::Runtime& rt){
    rt.acquire(AB, AC);
    counter_ab++;
    counter_ac++;
    rt.release(AB, AC);
  });
  
  taskflow.emplace([&](tf::Runtime& rt){
    rt.acquire(AB, BD);
    counter_ab++;
    counter_bd++;
    rt.release(AB, BD);
  });
  
  taskflow.emplace([&](tf::Runtime& rt){
    rt.acquire(AC, CD);
    counter_ac++;
    counter_cd++;
    rt.release(AC, CD);
  });
  
  taskflow.emplace([&](tf::Runtime& rt){
    rt.acquire(BD, CD);
    counter_bd++;
    counter_cd++;
    rt.release(BD, CD);
  });

  executor.run(taskflow).wait();

  REQUIRE(counter_ab == 2);
  REQUIRE(counter_ac == 2);
  REQUIRE(counter_bd == 2);
  REQUIRE(counter_cd == 2);

  for(size_t i=0; i<10; i++) {
    executor.run_n(taskflow, 10);
  }
  executor.wait_for_all();

  REQUIRE(counter_ab == 202);
  REQUIRE(counter_ac == 202);
  REQUIRE(counter_bd == 202);
  REQUIRE(counter_cd == 202);
}

TEST_CASE("ConflictGraph2.1thread") {
  conflict_graph_2(1);
}

TEST_CASE("ConflictGraph2.2threads") {
  conflict_graph_2(2);
}

TEST_CASE("ConflictGraph2.3threads") {
  conflict_graph_2(3);
}

TEST_CASE("ConflictGraph2.4threads") {
  conflict_graph_2(4);
}

// --------------------------------------------------------
// Testcase: Deadlock
// --------------------------------------------------------
/*
     A
    / \
   /   \
  B-----C
*/
void deadlock_1(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  tf::Semaphore AB(1), AC(1), BC(1);

  int counter(0);

  taskflow.emplace([&](tf::Runtime& rt){
    rt.acquire(AB, AC);
    counter++;
    rt.release(AB, AC);
  });
  
  taskflow.emplace([&](tf::Runtime& rt){
    rt.acquire(AC, BC);
    counter++;
    rt.release(AC, BC);
  });
  
  taskflow.emplace([&](tf::Runtime& rt){
    rt.acquire(BC, AB);
    counter++;
    rt.release(BC, AB);
  });

  executor.run(taskflow).wait();

  REQUIRE(counter == 3);

  for(size_t i=0; i<10; i++) {
    executor.run_n(taskflow, 10);
  }
  executor.wait_for_all();

  REQUIRE(counter == 303);
}

TEST_CASE("Deadlock1.1thread") {
  deadlock_1(1);
}

TEST_CASE("Deadlock1.2threads") {
  deadlock_1(2);
}

TEST_CASE("Deadlock1.3threads") {
  deadlock_1(3);
}

TEST_CASE("Deadlock1.4threads") {
  deadlock_1(4);
}

/*
     A
    / \
   /   \
  B-----C
*/
void ranged_deadlock_1(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  std::list<tf::Semaphore> semaphores;
  tf::Semaphore& AB = semaphores.emplace_back(1);
  tf::Semaphore& AC = semaphores.emplace_back(1);
  tf::Semaphore& BC = semaphores.emplace_back(1);

  auto beg = semaphores.begin();
  auto end = semaphores.end();
  int counter(0);

  REQUIRE(&(*beg) == &AB);
  REQUIRE(&(*std::next(beg, 1)) == &AC);
  REQUIRE(&(*std::next(beg, 2)) == &BC);

  taskflow.emplace([&](tf::Runtime& rt){
    rt.acquire(beg, std::next(beg, 2));
    counter++;
    rt.release(beg, std::next(beg, 2));
  });
  
  taskflow.emplace([&](tf::Runtime& rt){
    rt.acquire(std::next(beg, 1), end);
    counter++;
    rt.release(std::next(beg, 1), end);
  });
  
  taskflow.emplace([&](tf::Runtime& rt){
    rt.acquire(BC, AB);
    counter++;
    rt.release(BC, AB);
  });

  executor.run(taskflow).wait();

  REQUIRE(counter == 3);

  for(size_t i=0; i<10; i++) {
    executor.run_n(taskflow, 10);
  }
  executor.wait_for_all();

  REQUIRE(counter == 303);
}

TEST_CASE("RagnedDeadlock1.1thread") {
  ranged_deadlock_1(1);
}

TEST_CASE("RangedDeadlock1.2threads") {
  ranged_deadlock_1(2);
}

TEST_CASE("RangedDeadlock1.3threads") {
  ranged_deadlock_1(3);
}

TEST_CASE("RangedDeadlock1.4threads") {
  ranged_deadlock_1(4);
}

// ----------------------------------------------------------------------------
// Multiple Taskflows
// ----------------------------------------------------------------------------

void semaphore_by_multiple_tasks(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow1, taskflow2, taskflow3, taskflow4;
  tf::Semaphore s(1);

  int counter {0};
  size_t N = 50;

  for(size_t i=0; i<N; i++) {
    taskflow1.emplace([&](tf::Runtime& rt){
      rt.acquire(s);
      counter++;
      rt.release(s);
    });
    
    taskflow2.emplace([&](tf::Runtime& rt){
      rt.acquire(s);
      counter++;
      rt.release(s);
    });

    executor.async([&](tf::Runtime& rt){
      rt.acquire(s);
      counter++;
      rt.release(s);
    });
    
    executor.async([&](tf::Runtime& rt){
      rt.acquire(s);
      counter++;
      rt.release(s);
    });
    
    taskflow3.emplace([&](tf::Runtime& rt){
      rt.acquire(s);
      counter++;
      rt.release(s);
    });
    
    taskflow4.emplace([&](tf::Runtime& rt){
      rt.acquire(s);
      counter++;
      rt.release(s);
    });
  }
  
  executor.run(taskflow1);
  executor.run(taskflow2);
  executor.run(taskflow3);
  executor.run(taskflow4);

  executor.wait_for_all();
  REQUIRE(counter == N*6);
}

TEST_CASE("SemaphoreByMultipleTasks.1thread") {
  semaphore_by_multiple_tasks(1);
}

TEST_CASE("SemaphoreByMultipleTasks.2threads") {
  semaphore_by_multiple_tasks(2);
}

TEST_CASE("SemaphoreByMultipleTasks.3threads") {
  semaphore_by_multiple_tasks(3);
}

TEST_CASE("SemaphoreByMultipleTasks.4threads") {
  semaphore_by_multiple_tasks(4);
}

// ----------------------------------------------------------------------------
// Nonblocking Semaphores
// ----------------------------------------------------------------------------

void nonblocking_semaphore(unsigned W) {
  
  tf::Executor executor(W);
  tf::Semaphore semaphore(0);

  const size_t N = 50;

  for(size_t i=0; i<N; i++) {
    executor.async([&, i](tf::Runtime& rt){
      rt.acquire(semaphore);
    });
  }
  
  for(size_t i=0; i<N; i++) {
    executor.async([&, i](tf::Runtime& rt){
      rt.release(semaphore);
    });
  }

  executor.wait_for_all();

  REQUIRE(semaphore.count() == 0);

}

TEST_CASE("NonblockingSemaphore.1thread") {
  nonblocking_semaphore(1);
}

TEST_CASE("NonblockingSemaphore.2threads") {
  nonblocking_semaphore(2);
}

TEST_CASE("NonblockingSemaphore.3threads") {
  nonblocking_semaphore(3);
}

TEST_CASE("NonblockingSemaphore.4threads") {
  nonblocking_semaphore(4);
}

TEST_CASE("NonblockingSemaphore.5threads") {
  nonblocking_semaphore(5);
}

TEST_CASE("NonblockingSemaphore.6threads") {
  nonblocking_semaphore(6);
}

TEST_CASE("NonblockingSemaphore.7threads") {
  nonblocking_semaphore(7);
}

TEST_CASE("NonblockingSemaphore.8threads") {
  nonblocking_semaphore(8);
}





