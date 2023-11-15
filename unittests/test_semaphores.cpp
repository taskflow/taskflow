#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <chrono>

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/semaphore_guard.hpp>

// --------------------------------------------------------
// Testcase: CriticalSection
// --------------------------------------------------------

void critical_section(size_t W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);
  tf::CriticalSection section(1);

  int N = 1000;
  int counter = 0;

  for(int i=0; i<N; ++i) {
    tf::Task task = taskflow.emplace([&](){ counter++; })
                            .name(std::to_string(i));
    section.add(task);
  }

  executor.run(taskflow).wait();

  REQUIRE(counter == N);

  executor.run(taskflow);
  executor.run(taskflow);
  executor.run(taskflow);

  executor.wait_for_all();

  REQUIRE(counter == 4*N);
  REQUIRE(section.count() == 1);
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
// Testcase: Semaphore
// --------------------------------------------------------

void semaphore(size_t W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  tf::Semaphore semaphore(1);

  int N = 1000;
  int counter = 0;

  for(int i=0; i<N; i++) {
    auto f = taskflow.emplace([&](){ counter++; });
    auto t = taskflow.emplace([&](){ counter++; });
    f.precede(t);
    f.acquire(semaphore);
    t.release(semaphore);
  }

  executor.run(taskflow).wait();

  REQUIRE(counter == 2*N);

}

TEST_CASE("Semaphore.1thread") {
  semaphore(1);
}

TEST_CASE("Semaphore.2threads") {
  semaphore(2);
}

TEST_CASE("Semaphore.4threads") {
  semaphore(4);
}

TEST_CASE("Semaphore.8threads") {
  semaphore(8);
}

// --------------------------------------------------------
// Testcase: OverlappedSemaphore
// --------------------------------------------------------

void overlapped_semaphore(size_t W) {

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
  REQUIRE(semaphore1.count() == 1);
  REQUIRE(semaphore4.count() == 4);
}

TEST_CASE("OverlappedSemaphore.1thread") {
  overlapped_semaphore(1);
}

TEST_CASE("OverlappedSemaphore.2threads") {
  overlapped_semaphore(2);
}

TEST_CASE("OverlappedSemaphore.4threads") {
  overlapped_semaphore(4);
}

TEST_CASE("OverlappedSemaphore.8threads") {
  overlapped_semaphore(8);
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

TEST_CASE("ConflictGraph.1thread") {
  conflict_graph(1);
}

TEST_CASE("ConflictGraph.2threads") {
  conflict_graph(2);
}

TEST_CASE("ConflictGraph.3threads") {
  conflict_graph(3);
}

TEST_CASE("ConflictGraph.4threads") {
  conflict_graph(4);
}

// --------------------------------------------------------
// Testcase: In-task Semaphore
// --------------------------------------------------------

TEST_CASE("IntaskSemaphore.Happypass") {
  tf::Semaphore se(1);
  tf::Executor executor(2);
  tf::Taskflow taskflow;
  int32_t count = 0;
  taskflow.emplace([&](tf::Runtime& rt) {
            rt.acquire(se);
            --count;
            rt.release(se);
          })
          .priority(tf::TaskPriority::LOW);
  taskflow.emplace([&](tf::Runtime& rt) {
            rt.acquire(se);
            ++count;
            rt.release(se);
          })
          .priority(tf::TaskPriority::HIGH);
  executor.run(taskflow);
  executor.wait_for_all();
  REQUIRE(count == 0);
}

void intask_semaphore(size_t W) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;
  tf::Semaphore se(1);

  int N = 1000;
  int counter = 0;

  for (int i = 0; i < N; i++) {
    auto f = taskflow.emplace([&](tf::Runtime& rt) {
      rt.acquire(se);
      counter++;
    });
    auto t = taskflow.emplace([&](tf::Runtime& rt) {
      counter++;
      rt.release(se);
    });
    f.precede(t);
  }

  executor.run(taskflow).wait();

  REQUIRE(counter == 2 * N);
}

TEST_CASE("IntaskSemaphore.1thread") {
  intask_semaphore(1);
}

TEST_CASE("IntaskSemaphore.2threads") {
  intask_semaphore(2);
}

TEST_CASE("IntaskSemaphore.3threads") {
  intask_semaphore(3);
}

TEST_CASE("IntaskSemaphore.4threads") {
  intask_semaphore(4);
}

TEST_CASE("IntaskSemaphore.SemaphoreGuard") {
  tf::Semaphore se(1);
  tf::Executor executor(2);
  tf::Taskflow taskflow;
  int32_t count = 0;
  taskflow.emplace([&](tf::Runtime& rt) {
            tf::SemaphoreGuard gd(rt, se);
            --count;
          })
          .priority(tf::TaskPriority::LOW);
  taskflow.emplace([&](tf::Runtime& rt) {
            tf::SemaphoreGuard gd(rt, se);
            ++count;
          })
          .priority(tf::TaskPriority::HIGH);
  executor.run(taskflow);
  executor.wait_for_all();
  REQUIRE(count == 0);
}

void bench_semaphore(size_t W, size_t N) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;
  tf::Semaphore se_1(1);
  tf::Semaphore se_2(1);

  int counter = 0;

  for (size_t i = 0; i < N; i++) {
    auto f = taskflow.emplace([&]() { counter++; });
    auto t = taskflow.emplace([&]() { counter++; });
    f.acquire(se_1).release(se_1);
    t.acquire(se_1).release(se_1);
    f.acquire(se_2).release(se_2);
    t.acquire(se_2).release(se_2);
  }

  auto beg = std::chrono::high_resolution_clock::now();
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  // std::cout << "semaphore worker_num:" << W << " thread_num:" << N
  //           << " time cost : "
  //           << std::chrono::duration_cast<std::chrono::microseconds>(end - beg)
  //                      .count()
  //           << "us." << std::endl;
}

void bench_intask_semaphore(size_t W, size_t N) {
  tf::Executor executor(W);
  tf::Taskflow taskflow;
  tf::Semaphore se_1(1);
  tf::Semaphore se_2(1);

  int counter = 0;

  for (size_t i = 0; i < N; i++) {
    taskflow.emplace([&](tf::Runtime& rt) {
      rt.acquire(se_1);
      rt.acquire(se_2);
      counter++;
      rt.release(se_2);
      rt.release(se_1);
    });
    taskflow.emplace([&](tf::Runtime& rt) {
      rt.acquire(se_1);
      rt.acquire(se_2);
      counter++;
      rt.release(se_2);
      rt.release(se_1);
    });
  }

  auto beg = std::chrono::high_resolution_clock::now();
  executor.run(taskflow).wait();
  auto end = std::chrono::high_resolution_clock::now();
  // std::cout << "intask_semaphore worker_num:" << W << " thread_num:" << N
  //           << " time cost : "
  //           << std::chrono::duration_cast<std::chrono::microseconds>(end - beg)
  //                      .count()
  //           << "us." << std::endl;
}

#define TASKFLOW_TEST_SEMAPHORE_BENCHMARK(W, N)                                \
  TEST_CASE("IntaskSemaphore.BenchmarkW##vN##") {                              \
    bench_semaphore(W, N);                                                     \
    bench_intask_semaphore(W, N);                                              \
  }

TASKFLOW_TEST_SEMAPHORE_BENCHMARK(1, 10000);
TASKFLOW_TEST_SEMAPHORE_BENCHMARK(2, 10000);
TASKFLOW_TEST_SEMAPHORE_BENCHMARK(4, 10000);
TASKFLOW_TEST_SEMAPHORE_BENCHMARK(8, 10000);
TASKFLOW_TEST_SEMAPHORE_BENCHMARK(16, 10000);

TASKFLOW_TEST_SEMAPHORE_BENCHMARK(1, 20000);
TASKFLOW_TEST_SEMAPHORE_BENCHMARK(2, 20000);
TASKFLOW_TEST_SEMAPHORE_BENCHMARK(4, 20000);
TASKFLOW_TEST_SEMAPHORE_BENCHMARK(8, 20000);
TASKFLOW_TEST_SEMAPHORE_BENCHMARK(16, 20000);

#undef TASKFLOW_TEST_SEMAPHORE_BENCHMARK
