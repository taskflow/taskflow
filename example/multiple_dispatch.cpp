// An example to show dispatching multiple Taskflow graphs as
// separate batches (which will all run on the same Threadpool).
//
// It first dispatches 5 "independent" tasks (#100-104),
// then launches four batches (0-3) of a task graph with inter-dependencies,
// then it waits for the 100-104 tasks to finish before
// launching 5 more independent tasks (#200-204).

#include <taskflow/taskflow.hpp>

void syncLog(std::string const& msg)
{
  static std::mutex logMutex;
  std::lock_guard<std::mutex> lock(logMutex);
  std::cout << msg << '\n';
}

void dispatchBatch(tf::Taskflow& tf, int batch)
{
  auto taskMaker = [](std::string const& taskName, int batch) {
    return [=]() {
      // Simulate some work
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      syncLog("  Batch " + std::to_string(batch) + " - done task " + taskName);
    };
  };
  auto[A, B, C, D] = tf.silent_emplace(
    taskMaker("A", batch),
    taskMaker("B", batch),
    taskMaker("C", batch),
    taskMaker("D", batch)
  );

  A.precede(B);  // B runs after A
  A.precede(C);  // C runs after A
  B.precede(D);  // D runs after B
  C.precede(D);  // D runs after C

  // Schedule this independent graph of tasks (so they start running)
  tf.silent_dispatch();
}

int main()
{
  tf::Taskflow tf(std::thread::hardware_concurrency());
  auto const numIndependent = 5;
  for (auto indTask = 100; indTask < 100 + numIndependent; ++indTask) {
    tf.silent_emplace([=]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      syncLog("  Independent task " + std::to_string(indTask) + " - done");
    });
  }
  syncLog("Dispatching " + std::to_string(numIndependent) + " independent tasks (100 range)");
  auto independentTasksFuture = tf.dispatch();

  auto const numBatches = 4;
  for (auto batch = 0; batch < numBatches; ++batch) {
    dispatchBatch(tf, batch);
  }
  syncLog(std::to_string(numBatches) + " batches (task graphs) dispatched");

  // For some reason, we want to wait for the first set of
  // "independent tasks" to finish before dispatching more
  // of them...simulate that here:
  independentTasksFuture.get();
  syncLog("----- Independent tasks (100 range) completed");
  for (auto indTask = 200; indTask < 200 + numIndependent; ++indTask) {
    tf.silent_emplace([=]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      syncLog("  Independent task " + std::to_string(indTask) + " - done");
    });
  }
  syncLog("Dispatching " + std::to_string(numIndependent) + " independent tasks (200 range)");
  tf.silent_dispatch();

  syncLog("Waiting for all...");
  tf.wait_for_all();
  syncLog("...all tasks finished");

  return 0;
}
