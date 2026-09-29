module;

#include <atomic>
#include <chrono>
#include <functional>
#include <sstream>
#include <unordered_set>

export module test_tf;
import tf;

export int module_async(unsigned workers) {
  tf::Executor executor(workers);
  const int left = 19;
  const int right = 23;
  auto result = executor.async([left, right] { return left + right; });

  std::atomic<int> count{0};
  executor.silent_async([&count] { ++count; });
  auto runtime = executor.async([&count](tf::Runtime& rt) {
    rt.silent_async([&count] { ++count; });
  });
  auto first = executor.silent_dependent_async([&count] { ++count; });
  auto dependent = executor.dependent_async([&count] { ++count; }, first);
  dependent.second.get();
  runtime.get();
  executor.wait_for_all();
  return result.get() + count.load();
}

export int module_subflow(unsigned workers) {
  tf::Executor executor(workers);
  tf::Taskflow taskflow;
  tf::Graph graph;
  std::atomic<int> count{0};
  taskflow.emplace([&count](tf::Subflow& subflow) {
    auto a = subflow.emplace([&count] { ++count; });
    auto b = subflow.emplace([&count] { ++count; });
    a.precede(b);
    subflow.join();
    ++count;
  });
  executor.run(taskflow).get();
  return graph.empty() ? count.load() : -1;
}

export bool module_hash() {
  tf::Taskflow taskflow;
  auto task = taskflow.emplace([] {});
  std::unordered_set<tf::Task> tasks;
  tasks.insert(task);
  tasks.insert(task);
  tf::UUID id;
  std::unordered_set<tf::UUID> ids;
  ids.insert(id);
  return tasks.size() == 1 && tasks.count(task) == 1 && ids.count(id) == 1;
}

export bool module_profile(unsigned workers) {
  tf::Executor executor(workers);
  auto observer = executor.make_observer<tf::TFProfObserver>();
  tf::Taskflow taskflow;
  taskflow.emplace([] {});
  executor.run(taskflow).get();
  std::ostringstream output;
  observer->summary(output);
  return !output.str().empty();
}
