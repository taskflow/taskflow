#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

class CustomWorkerBehavior : public tf::WorkerInterface {

  public:

  CustomWorkerBehavior(std::atomic<size_t>& counter, std::vector<size_t>& ids) : 
    _counter {counter},
    _ids     {ids} {
  }
  
  void scheduler_prologue(tf::Worker& wv) override {
    _counter++;

    std::scoped_lock lock(_mutex);
    _ids.push_back(wv.id());
  }

  void scheduler_epilogue(tf::Worker&, std::exception_ptr) override {
    _counter++;
  }

  std::atomic<size_t>& _counter;
  std::vector<size_t>& _ids;

  std::mutex _mutex;

};

TEST_CASE("WorkerInterface" * doctest::timeout(300)) {

  const size_t N = 10;

  for(size_t n=1; n<=N; n++) {
    std::atomic<size_t> counter{0};
    std::vector<size_t> ids;

    {
      tf::Executor executor(n, std::make_shared<CustomWorkerBehavior>(counter, ids));
    }

    REQUIRE(counter == n*2);
    REQUIRE(ids.size() == n);

    std::sort(ids.begin(), ids.end(), std::less<int>{});

    for(size_t i=0; i<n; i++) {
      REQUIRE(ids[i] == i);
    }
  }

}














