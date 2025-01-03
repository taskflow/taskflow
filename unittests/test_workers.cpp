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

void worker_interface_basics(unsigned W) {

  std::atomic<size_t> counter{0};
  std::vector<size_t> ids;

  {
    tf::Executor executor(W, tf::make_worker_interface<CustomWorkerBehavior>(counter, ids));
  }

  REQUIRE(counter == W*2);
  REQUIRE(ids.size() == W);

  std::sort(ids.begin(), ids.end());

  for(size_t i=0; i<W; i++) {
    REQUIRE(ids[i] == i);
  }
}

TEST_CASE("WorkerInterface.Basics.1thread" * doctest::timeout(300)) {
  worker_interface_basics(1);
}

TEST_CASE("WorkerInterface.Basics.2threads" * doctest::timeout(300)) {
  worker_interface_basics(2);
}

TEST_CASE("WorkerInterface.Basics.3threads" * doctest::timeout(300)) {
  worker_interface_basics(3);
}

TEST_CASE("WorkerInterface.Basics.4threads" * doctest::timeout(300)) {
  worker_interface_basics(4);
}

TEST_CASE("WorkerInterface.Basics.5threads" * doctest::timeout(300)) {
  worker_interface_basics(5);
}

TEST_CASE("WorkerInterface.Basics.6threads" * doctest::timeout(300)) {
  worker_interface_basics(6);
}

TEST_CASE("WorkerInterface.Basics.7threads" * doctest::timeout(300)) {
  worker_interface_basics(7);
}

TEST_CASE("WorkerInterface.Basics.8threads" * doctest::timeout(300)) {
  worker_interface_basics(8);
}
