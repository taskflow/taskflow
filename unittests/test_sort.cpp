#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/sort.hpp>

// ----------------------------------------------------------------------------
// Data Type
// ----------------------------------------------------------------------------

struct MoveOnly1{

  int a {-1234};
  
  MoveOnly1() = default;

  MoveOnly1(const MoveOnly1&) = delete;
  MoveOnly1(MoveOnly1&&) = default;

  MoveOnly1& operator = (const MoveOnly1& rhs) = delete;
  MoveOnly1& operator = (MoveOnly1&& rhs) = default;

};

// ----------------------------------------------------------------------------
// parallel sort
// ----------------------------------------------------------------------------

template <typename T>
void ps_pod(size_t W, size_t N) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  std::vector<T> data(N);

  for(auto& d : data) {
    d = ::rand() % 1000 - 500;
  }

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  taskflow.sort(data.begin(), data.end());

  executor.run(taskflow).wait();

  REQUIRE(std::is_sorted(data.begin(), data.end()));
}

TEST_CASE("ParallelSort.int.1.100000" * doctest::timeout(300)) {
  ps_pod<int>(1, 100000);
}

TEST_CASE("ParallelSort.int.2.100000" * doctest::timeout(300)) {
  ps_pod<int>(2, 100000);
}

TEST_CASE("ParallelSort.int.3.100000" * doctest::timeout(300)) {
  ps_pod<int>(3, 100000);
}

TEST_CASE("ParallelSort.int.4.100000" * doctest::timeout(300)) {
  ps_pod<int>(4, 100000);
}

TEST_CASE("ParallelSort.ldouble.1.100000" * doctest::timeout(300)) {
  ps_pod<long double>(1, 100000);
}

TEST_CASE("ParallelSort.ldouble.2.100000" * doctest::timeout(300)) {
  ps_pod<long double>(2, 100000);
}

TEST_CASE("ParallelSort.ldouble.3.100000" * doctest::timeout(300)) {
  ps_pod<long double>(3, 100000);
}

TEST_CASE("ParallelSort.ldouble.4.100000" * doctest::timeout(300)) {
  ps_pod<long double>(4, 100000);
}

struct Object {

  std::array<int, 10> integers;

  int sum() const {
    int s = 0;
    for(const auto i : integers) {
      s += i;
    }
    return s;
  }
};

void ps_object(size_t W, size_t N) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  std::vector<Object> data(N);

  for(auto& d : data) {
    for(auto& i : d.integers) {
      i = ::rand();
    }
  }

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  taskflow.sort(data.begin(), data.end(), [](const auto& l, const auto& r){
    return l.sum() < r.sum();
  });

  executor.run(taskflow).wait();

  REQUIRE(std::is_sorted(data.begin(), data.end(),
    [](const auto& l, const auto& r){ return l.sum() < r.sum(); }
  ));
}

TEST_CASE("ParallelSort.object.1.100000" * doctest::timeout(300)) {
  ps_object(1, 100000);
}

TEST_CASE("ParallelSort.object.2.100000" * doctest::timeout(300)) {
  ps_object(2, 100000);
}

TEST_CASE("ParallelSort.object.3.100000" * doctest::timeout(300)) {
  ps_object(3, 100000);
}

TEST_CASE("ParallelSort.object.4.100000" * doctest::timeout(300)) {
  ps_object(4, 100000);
}

void move_only_ps(unsigned W) {
  
  std::vector<MoveOnly1> vec(1000000);
  for(auto& i : vec) {
    i.a = rand()%100;
  }

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  taskflow.sort(vec.begin(), vec.end(),
    [](const MoveOnly1& m1, const MoveOnly1&m2) {
      return m1.a < m2.a;
    }
  );

  executor.run(taskflow).wait();

  for(size_t i=1; i<vec.size(); i++) {
    REQUIRE(vec[i-1].a <= vec[i].a);
  }

}

TEST_CASE("ParallelSort.MoveOnlyObject.1thread" * doctest::timeout(300)) {
  move_only_ps(1);
}

TEST_CASE("ParallelSort.MoveOnlyObject.2threads" * doctest::timeout(300)) {
  move_only_ps(2);
}

TEST_CASE("ParallelSort.MoveOnlyObject.3threads" * doctest::timeout(300)) {
  move_only_ps(3);
}

TEST_CASE("ParallelSort.MoveOnlyObject.4threads" * doctest::timeout(300)) {
  move_only_ps(4);
}

// ----------------------------------------------------------------------------
// Parallel Sort with  Async Tasks
// ----------------------------------------------------------------------------

void async(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));
  
  tf::Executor executor(W);
  std::vector<int> data;

  for(size_t n=0; n < 100000; n = (n ? n*10 : 1)) {
    
    data.resize(n);

    for(auto& d : data) {
      d = ::rand() % 1000 - 500;
    }
  
    executor.async(tf::make_sort_task(data.begin(), data.end()));
    executor.wait_for_all();
    REQUIRE(std::is_sorted(data.begin(), data.end()));
  }
}

TEST_CASE("ParallelSort.Async.1thread" * doctest::timeout(300)) {
  async(1);
}

TEST_CASE("ParallelSort.Async.2threads" * doctest::timeout(300)) {
  async(2);
}

TEST_CASE("ParallelSort.Async.3threads" * doctest::timeout(300)) {
  async(3);
}

TEST_CASE("ParallelSort.Async.4threads" * doctest::timeout(300)) {
  async(4);
}

// ----------------------------------------------------------------------------
// Parallel Sort with Dependent Async Tasks
// ----------------------------------------------------------------------------

void dependent_async(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));
  
  tf::Executor executor(W);
  std::vector<int> data;

  for(size_t n=0; n < 100000; n = (n ? n*10 : 1)) {
    
    data.resize(n);

    for(auto& d : data) {
      d = ::rand() % 1000 - 500;
    }
  
    executor.dependent_async(tf::make_sort_task(data.begin(), data.end()));
    executor.wait_for_all();
    REQUIRE(std::is_sorted(data.begin(), data.end()));
  }
}

TEST_CASE("ParallelSort.DependentAsync.1thread" * doctest::timeout(300)) {
  dependent_async(1);
}

TEST_CASE("ParallelSort.DependentAsync.2threads" * doctest::timeout(300)) {
  dependent_async(2);
}

TEST_CASE("ParallelSort.DependentAsync.3threads" * doctest::timeout(300)) {
  dependent_async(3);
}

TEST_CASE("ParallelSort.DependentAsync.4threads" * doctest::timeout(300)) {
  dependent_async(4);
}

// ----------------------------------------------------------------------------
// Parallel Sort with Silent Async Tasks
// ----------------------------------------------------------------------------

void silent_async(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));
  
  tf::Executor executor(W);
  std::vector<int> data;

  for(size_t n=0; n < 100000; n = (n ? n*10 : 1)) {
    
    data.resize(n);

    for(auto& d : data) {
      d = ::rand() % 1000 - 500;
    }
  
    executor.silent_async(tf::make_sort_task(data.begin(), data.end()));
    executor.wait_for_all();
    REQUIRE(std::is_sorted(data.begin(), data.end()));
  }
}

TEST_CASE("ParallelSort.SilentAsync.1thread" * doctest::timeout(300)) {
  silent_async(1);
}

TEST_CASE("ParallelSort.SilentAsync.2threads" * doctest::timeout(300)) {
  silent_async(2);
}

TEST_CASE("ParallelSort.SilentAsync.3threads" * doctest::timeout(300)) {
  silent_async(3);
}

TEST_CASE("ParallelSort.SilentAsync.4threads" * doctest::timeout(300)) {
  silent_async(4);
}

// ----------------------------------------------------------------------------
// Parallel Sort with Silent Dependent Async Tasks
// ----------------------------------------------------------------------------

void silent_dependent_async(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));
  
  tf::Executor executor(W);
  std::vector<int> data;

  for(size_t n=0; n < 100000; n = (n ? n*10 : 1)) {
    
    data.resize(n);

    for(auto& d : data) {
      d = ::rand() % 1000 - 500;
    }
  
    executor.silent_dependent_async(tf::make_sort_task(data.begin(), data.end()));
    executor.wait_for_all();
    REQUIRE(std::is_sorted(data.begin(), data.end()));
  }
}

TEST_CASE("ParallelSort.SilentDependentAsync.1thread" * doctest::timeout(300)) {
  silent_dependent_async(1);
}

TEST_CASE("ParallelSort.SilentDependentAsync.2threads" * doctest::timeout(300)) {
  silent_dependent_async(2);
}

TEST_CASE("ParallelSort.SilentDependentAsync.3threads" * doctest::timeout(300)) {
  silent_dependent_async(3);
}

TEST_CASE("ParallelSort.SilentDependentAsync.4threads" * doctest::timeout(300)) {
  silent_dependent_async(4);
}



//// ----------------------------------------------------------------------------
//// Exception
//// ----------------------------------------------------------------------------
//
//void parallel_sort_exception(unsigned W) {
//
//  tf::Taskflow taskflow;
//  tf::Executor executor(W);
//
//  std::vector<int> data(1000000);
//
//  // for_each
//  taskflow.sort(data.begin(), data.end(), [](int a, int b){
//    throw std::runtime_error("x");
//    return a < b;
//  });
//  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "x", std::runtime_error);
//}
//
//TEST_CASE("ParallelSort.Exception.1thread") {
//  parallel_sort_exception(1);
//}
//
//TEST_CASE("ParallelSort.Exception.2threads") {
//  parallel_sort_exception(2);
//}
//
//TEST_CASE("ParallelSort.Exception.3threads") {
//  parallel_sort_exception(3);
//}
//
//TEST_CASE("ParallelSort.Exception.4threads") {
//  parallel_sort_exception(4);
//}


