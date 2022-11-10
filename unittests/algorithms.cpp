#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/reduce.hpp>
#include <taskflow/algorithm/sort.hpp>
#include <taskflow/algorithm/transform.hpp>

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

struct MoveOnly2{

  int b {-1234};

  MoveOnly2() = default;

  MoveOnly2(const MoveOnly2&) = delete;
  MoveOnly2(MoveOnly2&&) = default;

  MoveOnly2& operator = (const MoveOnly2& rhs) = delete;
  MoveOnly2& operator = (MoveOnly2&& rhs) = default;

};

// --------------------------------------------------------
// Testcase: for_each
// --------------------------------------------------------

void for_each(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::vector<int> vec(1024);
  for(int n = 0; n <= 150; n++) {

    std::fill_n(vec.begin(), vec.size(), -1);

    int beg = ::rand()%300 - 150;
    int end = beg + n;

    for(int s=1; s<=16; s*=2) {
      for(int c=0; c<=17; c=c*2+1) {
        taskflow.clear();
        std::atomic<int> counter {0};

        taskflow.for_each_index(beg, end, s, [&](int i){
          counter++;
          vec[i-beg] = i;
        });

        executor.run(taskflow).wait();
        REQUIRE(counter == (n + s - 1) / s);

        for(int i=beg; i<end; i+=s) {
          REQUIRE(vec[i-beg] == i);
          vec[i-beg] = -1;
        }

        for(const auto i : vec) {
          REQUIRE(i == -1);
        }
      }
    }
  }

  for(size_t n = 0; n < 150; n++) {
    for(size_t c=0; c<=17; c=c*2+1) {

      std::fill_n(vec.begin(), vec.size(), -1);

      taskflow.clear();
      std::atomic<int> counter {0};

      taskflow.for_each(vec.begin(), vec.begin() + n, [&](int& i){
        counter++;
        i = 1;
      });

      executor.run(taskflow).wait();
      REQUIRE(counter == n);

      for(size_t i=0; i<n; ++i) {
        REQUIRE(vec[i] == 1);
      }

      for(size_t i=n; i<vec.size(); ++i) {
        REQUIRE(vec[i] == -1);
      }
    }
  }
}

TEST_CASE("ParallelFor.1thread" * doctest::timeout(300)) {
  for_each(1);
}

TEST_CASE("ParallelFor.2threads" * doctest::timeout(300)) {
  for_each(2);
}

TEST_CASE("ParallelFor.3threads" * doctest::timeout(300)) {
  for_each(3);
}

TEST_CASE("ParallelFor.4threads" * doctest::timeout(300)) {
  for_each(4);
}

TEST_CASE("ParallelFor.5threads" * doctest::timeout(300)) {
  for_each(5);
}

TEST_CASE("ParallelFor.6threads" * doctest::timeout(300)) {
  for_each(6);
}

TEST_CASE("ParallelFor.7threads" * doctest::timeout(300)) {
  for_each(7);
}

TEST_CASE("ParallelFor.8threads" * doctest::timeout(300)) {
  for_each(8);
}

TEST_CASE("ParallelFor.9threads" * doctest::timeout(300)) {
  for_each(9);
}

TEST_CASE("ParallelFor.10threads" * doctest::timeout(300)) {
  for_each(10);
}

TEST_CASE("ParallelFor.11threads" * doctest::timeout(300)) {
  for_each(11);
}

TEST_CASE("ParallelFor.12threads" * doctest::timeout(300)) {
  for_each(12);
}

// ----------------------------------------------------------------------------
// stateful_for_each
// ----------------------------------------------------------------------------

void stateful_for_each(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  std::vector<int> vec;
  std::atomic<int> counter {0};

  for(size_t n = 0; n <= 150; n++) {
    for(size_t c=0; c<=17; c++) {

    std::vector<int>::iterator beg, end;
    size_t ibeg = 0, iend = 0;
    size_t half = n/2;

    taskflow.clear();

    auto init = taskflow.emplace([&](){
      vec.resize(n);
      std::fill_n(vec.begin(), vec.size(), -1);

      beg = vec.begin();
      end = beg + half;

      ibeg = half;
      iend = n;

      counter = 0;
    });

    tf::Task pf1, pf2;

    pf1 = taskflow.for_each(
      std::ref(beg), std::ref(end), [&](int& i){
      counter++;
      i = 8;
    });

    pf2 = taskflow.for_each_index(
      std::ref(ibeg), std::ref(iend), size_t{1}, [&] (size_t i) {
        counter++;
        vec[i] = -8;
    });

    init.precede(pf1, pf2);

    executor.run(taskflow).wait();
    REQUIRE(counter == n);

    for(size_t i=0; i<half; ++i) {
      REQUIRE(vec[i] == 8);
      vec[i] = 0;
    }

    for(size_t i=half; i<n; ++i) {
      REQUIRE(vec[i] == -8);
      vec[i] = 0;
    }
    }
  }
}

// guided
TEST_CASE("StatefulParallelFor.1thread" * doctest::timeout(300)) {
  stateful_for_each(1);
}

TEST_CASE("StatefulParallelFor.2threads" * doctest::timeout(300)) {
  stateful_for_each(2);
}

TEST_CASE("StatefulParallelFor.3threads" * doctest::timeout(300)) {
  stateful_for_each(3);
}

TEST_CASE("StatefulParallelFor.4threads" * doctest::timeout(300)) {
  stateful_for_each(4);
}

TEST_CASE("StatefulParallelFor.5threads" * doctest::timeout(300)) {
  stateful_for_each(5);
}

TEST_CASE("StatefulParallelFor.6threads" * doctest::timeout(300)) {
  stateful_for_each(6);
}

TEST_CASE("StatefulParallelFor.7threads" * doctest::timeout(300)) {
  stateful_for_each(7);
}

TEST_CASE("StatefulParallelFor.8threads" * doctest::timeout(300)) {
  stateful_for_each(8);
}

TEST_CASE("StatefulParallelFor.9threads" * doctest::timeout(300)) {
  stateful_for_each(9);
}

TEST_CASE("StatefulParallelFor.10threads" * doctest::timeout(300)) {
  stateful_for_each(10);
}

TEST_CASE("StatefulParallelFor.11threads" * doctest::timeout(300)) {
  stateful_for_each(11);
}

TEST_CASE("StatefulParallelFor.12threads" * doctest::timeout(300)) {
  stateful_for_each(12);
}

// --------------------------------------------------------
// Testcase: reduce
// --------------------------------------------------------

void reduce(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::vector<int> vec(1000);

  for(auto& i : vec) i = ::rand() % 100 - 50;

  for(size_t n=1; n<vec.size(); n++) {
    for(size_t c=0; c<=17; c=c*2+1) {

      int smin = std::numeric_limits<int>::max();
      int pmin = std::numeric_limits<int>::max();
      auto beg = vec.end();
      auto end = vec.end();

      taskflow.clear();
      auto stask = taskflow.emplace([&](){
        beg = vec.begin();
        end = vec.begin() + n;
        for(auto itr = beg; itr != end; itr++) {
          smin = std::min(*itr, smin);
        }
      });

      tf::Task ptask;

      ptask = taskflow.reduce(
        std::ref(beg), std::ref(end), pmin, [](int& l, int& r){
        return std::min(l, r);
      });

      stask.precede(ptask);

      executor.run(taskflow).wait();

      REQUIRE(smin != std::numeric_limits<int>::max());
      REQUIRE(pmin != std::numeric_limits<int>::max());
      REQUIRE(smin == pmin);
    }
  }
}

// guided
TEST_CASE("Reduce.1thread" * doctest::timeout(300)) {
  reduce(1);
}

TEST_CASE("Reduce.2threads" * doctest::timeout(300)) {
  reduce(2);
}

TEST_CASE("Reduce.3threads" * doctest::timeout(300)) {
  reduce(3);
}

TEST_CASE("Reduce.4threads" * doctest::timeout(300)) {
  reduce(4);
}

TEST_CASE("Reduce.5threads" * doctest::timeout(300)) {
  reduce(5);
}

TEST_CASE("Reduce.6threads" * doctest::timeout(300)) {
  reduce(6);
}

TEST_CASE("Reduce.7threads" * doctest::timeout(300)) {
  reduce(7);
}

TEST_CASE("Reduce.8threads" * doctest::timeout(300)) {
  reduce(8);
}

TEST_CASE("Reduce.9threads" * doctest::timeout(300)) {
  reduce(9);
}

TEST_CASE("Reduce.10threads" * doctest::timeout(300)) {
  reduce(10);
}

TEST_CASE("Reduce.11threads" * doctest::timeout(300)) {
  reduce(11);
}

TEST_CASE("Reduce.12threads" * doctest::timeout(300)) {
  reduce(12);
}

// ----------------------------------------------------------------------------
// transform_reduce
// ----------------------------------------------------------------------------

class Data {

  private:

    int _v {::rand() % 100 - 50};

  public:

    int get() const { return _v; }
};

void transform_reduce(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::vector<Data> vec(1000);

  for(size_t n=1; n<vec.size(); n++) {
    for(size_t c=0; c<=17; c=c*2+1) {

      int smin = std::numeric_limits<int>::max();
      int pmin = std::numeric_limits<int>::max();
      auto beg = vec.end();
      auto end = vec.end();

      taskflow.clear();
      auto stask = taskflow.emplace([&](){
        beg = vec.begin();
        end = vec.begin() + n;
        for(auto itr = beg; itr != end; itr++) {
          smin = std::min(itr->get(), smin);
        }
      });

      tf::Task ptask;

      ptask = taskflow.transform_reduce(
        std::ref(beg), std::ref(end), pmin,
        [] (int l, int r)   { return std::min(l, r); },
        [] (const Data& data) { return data.get(); }
      );

      stask.precede(ptask);

      executor.run(taskflow).wait();

      REQUIRE(smin != std::numeric_limits<int>::max());
      REQUIRE(pmin != std::numeric_limits<int>::max());
      REQUIRE(smin == pmin);
    }
  }
}

// guided
TEST_CASE("TransformReduce.1thread" * doctest::timeout(300)) {
  transform_reduce(1);
}

TEST_CASE("TransformReduce.2threads" * doctest::timeout(300)) {
  transform_reduce(2);
}

TEST_CASE("TransformReduce.3threads" * doctest::timeout(300)) {
  transform_reduce(3);
}

TEST_CASE("TransformReduce.4threads" * doctest::timeout(300)) {
  transform_reduce(4);
}

TEST_CASE("TransformReduce.5threads" * doctest::timeout(300)) {
  transform_reduce(5);
}

TEST_CASE("TransformReduce.6threads" * doctest::timeout(300)) {
  transform_reduce(6);
}

TEST_CASE("TransformReduce.7threads" * doctest::timeout(300)) {
  transform_reduce(7);
}

TEST_CASE("TransformReduce.8threads" * doctest::timeout(300)) {
  transform_reduce(8);
}

TEST_CASE("TransformReduce.9threads" * doctest::timeout(300)) {
  transform_reduce(9);
}

TEST_CASE("TransformReduce.10threads" * doctest::timeout(300)) {
  transform_reduce(10);
}

TEST_CASE("TransformReduce.11threads" * doctest::timeout(300)) {
  transform_reduce(11);
}

TEST_CASE("TransformReduce.12threads" * doctest::timeout(300)) {
  transform_reduce(12);
}

// ----------------------------------------------------------------------------
// Transform & Reduce on Movable Data
// ----------------------------------------------------------------------------

void move_only_transform_reduce(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  const size_t N = 100000;
  std::vector<MoveOnly1> vec(N);

  for(auto& i : vec) i.a = 1;

  MoveOnly2 res;
  res.b = 100;

  taskflow.transform_reduce(vec.begin(), vec.end(), res,
    [](MoveOnly2 m1, MoveOnly2 m2) {
      MoveOnly2 res;
      res.b = m1.b + m2.b;
      return res;
    },
    [](const MoveOnly1& m) {
      MoveOnly2 n;
      n.b = m.a;
      return n;
    }
  );

  executor.run(taskflow).wait();

  REQUIRE(res.b == N + 100);
  
  // change vec data
  taskflow.clear();
  res.b = 0;
  
  taskflow.transform_reduce(vec.begin(), vec.end(), res,
    [](MoveOnly2&& m1, MoveOnly2&& m2) {
      MoveOnly2 res;
      res.b = m1.b + m2.b;
      return res;
    },
    [](MoveOnly1& m) {
      MoveOnly2 n;
      n.b = m.a;
      m.a = -7;
      return n;
    }
  );

  executor.run(taskflow).wait();
  REQUIRE(res.b == N);

  for(const auto& i : vec) {
    REQUIRE(i.a == -7);
  }

  // reduce
  taskflow.clear();
  MoveOnly1 red;
  red.a = 0;

  taskflow.reduce(vec.begin(), vec.end(), red,
    [](MoveOnly1& m1, MoveOnly1& m2){
      MoveOnly1 res;
      res.a = m1.a + m2.a;
      return res;
    }
  );  

  executor.run(taskflow).wait();
  REQUIRE(red.a == -7*N);
  
  taskflow.clear();
  red.a = 0;

  taskflow.reduce(vec.begin(), vec.end(), red,
    [](const MoveOnly1& m1, const MoveOnly1& m2){
      MoveOnly1 res;
      res.a = m1.a + m2.a;
      return res;
    }
  );  

  executor.run(taskflow).wait();
  REQUIRE(red.a == -7*N);

}

TEST_CASE("TransformReduce.MoveOnlyData.1thread" * doctest::timeout(300)) {
  move_only_transform_reduce(1);
}

TEST_CASE("TransformReduce.MoveOnlyData.2threads" * doctest::timeout(300)) {
  move_only_transform_reduce(2);
}

TEST_CASE("TransformReduce.MoveOnlyData.3threads" * doctest::timeout(300)) {
  move_only_transform_reduce(3);
}

TEST_CASE("TransformReduce.MoveOnlyData.4threads" * doctest::timeout(300)) {
  move_only_transform_reduce(4);
}

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

TEST_CASE("ParallelSort.int.1.100000") {
  ps_pod<int>(1, 100000);
}

TEST_CASE("ParallelSort.int.2.100000") {
  ps_pod<int>(2, 100000);
}

TEST_CASE("ParallelSort.int.3.100000") {
  ps_pod<int>(3, 100000);
}

TEST_CASE("ParallelSort.int.4.100000") {
  ps_pod<int>(4, 100000);
}

TEST_CASE("ParallelSort.ldouble.1.100000") {
  ps_pod<long double>(1, 100000);
}

TEST_CASE("ParallelSort.ldouble.2.100000") {
  ps_pod<long double>(2, 100000);
}

TEST_CASE("ParallelSort.ldouble.3.100000") {
  ps_pod<long double>(3, 100000);
}

TEST_CASE("ParallelSort.ldouble.4.100000") {
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

TEST_CASE("ParallelSort.object.1.100000") {
  ps_object(1, 100000);
}

TEST_CASE("ParallelSort.object.2.100000") {
  ps_object(2, 100000);
}

TEST_CASE("ParallelSort.object.3.100000") {
  ps_object(3, 100000);
}

TEST_CASE("ParallelSort.object.4.100000") {
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

TEST_CASE("ParallelSort.MoveOnlyObject.1thread") {
  move_only_ps(1);
}

TEST_CASE("ParallelSort.MoveOnlyObject.2threads") {
  move_only_ps(2);
}

TEST_CASE("ParallelSort.MoveOnlyObject.3threads") {
  move_only_ps(3);
}

TEST_CASE("ParallelSort.MoveOnlyObject.4threads") {
  move_only_ps(4);
}

// ----------------------------------------------------------------------------
// parallel transform
// ----------------------------------------------------------------------------

template<class T>
void parallel_transform(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  for(size_t N=0; N<1000; N++) {

    typename T::const_iterator src_beg;
    typename T::const_iterator src_end;
    std::list<std::string>::iterator tgt_beg;

    T src;
    std::list<std::string> tgt;

    taskflow.clear();

    auto from = taskflow.emplace([&](){
      src.resize(N);
      for(auto& d : src) {
        d = ::rand() % 10;
        tgt.emplace_back("hi");
      }
      src_beg = src.begin();
      src_end = src.end();
      tgt_beg = tgt.begin();
    });

    auto to = taskflow.transform(
      std::ref(src_beg), std::ref(src_end), std::ref(tgt_beg),
      [] (const auto& in) {
        return std::to_string(in+10);
      }
    );

    from.precede(to);

    executor.run(taskflow).wait();

    auto s_itr = src.begin();
    auto d_itr = tgt.begin();
    while(s_itr != src.end()) {
      REQUIRE(*d_itr++ == std::to_string(*s_itr++ + 10));
    }

  }
}

TEST_CASE("ParallelTransform.1thread") {
  parallel_transform<std::vector<int>>(1);
  parallel_transform<std::list<int>>(1);
}

TEST_CASE("ParallelTransform.2threads") {
  parallel_transform<std::vector<int>>(2);
  parallel_transform<std::list<int>>(2);
}

TEST_CASE("ParallelTransform.3threads") {
  parallel_transform<std::vector<int>>(3);
  parallel_transform<std::list<int>>(3);
}

TEST_CASE("ParallelTransform.4threads") {
  parallel_transform<std::vector<int>>(4);
  parallel_transform<std::list<int>>(4);
}

template<class T>
void parallel_transform2(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  for(size_t N=0; N<1000; N++) {

    typename T::const_iterator src_beg;
    typename T::const_iterator src_end;
    std::list<std::string>::iterator tgt_beg;

    T src;
    std::list<std::string> tgt;

    taskflow.clear();

    auto from = taskflow.emplace([&](){
      src.resize(N);
      for(auto& d : src) {
        d = ::rand() % 10;
        tgt.emplace_back("hi");
      }
      src_beg = src.begin();
      src_end = src.end();
      tgt_beg = tgt.begin();
    });

    auto to = taskflow.transform(
      std::ref(src_beg), std::ref(src_end), std::ref(src_beg), std::ref(tgt_beg),
      [] (const auto& in1, const auto& in2) {
        return std::to_string(in1 + in2 + 10);
      }
    );

    from.precede(to);

    executor.run(taskflow).wait();

    auto s_itr = src.begin();
    auto d_itr = tgt.begin();
    while(s_itr != src.end()) {
      REQUIRE(*d_itr++ == std::to_string(2 * *s_itr++ + 10));
    }

  }
}

TEST_CASE("parallel_transform2.1thread") {
  parallel_transform2<std::vector<int>>(1);
  parallel_transform2<std::list<int>>(1);
}

TEST_CASE("parallel_transform2.2threads") {
  parallel_transform2<std::vector<int>>(2);
  parallel_transform2<std::list<int>>(2);
}

TEST_CASE("parallel_transform2.3threads") {
  parallel_transform2<std::vector<int>>(3);
  parallel_transform2<std::list<int>>(3);
}

TEST_CASE("parallel_transform2.4threads") {
  parallel_transform2<std::vector<int>>(4);
  parallel_transform2<std::list<int>>(4);
}

void parallel_transform3(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  using std::string;
  using std::size_t;

  for(size_t N=0; N<1000; N++) {

    std::multimap<int, size_t> src;

    /** Reference implementation with std::transform */
    std::vector<string> ref;

    /** Target implementation with Subflow::transform */
    std::vector<string> tgt;

    std::vector<string>::iterator tgt_beg;

    /** A generic function to cast integers to string */
    const auto myFunction = [](const size_t x) -> string {
      return "id_" + std::to_string(x);
    };

    taskflow.clear();

    /** Group integers 0..(N-1) into ten groups,
     * each having an unique key `d`.
     */
    auto from = taskflow.emplace([&, N](){
      for(size_t i = 0; i < N; i++) {
        const int d = ::rand() % 10;
        src.emplace(d, i);
      }

      ref.resize(N);

      tgt.resize(N);
      tgt_beg = tgt.begin();
    });

    auto to_ref = taskflow.emplace([&]() {

      // Find entries matching key = 0.
      // This can return empty results.
      const auto [src_beg, src_end] = src.equal_range(0);
      const size_t n_matching = std::distance(src_beg, src_end);
      ref.resize(n_matching);

      // Extract all values having matching key value.
      std::transform(src_beg, src_end, ref.begin(),
        [&](const auto& x) -> string {
          return myFunction(x.second);
      });
    });

    /** Dynamic scheduling with Subflow::transform */
    auto to_tgt = taskflow.emplace([&](tf::Subflow& subflow) {

      // Find entries matching key = 0
      const auto [src_beg, src_end] = src.equal_range(0);
      const size_t n_matching = std::distance(src_beg, src_end);
      tgt.resize(n_matching);

      subflow.transform(std::ref(src_beg), std::ref(src_end), std::ref(tgt_beg),
        [&] (const auto& x) -> string {
          return myFunction(x.second);
      });

      subflow.join();
    });

    from.precede(to_ref);
    from.precede(to_tgt);

    executor.run(taskflow).wait();

    /** Target entries much match. */
    REQUIRE(std::equal(tgt.begin(), tgt.end(), ref.begin()));
  }
}

TEST_CASE("parallel_transform3.1thread") {
  parallel_transform3(1);
  parallel_transform3(1);
}

TEST_CASE("parallel_transform3.2threads") {
  parallel_transform3(2);
  parallel_transform3(2);
}

TEST_CASE("parallel_transform3.3threads") {
  parallel_transform3(3);
  parallel_transform3(3);
}

TEST_CASE("parallel_transform3.4threads") {
  parallel_transform3(4);
  parallel_transform3(4);
}
